"""
main.py
--------
FastAPI backend for the Agentic Sentiment Analysis & Morale Health Index System.

Endpoints:
  POST /analyze   - upload a CSV or XLSX with (text, sentiment, emotion[, mhi_score]),
                     computes mhi_score for any row missing it, stores everything in
                     SQLite, aggregates stats, and calls the Groq agent for a summary.
  GET  /entries    - fetch stored per-row entries (optionally filtered by batch_id)
  GET  /insights   - fetch stored AI insights (optionally filtered by batch_id)
  GET  /batches    - list all batches with their average MHI (for the trend chart)
"""

import io
import random
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import database
import agent

app = FastAPI(title="Morale Health Index API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

# ---------------------------------------------------------------------------
# MHI scoring logic
# ---------------------------------------------------------------------------
# Base score per sentiment (0-100 scale), then nudged up/down slightly by the
# emotion label, then clipped back into [0, 100]. This is intentionally
# simple and transparent rather than a black box, so it's easy to explain
# and to tune later if needed.

SENTIMENT_BASE = {
    "negative": 20,
    "neutral": 50,
    "positive": 80,
}

EMOTION_ADJUSTMENT = {
    "joy": 15,
    "surprise": 8,
    "neutral": 0,
    "sadness": -8,
    "disgust": -10,
    "fear": -12,
    "anger": -15,
}


def compute_mhi_score(sentiment: str, emotion: str) -> float:
    s = str(sentiment).strip().lower()
    e = str(emotion).strip().lower()
    base = SENTIMENT_BASE.get(s, 50)  # unknown sentiment -> neutral base
    adj = EMOTION_ADJUSTMENT.get(e, 0)  # unknown emotion -> no adjustment
    return round(max(0, min(100, base + adj)), 1)


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------

def _read_upload_to_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw_bytes))
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw_bytes))
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .csv or .xlsx file.",
        )


def _detect_text_column(df: pd.DataFrame) -> str:
    for candidate in ("clean_text", "text", "comment", "tweet"):
        if candidate in df.columns:
            return candidate
    raise HTTPException(
        status_code=400,
        detail="Could not find a text column. Expected one of: clean_text, text, comment, tweet.",
    )


def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _detect_text_column(df)

    for required in ("sentiment", "emotion"):
        if required not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required column: '{required}'.",
            )

    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

    if "mhi_score" not in df.columns:
        df["mhi_score"] = None

    # Fill mhi_score only where it's missing/NaN — never overwrite existing values.
    missing_mask = df["mhi_score"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "mhi_score"] = df.loc[missing_mask].apply(
            lambda r: compute_mhi_score(r["sentiment"], r["emotion"]), axis=1
        )
    df["mhi_score"] = df["mhi_score"].astype(float)

    return df[["text", "sentiment", "emotion", "mhi_score"]]


def _build_aggregated_stats(df: pd.DataFrame) -> dict:
    """
    Aggregate ONLY — this is what gets sent to the AI agent.
    No raw rows leave this function except the small sample below.
    """
    total = len(df)
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    emotion_counts = df["emotion"].value_counts().to_dict()

    return {
        "total_rows": total,
        "avg_mhi_score": round(df["mhi_score"].mean(), 2),
        "min_mhi_score": round(df["mhi_score"].min(), 2),
        "max_mhi_score": round(df["mhi_score"].max(), 2),
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": {
            k: round(v / total * 100, 2) for k, v in sentiment_counts.items()
        },
        "emotion_counts": emotion_counts,
        "emotion_percentages": {
            k: round(v / total * 100, 2) for k, v in emotion_counts.items()
        },
        "avg_mhi_by_sentiment": {
            k: round(v, 2) for k, v in df.groupby("sentiment")["mhi_score"].mean().to_dict().items()
        },
    }


def _sample_comments(df: pd.DataFrame, n: int) -> list[str]:
    """Small, stratified-ish sample of example comments sent alongside the stats."""
    n = min(n, len(df))
    if n <= 0:
        return []
    sample_df = df.sample(n=n, random_state=42)
    return sample_df["text"].tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    batch_id: str
    row_count: int
    aggregated_stats: dict
    insight: dict


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    df = _read_upload_to_dataframe(file.filename, raw_bytes)
    df = _validate_and_normalize(df)

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows.")

    batch_id = database.new_batch_id()

    # Store every row in SQLite (executemany under the hood — fine for 100k+ rows).
    rows = df.to_dict(orient="records")
    database.insert_entries(batch_id, rows)

    # Aggregate + sample -> this is ALL that goes to the AI agent, never raw rows at scale.
    aggregated_stats = _build_aggregated_stats(df)
    sample = _sample_comments(df, config.SAMPLE_SIZE)

    try:
        insight = agent.generate_insights(aggregated_stats, sample)
    except Exception as exc:
        # Don't fail the whole upload if the AI call has an issue — data is already saved.
        insight = {
            "summary": f"AI insight generation failed: {exc}",
            "key_themes": [],
            "anomalies": [],
            "recommendations": [],
        }

    database.insert_insight(
        batch_id, insight, aggregated_stats["avg_mhi_score"], aggregated_stats["total_rows"]
    )

    return AnalyzeResponse(
        batch_id=batch_id,
        row_count=aggregated_stats["total_rows"],
        aggregated_stats=aggregated_stats,
        insight=insight,
    )


@app.get("/entries")
def entries(batch_id: Optional[str] = Query(None), limit: int = Query(1000, le=20000)):
    return database.get_entries(batch_id=batch_id, limit=limit)


@app.get("/insights")
def insights(batch_id: Optional[str] = Query(None)):
    return database.get_insights(batch_id=batch_id)


@app.get("/batches")
def batches():
    return database.list_batches()


@app.get("/health")
def health():
    return {"status": "ok"}
