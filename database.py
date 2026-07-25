"""
database.py
------------
SQLite storage layer for the project. No external DB server — everything
lives in a single local file (see config.DB_PATH).

Two tables:
  1. entries  -> one row per comment/tweet, after sentiment/emotion/mhi
                 scoring. This is the raw, granular data.
  2. insights -> one row per uploaded batch, holding the AI agent's
                 JSON summary (summary, key_themes, anomalies,
                 recommendations) about that batch.

Every write is scoped to a `batch_id` (one per CSV/XLSX upload), so the
dashboard can show trends across multiple uploads over time.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

import config


def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet. Safe to call on every startup."""
    with db_cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id    TEXT NOT NULL,
                text        TEXT NOT NULL,
                sentiment   TEXT NOT NULL,
                emotion     TEXT NOT NULL,
                mhi_score   REAL NOT NULL,
                created_at  TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS insights (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id        TEXT NOT NULL,
                summary         TEXT,
                key_themes      TEXT,   -- stored as JSON array
                anomalies       TEXT,   -- stored as JSON array
                recommendations TEXT,   -- stored as JSON array
                avg_mhi_score   REAL,
                row_count       INTEGER,
                created_at      TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_batch ON entries(batch_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_insights_batch ON insights(batch_id)")


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def insert_entries(batch_id: str, rows: list[dict]) -> int:
    """
    rows: list of dicts with keys: text, sentiment, emotion, mhi_score
    Uses executemany for speed on large files (100k+ rows).
    """
    now = datetime.now(timezone.utc).isoformat()
    payload = [
        (batch_id, r["text"], r["sentiment"], r["emotion"], float(r["mhi_score"]), now)
        for r in rows
    ]
    with db_cursor() as cur:
        cur.executemany(
            """
            INSERT INTO entries (batch_id, text, sentiment, emotion, mhi_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
    return len(payload)


def insert_insight(batch_id: str, insight: dict, avg_mhi_score: float, row_count: int) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO insights
                (batch_id, summary, key_themes, anomalies, recommendations,
                 avg_mhi_score, row_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                insight.get("summary", ""),
                json.dumps(insight.get("key_themes", [])),
                json.dumps(insight.get("anomalies", [])),
                json.dumps(insight.get("recommendations", [])),
                avg_mhi_score,
                row_count,
                now,
            ),
        )
        return cur.lastrowid


def get_entries(batch_id: str | None = None, limit: int = 1000) -> list[dict]:
    with db_cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT * FROM entries WHERE batch_id = ? ORDER BY id DESC LIMIT ?",
                (batch_id, limit),
            )
        else:
            cur.execute("SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def get_insights(batch_id: str | None = None) -> list[dict]:
    with db_cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT * FROM insights WHERE batch_id = ? ORDER BY id DESC", (batch_id,)
            )
        else:
            cur.execute("SELECT * FROM insights ORDER BY id DESC")
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["key_themes"] = json.loads(r["key_themes"] or "[]")
            r["anomalies"] = json.loads(r["anomalies"] or "[]")
            r["recommendations"] = json.loads(r["recommendations"] or "[]")
        return rows


def list_batches() -> list[dict]:
    """Returns one row per batch with its avg mhi and timestamp, for the trend chart."""
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT batch_id,
                   AVG(mhi_score) AS avg_mhi_score,
                   COUNT(*) AS row_count,
                   MIN(created_at) AS created_at
            FROM entries
            GROUP BY batch_id
            ORDER BY created_at ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]
