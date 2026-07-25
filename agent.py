"""
agent.py
---------
Calls Groq (LLaMA model) to interpret ALREADY-CLASSIFIED data.

>>> IMPORTANT — SCOPE OF THE AI AGENT <<<
The sentiment/emotion classification and the mhi_score calculation are
BOTH done deterministically in Python (see main.py), before this module
is ever called. The AI agent does NOT classify text and does NOT see
raw per-row data at scale. Its only job is to read a small AGGREGATED
statistical summary (counts, percentages, averages) plus a handful of
sample comments, and return ONE clean JSON object interpreting that
summary — a plain-language narrative, themes, anomalies, and
recommendations. It is a single request/response summarization call,
not a conversational or classification agent. This distinction matters
for the project documentation: the AI is a "reporting layer" on top of
already-computed analytics, not the source of the analytics itself.

This keeps the agent cheap and fast even when the underlying CSV/XLSX
has 100,000+ rows, because we never send raw rows to the model.
"""

import json
from groq import Groq

import config

_client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a workplace/social morale analyst. You will be given
AGGREGATED statistics (not raw data) about a batch of analyzed comments:
sentiment distribution, emotion distribution, an overall Morale Health
Index (MHI) score, and a small sample of example comments.

Your job is ONLY to summarize and interpret this aggregated data for a
human reader. You do not classify or re-score anything.

Respond with ONLY a valid JSON object, no markdown fences, no preamble,
matching exactly this schema:
{
  "summary": "2-4 sentence plain-language overview of overall morale/sentiment",
  "key_themes": ["short theme 1", "short theme 2", "..."],
  "anomalies": ["anything unusual or noteworthy in the distribution", "..."],
  "recommendations": ["actionable recommendation 1", "recommendation 2", "..."]
}
Keep each list to at most 5 concise items. Do not include any text outside the JSON object."""


def build_user_prompt(aggregated_stats: dict, sample_comments: list[str]) -> str:
    return json.dumps(
        {
            "aggregated_stats": aggregated_stats,
            "sample_comments": sample_comments,
        },
        indent=2,
        ensure_ascii=False,
    )


def _safe_parse_json(raw_text: str) -> dict:
    """Model occasionally wraps JSON in ```json fences despite instructions — strip them."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to a minimal, still-useful structure rather than crashing the request.
        return {
            "summary": raw_text.strip()[:500],
            "key_themes": [],
            "anomalies": [],
            "recommendations": [],
        }


def generate_insights(aggregated_stats: dict, sample_comments: list[str]) -> dict:
    """
    One clean, non-conversational call to Groq. Takes the aggregated stats
    + a small sample of comments, returns a parsed JSON dict with:
    summary, key_themes, anomalies, recommendations.
    """
    response = _client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(aggregated_stats, sample_comments)},
        ],
    )
    raw_text = response.choices[0].message.content
    return _safe_parse_json(raw_text)
