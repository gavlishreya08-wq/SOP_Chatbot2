"""Lightweight JSON-file store for feedback, failed queries, and analytics."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backend.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _data_dir() -> Path:
    path = Path(settings.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _append_json(filepath: Path, record: dict) -> None:
    with _lock:
        data = _read_json(filepath)
        data.append(record)
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_json(filepath: Path, data: list[dict]) -> None:
    with _lock:
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Feedback (thumbs up/down) ──────────────────────────────────────────

def save_feedback(
    question: str,
    answer: str,
    rating: Literal["up", "down"],
    active_sop: str | None = None,
    comment: str = "",
) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer[:500],
        "rating": rating,
        "active_sop": active_sop,
        "comment": comment,
    }
    _append_json(_data_dir() / "feedback.json", record)
    return record


def get_feedback(limit: int = 200) -> list[dict]:
    data = _read_json(_data_dir() / "feedback.json")
    return list(reversed(data[-limit:]))


# ── Failed queries ──────────────────────────────────────────────────────

def save_failed_query(
    question: str,
    confidence: str,
    active_sop: str | None = None,
    answer: str = "",
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer[:300],
        "confidence": confidence,
        "active_sop": active_sop,
    }
    _append_json(_data_dir() / "failed_queries.json", record)


def get_failed_queries(limit: int = 200) -> list[dict]:
    data = _read_json(_data_dir() / "failed_queries.json")
    return list(reversed(data[-limit:]))


def clear_failed_query(index: int) -> bool:
    filepath = _data_dir() / "failed_queries.json"
    data = _read_json(filepath)
    reversed_data = list(reversed(data))
    if 0 <= index < len(reversed_data):
        original_index = len(data) - 1 - index
        data.pop(original_index)
        _write_json(filepath, data)
        return True
    return False


# ── Query log (for analytics) ──────────────────────────────────────────

def log_query(
    question: str,
    active_sop: str | None,
    detected_sop: str | None,
    confidence: str,
    was_clarification: bool = False,
    answer_mode: str = "detailed",
    llm_provider: str = "",
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "active_sop": active_sop,
        "detected_sop": detected_sop,
        "confidence": confidence,
        "was_clarification": was_clarification,
        "answer_mode": answer_mode,
        "llm_provider": llm_provider,
    }
    _append_json(_data_dir() / "query_log.json", record)


def get_analytics_summary() -> dict:
    queries = _read_json(_data_dir() / "query_log.json")
    feedback = _read_json(_data_dir() / "feedback.json")
    failed = _read_json(_data_dir() / "failed_queries.json")

    total_queries = len(queries)
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    sop_counts: dict[str, int] = {}
    question_counts: dict[str, int] = {}
    clarification_count = 0

    for q in queries:
        conf = q.get("confidence", "low")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        sop = q.get("detected_sop")
        if sop:
            sop_counts[sop] = sop_counts.get(sop, 0) + 1
        question_key = q.get("question", "").strip().lower()[:100]
        if question_key:
            question_counts[question_key] = question_counts.get(question_key, 0) + 1
        if q.get("was_clarification"):
            clarification_count += 1

    top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_sops = sorted(sop_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    thumbs_up = sum(1 for f in feedback if f.get("rating") == "up")
    thumbs_down = sum(1 for f in feedback if f.get("rating") == "down")

    return {
        "total_queries": total_queries,
        "confidence_breakdown": confidence_counts,
        "clarification_count": clarification_count,
        "top_questions": [{"question": q, "count": c} for q, c in top_questions],
        "top_sops": [{"sop": s, "count": c} for s, c in top_sops],
        "feedback_summary": {
            "total": len(feedback),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
        },
        "failed_query_count": len(failed),
    }


# ── Conversation history ───────────────────────────────────────────────

def save_conversation(conversation_id: str, messages: list[dict], title: str = "") -> dict:
    filepath = _data_dir() / "conversations.json"
    data = _read_json(filepath)

    existing = next((c for c in data if c["id"] == conversation_id), None)
    if existing:
        existing["messages"] = messages
        existing["title"] = title or existing.get("title", "")
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        data.append({
            "id": conversation_id,
            "title": title or (messages[0]["content"][:60] if messages else "New chat"),
            "messages": messages,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    _write_json(filepath, data)
    return {"id": conversation_id, "saved": True}


def get_conversations(limit: int = 50) -> list[dict]:
    data = _read_json(_data_dir() / "conversations.json")
    summaries = []
    for conv in reversed(data[-limit:]):
        summaries.append({
            "id": conv["id"],
            "title": conv.get("title", ""),
            "message_count": len(conv.get("messages", [])),
            "created_at": conv.get("created_at", ""),
            "updated_at": conv.get("updated_at", ""),
        })
    return summaries


def get_conversation(conversation_id: str) -> dict | None:
    data = _read_json(_data_dir() / "conversations.json")
    return next((c for c in data if c["id"] == conversation_id), None)


def delete_conversation(conversation_id: str) -> bool:
    filepath = _data_dir() / "conversations.json"
    data = _read_json(filepath)
    new_data = [c for c in data if c["id"] != conversation_id]
    if len(new_data) < len(data):
        _write_json(filepath, new_data)
        return True
    return False


def search_conversations(query: str, limit: int = 20) -> list[dict]:
    data = _read_json(_data_dir() / "conversations.json")
    query_lower = query.lower()
    results = []
    for conv in reversed(data):
        title_match = query_lower in conv.get("title", "").lower()
        message_match = any(
            query_lower in msg.get("content", "").lower()
            for msg in conv.get("messages", [])
        )
        if title_match or message_match:
            results.append({
                "id": conv["id"],
                "title": conv.get("title", ""),
                "message_count": len(conv.get("messages", [])),
                "created_at": conv.get("created_at", ""),
                "updated_at": conv.get("updated_at", ""),
            })
            if len(results) >= limit:
                break
    return results
