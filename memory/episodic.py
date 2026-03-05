from __future__ import annotations

import uuid
from memory import store
from datetime import datetime, timezone


def save_episode(
    content: str,
    session_id: str,
    role: str = "student",
    emotion: str = "neutral",
    emotion_strength: float = 0.0,
):
    doc_id = f"ep-{uuid.uuid4().hex[:12]}"
    store.add(
        "episodic",
        doc_id=doc_id,
        document=content,
        metadata={
            "session_id": session_id,
            "role": role,
            "emotion": emotion,
            "emotion_strength": float(emotion_strength),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return doc_id


def recall(query_text: str, n: int = 5, session_id: str | None = None):
    """Retrieve episodes most relevant to *query_text*."""
    where = {"session_id": session_id} if session_id else None
    return store.query("episodic", query_text, n_results=n, where=where)
