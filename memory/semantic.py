import uuid
from memory import store
from __future__ import annotations
from datetime import datetime, timezone


def store_fact(proposition: str, confidence: float = 0.8, source: str = "student"):
    doc_id = f"sem-{uuid.uuid4().hex[:12]}"
    store.add(
        "semantic",
        doc_id=doc_id,
        document=proposition,
        metadata={
            "confidence": float(confidence),
            "source": source,
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        },
    )
    return doc_id


def recall(query_text: str, n: int = 5):
    return store.query("semantic", query_text, n_results=n)


def reinforce(doc_id: str, current_meta: dict):
    """Boost confidence for a fact that was re-confirmed."""
    new_conf = min(1.0, current_meta.get("confidence", 0.5) + 0.1)
    store.update_metadata(
        "semantic",
        doc_id,
        {
            **current_meta,
            "confidence": new_conf,
            "updated": datetime.now(timezone.utc).isoformat(),
        },
    )


def weaken(doc_id: str, current_meta: dict):
    """Lower confidence for a fact that was contradicted."""
    new_conf = max(0.0, current_meta.get("confidence", 0.5) - 0.2)
    store.update_metadata(
        "semantic",
        doc_id,
        {
            **current_meta,
            "confidence": new_conf,
            "updated": datetime.now(timezone.utc).isoformat(),
        },
    )
