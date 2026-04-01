from __future__ import annotations

import math
import uuid
import config
from memory import store
from datetime import datetime, timezone


def _hours_since(iso_ts: str) -> float:
    then = datetime.fromisoformat(iso_ts)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    return max(delta.total_seconds() / 3600.0, 0.001)


def _decay(timestamp: str) -> float:
    """Exponential decay based on hours elapsed."""
    hours = _hours_since(timestamp)
    return math.exp(-config.CONFIDENCE_DECAY_RATE * hours)


def store_belief(
    proposition: str,
    holder: str,
    truth_value: bool,
    confidence: float = 0.8,
    topic: str = "",
):
    doc_id = f"psp-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    store.add(
        "perspective",
        doc_id=doc_id,
        document=proposition,
        metadata={
            "holder": holder,
            "truth_value": str(truth_value),
            "confidence": float(confidence),
            "topic": topic,
            "negotiation_status": "open",
            "timestamp": now,
        },
    )
    return doc_id


def recall_by_topic(topic: str, n: int = 5):
    return store.query("perspective", topic, n_results=n)


def recall_active_divergences(topic: str, n: int = 5):
    """Return divergences that are still contested or open.
    Only entries with a prior_id are actual contradictions — freshly stored
    beliefs have no prior_id and must not be surfaced as divergences.
    """
    results = store.query("perspective", topic, n_results=n)
    return [
        r
        for r in results
        if r["metadata"].get("negotiation_status") in ("contested", "open")
        and r["metadata"].get("prior_id")
    ]


def resolve_divergence(
    new_proposition: str,
    new_holder: str,
    new_truth_value: bool,
    new_confidence: float,
    topic: str,
) -> str:
    """Check *new* claim against existing beliefs and assign negotiation status.

    Returns one of: "resolved", "contested", "open", "no_divergence".
    """
    existing = store.query("perspective", new_proposition, n_results=10)

    for entry in existing:
        meta = entry["metadata"]
        old_truth = meta.get("truth_value", "True") == "True"
        same_topic = _topics_overlap(meta.get("topic", ""), topic)

        if not same_topic:
            continue

        is_contradiction = old_truth != new_truth_value
        if not is_contradiction:
            continue

        w_new = new_confidence * _decay(datetime.now(timezone.utc).isoformat())
        w_old = meta.get("confidence", 0.5) * _decay(meta["timestamp"])

        eps = config.DIVERGENCE_EPSILON

        if w_new > w_old + eps:
            status = "resolved"
        elif abs(w_new - w_old) < eps:
            status = "contested"
        else:
            status = "open"

        now = datetime.now(timezone.utc).isoformat()
        div_id = f"psp-{uuid.uuid4().hex[:12]}"
        old_stance = "true" if old_truth else "not true"
        new_stance = "true" if new_truth_value else "not true"
        store.add(
            "perspective",
            doc_id=div_id,
            document=(
                f"Student's view shifted on '{entry['document']}': "
                f"was {old_stance}, now {new_stance}. topic: {topic}"
            ),
            metadata={
                "holder": new_holder,
                "truth_value": str(new_truth_value),
                "confidence": float(new_confidence),
                "topic": topic,
                "negotiation_status": status,
                "prior_id": entry["id"],
                "timestamp": now,
            },
        )

        if status == "resolved":
            store.update_metadata(
                "perspective",
                entry["id"],
                {
                    **meta,
                    "negotiation_status": "resolved",
                },
            )

        return status

    store_belief(new_proposition, new_holder, new_truth_value, new_confidence, topic)
    return "no_divergence"


def _topics_overlap(a: str, b: str) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return bool(a_words & b_words)
