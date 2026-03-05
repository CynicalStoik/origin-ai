from __future__ import annotations

import os
import json
import config
from memory import store

TECHNIQUES_PATH = os.path.join(config.BASE_DIR, "data", "techniques.json")


def load_techniques():
    """Populate the procedural collection from techniques.json (idempotent)."""
    col = store.get_collection("procedural")
    if col.count() > 0:
        return

    if not os.path.exists(TECHNIQUES_PATH):
        return

    with open(TECHNIQUES_PATH) as f:
        techniques = json.load(f)

    for tech in techniques:
        doc = (
            f"{tech['name']}: {tech['description']}\n"
            f"Best for: {tech['best_for']}\n"
            f"Steps: {' → '.join(tech['steps'])}"
        )
        store.add(
            "procedural",
            doc_id=f"proc-{tech['id']}",
            document=doc,
            metadata={
                "name": tech["name"],
                "best_for": tech["best_for"],
                "duration_minutes": tech.get("duration_minutes", 5),
            },
        )
    print(f"[procedural] Loaded {len(techniques)} techniques")


def recall(query_text: str, n: int = 3):
    return store.query("procedural", query_text, n_results=n)
