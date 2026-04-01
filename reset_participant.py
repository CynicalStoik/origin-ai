"""
Run this between participants to clear all participant-specific memory.
Keeps the procedural (techniques) collection intact.

Usage:
    python reset_participant.py
"""
import json
import config
from memory import store, ltm

PARTICIPANT_COLLECTIONS = ("episodic", "semantic", "perspective")

client = store._get_client()

for name in PARTICIPANT_COLLECTIONS:
    try:
        client.delete_collection(name)
        print(f"[reset] Cleared: {name}")
    except Exception:
        print(f"[reset] {name} did not exist — skipping")

ltm.save(dict(ltm._DEFAULT_PROFILE))
print(f"[reset] Cleared: persona.json")

proc = store.get_collection("procedural")
print(f"[reset] Kept:    procedural ({proc.count()} techniques)")
print("[reset] Ready for next participant.")
