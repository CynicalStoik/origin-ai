"""
Run this between participants to clear all participant-specific memory.
Keeps the procedural (techniques) collection intact.

Usage:
    python reset_participant.py
"""
import shutil
import config
from memory import store, ltm
from memory.procedural import load_techniques

# Wipe the entire chroma_db to avoid HNSW segment corruption errors
# that occur when collections are deleted but metadata references remain.
shutil.rmtree(config.CHROMA_PERSIST_DIR, ignore_errors=True)
print("[reset] Cleared: chroma_db")

# Reset persona
ltm.save(dict(ltm._DEFAULT_PROFILE))
print("[reset] Cleared: persona.json")

# Reload techniques into fresh procedural collection
load_techniques()
print("[reset] Ready for next participant.")
