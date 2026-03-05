from __future__ import annotations

import os
import json
import config

_DEFAULT_PROFILE = {
    "name": None,
    "background": None,
    "stressors": [],
    "preferences": [],
    "patterns": [],
    "session_count": 0,
    "notes": "",
}


def load() -> dict:
    if os.path.exists(config.LPM_PATH):
        with open(config.LPM_PATH) as f:
            return json.load(f)
    return dict(_DEFAULT_PROFILE)


def save(profile: dict):
    os.makedirs(os.path.dirname(config.LPM_PATH), exist_ok=True)
    with open(config.LPM_PATH, "w") as f:
        json.dump(profile, f, indent=2)


def format_for_prompt(profile: dict) -> str:
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("background"):
        parts.append(f"Background: {profile['background']}")
    if profile.get("stressors"):
        parts.append(f"Known stressors: {', '.join(profile['stressors'])}")
    if profile.get("preferences"):
        parts.append(f"Preferences: {', '.join(profile['preferences'])}")
    if profile.get("patterns"):
        parts.append(f"Observed patterns: {', '.join(profile['patterns'])}")
    if profile.get("notes"):
        parts.append(f"Notes: {profile['notes']}")
    return "\n".join(parts) if parts else "(No profile yet.)"
