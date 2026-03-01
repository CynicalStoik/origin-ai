import json
from config import llm

REREF_PROMPT = """You are analyzing a user message in a mindfulness coaching conversation.
Determine if the user is referencing or confirming something from a past session.

Signs of re-reference:
- Mentioning a technique or suggestion from before ("that breathing thing", "what you said last time")
- Reporting back on something they tried ("it helped", "I did it again", "didn't work for me")
- Explicitly connecting to a previous moment ("like before", "as we discussed", "remember when")

User message: "{message}"

Respond ONLY with valid JSON, nothing else.
If re-referencing: {{"is_rereference": true, "reference_hint": "short description of what they're referencing"}}
If not: {{"is_rereference": false, "reference_hint": ""}}
"""


def detect_rereference(user_message: str) -> tuple[bool, str]:
    prompt = REREF_PROMPT.format(message=user_message)
    result = llm(prompt)
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return bool(parsed.get("is_rereference", False)), str(parsed.get("reference_hint", ""))
    except Exception:
        return False, ""