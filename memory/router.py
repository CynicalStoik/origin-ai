import json
from config import llm

ROUTING_PROMPT = """You are a memory router for a mindfulness coaching agent.
Given the user's message, decide which memory layers are needed to respond well.

User message: "{message}"

Memory types available:
- episodic: retrieve if the message relates to past emotional moments, specific situations, or events the user described in prior sessions
- facts: retrieve if stable known info about the user would help (sleep habits, stressors, coping strategies, personal details)
- compact: retrieve if understanding the user's overall journey or long-term progress is needed
- none: use if this is a greeting, small talk, or a simple question that needs no prior memory

Rules:
- You may pick one, multiple, or none.
- Only pick what is genuinely needed. Do not over-retrieve.
- Respond ONLY with valid JSON, nothing else.

Example outputs:
{{"layers": ["episodic", "facts"]}}
{{"layers": ["compact"]}}
{{"layers": []}}
"""


def route_memory(user_message: str) -> list:
    prompt = ROUTING_PROMPT.format(message=user_message)
    result = llm(prompt)
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        layers = parsed.get("layers", [])
        valid = {"episodic", "facts", "compact"}
        return [l for l in layers if l in valid]
    except Exception:
        return []