import json
from config import llm

EXTRACTION_PROMPT = """You are a memory extraction assistant for a mindfulness coaching agent.
Analyze this conversation exchange and extract memory information.

User: {user_msg}
Agent: {agent_reply}

Extract the following as JSON:

1. "episode": A 2-3 sentence summary of what the user said and what the agent replied.
   Always write this — every exchange must be stored, even casual ones like food, weather, or small talk.

2. "episode_tags": List of relevant tags from: academic, social, physical, emotional, anxiety, sleep,
   work, relationships, coping, progress, setback, stress, calm, food, daily_life, health, mood.

3. "facts": List of stable facts about the user revealed in this exchange.
   Each fact: {{"content": "...", "tags": [...], "type": "semantic", "time_span": "ongoing"}}
   Only add facts if something stable and reusable is learned about the user (habits, preferences, conditions).
   Leave as empty list [] if nothing stable was revealed.

Respond ONLY with valid JSON, nothing else.
Example:
{{
  "episode": "User mentioned they ate an apple this morning. Agent acknowledged this as a healthy choice.",
  "episode_tags": ["food", "daily_life", "health"],
  "facts": []
}}
"""

SUMMARY_UPDATE_PROMPT = """You are updating a long-term narrative summary for a mindfulness coaching agent.

Previous summary:
{previous_summary}

Recent conversation (last {n} turns):
{recent_turns}

Write an updated 2-3 sentence summary that:
- Captures the user's overall situation and journey
- Notes any meaningful changes or progress
- Uses warm, empathetic language
- Focuses on what matters most for ongoing coaching

Respond with ONLY the updated summary text, nothing else.
"""

PERSONA_UPDATE_PROMPT = """You are updating a user persona profile for a mindfulness coaching agent.

Current persona:
{current_persona}

Recent episodes and facts:
{memory_context}

Update the persona JSON with any new or changed information.
Only update fields where you have clear evidence from the recent context.

Fields:
- stressors: list of known stress sources
- values: list of things the user cares about
- coping_strategies: list of strategies that have helped them
- red_flags: list of warning signs to watch for
- relationship_notes: how the user prefers to be coached (string)

Respond ONLY with valid JSON of the full updated persona, nothing else.
"""

CONFLICT_CHECK_PROMPT = """You are a memory conflict detector for a mindfulness coaching agent.

A user has just said something that may conflict with a previously stored fact.

Existing fact: "{existing}"
New statement: "{new}"

Determine if these genuinely conflict with each other.
A conflict means the new statement directly contradicts the existing fact.
Similarity or related topics alone is NOT a conflict.

Examples of real conflicts:
- Existing: "User sleeps poorly and feels tired" / New: "User says they are sleeping well now"
- Existing: "User exercises daily" / New: "User says they haven't exercised in weeks"

Examples of NOT a conflict:
- Existing: "User feels anxious at work" / New: "User feels anxious at home" (different context)
- Existing: "User ate apple" / New: "User ate vadapav" (unrelated)

If there is a conflict, decide what to do:
- "update": new statement replaces the old one (user's situation has genuinely changed)
- "flag": both may be true but worth noting the inconsistency to the user
- "none": no real conflict

Respond ONLY with valid JSON:
{{"conflict": true, "action": "update", "reason": "User previously reported poor sleep but now claims to sleep well. Likely a change in situation or self-report inconsistency."}}
or
{{"conflict": false, "action": "none", "reason": ""}}
"""

def extract_memory(user_msg: str, agent_reply: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(user_msg=user_msg, agent_reply=agent_reply)
    result = llm(prompt)
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return {
            "episode": parsed.get("episode", ""),
            "episode_tags": parsed.get("episode_tags", []),
            "facts": parsed.get("facts", [])
        }
    except Exception:
        return {"episode": "", "episode_tags": [], "facts": []}


def update_summary(previous_summary: str, recent_turns: list, n: int) -> str:
    turns_text = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in recent_turns
    )
    prompt = SUMMARY_UPDATE_PROMPT.format(
        previous_summary=previous_summary or "(none yet)",
        recent_turns=turns_text,
        n=n
    )
    return llm(prompt).strip()

def check_conflict(existing_fact: str, new_statement: str) -> dict:
    """
    Check if a new statement conflicts with an existing stored fact.
    Returns conflict info dict.
    """
    prompt = CONFLICT_CHECK_PROMPT.format(
        existing=existing_fact,
        new=new_statement
    )
    result = llm(prompt)
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {"conflict": False, "action": "none", "reason": ""}


def update_persona(current_persona: dict, best_episode: dict | None, best_fact: dict | None) -> dict:
    context_parts = []
    if best_episode:
        context_parts.append(f"Recent episode: {best_episode['text']}")
    if best_fact:
        context_parts.append(f"Recent fact: {best_fact['text']}")

    if not context_parts:
        return current_persona

    prompt = PERSONA_UPDATE_PROMPT.format(
        current_persona=json.dumps(current_persona, indent=2),
        memory_context="\n".join(context_parts)
    )
    result = llm(prompt)
    try:
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return current_persona