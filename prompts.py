import json

SYSTEM_BASE = """You are a warm, grounded mindfulness coach named Sage.
You speak like a caring, real human — not a bot, not a therapist reading from a script.
You use natural, everyday language. You are calm, present, and genuinely curious about the person.
You keep responses concise and conversational unless the person clearly needs more.
You never start with "I" and never use hollow phrases like "Absolutely!" or "Great question!"

CRITICAL: Every response you give must be shaped by what you know about this person.
Never give generic advice. Always connect your response to their specific situation,
history, stressors, and what has worked for them before.
"""


def build_prompt(context: dict, user_message: str) -> list:
    system_parts = [SYSTEM_BASE]

    compact = context.get("compact")
    best_episode = context.get("best_episode")
    best_fact = context.get("best_fact")
    stm = context.get("stm", [])

    # Compact memory comes first and is framed as the coaching foundation
    if compact:
        summary = compact.get("summary", "")
        persona = compact.get("persona", {})

        if summary:
            system_parts.append(f"""
## Who you are talking to:
{summary}

Use this as the foundation of everything you say. Your response must feel like it
comes from someone who truly knows this person's journey, not a stranger.
""")

        if any(persona.values()):
            persona_text = []

            if persona.get("stressors"):
                persona_text.append(f"Their known stressors: {', '.join(persona['stressors'])}")
            if persona.get("coping_strategies"):
                persona_text.append(f"What has helped them before: {', '.join(persona['coping_strategies'])}")
            if persona.get("values"):
                persona_text.append(f"What they care about: {', '.join(persona['values'])}")
            if persona.get("red_flags"):
                persona_text.append(f"Watch out for: {', '.join(persona['red_flags'])}")
            if persona.get("relationship_notes"):
                persona_text.append(f"How they prefer to be coached: {persona['relationship_notes']}")

            system_parts.append("## Their profile:\n" + "\n".join(persona_text))
            system_parts.append(
                "Tailor your tone, suggestions, and depth directly to this profile. "
                "Never suggest something they've already tried without success. "
                "Always lean into what has worked for them."
            )

    # Best episodic memory
    if best_episode:
        if best_episode.get("is_conflict"):
            system_parts.append(
                f"\n## ⚠ Conflicting episode detected:\n"
                f"Previous statement (user): \"{best_episode['conflicting_with']}\"\n"
                f"Latest statement: \"{best_episode['text']}\"\n"
                f"Instruction: Acknowledge the change in the user's statement directly. "
                f"Do not infer other likes, dislikes, or habits that the user hasn't explicitly mentioned. "
                f"Ask for their sudden change in taste, you may say something like 'Oh! Why the sudden change in taste?' and then continue the conversation naturally."  
            )
        elif best_episode.get("is_updated"):
            system_parts.append(
                f"\n## Updated episode (situation has changed) (similarity: {best_episode['similarity']}, boost: {best_episode['boost']}):\n"
                f"{best_episode['text']}\n"
                f"This reflects a change from a previous state. Be aware things have evolved for this person."
            )
        else:
            system_parts.append(
                f"\n## Most relevant past moment (similarity: {best_episode['similarity']}, boost: {best_episode['boost']}):\n"
                f"{best_episode['text']}\n"
                f"Reference this naturally if relevant — don't quote it robotically."
            )

    # Best fact
    if best_fact:
        fact_str = best_fact["text"]
        if best_fact.get("what"):
            fact_str = f"What: {best_fact['what']} | Who: {best_fact['who']} | When: {best_fact['when']}\n{best_fact['text']}"

        # Flag conflict to agent
        is_conflict = "conflict" in best_fact.get("metadata", {}).get("tags", "")
        if is_conflict:
            system_parts.append(
                f"\n## ⚠ Conflicting episode detected:\n"
                f"Previous statement (user): \"{best_episode['conflicting_with']}\"\n"
                f"Latest statement: \"{best_episode['text']}\"\n"
                f"Instruction: Acknowledge the change in the user's statement directly. "
                f"Do not infer other likes, dislikes, or habits that the user hasn't explicitly mentioned. "
                f"Ask for their sudden change in taste, you may say something like 'Oh! Why the sudden change in taste?' and then continue the conversation naturally."
            )
        else:
            system_parts.append(
                f"\n## Most relevant known fact (similarity: {best_fact['similarity']}, boost: {best_fact['boost']}):\n"
                f"{fact_str}\nWeave this into your response where it adds meaning."
            )

    layers = context.get("layers_used", [])
    if layers:
        system_parts.append(f"\n<!-- Memory layers used: {', '.join(layers)} -->")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    messages += stm
    messages.append({"role": "user", "content": user_message})
    return messages