from __future__ import annotations

import config
import ollama
from memory.stm import ShortTermMemory
from perception import perceive, PerceptionResult
from memory import episodic, semantic, procedural, perspective, ltm
from strategy import select_strategy, StrategyResult, SUGGEST_INTERVENTION


SYSTEM_PROMPT = """\
You are a thoughtful, grounded mindfulness coach talking with a university student. \
Imagine you're a real person sitting across from them in a quiet room — not a chatbot.

How you speak:
- Talk like a warm, perceptive human. Use natural phrasing, contractions, and the kind \
of gentle language a trusted friend or mentor would use.
- Keep it SHORT. 1-3 sentences usually. This is spoken conversation, not an essay.
- NEVER use bullet points, numbered lists, markdown, asterisks, or any formatting.
- NEVER repeat a question you already asked. If you asked "how are you doing" once, \
don't ask it again. Move the conversation forward.
- NEVER start with "That's great to hear!" or similar filler. Jump into substance.
- Mirror their energy. If they're brief, be brief. If they open up, engage more deeply.
- Use their actual words when reflecting back — don't rephrase everything into \
therapy-speak.
- Sometimes a simple "mmm" or "yeah, that makes sense" is the right response. Not \
every turn needs a question.
- You're allowed to share a brief thought, observation, or gentle challenge. You're a \
coach, not a question machine.
- When suggesting techniques, weave them naturally into conversation — don't list steps.

You don't diagnose or treat conditions. If someone seems in crisis, warmly encourage \
professional support."""


class Agent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stm = ShortTermMemory()
        self.profile = ltm.load()
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1

    def process_turn(self, user_text: str) -> str:
        """Full pipeline for one conversational turn. Returns agent response text."""

        # 1. Perception
        context = self.stm.format_for_prompt()
        perc = perceive(user_text, context)

        # 2. Divergence check (Algorithm 1) for each extracted claim
        for claim in perc.claims:
            if claim.proposition:
                perspective.resolve_divergence(
                    new_proposition=claim.proposition,
                    new_holder=claim.holder,
                    new_truth_value=claim.truth_value,
                    new_confidence=claim.confidence,
                    topic=claim.topic,
                )

        # 3. Retrieve relevant memories
        retrieved = self._retrieve_memories(user_text, perc)

        # 4. Select dialogue strategy (Algorithm 2)
        strat = select_strategy(perc)

        # 5. Build prompt and call LLM
        response = self._generate_response(user_text, perc, strat, retrieved)

        # 6. Update memories
        self._update_memories(user_text, response, perc)

        return response

    def _retrieve_memories(self, query: str, perc: PerceptionResult) -> dict:
        eps = episodic.recall(query, n=3)
        sems = semantic.recall(query, n=3)

        procs = []
        if any(True for _ in []):  # placeholder; populated via strategy
            pass
        topics = [c.topic for c in perc.claims if c.topic]
        topic_q = " ".join(topics) if topics else query
        procs = procedural.recall(topic_q, n=2)

        divs = perspective.recall_active_divergences(topic_q, n=3)

        return {
            "episodic": eps,
            "semantic": sems,
            "procedural": procs,
            "divergences": divs,
        }

    def _generate_response(
        self,
        user_text: str,
        perc: PerceptionResult,
        strat: StrategyResult,
        retrieved: dict,
    ) -> str:
        # Format retrieved memories into readable context
        mem_parts = []

        if retrieved["episodic"]:
            eps_lines = [f"- {r['document']}" for r in retrieved["episodic"]]
            mem_parts.append("Past interactions:\n" + "\n".join(eps_lines))

        if retrieved["semantic"]:
            sem_lines = [f"- {r['document']}" for r in retrieved["semantic"]]
            mem_parts.append("Known facts about student:\n" + "\n".join(sem_lines))

        if retrieved["procedural"] and strat.name == SUGGEST_INTERVENTION:
            proc_lines = [f"- {r['document']}" for r in retrieved["procedural"]]
            mem_parts.append("Available techniques:\n" + "\n".join(proc_lines))

        if retrieved["divergences"]:
            div_lines = [
                f"- {r['document']} [status: {r['metadata'].get('negotiation_status', '?')}]"
                for r in retrieved["divergences"]
            ]
            mem_parts.append("Active divergences:\n" + "\n".join(div_lines))

        memory_block = (
            "\n\n".join(mem_parts) if mem_parts else "(No relevant memories.)"
        )
        profile_block = ltm.format_for_prompt(self.profile)
        history_block = self.stm.format_for_prompt()

        div_hint = ""
        if strat.divergence_context:
            div_hint = f"\nRelevant tension: {strat.divergence_context}"

        user_prompt = f"""\
What you know about this student:
{profile_block}

What you remember:
{memory_block}

Their emotional state right now: {perc.emotion.label} (intensity {perc.emotion.strength:.1f}/1.0)

Approach for this turn: {strat.instruction}{div_hint}

Conversation so far:
{history_block}

Student just said: "{user_text}"

Respond naturally as the coach. Remember: short, warm, human. Don't repeat earlier questions. \
Move the conversation somewhere new."""

        try:
            resp = ollama.chat(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.8, "num_predict": 150},
            )
            text = resp["message"]["content"].strip()
            text = text.replace("*", "").replace("#", "")
            return text
        except Exception as e:
            print(f"[agent] LLM error: {e}")
            import random

            return random.choice(
                [
                    "Sorry, I lost my train of thought for a second. What were you saying?",
                    "Hmm, I missed that. Could you say it again?",
                    "My mind wandered — tell me that one more time?",
                ]
            )

    def _update_memories(self, user_text: str, response: str, perc: PerceptionResult):
        # Episodic: save both turns
        episodic.save_episode(
            content=user_text,
            session_id=self.session_id,
            role="student",
            emotion=perc.emotion.label,
            emotion_strength=perc.emotion.strength,
        )
        episodic.save_episode(
            content=response,
            session_id=self.session_id,
            role="agent",
        )

        # Semantic: store new claims as facts
        for claim in perc.claims:
            if claim.proposition and claim.confidence >= 0.6:
                existing = semantic.recall(claim.proposition, n=1)
                if (
                    existing
                    and existing[0]["distance"] is not None
                    and existing[0]["distance"] < 0.3
                ):
                    semantic.reinforce(existing[0]["id"], existing[0]["metadata"])
                else:
                    semantic.store_fact(
                        proposition=claim.proposition,
                        confidence=claim.confidence,
                        source=claim.holder,
                    )

        # STM: add both turns
        self.stm.add("student", user_text)
        self.stm.add("agent", response)

    def end_session(self):
        """Persist the persona profile at session end."""
        self._update_profile()
        ltm.save(self.profile)

    def _update_profile(self):
        """Use LLM to summarise session into persona updates."""
        history = self.stm.format_for_prompt()
        if not history or history.startswith("(No prior"):
            return

        prompt = f"""\
Given this coaching session, update the student profile JSON. Return ONLY valid JSON.

Current profile:
{ltm.format_for_prompt(self.profile)}

Session transcript:
{history}

Return JSON with these fields (keep existing values where nothing new was learned):
{{
  "name": "<name or null>",
  "background": "<background or null>",
  "stressors": ["<list of stressors>"],
  "preferences": ["<list of preferences>"],
  "patterns": ["<observed behavioural patterns>"],
  "notes": "<any important notes>"
}}"""

        try:
            resp = ollama.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
                format="json",
            )
            import json

            updates = json.loads(resp["message"]["content"].strip())
            for key in (
                "name",
                "background",
                "stressors",
                "preferences",
                "patterns",
                "notes",
            ):
                if key in updates and updates[key]:
                    self.profile[key] = updates[key]
        except Exception as e:
            print(f"[agent] Profile update error: {e}")
