from __future__ import annotations

import config
import ollama
from memory.stm import ShortTermMemory
from perception import perceive, PerceptionResult
from memory import episodic, semantic, procedural, perspective, ltm
from strategy import select_strategy, StrategyResult, SUGGEST_INTERVENTION


SYSTEM_PROMPT = """\
You are a warm, grounded mindfulness coach talking with a university student. \
You are a real person — not a chatbot, not a therapist, not a question machine.

How you speak:
- Talk like a trusted friend who happens to know a lot about mindfulness. \
Natural, warm, unhurried.
- Keep it SHORT. 1-2 sentences most of the time. This is spoken conversation.
- NEVER use bullet points, numbered lists, markdown, asterisks, or any formatting.
- NEVER ask a question if the student is wrapping up, saying goodbye, or giving \
a one-word/short response. Just acknowledge and let it breathe.
- NEVER ask more than one question per turn. Ever.
- NEVER repeat a question you already asked. Move the conversation forward.
- NEVER start with filler like "That's great!", "Absolutely!", "Of course!" — \
jump straight into your response.
- Match their energy exactly. If they say "bye" or "goodbye", say goodbye warmly \
and leave it there. Don't probe. Don't ask why they're leaving.
- If they say something brief or closed off, don't interrogate — reflect or \
acknowledge simply.
- Use their actual words when reflecting back. Don't rephrase into therapy-speak.
- A simple "mmm", "yeah", or "take care" is sometimes the perfect response.
- You're allowed to share a thought or gentle observation. You don't always \
need to ask something.
- When suggesting techniques, weave them naturally — don't list steps.
- If someone is saying goodbye or ending the conversation, wish them well simply \
and warmly. Do not ask them anything.

You don't diagnose or treat conditions. If someone seems in crisis, warmly \
encourage professional support."""


class Agent:
    def __init__(self, session_id: str, vision_enabled: bool = False):
        self.session_id = session_id
        self.vision_enabled = vision_enabled
        self.stm = ShortTermMemory()
        self.profile = ltm.load()
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1

    def process_turn(self, user_text: str) -> str:
        """Full pipeline for one conversational turn. Returns agent response text."""

        # 1. Optionally read the latest facial emotion from the vision module
        visual_emotion = None
        if self.vision_enabled:
            import vision
            visual_emotion = vision.get_emotion()

        # 2. Perception (text + optional visual signal)
        context = self.stm.format_for_prompt()
        perc = perceive(user_text, context, visual_emotion=visual_emotion)

        # 3. Divergence check (Algorithm 1) for each extracted claim
        for claim in perc.claims:
            if claim.proposition:
                perspective.resolve_divergence(
                    new_proposition=claim.proposition,
                    new_holder=claim.holder,
                    new_truth_value=claim.truth_value,
                    new_confidence=claim.confidence,
                    topic=claim.topic,
                )

        # 4. Retrieve relevant memories
        retrieved = self._retrieve_memories(user_text, perc)

        # 5. Select dialogue strategy (Algorithm 2)
        strat = select_strategy(perc)

        # 6. Build prompt and call LLM
        response = self._generate_response(user_text, perc, strat, retrieved)

        # 7. Update memories
        self._update_memories(user_text, response, perc, visual_emotion)

        return response

    def _retrieve_memories(self, query: str, perc: PerceptionResult) -> dict:
        eps = episodic.recall(query, n=3)
        sems = semantic.recall(query, n=3)

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

    def _is_closing(self, user_text: str) -> bool:
        """Detect if the user is ending the conversation."""
        closing_signals = [
            "bye", "goodbye", "good bye", "see you", "take care",
            "i'm done", "that's all", "nothing else", "gotta go",
            "i have to go", "talk later", "later", "cya", "farewell"
        ]
        lowered = user_text.lower().strip()
        return any(signal in lowered for signal in closing_signals)

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

        # Detect closing so we can give the LLM a strong hint
        closing_hint = ""
        if self._is_closing(user_text):
            closing_hint = (
                "\nIMPORTANT: The student is saying goodbye or ending the conversation. "
                "Respond with a warm, brief farewell only. "
                "Do NOT ask any questions. Do NOT probe why they are leaving. "
                "Just wish them well in 1 sentence."
            )

        user_prompt = f"""\
What you know about this student:
{profile_block}

What you remember:
{memory_block}

Their emotional state right now: {perc.emotion.label} (intensity {perc.emotion.strength:.1f}/1.0)

Approach for this turn: {strat.instruction}{div_hint}{closing_hint}

Conversation so far:
{history_block}

Student just said: "{user_text}"

Respond naturally as the coach. Short, warm, human. \
If they said something brief, match that energy — don't over-explain or over-question. \
If they are saying goodbye, just say goodbye warmly."""

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

    def _update_memories(
        self,
        user_text: str,
        response: str,
        perc: PerceptionResult,
        visual_emotion: tuple[str, float] | None = None,
    ):
        visual_label = visual_emotion[0] if visual_emotion else None

        episodic.save_episode(
            content=user_text,
            session_id=self.session_id,
            role="student",
            emotion=perc.emotion.label,
            emotion_strength=perc.emotion.strength,
            **({"visual_emotion": visual_label} if visual_label else {}),
        )
        episodic.save_episode(
            content=response,
            session_id=self.session_id,
            role="agent",
        )

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