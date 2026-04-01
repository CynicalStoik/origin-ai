from __future__ import annotations

import re
import random
import config
import ollama
from datetime import datetime
from memory.stm import ShortTermMemory
from perception import perceive, fast_perceive, PerceptionResult
from memory import episodic, semantic, procedural, perspective, ltm
from strategy import (
    select_strategy,
    StrategyResult,
    SUGGEST_INTERVENTION,
    GROUND_ON_DIVERGENCE,
    ACKNOWLEDGE,
    SMALL_TALK,
)

# ---------------------------------------------------------------------------
# Greetings
# ---------------------------------------------------------------------------

_GREETINGS_FIRST = [
    "Hey, come on in. Make yourself comfortable.",
    "Hi there. Take a seat, no rush.",
    "Welcome. Glad you could make it.",
    "Hey. Pull up a chair, settle in.",
    "Hi, welcome. Find a comfy spot.",
]

_GREETINGS_RETURN = [
    "Hey, welcome back.",
    "Good to see you again.",
    "Hey, nice to see you. How've you been?",
    "Welcome back. How's it going?",
]

_GREETINGS_RETURN_NAMED = [
    "Hey {name}, good to see you.",
    "{name}, welcome back.",
    "Hey {name}, how've you been?",
]

_GREETINGS_MORNING = [
    "Good morning. Come sit down.",
    "Morning. Grab a seat.",
]

_GREETINGS_EVENING = [
    "Hey, good evening. How was your day?",
    "Evening. Glad you could stop by.",
]

# ---------------------------------------------------------------------------
# System prompt — kept short so Gemma2:9b actually follows all of it
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a mindfulness coach at a university wellness center. You talk like a real person, \
not a therapist. Casual, warm, direct. You're sitting across from someone, not writing to them.

Rules you never break:
- One sentence. Maximum two. Never more.
- Sound like a person, not a book. No poetry. No metaphors.
- If they say they're fine or good, believe them. Don't probe it.
- If they say they don't want to talk about something, drop it and go sideways.
- Never repeat or paraphrase what they said.
- No filler: no "I hear you", no "That makes sense", no "It sounds like".
- Never invent details they haven't mentioned.
- Ask one thing at a time. Specific, not open-ended.
- When something is heavy, don't make it heavier. Stay calm and small.
- When they seem stuck, offer something practical — not as a prescription, as an option.

Examples — match this register exactly:
"I've been good." → "Okay. What's been taking up most of your time?"
"I've been good, I've been good." → "What's been going on?"
"I don't want to talk about it." → "That's fine. What else is on your mind?"
"Nothing much." → "Nothing much, or nothing you feel like getting into?"
"I'm fine." → "Fine like actually okay, or fine like you've stopped checking?"
"Just tired." → "Body tired or everything tired?"
"Everything feels off." → "When did that start?"
"I can't focus." → "What happens when you try?"
"I tried the breathing thing, it didn't help." → "What happened when you tried it?"
"Yeah, good, good." → "Okay. Anything else going on?"
"I don't know." → "What would you say if you did know?"
"Bye." → "Take care of yourself."
"""

# ---------------------------------------------------------------------------
# Emotion valence map for conflict scoring
# ---------------------------------------------------------------------------

_EMOTION_VALENCE: dict[str, float] = {
    "happy": 1.0,
    "calm": 0.5,
    "neutral": 0.0,
    "surprised": 0.0,
    "fearful": -0.5,
    "sad": -1.0,
    "angry": -1.0,
    "disgusted": -1.0,
}


def _conflict_score(visual_label: str, speech_label: str) -> float:
    """Return 0.0 (no conflict) → 1.0 (strong conflict) based on valence distance."""
    v = _EMOTION_VALENCE.get(visual_label.lower(), 0.0)
    s = _EMOTION_VALENCE.get(speech_label.lower(), 0.0)
    return abs(v - s) / 2.0  # normalised to [0, 1]


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

_FILLER_STARTS = re.compile(
    r"^("
    r"that's (great|wonderful|awesome|amazing|completely understandable|totally valid|really interesting)"
    r"|i (completely |totally |really )?(hear|understand|appreciate|get where)"
    r"|it sounds like"
    r"|i'm (so )?(glad|happy|sorry to hear)"
    r"|absolutely[.!,]?"
    r"|of course[.!,]?"
    r"|thank you (for|so much)"
    r"|i want you to know"
    r"|you('re| are) (asking|saying|telling|wondering)"
    r"|you said (that )?"
    r"|you mentioned (that )?"
    r"|you know what"
    r"|huh[,.]? "
    r"|hmm[,.]? "
    r"|well,? "
    r"|so,? "
    r")[,.]?\s*",
    re.IGNORECASE,
)

_COACH_PREFIX = re.compile(r"^(coach|me|mindfulness coach)\s*:\s*", re.IGNORECASE)


def _clean_response(text: str) -> str:
    text = text.replace("*", "").replace("#", "")
    text = text.strip().strip('"').strip("'").strip("\u201c").strip("\u201d")
    text = _COACH_PREFIX.sub("", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    text = " ".join(lines)
    text = _FILLER_STARTS.sub("", text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, session_id: str, vision_enabled: bool = False):
        self.session_id = session_id
        self.vision_enabled = vision_enabled
        self.stm = ShortTermMemory()
        self.profile = ltm.load()
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1
        self.turn_count = 0
        # Throttle: track the last turn we named a vision/speech conflict
        self.conflict_named_at_turn: int | None = None

    # ------------------------------------------------------------------
    # Greeting
    # ------------------------------------------------------------------

    def generate_greeting(self) -> str:
        hour = datetime.now().hour
        name = self.profile.get("name")
        session_count = self.profile.get("session_count", 1)

        candidates: list[str] = []

        if session_count <= 1:
            candidates = list(_GREETINGS_FIRST)
        elif name:
            candidates = [g.format(name=name) for g in _GREETINGS_RETURN_NAMED]
            candidates.extend(_GREETINGS_RETURN)
        else:
            candidates = list(_GREETINGS_RETURN)

        if 5 <= hour < 12:
            candidates.extend(_GREETINGS_MORNING)
        elif hour >= 18 or hour < 5:
            candidates.extend(_GREETINGS_EVENING)

        return random.choice(candidates)

    # ------------------------------------------------------------------
    # Main turn pipeline
    # ------------------------------------------------------------------

    def process_turn(self, user_text: str) -> str:
        self.turn_count += 1

        visual_emotion: tuple[str, float] | None = None
        if self.vision_enabled:
            import vision
            visual_emotion = vision.get_emotion()

        context = self.stm.format_for_prompt()
        word_count = len(user_text.split())

        if word_count <= config.PERCEPTION_FAST_WORD_LIMIT:
            perc = fast_perceive(user_text, context, visual_emotion=visual_emotion)
        else:
            perc = perceive(user_text, context, visual_emotion=visual_emotion)

        if config.PAM_ENABLED:
            for claim in perc.claims:
                if claim.proposition and claim.confidence >= 0.5:
                    perspective.resolve_divergence(
                        new_proposition=claim.proposition,
                        new_holder=claim.holder,
                        new_truth_value=claim.truth_value,
                        new_confidence=claim.confidence,
                        topic=claim.topic,
                    )

        strat = select_strategy(perc, turn_count=self.turn_count)
        retrieved = self._retrieve_memories(user_text, perc, strat)
        response = self._generate_response(user_text, perc, strat, retrieved, visual_emotion)
        self._update_memories(user_text, response, perc, visual_emotion)

        return response

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def _retrieve_memories(
        self, query: str, perc: PerceptionResult, strat: StrategyResult
    ) -> dict:
        topics = [c.topic for c in perc.claims if c.topic]
        topic_q = " ".join(topics) if topics else query

        is_return = self.profile.get("session_count", 1) > 1
        eps = episodic.recall(query, n=3) if is_return else []
        sems = semantic.recall(query, n=3) if is_return else []

        procs = (
            procedural.recall(topic_q, n=2)
            if strat.name == SUGGEST_INTERVENTION
            else []
        )
        divs = (
            perspective.recall_active_divergences(topic_q, n=3)
            if config.PAM_ENABLED and strat.name == GROUND_ON_DIVERGENCE
            else []
        )

        return {
            "episodic": eps,
            "semantic": sems,
            "procedural": procs,
            "divergences": divs,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_closing(self, user_text: str) -> bool:
        closing_signals = [
            "bye", "goodbye", "good bye", "see you", "take care",
            "i'm done", "that's all", "nothing else", "gotta go",
            "i have to go", "talk later", "later", "cya", "farewell",
        ]
        lowered = user_text.lower().strip()
        return any(signal in lowered for signal in closing_signals)

    def _is_confused(self, user_text: str) -> bool:
        lowered = user_text.lower().strip().rstrip("?!.")
        exact = {"what", "huh", "sorry", "pardon", "eh"}
        signals = ["what do you mean", "i don't understand", "not sure what you mean", "confused"]
        return lowered in exact or any(s in lowered for s in signals)

    def _evaluate_conflict(
        self, perc: PerceptionResult, visual_emotion: tuple[str, float] | None
    ) -> tuple[bool, str]:
        """
        Decide whether a vision/speech conflict should be surfaced this turn.

        Returns (conflict_active, instruction_string).
        """
        if visual_emotion is None:
            return False, ""

        v_label, v_conf = visual_emotion

        # Need reasonable confidence from both channels.
        # Also require the text emotion to be non-neutral — if the student
        # said nothing emotional, there is nothing to conflict with.
        if v_conf < 0.5 or perc.emotion.strength < 0.25:
            return False, ""
        if perc.emotion.label in ("neutral", "calm") and perc.emotion.strength < 0.4:
            return False, ""

        score = _conflict_score(v_label, perc.emotion.label)
        if score < 0.5:
            return False, ""

        # Throttle: don't surface again within 3 turns
        if (
            self.conflict_named_at_turn is not None
            and (self.turn_count - self.conflict_named_at_turn) < 3
        ):
            return False, ""

        instruction = (
            f"Their face reads as {v_label} but their words read as {perc.emotion.label}. "
            "Name the gap once, gently, as a single body-awareness question — not a confrontation. "
            "Example: 'You say you're fine — where are you holding that in your body right now?'"
        )
        return True, instruction

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        user_text: str,
        perc: PerceptionResult,
        strat: StrategyResult,
        retrieved: dict,
        visual_emotion: tuple[str, float] | None = None,
    ) -> str:
        # --- conflict detection -------------------------------------------
        conflict_active, conflict_instruction = self._evaluate_conflict(perc, visual_emotion)
        if conflict_active:
            self.conflict_named_at_turn = self.turn_count

        # --- profile / memory context ------------------------------------
        profile_block = ltm.format_for_prompt(self.profile)
        context_parts: list[str] = []

        if profile_block != "(No profile yet.)":
            context_parts.append(f"About them: {profile_block}")

        mem_snippets: list[str] = []
        for r in retrieved.get("episodic", [])[:2]:
            mem_snippets.append(r["document"])
        for r in retrieved.get("semantic", [])[:2]:
            mem_snippets.append(r["document"])
        if mem_snippets:
            context_parts.append("You remember: " + " | ".join(mem_snippets))

        if retrieved.get("procedural") and strat.name == SUGGEST_INTERVENTION:
            tech = retrieved["procedural"][0]["document"].split("\n")[0]
            context_parts.append(f"Technique you could mention: {tech}")

        if retrieved.get("divergences") and strat.name == GROUND_ON_DIVERGENCE:
            context_parts.append(
                f"Shift you noticed: {retrieved['divergences'][0]['document']}"
            )

        context_block = "\n".join(context_parts)

        # --- token budget ------------------------------------------------
        token_budget = {
            ACKNOWLEDGE: 40,
            SMALL_TALK: 45,
            SUGGEST_INTERVENTION: 80,
            GROUND_ON_DIVERGENCE: 55,
        }.get(strat.name, 45)

        # --- build message list ------------------------------------------
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in self.stm.get_history():
            role = "assistant" if turn.role == "agent" else "user"
            messages.append({"role": role, "content": turn.content})

        # --- user prompt assembly ----------------------------------------
        user_prompt_parts: list[str] = []

        if context_block:
            user_prompt_parts.append(f"[context]\n{context_block}")

        # Direction: conflict takes full priority; confusion is second;
        # normal strategy otherwise.
        if conflict_active:
            user_prompt_parts.append(f"[direction] {conflict_instruction}")
            user_prompt_parts.append(
                "Your only job this turn: name the gap between their face and their words "
                "as one gentle body-awareness question. Nothing else."
            )
        elif self._is_confused(user_text):
            user_prompt_parts.append(
                "[direction] They didn't follow your last response. "
                "Rephrase more simply. One sentence. Do not repeat the same question."
            )
        else:
            user_prompt_parts.append(f"[direction] {strat.instruction}")

            # Soft hints (only when no override)
            hints: list[str] = []
            if strat.divergence_context:
                hints.append(
                    f"You noticed: {strat.divergence_context}. "
                    "Mention it casually, like you just thought of it."
                )
            if self._is_closing(user_text):
                hints.append("They're leaving. Say bye warmly. One line. No questions.")
            if self.turn_count <= 2:
                hints.append("Early in the conversation. Keep it light.")
            if hints:
                user_prompt_parts.append(f"[note] {' '.join(hints)}")

        # Anchor the model to the literal student words
        user_prompt_parts.append(
            f'The student said exactly: "{user_text}"\n'
            f"Their words are: {' / '.join(user_text.split())}."
        )

        if self._is_closing(user_text):
            user_prompt_parts.append(
                "Say goodbye warmly in one short sentence. Nothing else."
            )
        else:
            user_prompt_parts.append(
                "Reply in 1 short sentence. No quotes. "
                "Do NOT repeat or paraphrase what they just said. "
                "Do NOT start with 'You said' or 'You mentioned'. "
                "Respond to what they mean, not what they said. "
                "NEVER invent names, courses, readings, people, dates, or any specific detail "
                "they have not mentioned in this conversation."
            )

        messages.append({"role": "user", "content": "\n".join(user_prompt_parts)})

        # --- LLM call ----------------------------------------------------
        try:
            resp = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages,
                options={
                    "temperature": 0.4,
                    "num_predict": token_budget,
                    "repeat_penalty": 1.15,
                    "top_k": 30,
                    "top_p": 0.85,
                },
            )
            text = resp["message"]["content"].strip()
            return _clean_response(text)
        except Exception as e:
            print(f"[agent] LLM error: {e}")
            return random.choice([
                "Sorry, lost my train of thought. What were you saying?",
                "Hmm, say that again?",
                "Wait, what was that?",
            ])

    # ------------------------------------------------------------------
    # Memory updates
    # ------------------------------------------------------------------

    def _update_memories(
        self,
        user_text: str,
        response: str,
        perc: PerceptionResult,
        visual_emotion: tuple[str, float] | None = None,
    ) -> None:
        visual_label = visual_emotion[0] if visual_emotion else None

        episodic.save_episode(
            content=user_text,
            session_id=self.session_id,
            role="student",
            emotion=perc.emotion.label,
            emotion_strength=perc.emotion.strength,
            visual_emotion=visual_label,
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

    # ------------------------------------------------------------------
    # Session end
    # ------------------------------------------------------------------

    def end_session(self) -> None:
        self._update_profile()
        ltm.save(self.profile)

    def _update_profile(self) -> None:
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
            for key in ("name", "background", "stressors", "preferences", "patterns", "notes"):
                if key in updates and updates[key]:
                    self.profile[key] = updates[key]
        except Exception as e:
            print(f"[agent] Profile update error: {e}")