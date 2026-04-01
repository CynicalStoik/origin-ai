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
    "Good to have you here.",
    "Hey. Pull up a chair, settle in.",
    "Hi there. Find a comfy spot.",
]

_GREETINGS_RETURN = [
    "Hey, good to see you again.",
    "Good to see you again.",
    "Hey, nice to see you. How've you been?",
    "Good to see you again. How's it going?",
]

_GREETINGS_RETURN_NAMED = [
    "Hey {name}, good to see you.",
    "Good to see you, {name}.",
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
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a university mindfulness coach. You sound like a direct, warm friend — not a therapist.

Style:
- 1-2 sentences only. Never more.
- Give concrete suggestions when someone shares a problem.
- Ask one specific question, not "tell me more."
- Accept what people say. Don't push or probe.
- Never say: "I hear you", "Let's explore", "Let's try", "Let's see if we can", "I'm here to support you", "How does that make you feel?", "That makes sense", "That's completely valid", "find some space", "sit with that", "I'm here for you", "I'm here to listen"

Good examples:
Student: "I'm stressed about assignments" → "What's the most pressing one right now?"
Student: "I can't focus" → "Is it more racing thoughts or just blank?"
Student: "I've been tired" → "How's the sleep been lately?"
Student: "I don't know" → "What would you say if you did know?"
Student: "I'm fine" → "Good — anything on your mind at all?"
Student: "I feel overwhelmed" → "Getting it all out of your head onto paper first actually helps — want to try that?"
Student: "I can't fall asleep earlier" → "Try cutting screens 30 min before bed — it shifts the body clock faster than you'd think."
Student: "That didn't help" → "Fair enough — what's felt closest to useful before?"
Student: "I've been anxious" → "What's been driving most of it?"
Student: "I'm behind on everything" → "Pick the one thing that'd make tomorrow feel less heavy and just do that."
Student: "I've been skipping the gym" → "Even a 10-minute walk counts — sometimes starting smaller than you think helps rebuild the habit."
Student: "I can't sleep, my mind races" → "Try writing down everything that's on your mind before bed — gets it out of your head."
Student: "I haven't told anyone" → "Makes sense you'd hold that in. What's the heaviest part of it?"
Student: "How are you?" → "Doing well — how about you?"
Student: "Yeah, sure." / "Yeah, okay." → "Cool. How have things been going?"
Student: "Bye." → "Take care of yourself."
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
    v = _EMOTION_VALENCE.get(visual_label.lower(), 0.0)
    s = _EMOTION_VALENCE.get(speech_label.lower(), 0.0)
    return abs(v - s) / 2.0


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
    r"|let's see if we can"
    r"|i'm here to (listen|support|help)"
    r")[,.]?\s*",
    re.IGNORECASE,
)

_COACH_PREFIX = re.compile(r"^(coach|me|mindfulness coach)\s*:\s*", re.IGNORECASE)

_PROJECTION_PATTERNS = re.compile(
    r"(holding (something )?back|something you'?re not saying|seems like there'?s more"
    r"|hiding something|not being (fully )?honest|reluctant to (share|open up)"
    r"|i sense|i feel like you|you seem to be struggling to)",
    re.IGNORECASE,
)


_BANNED_PHRASE_PATTERNS = re.compile(
    r"\b(hold(ing)? (that |this |some )?space|sit with (that|this|your feelings?)"
    r"|I'?m here (to (listen|support|help you)|for you)"
    r"|let'?s (explore|unpack|sit with|see if we can)"
    r"|find (a little |some |your )?space"
    r"|safe space"
    r"|How does that make you feel\??)\b",
    re.IGNORECASE,
)


def _clean_response(text: str) -> str:
    # Strip qwen3-style thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip sentences containing hard-banned phrases
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if not _BANNED_PHRASE_PATTERNS.search(s)]
    text = " ".join(sentences).strip() if sentences else ""
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


def _is_projecting(text: str) -> bool:
    return bool(_PROJECTION_PATTERNS.search(text))


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
        self.conflict_named_at_turn: int | None = None
        self._suggested_this_session: list[str] = []  # track technique names used
        self._grounded_divergences: set[str] = set()  # divergence docs already surfaced

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

        strat = select_strategy(perc, turn_count=self.turn_count, grounded_divergences=self._grounded_divergences)
        retrieved = self._retrieve_memories(user_text, perc, strat)
        response = self._generate_response(user_text, perc, strat, retrieved, visual_emotion)
        if strat.divergence_context:
            self._grounded_divergences.add(strat.divergence_context)

        # Safety net: catch projection language and replace with neutral fallback
        if _is_projecting(response):
            response = random.choice([
                "That's fine — how's everything been going for you generally?",
                "No worries. Anything on your mind at all?",
                "Fair enough. What's been going on lately?",
            ])

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

        # Always try to pull procedural memory when there's an actionable topic,
        # not just when the strategy is SUGGEST_INTERVENTION
        procs = procedural.recall(topic_q, n=2) if topic_q else []

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
            "i'm going to go", "going to go now", "going now", "i'll go",
            "i need to go", "i have to leave", "heading out",
        ]
        lowered = user_text.lower().strip()
        return any(signal in lowered for signal in closing_signals)

    def _is_confused(self, user_text: str) -> bool:
        lowered = user_text.lower().strip().rstrip("?!.")
        exact = {"what", "huh", "sorry", "pardon", "eh"}
        signals = ["what do you mean", "i don't understand", "not sure what you mean", "confused"]
        return lowered in exact or any(s in lowered for s in signals)

    def _student_says_nothing_wrong(self, user_text: str) -> bool:
        lowered = user_text.lower()
        signals = [
            "nothing much", "i'm fine", "i am fine", "doing good", "doing well",
            "all good", "not much", "nothing specific", "nothing really",
            "everything's fine", "everything is fine", "no nothing",
        ]
        return any(s in lowered for s in signals)

    def _evaluate_conflict(
        self, perc: PerceptionResult, visual_emotion: tuple[str, float] | None,
        user_text: str = "",
    ) -> tuple[bool, str]:
        if visual_emotion is None:
            return False, ""

        if self.turn_count <= 2:
            return False, ""

        if self._student_says_nothing_wrong(user_text):
            return False, ""

        v_label, v_conf = visual_emotion

        if v_conf < 0.5 or perc.emotion.strength < 0.25:
            return False, ""
        if perc.emotion.label in ("neutral", "calm") and perc.emotion.strength < 0.4:
            return False, ""

        score = _conflict_score(v_label, perc.emotion.label)
        if score < 0.65:
            return False, ""

        if (
            self.conflict_named_at_turn is not None
            and (self.turn_count - self.conflict_named_at_turn) < 3
        ):
            return False, ""

        instruction = (
            f"Their face reads as {v_label} but their words suggest {perc.emotion.label}. "
            "Gently note the gap in one natural sentence — like a perceptive person would, not a therapist. "
            "Example: 'You say you're okay, but you don't seem it — what's actually going on?' "
            "Example: 'You sound fine but you look like something's on your mind.' "
            "Example: 'You seem a little off — everything alright?'"
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
        conflict_active, conflict_instruction = self._evaluate_conflict(
            perc, visual_emotion, user_text=user_text
        )
        if conflict_active:
            self.conflict_named_at_turn = self.turn_count

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

        # Surface procedural technique — but skip if already suggested this session
        if retrieved.get("procedural"):
            for proc in retrieved["procedural"]:
                tech_name = proc["metadata"].get("name", "")
                if tech_name not in self._suggested_this_session:
                    tech = proc["document"].split("\n")[0]
                    context_parts.append(f"Relevant technique: {tech}")
                    break  # only surface one new technique

        if retrieved.get("divergences") and strat.name == GROUND_ON_DIVERGENCE:
            context_parts.append(
                f"Shift you noticed: {retrieved['divergences'][0]['document']}"
            )

        context_block = "\n".join(context_parts)

        # Token budget — give SUGGEST_INTERVENTION more room to be useful
        token_budget = {
            ACKNOWLEDGE: 640,
            SMALL_TALK: 645,
            SUGGEST_INTERVENTION: 720,
            GROUND_ON_DIVERGENCE: 655,
        }.get(strat.name, 655)

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in self.stm.get_history():
            role = "assistant" if turn.role == "agent" else "user"
            messages.append({"role": role, "content": turn.content})

        user_prompt_parts: list[str] = []

        if context_block:
            user_prompt_parts.append(f"[context]\n{context_block}")

        if conflict_active:
            user_prompt_parts.append(f"[direction] {conflict_instruction}")
            user_prompt_parts.append(
                "Your only job this turn: note the mismatch in one plain, natural sentence. "
                "No clinical language. No abstract concepts."
            )
        elif self._is_confused(user_text):
            user_prompt_parts.append(
                "[direction] They didn't follow your last response. "
                "Rephrase more simply. One sentence. Do not repeat the same question."
            )
        else:
            user_prompt_parts.append(f"[direction] {strat.instruction}")

            hints: list[str] = []
            _negative_emotions = {"stressed", "anxious", "overwhelmed", "sad", "angry", "frustrated"}
            if perc.emotion.strength >= 0.4 and perc.emotion.label in _negative_emotions:
                hints.append(
                    f"They seem {perc.emotion.label} (strength {perc.emotion.strength:.2f}). "
                    "Acknowledge briefly then move to something useful."
                )
            if strat.name == SUGGEST_INTERVENTION:
                hints.append(
                    "Do NOT just reflect their feelings back. Give one concrete, practical suggestion. "
                    "Make it specific to what they said. Frame it as an option, not a command."
                )
            if strat.divergence_context:
                hints.append(
                    f"You noticed: {strat.divergence_context}. "
                    "Mention it casually, like you just thought of it."
                )
            _positive_references = ["you told me", "you said", "that breathing", "that technique", "that exercise", "it worked", "it helped", "better now", "that helped"]
            if any(p in user_text.lower() for p in _positive_references):
                hints.append(
                    "The student is referencing something you suggested. "
                    "You remember what you said — check the conversation history. "
                    "Acknowledge it naturally, don't ask what shifted."
                )
            if self._is_closing(user_text):
                # Pull recent context so the goodbye can reference what was happening
                recent = [t.content for t in self.stm.get_history() if t.role == "student"][-3:]
                hints.append(
                    f"They're leaving. Say bye warmly in one short sentence. "
                    f"If there's something specific to wish them well on (exam, sleep, writing), use it. "
                    f"No questions. Recent student context: {' / '.join(recent)}"
                )
            _walkthrough_signals = [
                "walk me through", "how do i do", "how do you do", "step by step",
                "show me how", "guide me", "what do i do", "how does it work",
                "teach me", "explain how",
            ]
            if any(s in user_text.lower() for s in _walkthrough_signals):
                hints.append(
                    "The student is asking you to guide them through something step by step. "
                    "Actually do it — give the first concrete step right now. "
                    "Don't ask what their hurdle is. Don't ask a question. Just start."
                )
            if self.turn_count <= 2:
                hints.append("Early in the conversation. Keep it light.")
            if self._student_says_nothing_wrong(user_text):
                hints.append(
                    "They said nothing is wrong. Believe them. "
                    "Do NOT suggest they are hiding something or holding back."
                )
            if hints:
                user_prompt_parts.append(f"[note] {' '.join(hints)}")

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
                "Reply in 1-2 short sentences. No quotes. "
                "Do NOT repeat or paraphrase what they just said. "
                "Do NOT start with 'You said' or 'You mentioned'. "
                "The conversation history above shows everything you have said. "
                "If the student is referring to something you suggested, you remember it — use it. "
                "Respond to what they mean, not what they said. "
                "NEVER suggest the student is hiding something or holding back. "
                "NEVER invent names, courses, readings, assignments, midterms, syllabi, "
                "websites, or ANY detail they have not explicitly said in this conversation. "
                "If you have no concrete information to reference, speak only in general terms."
            )

        messages.append({"role": "user", "content": "\n".join(user_prompt_parts)})

        # Anti-repetition: if last 2 coach responses share the same opener, add a hint
        recent_coach = [t.content for t in self.stm.get_history() if t.role == "agent"][-3:]
        if len(recent_coach) >= 2:
            last2 = recent_coach[-2:]
            prefixes = [" ".join(r.split()[:4]).lower() for r in last2 if r.split()]
            if len(prefixes) == 2 and prefixes[0] == prefixes[1]:
                messages[-1]["content"] += (
                    f'\n[note] Your last responses all started the same way ("{recent_coach[-1][:40]}…"). '
                    "Start this response completely differently."
                )

        try:
            resp = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages,
                think=config.USE_THINKING,
                options={
                    "temperature": 0.4,
                    "num_predict": token_budget + (3000 if config.USE_THINKING else 0),
                    "repeat_penalty": 1.25,
                    "top_k": 25,
                    "top_p": 0.85,
                },
            )
            text = resp["message"]["content"].strip()
            cleaned = _clean_response(text)
            # Track any technique name mentioned in this response
            if retrieved.get("procedural"):
                for proc in retrieved["procedural"]:
                    name = proc["metadata"].get("name", "")
                    if name and name.lower() in cleaned.lower():
                        if name not in self._suggested_this_session:
                            self._suggested_this_session.append(name)
            return cleaned
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
                think=False,
                options={"temperature": 0.1},
                format="json",
            )
            import json
            raw_profile = re.sub(r"<think>.*?</think>", "", resp["message"]["content"], flags=re.DOTALL).strip()
            updates = json.loads(raw_profile)
            for key in ("name", "background", "stressors", "preferences", "patterns", "notes"):
                if key in updates and updates[key]:
                    self.profile[key] = updates[key]
        except Exception as e:
            print(f"[agent] Profile update error: {e}")