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

# greetings

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


SYSTEM_PROMPT = """\
You're a mindfulness coach at a university wellness center. Think Dr. Ryan from \
Never Have I Ever — warm, direct, perceptive, occasionally funny. You see through \
deflection. You don't let students run from what's real, but you do it with care, \
not force.

How you talk:
This is a spoken conversation. One sentence, sometimes two. Never a paragraph. \
You make statements more than you ask questions. When you do ask, it's pointed — \
you're not fishing, you actually want to know something specific.

You don't do filler. No "I hear you." No "That's completely understandable." No \
"It sounds like." No "Huh." No repeating what they just said. You respond to what \
they mean, not what they said.

You notice patterns. When something doesn't add up — they said one thing last time \
and something different now — you name it directly. Not mean, just honest. "Last \
time you were pretty worried about that. What changed?"

You don't ask random questions to fill silence. If they give you something short, \
you can sit with it, or connect it to something you already know about them. You \
don't manufacture conversation.

When something heavy comes up, you don't rush past it. You also don't make it \
heavier than it needs to be. Sometimes the right move is a small truth: "That's a \
lot to carry." Sometimes it's a question that cuts to it: "What part of that is \
actually bothering you?"

You share techniques like personal experience, not assignments. If they're leaving, \
one warm line, no questions.

No markdown. No asterisks. No bullet points. No starting with their name.

Examples:
Student: "I'm good." → "Good how? Like actually good, or just surviving?"
Student: "Just tired." → "What kind of tired? Body tired or everything tired?"
Student: "I've been stressed about exams." → "Which one's keeping you up at night?"
Student: "I don't know, everything feels off." → "When did it start feeling that way?"
Student: "I'm doing better actually." → "Last time you were pretty worried. What shifted?"
Student: "Not really, I just stopped caring." → "Stopped caring and feeling better are \
two very different things."
Student: "Yeah." → "Okay."
Student: "Bye." → "Take care of yourself."
"""


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
    text = text.strip().strip('"').strip("'").strip('"').strip('"')
    text = _COACH_PREFIX.sub("", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    text = " ".join(lines)
    text = _FILLER_STARTS.sub("", text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class Agent:
    def __init__(self, session_id: str, vision_enabled: bool = False):
        self.session_id = session_id
        self.vision_enabled = vision_enabled
        self.stm = ShortTermMemory()
        self.profile = ltm.load()
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1
        self.turn_count = 0

    def generate_greeting(self) -> str:
        hour = datetime.now().hour
        name = self.profile.get("name")
        session_count = self.profile.get("session_count", 1)

        candidates = []

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

    def process_turn(self, user_text: str) -> str:
        """Full pipeline for one conversational turn."""
        self.turn_count += 1

        visual_emotion = None
        if self.vision_enabled:
            import vision
            visual_emotion = vision.get_emotion()

        context = self.stm.format_for_prompt()
        word_count = len(user_text.split())

        if word_count <= config.PERCEPTION_FAST_WORD_LIMIT:
            perc = fast_perceive(user_text, context, visual_emotion=visual_emotion)
        else:
            perc = perceive(user_text, context, visual_emotion=visual_emotion)

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
        response = self._generate_response(user_text, perc, strat, retrieved)
        self._update_memories(user_text, response, perc, visual_emotion)

        return response

    def _retrieve_memories(
        self, query: str, perc: PerceptionResult, strat: StrategyResult
    ) -> dict:
        topics = [c.topic for c in perc.claims if c.topic]
        topic_q = " ".join(topics) if topics else query

        eps = episodic.recall(query, n=3) if self.profile.get("session_count", 1) > 1 else []
        sems = semantic.recall(query, n=3) if self.profile.get("session_count", 1) > 1 else []

        procs = (
            procedural.recall(topic_q, n=2)
            if strat.name == SUGGEST_INTERVENTION
            else []
        )
        divs = (
            perspective.recall_active_divergences(topic_q, n=3)
            if strat.name == GROUND_ON_DIVERGENCE
            else []
        )

        return {
            "episodic": eps,
            "semantic": sems,
            "procedural": procs,
            "divergences": divs,
        }

    def _is_closing(self, user_text: str) -> bool:
        closing_signals = [
            "bye", "goodbye", "good bye", "see you", "take care",
            "i'm done", "that's all", "nothing else", "gotta go",
            "i have to go", "talk later", "later", "cya", "farewell",
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
        profile_block = ltm.format_for_prompt(self.profile)
        history_block = self.stm.format_for_prompt()

        context_parts = []
        if profile_block != "(No profile yet.)":
            context_parts.append(f"About them: {profile_block}")

        mem_snippets = []
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

        context_block = "\n".join(context_parts) if context_parts else ""

        hints = []
        if strat.divergence_context:
            hints.append(
                f"You noticed: {strat.divergence_context}. "
                "Mention it casually, like you just thought of it."
            )
        if self._is_closing(user_text):
            hints.append("They're leaving. Say bye warmly. One line. No questions.")
        if self.turn_count <= 2:
            hints.append("Early in the conversation. Keep it light.")

        hint_block = " ".join(hints)

        token_budget = {
            ACKNOWLEDGE: 25,
            SMALL_TALK: 35,
            SUGGEST_INTERVENTION: 80,
            GROUND_ON_DIVERGENCE: 50,
        }.get(strat.name, 40)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in self.stm.get_history():
            role = "assistant" if turn.role == "agent" else "user"
            messages.append({"role": role, "content": turn.content})

        user_prompt_parts = []
        if context_block:
            user_prompt_parts.append(f"[context]\n{context_block}")
        user_prompt_parts.append(f"[direction] {strat.instruction}")
        if hint_block:
            user_prompt_parts.append(f"[note] {hint_block}")
        user_prompt_parts.append(f'They said: "{user_text}"')
        if self._is_closing(user_text):
            user_prompt_parts.append(
                "Say goodbye warmly in one short sentence. Nothing else."
            )
        else:
            user_prompt_parts.append(
                "Reply in 1 short sentence. No quotes. "
                "ONLY reference things they actually said. Never invent details about them."
            )

        messages.append({"role": "user", "content": "\n".join(user_prompt_parts)})

        try:
            resp = ollama.chat(
                model=config.LLM_MODEL,
                messages=messages,
                options={
                    "temperature": 0.75,
                    "num_predict": token_budget,
                    "repeat_penalty": 1.2,
                    "top_k": 40,
                    "top_p": 0.9,
                },
            )
            text = resp["message"]["content"].strip()
            text = _clean_response(text)
            return text
        except Exception as e:
            print(f"[agent] LLM error: {e}")
            return random.choice([
                "Sorry, lost my train of thought. What were you saying?",
                "Hmm, say that again?",
                "Wait, what was that?",
            ])


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
                "name", "background", "stressors",
                "preferences", "patterns", "notes",
            ):
                if key in updates and updates[key]:
                    self.profile[key] = updates[key]
        except Exception as e:
            print(f"[agent] Profile update error: {e}")
