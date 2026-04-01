from __future__ import annotations

import random
import config
from dataclasses import dataclass
from perception import PerceptionResult
from memory import perspective as psp_mem

GROUND_ON_DIVERGENCE = "ground_on_divergence"
SUGGEST_INTERVENTION = "suggest_intervention"
ELICIT_CLARIFICATION = "elicit_clarification"
REFLECT_AND_VALIDATE = "reflect_and_validate"
MATCH_AND_CONTINUE = "match_and_continue"
ACKNOWLEDGE = "acknowledge"
SMALL_TALK = "small_talk"

STRATEGY_INSTRUCTIONS = {
    GROUND_ON_DIVERGENCE: (
        "You noticed something shifted from what they said before. "
        "Name it once, casually, like you just thought of it. "
        "One line. Don't explain it or make a case. "
        "Example: 'Earlier you said things were fine — this sounds different.'"
    ),
    SUGGEST_INTERVENTION: (
        "They've shared a real problem. Don't just reflect it back — offer something useful. "
        "Suggest one concrete, practical thing they could try. Keep it short and specific. "
        "Frame it as an option, not a prescription. "
        "Example: student says 'I have a lot of assignments' → "
        "'One thing that helps is writing everything down and picking just one to start — takes the edge off.' "
        "Example: student says 'I've been stressed' → "
        "'Have you tried breaking your day into smaller chunks? Even 25-minute focused blocks can help.' "
        "Example: student says 'I feel overwhelmed' → "
        "'Sometimes just getting the tasks out of your head and onto paper helps — want to try that?'"
    ),
    ELICIT_CLARIFICATION: (
        "Something was half-said or unclear. Ask the one specific thing you want to know. "
        "Not 'tell me more'. One direct question. "
        "Example: 'What do you mean by that?'"
    ),
    REFLECT_AND_VALIDATE: (
        "They said something heavy or real. Don't fix it, don't minimise it. "
        "Say one short thing that shows you heard them. "
        "Example: student says 'I've been really struggling' → 'That's a lot to carry.'"
    ),
    MATCH_AND_CONTINUE: (
        "Conversation is moving. React directly to what they said — "
        "a statement or one specific question. Keep it grounded in their words. "
        "Example: student says 'I've been busy' → 'What's been taking most of it?'"
    ),
    ACKNOWLEDGE: (
        "The student said something brief, vague, or seems unsure what to talk about. "
        "If they seem to not know what to say or are asking what to talk about, "
        "warmly introduce yourself and let them know what you are here for. "
        "You are a mindfulness agent — you are here to support their wellbeing, "
        "help them reflect, manage stress, and have an open conversation. "
        "Keep it warm, brief, and inviting — one or two sentences max. "
        "Example: student says 'anything I need to talk about?' → "
        "'I'm a mindfulness coach here to support you — we can talk about anything "
        "on your mind, whether that's stress, how you're feeling, or just how your day went.' "
        "Example: student says 'yeah' → 'Yeah, go ahead — what's on your mind?' "
        "Example: student says 'I don't know' → "
        "'That's okay — we can start simple. How have you been feeling lately?'"
    ),
    SMALL_TALK: (
        "Nothing heavy yet. Ask one genuine, specific question about their life. "
        "Don't force depth. Don't repeat what they said. "
        "Example: 'What's been taking up most of your headspace lately?'"
    ),
}

# Emotions and topics that should trigger actionable suggestions
_INTERVENTION_EMOTIONS = frozenset({
    "stressed", "anxious", "overwhelmed", "frustrated", "sad"
})

_INTERVENTION_TOPICS = frozenset({
    "stress", "academics", "sleep", "exercise", "relationships",
    "assignments", "work", "workload", "exams", "deadlines",
})


@dataclass
class StrategyResult:
    name: str
    instruction: str
    divergence_context: str = ""


def select_strategy(
    perception: PerceptionResult,
    turn_count: int = 0,
) -> StrategyResult:
    topics = [c.topic for c in perception.claims if c.topic]
    query_topic = " ".join(topics) if topics else ""

    has_substance = any(
        c.topic and c.topic != "wellbeing" for c in perception.claims
    )

    # Does the student have a concrete stressor worth acting on?
    has_actionable_topic = any(
        c.topic and c.topic.lower() in _INTERVENTION_TOPICS
        for c in perception.claims
    )

    _no_reflect = {"confused", "neutral", "calm", "happy"}
    emotion_warrants_reflect = (
        perception.emotion.label not in _no_reflect
        and has_substance
    )

    # Strong emotion + substance → validate first
    if perception.emotion.strength > 0.75 and emotion_warrants_reflect:
        return StrategyResult(
            name=REFLECT_AND_VALIDATE,
            instruction=STRATEGY_INSTRUCTIONS[REFLECT_AND_VALIDATE],
        )

    divergences: list[dict] = []
    if config.PAM_ENABLED and query_topic and turn_count >= config.GROUNDING_MIN_TURNS:
        divergences = psp_mem.recall_active_divergences(query_topic, n=3)

    contested = [
        d
        for d in divergences
        if d["metadata"].get("negotiation_status") == "contested"
        and d["metadata"].get("confidence", 0) > config.DIVERGENCE_EPSILON
    ]

    if contested and random.random() < config.GROUNDING_PROBABILITY:
        return StrategyResult(
            name=GROUND_ON_DIVERGENCE,
            instruction=STRATEGY_INSTRUCTIONS[GROUND_ON_DIVERGENCE],
            divergence_context=contested[0]["document"],
        )

    # Moderate emotion + substance → validate or suggest
    if perception.emotion.strength > config.EMOTION_THRESHOLD and emotion_warrants_reflect:
        # If we've already reflected and the student has a concrete stressor,
        # move to suggesting something useful rather than reflecting again
        if (
            turn_count >= 3
            and perception.emotion.label in _INTERVENTION_EMOTIONS
            and (has_actionable_topic or perception.emotion.strength > 0.45)
        ):
            return StrategyResult(
                name=SUGGEST_INTERVENTION,
                instruction=STRATEGY_INSTRUCTIONS[SUGGEST_INTERVENTION],
            )
        return StrategyResult(
            name=REFLECT_AND_VALIDATE,
            instruction=STRATEGY_INSTRUCTIONS[REFLECT_AND_VALIDATE],
        )

    if not has_substance and perception.emotion.strength < 0.3:
        if not perception.claims:
            return StrategyResult(
                name=ACKNOWLEDGE,
                instruction=STRATEGY_INSTRUCTIONS[ACKNOWLEDGE],
            )
        return StrategyResult(
            name=SMALL_TALK,
            instruction=STRATEGY_INSTRUCTIONS[SMALL_TALK],
        )

    if not perception.claims:
        return StrategyResult(
            name=ACKNOWLEDGE,
            instruction=STRATEGY_INSTRUCTIONS[ACKNOWLEDGE],
        )

    open_divs = [
        d
        for d in divergences
        if d["metadata"].get("negotiation_status") == "open"
    ]
    if (
        config.PAM_ENABLED
        and open_divs
        and turn_count >= config.GROUNDING_MIN_TURNS
        and random.random() < 0.4
    ):
        return StrategyResult(
            name=GROUND_ON_DIVERGENCE,
            instruction=STRATEGY_INSTRUCTIONS[GROUND_ON_DIVERGENCE],
            divergence_context=open_divs[0]["document"],
        )

    if perception.is_continuation:
        # Suggest intervention when:
        # - student has shared a concrete stressor, OR
        # - emotion is present and we're past the opening turns
        if (
            turn_count >= 3
            and perception.emotion.label in _INTERVENTION_EMOTIONS
            and (has_actionable_topic or perception.emotion.strength > 0.3)
            and random.random() < 0.55  # raised from 0.2
        ):
            return StrategyResult(
                name=SUGGEST_INTERVENTION,
                instruction=STRATEGY_INSTRUCTIONS[SUGGEST_INTERVENTION],
            )
        return StrategyResult(
            name=MATCH_AND_CONTINUE,
            instruction=STRATEGY_INSTRUCTIONS[MATCH_AND_CONTINUE],
        )

    # Even on a new topic, if the student has a clear stressor, offer help
    if (
        turn_count >= 3
        and has_actionable_topic
        and perception.emotion.label in _INTERVENTION_EMOTIONS
        and random.random() < 0.5
    ):
        return StrategyResult(
            name=SUGGEST_INTERVENTION,
            instruction=STRATEGY_INSTRUCTIONS[SUGGEST_INTERVENTION],
        )

    return StrategyResult(
        name=MATCH_AND_CONTINUE,
        instruction=STRATEGY_INSTRUCTIONS[MATCH_AND_CONTINUE],
    )