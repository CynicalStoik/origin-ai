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
        "You must reference the specific shift you noticed — do not skip it. "
        "But bring it up like you just thought of it, not like you're making a case. "
        "Sound curious, not confrontational: 'Earlier you said X — this sounds different.' "
        "One line. Name the gap, then stop. Don't explain it or list reasons."
    ),
    SUGGEST_INTERVENTION: (
        "Share something practical like it helped you or someone you know. "
        "Not a prescription. One sentence."
    ),
    ELICIT_CLARIFICATION: (
        "Something felt half-said. Ask the specific thing you want to know. "
        "Not 'tell me more' — ask what you actually want to understand."
    ),
    REFLECT_AND_VALIDATE: (
        "They're feeling something real. Don't fix it. Don't minimize it. "
        "Name what you see. Maybe one short sentence."
    ),
    MATCH_AND_CONTINUE: (
        "Conversation is flowing. React to what they actually said. "
        "Make a statement or ask one specific thing. Don't wander."
    ),
    ACKNOWLEDGE: (
        "They gave you something brief. Match their energy. "
        "A word, a short reaction. Don't push. Don't ask a random question."
    ),
    SMALL_TALK: (
        "Nothing heavy is on the table. Connect to something you know about them — "
        "their classes, something they mentioned before. If you don't know anything, "
        "one genuine question about their life. Don't force it."
    ),
}


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

    if perception.emotion.strength > 0.75:
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

    if perception.emotion.strength > config.EMOTION_THRESHOLD:
        return StrategyResult(
            name=REFLECT_AND_VALIDATE,
            instruction=STRATEGY_INSTRUCTIONS[REFLECT_AND_VALIDATE],
        )

    has_substance = any(
        c.topic and c.topic != "wellbeing" for c in perception.claims
    )

    if not has_substance and perception.emotion.strength < 0.3:
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
    if config.PAM_ENABLED and open_divs and turn_count >= config.GROUNDING_MIN_TURNS and random.random() < 0.4:
        return StrategyResult(
            name=GROUND_ON_DIVERGENCE,
            instruction=STRATEGY_INSTRUCTIONS[GROUND_ON_DIVERGENCE],
            divergence_context=open_divs[0]["document"],
        )

    if perception.is_continuation:
        if (
            turn_count >= 5
            and perception.emotion.strength > 0.3
            and random.random() < 0.2
        ):
            return StrategyResult(
                name=SUGGEST_INTERVENTION,
                instruction=STRATEGY_INSTRUCTIONS[SUGGEST_INTERVENTION],
            )
        return StrategyResult(
            name=MATCH_AND_CONTINUE,
            instruction=STRATEGY_INSTRUCTIONS[MATCH_AND_CONTINUE],
        )

    return StrategyResult(
        name=MATCH_AND_CONTINUE,
        instruction=STRATEGY_INSTRUCTIONS[MATCH_AND_CONTINUE],
    )
