from __future__ import annotations

import config
from dataclasses import dataclass
from perception import PerceptionResult
from memory import perspective as psp_mem

GROUND_ON_DIVERGENCE = "ground_on_divergence"
SUGGEST_INTERVENTION = "suggest_intervention"
ELICIT_CLARIFICATION = "elicit_clarification"
REFLECT_AND_VALIDATE = "reflect_and_validate"
MATCH_AND_CONTINUE = "match_and_continue"

STRATEGY_INSTRUCTIONS = {
    GROUND_ON_DIVERGENCE: (
        "You've noticed something they said now doesn't quite line up with "
        "something from before. Gently bring that up — not as a gotcha, but "
        "like 'hm, I remember you mentioned X, and now it sounds like Y — "
        "what shifted?' Let them sit with it."
    ),
    SUGGEST_INTERVENTION: (
        "The moment feels right to offer something practical. Suggest a "
        "technique naturally, like you're sharing something that helped "
        "someone you know. Keep it casual, not prescriptive."
    ),
    ELICIT_CLARIFICATION: (
        "Something feels unclear or half-said. Draw it out gently — not "
        "with a therapist question, but with genuine curiosity. Like, "
        "'say more about that' or 'what do you mean when you say...'"
    ),
    REFLECT_AND_VALIDATE: (
        "They're feeling something strong right now. Don't try to fix it. "
        "Just be with them. Acknowledge what they're going through in a way "
        "that shows you actually heard them. A short, warm response."
    ),
    MATCH_AND_CONTINUE: (
        "Things are flowing naturally. Keep the conversation going — share "
        "a thought, make an observation, or gently explore what they said. "
        "Don't force a question if a simple acknowledgment fits better."
    ),
}


@dataclass
class StrategyResult:
    name: str
    instruction: str
    divergence_context: str = ""


def select_strategy(perception: PerceptionResult) -> StrategyResult:
    topics = [c.topic for c in perception.claims if c.topic]
    query_topic = " ".join(topics) if topics else ""

    divergences: list[dict] = []
    if query_topic:
        divergences = psp_mem.recall_active_divergences(query_topic, n=5)

    high_conf_divs = [
        d
        for d in divergences
        if d["metadata"].get("confidence", 0) > config.DIVERGENCE_EPSILON
    ]

    if high_conf_divs:
        top = high_conf_divs[0]
        status = top["metadata"].get("negotiation_status", "open")
        div_ctx = top["document"]

        if status == "contested":
            return StrategyResult(
                name=GROUND_ON_DIVERGENCE,
                instruction=STRATEGY_INSTRUCTIONS[GROUND_ON_DIVERGENCE],
                divergence_context=div_ctx,
            )
        elif status == "resolved":
            return StrategyResult(
                name=SUGGEST_INTERVENTION,
                instruction=STRATEGY_INSTRUCTIONS[SUGGEST_INTERVENTION],
                divergence_context=div_ctx,
            )
        else:  # open
            return StrategyResult(
                name=ELICIT_CLARIFICATION,
                instruction=STRATEGY_INSTRUCTIONS[ELICIT_CLARIFICATION],
                divergence_context=div_ctx,
            )

    if perception.emotion.strength > config.EMOTION_THRESHOLD:
        return StrategyResult(
            name=REFLECT_AND_VALIDATE,
            instruction=STRATEGY_INSTRUCTIONS[REFLECT_AND_VALIDATE],
        )

    if perception.is_continuation:
        return StrategyResult(
            name=MATCH_AND_CONTINUE,
            instruction=STRATEGY_INSTRUCTIONS[MATCH_AND_CONTINUE],
        )

    return StrategyResult(
        name=ELICIT_CLARIFICATION,
        instruction=STRATEGY_INSTRUCTIONS[ELICIT_CLARIFICATION],
    )
