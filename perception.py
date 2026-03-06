from __future__ import annotations

import json
import config
import ollama
from dataclasses import dataclass, field


@dataclass
class Emotion:
    label: str = "neutral"
    strength: float = 0.0


@dataclass
class Claim:
    proposition: str = ""
    holder: str = "student"
    truth_value: bool = True
    confidence: float = 0.8
    topic: str = ""


@dataclass
class PerceptionResult:
    emotion: Emotion = field(default_factory=Emotion)
    claims: list[Claim] = field(default_factory=list)
    is_continuation: bool = True


_PERCEPTION_PROMPT = """\
You are a perception module for a mindfulness coaching agent. Analyze the student's utterance and return ONLY valid JSON (no markdown, no extra text).

Student said: "{utterance}"

Recent context:
{context}
{visual_hint}
Return JSON with this exact structure:
{{
  "emotion": {{
    "label": "<one of: calm, stressed, anxious, sad, frustrated, angry, hopeful, confused, ashamed, overwhelmed, neutral>",
    "strength": <float 0.0-1.0>
  }},
  "claims": [
    {{
      "proposition": "<factual claim or belief expressed>",
      "holder": "student",
      "truth_value": <true or false>,
      "confidence": <float 0.0-1.0, how confident the student seems>,
      "topic": "<one or two word topic: sleep, stress, exercise, relationships, academics, mindfulness, etc.>"
    }}
  ],
  "is_continuation": <true if continuing the same topic as recent context, false if new topic>
}}

Rules:
- Extract 0-3 claims. Only extract clear factual statements or beliefs, not questions or pleasantries.
- Emotion strength: 0.0 = not at all, 1.0 = extremely strong.
- If the student contradicts something they said before, still record what they say NOW with truth_value reflecting their current stance.
- Return ONLY the JSON object, nothing else.
"""

_VISUAL_HINT_AGREE = (
    "Visual signal: their facial expression also shows {label} "
    "(confidence {strength:.2f}) — consistent with their words.\n"
)

_VISUAL_HINT_CONFLICT = (
    "Visual signal: their facial expression shows {visual_label} "
    "(confidence {visual_strength:.2f}), which may conflict with what "
    "they said. Consider this non-verbal cue when judging their emotional state.\n"
)


def _fuse_emotions(
    text_emotion: Emotion,
    visual_emotion: tuple[str, float] | None,
) -> Emotion:
    """
    Merge the LLM-derived text emotion with the camera-derived facial emotion.

    Rules:
    - No visual signal → return text emotion unchanged.
    - Labels agree     → average strengths + small confidence boost.
    - Labels disagree  → keep text label but reduce strength slightly to
                         signal uncertainty (the prompt already told the LLM
                         about the conflict, so the strategy layer will adapt).
    """
    if visual_emotion is None:
        return text_emotion

    v_label, v_strength = visual_emotion

    if text_emotion.label == v_label:
        fused_strength = min(1.0, (text_emotion.strength + v_strength) / 2 + 0.1)
        return Emotion(label=text_emotion.label, strength=fused_strength)
    else:
        # Conflict: preserve the text-derived label but dampen confidence
        fused_strength = max(0.0, text_emotion.strength - 0.15)
        return Emotion(label=text_emotion.label, strength=fused_strength)


def perceive(
    utterance: str,
    context: str = "",
    visual_emotion: tuple[str, float] | None = None,
) -> PerceptionResult:
    """Run the perception pipeline on a student utterance.

    Args:
        utterance:      The student's transcribed speech (or typed text).
        context:        Formatted recent conversation history from STM.
        visual_emotion: Optional (label, strength) tuple from vision.get_emotion().
                        When provided, it is fused with the text-derived emotion.
    """
    # Build the optional visual hint that goes into the LLM prompt
    visual_hint = ""
    if visual_emotion is not None:
        v_label, v_strength = visual_emotion
        # We don't know the text label yet, so we always include the raw hint;
        # the LLM sees the facial signal and can weigh it against the words.
        visual_hint = _VISUAL_HINT_CONFLICT.format(
            visual_label=v_label,
            visual_strength=v_strength,
        )

    prompt = _PERCEPTION_PROMPT.format(
        utterance=utterance,
        context=context,
        visual_hint=visual_hint,
    )

    try:
        resp = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            format="json",
        )
        raw = resp["message"]["content"].strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"[perception] LLM parse error: {e}")
        return PerceptionResult()

    text_emotion = Emotion(
        label=data.get("emotion", {}).get("label", "neutral"),
        strength=float(data.get("emotion", {}).get("strength", 0.0)),
    )

    # Fuse text and visual signals into a single Emotion
    fused_emotion = _fuse_emotions(text_emotion, visual_emotion)

    # Log the fusion result so it's visible during a session
    if visual_emotion is not None:
        v_label, _ = visual_emotion
        match = "" if text_emotion.label == v_label else f" (face: {v_label} — conflict)"
        print(f"  [vision: {v_label}{match}, text: {text_emotion.label} → fused: {fused_emotion.label} {fused_emotion.strength:.2f}]")

    claims = []
    for c in data.get("claims", []):
        claims.append(
            Claim(
                proposition=c.get("proposition", ""),
                holder=c.get("holder", "student"),
                truth_value=bool(c.get("truth_value", True)),
                confidence=float(c.get("confidence", 0.8)),
                topic=c.get("topic", ""),
            )
        )

    return PerceptionResult(
        emotion=fused_emotion,
        claims=claims,
        is_continuation=bool(data.get("is_continuation", True)),
    )
