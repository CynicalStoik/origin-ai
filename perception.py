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


def perceive(utterance: str, context: str = "") -> PerceptionResult:
    """Run the perception pipeline on a student utterance."""
    prompt = _PERCEPTION_PROMPT.format(utterance=utterance, context=context)

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

    emotion = Emotion(
        label=data.get("emotion", {}).get("label", "neutral"),
        strength=float(data.get("emotion", {}).get("strength", 0.0)),
    )

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
        emotion=emotion,
        claims=claims,
        is_continuation=bool(data.get("is_continuation", True)),
    )
