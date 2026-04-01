from __future__ import annotations

import json
import re as _re
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
    visual_emotion: tuple[str, float] | None = None


# keyword → (emotion_label, strength)
_EMOTION_KEYWORDS: dict[str, tuple[str, float]] = {
    "stressed": ("stressed", 0.7),
    "stress": ("stressed", 0.6),
    "stressful": ("stressed", 0.6),
    "anxious": ("anxious", 0.7),
    "anxiety": ("anxious", 0.6),
    "worried": ("anxious", 0.5),
    "nervous": ("anxious", 0.5),
    "panicking": ("anxious", 0.8),
    "panic": ("anxious", 0.7),
    "sad": ("sad", 0.7),
    "depressed": ("sad", 0.6),
    "down": ("sad", 0.4),
    "lonely": ("sad", 0.5),
    "angry": ("angry", 0.7),
    "mad": ("angry", 0.6),
    "frustrated": ("frustrated", 0.6),
    "annoyed": ("frustrated", 0.5),
    "irritated": ("frustrated", 0.5),
    "tired": ("stressed", 0.4),
    "exhausted": ("overwhelmed", 0.6),
    "drained": ("overwhelmed", 0.5),
    "overwhelmed": ("overwhelmed", 0.7),
    "confused": ("confused", 0.5),
    "lost": ("confused", 0.4),
    "stuck": ("frustrated", 0.5),
    "happy": ("happy", 0.5),   # was "calm" — must match _EMOTION_VALENCE in agent.py
    "good": ("calm", 0.3),
    "great": ("calm", 0.5),
    "fine": ("neutral", 0.2),
    "okay": ("neutral", 0.2),
    "ok": ("neutral", 0.2),
    "calm": ("calm", 0.5),
    "peaceful": ("calm", 0.6),
    "relaxed": ("calm", 0.5),
    "hopeful": ("hopeful", 0.5),
    "better": ("hopeful", 0.4),
    "scared": ("anxious", 0.6),
    "afraid": ("anxious", 0.6),
    "ashamed": ("ashamed", 0.6),
    "guilty": ("ashamed", 0.5),
    "worthless": ("sad", 0.8),
    "terrible": ("sad", 0.7),
    "awful": ("sad", 0.7),
    "terribly": ("stressed", 0.6),
    "horrible": ("sad", 0.7),
    "struggling": ("stressed", 0.6),
    "rough": ("stressed", 0.5),
    "miserable": ("sad", 0.7),
    "suffering": ("sad", 0.7),
    "hurting": ("sad", 0.6),
    "failing": ("frustrated", 0.6),
    "broken": ("sad", 0.7),
    "hopeless": ("sad", 0.8),
    "helpless": ("overwhelmed", 0.7),
    "upset": ("frustrated", 0.6),
    "crying": ("sad", 0.7),
    "panicked": ("anxious", 0.7),
    "restless": ("anxious", 0.5),
    "uneasy": ("anxious", 0.5),
    "numb": ("sad", 0.5),
    "empty": ("sad", 0.5),
}

_NEGATIVE_EMOTIONS = frozenset({
    "stressed", "anxious", "sad", "frustrated", "angry",
    "overwhelmed", "confused", "ashamed",
})
_POSITIVE_EMOTIONS = frozenset({"calm", "hopeful", "happy"})  # added "happy"


_CLAIM_RULES: list[tuple] = [
    # (compiled_regex, proposition, truth_value, topic)
    (_re.compile(r"\b(not|haven'?t|can'?t|don'?t)\b.{0,20}\bsleep(ing)?\b|\bsleep(ing)?\b.{0,15}\b(bad|poor|badly|awful|terrible|rough)\b|\bcan'?t sleep\b|\bno sleep\b", _re.I), "student is sleeping well", False, "sleep"),
    (_re.compile(r"\bsleep(ing)?\b.{0,15}\b(well|fine|good|great|okay|ok)\b|\bsleep(ing)? (well|fine|good|great)\b|\bbeen sleeping\b", _re.I), "student is sleeping well", True, "sleep"),
    (_re.compile(r"\bgo(ing)? to bed\b.{0,15}\b([12]\s*am|midnight|late|after midnight)\b|\bstay(ing)? up\b.{0,15}\b(late|all night)\b|\b2am\b|\b1am\b", _re.I), "student goes to bed at a reasonable time", False, "sleep"),
    (_re.compile(r"\b(stressed|stress(ed)? out|overwhelmed|overwhelm)\b", _re.I), "student is managing stress well", False, "stress"),
    (_re.compile(r"\bnot (stressed|overwhelmed)\b|\bless stressed\b|\bstress.free\b", _re.I), "student is managing stress well", True, "stress"),
    (_re.compile(r"\b(assignments?|homework|deadlines?|exams?|midterms?|papers?|projects?)\b.{0,20}\b(due|piling|a lot|behind|overdue)\b|\ba lot of (work|assignments?|homework)\b", _re.I), "student has manageable academic workload", False, "academics"),
    (_re.compile(r"\bprocrastinat\b|\bcan'?t (get myself to start|start|focus)\b|\bkeep putting\b|\bputting (it|them) off\b", _re.I), "student can manage their academic tasks proactively", False, "academics"),
    (_re.compile(r"\bcan'?t (focus|concentrate)\b|\bdistracte?d\b|\bcan'?t get anything done\b", _re.I), "student can focus on their work", False, "focus"),
    (_re.compile(r"\b(feeling|feel)\b.{0,10}\b(good|great|fine|well|okay|ok|better|happy|calm)\b|\bfeeling (much )?better\b", _re.I), "student is doing well emotionally", True, "wellbeing"),
    (_re.compile(r"\b(feeling|feel)\b.{0,10}\b(bad|terrible|awful|sad|down|low|depressed|anxious|worried|scared)\b|\bnot (feeling|doing) (well|good|great|fine)\b", _re.I), "student is doing well emotionally", False, "wellbeing"),
]

_PERCEPTION_PROMPT = """\
You are a perception module for a mindfulness coaching agent. Analyze the student's \
utterance and return ONLY valid JSON (no markdown, no extra text).

Student said: "{utterance}"

Recent context:
{context}
{visual_hint}
Return JSON with this exact structure:
{{
  "emotion": {{
    "label": "<one of: calm, happy, stressed, anxious, sad, frustrated, angry, hopeful, \
confused, ashamed, overwhelmed, neutral>",
    "strength": <float 0.0-1.0>
  }},
  "claims": [
    {{
      "proposition": "<factual claim or belief expressed>",
      "holder": "student",
      "truth_value": <true or false>,
      "confidence": <float 0.0-1.0, how confident the student seems>,
      "topic": "<one or two word topic: sleep, stress, exercise, relationships, \
academics, mindfulness, etc.>"
    }}
  ],
  "is_continuation": <true if continuing the same topic as recent context, false \
if new topic>
}}

Rules:
- Extract 0-3 claims. Only extract clear factual statements or beliefs, not \
questions or pleasantries.
- Emotion strength: 0.0 = not at all, 1.0 = extremely strong.
- Always phrase propositions in POSITIVE form. Use truth_value to indicate \
whether the student affirms or denies it. \
Example: "I'm not sleeping well" → proposition: "student is sleeping well", truth_value: false. \
Example: "I sleep fine" → proposition: "student is sleeping well", truth_value: true. \
This allows contradictions between turns to be detected reliably.
- If the student contradicts something they said before, still record what they \
say NOW with truth_value reflecting their current stance.
- Return ONLY the JSON object, nothing else.
"""


def _fuse_emotions(
    text_emotion: Emotion,
    visual_emotion: tuple[str, float] | None,
) -> Emotion:
    if visual_emotion is None:
        return text_emotion

    v_label, v_strength = visual_emotion

    if text_emotion.label == v_label:
        fused_strength = min(1.0, (text_emotion.strength + v_strength) / 2 + 0.1)
        return Emotion(label=text_emotion.label, strength=fused_strength)
    else:
        fused_strength = max(0.0, text_emotion.strength - 0.15)
        return Emotion(label=text_emotion.label, strength=fused_strength)


def fast_perceive(
    utterance: str,
    context: str = "",
    visual_emotion: tuple[str, float] | None = None,
) -> PerceptionResult:
    """Rule-based perception for short/simple utterances. No LLM call."""
    words = utterance.lower().split()

    best = Emotion()
    for word in words:
        clean = word.strip(".,!?'\"()[]")
        if clean in _EMOTION_KEYWORDS:
            label, strength = _EMOTION_KEYWORDS[clean]
            if strength > best.strength:
                best = Emotion(label=label, strength=strength)

    fused = _fuse_emotions(best, visual_emotion)
    is_cont = bool(context and not context.startswith("(No prior"))

    claims = []
    # Keyword-based topic claims — extracted without LLM
    seen_propositions: set[str] = set()
    for pattern, proposition, truth_value, topic in _CLAIM_RULES:
        if proposition in seen_propositions:
            continue
        if pattern.search(utterance):
            seen_propositions.add(proposition)
            claims.append(Claim(
                proposition=proposition,
                holder="student",
                truth_value=truth_value,
                confidence=0.75,
                topic=topic,
            ))
    # Generic wellbeing claim from emotion keywords (fallback)
    if fused.label in _NEGATIVE_EMOTIONS | _POSITIVE_EMOTIONS and fused.strength >= 0.3:
        if "student is doing well emotionally" not in seen_propositions:
            claims.append(Claim(
                proposition="student is doing well emotionally",
                holder="student",
                truth_value=fused.label in _POSITIVE_EMOTIONS,
                confidence=max(fused.strength, 0.6),
                topic="wellbeing",
            ))

    # Print fusion line for fast path too so terminal always shows vision state
    if visual_emotion is not None:
        v_label, _ = visual_emotion
        # Only flag as conflict when labels genuinely differ AND text emotion is present
        is_conflict = (best.label != v_label) and (best.strength >= 0.25) and (v_label != best.label)
        tag = f" (face: {v_label} — conflict)" if is_conflict else ""
        print(
            f"  [vision: {v_label}{tag}, text: {best.label} "
            f"→ fused: {fused.label} {fused.strength:.2f}]"
        )

    return PerceptionResult(
        emotion=fused, claims=claims, is_continuation=is_cont,
        visual_emotion=visual_emotion,
    )


def perceive(
    utterance: str,
    context: str = "",
    visual_emotion: tuple[str, float] | None = None,
) -> PerceptionResult:
    """Full LLM-based perception for substantive utterances."""
    # Fast path: if rule-based extraction already found topic-specific claims,
    # skip the LLM call entirely — one less qwen3 thinking phase per turn.
    fast_result = fast_perceive(utterance, context, visual_emotion=visual_emotion)
    if any(c.topic and c.topic != "wellbeing" for c in fast_result.claims):
        return fast_result

    # Visual intentionally excluded from perception prompt — injecting it biases
    # the speech emotion classification toward the visual signal. Visual-verbal
    # conflict is handled separately in the response layer (_evaluate_conflict).
    prompt = _PERCEPTION_PROMPT.format(
        utterance=utterance,
        context=context,
        visual_hint="",
    )

    try:
        resp = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 1800},
            format="json",
        )
        raw = resp["message"]["content"].strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"[perception] LLM error: {e}")
        return fast_perceive(utterance, context, visual_emotion)

    text_emotion = Emotion(
        label=data.get("emotion", {}).get("label", "neutral"),
        strength=float(data.get("emotion", {}).get("strength", 0.0)),
    )

    fused_emotion = _fuse_emotions(text_emotion, visual_emotion)

    if visual_emotion is not None:
        v_label, _ = visual_emotion
        is_conflict = text_emotion.label != v_label and text_emotion.strength >= 0.25
        tag = f" (face: {v_label} — conflict)" if is_conflict else ""
        print(
            f"  [vision: {v_label}{tag}, text: {text_emotion.label} "
            f"→ fused: {fused_emotion.label} {fused_emotion.strength:.2f}]"
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

    if fused_emotion.label in _NEGATIVE_EMOTIONS | _POSITIVE_EMOTIONS and fused_emotion.strength >= 0.3:
        claims.append(Claim(
            proposition="student is doing well emotionally",
            holder="student",
            truth_value=fused_emotion.label in _POSITIVE_EMOTIONS,
            confidence=max(fused_emotion.strength, 0.6),
            topic="wellbeing",
        ))

    return PerceptionResult(
        emotion=fused_emotion,
        claims=claims,
        is_continuation=bool(data.get("is_continuation", True)),
        visual_emotion=visual_emotion,
    )