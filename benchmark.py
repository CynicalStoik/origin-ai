#!/usr/bin/env python3
"""
Comprehensive benchmark: model × prompt × params
Usage: python benchmark.py --model qwen3:8b
Runs all prompt/param combos for a given model and prints a scored table.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import uuid
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

os.environ["LLM_MODEL"] = args.model
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config
config.LLM_MODEL = args.model

# Pre-load embedding model once before all runs so it doesn't hang mid-test
print("[benchmark] Pre-loading embedding model …")
from memory import store as _store
_store._get_embed_fn()
print("[benchmark] Embedding model ready.\n")

from memory.procedural import load_techniques
import agent as agent_mod

# ── Prompt variants ─────────────────────────────────────────────────────────

PROMPT_FULL = """You are a mindfulness coach at a university wellness center. \
You help students with stress, anxiety, sleep, focus, and emotional overwhelm. \
Talk like a real person — casual, warm, direct. Like a friend who happens to know about wellbeing. \
NOT like a therapist. NOT like a wellness pamphlet.

HARD RULES:
1. One or two sentences max. Never more.
2. If they say good/fine/okay/nothing much, believe them. Ask one light follow-up.
3. NEVER suggest they are hiding something. If they say nothing is wrong, accept it.
4. NEVER ask about a feeling they didn't mention.
5. NEVER invent any detail — no courses, assignments, names, or events they didn't say.
6. No filler phrases. No "I hear you." No metaphors. No poetry. No self-introduction mid-conversation.
7. If they deny something, drop it. Don't push.
8. If they're confused, say so simply and move on differently.
9. When someone shares a real problem, offer ONE concrete, practical suggestion. Don't just reflect it back.
10. "Yeah", "sure", "okay", "mhm", "why not", "sure why not" are agreements — move forward, don't restart.

BANNED PHRASES — never say these or anything like them:
- "find a little space", "create some space", "hold that space", "safe space"
- "sit with that", "sit with your feelings", "let's explore", "let's unpack"
- "I'm here to listen", "I'm here for you", "I'm here to support you"
- "How does that make you feel?", "That's completely valid", "That makes sense"
- Any sentence starting with "Let's see if we can"
- Any sentence starting with "Let's try"
- Any sentence starting with "Let's"

EXAMPLES of good responses:
Student: "I'm stressed about assignments" → "What's the most pressing one?"
Student: "I can't focus" → "Is it restless or just blank?"
Student: "I've been tired" → "How's your sleep been?"
Student: "I don't know" → "What would you say if you did?"
Student: "I'm fine" → "Good — anything on your mind at all?"
Student: "I can't sleep earlier" → "Have you tried cutting screens 30 min before bed? Shifts the body clock."
Student: "I feel overwhelmed" → "Write everything down first — just getting it out of your head helps."
"""

PROMPT_EXAMPLES = """You are a university mindfulness coach. You sound like a direct, warm friend — not a therapist.

Style:
- 1-2 sentences only. Never more.
- Give concrete suggestions when someone shares a problem.
- Ask one specific question, not "tell me more."
- Accept what people say. Don't push or probe.
- Never say: "I hear you", "Let's explore", "Let's try", "Let's see if we can", "I'm here to support you", "How does that make you feel?", "That makes sense", "That's completely valid", "find some space", "sit with that"

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
Student: "Bye" → "Take care."
"""

PROMPT_MINIMAL = """You are a mindfulness coach at a university. Be a direct, warm friend.

Rules (strict):
- MAX 2 sentences per reply
- Give ONE concrete action when they share a problem
- Ask ONE specific question when you need more info
- Never use: "Let's", "I hear you", "I'm here for you", "How does that make you feel", "That makes sense", "That's valid", "explore", "unpack", "space", "journey"
- Never invent details they didn't mention
- Accept what they say, don't probe for hidden feelings
"""

PROMPTS = {
    "full": PROMPT_FULL,
    "examples": PROMPT_EXAMPLES,
    "minimal": PROMPT_MINIMAL,
}

# ── Parameter variants ───────────────────────────────────────────────────────

PARAM_SETS = {
    "default": {"temperature": 0.55, "repeat_penalty": 1.15, "top_k": 30, "top_p": 0.85},
    "tight":   {"temperature": 0.3,  "repeat_penalty": 1.3,  "top_k": 20, "top_p": 0.8},
    "creative":{"temperature": 0.75, "repeat_penalty": 1.2,  "top_k": 40, "top_p": 0.92},
}

# ── Scripted conversation (20 turns) ─────────────────────────────────────────

SCRIPT = [
    "Hello, how are you?",
    "I've been doing okay I guess.",
    "I have a lot of assignments due this week and I can't focus.",
    "I just feel like there's too much and I don't know where to start.",
    "Yeah I've tried making lists before but it doesn't really help.",
    "I don't know, I just feel overwhelmed all the time.",
    "Maybe. I've also been having trouble sleeping.",
    "I usually go to sleep around 2am and wake up at 7.",
    "I know it's not great. I just can't fall asleep earlier.",
    "My mind just races when I lie down.",
    "I keep thinking about all the things I haven't done yet.",
    "Yeah I guess. It's been like this for a few weeks.",
    "I've been skipping the gym too, which I know doesn't help.",
    "I used to go three times a week but now I just don't have the energy.",
    "I tried going yesterday but I just sat in the parking lot and drove home.",
    "That's embarrassing to say out loud.",
    "I haven't told anyone else about this.",
    "I don't really know what I'm hoping to get out of this conversation.",
    "Maybe just to feel less alone with it.",
    "Bye, thanks.",
]

# ── Scoring ──────────────────────────────────────────────────────────────────

BANNED = [
    "let's explore", "let's try", "let's see if we can", "let's unpack",
    "i hear you", "i'm here to listen", "i'm here for you", "i'm here to support",
    "how does that make you feel", "that's completely valid", "that makes sense",
    "find a little space", "find some space", "safe space", "sit with that",
    "sit with your feelings", "hold that space",
]

ADVICE_SIGNALS = [
    "try ", "have you tried", "one thing", "might help", "could help",
    "writing down", "write down", "break", "earlier", "routine",
    "schedule", "minutes", "start with", "focus on", "pick one", "cut screens",
]

def score_response(text: str, prev_responses: list[str]) -> dict:
    lower = text.lower()
    words = text.split()
    length = len(words)

    if length <= 0:
        length_score = -10
    elif length <= 25:
        length_score = 10
    elif length <= 40:
        length_score = 5
    else:
        length_score = max(-5, 10 - (length - 25) // 5)

    banned_hits = [b for b in BANNED if b in lower]
    banned_score = -5 * len(banned_hits)

    prefix = " ".join(words[:4]).lower() if len(words) >= 4 else lower
    repeat_hits = sum(1 for r in prev_responses[-5:] if r.lower().startswith(prefix))
    repeat_score = -6 * repeat_hits

    advice_score = 5 if any(s in lower for s in ADVICE_SIGNALS) else 0
    empty_penalty = -20 if not text.strip() else 0

    total = length_score + banned_score + repeat_score + advice_score + empty_penalty
    return {
        "total": total,
        "length": length,
        "banned": banned_hits,
        "repeat": repeat_hits,
        "has_advice": advice_score > 0,
    }

# ── State reset between runs ─────────────────────────────────────────────────

def _reset_state():
    """Clear episodic/semantic/perspective collections and LTM profile between runs."""
    for col_name in ("episodic", "semantic", "perspective"):
        try:
            col = _store.get_collection(col_name)
            ids = col.get()["ids"]
            if ids:
                col.delete(ids=ids)
        except Exception:
            pass
    # Reset LTM profile
    if os.path.exists(config.LPM_PATH):
        os.remove(config.LPM_PATH)

# ── Run one config ────────────────────────────────────────────────────────────

import ollama as _ollama
_orig_chat = _ollama.chat

def run_config(prompt_name: str, params_name: str) -> tuple[float, list[dict]]:
    _reset_state()

    agent_mod.SYSTEM_PROMPT = PROMPTS[prompt_name]
    params = PARAM_SETS[params_name]

    sid = uuid.uuid4().hex[:8]
    ag = agent_mod.Agent(session_id=sid, vision_enabled=False)

    num_predict_base = 650

    def _patched_chat(model, messages, options=None, **kwargs):
        if options is None:
            options = {}
        options.update({
            "temperature": params["temperature"],
            "repeat_penalty": params["repeat_penalty"],
            "top_k": params["top_k"],
            "top_p": params["top_p"],
        })
        if "num_predict" not in options or options["num_predict"] < num_predict_base:
            options["num_predict"] = num_predict_base
        return _orig_chat(model=model, messages=messages, options=options, **kwargs)

    _ollama.chat = _patched_chat

    prev_responses: list[str] = []
    results: list[dict] = []

    for user_text in SCRIPT:
        response = ag.process_turn(user_text)
        sc = score_response(response, prev_responses)
        results.append({"user": user_text, "coach": response, "score": sc})
        prev_responses.append(response)

    _ollama.chat = _orig_chat

    avg = sum(r["score"]["total"] for r in results) / len(results)
    return avg, results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_techniques()

    print(f"\n{'='*70}")
    print(f"  BENCHMARK — MODEL: {args.model}")
    print(f"{'='*70}\n")

    summary: list[tuple[str, str, float]] = []

    for prompt_name in PROMPTS:
        for params_name in PARAM_SETS:
            label = f"prompt={prompt_name}  params={params_name}"
            print(f"\n{'─'*60}")
            print(f"  {label}")
            print(f"{'─'*60}")
            try:
                avg, results = run_config(prompt_name, params_name)
                summary.append((prompt_name, params_name, avg))

                for r in results:
                    sc = r["score"]
                    flags = []
                    if sc["banned"]:
                        flags.append(f"BANNED:{sc['banned']}")
                    if sc["repeat"] > 0:
                        flags.append(f"REPEAT×{sc['repeat']}")
                    if sc["has_advice"]:
                        flags.append("advice")
                    flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
                    print(f"  [{sc['total']:+3d}|{sc['length']:2d}w{flag_str}]  S: {r['user'][:50]}")
                    print(f"  {'':>12}  C: {r['coach']}")

                print(f"\n  *** AVG SCORE: {avg:.1f} ***")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                summary.append((prompt_name, params_name, -999.0))

    print(f"\n\n{'='*70}")
    print(f"  FINAL RANKING — {args.model}")
    print(f"{'='*70}")
    summary.sort(key=lambda x: x[2], reverse=True)
    for rank, (p, par, score) in enumerate(summary, 1):
        bar = "█" * max(0, int(score + 10))
        print(f"  #{rank}  prompt={p:<10} params={par:<10}  avg={score:+5.1f}  {bar}")
    print()

if __name__ == "__main__":
    main()
