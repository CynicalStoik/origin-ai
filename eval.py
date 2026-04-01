#!/usr/bin/env python3
"""
Quick eval harness — run a scripted conversation and print all responses.

Usage:
  python eval.py --model qwen3:8b
  python eval.py --model gemma3:4b
  python eval.py --model qwen3:8b --prompt simple
"""
from __future__ import annotations
import argparse
import os
import sys
import uuid

# Patch LLM_MODEL before any import reads config
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3:8b")
parser.add_argument(
    "--prompt",
    default="full",
    choices=["full", "simple"],
    help="'full' = current system prompt, 'simple' = stripped-down variant",
)
args = parser.parse_args()

os.environ["LLM_MODEL"] = args.model
os.environ["HF_HUB_OFFLINE"] = "1"

import config
config.LLM_MODEL = args.model

# Optionally patch system prompt before agent module is loaded
SIMPLE_PROMPT = """You are a university mindfulness coach. Talk like a real friend, not a therapist.

RULES:
1. 1-2 sentences max.
2. Be direct and warm.
3. Ask one specific follow-up question when appropriate.
4. Never invent details the student didn't mention.
5. Give one concrete suggestion when they share a real problem.
6. Never say: "I hear you", "That makes sense", "I'm here to support you", "Let's unpack", "How does that make you feel?", "That's completely valid".
"""

if args.prompt == "simple":
    import agent as _agent_mod
    _agent_mod  # import now so we can patch after
    import agent
    agent.SYSTEM_PROMPT = SIMPLE_PROMPT
else:
    import agent

from memory.procedural import load_techniques

SCRIPT = [
    "Hello, how are you?",
    "I've been doing okay I guess.",
    "I have a lot of assignments due this week and I can't focus.",
    "I just feel like there's too much and I don't know where to start.",
    "Yeah I've tried that before but it doesn't really help.",
    "I don't know, I just feel overwhelmed all the time.",
    "Maybe. I've also been having trouble sleeping.",
    "I usually go to sleep around 2am and wake up at 7.",
    "I know it's not great. I just can't fall asleep earlier.",
    "Bye, thanks.",
]

load_techniques()
session_id = uuid.uuid4().hex[:8]
ag = agent.Agent(session_id=session_id, vision_enabled=False)

print(f"\n{'='*60}")
print(f"  MODEL: {args.model}   PROMPT: {args.prompt}")
print(f"{'='*60}\n")

greeting = ag.generate_greeting()
print(f"Coach: {greeting}\n")

for turn_input in SCRIPT:
    print(f"Student: {turn_input}")
    response = ag.process_turn(turn_input)
    print(f"Coach: {response}\n")
