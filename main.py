from __future__ import annotations

import sys
import uuid
import signal
import argparse
from agent import Agent
from memory.procedural import load_techniques


def main():
    parser = argparse.ArgumentParser(description="Origon Mindfulness Coach")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Text-only mode (type instead of speak). Required for Docker.",
    )
    args = parser.parse_args()
    text_mode = args.text

    if not text_mode:
        import speech  # only import (and load audio deps) when needed

    session_id = uuid.uuid4().hex[:8]
    mode_label = "Type your message." if text_mode else "Speak naturally."
    print("=" * 60)
    print("  ORIGON – Mindfulness Coaching Agent")
    print(f"  {mode_label} Press Ctrl+C to end the session.")
    print("=" * 60)
    print()

    load_techniques()

    agent = Agent(session_id=session_id)

    greeting = "Hey, good to see you. What's on your mind today?"
    print(f"Coach: {greeting}")
    if not text_mode:
        speech.speak(greeting)

    def shutdown(signum=None, frame=None):
        print("\n\n[session] Wrapping up …")
        agent.end_session()
        print("[session] Profile saved. Take care!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    while True:
        try:
            if text_mode:
                user_text = input("\nYou: ").strip()
                if not user_text:
                    continue
            else:
                audio = speech.record_audio()
                if audio.size == 0:
                    continue
                user_text = speech.transcribe(audio)
                if not user_text.strip():
                    continue
                print(f"\nStudent: {user_text}")

            response = agent.process_turn(user_text)
            print(f"Coach: {response}")

            if not text_mode:
                speech.speak(response)

        except (KeyboardInterrupt, EOFError):
            shutdown()
        except Exception as e:
            print(f"[error] {e}")
            continue


if __name__ == "__main__":
    main()
