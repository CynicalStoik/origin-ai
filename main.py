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
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable webcam facial emotion detection (requires fer + opencv).",
    )
    args = parser.parse_args()
    text_mode = args.text
    vision_enabled = args.vision

    if not text_mode:
        import speech
    if vision_enabled:
        import vision
        vision.start()

    session_id = uuid.uuid4().hex[:8]
    modalities = []
    if not text_mode:
        modalities.append("voice")
    else:
        modalities.append("text")
    if vision_enabled:
        modalities.append("vision")

    mode_label = "Type your message." if text_mode else "Speak naturally."
    print("=" * 60)
    print("  ORIGON – Mindfulness Coaching Agent")
    print(f"  {mode_label} Press Ctrl+C to end the session.")
    print(f"  Modalities: {', '.join(modalities)}")
    print("=" * 60)
    print()

    load_techniques()
    agent = Agent(session_id=session_id, vision_enabled=vision_enabled)

    greeting = "Hey, good to see you. What's on your mind today?"
    print(f"Coach: {greeting}")
    if not text_mode:
        speech.speak(greeting)

    def shutdown(signum=None, frame=None):
        print("\n\n[session] Wrapping up …")
        agent.end_session()
        if vision_enabled:
            vision.stop()
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
                # Build STM context in the format speech.py expects
                stm_context = [
                    {"role": t.role, "content": t.content}
                    for t in agent.stm.get_history()
                ]
                audio = speech.record_audio(context=stm_context)
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