import argparse
from speech_io import record_from_mic, transcribe_wav
from config import llm_chat
from memory.manager import MemoryManager
from prompts import build_prompt


def _get_text_input() -> str:
    return input("You: ").strip()


def _get_speech_input() -> str:
    """Record microphone audio, transcribe to text, return transcript.

    Once transcribed, the rest of the agent runs exactly the same way as with
    typed input.
    """


    print("\nPress Enter to START recording, then press Enter again to STOP.")
    input("Ready…")
    wav_path, duration = record_from_mic()
    text, meta = transcribe_wav(wav_path, duration)
    if meta.get("confidence") is not None:
        print(f"  [asr_confidence: {meta['confidence']:.2f}]")
        print(f"Retrieved text: {text}")
    return (text or "").strip()


def run(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Mindfulness Agent (Sage)")
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Use microphone input (ASR) instead of typed input.",
    )
    args = parser.parse_args(argv)

    print("\nMindfulness Agent — type 'quit' to exit\n")
    memory = MemoryManager()

    input_fn = _get_speech_input if args.speech else _get_text_input

    while True:
        try:
            user_input = input_fn()
        except (EOFError, KeyboardInterrupt):
            print("\nTake care.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Sage: Take care of yourself. See you next time!")
            break

        context = memory.before_response(user_input)

        layers = context.get("layers_used", [])
        print(f"  [memory: {', '.join(layers) if layers else 'none'}]")

        messages = build_prompt(context, user_input)
        response = llm_chat(messages)

        print(f"\nSage: {response}\n")

        memory.after_response(user_input, response)


if __name__ == "__main__":
    run()