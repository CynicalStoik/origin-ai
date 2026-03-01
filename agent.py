from config import llm_chat
from memory.manager import MemoryManager
from prompts import build_prompt


def run():
    print("\nMindfulness Agent — type 'quit' to exit\n")
    memory = MemoryManager()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTake care.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Sage: Take care of yourself. See you next time!.")
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