from config import STM_MAX_TURNS


class ShortTermMemory:
    def __init__(self, max_turns: int = STM_MAX_TURNS):
        self.turns = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get(self) -> list:
        return self.turns

    def clear(self):
        self.turns = []