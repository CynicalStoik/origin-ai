from __future__ import annotations

import config
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "student" or "agent"
    content: str


class ShortTermMemory:
    def __init__(self, max_turns: int = config.STM_WINDOW_SIZE):
        self._turns: list[Turn] = []
        self._max = max_turns

    def add(self, role: str, content: str):
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self._max:
            self._turns = self._turns[-self._max :]

    def get_history(self) -> list[Turn]:
        return list(self._turns)

    def format_for_prompt(self) -> str:
        if not self._turns:
            return "(No prior turns this session.)"
        lines = []
        for t in self._turns:
            label = "Student" if t.role == "student" else "Coach"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def clear(self):
        self._turns.clear()
