import json
import os
from config import COMPACT_MEMORY_PATH

DEFAULT_COMPACT = {
    "summary": "",
    "persona": {
        "stressors": [],
        "values": [],
        "coping_strategies": [],
        "red_flags": [],
        "relationship_notes": ""
    }
}


class CompactMemory:
    def __init__(self, path: str = COMPACT_MEMORY_PATH):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = DEFAULT_COMPACT.copy()
            self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def update_summary(self, new_summary: str):
        self.data["summary"] = new_summary
        self._save()

    def update_persona(self, updates: dict):
        for key, value in updates.items():
            if key in self.data["persona"]:
                if isinstance(value, list):
                    existing = self.data["persona"][key]
                    self.data["persona"][key] = list(set(existing + value))
                else:
                    self.data["persona"][key] = value
        self._save()

    def get(self) -> dict:
        return self.data

    def get_summary(self) -> str:
        return self.data.get("summary", "")

    def get_persona(self) -> dict:
        return self.data.get("persona", {})

    def is_empty(self) -> bool:
        return not self.data["summary"] and not any(self.data["persona"].values())