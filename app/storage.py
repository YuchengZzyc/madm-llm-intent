from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONReminderStorage:
    def __init__(self, path: str | Path = "data/reminders.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> list[dict[str, Any]]:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, reminders: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(reminders, ensure_ascii=False, indent=2), encoding="utf-8")
