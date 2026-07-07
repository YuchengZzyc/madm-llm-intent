from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor


def main() -> None:
    service = ReminderService(JSONReminderStorage("data/reminders.json"))
    executor = ToolExecutor(service)

    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_demo_query_001",
                "type": "function",
                "function": {
                    "name": "query_reminder",
                    "arguments": json.dumps(
                        {
                            "task": "打球",
                            "target": "self",
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }

    tool_messages = executor.execute_tool_calls(assistant_message)
    print("后端返回：")
    print(json.dumps(tool_messages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
