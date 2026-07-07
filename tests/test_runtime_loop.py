from __future__ import annotations

import json

from app.reminder_service import ReminderService
from app.runtime_loop import RuntimeLoop
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor


def test_runtime_loop_pause_and_resume(tmp_path):
    service = ReminderService(JSONReminderStorage(tmp_path / "reminders.json"))
    executor = ToolExecutor(service)
    calls = {"n": 0}

    def mock_model(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "create_reminder",
                            "arguments": '{"time_text":"tomorrow morning","task":"run","target":"self"}',
                        },
                    }
                ],
            }
        tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
        payload = json.loads(tool_msg["content"])
        assert payload["status"] == "success"
        return {"role": "assistant", "content": "created"}

    loop = RuntimeLoop(mock_model, executor, max_tool_rounds=3)
    result = loop.run("system", [], "remind me to run tomorrow morning")

    assert result["final"]["role"] == "assistant"
    assert result["final"]["content"] == "created"
    assert len([m for m in result["messages"] if m.get("role") == "tool"]) == 1
