from __future__ import annotations

import json

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor


def make_executor(tmp_path):
    service = ReminderService(JSONReminderStorage(tmp_path / "reminders.json"))
    return ToolExecutor(service)


def test_tool_executor_create_success(tmp_path):
    ex = make_executor(tmp_path)
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "create_reminder",
                    "arguments": '{"time_text":"tomorrow 9am","task":"meeting","target":"self"}',
                },
            }
        ],
    }
    out = ex.execute_tool_calls(msg)
    payload = json.loads(out[0]["content"])
    assert out[0]["role"] == "tool"
    assert payload["status"] == "success"
    assert payload["state"] == "success"


def test_tool_executor_missing_fields(tmp_path):
    ex = make_executor(tmp_path)
    call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "create_reminder", "arguments": '{"time_text":"tomorrow 9am"}'},
    }
    out = ex.execute_tool_call(call)
    payload = json.loads(out["content"])
    assert payload["status"] == "missing_fields"
    assert payload["state"] is False
    assert "task" in payload["missing_fields"]


def test_tool_executor_not_found_and_ambiguous(tmp_path):
    ex = make_executor(tmp_path)

    q_call = {
        "id": "call_q",
        "type": "function",
        "function": {"name": "query_reminder", "arguments": '{"task":"not-exist"}'},
    }
    q_payload = json.loads(ex.execute_tool_call(q_call)["content"])
    assert q_payload["status"] == "not_found"
    assert q_payload["state"] is False

    ex.execute_tool_call({"id": "c1", "type": "function", "function": {"name": "create_reminder", "arguments": '{"time_text":"8am","task":"pill"}'}})
    ex.execute_tool_call({"id": "c2", "type": "function", "function": {"name": "create_reminder", "arguments": '{"time_text":"8pm","task":"pill"}'}})

    d_call = {
        "id": "call_d",
        "type": "function",
        "function": {"name": "delete_reminder", "arguments": '{"task":"pill"}'},
    }
    d_payload = json.loads(ex.execute_tool_call(d_call)["content"])
    assert d_payload["status"] == "ambiguous"
    assert d_payload["state"] is False
    assert len(d_payload["candidates"]) == 2
