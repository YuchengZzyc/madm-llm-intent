from __future__ import annotations

import json

from app.tool_registry import get_tools
from scripts.run_toolcall_param_eval_api import build_messages, parse_tool_call_output


def test_parse_native_tool_calls_message():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "create_reminder",
                    "arguments": "{\"time_text\":\"tomorrow\",\"task\":\"take medicine\",\"target\":\"self\"}",
                },
            }
        ],
    }

    parsed, _raw = parse_tool_call_output(message)

    assert parsed is not None
    assert parsed["tool_calls"][0]["function"]["name"] == "create_reminder"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"])["task"] == "take medicine"


def test_parse_json_content_tool_calls_message():
    content = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "delete_reminder",
                    "arguments": "{\"task\":\"water flowers\",\"target\":\"self\"}",
                },
            }
        ],
    }

    parsed, _raw = parse_tool_call_output({"role": "assistant", "content": json.dumps(content)})

    assert parsed is not None
    assert parsed["tool_calls"][0]["function"]["name"] == "delete_reminder"


def test_build_messages_includes_four_tool_descriptions_and_examples():
    messages = build_messages("What reminders do I have tomorrow?", get_tools(), few_shot_count=4)
    system = messages[0]["content"]

    assert "Available tools and descriptions" in system
    assert system.count('"name":') == 4
    assert "create_reminder" in system
    assert "query_reminder" in system
    assert "update_reminder" in system
    assert "delete_reminder" in system
    assert len([message for message in messages if message["role"] == "assistant"]) == 4
