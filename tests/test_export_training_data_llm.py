from __future__ import annotations

import json

import pytest

from app.tool_registry import get_tools
from scripts.export_training_data_llm import DataBuilder


def test_tools_description_and_local_tool_result(tmp_path):
    builder = DataBuilder(tmp_path / "tmp_reminders.json")
    bp = {
        "scenario": "create_success",
        "user_text": "Remind me tomorrow at 4 PM to take medicine",
        "assistant_text": None,
        "tool_name": "create_reminder",
        "args": {"time_text": "tomorrow 4 PM", "task": "take medicine", "target": "self"},
        "setup_reminders": [],
        "notes": "test",
    }

    sample = builder.build_sample(bp)
    builder.cleanup()

    assert sample["tools"] == get_tools()

    tool_msg = [m for m in sample["messages"] if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert "state" in payload
    assert "status" in payload


def test_scenario_status_mismatch_is_rejected(tmp_path):
    builder = DataBuilder(tmp_path / "tmp_reminders.json")
    bad_bp = {
        "scenario": "delete_ambiguous",
        "user_text": "Delete my reminder",
        "assistant_text": None,
        "tool_name": "delete_reminder",
        "args": {"target": "self"},
        "setup_reminders": [],
        "notes": "mismatch",
    }

    with pytest.raises(ValueError, match="expected status=ambiguous"):
        builder.build_sample(bad_bp)
    builder.cleanup()
