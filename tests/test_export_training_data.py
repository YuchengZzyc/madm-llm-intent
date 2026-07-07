from __future__ import annotations

import json
from pathlib import Path

from scripts.export_training_data import export_jsonl


def test_export_training_data_format(tmp_path):
    out = tmp_path / "train.jsonl"
    count = export_jsonl(out)

    assert count > 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count

    tool_sample_count = 0
    no_tool_count = 0

    for line in lines:
        obj = json.loads(line)
        assert "messages" in obj and isinstance(obj["messages"], list)
        assert "tools" in obj and isinstance(obj["tools"], list)

        has_tool = any(m.get("role") == "tool" for m in obj["messages"])
        if has_tool:
            tool_sample_count += 1
            tool_msg = [m for m in obj["messages"] if m.get("role") == "tool"][0]
            payload = json.loads(tool_msg["content"])
            assert "state" in payload
            assert "status" in payload
        else:
            no_tool_count += 1
            assert all("tool_calls" not in m for m in obj["messages"])

    assert tool_sample_count > 0
    assert no_tool_count > 0
