from __future__ import annotations

import json

from scripts.run_toolcall_param_eval import parse_tool_call_output


def test_parse_qwen_xml_tool_call_output():
    raw = """The user wants me to remind them tomorrow at 4:00 to play basketball.
</think>

<tool_call>
<function=create_reminder>
<parameter=time_text>
tomorrow at 4:00
</parameter>
<parameter=task>
play basketball
</parameter>
<parameter=target>
self
</parameter>
</function>
</tool_call>"""

    parsed = parse_tool_call_output(raw)

    assert parsed is not None
    tool_call = parsed["tool_calls"][0]
    assert tool_call["function"]["name"] == "create_reminder"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "time_text": "tomorrow at 4:00",
        "task": "play basketball",
        "target": "self",
    }


def test_parse_json_tool_call_output_still_works():
    raw = (
        '<tool_call>{"name":"create_reminder","arguments":'
        '{"time_text":"tomorrow at 4:00","task":"play basketball","target":"self"}}</tool_call>'
    )

    parsed = parse_tool_call_output(raw)

    assert parsed is not None
    tool_call = parsed["tool_calls"][0]
    assert tool_call["function"]["name"] == "create_reminder"
    assert json.loads(tool_call["function"]["arguments"])["task"] == "play basketball"
