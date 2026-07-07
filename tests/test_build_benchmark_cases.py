from __future__ import annotations

import json

from scripts.build_benchmark_cases import build_case


def test_build_case_has_initial_reminders_and_contract():
    row = {
        "id": "bench_001",
        "scenario": "delete_ambiguous_generic",
        "tool": "delete",
        "expected_outcome": "ambiguous",
        "prompt": "Cancel my reminder tomorrow.",
    }
    case = build_case(row)
    assert case["id"] == "bench_001"
    assert isinstance(case["initial_reminders"], list)
    assert len(case["initial_reminders"]) == 2
    assert case["judge_contract"]["must_call_tool"] is True
    assert case["judge_contract"]["expected_tool_status"] == "ambiguous"


def test_build_case_no_tool_has_empty_init():
    row = {
        "id": "bench_002",
        "scenario": "no_tool_daily_chat",
        "tool": "no_tool",
        "expected_outcome": "no_tool",
        "prompt": "I took a walk today.",
    }
    case = build_case(row)
    assert case["initial_reminders"] == []
    assert case["judge_contract"]["must_call_tool"] is False
    assert case["judge_contract"]["expected_tool_status"] is None

