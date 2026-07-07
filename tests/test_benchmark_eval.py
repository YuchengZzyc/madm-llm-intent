from __future__ import annotations

from app.benchmark_eval import StorageDelta, judge_tool_success


def test_judge_create_success_delta_ok():
    sample = {"tool": "create", "expected_outcome": "success"}
    delta = StorageDelta(before_active=1, after_active=2, created_count=1, deleted_count=0, changed_count=0)
    ok, _ = judge_tool_success(sample, "success", delta)
    assert ok


def test_judge_non_success_no_mutation_required():
    sample = {"tool": "update", "expected_outcome": "ambiguous"}
    delta = StorageDelta(before_active=1, after_active=1, created_count=0, deleted_count=0, changed_count=0)
    ok, _ = judge_tool_success(sample, "ambiguous", delta)
    assert ok


def test_judge_non_success_with_mutation_fails():
    sample = {"tool": "delete", "expected_outcome": "not_found"}
    delta = StorageDelta(before_active=1, after_active=0, created_count=0, deleted_count=0, changed_count=1)
    ok, _ = judge_tool_success(sample, "not_found", delta)
    assert not ok
