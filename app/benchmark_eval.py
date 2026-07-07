from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StorageDelta:
    before_active: int
    after_active: int
    created_count: int
    deleted_count: int
    changed_count: int


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _active(reminders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in reminders if r.get("status") == "active"]


def compute_storage_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> StorageDelta:
    before_map = {r.get("reminder_id"): r for r in before}
    after_map = {r.get("reminder_id"): r for r in after}

    created_count = sum(1 for rid in after_map if rid not in before_map)
    deleted_count = sum(1 for rid in before_map if rid not in after_map)

    changed_count = 0
    for rid, b in before_map.items():
        a = after_map.get(rid)
        if a is None:
            continue
        if b != a:
            changed_count += 1

    return StorageDelta(
        before_active=len(_active(before)),
        after_active=len(_active(after)),
        created_count=created_count,
        deleted_count=deleted_count,
        changed_count=changed_count,
    )


def judge_tool_success(sample: dict[str, Any], tool_status: str | None, delta: StorageDelta) -> tuple[bool, str]:
    expected = sample.get("expected_outcome")
    tool = sample.get("tool")

    if expected == "no_tool":
        if tool_status is not None:
            return False, f"no_tool case should not call tool, got status={tool_status}"
        if delta.created_count != 0 or delta.deleted_count != 0 or delta.changed_count != 0:
            return False, "no_tool case should not change storage"
        return True, "ok"

    if tool_status != expected:
        return False, f"status mismatch: expected={expected}, got={tool_status}"

    if expected != "success":
        # For non-success paths, storage should not mutate in this harness.
        if delta.created_count != 0 or delta.deleted_count != 0 or delta.changed_count != 0:
            return False, "non-success case should not change storage"
        return True, "ok"

    if tool == "create":
        if delta.after_active != delta.before_active + 1:
            return False, "create success should increase active reminders by 1"
    elif tool == "delete":
        if delta.after_active != delta.before_active - 1:
            return False, "delete success should decrease active reminders by 1"
    elif tool == "update":
        if delta.changed_count < 1:
            return False, "update success should modify at least one record"
    elif tool == "query":
        if delta.created_count != 0 or delta.deleted_count != 0 or delta.changed_count != 0:
            return False, "query success should not change storage"

    return True, "ok"


def judge_answer_accuracy(sample: dict[str, Any], final_text: str) -> tuple[bool, str]:
    # Lightweight lexical check; tool correctness remains primary.
    text = (final_text or "").lower()
    expected = sample.get("expected_outcome")

    if expected == "success":
        if any(k in text for k in ["done", "created", "updated", "deleted", "scheduled", "found", "cancelled", "canceled"]):
            return True, "ok"
        return False, "missing success confirmation phrase"

    if expected == "missing_fields":
        if any(k in text for k in ["what", "which", "when", "missing", "need"]):
            return True, "ok"
        return False, "missing clarification question"

    if expected == "ambiguous":
        if any(k in text for k in ["which one", "choose", "select", "multiple"]):
            return True, "ok"
        return False, "missing disambiguation question"

    if expected == "not_found":
        if any(k in text for k in ["not found", "no matching", "couldn't find", "cannot find"]):
            return True, "ok"
        return False, "missing not_found explanation"

    if expected == "no_tool":
        if text.strip():
            return True, "ok"
        return False, "empty no_tool reply"

    return True, "unchecked"
