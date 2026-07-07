from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {lineno}: expected JSON object")
            rows.append(obj)
    return rows


def require_gold(row: dict[str, Any], lineno: int) -> dict[str, Any]:
    gold = row.get("gold")
    if not isinstance(gold, dict):
        raise ValueError(
            f"line {lineno}: missing structured 'gold'. Regenerate prompts with generate_benchmark_prompts_structured.py"
        )
    return gold


def norm_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_expected(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    tool = str(row.get("tool") or "")
    expected_outcome = str(row.get("expected_outcome") or "")
    scenario = str(row.get("scenario") or "")
    should_call = bool(gold.get("should_call_tool", expected_outcome != "no_tool"))
    expected_tool = None if not should_call else {
        "create": "create_reminder",
        "query": "query_reminder",
        "update": "update_reminder",
        "delete": "delete_reminder",
    }.get(tool)

    if expected_outcome == "success":
        if tool == "create":
            state_change = "create_one"
            diff_assertions = [
                "after_active_count == before_active_count + 1",
                "new reminder.task == gold.task",
                "new reminder.target == 'self'",
                "new reminder.scheduled_time == resolve(gold.time_text, run_now)",
            ]
        elif tool == "query":
            state_change = "no_change"
            diff_assertions = [
                "after_state == before_state",
                "tool_result.status == 'success'",
                "tool_result.count >= 1",
            ]
        elif tool == "update":
            state_change = "update_one"
            diff_assertions = [
                "after_active_count == before_active_count",
                "exactly_one_existing_reminder_changed",
                "changed reminder matches gold.locator_task/time before update",
                "changed reminder contains gold.new_task and/or resolve(gold.new_time_text, run_now) after update",
            ]
        elif tool == "delete":
            state_change = "delete_or_cancel_one"
            diff_assertions = [
                "after_active_count == before_active_count - 1 OR target reminder.status in ['cancelled','deleted']",
                "deleted/cancelled reminder matches gold.locator_task/time_text",
            ]
        else:
            state_change = "no_change"
            diff_assertions = ["after_state == before_state"]
    elif expected_outcome in {"missing_fields", "ambiguous", "not_found"}:
        state_change = "no_change"
        diff_assertions = [
            "after_state == before_state",
            f"tool_result.status == '{expected_outcome}'",
        ]
    elif expected_outcome == "no_tool":
        state_change = "no_change"
        diff_assertions = [
            "no tool call should be emitted",
            "after_state == before_state",
        ]
    else:
        state_change = "unknown"
        diff_assertions = ["manual review required"]

    return {
        "should_call_tool": should_call,
        "expected_tool": expected_tool,
        "expected_status": expected_outcome,
        "state_change": state_change,
        "scenario": scenario,
        "diff_assertions": diff_assertions,
    }


def build_case(row: dict[str, Any], lineno: int) -> dict[str, Any]:
    gold = require_gold(row, lineno)
    normalized_gold = {
        "action": norm_or_none(gold.get("action")) or (str(row.get("tool")) if row.get("tool") != "no_tool" else "none"),
        "target": "self",
        "task": norm_or_none(gold.get("task")),
        "time_text": norm_or_none(gold.get("time_text")),
        "locator_task": norm_or_none(gold.get("locator_task")),
        "locator_time_text": norm_or_none(gold.get("locator_time_text")),
        "new_task": norm_or_none(gold.get("new_task")),
        "new_time_text": norm_or_none(gold.get("new_time_text")),
        "should_call_tool": bool(gold.get("should_call_tool", row.get("expected_outcome") != "no_tool")),
    }
    return {
        "id": str(row.get("id") or f"case_{lineno:05d}"),
        "tool": row.get("tool"),
        "scenario": row.get("scenario"),
        "expected_outcome": row.get("expected_outcome"),
        "prompt": row.get("prompt"),
        "gold": normalized_gold,
        "expected": build_expected(row, normalized_gold),
        "metadata": {
            "seed_utterance": row.get("seed_utterance"),
            "notes": row.get("notes"),
            "source_line": lineno,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build executable tool-use eval cases from structured prompt JSONL.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("benchmark/prompts_structured.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/tool_eval_cases_structured.jsonl"))
    args = parser.parse_args()

    rows = read_jsonl(args.prompts_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            case = build_case(row, idx)
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases -> {args.output}")


if __name__ == "__main__":
    main()
