from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTION_TO_TOOL = {
    "create": "create_reminder",
    "query": "query_reminder",
    "update": "update_reminder",
    "delete": "delete_reminder",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_expected_args(gold: dict[str, Any]) -> dict[str, Any]:
    action = gold.get("action")
    target = gold.get("target", "self")

    if action == "create":
        out = {"target": target}
        if gold.get("time_text") is not None:
            out["time_text"] = gold.get("time_text")
        if gold.get("task") is not None:
            out["task"] = gold.get("task")
        return out

    if action == "query":
        out = {"target": target}
        if gold.get("time_text") is not None:
            out["time_text"] = gold.get("time_text")
        if gold.get("task") is not None:
            out["task"] = gold.get("task")
        return out

    if action == "update":
        out = {"target": target}
        if gold.get("locator_time_text") is not None:
            out["time_text"] = gold.get("locator_time_text")
        if gold.get("locator_task") is not None:
            out["task"] = gold.get("locator_task")
        if gold.get("new_time_text") is not None:
            out["new_time_text"] = gold.get("new_time_text")
        if gold.get("new_task") is not None:
            out["new_task"] = gold.get("new_task")
        return out

    if action == "delete":
        out = {"target": target}
        if gold.get("locator_time_text") is not None:
            out["time_text"] = gold.get("locator_time_text")
        if gold.get("locator_task") is not None:
            out["task"] = gold.get("locator_task")
        return out

    return {}


def build_case(row: dict[str, Any]) -> dict[str, Any]:
    gold = row.get("gold") or {}
    action = gold.get("action")
    should_call_tool = bool(gold.get("should_call_tool", True))
    expected_tool = ACTION_TO_TOOL.get(action) if should_call_tool else None
    expected_args = build_expected_args(gold) if should_call_tool else {}

    return {
        "id": row.get("id"),
        "scenario": row.get("scenario"),
        "prompt": row.get("prompt"),
        "tool": row.get("tool"),
        "expected_outcome": row.get("expected_outcome"),
        "gold": gold,
        "eval": {
            "should_call_tool": should_call_tool,
            "expected_tool_name": expected_tool,
            "expected_args": expected_args,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tool-call parameter evaluation cases from structured prompts JSONL.")
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.prompts_jsonl)
    cases = [build_case(r) for r in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + ("\n" if cases else ""),
        encoding="utf-8",
    )
    print(json.dumps({"input": str(args.prompts_jsonl), "output": str(args.output), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

