from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_STATUS: dict[str, str] = {
    "create_success": "success",
    "create_missing_task": "missing_fields",
    "create_missing_time": "missing_fields",
    "query_success": "success",
    "query_not_found": "not_found",
    "query_vague_memory": "success",
    "update_success": "success",
    "update_ambiguous": "ambiguous",
    "update_missing": "missing_fields",
    "delete_success": "success",
    "delete_ambiguous": "ambiguous",
    "delete_missing": "missing_fields",
}

EXPECTED_TOOL: dict[str, str] = {
    "create_success": "create_reminder",
    "create_missing_task": "create_reminder",
    "create_missing_time": "create_reminder",
    "query_success": "query_reminder",
    "query_not_found": "query_reminder",
    "query_vague_memory": "query_reminder",
    "update_success": "update_reminder",
    "update_ambiguous": "update_reminder",
    "update_missing": "update_reminder",
    "delete_success": "delete_reminder",
    "delete_ambiguous": "delete_reminder",
    "delete_missing": "delete_reminder",
}


def _state_from_status(status: str) -> Any:
    return "success" if status == "success" else False


def _ensure_str(v: Any, default: str = "") -> str:
    if isinstance(v, str) and v.strip():
        return v
    return default


def _normalize_reminder_obj(rem: dict[str, Any], fallback_target: str = "self") -> dict[str, Any]:
    reminder_id = rem.get("reminder_id") or rem.get("id") or ""
    out: dict[str, Any] = {}
    if reminder_id:
        out["reminder_id"] = str(reminder_id)
    if "task" in rem:
        out["task"] = rem["task"]
    time_text = rem.get("time_text")
    if time_text is not None:
        out["time_text"] = time_text
    scheduled_time = rem.get("scheduled_time", rem.get("time"))
    if scheduled_time is not None:
        out["scheduled_time"] = scheduled_time
    out["target"] = rem.get("target", fallback_target) or "self"
    if "status" in rem:
        out["status"] = rem["status"]
    if "created_at" in rem:
        out["created_at"] = rem["created_at"]
    if "updated_at" in rem:
        out["updated_at"] = rem["updated_at"]
    return out


def _normalize_candidates(cands: Any) -> list[dict[str, Any]]:
    if not isinstance(cands, list):
        return []
    out: list[dict[str, Any]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        row: dict[str, Any] = {}
        rid = c.get("reminder_id") or c.get("id")
        if rid:
            row["reminder_id"] = rid
        if "task" in c:
            row["task"] = c["task"]
        st = c.get("scheduled_time", c.get("time", c.get("time_text")))
        if st is not None:
            row["scheduled_time"] = st
        out.append(row)
    return out


def normalize_payload(
    scenario: str,
    tool_name: str,
    tool_args: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        status = EXPECTED_STATUS.get(scenario, "error")
    expected_status = EXPECTED_STATUS.get(scenario)
    if expected_status and status != expected_status:
        raise ValueError(f"status mismatch: scenario={scenario}, got={status}, expected={expected_status}")

    out: dict[str, Any] = {"status": status, "state": _state_from_status(status)}

    if status == "success":
        if tool_name == "create_reminder":
            rem = payload.get("reminder")
            if not isinstance(rem, dict):
                rem = {}
            rem = _normalize_reminder_obj(rem, fallback_target=str(tool_args.get("target", "self")))
            reminder_id = payload.get("reminder_id") or rem.get("reminder_id")
            if not reminder_id:
                raise ValueError("create_success missing reminder_id")
            rem["reminder_id"] = reminder_id
            out["reminder_id"] = reminder_id
            out["reminder"] = rem
        elif tool_name == "query_reminder":
            rems = payload.get("reminders")
            if not isinstance(rems, list):
                raise ValueError("query_success missing reminders")
            out["reminders"] = [_normalize_reminder_obj(r if isinstance(r, dict) else {}) for r in rems]
        elif tool_name == "update_reminder":
            rem = payload.get("reminder")
            if not isinstance(rem, dict):
                rem = {}
            rem = _normalize_reminder_obj(rem, fallback_target=str(tool_args.get("target", "self")))
            reminder_id = payload.get("reminder_id") or rem.get("reminder_id")
            if not reminder_id:
                raise ValueError("update_success missing reminder_id")
            rem["reminder_id"] = reminder_id
            out["reminder_id"] = reminder_id
            out["reminder"] = rem
        elif tool_name == "delete_reminder":
            reminder_id = payload.get("reminder_id")
            if not reminder_id and isinstance(payload.get("deleted_reminder"), dict):
                reminder_id = payload["deleted_reminder"].get("reminder_id")
            if not reminder_id:
                raise ValueError("delete_success missing reminder_id")
            out["reminder_id"] = reminder_id
        else:
            raise ValueError(f"unknown tool for success: {tool_name}")
        return out

    if status == "missing_fields":
        mf = payload.get("missing_fields")
        if not isinstance(mf, list):
            mf = []
        out["missing_fields"] = mf
        out["message"] = _ensure_str(payload.get("message"), "Missing required field(s).")
        return out

    if status == "ambiguous":
        out["candidates"] = _normalize_candidates(payload.get("candidates"))
        if "message" in payload:
            out["message"] = _ensure_str(payload.get("message"), "Multiple reminders match.")
        return out

    if status == "not_found":
        out["message"] = _ensure_str(payload.get("message"), "No matching reminder found.")
        return out

    if status == "error":
        out["message"] = _ensure_str(payload.get("message"), "Tool execution error.")
        return out

    raise ValueError(f"unknown status: {status}")


def clean_record(obj: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    messages = obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be list")
    metadata = obj.get("metadata", {})
    scenario = metadata.get("scenario")

    assistant_call = None
    tool_msg = None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            m["content"] = None
            assistant_call = m
        if isinstance(m, dict) and m.get("role") == "tool":
            tool_msg = m

    if scenario in EXPECTED_TOOL:
        if not assistant_call or not tool_msg:
            raise ValueError("tool scenario missing assistant/tool message")
        tc = assistant_call["tool_calls"][0]
        tool_name = tc["function"]["name"]
        expected_tool = EXPECTED_TOOL[scenario]
        if tool_name != expected_tool:
            raise ValueError(f"tool mismatch: {tool_name} != {expected_tool}")
        args_raw = tc["function"].get("arguments", "{}")
        args = json.loads(args_raw) if isinstance(args_raw, str) else {}
        payload = json.loads(tool_msg.get("content", "{}"))
        fixed = normalize_payload(scenario, tool_name, args if isinstance(args, dict) else {}, payload if isinstance(payload, dict) else {})
        if fixed != payload:
            tool_msg["content"] = json.dumps(fixed, ensure_ascii=False)
            changed = True
    else:
        # no_tool: ensure no stray tool calls
        if assistant_call or tool_msg:
            raise ValueError("no_tool scenario contains tool messages")

    return obj, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean training_data_llm.jsonl to backend-consistent tool payloads.")
    parser.add_argument("--input", type=Path, default=Path("data/training_data_llm.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/training_data_llm.cleaned.jsonl"))
    parser.add_argument("--rejects", type=Path, default=Path("data/training_data_llm.rejects.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/training_data_llm.clean_report.json"))
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    rejects: list[str] = []
    changed = 0
    kept = 0
    bad = 0

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            fixed, was_changed = clean_record(obj)
            cleaned.append(json.dumps(fixed, ensure_ascii=False))
            kept += 1
            if was_changed:
                changed += 1
        except Exception as exc:
            bad += 1
            rejects.append(json.dumps({"line": idx, "error": str(exc), "raw": line}, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")
    args.rejects.write_text("\n".join(rejects) + ("\n" if rejects else ""), encoding="utf-8")
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "rejects": str(args.rejects),
        "total_lines": len([x for x in lines if x.strip()]),
        "kept": kept,
        "changed": changed,
        "rejected": bad,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

