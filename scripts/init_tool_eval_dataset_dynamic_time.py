from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Singapore"

DAYPART_DEFAULTS = {
    "early morning": (7, 0),
    "morning": (9, 0),
    "noon": (12, 0),
    "afternoon": (15, 0),
    "evening": (19, 0),
    "night": (20, 0),
    "tonight": (20, 0),
    "before bed": (21, 30),
    "after breakfast": (9, 0),
    "before breakfast": (7, 0),
    "after lunch": (13, 0),
    "before lunch": (11, 30),
    "after dinner": (19, 30),
    "before dinner": (17, 30),
}

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().replace("，", ",").split())


def parse_clock(text: str) -> tuple[int, int] | None:
    t = normalize_text(text)
    # 8:15 PM / 8.15pm / 20:30
    m = re.search(r"\b(\d{1,2})(?::|\.)(\d{2})\s*(a\.m\.|p\.m\.|am|pm)?\b", t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        suffix = (m.group(3) or "").replace(".", "")
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return hour % 24, minute
    # 8 PM / 7am / at 9
    m = re.search(r"\b(\d{1,2})\s*(a\.m\.|p\.m\.|am|pm)\b", t)
    if m:
        hour = int(m.group(1))
        suffix = m.group(2).replace(".", "")
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return hour % 24, 0
    m = re.search(r"\bat\s+(\d{1,2})\b", t)
    if m:
        hour = int(m.group(1))
        # infer PM from daypart words, otherwise keep as written
        if any(x in t for x in ["evening", "tonight", "night", "afternoon", "dinner"]) and hour < 12:
            hour += 12
        return hour % 24, 0
    return None


def parse_daypart_clock(text: str) -> tuple[int, int]:
    t = normalize_text(text)
    explicit = parse_clock(t)
    if explicit:
        return explicit
    for key, value in DAYPART_DEFAULTS.items():
        if key in t:
            return value
    return 9, 0


def next_weekday(base_date: datetime, weekday: int, include_today: bool) -> datetime:
    days = (weekday - base_date.weekday()) % 7
    if days == 0 and not include_today:
        days = 7
    return base_date + timedelta(days=days)


def resolve_time_text(time_text: str | None, now: datetime) -> str | None:
    """Resolve natural relative time_text into ISO-8601 scheduled_time.

    This intentionally handles the controlled phrases emitted by
    generate_benchmark_prompts_structured.py. For arbitrary production-grade
    parsing, plug in your own parser here.
    """
    if not time_text:
        return None
    t = normalize_text(time_text)
    base = now

    if "day after tomorrow" in t:
        target = base + timedelta(days=2)
    elif "tomorrow" in t:
        target = base + timedelta(days=1)
    elif "today" in t or "tonight" in t:
        target = base
    else:
        matched_weekday = None
        for name, idx in WEEKDAY_INDEX.items():
            if re.search(rf"\b{name}\b", t):
                matched_weekday = idx
                break
        if matched_weekday is not None:
            include_today = "this" in t and "next" not in t
            target = next_weekday(base, matched_weekday, include_today=include_today)
            if "next" in t:
                # If base is Tuesday and text is next Monday, this gives the coming Monday.
                # If text is next Tuesday on Tuesday, it means seven days later.
                target = next_weekday(base, matched_weekday, include_today=False)
        else:
            # Routine anchors without day: choose today if still future, otherwise tomorrow.
            target = base

    hour, minute = parse_daypart_clock(t)
    resolved = target.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # Keep initialized reminders in the future when the text does not explicitly request a past date.
    if resolved <= now and not any(x in t for x in ["yesterday", "last "]):
        resolved = resolved + timedelta(days=1)
    return resolved.isoformat()


def reminder(reminder_id: str, task: str, time_text: str, now: datetime, status: str = "active") -> dict[str, Any]:
    scheduled_time = resolve_time_text(time_text, now)
    return {
        "reminder_id": reminder_id,
        "task": task,
        "scheduled_time": scheduled_time,
        "time_text": time_text,
        "target": "self",
        "status": status,
        "created_at": (now - timedelta(days=2, hours=1)).isoformat(),
        "updated_at": (now - timedelta(days=1)).isoformat(),
    }


def distractors(now: datetime) -> list[dict[str, Any]]:
    return [
        reminder("rem_d001", "water the orchids", "tomorrow morning", now),
        reminder("rem_d002", "charge the phone", "tonight at 9:00 PM", now),
    ]


def active_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for r in rows if r.get("status") == "active")


def build_initial_reminders(case: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    tool = str(case.get("tool") or "")
    expected = str(case.get("expected_outcome") or "")
    scenario = str(case.get("scenario") or "")
    gold = case.get("gold") or {}
    rows = distractors(now)

    def add_match(rid: str, task_key: str = "task", time_key: str = "time_text", fallback_task: str = "play basketball") -> None:
        task = gold.get(task_key) or gold.get("locator_task") or fallback_task
        time_text = gold.get(time_key) or gold.get("locator_time_text") or "tomorrow at 4:00 PM"
        rows.insert(0, reminder(rid, str(task), str(time_text), now))

    if expected == "success":
        if tool == "query":
            add_match("rem_0001", "task", "time_text")
        elif tool == "update":
            add_match("rem_0001", "locator_task", "locator_time_text")
        elif tool == "delete":
            add_match("rem_0001", "locator_task", "locator_time_text")
        # create success intentionally starts without the new reminder.
    elif expected == "ambiguous":
        # Create at least two reminders that can match the vague locator.
        base_task = gold.get("locator_task") or gold.get("task") or "take medicine"
        rows.insert(0, reminder("rem_0001", str(base_task), "tomorrow morning", now))
        rows.insert(1, reminder("rem_0002", str(base_task), "tomorrow evening", now))
    elif expected == "missing_fields":
        # Keep neutral distractors only; no target reminder should be mutated.
        pass
    elif expected == "not_found":
        # Keep distractors that intentionally do not match the gold locator.
        pass
    elif expected == "no_tool":
        pass

    # A scenario may want the classic basketball baseline; keep it when explicit.
    if "basketball" in scenario and not any(r["task"] == "play basketball" for r in rows):
        rows.append(reminder("rem_basketball", "play basketball", "tomorrow at 4:00 PM", now))
    return rows


def enrich_case(case: dict[str, Any], reminders_before: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    gold = dict(case.get("gold") or {})
    for key in ["time_text", "locator_time_text", "new_time_text"]:
        iso = resolve_time_text(gold.get(key), now)
        if iso:
            gold[key.replace("time_text", "scheduled_time")] = iso
    expected = dict(case.get("expected") or {})
    expected["before_active_count"] = active_count(reminders_before)
    expected["run_now"] = now.isoformat()
    expected["timezone"] = str(now.tzinfo)
    return {
        **case,
        "gold": gold,
        "initial_reminders": reminders_before,
        "expected": expected,
    }


def write_sqlite(path: Path, reminders: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE reminders (
                reminder_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                scheduled_time TEXT,
                time_text TEXT,
                target TEXT NOT NULL DEFAULT 'self',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        for r in reminders:
            conn.execute(
                """
                INSERT INTO reminders(reminder_id, task, scheduled_time, time_text, target, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["reminder_id"],
                    r["task"],
                    r.get("scheduled_time"),
                    r.get("time_text"),
                    r.get("target", "self"),
                    r.get("status", "active"),
                    r["created_at"],
                    r["updated_at"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def parse_now(value: str | None, tz: ZoneInfo) -> datetime:
    if value:
        raw = datetime.fromisoformat(value)
        if raw.tzinfo is None:
            return raw.replace(tzinfo=tz)
        return raw.astimezone(tz)
    return datetime.now(tz)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize reminder tool-use eval runs with dynamic scheduled_time resolution.")
    parser.add_argument("--cases-jsonl", type=Path, default=Path("benchmark/tool_eval_cases_structured.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/eval_runs"))
    parser.add_argument("--timezone", default=DEFAULT_TZ)
    parser.add_argument("--now", default=None, help="Optional ISO datetime. If omitted, uses real current time in --timezone.")
    parser.add_argument("--with-sqlite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolved-cases-jsonl", type=Path, default=None, help="Optional output with initial_reminders and resolved gold scheduled_time embedded.")
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    now = parse_now(args.now, tz)
    cases = load_jsonl(args.cases_jsonl)

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    resolved_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        case_dir = args.out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        reminders_before = build_initial_reminders(case, now)
        resolved_case = enrich_case(case, reminders_before, now)
        resolved_rows.append(resolved_case)

        (case_dir / "input.json").write_text(
            json.dumps({"id": case_id, "prompt": case.get("prompt"), "tool": case.get("tool"), "scenario": case.get("scenario")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "reminders_before.json").write_text(json.dumps(reminders_before, ensure_ascii=False, indent=2), encoding="utf-8")
        (case_dir / "expected.json").write_text(json.dumps({"gold": resolved_case["gold"], "expected": resolved_case["expected"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.with_sqlite:
            write_sqlite(case_dir / "reminders.db", reminders_before)

    if args.resolved_cases_jsonl:
        args.resolved_cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.resolved_cases_jsonl.open("w", encoding="utf-8") as f:
            for row in resolved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"initialized {len(cases)} cases -> {args.out_dir}")
    print(f"run_now={now.isoformat()} timezone={args.timezone}")


if __name__ == "__main__":
    main()
