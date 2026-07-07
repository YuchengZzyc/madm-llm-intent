from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from dateutil import parser as dt_parser

from app.models import Reminder, ServiceResult, now_iso
from app.storage import JSONReminderStorage


class ReminderService:
    def __init__(self, storage: JSONReminderStorage) -> None:
        self.storage = storage

    def _next_id(self, reminders: list[dict[str, Any]]) -> str:
        if not reminders:
            return "rem_0001"
        nums = []
        for r in reminders:
            rid = r.get("reminder_id", "")
            if rid.startswith("rem_"):
                try:
                    nums.append(int(rid.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return f"rem_{(max(nums) + 1 if nums else 1):04d}"

    def _result(self, status: str, **kwargs: Any) -> dict[str, Any]:
        payload = ServiceResult(status=status, **kwargs).model_dump(exclude_none=True)
        payload["state"] = "success" if status == "success" else False
        return payload

    def _parse_time_text(self, time_text: str) -> str | None:
        text = time_text.strip()
        if not text:
            return None

        now = datetime.now().astimezone()
        low = text.lower()
        day_offset = 0
        if "tomorrow" in low:
            day_offset = 1
            low = low.replace("tomorrow", " ").strip()
        elif "today" in low:
            low = low.replace("today", " ").strip()

        low = low.replace("tonight", " ").strip()
        low = re.sub(r"\s+", " ", low).strip()

        if day_offset:
            base = now + timedelta(days=day_offset)
            if low:
                try:
                    parsed = dt_parser.parse(low, fuzzy=True, default=base)
                except (ValueError, TypeError):
                    return None
            else:
                parsed = base.replace(hour=21, minute=0, second=0, microsecond=0)
            return parsed.astimezone().isoformat(timespec="seconds")

        try:
            parsed = dt_parser.parse(text, fuzzy=True, default=now)
        except (ValueError, TypeError):
            return None
        return parsed.astimezone().isoformat(timespec="seconds")

    def create_reminder(self, time_text: str | None, task: str | None, target: str = "self") -> dict[str, Any]:
        missing = []
        if not time_text:
            missing.append("time_text")
        if not task:
            missing.append("task")
        if missing:
            return self._result(
                status="missing_fields",
                missing_fields=missing,
                message="Missing required fields: " + ", ".join(missing),
            )

        reminders = self.storage.load()
        ts = now_iso()
        parsed_time = self._parse_time_text(time_text)
        if not parsed_time:
            return self._result(
                status="missing_fields",
                missing_fields=["time_text"],
                message="Unable to parse time_text into a concrete datetime.",
            )

        reminder = Reminder(
            reminder_id=self._next_id(reminders),
            task=task,
            scheduled_time=parsed_time,
            time_text=time_text,
            target="self" if target != "self" else target,
            status="active",
            created_at=ts,
            updated_at=ts,
        )
        reminders.append(reminder.model_dump())
        self.storage.save(reminders)
        return self._result(
            status="success",
            reminder_id=reminder.reminder_id,
            reminder=reminder.model_dump(),
        )

    def _task_matches(self, query_task: str | None, reminder_task: str | None) -> bool:
        if not query_task:
            return True
        if not reminder_task:
            return False
        q = query_task.strip().lower()
        t = reminder_task.strip().lower()
        if q == t or q in t or t in q:
            return True

        def normalize_tokens(text: str) -> list[str]:
            toks = re.findall(r"[a-z0-9]+", text.lower())
            out: list[str] = []
            for tok in toks:
                base = tok
                if len(base) > 4 and base.endswith("ing"):
                    base = base[:-3]
                elif len(base) > 3 and base.endswith("ed"):
                    base = base[:-2]
                elif len(base) > 3 and base.endswith("s"):
                    base = base[:-1]
                out.append(base)
            return out

        q_tokens = normalize_tokens(q)
        t_tokens = normalize_tokens(t)
        if not q_tokens or not t_tokens:
            return False
        q_set = set(q_tokens)
        t_set = set(t_tokens)

        # Ignore very generic verbs/articles when deciding semantic match.
        stop = {"play", "do", "go", "have", "take", "make", "the", "a", "an", "my"}
        q_content = {x for x in q_set if x not in stop}
        t_content = {x for x in t_set if x not in stop}

        # If query has content tokens, require all of them to appear in reminder task.
        if q_content:
            return q_content.issubset(t_content)
        # Fallback for very short generic tasks: require at least one token overlap.
        return len(q_set & t_set) > 0

    def _time_matches(self, query_time_text: str | None, reminder: dict[str, Any]) -> bool:
        if not query_time_text:
            return True

        q = query_time_text.strip().lower()
        rt = str(reminder.get("time_text", "")).strip().lower()
        if q == rt or q in rt or rt in q:
            return True

        # Fallback: compare parsed day from query to reminder.scheduled_time day.
        parsed_query = self._parse_time_text(query_time_text)
        scheduled_raw = reminder.get("scheduled_time")
        if not parsed_query or not isinstance(scheduled_raw, str):
            return False
        try:
            query_dt = datetime.fromisoformat(parsed_query)
            sched_dt = datetime.fromisoformat(scheduled_raw)
        except ValueError:
            return False
        return query_dt.date() == sched_dt.date()

    def query_reminder(self, time_text: str | None = None, task: str | None = None, target: str = "self") -> dict[str, Any]:
        reminders = [r for r in self.storage.load() if r.get("status") == "active"]
        matches = [
            r for r in reminders
            if self._time_matches(time_text, r)
            and self._task_matches(task, r.get("task"))
            and r.get("target") == ("self" if target != "self" else target)
        ]
        if not matches:
            return self._result(status="not_found", message="No matching reminder found.")
        return self._result(status="success", reminders=matches)

    def _find_candidates(self, reminder_id: str | None = None, time_text: str | None = None, task: str | None = None, target: str = "self") -> list[dict[str, Any]]:
        reminders = [r for r in self.storage.load() if r.get("status") == "active"]
        if reminder_id:
            return [r for r in reminders if r.get("reminder_id") == reminder_id]
        return [
            r for r in reminders
            if self._time_matches(time_text, r)
            and self._task_matches(task, r.get("task"))
            and r.get("target") == ("self" if target != "self" else target)
        ]

    def delete_reminder(self, reminder_id: str | None = None, time_text: str | None = None, task: str | None = None, target: str = "self") -> dict[str, Any]:
        if not reminder_id and not time_text and not task:
            return self._result(
                status="missing_fields",
                missing_fields=["reminder_id|time_text|task"],
                message="Provide reminder_id or filter fields.",
            )

        candidates = self._find_candidates(reminder_id, time_text, task, target)
        if not candidates:
            return self._result(status="not_found", message="No matching reminder found.")
        if len(candidates) > 1:
            compact = [{"reminder_id": r["reminder_id"], "task": r["task"], "scheduled_time": r["scheduled_time"]} for r in candidates]
            return self._result(status="ambiguous", candidates=compact)

        reminders = self.storage.load()
        rid = candidates[0]["reminder_id"]
        ok = False
        for r in reminders:
            if r.get("reminder_id") == rid and r.get("status") == "active":
                r["status"] = "deleted"
                r["updated_at"] = now_iso()
                ok = True
                break
        self.storage.save(reminders)
        if not ok:
            return self._result(status="error", message="Failed to delete reminder.")
        return self._result(status="success", reminder_id=rid)

    def update_reminder(self, reminder_id: str | None = None, time_text: str | None = None, task: str | None = None, new_time_text: str | None = None, new_task: str | None = None, target: str = "self") -> dict[str, Any]:
        if not new_time_text and not new_task:
            return self._result(
                status="missing_fields",
                missing_fields=["new_time_text|new_task"],
                message="At least one update field is required.",
            )
        if not reminder_id and not time_text and not task:
            return self._result(
                status="missing_fields",
                missing_fields=["reminder_id|time_text|task"],
                message="Provide reminder_id or filter fields.",
            )

        candidates = self._find_candidates(reminder_id, time_text, task, target)
        if not candidates:
            return self._result(status="not_found", message="No matching reminder found.")
        if len(candidates) > 1:
            compact = [{"reminder_id": r["reminder_id"], "task": r["task"], "scheduled_time": r["scheduled_time"]} for r in candidates]
            return self._result(status="ambiguous", candidates=compact)

        reminders = self.storage.load()
        rid = candidates[0]["reminder_id"]
        updated = None
        for r in reminders:
            if r.get("reminder_id") == rid and r.get("status") == "active":
                if new_time_text:
                    parsed_time = self._parse_time_text(new_time_text)
                    if not parsed_time:
                        return self._result(
                            status="missing_fields",
                            missing_fields=["new_time_text"],
                            message="Unable to parse new_time_text into a concrete datetime.",
                        )
                    r["time_text"] = new_time_text
                    r["scheduled_time"] = parsed_time
                if new_task:
                    r["task"] = new_task
                r["updated_at"] = now_iso()
                updated = r
                break
        self.storage.save(reminders)
        if not updated:
            return self._result(status="error", message="Failed to update reminder.")
        return self._result(status="success", reminder_id=rid, reminder=updated)
