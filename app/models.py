from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

ResultStatus = Literal["success", "missing_fields", "not_found", "ambiguous", "error"]


class Reminder(BaseModel):
    reminder_id: str
    task: str
    scheduled_time: str
    time_text: str
    target: Literal["self"] = "self"
    status: Literal["active", "deleted"] = "active"
    created_at: str
    updated_at: str


class ServiceResult(BaseModel):
    status: ResultStatus
    message: str | None = None
    missing_fields: list[str] | None = None
    candidates: list[dict[str, Any]] | None = None
    reminder: dict[str, Any] | None = None
    reminders: list[dict[str, Any]] | None = None
    reminder_id: str | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
