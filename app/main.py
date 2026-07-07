from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor
from app.tool_registry import get_tools


class CreateReminderRequest(BaseModel):
    time_text: str | None = None
    task: str | None = None
    target: str = "self"


class UpdateReminderRequest(BaseModel):
    new_time_text: str | None = None
    new_task: str | None = None


class ExecuteToolsRequest(BaseModel):
    assistant_message: dict[str, Any]


def create_app(storage_path: str | Path = "data/reminders.json") -> FastAPI:
    app = FastAPI(title="Reminder Tool Harness")

    storage = JSONReminderStorage(storage_path)
    service = ReminderService(storage)
    executor = ToolExecutor(service)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    def tools() -> list[dict[str, Any]]:
        return get_tools()

    @app.post("/tools/execute")
    def execute_tools(req: ExecuteToolsRequest) -> dict[str, Any]:
        messages = executor.execute_tool_calls(req.assistant_message)
        return {"tool_messages": messages}

    @app.post("/reminders")
    def create_reminder(req: CreateReminderRequest) -> dict[str, Any]:
        return service.create_reminder(time_text=req.time_text, task=req.task, target=req.target)

    @app.get("/reminders")
    def query_reminders(time_text: str | None = None, task: str | None = None, target: str = "self") -> dict[str, Any]:
        return service.query_reminder(time_text=time_text, task=task, target=target)

    @app.patch("/reminders/{reminder_id}")
    def update_reminder(reminder_id: str, req: UpdateReminderRequest) -> dict[str, Any]:
        return service.update_reminder(
            reminder_id=reminder_id,
            new_time_text=req.new_time_text,
            new_task=req.new_task,
            target="self",
        )

    @app.delete("/reminders/{reminder_id}")
    def delete_reminder(reminder_id: str) -> dict[str, Any]:
        return service.delete_reminder(reminder_id=reminder_id, target="self")

    return app


app = create_app()
