from __future__ import annotations

import json
from typing import Any

from app.reminder_service import ReminderService
from app.tool_registry import get_tool
from app.models import ToolMessage


class ToolExecutor:
    def __init__(self, reminder_service: ReminderService) -> None:
        self.reminder_service = reminder_service
        self.tool_map = {
            "create_reminder": self.reminder_service.create_reminder,
            "query_reminder": self.reminder_service.query_reminder,
            "update_reminder": self.reminder_service.update_reminder,
            "delete_reminder": self.reminder_service.delete_reminder,
        }

    def _validate_required(self, schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        return [k for k in required if k not in args or args[k] in (None, "")]

    def _error_payload(self, status: str, message: str, missing_fields: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": False, "status": status, "message": message}
        if missing_fields is not None:
            payload["missing_fields"] = missing_fields
        return payload

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, str]:
        call_id = tool_call.get("id", "call_unknown")
        fn = tool_call.get("function", {})
        tool_name = fn.get("name")
        args_raw = fn.get("arguments", "{}")

        if tool_name not in self.tool_map:
            payload = self._error_payload(status="error", message=f"Unknown tool: {tool_name}")
            return ToolMessage(tool_call_id=call_id, name=tool_name or "unknown", content=json.dumps(payload)).model_dump()

        schema = get_tool(tool_name)
        if not schema:
            payload = self._error_payload(status="error", message=f"Schema not found for {tool_name}")
            return ToolMessage(tool_call_id=call_id, name=tool_name, content=json.dumps(payload)).model_dump()

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            payload = self._error_payload(status="error", message="Malformed JSON in tool arguments.")
            return ToolMessage(tool_call_id=call_id, name=tool_name, content=json.dumps(payload)).model_dump()

        missing = self._validate_required(schema, args)
        if missing:
            payload = self._error_payload(
                status="missing_fields",
                missing_fields=missing,
                message="Missing required field(s): " + ", ".join(missing),
            )
            return ToolMessage(tool_call_id=call_id, name=tool_name, content=json.dumps(payload)).model_dump()

        try:
            payload = self.tool_map[tool_name](**args)
        except TypeError as exc:
            payload = self._error_payload(status="error", message=f"Invalid tool arguments: {exc}")
        except Exception as exc:
            payload = self._error_payload(status="error", message=f"Tool execution error: {exc}")

        return ToolMessage(tool_call_id=call_id, name=tool_name, content=json.dumps(payload)).model_dump()

    def execute_tool_calls(self, assistant_message: dict[str, Any]) -> list[dict[str, str]]:
        return [self.execute_tool_call(tc) for tc in assistant_message.get("tool_calls", [])]
