from __future__ import annotations

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder from user time expression and task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_text": {"type": "string", "description": "Original user time expression."},
                    "task": {"type": "string", "description": "Reminder task."},
                    "target": {"type": "string", "enum": ["self"], "default": "self"},
                },
                "required": ["time_text", "task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_reminder",
            "description": "Query reminders by optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_text": {"type": "string"},
                    "task": {"type": "string"},
                    "target": {"type": "string", "enum": ["self"], "default": "self"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_reminder",
            "description": "Update reminder time or task by id/filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "string"},
                    "time_text": {"type": "string"},
                    "task": {"type": "string"},
                    "new_time_text": {"type": "string"},
                    "new_task": {"type": "string"},
                    "target": {"type": "string", "enum": ["self"], "default": "self"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reminder",
            "description": "Delete reminder by id/filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "string"},
                    "time_text": {"type": "string"},
                    "task": {"type": "string"},
                    "target": {"type": "string", "enum": ["self"], "default": "self"},
                },
                "additionalProperties": False,
            },
        },
    },
]

TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def get_tools() -> list[dict]:
    return TOOLS


def get_tool(name: str) -> dict | None:
    return TOOL_BY_NAME.get(name)
