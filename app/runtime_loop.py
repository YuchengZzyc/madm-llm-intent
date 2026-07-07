from __future__ import annotations

from typing import Any, Callable

from app.tool_executor import ToolExecutor
from app.tool_registry import get_tools


class RuntimeLoop:
    def __init__(self, model_callable: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]], executor: ToolExecutor, max_tool_rounds: int = 3) -> None:
        self.model_callable = model_callable
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds

    def run(self, system_prompt: str, history: list[dict[str, Any]], user_message: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}]
        tools = get_tools()

        for _ in range(self.max_tool_rounds):
            assistant = self.model_callable(messages, tools)
            messages.append(assistant)
            if not assistant.get("tool_calls"):
                return {"messages": messages, "final": assistant}
            messages.extend(self.executor.execute_tool_calls(assistant))

        return {
            "messages": messages,
            "final": {"role": "assistant", "content": "Tool loop stopped after max_tool_rounds to prevent infinite loop."},
        }
