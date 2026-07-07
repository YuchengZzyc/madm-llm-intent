from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor
from app.tool_registry import get_tools


SYSTEM_PROMPT = (
    "You are XiaoNuan, a warm and patient companion assistant for older adults. "
    "You may use reminder tools only when the user explicitly asks to create, query, update, or delete reminders. "
    "Do not call tools for normal chatting, emotional support, life sharing, or statements that merely mention time. "
    "Do not fabricate tool results. You can confirm success only after receiving a role=tool result. "
    "If the tool returns missing_fields, ask naturally for the missing information. "
    "If the tool returns ambiguous, ask the user to choose the specific reminder. "
    "If the tool returns not_found, explain that no matching reminder was found and offer a next step."
)

SCENARIOS = [
    "create_success",
    "create_missing_task",
    "create_missing_time",
    "query_success",
    "query_not_found",
    "query_vague_memory",
    "update_success",
    "update_ambiguous",
    "update_missing",
    "delete_success",
    "delete_ambiguous",
    "delete_missing",
    "no_tool_daily_chat",
    "no_tool_emotional_support",
    "no_tool_happy_share",
    "no_tool_time_word_but_no_reminder",
]

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


def load_env_file(path: str | Path) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}

    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_required(name: str, values: list[str | None]) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"missing required setting: {name}")


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        endpoint: str = "chat.completions",
        timeout: int = 120,
        auth_header: str = "Authorization",
        auth_scheme: str = "bearer",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme

    def _auth_value(self) -> str:
        if self.auth_scheme == "raw":
            return self.api_key
        return f"Bearer {self.api_key}"

    def chat(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        if self.endpoint == "responses":
            url = f"{self.base_url}/responses"
            payload = {
                "model": self.model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            self.auth_header: self._auth_value(),
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if "gzip" in content_encoding or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                raw = body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = e.read()
            if err_body[:2] == b"\x1f\x8b":
                err_body = gzip.decompress(err_body)
            err_text = err_body.decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTPError: status={e.code}, body={err_text[:1000]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"URLError: {e}") from e

        if not raw.strip():
            raise RuntimeError(
                f"Empty response body from model endpoint. "
                f"Check base_url/model/provider compatibility. endpoint={self.endpoint}, model={self.model}"
            )

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Model endpoint returned non-JSON response. "
                f"error={e}; raw_preview={raw[:1000]!r}"
            ) from e

        # OpenAI-compatible chat.completions
        if isinstance(obj.get("choices"), list) and obj["choices"]:
            message = obj["choices"][0].get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                joined = "\n".join(parts).strip()
                if joined:
                    return joined

        # OpenAI responses API
        if isinstance(obj.get("output_text"), str) and obj["output_text"].strip():
            return obj["output_text"].strip()

        if isinstance(obj.get("output"), list):
            parts: list[str] = []
            for block in obj["output"]:
                if not isinstance(block, dict):
                    continue
                for content in block.get("content", []) or []:
                    if isinstance(content, dict):
                        text = content.get("text")
                        if isinstance(text, str):
                            parts.append(text)
            joined = "\n".join(parts).strip()
            if joined:
                return joined

        raise RuntimeError(
            f"Model response did not contain readable text. keys={list(obj.keys())[:20]}, raw={raw[:1000]}"
        )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model output")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("model output JSON is not an object")
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text[:800]}")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("extracted JSON is not an object")
    return obj


def choose_scenario() -> str:
    weights = {
        "create_success": 0.16,
        "create_missing_task": 0.05,
        "create_missing_time": 0.05,
        "query_success": 0.12,
        "query_not_found": 0.06,
        "query_vague_memory": 0.08,
        "update_success": 0.08,
        "update_ambiguous": 0.05,
        "update_missing": 0.05,
        "delete_success": 0.08,
        "delete_ambiguous": 0.05,
        "delete_missing": 0.04,
        "no_tool_daily_chat": 0.06,
        "no_tool_emotional_support": 0.05,
        "no_tool_happy_share": 0.04,
        "no_tool_time_word_but_no_reminder": 0.08,
    }
    return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def build_prompt(scenario: str, tools: list[dict[str, Any]], recent_user_texts: list[str]) -> list[dict[str, Any]]:
    recent = "\n".join(f"- {x}" for x in recent_user_texts[-20:]) or "none"
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)

    generator_system = """You are a high-quality SFT data generator for tool-use training.

Return exactly one valid JSON object as plain text. Do not use markdown. Do not explain. Do not wrap it in code fences.

All user-facing text must be English.

You generate the final training sample directly. Do not generate a blueprint.

The target format is:
{
  "messages": [...],
  "tools": [...],
  "metadata": {...}
}

The script will overwrite the top-level "tools" with the real get_tools() output, but you must still follow the provided schema.

Hard rules:
1. Tool-use sample order must be:
   system -> user -> assistant.tool_calls -> tool -> assistant final reply
2. assistant.tool_calls must be immediately followed by role=tool.
3. No assistant natural-language text may appear between assistant.tool_calls and role=tool.
4. The role=tool content must be a JSON string, not an object.
5. The role=tool content JSON must include a "status" field.
6. No-tool samples must not contain tool_calls and must not contain role=tool.
7. The first system message content must be exactly the system prompt provided by the user prompt.
8. The assistant must not say an operation succeeded before role=tool.
"""

    user_prompt = f"""Generate one complete training sample for scenario: {scenario}.

System prompt to use as messages[0].content:
{SYSTEM_PROMPT}

Available tools schema:
{tools_json}

Recent user expressions to avoid repeating:
{recent}

Top-level metadata:
{{
  "scenario": "{scenario}",
  "selected_tool": "create_reminder | query_reminder | update_reminder | delete_reminder | null",
  "source": "direct_llm_final_sample"
}}

Scenario requirements:

create_success:
- User clearly asks to create a reminder.
- assistant.tool_calls calls create_reminder with time_text, task, target="self".
- role=tool content has status="success" and realistic reminder fields.
- Final assistant confirms warmly.

create_missing_task:
- User gives time and reminder intent, but no task.
- assistant.tool_calls calls create_reminder without task.
- role=tool content has status="missing_fields", missing_fields includes "task".
- Final assistant asks what to remind them about.

create_missing_time:
- User gives task but no time.
- assistant.tool_calls calls create_reminder without time_text.
- role=tool content has status="missing_fields", missing_fields includes "time_text".
- Final assistant asks when to remind them.

query_success:
- User asks to query reminders.
- assistant.tool_calls calls query_reminder.
- role=tool content has status="success", count, and reminders list with 1-3 reminders.
- Final assistant summarizes specific reminders.

query_not_found:
- User asks to query reminders.
- assistant.tool_calls calls query_reminder.
- role=tool content has status="not_found", count=0, reminders=[].
- Final assistant says no matching reminder was found and offers a next step.

query_vague_memory:
- User has fuzzy memory, e.g. "I forgot what I need to do tomorrow".
- assistant.tool_calls calls query_reminder with available fuzzy condition.
- role=tool content has status="success" and reminders.
- Final assistant gently summarizes.

update_success:
- User clearly asks to update a reminder.
- assistant.tool_calls calls update_reminder with locator plus new_time_text or new_task.
- role=tool content has status="success".
- Final assistant confirms update.

update_ambiguous:
- User asks to update but locator matches multiple reminders.
- assistant.tool_calls calls update_reminder.
- role=tool content has status="ambiguous" and candidates.
- Final assistant asks which one they mean.

update_missing:
- User asks to update but lacks locator or new value.
- assistant.tool_calls calls update_reminder with only the information actually given.
- role=tool content has status="missing_fields".
- Final assistant asks for missing details.

delete_success:
- User clearly asks to delete a reminder.
- assistant.tool_calls calls delete_reminder with locator information.
- role=tool content has status="success".
- Final assistant confirms deletion.

delete_ambiguous:
- User deletion request is ambiguous.
- assistant.tool_calls calls delete_reminder.
- role=tool content has status="ambiguous" and candidates.
- Final assistant asks which one to delete.

delete_missing:
- User wants to delete a reminder but does not identify which.
- assistant.tool_calls calls delete_reminder with only target="self".
- role=tool content has status="missing_fields".
- Final assistant asks which reminder to delete.

no_tool_daily_chat:
- Casual daily chat only.
- No tool call.
- Warm natural reply.

no_tool_emotional_support:
- User expresses loneliness, worry, sadness, or anxiety.
- No tool call.
- Supportive reply.

no_tool_happy_share:
- User shares good news.
- No tool call.
- Positive reply.

no_tool_time_word_but_no_reminder:
- User includes time words such as tomorrow, tonight, or this afternoon, but does not ask for a reminder.
- No tool call.
- Natural reply.

Use diverse realistic older-adult daily-life content:
medicine, blood pressure, blood sugar, doctor appointment, calling daughter/son/grandchild, groceries, utility bills, closing windows, pets, walking, community activities, vague memory, and emotional support.

Return JSON object only.
"""

    return [
        {"role": "system", "content": generator_system},
        {"role": "user", "content": user_prompt},
    ]


def normalize_sample(sample: dict[str, Any], tools: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError("sample must be a JSON object")

    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("sample.messages must be a list")

    # Normalize assistant tool-call messages to explicit OpenAI format.
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg["content"] = None

    # Always force real tools from current project.
    sample["tools"] = copy.deepcopy(tools)

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["scenario"] = scenario
    metadata.setdefault("source", "direct_llm_final_sample")

    # Infer selected_tool if missing.
    selected_tool = metadata.get("selected_tool")
    if selected_tool is None:
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                try:
                    selected_tool = msg["tool_calls"][0]["function"]["name"]
                except Exception:
                    selected_tool = None
                break
    metadata["selected_tool"] = selected_tool

    sample["metadata"] = metadata
    return sample


def _build_local_executor() -> ToolExecutor:
    tmp_dir = ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    storage_path = tmp_dir / f"llm2_exec_{uuid.uuid4().hex}.json"
    storage_path.write_text("[]", encoding="utf-8")
    storage = JSONReminderStorage(storage_path)
    service = ReminderService(storage)
    return ToolExecutor(service)


def _preseed_for_scenario(executor: ToolExecutor, scenario: str, tool_name: str, args: dict[str, Any]) -> None:
    service = executor.reminder_service
    if scenario in {"query_success", "query_vague_memory"} and tool_name == "query_reminder":
        service.create_reminder(time_text=args.get("time_text") or "tomorrow morning", task=args.get("task") or "take medicine", target="self")
        return
    if scenario == "update_success" and tool_name == "update_reminder":
        service.create_reminder(time_text=args.get("time_text") or "tomorrow morning", task=args.get("task") or "walk", target="self")
        return
    if scenario == "delete_success" and tool_name == "delete_reminder":
        service.create_reminder(time_text=args.get("time_text") or "tomorrow morning", task=args.get("task") or "walk", target="self")
        return
    if scenario == "update_ambiguous" and tool_name == "update_reminder":
        tt = args.get("time_text") or "tomorrow morning"
        tk = args.get("task") or "walk"
        service.create_reminder(time_text=tt, task=tk, target="self")
        service.create_reminder(time_text=tt, task=tk, target="self")
        return
    if scenario == "delete_ambiguous" and tool_name == "delete_reminder":
        tt = args.get("time_text") or "tomorrow morning"
        tk = args.get("task") or "walk"
        service.create_reminder(time_text=tt, task=tk, target="self")
        service.create_reminder(time_text=tt, task=tk, target="self")
        return


def inject_local_tool_result(sample: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario.startswith("no_tool"):
        return sample

    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("sample.messages must be list")

    assistant_idx = None
    assistant_msg = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_idx = i
            assistant_msg = msg
            break
    if assistant_idx is None or not isinstance(assistant_msg, dict):
        raise ValueError("tool scenario missing assistant tool_calls message")

    tool_calls = assistant_msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("assistant tool_calls must be non-empty list")
    tool_call = tool_calls[0]
    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool_name = fn.get("name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_call.function.name missing")
    args_raw = fn.get("arguments", "{}")
    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    if not isinstance(args, dict):
        raise ValueError("tool args must be object")

    executor = _build_local_executor()
    _preseed_for_scenario(executor, scenario=scenario, tool_name=tool_name, args=args)
    tool_msg = executor.execute_tool_call(tool_call if isinstance(tool_call, dict) else {})

    # Replace the first immediate tool message after assistant.tool_calls; insert if absent.
    if assistant_idx + 1 < len(messages) and isinstance(messages[assistant_idx + 1], dict) and messages[assistant_idx + 1].get("role") == "tool":
        messages[assistant_idx + 1] = tool_msg
    else:
        messages.insert(assistant_idx + 1, tool_msg)

    sample["messages"] = messages
    return sample


def validate_sample(sample: dict[str, Any], scenario: str) -> None:
    messages = sample["messages"]

    if len(messages) < 3:
        raise ValueError("sample must have at least 3 messages")

    if messages[0].get("role") != "system":
        raise ValueError("first message must be system")
    if messages[0].get("content") != SYSTEM_PROMPT:
        raise ValueError("system prompt does not match expected SYSTEM_PROMPT")
    if messages[1].get("role") != "user":
        raise ValueError("second message must be user")
    if not isinstance(messages[1].get("content"), str) or not messages[1]["content"].strip():
        raise ValueError("user content missing")

    has_tool_call = False
    has_tool_msg = False
    called_tool_name: str | None = None
    tool_statuses: list[str] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid role: {role}")

        if role == "assistant" and msg.get("tool_calls"):
            has_tool_call = True

            if "content" not in msg:
                raise ValueError("assistant tool call message must include content field")
            if msg.get("content") is not None:
                raise ValueError("assistant tool call message content must be null")

            if i + 1 >= len(messages) or messages[i + 1].get("role") != "tool":
                raise ValueError("assistant.tool_calls must be followed immediately by role=tool")

            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                raise ValueError("tool_calls must be a non-empty list")

            for tool_call in tool_calls:
                if tool_call.get("type") != "function":
                    raise ValueError("tool_call.type must be function")
                fn = tool_call.get("function")
                if not isinstance(fn, dict):
                    raise ValueError("tool_call.function must be object")
                if not isinstance(fn.get("name"), str) or not fn["name"].strip():
                    raise ValueError("tool_call.function.name missing")
                called_tool_name = fn["name"].strip()
                args_text = fn.get("arguments")
                if not isinstance(args_text, str):
                    raise ValueError("tool_call.function.arguments must be JSON string")
                json.loads(args_text)

        if role == "tool":
            has_tool_msg = True
            if not isinstance(msg.get("content"), str):
                raise ValueError("tool content must be JSON string")
            payload = json.loads(msg["content"])
            if "status" not in payload:
                raise ValueError("tool payload must include status")
            if not isinstance(payload["status"], str) or not payload["status"].strip():
                raise ValueError("tool payload status must be non-empty string")
            tool_statuses.append(payload["status"])

    if scenario.startswith("no_tool"):
        if has_tool_call or has_tool_msg:
            raise ValueError("no_tool scenario must not contain tool_calls or role=tool")
    else:
        if not has_tool_call or not has_tool_msg:
            raise ValueError("tool scenario must contain assistant.tool_calls and role=tool")
        expected_tool = EXPECTED_TOOL.get(scenario)
        if expected_tool and called_tool_name != expected_tool:
            raise ValueError(
                f"scenario={scenario} expected tool={expected_tool}, got tool={called_tool_name}"
            )
        expected_status = EXPECTED_STATUS.get(scenario)
        if expected_status:
            if len(tool_statuses) != 1:
                raise ValueError(
                    f"scenario={scenario} expects exactly one tool status, got {tool_statuses}"
                )
            actual_status = tool_statuses[0]
            if actual_status != expected_status:
                raise ValueError(
                    f"scenario={scenario} expected status={expected_status}, got status={actual_status}"
                )


def generate_sample(
    client: OpenAICompatibleClient,
    tools: list[dict[str, Any]],
    scenario: str,
    recent_user_texts: list[str],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    prompt = build_prompt(scenario=scenario, tools=tools, recent_user_texts=recent_user_texts)
    last_err: Exception | None = None
    last_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            text = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
            last_text = text
            sample = extract_json_object(text)
            sample = normalize_sample(sample, tools=tools, scenario=scenario)
            sample = inject_local_tool_result(sample, scenario=scenario)
            validate_sample(sample, scenario=scenario)
            return sample
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"failed to generate sample for scenario={scenario}: {last_err}; raw_preview={last_text[:1000]!r}"
    )


def export_jsonl(
    output_path: Path,
    n: int,
    seed: int,
    client: OpenAICompatibleClient,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    append_output: bool,
    save_raw: bool,
) -> int:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not append_output:
        output_path.write_text("", encoding="utf-8")

    raw_path = output_path.with_suffix(".raw.jsonl")
    if save_raw and not append_output:
        raw_path.write_text("", encoding="utf-8")

    tools = get_tools()
    stats: dict[str, int] = {}
    recent_user_texts: list[str] = []

    with output_path.open("a", encoding="utf-8") as fout:
        raw_file = raw_path.open("a", encoding="utf-8") if save_raw else None
        try:
            for idx in range(1, n + 1):
                scenario = choose_scenario()
                sample = generate_sample(
                    client=client,
                    tools=tools,
                    scenario=scenario,
                    recent_user_texts=recent_user_texts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                )
                sample["id"] = f"reminder_tooluse_{idx:06d}"

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()

                if raw_file:
                    raw_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    raw_file.flush()

                user_text = sample["messages"][1]["content"]
                recent_user_texts.append(user_text)
                stats[scenario] = stats.get(scenario, 0) + 1

                print(f"[{idx}/{n}] {scenario} | {user_text}")
        finally:
            if raw_file:
                raw_file.close()

    output_path.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct LLM generator for reminder tool-use SFT data.")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--output", type=Path, default=Path("data/training_data_llm.jsonl"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--auth-header", default="Authorization")
    parser.add_argument("--auth-scheme", choices=["bearer", "raw"], default="bearer")
    parser.add_argument("--append-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    env_file = load_env_file(args.api_env)

    try:
        base_url = resolve_required(
            "base_url",
            [
                args.base_url,
                env_file.get("DISTILL_API_BASE_URL"),
                env_file.get("OPENAI_BASE_URL"),
                os.getenv("DISTILL_API_BASE_URL"),
                os.getenv("OPENAI_BASE_URL"),
            ],
        )
        api_key = resolve_required(
            "api_key",
            [
                args.api_key,
                env_file.get("DISTILL_API_KEY"),
                env_file.get("OPENAI_API_KEY"),
                os.getenv("DISTILL_API_KEY"),
                os.getenv("OPENAI_API_KEY"),
            ],
        )
        model = resolve_required(
            "model",
            [
                args.model,
                env_file.get("DISTILL_API_MODEL"),
                env_file.get("OPENAI_MODEL"),
                os.getenv("DISTILL_API_MODEL"),
                os.getenv("OPENAI_MODEL"),
            ],
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=args.endpoint,
        timeout=args.timeout_sec,
        auth_header=args.auth_header,
        auth_scheme=args.auth_scheme,
    )

    count = export_jsonl(
        output_path=args.output,
        n=args.n,
        seed=args.seed,
        client=client,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        append_output=args.append_output,
        save_raw=args.save_raw,
    )

    print(f"Exported {count} samples to {args.output}")
    print(f"Stats written to {args.output.with_suffix('.stats.json')}")
    print(f"api_env={Path(args.api_env).resolve()}")
    print(f"base_url={base_url}")
    print(f"model={model}")
    print(f"endpoint={args.endpoint}")


if __name__ == "__main__":
    main()
