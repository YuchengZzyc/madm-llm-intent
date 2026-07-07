from __future__ import annotations

import argparse
import gzip
import json
import os
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

from app.tool_registry import get_tools


SYSTEM_PROMPT = (
    "You are a reminder tool-call parser. Decide whether the user needs one of the available reminder tools. "
    "The only available tools are create_reminder, query_reminder, update_reminder, and delete_reminder. "
    "Use a tool only for explicit reminder create/query/update/delete requests. "
    "Never invent tool execution results. Your job is only to produce the assistant's next turn.\n\n"
    "Available tools and descriptions:\n{tools_json}\n\n"
    "Output rules:\n"
    "1. Return only one JSON object. Do not use markdown.\n"
    "2. If a tool is needed, return OpenAI-style assistant JSON with content:null and tool_calls.\n"
    "3. function.arguments must be a JSON string.\n"
    "4. If no tool is needed, return {{\"role\":\"assistant\",\"content\":\"...\"}} without tool_calls."
)


FEW_SHOT_EXAMPLES: list[tuple[str, dict[str, Any]]] = [
    (
        "Remind me tomorrow morning at 8 to take medicine.",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_example_create",
                    "type": "function",
                    "function": {
                        "name": "create_reminder",
                        "arguments": json.dumps(
                            {"time_text": "tomorrow morning at 8", "task": "take medicine", "target": "self"},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
    ),
    (
        "What reminders do I have tomorrow?",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_example_query",
                    "type": "function",
                    "function": {
                        "name": "query_reminder",
                        "arguments": json.dumps({"time_text": "tomorrow", "target": "self"}, ensure_ascii=False),
                    },
                }
            ],
        },
    ),
    (
        "Move my medicine reminder tomorrow morning to 9 AM.",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_example_update",
                    "type": "function",
                    "function": {
                        "name": "update_reminder",
                        "arguments": json.dumps(
                            {
                                "time_text": "tomorrow morning",
                                "task": "medicine",
                                "new_time_text": "tomorrow at 9 AM",
                                "target": "self",
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
    ),
    (
        "Delete my reminder to water the flowers.",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_example_delete",
                    "type": "function",
                    "function": {
                        "name": "delete_reminder",
                        "arguments": json.dumps({"task": "water the flowers", "target": "self"}, ensure_ascii=False),
                    },
                }
            ],
        },
    ),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise SystemExit("PyYAML is required when --config is used.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def pick_setting(name: str, *values: Any, required: bool = True, default: Any = None) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    if required:
        raise SystemExit(f"missing required setting: {name}")
    return default


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    return base


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int = 120,
        auth_header: str = "Authorization",
        auth_scheme: str = "bearer",
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            self.auth_header: self._auth_value(),
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read()
                if "gzip" in (response.headers.get("Content-Encoding") or "").lower() or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                raw = body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            if err_body[:2] == b"\x1f\x8b":
                err_body = gzip.decompress(err_body)
            raise RuntimeError(f"HTTPError status={exc.code}: {err_body.decode('utf-8', errors='replace')[:1200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"URLError: {exc}") from exc

        obj = json.loads(raw)
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"chat response has no choices: {raw[:1000]}")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError(f"chat response message is not an object: {raw[:1000]}")
        return message, raw

    def _auth_value(self) -> str:
        if self.auth_scheme == "raw":
            return self.api_key
        return f"Bearer {self.api_key}"


def build_messages(user_prompt: str, tools: list[dict[str, Any]], few_shot_count: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools_json=json.dumps(tools, ensure_ascii=False, indent=2)),
        }
    ]
    for user_text, assistant_obj in FEW_SHOT_EXAMPLES[: max(0, few_shot_count)]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def extract_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_tool_call_output(message: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
        parsed = {"role": "assistant", "content": None, "tool_calls": message["tool_calls"]}
        return parsed, json.dumps(message, ensure_ascii=False)

    content = message.get("content")
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    else:
        text = str(content or "")

    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        msg = dict(obj)
        msg["role"] = "assistant"
        msg["content"] = None
        return msg, text

    qwen_xml = re.search(
        r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        text,
        flags=re.S,
    )
    if qwen_xml:
        name = qwen_xml.group(1).strip()
        body = qwen_xml.group(2)
        arguments = {
            match.group(1).strip(): match.group(2).strip()
            for match in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, flags=re.S)
        }
        if name:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                    }
                ],
            }, text

    tagged = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    if tagged:
        try:
            call = json.loads(tagged.group(1))
        except Exception:
            call = None
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            arguments = call.get("arguments", {})
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }, text

    return None, text


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def compare_args(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[int, int, list[str]]:
    checked = 0
    ok = 0
    notes: list[str] = []
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        checked += 1
        if key not in actual:
            notes.append(f"missing:{key}")
            continue
        actual_value = actual.get(key)
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if _norm(expected_value) == _norm(actual_value):
                ok += 1
            else:
                notes.append(f"mismatch:{key}")
        elif expected_value == actual_value:
            ok += 1
        else:
            notes.append(f"mismatch:{key}")
    return ok, checked, notes


def evaluate_rows(
    rows: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    report_path: Path,
    temperature: float,
    max_tokens: int,
    few_shot_count: int,
    native_tools: bool,
    sleep_sec: float,
) -> dict[str, Any]:
    tools = get_tools()
    total = 0
    tool_call_correct = 0
    tool_name_correct = 0
    args_json_ok = 0
    arg_match_total = 0
    arg_match_ok = 0
    details: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        eval_spec = row.get("eval", {})
        should_call = bool(eval_spec.get("should_call_tool", True))
        expected_tool = eval_spec.get("expected_tool_name")
        expected_args = eval_spec.get("expected_args") or {}

        try:
            message, raw_response = client.chat(
                messages=build_messages(str(row.get("prompt", "")), tools, few_shot_count),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if native_tools else None,
            )
            tool_msg, raw_output = parse_tool_call_output(message)
            error = None
        except Exception as exc:
            message = {}
            raw_response = ""
            tool_msg = None
            raw_output = ""
            error = str(exc)

        got_call = tool_msg is not None and bool(tool_msg.get("tool_calls"))
        tool_ok = False
        name_ok = False
        args_ok = False
        arg_ok = 0
        arg_checked = 0
        arg_notes: list[str] = []
        pred_tool = None

        if should_call and got_call:
            tool_ok = True
            tool_call_correct += 1
            tool_call = tool_msg["tool_calls"][0]
            function = tool_call.get("function", {})
            pred_tool = function.get("name")
            if pred_tool == expected_tool:
                name_ok = True
                tool_name_correct += 1
            args_raw = function.get("arguments", "{}")
            try:
                actual_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if isinstance(actual_args, dict):
                    args_ok = True
                    args_json_ok += 1
                    arg_ok, arg_checked, arg_notes = compare_args(expected_args, actual_args)
                    arg_match_ok += arg_ok
                    arg_match_total += arg_checked
                else:
                    arg_notes.append("arguments_not_object")
            except Exception:
                arg_notes.append("arguments_json_invalid")

        if not should_call and not got_call:
            tool_ok = True
            tool_call_correct += 1

        details.append(
            {
                "id": row.get("id"),
                "scenario": row.get("scenario"),
                "prompt": row.get("prompt"),
                "should_call_tool": should_call,
                "expected_tool": expected_tool,
                "pred_tool": pred_tool,
                "tool_called": got_call,
                "tool_call_ok": tool_ok,
                "tool_name_ok": name_ok,
                "args_json_ok": args_ok,
                "arg_match_ok": arg_ok,
                "arg_match_checked": arg_checked,
                "arg_notes": arg_notes,
                "error": error,
                "raw_output_preview": raw_output[:800],
                "raw_response_preview": raw_response[:800],
            }
        )
        print(f"[{total}/{len(rows)}] {row.get('id')} tool={pred_tool} ok={tool_ok} args={arg_ok}/{arg_checked}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    summary = {
        "total": total,
        "tool_call_accuracy": (tool_call_correct / total) if total else 0.0,
        "tool_name_accuracy_overall": (tool_name_correct / total) if total else 0.0,
        "args_json_rate_overall": (args_json_ok / total) if total else 0.0,
        "arg_match_rate": (arg_match_ok / arg_match_total) if arg_match_total else 0.0,
        "few_shot_count": few_shot_count,
        "native_tools": native_tools,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reminder tool-call parameters through an OpenAI-compatible API.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--api-env", type=Path, default=Path("configs/data/api_generation.env"))
    parser.add_argument("--cases-jsonl", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--few-shot", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--auth-header", default=None)
    parser.add_argument("--auth-scheme", choices=["bearer", "raw"], default=None)
    parser.add_argument("--native-tools", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sleep-sec", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_cfg = cfg.get("api", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})
    env_file = load_env_file(args.api_env)

    base_url = pick_setting(
        "base_url",
        args.base_url,
        api_cfg.get("base_url"),
        env_file.get("DISTILL_API_BASE_URL"),
        env_file.get("OPENAI_BASE_URL"),
        os.getenv("DISTILL_API_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
    )
    api_key = pick_setting(
        "api_key",
        args.api_key,
        api_cfg.get("api_key"),
        env_file.get("DISTILL_API_KEY"),
        env_file.get("OPENAI_API_KEY"),
        os.getenv("DISTILL_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    )
    model = pick_setting(
        "model",
        args.model,
        api_cfg.get("model"),
        env_file.get("DISTILL_API_MODEL"),
        env_file.get("OPENAI_MODEL"),
        os.getenv("DISTILL_API_MODEL"),
        os.getenv("OPENAI_MODEL"),
    )
    cases_jsonl = Path(
        pick_setting(
            "cases_jsonl",
            args.cases_jsonl,
            data_cfg.get("cases_jsonl"),
            default="benchmark/toolcall_param_cases.jsonl",
            required=False,
        )
    )
    report = Path(
        pick_setting(
            "report",
            args.report,
            output_cfg.get("report"),
            default="benchmark/reports/toolcall_param_eval_api.json",
            required=False,
        )
    )
    temperature = float(pick_setting("temperature", args.temperature, eval_cfg.get("temperature"), default=0.0, required=False))
    max_tokens = int(pick_setting("max_tokens", args.max_tokens, eval_cfg.get("max_tokens"), default=512, required=False))
    max_samples = int(pick_setting("max_samples", args.max_samples, eval_cfg.get("max_samples"), default=0, required=False))
    few_shot = int(pick_setting("few_shot", args.few_shot, eval_cfg.get("few_shot"), default=4, required=False))
    timeout_sec = int(pick_setting("timeout_sec", args.timeout_sec, api_cfg.get("timeout_sec"), default=120, required=False))
    auth_header = str(pick_setting("auth_header", args.auth_header, api_cfg.get("auth_header"), default="Authorization", required=False))
    auth_scheme = str(pick_setting("auth_scheme", args.auth_scheme, api_cfg.get("auth_scheme"), default="bearer", required=False))
    native_tools = bool(pick_setting("native_tools", args.native_tools, api_cfg.get("native_tools"), default=False, required=False))
    sleep_sec = float(pick_setting("sleep_sec", args.sleep_sec, eval_cfg.get("sleep_sec"), default=0.0, required=False))

    rows = load_jsonl(cases_jsonl)
    if max_samples > 0:
        rows = rows[:max_samples]

    client = OpenAICompatibleClient(
        base_url=str(base_url),
        api_key=str(api_key),
        model=str(model),
        timeout_sec=timeout_sec,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
    )
    summary = evaluate_rows(
        rows=rows,
        client=client,
        report_path=report,
        temperature=temperature,
        max_tokens=max_tokens,
        few_shot_count=few_shot,
        native_tools=native_tools,
        sleep_sec=sleep_sec,
    )
    summary["base_url"] = normalize_base_url(str(base_url))
    summary["model"] = str(model)
    summary["cases_jsonl"] = str(cases_jsonl)
    summary["report"] = str(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
