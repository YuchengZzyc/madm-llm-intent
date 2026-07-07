from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark_eval import compute_storage_delta, judge_answer_accuracy, judge_tool_success, load_jsonl, save_json
from app.reminder_service import ReminderService
from app.runtime_loop import RuntimeLoop
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor

SYSTEM_PROMPT = (
    "You are a reliable assistant for reminder tool use. "
    "Use reminder tools when needed. Never fabricate tool results."
)


class HFAssistant:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 256) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        else:
            prompt = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # minimal parser: accept either plain text or OpenAI-style assistant JSON with tool_calls
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                obj.setdefault("role", "assistant")
                obj["content"] = None
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("content"), str):
                return {"role": "assistant", "content": obj["content"]}
        except Exception:
            pass

        return {"role": "assistant", "content": text}


def build_initial_reminders(scenario: str, tool: str, expected_outcome: str) -> list[dict[str, Any]]:
    # Minimal deterministic presets; each sample runs in isolated storage.
    if expected_outcome == "no_tool":
        return []

    base = [
        {
            "reminder_id": "rem_0001",
            "task": "play basketball",
            "scheduled_time": "2026-05-21T18:00:00+08:00",
            "time_text": "today 6pm",
            "target": "self",
            "status": "active",
            "created_at": "2026-05-20T10:00:00+08:00",
            "updated_at": "2026-05-20T10:00:00+08:00",
        }
    ]

    if expected_outcome == "ambiguous" and tool in {"update", "delete"}:
        base.append(
            {
                "reminder_id": "rem_0002",
                "task": "play basketball",
                "scheduled_time": "2026-05-22T08:00:00+08:00",
                "time_text": "tomorrow 8am",
                "target": "self",
                "status": "active",
                "created_at": "2026-05-20T10:00:00+08:00",
                "updated_at": "2026-05-20T10:00:00+08:00",
            }
        )

    if expected_outcome == "not_found":
        for r in base:
            r["task"] = "pay water bill"

    return base


def extract_last_tool_status(messages: list[dict[str, Any]]) -> str | None:
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    if not tool_msgs:
        return None
    try:
        payload = json.loads(tool_msgs[-1].get("content", "{}"))
        return payload.get("status")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark prompts against a trained checkpoint and score tool-use correctness.")
    parser.add_argument("--prompts", type=Path, default=Path("benchmark/prompts.jsonl"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--report", type=Path, default=Path("benchmark/report.json"))
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    rows = load_jsonl(args.prompts)[: args.max_samples]
    model = HFAssistant(model_path=args.model_path, adapter_path=args.adapter_path)

    total = 0
    tool_ok = 0
    answer_ok = 0
    details: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        scenario = row["scenario"]
        before_data = row.get("initial_reminders")
        if not isinstance(before_data, list):
            before_data = build_initial_reminders(
                scenario=scenario,
                tool=str(row.get("tool", "")),
                expected_outcome=str(row.get("expected_outcome", "")),
            )

        with tempfile.TemporaryDirectory(prefix="bench_reminder_") as td:
            storage_path = Path(td) / "reminders.json"
            save_json(storage_path, before_data)

            service = ReminderService(JSONReminderStorage(storage_path))
            executor = ToolExecutor(service)
            loop = RuntimeLoop(model_callable=model, executor=executor, max_tool_rounds=3)

            result = loop.run(SYSTEM_PROMPT, history=[], user_message=row["prompt"])

            after_data = JSONReminderStorage(storage_path).load()
            delta = compute_storage_delta(before_data, after_data)
            final_text = (result.get("final", {}) or {}).get("content", "")
            tool_status = extract_last_tool_status(result.get("messages", []))

            ok_tool, why_tool = judge_tool_success(row, tool_status, delta)
            ok_ans, why_ans = judge_answer_accuracy(row, final_text)

            tool_ok += int(ok_tool)
            answer_ok += int(ok_ans)
            details.append(
                {
                    "id": row.get("id"),
                    "scenario": scenario,
                    "tool": row.get("tool"),
                    "expected_outcome": row.get("expected_outcome"),
                    "tool_status": tool_status,
                    "tool_ok": ok_tool,
                    "tool_reason": why_tool,
                    "answer_ok": ok_ans,
                    "answer_reason": why_ans,
                    "final_text": final_text,
                }
            )

    summary = {
        "total": total,
        "tool_success_rate": (tool_ok / total) if total else 0.0,
        "answer_accuracy_rate": (answer_ok / total) if total else 0.0,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
