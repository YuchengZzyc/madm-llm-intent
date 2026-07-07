from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark_eval import compute_storage_delta, judge_answer_accuracy, judge_tool_success, load_jsonl
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
    parser = argparse.ArgumentParser(description="Run tool-use eval from initialized eval_runs dataset.")
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()

    rows = load_jsonl(args.cases_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    model = HFAssistant(model_path=args.model_path, adapter_path=args.adapter_path)

    total = 0
    tool_ok = 0
    answer_ok = 0
    details: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        case_id = str(row.get("id") or f"case_{i:05d}")
        case_dir = args.eval_dir / case_id
        reminders_path = case_dir / "reminders.json"
        if not reminders_path.exists():
            total += 1
            details.append(
                {
                    "id": case_id,
                    "scenario": row.get("scenario"),
                    "tool": row.get("tool"),
                    "expected_outcome": row.get("expected_outcome"),
                    "tool_status": None,
                    "tool_ok": False,
                    "tool_reason": f"missing initialized reminders file: {reminders_path}",
                    "answer_ok": False,
                    "answer_reason": "evaluation skipped because initialization is missing",
                    "final_text": "",
                }
            )
            continue

        before_data = json.loads(reminders_path.read_text(encoding="utf-8"))

        service = ReminderService(JSONReminderStorage(reminders_path))
        executor = ToolExecutor(service)
        loop = RuntimeLoop(model_callable=model, executor=executor, max_tool_rounds=3)
        result = loop.run(SYSTEM_PROMPT, history=[], user_message=str(row.get("prompt", "")))

        after_data = JSONReminderStorage(reminders_path).load()
        delta = compute_storage_delta(before_data, after_data)
        final_text = (result.get("final", {}) or {}).get("content", "")
        tool_status = extract_last_tool_status(result.get("messages", []))

        ok_tool, why_tool = judge_tool_success(row, tool_status, delta)
        ok_ans, why_ans = judge_answer_accuracy(row, final_text)

        total += 1
        tool_ok += int(ok_tool)
        answer_ok += int(ok_ans)
        details.append(
            {
                "id": case_id,
                "scenario": row.get("scenario"),
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
        "cases_jsonl": str(args.cases_jsonl),
        "eval_dir": str(args.eval_dir),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
