from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tool_registry import get_tools


SYSTEM_PROMPT = (
    "You are XiaoNuan. Use reminder tools only for create/query/update/delete reminder intent. "
    "If tool is needed, output assistant tool_calls in OpenAI format. "
    "If no tool is needed, answer naturally."
)


def extract_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-turn smoke test for tool-call output format.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto")
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.text},
    ]
    tools = get_tools()
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("=== RAW OUTPUT ===")
    print(text)
    print()
    parsed = extract_json(text)
    if parsed and isinstance(parsed.get("tool_calls"), list):
        print("=== PARSED ===")
        print("tool_calls_detected=True")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print("=== PARSED ===")
        print("tool_calls_detected=False")


if __name__ == "__main__":
    main()

