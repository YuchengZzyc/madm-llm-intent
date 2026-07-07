from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.device_intent_web_demo import build_training_prompt, parse_intent_output


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


class TokenTimingStreamer:
    """Minimal transformers streamer that records generated-token arrival times."""

    def __init__(self, skip_prompt: bool = True) -> None:
        self.skip_prompt = skip_prompt
        self.next_tokens_are_prompt = True
        self.token_ids: list[int] = []
        self.token_times: list[float] = []
        self.done = threading.Event()

    def put(self, value: Any) -> None:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "dim") and value.dim() > 1:
            if value.shape[0] > 1:
                raise ValueError("TokenTimingStreamer only supports batch size 1.")
            value = value[0]
        token_ids = value.tolist() if hasattr(value, "tolist") else list(value)
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.next_tokens_are_prompt = False
        now = time.perf_counter()
        for token_id in token_ids:
            self.token_ids.append(int(token_id))
            self.token_times.append(now)

    def end(self) -> None:
        self.done.set()


class TimedDeviceIntentModel:
    def __init__(self, model_path: str, adapter_path: str | None, max_new_tokens: int) -> None:
        load_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        sync_if_cuda()
        self.load_seconds = time.perf_counter() - load_start

    def generate_once(self, message: str) -> dict[str, Any]:
        prompt = build_training_prompt(message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])
        streamer = TokenTimingStreamer(skip_prompt=True)
        result_holder: dict[str, Any] = {}

        def run_generate() -> None:
            try:
                with torch.inference_mode():
                    result_holder["output"] = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                    )
            except Exception as exc:
                result_holder["error"] = exc
                streamer.end()

        sync_if_cuda()
        start = time.perf_counter()
        worker = threading.Thread(target=run_generate, daemon=True)
        worker.start()
        worker.join()
        sync_if_cuda()
        end = time.perf_counter()
        latency_seconds = end - start

        if "error" in result_holder:
            raise result_holder["error"]
        output = result_holder["output"]

        output_ids = output[0][input_tokens:]
        output_tokens = int(output_ids.shape[0])
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed = parse_intent_output(text)
        tokens_per_second = output_tokens / latency_seconds if latency_seconds > 0 else 0.0
        first_token_seconds = (streamer.token_times[0] - start) if streamer.token_times else None
        last_token_seconds = (streamer.token_times[-1] - start) if streamer.token_times else None
        post_first_seconds = (end - streamer.token_times[0]) if streamer.token_times else None
        decode_after_first_seconds = (
            streamer.token_times[-1] - streamer.token_times[0]
            if len(streamer.token_times) > 1
            else 0.0
        )
        post_last_token_overhead_seconds = (end - streamer.token_times[-1]) if streamer.token_times else None
        subsequent_token_count = max(0, len(streamer.token_times) - 1)
        subsequent_tokens_per_second = (
            subsequent_token_count / post_first_seconds
            if post_first_seconds and post_first_seconds > 0 and subsequent_token_count > 0
            else 0.0
        )
        decode_only_tokens_per_second = (
            subsequent_token_count / decode_after_first_seconds
            if decode_after_first_seconds > 0 and subsequent_token_count > 0
            else 0.0
        )
        return {
            "latency_seconds": latency_seconds,
            "latency_ms": latency_seconds * 1000,
            "ttft_seconds": first_token_seconds,
            "ttft_ms": first_token_seconds * 1000 if first_token_seconds is not None else None,
            "last_token_seconds": last_token_seconds,
            "last_token_ms": last_token_seconds * 1000 if last_token_seconds is not None else None,
            "post_first_seconds": post_first_seconds,
            "post_first_ms": post_first_seconds * 1000 if post_first_seconds is not None else None,
            "decode_after_first_seconds": decode_after_first_seconds,
            "decode_after_first_ms": decode_after_first_seconds * 1000,
            "post_last_token_overhead_seconds": post_last_token_overhead_seconds,
            "post_last_token_overhead_ms": post_last_token_overhead_seconds * 1000 if post_last_token_overhead_seconds is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "streamed_token_events": len(streamer.token_times),
            "subsequent_token_count": subsequent_token_count,
            "tokens_per_second": tokens_per_second,
            "subsequent_tokens_per_second": subsequent_tokens_per_second,
            "decode_only_tokens_per_second": decode_only_tokens_per_second,
            "raw_output": text,
            "parsed_intent": parsed,
        }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in results]
    output_tokens = [int(item["output_tokens"]) for item in results]
    tokens_per_second = [float(item["tokens_per_second"]) for item in results]
    ttft_values = [float(item["ttft_ms"]) for item in results if item.get("ttft_ms") is not None]
    subsequent_tokens_per_second = [float(item["subsequent_tokens_per_second"]) for item in results]
    decode_only_tokens_per_second = [float(item["decode_only_tokens_per_second"]) for item in results]
    post_last_overhead = [
        float(item["post_last_token_overhead_ms"])
        for item in results
        if item.get("post_last_token_overhead_ms") is not None
    ]
    return {
        "runs": len(results),
        "latency_ms_min": min(latencies) if latencies else 0.0,
        "latency_ms_max": max(latencies) if latencies else 0.0,
        "latency_ms_avg": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "ttft_ms_min": min(ttft_values) if ttft_values else None,
        "ttft_ms_max": max(ttft_values) if ttft_values else None,
        "ttft_ms_avg": statistics.mean(ttft_values) if ttft_values else None,
        "ttft_ms_p50": percentile(ttft_values, 0.50) if ttft_values else None,
        "ttft_ms_p95": percentile(ttft_values, 0.95) if ttft_values else None,
        "output_tokens_avg": statistics.mean(output_tokens) if output_tokens else 0.0,
        "tokens_per_second_avg": statistics.mean(tokens_per_second) if tokens_per_second else 0.0,
        "subsequent_tokens_per_second_avg": statistics.mean(subsequent_tokens_per_second) if subsequent_tokens_per_second else 0.0,
        "decode_only_tokens_per_second_avg": statistics.mean(decode_only_tokens_per_second) if decode_only_tokens_per_second else 0.0,
        "post_last_token_overhead_ms_avg": statistics.mean(post_last_overhead) if post_last_overhead else None,
    }


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one-message latency for the device intent LoRA model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--message", default="Lock the door.", help="Single user message to measure.")
    parser.add_argument("--runs", type=int, default=1, help="Measured runs after warmup.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs not included in latency summary.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    model = TimedDeviceIntentModel(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    for _ in range(max(0, args.warmup)):
        model.generate_once(args.message)

    results = [model.generate_once(args.message) for _ in range(max(1, args.runs))]
    summary = build_summary(results)
    report = {
        "message": args.message,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": model.load_seconds,
        "summary": summary,
        "last_result": results[-1],
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"message: {args.message}")
    print(f"load_seconds: {model.load_seconds:.3f}")
    print(f"runs: {summary['runs']} warmup: {max(0, args.warmup)}")
    print(f"latency_ms_avg: {summary['latency_ms_avg']:.2f}")
    print(f"latency_ms_p50: {summary['latency_ms_p50']:.2f}")
    print(f"latency_ms_p95: {summary['latency_ms_p95']:.2f}")
    print(f"latency_ms_min/max: {summary['latency_ms_min']:.2f}/{summary['latency_ms_max']:.2f}")
    print(f"ttft_ms_avg: {fmt_ms(summary['ttft_ms_avg'])}")
    print(f"ttft_ms_p50: {fmt_ms(summary['ttft_ms_p50'])}")
    print(f"ttft_ms_p95: {fmt_ms(summary['ttft_ms_p95'])}")
    print(f"ttft_ms_min/max: {fmt_ms(summary['ttft_ms_min'])}/{fmt_ms(summary['ttft_ms_max'])}")
    print(f"output_tokens_avg: {summary['output_tokens_avg']:.1f}")
    print(f"tokens_per_second_avg_total: {summary['tokens_per_second_avg']:.2f}")
    print(f"tokens_per_second_avg_after_first_until_end: {summary['subsequent_tokens_per_second_avg']:.2f}")
    print(f"tokens_per_second_avg_decode_only: {summary['decode_only_tokens_per_second_avg']:.2f}")
    print(f"post_last_token_overhead_ms_avg: {fmt_ms(summary['post_last_token_overhead_ms_avg'])}")
    print("raw_output:")
    print(results[-1]["raw_output"])
    print("parsed_intent:")
    print(json.dumps(results[-1]["parsed_intent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
