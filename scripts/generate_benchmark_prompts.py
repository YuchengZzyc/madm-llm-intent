from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

        obj = json.loads(raw)
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

        err = obj.get("error")
        if err:
            raise RuntimeError(f"Model API error payload: {err}")
        raise RuntimeError(f"Model response did not contain readable text. keys={list(obj.keys())[:20]}")


def parse_seed_markdown(path: Path) -> list[dict[str, Any]]:
    lines = [x.rstrip("\n") for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    data_lines = [ln for ln in lines if ln.startswith("|")]
    rows: list[dict[str, Any]] = []
    for ln in data_lines[2:]:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) < 6:
            continue
        rows.append(
            {
                "tool": cols[0],
                "scenario": cols[1],
                "seed_utterance": cols[2],
                "number": int(cols[3]),
                "expected_outcome": cols[4],
                "notes": cols[5],
            }
        )
    return rows


def generate_prompt(
    client: OpenAICompatibleClient,
    seed: dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    variation_tag: str,
) -> str:
    topic_pool = (
        "medicine, blood pressure, blood sugar, walking, stretching, grocery shopping, "
        "paying utility bills, calling son, calling daughter, calling grandson, "
        "doctor appointment, community activity, cooking, watering plants, feeding pets, "
        "taking out trash, closing windows, charging phone, picking up delivery"
    )

    prompt = [
        {
            "role": "system",
            "content": (
                "You rewrite reminder benchmark prompts. English only. "
                "Return either JSON {\"prompt\":\"...\"} or plain text. "
                "You should freely paraphrase and vary wording while preserving intent. "
                "Do not copy key entities from the seed by default."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Tool: {seed['tool']}\n"
                f"Scenario: {seed['scenario']}\n"
                f"Expected outcome: {seed['expected_outcome']}\n"
                f"Seed utterance: {seed['seed_utterance']}\n"
                f"Notes: {seed['notes']}\n"
                f"Variation tag: {variation_tag}\n"
                f"Topic pool (prefer rotating across these): {topic_pool}\n"
                "Write one new user prompt for the same scenario. "
                "Keep scenario intent/outcome unchanged, but you may change concrete content "
                "(task, person, activity, life context, time expression) to increase diversity. "
                "Do not stay stuck on only medicine, daughter, or basketball unless required."
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            text = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
                    return obj["prompt"].strip()
            except Exception:
                raw = text.strip()
                if raw:
                    return raw
            return seed["seed_utterance"]
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"generate_prompt failed after retries: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse benchmark prompts from benchmark/seed.md")
    parser.add_argument("--seed-md", type=Path, default=Path("benchmark/seed.md"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/prompts.jsonl"))
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent API workers.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout per request.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per prompt on transient failures.")
    parser.add_argument("--heartbeat-sec", type=int, default=5, help="Print heartbeat when no completion arrives.")
    args = parser.parse_args()

    env_file = load_env_file(args.api_env)
    base_url = resolve_required("base_url", [args.base_url, env_file.get("DISTILL_API_BASE_URL"), env_file.get("OPENAI_BASE_URL"), os.getenv("DISTILL_API_BASE_URL"), os.getenv("OPENAI_BASE_URL")])
    api_key = resolve_required("api_key", [args.api_key, env_file.get("DISTILL_API_KEY"), env_file.get("OPENAI_API_KEY"), os.getenv("DISTILL_API_KEY"), os.getenv("OPENAI_API_KEY")])
    model = resolve_required("model", [args.model, env_file.get("DISTILL_API_MODEL"), env_file.get("OPENAI_MODEL"), os.getenv("DISTILL_API_MODEL"), os.getenv("OPENAI_MODEL")])

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=args.endpoint,
        timeout=args.timeout_sec,
    )

    seeds = parse_seed_markdown(args.seed_md)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    idx = 0
    for row in seeds:
        count = int(row["number"])
        for _ in range(count):
            idx += 1
            jobs.append(
                {
                    "idx": idx,
                    "row": dict(row),
                    "temp": min(1.1, max(0.6, args.temperature + rng.uniform(-0.15, 0.15))),
                }
            )
    total = len(jobs)
    print(f"starting generation: total={total}, workers={max(1, int(args.workers))}", flush=True)

    def _run_job(job: dict[str, Any]) -> dict[str, Any]:
        row = job["row"]
        sample = dict(row)
        sample["id"] = f"bench_prompt_{job['idx']:05d}"
        try:
            sample["prompt"] = generate_prompt(
                client,
                row,
                temperature=job["temp"],
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                variation_tag=f"{job['idx']}-{int(job['temp'] * 1000)}",
            )
            sample["_error"] = None
        except Exception as exc:
            sample["prompt"] = row["seed_utterance"]
            sample["_error"] = str(exc)
        sample["_idx"] = job["idx"]
        return sample

    workers = max(1, int(args.workers))
    completed = 0
    results_by_idx: dict[int, dict[str, Any]] = {}
    next_to_write = 1
    with args.output.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=max(1, args.heartbeat_sec),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    print(f"[heartbeat] completed={completed}/{total}, pending={len(pending)}", flush=True)
                    continue
                for future in done:
                    sample = future.result()
                    completed += 1
                    results_by_idx[int(sample["_idx"])] = sample
                    preview = sample["prompt"][:80].replace("\n", " ")
                    if sample.get("_error"):
                        print(f"[warn] idx={sample['_idx']} fallback-to-seed due to: {sample['_error']}", flush=True)
                    print(f"[{completed}/{total}] ({sample['_idx']}/{total}) {sample['scenario']} -> {preview}", flush=True)

                    while next_to_write in results_by_idx:
                        out = results_by_idx.pop(next_to_write)
                        out.pop("_idx", None)
                        out.pop("_error", None)
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        f.flush()
                        next_to_write += 1

    print(f"generated={total} -> {args.output} (workers={workers})", flush=True)


if __name__ == "__main__":
    main()
