from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import random
import re
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
                if "gzip" in (resp.headers.get("Content-Encoding") or "").lower() or body[:2] == b"\x1f\x8b":
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
            msg = obj["choices"][0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
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
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        parts.append(content["text"])
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        if obj.get("error"):
            raise RuntimeError(f"Model API error payload: {obj['error']}")
        raise RuntimeError(f"Model response did not contain readable text. keys={list(obj.keys())[:20]}")


def parse_seed_markdown(path: Path) -> list[dict[str, Any]]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    data_lines = [ln for ln in lines if ln.startswith("|")]
    rows: list[dict[str, Any]] = []
    for ln in data_lines:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) < 6:
            continue
        if cols[0].lower() == "tool" or set(cols[0]) <= {"-", ":"}:
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


def cycle_pick(items: list[str], index: int, rng: random.Random) -> str:
    if not items:
        return ""
    return items[(index + rng.randrange(len(items))) % len(items)]


TASK_POOLS: dict[str, list[str]] = {
    "medicine": ["take blood pressure medicine", "check blood sugar", "measure blood pressure", "take vitamins after dinner", "apply the knee patch"],
    "family": ["call my daughter", "call my son", "video call my sister", "send my granddaughter a birthday message", "ask my nephew about the clinic appointment"],
    "sports": ["play table tennis", "play basketball", "go to tai chi class", "join the square dancing group", "take an evening walk"],
    "personal_care": ["get a haircut", "pick up new glasses", "go to the foot massage appointment", "trim the potted plants", "collect medicine from the pharmacy"],
    "household": ["bring in the laundry", "close the balcony window", "turn off the stove", "water the porch plants", "take out the trash", "charge the phone"],
    "errand": ["pay the electricity bill", "pay the water bill", "renew the bus card", "return the library book", "pick up a parcel downstairs"],
    "appointment": ["go to the dentist", "visit the community clinic", "meet the bank clerk", "attend the neighborhood committee meeting", "go to the property office"],
    "food": ["put the leftover soup in the fridge", "soak beans before dinner", "buy vegetables on the way home", "defrost fish for dinner"],
}

TIME_TEXTS = [
    "today at 6:30 PM",
    "tonight at 8:15 PM",
    "tomorrow at 8:00 AM",
    "tomorrow morning",
    "tomorrow afternoon at 4:00 PM",
    "after breakfast tomorrow",
    "before dinner tonight",
    "this Friday evening",
    "next Monday at 10:15 AM",
    "the day after tomorrow at 9:30 AM",
]

STYLE_HINTS = [
    "direct and short",
    "polite elderly-user style",
    "spoken and casual",
    "slightly hesitant",
    "compressed command style",
    "context first, request second",
    "one small irrelevant detail but still clear",
]


def task_pool_for_scenario(scenario: str, occurrence_idx: int) -> str:
    s = scenario.lower()
    if any(x in s for x in ["sport", "basketball", "tennis", "table_tennis"]):
        return "sports"
    if any(x in s for x in ["haircut", "personal", "care"]):
        return "personal_care"
    if any(x in s for x in ["medicine", "blood", "doctor"]):
        return "medicine" if occurrence_idx % 2 == 0 else "appointment"
    if any(x in s for x in ["family", "call", "son", "daughter"]):
        return "family"
    if any(x in s for x in ["bill", "payment", "errand", "utility"]):
        return "errand"
    if any(x in s for x in ["house", "safety", "laundry", "window"]):
        return "household"
    if "appointment" in s:
        return "appointment"
    keys = list(TASK_POOLS.keys())
    return keys[occurrence_idx % len(keys)]


def build_gold(seed: dict[str, Any], occurrence_idx: int, rng: random.Random) -> dict[str, Any]:
    tool = str(seed["tool"])
    scenario = str(seed["scenario"])
    expected = str(seed["expected_outcome"])
    pool_name = task_pool_for_scenario(scenario, occurrence_idx)
    task = cycle_pick(TASK_POOLS[pool_name], occurrence_idx, rng)
    alt_pool = TASK_POOLS["food" if pool_name != "food" else "household"]
    alt_task = cycle_pick(alt_pool, occurrence_idx + 2, rng)
    time_text = cycle_pick(TIME_TEXTS, occurrence_idx, rng)
    alt_time_text = cycle_pick(TIME_TEXTS, occurrence_idx + 4, rng)

    gold: dict[str, Any] = {
        "action": tool if tool in {"create", "query", "update", "delete"} else "none",
        "target": "self",
        "task": None,
        "time_text": None,
        "locator_task": None,
        "locator_time_text": None,
        "new_task": None,
        "new_time_text": None,
        "should_call_tool": expected != "no_tool",
    }

    if tool == "create":
        if "missing_both" in scenario:
            pass
        elif "missing_task" in scenario:
            gold["time_text"] = time_text
        elif "missing_time" in scenario:
            gold["task"] = task
        else:
            gold["task"] = task
            gold["time_text"] = time_text
    elif tool == "query":
        if "vague" in scenario:
            gold["time_text"] = time_text if occurrence_idx % 2 else "tomorrow"
        elif "by_time" in scenario:
            gold["time_text"] = time_text
        else:
            gold["task"] = task
            if occurrence_idx % 2 == 0:
                gold["time_text"] = time_text
    elif tool == "update":
        if "missing_target" in scenario:
            gold["new_time_text"] = alt_time_text
        elif "task_change" in scenario:
            gold["locator_task"] = task
            gold["locator_time_text"] = time_text
            gold["new_task"] = alt_task
        elif "time_shift" in scenario or expected == "success":
            gold["locator_task"] = task
            gold["locator_time_text"] = time_text
            gold["new_time_text"] = alt_time_text
        elif expected in {"ambiguous", "not_found"}:
            gold["locator_task"] = task
            gold["new_time_text"] = alt_time_text
    elif tool == "delete":
        if "missing_target" in scenario:
            pass
        else:
            gold["locator_task"] = task
            if occurrence_idx % 2 == 0 or expected == "success":
                gold["locator_time_text"] = time_text
    else:
        gold["should_call_tool"] = False

    # Convenience aliases for create/query/delete locators.
    if tool in {"query", "delete"}:
        gold["task"] = gold.get("task") or gold.get("locator_task")
        gold["time_text"] = gold.get("time_text") or gold.get("locator_time_text")

    return gold


def normalize_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def parse_model_json(raw: str) -> dict[str, Any]:
    text = normalize_json_text(raw)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("model did not return a JSON object")
    return obj


def validate_and_merge(obj: dict[str, Any], planned_gold: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prompt = str(obj.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("missing prompt")
    gold = dict(planned_gold)
    # Let model provide task/time_text, but keep our planned values as fallback.
    for key in ["task", "time_text", "locator_task", "locator_time_text", "new_task", "new_time_text"]:
        if key in obj:
            value = obj.get(key)
            gold[key] = value.strip() if isinstance(value, str) and value.strip() else None
    if isinstance(obj.get("gold"), dict):
        for key in ["task", "time_text", "locator_task", "locator_time_text", "new_task", "new_time_text", "target"]:
            value = obj["gold"].get(key)
            if isinstance(value, str) and value.strip():
                gold[key] = value.strip()
            elif value is None and key in obj["gold"]:
                gold[key] = None
    gold["target"] = "self"
    return " ".join(prompt.split()), gold


def generate_structured_sample(
    client: OpenAICompatibleClient,
    seed: dict[str, Any],
    planned_gold: dict[str, Any],
    variation: dict[str, str],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You generate English user prompts for a reminder tool-use benchmark. "
                "Return one JSON object only. No markdown, no explanations. "
                "The JSON must contain prompt plus gold semantic fields. "
                "Fields: prompt, task, time_text, locator_task, locator_time_text, new_task, new_time_text. "
                "Use null for fields that are missing or intentionally unspecified. "
                "target is always self and should not be included unless needed."
            ),
        },
        {
            "role": "user",
            "content": (
                "Benchmark row:\n"
                f"tool={seed['tool']}\nscenario={seed['scenario']}\nexpected_outcome={seed['expected_outcome']}\nnotes={seed['notes']}\n"
                f"seed_utterance={seed['seed_utterance']}\n\n"
                "Planned gold semantics. Keep the prompt consistent with these values:\n"
                f"{json.dumps(planned_gold, ensure_ascii=False)}\n\n"
                "Style requirements:\n"
                f"- user_style: {variation['style']}\n"
                f"- complexity: {variation['complexity']}\n"
                f"- avoid: {variation['avoid']}\n\n"
                "Return JSON exactly like this shape:\n"
                '{"prompt":"...","task":null,"time_text":null,"locator_task":null,"locator_time_text":null,"new_task":null,"new_time_text":null}\n'
                "Important: task fields must be the normalized reminder task, not the whole sentence. "
                "time_text fields must preserve the relative/natural time phrase, such as 'tomorrow at 4:00 PM'."
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            return validate_and_merge(parse_model_json(raw), planned_gold)
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"structured generation failed: {last_err}")


def build_variation(seed: dict[str, Any], occurrence_idx: int, rng: random.Random) -> dict[str, str]:
    expected = str(seed["expected_outcome"])
    scenario = str(seed["scenario"])
    avoid = [
        "do not copy the seed sentence",
        "do not output assistant wording",
        "do not create near-duplicates",
    ]
    if expected == "no_tool":
        avoid.append("do not ask to create, query, update, or delete reminders")
    if "missing_task" in scenario:
        avoid.append("the prompt must not mention the task")
    if "missing_time" in scenario:
        avoid.append("the prompt must not mention the time/date/daypart")
    if expected == "ambiguous":
        avoid.append("leave the target reminder underspecified")
    return {
        "style": cycle_pick(STYLE_HINTS, occurrence_idx, rng),
        "complexity": cycle_pick([
            "simple one-clause sentence",
            "medium sentence with a reason",
            "longer spoken sentence with one extra detail",
            "elliptical real-user command",
            "indirect but still clear",
        ], occurrence_idx, rng),
        "avoid": "; ".join(avoid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured reminder benchmark prompts with gold task/time fields.")
    parser.add_argument("--seed-md", type=Path, default=Path("benchmark/seed.md"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/prompts_structured.jsonl"))
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--heartbeat-sec", type=int, default=5)
    args = parser.parse_args()

    env_file = load_env_file(args.api_env)
    base_url = resolve_required("base_url", [args.base_url, env_file.get("DISTILL_API_BASE_URL"), env_file.get("OPENAI_BASE_URL"), os.getenv("DISTILL_API_BASE_URL"), os.getenv("OPENAI_BASE_URL")])
    api_key = resolve_required("api_key", [args.api_key, env_file.get("DISTILL_API_KEY"), env_file.get("OPENAI_API_KEY"), os.getenv("DISTILL_API_KEY"), os.getenv("OPENAI_API_KEY")])
    model = resolve_required("model", [args.model, env_file.get("DISTILL_API_MODEL"), env_file.get("OPENAI_MODEL"), os.getenv("DISTILL_API_MODEL"), os.getenv("OPENAI_MODEL")])

    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model, endpoint=args.endpoint, timeout=args.timeout_sec)
    seeds = parse_seed_markdown(args.seed_md)
    rng = random.Random(args.seed)
    jobs: list[dict[str, Any]] = []
    idx = 0
    for row in seeds:
        for occurrence_idx in range(int(row["number"])):
            idx += 1
            local_rng = random.Random(args.seed * 100000 + idx)
            planned_gold = build_gold(row, occurrence_idx, local_rng)
            variation = build_variation(row, occurrence_idx, local_rng)
            jobs.append({
                "idx": idx,
                "row": dict(row),
                "occurrence_idx": occurrence_idx,
                "planned_gold": planned_gold,
                "variation": variation,
                "temp": min(1.05, max(0.65, args.temperature + rng.uniform(-0.10, 0.15))),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(jobs)
    print(f"starting structured generation: total={total}, workers={max(1, int(args.workers))}", flush=True)

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        row = job["row"]
        sample = dict(row)
        sample["id"] = f"bench_prompt_{job['idx']:05d}"
        try:
            prompt, gold = generate_structured_sample(
                client=client,
                seed=row,
                planned_gold=job["planned_gold"],
                variation=job["variation"],
                temperature=job["temp"],
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
            sample["prompt"] = prompt
            sample["gold"] = gold
            sample["_error"] = None
        except Exception as exc:
            # Fallback is still structured, but based on planned gold.
            sample["prompt"] = row["seed_utterance"]
            sample["gold"] = dict(job["planned_gold"])
            sample["_error"] = str(exc)
        sample["_idx"] = job["idx"]
        return sample

    workers = max(1, int(args.workers))
    completed = 0
    results_by_idx: dict[int, dict[str, Any]] = {}
    next_to_write = 1
    with args.output.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=max(1, args.heartbeat_sec), return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    print(f"[heartbeat] completed={completed}/{total}, pending={len(pending)}", flush=True)
                    continue
                for future in done:
                    sample = future.result()
                    completed += 1
                    results_by_idx[int(sample["_idx"])] = sample
                    preview = sample["prompt"][:80].replace("\n", " ")
                    if sample.get("_error"):
                        print(f"[warn] idx={sample['_idx']} fallback due to: {sample['_error']}", flush=True)
                    print(f"[{completed}/{total}] ({sample['_idx']}/{total}) {sample['scenario']} -> {preview}", flush=True)
                    while next_to_write in results_by_idx:
                        out = results_by_idx.pop(next_to_write)
                        out.pop("_idx", None)
                        out.pop("_error", None)
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        f.flush()
                        next_to_write += 1
    print(f"generated={total} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
