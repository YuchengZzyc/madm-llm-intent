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
    # utf-8-sig avoids dropping the first row when seed.md is saved with a BOM.
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



def _cycle_pick(items: list[str], index: int, rng: random.Random) -> str:
    """Pick deterministically with light shuffling so early samples do not collapse."""
    if not items:
        return ""
    offset = rng.randrange(len(items))
    return items[(index + offset) % len(items)]


def build_variation_profile(seed: dict[str, Any], occurrence_idx: int, rng: random.Random) -> dict[str, str]:
    """Build an explicit diversity profile before asking the model to write.

    The old version asked the model to paraphrase a single seed utterance. That often
    makes all samples in one scenario orbit around the same task/person/time. This
    profile turns each example into a constrained generation problem: same scenario
    and expected outcome, but different task domain, time expression, user style,
    complexity, and surface form.
    """
    scenario = str(seed["scenario"])
    tool = str(seed["tool"])
    expected = str(seed["expected_outcome"])

    elder_daily_tasks = [
        "take blood pressure medicine",
        "check blood sugar",
        "measure blood pressure",
        "do knee stretches",
        "drink warm water after breakfast",
        "bring in the laundry",
        "turn off the stove",
        "close the balcony window",
        "charge the phone",
        "water the orchids",
        "feed the cat",
        "take out the trash",
        "pick up a parcel downstairs",
        "prepare documents for a clinic visit",
        "pay the electricity bill",
        "pay the water bill",
        "call the property office",
        "call my daughter",
        "call my son",
        "call my grandson",
        "join the community tai chi class",
        "buy vegetables on the way home",
        "soak beans before cooking dinner",
        "put the soup in the fridge",
    ]
    family_tasks = [
        "call my daughter",
        "call my son",
        "call my grandson",
        "video call my sister",
        "ask my nephew about the hospital appointment",
        "send my granddaughter a birthday message",
        "remind my husband to bring the umbrella",
        "check whether my daughter arrived home",
    ]
    utility_tasks = [
        "pay the electricity bill",
        "pay the water bill",
        "pay the gas bill",
        "top up the phone bill",
        "check the property management fee",
        "renew the bus card",
        "submit the community service form",
    ]
    absent_tasks = [
        "tennis practice",
        "dentist visit",
        "haircut appointment",
        "bank card replacement",
        "library lecture",
        "parcel return",
        "pharmacy pickup",
        "neighborhood committee meeting",
    ]

    if "call_family" in scenario:
        task = _cycle_pick(family_tasks, occurrence_idx, rng)
    elif "utility_bill" in scenario:
        task = _cycle_pick(utility_tasks, occurrence_idx, rng)
    elif any(x in scenario for x in ["not_found", "haircut", "tennis", "doctor"]):
        task = _cycle_pick(absent_tasks, occurrence_idx, rng)
    elif "medicine" in scenario or "same_task" in scenario:
        task = _cycle_pick([
            "take the morning medicine",
            "take the evening medicine",
            "measure blood pressure",
            "check blood sugar",
            "apply the knee patch",
            "take vitamins after dinner",
        ], occurrence_idx, rng)
    elif "task_change" in scenario:
        old_task = _cycle_pick(["go walking", "water the plants", "buy vegetables", "take medicine", "call my son"], occurrence_idx, rng)
        new_task = _cycle_pick(["check blood sugar", "bring the umbrella", "pay the water bill", "charge the phone", "prepare clinic papers"], occurrence_idx + 3, rng)
        task = f"change from '{old_task}' to '{new_task}'"
    elif tool == "no_tool":
        task = _cycle_pick([
            "chat about a walk in the park",
            "share feeling lonely",
            "share happiness after a family visit",
            "mention possible tea with friends without asking for a reminder",
            "ask a general wellness question",
            "talk about cooking dinner",
            "mention seeing flowers bloom downstairs",
        ], occurrence_idx, rng)
    else:
        task = _cycle_pick(elder_daily_tasks, occurrence_idx, rng)

    time_styles = {
        "specific_clock": "use a concrete clock time such as 7:10 AM, 2:45 PM, or 8:30 tonight",
        "relative_daypart": "use a relative daypart such as tomorrow morning, this evening, Friday night, or after lunch",
        "natural_context": "anchor time to a daily routine such as after breakfast, before dinner, when I get back home, or before bed",
        "date_like": "use a date-like phrase such as next Monday, this Friday, or the day after tomorrow",
        "compressed": "write in a short command style with minimal grammar",
        "no_time": "do not include any time expression",
    }
    if "missing_time" in scenario or "missing_both" in scenario:
        time_style_key = "no_time"
    else:
        time_style_key = _cycle_pick(list(time_styles.keys())[:-1], occurrence_idx, rng)

    user_styles = [
        "direct and short",
        "polite elder style",
        "slightly hesitant",
        "spoken and casual",
        "compressed command",
        "context first, request second",
        "contains a small irrelevant detail but remains clear",
    ]
    complexity_levels = [
        "simple one-clause sentence",
        "medium sentence with a reason or context",
        "longer spoken sentence with one extra detail",
        "elliptical command, like a real user typing quickly",
        "indirect phrasing but still clearly implies the same tool intent",
    ]

    if "short_command" in scenario:
        style = "compressed command"
        complexity = "elliptical command, like a real user typing quickly"
    elif "polite" in scenario:
        style = "polite elder style"
        complexity = "medium sentence with a reason or context"
    elif tool == "no_tool":
        style = _cycle_pick(["spoken and casual", "context first, request second", "slightly emotional but not dramatic"], occurrence_idx, rng)
        complexity = _cycle_pick(["simple one-clause sentence", "longer spoken sentence with one extra detail"], occurrence_idx, rng)
    else:
        style = _cycle_pick(user_styles, occurrence_idx, rng)
        complexity = _cycle_pick(complexity_levels, occurrence_idx, rng)

    negative_constraints: list[str] = [
        "do not copy the seed utterance structure",
        "do not reuse the same concrete task/person/time unless the scenario name requires it",
        "write exactly one user prompt, not an assistant response",
    ]
    if "missing_task" in scenario or "missing_both" in scenario:
        negative_constraints.append("do not mention the task to be reminded about")
    if "missing_time" in scenario or "missing_both" in scenario:
        negative_constraints.append("do not mention any time, date, or daypart")
    if expected == "no_tool":
        negative_constraints.append("do not ask to create, query, update, or delete a reminder")
    if expected == "ambiguous":
        negative_constraints.append("leave the target reminder under-specified so multiple existing reminders could match")
    if expected == "not_found":
        negative_constraints.append("ask about a plausible reminder target that is absent or unmatched")

    return {
        "task_or_topic": task,
        "time_style": time_styles[time_style_key],
        "user_style": style,
        "complexity": complexity,
        "negative_constraints": "; ".join(negative_constraints),
    }


def clean_generated_prompt(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
            raw = obj["prompt"].strip()
    except Exception:
        pass
    raw = raw.strip().strip('"').strip("'").strip()
    for prefix in ["Prompt:", "User prompt:", "User:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip()
    return " ".join(raw.split())


def generate_prompt(
    client: OpenAICompatibleClient,
    seed: dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    variation_tag: str,
    variation_profile: dict[str, str],
) -> str:
    prompt = [
        {
            "role": "system",
            "content": (
                "You generate diverse user prompts for a reminder-tool benchmark. English only. "
                "Return JSON only: {\"prompt\":\"...\"}. "
                "The prompt must be a natural message from an elderly user or family assistant user. "
                "Preserve the scenario label and expected outcome exactly; vary the surface wording and concrete content. "
                "Avoid near-duplicate paraphrases. Never include explanations, labels, or multiple options."
            ),
        },
        {
            "role": "user",
            "content": (
                "Benchmark row:\n"
                f"- tool: {seed['tool']}\n"
                f"- scenario: {seed['scenario']}\n"
                f"- expected_outcome: {seed['expected_outcome']}\n"
                f"- seed_utterance: {seed['seed_utterance']}\n"
                f"- notes: {seed['notes']}\n\n"
                "Diversity profile for this exact sample:\n"
                f"- task_or_topic: {variation_profile['task_or_topic']}\n"
                f"- time_style: {variation_profile['time_style']}\n"
                f"- user_style: {variation_profile['user_style']}\n"
                f"- complexity: {variation_profile['complexity']}\n"
                f"- must_avoid: {variation_profile['negative_constraints']}\n"
                f"- variation_tag: {variation_tag}\n\n"
                "Generate one benchmark prompt. It should test the same tool behavior as the scenario, "
                "but it should feel like a new real-world case rather than a paraphrase of the seed."
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            text = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
            cleaned = clean_generated_prompt(text)
            if cleaned:
                return cleaned
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
        for occurrence_idx in range(count):
            idx += 1
            jobs.append(
                {
                    "idx": idx,
                    "row": dict(row),
                    "occurrence_idx": occurrence_idx,
                    "variation_profile": build_variation_profile(row, occurrence_idx, rng),
                    "temp": min(1.15, max(0.75, args.temperature + rng.uniform(-0.10, 0.20))),
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
                variation_tag=f"{job['idx']}-{job['occurrence_idx']}-{int(job['temp'] * 1000)}",
                variation_profile=job["variation_profile"],
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
