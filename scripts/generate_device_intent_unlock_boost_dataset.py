from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_device_intent_dataset import (  # noqa: E402
    OpenAICompatibleClient,
    SCENARIOS,
    build_training_sample,
    count_existing_rows,
    format_offline_prompt as format_base_offline_prompt,
    label,
    label_for_scenario,
    load_env_file,
    load_existing_utterances,
    resolve_required,
    write_dataset_report,
)


DEFAULT_SAMPLES = 1000
DEFAULT_CONTRAST_RATIO = 0.10
DEFAULT_DOOR_LOCK_RATIO = 0.50
DEFAULT_LOCK_RATIO = 0.20

BASE_DOOR_LOCK_SCENARIOS = {"unlock_remote", "unlock_during_call"}
OTHER_INTENT_SCENARIOS = [scenario for scenario in SCENARIOS if scenario["scenario"] not in BASE_DOOR_LOCK_SCENARIOS]


UNLOCK_BOOST_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "unlock_boost_plain_command",
        "weight": 16,
        "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.95),
        "notes": "Direct requests to remotely unlock or open the entrance door lock.",
    },
    {
        "scenario": "unlock_boost_visitor_waiting",
        "weight": 16,
        "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.95),
        "notes": "A visitor, neighbor, nurse, courier, family member, or delivery person is waiting outside.",
    },
    {
        "scenario": "unlock_boost_intercom_or_downstairs",
        "weight": 12,
        "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.95),
        "notes": "The intercom rang, someone is downstairs, or the user wants to buzz someone in.",
    },
    {
        "scenario": "unlock_boost_elderly_fuzzy_request",
        "weight": 12,
        "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.93),
        "notes": "Colloquial or fuzzy wording from an older adult, still clearly asking to open the door lock.",
    },
    {
        "scenario": "unlock_boost_urgent_short_phrase",
        "weight": 10,
        "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.95),
        "notes": "Short urgent phrases asking the assistant to open or unlock the door immediately.",
    },
    {
        "scenario": "unlock_boost_during_call_rich",
        "weight": 14,
        "label": label(True, 4, "unlock", {"method": "remote", "during_call": True}, confidence=0.95),
        "notes": "The user is on an active call or intercom session and asks to unlock during the call.",
    },
]


LOCK_BOOST_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "lock_boost_plain_command",
        "weight": 16,
        "label": label(True, 4, "lock", {"method": "remote"}, confidence=0.95),
        "notes": "Direct requests to remotely lock or secure the door lock.",
    },
    {
        "scenario": "lock_boost_leaving_home",
        "weight": 12,
        "label": label(True, 4, "lock", {"method": "remote"}, confidence=0.95),
        "notes": "The user is leaving, going to sleep, or wants the entrance secured.",
    },
    {
        "scenario": "lock_boost_elderly_fuzzy_request",
        "weight": 10,
        "label": label(True, 4, "lock", {"method": "remote"}, confidence=0.93),
        "notes": "Colloquial or fuzzy wording from an older adult, still clearly asking to lock the door.",
    },
]


DOOR_LOCK_CONTRAST_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "door_lock_contrast_status_question",
        "weight": 8,
        "label": label(False, None, None),
        "notes": "The user asks about lock status, memory, keys, or reminders but does not ask the assistant to control the lock.",
    },
    {
        "scenario": "door_lock_contrast_non_control",
        "weight": 6,
        "label": label(False, None, None),
        "notes": "The utterance mentions a door, lock, key, visitor, or entrance, but is not a control request.",
    },
]


EN_OFFLINE_TEMPLATES: dict[str, list[str]] = {
    "unlock_boost_plain_command": [
        "Unlock the front door for me.",
        "Open the door lock remotely.",
        "Please release the door lock.",
        "Can you open the entrance door?",
        "Let me unlock the door from here.",
        "Pop the front door open.",
    ],
    "unlock_boost_visitor_waiting": [
        "The delivery person is outside; unlock the door for them.",
        "My daughter is at the entrance, please let her in.",
        "The nurse has arrived downstairs, open the door.",
        "A neighbor is waiting by the door, buzz them in.",
        "The courier is here with medicine, unlock the entrance.",
        "Someone from the property office is at the door, let them in.",
    ],
    "unlock_boost_intercom_or_downstairs": [
        "The intercom just rang; open the door.",
        "They are calling from downstairs, unlock it.",
        "Buzz the lobby door open.",
        "The gate call is connected, please unlock the door.",
        "Someone is at the building entrance, let them up.",
    ],
    "unlock_boost_elderly_fuzzy_request": [
        "I can't get up right now, please open the door for me.",
        "My hands are full, help me let them in.",
        "The person outside has been waiting, open it for me.",
        "I am moving slowly today, unlock the door please.",
        "Could you take care of the door and let them come in?",
    ],
    "unlock_boost_urgent_short_phrase": [
        "Quick, open the door!",
        "Unlock it now.",
        "Open up, they are outside.",
        "Let them in right away.",
        "Door, open it now please.",
    ],
    "unlock_boost_during_call_rich": [
        "While I am on this call, unlock the door.",
        "During the intercom call, let them in.",
        "I am still talking to them, open the door now.",
        "Unlock the entrance before I hang up.",
        "This call is from downstairs; please open the door.",
    ],
    "lock_boost_plain_command": [
        "Lock the front door for me.",
        "Please secure the door lock.",
        "Lock up the entrance now.",
        "Can you lock the door remotely?",
        "Make the door locked.",
    ],
    "lock_boost_leaving_home": [
        "I am going out, lock the door behind me.",
        "Before I go to sleep, secure the front door.",
        "We are leaving now, please lock up.",
        "After they come in, lock the entrance again.",
        "It is late, please make sure the door is locked.",
    ],
    "lock_boost_elderly_fuzzy_request": [
        "I cannot get to the door, help me lock it.",
        "My hands are full, please lock the door.",
        "Don't leave the entrance open; lock it for me.",
        "Please take care of the lock and secure the door.",
    ],
    "door_lock_contrast_status_question": [
        "Did I lock the door earlier?",
        "I cannot remember whether the front door is locked.",
        "Remind me later to check the door lock.",
        "Where did I leave the front door key?",
        "Is the lock battery low?",
    ],
    "door_lock_contrast_non_control": [
        "I forgot whether I locked the door.",
        "The door lock looks loose today.",
        "Where did I put the front door key?",
        "The visitor at the door was very polite.",
        "Remind me later to check the lock battery.",
    ],
}


ZH_OFFLINE_TEMPLATES: dict[str, list[str]] = {
    "unlock_boost_plain_command": [
        "帮我远程开一下门。",
        "把门锁打开。",
        "麻烦开一下大门。",
        "给我开门锁。",
        "把入户门放开一下。",
        "帮我把门打开。",
    ],
    "unlock_boost_visitor_waiting": [
        "外卖到了，帮我开门。",
        "我女儿在门口，帮她开一下。",
        "护工到了楼下，给她开门。",
        "邻居在外面等着，帮我放进来。",
        "快递员送药来了，把门打开。",
        "物业的人在门口，给他开一下门。",
    ],
    "unlock_boost_intercom_or_downstairs": [
        "门禁响了，帮我开一下。",
        "楼下在呼叫，把门打开。",
        "对讲里的人到了，给他开门。",
        "门口有人按铃，帮我远程开门。",
        "楼栋门那边有人，放他上来。",
    ],
    "unlock_boost_elderly_fuzzy_request": [
        "我现在起不来，帮我把门开了。",
        "我手里拿着东西，帮我开门吧。",
        "外面的人等半天了，给他开一下。",
        "我腿脚慢，你帮我开门。",
        "门口那个人让他进来吧。",
    ],
    "unlock_boost_urgent_short_phrase": [
        "快开门！",
        "马上开锁。",
        "门口有人，快开。",
        "现在就放他进来。",
        "开门，快点。",
    ],
    "unlock_boost_during_call_rich": [
        "通话中帮我开锁。",
        "还在对讲的时候把门打开。",
        "我正跟楼下通话，给他开门。",
        "挂断前先把门锁打开。",
        "这个电话是门口来的，帮我开门。",
    ],
    "lock_boost_plain_command": [
        "帮我把门锁上。",
        "把入户门上锁。",
        "远程锁一下门。",
        "现在把门锁住。",
        "给门上个锁。",
    ],
    "lock_boost_leaving_home": [
        "我要出门了，帮我锁门。",
        "睡觉前把门锁好。",
        "人都进来了，再把门锁上。",
        "天晚了，帮我把入户门锁住。",
        "离家前先把门上锁。",
    ],
    "lock_boost_elderly_fuzzy_request": [
        "我走不动了，帮我把门锁一下。",
        "我手里拿着东西，你帮我锁门。",
        "别让门一直开着，锁上吧。",
        "麻烦你处理下门锁，把门锁好。",
    ],
    "door_lock_contrast_status_question": [
        "我刚才有没有锁门？",
        "我忘了门是不是锁好了。",
        "晚点提醒我检查门锁。",
        "门钥匙放在哪里了？",
        "门锁电池是不是快没电了？",
    ],
    "door_lock_contrast_non_control": [
        "我忘了刚才有没有锁门。",
        "这个门锁今天有点松。",
        "门钥匙放在哪里了？",
        "刚才门口那个人挺客气的。",
        "晚点提醒我检查门锁电池。",
    ],
}


VOICE_SHAPES = [
    "one breath, very short",
    "casual spoken request",
    "contains a reason before the request",
    "contains the requester or visitor identity",
    "urgent but still realistic",
    "elderly user wording with mild hesitation",
]

SITUATIONS = [
    "front door",
    "building lobby",
    "intercom call",
    "courier or delivery",
    "family member visiting",
    "caregiver or nurse arriving",
    "user cannot easily walk to the door",
]


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def clean_boost_utterance(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("utterance", "text", "user_text"):
                if isinstance(obj.get(key), str):
                    raw = obj[key]
                    break
    except Exception:
        pass
    raw = raw.strip().strip('"').strip("'").strip()
    for prefix in ["User:", "Utterance:", "Text:", "user:", "utterance:", "text:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip()
    return normalize_spaces(raw)


def choose_language(user_language: str, occurrence_idx: int, rng: random.Random) -> str:
    if user_language in {"english", "chinese"}:
        return user_language
    return "chinese" if (occurrence_idx + rng.randrange(4)) % 4 == 0 else "english"


def weighted_select(definitions: Sequence[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    selected: list[dict[str, Any]] = []
    if count >= len(definitions):
        selected.extend(definitions)
    weighted_pool: list[dict[str, Any]] = []
    for item in definitions:
        weighted_pool.extend([item] * max(1, int(item.get("weight", 1))))
    while len(selected) < count:
        selected.append(rng.choice(weighted_pool))
    rng.shuffle(selected)
    return selected[:count]


def split_mixed_counts(total_samples: int, door_lock_ratio: float, contrast_ratio: float, lock_ratio: float) -> tuple[int, int, int, int]:
    total_samples = max(0, int(total_samples))
    door_lock_ratio = max(0.0, min(float(door_lock_ratio), 1.0))
    contrast_ratio = max(0.0, min(float(contrast_ratio), 0.95))
    door_positive_ratio = max(0.0, door_lock_ratio - contrast_ratio)
    lock_ratio = max(0.0, min(float(lock_ratio), door_positive_ratio))
    contrast_count = int(math.floor(total_samples * contrast_ratio + 0.5))
    lock_count = int(math.floor(total_samples * lock_ratio + 0.5))
    door_positive_count = int(math.floor(total_samples * door_positive_ratio + 0.5))
    if contrast_ratio > 0 and total_samples > 1 and contrast_count == 0:
        contrast_count = 1
    if lock_ratio > 0 and total_samples > 1 and lock_count == 0:
        lock_count = 1
    if door_positive_ratio > 0 and total_samples > 1 and door_positive_count == 0:
        door_positive_count = 1
    if contrast_count + door_positive_count > total_samples:
        overflow = contrast_count + door_positive_count - total_samples
        door_positive_count = max(0, door_positive_count - overflow)
    lock_count = min(lock_count, door_positive_count)
    unlock_count = door_positive_count - lock_count
    other_count = total_samples - contrast_count - door_positive_count
    return other_count, unlock_count, lock_count, contrast_count


def split_door_lock_counts(total_samples: int, contrast_ratio: float, lock_ratio: float) -> tuple[int, int, int]:
    _, unlock_count, lock_count, contrast_count = split_mixed_counts(
        total_samples=total_samples,
        door_lock_ratio=1.0,
        contrast_ratio=contrast_ratio,
        lock_ratio=lock_ratio,
    )
    return unlock_count, lock_count, contrast_count


def read_door_lock_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    door_lock_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
            label_data = json.loads(row["messages"][-1]["content"])
        except Exception:
            continue
        if label_data.get("matched") is True and label_data.get("capability_id") == 4 and label_data.get("intent") in {"unlock", "lock"}:
            door_lock_count += 1
    return total, door_lock_count


def read_unlock_counts(path: Path) -> tuple[int, int]:
    return read_door_lock_counts(path)


def plan_boost_sample_count(
    existing_total: int,
    existing_unlock: int,
    target_door_lock_ratio: float,
    door_lock_positive_ratio: float,
) -> int:
    if not 0 < target_door_lock_ratio < 1:
        raise ValueError("target_door_lock_ratio must be between 0 and 1.")
    if not 0 < door_lock_positive_ratio <= 1:
        raise ValueError("door_lock_positive_ratio must be between 0 and 1.")
    if target_door_lock_ratio >= door_lock_positive_ratio:
        raise ValueError("target_door_lock_ratio must be lower than the generated door-lock positive ratio.")
    if existing_total <= 0:
        return 0
    if existing_unlock / existing_total >= target_door_lock_ratio:
        return 0
    required = (target_door_lock_ratio * existing_total - existing_unlock) / (door_lock_positive_ratio - target_door_lock_ratio)
    return max(0, int(math.ceil(required)))


def resolve_sample_count(
    output_path: Path,
    append: bool,
    samples: int | None,
    target_unlock_ratio: float | None,
    door_lock_positive_ratio: float,
) -> int:
    if target_unlock_ratio is None:
        return DEFAULT_SAMPLES if samples is None else max(0, int(samples))
    if not append:
        return DEFAULT_SAMPLES if samples is None else max(0, int(samples))
    existing_total, existing_unlock = read_door_lock_counts(output_path)
    if existing_total == 0:
        return DEFAULT_SAMPLES if samples is None else max(0, int(samples))
    planned = plan_boost_sample_count(existing_total, existing_unlock, target_unlock_ratio, door_lock_positive_ratio)
    return max(planned, 0 if samples is None else int(samples))


def build_boost_jobs(
    rng: random.Random,
    total_samples: int,
    door_lock_ratio: float,
    contrast_ratio: float,
    lock_ratio: float,
    user_language: str,
) -> list[dict[str, Any]]:
    other_count, unlock_count, lock_count, contrast_count = split_mixed_counts(total_samples, door_lock_ratio, contrast_ratio, lock_ratio)
    selected = (
        weighted_select(OTHER_INTENT_SCENARIOS, other_count, rng)
        + weighted_select(UNLOCK_BOOST_SCENARIOS, unlock_count, rng)
        + weighted_select(LOCK_BOOST_SCENARIOS, lock_count, rng)
        + weighted_select(DOOR_LOCK_CONTRAST_SCENARIOS, contrast_count, rng)
    )
    rng.shuffle(selected)

    occurrence_by_scenario: dict[str, int] = {}
    jobs: list[dict[str, Any]] = []
    for idx, scenario in enumerate(selected, 1):
        scenario_name = scenario["scenario"]
        occurrence_idx = occurrence_by_scenario.get(scenario_name, 0)
        occurrence_by_scenario[scenario_name] = occurrence_idx + 1
        language = choose_language(user_language, idx + occurrence_idx, rng)
        jobs.append(
            {
                "idx": idx,
                "scenario": scenario,
                "occurrence_idx": occurrence_idx,
                "label": label_for_scenario(scenario, occurrence_idx),
                "language": language,
                "voice_shape": VOICE_SHAPES[(idx + rng.randrange(len(VOICE_SHAPES))) % len(VOICE_SHAPES)],
                "situation": SITUATIONS[(idx + rng.randrange(len(SITUATIONS))) % len(SITUATIONS)],
                "temperature": rng.uniform(0.85, 1.15),
            }
        )
    return jobs


def format_offline_unlock_boost(job: dict[str, Any], rng: random.Random) -> str:
    scenario_name = job["scenario"]["scenario"]
    if not scenario_name.startswith(("unlock_boost_", "lock_boost_", "door_lock_contrast_")):
        return format_base_offline_prompt(scenario_name, job["label"], job["occurrence_idx"], rng, job["language"])
    templates = ZH_OFFLINE_TEMPLATES if job["language"] == "chinese" else EN_OFFLINE_TEMPLATES
    text = rng.choice(templates[scenario_name])
    occurrence_idx = int(job["occurrence_idx"])
    if job["language"] == "chinese":
        prefixes = ["", "麻烦你", "那个，", "帮个忙，", "现在"]
        suffixes = ["", "谢谢。", "现在就弄。", "别等了。"]
        if occurrence_idx % 4 == 1:
            text = prefixes[occurrence_idx % len(prefixes)] + text
        if occurrence_idx % 5 == 2:
            text = text.rstrip("。！?") + "，" + suffixes[occurrence_idx % len(suffixes)]
    else:
        prefixes = ["", "Please, ", "Hey, ", "Could you ", "I need you to "]
        suffixes = ["", " Thanks.", " Right now.", " They are waiting."]
        if occurrence_idx % 4 == 1:
            prefix = prefixes[occurrence_idx % len(prefixes)]
            text = prefix + (text[0].lower() + text[1:] if prefix and text else text)
        if occurrence_idx % 5 == 2:
            text = text.rstrip(".!?") + suffixes[occurrence_idx % len(suffixes)]
    return normalize_spaces(text)


def generate_unlock_boost_utterance(
    client: OpenAICompatibleClient,
    job: dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    scenario = job["scenario"]
    label_data = job["label"]
    is_negative = label_data.get("matched") is False
    is_door_lock_row = scenario["scenario"].startswith(("unlock_boost_", "lock_boost_", "door_lock_contrast_"))
    prompt = [
        {
            "role": "system",
            "content": (
                "You write raw smart-home voice snippets for a device-control intent dataset with extra door-lock coverage. "
                "Answer with compact JSON only: {\"utterance\":\"...\"}. "
                "Do not include labels, analysis, markdown, translations, or assistant replies. "
                "Use fresh phrasing; avoid generic dataset language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one user utterance for a fixed training label.\n"
                f"Focus row: {scenario['scenario']}\n"
                f"Field note: {scenario['notes']}\n"
                f"Fixed label, do not change: {json.dumps(label_data, ensure_ascii=False)}\n"
                f"Language: {job['language']}\n"
                f"Voice shape: {job['voice_shape']}\n"
                f"Situation texture: {job['situation']}\n"
                f"Sample number: {job['occurrence_idx']}\n\n"
                "Requirements:\n"
                "- The sentence must sound like a real smart-home or home-assistant user request.\n"
                "- Keep normalized label strings in English only inside the fixed label; the generated user sentence may use the requested language.\n"
                + (
                    "- For matched=true and intent=unlock, it must clearly mean remote door unlocking/opening.\n"
                    "- For matched=true and intent=lock, it must clearly mean remotely locking/securing the door.\n"
                    "- For during_call=true, mention an active call, intercom, downstairs call, or before hanging up.\n"
                    "- For unlock, use alternatives such as buzz them in, let them in, open the entrance, or release the lock when natural.\n"
                    "- For lock, use alternatives such as secure the door, lock up, make sure it is locked, or lock the entrance when natural.\n"
                    if is_door_lock_row
                    else "- Match the fixed non-door-lock device capability and slots exactly; do not drift into door-lock wording.\n"
                )
                + (
                    "- Because this row is matched=false, mention door/lock words only as status questions, memory, reminders, keys, batteries, or background talk; do not ask to unlock/open/lock the door.\n"
                    if is_negative and is_door_lock_row
                    else "- Because this row is matched=false, it must not express any supported device-control request.\n"
                    if is_negative
                    else ""
                )
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            utterance = clean_boost_utterance(client.chat(prompt, temperature=temperature, max_tokens=max_tokens))
            if utterance:
                return utterance
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"generate_unlock_boost_utterance failed after retries: {last_err}")


def generate_unlock_boost_dataset(
    output_path: Path,
    client: OpenAICompatibleClient | None,
    report_path: Path | None = None,
    samples: int | None = None,
    target_unlock_ratio: float | None = None,
    door_lock_ratio: float = DEFAULT_DOOR_LOCK_RATIO,
    contrast_ratio: float = DEFAULT_CONTRAST_RATIO,
    lock_ratio: float = DEFAULT_LOCK_RATIO,
    user_language: str = "mixed",
    seed: int = 4242,
    workers: int = 1,
    temperature: float = 0.95,
    max_tokens: int = 120,
    max_retries: int = 3,
    dedupe_retries: int = 2,
    offline: bool = False,
    append: bool = True,
) -> int:
    contrast_ratio = max(0.0, min(float(contrast_ratio), 0.95))
    door_lock_positive_ratio = max(0.0, min(float(door_lock_ratio), 1.0) - max(0.0, min(float(contrast_ratio), 0.95)))
    sample_count = resolve_sample_count(output_path, append, samples, target_unlock_ratio, door_lock_positive_ratio)
    rng = random.Random(seed)
    jobs = build_boost_jobs(rng, sample_count, door_lock_ratio, contrast_ratio, lock_ratio, user_language)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = count_existing_rows(output_path) if append else 0
    seen_utterances = load_existing_utterances(output_path) if append else set()
    seen_lock = threading.Lock()

    def _run_job(job: dict[str, Any]) -> dict[str, Any]:
        scenario = job["scenario"]
        label_data = job["label"]
        generation_source = "unlock_boost_offline" if offline or client is None else "unlock_boost_api"
        generation_error: str | None = None
        local_rng = random.Random(seed + int(job["idx"]) * 7919)
        if offline or client is None:
            utterance = format_offline_unlock_boost(job, local_rng)
        else:
            utterance = ""
            attempts = max(1, int(dedupe_retries) + 1)
            try:
                for attempt in range(attempts):
                    candidate = generate_unlock_boost_utterance(
                        client=client,
                        job=job,
                        temperature=max(0.2, min(1.2, temperature + job["temperature"] - 0.95 + attempt * 0.05)),
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )
                    normalized = normalize_spaces(candidate)
                    with seen_lock:
                        if normalized not in seen_utterances or attempt == attempts - 1:
                            seen_utterances.add(normalized)
                            utterance = candidate
                            break
            except Exception as exc:
                generation_source = "unlock_boost_offline_fallback"
                generation_error = str(exc)
                utterance = format_offline_unlock_boost(job, local_rng)

        sample = build_training_sample(utterance, label_data)
        sample["_idx"] = job["idx"]
        sample["scenario"] = scenario["scenario"]
        sample["generation_source"] = generation_source
        if generation_error:
            sample["generation_error"] = generation_error
        return sample

    completed = 0
    errors = 0
    mode = "a" if append else "w"
    print(
        f"starting unlock boost generation: total={len(jobs)}, existing={existing_rows}, mode={mode}, "
        f"workers={max(1, int(workers))}, offline={offline}, door_lock_ratio={door_lock_ratio}, "
        f"contrast_ratio={contrast_ratio}, lock_ratio={lock_ratio}",
        flush=True,
    )
    with output_path.open(mode, encoding="utf-8") as output_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                sample = future.result()
                completed += 1
                if sample.get("generation_error"):
                    errors += 1
                    print(
                        f"[warn] idx={sample['_idx']} scenario={sample['scenario']} used offline fallback: {sample['generation_error']}",
                        flush=True,
                    )
                out = dict(sample)
                out.pop("_idx", None)
                output_file.write(json.dumps(out, ensure_ascii=False) + "\n")
                output_file.flush()
                print(
                    f"[progress] completed={completed}/{len(jobs)} written_this_run={completed} "
                    f"file_rows={existing_rows + completed} errors={errors} scenario={sample['scenario']}",
                    flush=True,
                )

    if report_path is not None:
        write_dataset_report(output_path, report_path)
    return len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append door-lock focused device intent training rows without adding new capability classes.")
    parser.add_argument("--output", type=Path, default=Path("data/device_intent_dataset.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/device_intent_dataset.stats.json"))
    parser.add_argument("--samples", type=int, default=None, help="Rows to add. Defaults to 1000, or a computed count when --target-unlock-ratio is set.")
    parser.add_argument("--target-unlock-ratio", type=float, default=None, help="When appending, add enough rows to reach this final capability_id=4 door-lock ratio.")
    parser.add_argument("--door-lock-ratio", type=float, default=DEFAULT_DOOR_LOCK_RATIO, help="Fraction of added rows assigned to capability_id=4 door-lock coverage, including contrast negatives.")
    parser.add_argument("--contrast-ratio", type=float, default=DEFAULT_CONTRAST_RATIO, help="Fraction of added rows that are matched=false door/lock contrast negatives.")
    parser.add_argument("--lock-ratio", type=float, default=DEFAULT_LOCK_RATIO, help="Fraction of added rows that are positive lock-door rows. Door-lock positive remainder is unlock-door rows.")
    parser.add_argument("--user-language", choices=["english", "chinese", "mixed"], default="mixed")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dedupe-retries", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="Use local door-lock templates instead of an API.")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True, help="Append to the output JSONL by default. Use --no-append to overwrite.")
    args = parser.parse_args()

    client: OpenAICompatibleClient | None = None
    if not args.offline:
        env_file = load_env_file(args.api_env)
        base_url = resolve_required("base_url", [args.base_url, env_file.get("DISTILL_API_BASE_URL"), env_file.get("OPENAI_BASE_URL"), os.getenv("DISTILL_API_BASE_URL"), os.getenv("OPENAI_BASE_URL")])
        api_key = resolve_required("api_key", [args.api_key, env_file.get("DISTILL_API_KEY"), env_file.get("OPENAI_API_KEY"), os.getenv("DISTILL_API_KEY"), os.getenv("OPENAI_API_KEY")])
        model = resolve_required("model", [args.model, env_file.get("DISTILL_API_MODEL"), env_file.get("OPENAI_MODEL"), os.getenv("DISTILL_API_MODEL"), os.getenv("OPENAI_MODEL")])
        client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model, endpoint=args.endpoint, timeout=args.timeout_sec)

    count = generate_unlock_boost_dataset(
        output_path=args.output,
        client=client,
        report_path=args.report,
        samples=args.samples,
        target_unlock_ratio=args.target_unlock_ratio,
        door_lock_ratio=args.door_lock_ratio,
        contrast_ratio=args.contrast_ratio,
        lock_ratio=args.lock_ratio,
        user_language=args.user_language,
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        dedupe_retries=args.dedupe_retries,
        offline=args.offline,
        append=args.append,
    )
    total, door_lock_count = read_door_lock_counts(args.output)
    ratio = door_lock_count / total if total else 0.0
    print(f"generated={count} output={args.output} door_lock_rows={door_lock_count}/{total} door_lock_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
