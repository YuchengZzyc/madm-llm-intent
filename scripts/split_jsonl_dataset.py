from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def get_scenario(row: dict[str, Any]) -> str:
    md = row.get("metadata")
    if isinstance(md, dict):
        s = md.get("scenario")
        if isinstance(s, str) and s.strip():
            return s.strip()
    return "__unknown__"


def stratified_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(get_scenario(r), []).append(r)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for _, items in buckets.items():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        if len(items) > 1:
            n_val = max(1, min(n_val, len(items) - 1))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def random_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows2 = list(rows)
    rng.shuffle(rows2)
    n_val = int(round(len(rows2) * val_ratio))
    if len(rows2) > 1:
        n_val = max(1, min(n_val, len(rows2) - 1))
    return rows2[n_val:], rows2[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train/val JSONL files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio, e.g. 0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by-scenario", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not (0 < args.val_ratio < 1):
        raise SystemExit("--val-ratio must be between 0 and 1")

    rows = load_jsonl(args.input)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 samples to split")

    if args.stratify_by_scenario:
        train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed)
    else:
        train_rows, val_rows = random_split(rows, args.val_ratio, args.seed)

    save_jsonl(args.train_output, train_rows)
    save_jsonl(args.val_output, val_rows)

    report = {
        "input": str(args.input),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "stratify_by_scenario": args.stratify_by_scenario,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

