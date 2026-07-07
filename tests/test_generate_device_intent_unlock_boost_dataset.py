from __future__ import annotations

import json

from scripts.generate_device_intent_dataset import build_training_sample, label
from scripts.generate_device_intent_unlock_boost_dataset import (
    generate_unlock_boost_dataset,
    plan_boost_sample_count,
    read_door_lock_counts,
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assistant_label(row):
    return json.loads(row["messages"][-1]["content"])


def test_unlock_boost_dataset_offline_keeps_schema_and_increases_door_lock(tmp_path):
    out = tmp_path / "device_intent_dataset.jsonl"
    report = tmp_path / "device_intent_dataset.stats.json"

    count = generate_unlock_boost_dataset(
        output_path=out,
        client=None,
        report_path=report,
        samples=100,
        door_lock_ratio=0.5,
        contrast_ratio=0.1,
        lock_ratio=0.2,
        offline=True,
        workers=1,
        user_language="english",
        append=False,
    )

    rows = read_rows(out)
    labels = [assistant_label(row) for row in rows]
    door_lock_labels = [item for item in labels if item["matched"] is True and item["capability_id"] == 4]
    lock_labels = [item for item in door_lock_labels if item["intent"] == "lock"]
    unlock_labels = [item for item in door_lock_labels if item["intent"] == "unlock"]
    negative_labels = [item for item in labels if item["matched"] is False]

    assert count == 100
    assert len(rows) == 100
    assert len(door_lock_labels) == 40
    assert len(unlock_labels) == 20
    assert len(lock_labels) == 20
    assert len(negative_labels) >= 10
    assert len({item["capability_id"] for item in labels if item["capability_id"] is not None}) > 1
    assert all(item["slots"].get("method") == "remote" for item in door_lock_labels)
    assert any(item["slots"].get("during_call") is True for item in unlock_labels)
    assert all(item["capability_id"] is None and item["intent"] is None for item in negative_labels)

    for row in rows:
        assert row["task"] == "device_intent_slot_extraction"
        assert row["schema_version"] == "device_intent_v2"
        assert row["generation_source"] == "unlock_boost_offline"
        assert "tools" not in row
        assert all("tool_calls" not in message for message in row["messages"])
        assert set(assistant_label(row)) == {
            "matched",
            "capability_id",
            "capability",
            "intent",
            "slots",
            "missing_slots",
            "confidence",
        }

    stats = json.loads(report.read_text(encoding="utf-8"))
    assert stats["total"] == 100
    assert stats["capability_distribution"]["4"] == 40
    assert stats["intent_distribution"]["unlock"] == 20
    assert stats["intent_distribution"]["lock"] == 20
    assert "8" in stats["capability_distribution"]


def test_unlock_boost_appends_to_existing_dataset_and_reports_ratio(tmp_path):
    out = tmp_path / "device_intent_dataset.jsonl"
    base_rows = [
        build_training_sample("Turn the volume up.", label(True, 8, "set_volume", {"adjustment": "up"})),
        build_training_sample("Unlock the door.", label(True, 4, "unlock", {"method": "remote"}, confidence=0.95)),
        build_training_sample("Lock the door.", label(True, 4, "lock", {"method": "remote"}, confidence=0.95)),
        build_training_sample("Tell me a joke.", label(False, None, None)),
    ]
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in base_rows), encoding="utf-8")

    count = generate_unlock_boost_dataset(
        output_path=out,
        client=None,
        samples=7,
        door_lock_ratio=1.0,
        contrast_ratio=0.0,
        lock_ratio=0.3,
        offline=True,
        workers=1,
        user_language="english",
        append=True,
    )

    total, door_lock_count = read_door_lock_counts(out)
    assert count == 7
    assert total == 11
    assert door_lock_count == 9


def test_plan_boost_sample_count_reaches_target_ratio():
    planned = plan_boost_sample_count(
        existing_total=4000,
        existing_unlock=240,
        target_door_lock_ratio=0.2,
        door_lock_positive_ratio=0.4,
    )

    final_total = 4000 + planned
    final_door_lock = 240 + planned * 0.4
    assert planned == 2800
    assert final_door_lock / final_total >= 0.2
