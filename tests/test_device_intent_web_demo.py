from __future__ import annotations

import json

from scripts.device_intent_web_demo import (
    PrintOnlyDeviceExecutor,
    SimulatedDeviceExecutor,
    build_app,
    build_training_prompt,
    parse_intent_output,
    reply_for_result,
)


def test_build_training_prompt_matches_adapter_format():
    prompt = build_training_prompt("Turn the volume up.")

    assert prompt.startswith("<TOOLS>\n[]\n</TOOLS>")
    assert "<SYSTEM>\nExtract device-control intent and slots. Return JSON only.\n</SYSTEM>" in prompt
    assert "<USER>\nTurn the volume up.\n</USER>" in prompt
    assert prompt.endswith("<ASSISTANT>")


def test_parse_intent_output_extracts_json_with_assistant_tag():
    raw = (
        '{"matched":true,"capability_id":8,"capability":"Adjust volume",'
        '"intent":"set_volume","slots":{"adjustment":"up"},'
        '"missing_slots":[],"confidence":0.94}</ASSISTANT>'
    )

    parsed = parse_intent_output(raw)

    assert parsed == {
        "matched": True,
        "capability_id": 8,
        "capability": "Adjust volume",
        "intent": "set_volume",
        "slots": {"adjustment": "up"},
        "missing_slots": [],
        "confidence": 0.94,
    }


def test_executor_prints_only_for_matched_intent(capsys):
    executor = SimulatedDeviceExecutor()
    label = {
        "matched": True,
        "capability_id": 8,
        "capability": "Adjust volume",
        "intent": "set_volume",
        "slots": {"adjustment": "up"},
        "missing_slots": [],
        "confidence": 0.94,
    }

    result = executor.execute(label)
    printed = json.loads(capsys.readouterr().out)

    assert result["status"] == "printed"
    assert result["state"]["volume"] == 60
    assert printed["event"] == "device_intent_postprocess"
    assert printed["intent"] == "set_volume"
    assert printed["slots"] == {"adjustment": "up"}
    assert printed["state"]["volume"] == 60


def test_executor_skips_not_matched(capsys):
    executor = PrintOnlyDeviceExecutor()

    result = executor.execute({"matched": False})

    assert result["status"] == "skipped"
    assert result["reason"] == "not_matched"
    assert result["state"]["volume"] == 50
    assert capsys.readouterr().out == ""


def test_executor_sets_volume_level(capsys):
    executor = SimulatedDeviceExecutor()
    result = executor.execute(
        {
            "matched": True,
            "capability_id": 8,
            "capability": "Adjust volume",
            "intent": "set_volume",
            "slots": {"adjustment": "set", "level": 60},
            "missing_slots": [],
            "confidence": 0.9,
        }
    )

    assert result["state"]["volume"] == 60
    assert result["changes"] == ["volume"]
    assert "Volume set to 60%" in result["state"]["last_action"]
    capsys.readouterr()


def test_executor_missing_slots_does_not_change_state(capsys):
    executor = SimulatedDeviceExecutor()
    result = executor.execute(
        {
            "matched": True,
            "capability_id": 6,
            "capability": "Change wallpaper",
            "intent": "change_wallpaper",
            "slots": {},
            "missing_slots": ["wallpaper_type"],
            "confidence": 0.8,
        }
    )

    assert result["status"] == "missing_slots"
    assert result["state"]["wallpaper"] == "default"
    assert capsys.readouterr().out == ""


def test_executor_locks_and_unlocks_door(capsys):
    executor = SimulatedDeviceExecutor()

    unlock_result = executor.execute(
        {
            "matched": True,
            "capability_id": 4,
            "capability": "Door lock control",
            "intent": "unlock",
            "slots": {"method": "remote"},
            "missing_slots": [],
            "confidence": 0.95,
        }
    )
    capsys.readouterr()

    lock_result = executor.execute(
        {
            "matched": True,
            "capability_id": 4,
            "capability": "Door lock control",
            "intent": "lock",
            "slots": {"method": "remote"},
            "missing_slots": [],
            "confidence": 0.95,
        }
    )
    capsys.readouterr()

    assert unlock_result["state"]["door_locked"] is False
    assert unlock_result["changes"] == ["door_locked"]
    assert lock_result["state"]["door_locked"] is True
    assert lock_result["changes"] == ["door_locked"]
    assert lock_result["state"]["last_action"] == "Door locked."


def test_reply_for_parse_failure_uses_raw_output():
    assert reply_for_result("hello", None, None) == "hello"


def test_api_response_includes_latency(monkeypatch):
    from fastapi.testclient import TestClient
    import scripts.device_intent_web_demo as demo

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, user_message):
            return {
                "text": '{"matched":true,"capability_id":4,"capability":"Door lock control","intent":"lock","slots":{"method":"remote"},"missing_slots":[],"confidence":0.95}',
                "latency": {
                    "latency_ms": 123.4,
                    "ttft_ms": 12.3,
                    "decode_after_first_ms": 100.0,
                    "post_last_token_overhead_ms": 1.1,
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "streamed_token_events": 20,
                    "tokens_per_second_total": 10.0,
                    "tokens_per_second_decode_only": 20.0,
                },
            }

    monkeypatch.setattr(demo, "DeviceIntentModel", FakeModel)
    client = TestClient(build_app(model_path="fake", adapter_path=None))

    response = client.post("/api/intent", json={"message": "Lock the door"})
    data = response.json()

    assert response.status_code == 200
    assert data["latency"]["latency_ms"] == 123.4
    assert data["latency"]["ttft_ms"] == 12.3
    assert data["parsed_intent"]["intent"] == "lock"
    assert data["postprocess_result"]["state"]["door_locked"] is True
