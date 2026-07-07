from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app


def make_client(tmp_path):
    app = create_app(tmp_path / "api_reminders.json")
    return TestClient(app)


def test_health_and_tools(tmp_path):
    client = make_client(tmp_path)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    tools = client.get("/tools")
    assert tools.status_code == 200
    names = {t["function"]["name"] for t in tools.json()}
    assert {"create_reminder", "query_reminder", "update_reminder", "delete_reminder"}.issubset(names)


def test_reminder_crud_api(tmp_path):
    client = make_client(tmp_path)

    created = client.post(
        "/reminders",
        json={"time_text": "tomorrow 10am", "task": "doctor", "target": "self"},
    )
    body = created.json()
    assert created.status_code == 200
    assert body["state"] == "success"
    rid = body["reminder_id"]

    queried = client.get("/reminders", params={"task": "doctor"})
    qbody = queried.json()
    assert queried.status_code == 200
    assert qbody["status"] == "success"
    assert len(qbody["reminders"]) == 1

    updated = client.patch(f"/reminders/{rid}", json={"new_task": "hospital"})
    ubody = updated.json()
    assert updated.status_code == 200
    assert ubody["status"] == "success"
    assert ubody["reminder"]["task"] == "hospital"

    deleted = client.delete(f"/reminders/{rid}")
    dbody = deleted.json()
    assert deleted.status_code == 200
    assert dbody["status"] == "success"

    not_found = client.get("/reminders", params={"task": "hospital"}).json()
    assert not_found["status"] == "not_found"
    assert not_found["state"] is False


def test_execute_tools_endpoint(tmp_path):
    client = make_client(tmp_path)
    payload = {
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "create_reminder",
                        "arguments": json.dumps(
                            {"time_text": "tomorrow 7pm", "task": "call daughter", "target": "self"}
                        ),
                    },
                }
            ],
        }
    }

    resp = client.post("/tools/execute", json=payload)
    assert resp.status_code == 200
    tool_messages = resp.json()["tool_messages"]
    assert len(tool_messages) == 1

    tool_payload = json.loads(tool_messages[0]["content"])
    assert tool_messages[0]["role"] == "tool"
    assert tool_payload["status"] == "success"
    assert tool_payload["state"] == "success"


def test_execute_tools_missing_fields(tmp_path):
    client = make_client(tmp_path)
    payload = {
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "create_reminder",
                        "arguments": json.dumps({"time_text": "tomorrow 7pm"}),
                    },
                }
            ],
        }
    }
    resp = client.post("/tools/execute", json=payload)
    assert resp.status_code == 200
    tool_payload = json.loads(resp.json()["tool_messages"][0]["content"])
    assert tool_payload["status"] == "missing_fields"
    assert tool_payload["state"] is False
    assert "task" in tool_payload["missing_fields"]
