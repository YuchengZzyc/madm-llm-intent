# AGENTS.md

## Project Goal

Build a minimal but reliable backend and execution harness for a small local model to learn and use tools in a standard tool-use loop.

The final target is:

1. The user sends a natural language message.
2. The model receives the conversation history and tool descriptions.
3. The model may output a structured tool call.
4. The runtime detects the tool call and pauses natural language generation.
5. The backend executes the tool call.
6. The tool result is appended back into the conversation as a `tool` message.
7. The model continues and produces the final user-facing answer.
8. If required fields are missing, the tool returns `missing_fields`, and the model asks the user for the missing information.
9. If multiple reminders match, the tool returns `ambiguous`, and the model asks the user to choose one.
10. The model must never fabricate tool execution results.

The first tool domain is reminders:

- create reminder
- query reminder
- update reminder
- delete reminder

---

## Core Development Philosophy

This project follows a harness-driven development style.

The coding agent should not jump directly into a large implementation. It should:

1. Build a small, testable backend first.
2. Define stable tool schemas before model integration.
3. Make every feature independently testable.
4. Add progress notes after each meaningful change.
5. Keep implementation simple and inspectable.
6. Avoid premature UI work.
7. Avoid modifying tests to make code pass.
8. Prefer clear, boring code over clever abstractions.

The project should be designed so that a future coding agent can resume work from `feature.json` and `progress.md` without needing the original conversation.

---

## Agent Roles

### 1. Initializer Agent

Responsibilities:

- Inspect the repository structure.
- Create the initial project skeleton if missing.
- Create or update:
  - `AGENTS.md`
  - `feature.json`
  - `progress.md`
- Set up a minimal backend service.
- Set up a minimal test suite.
- Add a local storage mechanism for reminders.
- Define canonical tool schemas.
- Ensure the project can be run locally.
- Ensure every implemented feature has a simple verification command.

The initializer agent should not implement advanced model inference first. Backend tool correctness comes first.

Expected first milestone:

- A backend service exposes reminder CRUD endpoints.
- A tool registry exposes tool descriptions.
- Unit tests or simple API tests verify reminder creation, querying, updating, deletion, missing field handling, and ambiguous match handling.

---

### 2. Coding Agent

Responsibilities:

- Pick one unfinished feature from `feature.json`.
- Implement only that feature or a small coherent group of related features.
- Add or update tests.
- Run the relevant tests.
- Update `progress.md`.
- Mark the feature status in `feature.json`.
- Leave clear notes if something is incomplete.

The coding agent must not:

- Remove tests to make the build pass.
- Hide failures.
- Change public tool schema without updating tests and documentation.
- Hard-code model outputs as if they were real model calls.
- Allow the assistant to fabricate tool results.

---

### 3. Reviewer Agent

Responsibilities:

- Check whether the tool-use loop is logically correct.
- Check whether backend responses are machine-readable and stable.
- Check whether missing-field and ambiguous cases are handled.
- Check whether the tool schema matches actual backend behavior.
- Check whether `feature.json` and `progress.md` reflect reality.
- Check whether tests cover both success and failure cases.

---

## Required Architecture

The implementation should be organized around four layers.

### Layer 1: Reminder Backend

A deterministic backend that manages reminder records.

Suggested reminder object:

```json
{
  "reminder_id": "rem_0001",
  "task": "给女儿打电话",
  "scheduled_time": "2026-04-25T19:00:00+08:00",
  "time_text": "后天晚上7点",
  "target": "self",
  "status": "active",
  "created_at": "2026-04-23T09:00:00+08:00",
  "updated_at": "2026-04-23T09:00:00+08:00"
}
```

Required backend functions:

- `create_reminder(time_text, task, target="self")`
- `query_reminder(time_text=None, task=None, target="self")`
- `update_reminder(reminder_id=None, time_text=None, task=None, new_time_text=None, new_task=None, target="self")`
- `delete_reminder(reminder_id=None, time_text=None, task=None, target="self")`

Required backend result statuses:

- `success`
- `missing_fields`
- `not_found`
- `ambiguous`
- `error`

The backend must not return free-form-only results. Every result must be structured JSON.

---

### Layer 2: Tool Registry

A canonical tool registry should define the model-visible tools.

Each tool needs:

- `type`
- `function.name`
- `function.description`
- `function.parameters`

Use OpenAI-compatible function/tool schema style.

Example:

```json
{
  "type": "function",
  "function": {
    "name": "create_reminder",
    "description": "Create a reminder when the user clearly wants to be reminded about a task at a time.",
    "parameters": {
      "type": "object",
      "properties": {
        "time_text": {
          "type": "string",
          "description": "The user's original time expression, such as 明天晚上8点."
        },
        "task": {
          "type": "string",
          "description": "The thing the user wants to be reminded about."
        },
        "target": {
          "type": "string",
          "enum": ["self"],
          "default": "self"
        }
      },
      "required": ["time_text", "task"],
      "additionalProperties": false
    }
  }
}
```

The tool registry must be used by both:

1. The model prompt/chat template.
2. The backend validation layer.

Do not maintain two inconsistent copies of the tool schema.

---

### Layer 3: Tool-Use Runtime Loop

The runtime should implement the standard tool-use procedure.

Required loop:

1. Build messages:
   - `system`
   - conversation history
   - user message
   - tool descriptions
2. Call model.
3. Inspect assistant output.
4. If assistant output contains no tool call:
   - return assistant answer to user.
5. If assistant output contains tool call:
   - validate tool name and arguments.
   - execute backend function.
   - append tool result:
     ```json
     {
       "role": "tool",
       "tool_call_id": "call_xxx",
       "name": "create_reminder",
       "content": "{\"status\":\"success\", ...}"
     }
     ```
   - call model again with updated messages.
   - return final assistant answer.
6. Stop after a safe maximum number of tool iterations.

The runtime must prevent infinite tool-call loops.

Recommended max tool iterations:

```text
max_tool_rounds = 3
```

---

### Layer 4: Model Integration Script

The first version can use a mock model.

Do not require a real model before the backend and runtime are tested.

Implementation stages:

1. Mock model outputs hard-coded assistant tool calls for tests.
2. Local model integration through transformers or vLLM.
3. Qwen chat template integration with `tools`.
4. Tool-call parser and fallback parser.
5. Real end-to-end demo.

The model integration must use standard tool-use message format:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_0001",
      "type": "function",
      "function": {
        "name": "create_reminder",
        "arguments": "{\"time_text\":\"明天下午4点\",\"task\":\"吃药\",\"target\":\"self\"}"
      }
    }
  ]
}
```

Tool results must use:

```json
{
  "role": "tool",
  "tool_call_id": "call_0001",
  "name": "create_reminder",
  "content": "{\"status\":\"success\", ...}"
}
```

---

## Critical Rules

### Tool Result Rule

The assistant must never invent tool results.

Wrong:

```json
{
  "role": "assistant",
  "content": "已经帮您删除了。"
}
```

before the backend has returned success.

Correct:

```json
{
  "role": "assistant",
  "tool_calls": [...]
}
```

then:

```json
{
  "role": "tool",
  "content": "{\"status\":\"success\"}"
}
```

then:

```json
{
  "role": "assistant",
  "content": "已经帮您删除这个提醒了。"
}
```

---

### Missing Fields Rule

If required parameters are missing, the backend should return:

```json
{
  "status": "missing_fields",
  "missing_fields": ["task"],
  "message": "Missing required field: task."
}
```

The model should then ask the user naturally:

```text
好的，我可以提醒您。您想让我提醒什么事情呢？
```

---

### Ambiguous Match Rule

If delete or update matches multiple reminders, return:

```json
{
  "status": "ambiguous",
  "candidates": [
    {
      "reminder_id": "rem_0001",
      "task": "吃药",
      "scheduled_time": "2026-04-24T08:00:00+08:00"
    },
    {
      "reminder_id": "rem_0002",
      "task": "吃药",
      "scheduled_time": "2026-04-24T20:00:00+08:00"
    }
  ]
}
```

The model should ask the user to choose.

---

### Not Found Rule

If no reminder matches, return:

```json
{
  "status": "not_found",
  "message": "No matching reminder found."
}
```

The model should explain that no matching reminder was found and ask if the user wants to create a new one or try another query.

---

## Suggested Repository Structure

```text
.
├── AGENTS.md
├── feature.json
├── progress.md
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── storage.py
│   ├── reminder_service.py
│   ├── tool_registry.py
│   ├── tool_executor.py
│   └── runtime_loop.py
├── scripts/
│   ├── demo_backend.py
│   ├── demo_tool_loop.py
│   └── call_local_model.py
├── tests/
│   ├── test_reminder_service.py
│   ├── test_tool_registry.py
│   ├── test_tool_executor.py
│   └── test_runtime_loop.py
└── data/
    └── reminders.json
```

---

## Suggested Tech Stack

Default stack:

- Python 3.10+
- FastAPI for backend API
- Pydantic for schema validation
- JSON file storage for first version
- pytest for tests
- optional: transformers / vLLM for model integration later

Do not introduce a database until JSON storage is stable.

---

## First Implementation Order

1. Project skeleton.
2. Pydantic reminder models.
3. JSON storage.
4. Reminder service functions.
5. Tool registry.
6. Tool executor.
7. FastAPI endpoints.
8. Unit tests for reminder service.
9. Unit tests for tool executor.
10. Mock model runtime loop.
11. Local model integration script.
12. Standard tool-use training data exporter.

---

## Done Definition

A feature is done only when:

1. Code is implemented.
2. Basic tests pass.
3. The feature is marked as passed in `feature.json`.
4. `progress.md` records what changed, how to test it, and what remains.
