from app.tool_registry import get_tool, get_tools


def test_tool_registry_has_required_tools():
    tools = get_tools()
    names = {t["function"]["name"] for t in tools}
    assert {"create_reminder", "query_reminder", "update_reminder", "delete_reminder"}.issubset(names)


def test_create_schema_required_fields():
    schema = get_tool("create_reminder")
    assert schema is not None
    required = schema["function"]["parameters"]["required"]
    assert required == ["time_text", "task"]
