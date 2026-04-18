"""P0-5: Tool schema validation rejects hallucinated and malformed calls."""

from koa.llm.tool_validator import ToolSchemaValidator


def _schema(params):
    return [{"type": "function", "function": {"name": "add", "parameters": params}}]


def test_unknown_tool_rejected():
    v = ToolSchemaValidator.from_openai_tools(_schema({"type": "object"}))
    r = v.validate("evil_rm", {})
    assert not r.ok
    assert r.reason == "unknown_tool"


def test_bad_name_rejected():
    v = ToolSchemaValidator.from_openai_tools(_schema({"type": "object"}))
    r = v.validate("../etc/passwd", {})
    assert not r.ok
    assert r.reason == "invalid_tool_name"


def test_non_dict_args_rejected():
    v = ToolSchemaValidator.from_openai_tools(_schema({"type": "object"}))
    r = v.validate("add", "not-a-dict")
    assert not r.ok


def test_schema_mismatch_rejected():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    v = ToolSchemaValidator.from_openai_tools(_schema(schema))
    assert not v.validate("add", {}).ok
    assert not v.validate("add", {"x": "str"}).ok
    assert not v.validate("add", {"x": 1, "extra": 2}).ok
    assert v.validate("add", {"x": 1}).ok


def test_known_names():
    v = ToolSchemaValidator.from_openai_tools(_schema({"type": "object"}))
    assert v.known_names == ["add"]
