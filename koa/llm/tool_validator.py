"""Validate LLM tool_calls against the advertised schema.

Two classes of risk are mitigated here:

1. **Hallucinated tool names** — the model invents a tool that was not in
   the schema list sent to it.  Executing such a call could collide with a
   future tool name or crash the dispatcher.
2. **Malformed arguments** — arguments not matching the declared JSON
   schema can crash the tool or, worse, bypass intended validation in the
   tool handler.

Both are rejected at the boundary.  Invalid calls are never coerced; the
caller must decide whether to surface an error to the model, retry, or
fail the turn.

Validation runs at *two* points:

- Immediately after LiteLLM parses the model response.
- Just before the orchestrator dispatches the tool, using the
  *request-scoped* tool list (which may include dynamically injected
  tools like ``notify_user`` and MCP tools not present at the static
  LLM-client layer).

The validator accepts either a list of tool schemas in the OpenAI format
(``{"type": "function", "function": {"name": "...", "parameters": {...}}}``)
or a list of ``AgentTool``-like objects exposing ``.name`` and
``.parameters_schema``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..observability.metrics import counter

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import Draft202012Validator

    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover - jsonschema is in requirements-dev
    _HAS_JSONSCHEMA = False


# Reject names that contain control characters or look like attempts to
# hijack reserved identifiers.  Regular tool names are kebab/snake case.
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-:.]{0,63}$")


@dataclass
class ValidationResult:
    """Outcome of a single tool-call validation."""

    ok: bool
    reason: str = ""
    details: Optional[Dict[str, Any]] = None


@dataclass
class _SchemaEntry:
    name: str
    parameters: Dict[str, Any]


class ToolSchemaValidator:
    """Validate model-emitted tool calls against an allowlist + JSON schema.

    Usage::

        validator = ToolSchemaValidator.from_openai_tools(schemas)
        result = validator.validate("send_email", {"to": "a@b.com"})
        if not result.ok:
            raise ValueError(result.reason)
    """

    def __init__(self, entries: Iterable[_SchemaEntry]):
        self._entries: Dict[str, _SchemaEntry] = {e.name: e for e in entries}
        # Precompile validators where possible.
        self._validators: Dict[str, Any] = {}
        if _HAS_JSONSCHEMA:
            for name, entry in self._entries.items():
                schema = entry.parameters or {}
                try:
                    self._validators[name] = Draft202012Validator(schema)
                except jsonschema.SchemaError as exc:
                    logger.warning(
                        "Tool %r has an invalid JSON schema; argument validation disabled: %s",
                        name,
                        exc,
                    )

    # ---- Builders ----------------------------------------------------

    @classmethod
    def from_openai_tools(
        cls, tool_schemas: Optional[List[Dict[str, Any]]]
    ) -> "ToolSchemaValidator":
        """Build from the OpenAI ``tools=[{"type": "function", "function": {...}}]`` shape."""
        entries: List[_SchemaEntry] = []
        for tool in tool_schemas or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str):
                continue
            entries.append(_SchemaEntry(name=name, parameters=fn.get("parameters") or {}))
        return cls(entries)

    @classmethod
    def from_agent_tools(cls, tools: Iterable[Any]) -> "ToolSchemaValidator":
        """Build from objects exposing ``.name`` and ``.parameters_schema``/``.parameters``."""
        entries: List[_SchemaEntry] = []
        for t in tools:
            name = getattr(t, "name", None)
            if not isinstance(name, str):
                continue
            schema = (
                getattr(t, "parameters_schema", None)
                or getattr(t, "parameters", None)
                or getattr(t, "input_schema", None)
                or {}
            )
            entries.append(_SchemaEntry(name=name, parameters=schema))
        return cls(entries)

    # ---- Validation --------------------------------------------------

    @property
    def known_names(self) -> List[str]:
        return sorted(self._entries.keys())

    def validate(self, name: str, arguments: Any) -> ValidationResult:
        """Validate a single ``(name, arguments)`` pair.

        ``arguments`` must be a dict (or convertible) — LLM parsers should
        JSON-decode strings before calling.
        """
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            counter("koa_tool_validation_failed_total", {"reason": "bad_name"}, 1)
            return ValidationResult(False, "invalid_tool_name", {"name": name})

        if name not in self._entries:
            counter("koa_tool_validation_failed_total", {"reason": "unknown_tool"}, 1)
            return ValidationResult(
                False,
                "unknown_tool",
                {"name": name, "known": self.known_names},
            )

        if not isinstance(arguments, dict):
            counter("koa_tool_validation_failed_total", {"reason": "bad_args_type"}, 1)
            return ValidationResult(
                False,
                "arguments_not_object",
                {"name": name, "type": type(arguments).__name__},
            )

        validator = self._validators.get(name)
        if validator is None:
            # Either jsonschema is unavailable or the schema itself was invalid.
            # Accept on permissive grounds but emit a metric so operators can see it.
            counter("koa_tool_validation_unchecked_total", {"name": name[:32]}, 1)
            return ValidationResult(True)

        errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
        if errors:
            first = errors[0]
            counter("koa_tool_validation_failed_total", {"reason": "bad_args"}, 1)
            return ValidationResult(
                False,
                "arguments_schema_mismatch",
                {
                    "name": name,
                    "path": list(first.absolute_path),
                    "message": first.message,
                },
            )
        return ValidationResult(True)

    def validate_many(
        self,
        calls: Iterable[Tuple[str, Any]],
    ) -> List[Tuple[ValidationResult, Optional[Tuple[str, Any]]]]:
        """Validate multiple calls; returns one result per input call."""
        return [(self.validate(name, args), (name, args)) for name, args in calls]


# --- Convenience helpers used by the orchestrator ---------------------


def validate_or_raise(
    validator: ToolSchemaValidator,
    name: str,
    arguments: Union[Dict[str, Any], str],
) -> Dict[str, Any]:
    """Decode stringified arguments, validate, and raise on failure.

    Returns the parsed/validated argument dict on success.
    """
    import json

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            counter("koa_tool_validation_failed_total", {"reason": "bad_json"}, 1)
            raise ValueError(f"Tool {name!r} arguments are not valid JSON: {exc}") from exc
    else:
        parsed = arguments or {}

    result = validator.validate(name, parsed)
    if not result.ok:
        raise ValueError(f"Tool call rejected ({result.reason}): {result.details}")
    return parsed
