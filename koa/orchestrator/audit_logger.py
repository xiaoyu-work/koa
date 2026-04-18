"""
Structured audit logging for orchestrator decisions.

Produces JSON log entries via Python's standard logging module under
the ``koa.audit`` logger name.  Each entry includes a timestamp,
event_type, optional tenant_id, and event-specific fields.

Request correlation is driven by the ``koa.observability.context``
ContextVars (``request_id_var`` / ``tenant_id_var``).  This means every
``AuditLogger`` call automatically inherits the current request id even
when many requests share a single ``AuditLogger`` instance (which is the
case for the orchestrator — see ``Orchestrator._audit``).

Usage::

    from koa.observability import bind_request_context

    with bind_request_context(tenant_id="user-123"):
        audit.start_request(tenant_id="user-123", message="check my email")
        audit.log_phase("intent_analysis", {"domains": ["communication"]})
        audit.end_request(status="completed", token_usage={...})
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..observability.context import (
    get_request_id,
    get_tenant_id,
    new_request_id,
    request_id_var,
)

_audit_logger = logging.getLogger("koa.audit")


class AuditLogger:
    """Structured audit logger with per-request tracing.

    The current ``request_id`` is read from :data:`koa.observability.context.request_id_var`
    so concurrent requests on the same instance do not overwrite each other.
    ``start_request`` will set the ContextVar if it is unset; callers that use
    :func:`koa.observability.bind_request_context` can simply rely on the
    already-bound value.
    """

    def __init__(self, tenant_id: Optional[str] = None) -> None:
        self._default_tenant_id = tenant_id
        # Per-request start timestamps keyed by request_id so multiple
        # concurrent requests on the same logger instance don't collide.
        self._starts: Dict[str, float] = {}
        # Optional token set when start_request() binds the ContextVar itself
        # (i.e. the caller did not already set it).  Reset in end_request().
        self._ctx_tokens: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Request tracing
    # ------------------------------------------------------------------

    def start_request(
        self,
        tenant_id: str,
        message: str,
        request_id: Optional[str] = None,
    ) -> str:
        """Begin tracing a new request. Returns the request_id.

        If ``request_id_var`` is unset, binds it here (and stores the reset
        token for :meth:`end_request`).  If already set by an outer
        ``bind_request_context`` block, uses the bound value.
        """
        rid = request_id or get_request_id() or new_request_id()
        if get_request_id() != rid:
            token = request_id_var.set(rid)
            self._ctx_tokens[rid] = token
        self._starts[rid] = time.monotonic()
        self._default_tenant_id = tenant_id
        self._emit(
            "request_start",
            {
                "tenant_id": tenant_id,
                "message_preview": message[:120] if message else "",
            },
        )
        return rid

    def end_request(
        self,
        status: str = "completed",
        token_usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End tracing the current request."""
        rid = get_request_id()
        fields: Dict[str, Any] = {"status": status}
        start = self._starts.pop(rid, None) if rid else None
        if start is not None:
            fields["total_ms"] = int((time.monotonic() - start) * 1000)
        if token_usage:
            fields["token_usage"] = token_usage
        if error:
            fields["error"] = error
        self._emit("request_end", fields)
        if rid and rid in self._ctx_tokens:
            try:
                request_id_var.reset(self._ctx_tokens.pop(rid))
            except ValueError:
                # Token from a different context — ignore.
                pass

    def log_phase(
        self,
        phase: str,
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a named phase within the current request.

        Useful for: intent_analysis, domain_filtering, tool_loading,
        memory_recall, true_memory_extraction, post_process, etc.
        """
        fields: Dict[str, Any] = {
            "tenant_id": self._tid(tenant_id),
            "phase": phase,
        }
        rid = get_request_id()
        start = self._starts.get(rid) if rid else None
        if start is not None:
            fields["elapsed_ms"] = int((time.monotonic() - start) * 1000)
        if details:
            fields.update(details)
        self._emit("phase", fields)

    @property
    def request_id(self) -> Optional[str]:
        """Current request id from the ContextVar."""
        return get_request_id()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, fields: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }
        rid = get_request_id()
        if rid:
            entry["request_id"] = rid
        entry.update(fields)
        _audit_logger.info(json.dumps(entry, default=str))

    def _tid(self, tenant_id: Optional[str] = None) -> str:
        return tenant_id or get_tenant_id() or self._default_tenant_id or ""

    # ------------------------------------------------------------------
    # public API (existing)
    # ------------------------------------------------------------------

    def log_policy_decision(
        self,
        intent: str,
        must_use_tools: bool,
        selected_tools: List[str],
        reason_code: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a tool-policy routing decision."""
        self._emit(
            "policy_decision",
            {
                "tenant_id": self._tid(tenant_id),
                "intent": intent,
                "must_use_tools": must_use_tools,
                "selected_tools": selected_tools,
                "selected_tools_count": len(selected_tools),
                "reason_code": reason_code,
            },
        )

    def log_route_decision(
        self,
        tenant_id: str,
        target_agent_id: Optional[str],
        waiting_agents_count: int,
        reason: str,
    ) -> None:
        """Log a WAITING-agent routing decision."""
        self._emit(
            "route_decision",
            {
                "tenant_id": tenant_id,
                "target_agent_id": target_agent_id,
                "waiting_agents_count": waiting_agents_count,
                "reason": reason,
            },
        )

    def log_tool_execution(
        self,
        tool_name: str,
        args_summary: Dict[str, Any],
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a tool execution result."""
        fields: Dict[str, Any] = {
            "tenant_id": self._tid(tenant_id),
            "tool_name": tool_name,
            "args_summary": args_summary,
            "success": success,
            "duration_ms": duration_ms,
        }
        if error is not None:
            fields["error"] = error
        self._emit("tool_execution", fields)

    def log_approval_decision(
        self,
        agent_name: str,
        tool_name: str,
        risk_level: str,
        decision: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log an approval decision."""
        self._emit(
            "approval_decision",
            {
                "tenant_id": self._tid(tenant_id),
                "agent_name": agent_name,
                "tool_name": tool_name,
                "risk_level": risk_level,
                "decision": decision,
            },
        )

    def log_react_turn(
        self,
        turn: int,
        tool_calls: List[str],
        final_answer: bool,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a ReAct turn summary."""
        self._emit(
            "react_turn",
            {
                "tenant_id": self._tid(tenant_id),
                "turn": turn,
                "tool_calls": tool_calls,
                "tool_calls_count": len(tool_calls),
                "final_answer": final_answer,
            },
        )
