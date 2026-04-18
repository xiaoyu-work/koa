"""Unified tool execution pipeline with before/after hooks.

Provides a structured execution pipeline for tool calls:
1. Pre-execution hooks (auth validation, parameter normalization)
2. Idempotency check (when a store is configured and the tool is idempotent
   or the caller supplied an idempotency key)
3. Tool execution (with timeout)
4. Post-execution hooks (result truncation, audit logging)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..models import AgentTool, AgentToolContext
from ..observability.context import get_idempotency_key
from ..observability.metrics import counter
from ..tenant_gate.idempotency import make_idempotency_key

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Structured result from the tool pipeline."""

    tool_name: str
    result: Any
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    was_truncated: bool = False
    original_chars: int = 0


@dataclass
class BeforeHookResult:
    """Result from a before-execution hook."""

    proceed: bool = True
    error_message: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None


BeforeHook = Callable[..., Any]  # async (tool, args, context) -> BeforeHookResult
AfterHook = Callable[..., Any]  # async (tool, result, context) -> result


class ToolPipeline:
    """Structured pipeline for tool execution with before/after hooks.

    Hooks are run in registration order. Before-hooks can block execution
    or modify arguments. After-hooks can transform results.

    Args:
        idempotency_store: Optional :class:`koa.tenant_gate.IdempotencyStore`.
            When provided, tool calls that either (a) carry a caller-supplied
            idempotency key (``koa.observability.context.idempotency_key_var``)
            or (b) declare ``idempotent=True`` on the AgentTool are deduped
            against the store.  Duplicate calls return the cached result
            without re-executing.
        idempotency_ttl: TTL (seconds) for idempotency records.
    """

    def __init__(
        self,
        idempotency_store: Optional[Any] = None,
        idempotency_ttl: float = 3600.0,
    ) -> None:
        self._before_hooks: List[BeforeHook] = []
        self._after_hooks: List[AfterHook] = []
        self._idem_store = idempotency_store
        self._idem_ttl = idempotency_ttl

    def add_before_hook(self, hook: BeforeHook) -> None:
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: AfterHook) -> None:
        self._after_hooks.append(hook)

    async def execute(
        self,
        tool: AgentTool,
        args: Dict[str, Any],
        context: AgentToolContext,
        timeout: int = 30,
    ) -> ToolExecutionResult:
        """Execute a tool through the full pipeline."""
        tool_name = tool.name
        start = time.monotonic()

        # Phase 0: Idempotency lookup.  We compute the key before running
        # before-hooks so cached results skip credential checks too (the
        # cached result was produced under equivalent conditions earlier).
        idem_key = None
        caller_key = get_idempotency_key()
        tool_is_idempotent = bool(getattr(tool, "idempotent", False))
        if self._idem_store is not None and (caller_key or tool_is_idempotent):
            idem_key = make_idempotency_key(
                tenant_id=context.tenant_id,
                session_id=getattr(context, "session_id", None)
                or (context.metadata or {}).get("session_id"),
                tool_name=tool_name,
                arguments=args,
                caller_key=caller_key,
            )
            is_new, existing = await self._idem_store.begin(idem_key, self._idem_ttl)
            if not is_new and existing is not None and existing.status == "completed":
                counter("koa_tool_idempotency_hit_total", {"tool": tool_name[:32]}, 1)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    result=existing.result,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    success=True,
                )
            if not is_new and existing is not None and existing.status == "in_flight":
                counter("koa_tool_idempotency_conflict_total", {"tool": tool_name[:32]}, 1)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    result=f"Duplicate in-flight request for {tool_name!r}; "
                    f"please wait for the original to complete.",
                    duration_ms=int((time.monotonic() - start) * 1000),
                    success=False,
                    error="idempotency_in_flight",
                )

        # Phase 1: Before hooks
        effective_args = dict(args)
        for hook in self._before_hooks:
            try:
                hook_result = await hook(tool, effective_args, context)
                if isinstance(hook_result, BeforeHookResult):
                    if not hook_result.proceed:
                        if idem_key is not None and self._idem_store is not None:
                            await self._idem_store.fail(idem_key)
                        return ToolExecutionResult(
                            tool_name=tool_name,
                            result=hook_result.error_message or "Blocked by pre-execution check",
                            duration_ms=int((time.monotonic() - start) * 1000),
                            success=False,
                            error=hook_result.error_message,
                        )
                    if hook_result.modified_args is not None:
                        effective_args = hook_result.modified_args
            except Exception as e:
                logger.warning(f"[ToolPipeline] Before-hook failed for {tool_name}: {e}")

        # Phase 2: Execute
        try:
            result = await asyncio.wait_for(
                tool.executor(effective_args, context),
                timeout=timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - start) * 1000)
            if idem_key is not None and self._idem_store is not None:
                await self._idem_store.fail(idem_key)
            return ToolExecutionResult(
                tool_name=tool_name,
                result=f"Tool '{tool_name}' timed out after {timeout}s",
                duration_ms=duration_ms,
                success=False,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            if idem_key is not None and self._idem_store is not None:
                await self._idem_store.fail(idem_key)
            return ToolExecutionResult(
                tool_name=tool_name,
                result=e,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )

        # Phase 3: After hooks
        for hook in self._after_hooks:
            try:
                result = await hook(tool, result, context)
            except Exception as e:
                logger.warning(f"[ToolPipeline] After-hook failed for {tool_name}: {e}")

        # Phase 4: Record idempotency result
        if idem_key is not None and self._idem_store is not None:
            try:
                await self._idem_store.complete(idem_key, result)
            except Exception as exc:
                logger.debug("idempotency complete() failed: %s", exc)

        return ToolExecutionResult(
            tool_name=tool_name,
            result=result,
            duration_ms=duration_ms,
            success=True,
        )


# ── Built-in hooks ────────────────────────────────────────────────────


async def credential_check_hook(
    tool: AgentTool,
    args: Dict[str, Any],
    context: AgentToolContext,
) -> BeforeHookResult:
    """Check that required credentials are available before execution."""
    if not context.credentials:
        return BeforeHookResult(proceed=True)

    # Tools that explicitly declare they need OAuth
    if getattr(tool, "requires_oauth", False):
        tenant_id = context.tenant_id
        has_creds = await context.credentials.has_valid_credentials(tenant_id)
        if not has_creds:
            return BeforeHookResult(
                proceed=False,
                error_message=f"Tool '{tool.name}' requires authentication. Please connect your account first.",
            )

    return BeforeHookResult(proceed=True)


async def result_audit_hook(
    tool: AgentTool,
    result: Any,
    context: AgentToolContext,
) -> Any:
    """Log tool execution result for audit trail."""
    result_str = str(result) if result is not None else ""
    result_len = len(result_str)
    logger.debug(f"[ToolPipeline] {tool.name} completed: {result_len} chars")
    return result
