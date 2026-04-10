"""Unified tool execution pipeline with before/after hooks.

Provides a structured execution pipeline for tool calls:
1. Pre-execution hooks (auth validation, parameter normalization)
2. Tool execution (with timeout)
3. Post-execution hooks (result truncation, audit logging)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..models import AgentTool, AgentToolContext

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
    """

    def __init__(self) -> None:
        self._before_hooks: List[BeforeHook] = []
        self._after_hooks: List[AfterHook] = []

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

        # Phase 1: Before hooks
        effective_args = dict(args)
        for hook in self._before_hooks:
            try:
                hook_result = await hook(tool, effective_args, context)
                if isinstance(hook_result, BeforeHookResult):
                    if not hook_result.proceed:
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
            return ToolExecutionResult(
                tool_name=tool_name,
                result=f"Tool '{tool_name}' timed out after {timeout}s",
                duration_ms=duration_ms,
                success=False,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
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
