"""
Approval System - Structured approval requests per design doc section 13.2

Provides:
- ApprovalRequest dataclass for structured approval presentation
- build_approval_request() to create requests from agent state
- collect_batch_approvals() for batching multiple approval requests
- ApprovalStore protocol + InMemoryApprovalStore with TTL expiration,
  used by the orchestrator to enforce timeouts on pending approvals.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """Structured approval request per design doc section 13.2."""

    agent_name: str
    action_summary: str
    risk_level: str = "write"  # "read", "write", "destructive"
    details: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["approve", "edit", "cancel"])
    timeout_minutes: int = 30
    allow_modification: bool = True
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def build_approval_request(agent) -> ApprovalRequest:
    """Build ApprovalRequest from agent's get_approval_prompt() + collected_fields.

    Args:
        agent: A StandardAgent instance in WAITING_FOR_APPROVAL state.

    Returns:
        ApprovalRequest populated from the agent's state.
    """
    agent_name = agent.__class__.__name__

    # Get the approval prompt from the agent
    action_summary = ""
    if hasattr(agent, "get_approval_prompt"):
        action_summary = agent.get_approval_prompt()

    # Gather collected fields as details
    details: Dict[str, Any] = {}
    if hasattr(agent, "collected_fields"):
        details = dict(agent.collected_fields)

    # Extract risk_level from agent's pending tool call if available
    risk_level = "write"
    if hasattr(agent, "_pending_tool_call") and agent._pending_tool_call:
        _, tool, _ = agent._pending_tool_call
        if hasattr(tool, "risk_level"):
            risk_level = tool.risk_level

    return ApprovalRequest(
        agent_name=agent_name,
        action_summary=action_summary,
        risk_level=risk_level,
        details=details,
    )


# Risk ordering used by batch presentation (highest first so destructive
# actions surface to the user before low-risk reads).
_RISK_ORDER = {"destructive": 0, "write": 1, "read": 2}


def collect_batch_approvals(approval_requests: List[ApprovalRequest]) -> List[ApprovalRequest]:
    """Group approval requests by risk level, destructive-first.

    Args:
        approval_requests: List of individual ApprovalRequest objects.

    Returns:
        Requests sorted so highest-risk actions appear first.  This keeps
        user attention focused on the ones that matter and lets UIs render
        collapsible sections per risk tier.
    """
    if not approval_requests:
        return []

    logger.debug(
        "Collecting %d approval requests for batch presentation",
        len(approval_requests),
    )
    return sorted(
        approval_requests,
        key=lambda r: _RISK_ORDER.get((r.risk_level or "write").lower(), 99),
    )


# ---------------------------------------------------------------------------
# Approval store (P1-10)
# ---------------------------------------------------------------------------


class ApprovalStore(Protocol):
    """Persistent store for pending approvals.

    The orchestrator records an approval when an agent reaches
    ``WAITING_FOR_APPROVAL`` and removes it when the user approves/rejects
    or the request times out.  Production deployments should provide a
    durable implementation (Redis/Postgres); the default in-memory store
    is suitable only for single-process dev and tests.
    """

    async def put(self, request: ApprovalRequest) -> None: ...

    async def get(self, request_id: str) -> Optional[ApprovalRequest]: ...

    async def resolve(self, request_id: str, decision: str) -> None: ...

    async def expire_overdue(self) -> List[ApprovalRequest]: ...


@dataclass
class _Entry:
    request: ApprovalRequest
    created_at: float
    deadline: float


class InMemoryApprovalStore:
    """Default in-process :class:`ApprovalStore` with TTL expiration."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def put(self, request: ApprovalRequest) -> None:
        deadline = time.time() + max(1, request.timeout_minutes) * 60.0
        async with self._lock:
            self._entries[request.request_id] = _Entry(
                request=request, created_at=time.time(), deadline=deadline
            )

    async def get(self, request_id: str) -> Optional[ApprovalRequest]:
        async with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            if time.time() > entry.deadline:
                self._entries.pop(request_id, None)
                return None
            return entry.request

    async def resolve(self, request_id: str, decision: str) -> None:
        async with self._lock:
            self._entries.pop(request_id, None)
        logger.info("Approval %s resolved: %s", request_id, decision)

    async def expire_overdue(self) -> List[ApprovalRequest]:
        """Remove and return all approvals whose deadline has passed.

        Callers may log an audit event or notify the user that a pending
        approval was auto-rejected.
        """
        now = time.time()
        expired: List[ApprovalRequest] = []
        async with self._lock:
            for rid, entry in list(self._entries.items()):
                if now > entry.deadline:
                    expired.append(entry.request)
                    self._entries.pop(rid, None)
        if expired:
            logger.info("Auto-rejected %d overdue approval(s)", len(expired))
        return expired
