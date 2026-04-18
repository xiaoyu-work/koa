"""Per-tenant admission control for the orchestrator.

This package provides the gate sitting between the API entry point and the
ReAct loop.  Every request acquires a :class:`GateTicket` via
:meth:`TenantGate.acquire` which enforces:

- **Rate limit** — per-tenant RPM via sliding-window or token-bucket.
- **Concurrency** — max in-flight requests per tenant.
- **Token + cost budget** — cumulative usage per tenant per window.
- **Idempotency** — dedupe on a caller-supplied key so retries don't
  double-execute side effects.

Default implementations are in-memory (single-process).  Production
deployments with multiple workers must supply a shared backend (Redis,
Postgres) — see :class:`GateBackend` for the protocol.  The gate refuses
to initialize with an in-memory backend when ``strict=True`` so that
production configs fail-closed.
"""

from .backends import GateBackend, InMemoryGateBackend
from .budget import BudgetTracker, TokenCost
from .gate import (
    GateDecision,
    GateRejected,
    GateTicket,
    TenantGate,
    TenantGateConfig,
)
from .idempotency import (
    IdempotencyRecord,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    make_idempotency_key,
)
from .rate_limiter import SlidingWindowLimiter, TokenBucketLimiter

__all__ = [
    "BudgetTracker",
    "GateBackend",
    "GateDecision",
    "GateRejected",
    "GateTicket",
    "IdempotencyRecord",
    "IdempotencyStore",
    "InMemoryGateBackend",
    "InMemoryIdempotencyStore",
    "SlidingWindowLimiter",
    "TenantGate",
    "TenantGateConfig",
    "TokenBucketLimiter",
    "TokenCost",
    "make_idempotency_key",
]
