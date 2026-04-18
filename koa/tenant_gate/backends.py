"""Pluggable backend types for the tenant gate.

The gate delegates its state (rate-limit counters, concurrency counts,
budgets) to a backend so single-process installations can use an
in-memory implementation while horizontally-scaled deployments plug in
Redis/Postgres.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Protocol


class GateBackend(Protocol):
    """Storage backend for gate concurrency counters.

    Rate-limit + budget state live in :class:`TokenBucketLimiter` /
    :class:`BudgetTracker` which are already pluggable.  This protocol
    covers the remaining in-flight concurrency count.
    """

    #: Whether this backend is multi-process safe.  ``False`` for in-memory.
    multiprocess_safe: bool

    async def incr_inflight(self, tenant_id: str) -> int: ...

    async def decr_inflight(self, tenant_id: str) -> int: ...

    async def get_inflight(self, tenant_id: str) -> int: ...


class InMemoryGateBackend:
    """Process-local concurrency counter.  Not production-safe."""

    multiprocess_safe: bool = False

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def incr_inflight(self, tenant_id: str) -> int:
        async with self._lock:
            n = self._counts.get(tenant_id, 0) + 1
            self._counts[tenant_id] = n
            return n

    async def decr_inflight(self, tenant_id: str) -> int:
        async with self._lock:
            n = max(0, self._counts.get(tenant_id, 0) - 1)
            self._counts[tenant_id] = n
            return n

    async def get_inflight(self, tenant_id: str) -> int:
        async with self._lock:
            return self._counts.get(tenant_id, 0)
