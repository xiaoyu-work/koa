"""Rate limiters for tenant gate.

Two independent algorithms so callers can pick based on traffic shape:

- :class:`SlidingWindowLimiter` — exact count of requests in the last N
  seconds.  Use when strict bursts-per-minute SLAs matter.
- :class:`TokenBucketLimiter` — classic refill-over-time bucket.  Smoother
  behavior under bursty load but allows short spikes.

Both are ``asyncio``-safe and keyed by ``tenant_id``.  Both expose a
consistent ``acquire(tenant_id)`` interface that returns ``(allowed, retry_after)``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    """Token-bucket rate limiter.

    Args:
        capacity: Maximum tokens (burst size).
        refill_per_second: Steady-state rate.
    """

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill_per_second must be > 0")
        self.capacity = float(capacity)
        self.refill = float(refill_per_second)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, tenant_id: str, cost: float = 1.0) -> Tuple[bool, float]:
        """Try to consume ``cost`` tokens.

        Returns ``(allowed, retry_after_seconds)``.  ``retry_after`` is 0
        on success and the estimated wait until the request would succeed
        on rejection.
        """
        now = time.monotonic()
        async with self._lock:
            b = self._buckets.get(tenant_id)
            if b is None:
                b = _Bucket(tokens=self.capacity, updated_at=now)
                self._buckets[tenant_id] = b
            # Refill
            elapsed = max(0.0, now - b.updated_at)
            b.tokens = min(self.capacity, b.tokens + elapsed * self.refill)
            b.updated_at = now
            if b.tokens >= cost:
                b.tokens -= cost
                return True, 0.0
            deficit = cost - b.tokens
            return False, deficit / self.refill


class SlidingWindowLimiter:
    """Sliding-window request counter.

    Args:
        max_requests: Max requests allowed in ``window_seconds``.
        window_seconds: Sliding window size.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("max_requests and window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = float(window_seconds)
        self._events: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, tenant_id: str) -> Tuple[bool, float]:
        now = time.monotonic()
        async with self._lock:
            q = self._events.setdefault(tenant_id, deque())
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) < self.max_requests:
                q.append(now)
                return True, 0.0
            retry_after = self.window_seconds - (now - q[0])
            return False, max(0.0, retry_after)
