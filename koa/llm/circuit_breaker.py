"""Circuit breaker for LLM API calls.

Prevents cascading failures when an LLM provider is down by fast-failing
requests after a threshold of consecutive failures. Automatically recovers
by allowing a probe request after a cooldown period.

States:
    CLOSED   → Normal operation. Failures increment counter.
    OPEN     → Provider assumed down. All calls fast-fail.
    HALF_OPEN → After recovery_timeout, a single probe call is allowed.
                Success → CLOSED. Failure → OPEN.

Recovery timeout is randomized within ``[1-jitter, 1+jitter] * recovery_timeout``
to avoid a thundering herd of probes hitting a freshly-recovered provider.
Only one concurrent caller may hold the half-open probe slot at a time; the
rest continue to fast-fail until the probe resolves.
"""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Optional

from ..observability.metrics import counter

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""

    def __init__(self, provider: str = "", retry_after: float = 0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for '{provider}'. Retry after {retry_after:.0f}s.")


class CircuitBreaker:
    """Per-provider circuit breaker.

    Args:
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning to HALF_OPEN
            (actual wait is jittered — see ``jitter``).
        provider_name: Label for logging/metrics.
        jitter: Fractional jitter applied to ``recovery_timeout``.  A value
            of ``0.2`` means the effective timeout is uniformly sampled from
            ``[0.8 * recovery_timeout, 1.2 * recovery_timeout]`` each time
            the circuit opens.  Set to 0 to disable.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        provider_name: str = "",
        jitter: float = 0.2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.provider_name = provider_name
        self.jitter = max(0.0, min(jitter, 1.0))

        self.failure_count: int = 0
        self._state: CircuitState = CircuitState.CLOSED
        self._opened_at: float = 0.0
        self._effective_timeout: float = recovery_timeout
        # Only one concurrent caller may hold the half-open probe slot.
        self._probe_lock = asyncio.Lock()
        self._probe_in_flight: bool = False

    def _sample_timeout(self) -> float:
        if self.jitter <= 0:
            return self.recovery_timeout
        low = self.recovery_timeout * (1.0 - self.jitter)
        high = self.recovery_timeout * (1.0 + self.jitter)
        return random.uniform(low, high)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._effective_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Check if a call is allowed. Raises CircuitBreakerOpenError if not.

        In HALF_OPEN state, if a probe is already in flight this raises so
        only one caller exercises the recovery path at a time.
        """
        current = self.state
        if current == CircuitState.OPEN:
            retry_after = self._effective_timeout - (time.monotonic() - self._opened_at)
            raise CircuitBreakerOpenError(
                provider=self.provider_name,
                retry_after=max(0, retry_after),
            )
        if current == CircuitState.HALF_OPEN and self._probe_in_flight:
            raise CircuitBreakerOpenError(
                provider=self.provider_name,
                retry_after=1.0,
            )

    async def acquire_probe(self) -> Optional[asyncio.Lock]:
        """Claim the half-open probe slot; returns the lock if acquired.

        Callers that want strict single-probe semantics across concurrent
        coroutines should ``await cb.acquire_probe()`` before the call and
        ``cb.release_probe()`` in a ``finally`` block.  Only the first caller
        transitions from OPEN → HALF_OPEN.  Others continue to fast-fail.
        """
        if self.state != CircuitState.HALF_OPEN:
            return None
        await self._probe_lock.acquire()
        self._probe_in_flight = True
        return self._probe_lock

    def release_probe(self) -> None:
        """Release the half-open probe slot (idempotent)."""
        self._probe_in_flight = False
        if self._probe_lock.locked():
            try:
                self._probe_lock.release()
            except RuntimeError:
                pass

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state != CircuitState.CLOSED:
            logger.info(
                f"[CircuitBreaker] {self.provider_name}: {self.state.value} → closed (success)"
            )
            counter(
                "koa_circuit_breaker_transition_total",
                {"provider": self.provider_name, "to": "closed"},
                1,
            )
        self._state = CircuitState.CLOSED
        self.failure_count = 0
        self.release_probe()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        current = self.state

        if current == CircuitState.HALF_OPEN:
            self._open("half_open probe failed")
        elif current == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open(f"{self.failure_count} consecutive failures")
        self.release_probe()

    def _open(self, reason: str) -> None:
        self._effective_timeout = self._sample_timeout()
        logger.warning(
            f"[CircuitBreaker] {self.provider_name}: "
            f"opening circuit ({reason}). "
            f"Will probe again in {self._effective_timeout:.1f}s."
        )
        counter(
            "koa_circuit_breaker_transition_total",
            {"provider": self.provider_name, "to": "open"},
            1,
        )
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
