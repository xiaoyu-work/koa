"""Koa observability primitives.

Provides production-grade observability scaffolding:

- ``context``: request/tenant ContextVars (PEP 567) that propagate across
  ``await`` boundaries and are inherited by spawned ``asyncio.Task`` children.
- ``tracing``: optional OpenTelemetry integration (no-op if the SDK is not
  installed).  Call :func:`trace_span` wherever you want a span.
- ``metrics``: Prometheus integration with an in-memory fallback so callers
  can always record metrics even without ``prometheus_client`` installed.
- ``logging_setup``: opt-in JSON logger that injects the current request_id
  and tenant_id into every record.
- ``task_registry``: :class:`TaskRegistry` that wraps ``asyncio.create_task``
  with exception logging, cancellation tracking, and cancel-all on shutdown.

All components default to no-op behavior; importing this package never
performs network/IO side effects.  Integrate at orchestrator init:

.. code-block:: python

    from koa.observability import (
        configure_metrics, configure_tracing,
        request_id_var, tenant_id_var, TaskRegistry,
    )

    configure_metrics(enabled=True, prometheus=True)
    configure_tracing(enabled=True, service_name="koa")
"""

from .context import (
    bind_request_context,
    get_idempotency_key,
    get_request_id,
    get_tenant_id,
    idempotency_key_var,
    new_request_id,
    request_id_var,
    tenant_id_var,
)
from .logging_setup import configure_logging
from .metrics import (
    configure_metrics,
    counter,
    get_metrics_registry,
    histogram,
    observe,
)
from .task_registry import TaskRegistry, get_task_registry
from .tracing import configure_tracing, get_tracer, trace_span

__all__ = [
    "bind_request_context",
    "configure_logging",
    "configure_metrics",
    "configure_tracing",
    "counter",
    "get_idempotency_key",
    "get_metrics_registry",
    "get_request_id",
    "get_task_registry",
    "get_tenant_id",
    "get_tracer",
    "histogram",
    "idempotency_key_var",
    "new_request_id",
    "observe",
    "request_id_var",
    "TaskRegistry",
    "tenant_id_var",
    "trace_span",
]
