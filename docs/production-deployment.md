# Production Deployment Guide

This guide covers deploying Koa in production.  For dev/demo see the main
README.

## Required configuration

| Variable | Why | Default |
| --- | --- | --- |
| `KOA_ENV=production` | Enables fail-secure defaults across the stack | `development` |
| `KOA_API_KEY` | Auth for `/chat`, `/stream` endpoints | _(none — auth disabled)_ |
| `KOA_CREDENTIAL_KEY` | AES-256-GCM key for at-rest tool credentials | _(none — passthrough)_ |
| `KOA_REQUIRE_ENCRYPTION=1` | Fail startup if encryption not configured | auto-on in production |
| `KOA_MCP_CALL_TIMEOUT` | Per-RPC timeout for MCP servers (seconds) | `30` |
| `KOA_MCP_RESPONSE_CAP` | Max MCP tool response bytes | `1048576` |

When `KOA_ENV=production` and `KOA_CREDENTIAL_KEY` is absent, the server
**refuses to start**.  This is deliberate: silently storing plaintext
credentials in a production database is a critical security incident.
Generate a key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Vector store

The default (in-process) vector store is **NOT production-ready**: it is
process-local, non-durable, and does not survive restarts.  Choose one of:

### Option A — pgvector (recommended)

* Persistent alongside your Postgres tenancy metadata
* Supports `HNSW` and `IVFFLAT` indexes
* Single DB for both metadata and vectors → simpler backups

Tuning knobs:
```
CREATE INDEX ON koa_memory_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```
At query time use `SET hnsw.ef_search = 40;` for recall/latency balance.

### Option B — External vector DB (Qdrant / Weaviate / Pinecone)

Use when the embedding catalog dwarfs operational Postgres (≳10 M
vectors).  Pair with momex's `ExternalVectorBackend` (see
`koa/memory/momex.py`).

## Rate limiting & budgets

The orchestrator exposes a :class:`koa.tenant_gate.TenantGate` kwarg.
Configure:

* **Concurrency cap** — per-tenant active request limit
* **RPM** — sliding-window request count
* **Token budget** — rolling daily/monthly token cap
* **Cost budget** — rolling cost cap in USD

For multi-worker deployments you **must** supply a Redis-backed backend.
`TenantGateConfig(strict=True)` refuses to initialize with the in-memory
backend when `KOA_ENV=production`.

## Observability

Install the `observability` extra for OTel + Prometheus wiring:

```bash
pip install 'koa[observability]'
```

Then in your entry point:

```python
from koa.observability import configure_logging, configure_metrics, configure_tracing

configure_logging()            # JSON logs, request_id auto-injected
configure_metrics()            # Prometheus registry
configure_tracing("koa-api")   # OTel (OTEL_EXPORTER_OTLP_ENDPOINT=...)
```

## Graceful shutdown

`Orchestrator.shutdown()` cancels all registered background tasks
(memory writes, backups, cleanup loops).  Run it before process exit:

```python
import signal, asyncio

app = ...  # your Koa instance

async def shutdown():
    await app.orchestrator.shutdown()

for sig in (signal.SIGTERM, signal.SIGINT):
    asyncio.get_event_loop().add_signal_handler(
        sig, lambda: asyncio.create_task(shutdown())
    )
```

## Checklist

- [ ] `KOA_ENV=production`
- [ ] `KOA_API_KEY` set
- [ ] `KOA_CREDENTIAL_KEY` set (48+ random bytes)
- [ ] Persistent vector backend (pgvector recommended)
- [ ] Tenant gate wired with Redis backend
- [ ] OTel exporter configured (`OTEL_EXPORTER_OTLP_ENDPOINT`)
- [ ] MCP server timeouts reviewed (`KOA_MCP_CALL_TIMEOUT`)
- [ ] Memory governance moderator reviewed for your content policy
- [ ] Graceful shutdown handler installed
