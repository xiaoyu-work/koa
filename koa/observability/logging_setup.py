"""Structured JSON logging with request-context injection.

Install via :func:`configure_logging` at startup.  Each log record gains
``request_id`` and ``tenant_id`` fields sourced from :mod:`koa.observability.context`
ContextVars so every line is correlatable without threading manual state.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from .context import get_request_id, get_tenant_id


class _ContextInjectingFilter(logging.Filter):
    """Inject request_id / tenant_id onto every ``LogRecord``."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.request_id = get_request_id() or "-"
        record.tenant_id = get_tenant_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter.

    Emits one JSON object per line with: timestamp, level, logger, message,
    request_id, tenant_id, and any extra attributes attached via ``extra=``.
    """

    RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "tenant_id": getattr(record, "tenant_id", "-"),
        }
        for k, v in record.__dict__.items():
            if k in self.RESERVED or k in payload:
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    level: int = logging.INFO,
    json_format: bool = True,
    stream: Any = None,
) -> None:
    """Configure the root logger with optional JSON formatting + context injection.

    Safe to call multiple times; existing handlers on the root logger are
    replaced.  Callers who manage handlers themselves may skip this helper
    and simply add the ``_ContextInjectingFilter`` to their own handlers.
    """
    stream = stream or sys.stderr
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s "
                "[req=%(request_id)s tenant=%(tenant_id)s] %(message)s"
            )
        )
    handler.addFilter(_ContextInjectingFilter())
    root.addHandler(handler)
    root.setLevel(level)
