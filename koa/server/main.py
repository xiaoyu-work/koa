"""CLI argument parsing and uvicorn entry point."""

import json
import logging
import os
import sys


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Koa — AI Agent Framework")
    subparsers = parser.add_subparsers(dest="command")

    # koa serve (default when no subcommand)
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=os.getenv("KOA_HOST", "127.0.0.1"))
    serve_parser.add_argument("--port", type=int, default=int(os.getenv("KOA_PORT", "8000")))

    # koa chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with a running Koa server")
    chat_parser.add_argument(
        "--url",
        default=os.getenv("KOA_URL", "http://localhost:8000"),
        help="Koa server URL (default: http://localhost:8000)",
    )
    chat_parser.add_argument(
        "--api-key",
        default=os.getenv("KOA_API_KEY"),
        help="API key for authentication (default: $KOA_API_KEY)",
    )
    chat_parser.add_argument(
        "--tenant-id",
        default="default",
        help="Tenant ID for multi-tenant mode (default: 'default')",
    )

    # Backward compat: bare `koa` or `koa --host/--port` starts server
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    if args.command == "chat":
        _run_chat(args)
    else:
        _run_serve(args)


def _run_serve(args):
    """Start the Koa API server."""
    import uvicorn

    host = args.host or os.getenv("KOA_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("KOA_PORT", "8000"))

    if os.getenv("LOG_FORMAT", "json") == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")

    from .app import api

    uvicorn.run(api, host=host, port=port)


def _run_chat(args):
    """Interactive CLI chat client."""
    from .cli_chat import chat_loop

    chat_loop(url=args.url, api_key=args.api_key, tenant_id=args.tenant_id)
