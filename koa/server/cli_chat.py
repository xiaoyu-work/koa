"""Interactive CLI chat client for Koa.

Connects to a running Koa server's /stream endpoint and provides
a terminal-based chat interface with streaming output.

Usage:
    koa chat                              # connect to localhost:8000
    koa chat --url http://remote:8000     # connect to remote server
    koa chat --api-key sk-xxx             # with authentication
"""

import json
import sys
from typing import Optional


def chat_loop(
    url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    tenant_id: str = "default",
) -> None:
    """Run the interactive chat REPL."""
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx")
        sys.exit(1)

    stream_url = f"{url.rstrip('/')}/stream"
    health_url = f"{url.rstrip('/')}/health"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Check server connectivity
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(health_url)
            info = resp.json()
            version = info.get("version", "?")
            auth = info.get("auth_enabled", False)
            print(f"\033[90mConnected to Koa v{version} at {url}")
            if auth and not api_key:
                print("\033[33m⚠ Server has auth enabled but no --api-key provided\033[0m")
            print(f"\033[90mType your message and press Enter. Ctrl+C to quit.\033[0m")
    except Exception as e:
        print(f"\033[31m✗ Cannot connect to {url}: {e}\033[0m")
        print(f"\033[90mMake sure the server is running: koa serve\033[0m")
        sys.exit(1)

    print()

    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\n\033[90mBye!\033[0m")
            break

        if not user_input.strip():
            continue

        payload = json.dumps({
            "message": user_input,
            "tenant_id": tenant_id,
        })

        sys.stdout.write("\033[1;32mKoa:\033[0m ")
        sys.stdout.flush()

        try:
            _stream_response(stream_url, headers, payload)
        except KeyboardInterrupt:
            print("\n\033[90m(interrupted)\033[0m")
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")

        print()


def _stream_response(url: str, headers: dict, payload: str) -> None:
    """Stream SSE response and print to terminal."""
    import httpx

    full_text = ""
    in_message = False

    with httpx.Client(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10)) as client:
        with client.stream("POST", url, content=payload, headers=headers) as resp:
            if resp.status_code == 401:
                print("\033[31mAuthentication failed. Use --api-key.\033[0m")
                return
            if resp.status_code == 503:
                print("\033[31mServer not configured. Set up config.yaml first.\033[0m")
                return
            if resp.status_code != 200:
                print(f"\033[31mHTTP {resp.status_code}\033[0m")
                return

            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")
                    edata = event.get("data", "")

                    if etype == "message_start":
                        in_message = True

                    elif etype == "message_chunk" and in_message:
                        text = ""
                        if isinstance(edata, dict):
                            text = edata.get("chunk") or edata.get("content") or ""
                        elif isinstance(edata, str):
                            text = edata
                        if text:
                            full_text += text
                            sys.stdout.write(text)
                            sys.stdout.flush()

                    elif etype == "message_end":
                        in_message = False

                    elif etype == "tool_call_start":
                        name = ""
                        if isinstance(edata, dict):
                            name = edata.get("tool_name") or edata.get("name") or "tool"
                        elif isinstance(edata, str):
                            name = edata
                        display = _format_tool_name(name)
                        sys.stdout.write(f"\n  \033[33m⚙ {display}...\033[0m")
                        sys.stdout.flush()

                    elif etype in ("tool_call_end", "tool_result"):
                        success = True
                        if isinstance(edata, dict):
                            success = edata.get("success", True)
                        mark = "\033[32m✓\033[0m" if success else "\033[31m✗\033[0m"
                        sys.stdout.write(f" {mark}\n")
                        sys.stdout.flush()

                    elif etype == "error":
                        msg = ""
                        if isinstance(edata, dict):
                            msg = edata.get("error") or edata.get("message") or str(edata)
                        else:
                            msg = str(edata)
                        print(f"\n\033[31mError: {msg}\033[0m")

    if not full_text:
        pass  # tool-only response or empty


def _format_tool_name(name: str) -> str:
    """Format agent/tool name for display: 'EmailAgent' -> 'Email'."""
    import re
    name = re.sub(r"Agent$", "", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.replace("_", " ").strip() or name
