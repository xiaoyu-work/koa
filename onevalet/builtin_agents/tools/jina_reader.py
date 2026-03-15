"""
Jina Reader - Fetch and extract content from any URL via Jina Reader API.

Handles both static and JavaScript-rendered pages by using Jina's headless
browser infrastructure. Returns clean Markdown suitable for LLM consumption.

Usage:
    content = await jina_fetch("https://example.com")
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_JINA_BASE = "https://r.jina.ai/"
_TIMEOUT = 30.0
_MAX_CONTENT_CHARS = 15000


async def jina_fetch(
    url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = _TIMEOUT,
    max_chars: int = _MAX_CONTENT_CHARS,
) -> Optional[str]:
    """Fetch a URL via Jina Reader and return Markdown content.

    Returns None on failure so callers can decide how to handle it.
    """
    api_key = api_key or os.getenv("JINA_API_KEY", "")

    jina_url = f"{_JINA_BASE}{url}"
    headers: dict = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(jina_url, headers=headers, timeout=timeout)

        if resp.status_code >= 400:
            logger.warning("Jina Reader returned %d for %s", resp.status_code, url)
            return None

        content = resp.text.strip()
        if not content or len(content) < 50:
            logger.warning("Jina Reader returned empty/short content for %s", url)
            return None

        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated]"

        return content

    except httpx.TimeoutException:
        logger.warning("Jina Reader timed out for %s", url)
        return None
    except Exception as e:
        logger.error("Jina Reader error for %s: %s", url, e, exc_info=True)
        return None
