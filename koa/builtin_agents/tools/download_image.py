"""
Download Image Tool - Download an image and return base64 data for storage.

Used after the LLM selects the best image from search results.
The base64 data can be stored directly in the client's local database.
"""

import base64
import logging

import httpx

from koa.models import AgentToolContext, ToolOutput

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 10_000_000  # 10 MB
_TIMEOUT = 30.0
_USER_AGENT = (
    "Mozilla/5.0 (compatible; Koa/1.0; +https://github.com/withkoi/koa)"
)


async def download_image_executor(args: dict, context: AgentToolContext = None):
    """Download an image from a URL and return it as base64 data.

    Returns a ``ToolOutput`` so the image can be delivered to the client
    for local storage.
    """
    url = (args.get("url") or "").strip()
    if not url:
        return "Error: url is required."

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Block private networks
    import re
    _BLOCKED = re.compile(
        r"^https?://"
        r"(localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|0\.0\.0\.0|\[::1?\])",
        re.IGNORECASE,
    )
    if _BLOCKED.match(url):
        return "Error: cannot download from internal or private network URLs."

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)

        if resp.status_code >= 400:
            return f"Error: HTTP {resp.status_code} downloading {url}"

        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return f"Error: URL did not return an image (content-type: {content_type})"

        if len(resp.content) > _MAX_IMAGE_BYTES:
            return f"Error: Image too large ({len(resp.content)} bytes, max {_MAX_IMAGE_BYTES})"

        mime = content_type.split(";")[0].strip()
        b64 = base64.b64encode(resp.content).decode()

        return ToolOutput(
            text=f"Successfully downloaded image from {url} ({len(resp.content)} bytes, {mime}).",
            media=[{
                "type": "image",
                "data": f"data:{mime};base64,{b64}",
                "media_type": mime,
                "metadata": {
                    "source_url": url,
                    "size_bytes": len(resp.content),
                    "for_storage": True,
                },
            }],
        )

    except httpx.TimeoutException:
        return f"Error: download timed out for {url}"
    except Exception as e:
        logger.error(f"download_image error for {url}: {e}", exc_info=True)
        return f"Error downloading image: {e}"


DOWNLOAD_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL of the image to download. The image will be downloaded and returned as base64 data for local storage.",
        },
    },
    "required": ["url"],
}
