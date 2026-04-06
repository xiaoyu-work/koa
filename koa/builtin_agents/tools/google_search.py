"""
Google Search Tool - Web search via Google Custom Search API

Supports both regular web search and image search.

Requires environment variables:
- GOOGLE_SEARCH_API_KEY: Your Google API key
- GOOGLE_SEARCH_ENGINE_ID: Your Custom Search Engine ID
"""

import asyncio
import base64
import os
import logging
from typing import List, Dict, Any, Tuple

import httpx

from koa.models import AgentToolContext, ToolOutput

logger = logging.getLogger(__name__)

_THUMBNAIL_TIMEOUT = 8.0
_MAX_THUMBNAIL_BYTES = 300_000  # ~300 KB per thumbnail


async def _download_thumbnail(
    client: httpx.AsyncClient, url: str
) -> Tuple[str, str]:
    """Download a single thumbnail and return (base64_data, media_type).

    Returns empty strings on failure so the caller can skip it.
    """
    try:
        resp = await client.get(url, timeout=_THUMBNAIL_TIMEOUT)
        if resp.status_code >= 400 or len(resp.content) > _MAX_THUMBNAIL_BYTES:
            return "", ""
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        return base64.b64encode(resp.content).decode(), ct
    except Exception:
        return "", ""


async def _download_thumbnails(
    urls: List[str],
) -> List[Tuple[str, str]]:
    """Download multiple thumbnails concurrently."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "Koa/1.0"},
    ) as client:
        tasks = [_download_thumbnail(client, u) for u in urls]
        return await asyncio.gather(*tasks)


async def _brave_search_fallback(query: str, num_results: int = 5) -> str:
    """Fallback to Brave Search API when Google Search fails."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(num_results, 10)},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                timeout=15.0,
            )
            if response.status_code != 200:
                logger.warning(f"Brave Search fallback failed: {response.status_code}")
                return ""

            data = response.json()
            results = data.get("web", {}).get("results", [])
            if not results:
                return f"No results found for '{query}'."

            output = []
            for i, item in enumerate(results[:num_results], 1):
                title = item.get("title", "No title")
                url = item.get("url", "")
                description = item.get("description", "").replace("\n", " ")
                output.append(f"{i}. {title}\n   URL: {url}\n   {description}")

            return (
                f"Found results for '{query}' (via Brave Search).\n"
                f"Top {len(output)} results:\n\n" + "\n\n".join(output)
            )
    except Exception as e:
        logger.warning(f"Brave Search fallback error: {e}")
        return ""


async def google_search_executor(args: dict, context: AgentToolContext = None):
    """Search the web using Google Custom Search API.

    Returns a plain string for web searches, or a ``ToolOutput`` with
    embedded thumbnail media for image searches.
    """
    query = args.get("query", "")
    num_results = args.get("num_results", 5)
    search_type = args.get("search_type", "web")

    if not query:
        return "Error: No search query provided."

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        # Try Brave Search as primary if Google not configured
        brave_result = await _brave_search_fallback(query, num_results)
        if brave_result:
            return brave_result
        return (
            "Error: Google Search API not configured. "
            "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
        )

    params: Dict[str, Any] = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": min(num_results, 10),
    }
    if search_type == "image":
        params["searchType"] = "image"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(f"Google Search API error: {response.status_code} - {response.text}")
                # Fallback to Brave Search
                brave_result = await _brave_search_fallback(query, num_results)
                if brave_result:
                    return brave_result
                return f"Error: Search failed with status {response.status_code}"

            data = response.json()
            items = data.get("items", [])

            if not items:
                return f"No results found for '{query}'."

            # --- Web search (unchanged) ---
            if search_type != "image":
                output = []
                for i, item in enumerate(items, 1):
                    title = item.get("title", "No title")
                    link = item.get("link", "")
                    snippet = item.get("snippet", "").replace("\n", " ")
                    output.append(
                        f"{i}. {title}\n"
                        f"   URL: {link}\n"
                        f"   {snippet}"
                    )

                total_results = data.get("searchInformation", {}).get("totalResults", "?")
                return (
                    f"Found approximately {total_results} results for '{query}'.\n"
                    f"Top {len(items)} results:\n\n" + "\n\n".join(output)
                )

            # --- Image search ---
            thumb_urls = []
            image_records: List[Dict[str, Any]] = []
            for item in items:
                img_info = item.get("image", {})
                thumb_url = img_info.get("thumbnailLink", "")
                full_url = item.get("link", "")
                thumb_urls.append(thumb_url)
                image_records.append({
                    "title": item.get("title", ""),
                    "full_url": full_url,
                    "thumb_url": thumb_url,
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                    "context_url": item.get("image", {}).get("contextLink", ""),
                })

            # Download thumbnails so the LLM can visually review them
            thumb_results = await _download_thumbnails(thumb_urls)

            text_parts = []
            media_list: List[Dict[str, Any]] = []
            for i, (rec, (b64, mime)) in enumerate(
                zip(image_records, thumb_results), 1
            ):
                text_parts.append(
                    f"[Image {i}] {rec['title']}\n"
                    f"   Full URL: {rec['full_url']}\n"
                    f"   Size: {rec['width']}x{rec['height']}"
                )
                if b64:
                    media_list.append({
                        "type": "image",
                        "data": f"data:{mime};base64,{b64}",
                        "media_type": mime,
                        "metadata": {
                            "index": i,
                            "full_url": rec["full_url"],
                            "title": rec["title"],
                            "width": rec["width"],
                            "height": rec["height"],
                        },
                    })

            total_results = data.get("searchInformation", {}).get("totalResults", "?")
            text = (
                f"Found approximately {total_results} image results for '{query}'.\n"
                f"Showing {len(items)} results (thumbnails attached for review):\n\n"
                + "\n\n".join(text_parts)
                + "\n\nPlease review the attached thumbnails and pick the best image for the user."
            )

            return ToolOutput(text=text, media=media_list)

    except httpx.TimeoutException:
        brave_result = await _brave_search_fallback(query, num_results)
        if brave_result:
            return brave_result
        return "Error: Search request timed out"
    except Exception as e:
        logger.error(f"Google Search error: {e}", exc_info=True)
        brave_result = await _brave_search_fallback(query, num_results)
        if brave_result:
            return brave_result
        return f"Error: {e}"


GOOGLE_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query string",
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (max 10)",
            "default": 5,
        },
        "search_type": {
            "type": "string",
            "enum": ["web", "image"],
            "description": "Type of search: 'web' for regular results, 'image' for image results with thumbnails for review",
            "default": "web",
        },
    },
    "required": ["query"],
}
