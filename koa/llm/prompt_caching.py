"""Anthropic prompt caching — system_and_3 strategy.

Reduces input token costs by ~90% on multi-turn conversations by caching
the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across all turns)
  2-4. Last 3 non-system messages (rolling window)

First request pays 1.25x write cost; subsequent requests with the same
prefix pay only 0.1x read cost (within the TTL window).

Pure functions — no class state, no side effects.
"""

import copy
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _apply_cache_marker(msg: dict, marker: dict) -> None:
    """Add cache_control to a single message, handling all format variations.

    - String content → wrapped in [{"type": "text", "text": ..., "cache_control": ...}]
    - List content → marker added to the last content block
    - Tool messages → marker at message level
    - Empty/None content → marker at message level
    """
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        msg["cache_control"] = marker
        return

    if content is None or content == "":
        msg["cache_control"] = marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = marker


def apply_anthropic_cache_control(
    messages: List[Dict[str, Any]],
    ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints:
      - Breakpoint 1: system prompt (stable, cached across all turns)
      - Breakpoints 2-4: last 3 non-system messages (rolling window)

    Args:
        messages: The conversation messages (OpenAI format).
        ttl: Cache time-to-live. "5m" (default, 1.25x write) or "1h".

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
        Original list is never mutated.
    """
    messages = copy.deepcopy(messages)
    if not messages:
        return messages

    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl != "5m":
        marker["ttl"] = ttl

    breakpoints_used = 0

    # Breakpoint 1: system prompt
    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker)
        breakpoints_used += 1

    # Breakpoints 2-4: last N non-system messages
    remaining = 4 - breakpoints_used
    non_sys_indices = [
        i for i in range(len(messages)) if messages[i].get("role") != "system"
    ]
    for idx in non_sys_indices[-remaining:]:
        _apply_cache_marker(messages[idx], marker)

    return messages


def is_anthropic_model(provider: str, model: str) -> bool:
    """Check if the given provider/model combination supports Anthropic prompt caching."""
    if provider == "anthropic":
        return True
    model_lower = model.lower()
    return "claude" in model_lower
