"""Graceful error responses in Koi's persona style.

When the orchestrator hits an error, this module produces a natural,
casual reply instead of exposing raw errors or silence to the user.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)

FALLBACK_MESSAGES: list[str] = [
    "hmm my brain just glitched. try again?",
    "blanked out for a moment there. what were you saying?",
    "lost my train of thought lol. one more time?",
    "wait i totally spaced. say that again?",
    "ok that didn't work. wanna try once more?",
    "my head's not in it rn. hit me again?",
    "ugh i fumbled that one. go ahead and retry?",
]

_SYSTEM_PROMPT = (
    "you are koi, a chill personal assistant. respond like you're texting a friend. "
    "you just failed to answer something. write one short casual sentence acknowledging "
    "the hiccup without saying sorry, without mentioning being AI or having technical issues. "
    "all lowercase. keep it under 15 words."
)


def get_fallback_message() -> str:
    """Return a random preset fallback message."""
    return random.choice(FALLBACK_MESSAGES)


async def generate_graceful_error(
    error: BaseException,
    llm_client: Optional[Any] = None,
    timeout: float = 3.0,
) -> str:
    """Generate a persona-consistent error message.

    Tries to use the LLM for a unique reply; falls back to a preset
    message on failure, timeout, or when no client is available.
    """
    if llm_client is None:
        return get_fallback_message()

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "something went wrong, give me a casual one-liner"},
        ]
        response = await asyncio.wait_for(
            llm_client.chat_completion(messages=messages, max_tokens=60, temperature=0.9),
            timeout=timeout,
        )
        content = getattr(response, "content", None) or ""
        if content.strip():
            return content.strip()
        return get_fallback_message()
    except Exception:
        logger.debug("graceful_response: LLM fallback triggered", exc_info=True)
        return get_fallback_message()
