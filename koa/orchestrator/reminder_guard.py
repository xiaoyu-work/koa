"""Reminder Guard — post-process hook that detects unfulfilled reminder commitments.

When the AI response says "I'll set a reminder" or "I'll remember to follow up"
but no cron tool was actually called during the turn, this hook appends a
self-correction note so the user doesn't receive a false promise.

Usage:
    orchestrator = Orchestrator(
        ...,
        post_process_hooks=[reminder_guard_hook],
    )
"""

import logging
import re
from typing import Any, Dict

from ..result import AgentResult

logger = logging.getLogger(__name__)

UNSCHEDULED_REMINDER_NOTE = (
    "Note: I did not schedule a reminder in this turn, "
    "so this will not trigger automatically."
)

# Patterns that indicate the AI committed to scheduling a reminder/alert
# but may not have actually called the cron tool.
REMINDER_COMMITMENT_PATTERNS = [
    re.compile(
        r"\b(?:i\s*[''\u2019]?ll|i will)\s+(?:make sure to\s+)?"
        r"(?:remember|remind|ping|follow up|follow-up|check back|circle back)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s*[''\u2019]?ll|i will)\s+(?:set|create|schedule)\s+(?:a\s+)?reminder\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breminder\s+(?:has been|is)\s+(?:set|created|scheduled)\b",
        re.IGNORECASE,
    ),
]

# Tool names that count as "reminder was actually created"
CRON_TOOL_NAMES = {"cron_add", "cron_update", "cron_run"}


def _has_unbacked_reminder_commitment(text: str) -> bool:
    """Check if the response text contains a reminder commitment."""
    if not text.strip():
        return False
    # Don't double-append
    if UNSCHEDULED_REMINDER_NOTE.lower() in text.lower():
        return False
    return any(p.search(text) for p in REMINDER_COMMITMENT_PATTERNS)


def _cron_tool_was_called(context: Dict[str, Any]) -> bool:
    """Check if any cron tool was invoked during this execution."""
    tool_calls = context.get("tool_calls", [])
    if not tool_calls:
        return False
    for tc in tool_calls:
        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if name in CRON_TOOL_NAMES:
            return True
    return False


async def reminder_guard_hook(
    result: AgentResult,
    context: Dict[str, Any],
) -> AgentResult:
    """Post-process hook: detect unfulfilled reminder commitments.

    If the AI's response claims it set/created a reminder but no cron tool
    was called, append a self-correction note to the response.
    """
    if not result.raw_message:
        return result

    if not _has_unbacked_reminder_commitment(result.raw_message):
        return result

    if _cron_tool_was_called(context):
        return result

    # AI promised a reminder but didn't actually create one
    logger.warning(
        "Reminder guard triggered: response claims reminder but no cron tool was called"
    )
    result.raw_message = f"{result.raw_message.rstrip()}\n\n{UNSCHEDULED_REMINDER_NOTE}"
    return result
