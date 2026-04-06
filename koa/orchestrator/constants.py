"""Orchestrator-level constants.

Centralizes magic numbers used across the orchestrator modules so they
can be reviewed and tuned in one place.  Config-level defaults that live
in ``ReactLoopConfig`` are intentionally *not* duplicated here; this
module covers values embedded in logic code.
"""

# ── Token estimation ─────────────────────────────────────────────────
TOKENS_PER_MESSAGE_OVERHEAD = 4
"""Approximate token overhead per chat message (role + separators)."""

TOOL_CALL_STRUCTURE_OVERHEAD_TOKENS = 20
"""Approximate token overhead per tool-call entry (name + JSON wrapper)."""

JSON_CHARS_PER_TOKEN = 3
"""Chars-per-token ratio for JSON / code content."""

TEXT_CHARS_PER_TOKEN = 4
"""Chars-per-token ratio for natural-language content."""

JSON_DETECTION_SAMPLE_SIZE = 500
"""Number of leading characters sampled for the JSON-detection heuristic."""

JSON_DETECTION_RATIO = 0.15
"""If the fraction of special chars ({, [, ", :, ,) in the sample exceeds
this threshold the text is treated as JSON/code for estimation purposes."""

IMAGE_TOKEN_ESTIMATE = 170
"""Default per-image token cost (high-res estimate).

Low-res images are ~85 tokens, but resolution is usually unknown so we
default to the conservative high-res figure.
"""

# ── Tool-result size caps ────────────────────────────────────────────
TOOL_RESULT_HARD_CAP_CHARS = 400_000
"""Absolute character limit applied to any single tool result before
context-aware truncation kicks in."""

# ── Agent-tool result truncation (context isolation) ─────────────────
AGENT_RESULT_TRUNCATE_THRESHOLD = 2000
"""If an agent-tool result exceeds this many characters it will be
truncated to keep the orchestrator context lean."""

AGENT_RESULT_TRUNCATE_TO = 1500
"""Target length (in characters) when truncating agent-tool results."""
