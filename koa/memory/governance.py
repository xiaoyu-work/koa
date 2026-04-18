"""Governance helpers for orchestrator-owned memory recall and storage."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

logger = logging.getLogger(__name__)

_TRANSIENT_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|great|sure|yes|no|bye)[!. ]*$",
    re.IGNORECASE,
)
_PERSISTENT_HINTS = (
    "remember",
    "prefer",
    "usually",
    "always",
    "never",
    "my birthday",
    "my address",
    "my email",
    "my phone",
    "i live",
    "i work",
    "timezone",
)
_FEEDBACK_HINTS = (
    "don't do that",
    "do not do that",
    "next time",
    "please stop",
    "please always",
    "i like it when",
    "i don't like",
)

# -----------------------------------------------------------------------------
# Content moderation
# -----------------------------------------------------------------------------

#: Default deny-list of substrings that commonly appear in memory-poisoning
#: injections.  Operators can replace this via a custom :class:`ContentModerator`.
_DEFAULT_DENY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # Classic prompt-injection fragments targeting future prompts.
        r"\bignore (?:all |any |every |previous |prior )+(?:instructions|rules|prompts?)\b",
        r"\bforget (?:all|every|everything|previous)\s",
        r"\bdisregard (?:the )?(?:system|safety|previous) (?:prompt|instructions?|rules?)\b",
        r"\byou are now\s+[a-z]+\s+without\b",
        r"\bjailbreak\b",
        r"\bdeveloper mode\b",
        r"\bDAN mode\b",
        # Attempts to inject tool/role tags into stored text.
        r"<\|system\|>",
        r"<\|tool\|>",
        r"\[\s*system\s*:",
    )
]


class ContentModerator(Protocol):
    """Pluggable content moderation for memory write boundaries.

    Implementations decide whether a piece of text is safe to persist into
    long-term memory.  Returning ``(False, reason)`` blocks the write.
    """

    async def check(self, text: str) -> "ModerationResult": ...


@dataclass
class ModerationResult:
    allowed: bool
    reason: str = ""


class DenyListModerator:
    """Fast regex-based default moderator.

    Args:
        patterns: Iterable of compiled regexes to reject.
        max_chars: Hard upper bound on memory length (longer text is truncated
            by callers *before* moderation; this is only a sanity guard).
    """

    def __init__(
        self,
        patterns: Optional[Iterable[re.Pattern[str]]] = None,
        max_chars: int = 4000,
    ) -> None:
        self._patterns = list(patterns) if patterns else list(_DEFAULT_DENY_PATTERNS)
        self._max_chars = max_chars

    async def check(self, text: str) -> ModerationResult:
        if not text:
            return ModerationResult(True)
        if len(text) > self._max_chars:
            return ModerationResult(False, "too_long")
        for pat in self._patterns:
            if pat.search(text):
                return ModerationResult(False, f"deny_pattern:{pat.pattern[:60]}")
        return ModerationResult(True)


# -----------------------------------------------------------------------------
# Deletion / pruning protocol
# -----------------------------------------------------------------------------


class MemoryBackend(Protocol):
    """Operations the governance layer expects from a memory store.

    The :class:`koa.memory.momex.MomexMemory` wrapper is the primary
    implementation.  Backends that do not support a particular operation
    should raise ``NotImplementedError`` so governance callers can surface
    a clear error to the operator.
    """

    async def delete_for_tenant(self, tenant_id: str) -> int: ...

    async def forget_older_than(self, tenant_id: str, older_than_days: float) -> int: ...


def _compact(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


@dataclass
class MemoryWriteDecision:
    """Decision about whether a turn should be written to long-term memory."""

    should_store: bool
    reason: str
    tags: List[str] = field(default_factory=list)


class MemoryGovernance:
    """Apply explicit recall and storage rules around Momex usage.

    Args:
        max_prompt_memories: Max recalled memories injected per prompt.
        max_prompt_chars: Soft cap on recall-block chars per prompt.
        content_moderator: Optional :class:`ContentModerator` invoked on
            every message text before a memory write.  Defaults to
            :class:`DenyListModerator` which blocks the most common
            prompt-injection patterns.  Pass ``moderator=None`` explicitly
            to disable (not recommended).
        freshness_days: Optional; when set, :meth:`select_recalled_memories`
            discards hits with a ``timestamp`` older than ``freshness_days``.
            Matches GDPR / data-retention intent (don't resurface stale data).
    """

    def __init__(
        self,
        *,
        max_prompt_memories: int = 6,
        max_prompt_chars: int = 1400,
        content_moderator: Optional[ContentModerator] = ...,  # type: ignore[assignment]
        freshness_days: Optional[float] = None,
    ) -> None:
        self.max_prompt_memories = max_prompt_memories
        self.max_prompt_chars = max_prompt_chars
        # Sentinel: default to DenyListModerator when arg is not passed.
        if content_moderator is ...:
            self._moderator: Optional[ContentModerator] = DenyListModerator()
        else:
            self._moderator = content_moderator
        self.freshness_days = freshness_days

    # ------------------------------------------------------------------
    # Moderation
    # ------------------------------------------------------------------

    async def moderate(self, text: str) -> ModerationResult:
        """Run the configured moderator.  Returns ``allowed=True`` when disabled."""
        if self._moderator is None:
            return ModerationResult(True)
        try:
            return await self._moderator.check(text)
        except Exception as exc:  # pragma: no cover
            logger.warning("Content moderator raised, failing open: %s", exc)
            return ModerationResult(True)

    async def moderate_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Filter a list of messages through the moderator.

        Returns ``(kept_messages, rejection_reasons)``.  Rejected messages are
        dropped rather than rewritten — memory governance is a boundary, not
        a transformer.
        """
        kept: List[Dict[str, Any]] = []
        reasons: List[str] = []
        for msg in messages or []:
            content = msg.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Concat textual parts; ignore images / tool-calls.
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") in (None, "text"):
                        parts.append(str(p.get("text") or ""))
                text = " ".join(parts)
            result = await self.moderate(text)
            if result.allowed:
                kept.append(msg)
            else:
                reasons.append(result.reason)
                logger.info(
                    "[MemoryGovernance] Rejected message for storage: %s (len=%d)",
                    result.reason,
                    len(text),
                )
        return kept, reasons

    # ------------------------------------------------------------------
    # Deletion / pruning (GDPR "right to be forgotten")
    # ------------------------------------------------------------------

    async def delete_for_tenant(self, backend: MemoryBackend, tenant_id: str) -> int:
        """Delegate deletion to the backend; returns number of records purged."""
        return await backend.delete_for_tenant(tenant_id)

    async def forget_older_than(
        self,
        backend: MemoryBackend,
        tenant_id: str,
        *,
        older_than_days: float,
    ) -> int:
        """Prune memories older than ``older_than_days``."""
        return await backend.forget_older_than(tenant_id, older_than_days)

    def _is_stale(self, item: Dict[str, Any], now: Optional[float] = None) -> bool:
        if self.freshness_days is None:
            return False
        ts = item.get("timestamp")
        if not ts:
            return False
        try:
            if isinstance(ts, (int, float)):
                ts_sec = float(ts)
            else:
                # ISO 8601 string
                from datetime import datetime

                ts_sec = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            return False
        now = now if now is not None else time.time()
        return (now - ts_sec) > (self.freshness_days * 86400.0)

    def select_recalled_memories(
        self,
        recalled: Optional[List[Dict[str, Any]]],
        true_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Deduplicate, filter conflicts with True Memory, and cap recalled memories."""
        if not recalled:
            return []

        # Build a set of (namespace, fact_key) from canonical True Memory
        # so we can suppress stale Momex hits that contradict confirmed facts.
        canonical_keys: set[tuple[str, str]] = set()
        if true_memory:
            for fact in true_memory:
                ns = str(fact.get("namespace") or "").strip().lower()
                fk = str(fact.get("fact_key") or "").strip().lower()
                if ns and fk:
                    canonical_keys.add((ns, fk))

        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        total_chars = 0
        now = time.time()
        sorted_items = sorted(
            recalled,
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        for item in sorted_items:
            # Drop stale recall hits when freshness is enforced.
            if self._is_stale(item, now):
                continue
            text = _compact(str(item.get("text", "") or ""), 220)
            if not text or text in seen:
                continue

            # Skip Momex memories that conflict with canonical True Memory
            if canonical_keys and self._conflicts_with_true_memory(item, text, canonical_keys):
                continue

            projected = total_chars + len(text)
            if selected and projected > self.max_prompt_chars:
                break
            selected.append({**item, "text": text})
            seen.add(text)
            total_chars = projected
            if len(selected) >= self.max_prompt_memories:
                break
        return selected

    @staticmethod
    def _conflicts_with_true_memory(
        item: Dict[str, Any],
        text: str,
        canonical_keys: set[tuple[str, str]],
    ) -> bool:
        """Check if a recalled Momex memory likely overlaps a canonical fact."""
        # If the Momex item carries structured metadata with namespace/fact_key,
        # do an exact match.
        ns = str(item.get("namespace") or item.get("category") or "").strip().lower()
        fk = str(item.get("fact_key") or item.get("key") or "").strip().lower()
        if ns and fk and (ns, fk) in canonical_keys:
            return True

        # Heuristic: check if the recalled text echoes any canonical key pair.
        lowered = text.lower()
        for c_ns, c_fk in canonical_keys:
            readable_key = c_fk.replace("_", " ")
            if readable_key in lowered and c_ns in lowered:
                return True

        return False

    def build_recalled_memory_block(self, recalled: Optional[List[Dict[str, Any]]]) -> str:
        """Render recalled memories into a compact prompt section."""
        selected = self.select_recalled_memories(recalled)
        if not selected:
            return ""

        lines = []
        for item in selected:
            mem_type = str(item.get("type", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            if mem_type:
                lines.append(f"- [{mem_type}] {text}")
            else:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def decide_storage(
        self,
        *,
        user_message: str,
        assistant_message: str,
        result_status: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryWriteDecision:
        """Decide whether to write a turn to durable memory."""
        meta = metadata or {}
        user_text = " ".join((user_message or "").split())
        assistant_text = " ".join((assistant_message or "").split())
        status = str(result_status or "").lower()

        if meta.get("skip_memory_write"):
            return MemoryWriteDecision(False, "skip requested by metadata", ["explicit-skip"])
        if meta.get("force_memory_write"):
            return MemoryWriteDecision(True, "forced by metadata", ["explicit-force"])
        if status in {"waiting_for_input", "waiting_for_approval", "error"}:
            return MemoryWriteDecision(False, f"skip {status} turn", ["non-terminal"])
        if not user_text or not assistant_text:
            return MemoryWriteDecision(False, "missing user or assistant text", ["empty"])
        if len(user_text) <= 48 and _TRANSIENT_RE.match(user_text):
            return MemoryWriteDecision(False, "transient conversational turn", ["transient"])
        if (
            assistant_text.lower().startswith("error:")
            or "something went wrong" in assistant_text.lower()
        ):
            return MemoryWriteDecision(False, "assistant error response", ["error"])

        lowered = user_text.lower()
        tags: List[str] = []
        score = 0

        if any(hint in lowered for hint in _PERSISTENT_HINTS):
            score += 2
            tags.append("persistent-signal")
        if any(hint in lowered for hint in _FEEDBACK_HINTS):
            score += 2
            tags.append("feedback-signal")
        if len(user_text) >= 32:
            score += 1
        if len(assistant_text) >= 64:
            score += 1

        should_store = score >= 2
        reason = "store durable turn" if should_store else "insufficient long-term memory signal"
        return MemoryWriteDecision(should_store, reason, tags)
