"""Structured LLM error classification.

Replaces fragile string-matching with isinstance checks against
litellm's typed exception hierarchy.  Falls back to heuristic
classification for non-litellm errors.
"""

from enum import Enum
from typing import Optional


class LLMErrorKind(Enum):
    """Exhaustive classification of LLM call failures."""

    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    AUTH = "auth"
    TIMEOUT = "timeout"
    BAD_REQUEST = "bad_request"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


# Maps LLMErrorKind to user-facing error codes that the frontend
# can resolve via its errorMessages mapping.
_ERROR_KIND_TO_CODE = {
    LLMErrorKind.BAD_REQUEST.value: "agent_failed",
    LLMErrorKind.RATE_LIMIT.value: "service_unavailable",
    LLMErrorKind.AUTH.value: "config_error",
    LLMErrorKind.TIMEOUT.value: "service_unavailable",
    LLMErrorKind.SERVICE_UNAVAILABLE.value: "service_unavailable",
    LLMErrorKind.CONTEXT_OVERFLOW.value: "agent_failed",
    LLMErrorKind.TRANSIENT.value: "service_unavailable",
    LLMErrorKind.UNKNOWN.value: "internal_error",
}


def error_code_for_kind(kind: "LLMErrorKind") -> str:
    """Return a frontend-friendly error code for the given LLMErrorKind."""
    return _ERROR_KIND_TO_CODE.get(kind.value, "internal_error")


# Keep a single flag so the import check runs once.
_LITELLM_AVAILABLE: Optional[bool] = None


def _ensure_litellm_imports():
    global _LITELLM_AVAILABLE
    if _LITELLM_AVAILABLE is not None:
        return
    try:
        import litellm.exceptions  # noqa: F401

        _LITELLM_AVAILABLE = True
    except ImportError:
        _LITELLM_AVAILABLE = False


def classify_llm_error(exc: Exception) -> LLMErrorKind:
    """Classify an LLM call exception into a known error kind.

    Prefers litellm typed exceptions when available, falls back to
    heuristic name / message inspection for unknown exception types.
    """
    _ensure_litellm_imports()

    if _LITELLM_AVAILABLE:
        from litellm.exceptions import (
            APIConnectionError,
            AuthenticationError,
            BadGatewayError,
            BadRequestError,
            ContextWindowExceededError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        )

        if isinstance(exc, RateLimitError):
            return LLMErrorKind.RATE_LIMIT
        if isinstance(exc, ContextWindowExceededError):
            return LLMErrorKind.CONTEXT_OVERFLOW
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return LLMErrorKind.AUTH
        if isinstance(exc, Timeout):
            return LLMErrorKind.TIMEOUT
        if isinstance(exc, BadRequestError):
            return LLMErrorKind.BAD_REQUEST
        if isinstance(exc, (ServiceUnavailableError, BadGatewayError, InternalServerError)):
            return LLMErrorKind.SERVICE_UNAVAILABLE
        if isinstance(exc, APIConnectionError):
            return LLMErrorKind.TRANSIENT

    # Heuristic fallback for non-litellm exceptions
    return _heuristic_classify(exc)


def _heuristic_classify(exc: Exception) -> LLMErrorKind:
    """Fallback classification using exception name and message."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "ratelimit" in name or "rate_limit" in name or "429" in msg:
        return LLMErrorKind.RATE_LIMIT
    if "contextwindow" in name or "context_length" in msg or "maximum context length" in msg:
        return LLMErrorKind.CONTEXT_OVERFLOW
    if "auth" in name or "permission" in name or "401" in msg or "403" in msg:
        return LLMErrorKind.AUTH
    if "timeout" in name:
        return LLMErrorKind.TIMEOUT
    if "badrequest" in name or "invalid" in name:
        return LLMErrorKind.BAD_REQUEST
    if "serviceunavailable" in name or "502" in msg or "503" in msg:
        return LLMErrorKind.SERVICE_UNAVAILABLE

    return LLMErrorKind.UNKNOWN
