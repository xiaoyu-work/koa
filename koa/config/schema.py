"""Config schema validation.

Validates config.yaml structure and values at load time, reporting all
errors at once rather than failing on the first one.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = {"openai", "anthropic", "azure", "dashscope", "gemini", "ollama"}


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {'; '.join(errors)}")


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate a config dict. Returns a list of error messages (empty = valid)."""
    errors: List[str] = []

    # LLM section
    llm = cfg.get("llm")
    if not llm or not isinstance(llm, dict):
        errors.append("'llm' section is required")
    else:
        if not llm.get("provider"):
            errors.append("'llm.provider' is required")
        elif llm["provider"] not in _VALID_PROVIDERS:
            errors.append(
                f"'llm.provider' must be one of {sorted(_VALID_PROVIDERS)}, "
                f"got '{llm['provider']}'"
            )
        if not llm.get("model"):
            errors.append("'llm.model' is required")

    # Database
    db = cfg.get("database")
    if not db:
        errors.append("'database' connection URL is required")
    elif isinstance(db, str) and not db.startswith(("postgresql://", "postgres://")):
        errors.append(
            f"'database' must be a PostgreSQL connection URL "
            f"(starts with postgresql://), got '{db[:30]}...'"
        )

    # Model routing (optional)
    routing = cfg.get("model_routing")
    if routing and isinstance(routing, dict) and routing.get("enabled"):
        rules = routing.get("rules", [])
        if not rules:
            errors.append("'model_routing.rules' is required when routing is enabled")
        else:
            sorted_rules = sorted(rules, key=lambda r: r.get("score_range", [0])[0])
            for i in range(len(sorted_rules) - 1):
                curr = sorted_rules[i].get("score_range", [0, 0])
                nxt = sorted_rules[i + 1].get("score_range", [0, 0])
                if len(curr) == 2 and len(nxt) == 2 and curr[1] >= nxt[0]:
                    errors.append(
                        f"Model routing rules overlap: "
                        f"{curr} and {nxt}"
                    )

    if errors:
        for e in errors:
            logger.error(f"[Config] {e}")

    return errors