"""Subscription detector — rule-based filter + LLM extraction.

Two-stage detection:
1. Rule-based: match sender domain against KNOWN_SERVICES, or check
   subject/snippet for SUBSCRIPTION_KEYWORDS.
2. LLM extraction: only for matched emails, extract amount, cycle, etc.

This keeps LLM costs low — 90%+ of emails skip LLM entirely.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from .known_services import KNOWN_SERVICES, SUBSCRIPTION_KEYWORDS

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract subscription details from this email. Respond with ONLY a JSON object.

Email from: {sender}
Subject: {subject}
Preview: {snippet}

Extract:
- service_name: Name of the service (e.g. "Netflix", "Spotify")
- amount: Numeric amount charged (e.g. 15.99). null if not found.
- currency: Currency code (e.g. "USD", "EUR"). Default "USD".
- billing_cycle: One of "monthly", "yearly", "weekly", "one-time", or null.
- next_billing_date: Next charge date in YYYY-MM-DD format, or null.
- last_charged_date: Date of this charge in YYYY-MM-DD format, or null.
- status: One of "active", "cancelled", "trial", "paused". Default "active".
  Use "cancelled" if email confirms cancellation.
  Use "trial" if email mentions free trial.

Respond with ONLY this JSON (no markdown, no extra text):
{{"service_name": "...", "amount": ..., "currency": "...", "billing_cycle": "...", "next_billing_date": "...", "last_charged_date": "...", "status": "..."}}
"""


@dataclass
class SubscriptionInfo:
    """Detected subscription data."""
    service_name: str
    category: str = "other"
    amount: Optional[float] = None
    currency: str = "USD"
    billing_cycle: Optional[str] = None
    next_billing_date: Optional[str] = None
    last_charged_date: Optional[str] = None
    status: str = "active"
    source_email: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "category": self.category,
            "amount": self.amount,
            "currency": self.currency,
            "billing_cycle": self.billing_cycle,
            "next_billing_date": self.next_billing_date,
            "last_charged_date": self.last_charged_date,
            "status": self.status,
            "source_email": self.source_email,
        }


def _extract_domain(email_address: str) -> str:
    """Extract root domain from email address. 'noreply@mail.netflix.com' → 'netflix.com'"""
    match = re.search(r"@([\w.-]+)", email_address)
    if not match:
        return ""
    parts = match.group(1).lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return match.group(1).lower()


def _has_subscription_keywords(text: str) -> bool:
    """Check if text contains any subscription-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in SUBSCRIPTION_KEYWORDS)


class SubscriptionDetector:
    """Detect subscriptions from email metadata.

    Args:
        llm_client: LLM client for extracting structured data from matched emails.
            If None, only rule-based detection (service_name + category) is used.
    """

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    async def check_email(
        self, sender: str, subject: str, snippet: str
    ) -> Optional[SubscriptionInfo]:
        """Check if an email indicates a subscription.

        Returns SubscriptionInfo if detected, None otherwise.
        """
        domain = _extract_domain(sender)
        known = KNOWN_SERVICES.get(domain)

        if not known and not _has_subscription_keywords(f"{subject} {snippet}"):
            return None

        service_name = known["name"] if known else ""
        category = known["category"] if known else "other"

        # LLM extraction for amount, cycle, dates
        extracted = await self._extract_details(sender, subject, snippet)

        if extracted:
            if not service_name:
                service_name = extracted.get("service_name", "")
            if not service_name:
                return None
            return SubscriptionInfo(
                service_name=service_name,
                category=category,
                amount=extracted.get("amount"),
                currency=extracted.get("currency", "USD"),
                billing_cycle=extracted.get("billing_cycle"),
                next_billing_date=extracted.get("next_billing_date"),
                last_charged_date=extracted.get("last_charged_date"),
                status=extracted.get("status", "active"),
                source_email=sender,
            )

        # No LLM or LLM failed — return basic info from domain match
        if service_name:
            return SubscriptionInfo(
                service_name=service_name,
                category=category,
                source_email=sender,
            )

        return None

    async def _extract_details(
        self, sender: str, subject: str, snippet: str
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to extract subscription details from email content."""
        if not self._llm_client:
            return None

        prompt = _EXTRACTION_PROMPT.format(
            sender=sender, subject=subject, snippet=snippet,
        )

        try:
            response = await self._llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract subscription data from emails. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (response.content or "").strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)

            # Validate and clean
            if isinstance(data.get("amount"), str):
                try:
                    data["amount"] = float(data["amount"].replace(",", ""))
                except (ValueError, AttributeError):
                    data["amount"] = None

            valid_cycles = {"monthly", "yearly", "weekly", "one-time"}
            if data.get("billing_cycle") not in valid_cycles:
                data["billing_cycle"] = None

            valid_statuses = {"active", "cancelled", "trial", "paused"}
            if data.get("status") not in valid_statuses:
                data["status"] = "active"

            return data

        except Exception as e:
            logger.warning(f"Subscription LLM extraction failed: {e}")
            return None
