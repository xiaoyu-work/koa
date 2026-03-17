"""OneValet Email Event Handler — LLM-powered email importance evaluation."""

import json
import logging
from typing import Any, Dict, Optional, Set

import httpx

from ..llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_IMPORTANCE_SYSTEM_PROMPT = """\
You are an email classifier. Evaluate the email and respond with ONLY a JSON object.

TASK 1 — IMPORTANCE:
Rules for IMPORTANT emails (require immediate attention):
- OTP / verification codes
- Security alerts (login attempts, password resets, suspicious activity)
- Payment failures or billing issues
- Delivery problems (failed delivery, return to sender)
- Time-sensitive actions required (expiring offers that matter, deadlines)
- Personal urgent messages from real people

Rules for NOT IMPORTANT emails:
- Newsletters and digests
- Order confirmations and shipping updates (routine, no problems)
- Receipts for completed transactions
- Social media notifications
- Marketing and promotional emails
- Automated status updates that require no action

TASK 2 — SUBSCRIPTION DETECTION:
If this email is a receipt, invoice, billing confirmation, renewal notice, or cancellation
for a recurring subscription service, extract the subscription details.
Examples: Netflix, Spotify, iCloud, T-Mobile, Adobe, YouTube Premium, etc.
Do NOT include one-time purchases (e.g. buying a product on Amazon).

Respond with ONLY this JSON (no markdown, no extra text):
{"important": true/false, "reason": "brief reason", "summary": "one-line summary", "subscription": null}

If a subscription is detected, replace null with:
{"service_name": "Netflix", "category": "streaming", "amount": 15.99, "currency": "USD", "billing_cycle": "monthly", "status": "active"}

subscription fields:
- service_name: Name of the service
- category: One of "streaming", "cloud", "productivity", "saas", "developer", "telecom", "vpn", "fitness", "news", "gaming", "education", "finance", "home", "shopping", "other"
- amount: Charged amount as number, or null if not found
- currency: Currency code (default "USD")
- billing_cycle: One of "monthly", "yearly", "weekly", "one-time", or null
- status: "active" for receipts/renewals, "cancelled" for cancellation emails, "trial" for free trials
"""


class EmailEventHandler:
    """Handles email events by evaluating importance and detecting subscriptions via LLM.

    A single LLM call classifies importance AND detects subscriptions simultaneously.

    Args:
        llm_client: LLM client for evaluation
        callback_url: URL to POST important email notifications to
        database: Optional asyncpg pool for subscription storage
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        callback_url: str,
        database=None,
    ):
        self._llm_client = llm_client
        self._callback_url = callback_url
        self._processed_ids: Set[str] = set()
        self._database = database

    async def handle_email(self, tenant_id: str, data: Dict[str, Any]) -> None:
        """Process an incoming email event.

        Single LLM call evaluates importance and detects subscriptions.
        If important, POSTs a callback. If subscription detected, upserts to DB.
        """
        message_id = data.get("message_id", "")

        # Duplicate prevention
        if message_id and message_id in self._processed_ids:
            logger.debug(f"Skipping duplicate email: {message_id}")
            return
        if message_id:
            self._processed_ids.add(message_id)

        sender = data.get("sender", "")
        subject = data.get("subject", "")
        snippet = data.get("snippet", "")

        # Single LLM call for both importance and subscription detection
        evaluation = await self._evaluate_email(sender, subject, snippet)
        if evaluation is None:
            logger.warning(f"LLM evaluation failed for email {message_id}")
            return

        # Handle subscription if detected
        sub_data = evaluation.get("subscription")
        if sub_data and isinstance(sub_data, dict) and sub_data.get("service_name") and self._database:
            try:
                await self._upsert_subscription(tenant_id, sub_data, sender)
            except Exception as e:
                logger.warning(f"Subscription upsert failed: {e}")

        # Handle importance
        if not evaluation.get("important", False):
            logger.debug(f"Email not important: {subject} (reason: {evaluation.get('reason', 'N/A')})")
            return

        logger.info(f"Important email detected: {subject} — {evaluation.get('reason', '')}")
        await self._send_callback(
            tenant_id=tenant_id,
            summary=evaluation.get("summary", subject),
            sender=sender,
            subject=subject,
            message_id=message_id,
            reason=evaluation.get("reason", ""),
        )

    async def _upsert_subscription(self, tenant_id: str, sub: Dict[str, Any], sender: str) -> None:
        """Upsert detected subscription to database."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        # Validate billing_cycle
        valid_cycles = {"monthly", "yearly", "weekly", "one-time"}
        billing_cycle = sub.get("billing_cycle")
        if billing_cycle not in valid_cycles:
            billing_cycle = None

        # Validate status
        valid_statuses = {"active", "cancelled", "trial", "paused"}
        status = sub.get("status", "active")
        if status not in valid_statuses:
            status = "active"

        # Parse amount
        amount = sub.get("amount")
        if isinstance(amount, str):
            try:
                amount = float(amount.replace(",", ""))
            except (ValueError, AttributeError):
                amount = None

        await self._database.execute("""
            INSERT INTO subscriptions (
                tenant_id, service_name, category, amount, currency,
                billing_cycle, status, detected_from, source_email, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (tenant_id, service_name) DO UPDATE SET
                amount = COALESCE(EXCLUDED.amount, subscriptions.amount),
                currency = COALESCE(EXCLUDED.currency, subscriptions.currency),
                billing_cycle = COALESCE(EXCLUDED.billing_cycle, subscriptions.billing_cycle),
                status = EXCLUDED.status,
                source_email = COALESCE(EXCLUDED.source_email, subscriptions.source_email),
                is_active = TRUE,
                updated_at = $10
        """,
            tenant_id,
            sub["service_name"],
            sub.get("category", "other"),
            amount,
            sub.get("currency", "USD"),
            billing_cycle,
            status,
            "email",
            sender,
            now,
        )
        logger.info(f"Subscription detected: {sub['service_name']} for tenant {tenant_id}")

    async def _evaluate_email(
        self, sender: str, subject: str, snippet: str
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM to evaluate email importance and detect subscriptions.

        Returns:
            Dict with keys: important (bool), reason (str), summary (str), subscription (dict|null)
            None if the LLM call fails.
        """
        user_message = (
            f"Sender: {sender}\n"
            f"Subject: {subject}\n"
            f"Preview: {snippet}"
        )

        try:
            response = await self._llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": _IMPORTANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                config={"temperature": 0.0, "max_tokens": 512},
            )
            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Email importance evaluation failed: {e}")
            return None

    async def _send_callback(
        self,
        tenant_id: str,
        summary: str,
        sender: str,
        subject: str,
        message_id: str,
        reason: str,
    ) -> None:
        """POST important email notification to the callback URL."""
        payload = {
            "tenant_id": tenant_id,
            "message": summary,
            "priority": "urgent",
            "category": "email_alert",
            "metadata": {
                "subject": subject,
                "sender": sender,
                "message_id": message_id,
                "reason": reason,
            },
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._callback_url,
                    json=payload,
                    timeout=15.0,
                )
                resp.raise_for_status()
                logger.info(f"Email callback sent for message {message_id}")
        except Exception as e:
            logger.error(f"Email callback failed for {message_id}: {e}")
