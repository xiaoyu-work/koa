"""Subscription tools — query and manage user subscriptions from the DB."""

import json
import logging
from typing import Annotated, Any, Dict, List

from onevalet.models import AgentToolContext, ToolOutput
from onevalet.tool_decorator import tool

logger = logging.getLogger(__name__)


async def _get_db(context: AgentToolContext):
    """Get database from context hints."""
    return context.metadata.get("db") if context.metadata else None


def _format_amount(amount, currency="USD") -> str:
    if amount is None:
        return "N/A"
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "CNY": "¥", "JPY": "¥"}
    sym = symbols.get(currency, f"{currency} ")
    return f"{sym}{amount:,.2f}"


def _format_cycle(cycle) -> str:
    return cycle.replace("_", " ").title() if cycle else "Unknown"


@tool
async def query_subscriptions(
    status: Annotated[str, "Filter by status: 'active', 'cancelled', 'trial', 'all'. Default 'active'."] = "active",
    category: Annotated[str, "Filter by category (e.g. 'streaming', 'telecom'). Leave empty for all."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Query user's subscriptions. Shows service name, price, billing cycle, and category."""
    db = await _get_db(context)
    if not db:
        return "Subscription tracking is not available. Database not configured."

    try:
        query = "SELECT * FROM subscriptions WHERE tenant_id = $1"
        params: List[Any] = [context.tenant_id]

        if status and status != "all":
            query += " AND status = $2"
            params.append(status)
            if status == "active":
                query += " AND is_active = TRUE"

        if category:
            query += f" AND category = ${len(params) + 1}"
            params.append(category.strip().lower())

        query += " ORDER BY amount DESC NULLS LAST, service_name ASC"

        rows = await db.fetch(query, *params)
    except Exception as e:
        logger.error(f"Failed to query subscriptions: {e}", exc_info=True)
        return "Sorry, I couldn't retrieve your subscriptions. Please try again."

    if not rows:
        filters = []
        if status and status != "all":
            filters.append(f"status={status}")
        if category:
            filters.append(f"category={category}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        return f"No subscriptions found{filter_str}."

    # Build text summary
    lines = [f"Found {len(rows)} subscription(s):\n"]
    monthly_total = 0.0

    for row in rows:
        name = row["service_name"]
        amount = row.get("amount")
        currency = row.get("currency", "USD")
        cycle = row.get("billing_cycle")
        cat = row.get("category", "other")
        sub_status = row.get("status", "active")

        amount_str = _format_amount(amount, currency) if amount else "N/A"
        cycle_str = _format_cycle(cycle)

        line = f"- {name}: {amount_str}/{cycle_str} ({cat})"
        if sub_status != "active":
            line += f" [{sub_status}]"
        lines.append(line)

        # Estimate monthly cost
        if amount and sub_status == "active":
            if cycle == "monthly":
                monthly_total += amount
            elif cycle == "yearly":
                monthly_total += amount / 12
            elif cycle == "weekly":
                monthly_total += amount * 4.33

    if monthly_total > 0:
        lines.append(f"\nEstimated monthly total: {_format_amount(monthly_total, rows[0].get('currency', 'USD'))}")

    text_result = "\n".join(lines)

    # Build inline cards
    cards = []
    for row in rows:
        amount = row.get("amount")
        currency = row.get("currency", "USD")
        cycle = row.get("billing_cycle")
        card = {
            "card_type": "subscription",
            "service_name": row["service_name"],
            "category": row.get("category", "other"),
            "amount": f"{_format_amount(amount, currency)}/{_format_cycle(cycle)}" if amount else None,
            "status": row.get("status", "active"),
            "billing_cycle": cycle,
        }
        cards.append(card)

    media = []
    if cards:
        media.append({
            "type": "inline_cards",
            "data": json.dumps(cards),
            "media_type": "application/json",
            "metadata": {"for_storage": False},
        })

    return ToolOutput(text=text_result, media=media)
