"""Internal subscription API routes (service-to-service).

Mirrors the shipment API pattern — CRUD operations on the subscriptions table,
called by koi-backend proxy routes.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from ..app import require_app, verify_service_key

logger = logging.getLogger(__name__)
router = APIRouter()


class UpsertSubscriptionRequest(BaseModel):
    tenant_id: str
    service_name: str
    category: str = "other"
    amount: Optional[float] = None
    currency: str = "USD"
    billing_cycle: Optional[str] = None
    next_billing_date: Optional[str] = None
    last_charged_date: Optional[str] = None
    status: str = "active"
    detected_from: str = "email"
    source_email: Optional[str] = None


@router.get("/api/internal/subscriptions/list", dependencies=[Depends(verify_service_key)])
async def internal_list_subscriptions(
    tenant_id: str,
    active_only: bool = True,
):
    """List all subscriptions for a tenant."""
    app = require_app()
    db = app.database
    if not db:
        return []

    query = "SELECT * FROM subscriptions WHERE tenant_id = $1"
    if active_only:
        query += " AND is_active = TRUE"
    query += " ORDER BY last_charged_date DESC NULLS LAST, created_at DESC"

    rows = await db.fetch(query, tenant_id)
    return [dict(r) for r in rows]


@router.put("/api/internal/subscriptions", dependencies=[Depends(verify_service_key)])
async def internal_upsert_subscription(req: UpsertSubscriptionRequest):
    """Upsert a subscription (insert or update on conflict)."""
    app = require_app()
    db = app.database
    if not db:
        return {"status": "error", "message": "Database not available"}

    now = datetime.now(timezone.utc)

    await db.execute("""
        INSERT INTO subscriptions (
            tenant_id, service_name, category, amount, currency,
            billing_cycle, next_billing_date, last_charged_date,
            status, detected_from, source_email, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (tenant_id, service_name) DO UPDATE SET
            amount = COALESCE(EXCLUDED.amount, subscriptions.amount),
            currency = COALESCE(EXCLUDED.currency, subscriptions.currency),
            billing_cycle = COALESCE(EXCLUDED.billing_cycle, subscriptions.billing_cycle),
            next_billing_date = COALESCE(EXCLUDED.next_billing_date, subscriptions.next_billing_date),
            last_charged_date = COALESCE(EXCLUDED.last_charged_date, subscriptions.last_charged_date),
            status = EXCLUDED.status,
            source_email = COALESCE(EXCLUDED.source_email, subscriptions.source_email),
            is_active = TRUE,
            updated_at = $12
    """,
        req.tenant_id, req.service_name, req.category,
        req.amount, req.currency, req.billing_cycle,
        req.next_billing_date, req.last_charged_date,
        req.status, req.detected_from, req.source_email, now,
    )

    return {"status": "ok", "service_name": req.service_name}


@router.delete("/api/internal/subscriptions", dependencies=[Depends(verify_service_key)])
async def internal_delete_subscription(
    tenant_id: str,
    service_name: str,
):
    """Soft-delete a subscription."""
    app = require_app()
    db = app.database
    if not db:
        return {"status": "error", "message": "Database not available"}

    await db.execute(
        "UPDATE subscriptions SET is_active = FALSE, updated_at = NOW() "
        "WHERE tenant_id = $1 AND service_name = $2",
        tenant_id, service_name,
    )
    return {"status": "ok"}
