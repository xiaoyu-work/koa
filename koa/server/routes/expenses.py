"""Internal expense API routes (service-to-service).

Provides endpoints for querying and managing expenses, category summaries,
budgets with spending status, and receipts.
"""

import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from ..app import require_app, verify_service_key
from koa.builtin_agents.expense.repository import ExpenseRepository
from koa.builtin_agents.expense.budget_repository import BudgetRepository
from koa.builtin_agents.expense.receipt_repository import ReceiptRepository

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Helpers
# =============================================================================


async def _get_repos() -> tuple[ExpenseRepository, BudgetRepository, ReceiptRepository]:
    """Create expense, budget, and receipt repositories from the app database."""
    app = require_app()
    await app._ensure_initialized()
    db = app._database
    return ExpenseRepository(db), BudgetRepository(db), ReceiptRepository(db)


def _parse_period(period: str) -> tuple[date, date]:
    """Parse a period string into (start_date, end_date) inclusive.

    Supports:
        - "today"       -> today only
        - "this_week"   -> Monday through Sunday of the current week
        - "this_month"  -> first through last day of the current month
        - "last_month"  -> first through last day of last month
        - "YYYY-MM"     -> first through last day of that month
    """
    today = date.today()
    lower = period.strip().lower()

    if lower == "today":
        return today, today

    if lower == "this_week":
        start = today - timedelta(days=today.weekday())  # Monday
        end = start + timedelta(days=6)  # Sunday
        return start, end

    if lower in ("this_month", ""):
        start = today.replace(day=1)
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return start, end

    if lower == "last_month":
        first_this_month = today.replace(day=1)
        end = first_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    # Try YYYY-MM format
    try:
        parts = lower.split("-")
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            start = date(year, month, 1)
            if month == 12:
                end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(year, month + 1, 1) - timedelta(days=1)
            return start, end
    except (ValueError, IndexError):
        pass

    # Default to this month
    start = today.replace(day=1)
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return start, end


def _serialize_row(row: dict) -> dict:
    """Convert date/Decimal fields to JSON-safe types."""
    out = {}
    for k, v in row.items():
        if isinstance(v, date):
            out[k] = v.isoformat()
        elif hasattr(v, "as_tuple"):  # Decimal
            out[k] = float(v)
        else:
            out[k] = v
    return out


# =============================================================================
# Routes
# =============================================================================


@router.get("/api/internal/expenses")
async def internal_list_expenses(
    request: Request,
    tenant_id: str,
    period: Optional[str] = None,
    category: Optional[str] = None,
    merchant: Optional[str] = None,
    limit: int = 50,
):
    """Query expenses with optional filters. Internal use only."""
    verify_service_key(request)
    expense_repo, _, _ = await _get_repos()

    start_date = None
    end_date = None
    if period:
        start_date, end_date = _parse_period(period)

    rows = await expense_repo.query(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        merchant=merchant,
        limit=limit,
    )

    expenses = [_serialize_row(r) for r in rows]
    total = sum(float(r.get("amount", 0)) for r in rows)

    return {"expenses": expenses, "total": round(total, 2)}


@router.get("/api/internal/expenses/summary")
async def internal_expense_summary(
    request: Request,
    tenant_id: str,
    period: str = "this_month",
):
    """Category breakdown of expenses for a period. Internal use only."""
    verify_service_key(request)
    expense_repo, _, _ = await _get_repos()

    start_date, end_date = _parse_period(period)
    rows = await expense_repo.summary_by_category(tenant_id, start_date, end_date)

    categories = []
    grand_total = 0.0
    for row in rows:
        cat_total = float(row.get("total_amount", 0))
        categories.append({
            "category": row["category"],
            "total": round(cat_total, 2),
            "count": row["count"],
        })
        grand_total += cat_total

    return {
        "categories": categories,
        "grand_total": round(grand_total, 2),
        "period": period,
    }


@router.get("/api/internal/budgets")
async def internal_list_budgets(
    request: Request,
    tenant_id: str,
):
    """Get all budgets with current month spending status. Internal use only."""
    verify_service_key(request)
    expense_repo, budget_repo, _ = await _get_repos()

    budgets = await budget_repo.get_all_budgets(tenant_id)

    today = date.today()
    result = []
    for b in budgets:
        category = b["category"]
        spent = await expense_repo.monthly_total(
            tenant_id,
            year=today.year,
            month=today.month,
            category=category if category != "_total" else None,
        )
        result.append({
            "category": category,
            "monthly_limit": float(b["monthly_limit"]),
            "spent": round(spent, 2),
            "currency": b.get("currency", "USD"),
        })

    return {"budgets": result}


@router.get("/api/internal/receipts")
async def internal_list_receipts(
    request: Request,
    tenant_id: str,
    limit: int = 20,
):
    """List receipts for a tenant. Internal use only."""
    verify_service_key(request)
    _, _, receipt_repo = await _get_repos()

    rows = await receipt_repo._fetch_many(
        where="tenant_id = $1",
        args=(tenant_id,),
        order_by="created_at DESC",
        limit=limit,
    )

    return {"receipts": [_serialize_row(r) for r in rows]}


@router.post("/api/internal/expenses")
async def internal_create_expense(
    request: Request,
    tenant_id: str,
):
    """Create a new expense. Internal use only."""
    verify_service_key(request)
    expense_repo, _, _ = await _get_repos()

    body = await request.json()

    # Parse date if provided
    expense_date = None
    if body.get("date"):
        from datetime import datetime as dt
        try:
            expense_date = dt.strptime(body["date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    row = await expense_repo.add(
        tenant_id=tenant_id,
        amount=float(body["amount"]),
        category=body.get("category", "other"),
        description=body.get("description", ""),
        merchant=body.get("merchant", ""),
        date=expense_date,
        currency=body.get("currency", "USD"),
        receipt_id=body.get("receipt_id"),
    )
    return {"expense": _serialize_row(row)}


@router.patch("/api/internal/expenses/{expense_id}")
async def internal_update_expense(
    request: Request,
    expense_id: str,
    tenant_id: str,
):
    """Update specific fields of an expense. Internal use only."""
    verify_service_key(request)
    expense_repo, _, _ = await _get_repos()

    body = await request.json()

    # Build update dict from allowed fields
    allowed = {"amount", "category", "description", "merchant", "date", "currency", "receipt_id"}
    updates = {}
    for key in allowed:
        if key in body:
            if key == "amount":
                updates[key] = float(body[key])
            elif key == "date":
                from datetime import datetime as dt
                try:
                    updates[key] = dt.strptime(body[key], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue
            else:
                updates[key] = body[key]

    if not updates:
        raise HTTPException(400, "No valid fields to update")

    row = await expense_repo.update(
        tenant_id=tenant_id,
        expense_id=expense_id,
        data=updates,
    )

    if row is None:
        raise HTTPException(404, "Expense not found or not owned by this tenant")

    return {"expense": _serialize_row(row)}


@router.delete("/api/internal/expenses/{expense_id}")
async def internal_delete_expense(
    request: Request,
    expense_id: str,
    tenant_id: str,
):
    """Delete an expense. Internal use only."""
    verify_service_key(request)
    expense_repo, _, _ = await _get_repos()

    deleted = await expense_repo.delete(tenant_id=tenant_id, expense_id=expense_id)

    if not deleted:
        raise HTTPException(404, "Expense not found or not owned by this tenant")

    return {"deleted": True}
