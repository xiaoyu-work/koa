"""Plan state persistence for the Orchestrator.

Provides database-backed storage for pending plans so they survive
process restarts.  Falls back to in-memory dict when no database is
configured.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PlanStore:
    """Persistent plan storage backed by PostgreSQL.

    Provides the same dict-like interface as the in-memory
    ``_tenant_plans`` but persists state to the ``agent_plans`` table.
    """

    def __init__(self, database: Optional[Any] = None) -> None:
        self._db = database
        # In-memory fallback / cache
        self._cache: Dict[str, Any] = {}

    async def get(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get a pending plan for the tenant."""
        # Check cache first
        if tenant_id in self._cache:
            return self._cache[tenant_id]

        if self._db is None:
            return None

        try:
            row = await self._db.fetchrow(
                "SELECT plan_data FROM agent_plans "
                "WHERE tenant_id = $1 AND status = 'pending'",
                tenant_id,
            )
            if row:
                plan = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                self._cache[tenant_id] = plan
                return plan
        except Exception as e:
            logger.warning(f"[PlanStore] Failed to load plan for {tenant_id}: {e}")

        return None

    async def save(self, tenant_id: str, plan_data: Dict[str, Any]) -> None:
        """Save a pending plan for the tenant."""
        self._cache[tenant_id] = plan_data

        if self._db is None:
            return

        try:
            await self._db.execute(
                """
                INSERT INTO agent_plans (tenant_id, plan_data, status, updated_at)
                VALUES ($1, $2, 'pending', NOW())
                ON CONFLICT (tenant_id)
                DO UPDATE SET plan_data = $2, status = 'pending', updated_at = NOW()
                """,
                tenant_id,
                json.dumps(plan_data),
            )
        except Exception as e:
            logger.warning(f"[PlanStore] Failed to save plan for {tenant_id}: {e}")

    async def pop(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get and remove a pending plan for the tenant."""
        plan = self._cache.pop(tenant_id, None)

        if self._db is not None:
            try:
                if plan is None:
                    row = await self._db.fetchrow(
                        "DELETE FROM agent_plans "
                        "WHERE tenant_id = $1 AND status = 'pending' "
                        "RETURNING plan_data",
                        tenant_id,
                    )
                    if row:
                        plan = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                else:
                    await self._db.execute(
                        "DELETE FROM agent_plans WHERE tenant_id = $1",
                        tenant_id,
                    )
            except Exception as e:
                logger.warning(f"[PlanStore] Failed to pop plan for {tenant_id}: {e}")

        return plan

    def __contains__(self, tenant_id: str) -> bool:
        """Check if a plan exists in cache (synchronous check)."""
        return tenant_id in self._cache

    async def has_plan(self, tenant_id: str) -> bool:
        """Check if a pending plan exists (with DB fallback)."""
        if tenant_id in self._cache:
            return True
        plan = await self.get(tenant_id)
        return plan is not None

    async def mark_approved(self, tenant_id: str) -> None:
        """Mark a plan as approved (keep in DB for audit)."""
        if self._db is None:
            return

        try:
            await self._db.execute(
                "UPDATE agent_plans SET status = 'approved', updated_at = NOW() "
                "WHERE tenant_id = $1",
                tenant_id,
            )
        except Exception as e:
            logger.warning(f"[PlanStore] Failed to mark plan approved: {e}")

    async def cleanup_expired(self, max_age_minutes: int = 60) -> int:
        """Remove plans older than max_age_minutes."""
        if self._db is None:
            return 0

        try:
            result = await self._db.execute(
                "DELETE FROM agent_plans "
                "WHERE updated_at < NOW() - INTERVAL '1 minute' * $1",
                max_age_minutes,
            )
            # Parse the DELETE count if available
            count_str = result.split(" ")[-1] if isinstance(result, str) else "0"
            return int(count_str) if count_str.isdigit() else 0
        except Exception as e:
            logger.warning(f"[PlanStore] Cleanup failed: {e}")
            return 0
