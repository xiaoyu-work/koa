"""PostgresCronJobStore — database-backed cron job persistence.

Drop-in replacement for the file-based CronJobStore. Uses the cron_jobs
table in PostgreSQL so jobs survive container restarts and support
multi-user isolation natively.
"""

import json
import logging
from typing import Dict, List, Optional

from .models import CronJob

logger = logging.getLogger(__name__)


class PostgresCronJobStore:
    """PostgreSQL persistence for cron jobs.

    Implements the same interface as CronJobStore (get, list, add, update,
    remove, find_by_name, find_by_hint, get_next_due_time) but backed by
    the ``cron_jobs`` table.

    Jobs are cached in memory for fast access by the timer loop, and
    synced to the database on every write.
    """

    def __init__(self, db):
        """
        Args:
            db: Database instance (onevalet.db.Database) with asyncpg pool.
        """
        self._db = db
        self._jobs: Dict[str, CronJob] = {}
        self._pending_deletes: List[str] = []

    async def load(self) -> None:
        """Load all jobs from database into memory cache."""
        rows = await self._db.fetch(
            "SELECT id, data FROM cron_jobs"
        )
        self._jobs = {}
        for row in rows:
            try:
                data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
                data["id"] = row["id"]
                job = CronJob.from_dict(data)
                self._jobs[job.id] = job
            except Exception as e:
                logger.warning(f"Skipping invalid cron job {row['id']}: {e}")

        logger.info(f"Loaded {len(self._jobs)} cron jobs from database")

    async def _persist(self, job: CronJob) -> None:
        """Write a single job to the database."""
        data = job.to_dict()
        await self._db.execute(
            """
            INSERT INTO cron_jobs (id, user_id, name, enabled, data, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, to_timestamp($6 / 1000.0), to_timestamp($7 / 1000.0))
            ON CONFLICT (id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                name = EXCLUDED.name,
                enabled = EXCLUDED.enabled,
                data = EXCLUDED.data,
                updated_at = EXCLUDED.updated_at
            """,
            job.id,
            job.user_id,
            job.name,
            job.enabled,
            json.dumps(data, ensure_ascii=False),
            job.created_at_ms,
            job.updated_at_ms,
        )

    async def save(self) -> None:
        """Persist all jobs to database and flush pending deletes."""
        # Flush deletes
        for job_id in self._pending_deletes:
            try:
                await self._db.execute("DELETE FROM cron_jobs WHERE id = $1", job_id)
            except Exception as e:
                logger.error(f"Failed to delete cron job {job_id}: {e}")
        self._pending_deletes.clear()

        # Upsert all current jobs
        for job in self._jobs.values():
            try:
                await self._persist(job)
            except Exception as e:
                logger.error(f"Failed to persist cron job {job.id}: {e}")

    def get(self, job_id: str) -> Optional[CronJob]:
        return self._jobs.get(job_id)

    def list(
        self,
        user_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[CronJob]:
        """List jobs, optionally filtered by user_id and enabled status."""
        jobs = list(self._jobs.values())
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if not include_disabled:
            jobs = [j for j in jobs if j.enabled]
        jobs.sort(key=lambda j: j.state.next_run_at_ms or float("inf"))
        return jobs

    def add(self, job: CronJob) -> None:
        self._jobs[job.id] = job

    def update(self, job: CronJob) -> None:
        self._jobs[job.id] = job

    def remove(self, job_id: str) -> bool:
        removed = self._jobs.pop(job_id, None) is not None
        if removed:
            self._pending_deletes.append(job_id)
        return removed

    def get_next_due_time(self) -> Optional[int]:
        """Return the earliest next_run_at_ms across all enabled, non-running jobs."""
        earliest: Optional[int] = None
        for job in self._jobs.values():
            if not job.enabled:
                continue
            if job.state.running_at_ms is not None:
                continue
            nra = job.state.next_run_at_ms
            if nra is not None and (earliest is None or nra < earliest):
                earliest = nra
        return earliest

    def find_by_name(self, name: str, user_id: Optional[str] = None) -> Optional[CronJob]:
        """Find a job by exact or partial name match."""
        name_lower = name.lower()
        for job in self._jobs.values():
            if user_id and job.user_id != user_id:
                continue
            if job.name.lower() == name_lower:
                return job
        for job in self._jobs.values():
            if user_id and job.user_id != user_id:
                continue
            if name_lower in job.name.lower():
                return job
        return None

    def find_by_hint(self, hint: str, user_id: Optional[str] = None) -> Optional[CronJob]:
        """Find a job by ID or name hint."""
        job = self.get(hint)
        if job and (user_id is None or job.user_id == user_id):
            return job
        return self.find_by_name(hint, user_id)

