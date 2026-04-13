from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from koa.providers.local_backend import LocalBackendClient

logger = logging.getLogger(__name__)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class LocalCalendarProvider:
    def __init__(self, tenant_id: str, backend_client: LocalBackendClient):
        self.tenant_id = tenant_id
        self.backend_client = backend_client

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        return True

    def _map_event(self, row: dict) -> dict:
        return {
            "id": row.get("id"),
            "event_id": row.get("id"),
            "summary": row.get("title", "No title"),
            "description": row.get("description", ""),
            "start": _parse_iso_datetime(row.get("start_at")),
            "end": _parse_iso_datetime(row.get("end_at")),
            "location": row.get("location", ""),
        }

    async def list_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
        query: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.list_local_events(
                self.tenant_id,
                time_min=time_min.isoformat() if time_min else None,
                time_max=time_max.isoformat() if time_max else None,
                query=query,
                max_results=max_results,
            )
            events = [self._map_event(row) for row in result.get("events", [])]
            return {"success": True, "data": events, "count": len(events)}
        except Exception as e:
            logger.error(f"Failed to list local events: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.create_local_event(
                self.tenant_id,
                {
                    "title": summary,
                    "description": description,
                    "start_at": start.isoformat(),
                    "end_at": end.isoformat(),
                    "location": location,
                },
            )
            event = self._map_event(result["event"])
            return {
                "success": result.get("created", False),
                "event_id": event["event_id"],
                "data": event,
            }
        except Exception as e:
            logger.error(f"Failed to create local event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {}
        if summary is not None:
            payload["title"] = summary
        if start is not None:
            payload["start_at"] = start.isoformat()
        if end is not None:
            payload["end_at"] = end.isoformat()
        if description is not None:
            payload["description"] = description
        if location is not None:
            payload["location"] = location

        try:
            result = await self.backend_client.update_local_event(
                self.tenant_id,
                event_id,
                payload,
            )
            event = self._map_event(result["event"])
            return {
                "success": result.get("updated", False),
                "event_id": event["event_id"],
                "data": event,
            }
        except Exception as e:
            logger.error(f"Failed to update local event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.delete_local_event(self.tenant_id, event_id)
            return {"success": result.get("deleted", False)}
        except Exception as e:
            logger.error(f"Failed to delete local event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
