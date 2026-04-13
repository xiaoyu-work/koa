from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from koa.providers.local_backend import LocalBackendClient

_PRIORITY_MAP = {
    0: "none",
    1: "low",
    2: "medium",
    3: "high",
}

_REVERSE_PRIORITY_MAP = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "urgent": 3,
}

logger = logging.getLogger(__name__)


class LocalTodoProvider:
    def __init__(self, tenant_id: str, backend_client: LocalBackendClient):
        self.tenant_id = tenant_id
        self.backend_client = backend_client

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        return True

    def _map_task(self, row: dict) -> dict:
        return {
            "id": row.get("id"),
            "title": row.get("title", ""),
            "due": row.get("due_date"),
            "priority": _PRIORITY_MAP.get(row.get("priority", 0), "none"),
            "completed": bool(row.get("is_completed", False)),
            "description": row.get("notes", ""),
            "list_id": row.get("list_id"),
            "list_name": row.get("list_name", ""),
        }

    async def list_tasks(
        self,
        list_id: Optional[str] = None,
        completed: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.search_local_todos(
                self.tenant_id,
                list_id=list_id,
                completed=completed,
            )
            tasks = [self._map_task(row) for row in result.get("todos", [])[:max_results]]
            return {"success": True, "data": tasks}
        except Exception as e:
            logger.error(f"Failed to list local todos: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_tasks(
        self,
        query: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.search_local_todos(
                self.tenant_id,
                query=query,
                list_id=list_id,
            )
            tasks = [self._map_task(row) for row in result.get("todos", [])]
            return {"success": True, "data": tasks}
        except Exception as e:
            logger.error(f"Failed to search local todos: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.create_local_todo(
                self.tenant_id,
                {
                    "title": title,
                    "notes": description,
                    "due_date": due,
                    "priority": _REVERSE_PRIORITY_MAP.get((priority or "none").lower(), 0),
                    "list_id": list_id,
                },
            )
            return {"success": result.get("created", False), "data": self._map_task(result["todo"])}
        except Exception as e:
            logger.error(f"Failed to create local todo: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def complete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.update_local_todo(
                self.tenant_id,
                task_id,
                {"is_completed": True},
            )
            return {"success": result.get("updated", False), "data": self._map_task(result["todo"])}
        except Exception as e:
            logger.error(f"Failed to complete local todo: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            result = await self.backend_client.delete_local_todo(self.tenant_id, task_id)
            return {"success": result.get("deleted", False)}
        except Exception as e:
            logger.error(f"Failed to delete local todo: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
