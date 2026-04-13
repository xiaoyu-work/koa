from datetime import datetime, timezone

import pytest

from koa.providers.calendar.local import LocalCalendarProvider


class DummyBackend:
    def __init__(self):
        self.calls = []

    async def list_local_events(self, tenant_id, time_min=None, time_max=None, query=None, max_results=10):
        self.calls.append(
            (
                "list_local_events",
                tenant_id,
                {
                    "time_min": time_min,
                    "time_max": time_max,
                    "query": query,
                    "max_results": max_results,
                },
            )
        )
        return {
            "events": [
                {
                    "id": "evt-1",
                    "title": "Team sync",
                    "description": "Weekly team sync",
                    "start_at": "2026-04-12T15:00:00+00:00",
                    "end_at": "2026-04-12T16:00:00+00:00",
                    "location": "Room A",
                }
            ]
        }

    async def create_local_event(self, tenant_id, payload):
        self.calls.append(("create_local_event", tenant_id, payload))
        return {
            "created": True,
            "event": {
                "id": "evt-1",
                "title": payload["title"],
                "start_at": payload["start_at"],
                "end_at": payload["end_at"],
                "location": payload.get("location"),
                "description": payload.get("description"),
            },
        }

    async def update_local_event(self, tenant_id, event_id, payload):
        self.calls.append(("update_local_event", tenant_id, event_id, payload))
        return {
            "updated": True,
            "event": {
                "id": event_id,
                "title": payload.get("title", "Team sync"),
                "start_at": payload.get("start_at", "2026-04-12T15:00:00+00:00"),
                "end_at": payload.get("end_at", "2026-04-12T16:00:00+00:00"),
                "location": payload.get("location"),
                "description": payload.get("description"),
            },
        }

    async def delete_local_event(self, tenant_id, event_id):
        self.calls.append(("delete_local_event", tenant_id, event_id))
        return {"deleted": True}


class FailingBackend:
    async def list_local_events(self, tenant_id, time_min=None, time_max=None, query=None, max_results=10):
        raise RuntimeError("backend down")

    async def create_local_event(self, tenant_id, payload):
        raise RuntimeError("backend down")

    async def update_local_event(self, tenant_id, event_id, payload):
        raise RuntimeError("backend down")

    async def delete_local_event(self, tenant_id, event_id):
        raise RuntimeError("backend down")


class TestLocalCalendarProvider:
    @pytest.mark.asyncio
    async def test_list_events_uses_backend_client_and_maps_event_shape(self):
        backend = DummyBackend()
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.list_events(
            time_min=datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc),
            time_max=datetime(2026, 4, 13, 0, 0, tzinfo=timezone.utc),
            query="sync",
            max_results=5,
        )

        assert result["success"] is True
        assert backend.calls == [
            (
                "list_local_events",
                "user-1",
                {
                    "time_min": "2026-04-12T00:00:00+00:00",
                    "time_max": "2026-04-13T00:00:00+00:00",
                    "query": "sync",
                    "max_results": 5,
                },
            )
        ]

        event = result["data"][0]
        assert event["id"] == "evt-1"
        assert event["event_id"] == "evt-1"
        assert event["summary"] == "Team sync"
        assert event["description"] == "Weekly team sync"
        assert event["start"] == datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc)
        assert event["end"] == datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc)
        assert event["location"] == "Room A"

    @pytest.mark.asyncio
    async def test_create_event_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.create_event(
            summary="Team sync",
            start=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc),
            description="Weekly team sync",
            location="Room A",
        )

        assert result["success"] is True
        assert backend.calls[-1] == (
            "create_local_event",
            "user-1",
            {
                "title": "Team sync",
                "description": "Weekly team sync",
                "start_at": "2026-04-12T15:00:00+00:00",
                "end_at": "2026-04-12T16:00:00+00:00",
                "location": "Room A",
            },
        )
        assert result["event_id"] == "evt-1"
        assert result["data"]["summary"] == "Team sync"

    @pytest.mark.asyncio
    async def test_update_event_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.update_event(
            event_id="evt-1",
            summary="Weekly sync",
            location="Room B",
        )

        assert result["success"] is True
        assert backend.calls[-1] == (
            "update_local_event",
            "user-1",
            "evt-1",
            {
                "title": "Weekly sync",
                "location": "Room B",
            },
        )
        assert result["event_id"] == "evt-1"
        assert result["data"]["summary"] == "Weekly sync"

    @pytest.mark.asyncio
    async def test_delete_event_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.delete_event("evt-1")

        assert result == {"success": True}
        assert backend.calls[-1] == ("delete_local_event", "user-1", "evt-1")

    @pytest.mark.asyncio
    async def test_ensure_valid_token_is_always_true(self):
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=DummyBackend())

        assert await provider.ensure_valid_token() is True

    @pytest.mark.asyncio
    async def test_list_events_returns_failure_when_backend_raises(self):
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.list_events()

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_create_event_returns_failure_when_backend_raises(self):
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.create_event(
            summary="Team sync",
            start=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 12, 16, 0, tzinfo=timezone.utc),
        )

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_update_event_returns_failure_when_backend_raises(self):
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.update_event(event_id="evt-1", summary="Weekly sync")

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_event_returns_failure_when_backend_raises(self):
        provider = LocalCalendarProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.delete_event("evt-1")

        assert result["success"] is False
        assert "backend down" in result["error"]
