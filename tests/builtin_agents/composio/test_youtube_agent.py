"""Unit tests for koa.builtin_agents.composio.youtube_agent — pure logic only.

Tests _check_api_key, input validation, action constant wiring, and
success/failure formatting for every tool without making real API calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from koa.builtin_agents.composio.youtube_agent import (
    YouTubeComposioAgent,
    _check_api_key,
    download_captions,
    get_channel_activities,
    get_channel_by_handle,
    get_channel_stats,
    get_video_details,
    list_captions,
    list_channel_videos,
    list_playlists,
    list_subscriptions,
    search_videos,
    subscribe_channel,
)

# =========================================================================
# Helpers
# =========================================================================


def _make_context(tenant_id: str = "test-tenant") -> SimpleNamespace:
    """Create a minimal AgentToolContext-like object for tool tests."""
    return SimpleNamespace(tenant_id=tenant_id)


def _success_response(data: dict | None = None) -> dict:
    return {"successfull": True, "data": data or {"ok": True}}


def _failure_response(error: str = "Something went wrong") -> dict:
    return {"successfull": False, "error": error}


# =========================================================================
# _check_api_key
# =========================================================================


class TestCheckApiKey:
    def test_returns_none_when_set(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key-123")
        assert _check_api_key() is None

    def test_returns_error_when_missing(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = _check_api_key()
        assert result is not None
        assert "not configured" in result


# =========================================================================
# Agent class wiring
# =========================================================================


class TestAgentWiring:
    def test_all_tools_registered(self):
        tool_names = {t.__name__ for t in YouTubeComposioAgent.tools}
        expected = {
            "search_videos",
            "get_video_details",
            "list_playlists",
            "list_subscriptions",
            "list_channel_videos",
            "get_channel_stats",
            "get_channel_activities",
            "get_channel_by_handle",
            "subscribe_channel",
            "list_captions",
            "download_captions",
            "connect_youtube",
        }
        assert tool_names == expected

    def test_system_prompt_mentions_all_tools(self):
        prompt = YouTubeComposioAgent.domain_system_prompt
        for tool_fn in YouTubeComposioAgent.tools:
            assert tool_fn.__name__ in prompt, (
                f"Tool {tool_fn.__name__} not mentioned in domain_system_prompt"
            )


# =========================================================================
# Tool tests — each validates:
#   1. Input validation (missing required params)
#   2. Missing API key short-circuit
#   3. Successful execution with correct action constant
#   4. Failed execution formatting
#   5. Exception handling
# =========================================================================

_MODULE = "koa.builtin_agents.composio.youtube_agent"


class TestSearchVideos:
    @pytest.mark.asyncio
    async def test_empty_query(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await search_videos("", context=_make_context())
        assert "query is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await search_videos("test", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response({"items": "vid1"}))
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok: True"
            result = await search_videos("python", limit=5, context=_make_context())

        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        assert call_args[0][0] == "YOUTUBE_SEARCH_YOU_TUBE"
        assert call_args[1]["params"]["q"] == "python"
        assert "matching 'python'" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(return_value=_failure_response())
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await search_videos("test", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(side_effect=RuntimeError("timeout"))
            result = await search_videos("test", context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestGetVideoDetails:
    @pytest.mark.asyncio
    async def test_empty_video_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_video_details("", context=_make_context())
        assert "video_id is required" in result

    @pytest.mark.asyncio
    async def test_uses_correct_action(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            await get_video_details("abc123", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_VIDEO_DETAILS"


class TestListPlaylists:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_playlists(context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_LIST_USER_PLAYLISTS"
        assert "playlists" in result.lower()


class TestListSubscriptions:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_subscriptions(limit=5, context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_LIST_USER_SUBSCRIPTIONS"
        assert mock_exec.call_args[1]["params"]["maxResults"] == 5
        assert "subscriptions" in result.lower()


class TestListChannelVideos:
    @pytest.mark.asyncio
    async def test_empty_channel_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_channel_videos("", context=_make_context())
        assert "channel_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_channel_videos("UC123", limit=5, context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_LIST_CHANNEL_VIDEOS"
        assert mock_exec.call_args[1]["params"]["channelId"] == "UC123"
        assert "UC123" in result


class TestGetChannelStats:
    @pytest.mark.asyncio
    async def test_empty_channel_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_channel_stats("", context=_make_context())
        assert "channel_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_channel_stats("UC123", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_GET_CHANNEL_STATISTICS"
        assert "statistics" in result.lower()


class TestGetChannelActivities:
    @pytest.mark.asyncio
    async def test_empty_channel_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_channel_activities("", context=_make_context())
        assert "channel_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_channel_activities("UC123", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_GET_CHANNEL_ACTIVITIES"
        assert "activities" in result.lower()


class TestGetChannelByHandle:
    @pytest.mark.asyncio
    async def test_empty_handle(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_channel_by_handle("", context=_make_context())
        assert "handle is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_channel_by_handle("@mkbhd", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_GET_CHANNEL_ID_BY_HANDLE"
        assert mock_exec.call_args[1]["params"]["handle"] == "@mkbhd"
        assert "@mkbhd" in result


class TestSubscribeChannel:
    @pytest.mark.asyncio
    async def test_empty_channel_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await subscribe_channel("", context=_make_context())
        assert "channel_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await subscribe_channel("UC123", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_SUBSCRIBE_CHANNEL"
        assert "Subscribed" in result


class TestListCaptions:
    @pytest.mark.asyncio
    async def test_empty_video_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_captions("", context=_make_context())
        assert "video_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_captions("dQw4w9WgXcQ", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_LIST_CAPTION_TRACK"
        assert mock_exec.call_args[1]["params"]["videoId"] == "dQw4w9WgXcQ"
        assert "Caption tracks" in result


class TestDownloadCaptions:
    @pytest.mark.asyncio
    async def test_empty_caption_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await download_captions("", context=_make_context())
        assert "caption_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await download_captions("cap-123", context=_make_context())
        assert mock_exec.call_args[0][0] == "YOUTUBE_LOAD_CAPTIONS"
        assert mock_exec.call_args[1]["params"]["captionId"] == "cap-123"
        assert "Caption text" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response("Not found")
            )
            MockClient.format_action_result = lambda d: "Error: Not found"
            result = await download_captions("bad-id", context=_make_context())
        assert "Failed" in result
