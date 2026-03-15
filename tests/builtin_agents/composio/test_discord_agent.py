"""Unit tests for onevalet.builtin_agents.composio.discord_agent — pure logic only.

Tests _check_api_key, input validation, action constant wiring, and
success/failure formatting for every tool without making real API calls.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from onevalet.builtin_agents.composio.discord_agent import (
    DiscordComposioAgent,
    _check_api_key,
    send_message,
    list_channels,
    list_servers,
    get_my_profile,
    list_connections,
    get_guild_member,
    connect_discord,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_context(tenant_id: str = "test-tenant") -> SimpleNamespace:
    """Create a minimal AgentToolContext-like object for tool tests."""
    return SimpleNamespace(tenant_id=tenant_id)


def _success_response(data: dict = None) -> dict:
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
        tool_names = {t.__name__ for t in DiscordComposioAgent.tools}
        expected = {
            "send_message",
            "list_channels",
            "list_servers",
            "get_my_profile",
            "list_connections",
            "get_guild_member",
            "connect_discord",
        }
        assert tool_names == expected

    def test_system_prompt_mentions_all_tools(self):
        prompt = DiscordComposioAgent.domain_system_prompt
        for tool_fn in DiscordComposioAgent.tools:
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

_MODULE = "onevalet.builtin_agents.composio.discord_agent"


class TestSendMessage:

    @pytest.mark.asyncio
    async def test_empty_channel_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await send_message("", "hello", context=_make_context())
        assert "channel_id is required" in result

    @pytest.mark.asyncio
    async def test_empty_content(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await send_message("ch123", "", context=_make_context())
        assert "content is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await send_message("ch123", "hello", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await send_message("ch123", "hello", context=_make_context())

        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0] == "DISCORDBOT_CREATE_MESSAGE"
        assert "Message sent" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await send_message("ch123", "hello", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await send_message("ch123", "hello", context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestListChannels:

    @pytest.mark.asyncio
    async def test_empty_guild_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_channels("", context=_make_context())
        assert "guild_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_channels("guild-1", context=_make_context())
        assert mock_exec.call_args[0][0] == "DISCORDBOT_LIST_GUILD_CHANNELS"
        assert "Channels" in result


class TestListServers:

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_servers(context=_make_context())
        assert mock_exec.call_args[0][0] == "DISCORD_LIST_MY_GUILDS"
        assert "servers" in result.lower()


class TestGetMyProfile:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await get_my_profile(context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_my_profile(context=_make_context())
        assert mock_exec.call_args[0][0] == "DISCORD_GET_MY_USER"
        assert "profile" in result.lower()

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await get_my_profile(context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await get_my_profile(context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestListConnections:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await list_connections(context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_connections(context=_make_context())
        assert mock_exec.call_args[0][0] == "DISCORD_LIST_MY_CONNECTIONS"
        assert "connections" in result.lower()

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await list_connections(context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await list_connections(context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestGetGuildMember:

    @pytest.mark.asyncio
    async def test_empty_guild_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_guild_member("", context=_make_context())
        assert "guild_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await get_guild_member("guild-1", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_guild_member("guild-1", context=_make_context())
        assert mock_exec.call_args[0][0] == "DISCORD_GET_MY_GUILD_MEMBER"
        assert mock_exec.call_args[1]["params"]["guild_id"] == "guild-1"
        assert "guild-1" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await get_guild_member("guild-1", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await get_guild_member("guild-1", context=_make_context())
        assert "Error" in result
        assert "timeout" in result
