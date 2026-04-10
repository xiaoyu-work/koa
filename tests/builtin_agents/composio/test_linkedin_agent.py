"""Unit tests for koa.builtin_agents.composio.linkedin_agent — pure logic only.

Tests _check_api_key, input validation, action constant wiring, and
success/failure formatting for every tool without making real API calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from koa.builtin_agents.composio.linkedin_agent import (
    LinkedInComposioAgent,
    _check_api_key,
    create_post,
    delete_post,
    get_company_info,
    get_my_profile,
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
        tool_names = {t.__name__ for t in LinkedInComposioAgent.tools}
        expected = {
            "create_post",
            "get_my_profile",
            "delete_post",
            "get_company_info",
            "connect_linkedin",
        }
        assert tool_names == expected

    def test_system_prompt_mentions_all_tools(self):
        prompt = LinkedInComposioAgent.domain_system_prompt
        for tool_fn in LinkedInComposioAgent.tools:
            assert tool_fn.__name__ in prompt, (
                f"Tool {tool_fn.__name__} not mentioned in domain_system_prompt"
            )


# =========================================================================
# Tool tests
# =========================================================================

_MODULE = "koa.builtin_agents.composio.linkedin_agent"


class TestCreatePost:
    @pytest.mark.asyncio
    async def test_empty_text(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await create_post("", context=_make_context())
        assert "text is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await create_post("hello world", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await create_post("hello world", context=_make_context())

        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0] == "LINKEDIN_CREATE_LINKED_IN_POST"
        assert "post created" in result.lower()

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(return_value=_failure_response())
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await create_post("test", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(side_effect=RuntimeError("timeout"))
            result = await create_post("test", context=_make_context())
        assert "Error" in result
        assert "timeout" in result


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
        assert mock_exec.call_args[0][0] == "LINKEDIN_GET_MY_INFO"
        assert "profile" in result.lower()


class TestDeletePost:
    @pytest.mark.asyncio
    async def test_empty_share_id(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await delete_post("", context=_make_context())
        assert "share_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await delete_post("share-1", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await delete_post("share-1", context=_make_context())
        assert mock_exec.call_args[0][0] == "LINKEDIN_DELETE_LINKED_IN_POST"
        assert mock_exec.call_args[1]["params"]["share_id"] == "share-1"
        assert "deleted" in result.lower()

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(return_value=_failure_response())
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await delete_post("share-1", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(side_effect=RuntimeError("timeout"))
            result = await delete_post("share-1", context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestGetCompanyInfo:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await get_company_info(context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_company_info(context=_make_context())
        assert mock_exec.call_args[0][0] == "LINKEDIN_GET_COMPANY_INFO"
        assert "company" in result.lower()

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(return_value=_failure_response())
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await get_company_info(context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(side_effect=RuntimeError("timeout"))
            result = await get_company_info(context=_make_context())
        assert "Error" in result
        assert "timeout" in result
