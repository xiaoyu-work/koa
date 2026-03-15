"""Unit tests for onevalet.builtin_agents.composio.github_agent — pure logic only.

Tests _check_api_key, input validation, action constant wiring, and
success/failure formatting for every tool without making real API calls.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from onevalet.builtin_agents.composio.github_agent import (
    GitHubComposioAgent,
    _check_api_key,
    create_issue,
    list_issues,
    create_pull_request,
    list_pull_requests,
    search_repositories,
    get_repository,
    list_commits,
    merge_pull_request,
    list_branches,
    star_repo,
    list_notifications,
    create_issue_comment,
    list_my_repos,
    connect_github,
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
        tool_names = {t.__name__ for t in GitHubComposioAgent.tools}
        expected = {
            "create_issue",
            "list_issues",
            "create_pull_request",
            "list_pull_requests",
            "search_repositories",
            "get_repository",
            "list_commits",
            "merge_pull_request",
            "list_branches",
            "star_repo",
            "list_notifications",
            "create_issue_comment",
            "list_my_repos",
            "connect_github",
        }
        assert tool_names == expected

    def test_system_prompt_mentions_all_tools(self):
        prompt = GitHubComposioAgent.domain_system_prompt
        for tool_fn in GitHubComposioAgent.tools:
            assert tool_fn.__name__ in prompt, (
                f"Tool {tool_fn.__name__} not mentioned in domain_system_prompt"
            )


# =========================================================================
# Tool tests — existing tools
# =========================================================================

_MODULE = "onevalet.builtin_agents.composio.github_agent"


class TestCreateIssue:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await create_issue("", "repo", "title", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_empty_title(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await create_issue("owner", "repo", "", context=_make_context())
        assert "title is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await create_issue("owner", "repo", "title", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_CREATE_AN_ISSUE"
        assert "Issue created" in result


class TestListIssues:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_issues("", "repo", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_issues("owner", "repo", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_LIST_REPOSITORY_ISSUES"
        assert "Issues" in result


class TestSearchRepositories:

    @pytest.mark.asyncio
    async def test_empty_query(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await search_repositories("", context=_make_context())
        assert "query is required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await search_repositories("python", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_SEARCH_REPOSITORIES"
        assert "python" in result


# =========================================================================
# Tool tests — new tools
# =========================================================================


class TestGetRepository:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_repository("", "repo", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_empty_repo(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await get_repository("owner", "", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await get_repository("owner", "repo", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await get_repository("owner", "repo", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_GET_A_REPOSITORY"
        assert mock_exec.call_args[1]["params"]["owner"] == "owner"
        assert mock_exec.call_args[1]["params"]["repo"] == "repo"
        assert "owner/repo" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await get_repository("owner", "repo", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await get_repository("owner", "repo", context=_make_context())
        assert "Error" in result
        assert "timeout" in result


class TestListCommits:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_commits("", "repo", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_success_default_params(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_commits("owner", "repo", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_LIST_COMMITS"
        params = mock_exec.call_args[1]["params"]
        assert params["owner"] == "owner"
        assert params["per_page"] == 10
        assert "sha" not in params
        assert "Commits" in result

    @pytest.mark.asyncio
    async def test_success_with_sha(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_commits("owner", "repo", sha="main", per_page=5, context=_make_context())
        params = mock_exec.call_args[1]["params"]
        assert params["sha"] == "main"
        assert params["per_page"] == 5

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await list_commits("owner", "repo", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await list_commits("owner", "repo", context=_make_context())
        assert "Error" in result


class TestMergePullRequest:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await merge_pull_request("", "repo", 1, context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await merge_pull_request("owner", "repo", 1, context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await merge_pull_request("owner", "repo", 42, context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_MERGE_A_PULL_REQUEST"
        params = mock_exec.call_args[1]["params"]
        assert params["pull_number"] == 42
        assert params["merge_method"] == "merge"
        assert "#42 merged" in result

    @pytest.mark.asyncio
    async def test_squash_merge(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await merge_pull_request(
                "owner", "repo", 10, merge_method="squash", context=_make_context()
            )
        assert mock_exec.call_args[1]["params"]["merge_method"] == "squash"

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await merge_pull_request("owner", "repo", 1, context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await merge_pull_request("owner", "repo", 1, context=_make_context())
        assert "Error" in result


class TestListBranches:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await list_branches("", "repo", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_branches("owner", "repo", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_LIST_BRANCHES"
        assert "Branches" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await list_branches("owner", "repo", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await list_branches("owner", "repo", context=_make_context())
        assert "Error" in result


class TestStarRepo:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await star_repo("", "repo", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await star_repo("owner", "repo", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER"
        assert "Starred" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await star_repo("owner", "repo", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await star_repo("owner", "repo", context=_make_context())
        assert "Error" in result


class TestListNotifications:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await list_notifications(context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success_default(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_notifications(context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_LIST_NOTIFICATIONS_FOR_THE_AUTHENTICATED_USER"
        assert mock_exec.call_args[1]["params"]["all"] is False
        assert "notifications" in result.lower()

    @pytest.mark.asyncio
    async def test_success_all(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_notifications(all=True, context=_make_context())
        assert mock_exec.call_args[1]["params"]["all"] is True

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await list_notifications(context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await list_notifications(context=_make_context())
        assert "Error" in result


class TestCreateIssueComment:

    @pytest.mark.asyncio
    async def test_empty_owner(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await create_issue_comment("", "repo", 1, "comment", context=_make_context())
        assert "owner and repo are required" in result

    @pytest.mark.asyncio
    async def test_empty_body(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        result = await create_issue_comment("owner", "repo", 1, "", context=_make_context())
        assert "body is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await create_issue_comment("owner", "repo", 1, "comment", context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await create_issue_comment("owner", "repo", 5, "looks good", context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_CREATE_AN_ISSUE_COMMENT"
        params = mock_exec.call_args[1]["params"]
        assert params["issue_number"] == 5
        assert params["body"] == "looks good"
        assert "Comment added" in result
        assert "#5" in result

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await create_issue_comment("owner", "repo", 1, "test", context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await create_issue_comment("owner", "repo", 1, "test", context=_make_context())
        assert "Error" in result


class TestListMyRepos:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await list_my_repos(context=_make_context())
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success_default_params(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_my_repos(context=_make_context())
        assert mock_exec.call_args[0][0] == "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER"
        params = mock_exec.call_args[1]["params"]
        assert params["per_page"] == 10
        assert params["sort"] == "updated"
        assert "repositories" in result.lower()

    @pytest.mark.asyncio
    async def test_success_custom_params(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        mock_exec = AsyncMock(return_value=_success_response())
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = mock_exec
            MockClient.format_action_result = lambda d: "ok"
            result = await list_my_repos(per_page=5, sort="created", context=_make_context())
        params = mock_exec.call_args[1]["params"]
        assert params["per_page"] == 5
        assert params["sort"] == "created"

    @pytest.mark.asyncio
    async def test_failure(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                return_value=_failure_response()
            )
            MockClient.format_action_result = lambda d: "Error: fail"
            result = await list_my_repos(context=_make_context())
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        with patch(f"{_MODULE}.ComposioClient") as MockClient:
            MockClient.return_value.execute_action = AsyncMock(
                side_effect=RuntimeError("timeout")
            )
            result = await list_my_repos(context=_make_context())
        assert "Error" in result
