"""Tests for koa.builtin_agents.composio.twitter_agent — tool functions.

Each tool is tested for:
- Required-param validation (returns error string when missing)
- Missing API key guard (_check_api_key)
- Success path (mocked execute_action)
- Failure path (API returns unsuccessful)
- Exception path (network / unexpected error)

The @tool decorator wraps functions into AgentTool objects whose ``executor``
has the signature ``async def executor(args: dict, context) -> str``.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from koa.builtin_agents.composio.twitter_agent import (
    # existing tools
    post_tweet,
    get_timeline,
    search_tweets,
    lookup_user,
    connect_twitter,
    # new tools
    delete_post,
    like_post,
    unlike_post,
    retweet,
    get_followers,
    get_following,
    follow_user,
    get_bookmarks,
    add_bookmark,
    send_dm,
    get_recent_dms,
    get_user_tweets,
    # helpers / constants
    _check_api_key,
    _ACTION_DELETE_POST,
    _ACTION_LIKE_POST,
    _ACTION_UNLIKE_POST,
    _ACTION_RETWEET,
    _ACTION_FOLLOWERS,
    _ACTION_FOLLOWING,
    _ACTION_FOLLOW,
    _ACTION_BOOKMARKS,
    _ACTION_ADD_BOOKMARK,
    _ACTION_SEND_DM,
    _ACTION_GET_DM_EVENTS,
    _ACTION_USER_TWEETS,
    TwitterComposioAgent,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def ctx():
    """Minimal AgentToolContext stub."""
    c = MagicMock()
    c.tenant_id = "test-tenant"
    return c


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    """Ensure COMPOSIO_API_KEY is set for all tests (unless overridden)."""
    monkeypatch.setenv("COMPOSIO_API_KEY", "test-key")


def _success_data(extra=None):
    return {"successfull": True, "data": extra or {"id": "12345"}}


def _failure_data(error="Something went wrong"):
    return {"successfull": False, "error": error}


def _mock_client():
    """Return a patch context for ComposioClient."""
    return patch("koa.builtin_agents.composio.twitter_agent.ComposioClient")


# ---------------------------------------------------------------------------
# _check_api_key
# ---------------------------------------------------------------------------

class TestCheckApiKey:

    def test_returns_none_when_set(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "key")
        assert _check_api_key() is None

    def test_returns_error_when_missing(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = _check_api_key()
        assert result is not None
        assert "not configured" in result


# ---------------------------------------------------------------------------
# Agent class wiring
# ---------------------------------------------------------------------------

class TestAgentWiring:

    def test_all_tools_registered(self):
        tool_names = {t.name for t in TwitterComposioAgent.tools}
        expected = {
            "post_tweet", "delete_post", "get_timeline", "search_tweets",
            "lookup_user", "like_post", "unlike_post", "retweet",
            "get_followers", "get_following", "follow_user",
            "get_bookmarks", "add_bookmark", "send_dm", "get_recent_dms",
            "get_user_tweets", "connect_twitter",
        }
        assert tool_names == expected

    def test_system_prompt_mentions_all_tools(self):
        prompt = TwitterComposioAgent.domain_system_prompt
        for t in TwitterComposioAgent.tools:
            assert t.name in prompt, (
                f"Tool {t.name} not found in domain_system_prompt"
            )

    def test_tool_count(self):
        assert len(TwitterComposioAgent.tools) == 17


# ---------------------------------------------------------------------------
# delete_post
# ---------------------------------------------------------------------------

class TestDeletePost:

    @pytest.mark.asyncio
    async def test_missing_tweet_id(self, ctx):
        result = await delete_post.executor({"tweet_id": ""}, ctx)
        assert "tweet_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, ctx, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await delete_post.executor({"tweet_id": "123"}, ctx)
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "id: 12345"

            result = await delete_post.executor({"tweet_id": "123"}, ctx)
            assert "deleted successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_DELETE_POST,
                params={"tweet_id": "123"},
                entity_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_failure(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_failure_data())
            MockClient.format_action_result = lambda d: "Error: Something went wrong"

            result = await delete_post.executor({"tweet_id": "123"}, ctx)
            assert "Failed" in result

    @pytest.mark.asyncio
    async def test_exception(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(side_effect=RuntimeError("boom"))

            result = await delete_post.executor({"tweet_id": "123"}, ctx)
            assert "Error" in result
            assert "boom" in result


# ---------------------------------------------------------------------------
# like_post
# ---------------------------------------------------------------------------

class TestLikePost:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await like_post.executor({"user_id": "", "tweet_id": "t1"}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_tweet_id(self, ctx):
        result = await like_post.executor({"user_id": "u1", "tweet_id": ""}, ctx)
        assert "tweet_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "liked"

            result = await like_post.executor({"user_id": "u1", "tweet_id": "t1"}, ctx)
            assert "liked successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_LIKE_POST,
                params={"user_id": "u1", "tweet_id": "t1"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# unlike_post
# ---------------------------------------------------------------------------

class TestUnlikePost:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await unlike_post.executor({"user_id": "", "tweet_id": "t1"}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_tweet_id(self, ctx):
        result = await unlike_post.executor({"user_id": "u1", "tweet_id": ""}, ctx)
        assert "tweet_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "unliked"

            result = await unlike_post.executor({"user_id": "u1", "tweet_id": "t1"}, ctx)
            assert "unliked successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_UNLIKE_POST,
                params={"user_id": "u1", "tweet_id": "t1"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# retweet
# ---------------------------------------------------------------------------

class TestRetweet:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await retweet.executor({"user_id": "", "tweet_id": "t1"}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_tweet_id(self, ctx):
        result = await retweet.executor({"user_id": "u1", "tweet_id": ""}, ctx)
        assert "tweet_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "retweeted"

            result = await retweet.executor({"user_id": "u1", "tweet_id": "t1"}, ctx)
            assert "Retweeted successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_RETWEET,
                params={"user_id": "u1", "tweet_id": "t1"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# get_followers
# ---------------------------------------------------------------------------

class TestGetFollowers:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await get_followers.executor({"user_id": ""}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_success_default_limit(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "followers list"

            result = await get_followers.executor({"user_id": "u1"}, ctx)
            assert "Followers:" in result
            call_kwargs = inst.execute_action.call_args
            assert call_kwargs[1]["params"]["max_results"] == 20

    @pytest.mark.asyncio
    async def test_custom_limit(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "followers"

            await get_followers.executor({"user_id": "u1", "max_results": 5}, ctx)
            call_kwargs = inst.execute_action.call_args
            assert call_kwargs[1]["params"]["max_results"] == 5


# ---------------------------------------------------------------------------
# get_following
# ---------------------------------------------------------------------------

class TestGetFollowing:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await get_following.executor({"user_id": ""}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "following list"

            result = await get_following.executor({"user_id": "u1"}, ctx)
            assert "Following:" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_FOLLOWING,
                params={"user_id": "u1", "max_results": 20},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# follow_user
# ---------------------------------------------------------------------------

class TestFollowUser:

    @pytest.mark.asyncio
    async def test_missing_source(self, ctx):
        result = await follow_user.executor(
            {"source_user_id": "", "target_user_id": "t1"}, ctx,
        )
        assert "source_user_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_target(self, ctx):
        result = await follow_user.executor(
            {"source_user_id": "s1", "target_user_id": ""}, ctx,
        )
        assert "target_user_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "followed"

            result = await follow_user.executor(
                {"source_user_id": "s1", "target_user_id": "t1"}, ctx,
            )
            assert "Followed user successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_FOLLOW,
                params={"source_user_id": "s1", "target_user_id": "t1"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# get_bookmarks
# ---------------------------------------------------------------------------

class TestGetBookmarks:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await get_bookmarks.executor({"user_id": ""}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "bookmarks"

            result = await get_bookmarks.executor({"user_id": "u1"}, ctx)
            assert "Bookmarked tweets:" in result


# ---------------------------------------------------------------------------
# add_bookmark
# ---------------------------------------------------------------------------

class TestAddBookmark:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await add_bookmark.executor({"user_id": "", "tweet_id": "t1"}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_missing_tweet_id(self, ctx):
        result = await add_bookmark.executor({"user_id": "u1", "tweet_id": ""}, ctx)
        assert "tweet_id is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "bookmarked"

            result = await add_bookmark.executor({"user_id": "u1", "tweet_id": "t1"}, ctx)
            assert "bookmarked successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_ADD_BOOKMARK,
                params={"user_id": "u1", "tweet_id": "t1"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# send_dm
# ---------------------------------------------------------------------------

class TestSendDm:

    @pytest.mark.asyncio
    async def test_missing_participants(self, ctx):
        result = await send_dm.executor({"participant_ids": [], "text": "hi"}, ctx)
        assert "participant_ids is required" in result

    @pytest.mark.asyncio
    async def test_missing_text(self, ctx):
        result = await send_dm.executor({"participant_ids": ["u1"], "text": ""}, ctx)
        assert "text is required" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "sent"

            result = await send_dm.executor(
                {"participant_ids": ["u1", "u2"], "text": "hello"}, ctx,
            )
            assert "DM sent successfully" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_SEND_DM,
                params={"participant_ids": ["u1", "u2"], "text": "hello"},
                entity_id="test-tenant",
            )


# ---------------------------------------------------------------------------
# get_recent_dms
# ---------------------------------------------------------------------------

class TestGetRecentDms:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, ctx, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        result = await get_recent_dms.executor({}, ctx)
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_success(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "dm events"

            result = await get_recent_dms.executor({}, ctx)
            assert "Recent DM events:" in result
            inst.execute_action.assert_awaited_once_with(
                _ACTION_GET_DM_EVENTS,
                params={},
                entity_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_exception(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(side_effect=RuntimeError("network"))

            result = await get_recent_dms.executor({}, ctx)
            assert "Error" in result
            assert "network" in result


# ---------------------------------------------------------------------------
# get_user_tweets
# ---------------------------------------------------------------------------

class TestGetUserTweets:

    @pytest.mark.asyncio
    async def test_missing_user_id(self, ctx):
        result = await get_user_tweets.executor({"user_id": ""}, ctx)
        assert "user_id is required" in result

    @pytest.mark.asyncio
    async def test_success_default_limit(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "tweets"

            result = await get_user_tweets.executor({"user_id": "u1"}, ctx)
            assert "User tweets:" in result
            call_kwargs = inst.execute_action.call_args
            assert call_kwargs[1]["params"]["max_results"] == 10

    @pytest.mark.asyncio
    async def test_custom_limit(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "tweets"

            await get_user_tweets.executor({"user_id": "u1", "max_results": 25}, ctx)
            call_kwargs = inst.execute_action.call_args
            assert call_kwargs[1]["params"]["max_results"] == 25

    @pytest.mark.asyncio
    async def test_failure(self, ctx):
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_failure_data("rate limited"))
            MockClient.format_action_result = lambda d: "Error: rate limited"

            result = await get_user_tweets.executor({"user_id": "u1"}, ctx)
            assert "Failed" in result

    @pytest.mark.asyncio
    async def test_uses_tenant_id(self, ctx):
        """Verify entity_id is set from context.tenant_id."""
        ctx.tenant_id = "custom-tenant"
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "tweets"

            await get_user_tweets.executor({"user_id": "u1"}, ctx)
            inst.execute_action.assert_awaited_once_with(
                _ACTION_USER_TWEETS,
                params={"user_id": "u1", "max_results": 10},
                entity_id="custom-tenant",
            )

    @pytest.mark.asyncio
    async def test_default_entity_id_when_tenant_none(self, ctx):
        """When tenant_id is None, entity_id should default to 'default'."""
        ctx.tenant_id = None
        with _mock_client() as MockClient:
            inst = MockClient.return_value
            inst.execute_action = AsyncMock(return_value=_success_data())
            MockClient.format_action_result = lambda d: "tweets"

            await get_user_tweets.executor({"user_id": "u1"}, ctx)
            inst.execute_action.assert_awaited_once_with(
                _ACTION_USER_TWEETS,
                params={"user_id": "u1", "max_results": 10},
                entity_id="default",
            )
