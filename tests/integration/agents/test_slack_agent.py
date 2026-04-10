"""Integration tests for SlackComposioAgent.

Tests tool selection, argument extraction, and response quality for:
- send_message: Send a message to a Slack channel or user
- fetch_messages: Fetch recent messages from a channel
- fetch_thread: Fetch replies in a message thread
- delete_message: Delete a message from a channel
- list_channels: List all available Slack channels
- find_channels: Search for channels by name or topic
- create_channel: Create a new Slack channel
- archive_channel: Archive an existing Slack channel
- find_users: Search for Slack users by name or email
- find_user_by_email: Look up a Slack user by email address
- add_reaction: Add an emoji reaction to a message
- get_user_presence: Check if a user is active or away
- invite_to_channel: Invite users to a Slack channel
- set_status: Set user custom status
- create_reminder: Create a Slack reminder
- connect_slack: Connect Slack account via OAuth
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.communication]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Send a Slack message to #engineering saying 'Deploy is done'", ["send_message"]),
    ("Show me the latest messages in the #general Slack channel", ["fetch_messages"]),
    ("List all Slack channels in the workspace", ["list_channels"]),
    ("Find the Slack user John Doe", ["find_users"]),
    ("Set a Slack reminder to check PR in 30 minutes", ["create_reminder"]),
    ("Show me the thread replies for that message in #engineering", ["fetch_thread"]),
    ("Delete the last message I sent in #general", ["delete_message"]),
    ("Create a new Slack channel called #project-alpha", ["create_channel"]),
    ("Archive the #old-project Slack channel", ["archive_channel"]),
    ("Search for Slack channels related to marketing", ["find_channels"]),
    ("Find the Slack user with email alice@company.com", ["find_user_by_email"]),
    ("React to that message with a thumbsup emoji", ["add_reaction"]),
    ("Is user U12345 currently online on Slack?", ["get_user_presence"]),
    ("Invite Bob and Alice to the #new-project channel", ["invite_to_channel", "find_users"]),
    ("Set my Slack status to 'In a meeting' with a calendar emoji", ["set_status"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(orchestrator_factory, user_input, expected_tools):
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", user_input)
    tools_called = [c["tool_name"] for c in recorder.tool_calls]
    assert any(t in tools_called for t in expected_tools), (
        f"Expected one of {expected_tools}, got {tools_called}"
    )


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------


async def test_extracts_message_fields(orchestrator_factory):
    """send_message should receive the channel and message text."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Send a Slack message to #engineering saying 'Build passed successfully'",
    )

    msg_calls = [c for c in recorder.tool_calls if c["tool_name"] == "send_message"]
    assert msg_calls, "send_message was never called"

    args = msg_calls[0]["arguments"]
    channel = args.get("channel", "").lower()
    assert "engineering" in channel, (
        f"Expected channel to contain 'engineering', got '{args.get('channel')}'"
    )
    text = args.get("text", "").lower()
    assert "build" in text or "passed" in text, (
        f"Expected text to contain message content, got '{args.get('text')}'"
    )


async def test_extracts_create_channel_fields(orchestrator_factory):
    """create_channel should receive the channel name."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Create a private Slack channel called team-backend",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "create_channel"]
    assert calls, "create_channel was never called"

    args = calls[0]["arguments"]
    name = args.get("name", "").lower()
    assert "team-backend" in name, (
        f"Expected name to contain 'team-backend', got '{args.get('name')}'"
    )


async def test_extracts_find_user_by_email_fields(orchestrator_factory):
    """find_user_by_email should receive the email address."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message(
        "test_user",
        "Look up the Slack user with email alice@example.com",
    )

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "find_user_by_email"]
    assert calls, "find_user_by_email was never called"

    args = calls[0]["arguments"]
    email = args.get("email", "").lower()
    assert "alice@example.com" in email, (
        f"Expected email to be 'alice@example.com', got '{args.get('email')}'"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------


async def test_response_quality_fetch(orchestrator_factory, llm_judge):
    """Fetching messages should produce a readable summary of the conversation."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Show me recent messages in #general on Slack")

    passed = await llm_judge(
        "Show me recent messages in #general on Slack",
        result.raw_message,
        "The response should present Slack messages or channel information. "
        "It is acceptable if the response mentions retrieving messages or "
        "showing channel data. It should not be an error message.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
