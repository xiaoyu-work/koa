"""Tests for Anthropic prompt caching."""

import copy
import importlib
import importlib.util
import os


# Direct import to avoid koa/__init__.py which requires Python 3.9+
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_caching_module = _load_module(
    "prompt_caching",
    os.path.join(os.path.dirname(__file__), "..", "..", "koa", "llm", "prompt_caching.py"),
)
apply_anthropic_cache_control = _caching_module.apply_anthropic_cache_control
is_anthropic_model = _caching_module.is_anthropic_model


class TestApplyAnthropicCacheControl:
    """Tests for apply_anthropic_cache_control()."""

    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_system_only(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        result = apply_anthropic_cache_control(messages)

        # System prompt should be wrapped in list format with cache_control
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "You are helpful."
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_does_not_mutate_original(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        original = copy.deepcopy(messages)
        apply_anthropic_cache_control(messages)
        assert messages == original

    def test_system_and_3_strategy(self):
        """4 breakpoints: system + last 3 non-system messages."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        result = apply_anthropic_cache_control(messages)

        # System: breakpoint 1
        assert "cache_control" in result[0]["content"][0]

        # Messages 1-2 (Q1, A1): no breakpoints
        assert isinstance(result[1]["content"], str)  # not wrapped
        assert isinstance(result[2]["content"], str)

        # Messages 3-5 (Q2, A2, Q3): breakpoints 2-4
        assert result[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[4]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[5]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_message(self):
        """Without system, all 4 breakpoints go to last 4 non-system messages."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        result = apply_anthropic_cache_control(messages)

        # Q1: no breakpoint (5 messages, only last 4 get breakpoints)
        assert isinstance(result[0]["content"], str)

        # A1, Q2, A2, Q3: breakpoints 1-4
        for i in [1, 2, 3, 4]:
            assert result[i]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_message(self):
        """Tool messages get cache_control at message level."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "Let me search."},
            {"role": "tool", "content": '{"results": []}'},
        ]
        result = apply_anthropic_cache_control(messages)

        # Tool message: cache_control at message level
        assert result[3]["cache_control"] == {"type": "ephemeral"}
        # Content unchanged
        assert result[3]["content"] == '{"results": []}'

    def test_list_content_format(self):
        """Content already in list format gets marker on last block."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        result = apply_anthropic_cache_control(messages)

        content = result[0]["content"]
        assert len(content) == 2
        # First block: no marker
        assert "cache_control" not in content[0]
        # Last block: has marker
        assert content[1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_content(self):
        """Empty content gets marker at message level."""
        messages = [
            {"role": "assistant", "content": ""},
        ]
        result = apply_anthropic_cache_control(messages)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_none_content(self):
        """None content gets marker at message level."""
        messages = [
            {"role": "assistant", "content": None},
        ]
        result = apply_anthropic_cache_control(messages)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_custom_ttl(self):
        messages = [{"role": "system", "content": "System"}]
        result = apply_anthropic_cache_control(messages, ttl="1h")
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_few_messages_all_get_breakpoints(self):
        """With fewer than 4 non-system messages, all get breakpoints."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_anthropic_cache_control(messages)

        # System: breakpoint
        assert "cache_control" in result[0]["content"][0]
        # User: breakpoint
        assert "cache_control" in result[1]["content"][0]


class TestIsAnthropicModel:
    def test_anthropic_provider(self):
        assert is_anthropic_model("anthropic", "claude-3-5-sonnet") is True

    def test_claude_in_model_name(self):
        assert is_anthropic_model("openai", "claude-3-opus") is True

    def test_openai_model(self):
        assert is_anthropic_model("openai", "gpt-4o") is False

    def test_gemini_model(self):
        assert is_anthropic_model("gemini", "gemini-pro") is False
