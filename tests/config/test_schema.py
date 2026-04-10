"""Tests for config schema validation."""

import pytest
from koa.config.schema import validate_config


class TestConfigValidation:
    def test_minimal_valid_config(self):
        cfg = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test123",
            },
            "database": "postgresql://user:pass@localhost/db",
        }
        errors = validate_config(cfg)
        assert errors == []

    def test_missing_provider(self):
        cfg = {
            "llm": {"model": "gpt-4o"},
            "database": "postgresql://user:pass@localhost/db",
        }
        errors = validate_config(cfg)
        assert any("provider" in e for e in errors)

    def test_missing_model(self):
        cfg = {
            "llm": {"provider": "openai"},
            "database": "postgresql://user:pass@localhost/db",
        }
        errors = validate_config(cfg)
        assert any("model" in e for e in errors)

    def test_missing_database(self):
        cfg = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
        }
        errors = validate_config(cfg)
        assert any("database" in e for e in errors)

    def test_invalid_provider(self):
        cfg = {
            "llm": {"provider": "invalid_provider", "model": "x"},
            "database": "postgresql://user:pass@localhost/db",
        }
        errors = validate_config(cfg)
        assert any("provider" in e for e in errors)

    def test_invalid_database_url(self):
        cfg = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "database": "not-a-valid-url",
        }
        errors = validate_config(cfg)
        assert any("database" in e for e in errors)

    def test_model_routing_valid(self):
        cfg = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "database": "postgresql://user:pass@localhost/db",
            "model_routing": {
                "enabled": True,
                "classifier_provider": "fast",
                "default_provider": "fast",
                "rules": [
                    {"score_range": [1, 50], "provider": "cheap"},
                    {"score_range": [51, 100], "provider": "strong"},
                ],
            },
        }
        errors = validate_config(cfg)
        assert errors == []

    def test_model_routing_overlapping_ranges(self):
        cfg = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "database": "postgresql://user:pass@localhost/db",
            "model_routing": {
                "enabled": True,
                "classifier_provider": "fast",
                "default_provider": "fast",
                "rules": [
                    {"score_range": [1, 60], "provider": "cheap"},
                    {"score_range": [50, 100], "provider": "strong"},
                ],
            },
        }
        errors = validate_config(cfg)
        assert any("overlap" in e.lower() for e in errors)