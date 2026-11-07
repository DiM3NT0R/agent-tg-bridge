"""Tests for dynamic model discovery in src.models."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import (
    FALLBACK_MODELS,
    get_available_models,
    parse_models_output,
    strip_model_effort,
    deduplicate_models_by_base,
    extract_model_and_effort,
)


def test_parse_models_output_with_ansi_and_spinner() -> None:
    sample_output = """⠋ Fetching available models...⠙ Fetching available models...gemini-3.7-flash-high     Gemini 3.7 Flash (High)
gemini-3.7-flash-medium   Gemini 3.7 Flash (Medium)
claude-sonnet-4-6         Claude Sonnet 4.6 (Thinking)
invalid model with spaces
-invalid-dash             Invalid
gpt-oss-120b-medium       GPT-OSS 120B (Medium)
"""
    parsed = parse_models_output(sample_output)
    expected_ids = ["gemini-3.7-flash-high", "gemini-3.7-flash-medium", "claude-sonnet-4-6", "gpt-oss-120b-medium"]
    assert [p[0] for p in parsed] == expected_ids
    assert parsed[0] == ("gemini-3.7-flash-high", "Gemini 3.7 Flash (High)")
    assert parsed[2] == ("claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)")


def test_get_available_models_subprocess_success(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["agy", "models"],
        returncode=0,
        stdout="gemini-4.0-pro-high     Gemini 4.0 Pro (High)\nclaude-opus-5-0     Claude Opus 5.0\n",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", mock_run)

    models = get_available_models(force_refresh=True)
    assert models == [
        ("gemini-4.0-pro-high", "Gemini 4.0 Pro (High)"),
        ("claude-opus-5-0", "Claude Opus 5.0"),
    ]


def test_get_available_models_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_run_fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("agy binary not found")

    monkeypatch.setattr(subprocess, "run", mock_run_fail)
    import src.models
    src.models._cached_models = None

    models = get_available_models(force_refresh=True)
    assert models == FALLBACK_MODELS


def test_strip_model_effort() -> None:
    assert strip_model_effort("gemini-4-argon-medium", "Gemini 4 Argon (Medium)") == ("gemini-4-argon", "Gemini 4 Argon")
    assert strip_model_effort("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)") == ("gemini-3.8-flash", "Gemini 3.8 Flash")
    assert strip_model_effort("claude-opus-5-5-low", "Claude Opus 5.5 (Low)") == ("claude-opus-5-5", "Claude Opus 5.5")
    assert strip_model_effort("gpt-6.1-sol", "GPT-6.1 Sol") == ("gpt-6.1-sol", "GPT-6.1 Sol")


def test_deduplicate_models_by_base() -> None:
    raw = [
        ("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
        ("gemini-4-argon-high", "Gemini 4 Argon (High)"),
        ("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
        ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
        ("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
        ("gpt-6.1-sol", "GPT-6.1 Sol"),
    ]
    deduped = deduplicate_models_by_base(raw)
    assert deduped == [
        ("gemini-4-argon", "Gemini 4 Argon"),
        ("gemini-3.8-flash", "Gemini 3.8 Flash"),
        ("gpt-6.1-sol", "GPT-6.1 Sol"),
    ]


def test_extract_model_and_effort() -> None:
    assert extract_model_and_effort("gemini-4-argon-high", "") == ("gemini-4-argon", "high")
    assert extract_model_and_effort("gemini-4-argon-low", "high") == ("gemini-4-argon", "high")
    assert extract_model_and_effort("gemini-4-argon", "") == ("gemini-4-argon", "")
    assert extract_model_and_effort("gpt-6.1-sol", "medium") == ("gpt-6.1-sol", "medium")


def test_get_available_codex_models_filters_deprecated_and_ensures_gpt6(tmp_path: Path, monkeypatch) -> None:
    from src.models import get_available_codex_models

    fake_cache = tmp_path / "models_cache.json"
    fake_cache.write_text(json.dumps({
        "models": [
            {"slug": "gpt-5.4", "display_name": "GPT-5.4", "visibility": "list"},
            {"slug": "gpt-5.4-mini", "display_name": "GPT-5.4-Mini", "visibility": "list"},
            {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list"},
            {"slug": "gpt-5.6-sol", "display_name": "GPT-5.6-Sol", "visibility": "list"},
            {"slug": "gpt-reserve", "display_name": "GPT-Reserve", "visibility": "hide"},
            {"slug": "codex-auto-review", "display_name": "Codex Auto Review", "visibility": "hide"},
        ]
    }))
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_cache) if "models_cache.json" in p else p)

    models = get_available_codex_models(force_refresh=True)
    slugs = [slug for slug, _ in models]

    # Stale 5.4 / 5.5 and hidden models must be filtered out
    assert "gpt-5.4" not in slugs
    assert "gpt-5.4-mini" not in slugs
    assert "gpt-5.5" not in slugs
    assert "gpt-reserve" not in slugs
    assert "codex-auto-review" not in slugs

    # Modern flagship models must be injected at the top
    assert slugs[0] == "gpt-6.1-sol"
    assert "gpt-6-astra" in slugs
    assert "gpt-6.1-terra" in slugs
