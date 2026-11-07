"""Tests for Codex runner, models and context compactor."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codex_runner import (
    CodexResult,
    _build_codex_args,
    _split_thoughts_and_final,
    check_codex_auth,
    find_codex_bin,
    install_codex,
    run_codex_device_auth,
)
from src.context_compactor import build_handoff_digest, prepare_turn_prompt
from src.models import get_available_codex_models, get_engine_for_model
from src.state import ChatState


def test_build_codex_args():
    # Fresh session
    args = _build_codex_args(
        codex_path="/usr/local/bin/codex",
        prompt="Hello Codex",
        thread_id="",
        model="gpt-6.1-sol",
        chat_dir="/tmp/test",
    )
    assert args == [
        "/usr/local/bin/codex",
        "-s",
        "danger-full-access",
        "-C",
        "/tmp/test",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-m",
        "gpt-6.1-sol",
        "Hello Codex",
    ]

    # Resumed session with effort
    args_effort = _build_codex_args(
        codex_path="/usr/local/bin/codex",
        prompt="Test effort",
        thread_id="",
        model="gpt-6.1-sol",
        chat_dir="/tmp/test",
        effort="high",
    )
    assert "-c" in args_effort
    assert 'model_reasoning_effort="high"' in args_effort

    # Resumed session
    args_resume = _build_codex_args(
        codex_path="/usr/local/bin/codex",
        prompt="Next turn",
        thread_id="01a1adbc-1234",
        model="",
        chat_dir="/tmp/test",
    )
    assert args_resume == [
        "/usr/local/bin/codex",
        "-s",
        "danger-full-access",
        "-C",
        "/tmp/test",
        "exec",
        "resume",
        "--json",
        "--skip-git-repo-check",
        "01a1adbc-1234",
        "Next turn",
    ]


def test_get_engine_for_model():
    assert get_engine_for_model("gemini-4-argon") == "agy"
    assert get_engine_for_model("gemini-3.8-flash-high") == "agy"
    assert get_engine_for_model("claude-opus-5-5-high") == "agy"
    assert get_engine_for_model("gpt-oss-120b-medium") == "agy"

    assert get_engine_for_model("gpt-6.1-sol") == "codex"
    assert get_engine_for_model("gpt-6-astra") == "codex"
    assert get_engine_for_model("gpt-5.6-luna") == "codex"
    assert get_engine_for_model("codex-mini") == "codex"
    assert get_engine_for_model("o3") == "codex"


def test_get_available_codex_models():
    models = get_available_codex_models()
    assert len(models) > 0
    slugs = [slug for slug, _ in models]
    assert "gpt-6.1-sol" in slugs


def test_context_compactor_handoff():
    cs = ChatState(chat_dir="/tmp", engine="agy", last_turn_engine="agy", conversation_id="conv-123")
    prompt = "Fix the login bug"

    # Switching from agy to codex (even if cs.engine was already set to codex by /model picker!)
    cs.engine = "codex"
    prepared = prepare_turn_prompt(prompt, cs, target_engine="codex")
    assert "[Контекст: переключение движка с AGY на CODEX]" in prepared
    assert "Fix the login bug" in prepared

    # Same engine as last_turn_engine -> prompt unchanged
    cs.last_turn_engine = "codex"
    cs.codex_thread_id = "thread-xyz"
    same = prepare_turn_prompt("Another question", cs, target_engine="codex")
    assert same == "Another question"

    # Switching back from codex to agy, then to codex again WITH existing codex_thread_id
    # must STILL inject handoff digest so Codex sees the intermediate AGY turns!
    cs.last_turn_engine = "agy"
    switched_again = prepare_turn_prompt("Continue after agy", cs, target_engine="codex")
    assert "[Контекст: переключение движка с AGY на CODEX]" in switched_again
    assert "Continue after agy" in switched_again


def test_split_thoughts_and_final():
    # Multiple messages: intermediate thoughts + final answer
    chunks = [
        "Thinking: check VPN logs",
        "Thinking: ports are open",
        "Final conclusion: VPN is working",
    ]
    thoughts, final_text = _split_thoughts_and_final(chunks)
    assert thoughts == ("Thinking: check VPN logs", "Thinking: ports are open")
    assert final_text == "Final conclusion: VPN is working"

    # Single message
    single_thoughts, single_text = _split_thoughts_and_final(["Only one answer"])
    assert single_thoughts == ()
    assert single_text == "Only one answer"

    # Empty
    empty_thoughts, empty_text = _split_thoughts_and_final([])
    assert empty_thoughts == ()
    assert empty_text == ""


@pytest.mark.asyncio
async def test_check_codex_auth_logged_in():
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b"Logged in using ChatGPT\n"))
        proc.returncode = 0
        mock_exec.return_value = proc

        ok, msg = await check_codex_auth()
        assert ok is True
        assert "Logged in using ChatGPT" in msg


@pytest.mark.asyncio
async def test_run_codex_device_auth_with_ansi_colors():
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        proc = MagicMock()
        proc.pid = 1234
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[
                b"Welcome to Codex\n",
                b"1. Open this link in your browser and sign in to your account\n",
                b"   \x1b[94mhttps://auth.openai.com/codex/device\x1b[0m\n",
                b"2. Enter this one-time code (expires in 15 minutes)\n",
                b"   \x1b[94m3MZW-PYSFB\x1b[0m\n",
                b"",
            ]
        )
        proc.stdout = mock_stdout
        proc.wait = AsyncMock(return_value=0)
        proc.returncode = 0
        mock_exec.return_value = proc

        reported_url = ""
        reported_code = ""

        async def on_code(url: str, code: str) -> None:
            nonlocal reported_url, reported_code
            reported_url = url
            reported_code = code

        ok, msg = await run_codex_device_auth(on_code)
        assert ok is True
        assert reported_url == "https://auth.openai.com/codex/device"
        assert reported_code == "3MZW-PYSFB"


def test_find_codex_bin(tmp_path: Path, monkeypatch):
    # Non-existent path returns fallback or None
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert find_codex_bin("/non/existent/codex") is None

    # Custom executable exists
    fake_bin = tmp_path / "my_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0")
    fake_bin.chmod(0o755)
    monkeypatch.undo()
    assert find_codex_bin(str(fake_bin)) == str(fake_bin)


@pytest.mark.asyncio
async def test_install_codex_already_installed(monkeypatch):
    monkeypatch.setattr("src.codex_runner.find_codex_bin", lambda *_: "/usr/local/bin/codex")
    ok, msg = await install_codex()
    assert ok is True
    assert "уже установлен" in msg


@pytest.mark.asyncio
async def test_install_codex_npm_missing(monkeypatch):
    monkeypatch.setattr("src.codex_runner.find_codex_bin", lambda *_: None)
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    ok, msg = await install_codex()
    assert ok is False
    assert "Node.js / npm не найден" in msg


@pytest.mark.asyncio
async def test_install_codex_npm_success(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else None)
    monkeypatch.setattr("os.path.isfile", lambda p: p == "/usr/bin/npm")
    monkeypatch.setattr("os.access", lambda p, mode: True)

    calls = []
    found_after_install = False

    def fake_find(*_):
        return "/usr/bin/codex" if found_after_install else None

    monkeypatch.setattr("src.codex_runner.find_codex_bin", fake_find)

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        proc = MagicMock()
        proc.returncode = 0

        async def fake_comm():
            nonlocal found_after_install
            found_after_install = True
            return b"added 7 packages", b""

        proc.communicate = fake_comm
        mock_exec.return_value = proc

        ok, msg = await install_codex()
        assert ok is True
        assert "успешно установлен" in msg
        assert "/usr/bin/codex" in msg


@pytest.mark.asyncio
async def test_check_codex_auth_not_installed(monkeypatch):
    monkeypatch.setattr("src.codex_runner.find_codex_bin", lambda *_: None)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    ok, msg = await check_codex_auth("/path/to/missing/codex")
    assert ok is False
    assert "не установлен" in msg


@pytest.mark.asyncio
async def test_run_codex_device_auth_missing_triggers_install(monkeypatch):
    # Missing initially, install succeeds, then proc runs
    state = {"installed": False}

    def fake_find(*_):
        return "/usr/bin/codex" if state["installed"] else None

    async def fake_install():
        state["installed"] = True
        return True, "installed"

    monkeypatch.setattr("src.codex_runner.find_codex_bin", fake_find)
    monkeypatch.setattr("src.codex_runner.install_codex", fake_install)
    monkeypatch.setattr("os.path.exists", lambda p: False)

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        proc = MagicMock()
        proc.pid = 999
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[
                b"https://auth.openai.com/codex/device\n",
                b"CODE-12345\n",
                b"",
            ]
        )
        proc.stdout = mock_stdout
        proc.wait = AsyncMock(return_value=0)
        proc.returncode = 0
        mock_exec.return_value = proc

        async def on_code(u, c):
            pass

        ok, msg = await run_codex_device_auth(on_code, codex_path="/missing/codex")
        assert ok is True
        assert state["installed"] is True
