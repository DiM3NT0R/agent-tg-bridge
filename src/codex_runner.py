"""Spawn the OpenAI Codex CLI (`codex`) and stream its JSON output.

Supports:
  - codex exec [PROMPT] --json ...
  - codex exec resume <thread_id> [PROMPT] --json ...
  - codex login --device-auth
  - codex login status
"""
from __future__ import annotations

import asyncio
import gc
import glob
import html
import json
import logging
import os
import re
import shutil
import signal
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

LOG = logging.getLogger("antigravity_telegram_bridge")

_STDOUT_CAP_BYTES = 524_288  # 512 KiB
DEFAULT_OVERALL_TIMEOUT_S = 1800.0  # 30 minutes
DEFAULT_INACTIVITY_TIMEOUT_S = 600.0  # 10 minutes without model/system response
_codex_condition = asyncio.Condition()
_codex_active = 0
_codex_updating = False


@asynccontextmanager
async def codex_turn_guard():
    global _codex_active
    async with _codex_condition:
        await _codex_condition.wait_for(lambda: not _codex_updating)
        _codex_active += 1
    try:
        yield
    finally:
        async with _codex_condition:
            _codex_active -= 1
            _codex_condition.notify_all()


@asynccontextmanager
async def codex_update_guard():
    global _codex_updating
    async with _codex_condition:
        await _codex_condition.wait_for(lambda: not _codex_updating and _codex_active == 0)
        _codex_updating = True
    try:
        yield
    finally:
        async with _codex_condition:
            _codex_updating = False
            _codex_condition.notify_all()


@dataclass(frozen=True)
class CodexResult:
    text: str
    exit_code: int
    stderr: str
    thread_id: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    thoughts: tuple[str, ...] = ()


def _split_thoughts_and_final(chunks: list[str]) -> tuple[tuple[str, ...], str]:
    if len(chunks) > 1:
        return tuple(chunks[:-1]), chunks[-1].strip()
    elif chunks:
        return (), chunks[0].strip()
    return (), ""


def _reap_zombies() -> None:
    try:
        while True:
            pid, _status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except (ChildProcessError, OSError):
        pass


def find_codex_bin(custom_path: str | None = None) -> str | None:
    """Find the codex CLI executable in PATH or standard install locations."""
    managed = Path.home() / ".antigravity/bridge/codex-npm"
    managed_bin = managed / "bin/codex"
    if (not custom_path or custom_path == "/usr/local/bin/codex") and (managed / ".bridge-active").is_file():
        if managed_bin.is_file() and os.access(managed_bin, os.X_OK):
            return str(managed_bin)
    if custom_path:
        try:
            if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
                return custom_path
        except Exception:
            pass

    # Check PATH
    try:
        found = shutil.which("codex")
        if found and os.path.isfile(found) and os.access(found, os.X_OK):
            return found
    except Exception:
        pass

    # Check common system and npm paths
    candidates = [
        "/usr/local/bin/codex",
        "/usr/bin/codex",
        os.path.expanduser("~/.npm-global/bin/codex"),
        os.path.expanduser("~/.local/bin/codex"),
    ]
    for c in candidates:
        try:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        except Exception:
            pass

    # Wildcard search for custom/versioned directories like /usr/local/lib/codex-*/bin/codex
    try:
        for c in sorted(glob.glob("/usr/local/lib/codex*/bin/codex"), reverse=True):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
    except Exception:
        pass

    return None


async def install_codex() -> tuple[bool, str]:
    """Attempt to install OpenAI Codex CLI (@openai/codex) via npm.

    Returns:
        (success: bool, message: str)
    """
    existing = find_codex_bin()
    if existing:
        return True, f"Codex CLI уже установлен: {existing}"

    # 1. Locate npm
    npm_bin = shutil.which("npm")
    if not npm_bin:
        for c in ("/usr/bin/npm", "/usr/local/bin/npm"):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                npm_bin = c
                break

    # 2. If npm not found, check if apt-get is available to install nodejs and npm
    if not npm_bin:
        apt_bin = shutil.which("apt-get")
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        if apt_bin and is_root:
            LOG.info("npm not found. Attempting apt-get install -y nodejs npm...")
            try:
                p_up = await asyncio.create_subprocess_exec(
                    apt_bin, "update", "-qq",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await p_up.communicate()
                p_in = await asyncio.create_subprocess_exec(
                    apt_bin, "install", "-y", "-qq", "nodejs", "npm",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await p_in.communicate()
                npm_bin = shutil.which("npm") or ("/usr/bin/npm" if os.path.isfile("/usr/bin/npm") else None)
            except Exception as e:
                LOG.warning("apt-get install nodejs npm failed: %s", e)

    if not npm_bin:
        return False, "Node.js / npm не найден в системе. Установите npm командой: apt-get install -y npm nodejs"

    LOG.info("Installing @openai/codex via %s...", npm_bin)
    try:
        proc = await asyncio.create_subprocess_exec(
            npm_bin, "install", "-g", "@openai/codex@latest",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = (stdout.decode("utf-8", errors="replace") + "\n" + stderr.decode("utf-8", errors="replace")).strip()
    except Exception as e:
        return False, f"Ошибка запуска npm: {e}"

    found = find_codex_bin()
    if proc.returncode == 0 and found:
        if found != "/usr/local/bin/codex" and not os.path.exists("/usr/local/bin/codex"):
            try:
                os.symlink(found, "/usr/local/bin/codex")
            except Exception:
                pass
        return True, f"Codex CLI успешно установлен: {found}"

    # Fallback for unprivileged users: try prefix ~/.npm-global
    npm_global = os.path.expanduser("~/.npm-global")
    try:
        proc2 = await asyncio.create_subprocess_exec(
            npm_bin, "install", "-g", "--prefix", npm_global, "@openai/codex@latest",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc2.communicate()
        found = find_codex_bin()
        if proc2.returncode == 0 and found:
            return True, f"Codex CLI успешно установлен: {found}"
    except Exception as e:
        LOG.warning("npm prefix fallback failed: %s", e)

    return False, f"Установка @openai/codex не удалась (код {proc.returncode}):\n{out[-300:]}"


def _npm_codex_prefix(binary: str) -> str | None:
    """Identify the npm prefix owning the selected executable, without changing other installs."""
    path = Path(binary).resolve()
    for parent in path.parents:
        if parent.name != "codex" or parent.parent.name != "@openai":
            continue
        if parent.parent.parent.name != "node_modules" or parent.parent.parent.parent.name != "lib":
            continue
        try:
            package = json.loads((parent / "package.json").read_text(encoding="utf-8"))
            if package.get("name") == "@openai/codex":
                return str(parent.parent.parent.parent.parent)
        except (OSError, ValueError):
            return None
    return None


async def _codex_update_command(*args: str, timeout: float = 120) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.communicate()
        raise
    output = stdout if proc.returncode == 0 else stdout + b"\n" + stderr
    return proc.returncode, output.decode("utf-8", errors="replace").strip()


async def update_npm_codex() -> tuple[bool, str]:
    """Update only the npm package that owns the CLI selected by the bridge."""
    binary = find_codex_bin("/usr/local/bin/codex")
    npm = shutil.which("npm")
    if not binary or not npm:
        return False, "Codex CLI или npm не найден"
    prefix = _npm_codex_prefix(binary)
    if not prefix:
        return False, f"Codex CLI {binary} не является npm-установкой; автообновление пропущено"
    try:
        rc, installed = await _codex_update_command(binary, "--version", timeout=10)
        view_rc, latest = await _codex_update_command(npm, "view", "@openai/codex@latest", "version", timeout=30)
        installed_match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", installed)
        latest_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", latest)
        if rc or view_rc or not installed_match or not latest_match:
            return False, "Не удалось определить установленную или последнюю версию Codex CLI"
        if tuple(map(int, installed_match.groups())) >= tuple(map(int, latest_match.groups())):
            return False, f"Codex CLI уже актуален: {installed_match.group(0)}"
        rc, output = await _codex_update_command(npm, "install", "-g", "--prefix", prefix, "@openai/codex@latest")
        if rc:
            if not re.search(r"EACCES|EPERM|permission denied", output, re.IGNORECASE):
                return False, f"Обновление Codex CLI не удалось: {output[-300:]}"
            managed = Path.home() / ".antigravity/bridge/codex-npm"
            rc, output = await _codex_update_command(npm, "install", "-g", "--prefix", str(managed), "@openai/codex@latest")
            if rc:
                return False, f"Установка Codex CLI в домашний каталог не удалась: {output[-300:]}"
            binary = str(managed / "bin/codex")
        rc, updated = await _codex_update_command(binary, "--version", timeout=10)
        match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", updated)
        if rc or not match or tuple(map(int, match.groups())) < tuple(map(int, latest_match.groups())):
            return False, "npm завершился, но используемый мостом Codex CLI не обновился"
        if binary == str(Path.home() / ".antigravity/bridge/codex-npm/bin/codex"):
            (Path(binary).parent.parent / ".bridge-active").touch()
        from src import models
        models._cached_codex_models = None
        models._cached_codex_at = 0.0
        return True, f"Codex CLI обновлён: {installed_match.group(0)} → {match.group(0)}"
    except (OSError, asyncio.TimeoutError) as exc:
        return False, f"Не удалось проверить или обновить Codex CLI: {exc}"


async def codex_auto_update_loop(stop: asyncio.Event, *, is_busy: Callable[[], bool], interval_s: float = 86400, initial_delay_s: float = 10) -> None:
    """Check on startup and daily; do not run npm during a Codex turn."""
    delay = initial_delay_s
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except asyncio.TimeoutError:
            pass
        if is_busy():
            delay = 60
            continue
        async with codex_update_guard():
            if is_busy():
                delay = 60
                continue
            updated, message = await update_npm_codex()
            LOG.info("Codex auto-update: %s", message)
        delay = interval_s


def _build_codex_args(
    *,
    codex_path: str,
    prompt: str,
    thread_id: str,
    model: str,
    chat_dir: str,
    sandbox: str = "danger-full-access",
    effort: str = "",
) -> list[str]:
    args = [codex_path]
    if sandbox:
        args.extend(["-s", sandbox])
    if chat_dir:
        args.extend(["-C", chat_dir])
    if effort:
        args.extend(["-c", f'model_reasoning_effort="{effort}"'])

    if thread_id:
        args.extend(["exec", "resume", "--json", "--skip-git-repo-check"])
        if model:
            args.extend(["-m", model])
        args.extend([thread_id, prompt])
    else:
        args.extend(["exec", "--json", "--skip-git-repo-check"])
        if model:
            args.extend(["-m", model])
        args.append(prompt)

    return args


async def run_codex(
    prompt: str,
    *,
    chat_dir: str,
    thread_id: str = "",
    model: str = "",
    effort: str = "",
    codex_path: str = "/usr/local/bin/codex",
    timeout: float | None = None,
    inactivity_timeout: float | None = DEFAULT_INACTIVITY_TIMEOUT_S,
    on_event: "Callable[[dict], Awaitable[None]] | None" = None,
    stop_event: asyncio.Event | None = None,
) -> CodexResult:
    """Run codex exec (or resume) with JSON streaming output."""
    _reap_zombies()
    resolved_path = find_codex_bin(codex_path)
    if not resolved_path and not os.path.exists(codex_path):
        ok, _ = await install_codex()
        if ok:
            resolved_path = find_codex_bin()
    if not resolved_path and not os.path.exists(codex_path):
        return CodexResult(
            text="⚠️ Codex CLI не найден на сервере. Установите его через меню /settings или командой <code>npm install -g @openai/codex</code>.",
            exit_code=127,
            stderr="Codex CLI binary not found",
        )
    target_bin = resolved_path or codex_path
    args = _build_codex_args(
        codex_path=target_bin,
        prompt=prompt,
        thread_id=thread_id,
        model=model,
        effort=effort,
        chat_dir=chat_dir,
    )

    env = os.environ.copy()
    try:
        from src.codex_accounts import read_active_auth_data
        active_auth = read_active_auth_data()
        if active_auth and active_auth.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = str(active_auth["OPENAI_BASE_URL"])
        elif active_auth and active_auth.get("base_url"):
            env["OPENAI_BASE_URL"] = str(active_auth["base_url"])
    except Exception:
        pass
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=chat_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=2 * 1024 * 1024,  # 2MB buffer
            start_new_session=True,  # Ensure codex runs in its own process group
        )
    except Exception as e:
        return CodexResult(text="", exit_code=-1, stderr=str(e))

    observed_thread_id = thread_id
    message_chunks: list[str] = []
    stderr_chunks: list[bytes] = []
    codex_error_message = ""
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    reasoning_output_tokens = 0

    tool_active = False
    last_activity_time = asyncio.get_event_loop().time()

    def record_activity():
        nonlocal last_activity_time
        last_activity_time = asyncio.get_event_loop().time()

    async def _terminate_process_group(sig_first: int = signal.SIGTERM, grace_s: float = 1.5) -> None:
        if process.returncode is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, sig_first)
        except OSError:
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass

    async def read_stdout():
        nonlocal observed_thread_id, tool_active, input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens, codex_error_message
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            record_activity()
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            ev_type = data.get("type", "")

            if ev_type == "thread.started":
                tid = data.get("thread_id")
                if tid:
                    observed_thread_id = tid
                    if on_event:
                        await on_event({"event": "init", "conversation_id": tid})

            elif ev_type == "error":
                err_msg = str(data.get("message") or "").strip()
                if err_msg:
                    codex_error_message = err_msg

            elif ev_type == "turn.failed":
                err_obj = data.get("error", {})
                err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                if err_msg:
                    codex_error_message = str(err_msg).strip()

            elif ev_type == "item.started":
                item = data.get("item", {})
                itype = item.get("type")
                if itype == "command_execution":
                    tool_active = True
                    cmd = item.get("command", "")
                    if on_event:
                        await on_event({
                            "event": "step_update",
                            "step_update": {
                                "step_type": "tool",
                                "state": "ACTIVE",
                                "tool_name": "run_command",
                                "tool_info": {"parameters": {"CommandLine": cmd, "toolSummary": cmd[:50]}},
                            },
                        })

            elif ev_type == "item.completed":
                item = data.get("item", {})
                itype = item.get("type")
                if itype == "command_execution":
                    tool_active = False
                    record_activity()
                elif itype == "agent_message":
                    text = item.get("text", "")
                    if text:
                        message_chunks.append(text)
                        if on_event:
                            await on_event({
                                "event": "step_update",
                                "step_update": {
                                    "step_type": "thought",
                                    "text": text,
                                    "state": "ACTIVE",
                                },
                            })
                elif itype == "error":
                    err_msg = str(item.get("message") or "").strip()
                    if err_msg:
                        codex_error_message = err_msg

            elif ev_type == "turn.completed":
                tool_active = False
                record_activity()
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                cached_input_tokens = usage.get("cached_input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                reasoning_output_tokens = usage.get("reasoning_output_tokens", 0)

    async def read_stderr():
        assert process.stderr is not None
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            record_activity()
            stderr_chunks.append(line)

    activity_event = asyncio.Event()

    async def inactivity_watchdog():
        if inactivity_timeout is None or inactivity_timeout <= 0:
            return
        while True:
            await asyncio.sleep(0.5)
            if tool_active:
                continue
            now = asyncio.get_event_loop().time()
            elapsed = now - last_activity_time
            remaining = inactivity_timeout - elapsed
            if remaining <= 0:
                activity_event.set()
                break
            await asyncio.sleep(min(remaining, 0.5))

    inactivity_task = asyncio.create_task(activity_event.wait())
    inactivity_watchdog_task = asyncio.create_task(inactivity_watchdog())
    stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None

    async def run_all():
        await asyncio.gather(read_stdout(), read_stderr(), process.wait())

    main_task = asyncio.create_task(run_all())

    wait_tasks = [main_task, inactivity_task]
    if stop_task:
        wait_tasks.append(stop_task)

    try:
        done, pending = await asyncio.wait(
            wait_tasks,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )

        if stop_task and stop_task in done:
            await _terminate_process_group(signal.SIGTERM, grace_s=1.5)
            main_task.cancel()
            await process.wait()
            thoughts, text = _split_thoughts_and_final(message_chunks)
            if text:
                text += "\n\n🛑 [Остановлено пользователем]"
            stderr_out = b"".join(stderr_chunks).decode("utf-8", errors="replace") + "\n[stopped by user]"
            return CodexResult(
                text=text,
                exit_code=130,
                stderr=stderr_out,
                thread_id=observed_thread_id,
                thoughts=thoughts,
            )

        elif inactivity_task in done:
            await _terminate_process_group(signal.SIGTERM, grace_s=1.0)
            main_task.cancel()
            await process.wait()
            thoughts, text = _split_thoughts_and_final(message_chunks)
            msg = f"codex timed out: no status update for {int(inactivity_timeout)}s"
            raw_err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            stderr_out = f"{raw_err}\n{msg}" if raw_err else msg
            return CodexResult(
                text=text,
                exit_code=124,
                stderr=stderr_out,
                thread_id=observed_thread_id,
                thoughts=thoughts,
            )

        elif not done:  # Total timeout expired
            await _terminate_process_group(signal.SIGTERM, grace_s=1.0)
            main_task.cancel()
            await process.wait()
            thoughts, text = _split_thoughts_and_final(message_chunks)
            msg = f"codex timed out after {timeout}s"
            raw_err = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            stderr_out = f"{raw_err}\n{msg}" if raw_err else msg
            return CodexResult(
                text=text,
                exit_code=124,
                stderr=stderr_out,
                thread_id=observed_thread_id,
                thoughts=thoughts,
            )

        else:
            if main_task.exception() is not None:
                raise main_task.exception()

    except (asyncio.CancelledError, BaseException):
        if process.returncode is None:
            await _terminate_process_group(signal.SIGKILL, grace_s=0.5)
            try:
                await process.wait()
            except Exception:
                pass
        raise
    finally:
        inactivity_watchdog_task.cancel()
        inactivity_task.cancel()
        if stop_task:
            stop_task.cancel()
        if process.returncode is None:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        gc.collect()

    stderr_out = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    thoughts, final_text = _split_thoughts_and_final(message_chunks)

    err_text = final_text
    if not err_text and (process.returncode or 0) != 0:
        if codex_error_message:
            low_msg = codex_error_message.lower()
            if "usage limit" in low_msg or "quota" in low_msg or "rate limit" in low_msg:
                err_text = (
                    f"⚠️ <b>Исчерпан лимит использования OpenAI (Usage limit reached)</b>\n\n"
                    f"{html.escape(codex_error_message)}\n\n"
                    "💡 <i>Переключитесь на другую модель (например, Gemini Flash) через /model или смените движок на AGY через /engine agy.</i>"
                )
            else:
                err_text = f"⚠️ Ошибка Codex:\n<code>{html.escape(codex_error_message)}</code>"
        elif stderr_out.strip():
            clean_err = re.sub(r"\x1b\[[0-9;]*[mK]", "", stderr_out).strip()
            err_text = f"⚠️ Ошибка Codex (код {process.returncode}):\n<code>{html.escape(clean_err[:1000])}</code>"

    return CodexResult(
        text=err_text,
        exit_code=process.returncode or 0,
        stderr=stderr_out,
        thread_id=observed_thread_id,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        thoughts=thoughts,
    )


# --- Authentication & Device Auth flow ---

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_DEVICE_URL_RE = re.compile(r"https://auth\.openai\.com/[^\s\x1b]+")
_DEVICE_CODE_RE = re.compile(r"\b([A-Z0-9]{4,5}-[A-Z0-9]{4,5})\b")


async def check_codex_auth(codex_path: str = "/usr/local/bin/codex") -> tuple[bool, str]:
    """Check if codex is authenticated."""
    resolved_path = find_codex_bin(codex_path)
    if not resolved_path and not os.path.exists(codex_path):
        return False, "Codex CLI не установлен в системе"
    target = resolved_path or codex_path
    try:
        proc = await asyncio.create_subprocess_exec(
            target, "login", "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = (stdout.decode("utf-8", errors="replace") + "\n" + stderr.decode("utf-8", errors="replace")).strip()
        if proc.returncode == 0 and "Logged in" in out:
            # Extract just the "Logged in..." line
            for line in out.splitlines():
                if "logged in" in line.lower():
                    return True, line.strip()
            return True, out
        return False, out
    except FileNotFoundError:
        return False, "Codex CLI не установлен в системе"
    except Exception as e:
        return False, str(e)


async def run_codex_device_auth(
    on_code: Callable[[str, str], Awaitable[None]],
    codex_path: str = "/usr/local/bin/codex",
    cancel_event: asyncio.Event | None = None,
    timeout_s: float = 900.0,  # 15 minutes
) -> tuple[bool, str]:
    """Execute codex login --device-auth and invoke on_code(url, code).

    Waits for the user to authenticate in their browser.
    """
    _reap_zombies()
    resolved_path = find_codex_bin(codex_path)
    if not resolved_path and not os.path.exists(codex_path):
        ok, inst_msg = await install_codex()
        if ok:
            resolved_path = find_codex_bin()
        else:
            return False, f"Codex CLI не найден на сервере. Ошибка автоматической установки: {inst_msg}"
    target = resolved_path or codex_path
    try:
        env = dict(os.environ)
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        proc = await asyncio.create_subprocess_exec(
            target, "login", "--device-auth",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError:
        return False, f"Не удалось запустить codex login: [Errno 2] No such file or directory: '{target}'"
    except Exception as e:
        return False, f"Не удалось запустить codex login: {e}"

    captured_url = ""
    captured_code = ""
    code_reported = False

    async def read_lines():
        nonlocal captured_url, captured_code, code_reported
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            raw_text = line.decode("utf-8", errors="replace").strip()
            if not raw_text:
                continue
            text = _ANSI_RE.sub("", raw_text)

            url_match = _DEVICE_URL_RE.search(text)
            if url_match and not captured_url:
                captured_url = url_match.group(0)

            code_match = _DEVICE_CODE_RE.search(text)
            if code_match and not captured_code:
                captured_code = code_match.group(1)

            if captured_url and captured_code and not code_reported:
                code_reported = True
                try:
                    await on_code(captured_url, captured_code)
                except Exception as e:
                    LOG.warning("on_code callback failed: %s", e)

    read_task = asyncio.create_task(read_lines())
    wait_proc = asyncio.create_task(proc.wait())
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None

    wait_list = [read_task, wait_proc]
    if cancel_task:
        wait_list.append(cancel_task)

    try:
        done, _ = await asyncio.wait(wait_list, return_when=asyncio.FIRST_COMPLETED, timeout=timeout_s)

        if cancel_task and cancel_task in done:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            return False, "Авторизация отменена пользователем"

        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            rc = proc.returncode if proc.returncode is not None else 1

        if rc == 0:
            return True, "Авторизация успешно завершена!"
        return False, f"Ошибка авторизации (код {rc})"

        # Timeout
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        return False, "Время ожидания авторизации истекло (15 минут)"
    finally:
        read_task.cancel()
        wait_proc.cancel()
        if cancel_task:
            cancel_task.cancel()


async def queue_codex_message(
    thread_id: str,
    message: str,
    codex_path: str = "/usr/local/bin/codex",
) -> tuple[bool, str]:
    """Queue a message to an active or existing codex thread."""
    resolved_path = find_codex_bin(codex_path)
    if not resolved_path and not os.path.exists(codex_path):
        return False, "Codex CLI не установлен в системе"
    target = resolved_path or codex_path
    try:
        proc = await asyncio.create_subprocess_exec(
            target, "queue", "--thread", thread_id, "--message", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return True, stdout.decode().strip()
        return False, stderr.decode().strip()
    except FileNotFoundError:
        return False, "Codex CLI не установлен в системе"
    except Exception as e:
        return False, str(e)
