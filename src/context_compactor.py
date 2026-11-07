"""Context compaction and cross-engine handoff layer.

Optimizes token usage when switching between engines (agy <-> codex) or
when the user explicitly compacts a chat session (/compact).
Instead of passing bloated raw transcript dumps (50k-100k+ tokens),
this layer synthesizes a high-density structured summary (~400-800 tokens).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import ChatState

LOG = logging.getLogger("antigravity_telegram_bridge")

HISTORY_FILENAME = ".bridge_history.jsonl"
MAX_HISTORY_ENTRIES = 16


def append_chat_history(
    chat_dir: str,
    user_prompt: str,
    assistant_reply: str,
    engine: str = "agy",
    model: str = "",
) -> None:
    """Append a completed turn to the per-chat JSONL conversation log."""
    if not chat_dir or not user_prompt.strip():
        return
    # Strip any handoff digest prefix from the stored user prompt so it doesn't snowball
    clean_user = user_prompt.strip()
    if "[Запрос пользователя]:\n" in clean_user:
        clean_user = clean_user.split("[Запрос пользователя]:\n", 1)[1].strip()
    clean_reply = (assistant_reply or "").strip()
    # Strip trailing token stats footer (e.g. 🌐 [CheapAI: ...] or 📊 [Codex: ...])
    for footer_marker in ("\n\n🌐 <i>[", "\n\n📊 <i>["):
        if footer_marker in clean_reply:
            clean_reply = clean_reply.split(footer_marker, 1)[0].strip()
    if not clean_reply:
        return

    path = Path(chat_dir) / HISTORY_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.append(json.loads(line))
                except Exception:
                    continue
        existing.append({
            "user": clean_user[:4000],
            "assistant": clean_reply[:6000],
            "engine": engine,
            "model": model,
        })
        existing = existing[-MAX_HISTORY_ENTRIES:]
        content = "\n".join(json.dumps(item, ensure_ascii=False) for item in existing) + "\n"
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        LOG.debug("Failed to append chat history in %s: %s", chat_dir, e)


def load_chat_history(chat_dir: str, limit: int = 8) -> list[dict[str, str]]:
    """Load recent turns from the per-chat JSONL conversation log."""
    if not chat_dir:
        return []
    path = Path(chat_dir) / HISTORY_FILENAME
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("user") and obj.get("assistant"):
                    entries.append(obj)
            except Exception:
                continue
    except Exception:
        return []
    return entries[-limit:]


def clear_chat_history(chat_dir: str) -> None:
    """Clear the per-chat JSONL conversation log (e.g. on /reset or /compact)."""
    if not chat_dir:
        return
    path = Path(chat_dir) / HISTORY_FILENAME
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def get_agy_transcript_recent_turns(conversation_id: str, limit: int = 4) -> list[dict[str, str]]:
    """Read recent user/assistant turns directly from agy's transcript.jsonl if available."""
    if not conversation_id:
        return []
    transcript_path = Path(os.path.expanduser(
        f"~/.gemini/antigravity-cli/brain/{conversation_id}/.system_generated/logs/transcript.jsonl"
    ))
    if not transcript_path.exists():
        return []
    turns: list[dict[str, str]] = []
    cur_user = ""
    try:
        lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw in lines[-200:]:
            try:
                step = json.loads(raw)
            except Exception:
                continue
            stype = step.get("type")
            content = str(step.get("content") or "").strip()
            if (stype == "USER_INPUT" or (stype == "GENERIC" and "<USER_REQUEST>" in content)) and content:
                if "<USER_REQUEST>" in content and "</USER_REQUEST>" in content:
                    content = content.split("<USER_REQUEST>", 1)[1].split("</USER_REQUEST>", 1)[0].strip()
                if "\n\n[Please respond in Russian" in content:
                    content = content.split("\n\n[Please respond in Russian", 1)[0].strip()
                cur_user = content
            elif stype == "PLANNER_RESPONSE" and content and not step.get("tool_calls"):
                if cur_user:
                    turns.append({"user": cur_user[:2000], "assistant": content[:3000]})
                    cur_user = ""
    except Exception as e:
        LOG.debug("Failed to read agy transcript turns: %s", e)
    return turns[-limit:]


def get_openai_history_messages(
    chat_dir: str,
    conversation_id: str = "",
    limit: int = 6,
) -> list[dict[str, str]]:
    """Return OpenAI-format [{'role': 'user', ...}, {'role': 'assistant', ...}] history messages."""
    entries = load_chat_history(chat_dir, limit=limit)
    if len(entries) < limit and conversation_id:
        agy_turns = get_agy_transcript_recent_turns(conversation_id, limit=limit)
        if agy_turns:
            seen_users = {str(it.get("user") or "").strip()[:120] for it in entries}
            merged = [t for t in agy_turns if str(t.get("user") or "").strip()[:120] not in seen_users] + entries
            entries = merged[-limit:]
    msgs: list[dict[str, str]] = []
    for item in entries:
        u = str(item.get("user") or "").strip()
        a = str(item.get("assistant") or "").strip()
        if u and a:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
    return msgs


def get_git_changed_files(chat_dir: str) -> list[str]:
    """Return list of modified or untracked files in the workspace, if git repo."""
    try:
        res = subprocess.run(
            ["git", "status", "--short"],
            cwd=chat_dir,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            lines = res.stdout.strip().splitlines()
            return [line.strip() for line in lines[:10]]
    except Exception:
        pass
    return []


def get_agy_session_summary(conversation_id: str) -> str:
    """Fetch high-level preview/title from agy's conversation_summaries.db."""
    if not conversation_id:
        return ""
    db_path = os.path.expanduser("~/.gemini/antigravity-cli/conversation_summaries.db")
    if not os.path.exists(db_path):
        return ""
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT title, preview FROM conversation_summaries WHERE conversation_id = ? LIMIT 1",
                (conversation_id,),
            )
            row = c.fetchone()
            if row:
                title, preview = row
                parts = []
                if title:
                    parts.append(title.strip())
                if preview and preview != title:
                    parts.append(preview.strip())
                return " — ".join(parts)
    except Exception as e:
        LOG.debug("Failed to query agy conversation_summaries: %s", e)
    return ""


def get_codex_session_summary(thread_id: str) -> str:
    """Fetch recent agent summary/items from codex thread_history_1.sqlite."""
    if not thread_id:
        return ""
    db_path = os.path.expanduser("~/.codex/thread_history_1.sqlite")
    if not os.path.exists(db_path):
        return ""
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT item_json FROM thread_items WHERE thread_id = ? AND item_type = 'agent_message' ORDER BY rowid DESC LIMIT 2",
                (thread_id,),
            )
            rows = c.fetchall()
            summaries = []
            for (raw_json,) in rows:
                try:
                    data = json.loads(raw_json)
                    text = data.get("text", "").strip()
                    if text:
                        # Extract first 200 chars
                        snip = text[:200] + "..." if len(text) > 200 else text
                        summaries.append(snip)
                except Exception:
                    continue
            if summaries:
                return " | ".join(reversed(summaries))
    except Exception as e:
        LOG.debug("Failed to query codex thread_history: %s", e)
    return ""


def build_handoff_digest(
    chat_dir: str,
    from_engine: str,
    to_engine: str,
    conversation_id: str = "",
    codex_thread_id: str = "",
) -> str:
    """Build a compact (~400-600 tokens) cross-engine handoff memorandum."""
    lines = [
        f"[Контекст: переключение движка с {from_engine.upper()} на {to_engine.upper()}]",
        "Для оптимизации токенов сырая история очищена. Ключевое состояние проекта:",
    ]

    # 1. Summary from previous engine
    prev_summary = ""
    if from_engine == "agy":
        prev_summary = get_agy_session_summary(conversation_id)
    elif from_engine == "codex":
        prev_summary = get_codex_session_summary(codex_thread_id)
    elif from_engine in ("openai", "openai_compat"):
        prev_summary = "OpenAI-compatible turn"

    if prev_summary:
        lines.append(f"• Текущий контекст/задача: {prev_summary}")

    # 1b. Recent conversation turns from local history or transcript
    recent_turns = load_chat_history(chat_dir, limit=4)
    if conversation_id and (not recent_turns or from_engine == "agy"):
        agy_turns = get_agy_transcript_recent_turns(conversation_id, limit=4)
        if agy_turns:
            seen_users = {str(it.get("user") or "").strip()[:120] for it in recent_turns}
            merged = [t for t in agy_turns if str(t.get("user") or "").strip()[:120] not in seen_users] + recent_turns
            recent_turns = merged[-4:]
    if recent_turns:
        lines.append("• Недавние сообщения в диалоге (из предыдущего движка):")
        for rt in recent_turns:
            u_snip = str(rt.get("user") or "").strip().replace("\n", " ")[:400]
            a_snip = str(rt.get("assistant") or "").strip().replace("\n", " ")[:600]
            lines.append(f"  - Пользователь: {u_snip}")
            lines.append(f"  - Ассистент: {a_snip}")

    # 2. Changed files in workspace
    changed_files = get_git_changed_files(chat_dir)
    if changed_files:
        lines.append("• Измененные/созданные файлы в репозитории:")
        for cf in changed_files[:6]:
            lines.append(f"  {cf}")

    lines.append("Продолжай выполнение задачи с учетом вышеизложенного контекста.")
    return "\n".join(lines)


def prepare_turn_prompt(
    user_prompt: str,
    cs: "ChatState",
    target_engine: str,
) -> str:
    """Prefix user prompt with handoff digest whenever switching between engines."""
    prev_engine = getattr(cs, "last_turn_engine", "")
    if not prev_engine:
        # Fallback for chats that haven't recorded last_turn_engine yet:
        # check the last entry in .bridge_history.jsonl, or cs.engine, or infer from active session IDs
        hist = load_chat_history(getattr(cs, "chat_dir", ""), limit=1)
        if hist and hist[-1].get("engine"):
            prev_engine = str(hist[-1]["engine"])
        elif getattr(cs, "engine", "") and getattr(cs, "engine", "") != target_engine:
            prev_engine = getattr(cs, "engine", "")
        elif target_engine != "agy" and getattr(cs, "conversation_id", ""):
            prev_engine = "agy"
        elif target_engine != "codex" and getattr(cs, "codex_thread_id", ""):
            prev_engine = "codex"
        else:
            prev_engine = getattr(cs, "engine", "") or "agy"

    # Normalize engine names
    norm_prev = "openai" if prev_engine == "openai_compat" else prev_engine
    norm_target = "openai" if target_engine == "openai_compat" else target_engine

    # Always inject handoff digest when switching from one engine to another,
    # even if the target engine already has an older resumed session!
    # Otherwise a resumed session on the target engine misses all turns that happened on the previous engine.
    if norm_prev and norm_prev != norm_target:
        digest = build_handoff_digest(
            chat_dir=cs.chat_dir,
            from_engine=norm_prev,
            to_engine=norm_target,
            conversation_id=getattr(cs, "conversation_id", ""),
            codex_thread_id=getattr(cs, "codex_thread_id", ""),
        )
        return f"{digest}\n\n[Запрос пользователя]:\n{user_prompt}"

    return user_prompt
