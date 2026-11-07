"""Dynamic model discovery from agy CLI and Codex CLI."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time

from src.state import is_valid_model

LOG = logging.getLogger("antigravity_telegram_bridge")

FALLBACK_MODELS: list[tuple[str, str]] = [
    ("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
    ("gemini-4-argon-high", "Gemini 4 Argon (High)"),
    ("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
    ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
    ("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"),
    ("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
    ("claude-opus-5-5-low", "Claude Opus 5.5 (Low)"),
    ("claude-opus-5-5-medium", "Claude Opus 5.5 (Medium)"),
    ("claude-opus-5-5-high", "Claude Opus 5.5 (High)"),
    ("claude-sonnet-5-5-low", "Claude Sonnet 5.5 (Low)"),
    ("claude-sonnet-5-5-medium", "Claude Sonnet 5.5 (Medium)"),
    ("claude-sonnet-5-5-high", "Claude Sonnet 5.5 (High)"),
    ("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"),
]

FALLBACK_CODEX_MODELS: list[tuple[str, str]] = [
    ("gpt-6.1-sol", "GPT-6.1 Sol (Main)"),
    ("gpt-6-astra", "GPT-6 Astra (Deep Reasoning)"),
    ("gpt-6.1-terra", "GPT-6.1 Terra (Balanced)"),
    ("gpt-6-sol", "GPT-6 Sol (Fast)"),
    ("gpt-6-luna", "GPT-6 Luna (Ultra Fast)"),
]

_CACHE_TTL_S = 300.0  # 5 minutes
_cached_models: list[tuple[str, str]] | None = None
_cached_at: float = 0.0

_cached_codex_models: list[tuple[str, str]] | None = None
_cached_codex_at: float = 0.0

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_model_effort(model_id: str, display_name: str = "") -> tuple[str, str]:
    """Strip effort suffix (-low, -medium, -high, -xhigh, -max) from model id and display name."""
    efforts = ("-low", "-medium", "-high", "-xhigh", "-max")
    clean_id = model_id
    for eff in efforts:
        if clean_id.lower().endswith(eff):
            clean_id = clean_id[:-len(eff)]
            break

    clean_name = display_name or clean_id
    clean_name = re.sub(r"\s*\((Low|Medium|High|XHigh|Max)\)\s*$", "", clean_name, flags=re.IGNORECASE).strip()
    return clean_id, clean_name


def deduplicate_models_by_base(models: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicate model list by removing effort variants and keeping base models."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for mid, mname in models:
        base_id, base_name = strip_model_effort(mid, mname)
        if base_id not in seen:
            seen.add(base_id)
            result.append((base_id, base_name))
    return result


def extract_model_and_effort(model: str, effort: str = "") -> tuple[str, str]:
    """Separate base model and effort suffix if present."""
    efforts = ("-low", "-medium", "-high", "-xhigh", "-max")
    clean_model = model
    eff = effort
    for sfx in efforts:
        if clean_model.lower().endswith(sfx):
            if not eff:
                eff = sfx.lstrip("-")
            clean_model = clean_model[:-len(sfx)]
            break
    return clean_model, eff


def parse_models_output(output: str) -> list[tuple[str, str]]:
    """Parse output from `agy models` into (model_id, display_name) pairs."""
    models: list[tuple[str, str]] = []
    clean = _ANSI_ESCAPE_RE.sub("", output.replace("\r", "\n"))
    for line in clean.splitlines():
        line = line.strip()
        if not line:
            continue
        if "fetching available models" in line.lower():
            idx = line.lower().rfind("fetching available models...")
            if idx != -1:
                line = line[idx + len("fetching available models..."):].strip()
        if not line:
            continue
        parts = re.split(r"\s{2,}|\t+", line, maxsplit=1)
        if len(parts) == 2:
            model_id, display_name = parts[0].strip(), parts[1].strip()
            if is_valid_model(model_id):
                models.append((model_id, display_name))
        elif len(parts) == 1:
            model_id = parts[0].strip()
            if is_valid_model(model_id):
                models.append((model_id, model_id))
    return models


def get_available_models(
    agy_path: str = "agy",
    force_refresh: bool = False,
    dedup_efforts: bool = False,
) -> list[tuple[str, str]]:
    """Fetch available models from `agy models` with TTL cache and fallback."""
    global _cached_models, _cached_at
    now = time.time()
    raw = None
    if not force_refresh and _cached_models is not None and (now - _cached_at) < _CACHE_TTL_S:
        raw = _cached_models
    else:
        try:
            res = subprocess.run(
                [agy_path, "models"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                parsed = parse_models_output(res.stdout)
                if parsed:
                    _cached_models = parsed
                    _cached_at = now
                    raw = parsed
            if raw is None:
                LOG.warning("agy models returned non-zero (%d) or empty: stderr=%s", res.returncode, res.stderr)
        except Exception as e:
            LOG.warning("Failed to run '%s models': %s", agy_path, e)

    if raw is None:
        raw = _cached_models or list(FALLBACK_MODELS)

    if dedup_efforts:
        return deduplicate_models_by_base(raw)
    return raw


def parse_codex_models_output(raw_json: str) -> list[tuple[str, str]]:
    """Parse JSON output from `codex debug models` or ~/.codex/models_cache.json into (slug, display_name) pairs.

    Respects model visibility ('list' only) and sorts by official priority.
    """
    try:
        data = json.loads(raw_json)
        raw_models = data.get("models", [])
        if not isinstance(raw_models, list):
            return []

        def _sort_key(m: dict) -> int:
            p = m.get("priority")
            return p if isinstance(p, int) else 999

        models_sorted = sorted(raw_models, key=_sort_key)
        items: list[tuple[str, str]] = []
        seen: set[str] = set()

        for m in models_sorted:
            if not isinstance(m, dict):
                continue
            slug = str(m.get("slug") or "").strip()
            if not slug:
                continue

            # Skip models hidden by Codex catalog
            visibility = m.get("visibility", "list")
            if visibility == "hide":
                continue
            if slug in ("codex-auto-review", "gpt-reserve"):
                continue

            slug_lower = slug.lower()
            if slug_lower in seen:
                continue
            seen.add(slug_lower)

            name = str(m.get("display_name") or slug).strip()
            items.append((slug, name))

        return items
    except Exception as exc:
        LOG.warning("Failed to parse codex models JSON: %s", exc)
        return []


def get_available_codex_models(
    codex_path: str = "/usr/local/bin/codex",
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Fetch live available models directly from Codex CLI (`codex debug models`),
    falling back to ~/.codex/models_cache.json or static fallback if CLI is not yet installed.
    """
    global _cached_codex_models, _cached_codex_at
    now = time.time()
    if not force_refresh and _cached_codex_models is not None and (now - _cached_codex_at) < _CACHE_TTL_S:
        return _cached_codex_models

    # 1. Primary source: query live Codex CLI directly
    try:
        from src.codex_runner import find_codex_bin
        bin_path = find_codex_bin(codex_path)
        if bin_path:
            res = subprocess.run(
                [bin_path, "debug", "models"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                parsed = parse_codex_models_output(res.stdout)
                if parsed:
                    _cached_codex_models = parsed
                    _cached_codex_at = now
                    return parsed
    except Exception as exc:
        LOG.debug("Direct `codex debug models` execution failed: %s", exc)

    # 2. Secondary source: read cached models file if CLI didn't run
    cache_file = os.path.expanduser("~/.codex/models_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                parsed = parse_codex_models_output(f.read())
            if parsed:
                _cached_codex_models = parsed
                _cached_codex_at = now
                return parsed
        except Exception as exc:
            LOG.warning("Failed to read ~/.codex/models_cache.json: %s", exc)

    # 3. Fallback if codex is not installed and cache is missing
    if _cached_codex_models:
        return _cached_codex_models
    return list(FALLBACK_CODEX_MODELS)


def get_available_openai_models() -> list[tuple[str, str]]:
    """Return models from the active OpenAI-compatible provider."""
    try:
        from src.openai_providers import get_active_provider
        prov = get_active_provider()
        if not prov:
            return []
        res = []
        if prov.default_model:
            res.append((prov.default_model, f"{prov.default_model} ({prov.name})"))
        for m in prov.models:
            if m != prov.default_model:
                res.append((m, f"{m} ({prov.name})"))
        return res
    except Exception:
        return []


def get_engine_for_model(model: str) -> str:
    """Return 'codex', 'openai', or 'agy' depending on the model name."""
    if not model:
        return "agy"
    m = model.lower()
    if m.startswith(("openai/", "cheapai/", "ceapai/", "openrouter/", "deepseek/")):
        return "openai"
    try:
        from src.openai_providers import get_active_provider
        prov = get_active_provider()
        if prov:
            if prov.default_model and m == prov.default_model.lower():
                return "openai"
            if any(m == pm.lower() for pm in prov.models):
                return "openai"
    except Exception:
        pass

    try:
        from src.models_config import load_models_config
        mcfg = load_models_config()
        base_m, _ = strip_model_effort(m)
        for prov_name, custom_items in mcfg.custom_models.items():
            if any(m == str(it.get("id", "")).lower() or base_m == str(it.get("id", "")).lower() for it in custom_items):
                return prov_name
    except Exception:
        pass

    if m.startswith(("gpt-", "codex-", "o1", "o3", "o4")):
        if "gpt-oss" in m:
            return "agy"
        return "codex"
    return "agy"
