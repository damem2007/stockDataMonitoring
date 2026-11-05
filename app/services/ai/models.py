from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from app.config import OPENAI_CONNECT_TIMEOUT_SECONDS, RECOMMENDED_OPENAI_MODEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_MODELS_ENDPOINT = os.getenv("OPENAI_MODELS_ENDPOINT", "https://api.openai.com/v1/models")
MODEL_CATALOG_TTL_SECONDS = 3600

# Model tiers. "deep-synthesis" tasks produce long, multi-instrument plans and
# need the strongest reasoning model; "interpretation" tasks translate an
# already computed evidence bundle into short structured commentary and can
# use a faster model without losing decision quality. Environment overrides
# always win so a deployment can pin any model for any task.
MODEL_PREFERENCES: dict[str, tuple[str, ...]] = {
    "deep-synthesis": ("gpt-5.6-sol", "gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"),
    "interpretation": ("gpt-5.6-sol", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini", "gpt-4.1"),
}

TASK_TIER: dict[str, str] = {
    "market-explorer": "deep-synthesis",
    "consultant": "deep-synthesis",
    "prediction": "deep-synthesis",
    "instrument": "interpretation",
    "advanced-trading": "interpretation",
    "options": "interpretation",
}

# Task-specific environment variables, then the shared fallbacks they inherit.
TASK_ENV: dict[str, tuple[str, ...]] = {
    "market-explorer": ("OPENAI_MARKET_EXPLORER_MODEL",),
    "consultant": ("OPENAI_MARKET_EXPLORER_MODEL",),
    "instrument": ("OPENAI_INSTRUMENT_MODEL", "OPENAI_MARKET_EXPLORER_MODEL"),
    "advanced-trading": ("OPENAI_ADVANCED_TRADING_MODEL", "OPENAI_INSTRUMENT_MODEL", "OPENAI_MARKET_EXPLORER_MODEL"),
    "prediction": ("OPENAI_PREDICTION_MODEL", "OPENAI_MARKET_EXPLORER_MODEL"),
    "options": ("OPENAI_OPTIONS_MODEL", "OPENAI_PREDICTION_MODEL", "OPENAI_MARKET_EXPLORER_MODEL"),
}

_CATALOG: dict[str, Any] = {"status": "not-checked", "models": [], "fetchedAt": None, "error": None, "checkedAt": 0.0}
_CATALOG_LOCK = threading.Lock()


def task_tier(task: str) -> str:
    return TASK_TIER.get(task, "interpretation")


def configured_model(task: str) -> str | None:
    """Return the explicitly configured model for a task, if any."""

    for name in TASK_ENV.get(task, ()):
        value = os.getenv(name)
        if value:
            return value
    return os.getenv("OPENAI_MODEL") or None


def preferred_models(task: str) -> tuple[str, ...]:
    return MODEL_PREFERENCES.get(task_tier(task), MODEL_PREFERENCES["interpretation"])


def resolve_model(task: str) -> str:
    """Pick the best model for a task.

    Precedence: task environment override -> shared ``OPENAI_MODEL`` -> the
    first preferred model for the task tier that the account can actually use
    (when the model catalogue has been verified) -> the recommended default.
    A configured model that the verified catalogue does not list is replaced
    by the best available preference and logged, so a retired model name never
    silently sends every request to the local fallback.
    """

    explicit = configured_model(task)
    available = set(_CATALOG.get("models") or [])
    if explicit:
        if not available or explicit in available:
            return explicit
        replacement = next((model for model in preferred_models(task) if model in available), None)
        if replacement:
            logger.warning("ai_models task=%s configured_model=%s outcome=unavailable replacement=%s", task, explicit, replacement)
            return replacement
        return explicit
    for model in preferred_models(task):
        if not available or model in available:
            return model
    return RECOMMENDED_OPENAI_MODEL


def model_recommendations() -> dict[str, dict[str, Any]]:
    return {
        task: {
            "tier": task_tier(task),
            "model": resolve_model(task),
            "configured": configured_model(task),
            "preferences": list(preferred_models(task)),
        }
        for task in TASK_ENV
    }


def refresh_model_catalog(force: bool = False, timeout: int | None = None) -> dict[str, Any]:
    """Fetch the account's model list (read-only, cached for an hour)."""

    api_key = os.getenv("OPENAI_API_KEY")
    with _CATALOG_LOCK:
        if not api_key:
            _CATALOG.update({"status": "unconfigured", "models": [], "error": "OPENAI_API_KEY is not configured.", "checkedAt": time.time()})
            return dict(_CATALOG)
        fresh = time.time() - float(_CATALOG.get("checkedAt") or 0.0) < MODEL_CATALOG_TTL_SECONDS
        if fresh and not force and _CATALOG.get("status") == "ok":
            return dict(_CATALOG)
        try:
            response = requests.get(
                OPENAI_MODELS_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout or OPENAI_CONNECT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            rows = response.json().get("data", [])
            models = sorted({str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")})
            _CATALOG.update({"status": "ok", "models": models, "fetchedAt": datetime.now(timezone.utc).isoformat(), "error": None, "checkedAt": time.time()})
            logger.info("ai_models outcome=catalog_refreshed model_count=%s", len(models))
        except requests.RequestException as error:
            _CATALOG.update({"status": "error", "error": f"Model catalogue check failed: {type(error).__name__}.", "checkedAt": time.time()})
            logger.warning("ai_models outcome=catalog_error error_type=%s", type(error).__name__)
        except ValueError as error:
            _CATALOG.update({"status": "error", "error": f"Model catalogue response could not be parsed: {type(error).__name__}.", "checkedAt": time.time()})
        return dict(_CATALOG)


def model_catalog_status() -> dict[str, Any]:
    with _CATALOG_LOCK:
        recommended = {task: resolve_model(task) for task in TASK_ENV}
        available = set(_CATALOG.get("models") or [])
        return {
            "status": _CATALOG.get("status"),
            "fetchedAt": _CATALOG.get("fetchedAt"),
            "error": _CATALOG.get("error"),
            "modelCount": len(available),
            "verifiedModels": {task: (model in available) if available else None for task, model in recommended.items()},
        }


def start_background_catalog_refresh() -> None:
    """Verify the configured models once at startup without blocking requests."""

    if os.getenv("OPENAI_VERIFY_MODELS_ON_STARTUP", "true").strip().lower() == "false":
        return
    import sys

    if "pytest" in sys.modules:
        return
    if not os.getenv("OPENAI_API_KEY"):
        return
    thread = threading.Thread(target=refresh_model_catalog, kwargs={"force": True}, name="openai-model-catalog", daemon=True)
    thread.start()
