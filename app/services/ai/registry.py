from __future__ import annotations

import logging
import os
from typing import Any

from app.config import AI_DISCLOSURE, FUTURE_PROVIDER_NOTE, LOCAL_DISCLOSURE
from app.services.ai.models import model_catalog_status, model_recommendations, refresh_model_catalog
from app.services.ai.providers import (
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDER_QWEN,
    AIProvider,
    LocalRulesProvider,
    OpenAIProvider,
    ProviderRequest,
    ProviderResult,
    QwenProvider,
)

__all__ = [
    "ProviderRequest",
    "ProviderResult",
    "generate_structured_json",
    "provider_chain_status",
    "provider_order",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROVIDERS: dict[str, AIProvider] = {
    PROVIDER_OPENAI: OpenAIProvider(),
    PROVIDER_QWEN: QwenProvider(),
    PROVIDER_LOCAL: LocalRulesProvider(),
}
DEFAULT_ORDER = (PROVIDER_OPENAI, PROVIDER_QWEN, PROVIDER_LOCAL)


def provider_order() -> list[str]:
    raw = os.getenv("AI_PROVIDER_ORDER", ",".join(DEFAULT_ORDER))
    order = [item.strip().lower() for item in raw.split(",") if item.strip().lower() in PROVIDERS]
    if PROVIDER_LOCAL not in order:
        order.append(PROVIDER_LOCAL)
    return order or list(DEFAULT_ORDER)


def generate_structured_json(request: ProviderRequest) -> ProviderResult:
    """Try each configured provider in order and return the first usable result.

    The returned result always describes the primary attempt when every
    provider fails, so callers keep the same fallback reasons users already
    see ("OpenAI was not configured on the backend.", timeouts, HTTP errors).
    """

    attempts: list[dict[str, Any]] = []
    primary_failure: ProviderResult | None = None
    for provider_id in provider_order():
        if provider_id == PROVIDER_LOCAL:
            break
        provider = PROVIDERS[provider_id]
        if provider_id == PROVIDER_QWEN and not QwenProvider().enabled():
            attempts.append({"provider": provider_id, "outcome": "skipped", "reason": "reserved provider slot"})
            continue
        result = provider.generate_json(request)
        attempts.append({"provider": provider_id, "outcome": result.status, "reason": result.reason, "model": result.model, "elapsedMs": result.elapsed_ms})
        if result.ok:
            result.attempts = attempts
            return result
        if primary_failure is None:
            primary_failure = result
    if primary_failure is None:
        primary_failure = ProviderResult(status="unavailable", provider=PROVIDER_LOCAL, model="deterministic local rules", reason="No AI provider is enabled; deterministic local rules are in use.", failure_category="configuration")
    primary_failure.attempts = attempts
    return primary_failure


def provider_chain_status(verify_models: bool = False) -> dict[str, Any]:
    if verify_models:
        refresh_model_catalog()
    order = provider_order()
    statuses = [PROVIDERS[provider_id].status() for provider_id in order]
    active = next((status["id"] for status in statuses if status["id"] != PROVIDER_LOCAL and status.get("configured") and status.get("enabled")), PROVIDER_LOCAL)
    return {
        "order": order,
        "activeProvider": active,
        "providers": statuses,
        "models": model_recommendations(),
        "catalog": model_catalog_status(),
        "disclosure": {"ai": AI_DISCLOSURE, "local": LOCAL_DISCLOSURE, "futureProviderNote": FUTURE_PROVIDER_NOTE, "notFinancialAdvice": True},
    }
