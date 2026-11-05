from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services.ai.models import refresh_model_catalog
from app.services.ai.registry import provider_chain_status


router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/providers")
def ai_providers(verify: bool = Query(False, description="Refresh the provider model catalogue before reporting.")) -> dict[str, Any]:
    """Describe the interpretation provider chain the UI should disclose.

    The payload names the active provider and model, the reserved in-house
    (Qwen) slot, and the deterministic local fallback so every screen can show
    the same "AI in use / local fallback" markers.
    """

    return provider_chain_status(verify_models=verify)


@router.post("/providers/verify")
def verify_ai_models() -> dict[str, Any]:
    """Force a read-only refresh of the account's model catalogue."""

    refresh_model_catalog(force=True)
    return provider_chain_status()
