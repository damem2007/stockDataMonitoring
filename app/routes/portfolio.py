from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.routes.dependencies import require_user
from backend.app.schemas.roles import InstrumentRole
from backend.app.schemas.workspace import OnboardingRequest
from backend.app.services.auth import AuthUser
from backend.app.services.workspace.application import portfolio_payload, save_onboarding


router = APIRouter(prefix="/api", tags=["portfolio"])


@router.post("/onboarding/portfolio")
def create_portfolio(payload: OnboardingRequest, user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    try:
        instruments = save_onboarding(payload.instruments, payload.exchange, InstrumentRole.trading.value, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"instruments": instruments, "portfolio": portfolio_payload(user_id=user.id)}


@router.get("/portfolio")
def portfolio(user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    return portfolio_payload(user_id=user.id)
