from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.routes.dependencies import optional_user
from backend.app.schemas.workspace import InstrumentRequest, OnboardingRequest
from backend.app.services.auth import AuthUser
from backend.app.services.workspace.application import save_instruments, save_onboarding, workspace_payload


router = APIRouter(prefix="/api", tags=["workspace"])


@router.get("/workspace")
def workspace(exchange: Optional[str] = None, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return workspace_payload(user_id=user.id if user else None, exchange=exchange)


@router.post("/onboarding/watchlist")
def create_watchlist(payload: OnboardingRequest, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    try:
        instruments = save_onboarding(
            payload.instruments,
            payload.exchange,
            payload.role.value,
            user_id=user.id if user else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"instruments": instruments, "workspace": workspace_payload(user_id=user.id if user else None)}


@router.put("/instruments")
def update_instruments(payload: InstrumentRequest, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    try:
        instruments = save_instruments(payload.instruments, user_id=user.id if user else None)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"instruments": instruments}
