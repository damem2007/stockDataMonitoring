from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status

from backend.app.routes.dependencies import optional_user
from backend.app.schemas.workspace import InstrumentRequest, OnboardingRequest
from backend.app.services.auth import AuthUser
from backend.app.services.workspace.application import save_instruments, save_onboarding, workspace_payload
from backend.app.services.workspace.application import portfolio_payload
from backend.app.services.workspace.refresh_service import register_guest_symbols
from backend.app.services.workspace.watchlist_store import active_symbols


router = APIRouter(prefix="/api", tags=["workspace"])


@router.get("/workspace")
def workspace(exchange: Optional[str] = None, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return workspace_payload(user_id=user.id if user else None, exchange=exchange)


@router.post("/workspace/sync")
def sync_workspace(
    payload: Optional[InstrumentRequest] = Body(default=None),
    user: Optional[AuthUser] = Depends(optional_user),
) -> dict[str, Any]:
    signed_in_user_id = user.id if user and not user.is_guest else None
    scoped_instruments = payload.instruments if payload else None
    if scoped_instruments is not None and signed_in_user_id is None:
        scoped_instruments = [{**row, "role": "Watching"} for row in scoped_instruments]
        register_guest_symbols(active_symbols(scoped_instruments))
    data = portfolio_payload(instruments=scoped_instruments, user_id=signed_in_user_id, refresh=True)
    return {"synced": data}


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
