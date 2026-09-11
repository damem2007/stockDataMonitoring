from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.app.database import database_available, session_scope
from backend.app.models.orm import InstrumentTag
from backend.app.routes.dependencies import require_permission
from backend.app.schemas.roles import InstrumentRole
from backend.app.schemas.workspace import HoldingActivityRequest, HoldingCloseRequest, OnboardingRequest, PortfolioNameRequest
from backend.app.services.auth import AuthUser
from backend.app.services.workspace.application import (
    add_portfolio_activity,
    close_portfolio_holding,
    list_portfolio_activities,
    portfolio_payload,
    rename_default_portfolio,
    save_onboarding,
)


router = APIRouter(prefix="/api", tags=["portfolio"])


@router.post("/onboarding/portfolio")
def create_portfolio(payload: OnboardingRequest, user: AuthUser = Depends(require_permission("create_portfolio"))) -> dict[str, Any]:
    try:
        instruments = save_onboarding(payload.instruments, payload.exchange, InstrumentRole.trading.value, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"instruments": instruments, "portfolio": portfolio_payload(user_id=user.id)}


@router.get("/portfolio")
def portfolio(
    refresh: bool = Query(False),
    user: AuthUser = Depends(require_permission("view_portfolio")),
) -> dict[str, Any]:
    return portfolio_payload(user_id=user.id, refresh=refresh)


@router.get("/portfolio/instruments/{symbol}")
def portfolio_instrument_membership(symbol: str, user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    if not database_available():
        return {"symbol": normalized, "inPortfolio": False}
    try:
        with session_scope() as session:
            registered = session.execute(
                select(InstrumentTag.id)
                .where(
                    InstrumentTag.user_id == UUID(str(user.id)),
                    InstrumentTag.symbol == normalized,
                    InstrumentTag.role == InstrumentRole.trading.value,
                    InstrumentTag.active.is_(True),
                )
                .limit(1)
            ).first() is not None
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Portfolio membership could not be verified right now.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Portfolio membership could not be verified right now.") from exc
    return {"symbol": normalized, "inPortfolio": registered}


@router.patch("/portfolio")
def update_portfolio(payload: PortfolioNameRequest, user: AuthUser = Depends(require_permission("edit_portfolio"))) -> dict[str, Any]:
    try:
        account = rename_default_portfolio(user.id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"account": account, "portfolio": portfolio_payload(user_id=user.id)}


@router.get("/portfolio/activities")
def portfolio_activities(symbol: Optional[str] = None, user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, Any]:
    return {"activities": list_portfolio_activities(user.id, symbol=symbol)}


@router.post("/portfolio/holdings")
def add_holding(payload: HoldingActivityRequest, user: AuthUser = Depends(require_permission("edit_portfolio"))) -> dict[str, Any]:
    row = payload.dict()
    row["average_cost"] = row.get("average_cost") or row.get("price")
    row["purchase_date"] = row.get("purchase_date") or row.get("trade_date")
    try:
        activity = add_portfolio_activity(user.id, row, activity_type="BUY")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"activity": activity, "portfolio": portfolio_payload(user_id=user.id)}


@router.delete("/portfolio/holdings/{symbol}")
def remove_holding(symbol: str, payload: HoldingCloseRequest, user: AuthUser = Depends(require_permission("edit_portfolio"))) -> dict[str, Any]:
    try:
        activity = close_portfolio_holding(user.id, symbol, payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"activity": activity, "portfolio": portfolio_payload(user_id=user.id)}
