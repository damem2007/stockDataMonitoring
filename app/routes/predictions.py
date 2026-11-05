from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import REFRESH_INTERVAL_SECONDS
from app.database import database_available, session_scope
from app.models.orm import InstrumentTag
from app.routes.dependencies import require_permission
from app.schemas.roles import InstrumentRole
from app.services.auth import AuthUser
from app.services.prediction.engine import build_prediction, local_interpretation
from app.services.prediction.provider import interpret_prediction, list_diagnostics
from app.services.workspace.application import currency_for_symbol, market_for_symbol
from app.services.workspace.watchlist_store import read_instrument_tags


router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/config")
def prediction_config(user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, int]:
    return {"refreshIntervalSeconds": REFRESH_INTERVAL_SECONDS}


@router.get("/portfolio/instruments")
def prediction_portfolio_instruments(user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, Any]:
    rows = read_instrument_tags(user_id=user.id)
    instruments = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("role") or "") != "Trading" or row.get("active") is False:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        instruments.append({"symbol": symbol, "market": market_for_symbol(symbol), "currency": currency_for_symbol(symbol), "category": row.get("category") or "Unassigned", "intent": row.get("intent") or "Hold / Watch", "strategy": row.get("strategy") or "Buy-dip"})
    return {"instruments": instruments}


@router.post("/{symbol}/run")
def run_prediction(symbol: str, payload: Optional[dict[str, Any]] = None, user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A symbol is required.")
    if not database_available():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Portfolio membership could not be verified right now.")
    try:
        with session_scope() as session:
            instrument = session.execute(
                select(InstrumentTag)
                .where(
                    InstrumentTag.user_id == UUID(str(user.id)),
                    InstrumentTag.symbol == normalized,
                    InstrumentTag.role == InstrumentRole.trading.value,
                    InstrumentTag.active.is_(True),
                )
                .limit(1)
            ).scalar_one_or_none()
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Portfolio membership could not be verified right now.") from exc
    if instrument is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{normalized} is not in your portfolio. Add it to your portfolio first.")
    body = payload if isinstance(payload, dict) else {}
    prediction = build_prediction(
        normalized,
        refresh=bool(body.get("refresh", False)),
        horizons=body.get("horizons"),
        intent=str(instrument.intent or "Hold / Watch"),
        strategy=str(instrument.strategy or "Buy-dip"),
        risk_profile=str(body.get("riskProfile") or "Balanced"),
        account_size=_optional_float(body.get("accountSize")),
    )
    if not prediction.get("ok"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=prediction.get("error") or "Prediction evidence was unavailable.")
    provider = interpret_prediction(prediction, user.id)
    if provider.get("status") == "ready":
        prediction["provider"] = {**prediction.get("provider", {}), "interpretation": "openai", "model": provider.get("model"), "diagnosticId": None}
        prediction["interpretation"] = provider.get("interpretation")
    else:
        prediction["provider"] = {**prediction.get("provider", {}), "interpretation": "local-rules", "model": "deterministic prediction fallback", "diagnosticId": provider.get("diagnosticId"), "failureReason": provider.get("reason")}
        prediction["interpretation"] = local_interpretation(prediction)
    prediction["providerDiagnostic"] = provider if provider.get("status") != "ready" else None
    return prediction


@router.get("/diagnostics")
def prediction_diagnostics(symbol: Optional[str] = Query(default=None), user: AuthUser = Depends(require_permission("view_portfolio"))) -> dict[str, Any]:
    return {"diagnostics": list_diagnostics(user.id, symbol)}


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
