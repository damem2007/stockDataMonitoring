from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.routes.dependencies import optional_user, require_user
from app.services.analysis.market_explorer import (
    delete_saved_market_explorer_analysis,
    list_saved_market_explorer_analyses,
    market_explorer_flow_payload,
    market_explorer_recommendation_payload,
    refresh_saved_market_explorer_analysis,
    save_market_explorer_analysis,
)
from app.services.analysis.consultant_jobs import job_snapshot, start_consultant_job
from app.services.analysis.instrument_ai import enhance_instrument_analysis
from app.services.analysis.advanced_trading import build_advanced_trading_indicator
from app.services.auth import AuthUser
from app.services.workspace.application import correlation_payload, focused_analysis_payload


router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/instruments/{symbol}/analysis")
def instrument_analysis(
    symbol: str,
    period: str = Query("2y", pattern="^(5d|1mo|3mo|6mo|ytd|1y|2y|5y|10y)$"),
    interval: str = Query("1d", pattern="^(1m|2m|5m|15m|30m|60m|90m|1h|1d|1wk|1mo)$"),
    refresh: bool = False,
    intent: str = "Hold / Watch",
    strategy: str = "Buy-dip",
    risk_profile: str = "Balanced",
    ml_model: str = "logistic",
) -> dict[str, Any]:
    return focused_analysis_payload(
        symbol,
        period=period,
        interval=interval,
        refresh=refresh,
        intent=intent,
        strategy=strategy,
        risk_profile_name=risk_profile,
        ml_model=ml_model,
    )


@router.post("/instruments/{symbol}/ai-analysis")
def instrument_ai_analysis(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return non-blocking AI interpretation for an already loaded local analysis."""

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A local analysis payload is required.")
    portfolio_context = payload.get("portfolioContext") if isinstance(payload.get("portfolioContext"), dict) else None
    portfolio_rows = payload.get("portfolioRows") if isinstance(payload.get("portfolioRows"), list) else []
    analysis = {**analysis, "symbol": str(analysis.get("symbol") or symbol).upper()}
    return enhance_instrument_analysis(analysis, portfolio_context, portfolio_rows)


@router.post("/instruments/{symbol}/advanced-trading-indicator")
def advanced_trading_indicator(symbol: str, payload: dict[str, Any], user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    """Return the independent Advanced Trading chart overlay."""

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A local analysis payload is required.")
    return build_advanced_trading_indicator(symbol, analysis)


@router.get("/correlation")
def correlation(symbols: str, period: str = "2y", interval: str = "1d") -> dict[str, Any]:
    parsed = [item.strip().upper() for item in symbols.replace("\n", ",").split(",") if item.strip()]
    return correlation_payload(parsed, period=period, interval=interval)


@router.get("/ai-market-explorer/flow")
def ai_market_explorer_flow() -> dict[str, Any]:
    return market_explorer_flow_payload()


@router.post("/ai-market-explorer/analyze")
def ai_market_explorer_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    return market_explorer_recommendation_payload(payload)


@router.post("/ai-market-explorer/jobs")
def start_ai_market_explorer_job(payload: dict[str, Any], user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    """Start the consultant analysis as a live job and return its first snapshot."""

    return start_consultant_job(payload, user.id if user and not user.is_guest else None)


@router.get("/ai-market-explorer/jobs/{job_id}")
def ai_market_explorer_job(job_id: str, include_result: bool = True, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    snapshot = job_snapshot(job_id, user.id if user and not user.is_guest else None, include_result=include_result)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job was not found or has expired.")
    return snapshot


@router.get("/ai-market-explorer/saved")
def saved_market_explorer_analyses(user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    return {"analyses": list_saved_market_explorer_analyses(user.id)}


@router.post("/ai-market-explorer/saved")
def save_market_explorer(payload: dict[str, Any], user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    profile = payload.get("profile")
    result = payload.get("result")
    if not isinstance(profile, dict) or not isinstance(result, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A profile and result are required.")
    return {"analysis": save_market_explorer_analysis(user.id, profile, result, payload.get("title"))}


@router.delete("/ai-market-explorer/saved")
def delete_all_saved_market_explorer(user: AuthUser = Depends(require_user)) -> dict[str, int]:
    return {"deleted": delete_saved_market_explorer_analysis(user.id)}


@router.delete("/ai-market-explorer/saved/{analysis_id}")
def delete_saved_market_explorer(analysis_id: str, user: AuthUser = Depends(require_user)) -> dict[str, int]:
    try:
        deleted = delete_saved_market_explorer_analysis(user.id, analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved analysis was not found.") from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved analysis was not found.")
    return {"deleted": deleted}


@router.post("/ai-market-explorer/saved/{analysis_id}/refresh")
def refresh_saved_market_explorer(analysis_id: str, user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    try:
        analysis = refresh_saved_market_explorer_analysis(user.id, analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved analysis was not found.") from exc
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved analysis was not found.")
    return {"analysis": analysis}
