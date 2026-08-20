from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from backend.app.services.workspace.application import correlation_payload, focused_analysis_payload


router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/instruments/{symbol}/analysis")
def instrument_analysis(
    symbol: str,
    period: str = Query("2y", pattern="^(5d|1mo|3mo|6mo|1y|2y|5y|10y)$"),
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


@router.get("/correlation")
def correlation(symbols: str, period: str = "2y", interval: str = "1d") -> dict[str, Any]:
    parsed = [item.strip().upper() for item in symbols.replace("\n", ",").split(",") if item.strip()]
    return correlation_payload(parsed, period=period, interval=interval)
