from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from backend.app.services.workspace.application import markets_payload, ticker_search_payload


router = APIRouter(prefix="/api", tags=["markets"])


@router.get("/markets")
def markets() -> dict[str, Any]:
    return markets_payload()


@router.get("/tickers/search")
def search_tickers(
    q: str = Query("", min_length=1),
    markets: str = "TSX,NASDAQ,NYSE",
) -> dict[str, Any]:
    selected_markets = [item.strip().upper() for item in markets.split(",") if item.strip()]
    return ticker_search_payload(q, selected_markets)
