from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from app.config import REFRESH_INTERVAL_SECONDS
from app.database import list_user_ids_with_instruments
from app.services.market_data.data_sources import (
    fetch_earnings_calendar,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    fetch_yahoo_snapshot,
)
from app.services.market_data.news import fetch_news
from app.services.workspace.application import evaluate_alerts_payload, portfolio_payload
from app.services.workspace.watchlist_store import active_symbols, read_instrument_tags

ACTIVE_GUEST_SYMBOLS: set[str] = set()


def register_guest_symbols(symbols: list[str]) -> None:
    ACTIVE_GUEST_SYMBOLS.update(symbol.strip().upper() for symbol in symbols if symbol.strip())


async def start_workspace_refresh_loop(interval_seconds: int = REFRESH_INTERVAL_SECONDS) -> asyncio.Task[None]:
    task = asyncio.create_task(_refresh_loop(interval_seconds), name="workspace-refresh-loop")
    return task


async def stop_workspace_refresh_loop(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _refresh_loop(interval_seconds: int) -> None:
    while True:
        try:
            await refresh_all_workspaces_async()
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


async def refresh_all_workspaces_async() -> dict[str, Any]:
    return await asyncio.to_thread(refresh_all_workspaces)


def refresh_all_workspaces() -> dict[str, Any]:
    """Warm market caches for active persisted app instruments.

    Guests keep watchlists in browser storage, so the first scoped sync request
    registers those symbols in process memory. The service then keeps warming
    those active symbols while the backend process is running.
    """

    refreshed_symbols: set[str] = set()
    active_universe: set[str] = set()
    user_ids = list_user_ids_with_instruments()

    local_rows = read_instrument_tags()
    active_universe.update(active_symbols(local_rows))
    active_universe.update(ACTIVE_GUEST_SYMBOLS)

    for user_id in user_ids:
        rows = read_instrument_tags(user_id=user_id)
        active_universe.update(active_symbols(rows))

    for symbol in sorted(active_universe):
        refresh_symbol(symbol)
        refreshed_symbols.add(symbol)

    plan_counts: dict[str, int] = {}
    triggered_alerts: list[dict[str, Any]] = []
    users_to_analyze = [None, *user_ids]
    for user_id in users_to_analyze:
        key = str(user_id or "local")
        portfolio = portfolio_payload(user_id=user_id, refresh=False)
        plan_counts[key] = len(portfolio.get("executionPlans") or [])
        alert_result = evaluate_alerts_payload(user_id=user_id)
        triggered_alerts.extend(alert_result.get("triggered", []))

    return {
        "ok": True,
        "syncedAt": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(refreshed_symbols),
        "executionPlanCounts": plan_counts,
        "triggeredAlerts": triggered_alerts,
    }


def refresh_symbol(symbol: str) -> None:
    """Warm the data needed by dashboards without blocking on deep analytics."""

    fetch_yahoo_quote(symbol)
    fetch_yahoo_history(symbol, period="5d", interval="5m", refresh=True)
    fetch_yahoo_history(symbol, period="6mo", interval="1d", refresh=True)
    snapshot, _snapshot_status = fetch_yahoo_snapshot(symbol, refresh=False)
    company_name = str(snapshot.get("longName") or snapshot.get("shortName") or "")
    fetch_earnings_calendar(symbol, refresh=False)
    fetch_news(symbol, company_name=company_name, refresh=True)
