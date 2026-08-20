from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from backend.app.services.workspace.alerts import (
    PORTFOLIO_ALERT_SYMBOL,
    evaluate_alerts,
    latest_alert_inputs,
    read_alerts,
    write_alerts,
)
from backend.app.services.market_data.data_sources import DATA_DIR, fetch_yahoo_history, fetch_yahoo_quote
from backend.app.database import database_available, list_user_ids_with_instruments
from backend.app.services.analysis.indicators import add_indicators
from backend.app.services.workspace.watchlist_store import active_symbols, read_instrument_tags, trading_instruments


SYNC_STATUS_PATH = DATA_DIR / "sync" / "last_run.json"


def run_once() -> dict[str, object]:
    user_ids: list[str | None] = list_user_ids_with_instruments() if database_available() else []
    if not user_ids:
        user_ids = [None]
    all_triggered = []
    all_status = {}
    checked_symbols: set[str] = set()

    for user_id in user_ids:
        user_result = run_once_for_user(user_id)
        all_triggered.extend(user_result["triggered_alerts"])
        all_status.update(user_result["status"])
        checked_symbols.update(user_result["symbols_checked"])

    result = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "symbols_checked": sorted(checked_symbols),
        "triggered_alerts": all_triggered,
        "status": all_status,
    }
    SYNC_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATUS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_once_for_user(user_id: str | None = None) -> dict[str, object]:
    instruments = read_instrument_tags(user_id=user_id)
    symbols = active_symbols(instruments)
    latest_by_symbol: dict[str, dict[str, float | None]] = {}
    quote_status: dict[str, str] = {}
    histories = {}

    for symbol in symbols:
        quote, quote_state = fetch_yahoo_quote(symbol)
        history, history_state = fetch_yahoo_history(symbol, period="1y", interval="1d", refresh=False)
        prepared = add_indicators(history)
        histories[symbol] = prepared
        latest_inputs = latest_alert_inputs(symbol, prepared, quote)
        latest_inputs["previous_close"] = previous_close(prepared)
        latest_by_symbol[symbol] = latest_inputs
        quote_status[symbol] = f"{quote_state.source}: {quote_state.message}; {history_state.source}: {history_state.message}"

    portfolio_inputs = build_portfolio_alert_inputs(trading_instruments(instruments), latest_by_symbol)
    if portfolio_inputs:
        latest_by_symbol[PORTFOLIO_ALERT_SYMBOL] = portfolio_inputs

    alerts = read_alerts(user_id=user_id)
    updated, triggered = evaluate_alerts(alerts, latest_by_symbol)
    if updated != alerts:
        write_alerts(updated, user_id=user_id)

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "symbols_checked": symbols,
        "triggered_alerts": triggered,
        "status": quote_status,
    }


def build_portfolio_alert_inputs(
    trading_rows: list[dict[str, object]],
    latest_by_symbol: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    market_value = 0.0
    today_pl = 0.0
    since_purchase_pl = 0.0
    has_value = False
    for row in trading_rows:
        symbol = str(row.get("symbol", "")).upper()
        price = latest_by_symbol.get(symbol, {}).get("price")
        if price is None:
            continue
        shares = _float(row.get("shares"))
        average_cost = _float(row.get("average_cost"))
        book_cost = _float(row.get("book_cost"))
        if shares <= 0 and average_cost > 0 and book_cost > 0:
            shares = book_cost / average_cost
        if shares <= 0:
            continue
        value = shares * float(price)
        market_value += value
        previous = latest_by_symbol.get(symbol, {}).get("previous_close")
        if previous is not None:
            today_pl += shares * (float(price) - float(previous))
        since_purchase_pl += value - book_cost
        has_value = True
    if not has_value:
        return {}
    return {
        "portfolio_value": market_value,
        "portfolio_today_pl": today_pl,
        "portfolio_since_purchase_pl": since_purchase_pl,
    }


def _float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed != parsed:
        return 0.0
    return max(parsed, 0.0)


def previous_close(frame) -> float | None:
    if frame is None or frame.empty:
        return None
    series = frame.dropna(subset=["Close"]).sort_values("Date")
    if len(series) < 2:
        return None
    return float(series.iloc[-2]["Close"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run near-real-time stock dashboard alert sync.")
    parser.add_argument("--loop", action="store_true", help="Keep syncing until stopped.")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Polling interval when --loop is set.")
    args = parser.parse_args()

    while True:
        result = run_once()
        print(json.dumps({"ran_at": result["ran_at"], "triggered": len(result["triggered_alerts"])}, indent=2))
        if not args.loop:
            return
        time.sleep(max(args.interval_seconds, 15))


if __name__ == "__main__":
    main()
