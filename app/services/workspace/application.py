from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any

import pandas as pd

from backend.app.services.workspace.alerts import (
    PORTFOLIO_ALERT_SYMBOL,
    create_alert,
    evaluate_alerts,
    latest_alert_inputs,
    read_alerts,
    write_alerts,
)
from backend.app.services.analysis.backtest import backtest_pullback_strategy, walk_forward_backtest
from backend.app.services.market_data.data_sources import (
    days_until_earnings,
    fetch_earnings_calendar,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    fetch_yahoo_snapshot,
    load_kaggle_history,
    merge_source_preference,
    read_watchlist,
)
from backend.app.services.analysis.indicators import add_indicators, correlation_matrix, price_volume_profile
from backend.app.services.market_data.markets import DEFAULT_WATCHLISTS, EXCHANGES, benchmark_for_symbol, normalize_symbol
from backend.app.services.analysis.ml_signal import latest_model_signal, walk_forward_validate
from backend.app.services.market_data.news import fetch_news, news_sentiment_score
from backend.app.services.analysis.risk import RISK_PROFILES, get_risk_profile, market_regime
from backend.app.services.analysis.strategies import build_signal, strategy_table
from backend.app.services.analysis.summary import build_focused_stock_brief
from backend.app.services.workspace.watchlist_store import (
    active_symbols,
    merge_instrument_tags,
    read_instrument_tags,
    role_for_symbol,
    trading_instruments,
    write_instrument_tags,
)


DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"


def load_benchmark_frame(benchmark_symbol: str, period: str, interval: str, refresh: bool = False) -> pd.DataFrame:
    raw, _status = fetch_yahoo_history(benchmark_symbol, period=period, interval=interval, refresh=refresh)
    return add_indicators(raw)


def load_analysis_frame(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    benchmark_symbol, _benchmark_name = benchmark_for_symbol(symbol)
    yahoo, yahoo_status = fetch_yahoo_history(symbol, period=period, interval=interval, refresh=refresh)
    kaggle, kaggle_status = load_kaggle_history(symbol)
    merged = merge_source_preference(yahoo, kaggle)
    status = f"{yahoo_status.source}: {yahoo_status.message}"
    if not kaggle.empty:
        status += f" | {kaggle_status.source}: {kaggle_status.message}"
    benchmark = load_benchmark_frame(benchmark_symbol, period, interval, refresh)
    return add_indicators(merged, benchmark=benchmark), status


def workspace_payload(user_id: str | None = None, exchange: str | None = None) -> dict[str, Any]:
    store_user_id = storage_user_id(user_id)
    stored = read_instrument_tags(user_id=store_user_id)
    source_symbols = active_symbols(stored)
    if not user_id and not source_symbols:
        source_symbols = read_watchlist() or list(DEFAULT_WATCHLISTS["Mixed"])
    instruments = dedupe_instruments(_normalize_instrument_row(row) for row in merge_instrument_tags(source_symbols, stored))
    return {
        "instruments": instruments,
        "symbols": active_symbols(instruments) or [_clean_symbol(symbol) for symbol in source_symbols if _clean_symbol(symbol)],
        "markets": markets_payload(),
        "alerts": read_alerts(user_id=store_user_id),
    }


def markets_payload() -> dict[str, Any]:
    return {
        "exchanges": {
            key: {
                "name": exchange.name,
                "examples": list(exchange.examples),
                "benchmarkSymbol": exchange.benchmark_symbol,
                "benchmarkName": exchange.benchmark_name,
            }
            for key, exchange in EXCHANGES.items()
        },
        "defaultWatchlists": {key: list(values) for key, values in DEFAULT_WATCHLISTS.items()},
        "riskProfiles": {
            key: {
                "name": profile.name,
                "description": profile.description,
                "riskPctPerTrade": profile.risk_pct_per_trade,
                "buyConfidenceThreshold": profile.min_confidence_full_buy,
            }
            for key, profile in RISK_PROFILES.items()
        },
        "strategies": ["Short-term (1-4 weeks)", "Long-term (6-12 months)", "Buy-dip"],
        "intents": ["Buy / Add", "Sell / Trim", "Hold / Watch"],
    }


def ticker_search_payload(query: str, markets: list[str] | None = None) -> dict[str, Any]:
    selected = [market.upper() for market in (markets or list(EXCHANGES)) if market.upper() in EXCHANGES]
    cleaned = _clean_symbol(query)
    if not cleaned:
        return {"results": []}

    candidates: list[tuple[str, str]] = []
    for market in selected:
        for symbol in DEFAULT_WATCHLISTS.get(market, ()):
            if cleaned in symbol.upper():
                candidates.append((symbol, market))
        candidate = normalize_symbol(cleaned, market)
        if candidate:
            candidates.append((candidate, market))
    if not selected:
        candidates.append((cleaned, ""))

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for symbol, market in candidates:
        canonical = _clean_symbol(symbol)
        if canonical in seen:
            continue
        if not symbol_exists(canonical):
            continue
        seen.add(canonical)
        snapshot, _status = fetch_yahoo_snapshot(canonical)
        resolved_market = market or market_for_symbol(canonical)
        label_bits = [
            canonical,
            resolved_market,
            str(snapshot.get("shortName") or snapshot.get("longName") or snapshot.get("quoteType") or "").strip(),
        ]
        results.append(
            {
                "symbol": canonical,
                "market": resolved_market,
                "label": " | ".join(bit for bit in label_bits if bit),
                "currency": currency_for_symbol(canonical),
                "name": snapshot.get("shortName") or snapshot.get("longName") or "",
            }
        )
        if len(results) >= 8:
            break
    return {"results": results}


def save_instruments(rows: list[dict[str, Any]], user_id: str | None = None) -> list[dict[str, Any]]:
    store_user_id = storage_user_id(user_id)
    normalized_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        raw_symbol = str(row.get("symbol", "")).strip()
        if not raw_symbol:
            continue
        canonical = resolve_symbol(raw_symbol, str(row.get("exchange") or ""))
        if not canonical:
            errors.append(f"Row {index}: {raw_symbol} is not a valid Yahoo Finance ticker.")
            continue
        normalized_rows.append(_normalize_instrument_row({**row, "symbol": canonical}))
    if errors:
        raise ValueError("; ".join(errors))
    normalized = dedupe_instruments(normalized_rows)
    write_instrument_tags(normalized, user_id=store_user_id)
    return read_instrument_tags(user_id=store_user_id)


def save_onboarding(
    rows: list[dict[str, Any]],
    exchange: str,
    role: str,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    store_user_id = storage_user_id(user_id)
    normalized, errors = normalize_onboarding_rows(rows, exchange, role)
    if errors:
        raise ValueError("; ".join(errors))
    existing = read_instrument_tags(user_id=store_user_id)
    by_symbol = {str(row.get("symbol", "")).upper(): row for row in existing}
    for row in normalized:
        by_symbol[str(row["symbol"]).upper()] = row
    write_instrument_tags(list(by_symbol.values()), user_id=store_user_id)
    return list(by_symbol.values())


def normalize_onboarding_rows(
    records: list[dict[str, Any]] | pd.DataFrame,
    exchange: str,
    role: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(records, start=1):
        raw_symbol = _clean_symbol(row.get("symbol", ""))
        symbol = resolve_symbol(raw_symbol, exchange)
        if not symbol:
            if raw_symbol:
                errors.append(f"Row {index}: {raw_symbol} is not a valid Yahoo Finance ticker for {exchange}.")
            continue
        average_cost = _safe_nonnegative(row.get("average_cost"))
        shares = _safe_nonnegative(row.get("shares"))
        book_cost = shares * average_cost
        if role == "Trading":
            purchase_date = str(row.get("purchase_date") or "")
            if average_cost <= 0:
                errors.append(f"Row {index}: average purchase price is required.")
            if shares <= 0:
                errors.append(f"Row {index}: shares are required.")
            if not purchase_date:
                errors.append(f"Row {index}: purchase date is required.")
        else:
            purchase_date = ""
        rows.append(
            {
                "symbol": symbol,
                "role": role,
                "active": True,
                "shares": shares,
                "average_cost": average_cost,
                "book_cost": book_cost,
                "purchase_date": purchase_date,
                "category": "",
                "intent": str(row.get("intent") or "Hold / Watch"),
                "strategy": str(row.get("strategy") or "Buy-dip"),
                "watch_reason": str(row.get("watch_reason") or ""),
                "notes": str(row.get("notes") or ""),
            }
        )
    if not rows:
        errors.append("Add at least one ticker.")
    return rows, errors


def focused_analysis_payload(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    refresh: bool = False,
    intent: str = "Hold / Watch",
    strategy: str = "Buy-dip",
    risk_profile_name: str = "Balanced",
    ml_model: str = "logistic",
) -> dict[str, Any]:
    normalized_symbol = _clean_symbol(symbol)
    frame, source_status = load_analysis_frame(normalized_symbol, period, interval, refresh)
    if frame.empty:
        return {"symbol": normalized_symbol, "ok": False, "status": source_status, "history": []}

    news = fetch_news(normalized_symbol, refresh=refresh)
    sentiment = news_sentiment_score(news)
    snapshot, snapshot_status = fetch_yahoo_snapshot(normalized_symbol, refresh=refresh)
    earnings, earnings_status = fetch_earnings_calendar(normalized_symbol, refresh=refresh)
    enriched_snapshot = enrich_snapshot_with_earnings(snapshot, earnings)
    risk_profile = get_risk_profile(risk_profile_name)
    benchmark_symbol, benchmark_name = benchmark_for_symbol(normalized_symbol)
    benchmark = load_benchmark_frame(benchmark_symbol, period, interval, refresh)
    regime = market_regime(benchmark, profile=risk_profile)
    ml_horizon = strategy_model_horizon_days(strategy)
    ml_signal = latest_model_signal(frame, horizon_days=ml_horizon, n_splits=5, model=ml_model)
    days_to_earnings = days_until_earnings(earnings.get("next_earnings_date"))
    signal = build_signal(
        normalized_symbol,
        frame,
        news_sentiment=sentiment,
        intent=intent,
        horizon=strategy,
        news_items=news,
        market_regime=regime,
        ml_signal=ml_signal,
        risk_profile=risk_profile,
        analyst_snapshot=enriched_snapshot,
        days_until_earnings=days_to_earnings,
    )
    brief = build_focused_stock_brief(normalized_symbol, frame, news, enriched_snapshot, days_to_earnings)
    backtest_trades, backtest_stats = backtest_pullback_strategy(frame)
    wf_windows, wf_summary = walk_forward_backtest(frame)
    ml_validation = walk_forward_validate(frame, horizon_days=ml_horizon, n_splits=5, model=ml_model)
    ml_validation_summary = dataclass_to_dict(ml_validation)
    ml_validation_summary.pop("fold_scores", None)

    return {
        "ok": True,
        "symbol": normalized_symbol,
        "sourceStatus": source_status,
        "snapshotStatus": f"{snapshot_status.source}: {snapshot_status.message}",
        "earningsStatus": f"{earnings_status.source}: {earnings_status.message}",
        "benchmark": {"symbol": benchmark_symbol, "name": benchmark_name},
        "history": frame_to_records(frame),
        "volumeProfile": frame_to_records(price_volume_profile(frame)),
        "summary": {"title": brief.title, "subtitle": brief.subtitle, "markdown": brief.markdown, "html": brief.html},
        "signal": serialize_signal(signal),
        "strategyRows": frame_to_records(strategy_table(signal)),
        "news": frame_to_records(news),
        "newsSentiment": sentiment,
        "snapshot": json_safe(enriched_snapshot),
        "earnings": json_safe(earnings),
        "daysUntilEarnings": days_to_earnings,
        "marketRegime": json_safe(regime),
        "ml": json_safe(dataclass_to_dict(ml_signal)),
        "mlValidation": {
            **json_safe(ml_validation_summary),
            "foldScores": frame_to_records(ml_validation.fold_scores),
        },
        "backtest": {
            "trades": frame_to_records(backtest_trades),
            "stats": json_safe(backtest_stats),
            "walkForward": frame_to_records(wf_windows),
            "walkForwardSummary": json_safe(wf_summary),
        },
    }


def portfolio_payload(
    instruments: list[dict[str, Any]] | None = None,
    user_id: str | None = None,
    period: str = "6mo",
    interval: str = DEFAULT_INTERVAL,
    refresh: bool = False,
) -> dict[str, Any]:
    store_user_id = storage_user_id(user_id)
    rows = instruments if instruments is not None else read_instrument_tags(user_id=store_user_id)
    rows = enrich_instrument_categories(rows, refresh=refresh)
    histories = {}
    for symbol in active_symbols(rows):
        frame, _status = load_analysis_frame(symbol, period=period, interval=interval, refresh=refresh)
        histories[symbol] = frame
    portfolio = build_portfolio_table(trading_instruments(rows), histories)
    category = category_exposure_table(portfolio)
    watchlist = build_watchlist_table(rows, histories)
    return {
        "portfolio": frame_to_records(portfolio),
        "categoryExposure": frame_to_records(category),
        "watchlist": frame_to_records(watchlist),
        "metrics": portfolio_metrics(portfolio),
        "notes": portfolio_recommendations(portfolio, category),
        "executionPlans": portfolio_execution_plans(trading_instruments(rows), histories),
    }


def correlation_payload(symbols: list[str], period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> dict[str, Any]:
    histories = {}
    for symbol in symbols:
        normalized = _clean_symbol(symbol)
        frame, _status = load_analysis_frame(normalized, period=period, interval=interval)
        if not frame.empty:
            histories[normalized] = frame
    corr = correlation_matrix(histories)
    return {"symbols": list(histories), "matrix": frame_to_records(corr.reset_index(names="symbol"))}


def alerts_payload(user_id: str | None = None) -> list[dict[str, Any]]:
    return read_alerts(user_id=storage_user_id(user_id))


def create_alert_payload(payload: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    store_user_id = storage_user_id(user_id)
    alerts = read_alerts(user_id=store_user_id)
    alert = create_alert(**payload)
    write_alerts(alerts + [alert], user_id=store_user_id)
    return alert


def replace_alerts_payload(alerts: list[dict[str, Any]], user_id: str | None = None) -> list[dict[str, Any]]:
    store_user_id = storage_user_id(user_id)
    write_alerts(alerts, user_id=store_user_id)
    return read_alerts(user_id=store_user_id)


def delete_alert_payload(alert_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
    store_user_id = storage_user_id(user_id)
    alerts = [alert for alert in read_alerts(user_id=store_user_id) if str(alert.get("id")) != alert_id]
    write_alerts(alerts, user_id=store_user_id)
    return alerts


def evaluate_alerts_payload(user_id: str | None = None) -> dict[str, Any]:
    store_user_id = storage_user_id(user_id)
    instruments = read_instrument_tags(user_id=store_user_id)
    latest_by_symbol = {}
    histories = {}
    for symbol in active_symbols(instruments):
        frame, _status = load_analysis_frame(symbol, period="1y", interval=DEFAULT_INTERVAL)
        quote, _quote_status = fetch_yahoo_quote(symbol)
        histories[symbol] = frame
        latest_by_symbol[symbol] = latest_alert_inputs(symbol, frame, quote)
    portfolio = build_portfolio_table(trading_instruments(instruments), histories)
    if not portfolio.empty:
        latest_by_symbol[PORTFOLIO_ALERT_SYMBOL] = portfolio_alert_inputs(portfolio)
    alerts = read_alerts(user_id=store_user_id)
    updated, triggered = evaluate_alerts(alerts, latest_by_symbol)
    if updated != alerts:
        write_alerts(updated, user_id=store_user_id)
    return {"alerts": updated, "triggered": triggered}


def storage_user_id(user_id: str | None = None) -> str | None:
    return user_id


def _clean_symbol(symbol: object) -> str:
    cleaned = str(symbol or "").strip().upper()
    if cleaned.endswith(".TO"):
        base = cleaned.removesuffix(".TO")
        us_defaults = set(DEFAULT_WATCHLISTS["NASDAQ"]) | set(DEFAULT_WATCHLISTS["NYSE"])
        if base in us_defaults:
            return base
    return cleaned


def resolve_symbol(raw_symbol: object, exchange: str | None = None) -> str:
    cleaned = _clean_symbol(raw_symbol)
    if not cleaned:
        return ""
    exchange_key = str(exchange or "").upper()
    candidates: list[str] = []
    if "." in cleaned:
        candidates.append(cleaned)
    elif exchange_key == "TSX":
        candidates.append(normalize_symbol(cleaned, "TSX"))
    elif exchange_key in EXCHANGES:
        candidates.append(normalize_symbol(cleaned, exchange_key))
    else:
        candidates.append(cleaned)
        candidates.extend(normalize_symbol(cleaned, market) for market in ("TSX", "NASDAQ", "NYSE"))

    for candidate in dict.fromkeys(_clean_symbol(item) for item in candidates if item):
        if symbol_exists(candidate):
            return candidate
    return ""


def symbol_exists(symbol: str) -> bool:
    frame, _status = fetch_yahoo_history(symbol, period="5d", interval="1d", refresh=False)
    return not frame.empty


def market_for_symbol(symbol: str) -> str:
    cleaned = _clean_symbol(symbol)
    if cleaned.endswith(".TO") or cleaned.endswith(".V"):
        return "TSX"
    for market in ("NASDAQ", "NYSE"):
        if cleaned in DEFAULT_WATCHLISTS.get(market, ()):
            return market
    return "US"


def currency_for_symbol(symbol: str) -> str:
    return "CAD" if market_for_symbol(symbol) == "TSX" else "USD"


def dedupe_instruments(rows: Any) -> list[dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = _clean_symbol(row.get("symbol"))
        if not symbol:
            continue
        current = by_symbol.get(symbol)
        row = {**row, "symbol": symbol}
        if current is None:
            by_symbol[symbol] = row
            continue
        for key, value in row.items():
            if key == "role" and value == "Trading":
                current[key] = value
            elif key == "active":
                current[key] = bool(current.get(key, True)) or bool(value)
            elif _has_value(value) and not _has_value(current.get(key)):
                current[key] = value
    return list(by_symbol.values())


def _has_value(value: object) -> bool:
    return value is not None and value != "" and value != 0 and value != 0.0


def enrich_instrument_categories(rows: list[dict[str, Any]], refresh: bool = False) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    snapshot_cache: dict[str, dict[str, object]] = {}
    for row in rows:
        enriched = dict(row)
        symbol = _clean_symbol(enriched.get("symbol"))
        if symbol and not str(enriched.get("category") or "").strip():
            if symbol not in snapshot_cache:
                snapshot, _status = fetch_yahoo_snapshot(symbol, refresh=refresh)
                snapshot_cache[symbol] = snapshot
            snapshot = snapshot_cache[symbol]
            category = snapshot.get("sector") or snapshot.get("industry") or snapshot.get("quoteType")
            if category:
                enriched["category"] = str(category)
        enriched_rows.append(enriched)
    return enriched_rows


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [json_safe(row) for row in result.to_dict("records")]


def serialize_signal(signal: object) -> dict[str, Any]:
    data = dataclass_to_dict(signal)
    return json_safe(data)


def dataclass_to_dict(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if isinstance(value, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return frame_to_records(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


def enrich_snapshot_with_earnings(snapshot: dict[str, object], earnings: dict[str, object]) -> dict[str, object]:
    enriched = dict(snapshot)
    next_date = earnings.get("next_earnings_date")
    last_date = earnings.get("last_earnings_date")
    if next_date:
        enriched["nextEarningsDate"] = next_date
        enriched.pop("lastEarningsDate", None)
    elif last_date:
        enriched.pop("nextEarningsDate", None)
        enriched["lastEarningsDate"] = last_date
    return enriched


def strategy_model_horizon_days(horizon: str) -> int:
    key = horizon.lower()
    if "short" in key:
        return 10
    if "long" in key:
        return 30
    return 15


def build_portfolio_table(trading_rows: list[dict[str, object]], histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in trading_rows:
        symbol = str(item.get("symbol", "")).strip().upper()
        frame = histories.get(symbol, pd.DataFrame())
        shares = _safe_nonnegative(item.get("shares"))
        average_cost = _safe_nonnegative(item.get("average_cost"))
        book_cost = _safe_nonnegative(item.get("book_cost"))
        if book_cost <= 0 and average_cost > 0:
            book_cost = shares * average_cost
        if average_cost <= 0 and shares > 0 and book_cost > 0:
            average_cost = book_cost / shares
        if shares <= 0 and average_cost > 0 and book_cost > 0:
            shares = book_cost / average_cost
        if average_cost <= 0 and book_cost <= 0:
            continue
        if shares <= 0 or frame.empty:
            continue

        price = latest_close(frame)
        prev_close = previous_close(frame, 1)
        week_close = previous_close(frame, 5)
        market_value = shares * price
        today_dollars = shares * (price - prev_close) if prev_close == prev_close else float("nan")
        today_pct = price / prev_close - 1 if prev_close == prev_close and prev_close > 0 else float("nan")
        week_dollars = shares * (price - week_close) if week_close == week_close else float("nan")
        week_pct = price / week_close - 1 if week_close == week_close and week_close > 0 else float("nan")
        since_purchase = market_value - book_cost
        since_purchase_pct = since_purchase / book_cost if book_cost > 0 else float("nan")
        rows.append(
            {
                "Symbol": symbol,
                "Category": str(item.get("category") or "Unassigned"),
                "Shares": shares,
                "Purchase Date": str(item.get("purchase_date") or ""),
                "Avg Cost": average_cost,
                "Book Cost": book_cost,
                "Current Price": price,
                "Market Value": market_value,
                "Portfolio %": 0.0,
                "Today $": today_dollars,
                "Today %": today_pct,
                "Week $": week_dollars,
                "Week %": week_pct,
                "Since Purchase $": since_purchase,
                "Since Purchase %": since_purchase_pct,
                "Notes": str(item.get("notes") or ""),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    total_market = float(table["Market Value"].sum())
    if total_market > 0:
        table["Portfolio %"] = table["Market Value"] / total_market
    return table.sort_values("Market Value", ascending=False).reset_index(drop=True)


def build_watchlist_table(rows: list[dict[str, Any]], histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = []
    for item in rows:
        if str(item.get("role") or "").strip().title() != "Watching" or not bool(item.get("active", True)):
            continue
        symbol = str(item.get("symbol", "")).upper()
        frame = histories.get(symbol, pd.DataFrame())
        current = latest_close(frame) if not frame.empty else float("nan")
        prev = previous_close(frame, 1) if not frame.empty else float("nan")
        week = previous_close(frame, 5) if not frame.empty else float("nan")
        output.append(
            {
                "Symbol": symbol,
                "Category": str(item.get("category") or "Unassigned"),
                "Reason": str(item.get("watch_reason") or item.get("notes") or ""),
                "Intent": str(item.get("intent") or "Hold / Watch"),
                "Strategy": str(item.get("strategy") or "Buy-dip"),
                "Current Price": current,
                "Today %": current / prev - 1 if current == current and prev == prev and prev > 0 else float("nan"),
                "Week %": current / week - 1 if current == current and week == week and week > 0 else float("nan"),
            }
        )
    return pd.DataFrame(output)


def portfolio_metrics(portfolio: pd.DataFrame) -> dict[str, Any]:
    currency = portfolio_currency(portfolio)
    if portfolio.empty:
        return {
            "totalInvested": 0.0,
            "marketValue": 0.0,
            "sincePurchase": 0.0,
            "sincePurchasePct": None,
            "todayPl": 0.0,
            "weekPl": 0.0,
            "currency": currency,
        }
    total_book = float(portfolio["Book Cost"].sum())
    total_market = float(portfolio["Market Value"].sum())
    total_pl = float(portfolio["Since Purchase $"].sum())
    return {
        "totalInvested": total_book,
        "marketValue": total_market,
        "sincePurchase": total_pl,
        "sincePurchasePct": total_pl / total_book if total_book > 0 else None,
        "todayPl": float(portfolio["Today $"].fillna(0).sum()),
        "weekPl": float(portfolio["Week $"].fillna(0).sum()),
        "currency": currency,
    }


def portfolio_currency(portfolio: pd.DataFrame) -> str:
    if portfolio.empty or "Symbol" not in portfolio.columns:
        return "USD"
    currencies = {currency_for_symbol(str(symbol)) for symbol in portfolio["Symbol"].dropna()}
    if len(currencies) == 1:
        return next(iter(currencies))
    return "MIXED"


def portfolio_execution_plans(trading_rows: list[dict[str, object]], histories: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for item in trading_rows:
        symbol = str(item.get("symbol", "")).strip().upper()
        frame = histories.get(symbol, pd.DataFrame())
        if frame.empty:
            continue
        risk_profile = get_risk_profile("Balanced")
        benchmark_symbol, _benchmark_name = benchmark_for_symbol(symbol)
        benchmark = histories.get(benchmark_symbol, pd.DataFrame())
        regime = market_regime(benchmark if not benchmark.empty else frame, profile=risk_profile)
        news = fetch_news(symbol)
        sentiment = news_sentiment_score(news)
        snapshot, _snapshot_status = fetch_yahoo_snapshot(symbol)
        earnings, _earnings_status = fetch_earnings_calendar(symbol)
        signal = build_signal(
            symbol,
            frame,
            news_sentiment=sentiment,
            intent=str(item.get("intent") or "Hold / Watch"),
            horizon=str(item.get("strategy") or "Buy-dip"),
            news_items=news,
            market_regime=regime,
            ml_signal=latest_model_signal(frame, horizon_days=strategy_model_horizon_days(str(item.get("strategy") or "Buy-dip"))),
            risk_profile=risk_profile,
            analyst_snapshot=enrich_snapshot_with_earnings(snapshot, earnings),
            days_until_earnings=days_until_earnings(earnings.get("next_earnings_date")),
        )
        plans.append(
            {
                "symbol": symbol,
                "action": signal.action,
                "score": signal.score,
                "setup": signal.setup,
                "entryZone": signal.entry_zone,
                "exitZone": signal.exit_zone,
                "risk": signal.risk_note,
            }
        )
    return plans


def portfolio_alert_inputs(portfolio: pd.DataFrame) -> dict[str, float | None]:
    if portfolio.empty:
        return {"portfolio_value": None, "portfolio_today_pl": None, "portfolio_since_purchase_pl": None}
    return {
        "portfolio_value": float(portfolio["Market Value"].sum()),
        "portfolio_today_pl": float(portfolio["Today $"].fillna(0).sum()),
        "portfolio_since_purchase_pl": float(portfolio["Since Purchase $"].fillna(0).sum()),
    }


def latest_close(frame: pd.DataFrame) -> float:
    series = frame.dropna(subset=["Close"]).sort_values("Date")
    if series.empty:
        return float("nan")
    return float(series.iloc[-1]["Close"])


def previous_close(frame: pd.DataFrame, periods_back: int) -> float:
    series = frame.dropna(subset=["Close"]).sort_values("Date")
    if len(series) <= periods_back:
        return float("nan")
    return float(series.iloc[-1 - periods_back]["Close"])


def category_exposure_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()
    grouped = (
        portfolio.groupby("Category", dropna=False)
        .agg({"Market Value": "sum", "Since Purchase $": "sum"})
        .reset_index()
        .sort_values("Market Value", ascending=False)
    )
    total = float(grouped["Market Value"].sum())
    grouped["Portfolio %"] = grouped["Market Value"] / total if total > 0 else 0.0
    return grouped


def portfolio_recommendations(portfolio: pd.DataFrame, category: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if portfolio.empty:
        return ["No trading holdings were entered yet."]
    largest = portfolio.iloc[0]
    if float(largest["Portfolio %"]) >= 0.4:
        notes.append(f"{largest['Symbol']} is {float(largest['Portfolio %']):.1%} of the portfolio; confirm this concentration is intentional.")
    losers = portfolio[portfolio["Since Purchase %"] <= -0.08]
    if not losers.empty:
        worst = losers.sort_values("Since Purchase %").iloc[0]
        notes.append(f"{worst['Symbol']} is down {float(worst['Since Purchase %']):.1%} from purchase; review the thesis and stop level.")
    winners = portfolio[portfolio["Since Purchase %"] >= 0.15]
    if not winners.empty:
        best = winners.sort_values("Since Purchase %", ascending=False).iloc[0]
        notes.append(f"{best['Symbol']} is up {float(best['Since Purchase %']):.1%} from purchase; trim only if allocation or thesis risk requires it.")
    if not category.empty and float(category.iloc[0]["Portfolio %"]) >= 0.5:
        notes.append(f"{category.iloc[0]['Category']} is {float(category.iloc[0]['Portfolio %']):.1%} of portfolio value; diversify if this was not deliberate.")
    day_loss = float(portfolio["Today $"].fillna(0).sum())
    if day_loss < 0:
        notes.append(f"Portfolio is down ${abs(day_loss):,.2f} today; check whether the loss is broad-based or concentrated.")
    if not notes:
        notes.append("No major concentration or loss warning was triggered by the current holdings data.")
    return notes


def filter_frame_by_date_range(frame: pd.DataFrame, range_label: str, anchor: date | None = None) -> pd.DataFrame:
    if frame.empty or range_label == "All" or "Date" not in frame.columns:
        return frame
    prepared = frame.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"]).dt.tz_localize(None)
    latest = prepared["Date"].max()
    if pd.isna(latest):
        return prepared
    anchor_date = anchor or latest.date()
    anchor_ts = pd.Timestamp(anchor_date)
    if range_label == "YTD":
        start = pd.Timestamp(date(anchor_date.year, 1, 1))
    elif range_label.endswith("D"):
        start = anchor_ts - pd.Timedelta(days=int(range_label[:-1]))
    elif range_label.endswith("W"):
        start = anchor_ts - pd.Timedelta(days=7 * int(range_label[:-1]))
    elif range_label.endswith("M"):
        start = anchor_ts - pd.DateOffset(months=int(range_label[:-1]))
    elif range_label.endswith("Y"):
        start = anchor_ts - pd.DateOffset(years=int(range_label[:-1]))
    else:
        return prepared
    return prepared[prepared["Date"] >= start].reset_index(drop=True)


def _normalize_instrument_row(row: dict[str, Any]) -> dict[str, Any]:
    shares = _safe_nonnegative(row.get("shares"))
    average_cost = _safe_nonnegative(row.get("average_cost"))
    book_cost = _safe_nonnegative(row.get("book_cost"))
    return {
        "symbol": _clean_symbol(row.get("symbol", "")),
        "role": str(row.get("role") or "Watching"),
        "active": bool(row.get("active", True)),
        "shares": shares,
        "average_cost": average_cost,
        "book_cost": book_cost or shares * average_cost,
        "purchase_date": str(row.get("purchase_date") or ""),
        "category": str(row.get("category") or ""),
        "intent": str(row.get("intent") or ""),
        "strategy": str(row.get("strategy") or ""),
        "watch_reason": str(row.get("watch_reason") or ""),
        "notes": str(row.get("notes") or ""),
    }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_nonnegative(value: object) -> float:
    number = _safe_float(value)
    if number != number:
        return 0.0
    return max(number, 0.0)
