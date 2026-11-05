"""Shared, deterministic evidence used by prediction and portfolio views.

The module deliberately contains no provider calls.  It turns the already
loaded Yahoo-backed records into auditable features that both local rules and
OpenAI can interpret without receiving different evidence.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from math import floor, sqrt
from typing import Any


def build_deep_evidence(
    history: list[dict[str, Any]],
    snapshot: dict[str, Any] | None = None,
    financial_statements: dict[str, list[dict[str, Any]]] | None = None,
    earnings: dict[str, Any] | None = None,
    news: list[dict[str, Any]] | None = None,
    holding: dict[str, Any] | None = None,
    *,
    intent: str = "Hold / Watch",
    strategy: str = "Buy-dip",
    risk_profile: str = "Balanced",
) -> dict[str, Any]:
    """Build the full evidence package for one instrument.

    Values are intentionally conservative: missing fundamentals remain
    missing, scenario ranges are labelled heuristics, and dividend-derived
    returns are marked as estimates unless actual cash receipts are supplied.
    """

    snapshot = snapshot or {}
    points = _history_points(history)
    current_price = _number(snapshot.get("currentPrice") or snapshot.get("regularMarketPrice"))
    if current_price is None and points:
        current_price = points[-1][1]
    current_price = current_price or 0.0
    historical = _historical_context(points)
    pullback = _pullback_zones(historical, current_price, risk_profile, strategy, intent)
    fundamentals = _fundamental_evidence(snapshot, financial_statements or {}, earnings or {})
    income = _income_evidence(snapshot, holding or {}, current_price)
    thesis = _thesis_classification(snapshot, historical, fundamentals, income, pullback)
    scenarios = _scenario_model(historical, fundamentals, thesis, current_price, strategy, risk_profile)
    catalysts = _catalysts(fundamentals, earnings or {}, news or [])
    contradictions = _contradictions(historical, pullback, thesis)
    execution = _execution_guidance(intent, strategy, risk_profile, thesis, pullback, scenarios, catalysts)
    pestel = _pestel_context(snapshot)
    return {
        "asOf": _latest_as_of(points),
        "historicalReversion": historical,
        "pullbackEntryZones": pullback,
        "fundamentals": fundamentals,
        "pestel": pestel,
        "incomeAndCapitalPreservation": income,
        "thesis": thesis,
        "scenarios": scenarios,
        "catalysts": catalysts,
        "contradictions": contradictions,
        "execution": execution,
        "method": "Deterministic evidence layer: OHLCV, statement fields, earnings dates, news, and holding context are calculated before provider interpretation.",
    }


def _history_points(rows: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for row in rows:
        raw_date = row.get("Date") or row.get("Datetime")
        close = _number(row.get("Close"))
        if not raw_date or close is None or close <= 0:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        points.append((parsed, close))
    return sorted(points, key=lambda item: item[0])


def _historical_context(points: list[tuple[datetime, float]]) -> dict[str, Any]:
    if not points:
        return {"status": "unavailable", "read": "Multi-year trading-centre/reversion analysis was unavailable because no dated closes were returned."}
    closes = [value for _, value in points]
    latest_date, latest = points[-1]
    ytd = [value for stamp, value in points if stamp.year == latest_date.year] or closes
    recent = closes[-252:]
    daily_returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes)) if closes[index - 1] > 0]
    recent_returns = [recent[index] / recent[index - 1] - 1 for index in range(1, len(recent)) if recent[index - 1] > 0]
    centre = _percentile(closes, 50)
    low = _percentile(closes, 10)
    high = _percentile(closes, 90)
    containment = sum(low <= value <= high for value in closes) / len(closes) * 100
    distance = (latest / centre - 1) * 100 if centre else 0.0
    ytd_return = (latest / ytd[0] - 1) * 100 if ytd[0] else 0.0
    average_move = sum(abs(value) for value in daily_returns) / len(daily_returns) * 100 if daily_returns else 0.0
    recent_move = sum(abs(value) for value in recent_returns) / len(recent_returns) * 100 if recent_returns else 0.0
    volatility = _std(daily_returns) * 100
    recent_volatility = _std(recent_returns) * 100
    years = max(1.0, (latest_date - points[0][0]).days / 365.25)
    flat = abs(ytd_return) < 3 or (abs(distance) < 5 and abs(ytd_return) < 8)
    return {
        "status": "ready",
        "asOf": latest_date.isoformat(),
        "observations": len(points),
        "historyYears": round(years, 2),
        "ytdReturnPct": round(ytd_return, 2),
        "ytdStartPrice": round(ytd[0], 4),
        "ytdHigh": round(max(ytd), 4),
        "ytdLow": round(min(ytd), 4),
        "oneYearHigh": round(max(recent), 4),
        "oneYearLow": round(min(recent), 4),
        "multiYearHigh": round(max(closes), 4),
        "multiYearLow": round(min(closes), 4),
        "tradingCentre": round(centre, 4),
        "coreRange": {"low": round(_percentile(closes, 25), 4), "high": round(_percentile(closes, 75), 4)},
        "establishedRange": {"low": round(low, 4), "high": round(high, 4)},
        "distanceFromCentrePct": round(distance, 2),
        "averageDailyMovementPct": round(average_move, 3),
        "recentYearAverageDailyMovementPct": round(recent_move, 3),
        "dailyVolatilityPct": round(volatility, 3),
        "recentYearVolatilityPct": round(recent_volatility, 3),
        "historicalRangeContainmentPct": round(containment, 1),
        "isFlat": flat,
        "read": f"Across {years:.1f} years, the trading centre was {centre:.2f}; {containment:.0f}% of closes stayed within {low:.2f}-{high:.2f}. YTD is {'up' if ytd_return >= 0 else 'down'} {abs(ytd_return):.1f}%.",
    }


def _pullback_zones(historical: dict[str, Any], price: float, risk: str, strategy: str, intent: str) -> dict[str, Any]:
    reference = _number(historical.get("oneYearHigh")) or 0.0
    if reference <= 0 or price <= 0:
        return {"status": "unavailable", "bands": [], "preferredBand": None, "guidance": "A current price and one-year high are required for entry-zone analysis."}
    bands = []
    for label, shallow, deep in (("6-8% starter", 6, 8), ("10% better entry", 10, 10), ("12-15% strong dip", 12, 15)):
        upper = reference * (1 - shallow / 100)
        lower = reference * (1 - deep / 100)
        bands.append({"label": label, "priceRange": {"low": round(lower, 2), "high": round(upper, 2)}, "active": lower <= price <= upper, "passed": price <= upper})
    preferred = bands[2] if "conservative" in risk.lower() else bands[0] if "short" in strategy.lower() or "growth" in risk.lower() else bands[1]
    guidance = "Use the active zone as a staged reference, not as proof of value." if intent == "Buy / Add" else "Use these bands as watch levels and prioritize thesis confirmation before changing the position."
    return {"status": "ready", "referenceHigh": round(reference, 2), "currentPrice": round(price, 2), "pullbackPct": round((1 - price / reference) * 100, 2), "bands": bands, "preferredBand": preferred["label"], "guidance": guidance}


def _fundamental_evidence(snapshot: dict[str, Any], statements: dict[str, list[dict[str, Any]]], earnings: dict[str, Any]) -> dict[str, Any]:
    keys = ("totalRevenue", "revenueGrowth", "grossMargins", "profitMargins", "netIncomeToCommon", "totalCash", "totalDebt", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook", "payoutRatio", "targetMeanPrice", "targetMedianPrice", "numberOfAnalystOpinions")
    metrics = {key: snapshot.get(key) for key in keys if snapshot.get(key) is not None}
    statement_changes = []
    aliases = {"Revenue": ("total revenue", "operating revenue", "revenue"), "Net income": ("net income",), "Operating cash flow": ("operating cash flow", "total cash from operating activities"), "Free cash flow": ("free cash flow",)}
    for label, names in aliases.items():
        for row in statements.get("incomeStatement", []) + statements.get("cashFlow", []):
            metric = str(row.get("metric") or "").lower()
            if any(name in metric for name in names):
                values = sorted((str(period), _number(value)) for period, value in row.items() if period != "metric" and _number(value) is not None)
                if len(values) >= 2 and values[-2][1] not in (None, 0):
                    change = values[-1][1] / values[-2][1] - 1
                    statement_changes.append({"metric": label, "from": values[-2][0], "to": values[-1][0], "changePct": round(change * 100, 2)})
                break
    debt = _number(snapshot.get("totalDebt"))
    cash = _number(snapshot.get("totalCash"))
    net_debt = round(debt - cash, 2) if debt is not None and cash is not None else None
    negative_changes = [change["metric"] for change in statement_changes if (_number(change.get("changePct")) or 0) < 0]
    return {"metrics": metrics, "statementChanges": statement_changes, "netDebt": net_debt, "earningsEvent": earnings.get("next_earnings_date") or earnings.get("last_earnings_date"), "guidanceAndBalanceSheet": {"guidanceStatus": "No structured company guidance field was available; verify the latest filing and earnings call.", "balanceSheetStatus": "Net debt is positive and should be monitored." if net_debt is not None and net_debt > 0 else "No net-debt deterioration signal was calculated from the available snapshot.", "deteriorationSignals": negative_changes}, "dataQuality": "Statement metrics are provider-reported and should be checked against the latest filing." if metrics or statement_changes else "Detailed financial metrics were unavailable."}


def _pestel_context(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    from app.services.analysis.pestel import sector_pestel

    sector = snapshot.get("sector") or snapshot.get("industry") or "the selected sector"
    market = snapshot.get("exchangeName") or snapshot.get("exchange")
    return sector_pestel(sector, str(market) if market else None)


def _income_evidence(snapshot: dict[str, Any], holding: dict[str, Any], price: float) -> dict[str, Any]:
    rate = _number(snapshot.get("dividendRate") or snapshot.get("trailingAnnualDividendRate"))
    shares = _number(holding.get("shares") or holding.get("Shares")) or 0.0
    average_cost = _number(holding.get("averageCost") or holding.get("Avg Cost")) or 0.0
    book_cost = _number(holding.get("bookCost") or holding.get("Book Cost")) or 0.0
    annual_income = rate * shares if rate is not None and shares > 0 else None
    yield_current = rate / price if rate is not None and price > 0 else None
    yield_cost = rate / average_cost if rate is not None and average_cost > 0 else None
    price_pl = (price * shares - book_cost) if shares > 0 and book_cost > 0 else None
    policy = "No holding data; income fields are instrument-level estimates." if shares <= 0 else "Dividend income is estimated from the current annualized rate; record actual receipts for realized total return."
    return {"annualDividendPerShare": rate, "estimatedAnnualIncome": round(annual_income, 2) if annual_income is not None else None, "currentYieldPct": round(yield_current * 100, 2) if yield_current is not None else None, "yieldOnCostPct": round(yield_cost * 100, 2) if yield_cost is not None else None, "pricePnl": round(price_pl, 2) if price_pl is not None else None, "totalReturnRule": "Sustainable income + operating growth + valuation change", "capitalPreservationStatus": "price loss is not offset by income" if price_pl is not None and price_pl < 0 and (annual_income or 0) < abs(price_pl) else "insufficient evidence or income currently offsets the recorded price change", "dataQuality": policy}


def _thesis_classification(snapshot: dict[str, Any], historical: dict[str, Any], fundamentals: dict[str, Any], income: dict[str, Any], pullback: dict[str, Any]) -> dict[str, Any]:
    revenue = _number(snapshot.get("revenueGrowth"))
    flat = bool(historical.get("isFlat"))
    yield_pct = _number(income.get("currentYieldPct")) or 0.0
    if flat and (revenue is None or revenue < 0.05):
        classification = "Steady or range-bound business; not a confirmed growth entry"
    elif yield_pct >= 4 and (revenue is None or revenue < 0.08):
        classification = "Income / defensive value; total-return and capital-preservation test required"
    elif revenue is not None and revenue >= 0.05:
        classification = "Growth-supported thesis, subject to valuation and catalyst confirmation"
    else:
        classification = "Mixed thesis; price setup and business evidence require confirmation"
    tension = "A peak-based pullback can coexist with a price near its multi-year trading centre; lower price alone is not evidence of value." if flat and (_number(pullback.get("pullbackPct")) or 0) >= 10 else ""
    return {"classification": classification, "tension": tension, "adjustedConclusion": f"{classification}. {tension}".strip(), "evidenceBasis": ["multi-year trading centre", "fundamental growth and statement changes", "income versus capital preservation", "current pullback zone"]}


def _scenario_model(historical: dict[str, Any], fundamentals: dict[str, Any], thesis: dict[str, Any], price: float, strategy: str, risk: str) -> dict[str, Any]:
    centre = _number(historical.get("tradingCentre")) or price
    established = historical.get("establishedRange") or {}
    low = _number(established.get("low")) or price
    high = _number(established.get("high")) or price
    move = _number(historical.get("recentYearAverageDailyMovementPct")) or _number(historical.get("averageDailyMovementPct")) or 0.0
    days = 10 if "short" in strategy.lower() else 30 if "long" in strategy.lower() else 15
    multiplier = 0.75 if "conservative" in risk.lower() else 1.25 if "growth" in risk.lower() else 1.0
    expected = min(35.0, move * sqrt(days) * multiplier)
    return {"method": "Historical percentile range plus average absolute movement; heuristic, not a guaranteed forecast.", "horizonDays": days, "rangeProbabilityProxyPct": historical.get("historicalRangeContainmentPct"), "baseCase": {"low": round(min(low, centre), 2), "high": round(max(high, centre), 2), "anchor": round(centre, 2)}, "worstCase": {"price": round(min(low, price * (1 - expected / 100)), 2), "condition": "operating evidence or the established lower range deteriorates"}, "bestCase": {"price": round(max(high, price * (1 + expected / 100)), 2), "condition": "fundamental evidence stabilizes and price confirms a range break"}, "read": "The base case is a return toward the trading centre unless a catalyst produces a sustained breakout or breakdown."}


def _catalysts(fundamentals: dict[str, Any], earnings: dict[str, Any], news: list[dict[str, Any]]) -> dict[str, Any]:
    positive = []
    negative = []
    for change in fundamentals.get("statementChanges", []):
        target = positive if _number(change.get("changePct")) is not None and _number(change.get("changePct")) > 0 else negative
        target.append(f"{change.get('metric')} changed {float(change.get('changePct') or 0):+.1f}% from {change.get('from')} to {change.get('to')}")
    for item in news[:8]:
        title = str(item.get("Title") or item.get("title") or "").strip()
        sentiment = _number(item.get("Sentiment") if item.get("Sentiment") is not None else item.get("sentiment")) or 0.0
        if title and sentiment > 0.15:
            positive.append(title)
        elif title and sentiment < -0.15:
            negative.append(title)
    event = earnings.get("next_earnings_date") or earnings.get("last_earnings_date")
    return {"upside": positive[:8], "downside": negative[:8], "nextEarningsDate": event, "eventRisk": "Earnings can create gap risk." if event else "No dated earnings event was available."}


def _contradictions(historical: dict[str, Any], pullback: dict[str, Any], thesis: dict[str, Any]) -> list[str]:
    rows = []
    if historical.get("isFlat") and (_number(pullback.get("pullbackPct")) or 0) >= 10:
        rows.append("Recent peak drawdown conflicts with a steady multi-year trading centre; do not equate correction size with growth opportunity.")
    if thesis.get("tension"):
        rows.append(str(thesis["tension"]))
    return rows


def _execution_guidance(intent: str, strategy: str, risk: str, thesis: dict[str, Any], pullback: dict[str, Any], scenarios: dict[str, Any], catalysts: dict[str, Any]) -> dict[str, Any]:
    action = "STAGE / CONFIRM" if intent == "Buy / Add" else "REASSESS / TRIM" if intent == "Sell / Trim" else "HOLD / MONITOR"
    return {"action": action, "staging": pullback.get("guidance"), "confirmation": ["Price confirms the selected zone or range break.", "Fundamental and earnings evidence remain aligned.", "No new downside catalyst invalidates the thesis."], "invalidation": ["Free cash flow, earnings, or leverage deteriorates materially.", "The price breaks the established lower range without a credible recovery catalyst."], "rationale": thesis.get("adjustedConclusion"), "catalystWatch": catalysts.get("upside", [])[:3]}


def _latest_as_of(points: list[tuple[datetime, float]]) -> str | None:
    return points[-1][0].isoformat() if points else None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile / 100
    lower = floor(index)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
