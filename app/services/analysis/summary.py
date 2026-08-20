from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import pandas as pd


@dataclass(frozen=True)
class FocusedStockBrief:
    title: str
    subtitle: str
    markdown: str
    html: str


def build_focused_stock_brief(
    symbol: str,
    frame: pd.DataFrame,
    news: pd.DataFrame,
    snapshot: dict[str, object] | None = None,
    days_until_earnings: int | None = None,
) -> FocusedStockBrief:
    """Build a concise factual brief for the focused stock.

    The brief is intentionally extractive and conservative: it uses only
    locally loaded price data, Yahoo profile/fundamental fields, and scanned
    headline data. Missing facts are disclosed instead of guessed.
    """

    snapshot = snapshot or {}
    name = str(snapshot.get("longName") or snapshot.get("shortName") or symbol)
    currency = str(snapshot.get("currency") or "")
    title = f"\\${symbol} - {name} | Focused Stock Brief"
    subtitle = _subtitle(snapshot)

    latest = frame.sort_values("Date").iloc[-1] if not frame.empty else pd.Series(dtype=object)
    current_price = _num(snapshot.get("currentPrice")) or _num(snapshot.get("regularMarketPrice"))
    if current_price is None and not latest.empty:
        current_price = _num(latest.get("Close"))

    market_lines = [
        f"Current price: {_money(current_price, currency)}",
        f"1W / 1M / 3M move: {_pct(_period_return(frame, days=7))} / {_pct(_period_return(frame, days=30))} / {_pct(_period_return(frame, days=90))}",
        f"52W range: {_money(_num(snapshot.get('fiftyTwoWeekLow')), currency)} - {_money(_num(snapshot.get('fiftyTwoWeekHigh')), currency)}",
        f"Market cap: {_compact_money(_num(snapshot.get('marketCap')), currency)}",
        f"P/E: {_number(_num(snapshot.get('trailingPE')))} trailing / {_number(_num(snapshot.get('forwardPE')))} forward",
        f"Dividend yield: {_yield_pct(_num(snapshot.get('dividendYield')))}",
    ]

    financial_lines = [
        f"Revenue: {_compact_money(_num(snapshot.get('totalRevenue')), currency)}",
        f"Net income to common: {_compact_money(_num(snapshot.get('netIncomeToCommon')), currency)}",
        f"Book value/share: {_money(_num(snapshot.get('bookValue')), currency)}",
        f"Cash: {_compact_money(_num(snapshot.get('totalCash')), currency)}",
        f"Debt: {_compact_money(_num(snapshot.get('totalDebt')), currency)}",
        f"Profit margin: {_pct(_num(snapshot.get('profitMargins')))}",
    ]

    analyst_lines = _analyst_lines(snapshot, current_price)
    valuation_context_lines = _valuation_context_lines(frame, snapshot, current_price, currency)
    dividend_lines = _dividend_lines(snapshot, currency)
    earnings_lines = _earnings_lines(snapshot, days_until_earnings)
    technical_lines = _technical_lines(latest)
    news_lines = _news_lines(news)
    caveats = _caveat_lines(snapshot, news)

    markdown = "\n\n".join(
        [
            f"### {title}",
            subtitle,
            _section("Market Snapshot", market_lines),
            _section("Fundamental Snapshot", financial_lines),
            _section("Analyst Consensus", analyst_lines),
            _section("Valuation In Context", valuation_context_lines),
            _section("Dividend / Income", dividend_lines),
            _section("Earnings Event Risk", earnings_lines),
            _section("What Is Moving The Story", news_lines),
            _section("Technical Read", technical_lines),
            _section("Read The Fine Print", caveats),
        ]
    )
    html = _investor_lens_html(symbol, frame, news, snapshot, name, subtitle, currency, days_until_earnings)
    return FocusedStockBrief(title=title, subtitle=subtitle, markdown=markdown, html=html)


def _investor_lens_html(
    symbol: str,
    frame: pd.DataFrame,
    news: pd.DataFrame,
    snapshot: dict[str, object],
    name: str,
    subtitle: str,
    currency: str,
    days_until_earnings: int | None = None,
) -> str:
    latest = frame.sort_values("Date").iloc[-1] if not frame.empty else pd.Series(dtype=object)
    current_price = _num(snapshot.get("currentPrice")) or _num(snapshot.get("regularMarketPrice"))
    if current_price is None and not latest.empty:
        current_price = _num(latest.get("Close"))

    exchange = _exchange_label(str(snapshot.get("exchange") or ""), symbol)
    sector = str(snapshot.get("sector") or "Sector unavailable")
    industry = str(snapshot.get("industry") or "Industry unavailable")
    volume = _num(latest.get("Volume")) if not latest.empty else None

    hero_metrics = [
        ("Current Price", _html_money(current_price, currency), ""),
        ("Today's Change", _html_pct(_period_return(frame, days=1)), "positive" if (_period_return(frame, days=1) or 0) >= 0 else "negative"),
        ("Market Cap", _html_compact_money(_num(snapshot.get("marketCap")), currency), ""),
        ("Volume", _compact_number(volume), ""),
    ]

    key_metrics = [
        ("52 Week High", _html_money(_num(snapshot.get("fiftyTwoWeekHigh")), currency)),
        ("52 Week Low", _html_money(_num(snapshot.get("fiftyTwoWeekLow")), currency)),
        ("Avg Volume", _compact_number(_num(snapshot.get("averageVolume")))),
        ("Beta", _html_number(_num(snapshot.get("beta")))),
        ("Shares Outstanding", _compact_number(_num(snapshot.get("sharesOutstanding")))),
    ]

    valuation = [
        ("Market Cap", _html_compact_money(_num(snapshot.get("marketCap")), currency)),
        ("Enterprise Value", _html_compact_money(_num(snapshot.get("enterpriseValue")), currency)),
        ("P/E Ratio", _html_number(_num(snapshot.get("trailingPE")))),
        ("Forward P/E", _html_number(_num(snapshot.get("forwardPE")))),
        ("Price/Book", _html_number(_num(snapshot.get("priceToBook")))),
        ("Dividend Yield", _html_yield_pct(_num(snapshot.get("dividendYield")))),
    ]

    analyst_consensus = [
        ("Rating", escape(_rating_label(snapshot))),
        ("Rating Score", _html_number(_num(snapshot.get("recommendationMean")))),
        ("Mean Target", _html_money(_num(snapshot.get("targetMeanPrice")), currency)),
        ("Median Target", _html_money(_num(snapshot.get("targetMedianPrice")), currency)),
        ("Target Upside", _html_pct(_target_upside(snapshot, current_price))),
        ("Analyst Count", _html_number(_num(snapshot.get("numberOfAnalystOpinions")))),
        ("Target Range", _target_range_html(snapshot, currency)),
    ]

    income = [
        ("Dividend Rate", _html_money(_num(snapshot.get("dividendRate") or snapshot.get("trailingAnnualDividendRate")), currency)),
        ("Dividend Yield", _html_yield_pct(_num(snapshot.get("dividendYield") or snapshot.get("trailingAnnualDividendYield")))),
        ("5Y Avg Yield", _html_yield_pct(_num(snapshot.get("fiveYearAvgDividendYield")))),
        ("Payout Ratio", _html_pct(_num(snapshot.get("payoutRatio")))),
        ("Dividend Date", escape(_date_label(snapshot.get("dividendDate") or snapshot.get("dividendDateFormatted")))),
        ("Ex-Dividend", escape(_date_label(snapshot.get("exDividendDate") or snapshot.get("exDividendDateFormatted")))),
    ]

    valuation_context = _valuation_context_lines(frame, snapshot, current_price, currency)

    earnings = _earnings_kv_items(snapshot, days_until_earnings)
    earnings_banner = _earnings_banner(snapshot, days_until_earnings)

    financials = [
        ("Revenue", _html_compact_money(_num(snapshot.get("totalRevenue")), currency)),
        ("Revenue Growth", _html_pct(_num(snapshot.get("revenueGrowth")))),
        ("Gross Margin", _html_pct(_num(snapshot.get("grossMargins")))),
        ("Profit Margin", _html_pct(_num(snapshot.get("profitMargins")))),
        ("Net Income", _html_compact_money(_num(snapshot.get("netIncomeToCommon")), currency)),
        ("Cash", _html_compact_money(_num(snapshot.get("totalCash")), currency)),
        ("Debt", _html_compact_money(_num(snapshot.get("totalDebt")), currency)),
        ("Book Value/Share", _html_money(_num(snapshot.get("bookValue")), currency)),
    ]

    quick_take = _quick_take(snapshot, latest, subtitle)
    red_flags = _red_flags(snapshot, latest, news)
    opportunities = _opportunities(snapshot, latest, news)
    developments = _recent_developments(news)

    hero_cards = "".join(_metric_card(label, value, tone) for label, value, tone in hero_metrics)
    return f"""
    <div class="il-page">
      <section class="il-hero">
        <div class="il-hero-top">
          <span class="il-chip">{escape(exchange or 'LISTED')}</span>
          <div>
            <h2>{escape(name)}</h2>
            <div class="il-symbol">{escape(symbol)} <span>•</span> {escape(sector)} <span>•</span> {escape(industry)}</div>
          </div>
        </div>
        <div class="il-hero-metrics">{hero_cards}</div>
        {earnings_banner}
      </section>

      {_card('Quick Take', quick_take, icon='💡', footer='Based on loaded price data, Yahoo snapshot fields, and cached RSS headlines. Not financial advice.')}

      <div class="il-grid-2">
        {_card('Key Metrics', _kv_grid(key_metrics), icon='📊')}
        {_card('Valuation', _kv_grid(valuation), icon='🔥')}
      </div>

      <div class="il-grid-2">
        {_card('Analyst Consensus', _kv_grid(analyst_consensus), icon='🎯')}
        {_card('Dividend / Income', _kv_grid(income), icon='💵')}
      </div>

      <div class="il-grid-2">
        {_card('Valuation In Context', _bullet_cards(valuation_context, 'info'), icon='⚖️')}
        {_card('Earnings Event Risk', _kv_grid(earnings), icon='📅')}
      </div>

      {_card('Financial Metrics', _kv_grid(financials, columns=2), icon='📊')}

      <div class="il-grid-2">
        {_card('Red Flags', _bullet_cards(red_flags, 'risk'), icon='🚩')}
        {_card('Opportunities', _bullet_cards(opportunities, 'opportunity'), icon='🌟')}
      </div>

      {_card('Recent Developments', _developments_html(developments), icon='📰')}
    </div>
    """


def _subtitle(snapshot: dict[str, object]) -> str:
    sector = snapshot.get("sector")
    industry = snapshot.get("industry")
    summary = str(snapshot.get("longBusinessSummary") or "").strip()
    first_sentence = summary.split(". ")[0].strip()
    parts = [str(part) for part in (sector, industry) if part]
    if first_sentence:
        return first_sentence if first_sentence.endswith(".") else f"{first_sentence}."
    if parts:
        return " | ".join(parts)
    return "Company description unavailable from the current Yahoo snapshot."


def _exchange_label(exchange: str, symbol: str) -> str:
    code = exchange.upper()
    if symbol.upper().endswith(".TO") or code in {"TOR", "TSX", "TSE"}:
        return "TSX"
    if code in {"NMS", "NGM", "NCM", "NASDAQ", "NAS"}:
        return "NASDAQ"
    if code in {"NYQ", "NYSE", "ASE", "AMEX"}:
        return "NYSE"
    return exchange or "LISTED"


def _quick_take(snapshot: dict[str, object], latest: pd.Series, subtitle: str) -> str:
    notes = [escape(subtitle)]
    rsi = _num(latest.get("RSI14")) if not latest.empty else None
    close = _num(latest.get("Close")) if not latest.empty else None
    ema20 = _num(latest.get("EMA20")) if not latest.empty else None
    profit_margin = _num(snapshot.get("profitMargins"))
    debt = _num(snapshot.get("totalDebt"))
    cash = _num(snapshot.get("totalCash"))

    if rsi is not None:
        if rsi < 35:
            notes.append(f"RSI is oversold at {rsi:.1f}, so a reversal needs confirmation rather than assumption.")
        elif rsi > 70:
            notes.append(f"RSI is elevated at {rsi:.1f}, which raises chase risk.")
    if close is not None and ema20 is not None:
        notes.append("Price is above the 20-day EMA." if close >= ema20 else "Price is below the 20-day EMA.")
    if profit_margin is not None:
        notes.append(f"Yahoo snapshot profit margin is {_html_pct(profit_margin)}.")
    if debt is not None and cash is not None:
        notes.append("Debt exceeds cash in the Yahoo snapshot." if debt > cash else "Cash exceeds debt in the Yahoo snapshot.")
    return "<p>" + " ".join(notes) + "</p>"


def _red_flags(snapshot: dict[str, object], latest: pd.Series, news: pd.DataFrame) -> list[str]:
    flags = []
    debt = _num(snapshot.get("totalDebt"))
    cash = _num(snapshot.get("totalCash"))
    profit_margin = _num(snapshot.get("profitMargins"))
    close = _num(latest.get("Close")) if not latest.empty else None
    ema20 = _num(latest.get("EMA20")) if not latest.empty else None
    rsi = _num(latest.get("RSI14")) if not latest.empty else None

    if debt is not None and cash is not None and debt > cash:
        flags.append("Debt is higher than cash; balance-sheet leverage deserves review.")
    if profit_margin is not None and profit_margin < 0:
        flags.append("Profit margin is negative in the Yahoo snapshot.")
    if close is not None and ema20 is not None and close < ema20:
        flags.append("Price is below the 20-day EMA; momentum confirmation is missing.")
    if rsi is not None and rsi < 35:
        flags.append("RSI is oversold; selling pressure may still be active until price bases or reclaims trend.")
    if not news.empty and "SymbolSpecific" in news.columns and (~news["SymbolSpecific"]).any():
        flags.append("Some headlines are market-wide context, not company-specific evidence.")
    if not flags:
        flags.append("No major red flag was detected from the loaded snapshot, price data, and headlines.")
    return flags


def _opportunities(snapshot: dict[str, object], latest: pd.Series, news: pd.DataFrame) -> list[str]:
    items = []
    revenue_growth = _num(snapshot.get("revenueGrowth"))
    gross_margin = _num(snapshot.get("grossMargins"))
    close = _num(latest.get("Close")) if not latest.empty else None
    ema50 = _num(latest.get("EMA50")) if not latest.empty else None
    adx = _num(latest.get("ADX14")) if not latest.empty else None

    if revenue_growth is not None and revenue_growth > 0:
        items.append(f"Revenue growth is positive in the Yahoo snapshot ({_html_pct(revenue_growth)}).")
    if gross_margin is not None and gross_margin > 0.4:
        items.append(f"Gross margin is strong at {_html_pct(gross_margin)}.")
    if close is not None and ema50 is not None and close >= ema50:
        items.append("Price is holding above the 50-day EMA.")
    if adx is not None and adx >= 25:
        items.append(f"ADX at {adx:.0f} suggests a tradeable trend is present.")
    positive_news = [line for line in _news_lines(news) if "(positive)" in line]
    items.extend(positive_news[:2])
    if not items:
        items.append("No clear opportunity catalyst was detected from the currently loaded data.")
    return items


def _recent_developments(news: pd.DataFrame) -> list[dict[str, str]]:
    if news.empty:
        return []
    rows = news.sort_values("Published", ascending=False).head(4)
    items = []
    for _, row in rows.iterrows():
        published = pd.to_datetime(row.get("Published"), errors="coerce")
        date = published.strftime("%b %-d") if not pd.isna(published) else "Recent"
        items.append(
            {
                "title": str(row.get("Title") or "Untitled"),
                "source": str(row.get("Source") or "News"),
                "date": date,
                "url": str(row.get("Url") or ""),
            }
        )
    return items


def _card(title: str, body: str, icon: str = "", footer: str | None = None) -> str:
    footer_html = f'<div class="il-card-footer">{escape(footer)}</div>' if footer else ""
    return f"""
    <section class="il-card">
      <h3>{escape(icon)} {escape(title)}</h3>
      <div class="il-card-body">{body}</div>
      {footer_html}
    </section>
    """


def _metric_card(label: str, value: str, tone: str = "") -> str:
    tone_class = f" il-{tone}" if tone else ""
    return f"""
    <div class="il-metric">
      <div class="il-metric-label">{escape(label)}</div>
      <div class="il-metric-value{tone_class}">{value}</div>
    </div>
    """


def _kv_grid(items: list[tuple[str, str]], columns: int = 1) -> str:
    class_name = "il-kv-grid il-kv-grid-2" if columns == 2 else "il-kv-grid"
    rows = "".join(
        f'<div class="il-kv-row"><span>{escape(label)}</span><strong>{value}</strong></div>'
        for label, value in items
    )
    return f'<div class="{class_name}">{rows}</div>'


def _bullet_cards(items: list[str], tone: str) -> str:
    rows = "".join(
        f'<div class="il-bullet il-{tone}"><span></span><p>{_html_text(item)}</p></div>'
        for item in items
    )
    return f'<div class="il-bullet-list">{rows}</div>'


def _html_text(value: str) -> str:
    return escape(value.replace("\\$", "$"))


def _developments_html(items: list[dict[str, str]]) -> str:
    if not items:
        return "<p>No recent RSS headlines were available for this symbol.</p>"
    rows = []
    for item in items:
        title = escape(item["title"])
        url = escape(item["url"])
        title_html = f'<a href="{url}" target="_blank">{title}</a>' if url else title
        rows.append(
            f'<div class="il-news-item"><strong>{title_html}</strong>'
            f'<span>{escape(item["date"])} • {escape(item["source"])}</span></div>'
        )
    return "".join(rows)


def _analyst_lines(snapshot: dict[str, object], current_price: float | None) -> list[str]:
    target = _num(snapshot.get("targetMeanPrice"))
    if target is None:
        return ["Analyst target data unavailable from the current Yahoo snapshot."]
    return [
        f"Consensus rating: {_rating_label(snapshot)}",
        f"Recommendation score: {_number(_num(snapshot.get('recommendationMean')))}",
        f"Mean price target: {_money(target, str(snapshot.get('currency') or ''))}",
        f"Median price target: {_money(_num(snapshot.get('targetMedianPrice')), str(snapshot.get('currency') or ''))}",
        f"Implied upside/downside: {_pct(_target_upside(snapshot, current_price))}",
        f"Analyst opinions: {_number(_num(snapshot.get('numberOfAnalystOpinions')))}",
        f"Target range: {_money(_num(snapshot.get('targetLowPrice')), str(snapshot.get('currency') or ''))} - {_money(_num(snapshot.get('targetHighPrice')), str(snapshot.get('currency') or ''))}",
    ]


def _valuation_context_lines(
    frame: pd.DataFrame,
    snapshot: dict[str, object],
    current_price: float | None,
    currency: str,
) -> list[str]:
    enterprise_value = _num(snapshot.get("enterpriseValue"))
    revenue = _num(snapshot.get("totalRevenue"))
    lines: list[str] = []
    if frame.empty or "Close" not in frame.columns or current_price is None:
        lines.append("Insufficient price history to compute self-relative valuation context.")
    else:
        closes = frame["Close"].dropna()
        if len(closes) >= 20:
            percentile = float((closes < current_price).mean()) * 100
            lines.append(
                f"Current price sits around the {percentile:.0f}th percentile of its own loaded "
                f"price history ({len(closes)} trading days)."
            )
        else:
            lines.append("Loaded price history is too short for a reliable price-percentile read.")

    low_52w = _num(snapshot.get("fiftyTwoWeekLow"))
    high_52w = _num(snapshot.get("fiftyTwoWeekHigh"))
    if current_price is not None and low_52w is not None and high_52w is not None and high_52w > low_52w:
        position = (current_price - low_52w) / (high_52w - low_52w)
        lines.append(f"Price is {_pct(position)} of the way from its 52-week low to its 52-week high.")

    trailing_pe = _num(snapshot.get("trailingPE"))
    forward_pe = _num(snapshot.get("forwardPE"))
    if trailing_pe is not None and forward_pe is not None and trailing_pe > 0:
        pe_change = forward_pe / trailing_pe - 1
        direction = "cheaper" if pe_change < 0 else "more expensive"
        lines.append(
            f"Forward P/E ({forward_pe:.1f}) makes the stock look {abs(pe_change):.0%} "
            f"{direction} than trailing P/E ({trailing_pe:.1f})."
        )

    lines.extend(
        [
        f"Enterprise value / revenue: {_number(_ratio(enterprise_value, revenue))}",
        f"Price/book: {_number(_num(snapshot.get('priceToBook')))}",
        f"Net margin: {_pct(_num(snapshot.get('profitMargins')))}",
        f"Revenue growth: {_pct(_num(snapshot.get('revenueGrowth')))}",
        f"Enterprise value: {_compact_money(enterprise_value, currency)}",
        "This is a self-relative valuation read, not a sector-peer valuation model.",
        ]
    )
    return lines


def _dividend_lines(snapshot: dict[str, object], currency: str) -> list[str]:
    rate = _num(snapshot.get("dividendRate") or snapshot.get("trailingAnnualDividendRate"))
    yield_value = _num(snapshot.get("dividendYield") or snapshot.get("trailingAnnualDividendYield"))
    if rate is None and yield_value is None:
        return ["Dividend/income data unavailable from the current Yahoo snapshot."]
    return [
        f"Dividend rate: {_money(rate, currency)}",
        f"Dividend yield: {_yield_pct(yield_value)}",
        f"5Y average dividend yield: {_yield_pct(_num(snapshot.get('fiveYearAvgDividendYield')))}",
        f"Payout ratio: {_pct(_num(snapshot.get('payoutRatio')))}",
        f"Dividend date: {_date_label(snapshot.get('dividendDate') or snapshot.get('dividendDateFormatted'))}",
        f"Ex-dividend date: {_date_label(snapshot.get('exDividendDate') or snapshot.get('exDividendDateFormatted'))}",
    ]


def _earnings_lines(snapshot: dict[str, object], days_until_earnings: int | None = None) -> list[str]:
    event_date, event_kind = _earnings_date_value(snapshot)
    if days_until_earnings is None and event_date is None:
        return ["Next earnings date unavailable from the current Yahoo snapshot."]
    if days_until_earnings is not None:
        if days_until_earnings < 0:
            return [f"Most recently reported earnings {abs(days_until_earnings)} day(s) ago."]
        if days_until_earnings <= 10:
            return [
                f"Earnings expected in {days_until_earnings} day(s) — near-term volatility risk is elevated regardless of the technical setup."
            ]
        return [f"Next earnings expected in {days_until_earnings} day(s) — outside the near-term risk window."]
    if event_kind == "last":
        days = _days_from_today(event_date)
        if days is not None and days < 0:
            return [f"Most recently reported earnings {abs(days)} day(s) ago."]
        return [f"Most recent known earnings date: {_date_label(event_date)}"]
    return [
        f"Next known earnings date: {_date_label(event_date)}",
        f"Event window: {_earnings_window_label(snapshot, days_until_earnings)}",
    ]


def _rating_label(snapshot: dict[str, object]) -> str:
    rating = str(snapshot.get("recommendationKey") or snapshot.get("averageAnalystRating") or "").strip()
    return rating.replace("_", " ").title() if rating else "n/a"


def _target_upside(snapshot: dict[str, object], current_price: float | None) -> float | None:
    target = _num(snapshot.get("targetMeanPrice"))
    if target is None or current_price is None or current_price == 0:
        return None
    return target / current_price - 1


def _target_range_html(snapshot: dict[str, object], currency: str) -> str:
    low = _html_money(_num(snapshot.get("targetLowPrice")), currency)
    high = _html_money(_num(snapshot.get("targetHighPrice")), currency)
    if low == "n/a" and high == "n/a":
        return "n/a"
    return f"{low} - {high}"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _earnings_date_value(snapshot: dict[str, object]) -> tuple[object | None, str]:
    last = snapshot.get("lastEarningsDate")
    if last not in {None, ""}:
        return last, "last"
    for key in ("nextEarningsDate", "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        value = snapshot.get(key)
        if value not in {None, ""}:
            days = _days_from_today(value)
            if days is not None and days < 0:
                return value, "last"
            return value, "next"
    return None, "unknown"


def _earnings_kv_items(snapshot: dict[str, object], days_until_earnings: int | None = None) -> list[tuple[str, str]]:
    event_date, event_kind = _earnings_date_value(snapshot)
    if days_until_earnings is not None and days_until_earnings >= 0:
        label = "Next Earnings"
    elif event_kind == "last":
        label = "Last Reported"
    else:
        label = "Earnings Date"
    return [
        (label, escape(_date_label(event_date))),
        ("Event Window", escape(_earnings_window_label(snapshot, days_until_earnings))),
    ]


def _earnings_window_label(snapshot: dict[str, object], days_until_earnings: int | None = None) -> str:
    if days_until_earnings is not None:
        if days_until_earnings < 0:
            return f"Last known date was {abs(days_until_earnings)} days ago"
        if days_until_earnings <= 7:
            return f"High event risk: within {days_until_earnings} days"
        if days_until_earnings <= 21:
            return f"Moderate event risk: {days_until_earnings} days away"
        return f"{days_until_earnings} days away"
    value, event_kind = _earnings_date_value(snapshot)
    if event_kind == "last":
        days = _days_from_today(value)
        if days is not None and days < 0:
            return f"Last reported {abs(days)} days ago; not a forward event risk"
        return "Most recent known report"
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed) and isinstance(value, (int, float)):
        parsed = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    if pd.isna(parsed):
        return "n/a"
    days = (parsed.date() - pd.Timestamp.utcnow().date()).days
    if days < 0:
        return f"Last known date was {abs(days)} days ago"
    if days <= 7:
        return f"High event risk: within {days} days"
    if days <= 21:
        return f"Moderate event risk: {days} days away"
    return f"{days} days away"


def _earnings_banner(snapshot: dict[str, object], days_until_earnings: int | None = None) -> str:
    label = _earnings_window_label(snapshot, days_until_earnings)
    if "High event risk" not in label:
        return ""
    line = _earnings_lines(snapshot, days_until_earnings)[0]
    return f'<div class="il-earnings-banner">Earnings Event Risk: {escape(line)}</div>'


def _date_label(value: object) -> str:
    if value in {None, ""}:
        return "n/a"
    if isinstance(value, (int, float)) and value > 10_000:
        parsed = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "n/a"
    return parsed.strftime("%b %-d, %Y")


def _days_from_today(value: object) -> int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)) and value > 10_000:
        parsed = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if parsed is None or pd.isna(parsed):
        return None
    return (parsed.date() - pd.Timestamp.utcnow().date()).days


def _technical_lines(latest: pd.Series) -> list[str]:
    if latest.empty:
        return ["No price history loaded."]

    close = _num(latest.get("Close"))
    ema20 = _num(latest.get("EMA20"))
    ema50 = _num(latest.get("EMA50"))
    rsi = _num(latest.get("RSI14"))
    adx = _num(latest.get("ADX14"))
    volume_ratio = _num(latest.get("VolumeRatio"))

    position = "n/a"
    if close is not None and ema20 is not None and ema50 is not None:
        if close >= ema20 >= ema50:
            position = "price is above the 20-day and 50-day EMA"
        elif close < ema20:
            position = "price is below the 20-day EMA"
        else:
            position = "price is between the 20-day and 50-day EMA"

    return [
        f"RSI 14: {_number(rsi)}",
        f"20D / 50D EMA: {_money(ema20, '')} / {_money(ema50, '')}",
        f"Trend location: {position}",
        f"ADX 14: {_number(adx)}",
        f"Volume vs 20D average: {_number(volume_ratio)}x",
    ]


def _news_lines(news: pd.DataFrame) -> list[str]:
    if news.empty:
        return ["No recent RSS headlines were available for this symbol."]

    scoped = news
    if "SymbolSpecific" in news.columns:
        symbol_specific = news[news["SymbolSpecific"]]
        if not symbol_specific.empty:
            scoped = symbol_specific

    rows = scoped.sort_values("Published", ascending=False).head(5)
    lines = []
    for _, row in rows.iterrows():
        sentiment = _num(row.get("Sentiment"))
        sentiment_label = "positive" if sentiment is not None and sentiment > 0.15 else "negative" if sentiment is not None and sentiment < -0.15 else "neutral"
        lines.append(f"{row.get('Source', 'News')}: {row.get('Title', 'Untitled')} ({sentiment_label})")
    return lines or ["No recent RSS headlines were available for this symbol."]


def _caveat_lines(snapshot: dict[str, object], news: pd.DataFrame) -> list[str]:
    lines = [
        "This brief is generated from Yahoo profile/fundamental fields, local price history, and scanned RSS headlines; it is not a substitute for reading the issuer's filings.",
        "Rule-based scores and ML probabilities in this dashboard are decision-support signals, not true probabilities of profit.",
    ]

    total_debt = _num(snapshot.get("totalDebt"))
    total_cash = _num(snapshot.get("totalCash"))
    if total_debt is not None and total_cash is not None and total_debt > total_cash:
        lines.append("Debt is higher than cash in the Yahoo snapshot, so balance-sheet risk deserves review.")

    if not news.empty and "SymbolSpecific" in news.columns and (~news["SymbolSpecific"]).any():
        lines.append("Market-wide headlines are shown for context but excluded from symbol-specific sentiment when possible.")

    if not snapshot:
        lines.append("Yahoo profile/fundamental data was unavailable, so this brief is heavier on price/news evidence than company financials.")

    return lines


def _section(title: str, lines: list[str]) -> str:
    bullets = "\n".join(f"- {line}" for line in lines)
    return f"**{title}:**\n{bullets}"


def _period_return(frame: pd.DataFrame, days: int) -> float | None:
    if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
        return None
    data = frame.copy().sort_values("Date")
    data["Date"] = pd.to_datetime(data["Date"])
    latest = data.iloc[-1]
    cutoff = latest["Date"] - pd.Timedelta(days=days)
    earlier = data[data["Date"] <= cutoff]
    if earlier.empty:
        return None
    start = _num(earlier.iloc[-1]["Close"])
    end = _num(latest["Close"])
    if start is None or end is None or start == 0:
        return None
    return end / start - 1


def _num(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _money(value: float | None, currency: str) -> str:
    if value is None:
        return "n/a"
    prefix = "\\$" if currency in {"", "CAD", "USD"} else ""
    suffix = f" {currency}" if currency and currency not in {"CAD", "USD"} else ""
    return f"{prefix}{value:,.2f}{suffix}"


def _compact_money(value: float | None, currency: str) -> str:
    if value is None:
        return "n/a"
    prefix = "\\$" if currency in {"", "CAD", "USD"} else ""
    suffix_currency = "" if currency in {"", "CAD", "USD"} else f" {currency}"
    abs_value = abs(value)
    for suffix, divisor in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
        if abs_value >= divisor:
            return f"{prefix}{value / divisor:,.2f}{suffix}{suffix_currency}"
    return f"{prefix}{value:,.0f}{suffix_currency}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _yield_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    normalized = value / 100 if value > 1 else value
    return f"{normalized:.2%}"


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _html_money(value: float | None, currency: str) -> str:
    return escape(_money(value, currency).replace("\\$", "$"))


def _html_compact_money(value: float | None, currency: str) -> str:
    return escape(_compact_money(value, currency).replace("\\$", "$"))


def _html_pct(value: float | None) -> str:
    return escape(_pct(value))


def _html_yield_pct(value: float | None) -> str:
    return escape(_yield_pct(value))


def _html_number(value: float | None) -> str:
    return escape(_number(value))


def _compact_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    for suffix, divisor in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs_value >= divisor:
            return escape(f"{value / divisor:,.2f}{suffix}")
    return escape(f"{value:,.0f}")
