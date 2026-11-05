from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exchange:
    name: str
    yahoo_suffix: str
    examples: tuple[str, ...]
    benchmark_symbol: str
    benchmark_name: str


EXCHANGES = {
    "TSX": Exchange(
        "Toronto Stock Exchange",
        ".TO",
        ("ENB.TO", "SHOP.TO", "RY.TO", "TD.TO", "BNS.TO"),
        benchmark_symbol="^GSPTSE",
        benchmark_name="S&P/TSX Composite",
    ),
    "NASDAQ": Exchange(
        "NASDAQ",
        "",
        ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN"),
        benchmark_symbol="^IXIC",
        benchmark_name="NASDAQ Composite",
    ),
    "NYSE": Exchange(
        "New York Stock Exchange",
        "",
        ("JPM", "DIS", "BA", "KO", "JNJ"),
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500",
    ),
}

DEFAULT_BENCHMARK_SYMBOL = "^GSPC"
DEFAULT_BENCHMARK_NAME = "S&P 500"


def benchmark_for_symbol(symbol: str) -> tuple[str, str]:
    """Pick a sensible benchmark index for a given ticker's suffix.

    Falls back to the S&P 500 for anything not matching a known TSX-style
    suffix, since most NASDAQ/NYSE symbols carry no suffix at all.
    """

    cleaned = symbol.strip().upper()
    if cleaned.endswith(".TO") or cleaned.endswith(".V"):
        return EXCHANGES["TSX"].benchmark_symbol, EXCHANGES["TSX"].benchmark_name
    return DEFAULT_BENCHMARK_SYMBOL, DEFAULT_BENCHMARK_NAME

DEFAULT_WATCHLISTS = {
    "Mixed": ("ENB.TO", "SHOP.TO", "RY.TO", "AAPL", "MSFT", "NVDA", "JPM", "TSLA"),
    **{exchange: config.examples for exchange, config in EXCHANGES.items()},
}


def normalize_symbol(symbol: str, exchange: str | None = None) -> str:
    """Normalize a user-entered ticker for Yahoo Finance.

    Yahoo Finance uses `.TO` for many TSX listings. NASDAQ and NYSE symbols usually
    have no suffix, so we leave them as typed after uppercasing.
    """

    cleaned = symbol.strip().upper()
    if not cleaned:
        return cleaned

    if exchange == "TSX" and "." not in cleaned:
        return f"{cleaned}.TO"
    return cleaned


def split_symbols(raw: str, exchange: str | None = None) -> list[str]:
    return [
        normalize_symbol(part, exchange)
        for part in raw.replace("\n", ",").split(",")
        if normalize_symbol(part, exchange)
    ]


# ---------------------------------------------------------------------------
# Symbol classification: market, currency, exchange timezone, and session model
# ---------------------------------------------------------------------------

EXCHANGE_TIMEZONES: dict[str, str] = {
    "TSX": "America/Toronto",
    "NASDAQ": "America/New_York",
    "NYSE": "America/New_York",
    "US": "America/New_York",
    "INDEX": "America/New_York",
    "CRYPTO": "UTC",
    "TOKENIZED": "UTC",
    "FX": "UTC",
    "OTHER": "UTC",
}

KNOWN_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LTC", "BNB", "LINK", "MATIC", "POL", "USDT",
    "USDC", "TRX", "TON", "SHIB", "BCH", "XLM", "ATOM", "UNI", "NEAR", "APT", "ARB", "OP", "SUI", "HBAR",
}

CONTINUOUS_MARKETS = {"CRYPTO", "TOKENIZED"}


def classify_symbol(symbol: str) -> dict[str, object]:
    """Describe how a ticker trades so timestamps can be shown in the right context.

    Every supported market and currency gets the same shape: the market label,
    the settlement currency, the exchange's IANA timezone, and whether the
    instrument trades continuously (tokenized ``-USD`` equities and crypto
    pairs never close, so their intraday quote timestamps are only meaningful
    when shown with an explicit UTC anchor next to the viewer's local time).
    """

    cleaned = str(symbol or "").strip().upper()
    result: dict[str, object] = {
        "symbol": cleaned,
        "market": "US",
        "currency": "USD",
        "assetClass": "equity",
        "timezone": EXCHANGE_TIMEZONES["US"],
        "tradesContinuously": False,
        "sessionModel": "exchange-hours",
    }
    if not cleaned:
        return result
    if cleaned.endswith(".TO") or cleaned.endswith(".V"):
        result.update({"market": "TSX", "currency": "CAD", "timezone": EXCHANGE_TIMEZONES["TSX"]})
        return result
    if cleaned.startswith("^"):
        is_canadian = cleaned in {"^GSPTSE", "^TXCX", "^SPTSE"}
        result.update({
            "market": "INDEX",
            "assetClass": "index",
            "currency": "CAD" if is_canadian else "USD",
            "timezone": EXCHANGE_TIMEZONES["TSX"] if is_canadian else EXCHANGE_TIMEZONES["INDEX"],
        })
        return result
    if cleaned.endswith("=X"):
        pair = cleaned[:-2]
        result.update({
            "market": "FX",
            "assetClass": "fx",
            "currency": pair[-3:] if len(pair) >= 6 else "USD",
            "timezone": EXCHANGE_TIMEZONES["FX"],
            "tradesContinuously": False,
            "sessionModel": "24x5",
        })
        return result
    for quote_currency in ("USD", "CAD"):
        suffix = f"-{quote_currency}"
        if cleaned.endswith(suffix):
            base = cleaned[: -len(suffix)]
            tokenized = base not in KNOWN_CRYPTO_BASES and base.endswith("X") and len(base) >= 4
            result.update({
                "market": "TOKENIZED" if tokenized else "CRYPTO",
                "assetClass": "tokenized-equity" if tokenized else "crypto",
                "currency": quote_currency,
                "timezone": "UTC",
                "tradesContinuously": True,
                "sessionModel": "24x7",
            })
            return result
    for market in ("NASDAQ", "NYSE"):
        if cleaned in DEFAULT_WATCHLISTS.get(market, ()):
            result.update({"market": market, "timezone": EXCHANGE_TIMEZONES[market]})
            return result
    if "." in cleaned:
        result.update({"market": "OTHER", "timezone": EXCHANGE_TIMEZONES["OTHER"]})
    return result


def market_timezone(symbol: str) -> str:
    return str(classify_symbol(symbol)["timezone"])


def session_state(symbol: str, at: "datetime | None" = None) -> str:
    """Return a simple session label: open, closed, 24x7, or 24x5.

    This is a calendar/time-of-day check for the exchange's regular session;
    exchange holidays are not modelled here, so ``open`` means "regular session
    hours on a weekday" rather than a guaranteed live session.
    """

    from datetime import datetime, timezone as _timezone
    from zoneinfo import ZoneInfo

    info = classify_symbol(symbol)
    model = str(info["sessionModel"])
    if model == "24x7":
        return "24x7"
    moment = at or datetime.now(_timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_timezone.utc)
    if model == "24x5":
        local = moment.astimezone(ZoneInfo("America/New_York"))
        weekday = local.weekday()
        if weekday == 5 or (weekday == 6 and local.hour < 17) or (weekday == 4 and local.hour >= 17):
            return "closed"
        return "24x5"
    local = moment.astimezone(ZoneInfo(str(info["timezone"])))
    if local.weekday() >= 5:
        return "closed"
    minutes = local.hour * 60 + local.minute
    return "open" if 9 * 60 + 30 <= minutes < 16 * 60 else "closed"


# ---------------------------------------------------------------------------
# Curated explorer universe: liquid, sector-tagged names per supported market
# ---------------------------------------------------------------------------

EXPLORER_UNIVERSE: dict[str, tuple[tuple[str, str], ...]] = {
    "TSX": (
        ("RY.TO", "Financial Services"), ("TD.TO", "Financial Services"), ("BNS.TO", "Financial Services"), ("MFC.TO", "Financial Services"),
        ("SHOP.TO", "Technology"), ("CSU.TO", "Technology"), ("OTEX.TO", "Technology"),
        ("ENB.TO", "Energy"), ("CNQ.TO", "Energy"), ("SU.TO", "Energy"), ("TRP.TO", "Energy"),
        ("CNR.TO", "Industrials"), ("CP.TO", "Industrials"), ("WCN.TO", "Industrials"),
        ("L.TO", "Consumer Defensive"), ("ATD.TO", "Consumer Defensive"), ("DOL.TO", "Consumer Defensive"),
        ("QSR.TO", "Consumer Cyclical"), ("MG.TO", "Consumer Cyclical"), ("GIL.TO", "Consumer Cyclical"),
        ("FTS.TO", "Utilities"), ("EMA.TO", "Utilities"), ("H.TO", "Utilities"),
        ("BCE.TO", "Communication Services"), ("T.TO", "Communication Services"), ("RCI-B.TO", "Communication Services"),
        ("CAR-UN.TO", "Real Estate"), ("REI-UN.TO", "Real Estate"), ("GRT-UN.TO", "Real Estate"),
        ("ABX.TO", "Basic Materials"), ("NTR.TO", "Basic Materials"), ("TECK-B.TO", "Basic Materials"),
        ("WELL.TO", "Healthcare"), ("BHC.TO", "Healthcare"),
    ),
    "NASDAQ": (
        ("AAPL", "Technology"), ("MSFT", "Technology"), ("NVDA", "Technology"), ("AMD", "Technology"), ("AVGO", "Technology"), ("INTC", "Technology"),
        ("AMZN", "Consumer Cyclical"), ("TSLA", "Consumer Cyclical"), ("SBUX", "Consumer Cyclical"),
        ("GOOGL", "Communication Services"), ("META", "Communication Services"), ("NFLX", "Communication Services"), ("CMCSA", "Communication Services"),
        ("COST", "Consumer Defensive"), ("PEP", "Consumer Defensive"), ("MDLZ", "Consumer Defensive"),
        ("AMGN", "Healthcare"), ("GILD", "Healthcare"), ("REGN", "Healthcare"), ("VRTX", "Healthcare"), ("ISRG", "Healthcare"),
        ("PYPL", "Financial Services"), ("CME", "Financial Services"),
        ("HON", "Industrials"), ("CSX", "Industrials"), ("PCAR", "Industrials"),
        ("XEL", "Utilities"), ("EXC", "Utilities"),
        ("FANG", "Energy"),
        ("LIN", "Basic Materials"),
    ),
    "NYSE": (
        ("JPM", "Financial Services"), ("BAC", "Financial Services"), ("V", "Financial Services"), ("GS", "Financial Services"), ("BRK-B", "Financial Services"),
        ("JNJ", "Healthcare"), ("PFE", "Healthcare"), ("UNH", "Healthcare"), ("LLY", "Healthcare"), ("ABBV", "Healthcare"), ("MRK", "Healthcare"),
        ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
        ("CAT", "Industrials"), ("BA", "Industrials"), ("GE", "Industrials"), ("UNP", "Industrials"), ("RTX", "Industrials"),
        ("KO", "Consumer Defensive"), ("PG", "Consumer Defensive"), ("WMT", "Consumer Defensive"),
        ("DIS", "Consumer Cyclical"), ("HD", "Consumer Cyclical"), ("NKE", "Consumer Cyclical"), ("MCD", "Consumer Cyclical"),
        ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"),
        ("VZ", "Communication Services"), ("T", "Communication Services"),
        ("CRM", "Technology"), ("ORCL", "Technology"), ("IBM", "Technology"),
        ("PLD", "Real Estate"), ("O", "Real Estate"), ("AMT", "Real Estate"),
        ("FCX", "Basic Materials"), ("NEM", "Basic Materials"),
    ),
}

EXPLORER_SECTORS: tuple[str, ...] = (
    "Technology",
    "Financial Services",
    "Healthcare",
    "Energy",
    "Industrials",
    "Consumer Defensive",
    "Consumer Cyclical",
    "Utilities",
    "Communication Services",
    "Real Estate",
    "Basic Materials",
)

# Sector ordering used when the investor lets the analysis infer a mix. The
# order expresses a tilt, not an exclusion: every sector can still appear.
SECTOR_TILTS: dict[str, tuple[str, ...]] = {
    "Conservative": ("Utilities", "Consumer Defensive", "Healthcare", "Financial Services", "Communication Services", "Real Estate", "Industrials", "Energy", "Basic Materials", "Technology", "Consumer Cyclical"),
    "Balanced": ("Technology", "Financial Services", "Healthcare", "Industrials", "Consumer Defensive", "Energy", "Communication Services", "Consumer Cyclical", "Utilities", "Real Estate", "Basic Materials"),
    "Growth": ("Technology", "Consumer Cyclical", "Communication Services", "Healthcare", "Industrials", "Financial Services", "Energy", "Basic Materials", "Consumer Defensive", "Utilities", "Real Estate"),
    "Income": ("Utilities", "Energy", "Financial Services", "Real Estate", "Communication Services", "Consumer Defensive", "Healthcare", "Industrials", "Basic Materials", "Technology", "Consumer Cyclical"),
}


def explorer_universe_rows(markets: "list[str] | tuple[str, ...] | None" = None) -> list[dict[str, str]]:
    """Return sector-tagged universe rows for the selected markets."""

    selected = [str(market).upper() for market in (markets or list(EXCHANGES)) if str(market).upper() in EXPLORER_UNIVERSE]
    rows: list[dict[str, str]] = []
    for market in selected or list(EXPLORER_UNIVERSE):
        for symbol, sector in EXPLORER_UNIVERSE[market]:
            rows.append({"symbol": symbol, "sector": sector, "market": market})
    return rows
