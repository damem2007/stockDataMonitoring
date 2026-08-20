from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
YAHOO_CACHE_DIR = DATA_DIR / "cache" / "yahoo"
YAHOO_SNAPSHOT_CACHE_DIR = DATA_DIR / "cache" / "yahoo_snapshot"
EARNINGS_CACHE_DIR = DATA_DIR / "cache" / "earnings"
YFINANCE_CACHE_DIR = DATA_DIR / "cache" / "yfinance"
KAGGLE_DIR = DATA_DIR / "kaggle"
WATCHLIST_DIR = DATA_DIR / "watchlists"
SNAPSHOT_SCHEMA_VERSION = 3
EARNINGS_SCHEMA_VERSION = 2


def configure_yfinance_cache() -> None:
    """Keep yfinance's internal SQLite cache inside the project cache tree."""

    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for setter_name in ("set_tz_cache_location", "set_cookie_cache_location"):
        setter = getattr(yf, setter_name, None)
        if callable(setter):
            try:
                setter(str(YFINANCE_CACHE_DIR))
            except Exception:
                pass


configure_yfinance_cache()


@dataclass(frozen=True)
class SourceStatus:
    source: str
    message: str
    ok: bool = True


def read_watchlist(path: Path = WATCHLIST_DIR / "default.json") -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [str(symbol).strip().upper() for symbol in payload.get("symbols", []) if symbol]


def _cache_path(symbol: str, period: str, interval: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("/", "-")
    return YAHOO_CACHE_DIR / f"{safe}_{period}_{interval}.csv"


def history_cache_is_fresh(frame: pd.DataFrame, interval: str, today: date | None = None) -> bool:
    """Return whether cached history is recent enough to drive live decisions."""

    if frame.empty or "Date" not in frame.columns:
        return False
    prepared = _prepare_history(frame)
    if prepared.empty:
        return False
    today = today or _today_utc()
    latest_date = pd.to_datetime(prepared["Date"].max(), errors="coerce")
    if latest_date is None or pd.isna(latest_date):
        return False
    latest = latest_date.date()
    if latest > today:
        return False
    return today - latest <= _freshness_tolerance(interval)


def history_is_plausible(frame: pd.DataFrame, interval: str, today: date | None = None) -> bool:
    """Guard against caching obviously wrong market data."""

    if frame.empty:
        return False
    prepared = _prepare_history(frame)
    if prepared.empty:
        return False
    price_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close"] if col in prepared.columns]
    if price_cols and (prepared[price_cols] < 0).any().any():
        return False
    if "Volume" in prepared.columns and (prepared["Volume"].dropna() < 0).any():
        return False
    if prepared["Date"].max().date() > (today or _today_utc()):
        return False
    return history_cache_is_fresh(prepared, interval, today=today)


def fetch_yahoo_history(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    refresh: bool = False,
) -> tuple[pd.DataFrame, SourceStatus]:
    """Fetch Yahoo Finance history and store a local CSV cache.

    The cache makes repeated analysis reproducible and gives the app an offline
    fallback when Yahoo is unavailable.
    """

    YAHOO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(symbol, period, interval)

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, parse_dates=["Date"])
        prepared = _prepare_history(frame)
        if history_cache_is_fresh(prepared, interval):
            return prepared, SourceStatus("Yahoo Finance cache", str(cache_path))

    ticker = yf.Ticker(symbol)
    try:
        frame = ticker.history(period=period, interval=interval, auto_adjust=False)
    except Exception as exc:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            return _prepare_history(cached), SourceStatus(
                "Yahoo Finance cache",
                f"Live fetch failed ({type(exc).__name__}); using cached file {cache_path}",
                ok=False,
            )
        return pd.DataFrame(), SourceStatus(
            "Yahoo Finance",
            f"Live fetch failed: {type(exc).__name__}: {exc}",
            ok=False,
        )
    if frame.empty:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            return _prepare_history(cached), SourceStatus(
                "Yahoo Finance cache",
                f"Live fetch empty; using cached file {cache_path}",
                ok=False,
            )
        return pd.DataFrame(), SourceStatus("Yahoo Finance", "No data returned", ok=False)

    frame = frame.reset_index()
    if "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    if not history_is_plausible(frame, interval):
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            return _prepare_history(cached), SourceStatus(
                "Yahoo Finance cache",
                f"Live fetch was stale or implausible for {interval}; using cached file {cache_path}",
                ok=False,
            )
        return pd.DataFrame(), SourceStatus(
            "Yahoo Finance",
            "Live fetch was stale or implausible, so it was not cached",
            ok=False,
        )
    frame.to_csv(cache_path, index=False)
    return _prepare_history(frame), SourceStatus(
        "Yahoo Finance",
        f"Fetched {len(frame):,} rows at {datetime.now(timezone.utc).isoformat()}",
    )


def fetch_yahoo_quote(symbol: str) -> tuple[dict[str, object], SourceStatus]:
    """Fetch a short-lived quote for alert checks.

    Unlike historical data, quotes are intentionally not persisted to disk; the
    app-level cache gives them a low TTL so alerts can re-check without turning
    the data directory into a stale quote dump.
    """

    try:
        fast_info = yf.Ticker(symbol).fast_info
        price = _fast_info_get(fast_info, "last_price") or _fast_info_get(fast_info, "lastPrice")
        volume = _fast_info_get(fast_info, "last_volume") or _fast_info_get(fast_info, "lastVolume")
        market_time = _fast_info_get(fast_info, "last_trade_time") or _fast_info_get(fast_info, "lastTradeTime")
    except Exception as exc:
        return {}, SourceStatus(
            "Yahoo Finance quote",
            f"Quote fetch failed: {type(exc).__name__}: {exc}",
            ok=False,
        )

    quote = {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "market_time": str(market_time) if market_time else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return quote, SourceStatus("Yahoo Finance quote", f"Fetched near-real-time quote for {symbol}")


def fetch_yahoo_snapshot(symbol: str, refresh: bool = False) -> tuple[dict[str, object], SourceStatus]:
    """Fetch a compact Yahoo Finance profile/fundamental snapshot.

    This intentionally stores only the scalar `Ticker.info` fields we can use
    in the stock brief. Quarterly filings are uneven across exchanges and
    Yahoo occasionally changes schemas, so this function is defensive and
    never lets profile fetch failures crash the dashboard.
    """

    YAHOO_SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _snapshot_cache_path(symbol)
    if cache_path.exists() and not refresh:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("snapshot_schema_version") == SNAPSHOT_SCHEMA_VERSION:
                return payload, SourceStatus("Yahoo Finance snapshot cache", str(cache_path))
        except Exception:
            pass

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return payload, SourceStatus(
                    "Yahoo Finance snapshot cache",
                    f"Live profile fetch failed ({type(exc).__name__}); using cached file {cache_path}",
                    ok=False,
                )
            except Exception:
                pass
        return {}, SourceStatus(
            "Yahoo Finance snapshot",
            f"Live profile fetch failed: {type(exc).__name__}: {exc}",
            ok=False,
        )

    snapshot = _clean_snapshot(info)
    _normalize_snapshot_dates(snapshot)
    snapshot["symbol"] = symbol
    snapshot["snapshot_schema_version"] = SNAPSHOT_SCHEMA_VERSION
    snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cache_path.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot, SourceStatus("Yahoo Finance snapshot", f"Fetched profile fields for {symbol}")


def _prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return frame


def _snapshot_cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("/", "-")
    return YAHOO_SNAPSHOT_CACHE_DIR / f"{safe}_snapshot.json"


def _clean_snapshot(info: dict[str, object]) -> dict[str, object]:
    wanted = {
        "longName",
        "shortName",
        "sector",
        "industry",
        "longBusinessSummary",
        "currency",
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "averageVolume",
        "beta",
        "totalRevenue",
        "revenueGrowth",
        "grossMargins",
        "profitMargins",
        "netIncomeToCommon",
        "totalCash",
        "totalDebt",
        "bookValue",
        "sharesOutstanding",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "regularMarketPrice",
        "currentPrice",
        "quoteType",
        "exchange",
        "website",
         # Analyst consensus (Yahoo Finance's Analysis tab surfaces these
        # prominently; the dashboard previously had no independent
        # analyst-driven signal alongside the rule-based/ML ones).
        "targetMeanPrice",
        "targetHighPrice",
        "targetLowPrice",
        "targetMedianPrice",
        "recommendationKey",
        "recommendationMean",
        "numberOfAnalystOpinions",
        # Dividend / income (relevant for the TSX dividend names in the
        # default watchlist, e.g. ENB.TO, RY.TO, BNS.TO).
        "payoutRatio",
        "fiveYearAvgDividendYield",
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "dividendRate",
        "dividendDate",
        "exDividendDate",
    }
    clean: dict[str, object] = {}
    for key in wanted:
        value = info.get(key)
        if _is_json_scalar(value):
            clean[key] = value
    return clean


def fetch_earnings_calendar(symbol: str, refresh: bool = False) -> tuple[dict[str, object], SourceStatus]:
    """Fetch the next scheduled earnings date, if Yahoo publishes one.
 
    Earnings releases are one of the largest sources of near-term price
    volatility for an individual stock — Yahoo Finance surfaces this
    prominently, and a signal that's silent about "earnings in 3 days" is
    missing a material piece of context. This is intentionally minimal (just
    the date) since EPS/revenue estimate fields are inconsistent across
    yfinance versions and tickers; callers should treat a missing date as
    "unknown," not "no upcoming earnings."
    """
 
    EARNINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _earnings_cache_path(symbol)
    if cache_path.exists() and not refresh:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload = _normalize_earnings_payload(payload)
            if payload.get("earnings_schema_version") == EARNINGS_SCHEMA_VERSION:
                return payload, SourceStatus("Yahoo Finance earnings cache", str(cache_path))
        except Exception:
            pass
 
    try:
        ticker = yf.Ticker(symbol)
        next_date = None
        try:
            calendar = ticker.calendar
            if isinstance(calendar, dict):
                raw = calendar.get("Earnings Date")
                if isinstance(raw, (list, tuple)) and raw:
                    next_date = raw[0]
            elif calendar is not None and not getattr(calendar, "empty", True):
                # Older yfinance versions return a DataFrame with dates as columns.
                candidate = calendar.get("Earnings Date") if hasattr(calendar, "get") else None
                if candidate is not None and len(candidate) > 0:
                    next_date = candidate.iloc[0] if hasattr(candidate, "iloc") else candidate[0]
        except Exception:
            next_date = None
 
        result: dict[str, object] = {"earnings_schema_version": EARNINGS_SCHEMA_VERSION}
        if next_date is not None:
            parsed = pd.to_datetime(next_date, errors="coerce", utc=True)
            if parsed is not None and not pd.isna(parsed):
                date_value = parsed.date().isoformat()
                if parsed.date() >= _today_utc():
                    result["next_earnings_date"] = date_value
                else:
                    result["last_earnings_date"] = date_value
    except Exception as exc:
        return {}, SourceStatus(
            "Yahoo Finance earnings",
            f"Earnings calendar fetch failed: {type(exc).__name__}: {exc}",
            ok=False,
        )
 
    result = _normalize_earnings_payload(result)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    if not result.get("next_earnings_date") and not result.get("last_earnings_date"):
        return result, SourceStatus("Yahoo Finance earnings", "No upcoming earnings date published")
    if result.get("next_earnings_date"):
        return result, SourceStatus("Yahoo Finance earnings", f"Next earnings {result.get('next_earnings_date')}")
    return result, SourceStatus("Yahoo Finance earnings", f"Last known earnings {result.get('last_earnings_date')}")
 
 
def days_until_earnings(next_earnings_date: str | None) -> int | None:
    """Compute days from today to a cached earnings date string, freshly each call.
 
    Kept separate from `fetch_earnings_calendar` so the on-disk cache only
    ever stores the absolute date — caching a pre-computed day-count would
    go stale the moment it sat in cache for more than a few hours.
    """
 
    if not next_earnings_date:
        return None
    parsed = pd.to_datetime(next_earnings_date, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    return int((parsed.date() - _today_utc()).days)
 
 
def _earnings_cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("/", "-")
    return EARNINGS_CACHE_DIR / f"{safe}_earnings.json"


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _freshness_tolerance(interval: str) -> timedelta:
    interval = interval.lower()
    if interval.endswith("mo"):
        return timedelta(days=45)
    if interval.endswith("wk"):
        return timedelta(days=14)
    if interval.endswith("m") or interval.endswith("h"):
        return timedelta(days=1)
    return timedelta(days=5)


def _fast_info_get(fast_info: object, key: str) -> object:
    if hasattr(fast_info, "get"):
        try:
            return fast_info.get(key)
        except Exception:
            pass
    try:
        return getattr(fast_info, key)
    except Exception:
        return None


def _normalize_snapshot_dates(snapshot: dict[str, object]) -> None:
    ex_dividend = _date_from_value(snapshot.get("exDividendDate"))
    if ex_dividend:
        snapshot["exDividendDateFormatted"] = ex_dividend
    dividend = _date_from_value(snapshot.get("dividendDate"))
    if dividend:
        snapshot["dividendDateFormatted"] = dividend


def _normalize_earnings_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["earnings_schema_version"] = EARNINGS_SCHEMA_VERSION
    next_date = _date_from_value(normalized.get("next_earnings_date"))
    last_date = _date_from_value(normalized.get("last_earnings_date"))
    if next_date:
        parsed_next = pd.to_datetime(next_date, errors="coerce")
        if parsed_next is not None and not pd.isna(parsed_next):
            if parsed_next.date() >= _today_utc():
                normalized["next_earnings_date"] = next_date
            else:
                normalized.pop("next_earnings_date", None)
                normalized["last_earnings_date"] = next_date
    if last_date:
        normalized["last_earnings_date"] = last_date
    return normalized


def _date_from_value(value: object) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)) and value > 10_000:
        parsed = pd.to_datetime(value, unit="s", errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _is_json_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return False
        return True
    return False


def load_kaggle_history(symbol: str) -> tuple[pd.DataFrame, SourceStatus]:
    """Load an optional local Kaggle CSV for a ticker.

    Place files in `data/kaggle/` using names like `AAPL.csv`, `SHOP.TO.csv`, or
    a broader dataset with a `Symbol` column. The app avoids making assumptions
    about Kaggle credentials or dataset licenses.
    """

    if not KAGGLE_DIR.exists():
        return pd.DataFrame(), SourceStatus("Kaggle local CSV", "No data/kaggle directory", False)

    exact = list(KAGGLE_DIR.glob(f"{symbol}.csv"))
    candidates = exact or list(KAGGLE_DIR.glob("*.csv"))
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        if "Symbol" in frame.columns:
            frame = frame[frame["Symbol"].astype(str).str.upper() == symbol.upper()]

        date_col = next((col for col in frame.columns if col.lower() in {"date", "datetime"}), None)
        close_col = next((col for col in frame.columns if col.lower() in {"close", "adj close"}), None)
        if date_col and close_col and not frame.empty:
            rename_map = {date_col: "Date", close_col: "Close"}
            for source, target in {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
                "adj close": "Adj Close",
            }.items():
                matched = next((col for col in frame.columns if col.lower() == source), None)
                if matched:
                    rename_map[matched] = target
            return _prepare_history(frame.rename(columns=rename_map)), SourceStatus(
                "Kaggle local CSV",
                str(path),
            )

    return pd.DataFrame(), SourceStatus(
        "Kaggle local CSV",
        "No matching CSV found. Add licensed Kaggle exports to data/kaggle/.",
        False,
    )


def merge_source_preference(yahoo: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    """Prefer Yahoo rows, then fill missing dates from Kaggle when available."""

    if yahoo.empty:
        return kaggle
    if kaggle.empty:
        return yahoo

    combined = pd.concat([yahoo.assign(Source="Yahoo"), kaggle.assign(Source="Kaggle")])
    combined = combined.sort_values(["Date", "Source"]).drop_duplicates("Date", keep="first")
    return combined.drop(columns=["Source"], errors="ignore").sort_values("Date").reset_index(drop=True)
