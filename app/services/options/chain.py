from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.market_data.data_sources import DATA_DIR

OPTIONS_CACHE_DIR = DATA_DIR / "cache" / "options"
CHAIN_CACHE_TTL_SECONDS = 15 * 60
CONTRACT_MULTIPLIER = 100
DEFAULT_RISK_FREE_RATE = 0.04
MIN_SIGMA = 1e-4
MIN_TIME_YEARS = 1.0 / (365.0 * 24.0)

ChainLoader = Callable[[str, Optional[str]], Tuple[List[str], Any, Any]]


def risk_free_rate() -> float:
    """Annual continuously-compounded risk-free rate used for greeks (env override)."""

    raw = os.getenv("OPTIONS_RISK_FREE_RATE", str(DEFAULT_RISK_FREE_RATE))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_RISK_FREE_RATE
    return value if math.isfinite(value) else DEFAULT_RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Black-Scholes with the standard library only (math.erf, no scipy)
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, t_years: float, rate: float, sigma: float) -> Tuple[float, float]:
    sigma = max(float(sigma), MIN_SIGMA)
    t_years = max(float(t_years), MIN_TIME_YEARS)
    root_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * root_t)
    return d1, d1 - sigma * root_t


def black_scholes_price(spot: float, strike: float, t_years: float, rate: float, sigma: float, option_type: str) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if t_years <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, t_years, rate, sigma)
    discount = math.exp(-rate * t_years)
    if option_type == "call":
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)


def black_scholes_greeks(spot: float, strike: float, t_years: float, rate: float, sigma: float, option_type: str) -> Dict[str, Optional[float]]:
    """Return delta, gamma, theta (per calendar day), vega (per 1 vol point), rho (per 1% rate)."""

    if spot <= 0 or strike <= 0 or sigma is None or not math.isfinite(sigma) or sigma <= 0 or t_years <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    d1, d2 = _d1_d2(spot, strike, t_years, rate, sigma)
    t_years = max(t_years, MIN_TIME_YEARS)
    root_t = math.sqrt(t_years)
    discount = math.exp(-rate * t_years)
    pdf_d1 = norm_pdf(d1)
    gamma = pdf_d1 / (spot * sigma * root_t)
    vega = spot * pdf_d1 * root_t / 100.0
    if option_type == "call":
        delta = norm_cdf(d1)
        theta_year = -(spot * pdf_d1 * sigma) / (2.0 * root_t) - rate * strike * discount * norm_cdf(d2)
        rho = strike * t_years * discount * norm_cdf(d2) / 100.0
    else:
        delta = norm_cdf(d1) - 1.0
        theta_year = -(spot * pdf_d1 * sigma) / (2.0 * root_t) + rate * strike * discount * norm_cdf(-d2)
        rho = -strike * t_years * discount * norm_cdf(-d2) / 100.0
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 5),
        "theta": round(theta_year / 365.0, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
    }


def implied_volatility(price: float, spot: float, strike: float, t_years: float, rate: float, option_type: str) -> Optional[float]:
    """Bisection IV solver; returns None when the price is outside no-arbitrage bounds."""

    if price is None or price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    intrinsic = max(spot - strike * math.exp(-rate * t_years), 0.0) if option_type == "call" else max(strike * math.exp(-rate * t_years) - spot, 0.0)
    if price < intrinsic - 1e-9:
        return None
    low, high = 1e-3, 5.0
    if black_scholes_price(spot, strike, t_years, rate, high, option_type) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (low + high)
        value = black_scholes_price(spot, strike, t_years, rate, mid, option_type)
        if abs(value - price) < 1e-6:
            return round(mid, 6)
        if value > price:
            high = mid
        else:
            low = mid
    return round(0.5 * (low + high), 6)


# ---------------------------------------------------------------------------
# Expiry helpers
# ---------------------------------------------------------------------------


def parse_expiration(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def days_to_expiry(expiration: Any, as_of: Optional[datetime] = None) -> int:
    expiry = parse_expiration(expiration)
    if expiry is None:
        return 0
    moment = as_of or datetime.now(timezone.utc)
    return max((expiry - moment.date()).days, 0)


def years_to_expiry(expiration: Any, as_of: Optional[datetime] = None) -> float:
    """Time to the 4pm New York expiry close, in years (never below one hour)."""

    expiry = parse_expiration(expiration)
    if expiry is None:
        return MIN_TIME_YEARS
    moment = as_of or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        close = datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    except Exception:  # pragma: no cover - zoneinfo missing data
        close = datetime(expiry.year, expiry.month, expiry.day, 20, 0, tzinfo=timezone.utc)
    seconds = (close - moment).total_seconds()
    return max(seconds / (365.0 * 24.0 * 3600.0), MIN_TIME_YEARS)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> Optional[int]:
    number = _finite(value)
    return int(number) if number is not None else None


def _records(frame: Any) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        return [dict(row) for row in frame if isinstance(row, dict)]
    try:
        return [dict(row) for row in frame.to_dict("records")]
    except Exception:
        return []


def normalize_chain_rows(
    frame: Any,
    option_type: str,
    spot: float,
    expiration: str,
    as_of: Optional[datetime] = None,
    rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Turn a yfinance calls/puts frame (or list of dicts) into normalized rows with greeks."""

    option_type = "put" if str(option_type).lower().startswith("p") else "call"
    rate = risk_free_rate() if rate is None else float(rate)
    t_years = years_to_expiry(expiration, as_of)
    dte = days_to_expiry(expiration, as_of)
    rows: List[Dict[str, Any]] = []
    for raw in _records(frame):
        strike = _finite(raw.get("strike"))
        if strike is None or strike <= 0:
            continue
        bid = _finite(raw.get("bid")) or 0.0
        ask = _finite(raw.get("ask")) or 0.0
        last = _finite(raw.get("lastPrice"))
        if bid > 0 and ask > 0 and ask >= bid:
            mid = round((bid + ask) / 2.0, 4)
        elif last is not None and last > 0:
            mid = round(last, 4)
        elif ask > 0:
            mid = round(ask, 4)
        else:
            mid = 0.0
        spread_pct = round((ask - bid) / mid, 4) if mid > 0 and ask >= bid and bid > 0 else None
        iv = _finite(raw.get("impliedVolatility"))
        if iv is not None and (iv <= 0.001 or iv > 5):
            iv = None
        if iv is None and mid > 0 and spot > 0:
            iv = implied_volatility(mid, spot, strike, t_years, rate, option_type)
        greeks = black_scholes_greeks(spot, strike, t_years, rate, iv if iv else 0.0, option_type) if iv else black_scholes_greeks(0, 0, 0, 0, 0, option_type)
        in_the_money = raw.get("inTheMoney")
        if in_the_money is None:
            in_the_money = spot > strike if option_type == "call" else spot < strike
        rows.append(
            {
                "contractSymbol": str(raw.get("contractSymbol") or f"{option_type.upper()}-{strike}-{expiration}"),
                "type": option_type,
                "strike": round(strike, 4),
                "expiration": str(expiration)[:10],
                "dte": dte,
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "mid": mid,
                "last": round(last, 4) if last is not None else None,
                "impliedVolatility": round(iv, 4) if iv is not None else None,
                "openInterest": _int_or_none(raw.get("openInterest")),
                "volume": _int_or_none(raw.get("volume")),
                "inTheMoney": bool(in_the_money),
                "spreadPct": spread_pct,
                "moneyness": round(spot / strike, 4) if spot > 0 else None,
                "distancePct": round((strike - spot) / spot, 4) if spot > 0 else None,
                "currency": str(raw.get("currency") or "") or None,
                "lastTradeDate": str(raw.get("lastTradeDate"))[:19] if raw.get("lastTradeDate") is not None else None,
                **greeks,
            }
        )
    rows.sort(key=lambda row: row["strike"])
    return rows


# ---------------------------------------------------------------------------
# Fetching + caching
# ---------------------------------------------------------------------------


def _safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(symbol or "").strip().upper()).replace("^", "idx_")


def _cache_path(symbol: str, key: str, cache_dir: Optional[Path]) -> Path:
    root = cache_dir or OPTIONS_CACHE_DIR
    return root / f"{_safe_symbol(symbol)}_{key}.json"


def _read_cache(path: Path, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(str(payload.get("fetchedAt")))
    except Exception:
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - fetched_at > timedelta(seconds=ttl_seconds):
        return None
    return payload


def _write_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except Exception:
        pass


def _raw_rows(frame: Any) -> List[Dict[str, Any]]:
    keep = ("contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "inTheMoney", "currency", "lastTradeDate")
    rows = []
    for raw in _records(frame):
        row: Dict[str, Any] = {}
        for key in keep:
            value = raw.get(key)
            if key == "lastTradeDate" and value is not None:
                value = str(value)
            elif key in {"contractSymbol", "currency"}:
                value = str(value) if value is not None else None
            elif key == "inTheMoney":
                value = bool(value) if value is not None and value == value else None
            else:
                value = _finite(value)
            row[key] = value
        rows.append(row)
    return rows


def _yahoo_loader(symbol: str, expiration: Optional[str]) -> Tuple[List[str], Any, Any]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expirations = [str(item) for item in (ticker.options or ())]
    if not expirations:
        return [], None, None
    chosen = expiration if expiration in expirations else None
    if chosen is None:
        chosen = _default_expiration(expirations)
    chain = ticker.option_chain(chosen)
    return expirations, getattr(chain, "calls", None), getattr(chain, "puts", None)


def _default_expiration(expirations: List[str], as_of: Optional[datetime] = None, min_dte: int = 7) -> str:
    for candidate in expirations:
        if days_to_expiry(candidate, as_of) >= min_dte:
            return candidate
    return expirations[-1]


def _latest_spot(symbol: str) -> Optional[float]:
    try:
        from app.services.workspace.application import latest_market_record

        record = latest_market_record(symbol)
        return _finite(record.get("Close")) if isinstance(record, dict) else None
    except Exception:
        return None


def list_expirations(symbol: str, refresh: bool = False, loader: Optional[ChainLoader] = None, cache_dir: Optional[Path] = None) -> List[str]:
    path = _cache_path(symbol, "expirations", cache_dir)
    if not refresh:
        cached = _read_cache(path, CHAIN_CACHE_TTL_SECONDS)
        if cached and isinstance(cached.get("expirations"), list):
            return [str(item) for item in cached["expirations"]]
    try:
        import yfinance as yf

        expirations = [str(item) for item in (yf.Ticker(symbol).options or ())] if loader is None else list(loader(symbol, None)[0])
    except Exception:
        expirations = []
    if expirations:
        _write_cache(path, {"symbol": symbol, "expirations": expirations, "fetchedAt": datetime.now(timezone.utc).isoformat()})
    return expirations


def fetch_option_chain(
    symbol: str,
    expiration: Optional[str] = None,
    refresh: bool = False,
    *,
    as_of: Optional[datetime] = None,
    spot: Optional[float] = None,
    loader: Optional[ChainLoader] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fetch (or reuse a <15 minute old cached) option chain and return normalized rows with greeks."""

    normalized_symbol = str(symbol or "").strip().upper()
    now = as_of or datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "ok": False,
        "symbol": normalized_symbol,
        "spot": spot,
        "expiration": None,
        "expirations": [],
        "calls": [],
        "puts": [],
        "asOf": now.isoformat(),
        "source": "Yahoo Finance option chain",
        "cached": False,
        "riskFreeRate": risk_free_rate(),
        "message": "",
    }
    if not normalized_symbol:
        result["message"] = "A symbol is required."
        return result

    expirations = list_expirations(normalized_symbol, refresh=refresh, loader=loader, cache_dir=cache_dir)
    chosen = expiration if expiration and expiration in expirations else None
    if chosen is None and expirations:
        chosen = _default_expiration(expirations, now)
    if chosen is None and expiration:
        chosen = expiration

    payload: Optional[Dict[str, Any]] = None
    cache_path = _cache_path(normalized_symbol, chosen or "default", cache_dir)
    if not refresh:
        payload = _read_cache(cache_path, CHAIN_CACHE_TTL_SECONDS)
        if payload is not None:
            result["cached"] = True
    if payload is None:
        try:
            fetched_expirations, calls, puts = (loader or _yahoo_loader)(normalized_symbol, chosen)
        except Exception as exc:
            result["message"] = f"Option chain fetch failed: {type(exc).__name__}: {exc}"
            result["expirations"] = expirations
            return result
        if fetched_expirations:
            expirations = [str(item) for item in fetched_expirations]
            if chosen not in expirations:
                chosen = _default_expiration(expirations, now)
        if calls is None and puts is None:
            result["message"] = f"No listed options were returned for {normalized_symbol}."
            result["expirations"] = expirations
            return result
        payload = {
            "symbol": normalized_symbol,
            "expiration": chosen,
            "expirations": expirations,
            "calls": _raw_rows(calls),
            "puts": _raw_rows(puts),
            "fetchedAt": now.isoformat(),
        }
        _write_cache(cache_path, payload)

    spot_value = spot if spot is not None else _latest_spot(normalized_symbol)
    if spot_value is None:
        # Fall back to the strike closest to the calls/puts ITM boundary so greeks stay usable.
        boundary = [row["strike"] for row in payload.get("calls", []) if row.get("inTheMoney") is False]
        spot_value = min(boundary) if boundary else None
    result.update(
        {
            "ok": True,
            "spot": round(spot_value, 4) if spot_value else None,
            "expiration": payload.get("expiration") or chosen,
            "expirations": [str(item) for item in payload.get("expirations") or expirations],
            "dte": days_to_expiry(payload.get("expiration") or chosen, now),
            "fetchedAt": payload.get("fetchedAt"),
            "calls": normalize_chain_rows(payload.get("calls"), "call", spot_value or 0.0, str(payload.get("expiration") or chosen), now),
            "puts": normalize_chain_rows(payload.get("puts"), "put", spot_value or 0.0, str(payload.get("expiration") or chosen), now),
        }
    )
    if spot_value is None:
        result["message"] = "Underlying price unavailable; greeks were not computed."
    return result


def find_contract(chain: Dict[str, Any], contract_symbol: str) -> Optional[Dict[str, Any]]:
    target = str(contract_symbol or "").strip().upper()
    for row in list(chain.get("calls") or []) + list(chain.get("puts") or []):
        if str(row.get("contractSymbol") or "").upper() == target:
            return row
    return None


def quote_for_leg(chain: Dict[str, Any], option_type: str, strike: float, expiration: Optional[str] = None) -> Optional[Dict[str, Any]]:
    rows = chain.get("puts") if str(option_type).lower().startswith("p") else chain.get("calls")
    best: Optional[Dict[str, Any]] = None
    for row in rows or []:
        if expiration and str(row.get("expiration")) != str(expiration)[:10]:
            continue
        if best is None or abs(float(row["strike"]) - float(strike)) < abs(float(best["strike"]) - float(strike)):
            best = row
    if best is not None and abs(float(best["strike"]) - float(strike)) > 1e-6:
        return None
    return best


def atm_implied_volatility(chain: Dict[str, Any]) -> Optional[float]:
    spot = _finite(chain.get("spot"))
    if not spot:
        return None
    candidates = []
    for row in list(chain.get("calls") or []) + list(chain.get("puts") or []):
        if row.get("impliedVolatility"):
            candidates.append((abs(float(row["strike"]) - spot), float(row["impliedVolatility"])))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    nearest = [iv for _, iv in candidates[:4]]
    return round(sum(nearest) / len(nearest), 4)
