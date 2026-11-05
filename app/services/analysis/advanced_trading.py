from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.config import AI_DISCLOSURE, OPENAI_MAX_OUTPUT_TOKENS, OPENAI_REASONING_EFFORT
from app.services.ai.models import resolve_model
from app.services.ai.registry import ProviderRequest, generate_structured_json
from app.services.analysis.indicators import add_indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def advanced_trading_model() -> str:
    return resolve_model("advanced-trading")


def build_advanced_trading_indicator(symbol: str, analysis: dict[str, Any]) -> dict[str, Any]:
    """Build a validated chart-timing overlay from local evidence plus AI ranking.

    OpenAI may choose among locally computed candidate dates, but it cannot
    introduce a price or timestamp that is not present in the OHLCV evidence.
    That keeps the chart explainable and makes the local engine a usable final
    fallback rather than a second, competing source of market facts.
    """

    normalized_symbol = str(symbol or analysis.get("symbol") or "").strip().upper()
    frame = _history_frame(analysis.get("history"))
    if frame.empty or len(frame) < 30:
        return _unavailable_result(normalized_symbol, "At least 30 OHLCV observations are required for the timing overlay.")

    enriched = add_indicators(frame)
    local = _local_indicator_output(enriched)
    interpretation, failure_reason = _request_openai_interpretation(normalized_symbol, analysis, local)
    if interpretation:
        buy_point = _match_point(interpretation.get("buyDate"), local["buyCandidates"]) or local["buyPoint"]
        sell_point = _match_point(interpretation.get("sellDate"), local["sellCandidates"]) or local["sellPoint"]
        markers = _mark_primary(local["markers"], buy_point, sell_point)
        return {
            "ok": True,
            "symbol": normalized_symbol,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "provider": {
                "quantitative": "local-indicator-engine",
                "interpretation": str(interpretation.get("provider") or "openai"),
                "model": str(interpretation.get("model") or advanced_trading_model()),
            },
            "markers": markers,
            "buyPoint": buy_point,
            "sellPoint": sell_point,
            "summary": str(interpretation.get("summary") or local["summary"]),
            "conditions": _string_list(interpretation.get("conditions")) or local["conditions"],
            "risks": _string_list(interpretation.get("risks")) or local["risks"],
            "evidence": local["evidence"],
            "disclosure": AI_DISCLOSURE,
            "notFinancialAdvice": True,
        }

    local_result = _local_indicator_result(normalized_symbol, local)
    local_result["provider"]["failureReason"] = failure_reason
    return local_result


def build_local_advanced_trading_indicator(symbol: str, history: Any) -> dict[str, Any]:
    """Recalculate the local timing overlay from the latest sync history.

    This intentionally does not call OpenAI. Workspace sync can therefore keep
    BUY/SELL markers current without spending a model token on every refresh.
    """

    normalized_symbol = str(symbol or "").strip().upper()
    records = history.to_dict(orient="records") if isinstance(history, pd.DataFrame) else history
    frame = _history_frame(records)
    if frame.empty or len(frame) < 30:
        return _unavailable_result(normalized_symbol, "At least 30 OHLCV observations are required for the timing overlay.")
    local = _local_indicator_output(add_indicators(frame))
    return _local_indicator_result(normalized_symbol, local)


def _local_indicator_result(symbol: str, local: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "symbol": symbol,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "provider": {
            "quantitative": "local-indicator-engine",
            "interpretation": "local-rules",
            "model": "deterministic-chart-timing-v1",
        },
        "markers": local["markers"],
        "buyPoint": local["buyPoint"],
        "sellPoint": local["sellPoint"],
        "summary": local["summary"],
        "conditions": local["conditions"],
        "risks": local["risks"],
        "evidence": local["evidence"],
        "disclosure": AI_DISCLOSURE,
        "notFinancialAdvice": True,
    }


def _history_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, list):
        return pd.DataFrame()
    frame = pd.DataFrame([row for row in value if isinstance(row, dict)])
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
    for column in required - {"Date"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=list(required)).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def _local_indicator_output(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame["RollingLow40"] = frame["Low"].rolling(40, min_periods=10).min()
    frame["RollingHigh40"] = frame["High"].rolling(40, min_periods=10).max()
    buy_candidates: list[dict[str, Any]] = []
    sell_candidates: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        if index < 20:
            continue
        previous = frame.iloc[index - 1]
        buy_score, buy_reasons = _score_point(row, previous, "BUY")
        sell_score, sell_reasons = _score_point(row, previous, "SELL")
        buy_candidates.append(_point(row, "BUY", buy_score, buy_reasons))
        sell_candidates.append(_point(row, "SELL", sell_score, sell_reasons))

    buy_candidates = _rank_points(buy_candidates)
    sell_candidates = _rank_points(sell_candidates)
    buy_point = buy_candidates[0] if buy_candidates else _fallback_point(frame, "BUY")
    sell_point = sell_candidates[0] if sell_candidates else _fallback_point(frame, "SELL")
    markers = _select_markers(buy_candidates, sell_candidates)
    if not markers:
        markers = [buy_point, sell_point]
    markers = _mark_primary(markers, buy_point, sell_point)
    latest = frame.iloc[-1]
    latest_close = _number(latest.get("Close"))
    latest_rsi = _number(latest.get("RSI14"), 50.0)
    return {
        "buyCandidates": buy_candidates[:8],
        "sellCandidates": sell_candidates[:8],
        "buyPoint": buy_point,
        "sellPoint": sell_point,
        "markers": markers,
        "summary": (
            f"The local timing engine identifies {buy_point['price']:.2f} on {buy_point['date'][:10]} "
            f"as the strongest BUY setup and {sell_point['price']:.2f} on {sell_point['date'][:10]} "
            f"as the strongest SELL setup from the supplied OHLCV history. "
            f"The latest close is {latest_close:.2f} with RSI {latest_rsi:.1f}."
        ),
        "conditions": [
            "Treat the marker as a research trigger, then confirm price, volume, trend, and the latest company evidence.",
            "A BUY setup requires confirmation above the relevant support or momentum trigger; a SELL setup requires confirmation below resistance or momentum failure.",
        ],
        "risks": [
            "Historical indicator alignment does not guarantee a future move.",
            "Gaps, news, liquidity, and regime changes can invalidate a chart-timing setup quickly.",
        ],
        "evidence": {
            "engine": "StockSignal local indicator engine",
            "features": ["RSI14", "EMA20", "EMA50", "MACD", "MACDSignal", "BBLower", "BBUpper", "ATR14", "VolumeRatio", "40-bar support/resistance"],
            "observations": len(frame),
            "latestDate": _iso(frame.iloc[-1].get("Date")),
        },
    }


def _score_point(row: pd.Series, previous: pd.Series, side: str) -> tuple[float, list[str]]:
    close = _number(row.get("Close"))
    ema20 = _number(row.get("EMA20"), close)
    ema50 = _number(row.get("EMA50"), close)
    rsi = _number(row.get("RSI14"), 50.0)
    macd = _number(row.get("MACD"))
    macd_signal = _number(row.get("MACDSignal"))
    prev_macd = _number(previous.get("MACD"))
    prev_signal = _number(previous.get("MACDSignal"))
    lower = _number(row.get("BBLower"), close)
    upper = _number(row.get("BBUpper"), close)
    support = _number(row.get("RollingLow40"), close)
    resistance = _number(row.get("RollingHigh40"), close)
    volume_ratio = _number(row.get("VolumeRatio"), 1.0)
    score = 0.0
    reasons: list[str] = []
    if side == "BUY":
        if rsi <= 35:
            score += 2
            reasons.append(f"RSI is oversold at {rsi:.1f}")
        if close <= lower * 1.02:
            score += 2
            reasons.append("price is testing the lower Bollinger band")
        if prev_macd <= prev_signal and macd > macd_signal:
            score += 2
            reasons.append("MACD crossed upward")
        if close > ema20 and _number(previous.get("Close")) <= _number(previous.get("EMA20"), close):
            score += 1.5
            reasons.append("price reclaimed EMA20")
        if close <= support * 1.04:
            score += 1.5
            reasons.append("price is near 40-bar support")
        if volume_ratio >= 1.2:
            score += 0.5
            reasons.append("volume confirms participation")
    else:
        if rsi >= 65:
            score += 2
            reasons.append(f"RSI is overbought at {rsi:.1f}")
        if close >= upper * 0.98:
            score += 2
            reasons.append("price is testing the upper Bollinger band")
        if prev_macd >= prev_signal and macd < macd_signal:
            score += 2
            reasons.append("MACD crossed downward")
        if close < ema20 and _number(previous.get("Close")) >= _number(previous.get("EMA20"), close):
            score += 1.5
            reasons.append("price lost EMA20")
        if close >= resistance * 0.96:
            score += 1.5
            reasons.append("price is near 40-bar resistance")
        if volume_ratio >= 1.2:
            score += 0.5
            reasons.append("volume confirms participation")
    if close > ema50 and side == "BUY":
        score += 0.5
        reasons.append("price remains above EMA50")
    if close < ema50 and side == "SELL":
        score += 0.5
        reasons.append("price remains below EMA50")
    return score, reasons or ["Best available local alignment; confirmation is still required"]


def _point(row: pd.Series, side: str, score: float, reasons: list[str]) -> dict[str, Any]:
    return {
        "date": _iso(row.get("Date")),
        "price": round(_number(row.get("Close")), 6),
        "type": side,
        "score": round(score, 3),
        "confidence": round(min(0.96, 0.42 + score * 0.07), 3),
        "reasons": reasons,
        "isPrimary": False,
    }


def _fallback_point(frame: pd.DataFrame, side: str) -> dict[str, Any]:
    row = frame.iloc[-1]
    return _point(row, side, 0.0, ["No fully confirmed multi-factor setup was found; use the latest bar as a watch reference"])


def _rank_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(points, key=lambda item: (-float(item.get("score") or 0), -_date_value(str(item.get("date") or ""))))


def _select_markers(buys: list[dict[str, Any]], sells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for point in [*buys, *sells]:
        if any(point["type"] == item["type"] and abs(_date_value(point["date"]) - _date_value(item["date"])) < 10 * 86400 for item in selected):
            continue
        selected.append(point)
        if len(selected) >= 6:
            break
    return selected


def _mark_primary(markers: list[dict[str, Any]], buy: dict[str, Any], sell: dict[str, Any]) -> list[dict[str, Any]]:
    result = [dict(marker, isPrimary=marker.get("date") in {buy.get("date"), sell.get("date")} and marker.get("type") in {buy.get("type"), sell.get("type")}) for marker in markers]
    for point in (buy, sell):
        if not any(marker.get("date") == point.get("date") and marker.get("type") == point.get("type") for marker in result):
            result.append(dict(point, isPrimary=True))
    return result


def _match_point(value: Any, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    requested = value[:10]
    return next((dict(item, isPrimary=True) for item in candidates if str(item.get("date") or "")[:10] == requested), None)


ADVANCED_TRADING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "buyDate": {"type": "string"},
        "sellDate": {"type": "string"},
        "summary": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
}


def _request_openai_interpretation(symbol: str, analysis: dict[str, Any], local: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    model = advanced_trading_model()
    prompt = {
        "task": "Rank the best historical BUY and SELL timing candidates for an educational chart overlay.",
        "symbol": symbol,
        "rules": [
            "Choose buyDate only from buyCandidates and sellDate only from sellCandidates.",
            "Do not invent dates or prices. The application will use the candidate's verified OHLCV price.",
            "Balance RSI, EMA, MACD, Bollinger, ATR, volume, support/resistance, trend, and the supplied Signal/News/ML/Backtest evidence.",
            "Explain that a marker is a conditional research signal, not a guaranteed trade instruction.",
        ],
        "localEvidence": local["evidence"],
        "buyCandidates": local["buyCandidates"],
        "sellCandidates": local["sellCandidates"],
        "sourceAnalysis": {
            "signal": analysis.get("signal"),
            "featureEngineering": analysis.get("featureEngineering"),
            "marketRegime": analysis.get("marketRegime"),
            "news": (analysis.get("news") or [])[:12],
            "ml": analysis.get("ml"),
            "backtest": analysis.get("backtest"),
        },
        "requiredOutput": ["buyDate", "sellDate", "summary", "conditions", "risks"],
    }
    result = generate_structured_json(
        ProviderRequest(
            task="advanced-trading",
            prompt=prompt,
            schema=ADVANCED_TRADING_SCHEMA,
            schema_name="advanced_trading_indicator",
            strict=False,
            max_output_tokens=min(1200, OPENAI_MAX_OUTPUT_TOKENS),
            reasoning_effort=OPENAI_REASONING_EFFORT,
            model=model,
            log_prefix="advanced_trading_provider",
            logger=logger,
            context={"symbol": symbol},
        )
    )
    if not result.ok:
        return None, result.reason or "The AI provider did not return an advanced trading interpretation."
    return {**(result.payload or {}), "provider": result.provider, "model": result.model}, None


def _unavailable_result(symbol: str, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "symbol": symbol,
        "provider": {"quantitative": "local-indicator-engine", "interpretation": "unavailable", "model": "deterministic-chart-timing-v1", "failureReason": reason},
        "markers": [],
        "buyPoint": None,
        "sellPoint": None,
        "summary": reason,
        "conditions": [],
        "risks": [],
        "evidence": {"engine": "StockSignal local indicator engine", "observations": 0},
        "disclosure": AI_DISCLOSURE,
        "notFinancialAdvice": True,
    }


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


def _iso(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return str(value or "")
    return timestamp.isoformat().replace("+00:00", "Z")


def _date_value(value: str) -> float:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return timestamp.timestamp() if not pd.isna(timestamp) else 0.0
