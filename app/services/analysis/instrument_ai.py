from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from app.config import AI_DISCLOSURE, OPENAI_MAX_OUTPUT_TOKENS, OPENAI_REASONING_EFFORT, RECOMMENDED_OPENAI_MODEL
from app.services.ai.models import resolve_model
from app.services.ai.registry import ProviderRequest, generate_structured_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL = RECOMMENDED_OPENAI_MODEL
DISCLOSURE = AI_DISCLOSURE

SECTION_DEFAULTS = (
    "signal",
    "chart",
    "ml",
    "backtest",
    "news",
    "fit",
    "deepEvidence",
)

INSTRUMENT_ENHANCEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "summary": {"type": "string"},
        **{
            section: {
                "type": "object",
                "additionalProperties": True,
                "description": f"AI interpretation for the {section} tab; do not rewrite raw local metrics.",
            }
            for section in SECTION_DEFAULTS
        },
        "sources": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
    },
}


def instrument_ai_model() -> str:
    return resolve_model("instrument")


def enhance_instrument_analysis(
    analysis: dict[str, Any],
    portfolio_context: dict[str, Any] | None = None,
    portfolio_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Interpret the local evidence without replacing its raw calculations.

    The base instrument endpoint computes reproducible Yahoo-backed facts and
    validation metrics first. This separate call gives the active AI provider
    that evidence for tab-specific interpretation, while preserving local
    values as the authoritative facts and keeping deterministic local output
    as the fallback.
    """

    symbol = str(analysis.get("symbol") or "").upper()
    model = instrument_ai_model()
    if not os.getenv("OPENAI_API_KEY"):
        return _unavailable("OpenAI was not configured on the backend.", model=model, category="configuration")
    prompt = _build_prompt(analysis, portfolio_context, portfolio_rows or [])
    result = generate_structured_json(
        ProviderRequest(
            task="instrument",
            prompt=prompt,
            schema=INSTRUMENT_ENHANCEMENT_SCHEMA,
            schema_name="instrument_analysis_enhancement",
            strict=False,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            reasoning_effort=OPENAI_REASONING_EFFORT,
            model=model,
            log_prefix="instrument_analysis_provider",
            logger=logger,
            context={"symbol": symbol},
        )
    )
    if not result.ok:
        return _unavailable(
            result.reason or "The AI provider did not return a usable interpretation.",
            model=result.model,
            category=result.failure_category or "unavailable",
            elapsed_ms=result.elapsed_ms,
            symbol=symbol,
            provider=result.provider,
        )
    parsed = result.payload or {}
    return {
        "status": "ready",
        "provider": result.provider,
        "model": result.model,
        "symbol": symbol,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "insights": _normalize_insights(parsed),
        "sources": parsed.get("sources") if isinstance(parsed.get("sources"), list) else [],
        "disclosure": DISCLOSURE,
        "notFinancialAdvice": True,
        "elapsedMs": result.elapsed_ms,
        "fallbackUsed": False,
    }


def _build_prompt(
    analysis: dict[str, Any],
    portfolio_context: dict[str, Any] | None,
    portfolio_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = analysis.get("snapshot") if isinstance(analysis.get("snapshot"), dict) else {}
    history = analysis.get("history") if isinstance(analysis.get("history"), list) else []
    news = analysis.get("news") if isinstance(analysis.get("news"), list) else []
    return {
        "task": "Improve the interpretation of one instrument's local analysis for a stock research dashboard.",
        "symbol": analysis.get("symbol"),
        "method": [
            "Use Yahoo Finance OHLCV, profile/financial statement fields, benchmark and VIX context, and matched news as evidence.",
            "Use the locally engineered RSI, MACD, EMA, ADX, Bollinger, ATR, volume, relative-strength, and trend features.",
            "Use local ML validation, probability, backtest, buy-and-hold, Sharpe, drawdown, and walk-forward results.",
            "Do not invent missing earnings, prices, indicators, sources, or forecasts.",
            "Do not replace local numeric facts; explain agreement, disagreement, limitations, and what to monitor.",
            "Use the deepEvidence package to interpret total-return/income, investment role, multi-year reversion, scenarios, catalysts, contradictions, and confirmation-based execution conditions.",
        ],
        "dataAcquisition": {
            "sourceStatus": analysis.get("sourceStatus"),
            "snapshotStatus": analysis.get("snapshotStatus"),
            "earningsStatus": analysis.get("earningsStatus"),
            "macroContext": analysis.get("macroContext"),
            "ohlcv": history[-120:],
            "financialSnapshot": _pick(snapshot, (
                "currency", "sector", "industry", "marketCap", "enterpriseValue", "totalRevenue",
                "revenueGrowth", "grossMargins", "profitMargins", "netIncomeToCommon", "trailingPE",
                "forwardPE", "priceToBook", "dividendYield", "beta", "targetMeanPrice", "targetMedianPrice",
            )),
            "earnings": analysis.get("earnings"),
            "financialStatements": analysis.get("financialStatements"),
            "financialEvidence": analysis.get("financialEvidence"),
            "news": news[:12],
            "deepEvidence": analysis.get("deepEvidence"),
        },
        "featureEngineering": analysis.get("featureEngineering"),
        "localAnalysis": {
            "signal": analysis.get("signal"),
            "marketRegime": analysis.get("marketRegime"),
            "ml": analysis.get("ml"),
            "mlValidation": analysis.get("mlValidation"),
            "backtest": analysis.get("backtest"),
            "newsSentiment": analysis.get("newsSentiment"),
            "deepEvidence": analysis.get("deepEvidence"),
        },
        "portfolioContext": {
            "selectedHolding": portfolio_context or {},
            "portfolioRows": _compact_portfolio_rows(portfolio_rows),
        },
        "requiredOutput": {
            "summary": "One concise evidence-weighted synthesis.",
            "signal": "Assessment, agreement/disagreement with local action, entry/exit interpretation, drivers, risks.",
            "chart": "Trend, trading range, support/resistance, volatility, breakout or reversion interpretation.",
            "ml": "Feature interpretation, direction read, validation quality, and limitations.",
            "backtest": "Strategy versus buy-and-hold, Sharpe/drawdown/consistency interpretation, and sample limitations.",
            "news": "Current sentiment, catalysts, risks, evidence quality, and event relevance.",
            "fit": "Portfolio concentration, diversification, P/L context, and conditional action.",
            "deepEvidence": "Interpret every supplied deep-evidence module, preserving uncertainty and citing which evidence supports or conflicts with the conclusion.",
        },
    }


def _pick(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: values.get(key) for key in keys if values.get(key) is not None}


def _compact_portfolio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("Symbol", "Shares", "Avg Cost", "Book Cost", "Market Value", "Portfolio %", "Since Purchase %")
    return [{key: row.get(key) for key in fields if row.get(key) is not None} for row in rows[:30] if isinstance(row, dict)]


def _normalize_insights(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for section in SECTION_DEFAULTS:
        value = parsed.get(section)
        result[section] = value if isinstance(value, dict) else {}
    result["summary"] = {"assessment": str(parsed.get("summary") or "")}
    return result


def _unavailable(
    reason: str,
    model: str | None = None,
    category: str = "unavailable",
    elapsed_ms: int | None = None,
    symbol: str = "",
    provider: str = "openai",
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "provider": provider,
        "model": model or instrument_ai_model(),
        "symbol": symbol,
        "insights": {},
        "sources": [],
        "reason": reason,
        "failureCategory": category,
        "disclosure": DISCLOSURE,
        "notFinancialAdvice": True,
        "elapsedMs": elapsed_ms,
        "fallbackUsed": True,
    }
