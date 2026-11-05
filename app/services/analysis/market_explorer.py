from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import (
    AI_DISCLOSURE,
    FUTURE_PROVIDER_NOTE,
    LOCAL_DISCLOSURE,
    MARKET_EXPLORER_RESULT_SCHEMA_VERSION,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_REASONING_EFFORT,
)
from app.services.ai.models import resolve_model
from app.services.ai.providers import extract_json_object
from app.services.ai.registry import ProviderRequest, generate_structured_json
from app.services.analysis.deep_evidence import build_deep_evidence
from app.services.analysis.pestel import canonical_sector, profile_pestel, sector_pestel
from app.services.market_data.data_sources import fetch_yahoo_history
from app.services.market_data.markets import DEFAULT_WATCHLISTS, EXCHANGES, SECTOR_TILTS, explorer_universe_rows, normalize_symbol
from app.services.workspace.application import focused_analysis_payload

MARKET_EXPLORER_RESULT_LAYOUT: dict[str, Any] = {
    "version": MARKET_EXPLORER_RESULT_SCHEMA_VERSION,
    "sections": [
        {"id": "run", "required": True, "purpose": "Provider, model, mode, generation time, and fallback provenance."},
        {"id": "request", "required": True, "purpose": "The user's profile, budget policy, markets, and requested workspace instruments."},
        {"id": "summary", "required": True, "purpose": "Descriptive executive read and conditional predictive conclusion."},
        {"id": "evidence", "required": True, "purpose": "All reviewed candidates, per-instrument modules, local ranking, AI selection, final selection, and explicit exclusions."},
        {"id": "recommendation", "required": True, "purpose": "Final instrument reads, alternatives, and rationale."},
        {"id": "allocation", "required": True, "purpose": "Authoritative budget policy, currency conversion, weights, amounts, and cash reserve."},
        {"id": "execution", "required": True, "purpose": "A checklist for every final selected instrument, with a local completion path when AI output is incomplete."},
        {"id": "context", "required": True, "purpose": "PESTEL context, FX source, data-quality notes, and research caveats."},
        {"id": "disclosure", "required": True, "purpose": "AI/fallback status and educational-not-financial-advice disclosure."},
    ],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MARKET_EXPLORER_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "budget",
        "label": "Budget",
        "type": "number",
        "prompt": "How much capital do you want to assess for this plan?",
        "placeholder": "3062",
    },
    {
        "id": "budgetAllocation",
        "label": "Budget use",
        "type": "budget-policy",
        "options": ["full", "percentage", "cash"],
        "prompt": "Should we use the whole budget, invest a percentage, or keep a fixed cash reserve?",
    },
    {
        "id": "currency",
        "label": "Currency",
        "type": "choice",
        "options": ["CAD", "USD", "Both"],
        "prompt": "Which currency should the analysis use for its budget view?",
    },
    {
        "id": "riskAppetite",
        "label": "Risk appetite",
        "type": "choice",
        "options": ["Conservative", "Balanced", "Growth"],
        "prompt": "How much drawdown can you tolerate before you would reconsider the plan?",
    },
    {
        "id": "experience",
        "label": "Experience",
        "type": "choice",
        "options": ["New investor", "Some experience", "Active trader"],
        "prompt": "How familiar are you with position sizing, news risk, and technical indicators?",
    },
    {
        "id": "purpose",
        "label": "Purpose",
        "type": "choice",
        "options": ["Hobby", "Daily trading", "One-year hold", "Income", "Learning"],
        "prompt": "What is the main reason you are assessing investments?",
    },
    {
        "id": "strategy",
        "label": "Strategy",
        "type": "choice",
        "options": ["Short-term (1-4 weeks)", "Long-term (6-12 months)", "Buy-dip"],
        "prompt": "Which strategy should the market explorer optimize around?",
    },
    {
        "id": "intent",
        "label": "Investment intent",
        "type": "choice",
        "options": ["Buy / Add", "Sell / Trim", "Hold / Watch", "Not sure"],
        "prompt": "What action are you considering?",
    },
    {
        "id": "desiredSector",
        "label": "Desired sector",
        "type": "multi-choice",
        "prompt": "Do you want to target one or more sectors, or should the AI infer a balanced sector mix?",
        "options": ["Technology", "Financial Services", "Healthcare", "Energy", "Industrials", "Consumer Defensive", "Consumer Cyclical", "Utilities", "Communication Services", "Real Estate"],
    },
    {
        "id": "preferredInstruments",
        "label": "Preferred instruments",
        "type": "symbols",
        "prompt": "Do you already have instruments in mind, or are you exploring?",
        "placeholder": "SHOP.TO, TD.TO, AAPL",
    },
]


PROCESS_STEPS: list[dict[str, str]] = [
    {
        "id": "profile",
        "title": "Investor discovery",
        "detail": "Quiz the user on budget, risk appetite, experience, purpose, strategy, intent, sectors, and whether they are exploring or already have tickers.",
    },
    {
        "id": "universe",
        "title": "Instrument discovery",
        "detail": "Explorers get a market universe filtered by sector, risk, and strategy. Users with preferred instruments skip directly to deep instrument analysis.",
    },
    {
        "id": "evidence",
        "title": "Deep evidence analysis",
        "detail": "Run every module for every instrument: multi-year trading-centre and reversion analysis, explicit pullback bands and entry zones, technical evidence, detailed earnings metrics, event catalysts, rich news, and current/forward PESTEL context.",
    },
    {
        "id": "prediction",
        "title": "Descriptive then predictive output",
        "detail": "Explain what has happened first, then classify the thesis, reconcile contradictory observations, and provide base, worst, and best-case scenarios with explicit uncertainty.",
    },
    {
        "id": "alternatives",
        "title": "Compare secondary intents and strategies",
        "detail": "Show how the result changes under other intents and strategies so the user can compare across sectors and execution styles.",
    },
    {
        "id": "allocation",
        "title": "Budget distribution",
        "detail": "Convert the selected budget into target weights, dollar allocation, approximate shares, and execution notes.",
    },
    {
        "id": "monitor",
        "title": "Live monitoring",
        "detail": "Keep the plan fresh as price, news, developments, and alert conditions change in background services.",
    },
]


@dataclass(frozen=True)
class MarketExplorerProfile:
    budget: float
    budget_currency: str
    budget_allocation_mode: str
    budget_allocation_value: float
    currency: tuple[str, ...]
    risk_appetite: str
    experience: str
    purpose: str
    strategy: str
    intent: str
    intent_options: tuple[str, ...]
    desired_sector: str
    preferred_instruments: tuple[str, ...]
    markets: tuple[str, ...]
    exploring: bool


def market_explorer_flow_payload() -> dict[str, Any]:
    return {
        "questions": MARKET_EXPLORER_QUESTIONS,
        "steps": PROCESS_STEPS,
        "supportedProviders": ["openai", "local-rules"],
        "providerOrder": ["openai", "future-qwen", "local-rules"],
        "failSafeProviders": ["future-qwen", "local-rules"],
        "recommendedOpenAIModel": market_explorer_model(),
        "disclosure": AI_DISCLOSURE,
        "futureProviderNote": FUTURE_PROVIDER_NOTE,
        "resultSchemaVersion": MARKET_EXPLORER_RESULT_SCHEMA_VERSION,
        "resultLayout": MARKET_EXPLORER_RESULT_LAYOUT,
    }


ProgressCallback = Callable[[dict[str, Any]], None]


def emit_progress(on_progress: ProgressCallback | None, **event: Any) -> None:
    """Report analysis progress without ever letting the callback break the run."""

    if on_progress is None:
        return
    try:
        on_progress(event)
    except Exception:  # pragma: no cover - reporting must not affect the analysis
        logger.debug("market_explorer_progress outcome=callback_error", exc_info=True)


def market_explorer_recommendation_payload(payload: dict[str, Any], on_progress: ProgressCallback | None = None) -> dict[str, Any]:
    profile = parse_profile(payload)
    emit_progress(
        on_progress,
        stage="profile",
        message=(
            f"Investor profile ready: {profile.risk_appetite} risk, {profile.experience.lower()}, "
            f"{profile.purpose.lower()} purpose, {profile.strategy}, intent {profile.intent}, "
            f"budget {profile.budget_currency} {profile.budget:,.0f}."
        ),
        profile=profile_to_payload(profile),
    )
    candidates = analyze_candidates(profile, on_progress=on_progress)
    emit_progress(on_progress, stage="prediction", message="Drafting descriptive reads and conditional predictive conclusions from the local evidence...")
    local_plan = build_local_plan(profile, candidates)
    ai_plan, fallback_reason = maybe_generate_openai_plan(profile, candidates, local_plan, on_progress=on_progress)
    plan = ai_plan or build_local_plan(profile, candidates, fallback_reason=fallback_reason)
    emit_progress(on_progress, stage="alternatives", message="Comparing secondary intents and strategies across sectors...")
    emit_progress(on_progress, stage="allocation", message="Distributing the budget across the selected instruments and any cash reserve...")
    result = canonical_market_explorer_result(
        profile,
        candidates,
        local_plan,
        plan,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    emit_progress(
        on_progress,
        stage="monitor",
        message="Plan ready. Live quotes keep the results page current until you run the analysis again.",
        provider=result.get("provider"),
        model=result.get("model"),
        mode=result.get("mode"),
    )
    return result


def canonical_market_explorer_result(
    profile: MarketExplorerProfile,
    candidates: list[dict[str, Any]],
    local_plan: dict[str, Any],
    plan: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    """Normalize every provider into one loss-resistant result contract."""
    local_selected = rows_from_value(local_plan.get("selectedInstruments"))
    ai_selected = rows_from_value(plan.get("aiSelectedInstruments"))
    provider_selected = rows_from_value(plan.get("selectedInstruments"))
    selected = merge_selected_instruments(
        provider_selected or ai_selected,
        local_selected,
    )
    if not selected:
        selected = local_selected or ai_selected
    selected_symbols = [str(item.get("symbol") or "") for item in selected]
    excluded = excluded_instrument_rows(profile, candidates, selected_symbols)
    allocation = merge_ai_allocation_with_currency(
        profile,
        rows_from_value(plan.get("allocation")),
        rows_from_value(local_plan.get("allocation")),
    )
    execution = merge_execution_plans(
        rows_from_value(plan.get("executionPlan")),
        rows_from_value(local_plan.get("executionPlan")),
    )
    alternatives = rows_from_value(plan.get("alternatives")) or rows_from_value(local_plan.get("alternatives"))
    pestel_context = rows_from_value(plan.get("pestelContext")) or rows_from_value(local_plan.get("pestelContext"))
    executive_summary = str(plan.get("executiveSummary") or local_plan.get("executiveSummary") or "")
    ai_disclosure = plan.get("aiDisclosure") if isinstance(plan.get("aiDisclosure"), dict) else local_plan.get("aiDisclosure")
    ai_disclosure = ai_disclosure if isinstance(ai_disclosure, dict) else {}
    provider = str(plan.get("provider") or local_plan.get("provider") or "local-rules")
    model = str(plan.get("model") or local_plan.get("model") or "deterministic-technical-news-allocation-v1")
    mode = str(plan.get("mode") or local_plan.get("mode") or "local")
    fx_quote = plan.get("fxQuote") if isinstance(plan.get("fxQuote"), dict) else local_plan.get("fxQuote")
    allocation_policy = plan.get("allocationPolicy") if isinstance(plan.get("allocationPolicy"), dict) else local_plan.get("allocationPolicy")
    structured_analysis = {
        "schemaVersion": MARKET_EXPLORER_RESULT_SCHEMA_VERSION,
        "run": {
            "mode": mode,
            "provider": provider,
            "model": model,
            "generatedAt": generated_at,
            "fallbackReason": ai_disclosure.get("fallbackReason"),
        },
        "request": {
            "profile": profile_to_payload(profile),
            "requestedInstruments": list(profile.preferred_instruments),
            "markets": list(profile.markets),
            "budget": {
                "amount": profile.budget,
                "currency": profile.budget_currency,
                "allocationPolicy": allocation_policy or budget_allocation_policy(profile),
            },
        },
        "summary": {
            "executive": executive_summary,
            "descriptive": [str(item.get("descriptiveRead") or "") for item in selected if item.get("descriptiveRead")],
            "predictive": [str(item.get("predictiveRead") or "") for item in selected if item.get("predictiveRead")],
        },
        "evidence": {
            "reviewedInstruments": candidates,
            "localRankedInstruments": local_selected,
            "aiSelectedInstruments": ai_selected,
            "finalSelectedInstruments": selected,
            "excludedInstruments": excluded,
        },
        "recommendation": {
            "selectedInstruments": selected,
            "alternatives": alternatives,
        },
        "allocation": {
            "policy": allocation_policy or budget_allocation_policy(profile),
            "rows": allocation,
        },
        "execution": {
            "plan": execution,
            "completion": "AI execution steps are merged with local evidence so every final selected instrument has a checklist.",
        },
        "context": {
            "pestel": pestel_context,
            "fxQuote": fx_quote,
            "dataQuality": [
                {"symbol": item.get("symbol"), "issue": item.get("error")}
                for item in candidates
                if item.get("ok", True) is False or item.get("error")
            ],
        },
        "disclosure": {
            "ai": ai_disclosure,
            "notFinancialAdvice": True,
            "futureProviderNote": FUTURE_PROVIDER_NOTE,
        },
    }
    return {
        **plan,
        "resultSchemaVersion": MARKET_EXPLORER_RESULT_SCHEMA_VERSION,
        "resultLayout": MARKET_EXPLORER_RESULT_LAYOUT,
        "structuredAnalysis": structured_analysis,
        "profile": profile_to_payload(profile),
        "questions": MARKET_EXPLORER_QUESTIONS,
        "steps": PROCESS_STEPS,
        "generatedAt": generated_at,
        "selectedInstruments": selected,
        "excludedInstruments": excluded,
        "allocation": allocation,
        "executionPlan": execution,
        "alternatives": alternatives,
        "pestelContext": pestel_context,
        "fxQuote": fx_quote,
        "allocationPolicy": allocation_policy or budget_allocation_policy(profile),
    }


def rows_from_value(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _saved_analysis_payload(record: Any) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "title": record.title,
        "profile": record.profile,
        "result": record.result,
        "createdAt": record.created_at.isoformat() if record.created_at else None,
        "updatedAt": record.updated_at.isoformat() if record.updated_at else None,
    }


def list_saved_market_explorer_analyses(user_id: str) -> list[dict[str, Any]]:
    from app.database import session_scope
    from app.models.orm import SavedMarketExplorerAnalysis

    with session_scope() as session:
        rows = (
            session.query(SavedMarketExplorerAnalysis)
            .filter(SavedMarketExplorerAnalysis.user_id == uuid.UUID(str(user_id)))
            .order_by(SavedMarketExplorerAnalysis.updated_at.desc())
            .all()
        )
        return [_saved_analysis_payload(row) for row in rows]


def save_market_explorer_analysis(user_id: str, profile: dict[str, Any], result: dict[str, Any], title: str | None = None) -> dict[str, Any]:
    from app.database import session_scope
    from app.models.orm import SavedMarketExplorerAnalysis

    with session_scope() as session:
        row = SavedMarketExplorerAnalysis(
            user_id=uuid.UUID(str(user_id)),
            title=(title or "Market Explorer analysis").strip() or "Market Explorer analysis",
            profile=profile,
            result=result,
        )
        session.add(row)
        session.flush()
        return _saved_analysis_payload(row)


def delete_saved_market_explorer_analysis(user_id: str, analysis_id: str | None = None) -> int:
    from app.database import session_scope
    from app.models.orm import SavedMarketExplorerAnalysis

    with session_scope() as session:
        query = session.query(SavedMarketExplorerAnalysis).filter(SavedMarketExplorerAnalysis.user_id == uuid.UUID(str(user_id)))
        if analysis_id:
            row = query.filter(SavedMarketExplorerAnalysis.id == uuid.UUID(str(analysis_id))).one_or_none()
            if row is None:
                return 0
            session.delete(row)
            return 1
        rows = query.all()
        for row in rows:
            session.delete(row)
        return len(rows)


def refresh_saved_market_explorer_analysis(user_id: str, analysis_id: str) -> dict[str, Any] | None:
    from app.database import session_scope
    from app.models.orm import SavedMarketExplorerAnalysis

    with session_scope() as session:
        row = (
            session.query(SavedMarketExplorerAnalysis)
            .filter(
                SavedMarketExplorerAnalysis.id == uuid.UUID(str(analysis_id)),
                SavedMarketExplorerAnalysis.user_id == uuid.UUID(str(user_id)),
            )
            .one_or_none()
        )
        if row is None:
            return None
        row.result = market_explorer_recommendation_payload(row.profile)
        row.updated_at = datetime.now(timezone.utc)
        session.flush()
        return _saved_analysis_payload(row)


def parse_profile(payload: dict[str, Any]) -> MarketExplorerProfile:
    preferred = payload.get("preferredInstruments") or payload.get("preferred_instruments") or []
    if isinstance(preferred, str):
        preferred = [item.strip() for item in preferred.replace("\n", ",").split(",")]
    markets = payload.get("markets") or list(EXCHANGES)
    intent = str(payload.get("intent") or "Buy / Add")
    raw_intent_options = payload.get("intentOptions") or payload.get("intent_options") or []
    if isinstance(raw_intent_options, str):
        raw_intent_options = [item.strip() for item in raw_intent_options.replace(",", "|").split("|")]
    valid_intents = ("Buy / Add", "Sell / Trim", "Hold / Watch")
    intent_options = tuple(
        item for item in (str(value) for value in raw_intent_options) if item in valid_intents
    )
    if intent == "Not sure" and not intent_options:
        intent_options = valid_intents
    elif intent != "Not sure":
        intent_options = (intent,) if intent in valid_intents else ()
    budget_allocation_mode = normalize_budget_allocation_mode(payload.get("budgetAllocationMode") or payload.get("budget_allocation_mode"))
    raw_budget_allocation_value = payload.get("budgetAllocationValue")
    if raw_budget_allocation_value is None:
        raw_budget_allocation_value = payload.get("budget_allocation_value")
    default_budget_allocation_value = 100.0 if budget_allocation_mode == "percentage" else 0.0
    return MarketExplorerProfile(
        budget=max(0.0, _num(payload.get("budget"), 0.0)),
        budget_currency=normalize_budget_currency(payload.get("budgetCurrency") or payload.get("budget_currency") or payload.get("currency")),
        budget_allocation_mode=budget_allocation_mode,
        budget_allocation_value=max(0.0, _num(raw_budget_allocation_value, default_budget_allocation_value)),
        currency=normalize_currencies(payload.get("currency")),
        risk_appetite=str(payload.get("riskAppetite") or payload.get("risk_appetite") or "Balanced"),
        experience=str(payload.get("experience") or "New investor"),
        purpose=str(payload.get("purpose") or "One-year hold"),
        strategy=str(payload.get("strategy") or "Buy-dip"),
        intent=intent,
        intent_options=intent_options,
        desired_sector=str(payload.get("desiredSector") or payload.get("desired_sector") or "Infer").strip(),
        preferred_instruments=tuple(normalize_requested_symbols(preferred, markets)),
        markets=tuple(str(market).upper() for market in markets if str(market).upper() in EXCHANGES),
        exploring=bool(payload.get("exploring", not preferred)),
    )


def normalize_budget_currency(raw: Any) -> str:
    values = list(raw) if isinstance(raw, list) else [raw or "CAD"]
    for value in values:
        code = str(value or "").upper()
        if code in {"CAD", "USD"}:
            return code
    return "CAD"


def normalize_budget_allocation_mode(raw: Any) -> str:
    value = str(raw or "full").strip().lower()
    if value in {"percentage", "percent", "investment_percentage"}:
        return "percentage"
    if value in {"cash", "reserve", "cash_reserve", "fixed_cash"}:
        return "cash"
    return "full"


def normalize_currencies(raw: Any) -> tuple[str, ...]:
    values = list(raw) if isinstance(raw, list) else [raw or "CAD"]
    if any(str(value or "").upper() == "BOTH" for value in values):
        values.extend(["CAD", "USD"])
    normalized: list[str] = []
    for value in values:
        code = str(value or "").upper()
        if code == "BOTH":
            continue
        if code in {"CAD", "USD"} and code not in normalized:
            normalized.append(code)
    return tuple(normalized or ["CAD"])


def normalize_requested_symbols(symbols: list[Any], markets: list[Any]) -> list[str]:
    cleaned: list[str] = []
    selected_markets = [str(market).upper() for market in markets if str(market).upper() in EXCHANGES]
    default_market = selected_markets[0] if len(selected_markets) == 1 else None
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if not text:
            continue
        normalized = normalize_symbol(text, default_market)
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


MAX_EXPLORER_CANDIDATES = 10


def analyze_candidates(profile: MarketExplorerProfile, on_progress: ProgressCallback | None = None) -> list[dict[str, Any]]:
    if profile.preferred_instruments:
        symbols = list(profile.preferred_instruments)[:MAX_EXPLORER_CANDIDATES]
        emit_progress(
            on_progress,
            stage="universe",
            message=f"Using your {len(symbols)} preferred instrument(s): {', '.join(symbols)}. Market discovery is skipped and deep analysis starts now.",
            candidates=symbols,
            route="preferred-instruments",
        )
    else:
        universe = explorer_universe_details(profile)
        symbols = [row["symbol"] for row in universe][:MAX_EXPLORER_CANDIDATES]
        sectors = list(dict.fromkeys(row["sector"] for row in universe if row["symbol"] in symbols))
        emit_progress(
            on_progress,
            stage="universe",
            message=(
                f"Discovery universe: {len(symbols)} instruments across {', '.join(profile.markets or tuple(EXCHANGES))} "
                f"covering {', '.join(sectors)}."
            ),
            candidates=symbols,
            route="explorer",
            universe=universe[:MAX_EXPLORER_CANDIDATES],
        )
    rows: list[dict[str, Any]] = []
    total = len(symbols)
    for index, symbol in enumerate(symbols):
        emit_progress(
            on_progress,
            stage="evidence",
            message=f"Analyzing {symbol} ({index + 1}/{total}): 5-year history, trading centre, pullback bands, earnings, news, and PESTEL...",
            symbol=symbol,
            index=index,
            total=total,
        )
        try:
            analysis = focused_analysis_payload(
                symbol,
                period="5y",
                interval="1d",
                refresh=True,
                intent="Hold / Watch" if profile.intent == "Not sure" else profile.intent,
                strategy=profile.strategy,
                risk_profile_name=profile.risk_appetite,
            )
        except Exception as exc:
            row = {"symbol": symbol, "ok": False, "score": 0.0, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            emit_progress(on_progress, stage="evidence", message=f"{symbol}: analysis failed ({type(exc).__name__}).", symbol=symbol, index=index + 1, total=total, candidate=compact_candidate(row))
            continue
        if not analysis.get("ok"):
            row = {"symbol": symbol, "ok": False, "score": 0.0, "error": str(analysis.get("status") or "No analysis available")}
            rows.append(row)
            emit_progress(on_progress, stage="evidence", message=f"{symbol}: no market data was returned.", symbol=symbol, index=index + 1, total=total, candidate=compact_candidate(row))
            continue
        row = candidate_summary(analysis, profile)
        rows.append(row)
        emit_progress(
            on_progress,
            stage="evidence",
            message=f"{symbol}: {row.get('action')} · evidence score {row.get('score')} · {row.get('sector')}",
            symbol=symbol,
            index=index + 1,
            total=total,
            candidate=compact_candidate(row),
        )
    return sorted(rows, key=lambda item: float(item.get("score") or 0), reverse=True)


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("symbol", "name", "sector", "currency", "market", "currentPrice", "score", "action", "confidence", "ok", "error")
    return {key: row.get(key) for key in keys if key in row}


def desired_sectors(profile: MarketExplorerProfile) -> list[str]:
    parts = [item.strip() for item in str(profile.desired_sector or "").split(",") if item.strip()]
    result: list[str] = []
    for part in parts:
        if part.lower() in {"infer", "ai infer", "none", "any"}:
            continue
        sector = canonical_sector(part)
        if sector and sector not in result:
            result.append(sector)
    return result


def sector_tilt(profile: MarketExplorerProfile) -> tuple[str, ...]:
    if profile.purpose.lower() == "income":
        return SECTOR_TILTS["Income"]
    for key in ("Conservative", "Growth"):
        if profile.risk_appetite.lower().startswith(key.lower()):
            return SECTOR_TILTS[key]
    return SECTOR_TILTS["Balanced"]


def explorer_universe_details(profile: MarketExplorerProfile, limit: int = MAX_EXPLORER_CANDIDATES) -> list[dict[str, str]]:
    """Choose a sector-diverse, market-balanced discovery universe.

    Desired sectors are honoured first. Otherwise sectors are ordered by the
    investor's risk/purpose tilt and instruments are taken round-robin across
    sectors and markets, so one exchange or one theme cannot dominate the
    shortlist that goes into deep analysis.
    """

    markets = tuple(profile.markets) or tuple(EXCHANGES)
    rows = explorer_universe_rows(markets)
    if not rows:
        return [{"symbol": symbol, "sector": "Unassigned", "market": "Mixed"} for symbol in DEFAULT_WATCHLISTS["Mixed"][:limit]]
    wanted = desired_sectors(profile)
    ordered_sectors = wanted + [sector for sector in sector_tilt(profile) if sector not in wanted]
    by_sector: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        by_sector.setdefault(row["sector"], {}).setdefault(row["market"], []).append(row)
    selected: list[dict[str, str]] = []

    def take(sectors: list[str], cap: int) -> None:
        while len(selected) < cap:
            progressed = False
            for sector in sectors:
                if len(selected) >= cap:
                    break
                buckets = by_sector.get(sector) or {}
                market_order = [market for market in markets if buckets.get(market)]
                if not market_order:
                    continue
                market = market_order[len(selected) % len(market_order)]
                selected.append(buckets[market].pop(0))
                progressed = True
            if not progressed:
                break

    if wanted:
        take(wanted, limit)
        if len(selected) < 3:
            take([sector for sector in ordered_sectors if sector not in wanted], limit)
    else:
        take(ordered_sectors, limit)
    return selected[:limit]


def explorer_universe(profile: MarketExplorerProfile) -> list[str]:
    return [row["symbol"] for row in explorer_universe_details(profile)]


def candidate_summary(analysis: dict[str, Any], profile: MarketExplorerProfile) -> dict[str, Any]:
    symbol = str(analysis.get("symbol") or "")
    snapshot = analysis.get("snapshot") or {}
    signal = analysis.get("signal") or {}
    latest = latest_history_row(analysis.get("history") or [])
    volume_profile = analysis.get("volumeProfile") or []
    sector = str(snapshot.get("sector") or snapshot.get("industry") or "Unassigned")
    wanted_sectors = desired_sectors(profile)
    sector_match = not wanted_sectors or canonical_sector(sector) in wanted_sectors
    base_score = _num(signal.get("confidence"), 45.0)
    rsi = _num(latest.get("RSI14"), 50.0)
    trend = _num(latest.get("TrendScore"), 0.0)
    sentiment = _num(analysis.get("newsSentiment"), 0.0)
    sector_bonus = 5.0 if sector_match else -4.0
    score = max(1.0, min(99.0, base_score + sector_bonus + trend * 8.0 + sentiment * 5.0))
    historical = historical_trend_context(analysis.get("history") or [])
    modules = build_instrument_analysis_modules(analysis, profile, snapshot, latest, signal, historical)
    deep_evidence = build_deep_evidence(
        analysis.get("history") or [],
        snapshot,
        analysis.get("financialStatements") if isinstance(analysis.get("financialStatements"), dict) else {},
        analysis.get("earnings") if isinstance(analysis.get("earnings"), dict) else {},
        analysis.get("news") if isinstance(analysis.get("news"), list) else [],
        intent="Hold / Watch" if profile.intent == "Not sure" else profile.intent,
        strategy=profile.strategy,
        risk_profile=profile.risk_appetite,
    )
    return {
        "symbol": symbol,
        "name": str(snapshot.get("longName") or snapshot.get("shortName") or symbol),
        "sector": sector,
        "sectorGroup": canonical_sector(sector) or "Unassigned",
        "currency": str(snapshot.get("currency") or ""),
        "market": str(snapshot.get("exchange") or ""),
        "currentPrice": _num(snapshot.get("currentPrice") or snapshot.get("regularMarketPrice"), _num(latest.get("Close"), 0.0)),
        "intent": "Hold / Watch" if profile.intent == "Not sure" else profile.intent,
        "strategy": profile.strategy,
        "score": round(score, 1),
        "action": str(signal.get("action") or "WAIT"),
        "confidence": _num(signal.get("confidence"), score),
        "rsi14": round(rsi, 1),
        "ema20": _num(latest.get("EMA20"), 0.0),
        "volumeGapZones": len(volume_profile),
        "newsSentiment": round(sentiment, 3),
        "historicalTrend": historical,
        "analysisModules": modules,
        "pullbackAnalysis": modules["pullback"],
        "earningsEvidence": modules["earnings"],
        "newsEvidence": modules["news"],
        "pestelEvidence": modules["pestel"],
        "thesis": modules["thesis"],
        "scenarioModel": modules["scenarios"],
        "deepEvidence": deep_evidence,
        "descriptiveRead": descriptive_read(snapshot, latest, signal, historical),
        "predictiveRead": predictive_read(signal, score, historical),
        "drivers": list(signal.get("reasons") or [])[:4],
        "risks": [signal.get("risk_note") or signal.get("riskNote") or "Monitor position size, liquidity, and event risk."],
        "href": f"/watchlist/{symbol}/summary",
    }


def build_instrument_analysis_modules(
    analysis: dict[str, Any],
    profile: MarketExplorerProfile,
    snapshot: dict[str, Any],
    latest: dict[str, Any],
    signal: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    """Run every evidence module for one instrument before provider synthesis."""
    price = _num(snapshot.get("currentPrice") or snapshot.get("regularMarketPrice"), _num(latest.get("Close"), 0.0))
    pullback = pullback_analysis(profile, historical, price, signal)
    earnings = earnings_evidence(analysis, snapshot)
    news = news_evidence(analysis.get("news") or [], analysis.get("newsSentiment"))
    pestel = instrument_pestel_context(snapshot, profile, news)
    thesis = thesis_classification(profile, historical, snapshot, signal, pullback)
    scenarios = scenario_analysis(profile, historical, price, pullback, thesis)
    return {
        "historical": historical,
        "pullback": pullback,
        "earnings": earnings,
        "news": news,
        "pestel": pestel,
        "thesis": thesis,
        "scenarios": scenarios,
    }


def build_local_plan(
    profile: MarketExplorerProfile,
    candidates: list[dict[str, Any]],
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    viable = [item for item in candidates if item.get("ok", True) is not False and _num(item.get("currentPrice"), 0) > 0]
    selected = viable[:6]
    fx_quote = build_fx_quote(profile, selected)
    allocation = allocate_budget(profile, selected, fx_quote)
    alternatives = build_strategy_alternatives(profile, viable)
    summary = build_executive_summary(profile, selected)
    return {
        "mode": "local",
        "provider": "local-rules",
        "model": "deterministic-technical-news-allocation-v1",
        "allocationPolicy": budget_allocation_policy(profile),
        "aiDisclosure": {
            "aiUsed": False,
            "fallbackUsed": True,
            "label": "Local model",
            "message": (
                f"{fallback_reason} {LOCAL_DISCLOSURE}"
                if fallback_reason
                else LOCAL_DISCLOSURE
            ),
            "fallbackReason": fallback_reason,
        },
        "executiveSummary": summary,
        "candidates": candidates,
        "selectedInstruments": selected,
        "aiSelectedInstruments": [],
        "allocation": allocation,
        "alternatives": alternatives,
        "executionPlan": build_execution_plan(selected),
        "excludedInstruments": excluded_instrument_rows(
            profile,
            candidates,
            [str(item.get("symbol") or "") for item in selected],
        ),
        "pestelContext": local_pestel_context(profile, selected),
        "fxQuote": fx_quote,
        "notFinancialAdvice": True,
        "futureProviderNote": FUTURE_PROVIDER_NOTE,
    }


MARKET_EXPLORER_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "executiveSummary": {"type": "string"},
        "selectedInstruments": {"type": "array", "description": "Provider's preferred instruments; the application will union these with the local evidence baseline.", "items": {"type": "object", "additionalProperties": True}},
        "allocation": {"type": "array", "description": "Allocation rationale only; the application owns authoritative amounts, currencies, and cash reserve rows.", "items": {"type": "object", "additionalProperties": True}},
        "alternatives": {"type": "array", "description": "Secondary intent and strategy comparisons.", "items": {"type": "object", "additionalProperties": True}},
        "executionPlan": {"type": "array", "description": "Provider execution notes; the application will fill missing instruments from local evidence.", "items": {"type": "object", "additionalProperties": True}},
        "excludedInstruments": {"type": "array", "description": "Optional provider explanations; the application recomputes authoritative exclusions from the evidence set.", "items": {"type": "object", "additionalProperties": True}},
        "pestelContext": {"type": "array", "description": "Political, economic, social, technological, environmental, and legal context.", "items": {"type": "object", "additionalProperties": True}},
    },
}


def maybe_generate_openai_plan(
    profile: MarketExplorerProfile,
    candidates: list[dict[str, Any]],
    local_plan: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ask the active AI provider for the synthesized plan.

    The provider chain (OpenAI now, Qwen reserved, local rules last) lives in
    ``.services.ai``; this function only prepares the evidence
    prompt and reconciles the reply with the authoritative local plan.
    """

    model = market_explorer_model()
    prompt = {
        "task": "Create an AI market explorer output for an educational stock research . Do not provide financial advice.",
        "instruction": "When intent is Not sure, compare every intent in intentOptions before recommending a conditional action. Treat Not sure as uncertainty, not as a trade action.",
        "analysisMethod": "Run every module for every candidate instrument: multi-year trading-centre and range/reversion analysis; risk/strategy/intent-sensitive pullback bands and entry zones; earnings metrics and event catalysts; current and forward PESTEL; rich news evidence; thesis classification; and an adjusted conclusion that reconciles contradictory signals or observations. Then use those modules together to produce the recommended plan and execution checklist.",
        "profileCriteriaInstruction": "Risk appetite, strategy, and intent are decision inputs, not labels. They must change the preferred entry zone, scenario horizon, sizing caution, and execution recommendation for each instrument.",
        "outputStyle": "Write a descriptive read first (what happened and why), then an affirmative but conditional predictive conclusion with the base, worst, and best case and the evidence that would invalidate it.",
        "requiredSections": [
            "executiveSummary",
            "selectedInstruments",
            "allocation",
            "alternatives",
            "executionPlan",
            "pestelContext",
        ],
        "resultContract": MARKET_EXPLORER_RESULT_LAYOUT,
        "profile": profile_to_payload(profile),
        "allocationPolicy": budget_allocation_policy(profile),
        "candidateEvidence": candidates[:10],
        "localDraft": local_plan,
        "styleReference": "Use the attached PDF style: executive summary, exact allocation, rationale, scenarios, catalysts, risks, and checklist.",
        "historicalTrendInstruction": "Treat longer-horizon evidence as crucial. Compare YTD and one-year behavior, established trading range, trend persistence, and distance from the high; do not call a pullback a growth opportunity from the peak alone.",
        "scenarioInstruction": "For every instrument, explain the base range, worst-case range test, best-case range test, historical containment proxy, and what would invalidate the scenario. State clearly that these are evidence-based heuristics, not guarantees.",
        "preferredInstrumentInstruction": "When preferred instruments come from a workspace, do not silently omit them. Include each in selectedInstruments or explain it under excludedInstruments with a specific reason based on the supplied evidence.",
        "allocationInstruction": "The allocation policy is authoritative. Use exactly its investable amount and cash reserve. Do not create a synthetic cash-reserve row unless cashReserveAmount is greater than zero, and never replace a cash reserve with a tradable instrument.",
        "alternativesInstruction": "For alternatives, keep the localDraft ranking evidence and explain how each secondary strategy/intent changes the preferred instruments and sector mix so the user can compare across sectors.",
    }
    emit_progress(
        on_progress,
        stage="prediction",
        message=f"Requesting the AI synthesis from {model}: descriptive read, predictive conclusion, allocation rationale, and execution checklist...",
        provider="openai",
        model=model,
        outcome="attempt",
    )
    result = generate_structured_json(
        ProviderRequest(
            task="market-explorer",
            prompt=prompt,
            schema=MARKET_EXPLORER_PLAN_SCHEMA,
            schema_name="ai_market_explorer_plan",
            strict=False,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            reasoning_effort=OPENAI_REASONING_EFFORT,
            model=model,
            log_prefix="market_explorer_provider",
            logger=logger,
            context={"candidate_count": len(candidates)},
        )
    )
    if not result.ok:
        emit_progress(
            on_progress,
            stage="prediction",
            message=f"AI synthesis unavailable: {result.reason} Deterministic local rules will complete the plan.",
            provider=result.provider,
            model=result.model,
            outcome="fallback",
        )
        return None, result.reason
    try:
        parsed = reconcile_provider_plan(profile, candidates, local_plan, dict(result.payload or {}))
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        reason = "OpenAI returned a response the application could not parse."
        logger.warning(
            "market_explorer_provider provider=%s outcome=fallback reason=parse_error fallback=local-rules model=%s error_type=%s",
            result.provider,
            result.model,
            type(error).__name__,
        )
        emit_progress(on_progress, stage="prediction", message=f"AI synthesis could not be reconciled ({type(error).__name__}); using deterministic local rules.", provider=result.provider, model=result.model, outcome="fallback")
        return None, reason
    emit_progress(
        on_progress,
        stage="prediction",
        message=f"AI synthesis received from {result.model} in {(result.elapsed_ms or 0) / 1000:.1f}s; reconciling it with the local evidence baseline.",
        provider=result.provider,
        model=result.model,
        outcome="ready",
    )
    return {
        **local_plan,
        **parsed,
        "mode": "ai",
        "provider": result.provider,
        "model": result.model,
        "aiDisclosure": {
            "aiUsed": True,
            "fallbackUsed": False,
            "label": f"AI: {result.model}",
            "message": AI_DISCLOSURE,
            "provider": result.provider,
            "elapsedMs": result.elapsed_ms,
        },
        "fxQuote": local_plan.get("fxQuote"),
        "allocationPolicy": local_plan.get("allocationPolicy"),
        "notFinancialAdvice": True,
        "futureProviderNote": FUTURE_PROVIDER_NOTE,
    }, None


def reconcile_provider_plan(
    profile: MarketExplorerProfile,
    candidates: list[dict[str, Any]],
    local_plan: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """Keep provider narrative while local market facts stay authoritative."""

    candidate_currency = {
        str(item.get("symbol") or ""): str(item.get("currency") or inferred_currency(str(item.get("symbol") or "")))
        for item in candidates
    }
    if isinstance(parsed.get("allocation"), list):
        parsed["allocation"] = [
            {
                **row,
                "currency": candidate_currency.get(str(row.get("symbol") or ""), inferred_currency(str(row.get("symbol") or ""))),
            }
            for row in parsed["allocation"]
            if isinstance(row, dict)
        ]
        parsed["allocation"] = merge_ai_allocation_with_currency(
            profile,
            parsed["allocation"],
            local_plan.get("allocation") or [],
        )
    if isinstance(parsed.get("selectedInstruments"), list):
        default_intent = "Hold / Watch" if profile.intent == "Not sure" else profile.intent
        normalized_selected = [
            {
                **row,
                "currency": str(row.get("currency") or candidate_currency.get(str(row.get("symbol") or ""), inferred_currency(str(row.get("symbol") or "")))),
                "intent": str(row.get("intent")) if str(row.get("intent") or "") in {"Buy / Add", "Sell / Trim", "Hold / Watch"} else default_intent,
                "strategy": str(row.get("strategy")) if str(row.get("strategy") or "") in {"Short-term (1-4 weeks)", "Long-term (6-12 months)", "Buy-dip"} else profile.strategy,
            }
            for row in parsed["selectedInstruments"]
            if isinstance(row, dict) and str(row.get("symbol") or "").strip()
        ]
        if normalized_selected:
            parsed["selectedInstruments"] = normalized_selected
            parsed["aiSelectedInstruments"] = normalized_selected
        else:
            parsed.pop("selectedInstruments", None)
            parsed["aiSelectedInstruments"] = []
    ai_selected = rows_from_value(parsed.get("aiSelectedInstruments"))
    local_selected = rows_from_value(local_plan.get("selectedInstruments"))
    parsed["selectedInstruments"] = merge_selected_instruments(ai_selected, local_selected)
    selected_symbols = [str(item.get("symbol") or "") for item in parsed["selectedInstruments"]]
    parsed["executionPlan"] = merge_execution_plans(
        parsed.get("executionPlan") if isinstance(parsed.get("executionPlan"), list) else [],
        local_plan.get("executionPlan") if isinstance(local_plan.get("executionPlan"), list) else [],
    )
    parsed["excludedInstruments"] = excluded_instrument_rows(profile, candidates, selected_symbols)
    provider_alternatives = rows_from_value(parsed.get("alternatives"))
    local_alternatives = rows_from_value(local_plan.get("alternatives"))
    if local_alternatives:
        parsed["alternatives"] = merge_alternatives(provider_alternatives, local_alternatives)
    return parsed


def merge_alternatives(provider_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach provider commentary to the locally ranked strategy/intent comparisons."""

    provider_by_key = {
        (str(row.get("strategy") or "").strip(), str(row.get("intent") or "").strip()): row
        for row in provider_rows
    }
    merged: list[dict[str, Any]] = []
    for row in local_rows:
        key = (str(row.get("strategy") or "").strip(), str(row.get("intent") or "").strip())
        provider_row = provider_by_key.get(key)
        if provider_row:
            merged.append({**row, "providerNote": str(provider_row.get("note") or provider_row.get("comparison") or provider_row.get("summary") or "").strip() or None})
        else:
            merged.append(row)
    return merged


def extract_response_json(payload: dict[str, Any]) -> dict[str, Any] | None:
    return extract_json_object(payload)


def market_explorer_model() -> str:
    return resolve_model("market-explorer")


def inferred_currency(symbol: str) -> str:
    cleaned = str(symbol or "").upper()
    return "CAD" if cleaned.endswith((".TO", ".V")) else "USD"


def build_fx_quote(profile: MarketExplorerProfile, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    base_currency = profile.budget_currency
    instrument_currencies = {
        str(item.get("currency") or inferred_currency(str(item.get("symbol") or ""))).upper()
        for item in candidates
    }
    mixed_currency = any(currency != base_currency for currency in instrument_currencies)
    quote: dict[str, Any] = {
        "required": mixed_currency,
        "baseCurrency": base_currency,
        "warning": "Exchange rates can change; confirm the current rate before acting.",
    }
    if not mixed_currency:
        return quote

    try:
        frame, status = fetch_yahoo_history("USDCAD=X", period="5d", interval="1d", refresh=False)
        if frame.empty:
            raise ValueError("No FX quote returned")
        row = frame.sort_values("Date").iloc[-1]
        usdcad = _num(row.get("Close"), 0.0)
        if usdcad <= 0:
            raise ValueError("Invalid FX quote returned")
        raw_as_of = row.get("Date")
        as_of = str(raw_as_of)
        try:
            parsed_as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            if parsed_as_of.tzinfo is None:
                parsed_as_of = parsed_as_of.replace(tzinfo=timezone.utc)
            as_of = parsed_as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
        quote.update({
            "pair": "USDCAD=X",
            "rate": round(usdcad, 6),
            "rateLabel": f"1 USD = C${usdcad:.4f}" if base_currency == "CAD" else f"1 CAD = US${1 / usdcad:.4f}",
            "source": f"Yahoo Finance FX quote (USDCAD=X; {status.source})",
            "asOf": as_of,
        })
    except Exception as exc:
        logger.warning(
            "market_explorer_fx outcome=unavailable pair=USDCAD=X base_currency=%s "
            "error_type=%s",
            base_currency,
            type(exc).__name__,
        )
        quote.update({
            "pair": "USDCAD=X",
            "rate": None,
            "rateLabel": "FX quote unavailable",
            "source": "Yahoo Finance (USDCAD=X)",
            "asOf": None,
        })
    return quote


def allocation_conversion_rate(base_currency: str, instrument_currency: str, fx_quote: dict[str, Any]) -> float | None:
    if base_currency == instrument_currency:
        return 1.0
    usdcad = _num(fx_quote.get("rate"), 0.0)
    if usdcad <= 0:
        return None
    if base_currency == "CAD" and instrument_currency == "USD":
        return 1 / usdcad
    if base_currency == "USD" and instrument_currency == "CAD":
        return usdcad
    return None


def allocate_budget(profile: MarketExplorerProfile, candidates: list[dict[str, Any]], fx_quote: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not candidates or profile.budget <= 0:
        return []
    scores = [max(1.0, _num(item.get("score"), 1.0)) for item in candidates]
    total_score = sum(scores)
    max_weight = 0.2 if profile.risk_appetite.lower().startswith("conservative") else 0.3 if profile.risk_appetite.lower().startswith("balanced") else 0.4
    raw_weights = [score / total_score for score in scores]
    capped = [min(weight, max_weight) for weight in raw_weights]
    total_capped = sum(capped) or 1.0
    normalized = [weight / total_capped for weight in capped]
    fx_quote = fx_quote or build_fx_quote(profile, candidates)
    reserve_amount = budget_reserve_amount(profile)
    investable_budget = max(0.0, profile.budget - reserve_amount)
    investable_ratio = investable_budget / profile.budget if profile.budget else 0.0
    rows: list[dict[str, Any]] = []
    for item, weight in zip(candidates, normalized):
        price = _num(item.get("currentPrice"), 0.0)
        allocation_weight = weight * investable_ratio
        base_amount = investable_budget * weight
        instrument_currency = str(item.get("currency") or inferred_currency(str(item.get("symbol") or ""))).upper()
        conversion_rate = allocation_conversion_rate(profile.budget_currency, instrument_currency, fx_quote)
        amount = base_amount * conversion_rate if conversion_rate is not None else None
        rows.append(
            {
                "symbol": item.get("symbol"),
                "weight": round(allocation_weight, 4),
                "amount": round(amount, 2) if amount is not None else None,
                "budgetAmount": round(base_amount, 2),
                "budgetCurrency": profile.budget_currency,
                "approxShares": round(amount / price, 3) if amount is not None and price > 0 else None,
                "currency": instrument_currency,
                "fxApplied": instrument_currency != profile.budget_currency,
                "fxRate": round(conversion_rate, 8) if conversion_rate is not None and instrument_currency != profile.budget_currency else None,
                "fxSource": fx_quote.get("source") if instrument_currency != profile.budget_currency else None,
                "fxAsOf": fx_quote.get("asOf") if instrument_currency != profile.budget_currency else None,
                "rationale": f"{item.get('action', 'Monitor')} with score {item.get('score', 0)}; keep position size within {profile.risk_appetite.lower()} risk limits.",
                "href": item.get("href"),
            }
        )
    if reserve_amount > 0:
        reserve_symbol = f"CASH_RESERVE_{profile.budget_currency}"
        rows.append(
            {
                "symbol": reserve_symbol,
                "isCashReserve": True,
                "weight": round(reserve_amount / profile.budget, 4),
                "amount": round(reserve_amount, 2),
                "budgetAmount": round(reserve_amount, 2),
                "budgetCurrency": profile.budget_currency,
                "approxShares": None,
                "currency": profile.budget_currency,
                "fxApplied": False,
                "fxRate": None,
                "fxSource": None,
                "fxAsOf": None,
                "rationale": "Held as cash reserve according to the user's budget allocation preference; no security purchase is intended.",
                "href": None,
            }
        )
    return rows


def budget_reserve_amount(profile: MarketExplorerProfile) -> float:
    if profile.budget <= 0 or profile.budget_allocation_mode == "full":
        return 0.0
    if profile.budget_allocation_mode == "percentage":
        invested_percentage = min(100.0, max(0.0, profile.budget_allocation_value))
        return round(profile.budget * (1.0 - invested_percentage / 100.0), 2)
    return round(min(profile.budget, max(0.0, profile.budget_allocation_value)), 2)


def budget_allocation_policy(profile: MarketExplorerProfile) -> dict[str, Any]:
    reserve_amount = budget_reserve_amount(profile)
    return {
        "mode": profile.budget_allocation_mode,
        "value": profile.budget_allocation_value,
        "budget": round(profile.budget, 2),
        "budgetCurrency": profile.budget_currency,
        "investableAmount": round(max(0.0, profile.budget - reserve_amount), 2),
        "cashReserveAmount": reserve_amount,
        "cashReserveSymbol": f"CASH_RESERVE_{profile.budget_currency}" if reserve_amount > 0 else None,
    }


def merge_ai_allocation_with_currency(
    profile: MarketExplorerProfile,
    ai_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    local_by_symbol = {str(row.get("symbol") or "").upper(): row for row in local_rows}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ai_rows:
        symbol = str(row.get("symbol") or "").upper()
        local_row = local_by_symbol.get(symbol)
        if local_row:
            merged.append({**row, **local_row, "rationale": row.get("rationale") or local_row.get("rationale")})
            seen.add(symbol)
    merged.extend(row for row in local_rows if str(row.get("symbol") or "").upper() not in seen)
    return merged


def merge_selected_instruments(
    provider_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union provider and local rows while keeping local market facts authoritative."""
    local_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in local_rows
        if str(row.get("symbol") or "").strip()
    }
    provider_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in provider_rows
        if str(row.get("symbol") or "").strip()
    }
    ordered_symbols = list(provider_by_symbol)
    ordered_symbols.extend(symbol for symbol in local_by_symbol if symbol not in provider_by_symbol)
    local_fact_keys = {
        "symbol",
        "name",
        "sector",
        "currency",
        "market",
        "currentPrice",
        "rsi14",
        "ema20",
        "historicalTrend",
        "analysisModules",
        "pullbackAnalysis",
        "earningsEvidence",
        "newsEvidence",
        "pestelEvidence",
        "thesis",
        "scenarioModel",
        "deepEvidence",
        "href",
    }
    merged: list[dict[str, Any]] = []
    for symbol in ordered_symbols:
        local_row = local_by_symbol.get(symbol, {})
        provider_row = provider_by_symbol.get(symbol, {})
        row = {**local_row, **provider_row}
        for key in local_fact_keys:
            if local_row.get(key) is not None:
                row[key] = local_row[key]
        if provider_row and local_row:
            row["selectionSource"] = "provider-and-local"
        elif provider_row:
            row["selectionSource"] = "provider"
        else:
            row["selectionSource"] = "local-baseline"
            row["selectionNote"] = "Retained from local evidence because the provider response did not return this reviewed instrument."
        merged.append(row)
    return merged


def merge_execution_plans(
    ai_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep model-written execution notes while filling missing symbols locally."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ai_rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        merged.append(row)
        seen.add(symbol)
    for row in local_rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and symbol not in seen:
            merged.append(row)
            seen.add(symbol)
    return merged


def excluded_instrument_rows(
    profile: MarketExplorerProfile,
    candidates: list[dict[str, Any]],
    selected_symbols: list[str],
) -> list[dict[str, Any]]:
    """Make every reviewed-but-omitted instrument visible with an explanation."""
    selected = {str(symbol or "").strip().upper() for symbol in selected_symbols if str(symbol or "").strip()}
    rows: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol or symbol in selected:
            continue
        if item.get("ok", True) is False:
            reason = (
                "Excluded because current market analysis was unavailable: "
                f"{item.get('error') or 'No data returned.'}"
            )
        elif _num(item.get("currentPrice"), 0.0) <= 0:
            reason = "Excluded because a current price was unavailable, so allocation and execution checks could not be calculated."
        else:
            reason = "Reviewed but not included in the current recommendation because other instruments ranked higher for this plan."
        rows.append(
            {
                "symbol": symbol,
                "name": item.get("name") or symbol,
                "currency": item.get("currency") or inferred_currency(symbol),
                "market": item.get("market"),
                "ok": item.get("ok", True),
                "currentPrice": item.get("currentPrice"),
                "exclusionReason": reason,
                "href": item.get("href") or f"/watchlist/{symbol}/summary",
            }
        )
    requested = {str(symbol or "").strip().upper() for symbol in profile.preferred_instruments if str(symbol or "").strip()}
    known = {str(item.get("symbol") or "").strip().upper() for item in candidates if isinstance(item, dict)}
    for symbol in sorted(requested - known - selected):
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "currency": inferred_currency(symbol),
                "market": None,
                "ok": False,
                "currentPrice": None,
                "exclusionReason": "This workspace instrument was not returned by the market-data analysis, so it could not be ranked or allocated.",
                "href": f"/watchlist/{symbol}/summary",
            }
        )
    return rows


STRATEGY_OPTIONS = ("Short-term (1-4 weeks)", "Long-term (6-12 months)", "Buy-dip")
INTENT_OPTIONS = ("Buy / Add", "Sell / Trim", "Hold / Watch")


def build_strategy_alternatives(profile: MarketExplorerProfile, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank the reviewed instruments under every other strategy/intent pair.

    The comparison reuses the evidence already computed for each candidate
    (rule action, RSI, trend, pullback bands, thesis, fundamentals, income) so
    the user can see how the preferred instruments and sector mix shift when
    the objective changes, instead of receiving one generic sentence.
    """

    viable = [item for item in candidates if item.get("ok", True) is not False and _num(item.get("currentPrice"), 0) > 0]
    primary_intent = "Hold / Watch" if profile.intent == "Not sure" else profile.intent
    intents = list(INTENT_OPTIONS) if profile.intent == "Not sure" else list(dict.fromkeys([primary_intent, *INTENT_OPTIONS]))
    primary = rank_for_profile(viable, profile.strategy, primary_intent)
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGY_OPTIONS:
        for intent in intents:
            if strategy == profile.strategy and intent == primary_intent:
                continue
            ranked = rank_for_profile(viable, strategy, intent)
            rows.append(
                {
                    "strategy": strategy,
                    "intent": intent,
                    "note": alternative_note(strategy, intent, ranked),
                    "rankedInstruments": ranked[:4],
                    "sectorMix": sector_mix(ranked[:4]),
                    "leadSector": ranked[0]["sector"] if ranked else None,
                    "comparison": comparison_note(primary, ranked, profile.strategy, primary_intent, strategy, intent),
                }
            )
    return rows


def rank_for_profile(candidates: list[dict[str, Any]], strategy: str, intent: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in candidates:
        score, reasons = alternative_score(item, strategy, intent)
        ranked.append(
            {
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "sector": item.get("sector") or "Unassigned",
                "currency": item.get("currency") or inferred_currency(str(item.get("symbol") or "")),
                "market": item.get("market"),
                "currentPrice": item.get("currentPrice"),
                "action": item.get("action"),
                "baseScore": item.get("score"),
                "score": round(score, 1),
                "stance": stance_for(intent, score),
                "reason": "; ".join(reasons) if reasons else "Ranked on the base evidence score only.",
            }
        )
    return sorted(ranked, key=lambda row: float(row.get("score") or 0), reverse=True)


def alternative_score(item: dict[str, Any], strategy: str, intent: str) -> tuple[float, list[str]]:
    base = _num(item.get("score"), 40.0)
    history = item.get("historicalTrend") if isinstance(item.get("historicalTrend"), dict) else {}
    modules = item.get("analysisModules") if isinstance(item.get("analysisModules"), dict) else {}
    pullback = modules.get("pullback") if isinstance(modules.get("pullback"), dict) else {}
    thesis = modules.get("thesis") if isinstance(modules.get("thesis"), dict) else {}
    deep = item.get("deepEvidence") if isinstance(item.get("deepEvidence"), dict) else {}
    fundamentals = deep.get("fundamentals") if isinstance(deep.get("fundamentals"), dict) else {}
    metrics = fundamentals.get("metrics") if isinstance(fundamentals.get("metrics"), dict) else {}
    income = deep.get("incomeAndCapitalPreservation") if isinstance(deep.get("incomeAndCapitalPreservation"), dict) else {}
    rsi = _num(item.get("rsi14"), 50.0)
    sentiment = _num(item.get("newsSentiment"), 0.0)
    ytd = _num(history.get("ytdReturnPct"), 0.0)
    distance = _num(history.get("distanceFromCentrePct"), 0.0)
    volatility = _num(history.get("recentYearVolatilityPct") or history.get("dailyVolatilityPct"), 0.0)
    containment = _num(history.get("historicalRangeContainmentPct"), 0.0)
    pullback_pct = _num(pullback.get("pullbackPct"), 0.0)
    active_band = any(isinstance(band, dict) and band.get("active") for band in (pullback.get("bands") or []))
    revenue_growth = _num(metrics.get("revenueGrowth"), float("nan"))
    yield_pct = _num(income.get("currentYieldPct"), 0.0)
    classification = str(thesis.get("classification") or "")
    action = str(item.get("action") or "WAIT").upper()
    score = base
    reasons: list[str] = []
    lowered = strategy.lower()
    if "short" in lowered:
        if 45 <= rsi <= 65:
            score += 6
            reasons.append(f"RSI {rsi:.0f} leaves room for a short-term move")
        elif rsi > 70:
            score -= 6
            reasons.append(f"RSI {rsi:.0f} is stretched for a new short-term entry")
        elif rsi < 30:
            score += 3
            reasons.append(f"RSI {rsi:.0f} sets up a mean-reversion bounce")
        score += max(-8.0, min(8.0, sentiment * 10))
        if abs(sentiment) >= 0.15:
            reasons.append(f"news sentiment {sentiment:+.2f}")
        score += min(volatility, 3.0) * 1.5
        if volatility >= 1.5:
            reasons.append(f"{volatility:.1f}% daily volatility supports short-horizon trading")
        if ytd > 0:
            score += 3
    elif "long" in lowered:
        if classification.startswith("Growth"):
            score += 10
            reasons.append("growth-supported thesis")
        elif classification.startswith(("Range-bound", "Steady")):
            score -= 5
            reasons.append("range-bound thesis limits long-term upside")
        score += max(-8.0, min(8.0, ytd / 5))
        if abs(ytd) >= 5:
            reasons.append(f"YTD trend {ytd:+.1f}%")
        if revenue_growth == revenue_growth and revenue_growth >= 0.05:
            score += 6
            reasons.append(f"revenue growth {revenue_growth:+.1%}")
        if volatility and volatility < 1.5:
            score += 3
            reasons.append("lower volatility suits a 6-12 month hold")
        if yield_pct >= 3:
            score += 3
            reasons.append(f"{yield_pct:.1f}% yield adds total return")
    else:
        if active_band:
            score += 10
            reasons.append("price sits inside an active pullback band")
        if 6 <= pullback_pct <= 15:
            score += 6
            reasons.append(f"{pullback_pct:.1f}% below the one-year high")
        elif pullback_pct > 20:
            score -= 4
            reasons.append(f"{pullback_pct:.1f}% drawdown looks like a falling knife")
        if distance < 0:
            score += 4
            reasons.append("trading below its multi-year centre")
        if containment >= 80:
            score += 3
            reasons.append(f"{containment:.0f}% range containment supports reversion")
    if intent == "Buy / Add":
        if action.startswith("BUY"):
            score += 8
            reasons.append(f"rule action {action}")
        elif action.startswith("SELL") or "TRIM" in action:
            score -= 10
            reasons.append(f"rule action {action} conflicts with buying")
    elif intent == "Sell / Trim":
        if rsi > 70:
            score += 8
            reasons.append(f"RSI {rsi:.0f} is overbought")
        if action.startswith("SELL") or "TRIM" in action:
            score += 10
            reasons.append(f"rule action {action}")
        elif action.startswith("BUY"):
            score -= 8
            reasons.append(f"rule action {action} argues against trimming")
        if pullback_pct <= 3:
            score += 4
            reasons.append("price is near its one-year high")
        if sentiment < -0.15:
            score += 5
            reasons.append("negative news flow")
    else:
        if abs(distance) <= 5:
            score += 4
            reasons.append("price is close to its trading centre")
        if containment >= 80:
            score += 4
            reasons.append("stable historical range")
        if "WAIT" in action or "HOLD" in action:
            score += 4
            reasons.append(f"rule action {action}")
    return max(1.0, min(99.0, score)), reasons[:3]


def stance_for(intent: str, score: float) -> str:
    if intent == "Buy / Add":
        return "Strong buy candidate" if score >= 70 else "Staged entry candidate" if score >= 55 else "Watch only"
    if intent == "Sell / Trim":
        return "Trim candidate" if score >= 70 else "Review for trimming" if score >= 55 else "Hold for now"
    return "Core hold" if score >= 70 else "Monitor" if score >= 55 else "Reassess"


def sector_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for row in rows:
        sector = str(row.get("sector") or "Unassigned")
        mix[sector] = mix.get(sector, 0) + 1
    return mix


def comparison_note(
    primary: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    primary_strategy: str,
    primary_intent: str,
    strategy: str,
    intent: str,
) -> str:
    if not ranked:
        return "No candidate evidence is available for comparison yet."
    leader = ranked[0]
    primary_symbols = [str(row.get("symbol")) for row in primary]
    position = primary_symbols.index(str(leader.get("symbol"))) + 1 if str(leader.get("symbol")) in primary_symbols else None
    movement = (
        f"{leader.get('symbol')} stays the leader"
        if position == 1
        else f"{leader.get('symbol')} moves to the top from #{position}"
        if position
        else f"{leader.get('symbol')} enters the shortlist"
    )
    sectors = ", ".join(f"{sector} x{count}" for sector, count in sector_mix(ranked[:4]).items())
    return (
        f"Compared with your primary plan ({primary_strategy} / {primary_intent}), {strategy} / {intent} "
        f"means {movement} ({leader.get('stance')}, score {leader.get('score')}); the top four lean {sectors}."
    )


def build_execution_plan(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        price = _num(item.get("currentPrice"), 0.0)
        ema20 = _num(item.get("ema20"), 0.0)
        rsi = _num(item.get("rsi14"), 50.0)
        modules = item.get("analysisModules") if isinstance(item.get("analysisModules"), dict) else {}
        historical = modules.get("historical") if isinstance(modules.get("historical"), dict) else {}
        pullback = modules.get("pullback") if isinstance(modules.get("pullback"), dict) else {}
        scenarios = modules.get("scenarios") if isinstance(modules.get("scenarios"), dict) else {}
        thesis = modules.get("thesis") if isinstance(modules.get("thesis"), dict) else {}
        earnings = modules.get("earnings") if isinstance(modules.get("earnings"), dict) else {}
        news = modules.get("news") if isinstance(modules.get("news"), dict) else {}
        bands = pullback.get("bands") if isinstance(pullback.get("bands"), list) else []
        active_band = next((band.get("label") for band in bands if isinstance(band, dict) and band.get("active")), "none")
        event = earnings.get("event") if isinstance(earnings.get("event"), dict) else {}
        rows.append(
            {
                "symbol": item.get("symbol"),
                "action": item.get("action", "WAIT"),
                "confidence": item.get("confidence"),
                "flow": [
                    f"Current price: {price:.2f}",
                    f"Action: {item.get('action', 'WAIT')}",
                    f"Trading centre: {_num(historical.get('tradingCentre'), price):.2f}; established range: {(_num((historical.get('establishedRange') or {}).get('low'), price)):.2f}-{(_num((historical.get('establishedRange') or {}).get('high'), price)):.2f}.",
                    f"Scenario check: {str(scenarios.get('rangeStatement') or 'Range scenario unavailable')}",
                    f"Pullback entry: {str(pullback.get('guidance') or 'Wait for pullback or confirmation.')}",
                    f"Active pullback band: {active_band}.",
                    f"Thesis classification: {str(thesis.get('classification') or 'Mixed thesis')}",
                    f"Adjusted conclusion: {str(thesis.get('adjustedConclusion') or 'Reconcile price, business, and catalyst evidence before acting.')}",
                    f"Earnings event: {str(event.get('nextEarningsDate') or event.get('lastEarningsDate') or 'No dated event available')}.",
                    f"News evidence: {news.get('itemCount', 0)} recent symbol-specific item(s); sentiment {news.get('sentiment', 0):.2f}.",
                    f"Check RSI < 60: {'yes' if rsi < 60 else 'no'}",
                    f"Check price near EMA20: {'yes' if ema20 and abs(price / ema20 - 1) <= 0.03 else 'no'}",
                    "Execute partial scale-in only if the thesis, indicator, and risk checks agree.",
                ],
                "scenarioModel": scenarios,
                "entryZones": pullback,
                "thesis": thesis,
                "href": f"/watchlist/{item.get('symbol')}/signal",
            }
        )
    return rows


def build_executive_summary(profile: MarketExplorerProfile, candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No investable candidates were found from the selected markets and preferences. Add preferred instruments or widen the market selection."
    symbols = ", ".join(str(item.get("symbol")) for item in candidates[:4])
    currency_label = profile.budget_currency
    trend_note = str((candidates[0].get("historicalTrend") or {}).get("read") or "Longer-horizon trend context is included for each candidate.")
    return (
        f"For a {profile.risk_appetite.lower()} {profile.purpose.lower()} profile with a "
        f"{currency_label} {profile.budget:,.2f} budget, the strongest current candidates are {symbols}. "
        f"The plan combines descriptive technical/news evidence with a predictive confidence read. Longer-horizon check: {trend_note} "
        "Position sizing is constrained so no single instrument dominates the budget."
    )


def descriptive_read(snapshot: dict[str, Any], latest: dict[str, Any], signal: dict[str, Any], trend: dict[str, Any]) -> str:
    price = _num(snapshot.get("currentPrice") or latest.get("Close"), 0.0)
    rsi = _num(latest.get("RSI14"), 50.0)
    ema = _num(latest.get("EMA20"), 0.0)
    ema_read = "above" if ema and price >= ema else "below"
    return f"Price is {ema_read} the 20-day EMA with RSI at {rsi:.1f}; current rule action is {signal.get('action', 'WAIT')}. {trend.get('read', '')}"


def predictive_read(signal: dict[str, Any], score: float, trend: dict[str, Any]) -> str:
    action = str(signal.get("action") or "WAIT")
    trend_warning = " The longer-horizon trend is broadly flat, so a peak-based pullback alone is not enough to establish a growth entry." if trend.get("isFlat") else ""
    if score >= 70:
        return f"Affirmative but conditional: {action} has strong evidence, provided news and price confirmation remain aligned.{trend_warning}"
    if score >= 50:
        return f"Moderate conviction: {action} can be monitored, but use staged execution and confirmation checks.{trend_warning}"
    return f"Low conviction: {action} should wait until trend, RSI, and news evidence improve.{trend_warning}"


def historical_trend_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dated = history_points(rows)
    if not dated:
        return {
            "read": "Longer-horizon trend data is unavailable.",
            "isFlat": False,
            "dataQuality": "No valid dated closing prices were returned.",
        }
    latest_date, latest_close = dated[-1]
    closes = [close for _stamp, close in dated]
    ytd_rows = [(stamp, close) for stamp, close in dated if stamp.year == latest_date.year]
    baseline = ytd_rows[0][1] if ytd_rows else closes[0]
    ytd_return = (latest_close / baseline - 1) * 100 if baseline else 0.0
    ytd_closes = [close for _stamp, close in ytd_rows] or closes
    recent = dated[-252:]
    recent_closes = [close for _stamp, close in recent]
    daily_moves = [abs(closes[index] / closes[index - 1] - 1) * 100 for index in range(1, len(closes)) if closes[index - 1]]
    daily_returns = [(closes[index] / closes[index - 1] - 1) * 100 for index in range(1, len(closes)) if closes[index - 1]]
    recent_daily_moves = [abs(recent_closes[index] / recent_closes[index - 1] - 1) * 100 for index in range(1, len(recent_closes)) if recent_closes[index - 1]]
    recent_daily_returns = [(recent_closes[index] / recent_closes[index - 1] - 1) * 100 for index in range(1, len(recent_closes)) if recent_closes[index - 1]]
    average_move = sum(daily_moves) / len(daily_moves) if daily_moves else 0.0
    recent_average_move = sum(recent_daily_moves) / len(recent_daily_moves) if recent_daily_moves else 0.0
    volatility = (sum((value - (sum(daily_returns) / len(daily_returns))) ** 2 for value in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 0.0
    recent_volatility = (sum((value - (sum(recent_daily_returns) / len(recent_daily_returns))) ** 2 for value in recent_daily_returns) / len(recent_daily_returns)) ** 0.5 if recent_daily_returns else 0.0
    center = _percentile(closes, 50)
    lower_core = _percentile(closes, 25)
    upper_core = _percentile(closes, 75)
    established_low = _percentile(closes, 10)
    established_high = _percentile(closes, 90)
    containment = sum(established_low <= close <= established_high for close in closes) / len(closes) * 100 if closes else 0.0
    center_distance = (latest_close / center - 1) * 100 if center else 0.0
    is_flat = abs(ytd_return) < 3.0 or abs(center_distance) < 5.0 and abs(ytd_return) < 8.0
    direction = "up" if ytd_return >= 0 else "down"
    years = max(1.0, (latest_date - dated[0][0]).days / 365.25)
    range_read = (
        f"Across approximately {years:.1f} years, the trading centre is {center:.2f}; {containment:.0f}% of closes stayed within "
        f"the established {established_low:.2f}-{established_high:.2f} range. Average daily movement was {average_move:.2f}%."
    )
    if is_flat:
        range_read += " The price is near its established trading centre, so a pullback from a peak alone does not establish a growth trend."
    read = f"YTD trend is {direction} {abs(ytd_return):.1f}% from {baseline:.2f}, with a {min(ytd_closes):.2f}-{max(ytd_closes):.2f} range. {range_read}"
    return {
        "asOf": latest_date.isoformat(),
        "observations": len(dated),
        "ytdReturnPct": round(ytd_return, 2),
        "ytdStartPrice": round(baseline, 4),
        "ytdHigh": round(max(ytd_closes), 4),
        "ytdLow": round(min(ytd_closes), 4),
        "oneYearHigh": round(max(recent_closes), 4),
        "oneYearLow": round(min(recent_closes), 4),
        "multiYearHigh": round(max(closes), 4),
        "multiYearLow": round(min(closes), 4),
        "historyYears": round(years, 2),
        "tradingCentre": round(center, 4),
        "coreRange": {"low": round(lower_core, 4), "high": round(upper_core, 4)},
        "establishedRange": {"low": round(established_low, 4), "high": round(established_high, 4)},
        "distanceFromCentrePct": round(center_distance, 2),
        "averageDailyMovementPct": round(average_move, 3),
        "recentYearAverageDailyMovementPct": round(recent_average_move, 3),
        "dailyVolatilityPct": round(volatility, 3),
        "recentYearVolatilityPct": round(recent_volatility, 3),
        "historicalRangeContainmentPct": round(containment, 1),
        "isFlat": is_flat,
        "rangeRead": range_read,
        "read": read,
    }


def history_points(rows: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    dated: list[tuple[datetime, float]] = []
    for row in rows:
        raw_date = row.get("Date") or row.get("Datetime")
        close = _num(row.get("Close"), float("nan"))
        if not raw_date or close != close or close <= 0:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        except ValueError:
            continue
        dated.append((parsed, close))
    return sorted(dated, key=lambda item: item[0])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def pullback_analysis(
    profile: MarketExplorerProfile,
    historical: dict[str, Any],
    current_price: float,
    signal: dict[str, Any],
) -> dict[str, Any]:
    reference_high = _num(historical.get("oneYearHigh"), 0.0)
    if reference_high <= 0 or current_price <= 0:
        return {
            "referenceHigh": None,
            "bands": [],
            "entryZone": None,
            "guidance": "A current price and reference high are required before pullback bands can be calculated.",
        }
    bands: list[dict[str, Any]] = []
    for label, shallow, deep in (
        ("6-8% starter", 6, 8),
        ("10% better entry", 10, 10),
        ("12-15% strong dip", 12, 15),
    ):
        upper = reference_high * (1 - shallow / 100)
        lower = reference_high * (1 - deep / 100)
        bands.append({
            "label": label,
            "fromHighPct": {"low": deep, "high": shallow},
            "priceRange": {"low": round(lower, 2), "high": round(upper, 2)},
            "active": lower <= current_price <= upper,
            "passed": current_price <= upper,
        })
    risk = profile.risk_appetite.lower()
    strategy = profile.strategy.lower()
    if "conservative" in risk:
        preferred = bands[2]
    elif "short" in strategy or "growth" in risk:
        preferred = bands[0]
    else:
        preferred = bands[1]
    intent = profile.intent
    if intent == "Buy / Add":
        guidance = f"For {profile.risk_appetite} risk and {profile.strategy}, use {preferred['label']} as the preferred staging reference; do not treat a peak-only pullback as proof of value."
    elif intent == "Sell / Trim":
        guidance = "Pullback bands are secondary risk references for a trim decision; prioritize thesis deterioration, position size, and resistance/recovery behavior before adding or reducing."
    else:
        guidance = "Use the bands as watch levels and wait for price, thesis, and news confirmation before changing the position."
    signal_entry = signal.get("entry_zone") or signal.get("entryZone")
    return {
        "referenceHigh": round(reference_high, 2),
        "currentPrice": round(current_price, 2),
        "pullbackPct": round((1 - current_price / reference_high) * 100, 2),
        "bands": bands,
        "preferredBand": preferred["label"],
        "signalEntryZone": signal_entry,
        "guidance": guidance,
    }


def scenario_analysis(
    profile: MarketExplorerProfile,
    historical: dict[str, Any],
    current_price: float,
    pullback: dict[str, Any],
    thesis: dict[str, Any],
) -> dict[str, Any]:
    centre = _num(historical.get("tradingCentre"), current_price)
    established = historical.get("establishedRange") or {}
    low = _num(established.get("low"), current_price)
    high = _num(established.get("high"), current_price)
    avg_move = _num(historical.get("averageDailyMovementPct"), 0.0)
    horizon_days = 10 if "short" in profile.strategy.lower() else 30 if "long" in profile.strategy.lower() else 15
    risk_multiplier = 0.75 if profile.risk_appetite.lower().startswith("conservative") else 1.25 if profile.risk_appetite.lower().startswith("growth") else 1.0
    expected_move_pct = min(35.0, avg_move * math.sqrt(horizon_days) * risk_multiplier) if avg_move else 0.0
    downside = max(0.0, current_price * (1 - expected_move_pct / 100))
    upside = current_price * (1 + expected_move_pct / 100)
    worst = min(low, downside) if current_price else low
    best = max(high, upside) if current_price else high
    base_low = min(low, centre)
    base_high = max(high, centre)
    range_probability = _num(historical.get("historicalRangeContainmentPct"), 0.0)
    range_statement = (
        f"Historical closes contained within the established range about {range_probability:.0f}% of the time; "
        f"the base case is a return toward the {centre:.2f} trading centre unless a catalyst creates a breakout or breakdown."
        if range_probability
        else "Historical range evidence is limited; treat the scenarios as directional heuristics rather than a calibrated probability."
    )
    if thesis.get("classification") == "Range-bound / steady business; not a confirmed growth entry":
        range_statement += " The range-bound thesis lowers confidence that a pullback from the peak will become a durable growth breakout."
    return {
        "method": "Historical percentile range plus average absolute movement; not a guaranteed forecast.",
        "horizonDays": horizon_days,
        "riskMultiplier": risk_multiplier,
        "expectedMovePct": round(expected_move_pct, 2),
        "rangeProbabilityProxyPct": round(range_probability, 1),
        "baseCase": {"low": round(base_low, 2), "high": round(base_high, 2), "anchor": round(centre, 2)},
        "worstCase": {"price": round(worst, 2), "description": "Historical lower-range test or risk-adjusted move against the current price."},
        "bestCase": {"price": round(best, 2), "description": "Historical upper-range test or risk-adjusted move above the current price."},
        "rangeStatement": range_statement,
        "entryImplication": pullback.get("guidance"),
    }


def earnings_evidence(analysis: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    earnings = analysis.get("earnings") if isinstance(analysis.get("earnings"), dict) else {}
    metric_keys = (
        "totalRevenue",
        "revenueGrowth",
        "grossMargins",
        "profitMargins",
        "netIncomeToCommon",
        "trailingPE",
        "forwardPE",
        "targetMeanPrice",
        "targetMedianPrice",
        "numberOfAnalystOpinions",
        "payoutRatio",
    )
    metrics = {key: snapshot.get(key) for key in metric_keys if snapshot.get(key) is not None}
    days = analysis.get("daysUntilEarnings")
    event = {
        "nextEarningsDate": earnings.get("next_earnings_date") or snapshot.get("nextEarningsDate"),
        "lastEarningsDate": earnings.get("last_earnings_date") or snapshot.get("lastEarningsDate"),
        "daysUntilEarnings": days,
        "status": analysis.get("earningsStatus"),
    }
    catalysts: list[str] = []
    risks: list[str] = []
    growth = _num(snapshot.get("revenueGrowth"), float("nan"))
    if growth == growth:
        (catalysts if growth > 0 else risks).append(f"Revenue growth is {growth:+.1%} based on the current provider snapshot.")
    if days is not None and _num(days, 999) <= 21:
        risks.append("An earnings event is within the analysis horizon and can create gap risk.")
    if not metrics:
        risks.append("Detailed earnings metrics were not available in the current provider snapshot; verify the latest filing before acting.")
    return {"metrics": metrics, "event": event, "catalysts": catalysts, "risks": risks}


def news_evidence(rows: Any, sentiment: Any) -> dict[str, Any]:
    items = rows_from_value(rows)
    compact: list[dict[str, Any]] = []
    catalysts: list[str] = []
    risks: list[str] = []
    for item in items[:8]:
        title = str(item.get("Title") or item.get("title") or "").strip()
        summary = str(item.get("Summary") or item.get("summary") or "").strip()
        item_sentiment = _num(item.get("Sentiment") if item.get("Sentiment") is not None else item.get("sentiment"), 0.0)
        compact.append({
            "source": item.get("Source") or item.get("source"),
            "published": item.get("Published") or item.get("published"),
            "title": title,
            "summary": summary[:500],
            "sentiment": round(item_sentiment, 3),
            "url": item.get("Url") or item.get("url"),
        })
        if title:
            (catalysts if item_sentiment > 0.15 else risks if item_sentiment < -0.15 else []).append(title)
    return {
        "sentiment": round(_num(sentiment), 3),
        "itemCount": len(items),
        "items": compact,
        "catalysts": catalysts[:5],
        "risks": risks[:5],
        "dataQuality": "No symbol-specific headlines were returned." if not items else "Recent symbol-specific headlines were supplied for synthesis.",
    }


def instrument_pestel_context(snapshot: dict[str, Any], profile: MarketExplorerProfile, news: dict[str, Any]) -> list[dict[str, str]]:
    sector = snapshot.get("sector") or snapshot.get("industry") or "the selected sector"
    market = snapshot.get("exchangeName") or snapshot.get("exchange")
    return sector_pestel(sector, str(market) if market else None)


def thesis_classification(
    profile: MarketExplorerProfile,
    historical: dict[str, Any],
    snapshot: dict[str, Any],
    signal: dict[str, Any],
    pullback: dict[str, Any],
) -> dict[str, Any]:
    revenue_growth = _num(snapshot.get("revenueGrowth"), float("nan"))
    trend_return = _num(historical.get("ytdReturnPct"), 0.0)
    if historical.get("isFlat") and (revenue_growth != revenue_growth or revenue_growth < 0.05):
        classification = "Range-bound / steady business; not a confirmed growth entry"
    elif revenue_growth >= 0.05 and trend_return > 3:
        classification = "Growth-supported trend, subject to valuation and catalyst confirmation"
    else:
        classification = "Mixed thesis; price setup and business evidence require confirmation"
    tension = ""
    if pullback.get("pullbackPct", 0) >= 10 and historical.get("isFlat"):
        tension = "A meaningful decline from the high can coexist with a price near its multi-year trading centre; the pullback is not automatically a bargain."
    conclusion = f"{classification}. {tension}".strip()
    if profile.intent == "Buy / Add" and historical.get("isFlat"):
        conclusion += " Prefer a starter or staged entry only if the business thesis and current catalysts support the range break."
    elif profile.intent == "Hold / Watch":
        conclusion += " Monitor for a durable break from the established range before treating the setup as a new trend."
    return {
        "classification": classification,
        "signalAction": signal.get("action") or "WAIT",
        "pullbackVsCentreCheck": tension or "No direct pullback-versus-trading-centre conflict was detected from the available history.",
        "adjustedConclusion": conclusion,
    }


def local_pestel_context(profile: MarketExplorerProfile, candidates: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    sectors = desired_sectors(profile) or [str(item.get("sector")) for item in (candidates or []) if item.get("sector")]
    return profile_pestel(sectors)


def alternative_note(strategy: str, intent: str, candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No candidate evidence is available for comparison yet."
    leader = candidates[0]
    return (
        f"Under {strategy} / {intent}, start by re-checking {leader.get('symbol')} because it has the highest "
        f"evidence score for that combination ({leader.get('stance') or 'ranked first'})."
    )


def latest_history_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[-1] if rows else {}


def profile_to_payload(profile: MarketExplorerProfile) -> dict[str, Any]:
    return {
        "budget": profile.budget,
        "budgetCurrency": profile.budget_currency,
        "budgetAllocationMode": profile.budget_allocation_mode,
        "budgetAllocationValue": profile.budget_allocation_value,
        "currency": profile.currency[0] if len(profile.currency) == 1 else list(profile.currency),
        "riskAppetite": profile.risk_appetite,
        "experience": profile.experience,
        "purpose": profile.purpose,
        "strategy": profile.strategy,
        "intent": profile.intent,
        "intentOptions": list(profile.intent_options),
        "desiredSector": profile.desired_sector,
        "preferredInstruments": list(profile.preferred_instruments),
        "markets": list(profile.markets),
        "exploring": profile.exploring,
    }


def _loads_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return default
    return parsed
