from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.config import AI_DISCLOSURE, OPENAI_MAX_OUTPUT_TOKENS, OPENAI_REASONING_EFFORT, OPENAI_RESPONSES_ENDPOINT
from app.database import database_available, ensure_database, session_scope
from app.models.orm import PredictionProviderDiagnostic
from app.services.ai.models import resolve_model
from app.services.ai.registry import ProviderRequest, generate_structured_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PREDICTION_INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "directionalRead", "risks", "conditions", "actionContext"],
    "properties": {
        "summary": {"type": "string"},
        "directionalRead": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "actionContext": {"type": "string"},
    },
}


def prediction_model() -> str:
    return resolve_model("prediction")


def interpret_prediction(prediction: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Ask the active provider to interpret the local forecast; never replace it."""

    symbol = str(prediction.get("symbol") or "").upper()
    model = prediction_model()
    started = time.monotonic()
    result = generate_structured_json(
        ProviderRequest(
            task="prediction",
            prompt=_compact_prompt(prediction),
            schema=PREDICTION_INTERPRETATION_SCHEMA,
            schema_name="prediction_interpretation",
            strict=True,
            system=f"{AI_DISCLOSURE} Return only the requested JSON object.",
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            reasoning_effort=OPENAI_REASONING_EFFORT,
            model=model,
            log_prefix="prediction_provider",
            logger=logger,
            context={"symbol": symbol},
        )
    )
    if result.ok and _usable(result.payload):
        return {
            "status": "ready",
            "provider": result.provider,
            "model": result.model,
            "interpretation": result.payload,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "notFinancialAdvice": True,
            "elapsedMs": result.elapsed_ms,
        }
    if result.ok:
        category = "unusable_response"
        reason = "OpenAI responded, but the body did not contain all required prediction interpretation fields."
        user_reason = "OpenAI returned an unusable analysis response."
        status = "fallback"
    else:
        category = result.failure_category or "unknown"
        reason = result.reason or "The AI provider did not return an interpretation."
        user_reason = _user_reason(category, reason)
        status = "unavailable" if category == "configuration" else "fallback"
    diagnostic = _store_diagnostic(
        user_id,
        symbol,
        result.model,
        category,
        reason,
        result.request_bytes or 0,
        started,
        result.response_excerpt or "",
        status=result.response_status,
        elapsed_ms=result.elapsed_ms,
    )
    return {"status": status, "provider": result.provider, "model": result.model, "diagnosticId": diagnostic, "reason": user_reason}


def _user_reason(category: str, reason: str) -> str:
    if category == "timeout":
        return "OpenAI timed out before returning a complete interpretation."
    if category in {"request_error", "parse_error"}:
        return "OpenAI response could not be completed or parsed."
    return reason


def list_diagnostics(user_id: str, symbol: str | None = None) -> list[dict[str, Any]]:
    if not database_available() or not ensure_database():
        return []
    with session_scope() as session:
        query = session.query(PredictionProviderDiagnostic).filter(PredictionProviderDiagnostic.user_id == UUID(str(user_id)))
        if symbol:
            query = query.filter(PredictionProviderDiagnostic.symbol == symbol.upper())
        return [_diagnostic_payload(row) for row in query.order_by(PredictionProviderDiagnostic.created_at.desc()).limit(50).all()]


def _compact_prompt(prediction: dict[str, Any]) -> dict[str, Any]:
    source = prediction.get("sourceAnalysis") if isinstance(prediction.get("sourceAnalysis"), dict) else {}
    return {
        "task": "Interpret a quantitative stock prediction without inventing facts or replacing calculated values.",
        "symbol": prediction.get("symbol"),
        "input": prediction.get("input"),
        "features": prediction.get("features"),
        "forecast": prediction.get("forecast"),
        "modelEvaluation": prediction.get("modelEvaluation"),
        "guidance": prediction.get("guidance"),
        "tradePlans": [
            {key: item.get(key) for key in ("horizonDays", "label", "probabilityUp", "calibratedProbabilityUp", "edgeScore", "tradeable", "tradePlan", "hitRateAtCurrentConfidence")}
            for item in (prediction.get("forecast") or [])
            if isinstance(item, dict)
        ],
        "sourceEvidence": {key: source.get(key) for key in ("signal", "ml", "mlValidation", "backtest", "newsSentiment", "macroContext", "financialEvidence", "deepEvidence")},
        "required": [
            "Summarize historical and current evidence.",
            "Explain whether multi-year reversion and pullback zones support or contradict the direction.",
            "Interpret total-return, thesis, scenario, catalyst, contradiction, and execution evidence when available.",
            "Explain the conviction gate, hit rate at the current confidence, and the profit plan (entry, stop, target, reward-to-risk) for the best horizon.",
            "Call out model weakness, uncertainty, events, and conditions that would invalidate the read.",
            "Use conditional action context, not personalized financial advice.",
        ],
    }


def _usable(value: dict[str, Any] | None) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(field), (str, list)) and bool(value.get(field)) for field in ("summary", "directionalRead", "risks", "conditions", "actionContext"))


def _store_diagnostic(
    user_id: str,
    symbol: str,
    model: str,
    category: str,
    reason: str,
    request_bytes: int,
    started: float,
    excerpt: str,
    status: int | None = None,
    elapsed_ms: int | None = None,
) -> str | None:
    elapsed = elapsed_ms if elapsed_ms is not None else round((time.monotonic() - started) * 1000)
    logger.warning("prediction_provider provider=openai outcome=fallback symbol=%s model=%s reason=%s fallback=local-rules elapsed_ms=%s", symbol, model, category, elapsed)
    if not database_available() or not ensure_database():
        return None
    try:
        with session_scope() as session:
            row = PredictionProviderDiagnostic(
                user_id=UUID(str(user_id)),
                symbol=symbol,
                provider="openai",
                model=model,
                endpoint=OPENAI_RESPONSES_ENDPOINT,
                outcome="fallback",
                failure_category=category,
                failure_reason=reason,
                response_status=status,
                request_bytes=request_bytes,
                elapsed_ms=elapsed,
                response_excerpt=_redact(excerpt)[:8000],
            )
            session.add(row)
            session.flush()
            return str(row.id)
    except Exception:  # pragma: no cover - diagnostics must never block a prediction
        logger.exception("prediction_provider outcome=diagnostic_store_failed symbol=%s", symbol)
        return None


def _diagnostic_payload(row: PredictionProviderDiagnostic) -> dict[str, Any]:
    return {"id": str(row.id), "symbol": row.symbol, "provider": row.provider, "model": row.model, "endpoint": row.endpoint, "outcome": row.outcome, "failureCategory": row.failure_category, "failureReason": row.failure_reason, "responseStatus": row.response_status, "requestBytes": row.request_bytes, "elapsedMs": row.elapsed_ms, "responseExcerpt": row.response_excerpt, "createdAt": row.created_at.isoformat() if row.created_at else None}


def _redact(value: str) -> str:
    return re.sub(r"Bearer\s+\S+", "Bearer [redacted]", str(value or ""))[:8000]
