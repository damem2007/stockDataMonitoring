from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.services.analysis.backtest import walk_forward_backtest
from app.services.analysis.ml_signal import FEATURE_COLUMNS
from app.services.analysis.risk import RiskProfile, get_risk_profile, volatility_target_position_size
from app.services.workspace.application import focused_analysis_payload


PREDICTION_HORIZONS = (1, 5, 10, 20, 60)
CONFIDENCE_BUCKETS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))
HIT_RATE_TARGET = 0.95
MIN_BUCKET_SAMPLES = 20
DEFAULT_ACCOUNT_SIZE = 10_000.0
PREDICTION_FEATURES = tuple(FEATURE_COLUMNS) + (
    "LogReturn",
    "RangePct",
    "BodyPct",
    "Volatility20",
    "TradingCentre",
    "TradingCentreDistance",
    "RangePosition",
    "PullbackFromHigh",
    "VolumeZ20",
)


def build_prediction(
    symbol: str,
    *,
    refresh: bool = False,
    horizons: list[int] | None = None,
    intent: str = "Hold / Watch",
    strategy: str = "Long-term (6-12 months)",
    risk_profile: str = "Balanced",
    account_size: float | None = None,
) -> dict[str, Any]:
    """Build an auditable prediction from the existing local analysis payload.

    This deliberately terminates the existing analysis contract: raw OHLCV,
    indicators, statements, news, macro evidence, and the existing backtest
    are inputs. The prediction layer does not mutate or replace them.
    """

    requested = _valid_horizons(horizons)
    analysis = focused_analysis_payload(
        symbol,
        period="5y",
        interval="1d",
        refresh=refresh,
        intent=intent,
        strategy=strategy,
        risk_profile_name=risk_profile,
        ml_model="gradient_boosting",
    )
    if not analysis.get("ok"):
        return {
            "schemaVersion": "prediction.v1",
            "ok": False,
            "symbol": str(symbol).upper(),
            "generatedAt": _now(),
            "error": analysis.get("status") or "No analysis evidence was available.",
            "sourceAnalysis": {"ok": False, "status": analysis.get("status")},
        }

    frame = _history_frame(analysis.get("history"))
    engineered = engineer_prediction_features(frame)
    profile = get_risk_profile(risk_profile)
    sizing_account = float(account_size) if account_size and float(account_size) > 0 else _default_account_size()
    forecasts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    explanations: dict[str, list[dict[str, Any]]] = {}
    for horizon in requested:
        forecast, evaluation, drivers = predict_horizon(engineered, horizon)
        forecast["tradePlan"] = build_trade_plan(forecast, engineered, profile, sizing_account)
        forecasts.append(forecast)
        evaluations.append(evaluation)
        explanations[str(horizon)] = drivers
    guidance = build_guidance(str(analysis.get("symbol") or symbol).upper(), forecasts, evaluations, profile, analysis.get("deepEvidence"), sizing_account, intent=intent, strategy=strategy)

    latest = engineered.iloc[-1] if not engineered.empty else pd.Series(dtype=float)
    source_analysis = {
        "generatedAt": _now(),
        "sourceStatus": analysis.get("sourceStatus"),
        "snapshotStatus": analysis.get("snapshotStatus"),
        "earningsStatus": analysis.get("earningsStatus"),
        "latestQuote": analysis.get("latestQuote"),
        "signal": analysis.get("signal"),
        "ml": analysis.get("ml"),
        "mlValidation": analysis.get("mlValidation"),
        "backtest": analysis.get("backtest"),
        "newsSentiment": analysis.get("newsSentiment"),
        "macroContext": analysis.get("macroContext"),
        "financialEvidence": analysis.get("financialEvidence"),
        "deepEvidence": analysis.get("deepEvidence"),
    }
    prediction_backtest, prediction_backtest_summary = walk_forward_backtest(frame, hold_days=20, cost_bps=10)
    walk_forward = _safe_records(prediction_backtest)
    return {
        "schemaVersion": "prediction.v1",
        "ok": True,
        "symbol": str(analysis.get("symbol") or symbol).upper(),
        "generatedAt": _now(),
        "provider": {
            "quantitative": "StockSignal prediction engine",
            "interpretation": "pending",
            "model": "logistic + gradient_boosting + rule evidence",
            "diagnosticId": None,
        },
        "input": {
            "sourceAnalysis": "focused_analysis_payload",
            "sourceAnalysisGeneratedAt": source_analysis["generatedAt"],
            "period": "5y",
            "interval": "1d",
            "dataAsOf": _latest_date(frame),
            "horizons": requested,
            "riskProfile": profile.name,
            "accountSize": sizing_account,
            "intent": intent,
            "strategy": strategy,
            "featureCount": len(PREDICTION_FEATURES),
            "leakageControl": "Forward labels are shifted into the future; time-series folds never shuffle; rolling features are shifted one row.",
        },
        "features": {
            "used": list(PREDICTION_FEATURES),
            "latest": _safe_record(latest, PREDICTION_FEATURES),
            "provenance": [
                "Yahoo Finance OHLCV and benchmark history",
                "Existing StockSignal indicators",
                "Multi-year trading-centre/reversion features",
                "Pullback and range-position features",
                "Existing news, earnings, macro, portfolio, and backtest evidence",
            ],
        },
        "forecast": forecasts,
        "guidance": guidance,
        "modelEvaluation": {
            "perHorizon": evaluations,
            "baseline": "majority-direction and rule evidence are reported beside model metrics",
            "hitRateTarget": HIT_RATE_TARGET,
            "hitRateNote": "Confidence buckets report the out-of-sample hit rate of past predictions made at each confidence level; a 95% hit rate is only claimed where validation actually reached it with enough samples.",
        },
        "monitoring": build_monitoring_snapshot(engineered),
        "explanation": {
            "driversByHorizon": explanations,
            "cautions": [
                "Predictions are conditional probabilities, not guaranteed prices; no model can promise a 95% hit rate in advance.",
                "Short horizons are sensitive to stale quotes, gaps, news, and market regime changes.",
                "A model that does not beat its majority baseline is not treated as evidence of an edge.",
            ],
        },
        "backtest": {
            "source": "existing StockSignal local backtest, re-evaluated with 10 bps per-leg cost",
            "walkForward": walk_forward,
            "summary": _safe_record_dict(prediction_backtest_summary),
            "costs": {"costBpsPerLeg": 10, "note": "Prediction output reports a cost-aware local test; spread, slippage, fees, and execution liquidity can differ live."},
        },
        "sourceAnalysis": source_analysis,
        "deepEvidence": analysis.get("deepEvidence"),
        "interpretation": None,
        "disclosure": "Quantitative prediction and AI-assisted interpretation are educational decision support, not financial advice.",
    }


def local_interpretation(prediction: dict[str, Any]) -> dict[str, Any]:
    forecasts = prediction.get("forecast") if isinstance(prediction.get("forecast"), list) else []
    evaluations = prediction.get("modelEvaluation", {}).get("perHorizon", []) if isinstance(prediction.get("modelEvaluation"), dict) else []
    first = next((item for item in forecasts if isinstance(item, dict) and item.get("label") != "unavailable"), {})
    evaluation = next((item for item in evaluations if isinstance(item, dict) and item.get("horizonDays") == first.get("horizonDays")), {})
    label = str(first.get("label") or "unavailable")
    probability = first.get("probabilityUp")
    summary = f"The local ensemble reads {label} over the {first.get('horizonDays', 'selected')} day horizon with {float(probability) * 100:.1f}% estimated upside probability." if isinstance(probability, (int, float)) else "The local ensemble could not produce a sufficiently supported forecast for the selected history."
    deep = prediction.get("deepEvidence") if isinstance(prediction.get("deepEvidence"), dict) else {}
    thesis = deep.get("thesis") if isinstance(deep.get("thesis"), dict) else {}
    scenarios = deep.get("scenarios") if isinstance(deep.get("scenarios"), dict) else {}
    execution = deep.get("execution") if isinstance(deep.get("execution"), dict) else {}
    if thesis.get("adjustedConclusion"):
        summary = f"{summary} {thesis['adjustedConclusion']}"
    guidance = prediction.get("guidance") if isinstance(prediction.get("guidance"), dict) else {}
    if guidance.get("summary"):
        summary = f"{summary} {guidance['summary']}"
    conditions = [str(evaluation.get("note") or "Recheck the forecast after a meaningful price, earnings, or regime change.")]
    conditions.extend(str(item) for item in guidance.get("checklist", [])[:3])
    conditions.extend(str(item) for item in execution.get("confirmation", []) if item)
    risks = [
        "Local rules do not understand new qualitative developments beyond the supplied news and statement evidence.",
        "Forecast confidence is reduced when the model does not beat the majority-direction baseline.",
    ]
    worst_case = scenarios.get("worstCase") if isinstance(scenarios.get("worstCase"), dict) else {}
    if worst_case.get("condition"):
        risks.append(f"Worst-case trigger: {worst_case['condition']}")
    return {
        "summary": summary,
        "directionalRead": "This is a deterministic evidence read from sequential validation, trading-centre/reversion features, pullback context, and the existing instrument analysis.",
        "risks": risks,
        "conditions": conditions + ["Review the current quote, timestamp, news, and portfolio exposure before acting."],
        "actionContext": (
            f"{guidance.get('accuracyNote')} Use this as a research starting point and monitoring checklist, not as an instruction to trade."
            if guidance.get("accuracyNote")
            else "Use this as a research starting point and monitoring checklist, not as an instruction to trade."
        ),
    }


def engineer_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    df = frame.copy().sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df.get("Close"), errors="coerce")
    high = pd.to_numeric(df.get("High", close), errors="coerce")
    low = pd.to_numeric(df.get("Low", close), errors="coerce")
    open_price = pd.to_numeric(df.get("Open", close), errors="coerce")
    volume = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0)
    df["LogReturn"] = np.log(close.where(close > 0) / close.shift(1).where(close.shift(1) > 0))
    df["RangePct"] = (high - low) / close.replace(0, np.nan)
    df["BodyPct"] = (close - open_price) / open_price.replace(0, np.nan)
    df["Volatility20"] = df["LogReturn"].rolling(20, min_periods=10).std().shift(1) * np.sqrt(252)
    # All rolling evidence is shifted so the bar being predicted cannot leak
    # into its own feature value.
    centre = close.rolling(756, min_periods=60).median().shift(1)
    high_lookback = high.rolling(252, min_periods=30).max().shift(1)
    low_lookback = low.rolling(252, min_periods=30).min().shift(1)
    df["TradingCentre"] = centre
    df["TradingCentreDistance"] = close / centre.replace(0, np.nan) - 1
    df["RangePosition"] = (close - low_lookback) / (high_lookback - low_lookback).replace(0, np.nan)
    df["PullbackFromHigh"] = close / high_lookback.replace(0, np.nan) - 1
    volume_mean = volume.rolling(20, min_periods=10).mean().shift(1)
    volume_std = volume.rolling(20, min_periods=10).std().shift(1)
    df["VolumeZ20"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    return df


def predict_horizon(frame: pd.DataFrame, horizon: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _unavailable_horizon(horizon, "scikit-learn is not installed.")

    if frame.empty or "Close" not in frame:
        return _unavailable_horizon(horizon, "No OHLCV history was available.")
    df = frame.copy()
    df["ForwardReturn"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["Target"] = (df["ForwardReturn"] > 0).astype(int)
    labeled = df.iloc[:-horizon].copy() if horizon < len(df) else df.iloc[0:0].copy()
    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    if len(labeled) < 100:
        return _unavailable_horizon(horizon, f"Only {len(labeled)} labeled rows are available for a {horizon}-day horizon.")
    medians = labeled[list(PREDICTION_FEATURES)].median(numeric_only=True)
    X = labeled[list(PREDICTION_FEATURES)].fillna(medians).fillna(0.0)
    y = labeled["Target"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return _unavailable_horizon(horizon, "The labeled history contains only one direction class.")
    n_splits = min(5, max(2, len(labeled) // 100))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=min(horizon, 5))
    oos_y: list[int] = []
    oos_pred: list[int] = []
    oos_proba: list[float] = []
    for train_idx, test_idx in splitter.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        gradient = GradientBoostingClassifier(random_state=42, n_estimators=120, max_depth=2)
        logistic.fit(X.iloc[train_idx], y[train_idx])
        gradient.fit(X.iloc[train_idx], y[train_idx])
        logistic_probability = logistic.predict_proba(X.iloc[test_idx])[:, 1]
        gradient_probability = gradient.predict_proba(X.iloc[test_idx])[:, 1]
        weights = _regime_weights(labeled.iloc[train_idx])
        probabilities = logistic_probability * weights["logistic"] + gradient_probability * weights["gradientBoosting"]
        oos_y.extend(y[test_idx].tolist())
        oos_proba.extend(probabilities.tolist())
        oos_pred.extend((probabilities >= 0.5).astype(int).tolist())
    if not oos_y:
        return _unavailable_horizon(horizon, "No valid walk-forward folds were available.")
    true = np.asarray(oos_y)
    pred = np.asarray(oos_pred)
    proba = np.asarray(oos_proba)
    baseline = float(max(true.mean(), 1 - true.mean()))
    buckets = confidence_buckets(true, pred, proba)
    metrics = {
        "samples": int(len(labeled)),
        "oosSamples": int(len(true)),
        "folds": int(n_splits),
        "accuracy": _metric(accuracy_score(true, pred)),
        "balancedAccuracy": _metric(balanced_accuracy_score(true, pred)),
        "precision": _metric(precision_score(true, pred, zero_division=0)),
        "recall": _metric(recall_score(true, pred, zero_division=0)),
        "f1": _metric(f1_score(true, pred, zero_division=0)),
        "mcc": _metric(matthews_corrcoef(true, pred)),
        "rocAuc": _metric(roc_auc_score(true, proba)) if len(np.unique(true)) > 1 else None,
        "brier": _metric(brier_score_loss(true, proba)),
        "baselineAccuracy": baseline,
        "beatsBaseline": bool(float(np.mean(pred == true)) > baseline + 0.03),
        "calibration": "calibrated" if abs(float(proba.mean()) - float(true.mean())) <= 0.05 else "needs calibration",
        "confidenceBuckets": buckets,
        "highConvictionThreshold": high_conviction_threshold(buckets),
        "bestReliablePrecision": max((row["precision"] for row in buckets if row["reliable"] and row["precision"] is not None), default=None),
        "note": "Out-of-sample metrics use sequential folds; no random shuffle was used.",
    }
    latest = df.iloc[[-1]][list(PREDICTION_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    gradient = GradientBoostingClassifier(random_state=42, n_estimators=120, max_depth=2).fit(X, y)
    weights = _regime_weights(labeled)
    raw_probability_up = float(logistic.predict_proba(latest)[:, 1][0] * weights["logistic"] + gradient.predict_proba(latest)[:, 1][0] * weights["gradientBoosting"])
    calibrated_probability = platt_calibrate(proba, true, raw_probability_up)
    probability_up = calibrated_probability if calibrated_probability is not None else raw_probability_up
    forward = labeled["ForwardReturn"].dropna()
    positive_mean = float(forward[forward > 0].mean()) if (forward > 0).any() else 0.0
    negative_mean = float(forward[forward <= 0].mean()) if (forward <= 0).any() else 0.0
    predicted_return = probability_up * positive_mean + (1 - probability_up) * negative_mean
    dispersion = float(forward.std()) if len(forward) > 1 else 0.0
    current = float(df["Close"].iloc[-1])
    label = "up" if probability_up >= 0.55 else "down" if probability_up <= 0.45 else "neutral"
    confidence = max(probability_up, 1 - probability_up)
    hit_rate = bucket_for_confidence(buckets, confidence)
    gate = conviction_gate(label, metrics, hit_rate, calibrated_probability is not None)
    forecast = {
        "horizonDays": horizon,
        "target": "direction and forward return",
        "label": label,
        "probabilityUp": round(probability_up, 4),
        "probabilityDown": round(1 - probability_up, 4),
        "rawProbabilityUp": round(raw_probability_up, 4),
        "calibratedProbabilityUp": round(calibrated_probability, 4) if calibrated_probability is not None else None,
        "predictedReturn": _round_or_none(predicted_return),
        "predictedRange": {"low": _round_or_none(current * (1 + predicted_return - dispersion)), "high": _round_or_none(current * (1 + predicted_return + dispersion))},
        "conditionalReturns": {"up": _round_or_none(positive_mean), "down": _round_or_none(negative_mean), "dispersion": _round_or_none(dispersion)},
        "confidence": round(abs(probability_up - 0.5) * 2, 4),
        "modelAgreement": "ensemble agreement is directional" if abs(probability_up - 0.5) >= 0.08 else "models are close to neutral",
        "regimeAwareModelWeights": weights,
        "hitRateAtCurrentConfidence": hit_rate,
        "edgeScore": edge_score(probability_up, metrics),
        "gate": gate,
        "tradeable": gate["tradeable"],
    }
    coefficients = logistic[-1].coef_[0] if hasattr(logistic[-1], "coef_") else np.zeros(len(PREDICTION_FEATURES))
    importances = getattr(gradient, "feature_importances_", np.zeros(len(PREDICTION_FEATURES)))
    drivers = []
    for index, name in enumerate(PREDICTION_FEATURES):
        value = float(latest.iloc[0][name]) if pd.notna(latest.iloc[0][name]) else 0.0
        impact = float(coefficients[index] * value) + float(importances[index]) * (1 if value >= float(medians.get(name, 0) or 0) else -1)
        drivers.append({"feature": name, "value": _round_or_none(value), "impact": round(impact, 6), "direction": "upward" if impact > 0 else "downward" if impact < 0 else "neutral"})
    drivers.sort(key=lambda item: abs(float(item["impact"])), reverse=True)
    return forecast, metrics, drivers[:6]


def _history_frame(history: Any) -> pd.DataFrame:
    if not isinstance(history, list):
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame.get("Date"), errors="coerce", utc=True)
    for field in ("Open", "High", "Low", "Close", "Volume"):
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def _valid_horizons(horizons: list[int] | None) -> list[int]:
    if not horizons:
        return list(PREDICTION_HORIZONS)
    return [int(item) for item in dict.fromkeys(horizons) if int(item) in PREDICTION_HORIZONS] or list(PREDICTION_HORIZONS)


def _unavailable_horizon(horizon: int, note: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        {"horizonDays": horizon, "target": "direction and forward return", "label": "unavailable", "probabilityUp": None, "probabilityDown": None, "calibratedProbabilityUp": None, "predictedReturn": None, "predictedRange": {"low": None, "high": None}, "conditionalReturns": None, "confidence": None, "modelAgreement": note, "hitRateAtCurrentConfidence": None, "edgeScore": 0.0, "gate": {"tradeable": False, "status": "unavailable", "reasons": [note]}, "tradeable": False},
        {"horizonDays": horizon, "samples": 0, "folds": 0, "confidenceBuckets": [], "highConvictionThreshold": None, "note": note},
        [],
    )


def _safe_record(row: pd.Series, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: _round_or_none(row.get(field)) for field in fields if field in row.index}


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{str(key): _round_or_none(value) if isinstance(value, (int, float, np.number)) else str(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _safe_record_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _round_or_none(value) if isinstance(value, (int, float, np.number)) else value for key, value in values.items()}


def _regime_weights(frame: pd.DataFrame) -> dict[str, Any]:
    volatility = pd.to_numeric(frame.get("Volatility20"), errors="coerce").dropna() if "Volatility20" in frame else pd.Series(dtype=float)
    if not volatility.empty and float(volatility.iloc[-1]) > float(volatility.median()) * 1.25:
        return {"logistic": 0.35, "gradientBoosting": 0.65, "regime": "high-volatility"}
    return {"logistic": 0.65, "gradientBoosting": 0.35, "regime": "normal-volatility"}


def build_monitoring_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"driftFlag": True, "note": "No feature rows were available for monitoring."}
    latest = frame.iloc[-1]
    train = frame.iloc[:-1]
    shifts = []
    for feature in PREDICTION_FEATURES:
        values = pd.to_numeric(train.get(feature), errors="coerce").dropna() if feature in train else pd.Series(dtype=float)
        current = pd.to_numeric(pd.Series([latest.get(feature)]), errors="coerce").iloc[0]
        if values.empty or pd.isna(current) or float(values.std() or 0) == 0:
            continue
        shifts.append({"feature": feature, "zScore": round(float((current - values.mean()) / values.std()), 3)})
    shifts.sort(key=lambda item: abs(item["zScore"]), reverse=True)
    flag = any(abs(item["zScore"]) >= 3 for item in shifts)
    return {"driftFlag": flag, "topFeatureShifts": shifts[:5], "note": "Monitoring compares the latest feature snapshot with the historical training distribution; it is a review signal, not an automatic retraining trigger."}


def confidence_buckets(true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> list[dict[str, Any]]:
    """Out-of-sample hit rate of past predictions grouped by their confidence."""

    confidence = np.maximum(proba, 1 - proba)
    rows: list[dict[str, Any]] = []
    for low, high in CONFIDENCE_BUCKETS:
        mask = (confidence >= low) & ((confidence < high) if high < 1.0 else (confidence <= high))
        samples = int(mask.sum())
        precision = float(np.mean(pred[mask] == true[mask])) if samples else None
        rows.append({
            "confidenceFrom": low,
            "confidenceTo": high,
            "samples": samples,
            "precision": round(precision, 4) if precision is not None else None,
            "reliable": samples >= MIN_BUCKET_SAMPLES,
        })
    return rows


def high_conviction_threshold(buckets: list[dict[str, Any]], target: float = HIT_RATE_TARGET) -> float | None:
    """Lowest confidence level whose validated hit rate reached the target."""

    for row in buckets:
        if row["reliable"] and row["precision"] is not None and row["precision"] >= target:
            return float(row["confidenceFrom"])
    return None


def bucket_for_confidence(buckets: list[dict[str, Any]], confidence: float) -> dict[str, Any] | None:
    for row in buckets:
        upper = row["confidenceTo"]
        if row["confidenceFrom"] <= confidence and (confidence < upper or (upper >= 1.0 and confidence <= upper)):
            return row
    return buckets[-1] if buckets and confidence >= buckets[-1]["confidenceFrom"] else None


def platt_calibrate(proba: np.ndarray, true: np.ndarray, probability_up: float) -> float | None:
    """Map the live ensemble probability through a Platt fit of the OOS predictions."""

    if len(true) < 120 or len(np.unique(true)) < 2 or float(np.std(proba)) < 1e-6:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=500).fit(proba.reshape(-1, 1), true)
        value = float(model.predict_proba(np.array([[probability_up]]))[:, 1][0])
        return value if np.isfinite(value) else None
    except Exception:
        return None


def edge_score(probability_up: float, metrics: dict[str, Any]) -> float:
    """0-100 score combining directional margin, ranking quality, and baseline beat."""

    margin = min(1.0, abs(probability_up - 0.5) / 0.25)
    auc = metrics.get("rocAuc")
    auc_norm = max(0.0, min(1.0, ((float(auc) if auc is not None else 0.5) - 0.5) / 0.2))
    mcc = metrics.get("mcc")
    mcc_norm = max(0.0, min(1.0, (float(mcc) if mcc is not None else 0.0) / 0.3))
    baseline_bonus = 1.0 if metrics.get("beatsBaseline") else 0.0
    return round(100 * (0.4 * margin + 0.3 * auc_norm + 0.2 * mcc_norm + 0.1 * baseline_bonus), 1)


def conviction_gate(label: str, metrics: dict[str, Any], hit_rate: dict[str, Any] | None, calibrated: bool) -> dict[str, Any]:
    """Decide whether a forecast is strong enough to be treated as a tradeable setup."""

    reasons: list[str] = []
    checks: dict[str, bool] = {}
    checks["directional"] = label in {"up", "down"}
    if not checks["directional"]:
        reasons.append("The ensemble is close to neutral, so there is no directional edge to act on.")
    checks["beatsBaseline"] = bool(metrics.get("beatsBaseline"))
    if not checks["beatsBaseline"]:
        reasons.append(f"Walk-forward accuracy {_pct(metrics.get('accuracy'))} does not beat the majority baseline {_pct(metrics.get('baselineAccuracy'))} by three points.")
    auc = metrics.get("rocAuc")
    checks["rankingQuality"] = auc is not None and float(auc) >= 0.55
    if not checks["rankingQuality"]:
        reasons.append(f"ROC-AUC {auc if auc is not None else 'n/a'} is below the 0.55 minimum for a usable ranking signal.")
    checks["calibration"] = calibrated or metrics.get("calibration") == "calibrated"
    if not checks["calibration"]:
        reasons.append("Probabilities need calibration and no calibration fit was available.")
    checks["hitRate"] = True
    if hit_rate and hit_rate.get("reliable") and hit_rate.get("precision") is not None and float(hit_rate["precision"]) < 0.55:
        checks["hitRate"] = False
        reasons.append(f"Historical hit rate at this confidence is only {_pct(hit_rate['precision'])} over {hit_rate['samples']} out-of-sample cases.")
    tradeable = all(checks.values())
    return {
        "tradeable": tradeable,
        "status": "tradeable setup" if tradeable else "research only",
        "reasons": reasons or ["All conviction checks passed: directional edge, baseline beat, ranking quality, calibration, and hit rate."],
        "checks": checks,
    }


def build_trade_plan(forecast: dict[str, Any], frame: pd.DataFrame, profile: RiskProfile, account_size: float) -> dict[str, Any] | None:
    """Turn a directional forecast into entry, stop, target, reward:risk, and sizing."""

    if frame.empty or forecast.get("label") in {None, "unavailable"}:
        return None
    price = float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])
    atr_series = pd.to_numeric(frame.get("ATR14"), errors="coerce").dropna() if "ATR14" in frame else pd.Series(dtype=float)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty and float(atr_series.iloc[-1]) > 0 else price * 0.02
    conditional = forecast.get("conditionalReturns") or {}
    up_mean = float(conditional.get("up") or 0.0)
    down_mean = float(conditional.get("down") or 0.0)
    probability_up = float(forecast.get("probabilityUp") or 0.5)
    label = str(forecast.get("label"))
    stop_distance = max(atr * profile.stop_atr_multiple, price * 0.005)
    horizon = int(forecast.get("horizonDays") or 0)
    if label == "up":
        entry = price
        stop = max(entry - stop_distance, 0.01)
        target = entry * (1 + max(up_mean, 0.005))
        reward = target - entry
        risk = entry - stop
        expected_value_pct = probability_up * up_mean + (1 - probability_up) * down_mean
        sizing = volatility_target_position_size(price, atr, account_size, profile.risk_pct_per_trade, profile.stop_atr_multiple)
        stance = "Long setup"
        instructions = [
            f"Entry reference {entry:.2f}; prefer a fill at or below this level.",
            f"Protective stop {stop:.2f} ({profile.stop_atr_multiple:.1f}x ATR below entry).",
            f"Profit target {target:.2f} (historical average up-move over {horizon} days); take partial profit there.",
        ]
    elif label == "down":
        entry = price
        stop = entry + stop_distance
        target = entry * (1 + min(down_mean, -0.005))
        reward = entry - target
        risk = stop - entry
        expected_value_pct = probability_up * up_mean + (1 - probability_up) * down_mean
        sizing = volatility_target_position_size(price, atr, account_size, profile.risk_pct_per_trade, profile.stop_atr_multiple)
        stance = "Defensive: trim, hedge, or wait"
        instructions = [
            f"Exit or hedge reference {entry:.2f}; avoid adding while the read is negative.",
            f"Re-entry trigger {stop:.2f} (a reclaim {profile.stop_atr_multiple:.1f}x ATR above the current price invalidates the bearish read).",
            f"Downside objective {target:.2f} (historical average down-move over {horizon} days) for hedges or re-entry planning.",
        ]
    else:
        return {
            "stance": "Stand aside",
            "entry": round(price, 4),
            "stop": None,
            "target": None,
            "rewardToRisk": None,
            "expectedValuePct": _round_or_none(probability_up * up_mean + (1 - probability_up) * down_mean),
            "atr": round(atr, 4),
            "positionSize": {"shares": 0, "dollarAllocation": 0.0, "riskDollars": 0.0, "note": "No directional edge; no position is sized."},
            "instructions": ["Wait for a directional read with a passing conviction gate before sizing a trade.", f"Watch levels: {price - stop_distance:.2f} support test and {price + stop_distance:.2f} breakout confirmation."],
        }
    reward_to_risk = round(reward / risk, 2) if risk > 0 else None
    return {
        "stance": stance,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "rewardToRisk": reward_to_risk,
        "expectedValuePct": _round_or_none(expected_value_pct),
        "expectedValuePerShare": _round_or_none(expected_value_pct * price),
        "atr": round(atr, 4),
        "positionSize": {
            "shares": int(sizing.shares),
            "dollarAllocation": round(float(sizing.dollar_allocation), 2),
            "riskDollars": round(float(sizing.risk_dollars), 2),
            "stopPrice": round(float(sizing.stop_price), 4),
            "note": sizing.note,
        },
        "instructions": instructions,
    }


def build_guidance(
    symbol: str,
    forecasts: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    profile: RiskProfile,
    deep_evidence: Any,
    account_size: float,
    *,
    intent: str = "Hold / Watch",
    strategy: str = "Buy-dip",
) -> dict[str, Any]:
    """Summarize where the edge is, whether it clears the conviction gate, and how to manage it."""

    usable = [item for item in forecasts if item.get("label") not in {None, "unavailable"}]
    tradeable = [item for item in usable if item.get("tradeable")]
    pool = tradeable or usable
    best = max(pool, key=lambda item: float(item.get("edgeScore") or 0), default=None)
    checklist = [
        "Confirm the live quote is still close to the plan's entry reference before acting.",
        f"Risk no more than {profile.risk_pct_per_trade:.1%} of the account on the stop ({profile.name} profile).",
        "Take partial profit at the target and move the stop to breakeven after a one-R move.",
        "Exit if the deep-evidence invalidation triggers or the stop is hit; do not average down against the plan.",
        "Re-run the prediction after earnings, a regime change, or a move larger than one ATR.",
    ]
    profit_rules = [
        "Only treat a horizon as actionable when its conviction gate passes and the reward-to-risk is at least 1.5.",
        "Prefer the horizon with the highest edge score; shorter horizons are more sensitive to stale quotes and gaps.",
        "Size by volatility, not by conviction: the stop distance sets the share count.",
    ]
    if best is None:
        return {
            "status": "insufficient",
            "symbol": symbol,
            "summary": "The local engine could not produce a supported forecast for any horizon, so no profit plan is available.",
            "accuracyNote": "Not enough validated history to report hit rates.",
            "checklist": checklist,
            "profitRules": profit_rules,
            "hitRateTarget": HIT_RATE_TARGET,
            "riskProfile": profile.name,
            "accountSize": account_size,
            "disclaimer": "Educational decision support, not financial advice.",
        }
    evaluation = next((row for row in evaluations if row.get("horizonDays") == best.get("horizonDays")), {})
    hit_rate = best.get("hitRateAtCurrentConfidence") or {}
    threshold = evaluation.get("highConvictionThreshold")
    plan = best.get("tradePlan") or {}
    accuracy_note = (
        f"Walk-forward accuracy {_pct(evaluation.get('accuracy'))} versus a {_pct(evaluation.get('baselineAccuracy'))} majority baseline; "
        f"hit rate at the current confidence {_pct(hit_rate.get('precision'))} over {hit_rate.get('samples', 0)} out-of-sample cases."
    )
    if threshold is not None:
        accuracy_note += f" A {HIT_RATE_TARGET:.0%} historical hit rate was reached at or above {threshold:.0%} confidence; only forecasts at that level should be treated as high conviction."
    else:
        accuracy_note += f" No confidence level reached a {HIT_RATE_TARGET:.0%} historical hit rate in validation, so every forecast should be treated as probabilistic rather than near-certain."
    thesis = deep_evidence.get("thesis") if isinstance(deep_evidence, dict) and isinstance(deep_evidence.get("thesis"), dict) else {}
    summary = (
        f"The {best.get('horizonDays')}-day horizon carries the strongest edge ({best.get('edgeScore')}/100, "
        f"{_pct(best.get('probabilityUp'))} up probability): {plan.get('stance', 'no stance')}. "
        f"Conviction gate: {(best.get('gate') or {}).get('status', 'unknown')}."
    )
    if plan.get("rewardToRisk") is not None:
        summary += f" Reward-to-risk {plan['rewardToRisk']}:1 with expected value {_pct(plan.get('expectedValuePct'))} per trade."
    if thesis.get("adjustedConclusion"):
        summary += f" Thesis: {thesis['adjustedConclusion']}"
    return {
        "status": "ready" if tradeable else "research-only",
        "symbol": symbol,
        "bestHorizonDays": best.get("horizonDays"),
        "stance": plan.get("stance"),
        "edgeScore": best.get("edgeScore"),
        "tradeableHorizons": [item.get("horizonDays") for item in tradeable],
        "summary": summary,
        "accuracyNote": accuracy_note,
        "checklist": checklist,
        "profitRules": profit_rules,
        "hitRateTarget": HIT_RATE_TARGET,
        "highConvictionThreshold": threshold,
        "riskProfile": profile.name,
        "accountSize": account_size,
        "intent": intent,
        "strategy": strategy,
        "disclaimer": "Educational decision support, not financial advice.",
    }


def _default_account_size() -> float:
    import os

    try:
        return max(100.0, float(os.getenv("PREDICTION_DEFAULT_ACCOUNT_SIZE", DEFAULT_ACCOUNT_SIZE)))
    except (TypeError, ValueError):
        return DEFAULT_ACCOUNT_SIZE


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.0%}" if np.isfinite(number) else "n/a"


def _metric(value: Any) -> float | None:
    try:
        value = float(value)
        return round(value, 6) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _round_or_none(value: Any) -> float | None:
    return round(float(value), 6) if value is not None and pd.notna(value) and np.isfinite(float(value)) else None


def _latest_date(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    value = frame["Date"].iloc[-1]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
