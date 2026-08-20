from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from backend.app.services.analysis.indicators import detect_gap_zones
from backend.app.services.analysis.risk import RiskProfile, get_risk_profile
 
 
@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    current_price: float
    intent: str
    horizon: str
    action: str
    confidence: float
    setup: str
    action_justification: str
    entry_zone: str
    exit_zone: str
    risk_note: str
    reasons: tuple[str, ...]
    news_drivers: tuple[str, ...]
    model_probability_up: float | None = None
    model_note: str = ""
    regime_note: str = ""
    analyst_note: str = ""
    earnings_note: str = ""
    risk_profile_name: str = "Balanced"
    forecast_note: str = ""
 
 
@dataclass(frozen=True)
class CompositeOutlook:
    symbol: str
    current_price: float
    horizon: str
    risk_profile_name: str
    direction: str
    confidence_label: str
    rule_score: float
    composite_score: float
    summary: str
    drivers: tuple[str, ...]
    cautions: tuple[str, ...]
    model_probability_up: float | None = None
    analyst_upside: float | None = None
    news_sentiment: float = 0.0
    days_until_earnings: int | None = None


def build_composite_outlook(
    symbol: str,
    frame: pd.DataFrame,
    rule_score: float,
    action: str,
    horizon: str,
    news_sentiment: float = 0.0,
    market_regime: dict[str, object] | None = None,
    ml_signal: object | None = None,
    analyst_snapshot: dict[str, object] | None = None,
    days_until_earnings: int | None = None,
    risk_profile: RiskProfile | str | None = None,
) -> CompositeOutlook:
    """Synthesize technical, model, analyst, news, and event-risk evidence.

    This intentionally reports a scored outlook rather than a guaranteed
    probability. The rule score is still the primary signal; the ML model can
    nudge the composite only when its walk-forward validation beats baseline.
    """

    profile = risk_profile if isinstance(risk_profile, RiskProfile) else get_risk_profile(risk_profile or "Balanced")
    horizon_profile = get_horizon_profile(horizon)
    snapshot = analyst_snapshot or {}
    if frame.empty:
        return CompositeOutlook(
            symbol=symbol,
            current_price=0.0,
            horizon=horizon,
            risk_profile_name=profile.name,
            direction="No data",
            confidence_label="Unavailable",
            rule_score=0.0,
            composite_score=0.0,
            summary="No outlook available because historical price data was not loaded.",
            drivers=(),
            cautions=("Fetch historical price data before making a decision.",),
            news_sentiment=news_sentiment,
            days_until_earnings=days_until_earnings,
        )

    latest = frame.iloc[-1]
    price = float(latest.get("Close", 0.0))
    ema20 = float(latest.get("EMA20", np.nan))
    ema50 = float(latest.get("EMA50", np.nan))
    rsi = float(latest.get("RSI14", np.nan))
    trend = float(latest.get("TrendScore", 0.0))
    model_probability_up = getattr(ml_signal, "probability_up", None) if ml_signal is not None else None
    model_beats_baseline = bool(getattr(ml_signal, "beats_baseline", False)) if ml_signal is not None else False
    target = _num(snapshot.get("targetMeanPrice"))
    analyst_upside = (target / price - 1) if target and price > 0 else None
    resolved_days_until_earnings = resolve_days_until_earnings(snapshot, days_until_earnings)

    composite_score = float(rule_score)
    drivers: list[str] = []
    cautions: list[str] = []

    if rsi == rsi:
        if rsi < 35:
            drivers.append(f"RSI {rsi:.1f} is oversold, which can support a reversal setup after confirmation.")
        elif 40 <= rsi < 60:
            drivers.append(f"RSI {rsi:.1f} is constructive without being overheated.")
        elif rsi >= 70:
            cautions.append(f"RSI {rsi:.1f} is overbought, increasing chase risk.")
    if ema20 == ema20 and price > 0:
        if price >= ema20:
            drivers.append("Price is above the 20-day EMA, supporting short-term trend alignment.")
        else:
            cautions.append("Price is below the 20-day EMA, so timing confirmation is still incomplete.")
    if ema50 == ema50 and price >= ema50 and trend > 0:
        drivers.append("Price and 20/50-day trend structure are aligned upward.")
    elif ema50 == ema50 and price < ema50:
        cautions.append("Price is below the 50-day EMA, weakening medium-term confirmation.")

    if model_probability_up is not None and model_probability_up == model_probability_up:
        probability_up = float(model_probability_up)
        if model_beats_baseline:
            model_delta = (probability_up - 0.5) * 18
            composite_score += model_delta
            action_leans_up = action.startswith(("BUY", "ACCUMULATE"))
            if probability_up >= 0.55 and action_leans_up:
                drivers.append(f"Validated ML model agrees with the setup at P(up) {probability_up:.0%}.")
            elif probability_up >= 0.55 and not action_leans_up:
                cautions.append(
                    f"Validated ML model leans bullish at P(up) {probability_up:.0%}, "
                    f"but the rule action is {action.lower()}."
                )
            elif probability_up <= 0.45 and action_leans_up:
                cautions.append(f"Validated ML model disagrees with the buy setup at P(up) {probability_up:.0%}.")
            elif probability_up <= 0.45 and not action_leans_up:
                drivers.append(f"Validated ML model supports caution at P(up) {probability_up:.0%}.")
            else:
                cautions.append(f"Validated ML model is close to neutral at P(up) {probability_up:.0%}.")
        else:
            cautions.append(f"ML model shows P(up) {probability_up:.0%}, but without validated edge over baseline.")

    if analyst_upside is not None:
        if analyst_upside >= 0.08:
            drivers.append(f"Analyst mean target implies {analyst_upside:+.1%} upside.")
        elif analyst_upside <= -0.05:
            cautions.append(f"Analyst mean target implies {analyst_upside:+.1%} downside.")

    if news_sentiment > 0.15:
        drivers.append(f"Recent scanned news sentiment is positive ({news_sentiment:+.2f}).")
    elif news_sentiment < -0.15:
        cautions.append(f"Recent scanned news sentiment is negative ({news_sentiment:+.2f}).")

    if market_regime:
        regime_note = str(market_regime.get("note", ""))
        if market_regime.get("dampen_buy_confidence"):
            cautions.append(f"Market regime dampener: {regime_note}")
        elif regime_note:
            drivers.append(f"Market regime context: {regime_note}")

    if resolved_days_until_earnings is not None:
        if 0 <= resolved_days_until_earnings <= 14:
            cautions.append("Upcoming earnings can override technical and model signals.")
        elif resolved_days_until_earnings < 0:
            cautions.append("Last known earnings date is in the past; verify fresh results are reflected in price.")

    composite_score = float(np.clip(composite_score, 1, 99))
    if action.startswith(("BUY", "ACCUMULATE")):
        direction = "Constructive"
    elif "SELL" in action or "REDUCE" in action or "TRIM" in action:
        direction = "Risk-reduction"
    elif "AVOID" in action:
        direction = "Unfavorable"
    elif composite_score >= 65:
        direction = "Constructive watch"
    else:
        direction = "Confirmation-first"

    if composite_score >= 75:
        confidence_label = "High"
    elif composite_score >= 55:
        confidence_label = "Moderate"
    elif composite_score >= 40:
        confidence_label = "Low"
    else:
        confidence_label = "Weak"

    primary_driver = drivers[0] if drivers else "Evidence is mixed."
    primary_caution = f" Caution: {cautions[0]}" if cautions else ""
    summary = (
        f"{direction} {confidence_label.lower()} outlook for a {profile.name.lower()} profile "
        f"using {horizon_profile.name.lower()} strategy rules: composite score {composite_score:.0f}/99 "
        f"(rule score {float(rule_score):.0f}/99). {primary_driver}{primary_caution}"
    )

    return CompositeOutlook(
        symbol=symbol,
        current_price=price,
        horizon=horizon,
        risk_profile_name=profile.name,
        direction=direction,
        confidence_label=confidence_label,
        rule_score=float(rule_score),
        composite_score=composite_score,
        summary=summary,
        drivers=tuple(drivers),
        cautions=tuple(cautions),
        model_probability_up=float(model_probability_up) if model_probability_up is not None and model_probability_up == model_probability_up else None,
        analyst_upside=analyst_upside,
        news_sentiment=news_sentiment,
        days_until_earnings=resolved_days_until_earnings,
    )


@dataclass(frozen=True)
class HorizonProfile:
    name: str
    full_buy_adjustment: float
    momentum_buy_adjustment: float
    stop_multiplier: float
    short_target_atr: float
    long_target_multiplier: float
    score_bias: float
    description: str


def build_signal(
    symbol: str,
    frame: pd.DataFrame,
    news_sentiment: float = 0.0,
    intent: str = "Buy / Add",
    horizon: str = "Buy-dip",
    news_items: pd.DataFrame | None = None,
    market_regime: dict[str, object] | None = None,
    ml_signal: object | None = None,
    risk_profile: RiskProfile | str | None = None,
    analyst_snapshot: dict[str, object] | None = None,
    days_until_earnings: int | None = None,
) -> StrategySignal:
    """Build the rule-based signal.
 
    `market_regime` (from risk.market_regime) and `ml_signal` (from
    ml_signal.latest_model_signal) are both optional so this function stays
    usable standalone. When provided, they add regime dampening and an
    independently-validated model probability alongside the rule-based
    score rather than replacing it — the two are complementary checks, and
    disagreement between them is itself useful information.
 
    `risk_profile` (a RiskProfile or its name — "Conservative"/"Balanced"/
    "Growth") controls how much confirmation is required before a BUY is
    suggested and how hard a market-downtrend regime dampens the score.
    `analyst_snapshot` is the cleaned Yahoo snapshot dict; if it carries
    analyst consensus fields, they're folded in as a third independent
    check alongside the rule-based score and the ML model. `days_until_earnings`
    surfaces near-term earnings-date event risk when known.
    """
 
    profile = risk_profile if isinstance(risk_profile, RiskProfile) else get_risk_profile(risk_profile or "Balanced")
    horizon_profile = get_horizon_profile(horizon)
 
    if frame.empty:
        return StrategySignal(
            symbol=symbol,
            current_price=0,
            intent=intent,
            horizon=horizon,
            action="NO DATA",
            confidence=0,
            setup="Insufficient data",
            action_justification="No historical price data was available for the selected instrument.",
            entry_zone="N/A",
            exit_zone="N/A",
            risk_note="Fetch historical data before making a decision.",
            reasons=("No price history available.",),
            news_drivers=(),
            risk_profile_name=profile.name,
        )
 
    latest = frame.iloc[-1]
    price = float(latest["Close"])
    ema20 = float(latest.get("EMA20", np.nan))
    ema50 = float(latest.get("EMA50", np.nan))
    rsi = float(latest.get("RSI14", 50))
    atr = float(latest.get("ATR14", np.nan))
    volume_ratio = float(latest.get("VolumeRatio", np.nan))
    trend = float(latest.get("TrendScore", 0))
    near_ema20 = abs(price - ema20) / price <= 0.025 if ema20 == ema20 else False
    above_ema20 = price >= ema20 if ema20 == ema20 else False
    above_ema50 = price >= ema50 if ema50 == ema50 else False
 
    score = 50.0
    reasons: list[str] = []
    news_drivers = summarize_news_drivers(news_items)
    score += horizon_profile.score_bias
    reasons.append(f"Strategy profile: {horizon_profile.description}")
 
    if 40 <= rsi < 60:
        score += 10
        reasons.append(f"RSI {rsi:.1f} is constructive without being overheated.")
    elif rsi < 35:
        score += 7
        reasons.append(f"RSI {rsi:.1f} is oversold; reversal possible but needs confirmation.")
    elif rsi >= 70:
        score -= 18
        reasons.append(f"RSI {rsi:.1f} is overbought; chase risk is elevated.")
    else:
        reasons.append(f"RSI {rsi:.1f} is neutral-to-watch.")
 
    if near_ema20:
        score += 12
        reasons.append("Price is near the 20-day EMA, matching the pullback-entry rule.")
    elif above_ema20:
        score += 5
        reasons.append("Price is above the 20-day EMA, showing short-term trend support.")
    else:
        score -= 8
        reasons.append(
            "Price is already below the 20-day EMA; wait for reclaim, basing, or deeper support."
        )
 
    if above_ema50 and trend > 0:
        score += 8
        reasons.append("20/50-day trend structure is positive.")
    elif trend < -0.15:
        score -= 10
        reasons.append("Medium-term momentum is negative.")
 
    if volume_ratio == volume_ratio and volume_ratio > 1.4:
        score += 5
        reasons.append(f"Volume is {volume_ratio:.1f}x the 20-day average.")
 
    macd_hist = float(latest.get("MACDHist", np.nan))
    if macd_hist == macd_hist:
        if macd_hist > 0:
            score += 4
            reasons.append("MACD histogram is positive, confirming upward momentum.")
        elif macd_hist < 0:
            score -= 4
            reasons.append("MACD histogram is negative, momentum is fading or reversing down.")
 
    adx_val = float(latest.get("ADX14", np.nan))
    trend_regime = latest.get("TrendRegime", None)
    if adx_val == adx_val:
        if trend_regime == "Choppy":
            score -= 6
            reasons.append(
                f"ADX {adx_val:.0f} indicates a choppy, range-bound market; trend-following "
                "rules like this one are more prone to whipsaws here."
            )
        elif adx_val >= 25:
            reasons.append(f"ADX {adx_val:.0f} confirms a tradeable trend is in place.")
 
    relative_strength = float(latest.get("RelativeStrength", np.nan))
    if relative_strength == relative_strength:
        if relative_strength > 1.05:
            score += 4
            reasons.append(f"Stock is outperforming its benchmark (RS {relative_strength:.2f}).")
        elif relative_strength < 0.95:
            score -= 4
            reasons.append(f"Stock is underperforming its benchmark (RS {relative_strength:.2f}).")
 
    regime_note = ""
    if market_regime and market_regime.get("dampen_buy_confidence"):
        score -= float(market_regime.get("dampen_points", profile.regime_dampen_points))
        regime_note = str(market_regime.get("note", ""))
        reasons.append(f"Market regime check: {regime_note}")
    elif market_regime:
        regime_note = str(market_regime.get("note", ""))
 
    if news_sentiment > 0.15:
        score += 5
        reasons.append("Recent scanned news sentiment is positive.")
    elif news_sentiment < -0.15:
        score -= 7
        reasons.append("Recent scanned news sentiment is negative.")
    elif news_items is not None and not news_items.empty:
        reasons.append("Recent scanned news sentiment is mixed or neutral.")
 
    gaps = detect_gap_zones(frame)
    if not gaps.empty:
        top_gap = gaps.iloc[0]
        reasons.append(
            f"Unfilled {top_gap['Direction'].lower()} zone remains near "
            f"{top_gap['Lower']:.2f}-{top_gap['Upper']:.2f}."
        )
 
    model_probability_up = None
    model_note = ""
    if ml_signal is not None:
        model_probability_up = getattr(ml_signal, "probability_up", None)
        model_note = getattr(ml_signal, "note", "")
        beats_baseline = getattr(ml_signal, "beats_baseline", False)
        if model_probability_up == model_probability_up:  # not NaN
            direction = "up" if model_probability_up >= 0.5 else "down"
            confidence_tag = "validated edge" if beats_baseline else "no validated edge — treat as noise"
            reasons.append(
                f"ML model (walk-forward validated) puts P({direction}) at "
                f"{max(model_probability_up, 1 - model_probability_up):.0%} over the model's horizon "
                f"[{confidence_tag}]."
            )
            rule_leans_up = score >= 55
            model_leans_up = model_probability_up >= 0.5
            if beats_baseline and rule_leans_up != model_leans_up:
                reasons.append(
                    "Note: the rule-based signal and the ML model disagree on direction — "
                    "treat this setup with extra caution."
                )

    analyst_note = build_analyst_note(analyst_snapshot or {}, price)
    analyst_score_adjustment = analyst_score_delta(analyst_snapshot or {}, price)
    if analyst_note:
        score += analyst_score_adjustment
        reasons.append(analyst_note)

    earnings_note = build_earnings_note(analyst_snapshot or {}, days_until_earnings)
    days_to_earnings = resolve_days_until_earnings(analyst_snapshot or {}, days_until_earnings)
    if earnings_note:
        reasons.append(earnings_note)
        if days_to_earnings is not None and days_to_earnings <= 7:
            if profile.name == "Conservative":
                score -= 14
            elif profile.name == "Growth":
                score -= 8
            else:
                score -= 12
        elif days_to_earnings is not None and days_to_earnings <= 14 and profile.name == "Conservative":
            score -= 5

    confidence = float(np.clip(score, 1, 99))
    action, setup, action_justification = choose_action(
        confidence=confidence,
        rsi=rsi,
        near_ema20=near_ema20,
        above_ema20=above_ema20,
        above_ema50=above_ema50,
        trend=trend,
        news_sentiment=news_sentiment,
        intent=intent,
        horizon=horizon,
        risk_profile=profile,
        horizon_profile=horizon_profile,
    )
 
    stop_multiple = profile.stop_atr_multiple * horizon_profile.stop_multiplier
    stop = price - (stop_multiple * atr) if atr == atr else price * 0.94
    target_short = price + (horizon_profile.short_target_atr * atr) if atr == atr else price * 1.08
    target_long = price * horizon_profile.long_target_multiplier if trend > 0 else price * 1.08
    analyst_target = _num((analyst_snapshot or {}).get("targetMeanPrice"))
    target_note = ""
    if analyst_target is not None and analyst_target > 0:
        target_note = f"; analyst mean target {analyst_target:.2f}"
 
    entry_zone = build_entry_zone(action, price, ema20, atr, rsi, near_ema20, above_ema20)
    composite_outlook = build_composite_outlook(
        symbol=symbol,
        frame=frame,
        rule_score=confidence,
        action=action,
        horizon=horizon,
        news_sentiment=news_sentiment,
        market_regime=market_regime,
        ml_signal=ml_signal,
        analyst_snapshot=analyst_snapshot or {},
        days_until_earnings=days_to_earnings,
        risk_profile=profile,
    )
    forecast_note = composite_outlook.summary
 
    return StrategySignal(
        symbol=symbol,
        current_price=price,
        intent=intent,
        horizon=horizon,
        action=action,
        confidence=confidence,
        setup=setup,
        action_justification=action_justification,
        entry_zone=entry_zone,
        reasons=tuple(reasons),
        news_drivers=news_drivers,
        model_probability_up=model_probability_up,
        model_note=model_note,
        regime_note=regime_note,
        analyst_note=analyst_note,
        earnings_note=earnings_note,
        risk_profile_name=profile.name,
        forecast_note=forecast_note,
        exit_zone=f"Short-term target {target_short:.2f}; long-term checkpoint {target_long:.2f}{target_note}",
        risk_note=(
            f"Suggested invalidation/stop area: {stop:.2f} "
            f"({stop_multiple:.1f}x ATR {profile.name.lower()} + {horizon_profile.name.lower()} stop). "
            "Size positions so this loss is acceptable."
        ),
    )
 
 
def get_horizon_profile(horizon: str) -> HorizonProfile:
    key = horizon.lower()
    if "short" in key:
        return HorizonProfile(
            name="Short-term",
            full_buy_adjustment=4.0,
            momentum_buy_adjustment=3.0,
            stop_multiplier=0.9,
            short_target_atr=1.5,
            long_target_multiplier=1.08,
            score_bias=-2.0,
            description=(
                "short-term mode prioritizes cleaner technical timing and tighter risk because the "
                "holding window is only 1-4 weeks."
            ),
        )
    if "long" in key:
        return HorizonProfile(
            name="Long-term",
            full_buy_adjustment=-3.0,
            momentum_buy_adjustment=0.0,
            stop_multiplier=1.25,
            short_target_atr=2.0,
            long_target_multiplier=1.22,
            score_bias=2.0,
            description=(
                "long-term mode allows wider volatility bands and gives more room for valuation, "
                "analyst targets, income, and 6-12 month trend development."
            ),
        )
    return HorizonProfile(
        name="Buy-dip",
        full_buy_adjustment=0.0,
        momentum_buy_adjustment=0.0,
        stop_multiplier=1.0,
        short_target_atr=2.0,
        long_target_multiplier=1.18,
        score_bias=0.0,
        description="buy-dip mode waits for either an attractive pullback zone or reversal confirmation.",
    )


def _num(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or np.isinf(number):
        return None
    return number


def analyst_score_delta(snapshot: dict[str, object], price: float) -> float:
    target = _num(snapshot.get("targetMeanPrice"))
    rating = str(snapshot.get("recommendationKey") or "").lower()
    rating_score = _num(snapshot.get("recommendationMean"))
    opinions = _num(snapshot.get("numberOfAnalystOpinions")) or 0
    if not target or price <= 0 or opinions < 2:
        return 0.0

    upside = target / price - 1
    delta = 0.0
    if upside >= 0.2:
        delta += 6
    elif upside >= 0.08:
        delta += 3
    elif upside <= -0.1:
        delta -= 5

    if rating in {"buy", "strong_buy"}:
        delta += 3
    elif rating in {"sell", "strong_sell", "underperform"}:
        delta -= 5
    elif rating in {"hold", "neutral"}:
        delta -= 1
    elif rating_score is not None:
        if rating_score <= 2.0:
            delta += 3
        elif rating_score >= 3.5:
            delta -= 3
    return delta


def build_analyst_note(snapshot: dict[str, object], price: float) -> str:
    target = _num(snapshot.get("targetMeanPrice"))
    median_target = _num(snapshot.get("targetMedianPrice"))
    rating = str(snapshot.get("recommendationKey") or snapshot.get("averageAnalystRating") or "").replace("_", " ")
    rating_score = _num(snapshot.get("recommendationMean"))
    opinions = _num(snapshot.get("numberOfAnalystOpinions"))
    if not target or price <= 0:
        return ""

    upside = target / price - 1
    coverage = f" across {opinions:.0f} analysts" if opinions else ""
    rating_part = f", consensus {rating}" if rating else ""
    median_part = f", median target {median_target:.2f}" if median_target is not None else ""
    score_part = f", rating score {rating_score:.2f}" if rating_score is not None else ""
    return (
        f"Analyst consensus: mean target {target:.2f} implies {upside:+.1%} "
        f"upside/downside{coverage}{rating_part}{median_part}{score_part}."
    )


def resolve_days_until_earnings(snapshot: dict[str, object], explicit_days: int | None = None) -> int | None:
    if explicit_days is not None:
        return explicit_days
    for key in ("nextEarningsDate", "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        value = snapshot.get(key)
        if value in {None, ""}:
            continue
        parsed = _parse_event_date(value)
        if parsed is None:
            continue
        return (parsed - datetime.now(timezone.utc).date()).days
    return None


def build_earnings_note(snapshot: dict[str, object], explicit_days: int | None = None) -> str:
    days = resolve_days_until_earnings(snapshot, explicit_days)
    if days is None:
        return ""
    if days < 0:
        return f"Earnings event risk: last known earnings date was {abs(days)} days ago; verify whether fresh results are already reflected in price."
    if days <= 7:
        return f"Earnings event risk: next known earnings date is within {days} days, so gap risk is high."
    if days <= 21:
        return f"Earnings event risk: next known earnings date is in {days} days; consider smaller sizing or waiting for the event."
    return f"Earnings event risk: next known earnings date is about {days} days away."


def _parse_event_date(value: object) -> date | None:
    if isinstance(value, (int, float)) and value > 10_000:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def build_forecast_note(
    confidence: float,
    action: str,
    model_probability_up: float | None,
    analyst_snapshot: dict[str, object],
    price: float,
    rsi: float,
    above_ema20: bool,
    above_ema50: bool,
    days_until_earnings: int | None,
    risk_profile: RiskProfile,
    horizon_profile: HorizonProfile,
) -> str:
    evidence: list[str] = []
    if model_probability_up is not None and model_probability_up == model_probability_up:
        evidence.append(f"ML directional probability is {model_probability_up:.0%}.")
    target = _num(analyst_snapshot.get("targetMeanPrice"))
    if target and price > 0:
        evidence.append(f"Analyst mean target implies {(target / price - 1):+.1%}.")
    if rsi < 35:
        evidence.append("RSI is oversold, so rebound odds improve only after confirmation.")
    elif 40 <= rsi < 60:
        evidence.append("RSI is constructive without being overheated.")
    if above_ema20 and above_ema50:
        evidence.append("Price is above the 20-day and 50-day EMA.")
    elif not above_ema20:
        evidence.append("Price is below the 20-day EMA, so timing risk remains elevated.")
    if days_until_earnings is not None and 0 <= days_until_earnings <= 14:
        evidence.append("Upcoming earnings can override technical signals.")

    if action.startswith("BUY"):
        stance = "Favorable but still conditional"
    elif "WAIT" in action or "AVOID" in action:
        stance = "Watchlist/confirmation-first"
    elif "SELL" in action:
        stance = "Risk-reduction biased"
    else:
        stance = "Neutral monitoring"
    return (
        f"{stance} forecast for a {risk_profile.name.lower()} profile using {horizon_profile.name.lower()} "
        f"strategy rules: rule score {confidence:.0f}/99. " + " ".join(evidence)
    )


def choose_action(
    confidence: float,
    rsi: float,
    near_ema20: bool,
    above_ema20: bool,
    above_ema50: bool,
    trend: float,
    news_sentiment: float,
    intent: str,
    horizon: str,
    risk_profile: RiskProfile | None = None,
    horizon_profile: HorizonProfile | None = None,
) -> tuple[str, str, str]:
    profile = risk_profile or get_risk_profile("Balanced")
    horizon_rules = horizon_profile or get_horizon_profile(horizon)
    intent_key = intent.lower()
    horizon_key = horizon.lower()
    weak_trend = not above_ema20 or (not above_ema50 and trend < 0)
    rsi_buy_ceiling = 55 if profile.name == "Conservative" else 60 if profile.name == "Balanced" else 65
    momentum_rsi_ceiling = 60 if profile.name == "Conservative" else 65 if profile.name == "Balanced" else 70
    full_buy_threshold = profile.min_confidence_full_buy + horizon_rules.full_buy_adjustment
    momentum_buy_threshold = profile.min_confidence_momentum_buy + horizon_rules.momentum_buy_adjustment
    strong_buy_setup = (
        confidence >= full_buy_threshold
        and rsi < rsi_buy_ceiling
        and near_ema20
        and above_ema20
    )
    momentum_setup = (
        confidence >= momentum_buy_threshold
        and rsi < momentum_rsi_ceiling
        and above_ema20
    )
 
    if "sell" in intent_key or "trim" in intent_key:
        if rsi >= 70:
            return (
                "SELL / TRIM INTO STRENGTH",
                "Overbought exit setup",
                "The user selected a sell/trim workflow and RSI is overbought, so the data supports taking risk off rather than adding exposure.",
            )
        if weak_trend and news_sentiment < -0.15:
            return (
                "SELL / REDUCE RISK",
                "Weak trend with negative news",
                "The user selected a sell/trim workflow, price trend is weak, and scanned news sentiment is negative.",
            )
        if rsi < 35:
            return (
                "WAIT / REVIEW STOP",
                "Oversold sell-risk zone",
                "The user selected a sell/trim workflow, but RSI is already oversold; selling now may be reactive unless a predefined stop or thesis break has triggered.",
            )
        return (
            "HOLD / SET EXIT ALERT",
            "No urgent sell trigger",
            "The user selected a sell/trim workflow, but the technical and news evidence does not show a high-conviction exit trigger.",
        )
 
    if "hold" in intent_key or "watch" in intent_key:
        if weak_trend:
            return (
                "HOLD / WATCH CONFIRMATION",
                "Trend confirmation pending",
                "The user selected a hold/watch workflow and the price has not confirmed strength above the short-term trend line.",
            )
        return (
            "HOLD / MONITOR",
            "Position monitoring",
            "The user selected a hold/watch workflow and the current evidence supports monitoring rather than forcing a new trade.",
        )
 
    if rsi < 35 and not above_ema20:
        if "dip" in horizon_key:
            return (
                "WAIT / REVERSAL WATCH",
                "Oversold below 20-day EMA",
                "The stock is oversold but still below the 20-day EMA; for a buy-dip plan, wait for basing, a reclaim, or a risk-defined dip entry rather than treating oversold alone as a buy.",
            )
        return (
            "WAIT / DO NOT CHASE",
            "Oversold without confirmation",
            "The user selected a buy workflow, but price is below the 20-day EMA and RSI is oversold, so confirmation is missing.",
        )
    if strong_buy_setup:
        if "long" in horizon_key and above_ema50:
            return (
                "ACCUMULATE / START POSITION",
                "Long-term accumulation setup",
                "The user selected a long-term workflow, valuation/consensus and trend evidence meet the profile-adjusted threshold, and the setup supports staged accumulation rather than a short-term trade.",
            )
        return (
            "BUY 50% SCALE-IN",
            "Pullback entry",
            "The user selected a buy workflow and the setup matches the selected strategy thresholds: RSI is controlled, price is near the 20-day EMA, and short-term trend support is intact.",
        )
    if momentum_setup:
        if "short" in horizon_key:
            return (
                "TRADE SMALL / TIGHT STOP",
                "Short-term momentum setup",
                "The user selected a short-term workflow and momentum is constructive, but the strategy requires tighter sizing and faster invalidation.",
            )
        if "long" in horizon_key and above_ema50:
            return (
                "ACCUMULATE SMALL",
                "Long-term starter setup",
                "The user selected a long-term workflow; evidence is constructive enough for a starter position, but not strong enough for full allocation.",
            )
        return (
            "BUY SMALL / WAIT FOR EMA20",
            "Momentum with chase control",
            "The user selected a buy workflow and trend is constructive, but the entry is less attractive than an EMA20 pullback, so position size should be smaller.",
        )
    if rsi >= 70:
        return (
            "WAIT / AVOID NEW BUY",
            "Overbought risk",
            "The user selected a buy workflow, but RSI is overbought, so new entries have poor chase-risk control.",
        )
    if confidence <= 40:
        return (
            "AVOID / WAIT",
            "Weak confirmation",
            "The user selected a buy workflow, but the combined technical/news score is too weak for a new entry.",
        )
    return (
        "WAIT",
        "Needs cleaner entry",
        "The user selected a buy workflow, but the current data does not meet the RSI, EMA, trend, and news confirmation needed for a cleaner entry.",
    )
 
 
def summarize_news_drivers(news_items: pd.DataFrame | None, limit: int = 3) -> tuple[str, ...]:
    if news_items is None or news_items.empty:
        return ()
 
    required = {"Source", "Title", "Sentiment"}
    if not required.issubset(news_items.columns):
        return ()
 
    sorted_news = news_items.copy()
    sorted_news["AbsSentiment"] = sorted_news["Sentiment"].abs()
    sort_columns = ["AbsSentiment"]
    sort_order = [False]
    if "Published" in sorted_news.columns:
        sort_columns.append("Published")
        sort_order.append(False)
    sorted_news = sorted_news.sort_values(sort_columns, ascending=sort_order)
 
    drivers: list[str] = []
    for _, row in sorted_news.head(limit).iterrows():
        sentiment = float(row["Sentiment"])
        label = "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral"
        drivers.append(f"{row['Source']}: {row['Title']} ({label}, {sentiment:+.2f})")
    return tuple(drivers)
 
 
def build_entry_zone(
    action: str,
    price: float,
    ema20: float,
    atr: float,
    rsi: float,
    near_ema20: bool,
    above_ema20: bool,
) -> str:
    has_ema20 = ema20 == ema20
    has_atr = atr == atr and atr > 0
 
    if action.startswith("BUY"):
        if has_ema20:
            return f"{min(price, ema20):.2f}-{max(price, ema20):.2f}"
        return f"near {price:.2f}"
 
    if not has_ema20:
        return "Wait for cleaner support"
 
    if not above_ema20 and rsi < 35:
        if has_atr:
            dip_low = max(price - (0.5 * atr), 0)
            dip_high = price + (0.25 * atr)
            reclaim = min(ema20, price + atr)
            return (
                f"Aggressive dip watch {dip_low:.2f}-{dip_high:.2f}; "
                f"confirmation only after reclaiming {reclaim:.2f} or EMA20 {ema20:.2f}"
            )
        return (
            f"Price is below EMA20 {ema20:.2f}; wait for basing near current price "
            "or a reclaim before buying"
        )
 
    if not above_ema20:
        if has_atr:
            support = max(price - atr, 0)
            return f"Wait for reclaim of EMA20 {ema20:.2f} or support to hold near {support:.2f}-{price:.2f}"
        return f"Wait for reclaim of EMA20 around {ema20:.2f}"
 
    if near_ema20:
        return f"Watch current EMA20 zone around {ema20:.2f}"
 
    return f"Wait for pullback toward EMA20 around {ema20:.2f}"
 
 
def strategy_table(signal: StrategySignal) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Horizon": "Short-term",
                "Window": "1-4 weeks",
                "Plan": signal.action if signal.confidence >= 60 else "WAIT",
                "Entry": signal.entry_zone,
                "Exit / Checkpoint": signal.exit_zone.split(";")[0],
            },
            {
                "Horizon": "Long-term",
                "Window": "6-12 months",
                "Plan": "Accumulate only on confirmed uptrend/pullbacks" if signal.confidence >= 55 else "Watchlist only",
                "Entry": signal.entry_zone,
                "Exit / Checkpoint": signal.exit_zone.split(";")[-1].strip(),
            },
            {
                "Horizon": "Buy-dip",
                "Window": "Event-driven",
                "Plan": "Scale first 50%; add after reclaim/hold" if signal.action.startswith("BUY") else "Set alert near EMA20/support",
                "Entry": signal.entry_zone,
                "Exit / Checkpoint": signal.risk_note,
            },
        ]
    )
