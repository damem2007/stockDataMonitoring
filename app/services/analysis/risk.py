from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
 
 
@dataclass(frozen=True)
class RiskProfile:
    """A named bundle of thresholds, modeled loosely on Wealthsimple's
    risk-tolerance-quiz pattern (Conservative/Balanced/Growth portfolios).
 
    The dashboard previously applied the same fixed thresholds to every
    user — a 72+ rule-based score for a full BUY signal, a flat 1% account
    risk per trade, a 0.7 correlation warning threshold — regardless of how
    much risk someone actually wants to take. This makes "personal" in
    "personal stock advisor" mean something: the same underlying evidence
    can lead to a different suggested action depending on the profile.
    """
 
    name: str
    description: str
    risk_pct_per_trade: float
    stop_atr_multiple: float
    min_confidence_full_buy: float
    min_confidence_momentum_buy: float
    correlation_warning_threshold: float
    regime_dampen_points: float
 
 
RISK_PROFILES: dict[str, RiskProfile] = {
    "Conservative": RiskProfile(
        name="Conservative",
        description=(
            "Prioritizes capital preservation: smaller position sizes, wider stops, and a "
            "higher confirmation bar before suggesting a new buy."
        ),
        risk_pct_per_trade=0.005,
        stop_atr_multiple=2.0,
        min_confidence_full_buy=80.0,
        min_confidence_momentum_buy=75.0,
        correlation_warning_threshold=0.6,
        regime_dampen_points=15.0,
    ),
    "Balanced": RiskProfile(
        name="Balanced",
        description="The dashboard's original default thresholds — a middle ground between caution and growth.",
        risk_pct_per_trade=0.01,
        stop_atr_multiple=1.5,
        min_confidence_full_buy=72.0,
        min_confidence_momentum_buy=68.0,
        correlation_warning_threshold=0.7,
        regime_dampen_points=10.0,
    ),
    "Growth": RiskProfile(
        name="Growth",
        description=(
            "Prioritizes not missing moves: larger position sizes, tighter stops, and a lower "
            "confirmation bar — accepts more false signals in exchange for catching more real ones."
        ),
        risk_pct_per_trade=0.02,
        stop_atr_multiple=1.2,
        min_confidence_full_buy=64.0,
        min_confidence_momentum_buy=60.0,
        correlation_warning_threshold=0.8,
        regime_dampen_points=6.0,
    ),
}
 
 
def get_risk_profile(name: str) -> RiskProfile:
    return RISK_PROFILES.get(name, RISK_PROFILES["Balanced"])
 
 
@dataclass(frozen=True)
class PositionSizeSuggestion:
    shares: int
    dollar_allocation: float
    risk_dollars: float
    stop_price: float
    note: str
 
 
def volatility_target_position_size(
    price: float,
    atr: float,
    account_size: float,
    risk_pct_per_trade: float = 0.01,
    stop_atr_multiple: float = 1.5,
) -> PositionSizeSuggestion:
    """Size a position so a stop-out risks a fixed % of the account, not a fixed share count.
 
    This is the standard volatility-targeting approach: riskier (higher-ATR)
    stocks get smaller position sizes for the same dollar risk, instead of
    every symbol getting the same arbitrary "50% scale-in." `risk_pct_per_trade`
    is the fraction of the account you're willing to lose if the stop is hit;
    1% is a common conservative default for discretionary trading.
    """
 
    if price <= 0 or account_size <= 0:
        return PositionSizeSuggestion(0, 0.0, 0.0, 0.0, "Invalid price or account size.")
 
    if atr <= 0 or atr != atr:
        # No usable volatility estimate; fall back to a flat stop distance.
        stop_distance = price * 0.06
    else:
        stop_distance = atr * stop_atr_multiple
 
    stop_price = max(price - stop_distance, 0.01)
    risk_dollars = account_size * risk_pct_per_trade
    shares = int(risk_dollars // stop_distance) if stop_distance > 0 else 0
    dollar_allocation = shares * price
 
    note = (
        f"Sized so a stop at {stop_price:.2f} ({stop_atr_multiple:.1f}x ATR below entry) "
        f"risks ~{risk_pct_per_trade:.1%} of account equity."
    )
    return PositionSizeSuggestion(shares, dollar_allocation, risk_dollars, stop_price, note)
 
 
def correlation_warnings(
    correlation: pd.DataFrame,
    candidate_symbol: str,
    threshold: float = 0.7,
) -> list[tuple[str, float]]:
    """Flag existing watchlist symbols that are highly correlated with a candidate.
 
    Adding a new "diversifying" position that actually moves in lockstep with
    something you already hold doesn't reduce portfolio risk the way it looks
    like it should. Returns (symbol, correlation) pairs above `threshold`,
    sorted by strength.
    """
 
    if correlation.empty or candidate_symbol not in correlation.columns:
        return []
 
    column = correlation[candidate_symbol].drop(labels=[candidate_symbol], errors="ignore")
    flagged = column[column.abs() >= threshold].sort_values(ascending=False)
    return [(symbol, float(value)) for symbol, value in flagged.items()]
 
 
def market_regime(benchmark_frame: pd.DataFrame, profile: RiskProfile | None = None) -> dict[str, object]:
    """Classify the broad market regime from a benchmark index's own indicators.
 
    Buy signals generated during a benchmark downtrend are working against a
    strong headwind ("don't fight the tape"); this doesn't block trades, but
    gives the app a basis to dampen confidence and surface a warning rather
    than presenting a buy-dip signal on a single stock as if the rest of the
    market weren't falling apart at the same time. A Conservative profile
    dampens harder than a Growth profile for the same downtrend.
    """
 
    dampen_points = (profile or get_risk_profile("Balanced")).regime_dampen_points
 
    if benchmark_frame.empty or "EMA50" not in benchmark_frame.columns:
        return {
            "regime": "Unknown",
            "dampen_buy_confidence": False,
            "dampen_points": 0.0,
            "note": "No benchmark data available.",
        }
 
    latest = benchmark_frame.iloc[-1]
    price = float(latest["Close"])
    ema50 = float(latest.get("EMA50", np.nan))
    adx_val = float(latest.get("ADX14", np.nan))
    trend_regime = latest.get("TrendRegime", "Unknown")
 
    if ema50 != ema50:
        return {
            "regime": "Unknown",
            "dampen_buy_confidence": False,
            "dampen_points": 0.0,
            "note": "Insufficient benchmark history.",
        }
 
    if price < ema50 and trend_regime == "Trending":
        return {
            "regime": "Downtrend",
            "dampen_buy_confidence": True,
            "dampen_points": dampen_points,
            "note": (
                f"Benchmark is below its 50-day EMA in a trending (ADX {adx_val:.0f}) market — "
                "broad conditions are unfavorable for new long entries."
            ),
        }
    if price >= ema50 and trend_regime == "Trending":
        return {
            "regime": "Uptrend",
            "dampen_buy_confidence": False,
            "dampen_points": 0.0,
            "note": f"Benchmark is above its 50-day EMA in a trending (ADX {adx_val:.0f}) market.",
        }
    return {
        "regime": "Choppy",
        "dampen_buy_confidence": False,
        "dampen_points": 0.0,
        "note": f"Benchmark is range-bound (ADX {adx_val:.0f}) — no strong directional bias either way.",
    }
 