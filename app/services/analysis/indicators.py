from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(frame: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the full indicator set used across the app.

    `benchmark` is an optional OHLC frame (e.g. the S&P 500 or TSX Composite)
    used to compute relative strength. Pass it whenever you want RS available;
    the app fetches the benchmark once per exchange and reuses it.
    """

    if frame.empty:
        return frame

    df = frame.copy()
    df["Return"] = df["Close"].pct_change()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["VolumeAvg20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    df["VolumeRatio"] = df["Volume"] / df["VolumeAvg20"] if "Volume" in df.columns else np.nan
    df["TrendScore"] = trend_score(df)

    macd_line, signal_line, histogram = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACDSignal"] = signal_line
    df["MACDHist"] = histogram

    mid, upper, lower, bandwidth, percent_b = bollinger_bands(df["Close"])
    df["BBMid"] = mid
    df["BBUpper"] = upper
    df["BBLower"] = lower
    df["BBBandwidth"] = bandwidth
    df["BBPercentB"] = percent_b

    plus_di, minus_di, adx_val = adx(df)
    df["PlusDI"] = plus_di
    df["MinusDI"] = minus_di
    df["ADX14"] = adx_val
    df["TrendRegime"] = np.where(df["ADX14"] >= 25, "Trending", "Choppy")

    df["VWAP20"] = rolling_vwap(df, window=20)

    donchian_high, donchian_low, donchian_mid = donchian_channels(df, window=20)
    df["DonchianHigh20"] = donchian_high
    df["DonchianLow20"] = donchian_low
    df["DonchianMid20"] = donchian_mid

    if benchmark is not None and not benchmark.empty:
        df["RelativeStrength"] = relative_strength(df, benchmark)
    else:
        df["RelativeStrength"] = np.nan

    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = frame["High"] - frame["Low"]
    high_close = (frame["High"] - frame["Close"].shift()).abs()
    low_close = (frame["Low"] - frame["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def trend_score(frame: pd.DataFrame) -> pd.Series:
    close = frame["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    momentum = close.pct_change(20).fillna(0)
    ema_spread = ((ema20 - ema50) / close).replace([np.inf, -np.inf], 0).fillna(0)
    raw = (momentum * 2.5) + (ema_spread * 4)
    return raw.clip(-1, 1)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence: trend + momentum confirmation.

    Crossovers of MACD above/below its signal line are the classic trigger;
    the histogram (MACD - signal) shows whether momentum is accelerating or
    fading even before a crossover happens.
    """

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: volatility envelope around a moving average.

    Returns (mid, upper, lower, bandwidth, percent_b). Bandwidth (band width
    relative to price) flags volatility squeezes; percent_b shows where price
    sits within the band (0 = lower band, 1 = upper band) and is useful for
    mean-reversion setups.
    """

    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    bandwidth = ((upper - lower) / mid).replace([np.inf, -np.inf], np.nan)
    percent_b = ((series - lower) / (upper - lower)).replace([np.inf, -np.inf], np.nan)
    return mid, upper, lower, bandwidth, percent_b


def adx(frame: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index + Directional Indicators.

    ADX measures trend *strength* regardless of direction (values above ~25
    typically indicate a tradeable trend; below ~20 suggests a choppy,
    range-bound market where trend-following rules like RSI+EMA pullbacks
    tend to whipsaw). +DI/-DI show which direction currently dominates.
    """

    high, low, close = frame["High"], frame["Low"], frame["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=frame.index)
    minus_dm = pd.Series(minus_dm, index=frame.index)

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_smoothed = true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * (plus_dm_smoothed / atr_smoothed.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smoothed / atr_smoothed.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    return plus_di.fillna(0), minus_di.fillna(0), adx_value.fillna(0)


def rolling_vwap(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling N-day volume-weighted average price.

    Daily bars have no intraday session to anchor a "true" VWAP to, so this
    is a rolling window VWAP: it weights each day's typical price by that
    day's volume, giving a volume-aware alternative to a plain moving
    average. Price sustained above VWAP suggests real buying interest rather
    than just drift on light volume.
    """

    if "Volume" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)

    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    pv = typical_price * frame["Volume"]
    rolling_pv = pv.rolling(window).sum()
    rolling_vol = frame["Volume"].rolling(window).sum()
    return rolling_pv / rolling_vol.replace(0, np.nan)


def donchian_channels(frame: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels: highest-high / lowest-low breakout levels.

    A close beyond the prior N-day high/low is a classic breakout trigger
    (the core of trend-following systems like the Turtle strategy) and is a
    cleaner breakout signal than a moving-average crossover.
    """

    high = frame["High"].rolling(window).max()
    low = frame["Low"].rolling(window).min()
    mid = (high + low) / 2
    return high, low, mid


def relative_strength(frame: pd.DataFrame, benchmark: pd.DataFrame, window: int = 63) -> pd.Series:
    """Relative strength vs. a benchmark index (e.g. S&P 500 / TSX Composite).

    Computes the ratio of the stock's cumulative return to the benchmark's
    cumulative return over a rolling window (default ~1 trading quarter).
    Values above 1 mean the stock is outperforming its market; this catches
    stocks that are technically "healthy" on their own chart but quietly
    losing ground to the index, and vice versa.
    """

    if benchmark.empty or "Date" not in benchmark.columns:
        return pd.Series(np.nan, index=frame.index)

    stock = frame.set_index("Date")["Close"]
    bench = benchmark.set_index("Date")["Close"].reindex(stock.index).ffill()

    stock_roll_return = stock / stock.shift(window) - 1
    bench_roll_return = bench / bench.shift(window) - 1
    rs = (1 + stock_roll_return) / (1 + bench_roll_return)
    return rs.reset_index(drop=True)


def detect_gap_zones(frame: pd.DataFrame, min_gap_pct: float = 0.02) -> pd.DataFrame:
    """Detect unfilled daily open/close gaps and estimate their importance."""

    if frame.empty or len(frame) < 2:
        return pd.DataFrame()

    df = frame.copy()
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close
    zones: list[dict[str, float | str | pd.Timestamp]] = []

    for idx in df.index[gap_pct.abs() >= min_gap_pct]:
        previous_close = float(prev_close.loc[idx])
        open_price = float(df.loc[idx, "Open"])
        lower, upper = sorted((previous_close, open_price))
        future = df.loc[idx + 1 :]
        filled = bool(((future["Low"] <= upper) & (future["High"] >= lower)).any())
        if filled:
            continue

        volume_ratio = float(df.loc[idx, "VolumeRatio"]) if "VolumeRatio" in df.columns else np.nan
        zones.append(
            {
                "Date": df.loc[idx, "Date"],
                "Direction": "Gap Up" if open_price > previous_close else "Gap Down",
                "Lower": lower,
                "Upper": upper,
                "GapPct": float(gap_pct.loc[idx]),
                "VolumeRatio": volume_ratio,
                "Priority": abs(float(gap_pct.loc[idx])) * (volume_ratio if volume_ratio == volume_ratio else 1),
            }
        )

    return pd.DataFrame(zones).sort_values("Priority", ascending=False) if zones else pd.DataFrame()


def price_volume_profile(frame: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    if frame.empty or "Volume" not in frame.columns:
        return pd.DataFrame()

    prices = frame["Close"].dropna()
    if prices.empty or prices.min() == prices.max():
        return pd.DataFrame()

    bucket = pd.cut(frame["Close"], bins=bins)
    profile = frame.groupby(bucket, observed=True)["Volume"].sum().reset_index()
    intervals = profile["Close"].tolist()
    profile["PriceLow"] = [float(interval.left) for interval in intervals]
    profile["PriceHigh"] = [float(interval.right) for interval in intervals]
    profile["MidPrice"] = (profile["PriceLow"] + profile["PriceHigh"]) / 2
    return profile.drop(columns=["Close"]).sort_values("Volume", ascending=False)


def correlation_matrix(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = []
    for symbol, frame in histories.items():
        if not frame.empty:
            closes.append(frame.set_index("Date")["Close"].rename(symbol).pct_change())

    if not closes:
        return pd.DataFrame()
    returns = pd.concat(closes, axis=1).dropna(how="all")
    return returns.corr()
