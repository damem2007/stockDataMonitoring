from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_pullback_strategy(
    frame: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest the RSI + EMA20 pullback rule with fixed-horizon exits.

    `cost_bps` is applied once at entry and once at exit. The strategy is
    intentionally fixed-rule here; it is not fit on the same data it is tested
    on, and the UI separately surfaces walk-forward window consistency.
    """

    if frame.empty or len(frame) < hold_days + 50:
        return pd.DataFrame(), {}

    df = frame.copy().sort_values("Date").reset_index(drop=True)
    entries = (df["RSI14"] < 60) & ((df["Close"] - df["EMA20"]).abs() / df["Close"] <= 0.025)
    entries &= df["Close"] >= df["EMA50"]
    round_trip_cost = (float(cost_bps) / 10_000) * 2

    trades = []
    last_exit = -1
    for idx in df.index[entries]:
        if idx <= last_exit or idx + hold_days >= len(df):
            continue
        entry = float(df.loc[idx, "Close"])
        exit_price = float(df.loc[idx + hold_days, "Close"])
        low_path = float(df.loc[idx : idx + hold_days, "Low"].min())
        gross_ret = (exit_price - entry) / entry
        ret = gross_ret - round_trip_cost
        max_drawdown = (low_path - entry) / entry
        trades.append(
            {
                "EntryDate": df.loc[idx, "Date"],
                "ExitDate": df.loc[idx + hold_days, "Date"],
                "Entry": entry,
                "Exit": exit_price,
                "GrossReturn": gross_ret,
                "CostBpsRoundTrip": float(cost_bps) * 2,
                "Return": ret,
                "MaxDrawdown": max_drawdown,
            }
        )
        last_exit = idx + hold_days

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return trade_frame, {}

    returns = trade_frame["Return"]
    equity = (1 + returns).cumprod()
    equity_drawdown = equity / equity.cummax() - 1
    strategy_total_return = float(equity.iloc[-1] - 1)
    matched = _matched_price_window(df, trade_frame)
    benchmark = buy_and_hold_stats(matched, cost_bps=cost_bps)
    annualization = _annualization_factor(trade_frame)
    annualized_return = _annualize_return(strategy_total_return, trade_frame)
    benchmark_annualized = benchmark.get("annualized_return", float("nan"))

    stats = {
        "trades": float(len(trade_frame)),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "best_return": float(returns.max()),
        "worst_return": float(returns.min()),
        "avg_drawdown": float(trade_frame["MaxDrawdown"].mean()),
        "profit_factor": _profit_factor(returns),
        "strategy_total_return": strategy_total_return,
        "annualized_return": annualized_return,
        "benchmark_total_return": benchmark.get("total_return", float("nan")),
        "benchmark_annualized_return": benchmark_annualized,
        "excess_annualized_return": (
            annualized_return - benchmark_annualized
            if annualized_return == annualized_return and benchmark_annualized == benchmark_annualized
            else float("nan")
        ),
        "max_equity_drawdown": float(equity_drawdown.min()),
        "sharpe_ratio": _sharpe_ratio(returns, annualization),
        "sortino_ratio": _sortino_ratio(returns, annualization),
        "calmar_ratio": (
            annualized_return / abs(float(equity_drawdown.min()))
            if equity_drawdown.min() < 0 and annualized_return == annualized_return
            else float("nan")
        ),
        "low_sample_warning": bool(len(trade_frame) < 30),
        "cost_bps_per_leg": float(cost_bps),
    }
    return trade_frame, stats


def buy_and_hold_stats(frame: pd.DataFrame, cost_bps: float = 0.0) -> dict[str, float]:
    """Buy-and-hold baseline over the supplied frame, net of entry/exit costs."""

    if frame.empty or len(frame) < 2:
        return {"total_return": float("nan"), "annualized_return": float("nan")}

    df = frame.copy().sort_values("Date")
    first = float(df["Close"].iloc[0])
    last = float(df["Close"].iloc[-1])
    if first <= 0:
        return {"total_return": float("nan"), "annualized_return": float("nan")}

    total_return = (last / first - 1) - ((float(cost_bps) / 10_000) * 2)
    days = max((pd.to_datetime(df["Date"].iloc[-1]) - pd.to_datetime(df["Date"].iloc[0])).days, 1)
    annualized = (1 + total_return) ** (365 / days) - 1 if total_return > -1 else -1.0
    return {"total_return": float(total_return), "annualized_return": float(annualized)}


def walk_forward_backtest(
    frame: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 0.0,
    n_windows: int = 4,
    min_window_rows: int = 80,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the same fixed strategy across sequential non-overlapping windows."""

    if frame.empty or len(frame) < min_window_rows * 2:
        return pd.DataFrame(), {}

    df = frame.copy().sort_values("Date").reset_index(drop=True)
    windows = np.array_split(df.index.to_numpy(), n_windows)
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for window_id, index_values in enumerate(windows, start=1):
        if len(index_values) < min_window_rows:
            continue
        window = df.iloc[index_values].reset_index(drop=True)
        trades, stats = backtest_pullback_strategy(window, hold_days=hold_days, cost_bps=cost_bps)
        baseline = buy_and_hold_stats(window, cost_bps=cost_bps)
        strategy_return = stats.get("strategy_total_return", float("nan")) if stats else float("nan")
        benchmark_return = baseline.get("total_return", float("nan"))
        rows.append(
            {
                "Window": window_id,
                "Start": window["Date"].iloc[0],
                "End": window["Date"].iloc[-1],
                "Trades": float(len(trades)),
                "StrategyReturn": strategy_return,
                "BuyHoldReturn": benchmark_return,
                "ExcessReturn": (
                    strategy_return - benchmark_return
                    if strategy_return == strategy_return and benchmark_return == benchmark_return
                    else float("nan")
                ),
                "WinRate": stats.get("win_rate", float("nan")) if stats else float("nan"),
            }
        )

    window_frame = pd.DataFrame(rows)
    if window_frame.empty:
        return window_frame, {}

    with_trades = window_frame[window_frame["Trades"] > 0]
    beating = with_trades["ExcessReturn"] > 0 if not with_trades.empty else pd.Series(dtype=bool)
    summary = {
        "windows_with_trades": float(len(with_trades)),
        "windows_beating_benchmark": float(beating.sum()) if not beating.empty else 0.0,
        "consistency_rate": float(beating.mean()) if not beating.empty else float("nan"),
        "worst_window_return": float(with_trades["StrategyReturn"].min()) if not with_trades.empty else float("nan"),
    }
    return window_frame, summary


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _matched_price_window(frame: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return frame.iloc[0:0].copy()
    start = pd.to_datetime(trades["EntryDate"].min())
    end = pd.to_datetime(trades["ExitDate"].max())
    dates = pd.to_datetime(frame["Date"])
    return frame[(dates >= start) & (dates <= end)].copy()


def _annualization_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    durations = (
        pd.to_datetime(trades["ExitDate"]) - pd.to_datetime(trades["EntryDate"])
    ).dt.days.clip(lower=1)
    avg_days = float(durations.mean())
    return 365 / avg_days if avg_days > 0 else float("nan")


def _annualize_return(total_return: float, trades: pd.DataFrame) -> float:
    if trades.empty or total_return != total_return:
        return float("nan")
    start = pd.to_datetime(trades["EntryDate"].min())
    end = pd.to_datetime(trades["ExitDate"].max())
    days = max((end - start).days, 1)
    return float((1 + total_return) ** (365 / days) - 1) if total_return > -1 else -1.0


def _sharpe_ratio(returns: pd.Series, annualization: float) -> float:
    std = returns.std(ddof=1)
    if std == 0 or std != std or annualization != annualization:
        return float("nan")
    return float((returns.mean() / std) * np.sqrt(annualization))


def _sortino_ratio(returns: pd.Series, annualization: float) -> float:
    downside = returns[returns < 0]
    std = downside.std(ddof=1)
    if std == 0 or std != std or annualization != annualization:
        return float("nan")
    return float((returns.mean() / std) * np.sqrt(annualization))
