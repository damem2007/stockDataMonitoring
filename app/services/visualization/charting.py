from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def resolve_chart_type(choice: str, range_label: str) -> str:
    if choice != "Auto":
        return choice
    if range_label in {"1D", "5D", "1W", "1M"}:
        return "Candlestick"
    return "Line"


def price_chart(
    symbol: str,
    frame: pd.DataFrame,
    comparison_histories: dict[str, pd.DataFrame] | None = None,
    show_bbands: bool = True,
    show_donchian: bool = False,
    show_vwap: bool = False,
    chart_type: str = "Candlestick",
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=(None, "RSI 14 / ADX 14", "MACD"),
    )

    fig.add_trace(primary_price_trace(symbol, frame, chart_type), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["EMA20"], name="EMA20", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["EMA50"], name="EMA50", line=dict(width=1.5)), row=1, col=1)

    if show_bbands and "BBUpper" in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame["Date"], y=frame["BBUpper"], name="BB Upper", line=dict(width=1, dash="dot"), opacity=0.6),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=frame["Date"], y=frame["BBLower"], name="BB Lower", line=dict(width=1, dash="dot"), opacity=0.6),
            row=1,
            col=1,
        )

    if show_donchian and "DonchianHigh20" in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame["Date"], y=frame["DonchianHigh20"], name="Donchian High", line=dict(width=1, dash="dash"), opacity=0.6),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=frame["Date"], y=frame["DonchianLow20"], name="Donchian Low", line=dict(width=1, dash="dash"), opacity=0.6),
            row=1,
            col=1,
        )

    if show_vwap and "VWAP20" in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame["Date"], y=frame["VWAP20"], name="VWAP (20d)", line=dict(width=1.5, color="orange")),
            row=1,
            col=1,
        )

    for comparison_symbol, comparison_frame in (comparison_histories or {}).items():
        if comparison_frame.empty:
            continue
        series = comparison_frame.dropna(subset=["Date", "Close"]).sort_values("Date")
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series["Date"],
                y=series["Close"],
                mode="lines",
                name=f"{comparison_symbol} Close",
                line=dict(width=2),
                opacity=0.85,
                hovertemplate="%{x}<br>Close %{y:.2f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )

    if "RSI14" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame["RSI14"], name="RSI 14", line=dict(width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(220,50,50,0.4)", dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(50,150,50,0.4)", dash="dot"), row=2, col=1)
    if "ADX14" in frame.columns:
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame["ADX14"], name="ADX 14", line=dict(width=1.5, color="purple")), row=2, col=1)
        fig.add_hline(y=25, line=dict(color="rgba(120,80,200,0.3)", dash="dash"), row=2, col=1)

    if "MACDHist" in frame.columns:
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in frame["MACDHist"].fillna(0)]
        fig.add_trace(go.Bar(x=frame["Date"], y=frame["MACDHist"], name="MACD Hist", marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame["MACD"], name="MACD", line=dict(width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame["MACDSignal"], name="Signal", line=dict(width=1.2)), row=3, col=1)

    add_chart_range_selector(fig, height=760)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#171717"),
        yaxis_title="Price",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


def primary_price_trace(symbol: str, frame: pd.DataFrame, chart_type: str):
    if chart_type == "OHLC":
        return go.Ohlc(
            x=frame["Date"],
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name=symbol,
        )
    if chart_type == "Line":
        return go.Scatter(
            x=frame["Date"],
            y=frame["Close"],
            mode="lines",
            name=f"{symbol} Close",
            line=dict(width=2.2),
        )
    if chart_type == "Area":
        return go.Scatter(
            x=frame["Date"],
            y=frame["Close"],
            mode="lines",
            name=f"{symbol} Close",
            fill="tozeroy",
            line=dict(width=2.1),
            opacity=0.82,
        )
    if chart_type == "Bar":
        colors = ["#16a34a" if close >= open_ else "#ef4444" for open_, close in zip(frame["Open"], frame["Close"])]
        return go.Bar(
            x=frame["Date"],
            y=frame["Close"],
            name=f"{symbol} Close",
            marker_color=colors,
            hovertemplate="%{x}<br>Close %{y:.2f}<extra>%{fullData.name}</extra>",
        )
    return go.Candlestick(
        x=frame["Date"],
        open=frame["Open"],
        high=frame["High"],
        low=frame["Low"],
        close=frame["Close"],
        name=symbol,
    )


def add_chart_range_selector(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=72),
        xaxis_rangeslider_visible=False,
    )


def volume_profile_chart(profile: pd.DataFrame) -> go.Figure:
    chart_data = profile.sort_values("MidPrice").copy()
    fig = go.Figure(
        go.Bar(
            x=chart_data["MidPrice"],
            y=chart_data["Volume"],
            hovertemplate="Price %{x:.2f}<br>Volume %{y:,.0f}<extra></extra>",
            name="Volume",
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#171717"),
        height=320,
        xaxis_title="Price bucket",
        yaxis_title="Volume",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_xaxes(tickformat=".2f")
    return fig
