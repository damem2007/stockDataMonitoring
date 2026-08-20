from __future__ import annotations

from enum import Enum


class InstrumentRole(str, Enum):
    watching = "Watching"
    trading = "Trading"


class TradeIntent(str, Enum):
    buy_add = "Buy / Add"
    sell_trim = "Sell / Trim"
    hold_watch = "Hold / Watch"


class StrategyHorizon(str, Enum):
    short_term = "Short-term (1-4 weeks)"
    long_term = "Long-term (6-12 months)"
    buy_dip = "Buy-dip"
