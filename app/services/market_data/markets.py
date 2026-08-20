from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exchange:
    name: str
    yahoo_suffix: str
    examples: tuple[str, ...]
    benchmark_symbol: str
    benchmark_name: str


EXCHANGES = {
    "TSX": Exchange(
        "Toronto Stock Exchange",
        ".TO",
        ("ENB.TO", "SHOP.TO", "RY.TO", "TD.TO", "BNS.TO"),
        benchmark_symbol="^GSPTSE",
        benchmark_name="S&P/TSX Composite",
    ),
    "NASDAQ": Exchange(
        "NASDAQ",
        "",
        ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN"),
        benchmark_symbol="^IXIC",
        benchmark_name="NASDAQ Composite",
    ),
    "NYSE": Exchange(
        "New York Stock Exchange",
        "",
        ("JPM", "DIS", "BA", "KO", "JNJ"),
        benchmark_symbol="^GSPC",
        benchmark_name="S&P 500",
    ),
}

DEFAULT_BENCHMARK_SYMBOL = "^GSPC"
DEFAULT_BENCHMARK_NAME = "S&P 500"


def benchmark_for_symbol(symbol: str) -> tuple[str, str]:
    """Pick a sensible benchmark index for a given ticker's suffix.

    Falls back to the S&P 500 for anything not matching a known TSX-style
    suffix, since most NASDAQ/NYSE symbols carry no suffix at all.
    """

    cleaned = symbol.strip().upper()
    if cleaned.endswith(".TO") or cleaned.endswith(".V"):
        return EXCHANGES["TSX"].benchmark_symbol, EXCHANGES["TSX"].benchmark_name
    return DEFAULT_BENCHMARK_SYMBOL, DEFAULT_BENCHMARK_NAME

DEFAULT_WATCHLISTS = {
    "Mixed": ("ENB.TO", "SHOP.TO", "RY.TO", "AAPL", "MSFT", "NVDA", "JPM", "TSLA"),
    **{exchange: config.examples for exchange, config in EXCHANGES.items()},
}


def normalize_symbol(symbol: str, exchange: str | None = None) -> str:
    """Normalize a user-entered ticker for Yahoo Finance.

    Yahoo Finance uses `.TO` for many TSX listings. NASDAQ and NYSE symbols usually
    have no suffix, so we leave them as typed after uppercasing.
    """

    cleaned = symbol.strip().upper()
    if not cleaned:
        return cleaned

    if exchange == "TSX" and "." not in cleaned:
        return f"{cleaned}.TO"
    return cleaned


def split_symbols(raw: str, exchange: str | None = None) -> list[str]:
    return [
        normalize_symbol(part, exchange)
        for part in raw.replace("\n", ",").split(",")
        if normalize_symbol(part, exchange)
    ]
