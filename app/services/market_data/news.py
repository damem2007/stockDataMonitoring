from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .data_sources import DATA_DIR

#ROOT = Path(__file__).resolve().parents[2]
NEWS_CACHE_DIR = DATA_DIR / "cache" / "news"
CACHE_TTL_MINUTES = 15
NEWS_CACHE_SCHEMA_VERSION = 2

_VADER = SentimentIntensityAnalyzer()
_FINBERT_PIPELINE = None
_FINBERT_LOAD_ATTEMPTED = False


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    symbol_specific: bool = True


def news_sources(symbol: str, company_name: str | None = None) -> list[NewsSource]:
    """RSS sources used for per-symbol sentiment scanning."""

    search_query = quote_plus(news_search_query(symbol, company_name))
    return [
        NewsSource(
            "Yahoo Finance",
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            symbol_specific=True,
        ),
        NewsSource(
            "Google News Finance",
            f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en",
            symbol_specific=True,
        ),
    ]


def _load_finbert():
    """Lazily load a FinBERT sentiment pipeline, if `transformers` is available.

    FinBERT (ProsusAI/finbert) is trained on financial text and handles
    domain phrasing ("beats on EPS but guides lower", "misses revenue
    estimates") far more reliably than a general-purpose model like VADER,
    which was built for social-media text. Loading requires the
    `transformers`/`torch` packages and, on first use, a network fetch of
    model weights from Hugging Face — both optional, so we fail soft to
    VADER if either is unavailable rather than crashing the app.
    """

    global _FINBERT_PIPELINE, _FINBERT_LOAD_ATTEMPTED
    if _FINBERT_LOAD_ATTEMPTED:
        return _FINBERT_PIPELINE

    _FINBERT_LOAD_ATTEMPTED = True
    try:
        from transformers import pipeline

        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    except Exception:
        _FINBERT_PIPELINE = None
    return _FINBERT_PIPELINE


def analyze_sentiment(text: str) -> tuple[float, str]:
    """Score `text` in [-1, 1] plus which engine produced the score.

    Tries FinBERT first (finance-tuned); falls back to VADER (general
    purpose) if `transformers`/`torch` aren't installed or model download is
    unavailable in this environment. The engine used is returned alongside
    the score so the UI can be transparent about which one was active.
    """

    text = (text or "").strip()
    if not text:
        return 0.0, "none"

    finbert = _load_finbert()
    if finbert is not None:
        try:
            result = finbert(text[:512])[0]
            label = result["label"].lower()
            prob = float(result["score"])
            if label == "positive":
                return prob, "finbert"
            if label == "negative":
                return -prob, "finbert"
            return 0.0, "finbert"
        except Exception:
            pass  # fall through to VADER on any runtime failure

    return float(_VADER.polarity_scores(text)["compound"]), "vader"


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("/", "-")
    return NEWS_CACHE_DIR / f"{safe}_news.json"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema_version") != NEWS_CACHE_SCHEMA_VERSION:
        return None

    fetched_at = pd.to_datetime(payload.get("fetched_at"))
    if fetched_at is None or pd.isna(fetched_at):
        return None
    if datetime.now(timezone.utc) - fetched_at.to_pydatetime() > timedelta(minutes=CACHE_TTL_MINUTES):
        return None

    rows = payload.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["Source", "Published", "Title", "Summary", "Url", "Sentiment", "SentimentEngine", "SymbolSpecific"])
    frame = pd.DataFrame(rows)
    frame["Published"] = pd.to_datetime(frame["Published"])
    return frame


def _write_cache(symbol: str, frame: pd.DataFrame) -> None:
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": NEWS_CACHE_SCHEMA_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": json.loads(frame.to_json(orient="records", date_format="iso")),
    }
    _cache_path(symbol).write_text(json.dumps(payload), encoding="utf-8")


def fetch_news(symbol: str, company_name: str | None = None, max_items: int = 25, refresh: bool = False) -> pd.DataFrame:
    """Fetch and score recent news, using a short-lived on-disk cache.

    Live RSS feeds are slow and occasionally flaky; caching for
    `CACHE_TTL_MINUTES` avoids re-hitting four feeds on every symbol switch
    while still staying reasonably current for intraday use.
    """

    if not refresh:
        cached = _read_cache(symbol)
        if cached is not None:
            return cached

    aliases = relevance_aliases(symbol, company_name)
    rows: list[dict[str, object]] = []
    for source in news_sources(symbol, company_name):
        try:
            parsed = feedparser.parse(source.url)
        except Exception:
            continue
        for entry in parsed.entries[:max_items]:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            published = getattr(entry, "published", None)
            link = getattr(entry, "link", "")
            text = f"{title}. {summary}"
            if not is_symbol_relevant(text, symbol, aliases):
                continue
            score, engine = analyze_sentiment(text)
            rows.append(
                {
                    "Source": source.name,
                    "Published": _parse_date(published),
                    "Title": title,
                    "Summary": summary,
                    "Url": link,
                    "Sentiment": score,
                    "SentimentEngine": engine,
                    "SymbolSpecific": source.symbol_specific,
                    "MatchedAliases": ", ".join(matched_aliases(text, aliases)[:4]),
                }
            )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "Source",
                "Published",
                "Title",
                "Summary",
                "Url",
                "Sentiment",
                "SentimentEngine",
                "SymbolSpecific",
                "MatchedAliases",
            ]
        )
        return empty

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["Title", "Url"]).sort_values("Published", ascending=False)
    frame = frame.head(max_items).reset_index(drop=True)
    _write_cache(symbol, frame)
    return frame


def _parse_date(raw: str | None) -> pd.Timestamp:
    if not raw:
        return pd.Timestamp(datetime.utcnow())
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.Timestamp(datetime.utcnow())
    return parsed.tz_convert(None)


def news_sentiment_score(news: pd.DataFrame, symbol_specific_only: bool = True) -> float:
    """Average sentiment of the most recent items, decayed by recency.

    Only symbol-specific sources are used by default so a generic
    market-wide headline doesn't move an individual stock's score. Recent
    items are weighted more heavily than older ones via a simple linear
    decay across the selected window, rather than a flat unweighted mean.
    """

    if news.empty or "Sentiment" not in news.columns:
        return 0.0

    scoped = news
    if symbol_specific_only and "SymbolSpecific" in news.columns:
        scoped = news[news["SymbolSpecific"]]
        if scoped.empty:
            return 0.0

    recent = scoped.sort_values("Published", ascending=False).head(10).reset_index(drop=True)
    if recent.empty:
        return 0.0

    weights = pd.Series(range(len(recent), 0, -1), dtype=float)  # most recent gets highest weight
    weighted_avg = float((recent["Sentiment"] * weights).sum() / weights.sum())
    return weighted_avg


def news_search_query(symbol: str, company_name: str | None = None) -> str:
    cleaned = symbol.strip().upper()
    base = cleaned.split(".", 1)[0]
    terms: list[str] = []
    if company_name:
        terms.append(f'"{company_name}"')
        short_company = company_without_suffix(company_name)
        if short_company and short_company.lower() != company_name.lower():
            terms.append(f'"{short_company}"')
    terms.extend([f'"{cleaned}"', f'"{base}"'])
    if cleaned.endswith(".TO"):
        terms.extend([f'"TSX:{base}"', f'"TSE:{base}"'])
    joined = " OR ".join(dict.fromkeys(terms))
    return f"({joined}) (stock OR finance OR earnings OR acquisition)"


def relevance_aliases(symbol: str, company_name: str | None = None) -> list[str]:
    cleaned = symbol.strip().upper()
    base = cleaned.split(".", 1)[0]
    aliases = [cleaned]
    if cleaned.endswith(".TO"):
        aliases.extend([f"TSX:{base}", f"TSE:{base}", f"({base})"])
    else:
        aliases.extend([f"NASDAQ:{base}", f"NYSE:{base}", f"({base})"])
        if len(base) >= 4:
            aliases.append(base)
    if company_name:
        aliases.extend(company_aliases(company_name))
    return [alias for alias in dict.fromkeys(alias.strip() for alias in aliases) if alias]


def company_aliases(company_name: str) -> list[str]:
    name = re.sub(r"\s+", " ", company_name).strip()
    if not name:
        return []
    aliases = [name]
    without_suffix = company_without_suffix(name)
    if without_suffix:
        aliases.append(without_suffix)
    hyphen_parts = [
        part.strip()
        for part in re.split(r"[-|]", without_suffix or name)
        if len(part.strip()) >= 6 or " " in part.strip()
    ]
    aliases.extend(hyphen_parts)
    words = [word for word in re.split(r"\W+", without_suffix or name) if len(word) >= 5]
    if len(words) >= 2:
        aliases.append(" ".join(words[:2]))
    return aliases


def company_without_suffix(company_name: str) -> str:
    name = re.sub(r"\s+", " ", company_name).strip()
    suffix_pattern = r"\b(incorporated|inc|corp|corporation|company|co|ltd|limited|plc|class\s+[a-z])\.?$"
    previous = ""
    while name and previous != name:
        previous = name
        name = re.sub(suffix_pattern, "", name, flags=re.IGNORECASE).strip(" ,.-")
    return name


def is_symbol_relevant(text: str, symbol: str, aliases: list[str]) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    for alias in aliases:
        if alias_matches(lowered, alias):
            return True
    base = symbol.strip().upper().split(".", 1)[0]
    if len(base) <= 3:
        return bool(
            re.search(rf"\b(tsx|tse|nasdaq|nyse)\s*[:\-]\s*{re.escape(base.lower())}\b", lowered)
            or re.search(rf"\({re.escape(base.lower())}\)", lowered)
        )
    return bool(re.search(rf"\b{re.escape(base.lower())}\b", lowered))


def matched_aliases(text: str, aliases: list[str]) -> list[str]:
    lowered = (text or "").lower()
    return [alias for alias in aliases if alias_matches(lowered, alias)]


def alias_matches(lowered_text: str, alias: str) -> bool:
    lowered_alias = alias.lower()
    if not lowered_alias:
        return False
    if any(separator in lowered_alias for separator in (":", ".", "(", ")", "-", " ")):
        return lowered_alias in lowered_text
    return bool(re.search(rf"\b{re.escape(lowered_alias)}\b", lowered_text))
