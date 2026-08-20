from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[2]
NEWS_CACHE_DIR = ROOT / "data" / "cache" / "news"
CACHE_TTL_MINUTES = 15

_VADER = SentimentIntensityAnalyzer()
_FINBERT_PIPELINE = None
_FINBERT_LOAD_ATTEMPTED = False


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    symbol_specific: bool = True


def news_sources(symbol: str, company_name: str | None = None) -> list[NewsSource]:
    """RSS sources used for sentiment scanning.

    Only `symbol_specific=True` sources are used when computing a per-symbol
    sentiment score. The market-wide feeds (MarketWatch top stories, Reuters
    business front page) are kept for the News tab's general context, but
    they are NOT symbol-specific and mixing them into a single stock's
    sentiment score would silently dilute or contaminate it with unrelated
    headlines.
    """

    query = quote_plus(company_name or symbol)
    return [
        NewsSource(
            "Yahoo Finance",
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            symbol_specific=True,
        ),
        NewsSource(
            "Google News Finance",
            f"https://news.google.com/rss/search?q={query}%20stock%20finance&hl=en-US&gl=US&ceid=US:en",
            symbol_specific=True,
        ),
        NewsSource(
            "MarketWatch (market-wide)",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            symbol_specific=False,
        ),
        NewsSource(
            "Reuters Business (market-wide)",
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            symbol_specific=False,
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
                }
            )

    if not rows:
        empty = pd.DataFrame(
            columns=["Source", "Published", "Title", "Summary", "Url", "Sentiment", "SentimentEngine", "SymbolSpecific"]
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
            scoped = news  # fall back rather than silently returning 0

    recent = scoped.sort_values("Published", ascending=False).head(10).reset_index(drop=True)
    if recent.empty:
        return 0.0

    weights = pd.Series(range(len(recent), 0, -1), dtype=float)  # most recent gets highest weight
    weighted_avg = float((recent["Sentiment"] * weights).sum() / weights.sum())
    return weighted_avg
