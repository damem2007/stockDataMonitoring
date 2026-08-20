from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import delete, select

from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import InstrumentTag as InstrumentTagModel
from backend.app.services.market_data.data_sources import WATCHLIST_DIR


WATCHLIST_METADATA_PATH = WATCHLIST_DIR / "instruments.json"
WATCHLIST_ROLES = ("Watching", "Trading")


@dataclass(frozen=True)
class InstrumentTag:
    symbol: str
    role: str = "Watching"
    active: bool = True
    notes: str = ""
    shares: float = 0.0
    average_cost: float = 0.0
    book_cost: float = 0.0
    category: str = ""
    purchase_date: str = ""
    intent: str = ""
    strategy: str = ""
    watch_reason: str = ""


def read_instrument_tags(
    path: Path = WATCHLIST_METADATA_PATH,
    user_id: str | None = None,
) -> list[dict[str, object]]:
    if user_id and path == WATCHLIST_METADATA_PATH and database_available():
        if not ensure_database():
            return []
        parsed_user_id = _uuid(user_id)
        with session_scope() as session:
            rows = session.execute(
                select(InstrumentTagModel)
                .where(InstrumentTagModel.user_id == parsed_user_id)
                .order_by(InstrumentTagModel.created_at, InstrumentTagModel.symbol)
            ).scalars()
            return [_normalize_row(_model_to_row(row)) for row in rows]
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = payload.get("instruments", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        return []
    return [_normalize_row(row) for row in rows if isinstance(row, dict) and row.get("symbol")]


def write_instrument_tags(
    rows: list[dict[str, object]],
    path: Path = WATCHLIST_METADATA_PATH,
    user_id: str | None = None,
) -> None:
    normalized = [_normalize_row(row) for row in rows if str(row.get("symbol", "")).strip()]
    if user_id and path == WATCHLIST_METADATA_PATH and database_available():
        if not ensure_database():
            raise RuntimeError("Database storage is configured but unreachable; watchlist changes were not saved.")
        parsed_user_id = _uuid(user_id)
        with session_scope() as session:
            session.execute(delete(InstrumentTagModel).where(InstrumentTagModel.user_id == parsed_user_id))
            for row in normalized:
                purchase_date = row.get("purchase_date") or None
                session.add(
                    InstrumentTagModel(
                        user_id=parsed_user_id,
                        symbol=str(row["symbol"]),
                        role=str(row["role"]),
                        active=bool(row["active"]),
                        notes=str(row["notes"]),
                        shares=float(row["shares"]),
                        average_cost=float(row["average_cost"]),
                        book_cost=float(row["book_cost"]),
                        category=str(row["category"]),
                        purchase_date=_date_or_none(purchase_date),
                        intent=str(row["intent"]),
                        strategy=str(row["strategy"]),
                        watch_reason=str(row["watch_reason"]),
                    )
                )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "instruments": normalized}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_instrument_tags(symbols: list[str], stored_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stored_by_symbol = {
        str(row.get("symbol", "")).strip().upper(): _normalize_row(row)
        for row in stored_rows
        if str(row.get("symbol", "")).strip()
    }
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for symbol in symbols:
        cleaned = symbol.strip().upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        row = stored_by_symbol.get(cleaned, _default_row(cleaned))
        row["symbol"] = cleaned
        merged.append(row)
    return merged


def active_symbols(rows: list[dict[str, object]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen or not bool(row.get("active", True)):
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def role_for_symbol(rows: list[dict[str, object]], symbol: str) -> str:
    cleaned = symbol.strip().upper()
    for row in rows:
        if str(row.get("symbol", "")).strip().upper() == cleaned:
            return str(row.get("role") or "Watching")
    return "Watching"


def trading_instruments(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        _normalize_row(row)
        for row in rows
        if str(row.get("role") or "").strip().title() == "Trading"
        and bool(row.get("active", True))
    ]


def _default_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "role": "Watching",
        "active": True,
        "notes": "",
        "shares": 0.0,
        "average_cost": 0.0,
        "book_cost": 0.0,
        "category": "",
        "purchase_date": "",
        "intent": "",
        "strategy": "",
        "watch_reason": "",
    }


def _normalize_row(row: dict[str, object]) -> dict[str, object]:
    role = str(row.get("role") or row.get("tag") or "Watching").strip().title()
    if role not in WATCHLIST_ROLES:
        role = "Watching"
    shares = _float_or_zero(row.get("shares"))
    average_cost = _float_or_zero(row.get("average_cost") or row.get("avg_cost"))
    book_cost = _float_or_zero(row.get("book_cost"))
    if book_cost <= 0 and shares > 0 and average_cost > 0:
        book_cost = shares * average_cost
    if average_cost <= 0 and shares > 0 and book_cost > 0:
        average_cost = book_cost / shares
    return {
        "symbol": str(row.get("symbol", "")).strip().upper(),
        "role": role,
        "active": bool(row.get("active", True)),
        "notes": str(row.get("notes") or ""),
        "shares": shares,
        "average_cost": average_cost,
        "book_cost": book_cost,
        "category": str(row.get("category") or row.get("sector") or ""),
        "purchase_date": str(row.get("purchase_date") or ""),
        "intent": str(row.get("intent") or ""),
        "strategy": str(row.get("strategy") or row.get("horizon") or ""),
        "watch_reason": str(row.get("watch_reason") or row.get("reason") or ""),
    }


def _float_or_zero(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed != parsed:
        return 0.0
    return max(parsed, 0.0)


def _uuid(value: object) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _date_or_none(value: object) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _model_to_row(row: InstrumentTagModel) -> dict[str, object]:
    return {
        "symbol": row.symbol,
        "role": row.role,
        "active": row.active,
        "notes": row.notes,
        "shares": float(row.shares or 0),
        "average_cost": float(row.average_cost or 0),
        "book_cost": float(row.book_cost or 0),
        "category": row.category,
        "purchase_date": row.purchase_date.isoformat() if row.purchase_date else "",
        "intent": row.intent,
        "strategy": row.strategy,
        "watch_reason": row.watch_reason,
    }
