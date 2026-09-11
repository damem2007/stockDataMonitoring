from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable

import requests
from sqlalchemy import select

from backend.app.config import PUBLIC_HOLIDAY_COUNTRIES, PUBLIC_HOLIDAY_SOURCE_URL, PUBLIC_HOLIDAY_SYNC_YEARS
from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import PublicHoliday


def sync_public_holidays(countries: Iterable[str] | None = None, years: Iterable[int] | None = None) -> dict[str, Any]:
    selected_countries = tuple(str(item).strip().upper() for item in (countries or PUBLIC_HOLIDAY_COUNTRIES) if str(item).strip())
    selected_years = tuple(years or range(date.today().year, date.today().year + PUBLIC_HOLIDAY_SYNC_YEARS))
    statuses: list[dict[str, Any]] = []
    total = 0
    for country in selected_countries:
        for year in selected_years:
            try:
                response = requests.get(PUBLIC_HOLIDAY_SOURCE_URL.format(year=year, country=country), timeout=(5, 15))
                response.raise_for_status()
                payload = response.json()
                rows = _save_rows(country, int(year), payload if isinstance(payload, list) else [])
                total += rows
                statuses.append({"country": country, "year": year, "ok": True, "rows": rows})
            except Exception as exc:
                statuses.append({"country": country, "year": year, "ok": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": all(item["ok"] for item in statuses) if statuses else True, "countries": list(selected_countries), "years": list(selected_years), "synced": total, "statuses": statuses}


def list_public_holidays(year: int, countries: Iterable[str] | None = None) -> list[dict[str, Any]]:
    if not database_available() or not ensure_database():
        return []
    selected = tuple(str(item).strip().upper() for item in (countries or PUBLIC_HOLIDAY_COUNTRIES) if str(item).strip())
    with session_scope() as session:
        query = select(PublicHoliday).where(PublicHoliday.year == year)
        if selected:
            query = query.where(PublicHoliday.country_code.in_(selected))
        rows = session.execute(query.order_by(PublicHoliday.holiday_date, PublicHoliday.country_code, PublicHoliday.name)).scalars().all()
        return [_holiday_payload(row) for row in rows]


def _save_rows(country: str, year: int, payload: list[object]) -> int:
    if not database_available() or not ensure_database():
        return 0
    parsed: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("date") or not item.get("name"):
            continue
        try:
            holiday_date = date.fromisoformat(str(item["date"])[:10])
        except ValueError:
            continue
        parsed.append({"country_code": country, "holiday_date": holiday_date, "name": str(item.get("name") or ""), "local_name": str(item.get("localName") or ""), "global_holiday": bool(item.get("global", True)), "types": [str(value) for value in (item.get("types") or [])], "source": PUBLIC_HOLIDAY_SOURCE_URL.format(year=year, country=country), "year": year, "fetched_at": datetime.now(timezone.utc)})
    with session_scope() as session:
        existing = session.execute(select(PublicHoliday).where(PublicHoliday.country_code == country, PublicHoliday.year == year)).scalars().all()
        by_key = {(row.holiday_date, row.name): row for row in existing}
        for item in parsed:
            row = by_key.get((item["holiday_date"], item["name"]))
            if row is None:
                session.add(PublicHoliday(**item))
            else:
                for key, value in item.items():
                    setattr(row, key, value)
    return len(parsed)


def _holiday_payload(row: PublicHoliday) -> dict[str, Any]:
    return {"id": str(row.id), "country": row.country_code, "date": row.holiday_date.isoformat(), "name": row.name, "localName": row.local_name, "global": row.global_holiday, "types": row.types or [], "source": row.source, "fetchedAt": row.fetched_at.isoformat() if row.fetched_at else None}
