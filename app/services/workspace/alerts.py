from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import PriceAlert as PriceAlertModel
from backend.app.services.market_data.data_sources import DATA_DIR


ALERTS_PATH = DATA_DIR / "alerts" / "price_alerts.json"
PORTFOLIO_ALERT_SYMBOL = "__PORTFOLIO__"
ALERT_METRICS = (
    "Price",
    "Volume",
    "SMA 200",
    "RSI 14",
    "Portfolio Value",
    "Portfolio Today P/L",
    "Portfolio Since Purchase P/L",
)
ALERT_OPERATORS = ("Crossing", "Crossing Up", "Crossing Down", "Above", "Below")
ALERT_TRIGGERS = ("Once only", "Every sync")


@dataclass
class PriceAlert:
    symbol: str
    metric: str
    operator: str
    threshold: float
    trigger: str = "Once only"
    expiration: str | None = None
    message: str = ""
    notifications: list[str] = field(default_factory=lambda: ["In-app", "Toast"])
    enabled: bool = True
    role: str = "Watching"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_value: float | None = None
    last_triggered_at: str | None = None


def read_alerts(path: Path = ALERTS_PATH, user_id: str | None = None) -> list[dict[str, Any]]:
    if user_id and path == ALERTS_PATH and database_available():
        if not ensure_database():
            return []
        parsed_user_id = _uuid(user_id)
        with session_scope() as session:
            rows = session.execute(
                select(PriceAlertModel)
                .where(PriceAlertModel.user_id == parsed_user_id)
                .order_by(PriceAlertModel.created_at)
            ).scalars()
            return [_normalize_alert(_model_to_alert(row)) for row in rows]
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("alerts", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        return []
    return [_normalize_alert(row) for row in rows if isinstance(row, dict) and row.get("symbol")]


def write_alerts(
    alerts: list[dict[str, Any]],
    path: Path = ALERTS_PATH,
    user_id: str | None = None,
) -> None:
    normalized = [_normalize_alert(alert) for alert in alerts if str(alert.get("symbol", "")).strip()]
    if user_id and path == ALERTS_PATH and database_available():
        if not ensure_database():
            raise RuntimeError("Database storage is configured but unreachable; alert changes were not saved.")
        parsed_user_id = _uuid(user_id)
        with session_scope() as session:
            session.execute(delete(PriceAlertModel).where(PriceAlertModel.user_id == parsed_user_id))
            for alert in normalized:
                session.add(
                    PriceAlertModel(
                        id=_uuid(alert["id"]),
                        user_id=parsed_user_id,
                        symbol=str(alert["symbol"]),
                        role=str(alert["role"]),
                        metric=str(alert["metric"]),
                        operator=str(alert["operator"]),
                        threshold=float(alert["threshold"]),
                        trigger=str(alert["trigger"]),
                        expiration=_date_or_none(alert.get("expiration")),
                        message=str(alert["message"]),
                        notifications=list(alert["notifications"]),
                        enabled=bool(alert["enabled"]),
                        created_at=_datetime_or_now(alert.get("created_at")),
                        last_value=alert.get("last_value"),
                        last_triggered_at=_datetime_or_none(alert.get("last_triggered_at")),
                    )
                )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "alerts": normalized}, indent=2), encoding="utf-8")


def create_alert(**kwargs: Any) -> dict[str, Any]:
    alert = PriceAlert(**kwargs)
    return _normalize_alert(alert.__dict__)


def evaluate_alerts(
    alerts: list[dict[str, Any]],
    latest_by_symbol: dict[str, dict[str, float | None]],
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now = now or datetime.now(timezone.utc)
    updated: list[dict[str, Any]] = []
    triggered: list[dict[str, Any]] = []
    for raw in alerts:
        alert = _normalize_alert(raw)
        if not alert.get("enabled", True) or _is_expired(alert.get("expiration"), now.date()):
            updated.append(alert)
            continue

        symbol = str(alert["symbol"]).upper()
        metric_key = _metric_key(str(alert["metric"]))
        current_value = latest_by_symbol.get(symbol, {}).get(metric_key)
        if current_value is None:
            updated.append(alert)
            continue

        current = float(current_value)
        previous = alert.get("last_value")
        fires = _condition_fires(
            previous=float(previous) if previous is not None else None,
            current=current,
            threshold=float(alert["threshold"]),
            operator=str(alert["operator"]),
        )
        alert["last_value"] = current
        if fires:
            alert["last_triggered_at"] = now.isoformat()
            triggered.append(alert.copy())
            if alert.get("trigger") == "Once only":
                alert["enabled"] = False
        updated.append(alert)
    return updated, triggered


def latest_alert_inputs(symbol: str, frame, live_quote: dict[str, object] | None = None) -> dict[str, float | None]:
    if frame is None or frame.empty:
        return {"price": None, "volume": None, "sma200": None, "rsi14": None}
    latest = frame.iloc[-1]
    quote = live_quote or {}
    return {
        "price": _float_or_none(quote.get("price")) or _float_or_none(latest.get("Close")),
        "volume": _float_or_none(quote.get("volume")) or _float_or_none(latest.get("Volume")),
        "sma200": _float_or_none(latest.get("SMA200")),
        "rsi14": _float_or_none(latest.get("RSI14")),
    }


def _normalize_alert(alert: dict[str, Any]) -> dict[str, Any]:
    metric = str(alert.get("metric") or "Price")
    operator = str(alert.get("operator") or "Crossing")
    trigger = str(alert.get("trigger") or "Once only")
    raw_notifications = alert.get("notifications") or ["In-app", "Toast"]
    if isinstance(raw_notifications, str):
        try:
            parsed_notifications = json.loads(raw_notifications)
        except json.JSONDecodeError:
            parsed_notifications = [item.strip() for item in raw_notifications.split(",")]
    else:
        parsed_notifications = list(raw_notifications)
    notifications = [str(item).strip() for item in parsed_notifications if str(item).strip()]
    return {
        "id": str(alert.get("id") or uuid.uuid4().hex),
        "symbol": str(alert.get("symbol", "")).strip().upper(),
        "role": str(alert.get("role") or "Watching"),
        "metric": metric if metric in ALERT_METRICS else "Price",
        "operator": operator if operator in ALERT_OPERATORS else "Crossing",
        "threshold": float(alert.get("threshold") or 0),
        "trigger": trigger if trigger in ALERT_TRIGGERS else "Once only",
        "expiration": str(alert.get("expiration")) if alert.get("expiration") else None,
        "message": str(alert.get("message") or ""),
        "notifications": notifications or ["In-app", "Toast"],
        "enabled": bool(alert.get("enabled", True)),
        "created_at": str(alert.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "last_value": _float_or_none(alert.get("last_value")),
        "last_triggered_at": str(alert.get("last_triggered_at")) if alert.get("last_triggered_at") else None,
    }


def _condition_fires(previous: float | None, current: float, threshold: float, operator: str) -> bool:
    if operator == "Above":
        return current > threshold
    if operator == "Below":
        return current < threshold
    if previous is None:
        return False
    if operator == "Crossing Up":
        return previous < threshold <= current
    if operator == "Crossing Down":
        return previous > threshold >= current
    return (previous < threshold <= current) or (previous > threshold >= current)


def _is_expired(value: object, today: date) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value)).date() if "T" in str(value) else date.fromisoformat(str(value))
    except ValueError:
        return False
    return parsed < today


def _metric_key(metric: str) -> str:
    return {
        "Price": "price",
        "Volume": "volume",
        "SMA 200": "sma200",
        "RSI 14": "rsi14",
        "Portfolio Value": "portfolio_value",
        "Portfolio Today P/L": "portfolio_today_pl",
        "Portfolio Since Purchase P/L": "portfolio_since_purchase_pl",
    }.get(metric, "price")


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    text = str(value)
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.UUID(hex=text)


def _date_or_none(value: object) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)).date() if "T" in str(value) else date.fromisoformat(str(value))
    except ValueError:
        return None


def _datetime_or_none(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed


def _datetime_or_now(value: object) -> datetime:
    return _datetime_or_none(value) or datetime.now(timezone.utc)


def _model_to_alert(row: PriceAlertModel) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "symbol": row.symbol,
        "role": row.role,
        "metric": row.metric,
        "operator": row.operator,
        "threshold": float(row.threshold or 0),
        "trigger": row.trigger,
        "expiration": row.expiration.isoformat() if row.expiration else None,
        "message": row.message,
        "notifications": row.notifications,
        "enabled": row.enabled,
        "created_at": row.created_at.isoformat() if row.created_at else datetime.now(timezone.utc).isoformat(),
        "last_value": _float_or_none(row.last_value),
        "last_triggered_at": row.last_triggered_at.isoformat() if row.last_triggered_at else None,
    }
