from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import AppUser
from backend.app.services.email_service import send_email, send_sms


def dispatch_alert_notifications(user_id: str | None, triggered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deliver external alert channels without allowing a provider outage to stop sync."""
    if not user_id or not triggered or not database_available() or not ensure_database():
        return []
    try:
        parsed_user_id = uuid.UUID(str(user_id))
    except ValueError:
        return []
    with session_scope() as session:
        user = session.execute(select(AppUser).where(AppUser.id == parsed_user_id)).scalar_one_or_none()
        if not user:
            return []
        deliveries: list[dict[str, Any]] = []
        for alert in triggered:
            body = _alert_message(alert)
            channels = {str(item) for item in (alert.get("notifications") or [])}
            if "Email" in channels:
                delivered = send_email(user.email, f"StockSignal alert: {alert.get('symbol')}", body)
                deliveries.append({"alertId": alert.get("id"), "channel": "Email", "status": "sent" if delivered else "skipped"})
            if "SMS" in channels:
                delivered = send_sms(str(user.phone_number or ""), body)
                deliveries.append({"alertId": alert.get("id"), "channel": "SMS", "status": "sent" if delivered else "skipped"})
        return deliveries


def _alert_message(alert: dict[str, Any]) -> str:
    return f"{alert.get('symbol')} {alert.get('metric')} alert: {alert.get('operator')} {alert.get('threshold')}. {alert.get('message') or 'Review your StockSignal workspace.'}"
