from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends

from backend.app.routes.dependencies import optional_user
from backend.app.schemas.workspace import AlertListRequest, AlertRequest
from backend.app.services.auth import AuthUser
from backend.app.services.workspace.application import (
    alerts_payload,
    create_alert_payload,
    delete_alert_payload,
    evaluate_alerts_payload,
    replace_alerts_payload,
)


router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("")
def list_alerts(user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return {"alerts": alerts_payload(user_id=user.id if user else None)}


@router.post("")
def create_alert_route(payload: AlertRequest, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return {"alert": create_alert_payload(payload.alert, user_id=user.id if user else None)}


@router.put("")
def replace_alerts_route(payload: AlertListRequest, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return {"alerts": replace_alerts_payload(payload.alerts, user_id=user.id if user else None)}


@router.delete("/{alert_id}")
def delete_alert_route(alert_id: str, user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return {"alerts": delete_alert_payload(alert_id, user_id=user.id if user else None)}


@router.post("/evaluate")
def evaluate_alerts_route(user: Optional[AuthUser] = Depends(optional_user)) -> dict[str, Any]:
    return evaluate_alerts_payload(user_id=user.id if user else None)
