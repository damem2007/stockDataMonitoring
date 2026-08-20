from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from backend.app.schemas.roles import InstrumentRole


class OnboardingRequest(BaseModel):
    exchange: str = "TSX"
    role: InstrumentRole = InstrumentRole.watching
    instruments: list[dict[str, Any]]


class InstrumentRequest(BaseModel):
    instruments: list[dict[str, Any]]


class AlertRequest(BaseModel):
    alert: dict[str, Any]


class AlertListRequest(BaseModel):
    alerts: list[dict[str, Any]]
