from __future__ import annotations

from typing import Any, Optional

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


class PortfolioNameRequest(BaseModel):
    name: str


class HoldingActivityRequest(BaseModel):
    symbol: str
    trade_date: Optional[str] = None
    purchase_date: Optional[str] = None
    shares: float
    price: Optional[float] = None
    average_cost: Optional[float] = None
    fees: float = 0
    note: str = ""
    intent: str = "Hold / Watch"
    strategy: str = "Buy-dip"


class HoldingCloseRequest(BaseModel):
    action: str
    trade_date: Optional[str] = None
    shares: Optional[float] = None
    selling_price: Optional[float] = None
    fees: float = 0
    note: str = ""
