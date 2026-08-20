from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    instruments: Mapped[list["InstrumentTag"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[list["PriceAlert"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class InstrumentTag(Base):
    __tablename__ = "instrument_tags"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="instrument_tags_user_symbol_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="Watching")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    shares: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    average_cost: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    book_cost: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    category: Mapped[str] = mapped_column(String, nullable=False, default="")
    purchase_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    intent: Mapped[str] = mapped_column(String, nullable=False, default="")
    strategy: Mapped[str] = mapped_column(String, nullable=False, default="")
    watch_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[AppUser] = relationship(back_populates="instruments")


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="Watching")
    metric: Mapped[str] = mapped_column(String, nullable=False, default="Price")
    operator: Mapped[str] = mapped_column(String, nullable=False, default="Crossing")
    threshold: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    trigger: Mapped[str] = mapped_column(String, nullable=False, default="Once only")
    expiration: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    notifications: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=lambda: ["In-app", "Toast"])
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_value: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[AppUser] = relationship(back_populates="alerts")
