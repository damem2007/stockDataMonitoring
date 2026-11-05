from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from app.database import Base
except ModuleNotFoundError:
    from app.database import Base


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    phone_number: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    onboarding_status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="active")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    instruments: Mapped[list["InstrumentTag"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[list["PriceAlert"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    portfolios: Mapped[list["PortfolioAccount"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    activities: Mapped[list["PortfolioActivity"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    saved_market_explorer_analyses: Mapped[list["SavedMarketExplorerAnalysis"]] = relationship(back_populates="user", cascade="all, delete-orphan")


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


class PortfolioAccount(Base):
    __tablename__ = "portfolio_accounts"
    __table_args__ = (UniqueConstraint("user_id", "name", name="portfolio_accounts_user_name_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    account_type: Mapped[str] = mapped_column(String, nullable=False, default="Trading")
    currency: Mapped[str] = mapped_column(String, nullable=False, default="CAD")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[AppUser] = relationship(back_populates="portfolios")
    activities: Mapped[list["PortfolioActivity"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class PortfolioActivity(Base):
    __tablename__ = "portfolio_activities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False)
    portfolio_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("portfolio_accounts.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    activity_type: Mapped[str] = mapped_column(String, nullable=False, default="BUY")
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    shares: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    price: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    amount: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    fees: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False, default="Completed")
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    realized_pl: Mapped[Decimal] = mapped_column(Numeric, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user: Mapped[AppUser] = relationship(back_populates="activities")
    portfolio: Mapped[PortfolioAccount] = relationship(back_populates="activities")


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


class SavedMarketExplorerAnalysis(Base):
    __tablename__ = "saved_market_explorer_analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="Market Explorer analysis")
    profile: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    user: Mapped[AppUser] = relationship(back_populates="saved_market_explorer_analyses")


class PublicHoliday(Base):
    __tablename__ = "public_holidays"
    __table_args__ = (UniqueConstraint("country_code", "holiday_date", "name", name="public_holidays_country_date_name_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    holiday_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    local_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    global_holiday: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    types: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    source: Mapped[str] = mapped_column(String, nullable=False, default="")
    year: Mapped[int] = mapped_column(nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PredictionProviderDiagnostic(Base):
    __tablename__ = "prediction_provider_diagnostics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String, nullable=False, default="openai")
    model: Mapped[str] = mapped_column(String, nullable=False, default="")
    endpoint: Mapped[str] = mapped_column(String, nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String, nullable=False, default="failure")
    failure_category: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    failure_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    response_status: Mapped[Optional[int]] = mapped_column(nullable=True)
    request_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    elapsed_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    response_excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

class AccountVerificationToken(Base):
    __tablename__ = "account_verification_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
