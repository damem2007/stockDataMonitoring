from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


load_dotenv()


class Base(DeclarativeBase):
    pass


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def database_configured() -> bool:
    return bool(database_url())


def database_available() -> bool:
    return database_configured()


def database_status() -> str:
    if not database_url():
        return "DATABASE_URL is not set; using local JSON storage."
    return "DATABASE_URL is set; Supabase/Postgres storage will be used when reachable."


def _engine():
    url = database_url()
    if not url:
        raise RuntimeError(database_status())
    return create_engine(url, pool_pre_ping=True, future=True)


engine = _engine() if database_url() else None
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True) if engine else None


@contextmanager
def session_scope() -> Iterator[Session]:
    if SessionLocal is None:
        raise RuntimeError(database_status())
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_database(strict: bool = False) -> bool:
    if engine is None:
        return False
    try:
        from backend.app.models.orm import AppUser, InstrumentTag, PriceAlert  # noqa: F401

        with engine.begin() as conn:
            conn.execute(text("create extension if not exists pgcrypto"))
            Base.metadata.create_all(bind=conn)
        return True
    except SQLAlchemyError:
        if strict:
            raise
        return False


def list_user_ids_with_instruments() -> list[str]:
    if not ensure_database():
        return []
    from backend.app.models.orm import InstrumentTag

    with session_scope() as session:
        rows = session.query(InstrumentTag.user_id).filter(InstrumentTag.active.is_(True)).distinct().all()
        return [str(row[0]) for row in rows]
