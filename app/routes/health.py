from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.app.database import database_status, ensure_database


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "storage": database_status(),
        "databaseReachable": ensure_database(),
    }
