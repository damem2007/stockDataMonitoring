from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import ensure_database
from backend.app.routes.alerts import router as alerts_router
from backend.app.routes.analysis import router as analysis_router
from backend.app.routes.auth import router as auth_router
from backend.app.routes.health import router as health_router
from backend.app.routes.markets import router as markets_router
from backend.app.routes.portfolio import router as portfolio_router
from backend.app.routes.workspace import router as workspace_router


load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_database()
    yield


def get_allowed_origins() -> list[str]:
    local_dev_origins = [
        "http://localhost:8520",
        "http://127.0.0.1:8520",
        "http://192.168.1.67:8520",
    ]
    configured_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys([*configured_origins, *local_dev_origins]))


app = FastAPI(
    title="Stock Signal Dashboard API",
    version="0.3.0",
    description="FastAPI backend for the TSX, NASDAQ, and NYSE watchlist and portfolio workflow.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(markets_router)
app.include_router(workspace_router)
app.include_router(portfolio_router)
app.include_router(analysis_router)
app.include_router(alerts_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Stock Signal Dashboard API Running"}
