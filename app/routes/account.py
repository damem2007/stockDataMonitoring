from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.routes.dependencies import require_permission, require_user
from app.schemas.auth import PasswordUpdateRequest, ProfileUpdateRequest
from app.services.auth import AuthUser, change_user_password, create_access_token, update_user_profile
from app.services.market_data.public_holidays import list_public_holidays, sync_public_holidays
from app.config import PUBLIC_HOLIDAY_COUNTRIES


router = APIRouter(prefix="/api/account", tags=["account"])


@router.get("")
def account(user: AuthUser = Depends(require_user)) -> dict[str, object]:
    return {"user": user.as_dict()}


@router.patch("/profile")
def update_profile(
    payload: ProfileUpdateRequest,
    user: AuthUser = Depends(require_permission("update_account_profile")),
) -> dict[str, object]:
    try:
        updated = update_user_profile(user.id, email=payload.email, username=payload.username, full_name=payload.full_name, phone_number=payload.phone_number)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"user": updated.as_dict(), "access_token": create_access_token(updated), "token_type": "bearer"}


@router.patch("/security/password")
def update_password(
    payload: PasswordUpdateRequest,
    user: AuthUser = Depends(require_permission("update_password")),
) -> dict[str, str]:
    try:
        change_user_password(user.id, payload.current_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"status": "password_updated"}


@router.get("/subscription")
def subscription(user: AuthUser = Depends(require_permission("view_subscription"))) -> dict[str, object]:
    return {
        "subscription": {
            "plan": "premium",
            "status": "active",
            "features": ["portfolio_dashboard", "realtime_analysis", "buy_sell_indicators", "invoices"],
            "owner": user.id,
        }
    }


@router.get("/invoices")
def invoices(user: AuthUser = Depends(require_permission("view_invoices"))) -> dict[str, object]:
    return {"invoices": [], "owner": user.id}


@router.get("/public-holidays")
def public_holidays(
    year: Optional[int] = Query(None, ge=2000, le=2200),
    country: Optional[str] = Query(None, min_length=2, max_length=2),
    user: AuthUser = Depends(require_user),
) -> dict[str, object]:
    del user
    selected_year = year or date.today().year
    selected_countries = (country.upper(),) if country else PUBLIC_HOLIDAY_COUNTRIES
    rows = list_public_holidays(selected_year, selected_countries)
    if not rows:
        sync_public_holidays(countries=selected_countries, years=(selected_year,))
        rows = list_public_holidays(selected_year, selected_countries)
    return {"year": selected_year, "countries": list(selected_countries), "holidays": rows}


@router.post("/public-holidays/sync")
def sync_public_holidays_route(year: Optional[int] = Query(None, ge=2000, le=2200), user: AuthUser = Depends(require_user)) -> dict[str, object]:
    del user
    selected_year = year or date.today().year
    return sync_public_holidays(years=(selected_year,))
