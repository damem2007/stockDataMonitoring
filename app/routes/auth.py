from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.routes.dependencies import require_user
from backend.app.schemas.auth import LoginRequest, TokenResponse
from backend.app.services.auth import AuthUser, authenticate_user, create_access_token


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(payload.login, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username/email or password.")
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.get("/me")
def me(user: AuthUser = Depends(require_user)) -> dict[str, str]:
    return user.as_dict()
