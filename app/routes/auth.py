from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.routes.dependencies import require_user
from backend.app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from backend.app.services.auth import AuthUser, authenticate_user, create_access_token, create_guest_user, register_user


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(payload.login, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username/email or password.")
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.post("/guest", response_model=TokenResponse)
def guest_session() -> TokenResponse:
    user = create_guest_user()
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    try:
        user = register_user(payload.email, payload.username, payload.full_name, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.get("/me")
def me(user: AuthUser = Depends(require_user)) -> dict[str, object]:
    return user.as_dict()
