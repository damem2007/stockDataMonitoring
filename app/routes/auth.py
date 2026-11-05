from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.routes.dependencies import require_user
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, VerificationLinkRequest
from app.services.auth import (
    AuthUser,
    authenticate_user,
    create_access_token,
    create_guest_user,
    register_user,
    request_account_verification_link,
    verify_account_token,
)


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
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.get("/verify-account", response_model=TokenResponse)
def verify_account(token: str) -> TokenResponse:
    try:
        user = verify_account_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return TokenResponse(access_token=create_access_token(user), user=user.as_dict())


@router.post("/request-verification")
def request_verification(payload: VerificationLinkRequest) -> dict[str, str]:
    try:
        request_account_verification_link(payload.login, payload.token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {"status": "queued"}


@router.get("/me")
def me(user: AuthUser = Depends(require_user)) -> dict[str, object]:
    return user.as_dict()
