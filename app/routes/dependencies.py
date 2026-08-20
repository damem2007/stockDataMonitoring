from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from backend.app.services.auth import AuthUser, decode_access_token


def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[AuthUser]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return decode_access_token(token)


def require_user(user: Optional[AuthUser] = Depends(optional_user)) -> AuthUser:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in is required for portfolio monitoring.",
        )
    return user
