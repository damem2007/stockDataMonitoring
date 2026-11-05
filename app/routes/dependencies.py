from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.services.auth import AuthUser, decode_access_token
from app.services.rbac import role_has_permission


def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[AuthUser]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return decode_access_token(token)


def require_user(user: Optional[AuthUser] = Depends(optional_user)) -> AuthUser:
    if not user or user.is_guest:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in is required for portfolio monitoring.",
        )
    return user


def require_permission(permission: str):
    def dependency(user: AuthUser = Depends(require_user)) -> AuthUser:
        if not user.has_permission(permission) and not role_has_permission(user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}",
            )
        return user

    return dependency


def require_any_permission(*permissions: str):
    def dependency(user: AuthUser = Depends(require_user)) -> AuthUser:
        if not any(user.has_permission(permission) or role_has_permission(user.role, permission) for permission in permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One permission is required: {', '.join(permissions)}",
            )
        return user

    return dependency
