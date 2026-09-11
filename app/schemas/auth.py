from __future__ import annotations

from typing import Any
from typing import Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    login: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    username: str
    full_name: str = ""
    password: str


class VerificationLinkRequest(BaseModel):
    login: Optional[str] = None
    token: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None


class PasswordUpdateRequest(BaseModel):
    current_password: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]
