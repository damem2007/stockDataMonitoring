from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import or_, select

from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import AppUser


load_dotenv()

PASSWORD_ITERATIONS = 210_000


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    username: str
    name: str
    role: str = "user"

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "name": self.name,
            "role": self.role,
        }


def authenticate_user(login: str, password: str) -> AuthUser | None:
    login = login.strip().lower()
    if not login or not password:
        return None
    if database_available() and ensure_database():
        bootstrap_local_root_user()
        user = find_user(login)
        if user and verify_password(password, str(user["password_hash"])):
            return AuthUser(
                id=str(user["id"]),
                email=str(user["email"]),
                username=str(user["username"]),
                name=str(user["full_name"] or user["username"]),
                role=str(user["role"] or "user"),
            )
        return None
    return authenticate_local_root(login, password)


def default_storage_user_id() -> str | None:
    login = os.getenv("LOCAL_ROOT_EMAIL", "").strip().lower() or os.getenv("LOCAL_ROOT_USERNAME", "").strip().lower()
    if not login or not (database_available() and ensure_database()):
        return None
    bootstrap_local_root_user()
    user = find_user(login)
    return str(user["id"]) if user else None


def bootstrap_local_root_user() -> None:
    email = os.getenv("LOCAL_ROOT_EMAIL", "").strip().lower()
    username = os.getenv("LOCAL_ROOT_USERNAME", "").strip().lower()
    password = os.getenv("LOCAL_ROOT_PASSWORD", "")
    name = os.getenv("LOCAL_ROOT_NAME", "Local Root Administrator").strip()
    if not (email and username and password and database_available()):
        return
    password_hash = hash_password(password)
    with session_scope() as session:
        existing = session.execute(
            select(AppUser).where(or_(AppUser.email == email, AppUser.username == username))
        ).scalar_one_or_none()
        if existing:
            existing.email = email
            existing.username = username
            existing.full_name = name
            existing.password_hash = password_hash
            existing.role = "superadmin"
        else:
            session.add(
                AppUser(
                    email=email,
                    username=username,
                    full_name=name,
                    password_hash=password_hash,
                    role="superadmin",
                )
            )


def find_user(login: str) -> dict[str, Any] | None:
    with session_scope() as session:
        user = session.execute(
            select(AppUser).where(or_(AppUser.email.ilike(login), AppUser.username.ilike(login))).limit(1)
        ).scalar_one_or_none()
        if not user:
            return None
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "password_hash": user.password_hash,
            "role": user.role,
        }


def authenticate_local_root(login: str, password: str) -> AuthUser | None:
    email = os.getenv("LOCAL_ROOT_EMAIL", "").strip().lower()
    username = os.getenv("LOCAL_ROOT_USERNAME", "").strip().lower()
    expected_password = os.getenv("LOCAL_ROOT_PASSWORD", "")
    if not (expected_password and (login == email or login == username)):
        return None
    if not hmac.compare_digest(password, expected_password):
        return None
    return AuthUser(
        id="local-root",
        email=email or "root@local",
        username=username or "root",
        name=os.getenv("LOCAL_ROOT_NAME", "Local Root Administrator"),
        role="superadmin",
    )


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    parts = stored_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    _, raw_iterations, salt, expected = parts
    try:
        iterations = int(raw_iterations)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(digest, expected)


def token_expiry() -> datetime:
    minutes = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "60") or "60")
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "email": user.email,
        "username": user.username,
        "name": user.name,
        "role": user.role,
        "iss": os.getenv("JWT_ISSUER", "stock-signal-local"),
        "aud": os.getenv("JWT_AUDIENCE", "stock-signal-api"),
        "iat": int(now.timestamp()),
        "exp": int(token_expiry().timestamp()),
    }
    header = {"alg": os.getenv("JWT_ALGORITHM", "HS256"), "typ": "JWT"}
    signing_input = ".".join([_b64_json(header), _b64_json(payload)])
    signature = _sign(signing_input)
    return f"{signing_input}.{signature}"


def decode_access_token(token: str) -> AuthUser | None:
    try:
        header_raw, payload_raw, signature = token.split(".")
    except ValueError:
        return None
    signing_input = f"{header_raw}.{payload_raw}"
    if not hmac.compare_digest(_sign(signing_input), signature):
        return None
    try:
        payload = json.loads(_b64_decode(payload_raw))
    except Exception:
        return None
    if payload.get("iss") != os.getenv("JWT_ISSUER", "stock-signal-local"):
        return None
    if payload.get("aud") != os.getenv("JWT_AUDIENCE", "stock-signal-api"):
        return None
    try:
        if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            return None
    except (TypeError, ValueError):
        return None
    return AuthUser(
        id=str(payload.get("sub") or ""),
        email=str(payload.get("email") or ""),
        username=str(payload.get("username") or ""),
        name=str(payload.get("name") or payload.get("username") or ""),
        role=str(payload.get("role") or "user"),
    )


def _b64_json(payload: dict[str, object]) -> str:
    return _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(value: str) -> str:
    secret = os.getenv("JWT_SECRET", "dev-only-change-me").encode("utf-8")
    digest = hmac.new(secret, value.encode("utf-8"), hashlib.sha256).digest()
    return _b64_encode(digest)
