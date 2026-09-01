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

from backend.app.services.email_service import send_local_onboarding
from dotenv import load_dotenv
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database import database_available, ensure_database, session_scope
from backend.app.models.orm import AccountVerificationToken, AppUser
from backend.app.services.rbac import normalize_role, permissions_for_role


load_dotenv()

PASSWORD_ITERATIONS = 210_000


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    username: str
    name: str
    role: str = "user"
    permissions: tuple[str, ...] = ()
    onboarding_status: str = "pending"
    status: str = "active"
    is_guest: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "name": self.name,
            "role": self.role,
            "permissions": list(self.permissions or sorted(permissions_for_role(self.role))),
            "onboardingStatus": self.onboarding_status,
            "status": self.status,
            "isGuest": self.is_guest,
        }

    def has_permission(self, permission: str) -> bool:
        return permission in set(self.permissions or sorted(permissions_for_role(self.role)))


def authenticate_user(login: str, password: str) -> AuthUser | None:
    login = login.strip().lower()
    if not login or not password:
        return None
    if database_available() and ensure_database():
        bootstrap_local_root_user()
        user = find_user(login)
        if user and verify_password(password, str(user["password_hash"])):
            return auth_user_from_row(user)
        return None
    return authenticate_local_root(login, password)


def register_user(email: str, username: str, full_name: str, password: str,  role: str | None = None) -> AuthUser:
    if not database_available() or not ensure_database():
        raise RuntimeError("Database storage is required for account registration.")
    normalized_email = email.strip().lower()
    normalized_username = username.strip().lower()
    normalized_role = "user" if role is None or role.strip().lower() == "" else role.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise ValueError("A valid email is required.")
    if not normalized_username:
        raise ValueError("Username is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with session_scope() as session:
        exists = session.execute(
            select(AppUser).where(or_(AppUser.email == normalized_email, AppUser.username == normalized_username))
        ).scalar_one_or_none()
        if exists:
            raise ValueError("An account with that email or username already exists.")
        user = AppUser(
            email=normalized_email,
            username=normalized_username,
            full_name=full_name.strip() or normalized_username,
            password_hash=hash_password(password),
            role=normalized_role,
            onboarding_status="pending",
            status="active",
        )
        session.add(user)
        try:
            session.flush()
            session.refresh(user)
        except IntegrityError as exc:
            raise ValueError("An account with that email or username already exists.") from exc
        if normalized_role == "user":
            verify_account = create_account_verification_token(session, user) #if request.auth_source == "local" else None
            frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8520")
            delivered = send_local_onboarding(user.email, user.username, f"{frontend_url}/account-verification?token={verify_account}")
            user.onboarding_status = "email_sent" if delivered else "email_pending_configuration"
        return auth_user_from_model(user)


def request_account_verification_link(login: str | None = None, raw_token: str | None = None) -> bool:
    if not database_available() or not ensure_database():
        raise RuntimeError("Database storage is required for account verification.")
    normalized_login = (login or "").strip().lower()
    normalized_token = (raw_token or "").strip()
    if not normalized_login and not normalized_token:
        raise ValueError("Verification token or email/username is required.")
    with session_scope() as session:
        user: AppUser | None = None
        if normalized_token:
            token_hash = hashlib.sha256(normalized_token.encode()).hexdigest()
            token = session.execute(
                select(AccountVerificationToken).where(AccountVerificationToken.token_hash == token_hash).limit(1)
            ).scalar_one_or_none()
            if token:
                user = session.get(AppUser, token.user_id)
                if not user:
                    raise ValueError("This verification link no longer matches an active account.")
            elif not normalized_login:
                raise ValueError("Verification link is invalid. Enter your email or username to request a new link.")

        if user is None and normalized_login:
            user = session.execute(
                select(AppUser).where(or_(AppUser.email.ilike(normalized_login), AppUser.username.ilike(normalized_login))).limit(1)
            ).scalar_one_or_none()
        if not user:
            return False
        if str(user.onboarding_status or "").lower() == "verified":
            return True
        verify_account = create_account_verification_token(session, user)
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8520")
        delivered = send_local_onboarding(user.email, user.username, f"{frontend_url}/account-verification?token={verify_account}")
        user.onboarding_status = "email_sent" if delivered else "email_pending_configuration"
        return delivered


def update_user_profile(user_id: str, email: str | None = None, username: str | None = None, full_name: str | None = None) -> AuthUser:
    if not database_available() or not ensure_database():
        raise RuntimeError("Database storage is required for account management.")
    with session_scope() as session:
        user = session.get(AppUser, _uuid(user_id))
        if not user:
            raise ValueError("Account was not found.")
        if email is not None:
            normalized_email = email.strip().lower()
            if not normalized_email or "@" not in normalized_email:
                raise ValueError("A valid email is required.")
            user.email = normalized_email
        if username is not None:
            normalized_username = username.strip().lower()
            if not normalized_username:
                raise ValueError("Username is required.")
            user.username = normalized_username
        if full_name is not None:
            user.full_name = full_name.strip()
        try:
            session.flush()
        except IntegrityError as exc:
            raise ValueError("An account with that email or username already exists.") from exc
        return auth_user_from_model(user)


def change_user_password(user_id: str, current_password: str, new_password: str) -> None:
    if not database_available() or not ensure_database():
        raise RuntimeError("Database storage is required for account management.")
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters.")
    with session_scope() as session:
        user = session.get(AppUser, _uuid(user_id))
        if not user:
            raise ValueError("Account was not found.")
        if not verify_password(current_password, user.password_hash):
            raise ValueError("Current password is incorrect.")
        user.password_hash = hash_password(new_password)


def create_guest_user() -> AuthUser:
    guest_id = f"guest-{secrets.token_urlsafe(12)}"
    return AuthUser(
        id=guest_id,
        email="guest@local",
        username="guest",
        name="Guest",
        role="guest",
        permissions=tuple(sorted(permissions_for_role("guest"))),
        is_guest=True,
    )


def bootstrap_local_root_user() -> None:
    email = os.getenv("LOCAL_ROOT_EMAIL", "root@stockdata.local").strip().lower()
    username = os.getenv("LOCAL_ROOT_USERNAME", "root").strip().lower()
    password = os.getenv("LOCAL_ROOT_PASSWORD", "")
    name = os.getenv("LOCAL_ROOT_NAME", "Local Root Administrator").strip()
    if not (email and username and password and database_available()):
        return
    with session_scope() as session:
        existing = session.execute(
            select(AppUser).where(or_(AppUser.email == email, AppUser.username == username))
        ).scalar_one_or_none()
        if existing:
            existing.email = email
            existing.username = username
            existing.full_name = name
            existing.password_hash = hash_password(password)
            existing.role = "superadmin"
        else:
            session.add(
                AppUser(
                    email=email,
                    username=username,
                    full_name=name,
                    password_hash=hash_password(password),
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
            "onboarding_status": user.onboarding_status,
            "status": user.status,
        }


def authenticate_local_root(login: str, password: str) -> AuthUser | None:
    email = os.getenv("LOCAL_ROOT_EMAIL", "root@stockdata.local").strip().lower()
    username = os.getenv("LOCAL_ROOT_USERNAME", "root").strip().lower()
    expected_password = os.getenv("LOCAL_ROOT_PASSWORD", "")
    if not (expected_password and (login == email or login == username)):
        return None
    if not hmac.compare_digest(password, expected_password):
        return None
    return AuthUser(
        id="local-root",
        email=email,
        username=username,
        name=os.getenv("LOCAL_ROOT_NAME", "Local Root Administrator"),
        role="superadmin",
        permissions=tuple(sorted(permissions_for_role("superadmin"))),
    )


def auth_user_from_row(row: dict[str, Any]) -> AuthUser:
    role = normalize_role(str(row.get("role") or "user"))
    return AuthUser(
        id=str(row["id"]),
        email=str(row["email"]),
        username=str(row["username"]),
        name=str(row.get("full_name") or row["username"]),
        role=role,
        permissions=tuple(sorted(permissions_for_role(role))),
        onboarding_status=str(row.get("onboarding_status") or "pending"),
        status=str(row.get("status") or "active"),
    )


def auth_user_from_model(user: AppUser) -> AuthUser:
    return auth_user_from_row(
        {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "onboarding_status": user.onboarding_status,
            "status": user.status,
        }
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


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=token_minutes(user))
    payload = {
        "sub": user.id,
        "email": user.email,
        "username": user.username,
        "name": user.name,
        "role": user.role,
        "permissions": list(user.permissions or sorted(permissions_for_role(user.role))),
        "guest": user.is_guest,
        "iss": os.getenv("JWT_ISSUER", "stock-signal-local"),
        "aud": os.getenv("JWT_AUDIENCE", "stock-signal-api"),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_urlsafe(16),
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
        now = int(datetime.now(timezone.utc).timestamp())
        if int(payload.get("nbf", 0)) > now or int(payload.get("exp", 0)) < now:
            return None
    except (TypeError, ValueError):
        return None
    role = normalize_role(str(payload.get("role") or "guest"))
    token_permissions = payload.get("permissions")
    permissions = token_permissions if isinstance(token_permissions, list) else sorted(permissions_for_role(role))
    return AuthUser(
        id=str(payload.get("sub") or ""),
        email=str(payload.get("email") or ""),
        username=str(payload.get("username") or ""),
        name=str(payload.get("name") or payload.get("username") or ""),
        role=role,
        permissions=tuple(str(item) for item in permissions),
        is_guest=bool(payload.get("guest") or role == "guest"),
    )


def token_minutes(user: AuthUser) -> int:
    if user.role == "guest":
        return max(5, int(os.getenv("JWT_GUEST_TOKEN_MINUTES", "120")))
    return max(5, int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "60")))


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


def _uuid(value: object):
    import uuid

    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))

def create_account_verification_token(db: Session, user: AppUser, hours: int = 24) -> str:
    now = datetime.now(timezone.utc)
    db.query(AccountVerificationToken).filter(
        AccountVerificationToken.user_id == user.id,
        AccountVerificationToken.used_at.is_(None),
    ).update({AccountVerificationToken.used_at: now}, synchronize_session=False)
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    db.add(
        AccountVerificationToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=now + timedelta(hours=hours),
        )
    )
    return raw_token


def verify_account_token(raw_token: str) -> AuthUser:
    if not raw_token.strip():
        raise ValueError("Verification token is required.")
    if not database_available() or not ensure_database():
        raise RuntimeError("Database storage is required for account verification.")
    token_hash = hashlib.sha256(raw_token.strip().encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with session_scope() as session:
        token = session.execute(
            select(AccountVerificationToken).where(
                AccountVerificationToken.token_hash == token_hash,
                AccountVerificationToken.used_at.is_(None),
            )
        ).scalar_one_or_none()
        if not token:
            raise ValueError("Verification link is invalid or has already been used.")
        expires_at = token.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < now:
            token.used_at = now
            raise ValueError("Verification link has expired.")
        user = session.get(AppUser, token.user_id)
        if not user:
            token.used_at = now
            raise ValueError("This verification link no longer matches an active account.")
        token.used_at = now
        user.onboarding_status = "verified"
        user.status = "active"
        session.flush()
        return auth_user_from_model(user)
