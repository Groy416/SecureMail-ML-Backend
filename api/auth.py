"""Password and JWT session primitives for user authentication."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import AuthSession, User
from api.dependencies import get_db

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, _unb64(expected))
    except (ValueError, TypeError):
        return False


def allowed_email(email: str) -> bool:
    normalized = email.strip().lower()
    domains = {item.strip().lower().lstrip("@").rstrip(".") for item in os.environ.get("AUTH_ALLOWED_EMAIL_DOMAINS", settings.AUTH_ALLOWED_EMAIL_DOMAINS).split(",") if item.strip()}
    return bool(domains) and normalized.rsplit("@", 1)[-1] in domains and "@" in normalized


def _secret() -> str:
    secret = os.environ.get("AUTH_JWT_SECRET") or settings.AUTH_JWT_SECRET
    if not secret:
        raise RuntimeError("AUTH_JWT_SECRET is required")
    return secret


def create_access_token(email: str, *, expires_delta: timedelta | None = None, jti: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(minutes=int(os.environ.get("AUTH_ACCESS_TOKEN_MINUTES", settings.AUTH_ACCESS_TOKEN_MINUTES))))
    return jwt.encode({"sub": email, "jti": jti or str(uuid.uuid4()), "iat": now, "exp": expires}, _secret(), algorithm="HS256")


async def issue_access_token(db: AsyncSession, user: User) -> str:
    jti = str(uuid.uuid4())
    minutes = int(os.environ.get("AUTH_ACCESS_TOKEN_MINUTES", settings.AUTH_ACCESS_TOKEN_MINUTES))
    expires = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    db.add(AuthSession(user_id=user.id, jti=jti, expires_at=expires))
    return create_access_token(user.email, expires_delta=timedelta(minutes=minutes), jti=jti)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="authentication_required")
    try:
        payload = jwt.decode(authorization[7:], _secret(), algorithms=["HS256"])
        jti = payload["jti"]
        email = payload["sub"]
    except (jwt.PyJWTError, KeyError, RuntimeError):
        raise HTTPException(status_code=401, detail="invalid_token")
    session = await db.scalar(select(AuthSession).where(AuthSession.jti == jti))
    user = await db.scalar(select(User).where(User.email == email))
    if session is None or session.revoked_at is not None or _as_utc(session.expires_at) <= datetime.now(timezone.utc) or user is None or session.user_id != user.id:
        raise HTTPException(status_code=401, detail="invalid_token")
    return user


async def revoke_access_token(authorization: str | None, db: AsyncSession) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        return
    try:
        payload = jwt.decode(authorization[7:], _secret(), algorithms=["HS256"], options={"verify_exp": False})
        session = await db.scalar(select(AuthSession).where(AuthSession.jti == payload["jti"]))
        if session is not None and session.revoked_at is None:
            session.revoked_at = datetime.now(timezone.utc)
    except (jwt.PyJWTError, KeyError, RuntimeError):
        return
