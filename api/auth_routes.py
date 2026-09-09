"""Login, logout, and profile endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import allowed_email, get_current_user, hash_password, issue_access_token, revoke_access_token, verify_password
from api.database import User
from api.dependencies import get_db
from api.schemas import AuthLoginRequest, AuthRegisterRequest, AuthResponse, LogoutResponse, ProfileResponse, ProfileUpdateRequest

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _profile(user: User) -> ProfileResponse:
    return ProfileResponse(id=user.id, email=user.email, display_name=user.display_name)


@router.post("/register", response_model=AuthResponse)
async def register(body: AuthRegisterRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    email = body.email.strip().lower()
    if not allowed_email(email):
        raise HTTPException(status_code=422, detail="enterprise_email_required")
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="user_already_exists")
    display_name = (body.display_name or email.split("@")[0]).strip()
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        display_name=display_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    token = await issue_access_token(db, user)
    return AuthResponse(access_token=token, profile=_profile(user))


@router.post("/login", response_model=AuthResponse)
async def login(body: AuthLoginRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    email = body.email.strip().lower()
    if not allowed_email(email):
        raise HTTPException(status_code=422, detail="enterprise_email_required")
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = await issue_access_token(db, user)
    return AuthResponse(access_token=token, profile=_profile(user))


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> LogoutResponse:
    await revoke_access_token(authorization, db)
    return LogoutResponse(logged_out=True)


@router.get("/profile", response_model=ProfileResponse)
async def profile(user: User = Depends(get_current_user)) -> ProfileResponse:
    return _profile(user)


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(
    body: ProfileUpdateRequest,
    user: User = Depends(get_current_user),
) -> ProfileResponse:
    user.display_name = body.display_name.strip()
    return _profile(user)
