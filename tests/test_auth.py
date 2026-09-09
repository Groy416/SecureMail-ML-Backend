from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.auth import hash_password, verify_password
from api.database import AuthSession, User
from api.dependencies import AsyncSessionLocal, init_db


@pytest.fixture(scope="module")
def client():
    os.environ["AUTH_ALLOWED_EMAIL_DOMAINS"] = "example.com,enterprise.test"
    os.environ["AUTH_JWT_SECRET"] = "test-auth-secret-please-change-32-bytes"
    os.environ["AUTH_ACCESS_TOKEN_MINUTES"] = "30"
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    for key in ("AUTH_ALLOWED_EMAIL_DOMAINS", "AUTH_JWT_SECRET", "AUTH_ACCESS_TOKEN_MINUTES"):
        os.environ.pop(key, None)


@pytest.fixture(scope="module", autouse=True)
def seed_user():
    import asyncio

    async def seed():
        await init_db()
        async with AsyncSessionLocal() as db:
            db.add(User(email="alice@example.com", password_hash=hash_password("correct-password"), display_name="Alice"))
            await db.commit()

    asyncio.run(seed())
    yield


def test_password_hash_is_verified_without_storing_plaintext():
    hashed = hash_password("secret")
    assert hashed != "secret"
    assert verify_password("secret", hashed)
    assert not verify_password("wrong", hashed)


def test_login_rejects_non_enterprise_email(client: TestClient):
    response = client.post("/api/v1/auth/login", json={"email": "alice@outside.test", "password": "secret"})
    assert response.status_code == 422


def test_login_profile_update_and_logout_revoke_token(client: TestClient):
    login = client.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": "correct-password"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    profile = client.get("/api/v1/auth/profile", headers=headers)
    assert profile.status_code == 200
    assert profile.json() == {"id": profile.json()["id"], "email": "alice@example.com", "display_name": "Alice"}

    updated = client.patch("/api/v1/auth/profile", headers=headers, json={"display_name": "Alice Smith"})
    assert updated.status_code == 200
    assert updated.json()["display_name"] == "Alice Smith"

    logout = client.post("/api/v1/auth/logout", headers=headers)
    assert logout.status_code == 200
    assert logout.json() == {"logged_out": True}
    assert client.get("/api/v1/auth/profile", headers=headers).status_code == 401


def test_login_rejects_invalid_credentials_without_user_enumeration(client: TestClient):
    response = client.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_credentials"


def test_malformed_token_is_rejected(client: TestClient):
    response = client.get("/api/v1/auth/profile", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401


def test_expired_token_is_rejected(client: TestClient):
    from api.auth import create_access_token

    token = create_access_token("alice@example.com", expires_delta=timedelta(seconds=-1))
    response = client.get("/api/v1/auth/profile", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
