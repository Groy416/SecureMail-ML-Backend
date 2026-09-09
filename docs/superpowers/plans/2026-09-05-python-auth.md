# Native Python Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JWT login/logout/profile APIs with configured enterprise-email enforcement, without registration routes.

**Architecture:** Add `users` and `auth_sessions` SQLAlchemy models. A focused auth module will hash passwords, issue/verify JWTs, and revoke session JTIs; auth routes will consume these contracts. Existing API-key dependencies and ML/data routes remain unchanged.

**Tech Stack:** FastAPI, SQLAlchemy async, Pydantic, PyJWT, stdlib `hashlib.scrypt`.

**Spec:** `docs/superpowers/specs/2026-09-05-python-auth-design.md`

## Global Constraints

- `AUTH_ALLOWED_EMAIL_DOMAINS` is a comma-separated enterprise domain allowlist.
- `AUTH_JWT_SECRET` is required for auth routes.
- Passwords and tokens are never logged or returned as password fields.
- Registration, email delivery, refresh tokens, roles, and MFA are out of scope.

---

### Task 1: Auth contracts and persistence

**Files:**
- Modify: `api/config.py`
- Modify: `api/database.py`
- Create: `api/auth.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- `authenticate_user(db, email, password) -> User | None`
- `issue_access_token(db, user) -> str`
- `get_current_user(authorization, db) -> User`
- `revoke_access_token(authorization, db) -> None`

- [ ] Write failing unit/API tests for missing configuration, password verification, and token revocation.
- [ ] Run `.venv/bin/pytest tests/test_auth.py -v` and confirm failure because auth contracts are absent.
- [ ] Add settings, `User` and `AuthSession` models, scrypt password hashing, PyJWT encode/decode, and persisted JTI checks.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the persistence/auth contract change.

### Task 2: Auth routes and API schemas

**Files:**
- Modify: `api/schemas.py`
- Create: `api/auth_routes.py`
- Modify: `api/app.py`
- Modify: `.env.example`
- Test: `tests/test_auth.py`

**Interfaces:**
- `POST /api/v1/auth/login` accepts `{email, password}` and returns `{access_token, token_type, profile}`.
- `POST /api/v1/auth/logout` revokes the current bearer token and returns `{logged_out: true}`.
- `GET /api/v1/auth/profile` returns the current profile.
- `PATCH /api/v1/auth/profile` accepts `{display_name}` and returns the updated profile.

- [ ] Add failing endpoint tests using a seeded user fixture for login, profile, update, logout, disallowed email domains, invalid credentials, and malformed/revoked tokens.
- [ ] Run the focused endpoint tests and confirm expected failures.
- [ ] Implement the four routes and register the router; require `AUTH_ALLOWED_EMAIL_DOMAINS` for login policy and omit registration.
- [ ] Run the focused tests and the existing API authentication tests.
- [ ] Commit the route/config change.

### Task 3: Migration and regression verification

**Files:**
- Create: `migrations/versions/<generated-auth-migration>.py`
- Modify: `tests/test_auth.py` only if fixture isolation needs adjustment.

- [ ] Generate an Alembic migration from the two auth tables and inspect its upgrade/downgrade operations.
- [ ] Run `.venv/bin/pytest tests/test_auth.py tests/test_api.py -q`.
- [ ] Run the full `.venv/bin/pytest -q` suite.
- [ ] Inspect `git diff` and confirm no secrets or unrelated files changed.
- [ ] Commit the migration and verification-ready change.
