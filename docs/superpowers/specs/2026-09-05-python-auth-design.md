# Native Python Authentication Design

## Objective
Add native FastAPI authentication for enterprise users with JWT access tokens, login/logout, and profile APIs while preserving the existing API-key protection for ML/data endpoints.

## Configuration
- `AUTH_ALLOWED_EMAIL_DOMAINS`: required comma-separated domain allowlist, e.g. `company.com,subsidiary.com`.
- `AUTH_JWT_SECRET`: required signing secret.
- `AUTH_ACCESS_TOKEN_MINUTES`: access-token lifetime, default 30.

Emails are normalized to lowercase and accepted only when their domain is in the configured allowlist.

## Data model
- `users`: unique normalized email, password hash, display name, timestamps.
- `auth_sessions`: user ID, JWT `jti`, expiry, optional revocation timestamp.

Passwords are never returned or logged. JWT sessions are persisted so logout can revoke the current token.

## API contract
- `POST /api/v1/auth/register`: create an allowlisted user and return a bearer token plus profile.
- `POST /api/v1/auth/login`: authenticate credentials and return a bearer token plus profile.
- `POST /api/v1/auth/logout`: revoke the current bearer session; repeated logout is safe.
- `GET /api/v1/auth/profile`: return the current user's profile.
- `PATCH /api/v1/auth/profile`: update the current user's display name.

Invalid credentials, invalid tokens, expired tokens, and revoked tokens return `401` without revealing whether an account exists. Disallowed domains return `422`.

## Verification
Tests cover domain enforcement, registration/login, profile access/update, logout revocation, malformed/expired tokens, and regression coverage for existing API-key auth and API tests.

## Scope boundary
No email delivery, password reset, refresh tokens, roles, MFA, or external Better Auth service are included.
