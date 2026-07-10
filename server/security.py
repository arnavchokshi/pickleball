"""Password hashing, JWT access tokens, and refresh-token primitives (INFRA-1).

Per the approved design (runs/archive/root_docs_20260709/PRODUCT_INFRA_DESIGN_20260707.md
Sec 6): argon2id password hashing, 15-min HS256 JWT access tokens, 30-day
rotating refresh tokens hashed at rest (sha256 hex) with reuse-of-a-rotated
token revoking the whole user's chain. The chain-revocation logic itself lives
in `server/routes/auth.py` next to the Mongo collection it operates on; this
module only holds the stateless crypto primitives, mirroring how
`server/gpu_runner.py` keeps `runner_from_env` config-only and execution
elsewhere.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Header, HTTPException

_HASHER = PasswordHasher()

REFRESH_COOKIE_NAME = "pickleball_refresh_token"


@dataclass(frozen=True)
class AuthConfig:
    jwt_secret: str
    jwt_access_ttl_s: int
    jwt_refresh_ttl_days: int
    invite_code: str
    accounts_enabled: bool


def auth_config_from_env(env: Mapping[str, str] | None = None) -> AuthConfig:
    resolved_env = os.environ if env is None else env
    return AuthConfig(
        jwt_secret=resolved_env.get("PICKLEBALL_JWT_SECRET", "").strip(),
        jwt_access_ttl_s=int(resolved_env.get("PICKLEBALL_JWT_ACCESS_TTL_S", "900")),
        jwt_refresh_ttl_days=int(resolved_env.get("PICKLEBALL_JWT_REFRESH_TTL_DAYS", "30")),
        invite_code=resolved_env.get("PICKLEBALL_INVITE_CODE", "").strip(),
        accounts_enabled=resolved_env.get("PICKLEBALL_ACCOUNTS_ENABLED", "0").strip() == "1",
    )


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:  # noqa: BLE001 - a malformed hash must fail closed, not 500
        return False


def needs_rehash(password_hash: str) -> bool:
    return _HASHER.check_needs_rehash(password_hash)


def encode_access_token(*, user_id: str, secret: str, ttl_s: int) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + ttl_s}
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, *, secret: str) -> dict[str, Any]:
    return jwt.decode(token, secret, algorithms=["HS256"])


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def bearer_auth_dependency(config: AuthConfig):
    """Build a FastAPI dependency that resolves the caller's user_id from a
    `Authorization: Bearer <jwt>` header, or raises 401.
    """

    def _require_user_id(authorization: str | None = Header(default=None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        try:
            payload = decode_access_token(token, secret=config.jwt_secret)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="invalid or expired token") from None
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(status_code=401, detail="invalid token subject")
        return str(subject)

    return _require_user_id
