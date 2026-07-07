"""Auth routes (INFRA-1): register / login / refresh / logout.

Semantics per the approved design (Sec 6):
- registration is invite-gated (403 on bad invite), 409 on duplicate email;
- login returns a short-lived JWT access token in the body plus a rotating
  refresh token in an httpOnly SameSite=Lax Secure cookie;
- refresh rotates the token (old doc gets `rotated_at`, new doc records
  `rotated_from`); presenting an already-rotated token is treated as theft and
  revokes the user's whole chain;
- logout revokes the chain and clears the cookie.

All endpoints are plain `def` (sync pymongo, small JSON bodies).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError
from slowapi import Limiter

from ..security import (
    REFRESH_COOKIE_NAME,
    AuthConfig,
    encode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)

REGISTER_RATE_LIMIT = "5/hour"
LOGIN_RATE_LIMIT = "10/minute"


class RegisterBody(BaseModel):
    email: str
    password: str
    invite_code: str


class LoginBody(BaseModel):
    email: str
    password: str
    device_label: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Mongo round-trips datetimes as naive UTC; normalize before comparing."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _normalized_email(raw: str) -> str:
    email = raw.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=422, detail="invalid email address")
    return email


def build_auth_router(*, db: Any, auth_config: AuthConfig, limiter: Limiter) -> APIRouter:
    router = APIRouter()

    def _set_refresh_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            REFRESH_COOKIE_NAME,
            token,
            max_age=auth_config.jwt_refresh_ttl_days * 86400,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/api/auth",
        )

    def _issue_session(response: Response, *, user_id: str, rotated_from: str | None, device_label: str | None) -> dict[str, Any]:
        now = _utcnow()
        refresh_token = generate_refresh_token()
        db.refresh_tokens.insert_one(
            {
                "token_hash": hash_refresh_token(refresh_token),
                "user_id": user_id,
                "device_label": device_label,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + timedelta(days=auth_config.jwt_refresh_ttl_days),
                "rotated_from": rotated_from,
                "rotated_at": None,
                "revoked_at": None,
            }
        )
        _set_refresh_cookie(response, refresh_token)
        access_token = encode_access_token(
            user_id=user_id,
            secret=auth_config.jwt_secret,
            ttl_s=auth_config.jwt_access_ttl_s,
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": auth_config.jwt_access_ttl_s,
        }

    def _revoke_user_chain(user_id: str) -> None:
        now = _utcnow()
        db.refresh_tokens.update_many(
            {"user_id": user_id, "revoked_at": None},
            {"$set": {"revoked_at": now, "updated_at": now}},
        )

    @router.post("/api/auth/register", status_code=201)
    @limiter.limit(REGISTER_RATE_LIMIT)
    def register(request: Request, body: RegisterBody) -> dict[str, Any]:
        if not auth_config.invite_code or body.invite_code != auth_config.invite_code:
            raise HTTPException(status_code=403, detail="invalid invite code")
        email = _normalized_email(body.email)
        if len(body.password) < 8:
            raise HTTPException(status_code=422, detail="password must be at least 8 characters")
        now = _utcnow()
        user_id = f"user_{uuid.uuid4().hex[:16]}"
        try:
            db.users.insert_one(
                {
                    "_id": user_id,
                    "email": email,
                    "password_hash": hash_password(body.password),
                    "created_at": now,
                    "updated_at": now,
                    "last_login_at": None,
                    "stripe_customer_id": None,
                    "deleted_at": None,
                }
            )
        except DuplicateKeyError:
            raise HTTPException(status_code=409, detail="email already registered") from None
        return {"id": user_id, "email": email}

    @router.post("/api/auth/login")
    @limiter.limit(LOGIN_RATE_LIMIT)
    def login(request: Request, response: Response, body: LoginBody) -> dict[str, Any]:
        email = _normalized_email(body.email)
        user = db.users.find_one({"email": email, "deleted_at": None})
        if user is None or not verify_password(user["password_hash"], body.password):
            raise HTTPException(status_code=401, detail="invalid email or password")
        now = _utcnow()
        db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login_at": now, "updated_at": now}},
        )
        return _issue_session(
            response,
            user_id=user["_id"],
            rotated_from=None,
            device_label=body.device_label,
        )

    @router.post("/api/auth/refresh")
    def refresh(
        response: Response,
        refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
    ) -> dict[str, Any]:
        if not refresh_token:
            raise HTTPException(status_code=401, detail="missing refresh token")
        token_hash = hash_refresh_token(refresh_token)
        now = _utcnow()
        claimed = db.refresh_tokens.find_one_and_update(
            {
                "token_hash": token_hash,
                "rotated_at": None,
                "revoked_at": None,
                "expires_at": {"$gt": now},
            },
            {"$set": {"rotated_at": now, "updated_at": now}},
        )
        if claimed is None:
            stale = db.refresh_tokens.find_one({"token_hash": token_hash})
            if (
                stale is not None
                and stale.get("rotated_at") is not None
                and stale.get("revoked_at") is None
            ):
                # A token that was already rotated is being replayed: assume
                # theft and revoke this user's entire chain (design Sec 6).
                _revoke_user_chain(stale["user_id"])
                raise HTTPException(
                    status_code=401,
                    detail="refresh token reuse detected; all sessions revoked",
                )
            raise HTTPException(status_code=401, detail="invalid refresh token")
        return _issue_session(
            response,
            user_id=claimed["user_id"],
            rotated_from=token_hash,
            device_label=claimed.get("device_label"),
        )

    @router.post("/api/auth/logout", status_code=204)
    def logout(
        response: Response,
        refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
    ) -> None:
        if refresh_token:
            doc = db.refresh_tokens.find_one({"token_hash": hash_refresh_token(refresh_token)})
            if doc is not None:
                _revoke_user_chain(doc["user_id"])
        response.delete_cookie(REFRESH_COOKIE_NAME, path="/api/auth")
        return None

    return router
