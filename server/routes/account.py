"""Account routes (INFRA-1): stub only.

`DELETE /api/account` returns 501 until INFRA-5 lands the full delete-cascade
(user tombstone + S3 prefix deletes + derived-doc drops, design Sec 4). It is
still JWT-gated so the auth surface matches the final endpoint from day one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..security import AuthConfig, bearer_auth_dependency


def build_account_router(*, auth_config: AuthConfig) -> APIRouter:
    router = APIRouter()
    require_user = bearer_auth_dependency(auth_config)

    @router.delete("/api/account")
    def delete_account(user_id: str = Depends(require_user)) -> None:
        raise HTTPException(
            status_code=501,
            detail="account deletion (delete-cascade) ships in INFRA-5",
        )

    return router
