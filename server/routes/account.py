"""Account routes (INFRA-5): the full delete-cascade.

`DELETE /api/account` (JWT + password re-confirm in the body): tombstones
the user, revokes the refresh-token chain, batch-deletes every S3 object the
user owns (`raw/{user_id}/...`, `artifacts/{job_id}/...` per job,
`bundles/{clip_id}/...` per clip), then drops the clips/jobs/entitlements/
profiles Mongo docs. Wrong password -> 403 with NOTHING mutated (the
password check runs before any write). Every step after the password check
is naturally idempotent -- Mongo `delete_many`/`update_many` and
`server.s3.delete_prefix` are all no-ops against data that is already gone
-- so a retried call against an already-deleted account is safe.

Design ref: `~/.claude/plans/replicated-forging-cascade.md` INFRA-5 / the
approved product-infra spec Sec 4.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ..profile_store import ProfileStore
from ..s3 import delete_prefix
from ..security import AuthConfig, bearer_auth_dependency, verify_password


class DeleteAccountBody(BaseModel):
    password: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_account_router(*, db: Any, s3_client: Any, bucket: str, auth_config: AuthConfig) -> APIRouter:
    router = APIRouter()
    require_user = bearer_auth_dependency(auth_config)
    profile_store = ProfileStore(db)

    @router.delete("/api/account", status_code=204, response_model=None)
    def delete_account(body: DeleteAccountBody, user_id: str = Depends(require_user)) -> Response:
        user = db.users.find_one({"_id": user_id})
        if user is None:
            raise HTTPException(status_code=404, detail="account not found")
        if not verify_password(user["password_hash"], body.password):
            raise HTTPException(status_code=403, detail="incorrect password")

        now = _utcnow()
        clip_ids = [clip["_id"] for clip in db.clips.find({"user_id": user_id})]
        job_ids = [job["_id"] for job in db.jobs.find({"user_id": user_id})]

        # Batched S3 delete over every prefix this user's data can live
        # under. `raw/{user_id}/...` covers ALL of the user's clips in one
        # prefix delete; artifacts/bundles are keyed by job_id/clip_id
        # respectively, so those need one delete_prefix call each.
        delete_prefix(s3_client, bucket=bucket, prefix=f"raw/{user_id}/")
        for job_id in job_ids:
            delete_prefix(s3_client, bucket=bucket, prefix=f"artifacts/{job_id}/")
        for clip_id in clip_ids:
            delete_prefix(s3_client, bucket=bucket, prefix=f"bundles/{clip_id}/")

        db.users.update_one({"_id": user_id}, {"$set": {"deleted_at": now, "updated_at": now}})
        db.refresh_tokens.update_many(
            {"user_id": user_id, "revoked_at": None},
            {"$set": {"revoked_at": now, "updated_at": now}},
        )
        db.clips.delete_many({"user_id": user_id})
        db.jobs.delete_many({"user_id": user_id})
        db.entitlements.delete_many({"user_id": user_id})
        profile_store.delete(user_id)

        return Response(status_code=204)

    return router
