"""Worker profile endpoint (INFRA-5): read-only profile-registry access for
the pull-worker daemon.

`GET /api/worker/profiles/{account_id}` returns the account's profile
registry document as JSON so the worker can materialize it to local disk
before invoking `process_video.py` (the pipeline's actual profile CLI flag
still needs confirming against argparse before that wiring lands -- out of
scope here, this endpoint only exposes the data). Bearer-gated on the SAME
`PICKLEBALL_WORKER_BEARER_TOKEN` machine-principal credential as
`server/routes/worker.py`'s job-queue endpoints -- a separate concern from
user JWTs (`server/security.py`'s `bearer_auth_dependency`); the worker never
authenticates as a user.

An account with no profiles yet returns an empty-but-valid registry document
(200, not 404) rather than failing the worker's pre-flight -- no profiles is
a normal state, not an error.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from ..profile_store import ProfileStore, empty_registry


def build_profiles_worker_router(*, db: Any, worker_token: str) -> APIRouter:
    router = APIRouter()
    expected_authorization = f"Bearer {worker_token}" if worker_token else None
    profile_store = ProfileStore(db)

    def require_worker(authorization: str | None = Header(default=None)) -> str:
        if expected_authorization is None or authorization != expected_authorization:
            raise HTTPException(status_code=403, detail="invalid worker bearer token")
        return "worker"

    @router.get("/api/worker/profiles/{account_id}")
    def get_account_profiles(account_id: str, _: str = Depends(require_worker)) -> dict[str, Any]:
        registry = profile_store.load(account_id) or empty_registry(account_id)
        return registry.model_dump(mode="json")

    return router


__all__ = ["build_profiles_worker_router"]
