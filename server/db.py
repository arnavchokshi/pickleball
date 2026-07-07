"""MongoDB wiring for the accounts/clips/jobs product-infra plumbing (INFRA-1).

Per-concern `*_from_env(env: Mapping | None = None)` factory, mirroring the
`runner_from_env` pattern in `server/gpu_runner.py`: real env is read only when
`env=None`, so tests inject a literal dict instead of monkeypatching os.environ.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database


def mongo_client_from_env(env: Mapping[str, str] | None = None) -> MongoClient | None:
    """Build a MongoClient from `PICKLEBALL_MONGODB_URI`, or None if unset.

    None (not a lazily-failing client) is returned when the URI is missing so
    callers can decide how to degrade (e.g. accounts_enabled but no Mongo
    configured yet should fail loudly, not silently hang on first query).
    """
    resolved_env = os.environ if env is None else env
    uri = resolved_env.get("PICKLEBALL_MONGODB_URI", "").strip()
    if not uri:
        return None
    # Bounded server selection so the health probe reports "unreachable" in
    # seconds instead of hanging for pymongo's 30s default.
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_db(client: MongoClient, env: Mapping[str, str] | None = None) -> Database:
    resolved_env = os.environ if env is None else env
    db_name = resolved_env.get("PICKLEBALL_MONGODB_DB_NAME", "pickleball").strip() or "pickleball"
    return client[db_name]


def ensure_indexes(db: Database) -> None:
    """Create the collection indexes from the approved data model (idempotent).

    `create_index` is a no-op when an equivalent index already exists, so this
    is safe to call on every app startup.
    """
    db.users.create_index("email", unique=True, name="users_email_unique")
    db.refresh_tokens.create_index("token_hash", unique=True, name="refresh_tokens_token_hash_unique")
    db.refresh_tokens.create_index("user_id", name="refresh_tokens_user_id")
    db.refresh_tokens.create_index(
        "expires_at", expireAfterSeconds=0, name="refresh_tokens_expires_at_ttl"
    )
    db.jobs.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)], name="jobs_status_created_at"
    )
    db.jobs.create_index(
        [("worker_id", ASCENDING), ("heartbeat_at", ASCENDING)], name="jobs_worker_id_heartbeat_at"
    )
    db.clips.create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)], name="clips_user_id_created_at"
    )
    db.entitlements.create_index("user_id", unique=True, name="entitlements_user_id_unique")


def mongo_health(db: Database | None) -> dict[str, Any]:
    """Cheap reachability probe for the health endpoint. Never raises."""
    if db is None:
        return {"ok": False, "detail": "mongo not configured"}
    try:
        db.command("ping")
    except Exception as exc:  # noqa: BLE001 - health checks must never 500
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"ok": True}
