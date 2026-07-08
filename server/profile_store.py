"""Mongo-backed profile registry store (INFRA-5): P0-9 profile registry port.

Storage-layer port of `threed/racketsport/profile_registry.py`'s flat
per-account JSON registry (`runs/profiles/{account_id}/profile_registry.json`)
to a `profiles` Mongo collection, one document per account keyed by
`account_id`. The five Pydantic profile schemas + `RetentionPolicy` consent
enforcement (`_validate_persistence_rules`) are imported UNCHANGED from the
canonical module -- this file only replaces the storage mechanism. Behavioral
parity with the file backend is the bar (per the lane spec): same versioning
semantics (per-profile `version` increments on update, `created_at_utc`
preserved across updates, registry-level `registry_version` increments on
every write), same consent enforcement (persisting a non-owner player's
biometric refs without granted consent still raises `ProfileConsentError`).

New behavior not present in the file backend (spec-directed): a profile whose
`retention.scope == "session_only"` is never written to Mongo -- it must not
persist past the request that produced it. `ProfileStore.update` still runs
consent validation and returns the account's current registry (so callers get
a consistent response shape), but nothing is mutated for a session_only
profile.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from threed.racketsport.profile_registry import (
    Profile,
    ProfileRegistryDocument,
    ProfileRegistryError,
    ProfileType,
    _validate_persistence_rules,
)

UTC = timezone.utc

_COLLECTION_BY_TYPE: dict[str, str] = {
    "court": "court_profiles",
    "device": "device_profiles",
    "player": "player_profiles",
    "gear": "gear_profiles",
    "session_cache": "session_caches",
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _empty_registry_payload(account_id: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_profile_registry",
        "account_id": account_id,
        "registry_version": 1,
        "created_at_utc": now,
        "updated_at_utc": now,
        "court_profiles": {},
        "device_profiles": {},
        "player_profiles": {},
        "gear_profiles": {},
        "session_caches": {},
    }


def empty_registry(account_id: str) -> ProfileRegistryDocument:
    """Build an empty registry document in-memory, without touching Mongo.

    Used by read paths (e.g. the worker profile endpoint) that must not have
    a write side effect just because an account has no profiles yet.
    """
    return ProfileRegistryDocument.model_validate(_empty_registry_payload(account_id))


def _strip_mongo_id(doc: dict[str, Any]) -> dict[str, Any]:
    payload = dict(doc)
    payload.pop("_id", None)
    return payload


class ProfileStore:
    """Mongo-backed profile registry, one document per account in `db.profiles`."""

    def __init__(self, db: Any) -> None:
        self._collection = db.profiles

    def load(self, account_id: str) -> ProfileRegistryDocument | None:
        """Read-only load. Returns None if the account has no registry doc yet."""
        doc = self._collection.find_one({"_id": account_id})
        if doc is None:
            return None
        return ProfileRegistryDocument.model_validate(_strip_mongo_id(doc))

    def get_or_create(self, account_id: str) -> ProfileRegistryDocument:
        """Load the account's registry, creating (and persisting) an empty one
        if it does not exist yet -- mirrors `create_profile_registry`'s
        create-if-missing behavior in the file backend."""
        registry = self.load(account_id)
        if registry is not None:
            return registry
        registry = empty_registry(account_id)
        self._collection.update_one(
            {"_id": account_id},
            {"$set": {"_id": account_id, **registry.model_dump(mode="json")}},
            upsert=True,
        )
        return registry

    def list_profiles(self, account_id: str, profile_type: ProfileType) -> list[Profile]:
        registry = self.load(account_id)
        if registry is None:
            return []
        collection_name = _COLLECTION_BY_TYPE[profile_type]
        return list(getattr(registry, collection_name).values())

    def update(self, account_id: str, profile: Profile) -> ProfileRegistryDocument:
        if profile.account_id != account_id:
            raise ProfileRegistryError(
                f"profile account_id {profile.account_id!r} does not match requested account_id {account_id!r}"
            )
        # Consent enforcement is shared, unchanged logic from the canonical
        # module -- runs BEFORE any persistence decision so a rejected write
        # (e.g. missing consent) never reaches Mongo, matching the file
        # backend's fail-closed ordering.
        _validate_persistence_rules(profile)

        registry = self.get_or_create(account_id)
        if profile.retention.scope == "session_only":
            # Never persists past request scope: return the CURRENT stored
            # registry untouched. No Mongo write happens for this profile.
            return registry

        collection_name = _COLLECTION_BY_TYPE[profile.profile_type]
        collection = dict(getattr(registry, collection_name))
        existing = collection.get(profile.profile_id)
        now = _utc_now()
        version = existing.version + 1 if existing is not None else max(1, profile.version)
        created_at = existing.created_at_utc if existing is not None else profile.created_at_utc
        updated_profile = profile.model_copy(
            update={"version": version, "created_at_utc": created_at, "updated_at_utc": now}
        )
        collection[profile.profile_id] = updated_profile

        payload = registry.model_dump(mode="json")
        payload[collection_name] = {pid: item.model_dump(mode="json") for pid, item in collection.items()}
        payload["registry_version"] = registry.registry_version + 1
        payload["updated_at_utc"] = now
        updated_registry = ProfileRegistryDocument.model_validate(payload)

        self._collection.update_one(
            {"_id": account_id},
            {"$set": {"_id": account_id, **updated_registry.model_dump(mode="json")}},
            upsert=True,
        )
        return updated_registry

    def delete(self, account_id: str) -> int:
        """Hard-delete the account's profile registry doc (delete-cascade).

        Returns the number of documents deleted (0 or 1) -- idempotent, so a
        second call against an already-deleted account is a no-op.
        """
        result = self._collection.delete_one({"_id": account_id})
        return int(getattr(result, "deleted_count", 0) or 0)


__all__ = ["ProfileStore", "empty_registry"]
