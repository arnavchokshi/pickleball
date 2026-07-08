"""INFRA-5 profile-store tests: Mongo-backend parity with the flat-file
registry (`threed/racketsport/profile_registry.py`) that `ProfileStore` ports
storage for, plus the worker-facing HTTP endpoint that exposes it.

Section 1 exercises `ProfileStore` directly against a mongomock database
(store->load round-trip, versioning, consent enforcement, the
session_only-never-persists rule, and that all 5 profile schemas validate
through the Mongo round-trip). Section 2 exercises
`GET /api/worker/profiles/{account_id}` via `create_app` + TestClient +
moto, matching the bearer-auth conventions in test_worker_endpoints.py.
"""

from pathlib import Path

import mongomock
import pytest
from fastapi.testclient import TestClient

from server.profile_store import ProfileStore, empty_registry
from server.render_app import create_app
from threed.racketsport.profile_registry import (
    CourtProfile,
    DeviceProfile,
    GearProfile,
    PlayerProfile,
    ProfileConsentError,
    ProfileRegistryDocument,
    SessionCache,
    SourceTrace,
)

ACCOUNT_ID = "user_profile_store_test"


def _retention(scope: str = "account_lifetime") -> dict[str, object]:
    return {
        "scope": scope,
        "delete_with_source_clip": False,
        "delete_with_source_profile": False,
        "retention_days": None,
        "legal_basis": "owner_setup",
    }


def _source_trace() -> SourceTrace:
    return SourceTrace(source_clip_id="seed_clip")


def _owner_player_profile(*, account_id: str = ACCOUNT_ID, profile_id: str = "player_self") -> PlayerProfile:
    return PlayerProfile(
        schema_version=1,
        artifact_type="racketsport_player_profile",
        account_id=account_id,
        profile_id=profile_id,
        display_name="Account Owner",
        source_trace=_source_trace(),
        retention=_retention(),
        is_account_owner=True,
        height_m=1.80,
        height_provenance="self_reported",
        handedness="right",
        cross_account_shareable=False,
        consent_status="owner",
    )


def _non_owner_player_profile(*, consent_status: str, account_id: str = ACCOUNT_ID) -> PlayerProfile:
    return PlayerProfile(
        schema_version=1,
        artifact_type="racketsport_player_profile",
        account_id=account_id,
        profile_id="player_friend",
        display_name="Regular Partner",
        source_trace=_source_trace(),
        retention=_retention(),
        is_account_owner=False,
        height_m=1.78,
        height_provenance="tape_measured",
        frozen_shape_betas_ref={
            "uri": "runs/body_profiles/player_friend/betas.json",
            "artifact_type": "frozen_shape_betas",
            "source_trace": _source_trace().model_dump(),
        },
        handedness="left",
        cross_account_shareable=False,
        consent_status=consent_status,
        consent_source_trace=_source_trace() if consent_status == "granted" else None,
    )


def _court_profile(*, account_id: str = ACCOUNT_ID) -> CourtProfile:
    return CourtProfile(
        schema_version=1,
        artifact_type="racketsport_court_profile",
        account_id=account_id,
        profile_id="court_home",
        display_name="Home Court",
        source_trace=_source_trace(),
        retention=_retention(),
        frozen_calibration_ref={
            "uri": "runs/calibration/court_home/court_calibration.json",
            "artifact_type": "court_calibration",
            "source_trace": _source_trace().model_dump(),
        },
        line_paint_color_lab={"l": 92.0, "a": -4.0, "b": 6.0},
        background_frame_ref={
            "uri": "runs/backgrounds/court_home/frame_000120.jpg",
            "artifact_type": "background_frame",
            "source_trace": _source_trace().model_dump(),
        },
        camera_fingerprint="fp_abc123",
    )


def _device_profile(*, account_id: str = ACCOUNT_ID) -> DeviceProfile:
    return DeviceProfile(
        schema_version=1,
        artifact_type="racketsport_device_profile",
        account_id=account_id,
        profile_id="iphone_15pro_wide_1x",
        display_name="iPhone 15 Pro Wide 1x",
        source_trace=_source_trace(),
        retention=_retention(),
        device_key="iphone15pro-wide-1x",
        intrinsics_by_lens_zoom=[
            {
                "lens": "wide",
                "zoom": 1.0,
                "intrinsics": {
                    "fx": 1510.0,
                    "fy": 1508.5,
                    "cx": 960.0,
                    "cy": 540.0,
                    "dist": [0.01, -0.02, 0.0, 0.0],
                    "source": "charuco_sweep",
                },
                "source_trace": _source_trace().model_dump(),
            }
        ],
        exposure_constant=1.0,
    )


def _gear_profile(*, account_id: str = ACCOUNT_ID) -> GearProfile:
    return GearProfile(
        schema_version=1,
        artifact_type="racketsport_gear_profile",
        account_id=account_id,
        profile_id="gear_owner_set",
        display_name="Owner Gear",
        source_trace=_source_trace(),
        retention=_retention(),
        paddle_scan_ref={
            "uri": "runs/paddle_scans/owner_paddle/scan.glb",
            "artifact_type": "paddle_scan",
            "source_trace": _source_trace().model_dump(),
        },
        paddle_dims={"length_in": 15.9, "width_in": 7.85, "thickness_in": 0.55},
        ball_sku="dura-fast-40-neon",
        ball_color_window_lab={"min": {"l": 70.0, "a": -36.0, "b": 55.0}, "max": {"l": 94.0, "a": -12.0, "b": 95.0}},
        ball_diameter_mm=74.0,
    )


def _session_cache(*, account_id: str = ACCOUNT_ID, scope: str = "clip_lifetime") -> SessionCache:
    return SessionCache(
        schema_version=1,
        artifact_type="racketsport_session_cache",
        account_id=account_id,
        profile_id="session_20260706",
        display_name="2026-07-06 evening doubles",
        source_trace=_source_trace(),
        retention=_retention(scope),
        session_id="session_20260706",
    )


# ---------------------------------------------------------------------------
# Section 1: ProfileStore unit tests (mongomock)
# ---------------------------------------------------------------------------


def _db():
    return mongomock.MongoClient()["pickleball"]


def test_load_missing_account_returns_none_and_list_returns_empty():
    store = ProfileStore(_db())

    assert store.load("no_such_account") is None
    assert store.list_profiles("no_such_account", "player") == []


def test_get_or_create_persists_empty_registry_for_new_account():
    db = _db()
    store = ProfileStore(db)

    registry = store.get_or_create(ACCOUNT_ID)

    assert isinstance(registry, ProfileRegistryDocument)
    assert registry.account_id == ACCOUNT_ID
    assert registry.registry_version == 1
    assert registry.player_profiles == {}
    # Persisted: a second get_or_create loads the SAME doc, not a new one.
    again = store.get_or_create(ACCOUNT_ID)
    assert again.created_at_utc == registry.created_at_utc


def test_update_then_load_round_trips_per_account():
    db = _db()
    store = ProfileStore(db)

    store.update(ACCOUNT_ID, _owner_player_profile())
    loaded = store.load(ACCOUNT_ID)

    assert loaded is not None
    assert set(loaded.player_profiles) == {"player_self"}
    assert loaded.player_profiles["player_self"].display_name == "Account Owner"
    assert loaded.registry_version == 2  # 1 (create) + 1 (this update)


def test_update_increments_version_and_preserves_created_at():
    db = _db()
    store = ProfileStore(db)

    first = store.update(ACCOUNT_ID, _owner_player_profile())
    created_at = first.player_profiles["player_self"].created_at_utc
    assert first.player_profiles["player_self"].version == 1

    updated_profile = _owner_player_profile().model_copy(update={"height_m": 1.85})
    second = store.update(ACCOUNT_ID, updated_profile)

    assert second.player_profiles["player_self"].version == 2
    assert second.player_profiles["player_self"].height_m == 1.85
    assert second.player_profiles["player_self"].created_at_utc == created_at
    assert second.registry_version == first.registry_version + 1


def test_session_only_scope_never_persists():
    db = _db()
    store = ProfileStore(db)

    ephemeral = _owner_player_profile(profile_id="player_session_only")
    ephemeral = ephemeral.model_copy(
        update={
            "retention": ephemeral.retention.model_copy(
                update={"scope": "session_only", "delete_with_source_clip": False, "delete_with_source_profile": False}
            )
        }
    )

    result = store.update(ACCOUNT_ID, ephemeral)

    # The call succeeds (so a caller CAN use the profile in-request-scope),
    # but nothing about it is ever written to Mongo.
    assert "player_session_only" not in result.player_profiles
    persisted = store.load(ACCOUNT_ID)
    assert persisted is None or "player_session_only" not in persisted.player_profiles


def test_consent_enforcement_rejects_ungranted_biometric_persistence():
    db = _db()
    store = ProfileStore(db)

    with pytest.raises(ProfileConsentError, match="consent_status='granted'"):
        store.update(ACCOUNT_ID, _non_owner_player_profile(consent_status="unknown"))

    # Nothing was written by the rejected call.
    assert store.load(ACCOUNT_ID) is None

    # Granted consent persists fine.
    store.update(ACCOUNT_ID, _non_owner_player_profile(consent_status="granted"))
    loaded = store.load(ACCOUNT_ID)
    assert loaded is not None
    assert loaded.player_profiles["player_friend"].consent_status == "granted"


def test_update_rejects_account_id_mismatch():
    store = ProfileStore(_db())
    profile = _owner_player_profile(account_id="other_account")

    with pytest.raises(Exception, match="does not match requested account_id"):
        store.update(ACCOUNT_ID, profile)


def test_all_five_schemas_validate_through_mongo_round_trip():
    db = _db()
    store = ProfileStore(db)

    store.update(ACCOUNT_ID, _court_profile())
    store.update(ACCOUNT_ID, _device_profile())
    store.update(ACCOUNT_ID, _owner_player_profile())
    store.update(ACCOUNT_ID, _gear_profile())
    store.update(ACCOUNT_ID, _session_cache())

    loaded = store.load(ACCOUNT_ID)
    assert loaded is not None
    assert set(loaded.court_profiles) == {"court_home"}
    assert set(loaded.device_profiles) == {"iphone_15pro_wide_1x"}
    assert set(loaded.player_profiles) == {"player_self"}
    assert set(loaded.gear_profiles) == {"gear_owner_set"}
    assert set(loaded.session_caches) == {"session_20260706"}
    assert store.list_profiles(ACCOUNT_ID, "gear")[0].profile_id == "gear_owner_set"


def test_delete_is_idempotent():
    db = _db()
    store = ProfileStore(db)
    store.update(ACCOUNT_ID, _owner_player_profile())

    assert store.delete(ACCOUNT_ID) == 1
    assert store.load(ACCOUNT_ID) is None
    assert store.delete(ACCOUNT_ID) == 0


def test_empty_registry_helper_does_not_touch_mongo():
    db = _db()
    registry = empty_registry("never_persisted_account")

    assert registry.account_id == "never_persisted_account"
    assert ProfileStore(db).load("never_persisted_account") is None


# ---------------------------------------------------------------------------
# Section 2: GET /api/worker/profiles/{account_id}
# ---------------------------------------------------------------------------

JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
WORKER_TOKEN = "test-worker-bearer-token"


def _make_app(tmp_path: Path):
    db = mongomock.MongoClient()["pickleball"]
    app = create_app(
        upload_root=tmp_path,
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=db,
        s3_client=None,
        accounts_enabled=True,
        env={
            "PICKLEBALL_JWT_SECRET": JWT_SECRET,
            "PICKLEBALL_INVITE_CODE": INVITE_CODE,
            "PICKLEBALL_S3_BUCKET": "test-bucket",
            "PICKLEBALL_WORKER_BEARER_TOKEN": WORKER_TOKEN,
        },
    )
    return TestClient(app), db


def _worker_auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {WORKER_TOKEN}"}


def test_worker_profiles_endpoint_403_without_token(tmp_path: Path) -> None:
    client, _db = _make_app(tmp_path)

    assert client.get(f"/api/worker/profiles/{ACCOUNT_ID}").status_code == 403


def test_worker_profiles_endpoint_403_wrong_token(tmp_path: Path) -> None:
    client, _db = _make_app(tmp_path)

    response = client.get(
        f"/api/worker/profiles/{ACCOUNT_ID}", headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 403


def test_worker_profiles_endpoint_200_empty_registry_for_unknown_account(tmp_path: Path) -> None:
    client, db = _make_app(tmp_path)

    response = client.get(f"/api/worker/profiles/{ACCOUNT_ID}", headers=_worker_auth())

    assert response.status_code == 200
    payload = response.json()
    assert payload["account_id"] == ACCOUNT_ID
    assert payload["player_profiles"] == {}
    # A read must not have a write side effect.
    assert db.profiles.find_one({"_id": ACCOUNT_ID}) is None


def test_worker_profiles_endpoint_200_returns_seeded_doc(tmp_path: Path) -> None:
    client, db = _make_app(tmp_path)
    ProfileStore(db).update(ACCOUNT_ID, _owner_player_profile())

    response = client.get(f"/api/worker/profiles/{ACCOUNT_ID}", headers=_worker_auth())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["player_profiles"]) == {"player_self"}
    assert payload["player_profiles"]["player_self"]["display_name"] == "Account Owner"
