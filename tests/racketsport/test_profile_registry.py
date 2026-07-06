from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.owner_capture_intake import OwnerCaptureVideoMetadata, camera_fingerprint
from threed.racketsport.profile_registry import (
    CourtProfile,
    DeviceProfile,
    GearProfile,
    PlayerProfile,
    ProfileConsentError,
    ProfileRegistryDocument,
    SessionCache,
    create_profile_registry,
    list_accounts,
    list_profiles,
    load_profile_registry,
    lookup_court_profile,
    profile_registry_json_schema,
    update_profile,
)


def test_court_profile_round_trips_and_matches_fingerprint_hint(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    create_profile_registry("owner_1", profiles_root=profiles_root)
    fingerprint = _camera_fingerprint()
    court = _court_profile(account_id="owner_1", camera_fingerprint=fingerprint)

    update_profile("owner_1", court, profiles_root=profiles_root)

    matched = lookup_court_profile(
        "owner_1",
        camera_fingerprint=fingerprint,
        gps_hint=court.gps_hint,
        wifi_hint=court.wifi_hint,
        profiles_root=profiles_root,
    )
    assert matched is not None
    assert matched.profile_id == "court_home"
    assert matched.frozen_calibration_ref.uri == "runs/calibration/court_home/court_calibration.json"
    assert matched.net_post_height_in == 36.0
    assert matched.net_center_height_in == 34.0
    assert matched.net_height_provenance == "tape_measured"

    assert lookup_court_profile("missing", camera_fingerprint=fingerprint, profiles_root=profiles_root) is None
    assert (
        lookup_court_profile(
            "owner_1",
            camera_fingerprint=fingerprint,
            gps_hint={"latitude": 40.0, "longitude": -73.0, "radius_m": 25.0},
            profiles_root=profiles_root,
        )
        is None
    )


def test_player_profile_persistence_requires_granted_consent_for_non_owner_biometrics(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    player = _player_profile(account_id="friend_account", consent_status="unknown")

    with pytest.raises(ProfileConsentError, match="consent_status='granted'"):
        update_profile("friend_account", player, profiles_root=profiles_root)

    granted = player.model_copy(update={"consent_status": "granted"})
    update_profile("friend_account", granted, profiles_root=profiles_root)

    persisted = list_profiles("friend_account", "player", profiles_root=profiles_root)
    assert len(persisted) == 1
    assert persisted[0].profile_id == "player_regular_partner"
    assert persisted[0].retention.scope == "account_lifetime"


def test_registry_schema_file_matches_model_and_validates_examples() -> None:
    schema_path = Path("docs/racketsport/profile_registry_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema == profile_registry_json_schema()
    assert schema["$defs"]["PlayerProfile"]["required"].count("consent_status") == 1
    assert schema["$defs"]["PlayerProfile"]["required"].count("retention") == 1

    payload = _registry_document().model_dump(mode="json")
    round_trip = ProfileRegistryDocument.model_validate(payload)
    assert set(round_trip.court_profiles) == {"court_home"}
    assert set(round_trip.device_profiles) == {"iphone_15pro_wide_1x"}
    assert set(round_trip.player_profiles) == {"player_regular_partner"}
    assert set(round_trip.gear_profiles) == {"gear_owner_set"}
    assert set(round_trip.session_caches) == {"session_20260706"}


def test_update_profile_writes_versioned_flat_per_account_json(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    registry = create_profile_registry("owner_1", profiles_root=profiles_root)
    assert registry.registry_version == 1

    device = _device_profile(account_id="owner_1")
    registry = update_profile("owner_1", device, profiles_root=profiles_root)
    assert registry.registry_version == 2
    assert registry.device_profiles["iphone_15pro_wide_1x"].version == 1

    updated_device = device.model_copy(update={"exposure_constant": 1.25})
    registry = update_profile("owner_1", updated_device, profiles_root=profiles_root)
    assert registry.registry_version == 3
    assert registry.device_profiles["iphone_15pro_wide_1x"].version == 2
    assert registry.device_profiles["iphone_15pro_wide_1x"].exposure_constant == 1.25

    account_dir = profiles_root / "owner_1"
    assert sorted(path.name for path in account_dir.iterdir()) == [
        "profile_registry.json",
        "profile_registry_v000001.json",
        "profile_registry_v000002.json",
        "profile_registry_v000003.json",
    ]
    assert all(path.is_file() for path in account_dir.iterdir())
    assert list_accounts(profiles_root=profiles_root) == ["owner_1"]
    assert load_profile_registry("owner_1", profiles_root=profiles_root).registry_version == 3


def _registry_document() -> ProfileRegistryDocument:
    account_id = "owner_1"
    fingerprint = _camera_fingerprint()
    return ProfileRegistryDocument(
        schema_version=1,
        artifact_type="racketsport_profile_registry",
        account_id=account_id,
        registry_version=5,
        court_profiles={"court_home": _court_profile(account_id=account_id, camera_fingerprint=fingerprint)},
        device_profiles={"iphone_15pro_wide_1x": _device_profile(account_id=account_id)},
        player_profiles={"player_regular_partner": _player_profile(account_id=account_id, consent_status="granted")},
        gear_profiles={"gear_owner_set": _gear_profile(account_id=account_id)},
        session_caches={"session_20260706": _session_cache(account_id=account_id)},
    )


def _camera_fingerprint() -> str:
    metadata = OwnerCaptureVideoMetadata(width=1920, height=1080, fps=29.970, duration_s=12.0, frame_count=360)
    sidecar = {
        "intrinsics": {
            "fx": 1510.0,
            "fy": 1508.5,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.01, -0.02, 0.0, 0.0],
            "source": "charuco_sweep",
        }
    }
    return camera_fingerprint(metadata, sidecar)


def _source_trace() -> dict[str, object]:
    return {
        "source_clip_id": "owner_empty_court_20260706",
        "source_clip_ref": "runs/owner_data/owner_empty_court_20260706/clip.mov",
        "source_profile_id": None,
    }


def _retention(scope: str = "account_lifetime") -> dict[str, object]:
    return {
        "scope": scope,
        "delete_with_source_clip": True,
        "delete_with_source_profile": True,
        "retention_days": None,
        "legal_basis": "owner_setup",
    }


def _artifact_ref(uri: str, artifact_type: str) -> dict[str, object]:
    return {"uri": uri, "artifact_type": artifact_type, "source_trace": _source_trace()}


def _court_profile(*, account_id: str, camera_fingerprint: str) -> CourtProfile:
    return CourtProfile(
        schema_version=1,
        artifact_type="racketsport_court_profile",
        account_id=account_id,
        profile_id="court_home",
        display_name="Home Court",
        version=1,
        source_trace=_source_trace(),
        retention=_retention(),
        frozen_calibration_ref=_artifact_ref("runs/calibration/court_home/court_calibration.json", "court_calibration"),
        line_paint_color_lab={"l": 92.0, "a": -4.0, "b": 6.0},
        background_frame_ref=_artifact_ref("runs/backgrounds/court_home/frame_000120.jpg", "background_frame"),
        gps_hint={"latitude": 37.3318, "longitude": -122.0312, "radius_m": 30.0},
        wifi_hint={"ssid": "CourtWiFi", "bssid": "aa:bb:cc:dd:ee:ff"},
        camera_fingerprint=camera_fingerprint,
        net_post_height_in=36.0,
        net_center_height_in=34.0,
        net_height_provenance="tape_measured",
        net_height_source_trace=_source_trace(),
    )


def _device_profile(*, account_id: str) -> DeviceProfile:
    return DeviceProfile(
        schema_version=1,
        artifact_type="racketsport_device_profile",
        account_id=account_id,
        profile_id="iphone_15pro_wide_1x",
        display_name="iPhone 15 Pro Wide 1x",
        version=1,
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
                "source_trace": _source_trace(),
            }
        ],
        exposure_constant=1.0,
    )


def _player_profile(*, account_id: str, consent_status: str) -> PlayerProfile:
    return PlayerProfile(
        schema_version=1,
        artifact_type="racketsport_player_profile",
        account_id=account_id,
        profile_id="player_regular_partner",
        display_name="Regular Partner",
        version=1,
        source_trace=_source_trace(),
        retention=_retention(),
        is_account_owner=False,
        height_m=1.78,
        height_provenance="tape_measured",
        frozen_shape_betas_ref=_artifact_ref("runs/body_profiles/player_regular_partner/betas.json", "frozen_shape_betas"),
        reid_gallery_ref=_artifact_ref("runs/reid/player_regular_partner/gallery.json", "reid_gallery"),
        handedness="right",
        cross_account_shareable=True,
        consent_status=consent_status,
        consent_source_trace=_source_trace(),
    )


def _gear_profile(*, account_id: str) -> GearProfile:
    return GearProfile(
        schema_version=1,
        artifact_type="racketsport_gear_profile",
        account_id=account_id,
        profile_id="gear_owner_set",
        display_name="Owner Gear",
        version=1,
        source_trace=_source_trace(),
        retention=_retention(),
        paddle_scan_ref=_artifact_ref("runs/paddle_scans/owner_paddle/scan.glb", "paddle_scan"),
        paddle_dims={"length_in": 15.9, "width_in": 7.85, "thickness_in": 0.55},
        ball_sku="dura-fast-40-neon",
        ball_color_window_lab={"min": {"l": 70.0, "a": -36.0, "b": 55.0}, "max": {"l": 94.0, "a": -12.0, "b": 95.0}},
        ball_diameter_mm=74.0,
    )


def _session_cache(*, account_id: str) -> SessionCache:
    return SessionCache(
        schema_version=1,
        artifact_type="racketsport_session_cache",
        account_id=account_id,
        profile_id="session_20260706",
        display_name="2026-07-06 evening doubles",
        version=1,
        source_trace=_source_trace(),
        retention=_retention("clip_lifetime"),
        session_id="session_20260706",
        court_profile_id="court_home",
        device_profile_id="iphone_15pro_wide_1x",
        player_profile_ids=["player_regular_partner"],
        gear_profile_id="gear_owner_set",
        background_model_ref=_artifact_ref("runs/session_cache/session_20260706/background.json", "background_model"),
        lighting_variant_ref=_artifact_ref("runs/session_cache/session_20260706/lighting.json", "lighting_variant"),
        apparel_color_lab_by_player={"player_regular_partner": {"l": 35.0, "a": 12.0, "b": -30.0}},
    )
