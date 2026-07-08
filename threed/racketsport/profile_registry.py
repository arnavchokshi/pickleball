"""Profile registry schemas and flat per-account storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .owner_capture_intake import OwnerCaptureVideoMetadata, camera_fingerprint
from .schemas import CameraIntrinsics, FiniteFloat

UTC = timezone.utc
DEFAULT_PROFILE_ROOT = Path("runs/profiles")
CURRENT_REGISTRY_FILENAME = "profile_registry.json"
COURT_PROFILE_COLOR_DELTA_E_THRESHOLD = 10.0


class ProfileRegistryError(ValueError):
    """Base error for profile registry validation and storage problems."""


class ProfileConsentError(ProfileRegistryError):
    """Raised when persistence would violate biometric consent rules."""


class SourceTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_clip_id: str | None = None
    source_clip_ref: str | None = None
    source_profile_id: str | None = None

    @model_validator(mode="after")
    def _must_trace_to_clip_or_profile(self) -> SourceTrace:
        if not any((self.source_clip_id, self.source_clip_ref, self.source_profile_id)):
            raise ValueError("source_trace must include source_clip_id, source_clip_ref, or source_profile_id")
        return self


class RetentionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["account_lifetime", "clip_lifetime", "session_only", "delete_after_days"]
    delete_with_source_clip: bool
    delete_with_source_profile: bool
    retention_days: int | None = Field(default=None, ge=1)
    legal_basis: str

    @model_validator(mode="after")
    def _retention_days_match_scope(self) -> RetentionPolicy:
        if self.scope == "delete_after_days" and self.retention_days is None:
            raise ValueError("retention_days is required when scope='delete_after_days'")
        if self.scope != "delete_after_days" and self.retention_days is not None:
            raise ValueError("retention_days is only allowed when scope='delete_after_days'")
        return self


class ProfileArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    source_trace: SourceTrace


class LabColor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l: FiniteFloat = Field(ge=0.0, le=100.0)
    a: FiniteFloat = Field(ge=-128.0, le=127.0)
    b: FiniteFloat = Field(ge=-128.0, le=127.0)


class LabColorWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: LabColor
    max: LabColor

    @model_validator(mode="after")
    def _ordered_window(self) -> LabColorWindow:
        for channel in ("l", "a", "b"):
            if getattr(self.min, channel) > getattr(self.max, channel):
                raise ValueError(f"ball_color_window_lab min.{channel} must be <= max.{channel}")
        return self


class GpsHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude: FiniteFloat = Field(ge=-180.0, le=180.0)
    radius_m: FiniteFloat = Field(gt=0.0)


class WifiHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ssid: str | None = None
    bssid: str | None = None

    @model_validator(mode="after")
    def _must_have_wifi_identifier(self) -> WifiHint:
        if not self.ssid and not self.bssid:
            raise ValueError("wifi_hint must include ssid or bssid")
        return self


class ProfileBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    account_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    created_at_utc: str = Field(default_factory=lambda: _utc_now())
    updated_at_utc: str = Field(default_factory=lambda: _utc_now())
    source_trace: SourceTrace
    retention: RetentionPolicy


class CourtProfile(ProfileBase):
    artifact_type: Literal["racketsport_court_profile"]
    profile_type: Literal["court"] = "court"
    frozen_calibration_ref: ProfileArtifactRef
    line_paint_color_lab: LabColor
    background_frame_ref: ProfileArtifactRef
    gps_hint: GpsHint | None = None
    wifi_hint: WifiHint | None = None
    camera_fingerprint: str = Field(min_length=1)
    net_post_height_in: FiniteFloat = Field(default=36.0, gt=0.0)
    net_center_height_in: FiniteFloat = Field(default=34.0, gt=0.0)
    net_height_provenance: Literal["regulation_default", "tape_measured"] = "regulation_default"
    net_height_source_trace: SourceTrace | None = None

    @model_validator(mode="after")
    def _tape_measurement_requires_trace(self) -> CourtProfile:
        if self.net_height_provenance == "tape_measured" and self.net_height_source_trace is None:
            raise ValueError("tape-measured net heights require net_height_source_trace")
        return self


class LensZoomIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lens: str = Field(min_length=1)
    zoom: FiniteFloat = Field(gt=0.0)
    intrinsics: CameraIntrinsics
    source_trace: SourceTrace


class DeviceProfile(ProfileBase):
    artifact_type: Literal["racketsport_device_profile"]
    profile_type: Literal["device"] = "device"
    device_key: str = Field(min_length=1)
    intrinsics_by_lens_zoom: list[LensZoomIntrinsics] = Field(min_length=1)
    exposure_constant: FiniteFloat = Field(gt=0.0)


class PlayerProfile(ProfileBase):
    artifact_type: Literal["racketsport_player_profile"]
    profile_type: Literal["player"] = "player"
    is_account_owner: bool
    height_m: FiniteFloat = Field(gt=0.0)
    height_provenance: Literal["tape_measured", "self_reported", "estimated"]
    frozen_shape_betas_ref: ProfileArtifactRef | None = None
    reid_gallery_ref: ProfileArtifactRef | None = None
    handedness: Literal["right", "left", "ambidextrous", "unknown"]
    cross_account_shareable: bool
    consent_status: Literal["owner", "granted", "denied", "unknown", "revoked"]
    consent_source_trace: SourceTrace | None = None

    @property
    def has_persistent_biometrics(self) -> bool:
        return self.frozen_shape_betas_ref is not None or self.reid_gallery_ref is not None


class PaddleDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    length_in: FiniteFloat = Field(gt=0.0)
    width_in: FiniteFloat = Field(gt=0.0)
    thickness_in: FiniteFloat = Field(gt=0.0)


class GearProfile(ProfileBase):
    artifact_type: Literal["racketsport_gear_profile"]
    profile_type: Literal["gear"] = "gear"
    paddle_scan_ref: ProfileArtifactRef
    paddle_dims: PaddleDimensions
    ball_sku: str = Field(min_length=1)
    ball_color_window_lab: LabColorWindow
    ball_diameter_mm: FiniteFloat = Field(gt=0.0)


class SessionCache(ProfileBase):
    artifact_type: Literal["racketsport_session_cache"]
    profile_type: Literal["session_cache"] = "session_cache"
    session_id: str = Field(min_length=1)
    court_profile_id: str | None = None
    device_profile_id: str | None = None
    player_profile_ids: list[str] = Field(default_factory=list)
    gear_profile_id: str | None = None
    background_model_ref: ProfileArtifactRef | None = None
    lighting_variant_ref: ProfileArtifactRef | None = None
    apparel_color_lab_by_player: dict[str, LabColor] = Field(default_factory=dict)


Profile: TypeAlias = CourtProfile | DeviceProfile | PlayerProfile | GearProfile | SessionCache
ProfileType: TypeAlias = Literal["court", "device", "player", "gear", "session_cache"]


@dataclass(frozen=True)
class CourtProfileMatch:
    profile: CourtProfile
    match_confidence: float
    color_delta_e2000: float | None
    gps_distance_m: float | None


class ProfileRegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Racket-sport profile registry")

    schema_version: Literal[1]
    artifact_type: Literal["racketsport_profile_registry"]
    account_id: str = Field(min_length=1)
    registry_version: int = Field(default=1, ge=1)
    created_at_utc: str = Field(default_factory=lambda: _utc_now())
    updated_at_utc: str = Field(default_factory=lambda: _utc_now())
    court_profiles: dict[str, CourtProfile] = Field(default_factory=dict)
    device_profiles: dict[str, DeviceProfile] = Field(default_factory=dict)
    player_profiles: dict[str, PlayerProfile] = Field(default_factory=dict)
    gear_profiles: dict[str, GearProfile] = Field(default_factory=dict)
    session_caches: dict[str, SessionCache] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _profile_keys_match_ids_and_account(self) -> ProfileRegistryDocument:
        for collection in (
            self.court_profiles,
            self.device_profiles,
            self.player_profiles,
            self.gear_profiles,
            self.session_caches,
        ):
            for key, profile in collection.items():
                if key != profile.profile_id:
                    raise ValueError(f"profile map key {key!r} does not match profile_id {profile.profile_id!r}")
                if profile.account_id != self.account_id:
                    raise ValueError(
                        f"profile {profile.profile_id!r} account_id {profile.account_id!r} "
                        f"does not match registry account_id {self.account_id!r}"
                    )
        return self


def create_profile_registry(
    account_id: str,
    *,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
    overwrite: bool = False,
) -> ProfileRegistryDocument:
    safe_account_id = _safe_id(account_id, label="account_id")
    path = _registry_path(safe_account_id, profiles_root)
    if path.exists() and not overwrite:
        return load_profile_registry(safe_account_id, profiles_root=profiles_root)
    now = _utc_now()
    registry = ProfileRegistryDocument(
        schema_version=1,
        artifact_type="racketsport_profile_registry",
        account_id=safe_account_id,
        registry_version=1,
        created_at_utc=now,
        updated_at_utc=now,
    )
    _write_registry(registry, profiles_root=profiles_root)
    return registry


def load_profile_registry(
    account_id: str,
    *,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> ProfileRegistryDocument:
    path = _registry_path(_safe_id(account_id, label="account_id"), profiles_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProfileRegistryDocument.model_validate(payload)


def update_profile(
    account_id: str,
    profile: Profile,
    *,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> ProfileRegistryDocument:
    safe_account_id = _safe_id(account_id, label="account_id")
    if profile.account_id != safe_account_id:
        raise ProfileRegistryError(
            f"profile account_id {profile.account_id!r} does not match requested account_id {safe_account_id!r}"
        )
    _validate_persistence_rules(profile)
    path = _registry_path(safe_account_id, profiles_root)
    registry = (
        load_profile_registry(safe_account_id, profiles_root=profiles_root)
        if path.exists()
        else create_profile_registry(safe_account_id, profiles_root=profiles_root)
    )

    collection_name = _collection_name(profile.profile_type)
    collections = _registry_collections(registry)
    collection = dict(collections[collection_name])
    existing = collection.get(profile.profile_id)
    now = _utc_now()
    version = existing.version + 1 if existing is not None else max(1, profile.version)
    created_at = existing.created_at_utc if existing is not None else profile.created_at_utc
    updated_profile = profile.model_copy(
        update={"version": version, "created_at_utc": created_at, "updated_at_utc": now}
    )
    collection[profile.profile_id] = updated_profile

    payload = registry.model_dump(mode="json")
    payload[collection_name] = {profile_id: item.model_dump(mode="json") for profile_id, item in collection.items()}
    payload["registry_version"] = registry.registry_version + 1
    payload["updated_at_utc"] = now
    updated_registry = ProfileRegistryDocument.model_validate(payload)
    _write_registry(updated_registry, profiles_root=profiles_root)
    return updated_registry


def list_profiles(
    account_id: str,
    profile_type: ProfileType,
    *,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> list[Profile]:
    path = _registry_path(_safe_id(account_id, label="account_id"), profiles_root)
    if not path.exists():
        return []
    registry = load_profile_registry(account_id, profiles_root=profiles_root)
    return list(_registry_collections(registry)[_collection_name(profile_type)].values())


def list_accounts(*, profiles_root: str | Path = DEFAULT_PROFILE_ROOT) -> list[str]:
    root = Path(profiles_root)
    if not root.exists():
        return []
    accounts = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / CURRENT_REGISTRY_FILENAME).is_file()
    ]
    return sorted(accounts)


def lookup_court_profile(
    account_id: str,
    *,
    camera_fingerprint: str,
    line_color_lab: LabColor | dict[str, Any] | None = None,
    gps_hint: GpsHint | dict[str, Any] | None = None,
    wifi_hint: WifiHint | dict[str, Any] | None = None,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> CourtProfile | None:
    """Return the best matching court profile while preserving the legacy return shape."""

    profile, _confidence = match_court_profile(
        account_id,
        camera_fingerprint=camera_fingerprint,
        line_color_lab=line_color_lab,
        gps_hint=gps_hint,
        wifi_hint=wifi_hint,
        profiles_root=profiles_root,
    )
    return profile


def match_court_profile(
    account_id: str,
    *,
    camera_fingerprint: str,
    line_color_lab: LabColor | dict[str, Any] | None = None,
    gps_hint: GpsHint | dict[str, Any] | None = None,
    wifi_hint: WifiHint | dict[str, Any] | None = None,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> tuple[CourtProfile | None, float]:
    """Return the top ranked court profile and advisory match confidence."""

    matches = rank_court_profile_matches(
        account_id,
        camera_fingerprint=camera_fingerprint,
        line_color_lab=line_color_lab,
        gps_hint=gps_hint,
        wifi_hint=wifi_hint,
        profiles_root=profiles_root,
    )
    if not matches:
        return None, 0.0
    best = matches[0]
    return best.profile, best.match_confidence


def rank_court_profile_matches(
    account_id: str,
    *,
    camera_fingerprint: str,
    line_color_lab: LabColor | dict[str, Any] | None = None,
    gps_hint: GpsHint | dict[str, Any] | None = None,
    wifi_hint: WifiHint | dict[str, Any] | None = None,
    profiles_root: str | Path = DEFAULT_PROFILE_ROOT,
) -> list[CourtProfileMatch]:
    """Rank candidate court profiles by exact camera fingerprint, line color, and optional location."""

    path = _registry_path(_safe_id(account_id, label="account_id"), profiles_root)
    if not path.exists():
        return []
    registry = load_profile_registry(account_id, profiles_root=profiles_root)
    expected_color = LabColor.model_validate(line_color_lab) if line_color_lab is not None else None
    expected_gps = GpsHint.model_validate(gps_hint) if gps_hint is not None else None
    expected_wifi = WifiHint.model_validate(wifi_hint) if wifi_hint is not None else None
    matches: list[CourtProfileMatch] = []
    for profile in registry.court_profiles.values():
        if profile.camera_fingerprint != camera_fingerprint:
            continue
        if expected_wifi is not None and profile.wifi_hint != expected_wifi:
            continue
        gps_distance_m: float | None = None
        if expected_gps is not None:
            if profile.gps_hint is None:
                continue
            gps_distance_m = _gps_distance_m(expected_gps, profile.gps_hint)
            if gps_distance_m > float(expected_gps.radius_m) + float(profile.gps_hint.radius_m):
                continue
        color_delta: float | None = None
        if expected_color is not None:
            color_delta = delta_e2000(profile.line_paint_color_lab, expected_color)
            if color_delta > COURT_PROFILE_COLOR_DELTA_E_THRESHOLD:
                continue
            match_confidence = max(0.0, 1.0 - color_delta / COURT_PROFILE_COLOR_DELTA_E_THRESHOLD)
        else:
            match_confidence = 1.0
        matches.append(
            CourtProfileMatch(
                profile=profile,
                match_confidence=match_confidence,
                color_delta_e2000=color_delta,
                gps_distance_m=gps_distance_m,
            )
        )
    return sorted(
        matches,
        key=lambda item: (
            item.color_delta_e2000 if item.color_delta_e2000 is not None else 0.0,
            item.gps_distance_m if item.gps_distance_m is not None else 0.0,
            item.profile.profile_id,
        ),
    )


def delta_e2000(first: LabColor | dict[str, Any], second: LabColor | dict[str, Any]) -> float:
    """Return CIEDE2000 color difference for two CIELAB colors."""

    lab1 = LabColor.model_validate(first)
    lab2 = LabColor.model_validate(second)
    l1, a1, b1 = float(lab1.l), float(lab1.a), float(lab1.b)
    l2, a2, b2 = float(lab2.l), float(lab2.a), float(lab2.b)

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - math.sqrt(c_bar7 / (c_bar7 + 25.0**7))) if c_bar > 0.0 else 0.0
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = math.hypot(a1_prime, b1)
    c2_prime = math.hypot(a2_prime, b2)
    h1_prime = _lab_hue_degrees(a1_prime, b1) if c1_prime > 0.0 else 0.0
    h2_prime = _lab_hue_degrees(a2_prime, b2) if c2_prime > 0.0 else 0.0

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_h_prime = _delta_h_prime(h1_prime, h2_prime, c1_prime, c2_prime)
    delta_h_big_prime = 2.0 * math.sqrt(c1_prime * c2_prime) * math.sin(math.radians(delta_h_prime / 2.0))

    l_bar_prime = (l1 + l2) / 2.0
    c_bar_prime = (c1_prime + c2_prime) / 2.0
    h_bar_prime = _mean_h_prime(h1_prime, h2_prime, c1_prime, c2_prime)
    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_prime - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_prime - 63.0))
    )
    delta_theta = 30.0 * math.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    c_bar_prime7 = c_bar_prime**7
    r_c = 2.0 * math.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0**7)) if c_bar_prime > 0.0 else 0.0
    s_l = 1.0 + (0.015 * ((l_bar_prime - 50.0) ** 2)) / math.sqrt(20.0 + ((l_bar_prime - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c

    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_h_big_prime / s_h
    return math.sqrt(l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term)


def fingerprint_capture(metadata: OwnerCaptureVideoMetadata, sidecar: dict[str, Any] | None) -> str:
    """Expose the shared capture fingerprint function for registry callers."""

    return camera_fingerprint(metadata, sidecar)


def profile_registry_json_schema() -> dict[str, Any]:
    schema = ProfileRegistryDocument.model_json_schema(ref_template="#/$defs/{model}")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://example.local/racketsport/profile_registry_schema.json"
    return schema


def _validate_persistence_rules(profile: Profile) -> None:
    if not isinstance(profile, PlayerProfile):
        return
    if profile.is_account_owner:
        return
    if profile.has_persistent_biometrics and profile.consent_status != "granted":
        raise ProfileConsentError(
            "persisting non-owner biometric profile refs requires consent_status='granted'"
        )
    if profile.cross_account_shareable and profile.consent_status != "granted":
        raise ProfileConsentError("cross-account-shareable player profiles require consent_status='granted'")
    if profile.consent_status == "granted" and profile.consent_source_trace is None:
        raise ProfileConsentError("consent_status='granted' requires consent_source_trace")


def _registry_path(account_id: str, profiles_root: str | Path) -> Path:
    return Path(profiles_root) / account_id / CURRENT_REGISTRY_FILENAME


def _snapshot_path(registry: ProfileRegistryDocument, profiles_root: str | Path) -> Path:
    return Path(profiles_root) / registry.account_id / f"profile_registry_v{registry.registry_version:06d}.json"


def _write_registry(registry: ProfileRegistryDocument, *, profiles_root: str | Path) -> None:
    account_dir = Path(profiles_root) / registry.account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (account_dir / CURRENT_REGISTRY_FILENAME).write_text(payload, encoding="utf-8")
    _snapshot_path(registry, profiles_root).write_text(payload, encoding="utf-8")


def _registry_collections(registry: ProfileRegistryDocument) -> dict[str, dict[str, Any]]:
    return {
        "court_profiles": registry.court_profiles,
        "device_profiles": registry.device_profiles,
        "player_profiles": registry.player_profiles,
        "gear_profiles": registry.gear_profiles,
        "session_caches": registry.session_caches,
    }


def _collection_name(profile_type: ProfileType) -> str:
    return {
        "court": "court_profiles",
        "device": "device_profiles",
        "player": "player_profiles",
        "gear": "gear_profiles",
        "session_cache": "session_caches",
    }[profile_type]


def _safe_id(value: str, *, label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    if not safe:
        raise ProfileRegistryError(f"{label} cannot be empty")
    return safe


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _lab_hue_degrees(a_value: float, b_value: float) -> float:
    return math.degrees(math.atan2(b_value, a_value)) % 360.0


def _delta_h_prime(h1_prime: float, h2_prime: float, c1_prime: float, c2_prime: float) -> float:
    if c1_prime * c2_prime == 0.0:
        return 0.0
    diff = h2_prime - h1_prime
    if abs(diff) <= 180.0:
        return diff
    if diff > 180.0:
        return diff - 360.0
    return diff + 360.0


def _mean_h_prime(h1_prime: float, h2_prime: float, c1_prime: float, c2_prime: float) -> float:
    if c1_prime * c2_prime == 0.0:
        return h1_prime + h2_prime
    diff = abs(h1_prime - h2_prime)
    if diff <= 180.0:
        return (h1_prime + h2_prime) / 2.0
    if h1_prime + h2_prime < 360.0:
        return (h1_prime + h2_prime + 360.0) / 2.0
    return (h1_prime + h2_prime - 360.0) / 2.0


def _gps_distance_m(first: GpsHint, second: GpsHint) -> float:
    lat1 = math.radians(float(first.latitude))
    lat2 = math.radians(float(second.latitude))
    delta_lat = lat2 - lat1
    delta_lon = math.radians(float(second.longitude) - float(first.longitude))
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 6_371_000.0 * 2.0 * math.atan2(math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine)))


__all__ = [
    "CURRENT_REGISTRY_FILENAME",
    "COURT_PROFILE_COLOR_DELTA_E_THRESHOLD",
    "DEFAULT_PROFILE_ROOT",
    "CourtProfile",
    "CourtProfileMatch",
    "DeviceProfile",
    "GearProfile",
    "GpsHint",
    "LabColor",
    "LabColorWindow",
    "LensZoomIntrinsics",
    "PaddleDimensions",
    "PlayerProfile",
    "Profile",
    "ProfileArtifactRef",
    "ProfileConsentError",
    "ProfileRegistryDocument",
    "ProfileRegistryError",
    "RetentionPolicy",
    "SessionCache",
    "SourceTrace",
    "WifiHint",
    "create_profile_registry",
    "delta_e2000",
    "fingerprint_capture",
    "list_accounts",
    "list_profiles",
    "load_profile_registry",
    "lookup_court_profile",
    "match_court_profile",
    "profile_registry_json_schema",
    "rank_court_profile_matches",
    "update_profile",
]
