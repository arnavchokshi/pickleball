#!/usr/bin/env python3
"""Build the CPU-only plan for the owner-directed PERSON mixed pseudo pack.

This command never runs the teacher and never emits a trainable ``data.yaml``.
It pins the closed P1 human split, plans uniformly spaced pseudo frames, and
writes the exact materialization/interleave contracts for the later GPU lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260722
DEFAULT_TARGET_PSEUDO_FRAMES = 7200
DEFAULT_MAX_FRAMES_PER_SOURCE = 400
DEFAULT_MAX_FAMILY_FRACTION = 0.15
DEFAULT_TEACHER_CONFIDENCE = 0.60
DEFAULT_ANCHOR_PSEUDO_RATIO = (1, 1)
PSEUDO_TARGET_RANGE = (6000, 12000)
MINIMUM_SOURCE_FAMILIES = 15
EXPECTED_PB_VISION_SOURCE_COUNT = 10
EXPECTED_HARVEST_SOURCE_COUNT = 8
EXPECTED_ANCHOR_FAMILY_COUNT = 7

BLIND_SPOT_CAVEAT = "teacher misses become background holes in pseudo frames"
YOLO26M_ID = "yolo26m"
YOLO26M_RELATIVE_PATH = Path("models/checkpoints/yolo26m.pt")
FINAL_LIST_VALIDATOR_RELATIVE_PATH = Path(
    "scripts/racketsport/build_person_mixed_pseudo_pack.py"
)
OD8AL_FAMILY = "family:pickleball-od8al/pickleball-seg"
HEMEL_FAMILY = "family:hemel/pickleball-cedmo"

COMPARE_ONLY_PBVISION_IDS = frozenset(
    {"83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"}
)
PBVISION_TRAIN_IDS = frozenset(
    {
        "0tmdeghtfvjx",
        "143sf3gdwxsa",
        "98z43hspqz13",
        "bewqc0glhgpq",
        "pldtjpw3h0jw",
        "st0epgnab7dr",
        "td2szayjwtrj",
        "tqjlrcntpjvt",
        "utasf5hnozwz",
        "xkadsq9bli3h",
    }
)
HARVEST_TRAIN_IDS = frozenset(
    {
        "73VurrTKCZ8",
        "Ezz6HDNHlnk",
        "HyUqT7zFiwk",
        "_L0HVmAlCQI",
        "pwxNwFfYQlQ",
        "vQhtz8l6VqU",
        "wBu8bC4OfUY",
        "zwCtH_i1_S4",
    }
)
HARVEST_EXPECTED_SHA256 = {
    "73VurrTKCZ8": "e0507678976d31fc04f196467a51bd73dd1bf522d921df915713ad6e7dbc28b7",
    "Ezz6HDNHlnk": "03c82d9769b380dead2b6bd38e9f42bea04a83364c91673fa2ff737aeb71853e",
    "HyUqT7zFiwk": "8679a62299d7a50684f04497a371acd4b0db0417a24bc2e59c9d5ec8cd485b4e",
    "_L0HVmAlCQI": "9e454f99bfac00c56eca2bd339a585b9d2ba30c9a8b37868c9a28f49b0ac708a",
    "pwxNwFfYQlQ": "1fdd6419366d144858be3b1d9da24b073e5eafc495540f43e604f0bfb2e456c8",
    "vQhtz8l6VqU": "f2b4e74b1fb23a3db5cce78f4563f8c466a6b6ed414366c527be1ae7df7ee426",
    "wBu8bC4OfUY": "91758a50be32bd5e40ce46e45ef1222f0c56e9ab84fecdce72f65be37b760e7e",
    "zwCtH_i1_S4": "9528b5d82606735e457bad1ba870fb31a6a61a31d94ccbd6e2103e19a6a5a445",
}
CROSS_COMPONENT_HOLDOUT_ROLES = {
    "Ezz6HDNHlnk": ("BALL_judge_holdout",),
    "HyUqT7zFiwk": ("BALL_judge_holdout",),
}
PROTECTED_CLIP_IDS = (
    "burlington_gold_0300_low_steep_corner",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "wolverine_mixed_0200_mid_steep_corner",
)
PROTECTED_MEDIA_SHA256 = frozenset(
    {
        "fc329b53a8d522046779a45fba4e695ee953421e1187070a4ce9a36239cb1aaa",
        "22955134f7bf9bdc9392bdde868173fbc6ec9afa4d7a8c58f3e7e0ed33d4e0f1",
        "8b0265f5dc3bf3e3b5b5a1423bf7e58ac7972481dc163b8398dcb2f20bf070c9",
        "7f6c33b7cfd94a063405b68708d37d968cc1850e7435aa875f5b30f0afb6cb4b",
    }
)
IYNBD_DERIVATIVE_TOKEN = "IYnbdRs1Jdk"

# This registry is independent of the ignored gallery inventory. It binds every
# preregistered pb.vision ID to media bytes, and the compare-only SHA set is
# denied even when a manifest falsely pairs those bytes with a permitted ID.
PBVISION_MEDIA_SHA256_BY_ID = {
    "0tmdeghtfvjx": "8b007124fa949defff85b11f70de5bf4c4c0e43ba64c085c7eded18f0041dfd1",
    "143sf3gdwxsa": "03fbdc2b056c1b1ed665c71994c06bc485f385b44a2fee892338360c666f845c",
    "83gyqyc10y8f": "272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383",
    "98z43hspqz13": "006eb7d0e7e7c5c351ea72b88c946a452660adb24eff87e77d12419b7330b11f",
    "bewqc0glhgpq": "e6b73a38535aea5d3644c3a94091b3c5d261b6c2b60e5d80a21514ad502b69cf",
    "iottnc0h3ekn": "1f3109f5764c86bc36dffbd1613b3f98750df8088fd6d5191ee3037476b1587a",
    "o4dee9dn0ccr": "21b82040d55fcc662665a45d1d5351cdc74793e2774f4083e6a9621fdd1a6dd1",
    "pldtjpw3h0jw": "4d55d822c0b0bbaedbf27e16301b035beef2542df0b11b416b87f898ba8ff59c",
    "st0epgnab7dr": "2803b4a18c97e3d3165cdbacbe7bcbe6c4b0c273820aa6840b7e731aea98ff04",
    "td2szayjwtrj": "9594260561b334937a1dfb62c1450315fdcb1ee3e1ece304961416c7d15a2d79",
    "tqjlrcntpjvt": "176cb66c13e2fa481839815c1dc41c063b2a0cc17758e75dd9c7f39627f31490",
    "utasf5hnozwz": "614580f5b3a2f634a76f5483e10b8c1f7919fd5affbd3ee86532a539f3f58197",
    "xkadsq9bli3h": "5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181",
}
COMPARE_ONLY_MEDIA_SHA256 = frozenset(
    PBVISION_MEDIA_SHA256_BY_ID[source_id] for source_id in COMPARE_ONLY_PBVISION_IDS
)
QUARANTINED_MEDIA_SHA256 = PROTECTED_MEDIA_SHA256 | COMPARE_ONLY_MEDIA_SHA256

# Immutable build-time closure for the closed P1 lane. These are the six
# hashes reviewed on 2026-07-22; structurally similar replacement data is not
# accepted as a new baseline.
CLOSED_P1_HASH_REGISTRY_ID = "person_p1_roboflow_20260721.closed.reviewed_20260722"
CLOSED_P1_FIDELITY_SHA256 = {
    "closed_manifest_sha256": "0803d1bab92fb8f501086090bdcda57aee5a689f14080457182ed20b20cded60",
    "closed_train_list_sha256": "954a704aaf1e5c12d7dc1bb972609d048bc16763386139b92607622382918116",
    "closed_train_all_rows_sha256": "61b39fe2cf1320848236c092fd30a382b2670abed3d69b6454ff66b83d23529e",
    "anchor_exposure_rows_sha256": "1046db282dcc22de6d85a8706220b56d41fca8db27caa9d0518a9eeeee0dd3a9",
    "human_val_od8al_rows_sha256": "8a8dc3ce62c0b8d5ec0d1d92470fba5085f05f6c07bff4a2d9fe3dc21052dcee",
    "human_val_hemel_rows_sha256": "a25038b00baa0088927fc94cbbbddabfa87246c057f9e4731cf80d98a81ec039",
}
FROZEN_FIDELITY_HASH_KEYS = tuple(CLOSED_P1_FIDELITY_SHA256)
MAX_CANONICAL_DECODE_PASSES = 8


class QuarantinedSourceError(ValueError):
    """Raised when a protected, compare-only, or IYnbd-derived source is offered."""


class FidelityError(ValueError):
    """Raised when the closed P1 split does not satisfy the frozen contract."""


@dataclass(frozen=True)
class Protocol:
    target_pseudo_frames: int = DEFAULT_TARGET_PSEUDO_FRAMES
    max_frames_per_source: int = DEFAULT_MAX_FRAMES_PER_SOURCE
    max_family_fraction: float = DEFAULT_MAX_FAMILY_FRACTION
    teacher_confidence: float = DEFAULT_TEACHER_CONFIDENCE
    anchor_pseudo_ratio: tuple[int, int] = DEFAULT_ANCHOR_PSEUDO_RATIO
    seed: int = DEFAULT_SEED
    cli_protocol_override: bool = False


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_decoded_text(value: Any) -> str:
    """Canonicalize an identity/path scalar before any quarantine comparison."""

    text = str(value)
    for _ in range(MAX_CANONICAL_DECODE_PASSES):
        decoded = unicodedata.normalize("NFKC", unquote(text, errors="replace"))
        if decoded == text:
            break
        text = decoded
    return unicodedata.normalize("NFKC", text).casefold()


def _guard_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _guard_scalars(child)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            yield from _guard_scalars(child)
    else:
        yield value


def _normalise_guard_text(value: Any) -> str:
    return "".join(
        "".join(character for character in _canonical_decoded_text(scalar) if character.isalnum())
        for scalar in _guard_scalars(value)
    )


def _normalise_sha256(value: Any) -> str:
    candidate = _canonical_decoded_text(value).strip()
    return candidate if re.fullmatch(r"[0-9a-f]{64}", candidate) else ""


def _source_media_sha256s(source: Mapping[str, Any]) -> set[str]:
    digests: set[str] = set()
    for key, value in source.items():
        key_text = _normalise_guard_text(key)
        if "sha256" in key_text and ("media" in key_text or "video" in key_text or "source" in key_text):
            digest = _normalise_sha256(value)
            if digest:
                digests.add(digest)
        if isinstance(value, Mapping):
            digests.update(_source_media_sha256s(value))
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, Mapping):
                    digests.update(_source_media_sha256s(child))
    return digests


def quarantine_reason(source: Mapping[str, Any]) -> str | None:
    """Return the binding quarantine reason for any source identity/lineage."""

    source_id = str(source.get("source_id") or source.get("video_id") or source.get("id") or "")
    canonical_source_id = _normalise_guard_text(source_id)
    if canonical_source_id in {
        _normalise_guard_text(value) for value in COMPARE_ONLY_PBVISION_IDS
    }:
        return "PBVISION_COMPARE_ONLY"
    media_shas = _source_media_sha256s(source)
    if media_shas & PROTECTED_MEDIA_SHA256:
        return "PROTECTED_EVAL_MEDIA_SHA256"
    if media_shas & COMPARE_ONLY_MEDIA_SHA256:
        return "PBVISION_COMPARE_ONLY_MEDIA_SHA256"

    normalised = _normalise_guard_text(source)
    if _normalise_guard_text(IYNBD_DERIVATIVE_TOKEN) in normalised:
        return "IYNBDRS1JDK_DERIVATIVE"
    for clip_id in PROTECTED_CLIP_IDS:
        if _normalise_guard_text(clip_id) in normalised:
            return "PROTECTED_EVAL_CLIP_ID"
    for compare_id in COMPARE_ONLY_PBVISION_IDS:
        if _normalise_guard_text(compare_id) in normalised:
            return "PBVISION_COMPARE_ONLY_DERIVATIVE"
    return None


def assert_source_allowed(source: Mapping[str, Any]) -> None:
    reason = quarantine_reason(source)
    if reason is not None:
        source_id = source.get("source_id") or source.get("video_id") or source.get("id") or "unknown"
        raise QuarantinedSourceError(f"refused source {source_id}: {reason}")


def _parse_fps(resolution: str) -> float:
    match = re.search(r"@([0-9]+(?:\.[0-9]+)?)(?:/([0-9]+))?(?:fps)?$", resolution)
    if not match:
        raise ValueError(f"cannot parse fps from resolution {resolution!r}")
    numerator = float(match.group(1))
    denominator = float(match.group(2) or 1.0)
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"invalid fps in resolution {resolution!r}")
    return numerator / denominator


def _expected_frame_count(duration_s: float, fps: float) -> int:
    if not math.isfinite(duration_s) or not math.isfinite(fps) or duration_s <= 0 or fps <= 0:
        raise ValueError(f"invalid duration/fps: duration={duration_s}, fps={fps}")
    return max(1, int(round(duration_s * fps)))


def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Return unique, endpoint-inclusive uniform frame indices."""

    if frame_count <= 0 or sample_count < 0 or sample_count > frame_count:
        raise ValueError(f"invalid frame/sample counts: {frame_count}/{sample_count}")
    if sample_count == 0:
        return []
    if sample_count == 1:
        return [0]
    indices = [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]
    if len(indices) != len(set(indices)):
        raise AssertionError("uniform sampler produced duplicate frame indices")
    return indices


def allocate_source_counts(
    sources: Sequence[Mapping[str, Any]], protocol: Protocol
) -> dict[str, int]:
    """Evenly allocate frames without exceeding source or family caps."""

    if protocol.target_pseudo_frames <= 0:
        raise ValueError("target_pseudo_frames must be positive")
    if protocol.max_frames_per_source <= 0:
        raise ValueError("max_frames_per_source must be positive")
    if not 0 < protocol.max_family_fraction <= 1:
        raise ValueError("max_family_fraction must be in (0, 1]")

    ordered = sorted(sources, key=lambda row: str(row["source_family_id"]))
    family_capacity: dict[str, int] = defaultdict(int)
    source_capacity: dict[str, int] = {}
    for row in ordered:
        source_id = str(row["source_id"])
        family_id = str(row.get("venue_source_family_id") or row["source_family_id"])
        capacity = min(int(row["expected_frame_count"]), protocol.max_frames_per_source)
        if source_id in source_capacity:
            raise ValueError(f"duplicate source_id {source_id}")
        source_capacity[source_id] = capacity
        family_capacity[family_id] += capacity

    total_capacity = sum(source_capacity.values())
    desired = min(protocol.target_pseudo_frames, total_capacity)
    counts = {source_id: 0 for source_id in source_capacity}
    if desired == 0:
        return counts

    # The final-total family cap is strict. Using the desired final count here
    # is valid because a shortfall is reported rather than silently redefined.
    family_limit = max(1, math.floor(desired * protocol.max_family_fraction))
    family_counts: Counter[str] = Counter()
    remaining = desired
    while remaining:
        made_progress = False
        for row in ordered:
            source_id = str(row["source_id"])
            family_id = str(row["source_family_id"])
            if counts[source_id] >= source_capacity[source_id]:
                continue
            if family_counts[family_id] >= family_limit:
                continue
            counts[source_id] += 1
            family_counts[family_id] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            break

    planned = sum(counts.values())
    if planned:
        observed = Counter()
        source_to_family = {
            str(row["source_id"]): str(
                row.get("venue_source_family_id") or row["source_family_id"]
            )
            for row in ordered
        }
        for source_id, count in counts.items():
            observed[source_to_family[source_id]] += count
        # If the desired total was unattainable, its provisional family limit
        # can be too high relative to the smaller achieved total. Trim only
        # the overflowing family until the cap is true for the reported total.
        while planned and max(observed.values(), default=0) / planned > protocol.max_family_fraction + 1e-12:
            limit = math.floor(planned * protocol.max_family_fraction)
            overflowing = sorted(
                (family_id for family_id, count in observed.items() if count > limit),
                key=lambda family_id: (-observed[family_id], family_id),
            )
            if not overflowing:
                break
            family_id = overflowing[0]
            candidates = sorted(
                (
                    source_id
                    for source_id, count in counts.items()
                    if count and source_to_family[source_id] == family_id
                ),
                key=lambda source_id: (-counts[source_id], source_id),
            )
            counts[candidates[0]] -= 1
            observed[family_id] -= 1
            planned -= 1
        if planned and max(observed.values(), default=0) / planned > protocol.max_family_fraction + 1e-12:
            raise AssertionError("family allocation cap was violated")
    return counts


def _source_common(row: Mapping[str, Any], *, source_pool: str, source_id: str, fps: float) -> dict[str, Any]:
    duration_s = float(row["duration_s"])
    result = {
        "source_id": source_id,
        "source_pool": source_pool,
        "source_family_id": f"{source_pool}:{source_id}",
        "venue_source_family_id": f"{source_pool}:{source_id}",
        "duration_s": duration_s,
        "fps_inventory": fps,
        "expected_frame_count": _expected_frame_count(duration_s, fps),
        "cross_component_holdout_roles": list(CROSS_COMPONENT_HOLDOUT_ROLES.get(source_id, ())),
        "teacher_derived": True,
        "ground_truth": False,
        "split": "train",
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }
    return result


def load_pbvision_sources(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    videos = manifest.get("videos") if isinstance(manifest, Mapping) else None
    if not isinstance(videos, list):
        raise ValueError("pb.vision manifest must contain a videos list")

    canonical_ids = {
        _normalise_guard_text(source_id): source_id
        for source_id in PBVISION_TRAIN_IDS | COMPARE_ONLY_PBVISION_IDS
    }
    selected: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for row in videos:
        canonical_id = _normalise_guard_text(row.get("video_id") or "")
        source_id = canonical_ids.get(canonical_id)
        if source_id is not None:
            if source_id in observed_ids:
                raise ValueError(f"duplicate canonical pb.vision ID {source_id}")
            observed_ids.add(source_id)
    missing = (PBVISION_TRAIN_IDS | COMPARE_ONLY_PBVISION_IDS) - observed_ids
    if missing:
        raise ValueError(f"pb.vision manifest is missing preregistered IDs: {sorted(missing)}")

    for raw in videos:
        raw_source_id = str(raw.get("video_id") or "")
        source_id = canonical_ids.get(_normalise_guard_text(raw_source_id))
        if source_id is None:
            raise ValueError(f"unexpected non-preregistered pb.vision ID {raw_source_id}")
        sha = _normalise_sha256(raw.get("video_sha256") or "")
        if not sha:
            raise ValueError(f"pb.vision source {source_id} lacks an expected media SHA-256")
        offered_identity = {
            **dict(raw),
            "source_id": source_id,
            "expected_media_sha256": sha,
        }
        if source_id in COMPARE_ONLY_PBVISION_IDS:
            reason = quarantine_reason(offered_identity)
            if reason is None:
                raise AssertionError(f"compare-only source {source_id} escaped quarantine")
            refusals.append(
                {
                    "source_id": source_id,
                    "source_pool": "pbvision",
                    "expected_media_sha256": sha,
                    "reason": reason,
                    "structurally_excluded": True,
                    "verified": False,
                    "do_not_promote": True,
                    "production_eligible": False,
                }
            )
            continue
        assert_source_allowed(offered_identity)
        pinned_sha = PBVISION_MEDIA_SHA256_BY_ID[source_id]
        if sha != pinned_sha:
            raise FidelityError(
                f"pb.vision source {source_id} media SHA-256 is not bound to the committed registry: "
                f"expected {pinned_sha}, got {sha}"
            )
        source = _source_common(
            raw,
            source_pool="pbvision",
            source_id=source_id,
            fps=_parse_fps(str(raw["resolution"])),
        )
        source.update(
            {
                "title": raw.get("title"),
                "expected_media_sha256": sha,
                "expected_media_location": raw.get("video_location"),
                "media_identity_binding": "committed_pbvision_id_to_sha256_registry",
                "rights_posture": "owner_signed_full_usage_experiment_only",
            }
        )
        assert_source_allowed(source)
        selected.append(source)

    if len(selected) != EXPECTED_PB_VISION_SOURCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_PB_VISION_SOURCE_COUNT} non-compare pb.vision sources, got {len(selected)}"
        )
    return sorted(selected, key=lambda row: str(row["source_id"])), sorted(
        refusals, key=lambda row: str(row["source_id"])
    )


def load_harvest_sources(
    manifest_path: Path,
    repo_root: Path,
    *,
    expected_sha256: Mapping[str, str] = HARVEST_EXPECTED_SHA256,
) -> list[dict[str, Any]]:
    rows = _load_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError("harvest manifest must be a list")
    downloaded = [row for row in rows if row.get("status") == "downloaded"]
    observed = {str(row.get("id")) for row in downloaded}
    if observed != HARVEST_TRAIN_IDS:
        raise ValueError(
            f"downloaded harvest IDs differ from preregistration: missing={sorted(HARVEST_TRAIN_IDS - observed)}, "
            f"extra={sorted(observed - HARVEST_TRAIN_IDS)}"
        )

    selected: list[dict[str, Any]] = []
    for raw in downloaded:
        source_id = str(raw["id"])
        expected_sha = str(expected_sha256.get(source_id) or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError(f"harvest source {source_id} lacks a pinned media SHA-256")
        media_path = manifest_path.parent / str(raw["file"])
        if not media_path.is_file():
            raise FileNotFoundError(f"downloaded harvest media is absent: {media_path}")
        actual_sha = sha256_file(media_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"harvest source {source_id} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        source = _source_common(
            raw,
            source_pool="online_harvest_20260706",
            source_id=source_id,
            fps=float(raw["fps"]),
        )
        try:
            media_ref = media_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            media_ref = str(media_path.resolve())
        source.update(
            {
                "title": raw.get("title"),
                "channel": raw.get("channel"),
                "expected_media_sha256": expected_sha,
                "expected_media_location": media_ref,
                "license_field": raw.get("license_field"),
                "rights_posture": "owner_directed_experiment_only_source_license_unresolved",
            }
        )
        channel_slug = re.sub(r"[^a-z0-9]+", "-", str(raw.get("channel") or "unknown").lower()).strip("-")
        source["venue_source_family_id"] = (
            f"online_harvest_20260706:channel:{channel_slug}"
        )
        assert_source_allowed(source)
        selected.append(source)

    if len(selected) != EXPECTED_HARVEST_SOURCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_HARVEST_SOURCE_COUNT} downloaded harvest sources, got {len(selected)}"
        )
    return sorted(selected, key=lambda row: str(row["source_id"]))


def verify_teacher_checkpoint(model_manifest_path: Path, teacher_path: Path) -> dict[str, Any]:
    manifest = _load_json(model_manifest_path)
    models = manifest.get("models") if isinstance(manifest, Mapping) else None
    if not isinstance(models, list):
        raise ValueError("model manifest must contain a models list")
    matches = [row for row in models if row.get("id") == YOLO26M_ID]
    if len(matches) != 1:
        raise ValueError(f"model manifest must contain exactly one {YOLO26M_ID} entry")
    expected_sha = str(matches[0].get("sha256") or "").lower()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint is absent: {teacher_path}")
    actual_sha = sha256_file(teacher_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"teacher checkpoint SHA-256 mismatch: manifest {expected_sha}, local {actual_sha}"
        )
    return {
        "model_id": YOLO26M_ID,
        "checkpoint_path": teacher_path.as_posix(),
        "checkpoint_sha256": actual_sha,
        "manifest_path": model_manifest_path.as_posix(),
        "manifest_sha256": sha256_file(model_manifest_path),
        "class_filter": {"class_id": 0, "class_name": "person"},
        "confidence_min": DEFAULT_TEACHER_CONFIDENCE,
        "nms": "ultralytics_defaults_no_override",
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }


def _resolve_closed_path(p1_root: Path, relative: str) -> Path:
    candidate = (p1_root / relative).resolve()
    try:
        candidate.relative_to(p1_root.resolve())
    except ValueError as exc:
        raise FidelityError(f"P1 row escapes the closed artifact: {relative}") from exc
    if not candidate.is_file():
        raise FidelityError(f"P1 row file is missing: {candidate}")
    return candidate


def _row_file_digests(
    rows: Sequence[Mapping[str, Any]],
    p1_root: Path,
    cache: dict[Path, str],
    *,
    sort_rows: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    ordered_rows = sorted(rows, key=lambda value: str(value["sample_id"])) if sort_rows else rows
    for row in ordered_rows:
        record = {
            "sample_id": str(row["sample_id"]),
            "family_id": str(row["family_id"]),
            "source": str(row["source"]),
            "image": str(row["image"]),
            "label": str(row["label"]),
        }
        for kind in ("image", "label"):
            path = _resolve_closed_path(p1_root, record[kind])
            if path not in cache:
                cache[path] = sha256_file(path)
            record[f"{kind}_sha256"] = cache[path]
        digest.update((json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"))
        records.append(record)
    return digest.hexdigest(), records


def _walk_row_fields(value: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, child in value.items():
        yield str(key), child
        if isinstance(child, Mapping):
            yield from _walk_row_fields(child)


def _assert_human_validation_provenance(row: Mapping[str, Any], expected_split: str) -> None:
    if row.get("split") != expected_split:
        raise FidelityError(
            f"human validation row {row.get('sample_id')} changed split: "
            f"expected {expected_split}, got {row.get('split')}"
        )
    for raw_key, value in _walk_row_fields(row):
        key = _normalise_guard_text(raw_key)
        if key == "teacherderived" and value is not False:
            raise FidelityError(
                f"human-only validation provenance conflict for {row.get('sample_id')}: "
                "teacher_derived must be false"
            )
        if key == "groundtruth" and value is not True:
            raise FidelityError(
                f"human-only validation provenance conflict for {row.get('sample_id')}: "
                "ground_truth must be true"
            )
        if key in {"labelorigin", "annotationorigin"} and value is not None:
            if _normalise_guard_text(value) != _normalise_guard_text("roboflow_human_annotation"):
                raise FidelityError(
                    f"human-only validation provenance conflict for {row.get('sample_id')}: "
                    f"{raw_key}={value!r}"
                )
        if key in {
            "teachermodelid",
            "teachercheckpointsha256",
            "teacherconf",
            "teacherconfidencemin",
            "pseudolabel",
            "pseudoderived",
        } and value is not None:
            raise FidelityError(
                f"human-only validation provenance conflict for {row.get('sample_id')}: "
                f"unexpected {raw_key}"
            )
        if key in {"provenance", "origin"} and isinstance(value, str):
            canonical_value = _normalise_guard_text(value)
            if "teacher" in canonical_value or "pseudo" in canonical_value:
                raise FidelityError(
                    f"human-only validation provenance conflict for {row.get('sample_id')}: "
                    f"{raw_key}={value!r}"
                )


def _inspect_closed_p1(p1_root: Path) -> dict[str, Any]:
    manifest_path = p1_root / "dataset_manifest.json"
    train_list_path = p1_root / "train_family_balanced.txt"
    manifest = _load_json(manifest_path)
    rows = manifest.get("rows") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise FidelityError("closed P1 dataset manifest has no rows")
    if not train_list_path.is_file():
        raise FidelityError("closed P1 family-balanced train list is absent")

    split_rows = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "val", "test")
    }
    for row in split_rows["val"]:
        _assert_human_validation_provenance(row, "val")
    for row in split_rows["test"]:
        _assert_human_validation_provenance(row, "test")
    train_families = {str(row["family_id"]) for row in split_rows["train"]}
    val_families = {str(row["family_id"]) for row in split_rows["val"]}
    test_families = {str(row["family_id"]) for row in split_rows["test"]}
    if len(train_families) != EXPECTED_ANCHOR_FAMILY_COUNT:
        raise FidelityError(f"closed P1 must have seven anchor families, got {sorted(train_families)}")
    if val_families != {OD8AL_FAMILY} or test_families != {HEMEL_FAMILY}:
        raise FidelityError(
            f"closed P1 human holdouts changed: val={sorted(val_families)}, test={sorted(test_families)}"
        )

    by_image = {str(row["image"]): row for row in split_rows["train"]}
    train_list_bytes = train_list_path.read_bytes()
    train_entries = [line.strip() for line in train_list_bytes.decode("utf-8").splitlines() if line.strip()]
    anchor_rows: list[Mapping[str, Any]] = []
    for entry in train_entries:
        normalised = entry[2:] if entry.startswith("./") else entry
        row = by_image.get(normalised)
        if row is None:
            raise FidelityError(f"family-balanced entry is not in the closed train split: {entry}")
        anchor_rows.append(row)

    cache: dict[Path, str] = {}
    train_hash, _ = _row_file_digests(split_rows["train"], p1_root, cache)
    od8al_hash, od8al_records = _row_file_digests(split_rows["val"], p1_root, cache)
    hemel_hash, hemel_records = _row_file_digests(split_rows["test"], p1_root, cache)
    anchor_hash, anchor_records = _row_file_digests(
        anchor_rows, p1_root, cache, sort_rows=False
    )
    fidelity = {
        "hash_algorithm": (
            "sha256_over_path_and_file_sha256_records; closed split rows sorted by sample_id; "
            "anchor exposure rows retain frozen train-list order"
        ),
        "closed_manifest_sha256": sha256_file(manifest_path),
        "closed_train_list_sha256": _sha256_bytes(train_list_bytes),
        "closed_train_all_rows_sha256": train_hash,
        "anchor_exposure_rows_sha256": anchor_hash,
        "human_val_od8al_rows_sha256": od8al_hash,
        "human_val_hemel_rows_sha256": hemel_hash,
        "counts": {
            "closed_train_all": len(split_rows["train"]),
            "anchor_train_exposures": len(anchor_rows),
            "human_val_od8al": len(split_rows["val"]),
            "human_val_hemel": len(split_rows["test"]),
        },
        "direct_closed_file_references": True,
        "images_or_labels_reencoded": False,
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "train_list_path": train_list_path,
        "train_list_bytes": train_list_bytes,
        "train_list_sha256": _sha256_bytes(train_list_bytes),
        "train_entries": train_entries,
        "anchor_records": anchor_records,
        "od8al_records": od8al_records,
        "hemel_records": hemel_records,
        "fidelity": fidelity,
    }


def load_closed_p1(
    p1_root: Path,
    *,
    expected_fidelity: Mapping[str, str] = CLOSED_P1_FIDELITY_SHA256,
    registry_id: str = CLOSED_P1_HASH_REGISTRY_ID,
) -> dict[str, Any]:
    result = _inspect_closed_p1(p1_root)
    actual = result["fidelity"]
    for key in FROZEN_FIDELITY_HASH_KEYS:
        expected_sha = _normalise_sha256(expected_fidelity.get(key))
        if not expected_sha:
            raise FidelityError(f"closed P1 hash registry {registry_id} lacks valid {key}")
        if actual[key] != expected_sha:
            raise FidelityError(
                f"closed P1 fidelity drift for {key}: expected {expected_sha}, got {actual[key]}"
            )
    actual["closed_hash_registry_id"] = registry_id
    actual["closed_hash_registry_sha256"] = _sha256_bytes(
        _json_bytes({key: expected_fidelity[key] for key in FROZEN_FIDELITY_HASH_KEYS})
    )
    actual["closed_hash_binding_passed"] = True
    actual["closed_hashes_expected"] = {
        key: expected_fidelity[key] for key in FROZEN_FIDELITY_HASH_KEYS
    }
    return result


def assert_no_pseudo_in_validation(
    pseudo_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]
) -> None:
    pseudo_ids = {str(row["sample_id"]) for row in pseudo_rows}
    validation_ids = {str(row["sample_id"]) for row in validation_rows}
    if pseudo_ids & validation_ids:
        raise ValueError("pseudo and validation sample IDs overlap")
    for row in pseudo_rows:
        if row.get("split") != "train" or row.get("teacher_derived") is not True or row.get("ground_truth") is not False:
            raise ValueError("pseudo rows must be teacher-derived, non-ground-truth train rows")
    for row in validation_rows:
        _assert_human_validation_provenance(row, str(row.get("split")))
        if row.get("split") not in {"val", "test"}:
            raise ValueError("validation rows must retain their closed val/test split")
        if row.get("teacher_derived") is not False or row.get("ground_truth") is not True:
            raise ValueError("validation must be human-only and ground-truth")

    pseudo_content = _content_identity_index(pseudo_rows)
    validation_content = _content_identity_index(validation_rows)
    collisions = sorted(set(pseudo_content) & set(validation_content))
    if collisions:
        identity = collisions[0]
        raise ValueError(
            "pseudo and validation content identity overlap: "
            f"{identity} ({pseudo_content[identity]} vs {validation_content[identity]})"
        )

    pseudo_families = _source_family_identities(pseudo_rows)
    validation_families = _source_family_identities(validation_rows)
    family_overlap = sorted(pseudo_families & validation_families)
    if family_overlap:
        raise ValueError(
            f"pseudo and validation source-family lineage overlap: {family_overlap[0]}"
        )


def _content_identity_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    direct_digest_keys = {
        "contentsha256",
        "imagesha256",
        "decodedframesha256",
        "framesha256",
        "mediaframesha256",
    }
    media_digest_keys = {
        "expectedmediasha256",
        "mediasha256",
        "videosha256",
        "sourcesha256",
    }
    frame_keys = {"frameindex", "sourceframeindex", "decodedframeindex"}
    for row in rows:
        sample_id = str(row.get("sample_id") or "unknown")
        media_digests: set[str] = set()
        frame_values: set[str] = set()
        for raw_key, value in _walk_row_fields(row):
            key = _normalise_guard_text(raw_key)
            if key in direct_digest_keys:
                digest = _normalise_sha256(value)
                if digest:
                    result.setdefault(f"bytes:sha256:{digest}", sample_id)
            elif key in media_digest_keys:
                digest = _normalise_sha256(value)
                if digest:
                    media_digests.add(digest)
            elif key in frame_keys and value is not None:
                frame_values.add(_canonical_decoded_text(value).strip())
            elif key in {"contentidentity", "frameidentity"} and value is not None:
                result.setdefault(
                    f"declared:{_normalise_guard_text(value)}",
                    sample_id,
                )
        for media_digest in media_digests:
            result.setdefault(f"media:sha256:{media_digest}", sample_id)
            for frame_value in frame_values:
                result.setdefault(
                    f"media-frame:sha256:{media_digest}:frame:{frame_value}",
                    sample_id,
                )
    return result


def _source_family_identities(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        for key in ("source_family_id", "venue_source_family_id", "family_id"):
            if row.get(key) is not None:
                result.add(_normalise_guard_text(row[key]))
    return result


def _repo_reference(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validation_rows(records: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    validation_rows: list[dict[str, Any]] = []
    for row in records:
        if "split" in row and row.get("split") != split:
            raise FidelityError(
                f"human-only validation provenance conflict for {row.get('sample_id')}: "
                f"split must remain {split}"
            )
        candidate = {**dict(row), "split": split}
        _assert_human_validation_provenance(candidate, split)
        validation_rows.append(
            {
                **candidate,
                "teacher_derived": False,
                "ground_truth": True,
                "label_origin": "roboflow_human_annotation",
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            }
        )
    return validation_rows


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FidelityError(f"required final-list artifact is absent: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FidelityError(f"final list is not UTF-8: {path}") from exc
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise FidelityError(f"final list contains blank entries: {path}")
    return lines


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_list(path), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FidelityError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _resolve_final_image(entry: str, repo_root: Path) -> Path:
    decoded = entry
    for _ in range(MAX_CANONICAL_DECODE_PASSES):
        candidate = unicodedata.normalize("NFKC", unquote(decoded, errors="strict"))
        if candidate == decoded:
            break
        decoded = candidate
    path = Path(decoded)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise FidelityError(f"final-list image escapes repo root: {entry}") from exc
    if not resolved.is_file():
        raise FidelityError(f"final-list image is absent: {entry}")
    return resolved


def _verify_plan_artifact_index(out: Path) -> None:
    index_path = out / "artifact_index.json"
    index = _load_json(index_path)
    artifacts = index.get("artifacts") if isinstance(index, Mapping) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise FidelityError("person mixed artifact index is absent or empty")
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise FidelityError("person mixed artifact index contains a non-object row")
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise FidelityError(f"unsafe artifact-index path: {relative}")
        path = (out / relative).resolve()
        try:
            path.relative_to(out.resolve())
        except ValueError as exc:
            raise FidelityError(f"artifact-index path escapes pack: {relative}") from exc
        if not path.is_file():
            raise FidelityError(f"indexed plan artifact is absent: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise FidelityError(f"indexed plan artifact drifted: {relative}")


def validate_final_lists(
    *,
    repo_root: Path,
    p1_root: Path,
    out: Path,
    pseudo_train_list_path: Path,
    mixed_train_list_path: Path,
    report_path: Path | None = None,
    expected_closed_p1_fidelity: Mapping[str, str] = CLOSED_P1_FIDELITY_SHA256,
    closed_p1_registry_id: str = CLOSED_P1_HASH_REGISTRY_ID,
) -> dict[str, Any]:
    """Executable, fail-closed GPU materialization barrier for final YOLO lists."""

    repo_root = repo_root.resolve()
    p1_root = p1_root.resolve()
    out = out.resolve()
    if (out / "data.yaml").exists():
        raise FidelityError("data.yaml already exists; final-list validator must pass first")
    _verify_plan_artifact_index(out)
    pack_manifest = _load_json(out / "pack_manifest.json")
    validator_contract = pack_manifest.get("gpu_materialization_gate", {}).get(
        "final_list_validator"
    )
    if not isinstance(validator_contract, Mapping):
        raise FidelityError("pack manifest lacks the executable final-list validator contract")
    if validator_contract.get("path") != FINAL_LIST_VALIDATOR_RELATIVE_PATH.as_posix():
        raise FidelityError("pack manifest references the wrong final-list validator")
    current_validator_sha = sha256_file(Path(__file__).resolve())
    if validator_contract.get("sha256") != current_validator_sha:
        raise FidelityError("final-list validator code SHA-256 drifted after pack generation")
    if validator_contract.get("executable") is not True:
        raise FidelityError("pack manifest does not require an executable final-list validator")

    p1 = load_closed_p1(
        p1_root,
        expected_fidelity=expected_closed_p1_fidelity,
        registry_id=closed_p1_registry_id,
    )
    for key in FROZEN_FIDELITY_HASH_KEYS:
        if pack_manifest.get("anchor_and_validation_fidelity", {}).get(key) != p1["fidelity"][key]:
            raise FidelityError(f"pack manifest closed-P1 binding drifted for {key}")

    decode_rows = _read_jsonl(out / str(pack_manifest["artifacts"]["decode_plan"]))
    pseudo_entries = _read_list(pseudo_train_list_path)
    if len(pseudo_entries) != len(decode_rows):
        raise FidelityError(
            f"pseudo final-list length mismatch: expected {len(decode_rows)}, got {len(pseudo_entries)}"
        )

    od8al_validation = _validation_rows(p1["od8al_records"], "val")
    hemel_validation = _validation_rows(p1["hemel_records"], "test")
    validation_rows = od8al_validation + hemel_validation
    expected_od8al = [
        _repo_reference(p1_root / str(row["image"]), repo_root)
        for row in p1["od8al_records"]
    ]
    expected_hemel = [
        _repo_reference(p1_root / str(row["image"]), repo_root)
        for row in p1["hemel_records"]
    ]
    expected_validation_lists = {
        "od8al_val": expected_od8al,
        "hemel_val": expected_hemel,
        "all_val": expected_od8al + expected_hemel,
    }
    for artifact_key, expected_entries in expected_validation_lists.items():
        actual_entries = _read_list(out / str(pack_manifest["artifacts"][artifact_key]))
        if actual_entries != expected_entries:
            raise FidelityError(
                f"{artifact_key} is not derived exactly from the pinned closed P1 holdout rows"
            )

    resolved_pseudo: set[Path] = set()
    materialized_pseudo_rows: list[dict[str, Any]] = []
    for decode_row, entry in zip(decode_rows, pseudo_entries, strict=True):
        image_path = _resolve_final_image(entry, repo_root)
        if image_path in resolved_pseudo:
            raise FidelityError(f"pseudo final list repeats materialized image path: {entry}")
        resolved_pseudo.add(image_path)
        materialized_pseudo_rows.append(
            {
                **decode_row,
                "image": entry,
                "image_sha256": sha256_file(image_path),
            }
        )

    assert_no_pseudo_in_validation(materialized_pseudo_rows, validation_rows)

    anchor_refs = [
        _repo_reference(p1_root / str(row["image"]), repo_root)
        for row in p1["anchor_records"]
    ]
    expected_mixed: list[str] = []
    for index, pseudo_entry in enumerate(pseudo_entries):
        expected_mixed.extend([anchor_refs[index % len(anchor_refs)], pseudo_entry])
    actual_mixed = _read_list(mixed_train_list_path)
    if actual_mixed != expected_mixed:
        raise FidelityError(
            "mixed final list does not exactly realize the frozen 1:1 anchor/pseudo interleave"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_mixed_final_list_validation",
        "status": "PASS",
        "structural_validation_passed": True,
        "validator_path": FINAL_LIST_VALIDATOR_RELATIVE_PATH.as_posix(),
        "validator_sha256": current_validator_sha,
        "closed_p1_hash_registry_id": closed_p1_registry_id,
        "closed_p1_hash_binding_passed": True,
        "pseudo_rows": len(materialized_pseudo_rows),
        "mixed_exposures": len(actual_mixed),
        "human_validation_rows": len(validation_rows),
        "pseudo_train_list_sha256": sha256_file(pseudo_train_list_path),
        "mixed_train_list_sha256": sha256_file(mixed_train_list_path),
        "content_identity_overlap": 0,
        "source_family_overlap": 0,
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }
    if report_path is not None:
        _write(report_path, _json_bytes(result))
    return result


def build_pack(
    *,
    repo_root: Path,
    p1_root: Path,
    pbvision_manifest_path: Path,
    harvest_manifest_path: Path,
    model_manifest_path: Path,
    teacher_path: Path,
    out: Path,
    protocol: Protocol = Protocol(),
    expected_harvest_sha256: Mapping[str, str] = HARVEST_EXPECTED_SHA256,
    expected_closed_p1_fidelity: Mapping[str, str] = CLOSED_P1_FIDELITY_SHA256,
    closed_p1_registry_id: str = CLOSED_P1_HASH_REGISTRY_ID,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    p1_root = p1_root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    teacher = verify_teacher_checkpoint(model_manifest_path, teacher_path)
    teacher["confidence_min"] = protocol.teacher_confidence
    p1 = load_closed_p1(
        p1_root,
        expected_fidelity=expected_closed_p1_fidelity,
        registry_id=closed_p1_registry_id,
    )
    pbvision_sources, observed_refusals = load_pbvision_sources(pbvision_manifest_path)
    harvest_sources = load_harvest_sources(
        harvest_manifest_path, repo_root, expected_sha256=expected_harvest_sha256
    )
    sources = sorted(pbvision_sources + harvest_sources, key=lambda row: str(row["source_family_id"]))
    for source in sources:
        assert_source_allowed(source)

    counts = allocate_source_counts(sources, protocol)
    pseudo_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for source in sources:
        source_id = str(source["source_id"])
        count = counts[source_id]
        indices = uniform_frame_indices(int(source["expected_frame_count"]), count)
        per_source_rows.append(
            {
                "source_id": source_id,
                "source_pool": source["source_pool"],
                "source_family_id": source["source_family_id"],
                "venue_source_family_id": source["venue_source_family_id"],
                "expected_media_sha256": source["expected_media_sha256"],
                "expected_frame_count": source["expected_frame_count"],
                "planned_pseudo_frames": count,
                "fraction_of_pack": None,
                "cross_component_holdout_roles": source["cross_component_holdout_roles"],
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            }
        )
        for frame_index in indices:
            pseudo_rows.append(
                {
                    "sample_id": f"pseudo:{source['source_pool']}:{source_id}:frame:{frame_index:08d}",
                    "source_id": source_id,
                    "source_pool": source["source_pool"],
                    "source_family_id": source["source_family_id"],
                    "venue_source_family_id": source["venue_source_family_id"],
                    "frame_index": frame_index,
                    "expected_frame_count": source["expected_frame_count"],
                    "expected_media_location": source["expected_media_location"],
                    "expected_media_sha256": source["expected_media_sha256"],
                    "cross_component_holdout_roles": source["cross_component_holdout_roles"],
                    "split": "train",
                    "teacher_derived": True,
                    "ground_truth": False,
                    "teacher_model_id": YOLO26M_ID,
                    "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
                    "teacher_conf": None,
                    "teacher_confidence_min": protocol.teacher_confidence,
                    "teacher_class_id": 0,
                    "teacher_class_name": "person",
                    "teacher_nms": "ultralytics_defaults_no_override",
                    "materialization_status": "planned_gpu_teacher_inference_required",
                    "blind_spot_caveat": BLIND_SPOT_CAVEAT,
                    "verified": False,
                    "do_not_promote": True,
                    "production_eligible": False,
                }
            )

    planned_total = len(pseudo_rows)
    for row in per_source_rows:
        row["fraction_of_pack"] = round(int(row["planned_pseudo_frames"]) / planned_total, 12)
    per_family = []
    family_counts: Counter[str] = Counter(
        str(row["source_family_id"]) for row in pseudo_rows
    )
    for family_id in sorted(family_counts):
        per_family.append(
            {
                "source_family_id": family_id,
                "planned_pseudo_frames": family_counts[family_id],
                "fraction_of_pack": round(family_counts[family_id] / planned_total, 12),
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            }
        )

    per_venue_family = []
    venue_family_counts: Counter[str] = Counter(
        str(row["venue_source_family_id"]) for row in pseudo_rows
    )
    for family_id in sorted(venue_family_counts):
        per_venue_family.append(
            {
                "venue_source_family_id": family_id,
                "planned_pseudo_frames": venue_family_counts[family_id],
                "fraction_of_pack": round(venue_family_counts[family_id] / planned_total, 12),
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            }
        )

    od8al_validation = _validation_rows(p1["od8al_records"], "val")
    hemel_validation = _validation_rows(p1["hemel_records"], "test")
    validation_rows = od8al_validation + hemel_validation
    assert_no_pseudo_in_validation(pseudo_rows, validation_rows)

    anchor_shard_rows = [
        {
            **dict(row),
            "closed_list_position": index,
            "split": "train",
            "teacher_derived": False,
            "ground_truth": True,
            "label_origin": "roboflow_human_annotation",
            "verified": False,
            "do_not_promote": True,
            "production_eligible": False,
        }
        for index, row in enumerate(p1["anchor_records"])
    ]
    if not anchor_shard_rows:
        raise FidelityError("closed anchor shard is empty")

    anchor_ratio, pseudo_ratio = protocol.anchor_pseudo_ratio
    if (anchor_ratio, pseudo_ratio) != DEFAULT_ANCHOR_PSEUDO_RATIO:
        raise ValueError("only the preregistered 1:1 anchor:pseudo interleave is implemented")
    interleave_rows: list[dict[str, Any]] = []
    for pseudo_index, pseudo_row in enumerate(pseudo_rows):
        anchor_index = pseudo_index % len(anchor_shard_rows)
        anchor_row = anchor_shard_rows[anchor_index]
        interleave_rows.extend(
            [
                {
                    "position": len(interleave_rows),
                    "shard": "anchor",
                    "source_row_index": anchor_index,
                    "sample_id": anchor_row["sample_id"],
                    "image": anchor_row["image"],
                    "verified": False,
                    "do_not_promote": True,
                    "production_eligible": False,
                },
                {
                    "position": len(interleave_rows) + 1,
                    "shard": "pseudo",
                    "source_row_index": pseudo_index,
                    "sample_id": pseudo_row["sample_id"],
                    "image": None,
                    "verified": False,
                    "do_not_promote": True,
                    "production_eligible": False,
                },
            ]
        )

    source_family_count = sum(1 for count in family_counts.values() if count > 0)
    maximum_fraction = max(
        (row["fraction_of_pack"] for row in per_venue_family), default=0.0
    )
    target_range_met = PSEUDO_TARGET_RANGE[0] <= planned_total <= PSEUDO_TARGET_RANGE[1]
    planning_bars = {
        "pseudo_frame_target_range": {
            "after": planned_total,
            "target_min": PSEUDO_TARGET_RANGE[0],
            "target_max": PSEUDO_TARGET_RANGE[1],
            "pass": target_range_met,
        },
        "distinct_source_families": {
            "after": source_family_count,
            "target_min": MINIMUM_SOURCE_FAMILIES,
            "pass": source_family_count >= MINIMUM_SOURCE_FAMILIES,
        },
        "per_source_cap": {
            "after_max": max(counts.values(), default=0),
            "target_max": protocol.max_frames_per_source,
            "pass": max(counts.values(), default=0) <= protocol.max_frames_per_source,
        },
        "family_fraction_cap": {
            "after_max": maximum_fraction,
            "target_max": protocol.max_family_fraction,
            "pass": maximum_fraction <= protocol.max_family_fraction + 1e-12,
        },
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }

    count_tables = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_mixed_pseudo_pack_count_tables",
        "planned_pseudo_total": planned_total,
        "per_source": per_source_rows,
        "per_source_family": per_family,
        "per_venue_source_family": per_venue_family,
        "pool_totals": [
            {"source_pool": pool, "planned_pseudo_frames": count}
            for pool, count in sorted(Counter(str(row["source_pool"]) for row in pseudo_rows).items())
        ],
        "planning_bars": planning_bars,
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }

    quarantine_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_mixed_pseudo_pack_quarantine_refusals",
        "binding_rules": {
            "canonical_identity": (
                "bounded_repeated_percent_decode_then_unicode_NFKC_casefold_and_alnum_normalization"
            ),
            "protected_clip_ids": list(PROTECTED_CLIP_IDS),
            "protected_media_sha256": sorted(PROTECTED_MEDIA_SHA256),
            "pbvision_compare_only_ids": sorted(COMPARE_ONLY_PBVISION_IDS),
            "pbvision_compare_only_media_sha256": sorted(COMPARE_ONLY_MEDIA_SHA256),
            "pbvision_id_to_media_sha256": dict(sorted(PBVISION_MEDIA_SHA256_BY_ID.items())),
            "pbvision_id_to_media_sha256_registry_sha256": _sha256_bytes(
                _json_bytes(dict(sorted(PBVISION_MEDIA_SHA256_BY_ID.items())))
            ),
            "iynbd_derivative_token": IYNBD_DERIVATIVE_TOKEN,
        },
        "observed_refusals": observed_refusals,
        "all_selected_sources_passed": True,
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }

    data_steward_rows = [
        {
            **source,
            "planned_pseudo_frames": counts[str(source["source_id"])],
            "consumed_for": ["PERSON_mixed_pool_pseudo_train_experiment"],
            "never_allowed_for": ["validation", "evaluation", "promotion"],
            "blind_spot_caveat": BLIND_SPOT_CAVEAT,
        }
        for source in sources
    ]

    relative_out = _repo_reference(out, repo_root)
    human_val_paths = {
        "od8al": [
            _repo_reference(p1_root / str(row["image"]), repo_root) for row in p1["od8al_records"]
        ],
        "hemel": [
            _repo_reference(p1_root / str(row["image"]), repo_root) for row in p1["hemel_records"]
        ],
    }
    template_text = (
        "# TEMPLATE ONLY: GPU teacher inference must materialize and SHA-verify every pseudo row.\n"
        "# DO_NOT_PROMOTE; VERIFIED=0; no promotion path.\n"
        f"path: {repo_root.as_posix()}\n"
        f"train: {relative_out}/mixed_train.txt  # ABSENT until GPU materialization gate passes\n"
        f"val: {relative_out}/human_val_all.txt  # human-only closed P1 holdouts\n"
        "nc: 1\n"
        "names:\n"
        "  0: person\n"
    ).encode("utf-8")

    final_validation_report = out / "final_list_validation.json"
    final_list_validator = {
        "path": FINAL_LIST_VALIDATOR_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(Path(__file__).resolve()),
        "executable": True,
        "must_run_on_gpu_lane_final_materialized_files": True,
        "must_pass_before_data_yaml": True,
        "report_path": _repo_reference(final_validation_report, repo_root),
        "command": [
            ".venv/bin/python",
            FINAL_LIST_VALIDATOR_RELATIVE_PATH.as_posix(),
            "--repo-root",
            ".",
            "--validate-final-lists",
            "--p1-root",
            _repo_reference(p1_root, repo_root),
            "--out",
            relative_out,
            "--pseudo-train-list",
            _repo_reference(out / "pseudo_train.txt", repo_root),
            "--mixed-train-list",
            _repo_reference(out / "mixed_train.txt", repo_root),
            "--final-validation-report",
            _repo_reference(final_validation_report, repo_root),
        ],
        "checks": [
            "closed-P1 six-hash binding",
            "pack artifact-index integrity",
            "validation lists derived exactly from pinned closed P1",
            "pseudo versus validation decoded-image SHA-256 overlap",
            "pseudo versus validation media/frame identity overlap",
            "pseudo versus validation source-family lineage overlap",
            "exact frozen 1:1 anchor/pseudo final interleave",
        ],
    }

    paths = {
        "decode_plan": out / "decode_plan.jsonl",
        "anchor_shard": out / "anchor_train_shard.jsonl",
        "anchor_byte_copy": out / "anchor_train_closed_byte_copy.txt",
        "od8al_val": out / "human_val_od8al.txt",
        "hemel_val": out / "human_val_hemel.txt",
        "all_val": out / "human_val_all.txt",
        "interleave": out / "interleave_plan.jsonl",
        "count_tables": out / "count_tables.json",
        "ledger": out / "data_steward_ledger_rows.jsonl",
        "quarantines": out / "quarantine_refusals.json",
        "data_yaml_template": out / "data.yaml.template",
    }
    _write(paths["decode_plan"], _jsonl_bytes(pseudo_rows))
    _write(paths["anchor_shard"], _jsonl_bytes(anchor_shard_rows))
    _write(paths["anchor_byte_copy"], p1["train_list_bytes"])
    _write(paths["od8al_val"], ("\n".join(human_val_paths["od8al"]) + "\n").encode("utf-8"))
    _write(paths["hemel_val"], ("\n".join(human_val_paths["hemel"]) + "\n").encode("utf-8"))
    _write(
        paths["all_val"],
        ("\n".join(human_val_paths["od8al"] + human_val_paths["hemel"]) + "\n").encode("utf-8"),
    )
    _write(paths["interleave"], _jsonl_bytes(interleave_rows))
    _write(paths["count_tables"], _json_bytes(count_tables))
    _write(paths["ledger"], _jsonl_bytes(data_steward_rows))
    _write(paths["quarantines"], _json_bytes(quarantine_manifest))
    _write(paths["data_yaml_template"], template_text)

    protocol_override_reasons = []
    if protocol.cli_protocol_override:
        protocol_override_reasons.append("CLI_PROTOCOL_OVERRIDE")
    production_ineligibility_reasons = ["OWNER_DIRECTED_EXPERIMENT_NO_PROMOTION_PATH"]
    production_ineligibility_reasons.extend(protocol_override_reasons)
    pack_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_mixed_pseudo_pack_plan",
        "lane": "person_mixed_20260722",
        "status": "GPU_TEACHER_INFERENCE_REQUIRED",
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
        "production_ineligibility_reasons": production_ineligibility_reasons,
        "best_stack_delta": None,
        "amendment": {
            "roboflow_only_p2_closed": True,
            "mixed_pool_self_training_experiment_open": True,
            "PERSON_RF_POOL_TOO_THIN_stands": True,
            "no_promotion_path": True,
        },
        "blind_spot_caveat": BLIND_SPOT_CAVEAT,
        "protocol": {
            "seed": protocol.seed,
            "target_pseudo_frames": protocol.target_pseudo_frames,
            "max_frames_per_source": protocol.max_frames_per_source,
            "max_family_fraction": protocol.max_family_fraction,
            "uniform_sampling": "endpoint_inclusive_even_spacing_over_inventory_expected_frame_count",
            "anchor_pseudo_exposure_ratio": "1:1",
            "cli_protocol_override": protocol.cli_protocol_override,
            "protocol_defaults_unchanged": not protocol.cli_protocol_override,
            "verified": False,
            "do_not_promote": True,
            "production_eligible": False,
        },
        "teacher": teacher,
        "anchor_and_validation_fidelity": p1["fidelity"],
        "shards": {
            "anchor_train": {
                "status": "materialized_closed_p1_references",
                "rows": len(anchor_shard_rows),
                "path": paths["anchor_shard"].name,
                "closed_byte_copy_path": paths["anchor_byte_copy"].name,
                "closed_byte_copy_sha256": sha256_file(paths["anchor_byte_copy"]),
                "teacher_derived": False,
                "ground_truth": True,
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            },
            "pseudo_train": {
                "status": "planned_gpu_teacher_inference_required",
                "planned_rows": planned_total,
                "decode_plan_path": paths["decode_plan"].name,
                "materialized_yolo_list_path": "pseudo_train.txt",
                "teacher_derived": True,
                "ground_truth": False,
                "required_materialized_teacher_conf": f">={protocol.teacher_confidence:.2f}",
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            },
            "mixed_train": {
                "status": "blocked_until_pseudo_train_materialized",
                "planned_exposures": len(interleave_rows),
                "anchor_exposures": planned_total,
                "pseudo_exposures": planned_total,
                "anchor_pseudo_exposure_ratio": "1:1",
                "interleave_plan_path": paths["interleave"].name,
                "materialized_yolo_list_path": "mixed_train.txt",
                "verified": False,
                "do_not_promote": True,
                "production_eligible": False,
            },
        },
        "validation": {
            "human_only": True,
            "pseudo_rows": 0,
            "od8al_family": {
                "family_id": OD8AL_FAMILY,
                "rows": len(od8al_validation),
                "path": paths["od8al_val"].name,
            },
            "hemel_family": {
                "family_id": HEMEL_FAMILY,
                "rows": len(hemel_validation),
                "path": paths["hemel_val"].name,
            },
            "combined_path": paths["all_val"].name,
            "teacher_derived": False,
            "ground_truth": True,
            "verified": False,
            "do_not_promote": True,
            "production_eligible": False,
        },
        "count_tables": count_tables,
        "quarantines": quarantine_manifest,
        "experiment_bars": {
            "control": "anchor_only_same_closed_P1_anchor_and_human_holdouts",
            "candidate": "mixed_anchor_plus_pseudo",
            "aggregate_required": {
                "heldout_family_macro_F1_delta": ">0",
                "heldout_family_macro_mAP50_delta": ">0",
            },
            "per_family_required": {
                OD8AL_FAMILY: {"F1_delta": ">=0", "mAP50_delta": ">=0"},
                HEMEL_FAMILY: {"F1_delta": ">=0", "mAP50_delta": ">=0"},
            },
            "selection_or_promotion_allowed": False,
            "verified": False,
            "do_not_promote": True,
            "production_eligible": False,
        },
        "gpu_materialization_gate": {
            "data_yaml_template": paths["data_yaml_template"].name,
            "data_yaml_must_not_exist_before_gate": True,
            "final_list_validator": final_list_validator,
            "required": [
                "media SHA-256 verified for every source",
                "stock YOLO26m person-only inference at confidence >= 0.60",
                "actual teacher_conf recorded for every retained pseudo label",
                "pseudo_train.txt and mixed_train.txt materialized from the frozen plans",
                "referenced executable final-list validator passes on materialized lists",
            ],
            "complete": False,
            "verified": False,
            "do_not_promote": True,
            "production_eligible": False,
        },
        "artifacts": {key: path.name for key, path in paths.items()},
    }
    pack_manifest_path = out / "pack_manifest.json"
    _write(pack_manifest_path, _json_bytes(pack_manifest))

    indexed_paths = {**paths, "pack_manifest": pack_manifest_path}
    artifact_index = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_mixed_pseudo_pack_artifact_index",
        "artifacts": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for _, path in sorted(indexed_paths.items())
        ],
        "verified": False,
        "do_not_promote": True,
        "production_eligible": False,
    }
    _write(out / "artifact_index.json", _json_bytes(artifact_index))
    return pack_manifest


def _protocol_from_args(args: argparse.Namespace) -> Protocol:
    override = any(
        value is not None
        for value in (
            args.target_pseudo_frames,
            args.max_frames_per_source,
            args.max_family_fraction,
            args.teacher_confidence,
            args.seed,
        )
    )
    return Protocol(
        target_pseudo_frames=(
            DEFAULT_TARGET_PSEUDO_FRAMES
            if args.target_pseudo_frames is None
            else args.target_pseudo_frames
        ),
        max_frames_per_source=(
            DEFAULT_MAX_FRAMES_PER_SOURCE
            if args.max_frames_per_source is None
            else args.max_frames_per_source
        ),
        max_family_fraction=(
            DEFAULT_MAX_FAMILY_FRACTION
            if args.max_family_fraction is None
            else args.max_family_fraction
        ),
        teacher_confidence=(
            DEFAULT_TEACHER_CONFIDENCE
            if args.teacher_confidence is None
            else args.teacher_confidence
        ),
        seed=DEFAULT_SEED if args.seed is None else args.seed,
        cli_protocol_override=override,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the CPU decode/interleave plan for the owner-directed PERSON pseudo pack."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--p1-root",
        type=Path,
        default=Path("runs/lanes/person_p1_roboflow_20260721/roboflow_person"),
    )
    parser.add_argument(
        "--pbvision-manifest",
        type=Path,
        default=Path("data/pbvision_gallery_20260719/MANIFEST.json"),
    )
    parser.add_argument(
        "--harvest-manifest",
        type=Path,
        default=Path("data/online_harvest_20260706/manifest.json"),
    )
    parser.add_argument("--model-manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--teacher", type=Path, default=YOLO26M_RELATIVE_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/lanes/person_mixed_20260722"),
    )
    parser.add_argument(
        "--validate-final-lists",
        action="store_true",
        help="Run the executable fail-closed GPU final-list validation barrier.",
    )
    parser.add_argument("--pseudo-train-list", type=Path, default=None)
    parser.add_argument("--mixed-train-list", type=Path, default=None)
    parser.add_argument("--final-validation-report", type=Path, default=None)
    parser.add_argument("--target-pseudo-frames", type=int, default=None)
    parser.add_argument("--max-frames-per-source", type=int, default=None)
    parser.add_argument("--max-family-fraction", type=float, default=None)
    parser.add_argument("--teacher-confidence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    resolve = lambda value: value if value.is_absolute() else repo_root / value
    if args.validate_final_lists:
        resolved_out = resolve(args.out)
        pseudo_train_list = resolve(args.pseudo_train_list or resolved_out / "pseudo_train.txt")
        mixed_train_list = resolve(args.mixed_train_list or resolved_out / "mixed_train.txt")
        validation_report = resolve(
            args.final_validation_report or resolved_out / "final_list_validation.json"
        )
        result = validate_final_lists(
            repo_root=repo_root,
            p1_root=resolve(args.p1_root),
            out=resolved_out,
            pseudo_train_list_path=pseudo_train_list,
            mixed_train_list_path=mixed_train_list,
            report_path=validation_report,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    result = build_pack(
        repo_root=repo_root,
        p1_root=resolve(args.p1_root),
        pbvision_manifest_path=resolve(args.pbvision_manifest),
        harvest_manifest_path=resolve(args.harvest_manifest),
        model_manifest_path=resolve(args.model_manifest),
        teacher_path=resolve(args.teacher),
        out=resolve(args.out),
        protocol=_protocol_from_args(args),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "planned_pseudo_frames": result["count_tables"]["planned_pseudo_total"],
                "source_families": len(result["count_tables"]["per_source_family"]),
                "anchor_train_exposures": result["shards"]["anchor_train"]["rows"],
                "human_val_od8al": result["validation"]["od8al_family"]["rows"],
                "human_val_hemel": result["validation"]["hemel_family"]["rows"],
                "out": str(resolve(args.out)),
                "verified": False,
                "do_not_promote": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
