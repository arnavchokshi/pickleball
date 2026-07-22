#!/usr/bin/env python3
"""Ingest reviewed CVAT image-task court keypoints into the existing trainer format.

The emitted label contract is defined by
``scripts/racketsport/train_court_keypoint_heatmap.py``
(``load_real_court_keypoint_labels`` / ``court_keypoint_label_rows``): every
``<source>/labels/court_keypoints.json`` item contains exactly the 15 canonical
keypoint names, with visible points encoded as ``[x, y]`` and every unavailable
point encoded as JSON ``null``.

Quarantine order is deliberate and test-pinned:

1. freeze connected source/channel/venue families and the eight holdouts;
2. deny the permanent source and quarantine train/holdout family collisions;
3. densely compare every remaining package image pHash against every protected
   image pHash, still before looking at shapes;
4. inspect CVAT labels only for rows that survived both quarantines.

Protected image files and every decoded protected-video frame are opened only to
compute their pHash. They are never copied, linked, or otherwise staged in the
output corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


PRODUCTION_MANIFEST_ARTIFACT_TYPE = "racketsport_court_diversity_20260712_package_manifest"
PRODUCTION_MANIFEST_SHA256 = "c0243e9146152c5c46b5d0aebca9d571bfd39b6e90b34227d4024d09eabcdd7e"
SYNTHETIC_MANIFEST_ARTIFACT_TYPE = "synthetic_court_diversity_fixture"
PERMANENT_DENY_SOURCE_IDS: frozenset[str] = frozenset({"IYnbdRs1Jdk"})
EXPECTED_PERMANENT_DENY_ROW_COUNT = 3

HOLDOUT_SOURCE_IDS: tuple[str, ...] = (
    "1or-bXVM80M",
    "4qSoA-jwpVM",
    "C5YUQlqZqBY",
    "q3575jnmjJQ",
    "A9H6EWfXht0",
    "Se7M6ZKaC4Y",
    "a_HzWrwK6vM",
    "wv3aPJrDwK4",
)
REQUIRED_FLOOR_ANCHORS: frozenset[str] = frozenset(
    {"near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner"}
)
CANONICAL_KEYPOINT_NAMES: tuple[str, ...] = tuple(PICKLEBALL_COURT_KEYPOINT_NAMES)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
DEFAULT_PHASH_MAX_DISTANCE = 3
MIN_FLOOR_QUADRILATERAL_AREA_PX2 = 4.0
MIN_FLOOR_QUADRILATERAL_AREA_RATIO = 0.001
MIN_NORMALIZED_FLOOR_EDGE_LENGTH = 0.001
MAX_NORMALIZED_HOMOGRAPHY_CONDITION_NUMBER = 5.0e3

# ORCHESTRATOR RULING 2026-07-21: "PPA Tour" and "PPA Tour Asia" are one
# organizational family for conservative train/holdout collision detection.
ORGANIZATIONAL_FAMILY_ALIAS_MAP: dict[str, str] = {"PPA Tour Asia": "PPA Tour"}
ORGANIZATIONAL_FAMILY_ALIAS_MAP_SHA256 = "d947cb3a975cef27450bf7d77dfed8df9d33279ea9f240914ee1150791db1bdc"
_alias_map_bytes = json.dumps(
    ORGANIZATIONAL_FAMILY_ALIAS_MAP,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
if hashlib.sha256(_alias_map_bytes).hexdigest() != ORGANIZATIONAL_FAMILY_ALIAS_MAP_SHA256:
    raise RuntimeError("organizational family alias map does not match its pinned SHA-256")
_NORMALIZED_ORGANIZATIONAL_FAMILY_ALIAS_MAP = {
    " ".join(alias.casefold().split()): " ".join(canonical.casefold().split())
    for alias, canonical in ORGANIZATIONAL_FAMILY_ALIAS_MAP.items()
}

MIN_TRAIN_ROWS = 60
MIN_TRAIN_SOURCE_GROUPS = 15
MIN_HOLDOUT_ROWS_PER_GROUP = 2

SUFFICIENT_VERDICT = "COURT_DIVERSITY_ROWS_SUFFICIENT"
INSUFFICIENT_VERDICT = "COURT_DIVERSITY_ROWS_INSUFFICIENT"
ERROR_VERDICT = "COURT_DIVERSITY_INGEST_ERROR"

STATE_DENIED_PERMANENT_SOURCE = "DENIED_PERMANENT_SOURCE"
STATE_QUARANTINED_FAMILY_COLLISION = "QUARANTINED_FAMILY_COLLISION"
STATE_FROZEN_HOLDOUT = "FROZEN_HOLDOUT"
STATE_TRAIN_CANDIDATE = "TRAIN_CANDIDATE"


@dataclass(frozen=True)
class ManifestImage:
    file_name: str
    source_id: str
    path: Path
    width: int
    height: int
    frame_sha256: str
    declared_phash64_hex: str | None
    channel: str | None
    venue_group: str | None
    indoor_outdoor: str | None
    source_video_url: str | None

    @property
    def split(self) -> str:
        return "holdout" if self.source_id in HOLDOUT_SOURCE_IDS else "train"


@dataclass(frozen=True)
class ReviewedRow:
    image: ManifestImage
    keypoints: dict[str, list[float] | None]
    source_family_key: str | None = None


@dataclass(frozen=True)
class SourceAssignment:
    source_id: str
    source_family_key: str
    family_source_ids: tuple[str, ...]
    state: str
    split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be null or a non-empty string")
    return value


def _validate_manifest_mode(
    payload: dict[str, Any],
    manifest_sha256: str,
    *,
    synthetic_fixture: bool,
) -> None:
    artifact_type = payload.get("artifact_type")
    if synthetic_fixture:
        if artifact_type != SYNTHETIC_MANIFEST_ARTIFACT_TYPE:
            raise ValueError(
                "--synthetic-fixture requires artifact_type "
                f"{SYNTHETIC_MANIFEST_ARTIFACT_TYPE!r}; got {artifact_type!r}"
            )
        return
    if artifact_type != PRODUCTION_MANIFEST_ARTIFACT_TYPE:
        raise ValueError(
            "production package manifest artifact_type mismatch: "
            f"expected {PRODUCTION_MANIFEST_ARTIFACT_TYPE!r}, got {artifact_type!r}"
        )
    if manifest_sha256 != PRODUCTION_MANIFEST_SHA256:
        raise ValueError(
            "production package manifest SHA-256 mismatch: "
            f"expected {PRODUCTION_MANIFEST_SHA256}, got {manifest_sha256}"
        )


def _validate_runtime_contract(
    *,
    deny_sources: set[str],
    phash_max_distance: int,
    synthetic_fixture: bool,
) -> None:
    if deny_sources != PERMANENT_DENY_SOURCE_IDS:
        raise ValueError(
            "--deny-source must be exactly the permanent deny set "
            f"{sorted(PERMANENT_DENY_SOURCE_IDS)}; got {sorted(deny_sources)}"
        )
    if phash_max_distance < 0 or phash_max_distance > 64:
        raise ValueError("pHash max distance must be between 0 and 64")
    if not synthetic_fixture and phash_max_distance < DEFAULT_PHASH_MAX_DISTANCE:
        raise ValueError(
            "production --phash-max-distance cannot weaken the frozen threshold: "
            f"got {phash_max_distance}, require >= {DEFAULT_PHASH_MAX_DISTANCE}"
        )


def load_package_manifest(
    path: Path,
    *,
    synthetic_fixture: bool = False,
) -> tuple[dict[str, Any], str, list[ManifestImage]]:
    if not path.is_file():
        raise ValueError(f"package manifest does not exist: {path}")
    manifest_sha256 = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("package manifest root must be an object")
    _validate_manifest_mode(payload, manifest_sha256, synthetic_fixture=synthetic_fixture)
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("package manifest images must be a non-empty list")
    declared_count = payload.get("image_count")
    if declared_count is not None and declared_count != len(raw_images):
        raise ValueError(f"package manifest image_count mismatch: {declared_count} != {len(raw_images)}")

    frame_root = path.parent / "frames"
    images: list[ManifestImage] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_images):
        if not isinstance(raw, dict):
            raise ValueError(f"package manifest images[{index}] must be an object")
        file_name = raw.get("file_name")
        source_id = raw.get("source_id")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise ValueError(f"package manifest images[{index}].file_name must be a safe basename")
        if file_name in seen_names:
            raise ValueError(f"duplicate package image file_name: {file_name}")
        if not isinstance(source_id, str) or not source_id or any(char in source_id for char in "/\\"):
            raise ValueError(f"package manifest images[{index}].source_id must be a safe non-empty ID")
        if not file_name.startswith(source_id + "__"):
            raise ValueError(f"package image {file_name} does not begin with its source_id {source_id}__")
        resolution = raw.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"package manifest images[{index}].resolution must be [width, height]")
        width = _positive_int(resolution[0], f"images[{index}].resolution[0]")
        height = _positive_int(resolution[1], f"images[{index}].resolution[1]")
        frame_sha256 = raw.get("frame_sha256")
        if not isinstance(frame_sha256, str) or len(frame_sha256) != 64:
            raise ValueError(f"package manifest images[{index}].frame_sha256 must be a SHA-256 hex digest")
        try:
            int(frame_sha256, 16)
        except ValueError as exc:
            raise ValueError(f"package manifest images[{index}].frame_sha256 must be hexadecimal") from exc
        declared_phash = raw.get("phash64_hex")
        if declared_phash is not None:
            if not isinstance(declared_phash, str) or len(declared_phash) != 16:
                raise ValueError(f"package manifest images[{index}].phash64_hex must be 16 hex characters")
            try:
                int(declared_phash, 16)
            except ValueError as exc:
                raise ValueError(f"package manifest images[{index}].phash64_hex must be hexadecimal") from exc
        image_path = frame_root / file_name
        if not image_path.is_file():
            raise ValueError(f"package image does not exist: {image_path}")
        actual_sha256 = _sha256(image_path)
        if actual_sha256 != frame_sha256.lower():
            raise ValueError(
                f"package image SHA-256 mismatch for {file_name}: expected {frame_sha256}, got {actual_sha256}"
            )
        strata = raw.get("strata")
        if strata is not None and not isinstance(strata, dict):
            raise ValueError(f"package manifest images[{index}].strata must be an object when present")
        images.append(
            ManifestImage(
                file_name=file_name,
                source_id=source_id,
                path=image_path,
                width=width,
                height=height,
                frame_sha256=actual_sha256,
                declared_phash64_hex=declared_phash.lower() if declared_phash is not None else None,
                channel=_optional_string(raw.get("channel"), f"images[{index}].channel"),
                venue_group=_optional_string(raw.get("venue_group"), f"images[{index}].venue_group"),
                indoor_outdoor=_optional_string(
                    (strata or {}).get("indoor_outdoor"), f"images[{index}].strata.indoor_outdoor"
                ),
                source_video_url=_optional_string(raw.get("source_video_url"), f"images[{index}].source_video_url"),
            )
        )
        seen_names.add(file_name)

    declared_sources = payload.get("distinct_source_video_count")
    actual_sources = len({image.source_id for image in images})
    if declared_sources is not None and declared_sources != actual_sources:
        raise ValueError(
            f"package manifest distinct_source_video_count mismatch: {declared_sources} != {actual_sources}"
        )
    permanent_deny_rows = [image for image in images if image.source_id in PERMANENT_DENY_SOURCE_IDS]
    if len(permanent_deny_rows) != EXPECTED_PERMANENT_DENY_ROW_COUNT:
        raise ValueError(
            "package manifest must contain exactly "
            f"{EXPECTED_PERMANENT_DENY_ROW_COUNT} rows from the permanent deny set "
            f"{sorted(PERMANENT_DENY_SOURCE_IDS)}; got {len(permanent_deny_rows)}"
        )
    return payload, manifest_sha256, images


def protected_media_paths(protected_root: Path) -> tuple[list[Path], list[Path]]:
    if not protected_root.is_dir():
        raise ValueError(f"protected root does not exist or is not a directory: {protected_root}")
    image_paths = sorted(
        path for path in protected_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    video_paths = sorted(
        path for path in protected_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not image_paths and not video_paths:
        raise ValueError(f"protected root contains no image files or videos to pHash: {protected_root}")
    return image_paths, video_paths


def _phash_is_protected(candidate_hash: Any, protected_hash: Any, *, max_distance: int) -> bool:
    """Keep the inclusive 1-through-3 production collision boundary explicit and testable."""

    return int(candidate_hash - protected_hash) <= max_distance


def dense_phash_screen(
    images: Iterable[ManifestImage],
    *,
    protected_root: Path,
    max_distance: int,
) -> dict[str, Any]:
    if max_distance < 0 or max_distance > 64:
        raise ValueError("pHash max distance must be between 0 and 64")
    try:
        import cv2
        import imagehash
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency is present in the project venv
        raise RuntimeError("dense pHash screening requires OpenCV, Pillow, and ImageHash") from exc

    protected_image_paths, protected_video_paths = protected_media_paths(protected_root)
    protected_hashes: list[tuple[str, Any]] = []
    for path in protected_image_paths:
        try:
            with Image.open(path) as pil_image:
                protected_hashes.append((str(path), imagehash.phash(pil_image)))
        except Exception as exc:
            raise ValueError(f"failed to pHash protected frame {path}: {exc}") from exc

    protected_video_rows: list[dict[str, Any]] = []
    for path in protected_video_paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"failed to open protected video for pHash: {path}")
        advertised_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded_frames = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                protected_hashes.append(
                    (f"{path}#frame={decoded_frames}", imagehash.phash(Image.fromarray(frame_rgb)))
                )
                decoded_frames += 1
        finally:
            capture.release()
        if advertised_frames > 0 and decoded_frames != advertised_frames:
            raise ValueError(
                f"protected video decode was not exhaustive for {path}: "
                f"advertised {advertised_frames}, decoded {decoded_frames}"
            )
        if decoded_frames == 0:
            raise ValueError(f"protected video decoded zero frames for pHash: {path}")
        protected_video_rows.append(
            {
                "path": str(path),
                "advertised_frames": advertised_frames,
                "decoded_frames": decoded_frames,
            }
        )

    hits: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    comparison_count = 0
    for image in images:
        try:
            with Image.open(image.path) as pil_image:
                width, height = pil_image.size
                if (width, height) != (image.width, image.height):
                    raise ValueError(
                        f"manifest resolution {(image.width, image.height)} != decoded resolution {(width, height)}"
                    )
                candidate_hash = imagehash.phash(pil_image)
        except Exception as exc:
            raise ValueError(f"failed to pHash package image {image.file_name}: {exc}") from exc
        candidate_hex = str(candidate_hash)
        if image.declared_phash64_hex is not None and candidate_hex != image.declared_phash64_hex:
            raise ValueError(
                f"package image pHash mismatch for {image.file_name}: "
                f"expected {image.declared_phash64_hex}, got {candidate_hex}"
            )
        image_hashes[image.file_name] = candidate_hex
        # Do not short-circuit after a hit: the contract is dense all-pairs comparison.
        for protected_frame, protected_hash in protected_hashes:
            comparison_count += 1
            distance = int(candidate_hash - protected_hash)
            if _phash_is_protected(candidate_hash, protected_hash, max_distance=max_distance):
                hits.append(
                    {
                        "file_name": image.file_name,
                        "source_id": image.source_id,
                        "package_phash64_hex": candidate_hex,
                        "protected_frame": protected_frame,
                        "protected_phash64_hex": str(protected_hash),
                        "hamming_distance": distance,
                    }
                )
    return {
        "algorithm": "imagehash.phash_8x8_64bit",
        "max_hamming_distance": max_distance,
        "protected_image_file_count": len(protected_image_paths),
        "protected_video_count": len(protected_video_paths),
        "protected_video_frame_count": sum(row["decoded_frames"] for row in protected_video_rows),
        "protected_frame_count": len(protected_hashes),
        "protected_videos": protected_video_rows,
        "candidate_image_count": len(image_hashes),
        "comparison_count": comparison_count,
        "hits": hits,
        "hit_file_names": sorted({str(hit["file_name"]) for hit in hits}),
        "image_hashes": image_hashes,
    }


def _expected_export_shards(
    manifest_payload: dict[str, Any],
    *,
    images: list[ManifestImage],
    required: bool,
) -> dict[str, frozenset[str]] | None:
    raw_shards = manifest_payload.get("shards")
    if raw_shards is None and not required:
        return None
    if not isinstance(raw_shards, list) or len(raw_shards) != 4:
        raise ValueError("package manifest shards must define exactly four CVAT export shards")
    expected: dict[str, frozenset[str]] = {}
    all_names: list[str] = []
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, dict):
            raise ValueError(f"package manifest shards[{index}] must be an object")
        shard_name = raw.get("shard_name")
        file_names = raw.get("file_names")
        if not isinstance(shard_name, str) or not shard_name or Path(shard_name).name != shard_name:
            raise ValueError(f"package manifest shards[{index}].shard_name must be a safe basename")
        if shard_name in expected:
            raise ValueError(f"duplicate package manifest shard_name: {shard_name}")
        if not isinstance(file_names, list) or not file_names or not all(
            isinstance(name, str) and Path(name).name == name for name in file_names
        ):
            raise ValueError(f"package manifest shards[{index}].file_names must be safe basenames")
        if len(set(file_names)) != len(file_names):
            raise ValueError(f"package manifest shard {shard_name} contains duplicate file names")
        declared_count = raw.get("image_count")
        if declared_count is not None and declared_count != len(file_names):
            raise ValueError(
                f"package manifest shard {shard_name} image_count mismatch: {declared_count} != {len(file_names)}"
            )
        expected[shard_name] = frozenset(file_names)
        all_names.extend(file_names)
    duplicates = sorted(name for name in set(all_names) if all_names.count(name) > 1)
    if duplicates:
        raise ValueError(f"package manifest shards duplicate images across shards: {duplicates[:10]}")
    manifest_names = {image.file_name for image in images}
    if set(all_names) != manifest_names:
        missing = sorted(manifest_names - set(all_names))
        extra = sorted(set(all_names) - manifest_names)
        raise ValueError(f"package manifest shards do not partition images: missing={missing[:10]}, extra={extra[:10]}")
    return expected


def _xml_documents(
    export_path: Path,
    *,
    expected_shards: dict[str, frozenset[str]] | None = None,
) -> list[tuple[str, bytes, frozenset[str] | None]]:
    paths: list[Path]
    shard_by_path: dict[Path, tuple[str, frozenset[str]]] = {}
    if expected_shards is not None:
        if not export_path.is_dir():
            raise ValueError("production CVAT export must be the four-zip export directory")
        expected_file_names = {f"{shard_name}_annotations.zip" for shard_name in expected_shards}
        actual_paths = sorted(export_path.glob("*.zip"))
        actual_file_names = {path.name for path in actual_paths}
        if actual_file_names != expected_file_names:
            missing = sorted(expected_file_names - actual_file_names)
            extra = sorted(actual_file_names - expected_file_names)
            raise ValueError(f"CVAT export shard set mismatch: missing={missing}, extra={extra}")
        paths = actual_paths
        for path in paths:
            shard_name = path.name[: -len("_annotations.zip")]
            shard_by_path[path] = (shard_name, expected_shards[shard_name])
    elif export_path.is_dir():
        paths = sorted(export_path.glob("*.zip"))
        if not paths:
            raise ValueError(f"CVAT export directory contains no zip files: {export_path}")
    elif export_path.is_file():
        paths = [export_path]
    else:
        raise ValueError(f"CVAT export does not exist: {export_path}")

    documents: list[tuple[str, bytes, frozenset[str] | None]] = []
    for path in paths:
        if path.suffix.lower() == ".xml":
            documents.append((str(path), path.read_bytes(), None))
            continue
        if path.suffix.lower() != ".zip":
            raise ValueError(f"CVAT export must be an XML file, zip file, or directory of zip files: {path}")
        try:
            with zipfile.ZipFile(path) as archive:
                xml_names = sorted(name for name in archive.namelist() if name.lower().endswith(".xml"))
                if not xml_names:
                    raise ValueError(f"CVAT export zip contains no XML document: {path}")
                if expected_shards is not None and len(xml_names) != 1:
                    raise ValueError(f"CVAT shard export must contain exactly one XML document: {path}")
                for name in xml_names:
                    expected_names = shard_by_path[path][1] if path in shard_by_path else None
                    documents.append((f"{path}:{name}", archive.read(name), expected_names))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid CVAT export zip: {path}") from exc
    return documents


def _float_coordinate(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _shape_attribute(shape: ET.Element, name: str) -> str | None:
    values = [attribute.text or "" for attribute in shape.findall("attribute") if attribute.get("name") == name]
    if len(values) > 1:
        raise ValueError(f"point has duplicate {name!r} attributes")
    return values[0] if values else None


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_properly_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (ab_c > 0 > ab_d or ab_d > 0 > ab_c) and (cd_a > 0 > cd_b or cd_b > 0 > cd_a)


def _validate_floor_geometry(
    keypoints: dict[str, list[float] | None],
    *,
    image: ManifestImage,
) -> None:
    """Reject four-corner layouts that cannot support a finite invertible court homography."""

    ordered_names = (
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
    )
    points = [tuple(float(value) for value in keypoints[name] or []) for name in ordered_names]
    if any(len(point) != 2 for point in points):
        raise ValueError("missing required floor anchors")
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance_sq = sum((points[left][axis] - points[right][axis]) ** 2 for axis in (0, 1))
            if distance_sq <= 1.0e-12:
                raise ValueError(
                    "duplicate floor anchors: "
                    f"{ordered_names[left]} and {ordered_names[right]} are coincident"
                )

    if _segments_properly_intersect(points[0], points[1], points[2], points[3]) or _segments_properly_intersect(
        points[1], points[2], points[3], points[0]
    ):
        raise ValueError("self-intersecting/crossed four-corner configuration")

    cross_products = [_orientation(points[index], points[(index + 1) % 4], points[(index + 2) % 4]) for index in range(4)]
    if any(abs(value) <= 1.0e-10 for value in cross_products):
        raise ValueError("collinear floor anchors cannot define an invertible court homography")
    if not (all(value > 0 for value in cross_products) or all(value < 0 for value in cross_products)):
        raise ValueError("four-corner configuration is not a convex, invertible court quadrilateral")

    signed_area_twice = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    area_px2 = abs(signed_area_twice) * 0.5
    minimum_area = max(
        MIN_FLOOR_QUADRILATERAL_AREA_PX2,
        image.width * image.height * MIN_FLOOR_QUADRILATERAL_AREA_RATIO,
    )
    if area_px2 < minimum_area:
        raise ValueError(
            f"near-zero-area four-corner configuration: {area_px2:.6g}px^2 < {minimum_area:.6g}px^2"
        )

    normalized_points = [
        (point[0] / float(image.width), point[1] / float(image.height))
        for point in points
    ]
    normalized_edge_lengths = [
        math.dist(normalized_points[index], normalized_points[(index + 1) % 4])
        for index in range(4)
    ]
    minimum_edge = min(normalized_edge_lengths)
    if minimum_edge < MIN_NORMALIZED_FLOOR_EDGE_LENGTH:
        raise ValueError(
            "subpixel/short floor edge cannot support stable court geometry: "
            f"normalized length {minimum_edge:.6g} < {MIN_NORMALIZED_FLOOR_EDGE_LENGTH:.6g}"
        )

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a project dependency
        raise RuntimeError("court geometry validation requires NumPy") from exc
    canonical = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    matrix_rows: list[list[float]] = []
    targets: list[float] = []
    for (u, v), (x, y) in zip(canonical, normalized_points):
        matrix_rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        targets.append(x)
        matrix_rows.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        targets.append(y)
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > MAX_NORMALIZED_HOMOGRAPHY_CONDITION_NUMBER:
        raise ValueError(
            "non-invertible/ill-conditioned normalized court homography "
            f"(condition={condition:.6g}, max={MAX_NORMALIZED_HOMOGRAPHY_CONDITION_NUMBER:.6g})"
        )
    try:
        solution = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("non-invertible court homography") from exc
    homography = np.asarray(
        [solution[0:3], solution[3:6], [solution[6], solution[7], 1.0]],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(homography))
    if not np.isfinite(homography).all() or not math.isfinite(determinant) or abs(determinant) <= 1.0e-12:
        raise ValueError("non-finite or non-invertible court homography")


def parse_reviewed_keypoints(image_element: ET.Element, image: ManifestImage) -> dict[str, list[float] | None]:
    width_text = image_element.get("width")
    height_text = image_element.get("height")
    try:
        xml_width = int(width_text or "")
        xml_height = int(height_text or "")
    except ValueError as exc:
        raise ValueError("CVAT image width/height must be integers") from exc
    if (xml_width, xml_height) != (image.width, image.height):
        raise ValueError(
            f"CVAT image resolution {(xml_width, xml_height)} != manifest resolution {(image.width, image.height)}"
        )

    keypoints: dict[str, list[float] | None] = {name: None for name in CANONICAL_KEYPOINT_NAMES}
    seen_labels: set[str] = set()
    for shape in image_element:
        if shape.tag != "points":
            raise ValueError(f"unexpected CVAT shape type {shape.tag!r}; court labels must be points")
        label = shape.get("label")
        if label not in keypoints:
            raise ValueError(f"unexpected court keypoint label {label!r}")
        if label in seen_labels:
            raise ValueError(f"duplicate court keypoint label {label!r}")
        seen_labels.add(label)

        custom_source = _shape_attribute(shape, "source")
        is_owner = custom_source in {None, "owner"} and shape.get("source", "manual") == "manual"
        is_visible = shape.get("occluded", "0") != "1" and shape.get("outside", "0") != "1"
        if not is_owner or not is_visible:
            continue
        raw_points = shape.get("points")
        if not isinstance(raw_points, str) or ";" in raw_points:
            raise ValueError(f"court keypoint {label!r} must contain exactly one x,y point")
        parts = raw_points.split(",")
        if len(parts) != 2:
            raise ValueError(f"court keypoint {label!r} must contain exactly one x,y point")
        x = _float_coordinate(parts[0], f"{label}.x")
        y = _float_coordinate(parts[1], f"{label}.y")
        if x < 0 or x > image.width or y < 0 or y > image.height:
            raise ValueError(f"court keypoint {label!r} is outside the image bounds")
        keypoints[label] = [x, y]

    missing_required = sorted(name for name in REQUIRED_FLOOR_ANCHORS if keypoints[name] is None)
    if missing_required:
        raise ValueError("missing required floor anchors: " + ", ".join(missing_required))
    _validate_floor_geometry(keypoints, image=image)
    return keypoints


def read_cvat_export(
    export_path: Path,
    *,
    images: list[ManifestImage],
    prelabel_excluded_states: dict[str, str],
    phash_denied_names: set[str],
    assignments: dict[str, SourceAssignment],
    expected_shards: dict[str, frozenset[str]] | None,
) -> tuple[list[ReviewedRow], list[dict[str, str]], set[str]]:
    by_name = {image.file_name: image for image in images}
    seen: set[str] = set()
    usable: list[ReviewedRow] = []
    rejected: list[dict[str, str]] = []
    for document_name, raw_xml, expected_names in _xml_documents(export_path, expected_shards=expected_shards):
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            raise ValueError(f"invalid CVAT XML in {document_name}: {exc}") from exc
        if root.tag != "annotations" or root.findtext("version") != "1.1":
            raise ValueError(f"{document_name} is not a CVAT for images 1.1 export")
        document_image_names = [
            Path(element.get("name") or "").name for element in root.findall("image")
        ]
        if expected_names is not None and set(document_image_names) != expected_names:
            missing = sorted(expected_names - set(document_image_names))
            extra = sorted(set(document_image_names) - expected_names)
            raise ValueError(
                f"CVAT shard image reconciliation failed for {document_name}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        if len(document_image_names) != len(set(document_image_names)):
            raise ValueError(f"CVAT shard contains duplicate images: {document_name}")
        for image_element in root.findall("image"):
            raw_name = image_element.get("name")
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"CVAT image without a name in {document_name}")
            file_name = Path(raw_name).name
            if file_name not in by_name:
                raise ValueError(f"CVAT export contains image absent from package manifest: {raw_name}")
            if file_name in seen:
                raise ValueError(f"CVAT export contains duplicate image: {file_name}")
            seen.add(file_name)
            image = by_name[file_name]

            # These branches intentionally precede iteration over child shapes. Tests put invalid
            # poison shapes on quarantined rows so this ordering cannot regress silently.
            if file_name in prelabel_excluded_states:
                continue
            if file_name in phash_denied_names:
                continue
            try:
                keypoints = parse_reviewed_keypoints(image_element, image)
            except ValueError as exc:
                rejected.append({"file_name": file_name, "source_id": image.source_id, "reason": str(exc)})
                continue
            usable.append(
                ReviewedRow(
                    image=image,
                    keypoints=keypoints,
                    source_family_key=assignments[image.source_id].source_family_key,
                )
            )

    expected = set(by_name)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"CVAT export is missing {len(missing)} package images: {missing[:10]}")
    return usable, rejected, seen


def _lineage_for_group(rows: list[ManifestImage]) -> dict[str, Any]:
    def one(field: str) -> str | None:
        values = {getattr(row, field) for row in rows}
        if len(values) != 1:
            raise ValueError(f"source {rows[0].source_id} has inconsistent {field} lineage: {sorted(values)}")
        return next(iter(values))

    return {
        "source_id": rows[0].source_id,
        "channel_group": one("channel"),
        "venue_group": one("venue_group"),
        "indoor_outdoor": one("indoor_outdoor"),
        "source_video_url": one("source_video_url"),
    }


def _normalized_family_value(value: str | None, *, field: str, source_id: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"source {source_id} is missing required {field} family lineage")
    normalized = " ".join(value.casefold().split())
    if field == "channel":
        return _NORMALIZED_ORGANIZATIONAL_FAMILY_ALIAS_MAP.get(normalized, normalized)
    return normalized


def build_source_assignments(
    images: list[ManifestImage],
    *,
    deny_sources: set[str],
) -> tuple[dict[str, SourceAssignment], dict[str, dict[str, Any]]]:
    """Build connected source families sharing either channel or venue lineage."""

    manifest_by_source: dict[str, list[ManifestImage]] = defaultdict(list)
    for image in images:
        manifest_by_source[image.source_id].append(image)
    missing_holdouts = sorted(set(HOLDOUT_SOURCE_IDS) - set(manifest_by_source))
    if missing_holdouts:
        raise ValueError(f"package manifest is missing frozen holdout source IDs: {missing_holdouts}")
    lineages = {source_id: _lineage_for_group(rows) for source_id, rows in manifest_by_source.items()}
    parent = {source_id: source_id for source_id in manifest_by_source}

    def find(source_id: str) -> str:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen_token: dict[tuple[str, str], str] = {}
    normalized_by_source: dict[str, tuple[str, str]] = {}
    for source_id, lineage in sorted(lineages.items()):
        channel = _normalized_family_value(lineage["channel_group"], field="channel", source_id=source_id)
        venue = _normalized_family_value(lineage["venue_group"], field="venue", source_id=source_id)
        normalized_by_source[source_id] = (channel, venue)
        for token in (("channel", channel), ("venue", venue)):
            if token in seen_token:
                union(source_id, seen_token[token])
            else:
                seen_token[token] = source_id

    component_sources: dict[str, list[str]] = defaultdict(list)
    for source_id in manifest_by_source:
        component_sources[find(source_id)].append(source_id)

    source_to_family: dict[str, str] = {}
    families: dict[str, dict[str, Any]] = {}
    for source_ids in sorted((sorted(values) for values in component_sources.values()), key=lambda values: values[0]):
        channels = sorted({normalized_by_source[source_id][0] for source_id in source_ids})
        venues = sorted({normalized_by_source[source_id][1] for source_id in source_ids})
        canonical = json.dumps({"channels": channels, "venues": venues}, sort_keys=True, separators=(",", ":"))
        family_key = f"family:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
        family = {
            "source_family_key": family_key,
            "source_ids": source_ids,
            "channel_groups": sorted({lineages[source_id]["channel_group"] for source_id in source_ids}),
            "venue_groups": sorted({lineages[source_id]["venue_group"] for source_id in source_ids}),
        }
        families[family_key] = family
        for source_id in source_ids:
            source_to_family[source_id] = family_key

    holdout_family_keys = {source_to_family[source_id] for source_id in HOLDOUT_SOURCE_IDS if source_id in source_to_family}
    assignments: dict[str, SourceAssignment] = {}
    for source_id in sorted(manifest_by_source):
        family_key = source_to_family[source_id]
        family_source_ids = tuple(families[family_key]["source_ids"])
        if source_id in deny_sources:
            state = STATE_DENIED_PERMANENT_SOURCE
            split = "denied"
        elif source_id in HOLDOUT_SOURCE_IDS:
            state = STATE_FROZEN_HOLDOUT
            split = "holdout"
        elif family_key in holdout_family_keys:
            state = STATE_QUARANTINED_FAMILY_COLLISION
            split = "quarantined"
        else:
            state = STATE_TRAIN_CANDIDATE
            split = "train"
        assignments[source_id] = SourceAssignment(
            source_id=source_id,
            source_family_key=family_key,
            family_source_ids=family_source_ids,
            state=state,
            split=split,
        )
    return assignments, families


def build_source_split(
    images: list[ManifestImage],
    *,
    deny_sources: set[str],
    usable_rows: list[ReviewedRow],
    assignments: dict[str, SourceAssignment],
    families: dict[str, dict[str, Any]],
    corpus_root: Path,
) -> dict[str, Any]:
    manifest_by_source: dict[str, list[ManifestImage]] = defaultdict(list)
    usable_by_source: dict[str, list[ReviewedRow]] = defaultdict(list)
    for image in images:
        manifest_by_source[image.source_id].append(image)
    for row in usable_rows:
        usable_by_source[row.image.source_id].append(row)

    groups: list[dict[str, Any]] = []
    for source_id, source_images in sorted(manifest_by_source.items()):
        lineage = _lineage_for_group(source_images)
        assignment = assignments[source_id]
        groups.append(
            {
                **lineage,
                "split": assignment.split,
                "state": assignment.state,
                "pre_label_inspection_assignment": True,
                "package_row_count": len(source_images),
                "usable_row_count": len(usable_by_source[source_id]),
                "source_family_key": assignment.source_family_key,
                "family_source_ids": list(assignment.family_source_ids),
            }
        )
    train_datasets = sorted(
        source_id
        for source_id, assignment in assignments.items()
        if assignment.split == "train" and usable_by_source[source_id]
    )
    holdout_datasets = list(HOLDOUT_SOURCE_IDS)
    repo_relative_corpus_root = Path(os.path.relpath(corpus_root.resolve(), ROOT)).as_posix()
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diversity_source_split",
        "partition_unit": "connected_source_channel_venue_family",
        "frame_random_split": False,
        "path_base": "repo_root",
        "corpus_root": repo_relative_corpus_root,
        "real_root": (Path(repo_relative_corpus_root) / "train").as_posix(),
        "train_datasets": train_datasets,
        "holdout_datasets": holdout_datasets,
        "holdout_source_ids": list(HOLDOUT_SOURCE_IDS),
        "train_source_ids": train_datasets,
        "denied_source_ids": sorted(source_id for source_id in manifest_by_source if source_id in deny_sources),
        "quarantined_family_collision_source_ids": sorted(
            source_id
            for source_id, assignment in assignments.items()
            if assignment.state == STATE_QUARANTINED_FAMILY_COLLISION
        ),
        "holdout": {
            "frozen": True,
            "source_ids": list(HOLDOUT_SOURCE_IDS),
            "datasets": holdout_datasets,
            "family_keys": sorted({assignments[source_id].source_family_key for source_id in HOLDOUT_SOURCE_IDS}),
        },
        "organizational_family_aliases": {
            "ruling": "ORCHESTRATOR RULING 2026-07-21",
            "map": ORGANIZATIONAL_FAMILY_ALIAS_MAP,
            "sha256": ORGANIZATIONAL_FAMILY_ALIAS_MAP_SHA256,
        },
        "families": [families[key] for key in sorted(families)],
        "groups": groups,
    }


def evaluate_gate(usable_rows: list[ReviewedRow]) -> dict[str, Any]:
    train_rows = [row for row in usable_rows if row.image.split == "train"]
    holdout_rows = [row for row in usable_rows if row.image.split == "holdout"]
    train_families = sorted(
        {
            row.source_family_key
            or "family:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "channels": [_normalized_family_value(row.image.channel, field="channel", source_id=row.image.source_id)],
                        "venues": [
                            _normalized_family_value(row.image.venue_group, field="venue", source_id=row.image.source_id)
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:16]
            for row in train_rows
        }
    )
    holdout_families = sorted({row.source_family_key for row in holdout_rows if row.source_family_key is not None})
    holdout_counts = {
        source_id: sum(1 for row in holdout_rows if row.image.source_id == source_id)
        for source_id in HOLDOUT_SOURCE_IDS
    }
    failures: list[str] = []
    if len(train_rows) < MIN_TRAIN_ROWS:
        failures.append(f"usable train rows {len(train_rows)} < {MIN_TRAIN_ROWS}")
    if len(train_families) < MIN_TRAIN_SOURCE_GROUPS:
        failures.append(f"usable train family groups {len(train_families)} < {MIN_TRAIN_SOURCE_GROUPS}")
    thin_holdouts = {source_id: count for source_id, count in holdout_counts.items() if count < MIN_HOLDOUT_ROWS_PER_GROUP}
    if thin_holdouts:
        failures.append(
            "holdout groups below two valid rows: "
            + ", ".join(f"{source_id}={count}" for source_id, count in sorted(thin_holdouts.items()))
        )
    return {
        "passed": not failures,
        "verdict": SUFFICIENT_VERDICT if not failures else INSUFFICIENT_VERDICT,
        "thresholds": {
            "min_usable_train_rows": MIN_TRAIN_ROWS,
            "min_train_family_groups": MIN_TRAIN_SOURCE_GROUPS,
            "min_valid_rows_per_holdout_group": MIN_HOLDOUT_ROWS_PER_GROUP,
            "required_holdout_group_count": len(HOLDOUT_SOURCE_IDS),
        },
        "observed": {
            "usable_train_rows": len(train_rows),
            "train_family_group_count": len(train_families),
            "train_family_groups": train_families,
            "usable_holdout_rows": len(holdout_rows),
            "holdout_family_group_count": len(holdout_families),
            "holdout_family_groups": holdout_families,
            "holdout_rows_by_source": holdout_counts,
        },
        "failure_reasons": failures,
    }


def write_corpus(
    staging_root: Path,
    *,
    output_root: Path,
    usable_rows: list[ReviewedRow],
    manifest_sha256: str,
) -> None:
    rows_by_split_source: dict[tuple[str, str], list[ReviewedRow]] = defaultdict(list)
    for row in usable_rows:
        rows_by_split_source[(row.image.split, row.image.source_id)].append(row)

    for (split, source_id), source_rows in sorted(rows_by_split_source.items()):
        resolutions = {(row.image.width, row.image.height) for row in source_rows}
        if len(resolutions) != 1:
            raise ValueError(f"source {source_id} has multiple image resolutions, which one label payload cannot encode")
        width, height = next(iter(resolutions))
        labels_dir = staging_root / split / source_id / "labels"
        staging_frame_dir = labels_dir / "court_keypoint_frames"
        final_frame_dir = output_root / split / source_id / "labels" / "court_keypoint_frames"
        real_root_relative_frame_dir = final_frame_dir.relative_to(output_root / split)
        items: list[dict[str, Any]] = []
        for index, row in enumerate(sorted(source_rows, key=lambda value: value.image.file_name), start=1):
            frame_name = f"frame_{index:06d}{row.image.path.suffix.lower()}"
            staging_frame_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.image.path, staging_frame_dir / frame_name)
            items.append(
                {
                    "frame": frame_name,
                    "status": "reviewed",
                    "keypoints": row.keypoints,
                    "provenance": {
                        "original_file_name": row.image.file_name,
                        "original_source_id": row.image.source_id,
                        "source_video_url": row.image.source_video_url,
                        "channel_group": row.image.channel,
                        "venue_group": row.image.venue_group,
                        "indoor_outdoor": row.image.indoor_outdoor,
                        "frame_sha256": row.image.frame_sha256,
                        "package_manifest_sha256": manifest_sha256,
                        "partition": split,
                        "partition_unit": "connected_source_channel_venue_family",
                        "source_family_key": row.source_family_key,
                    },
                }
            )
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_court_keypoint_labels",
            "clip": source_id,
            "annotation": {"items": items},
            "frames": {
                "available_review_frame_count": len(items),
                "frame_count": len(items),
                "frame_dir": real_root_relative_frame_dir.as_posix(),
                "path_base": "corpus_root",
                "label_coordinate_space": [width, height],
                "source_resolution": [width, height],
            },
            "review": {
                "status": "reviewed",
                "reviewer": "owner_cvat_court_diversity_20260721",
                "independent_reviewed_count": len(items),
                "note": (
                    "CVAT image-task points inspected after source-ID denial and dense protected-frame pHash screening. "
                    "All 15 canonical names are present; unavailable visible points are JSON null."
                ),
            },
        }
        _write_json(labels_dir / "court_keypoints.json", payload)


def ingest_cvat_court_images(
    *,
    package_manifest: Path,
    cvat_export: Path,
    deny_sources: set[str],
    protected_root: Path,
    out: Path,
    phash_max_distance: int = DEFAULT_PHASH_MAX_DISTANCE,
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    _validate_runtime_contract(
        deny_sources=deny_sources,
        phash_max_distance=phash_max_distance,
        synthetic_fixture=synthetic_fixture,
    )
    out = out.resolve()
    if not synthetic_fixture:
        try:
            out.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("production output must live under the repository root for portable C1 paths") from exc
    if out.exists():
        raise ValueError(f"output path already exists; refusing to overwrite: {out}")

    manifest_payload, manifest_sha256, images = load_package_manifest(
        package_manifest,
        synthetic_fixture=synthetic_fixture,
    )
    assignments, families = build_source_assignments(images, deny_sources=deny_sources)
    expected_shards = _expected_export_shards(
        manifest_payload,
        images=images,
        required=not synthetic_fixture,
    )

    # Freeze family-aware assignments entirely from manifest lineage before pHash and labels.
    prelabel_excluded_states = {
        image.file_name: assignments[image.source_id].state
        for image in images
        if assignments[image.source_id].state
        in {STATE_DENIED_PERMANENT_SOURCE, STATE_QUARANTINED_FAMILY_COLLISION}
    }
    source_denied_names = {
        image.file_name
        for image in images
        if assignments[image.source_id].state == STATE_DENIED_PERMANENT_SOURCE
    }
    permanent_manifest_names = {
        image.file_name for image in images if image.source_id in PERMANENT_DENY_SOURCE_IDS
    }
    if (
        len(source_denied_names) != EXPECTED_PERMANENT_DENY_ROW_COUNT
        or source_denied_names != permanent_manifest_names
    ):
        raise AssertionError(
            "permanent deny exclusion mismatch: "
            f"manifest={sorted(permanent_manifest_names)}, excluded={sorted(source_denied_names)}"
        )
    family_quarantined_names = {
        image.file_name
        for image in images
        if assignments[image.source_id].state == STATE_QUARANTINED_FAMILY_COLLISION
    }
    phash_candidates = [image for image in images if image.file_name not in prelabel_excluded_states]
    phash = dense_phash_screen(phash_candidates, protected_root=protected_root, max_distance=phash_max_distance)
    phash_denied_names = set(phash["hit_file_names"])

    usable_rows, rejected_rows, seen = read_cvat_export(
        cvat_export,
        images=images,
        prelabel_excluded_states=prelabel_excluded_states,
        phash_denied_names=phash_denied_names,
        assignments=assignments,
        expected_shards=expected_shards,
    )
    if any(assignments[row.image.source_id].split != row.image.split for row in usable_rows):
        raise AssertionError("pre-label family split changed during label inspection")

    gate = evaluate_gate(usable_rows)
    train_rows = [row for row in usable_rows if row.image.split == "train"]
    holdout_rows = [row for row in usable_rows if row.image.split == "holdout"]
    label_rejected_count = len(rejected_rows)
    counts = {
        "cvat_present": len(seen),
        # A row earns reviewed/usable only after owner-manual labels and geometry validate.
        "reviewed": len(usable_rows),
        "usable": len(usable_rows),
        "protected_denied": len(source_denied_names) + len(phash_denied_names),
        "train": len(train_rows),
        "holdout": len(holdout_rows),
        "rejected": label_rejected_count + len(family_quarantined_names),
        "label_rejected": label_rejected_count,
        "source_denied": len(source_denied_names),
        "phash_denied": len(phash_denied_names),
        "family_quarantined": len(family_quarantined_names),
    }
    if counts["cvat_present"] != counts["usable"] + counts["protected_denied"] + counts["rejected"]:
        raise AssertionError(f"count reconciliation failed: {counts}")

    source_split = build_source_split(
        images,
        deny_sources=deny_sources,
        usable_rows=usable_rows,
        assignments=assignments,
        families=families,
        corpus_root=out,
    )
    repo_relative_corpus_root = Path(os.path.relpath(out, ROOT)).as_posix()
    artifacts = {
        "path_base": "repo_root",
        "corpus_root": repo_relative_corpus_root,
        "train_root": (Path(repo_relative_corpus_root) / "train").as_posix(),
        "holdout_root": (Path(repo_relative_corpus_root) / "holdout").as_posix(),
        "source_split": (Path(repo_relative_corpus_root) / "source_split.json").as_posix(),
        "ingest_report": (Path(repo_relative_corpus_root) / "ingest_report.json").as_posix(),
    }
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_court_image_ingest_report",
        "verdict": gate["verdict"],
        "counts": counts,
        "gate": gate,
        "package_manifest": str(package_manifest),
        "package_manifest_sha256": manifest_sha256,
        "cvat_export": str(cvat_export),
        "synthetic_fixture": synthetic_fixture,
        "denied_source_ids": sorted(deny_sources),
        "fixed_holdout_source_ids": list(HOLDOUT_SOURCE_IDS),
        "family_collision_quarantine": {
            "state": STATE_QUARANTINED_FAMILY_COLLISION,
            "source_ids": source_split["quarantined_family_collision_source_ids"],
            "row_count": len(family_quarantined_names),
        },
        "organizational_family_aliases": source_split["organizational_family_aliases"],
        "court_keypoints_format_definition": "scripts/racketsport/train_court_keypoint_heatmap.py",
        "court_keypoints_format_loader": "load_real_court_keypoint_labels / court_keypoint_label_rows",
        "phash_guard": {
            key: value for key, value in phash.items() if key != "image_hashes"
        },
        "rejected_rows": rejected_rows,
        "artifacts": artifacts,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        write_corpus(
            staging,
            output_root=out,
            usable_rows=usable_rows,
            manifest_sha256=manifest_sha256,
        )
        _write_json(staging / "source_split.json", source_split)
        _write_json(staging / "ingest_report.json", report)
        staging.replace(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest CVAT for images 1.1 court-keypoint labels with source and pHash quarantine guards."
    )
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--cvat-export", type=Path, required=True)
    parser.add_argument("--deny-source", action="append", dest="deny_sources", required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help=(
            "Explicit test-only mode for artifact_type synthetic_court_diversity_fixture. "
            "Normal invocation always enforces the pinned production artifact type, SHA, four shards, and pHash floor."
        ),
    )
    parser.add_argument(
        "--phash-max-distance",
        type=int,
        default=DEFAULT_PHASH_MAX_DISTANCE,
        help=(
            "Maximum 64-bit pHash Hamming distance treated as a protected-frame hit (default: 3). "
            "Production accepts 3 or a stricter/larger value, never 0-2."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = ingest_cvat_court_images(
            package_manifest=args.package_manifest,
            cvat_export=args.cvat_export,
            deny_sources=set(args.deny_sources),
            protected_root=args.protected_root,
            out=args.out,
            phash_max_distance=args.phash_max_distance,
            synthetic_fixture=args.synthetic_fixture,
        )
    except Exception as exc:
        print(json.dumps({"verdict": ERROR_VERDICT, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
