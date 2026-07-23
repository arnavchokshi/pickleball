#!/usr/bin/env python3
"""Ingest owner sequential court labels without fabricating invisible points.

The owner UI records visible keypoints, explicit point skips, and whole-frame
unsupported-view exclusions.  This adapter converts positive frames into the
existing ``court_keypoints.json`` training contract while retaining rejected
frames in a separate negative-view manifest for the structured-v3 supported-
view head.

Missing point decisions are accepted only with the explicit
``--owner-omissions-are-invisible`` flag.  They become JSON ``null`` and never
receive guessed coordinates.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_structured_solver import (
    FLOOR_KEYPOINT_NAMES,
    solve_best_floor_court,
)
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


PACKAGE_ARTIFACT_TYPE = "racketsport_court_labelpack3_owner_click_package"
RESULTS_ARTIFACT_TYPE = "racketsport_court_diversity_owner_sequential_labels"
RESULTS_AUTHORITY = "owner_reviewed"
VALID_STATUSES = {"unreviewed", "in_progress", "reviewed", "reviewed_partial", "excluded"}
VALID_EXCLUSION_REASONS = {"sideways_view", "fisheye", "camera_too_low", "bad_angle"}
CANONICAL_NAMES = tuple(PICKLEBALL_COURT_KEYPOINT_NAMES)
CANONICAL_SET = frozenset(CANONICAL_NAMES)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


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


def _xy(value: Any, *, width: int, height: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item coordinate")
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"{field} coordinates must be numeric")
    result = [float(x), float(y)]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field} coordinates must be finite")
    if not (0.0 <= result[0] <= width and 0.0 <= result[1] <= height):
        raise ValueError(f"{field} is outside the declared {width}x{height} image")
    return [round(component, 3) for component in result]


def _prior_training_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = _read_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("prior training manifest must contain a rows list")
    hashes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("image_sha256"), str):
            raise ValueError(f"prior training row {index} requires image_sha256")
        hashes.add(row["image_sha256"])
    return hashes


def _geometry_audit(keypoints: dict[str, list[float] | None]) -> dict[str, Any]:
    observations = {
        name: {
            "xy": value,
            "confidence": 1.0,
            "visibility": 1.0,
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
        }
        for name, value in keypoints.items()
        if name in FLOOR_KEYPOINT_NAMES and value is not None
    }
    if len(observations) < 4:
        return {
            "available": False,
            "floor_observation_count": len(observations),
            "reason": "fewer_than_four_visible_floor_points",
        }
    solved = solve_best_floor_court(
        observations,
        max_hypotheses=64,
        shortlist_size=8,
        inlier_threshold_px=10.0,
    )
    ignored = [
        item
        for item in solved.get("ignored_observations", [])
        if isinstance(item, dict) and item.get("semantic") in observations
    ]
    residual_outliers = [
        item
        for item in ignored
        if item.get("reason") in {"residual_outlier", "duplicate_location"}
    ]
    residuals = solved.get("residual_stats_px", {})
    p95 = residuals.get("p95") if isinstance(residuals, dict) else None
    flagged = (
        solved.get("status") not in {"solved_best_effort", "prior_only_best_effort"}
        or len(residual_outliers) >= 2
        or (isinstance(p95, (int, float)) and float(p95) > 10.0)
    )
    return {
        "available": True,
        "floor_observation_count": len(observations),
        "solver_status": solved.get("status"),
        "inlier_count": len(solved.get("inliers", [])),
        "residual_outlier_count": len(residual_outliers),
        "residual_outliers": [
            {
                "name": item.get("semantic"),
                "reason": item.get("reason"),
                "residual_px": item.get("residual_px"),
            }
            for item in residual_outliers
        ],
        "residual_stats_px": residuals,
        "flagged": flagged,
    }


def ingest_labelpack(
    *,
    package_manifest_path: Path,
    results_path: Path,
    out: Path,
    owner_omissions_are_invisible: bool,
    prior_training_manifest: Path | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    package = _read_object(package_manifest_path)
    results = _read_object(results_path)
    if package.get("artifact_type") != PACKAGE_ARTIFACT_TYPE or package.get("schema_version") != 1:
        raise ValueError("unsupported court label-pack manifest")
    if results.get("artifact_type") != RESULTS_ARTIFACT_TYPE or results.get("schema_version") != 1:
        raise ValueError("unsupported court label-pack results")
    if results.get("authority") != RESULTS_AUTHORITY:
        raise ValueError("court label-pack results must be owner_reviewed")
    if tuple(results.get("label_order", [])) != tuple(package.get("label_order", [])):
        raise ValueError("package and results label order differ")
    if set(results.get("label_order", [])) != CANONICAL_SET:
        raise ValueError("results label order is not the canonical 15-point taxonomy")
    if package.get("protocol_exclusions", {}).get("selected_identity_overlap_count") != 0:
        raise ValueError("package reports protected identity overlap")

    raw_images = package.get("images")
    raw_items = results.get("items")
    if not isinstance(raw_images, list) or not isinstance(raw_items, dict):
        raise ValueError("package images and results items are required")
    images: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_images):
        if not isinstance(raw, dict) or not isinstance(raw.get("file_name"), str):
            raise ValueError(f"package image {index} is malformed")
        name = raw["file_name"]
        if name in images:
            raise ValueError(f"duplicate package image: {name}")
        images[name] = raw
    if set(raw_items) != set(images):
        missing = sorted(set(images) - set(raw_items))
        extra = sorted(set(raw_items) - set(images))
        raise ValueError(f"results/package item mismatch missing={missing} extra={extra}")

    prior_hashes = _prior_training_hashes(prior_training_manifest)
    package_root = package_manifest_path.parent
    normalized: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    omitted_point_count = 0
    explicitly_skipped_point_count = 0
    labeled_point_count = 0
    geometry_flagged: list[dict[str, Any]] = []

    for file_name in sorted(images):
        image = images[file_name]
        raw = raw_items[file_name]
        if not isinstance(raw, dict):
            raise ValueError(f"result item must be an object: {file_name}")
        status = str(raw.get("status", "unreviewed"))
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}: {file_name}")
        status_counts[status] += 1
        resolution = image.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"missing resolution: {file_name}")
        width = _positive_int(resolution[0], f"{file_name}.width")
        height = _positive_int(resolution[1], f"{file_name}.height")
        relative_path = image.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"missing relative_path: {file_name}")
        frame_path = package_root / relative_path
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        frame_sha = _sha256(frame_path)
        if frame_sha != image.get("frame_sha256"):
            raise ValueError(f"frame SHA-256 mismatch: {file_name}")

        raw_points = raw.get("keypoints", {})
        raw_skipped = raw.get("skipped_points", {})
        raw_reasons = raw.get("exclusion_reasons", [])
        if not isinstance(raw_points, dict) or not isinstance(raw_skipped, dict) or not isinstance(raw_reasons, list):
            raise ValueError(f"malformed result fields: {file_name}")
        extra_points = sorted(set(raw_points) - CANONICAL_SET)
        extra_skips = sorted(set(raw_skipped) - CANONICAL_SET)
        if extra_points or extra_skips:
            raise ValueError(f"unexpected point names for {file_name}: points={extra_points} skips={extra_skips}")
        overlap = sorted(set(raw_points) & {name for name, value in raw_skipped.items() if value})
        if overlap:
            raise ValueError(f"points cannot be both labeled and skipped for {file_name}: {overlap}")
        reasons = [str(reason) for reason in raw_reasons]
        if any(reason not in VALID_EXCLUSION_REASONS for reason in reasons):
            raise ValueError(f"invalid exclusion reason: {file_name}")

        if status == "excluded":
            if raw_points:
                raise ValueError(f"excluded frame contains keypoints: {file_name}")
            if not reasons:
                raise ValueError(f"excluded frame requires an exclusion reason: {file_name}")
            for reason in reasons:
                exclusion_counts[reason] += 1
            normalized.append(
                {
                    "kind": "unsupported",
                    "file_name": file_name,
                    "frame_path": frame_path,
                    "frame_sha256": frame_sha,
                    "source_sha256": image.get("source_sha256"),
                    "width": width,
                    "height": height,
                    "venue_id": image.get("venue_id"),
                    "venue": image.get("venue"),
                    "workspace": image.get("workspace"),
                    "reasons": reasons,
                }
            )
            continue
        if not raw_points:
            raise ValueError(f"non-excluded frame contains no visible labels: {file_name}")

        points = {
            name: _xy(value, width=width, height=height, field=f"{file_name}.{name}")
            for name, value in raw_points.items()
        }
        explicit_skips = {name for name, value in raw_skipped.items() if value}
        missing = CANONICAL_SET - set(points) - explicit_skips
        if missing and not owner_omissions_are_invisible:
            raise ValueError(
                f"{file_name} has undecided points {sorted(missing)}; "
                "pass --owner-omissions-are-invisible only after explicit owner confirmation"
            )
        keypoints: dict[str, list[float] | None] = {
            name: points.get(name) for name in CANONICAL_NAMES
        }
        omitted_point_count += len(missing)
        explicitly_skipped_point_count += len(explicit_skips)
        labeled_point_count += len(points)
        audit = _geometry_audit(keypoints)
        if audit.get("flagged"):
            geometry_flagged.append({"file_name": file_name, **audit})
        source_sha = image.get("source_sha256")
        normalized.append(
            {
                "kind": "positive",
                "file_name": file_name,
                "frame_path": frame_path,
                "frame_sha256": frame_sha,
                "source_sha256": source_sha,
                "prior_training_exact_source_overlap": isinstance(source_sha, str) and source_sha in prior_hashes,
                "width": width,
                "height": height,
                "venue_id": image.get("venue_id"),
                "venue": image.get("venue"),
                "workspace": image.get("workspace"),
                "source_path": image.get("source_path"),
                "owner_result_status": status,
                "explicitly_skipped_points": sorted(explicit_skips),
                "omitted_points": sorted(missing),
                "keypoints": keypoints,
                "geometry_audit": audit,
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        train_root = temporary / "train"
        unsupported_root = temporary / "unsupported"
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        unsupported_rows: list[dict[str, Any]] = []
        for row in normalized:
            venue_id = row.get("venue_id")
            if not isinstance(venue_id, str) or not venue_id:
                raise ValueError(f"missing venue_id: {row['file_name']}")
            if row["kind"] == "positive":
                grouped[(venue_id, int(row["width"]), int(row["height"]))].append(row)
            else:
                target_dir = unsupported_root / venue_id / "frames"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / row["file_name"]
                shutil.copy2(row["frame_path"], target)
                unsupported_rows.append(
                    {
                        "file_name": row["file_name"],
                        "image": target.relative_to(temporary).as_posix(),
                        "frame_sha256": row["frame_sha256"],
                        "source_sha256": row["source_sha256"],
                        "venue_id": venue_id,
                        "venue": row["venue"],
                        "workspace": row["workspace"],
                        "reasons": row["reasons"],
                        "supported_view": False,
                    }
                )

        clip_count = 0
        for (venue_id, width, height), rows in sorted(grouped.items()):
            clip = venue_id if len({(w, h) for v, w, h in grouped if v == venue_id}) == 1 else f"{venue_id}__{width}x{height}"
            clip_root = train_root / clip
            frame_dir = clip_root / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            items: list[dict[str, Any]] = []
            for frame_index, row in enumerate(sorted(rows, key=lambda value: value["file_name"]), start=1):
                suffix = row["frame_path"].suffix.lower() or ".jpg"
                frame_name = f"frame_{frame_index:06d}{suffix}"
                shutil.copy2(row["frame_path"], frame_dir / frame_name)
                items.append(
                    {
                        "frame": frame_name,
                        "keypoints": row["keypoints"],
                        "status": "reviewed",
                        "provenance": {
                            "original_file_name": row["file_name"],
                            "frame_sha256": row["frame_sha256"],
                            "source_sha256": row["source_sha256"],
                            "source_path": row["source_path"],
                            "workspace": row["workspace"],
                            "venue_id": venue_id,
                            "venue": row["venue"],
                            "owner_result_status": row["owner_result_status"],
                            "explicitly_skipped_points": row["explicitly_skipped_points"],
                            "omitted_points_interpreted_as": {
                                "names": row["omitted_points"],
                                "meaning": "occluded_or_not_visible_per_owner_message_2026-07-23",
                            },
                            "prior_training_exact_source_overlap": row[
                                "prior_training_exact_source_overlap"
                            ],
                            "geometry_audit": row["geometry_audit"],
                        },
                    }
                )
            payload = {
                "schema_version": 1,
                "artifact_type": "racketsport_court_keypoint_labels",
                "clip": clip,
                "annotation": {"items": items},
                "frames": {
                    "frame_count": len(items),
                    "available_review_frame_count": len(items),
                    "frame_dir": f"{clip}/frames",
                    "path_base": "corpus_root",
                    "label_coordinate_space": [width, height],
                    "source_resolution": [width, height],
                },
                "review": {
                    "status": "reviewed",
                    "reviewer": "owner_sequential_court_labelpack3_20260723",
                    "independent_reviewed_count": len(items),
                    "note": (
                        "Owner clicked every visible point. Explicit skips and unfilled points "
                        "are JSON null because the owner confirmed they were covered or not visible; "
                        "no coordinate was fabricated."
                    ),
                },
            }
            _write_json(clip_root / "labels" / "court_keypoints.json", payload)
            clip_count += 1

        unsupported_manifest = {
            "schema_version": 1,
            "artifact_type": "racketsport_court_unsupported_view_labels",
            "authority": "owner_reviewed",
            "items": unsupported_rows,
        }
        _write_json(temporary / "unsupported_view_manifest.json", unsupported_manifest)
        shutil.copy2(results_path, temporary / "source_results.json")
        shutil.copy2(package_manifest_path, temporary / "source_package_manifest.json")

        positives = [row for row in normalized if row["kind"] == "positive"]
        report = {
            "schema_version": 1,
            "artifact_type": "racketsport_court_labelpack_owner_ingest_report",
            "status": "READY",
            "inputs": {
                "package_manifest": str(package_manifest_path),
                "package_manifest_sha256": _sha256(package_manifest_path),
                "results": str(results_path),
                "results_sha256": _sha256(results_path),
                "prior_training_manifest": str(prior_training_manifest) if prior_training_manifest else None,
            },
            "counts": {
                "package_frames": len(images),
                "positive_frames": len(positives),
                "unsupported_view_frames": len(unsupported_rows),
                "positive_venue_groups": len({row["venue_id"] for row in positives}),
                "trainer_clip_groups": clip_count,
                "labeled_points": labeled_point_count,
                "explicitly_skipped_points": explicitly_skipped_point_count,
                "omitted_points_interpreted_as_invisible": omitted_point_count,
                "prior_training_exact_source_overlap_positive_frames": sum(
                    bool(row["prior_training_exact_source_overlap"]) for row in positives
                ),
                "new_source_positive_frames": sum(
                    not bool(row["prior_training_exact_source_overlap"]) for row in positives
                ),
                "geometry_audited_positive_frames": sum(
                    bool(row["geometry_audit"].get("available")) for row in positives
                ),
                "geometry_flagged_positive_frames": len(geometry_flagged),
            },
            "status_counts": dict(sorted(status_counts.items())),
            "unsupported_reason_counts": dict(sorted(exclusion_counts.items())),
            "geometry_flags": geometry_flagged,
            "policies": {
                "raw_owner_results_immutable": True,
                "missing_points_fabricated": False,
                "owner_omissions_are_invisible": owner_omissions_are_invisible,
                "unsupported_views_enter_keypoint_training": False,
                "unsupported_views_preserved_for_supported_view_training": True,
                "source_grouping": "venue_id plus resolution; no venue crosses partitions",
                "independent_gate_usage": "forbidden; training-only rows",
            },
            "artifacts": {
                "train_root": f"{out}/train",
                "unsupported_manifest": f"{out}/unsupported_view_manifest.json",
                "source_results_copy": f"{out}/source_results.json",
                "source_package_manifest_copy": f"{out}/source_package_manifest.json",
            },
        }
        _write_json(temporary / "ingest_report.json", report)
        os.replace(temporary, out)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-training-manifest", type=Path)
    parser.add_argument(
        "--owner-omissions-are-invisible",
        action="store_true",
        help=(
            "Treat every canonical point absent from both keypoints and skipped_points as "
            "occluded/not visible. Use only after an explicit owner statement."
        ),
    )
    args = parser.parse_args()
    report = ingest_labelpack(
        package_manifest_path=args.package_manifest,
        results_path=args.results,
        out=args.out,
        owner_omissions_are_invisible=args.owner_omissions_are_invisible,
        prior_training_manifest=args.prior_training_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
