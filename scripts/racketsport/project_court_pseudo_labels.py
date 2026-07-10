#!/usr/bin/env python3
"""Project calibrated metric-15 court keypoints onto harvest rally videos.

The output keeps every sampled row in the full JSONL manifest. Clips whose
measured court-region camera drift p95 exceeds 2px remain auditable there but
are omitted from the included-only default view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.calibrate_harvest_courts import (  # noqa: E402
    CALIBRATED_GRADES,
    GRADE_AUTO_BAR,
    GRADE_MANUAL_BAR,
)
from threed.racketsport.court_keypoint_lines import COURT_LINE_FAMILIES  # noqa: E402
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME  # noqa: E402
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES  # noqa: E402


PROJECTOR_VERSION = "court_metric15_projector_v1"
STATIC_METHOD_VERSION = "orb_scene_homography_court_region_v2"
STATIC_DRIFT_P95_GATE_PX = 2.0

DEFAULT_LANE_DIR = Path("runs/lanes/court_data1_20260709")
DEFAULT_CALIBRATION_DIR = DEFAULT_LANE_DIR / "court_calibrations_r2"
DEFAULT_RALLIES_DIR = Path("data/online_harvest_20260706/rallies")
DEFAULT_MANIFEST = DEFAULT_LANE_DIR / "court_pseudo_labels.jsonl"
DEFAULT_VIEW = DEFAULT_LANE_DIR / "court_pseudo_labels_default.jsonl"
DEFAULT_REPORT = DEFAULT_LANE_DIR / "projector_report.json"
DEFAULT_QA_DIR = DEFAULT_LANE_DIR / "qa_overlays"


def project_metric_template(calibration: Mapping[str, Any]) -> dict[str, list[float]]:
    """Project the named 15-point metric template through a solved calibration."""

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    intrinsics = calibration["intrinsics"]
    extrinsics = calibration["extrinsics"]
    camera = np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_values = [float(value) for value in intrinsics.get("dist", [])]
    dist = np.asarray((dist_values + [0.0] * 4)[:4], dtype=np.float64)
    rotation = np.asarray(extrinsics["R"], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.asarray(extrinsics["t"], dtype=np.float64).reshape(3, 1)
    world = np.asarray(
        [PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m for name in PICKLEBALL_COURT_KEYPOINT_NAMES],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(world, rvec, translation, camera, dist)
    return {
        name: [float(point[0]), float(point[1])]
        for name, point in zip(PICKLEBALL_COURT_KEYPOINT_NAMES, projected.reshape(-1, 2), strict=True)
    }


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "source_id",
        "clip_path",
        "frame_index",
        "image_size",
        "keypoints",
        "source_grade",
        "calibration_residual_summary",
        "projector_version",
        "static_camera_check_id",
        "included",
        "excluded_reason",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"manifest row missing fields: {missing}")
    if row["schema_version"] != 1:
        raise ValueError("manifest row schema_version must be 1")
    if not isinstance(row["source_id"], str) or not row["source_id"]:
        raise ValueError("manifest row source_id must be non-empty")
    clip_path = Path(str(row["clip_path"]))
    if clip_path.is_absolute():
        raise ValueError("manifest clip_path must be relative")
    if isinstance(row["frame_index"], bool) or not isinstance(row["frame_index"], int) or row["frame_index"] < 0:
        raise ValueError("manifest frame_index must be a non-negative integer")
    size = row["image_size"]
    if not isinstance(size, list) or len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("manifest image_size must be two positive integers")
    if row["source_grade"] not in {GRADE_MANUAL_BAR, GRADE_AUTO_BAR}:
        raise ValueError("manifest source_grade must be calibrated")
    if row["projector_version"] != PROJECTOR_VERSION:
        raise ValueError("manifest projector_version mismatch")
    if not isinstance(row["included"], bool):
        raise ValueError("manifest included must be boolean")
    reason = row["excluded_reason"]
    if row["included"] and reason is not None:
        raise ValueError("included manifest rows cannot carry excluded_reason")
    if not row["included"] and not isinstance(reason, str):
        raise ValueError("excluded manifest rows require excluded_reason")
    keypoints = row["keypoints"]
    if not isinstance(keypoints, Mapping) or tuple(keypoints) != PICKLEBALL_COURT_KEYPOINT_NAMES:
        raise ValueError("manifest keypoints must contain the canonical 15 names in order")
    for name, item in keypoints.items():
        if not isinstance(item, Mapping) or set(item) != {"px", "in_frame"}:
            raise ValueError(f"manifest keypoint {name} must contain px and in_frame")
        px = item["px"]
        if not isinstance(px, list) or len(px) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in px):
            raise ValueError(f"manifest keypoint {name} px must be two finite numbers")
        if not isinstance(item["in_frame"], bool):
            raise ValueError(f"manifest keypoint {name} in_frame must be boolean")
    residual = row["calibration_residual_summary"]
    if not isinstance(residual, Mapping) or not {"median_px", "p95_px"} <= set(residual):
        raise ValueError("manifest calibration_residual_summary needs median_px and p95_px")


def measure_static_camera(
    reference_image: Any,
    video_path: Path,
    court_points: Mapping[str, Sequence[float]],
    *,
    sample_count: int = 8,
) -> dict[str, Any]:
    """Estimate camera drift using ORB/RANSAC homographies in the court region."""

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    if sample_count < 8:
        raise ValueError("static-camera verification requires at least 8 sampled frames")
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        return _static_unavailable(video_path, frame_count, "video_metadata_unavailable")
    if reference_image is None or reference_image.shape[1] != width or reference_image.shape[0] != height:
        capture.release()
        return _static_unavailable(video_path, frame_count, "reference_image_size_mismatch")

    sample_indexes = sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, sample_count)})
    if len(sample_indexes) < 8:
        capture.release()
        return _static_unavailable(video_path, frame_count, "fewer_than_8_distinct_frames")

    scale = min(1.0, 640.0 / width)
    small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    ref_gray = cv2.cvtColor(cv2.resize(reference_image, small_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    court = np.asarray([court_points[name] for name in PICKLEBALL_COURT_KEYPOINT_NAMES], dtype=np.float32)
    court_small = court * scale
    # Estimate the scene transform from all available static texture (trees,
    # apron, fence, and court). Restricting feature detection to the playable
    # polygon lets moving players dominate RANSAC. The reported displacement is
    # still evaluated only at the 15 court-region template points.
    feature_mask = np.full(ref_gray.shape, 255, dtype=np.uint8)

    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, feature_mask)
    sample_rows: list[dict[str, Any]] = []
    all_drifts: list[float] = []
    for frame_index in sample_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            sample_rows.append({"frame_index": frame_index, "status": "decode_failed"})
            continue
        gray = cv2.cvtColor(cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, feature_mask)
        homography, inliers, match_count, method = _estimate_court_homography(
            ref_gray,
            gray,
            feature_mask,
            ref_keypoints,
            ref_descriptors,
            keypoints,
            descriptors,
        )
        if homography is None:
            sample_rows.append(
                {
                    "frame_index": frame_index,
                    "status": "motion_estimate_failed",
                    "match_count": match_count,
                    "inlier_count": inliers,
                }
            )
            continue
        moved = cv2.perspectiveTransform(court_small.reshape(-1, 1, 2), homography).reshape(-1, 2)
        drifts = np.linalg.norm(moved - court_small, axis=1) / scale
        values = [float(value) for value in drifts]
        all_drifts.extend(values)
        sample_rows.append(
            {
                "frame_index": frame_index,
                "status": "measured",
                "method": method,
                "match_count": match_count,
                "inlier_count": inliers,
                "drift_median_px": _percentile(values, 50.0),
                "drift_p95_px": _percentile(values, 95.0),
                "drift_max_px": max(values),
            }
        )
    capture.release()

    measured_count = sum(row.get("status") == "measured" for row in sample_rows)
    if measured_count < 8:
        return {
            "method": STATIC_METHOD_VERSION,
            "video_path": _relative_path(video_path),
            "frame_count": frame_count,
            "sample_count_requested": sample_count,
            "sample_count_measured": measured_count,
            "sample_frames": sample_rows,
            "drift_p95_px": None,
            "gate_px": STATIC_DRIFT_P95_GATE_PX,
            "static_camera": False,
            "status": "unavailable",
            "excluded_reason": "static_check_unavailable",
            "method_note": _static_method_note(),
        }
    drift_p95 = _percentile(all_drifts, 95.0)
    static_camera = drift_p95 <= STATIC_DRIFT_P95_GATE_PX
    return {
        "method": STATIC_METHOD_VERSION,
        "video_path": _relative_path(video_path),
        "frame_count": frame_count,
        "sample_count_requested": sample_count,
        "sample_count_measured": measured_count,
        "sample_frames": sample_rows,
        "drift_median_px": _percentile(all_drifts, 50.0),
        "drift_p95_px": drift_p95,
        "drift_max_px": max(all_drifts),
        "gate_px": STATIC_DRIFT_P95_GATE_PX,
        "static_camera": static_camera,
        "status": "pass" if static_camera else "fail",
        "excluded_reason": None if static_camera else "camera_motion",
        "method_note": _static_method_note(),
    }


def _estimate_court_homography(
    reference_gray: Any,
    frame_gray: Any,
    mask: Any,
    reference_keypoints: Any,
    reference_descriptors: Any,
    keypoints: Any,
    descriptors: Any,
) -> tuple[Any | None, int, int, str | None]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    if reference_descriptors is not None and descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.78 * second.distance]
        if len(good) >= 8:
            src = np.float32([reference_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
            dst = np.float32([keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            required_inliers = max(20, int(math.ceil(0.25 * len(good))))
            if homography is not None and inliers >= required_inliers:
                return homography, inliers, len(good), "orb_ransac_homography"

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
    try:
        _score, warp = cv2.findTransformECC(
            reference_gray,
            frame_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None, 0, 0, None
    homography = np.vstack([warp, [0.0, 0.0, 1.0]]).astype(np.float64)
    return homography, 0, 0, "ecc_affine_fallback"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.static_samples < 8:
        raise ValueError("--static-samples must be at least 8")
    if args.qa_per_source < 0:
        raise ValueError("--qa-per-source must be non-negative")

    calibrations: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(args.calibration_dir.glob("*.json")):
        if path.name == "coverage_report.json":
            continue
        calibration = json.loads(path.read_text(encoding="utf-8"))
        source_id = str(calibration.get("source_id") or path.stem)
        if source_id in {"pwxNwFfYQlQ", "vQhtz8l6VqU"}:
            raise ValueError(f"held-out source calibration is forbidden: {source_id}")
        if calibration.get("calibration_grade") in CALIBRATED_GRADES:
            calibrations[source_id] = (calibration, path)
    if not calibrations:
        raise ValueError("no manual_bar/auto_bar calibrations found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.default_view_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.qa_dir.mkdir(parents=True, exist_ok=True)

    static_checks: list[dict[str, Any]] = []
    counts_by_source: dict[str, dict[str, int]] = {}
    counts_by_clip: dict[str, dict[str, Any]] = {}
    qa_candidates: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in calibrations}
    full_count = 0
    included_count = 0
    with args.out.open("w", encoding="utf-8") as full_file, args.default_view_out.open("w", encoding="utf-8") as default_file:
        for source_id, (calibration, calibration_path) in sorted(calibrations.items()):
            projected = project_metric_template(calibration)
            reference_path = Path(str(calibration.get("reference_image_path") or ""))
            if not reference_path.is_file():
                raise ValueError(f"{calibration_path}: reference_image_path missing or unreadable: {reference_path}")
            reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            source_dir = args.rallies_dir / source_id
            clips = sorted(source_dir.glob("*.mp4")) if source_dir.is_dir() else []
            if not clips:
                raise ValueError(f"no rally clips found for calibrated source {source_id}: {source_dir}")
            counts_by_source[source_id] = {"full_rows": 0, "included_rows": 0, "excluded_rows": 0, "clip_count": len(clips)}
            for clip_path in clips:
                check = measure_static_camera(
                    reference_image,
                    clip_path,
                    projected,
                    sample_count=args.static_samples,
                )
                check_id = f"{source_id}/{clip_path.stem}"
                check["static_camera_check_id"] = check_id
                static_checks.append(check)
                excluded_reason = check["excluded_reason"]
                included = excluded_reason is None
                capture = cv2.VideoCapture(str(clip_path))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                clip_counts = {
                    "source_id": source_id,
                    "clip_path": _relative_path(clip_path),
                    "frame_count": frame_count,
                    "full_rows": 0,
                    "included_rows": 0,
                    "excluded_rows": 0,
                    "excluded_reason": excluded_reason,
                    "static_camera_check_id": check_id,
                }
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % args.stride == 0:
                        row = _manifest_row(
                            source_id=source_id,
                            clip_path=clip_path,
                            frame_index=frame_index,
                            image_size=(width, height),
                            projected=projected,
                            calibration=calibration,
                            static_check_id=check_id,
                            included=included,
                            excluded_reason=excluded_reason,
                        )
                        validate_manifest_row(row)
                        line = json.dumps(row, sort_keys=False, separators=(",", ":")) + "\n"
                        full_file.write(line)
                        full_count += 1
                        clip_counts["full_rows"] += 1
                        counts_by_source[source_id]["full_rows"] += 1
                        if included:
                            default_file.write(line)
                            included_count += 1
                            clip_counts["included_rows"] += 1
                            counts_by_source[source_id]["included_rows"] += 1
                        else:
                            clip_counts["excluded_rows"] += 1
                            counts_by_source[source_id]["excluded_rows"] += 1
                        qa_candidates[source_id].append(
                            {
                                "clip_path": clip_path,
                                "frame_index": frame_index,
                                "projected": projected,
                                "included": included,
                                "excluded_reason": excluded_reason,
                            }
                        )
                    frame_index += 1
                capture.release()
                counts_by_clip[check_id] = clip_counts

    qa_counts = _render_qa_overlays(qa_candidates, args.qa_dir, args.qa_per_source, args.seed)
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_pseudo_label_projection_report",
        "projector_version": PROJECTOR_VERSION,
        "stride": args.stride,
        "calibration_dir": args.calibration_dir.as_posix(),
        "rallies_dir": args.rallies_dir.as_posix(),
        "manifest_jsonl": args.out.as_posix(),
        "default_view_jsonl": args.default_view_out.as_posix(),
        "calibrated_source_count": len(calibrations),
        "calibrated_source_ids": sorted(calibrations),
        "full_row_count": full_count,
        "included_row_count": included_count,
        "excluded_row_count": full_count - included_count,
        "counts_by_source": counts_by_source,
        "counts_by_clip": counts_by_clip,
        "static_camera_method": _static_method_note(),
        "static_camera_gate_px": STATIC_DRIFT_P95_GATE_PX,
        "static_camera_checks": static_checks,
        "qa_overlay_count_by_source": qa_counts,
        "schema_validation": {"rows_checked": full_count, "failures": 0},
    }
    _write_json(args.report_json, report)
    return report


def _manifest_row(
    *,
    source_id: str,
    clip_path: Path,
    frame_index: int,
    image_size: tuple[int, int],
    projected: Mapping[str, Sequence[float]],
    calibration: Mapping[str, Any],
    static_check_id: str,
    included: bool,
    excluded_reason: str | None,
) -> dict[str, Any]:
    width, height = image_size
    reprojection = calibration["reprojection_error_px"]
    return {
        "schema_version": 1,
        "source_id": source_id,
        "clip_path": _relative_path(clip_path),
        "frame_index": frame_index,
        "image_size": [width, height],
        "keypoints": {
            name: {
                "px": [float(projected[name][0]), float(projected[name][1])],
                "in_frame": 0.0 <= float(projected[name][0]) < width and 0.0 <= float(projected[name][1]) < height,
            }
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        },
        "source_grade": calibration["calibration_grade"],
        "calibration_residual_summary": {
            "median_px": float(reprojection["median"]),
            "p95_px": float(reprojection["p95"]),
            "labeled_frame_count": len(calibration.get("per_frame_reprojection_stats", [])),
        },
        "projector_version": PROJECTOR_VERSION,
        "static_camera_check_id": static_check_id,
        "included": included,
        "excluded_reason": excluded_reason,
    }


def _render_qa_overlays(
    candidates_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    qa_dir: Path,
    count: int,
    seed: int,
) -> dict[str, int]:
    import cv2  # type: ignore[import-not-found]

    rng = random.Random(seed)
    rendered: dict[str, int] = {}
    for source_id, candidates in sorted(candidates_by_source.items()):
        source_dir = qa_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        selected = rng.sample(list(candidates), min(count, len(candidates)))
        output_count = 0
        for idx, candidate in enumerate(selected):
            capture = cv2.VideoCapture(str(candidate["clip_path"]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(candidate["frame_index"]))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                continue
            _draw_court_overlay(frame, candidate["projected"])
            status = "included" if candidate["included"] else str(candidate["excluded_reason"])
            cv2.putText(
                frame,
                f"{source_id} | {Path(candidate['clip_path']).stem} f={candidate['frame_index']} | {status}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            output_path = source_dir / f"overlay_{idx:02d}_{Path(candidate['clip_path']).stem}_f{int(candidate['frame_index']):06d}.jpg"
            if cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                output_count += 1
        rendered[source_id] = output_count
    return rendered


def _draw_court_overlay(frame: Any, projected: Mapping[str, Sequence[float]]) -> None:
    import cv2  # type: ignore[import-not-found]

    for family in COURT_LINE_FAMILIES:
        color = (0, 165, 255) if family.name == "net" else (50, 240, 80)
        for start_name, end_name in family.segment_pairs:
            start = tuple(int(round(value)) for value in projected[start_name])
            end = tuple(int(round(value)) for value in projected[end_name])
            cv2.line(frame, start, end, color, 4, cv2.LINE_AA)
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        point = tuple(int(round(value)) for value in projected[name])
        cv2.circle(frame, point, 6, (255, 80, 40), -1, cv2.LINE_AA)


def _static_unavailable(video_path: Path, frame_count: int, reason: str) -> dict[str, Any]:
    return {
        "method": STATIC_METHOD_VERSION,
        "video_path": _relative_path(video_path),
        "frame_count": frame_count,
        "sample_count_requested": 8,
        "sample_count_measured": 0,
        "sample_frames": [],
        "drift_p95_px": None,
        "gate_px": STATIC_DRIFT_P95_GATE_PX,
        "static_camera": False,
        "status": "unavailable",
        "excluded_reason": "static_check_unavailable",
        "unavailable_reason": reason,
        "method_note": _static_method_note(),
    }


def _static_method_note() -> str:
    return (
        "Detect up to 2500 ORB scene features (static court, apron, fence, and background), match "
        "reference-to-sample descriptors with a 0.78 ratio test, require at least 20 and 25% RANSAC "
        "inliers for a 1.5px homography, and measure displacement at all 15 court-region template points. "
        "ECC affine is a failover only. "
        "At least 8 uniformly spaced frames must measure; pooled court-point drift p95 must be <=2.0px."
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT.resolve())).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--rallies-dir", type=Path, default=DEFAULT_RALLIES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--default-view-out", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--static-samples", type=int, default=8)
    parser.add_argument("--qa-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--json", action="store_true", help="Print the projection report JSON.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.calibration_dir.is_dir():
        parser.exit(2, f"{parser.prog}: error: --calibration-dir not found: {args.calibration_dir}\n")
    if not args.rallies_dir.is_dir():
        parser.exit(2, f"{parser.prog}: error: --rallies-dir not found: {args.rallies_dir}\n")
    try:
        report = run(args)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps({"manifest_jsonl": report["manifest_jsonl"], "full_row_count": report["full_row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
