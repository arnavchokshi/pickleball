"""Ground-truth-free diagnostics for an existing court calibration run.

These metrics measure internal image evidence, temporal consistency, optional
net evidence, an explicitly diagnostic PB Vision comparison, and local metric
sensitivity.  They are not calibration promotion gates and never solve or
modify a calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .court_calibration import homography_from_planar_points, project_image_points_to_world, project_planar_points
from .court_templates import get_court_template


MISSING = "absent"
PRESENT = "present"
SCORER_VERSION = "cpm_v2_frozen_20260712"
SCORER_VERSION_POLICY = "manager_bump_required"
PAINT_EXCLUDED_LINE_IDS = frozenset({"net"})
EDGE_LINE_IDS = frozenset({"near_baseline", "far_baseline", "left_sideline", "right_sideline"})
CANONICAL_SENSITIVITY_POINTS_M: dict[str, tuple[float, float]] = {
    "near_baseline_left_corner": (-3.048, -6.7056),
    "near_baseline_center": (0.0, -6.7056),
    "near_baseline_right_corner": (3.048, -6.7056),
    "far_baseline_left_corner": (-3.048, 6.7056),
    "far_baseline_center": (0.0, 6.7056),
    "far_baseline_right_corner": (3.048, 6.7056),
    "near_kitchen_center": (0.0, -2.1336),
    "far_kitchen_center": (0.0, 2.1336),
    "left_sideline_mid": (-3.048, 0.0),
    "right_sideline_mid": (3.048, 0.0),
}


def score_court_precision_run(
    run_dir: str | Path,
    *,
    sample_count: int = 12,
    search_window_px: int = 14,
    pb_export_path: str | Path | None = None,
    net_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score artifacts already present in ``run_dir`` without recalibration."""

    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    calibration_path = root / "court_calibration.json"
    if not calibration_path.is_file():
        raise ValueError(f"run is missing court_calibration.json: {root}")
    calibration = _read_json(calibration_path)
    video_path = discover_run_video(root)
    clip_id = _clip_id(root, calibration)
    frame_meta = probe_video(video_path) if video_path is not None else None
    sample_indexes = (
        evenly_spaced_frame_indexes(frame_meta["frame_count"], sample_count)
        if frame_meta is not None
        else []
    )
    per_frame_calibrations = load_per_frame_calibrations(root)
    camera_motion_path = root / "camera_motion.json"
    camera_motion = _read_json(camera_motion_path) if camera_motion_path.is_file() else None

    if video_path is None:
        m1 = _absent("video_or_materialized_frames_missing", required_artifacts=["source.mp4 or source.mov"])
    else:
        m1 = line_evidence_residual(
            video_path,
            calibration,
            sample_indexes,
            seed_calibration=calibration,
            per_frame_calibrations=per_frame_calibrations,
            search_window_px=search_window_px,
        )

    m2 = temporal_stability(
        per_frame_calibrations,
        camera_motion=camera_motion,
        static_calibration_path=calibration_path,
    )
    m3 = net_consistency(
        calibration,
        root=root,
        explicit_evidence_path=Path(net_evidence_path) if net_evidence_path else None,
    )
    if "wolverine" not in clip_id.lower():
        m4 = _absent("pbvision_comparison_is_wolverine_only")
    elif pb_export_path is None:
        m4 = _absent("pbvision_export_not_supplied")
    else:
        m4 = pbvision_court_comparison(calibration, Path(pb_export_path))
    m5 = calibration_sensitivity_diagnostics(calibration)

    input_hashes = _input_hashes(
        calibration_path=calibration_path,
        video_path=video_path,
        camera_motion_path=camera_motion_path if camera_motion_path.is_file() else None,
        net_evidence_path=Path(net_evidence_path) if net_evidence_path else None,
        pb_export_path=Path(pb_export_path) if pb_export_path is not None and "wolverine" in clip_id.lower() else None,
    )
    freeze_contract = {
        "scorer_version": SCORER_VERSION,
        "scorer_version_policy": SCORER_VERSION_POLICY,
        "frozen_frame_indexes": sample_indexes,
        "input_sha256": input_hashes,
        "m1_frozen_visible_sample_set_sha256": m1.get("frozen_visible_sample_set_sha256"),
        "m1_frozen_evidence_fits_sha256": m1.get("frozen_evidence_fits_sha256"),
        "coordinate_space": "source_video_native_pixels; calibration-declared distortion handling",
        "diagnostic_only": True,
        "promotion_gate": False,
    }

    return {
        "schema_version": 1,
        "artifact_type": "court_precision_clip_diagnostics",
        "scorer_version": SCORER_VERSION,
        "clip": clip_id,
        "run_dir": str(root),
        "diagnostic_only": True,
        "promotion_gate": False,
        "freeze_contract": freeze_contract,
        "metrics": {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5},
        "inputs": {
            "calibration": str(calibration_path),
            "video": str(video_path) if video_path is not None else None,
            "frame_metadata": frame_meta,
            "sampled_frame_indexes": sample_indexes,
            "per_frame_calibration_count": len(per_frame_calibrations),
            "camera_motion": str(camera_motion_path) if camera_motion_path.is_file() else None,
            "net_evidence": m3.get("evidence_path"),
            "pbvision_export": str(pb_export_path) if pb_export_path is not None and "wolverine" in clip_id.lower() else None,
        },
    }


def discover_run_video(root: Path) -> Path | None:
    for name in ("source.mp4", "source.mov", "source.m4v", "input.mp4"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def evenly_spaced_frame_indexes(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(frame_count, sample_count)
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})


def line_evidence_residual(
    video_path: Path,
    calibration: Mapping[str, Any],
    frame_indexes: Sequence[int],
    *,
    seed_calibration: Mapping[str, Any] | None = None,
    per_frame_calibrations: Mapping[int, Mapping[str, Any]] | None = None,
    search_window_px: int = 14,
) -> dict[str, Any]:
    """M1: score a candidate against seed-frozen paired-edge paint evidence.

    The seed calibration defines visibility and evidence extraction exactly
    once. Candidate projections are then compared with those frozen image
    observations; a candidate cannot move samples out of the denominator or
    re-run the paint search around its own projection.
    """

    if search_window_px < 1:
        raise ValueError("search_window_px must be positive")
    seed = seed_calibration or calibration
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    frozen_frames: list[dict[str, Any]] = []
    try:
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                frozen_frames.append(
                    {"frame_index": int(frame_index), "decode_status": MISSING, "reason": "frame_decode_failed", "samples": []}
                )
                continue
            frozen_frames.append(
                freeze_line_evidence_frame(
                    frame,
                    seed,
                    frame_index=int(frame_index),
                    search_window_px=search_window_px,
                )
            )
    finally:
        capture.release()

    visible_set = [
        {
            "frame_index": frame["frame_index"],
            "line_id": sample["line_id"],
            "sample_index": sample["sample_index"],
            "world_xyz_m": sample["world_xyz_m"],
            "seed_image_xy_px": sample["seed_image_xy_px"],
            "seed_normal_xy": sample["seed_normal_xy"],
            "near_far_bucket": sample["near_far_bucket"],
            "edge_center_bucket": sample["edge_center_bucket"],
        }
        for frame in frozen_frames
        for sample in frame.get("samples", [])
    ]
    evidence_fits = [
        {
            "frame_index": frame["frame_index"],
            "line_id": sample["line_id"],
            "sample_index": sample["sample_index"],
            "evidence_status": sample["evidence_status"],
            "observed_center_xy_px": sample.get("observed_center_xy_px"),
            "signed_center_offset_from_seed_px": sample.get("signed_center_offset_from_seed_px"),
            "edge_offsets_from_seed_px": sample.get("edge_offsets_from_seed_px"),
            "paint_band_width_px": sample.get("paint_band_width_px"),
            "edge_strength": sample.get("edge_strength"),
            "interior_contrast": sample.get("interior_contrast"),
        }
        for frame in frozen_frames
        for sample in frame.get("samples", [])
    ]
    rows = []
    for frozen_frame in frozen_frames:
        frame_index = int(frozen_frame["frame_index"])
        candidate = (per_frame_calibrations or {}).get(frame_index, calibration)
        rows.append(score_frozen_line_evidence_frame(frozen_frame, candidate))
    result = _aggregate_m1_rows(rows)
    result.update(
        {
            "method": "seed_frozen_visibility_and_paired_gradient_edge_paint_centers",
            "evidence_extractor": "court_precision_metrics.private_paired_gradient_edges_v1",
            "candidate_machinery_imported": False,
            "visibility_policy": "computed_once_from_input_seed_calibration_and_frozen_for_all_candidates",
            "evidence_policy": "paired signed gradient edges with parabolic subpixel peaks; midpoint is paint center",
            "overflow_policy": "no pair inside the bounded seed search is explicit overflow; residual maxima exclude overflow and are never clipped",
            "distortion_aware": bool(_has_distortion(seed)),
            "search_window_px": int(search_window_px),
            "frozen_visible_sample_set": visible_set,
            "frozen_visible_sample_set_sha256": _canonical_sha256(visible_set),
            "frozen_evidence_fits": evidence_fits,
            "frozen_evidence_fits_sha256": _canonical_sha256(evidence_fits),
        }
    )
    return result


def score_line_evidence_frame(
    frame_bgr: np.ndarray,
    calibration: Mapping[str, Any],
    *,
    frame_index: int,
    search_window_px: int,
    seed_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = freeze_line_evidence_frame(
        frame_bgr,
        seed_calibration or calibration,
        frame_index=frame_index,
        search_window_px=search_window_px,
    )
    row = score_frozen_line_evidence_frame(frozen, calibration)
    visible_set = [
        {
            "frame_index": frame_index,
            "line_id": sample["line_id"],
            "sample_index": sample["sample_index"],
            "world_xyz_m": sample["world_xyz_m"],
            "seed_image_xy_px": sample["seed_image_xy_px"],
            "seed_normal_xy": sample["seed_normal_xy"],
        }
        for sample in frozen.get("samples", [])
    ]
    row["frozen_visible_sample_set_sha256"] = _canonical_sha256(visible_set)
    return row


def freeze_line_evidence_frame(
    frame_bgr: np.ndarray,
    seed_calibration: Mapping[str, Any],
    *,
    frame_index: int,
    search_window_px: int,
) -> dict[str, Any]:
    """Extract one immutable evidence record per seed-visible template sample."""

    if frame_bgr is None or frame_bgr.ndim < 2:
        raise ValueError("frame_bgr must be an image array")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) if frame_bgr.ndim == 3 else frame_bgr.astype(np.float64)
    height, width = gray.shape[:2]
    template = get_court_template(str(seed_calibration.get("sport", "pickleball")))
    samples: list[dict[str, Any]] = []
    for line_id, endpoints in template.line_segments_m.items():
        if line_id in PAINT_EXCLUDED_LINE_IDS:
            continue
        world_points = _sample_world_segment(endpoints, count=65)
        seed_projected = project_world_points_distortion_aware(seed_calibration, world_points)
        for sample_index in range(1, len(seed_projected) - 1):
            point = seed_projected[sample_index]
            tangent = seed_projected[sample_index + 1] - seed_projected[sample_index - 1]
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= 1e-9 or not np.isfinite(point).all():
                continue
            if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
                continue
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64) / tangent_length
            fit = _paired_gradient_paint_center(gray, point, normal, search_window_px)
            world = world_points[sample_index]
            row: dict[str, Any] = {
                "line_id": line_id,
                "sample_index": sample_index,
                "world_xyz_m": [float(value) for value in world],
                "seed_image_xy_px": [float(value) for value in point],
                "seed_normal_xy": [float(value) for value in normal],
                "near_far_bucket": _near_far_bucket(float(world[1])),
                "edge_center_bucket": "edge" if line_id in EDGE_LINE_IDS else "center",
                **fit,
            }
            if fit["evidence_status"] == PRESENT:
                center = point + float(fit["signed_center_offset_from_seed_px"]) * normal
                row["observed_center_xy_px"] = [float(value) for value in center]
            samples.append(row)
    return {"frame_index": int(frame_index), "decode_status": PRESENT, "samples": samples}


def score_frozen_line_evidence_frame(
    frozen_frame: Mapping[str, Any], candidate_calibration: Mapping[str, Any]
) -> dict[str, Any]:
    frame_index = int(frozen_frame["frame_index"])
    if frozen_frame.get("decode_status") != PRESENT:
        return {
            "frame_index": frame_index,
            "status": MISSING,
            "reason": frozen_frame.get("reason", "frame_decode_failed"),
            "visible_sample_count": 0,
            "evidence_sample_count": 0,
            "overflow_count": 0,
            "evidence_coverage_fraction": 0.0,
            "lines": [],
            "buckets": {},
        }
    samples = list(frozen_frame.get("samples", []))
    world = [sample["world_xyz_m"] for sample in samples]
    candidate_points = project_world_points_distortion_aware(candidate_calibration, world) if world else np.empty((0, 2))
    scored: list[dict[str, Any]] = []
    for sample, candidate_point in zip(samples, candidate_points, strict=True):
        item = dict(sample)
        item["candidate_image_xy_px"] = [float(value) for value in candidate_point]
        if sample.get("evidence_status") == PRESENT:
            observed = np.asarray(sample["observed_center_xy_px"], dtype=np.float64)
            normal = np.asarray(sample["seed_normal_xy"], dtype=np.float64)
            signed = float(np.dot(observed - candidate_point, normal))
            item["signed_residual_px"] = signed
            item["abs_residual_px"] = abs(signed)
            item["occlusion_bucket"] = "evidence_present"
        else:
            item["signed_residual_px"] = None
            item["abs_residual_px"] = None
            item["occlusion_bucket"] = "evidence_absent_or_occluded"
        scored.append(item)
    line_rows = _group_m1_samples(scored, key="line_id")
    bucket_rows = {
        "near_far_side": _group_m1_samples(scored, key="near_far_bucket"),
        "edge_center": _group_m1_samples(scored, key="edge_center_bucket"),
        "occlusion": _group_m1_samples(scored, key="occlusion_bucket"),
    }
    residuals = [float(sample["abs_residual_px"]) for sample in scored if sample.get("abs_residual_px") is not None]
    visible = len(scored)
    evidence = len(residuals)
    return {
        "frame_index": frame_index,
        "status": PRESENT if residuals else MISSING,
        "reason": None if residuals else "no_paired_edge_evidence_in_frozen_seed_windows",
        "residual_px": _summary(residuals, p90=True) if residuals else None,
        "residuals_px": residuals,
        "visible_sample_count": visible,
        "evidence_sample_count": evidence,
        "overflow_count": visible - evidence,
        "evidence_coverage_fraction": evidence / visible if visible else 0.0,
        "coverage_weighted_rollup": _coverage_weighted_rollup(line_rows),
        "lines": line_rows,
        "buckets": bucket_rows,
    }


def project_world_points_distortion_aware(
    calibration: Mapping[str, Any], world_points: Sequence[Sequence[float]]
) -> np.ndarray:
    intrinsics = calibration.get("intrinsics")
    extrinsics = calibration.get("extrinsics")
    if _has_distortion(calibration) and isinstance(intrinsics, Mapping) and isinstance(extrinsics, Mapping):
        camera = np.asarray(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(extrinsics["R"], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = np.asarray(extrinsics["t"], dtype=np.float64).reshape(3, 1)
        distortion = np.asarray(intrinsics.get("dist", []), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            np.asarray(world_points, dtype=np.float64), rvec, tvec, camera, distortion
        )
        return projected.reshape(-1, 2)
    return np.asarray(project_planar_points(calibration["homography"], world_points), dtype=np.float64)


def load_per_frame_calibrations(root: Path) -> dict[int, Mapping[str, Any]]:
    """Load only explicit per-frame solves; never derive them from static artifacts."""

    for name in ("court_calibrations.json", "court_calibration_frames.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = _read_json(path)
        raw_frames = payload.get("frames") if isinstance(payload, Mapping) else None
        if isinstance(raw_frames, list):
            result: dict[int, Mapping[str, Any]] = {}
            for row in raw_frames:
                if not isinstance(row, Mapping):
                    continue
                frame_index = row.get("frame_index", row.get("frame_idx"))
                calibration = row.get("calibration", row)
                if frame_index is not None and isinstance(calibration, Mapping) and "homography" in calibration:
                    result[int(frame_index)] = calibration
            return result
        if isinstance(raw_frames, Mapping):
            return {
                int(frame): value
                for frame, value in raw_frames.items()
                if isinstance(value, Mapping) and "homography" in value
            }
    return {}


def _project_standard_keypoints(calibration: Mapping[str, Any]) -> dict[str, list[float]]:
    template = get_court_template(str(calibration.get("sport", "pickleball")))
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    nvz = float(template.non_volley_zone_ft or 0.0) * 0.3048
    points = {
        "near_left_corner": (-half_width, -half_length),
        "near_baseline_center": (0.0, -half_length),
        "near_right_corner": (half_width, -half_length),
        "far_right_corner": (half_width, half_length),
        "far_baseline_center": (0.0, half_length),
        "far_left_corner": (-half_width, half_length),
        "near_nvz_left": (-half_width, -nvz),
        "near_nvz_center": (0.0, -nvz),
        "near_nvz_right": (half_width, -nvz),
        "net_left_sideline": (-half_width, 0.0),
        "net_center": (0.0, 0.0),
        "net_right_sideline": (half_width, 0.0),
        "far_nvz_left": (-half_width, nvz),
        "far_nvz_center": (0.0, nvz),
        "far_nvz_right": (half_width, nvz),
    }
    projected = project_world_points_distortion_aware(
        calibration, [[x, y, 0.0] for x, y in points.values()]
    )
    return {
        name: [float(value) for value in xy]
        for name, xy in zip(points, projected, strict=True)
    }


def temporal_stability(
    per_frame_calibrations: Mapping[int, Mapping[str, Any]],
    *,
    camera_motion: Mapping[str, Any] | None,
    static_calibration_path: Path,
) -> dict[str, Any]:
    """M2: keypoint deltas and drift, only when true per-frame solves exist."""

    if len(per_frame_calibrations) < 2:
        return _absent(
            "per_frame_calibration_missing",
            detail="only a static court_calibration.json is available; temporal precision is not inferred",
            static_calibration=str(static_calibration_path),
            per_frame_calibration_count=len(per_frame_calibrations),
            camera_motion_available=camera_motion is not None,
        )
    frame_indexes = sorted(per_frame_calibrations)
    projected_by_frame = {
        frame: _project_standard_keypoints(per_frame_calibrations[frame])
        for frame in frame_indexes
    }
    names = sorted(set.intersection(*(set(points) for points in projected_by_frame.values())))
    arrays = {
        frame: np.asarray([projected_by_frame[frame][name] for name in names], dtype=np.float64)
        for frame in frame_indexes
    }
    robust_center = np.median(np.stack(list(arrays.values())), axis=0)
    per_frame: list[dict[str, Any]] = []
    deltas: list[float] = []
    drifts: list[float] = []
    previous: np.ndarray | None = None
    motion_labels = _camera_motion_labels(camera_motion)
    split: dict[str, list[float]] = {"static": [], "camera_moving": [], "unknown": []}
    for frame in frame_indexes:
        array = arrays[frame]
        frame_delta = np.linalg.norm(array - previous, axis=1) if previous is not None else np.asarray([])
        frame_drift = np.linalg.norm(array - robust_center, axis=1)
        label = motion_labels.get(frame, "unknown")
        deltas.extend(frame_delta.tolist())
        drifts.extend(frame_drift.tolist())
        split[label].extend(frame_delta.tolist())
        per_frame.append(
            {
                "frame_index": frame,
                "motion_class": label,
                "delta_px": _summary(frame_delta) if frame_delta.size else None,
                "drift_vs_robust_clip_median_px": _summary(frame_drift),
            }
        )
        previous = array
    return {
        "status": PRESENT,
        "keypoint_count": len(names),
        "frame_count": len(frame_indexes),
        "frame_to_frame_keypoint_delta_px": _summary(deltas),
        "drift_vs_robust_clip_median_px": _summary(drifts),
        "split_frame_to_frame_delta_px": {
            label: (_summary(values) if values else _absent("no_frames_in_class"))
            for label, values in split.items()
        },
        "per_frame": per_frame,
    }


def net_consistency(
    calibration: Mapping[str, Any],
    *,
    root: Path,
    explicit_evidence_path: Path | None,
) -> dict[str, Any]:
    """M3: compare existing net evidence with the calibrated net top."""

    candidates = [explicit_evidence_path] if explicit_evidence_path is not None else []
    candidates.extend([root / "court_line_evidence.json", root / "net_anchor_court.json", root / "court_proposal.json"])
    evidence_path = next((path for path in candidates if path is not None and path.is_file()), None)
    if evidence_path is None:
        return _absent("net_evidence_or_solver_artifact_missing")
    payload = _read_json(evidence_path)
    audit = _audit_net_evidence_payload(payload)
    if audit["comparable"] is not True:
        return _absent(
            str(audit["reason"]),
            evidence_path=str(evidence_path),
            evidence_sha256=_file_sha256(evidence_path),
            semantics_audit=audit,
        )
    observed = np.asarray(audit["semantic_top_net_triplet_px"], dtype=np.float64)
    source_size = audit.get("source_image_size")
    target_size = calibration.get("image_size", [1920, 1080])
    if source_size is not None:
        observed[:, 0] *= float(target_size[0]) / float(source_size[0])
        observed[:, 1] *= float(target_size[1]) / float(source_size[1])
    template = get_court_template(str(calibration.get("sport", "pickleball")))
    x = template.net_width_m / 2.0
    expected_world = [
        [-x, 0.0, template.post_net_height_m],
        [0.0, 0.0, template.center_net_height_m],
        [x, 0.0, template.post_net_height_m],
    ]
    expected = project_world_points_distortion_aware(calibration, expected_world)
    direct = np.linalg.norm(observed - expected, axis=1)
    flipped_observed = observed[::-1]
    flipped = np.linalg.norm(flipped_observed - expected, axis=1)
    if float(np.sum(flipped)) < float(np.sum(direct)):
        observed = flipped_observed
        direct = flipped
    return {
        "status": PRESENT,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _file_sha256(evidence_path),
        "semantics_audit": audit,
        "expected_calibrated_top_net_triplet_px": expected.tolist(),
        "observed_semantic_top_net_triplet_px": observed.tolist(),
        "residual_px": _summary(direct, p90=True),
        "net_shape": "post_height_to_center_sag_to_post_height",
    }


def pbvision_court_comparison(calibration: Mapping[str, Any], pb_export_path: Path) -> dict[str, Any]:
    """M4: reproduce the frozen matched-Wolverine 12-point camera protocol."""

    payload = _read_json(pb_export_path)
    camera = payload.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("pb export has no camera object")
    segments = camera.get("cameraSegments")
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], Mapping):
        raise ValueError("pb export has no camera segment")
    segment = segments[0]
    pb_points = segment.get("court_points")
    image_points = calibration.get("image_pts")
    world_points = calibration.get("world_pts")
    image_size = calibration.get("image_size", [1920, 1080])
    if not isinstance(pb_points, list) or len(pb_points) != 12:
        raise ValueError("pb camera segment must contain exactly 12 court_points")
    if not isinstance(image_points, list) or not isinstance(world_points, list) or len(image_points) != len(world_points):
        raise ValueError("our calibration lacks paired image_pts/world_pts")
    width, height = int(image_size[0]), int(image_size[1])
    pb_observed = np.asarray([[float(point["u"]) * width, float(point["v"]) * height] for point in pb_points])
    ours_observed = np.asarray(image_points, dtype=np.float64)
    nearest = np.linalg.norm(pb_observed[:, None] - ours_observed[None, :], axis=2).argmin(axis=1)
    if len(set(nearest.tolist())) != 12:
        return _absent("pb_12_point_nearest_assignment_not_unique")
    shared_world = np.asarray(world_points, dtype=np.float64)[nearest]
    ours_projected = project_world_points_distortion_aware(calibration, shared_world)
    pb_projected = _project_pb_camera(_ours_m_to_pb_feet(shared_world), segment, width, height)
    theirs = np.linalg.norm(pb_observed - ours_observed[nearest], axis=1)
    ours = np.linalg.norm(ours_projected - ours_observed[nearest], axis=1)
    pb_self = np.linalg.norm(pb_projected - pb_observed, axis=1)
    camera_to_camera = np.linalg.norm(pb_projected - ours_projected, axis=1)
    theirs_summary = _summary(theirs)
    ours_summary = _summary(ours)
    frozen_mapping = [
        {
            "pb_point_index": int(pb_index),
            "our_reviewed_point_index": int(our_index),
            "world_xyz_m": [float(value) for value in shared_world[pb_index]],
        }
        for pb_index, our_index in enumerate(nearest.tolist())
    ]
    return {
        "status": PRESENT,
        "competitor_reference_only": True,
        "promotion_gate": False,
        "protocol": "reveng_matched_wolverine_12_unnamed_points_nearest_unique_assignment",
        "frame_protocol": {
            "camera_segment_start": int(segment.get("s", 0)),
            "camera_segment_end": int(segment.get("e", 299)),
            "fps": float(camera.get("fps", 30.0)),
            "static_camera_segment": True,
        },
        "point_count": 12,
        "frozen_point_mapping": frozen_mapping,
        "frozen_point_mapping_sha256": _canonical_sha256(frozen_mapping),
        "pb_export_sha256": _file_sha256(pb_export_path),
        "our_calibration_reprojection_to_our_reviewed_points_px": ours_summary,
        "pbvision_observation_to_our_reviewed_point_px": theirs_summary,
        "delta_ours_minus_pbvision_median_px": float(ours_summary["median"] - theirs_summary["median"]),
        "pbvision_camera_self_reprojection_px": _summary(pb_self),
        "pbvision_camera_to_our_camera_projection_px": _summary(camera_to_camera),
        "banked_reference_median_px": 5.666187383697394,
        "banked_reference_delta_px": float(theirs_summary["median"] - 5.666187383697394),
    }


def calibration_sensitivity_diagnostics(
    calibration: Mapping[str, Any],
    *,
    bootstrap_draws: int = 200,
    observation_sigma_px: float = 1.0,
    random_seed: int = 20260712,
) -> dict[str, Any]:
    """M5: separate local scale from observation-refit uncertainty."""

    return {
        "status": PRESENT,
        "image_to_world_scale_jacobian": image_to_world_scale_jacobian(calibration),
        "observation_perturbation_bootstrap": observation_perturbation_bootstrap(
            calibration,
            draws=bootstrap_draws,
            sigma_px=observation_sigma_px,
            random_seed=random_seed,
        ),
    }


def image_to_world_scale_jacobian(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Local image-to-world scale only; explicitly not calibration uncertainty."""

    rows: list[dict[str, Any]] = []
    for name, world_xy in CANONICAL_SENSITIVITY_POINTS_M.items():
        image_xy = project_world_points_distortion_aware(calibration, [[world_xy[0], world_xy[1], 0.0]])[0]
        perturbed = np.asarray(
            [
                [image_xy[0] - 1.0, image_xy[1]],
                [image_xy[0] + 1.0, image_xy[1]],
                [image_xy[0], image_xy[1] - 1.0],
                [image_xy[0], image_xy[1] + 1.0],
            ],
            dtype=np.float64,
        )
        world_perturbed = _image_pixels_to_world_distortion_aware(calibration, perturbed)
        displacements_cm = np.linalg.norm(world_perturbed - np.asarray(world_xy), axis=1) * 100.0
        rows.append(
            {
                "point": name,
                "world_xy_m": list(world_xy),
                "image_xy_px": image_xy.tolist(),
                "minus_x_cm": float(displacements_cm[0]),
                "plus_x_cm": float(displacements_cm[1]),
                "minus_y_cm": float(displacements_cm[2]),
                "plus_y_cm": float(displacements_cm[3]),
                "worst_1px_displacement_cm": float(np.max(displacements_cm)),
            }
        )
    values = [row["worst_1px_displacement_cm"] for row in rows]
    return {
        "status": PRESENT,
        "metric_name": "image_to_world_scale_jacobian",
        "calibration_uncertainty": False,
        "method": "four_direction_local_finite_difference_through_seed_inverse_projection",
        "distortion_aware": bool(_has_distortion(calibration)),
        "perturbation_px": 1.0,
        "summary_worst_direction_cm": _summary(values),
        "table": rows,
    }


def observation_perturbation_bootstrap(
    calibration: Mapping[str, Any],
    *,
    draws: int = 200,
    sigma_px: float = 1.0,
    random_seed: int = 20260712,
) -> dict[str, Any]:
    """Refit H after perturbing the seed's actual image observations."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if sigma_px <= 0.0:
        raise ValueError("observation sigma must be positive")
    image_points = calibration.get("image_pts")
    world_points = calibration.get("world_pts")
    if not isinstance(image_points, list) or not isinstance(world_points, list) or len(image_points) != len(world_points):
        return _absent("paired_seed_image_pts_world_pts_missing")
    if len(image_points) < 4:
        return _absent("fewer_than_four_seed_observations", observation_count=len(image_points))
    image = np.asarray(image_points, dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64)
    canonical_names = list(CANONICAL_SENSITIVITY_POINTS_M)
    canonical_world = np.asarray([CANONICAL_SENSITIVITY_POINTS_M[name] for name in canonical_names], dtype=np.float64)
    seed_pixels = np.asarray(project_planar_points(calibration["homography"], canonical_world), dtype=np.float64)
    values_by_point: dict[str, list[float]] = {name: [] for name in canonical_names}
    components_by_point: dict[str, list[list[float]]] = {name: [] for name in canonical_names}
    rng = np.random.default_rng(random_seed)
    failed = 0
    for _draw_index in range(draws):
        perturbed = image + rng.normal(0.0, sigma_px, size=image.shape)
        try:
            fitted = homography_from_planar_points(world.tolist(), perturbed.tolist())
            estimated_world = np.asarray(project_image_points_to_world(fitted, seed_pixels), dtype=np.float64)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            failed += 1
            continue
        deltas_cm = (estimated_world - canonical_world) * 100.0
        for index, name in enumerate(canonical_names):
            axes = _boundary_normal_axes(name)
            components = [float(deltas_cm[index, axis]) for axis in axes]
            values_by_point[name].append(max(abs(value) for value in components))
            components_by_point[name].append(components)
    table = []
    for name in canonical_names:
        axes = _boundary_normal_axes(name)
        values = values_by_point[name]
        table.append(
            {
                "point": name,
                "world_xy_m": list(CANONICAL_SENSITIVITY_POINTS_M[name]),
                "boundary_normal_axes": ["x" if axis == 0 else "y" for axis in axes],
                "boundary_normal_abs_displacement_cm": _summary(values),
                "signed_components_cm": {
                    ("x" if axis == 0 else "y"): _signed_summary(
                        row[position] for row in components_by_point[name]
                    )
                    for position, axis in enumerate(axes)
                },
            }
        )
    return {
        "status": PRESENT if draws - failed > 0 else MISSING,
        "metric_name": "observation_perturbation_bootstrap",
        "solver": "court_calibration.homography_from_planar_points",
        "observation_model": "independent_zero_mean_gaussian_xy_pixels",
        "sigma_px": float(sigma_px),
        "draws_requested": int(draws),
        "draws_completed": int(draws - failed),
        "refit_failure_count": int(failed),
        "random_seed": int(random_seed),
        "observation_count": int(len(image)),
        "boundary_normal_definition": "maximum absolute component across every call-line normal incident to the canonical point",
        "table": table,
    }


def render_m1_overlay(
    video_path: Path,
    calibration: Mapping[str, Any],
    frame_index: int,
    out_path: Path,
    *,
    title: str,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"failed to decode overlay frame {frame_index} from {video_path}")
    template = get_court_template(str(calibration.get("sport", "pickleball")))
    for line_id, endpoints in template.line_segments_m.items():
        points = project_world_points_distortion_aware(calibration, _sample_world_segment(endpoints, 65))
        visible = np.isfinite(points).all(axis=1)
        if not visible.any():
            continue
        color = (0, 165, 255) if line_id == "net" else (40, 230, 70)
        cv2.polylines(frame, [np.round(points[visible]).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, f"{title} | reported M1 worst frame {frame_index}", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise ValueError(f"failed to write overlay: {out_path}")


def source_provenance(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    result = []
    for value in paths:
        path = Path(value)
        result.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                "worktree_policy": "current_worktree_state_imported_read_only",
            }
        )
    return result


def _aggregate_m1_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residuals = [
        float(value)
        for row in rows
        if row.get("status") == PRESENT
        for value in row.get("residuals_px", [])
    ]
    visible = sum(int(row.get("visible_sample_count", 0)) for row in rows)
    evidence = sum(int(row.get("evidence_sample_count", 0)) for row in rows)
    overflow = sum(int(row.get("overflow_count", 0)) for row in rows)
    line_samples: list[dict[str, Any]] = []
    for row in rows:
        for line in row.get("lines", []):
            line_samples.append({**line, "frame_index": row.get("frame_index")})
    aggregate_lines = _merge_m1_group_rows(line_samples, group_key="line_id")
    aggregate_buckets: dict[str, list[dict[str, Any]]] = {}
    for bucket_family, group_key in (
        ("near_far_side", "near_far_bucket"),
        ("edge_center", "edge_center_bucket"),
        ("occlusion", "occlusion_bucket"),
    ):
        bucket_rows = [
            bucket
            for row in rows
            for bucket in row.get("buckets", {}).get(bucket_family, [])
        ]
        aggregate_buckets[bucket_family] = _merge_m1_group_rows(bucket_rows, group_key=group_key)
    scored_rows = [row for row in rows if row.get("status") == PRESENT and row.get("residual_px")]
    worst = sorted(
        scored_rows,
        key=lambda row: (float(row["residual_px"]["median"]), -float(row["evidence_coverage_fraction"])),
        reverse=True,
    )
    return {
        "status": PRESENT if residuals else MISSING,
        "reason": None if residuals else "no_paired_edge_evidence_in_frozen_seed_windows",
        "residual_px": _summary(residuals, p90=True) if residuals else None,
        "visible_sample_count": visible,
        "evidence_sample_count": evidence,
        "overflow_count": overflow,
        "evidence_coverage_fraction": evidence / visible if visible else 0.0,
        "coverage_weighted_rollup": _coverage_weighted_rollup(aggregate_lines),
        "per_line": aggregate_lines,
        "buckets": aggregate_buckets,
        "per_frame": list(rows),
        "worst_frame_indexes": [int(row["frame_index"]) for row in worst],
    }


def _group_m1_samples(samples: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample.get(key, "unknown")), []).append(sample)
    rows = []
    for name in sorted(groups):
        group = groups[name]
        residuals = [float(item["abs_residual_px"]) for item in group if item.get("abs_residual_px") is not None]
        visible = len(group)
        evidence = len(residuals)
        rows.append(
            {
                key: name,
                "visible_sample_count": visible,
                "evidence_sample_count": evidence,
                "overflow_count": visible - evidence,
                "evidence_coverage_fraction": evidence / visible if visible else 0.0,
                "residual_px": _summary(residuals, p90=True) if residuals else None,
                "residuals_px": residuals,
            }
        )
    return rows


def _merge_m1_group_rows(rows: Sequence[Mapping[str, Any]], *, group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    merged = []
    for name in sorted(groups):
        group = groups[name]
        residuals = [float(value) for row in group for value in row.get("residuals_px", [])]
        visible = sum(int(row.get("visible_sample_count", 0)) for row in group)
        evidence = len(residuals)
        merged.append(
            {
                group_key: name,
                "visible_sample_count": visible,
                "evidence_sample_count": evidence,
                "overflow_count": visible - evidence,
                "evidence_coverage_fraction": evidence / visible if visible else 0.0,
                "residual_px": _summary(residuals, p90=True) if residuals else None,
                "residuals_px": residuals,
            }
        )
    return merged


def _coverage_weighted_rollup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weighted_rows = [
        row
        for row in rows
        if isinstance(row.get("residual_px"), Mapping)
        and row["residual_px"].get("median") is not None
        and int(row.get("evidence_sample_count", 0)) > 0
    ]
    denominator = sum(int(row["evidence_sample_count"]) for row in weighted_rows)
    if denominator <= 0:
        return _absent("no_covered_groups")

    def weighted(field: str) -> float:
        return sum(
            float(row["residual_px"][field]) * int(row["evidence_sample_count"])
            for row in weighted_rows
        ) / denominator

    visible = sum(int(row.get("visible_sample_count", 0)) for row in rows)
    evidence = sum(int(row.get("evidence_sample_count", 0)) for row in rows)
    return {
        "method": "per-group residual summaries weighted by evidenced sample count; coverage and overflow remain separate",
        "weighted_median_px": weighted("median"),
        "weighted_p90_px": weighted("p90"),
        "weighted_max_px": weighted("max"),
        "visible_sample_count": visible,
        "evidence_sample_count": evidence,
        "overflow_count": visible - evidence,
        "evidence_coverage_fraction": evidence / visible if visible else 0.0,
    }


def _near_far_bucket(world_y_m: float) -> str:
    if world_y_m < -1e-9:
        return "near"
    if world_y_m > 1e-9:
        return "far"
    return "center"


def _paired_gradient_paint_center(
    gray: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    window: int,
) -> dict[str, Any]:
    """Fit both paint-band edges and return their subpixel midpoint."""

    step = 0.25
    pad = 3.0
    offsets = np.arange(-window - pad, window + pad + step * 0.5, step, dtype=np.float64)
    coordinates = point[None, :] + offsets[:, None] * normal[None, :]
    profile = _bilinear(gray, coordinates)
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    smooth = np.convolve(profile, kernel, mode="same")
    gradient = np.gradient(smooth, step)
    positive = _local_peak_indexes(gradient)
    negative = _local_peak_indexes(-gradient)
    pairs: list[dict[str, float]] = []
    for left_index in positive:
        left_offset = _parabolic_peak_offset(offsets, gradient, left_index)
        left_strength = _parabolic_peak_value(gradient, left_index)
        for right_index in negative:
            if right_index <= left_index:
                continue
            right_offset = _parabolic_peak_offset(offsets, -gradient, right_index)
            width = right_offset - left_offset
            if not 1.25 <= width <= 16.0:
                continue
            right_strength = _parabolic_peak_value(-gradient, right_index)
            edge_strength = min(left_strength, right_strength)
            if edge_strength < 6.0:
                continue
            interior = profile[(offsets >= left_offset + 0.4) & (offsets <= right_offset - 0.4)]
            outside = profile[
                ((offsets >= left_offset - 2.5) & (offsets <= left_offset - 0.5))
                | ((offsets >= right_offset + 0.5) & (offsets <= right_offset + 2.5))
            ]
            if interior.size == 0 or outside.size == 0:
                continue
            contrast = float(np.median(interior) - np.median(outside))
            if contrast < 8.0:
                continue
            center = (left_offset + right_offset) / 2.0
            if abs(center) > window:
                continue
            symmetry = min(left_strength, right_strength) / max(left_strength, right_strength)
            if symmetry < 0.28:
                continue
            pairs.append(
                {
                    "center": float(center),
                    "left": float(left_offset),
                    "right": float(right_offset),
                    "width": float(width),
                    "edge_strength": float(edge_strength),
                    "contrast": contrast,
                    "symmetry": float(symmetry),
                }
            )
    if not pairs:
        return {
            "evidence_status": MISSING,
            "overflow_reason": "no_valid_paired_gradient_edges_inside_seed_window",
            "signed_center_offset_from_seed_px": None,
            "edge_offsets_from_seed_px": None,
            "paint_band_width_px": None,
            "edge_strength": None,
            "interior_contrast": None,
        }
    # Semantic identity is frozen around the seed. Prefer the closest valid
    # center; strength breaks only near-ties, so a remote parallel distractor
    # cannot capture a fit merely by being brighter.
    best = min(pairs, key=lambda pair: (round(abs(pair["center"]), 3), -(pair["edge_strength"] + pair["contrast"])))
    return {
        "evidence_status": PRESENT,
        "overflow_reason": None,
        "signed_center_offset_from_seed_px": best["center"],
        "edge_offsets_from_seed_px": [best["left"], best["right"]],
        "paint_band_width_px": best["width"],
        "edge_strength": best["edge_strength"],
        "interior_contrast": best["contrast"],
        "edge_strength_symmetry": best["symmetry"],
    }


def _local_peak_indexes(values: np.ndarray) -> list[int]:
    if len(values) < 3:
        return []
    return [
        index
        for index in range(1, len(values) - 1)
        if values[index] >= values[index - 1] and values[index] > values[index + 1]
    ]


def _parabolic_peak_offset(offsets: np.ndarray, values: np.ndarray, index: int) -> float:
    delta = _parabolic_delta(values, index)
    step = float(offsets[1] - offsets[0])
    return float(offsets[index] + delta * step)


def _parabolic_peak_value(values: np.ndarray, index: int) -> float:
    delta = _parabolic_delta(values, index)
    left, center, right = [float(value) for value in values[index - 1 : index + 2]]
    return float(center - 0.25 * (left - right) * delta)


def _parabolic_delta(values: np.ndarray, index: int) -> float:
    left, center, right = [float(value) for value in values[index - 1 : index + 2]]
    denominator = left - 2.0 * center + right
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _bilinear(image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(points_xy[:, 0], 0.0, width - 1.001)
    y = np.clip(points_xy[:, 1], 0.0, height - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    return (
        image[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + image[y0, x1] * dx * (1.0 - dy)
        + image[y1, x0] * (1.0 - dx) * dy
        + image[y1, x1] * dx * dy
    )


def _sample_world_segment(endpoints: Sequence[Sequence[float]], count: int) -> list[list[float]]:
    first = np.asarray(endpoints[0], dtype=np.float64)
    second = np.asarray(endpoints[1], dtype=np.float64)
    return [(first + t * (second - first)).tolist() for t in np.linspace(0.0, 1.0, count)]


def _has_distortion(calibration: Mapping[str, Any]) -> bool:
    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        return False
    return any(abs(float(value)) > 1e-12 for value in intrinsics.get("dist", []) or [])


def _image_pixels_to_world_distortion_aware(calibration: Mapping[str, Any], pixels: np.ndarray) -> np.ndarray:
    if not _has_distortion(calibration):
        return np.asarray(project_image_points_to_world(calibration["homography"], pixels), dtype=np.float64)
    intrinsics = calibration["intrinsics"]
    extrinsics = calibration.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        raise ValueError("distortion-aware inverse projection requires calibration extrinsics")
    camera = np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.asarray(intrinsics["dist"], dtype=np.float64)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera, dist).reshape(-1, 2)
    rays_camera = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    rotation = np.asarray(extrinsics["R"], dtype=np.float64)
    translation = np.asarray(extrinsics["t"], dtype=np.float64)
    camera_center_world = -(rotation.T @ translation)
    rays_world = (rotation.T @ rays_camera.T).T
    scales = -camera_center_world[2] / rays_world[:, 2]
    points_world = camera_center_world[None, :] + scales[:, None] * rays_world
    return points_world[:, :2]


def _camera_motion_labels(payload: Mapping[str, Any] | None) -> dict[int, str]:
    if payload is None:
        return {}
    result: dict[int, str] = {}
    for row in payload.get("frames", []):
        if not isinstance(row, Mapping) or row.get("frame_idx", row.get("frame_index")) is None:
            continue
        frame = int(row.get("frame_idx", row.get("frame_index")))
        matrix = np.asarray(row.get("matrix_t_to_ref", np.eye(3)), dtype=np.float64)
        magnitude = float(np.linalg.norm(matrix[:2, 2])) if matrix.shape == (3, 3) else 0.0
        result[frame] = "camera_moving" if magnitude > 1.0 else "static"
    return result


def _audit_net_evidence_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact_type = str(payload.get("artifact_type", "unknown"))
    for row in payload.get("net_observations", []) if isinstance(payload.get("net_observations"), list) else []:
        if isinstance(row, Mapping) and row.get("net_id") == "top_net":
            points = row.get("image_points")
            confidence = float(row.get("confidence", 0.0))
            if isinstance(points, list) and len(points) == 3 and confidence >= 0.7:
                return {
                    "comparable": True,
                    "reason": "trusted_semantic_top_net_triplet",
                    "artifact_type": artifact_type,
                    "coordinate_space": "calibration_native_pixels",
                    "source_image_size": None,
                    "segment_identity": "semantic_left_post_top_center_sag_right_post_top",
                    "confidence": confidence,
                    "semantic_top_net_triplet_px": [[float(v) for v in point] for point in points],
                }
            return {
                "comparable": False,
                "reason": "top_net_observation_lacks_trusted_semantic_post_center_triplet",
                "artifact_type": artifact_type,
                "coordinate_space": "calibration_native_pixels",
                "segment_identity": "declared_top_net_but_triplet_or_confidence_inadequate",
                "point_count": len(points) if isinstance(points, list) else 0,
                "confidence": confidence,
            }
    net = payload.get("net")
    if isinstance(net, Mapping) and isinstance(net.get("tape_line"), list) and len(net["tape_line"]) == 2:
        source = payload.get("source") if isinstance(payload.get("source"), Mapping) else {}
        size = source.get("image_size") if isinstance(source, Mapping) else None
        verification = payload.get("self_verification") if isinstance(payload.get("self_verification"), Mapping) else {}
        solver = payload.get("solver") if isinstance(payload.get("solver"), Mapping) else {}
        evidence = net.get("evidence") if isinstance(net.get("evidence"), Mapping) else {}
        return {
            "comparable": False,
            "reason": "net_anchor_tape_segment_extents_are_not_verified_semantic_posts",
            "artifact_type": artifact_type,
            "coordinate_space": "source_image_pixels",
            "source_image_size": size,
            "frame_role": source.get("frame_role") if isinstance(source, Mapping) else None,
            "segment_identity": "hough_tape_candidate_segment_extent; post_tops copy segment endpoints",
            "units": "pixels",
            "solver": solver.get("name"),
            "solver_strategy": solver.get("strategy"),
            "solver_confidence": payload.get("solver_confidence"),
            "net_candidate_confidence": net.get("confidence"),
            "detected_post_count": evidence.get("post_count"),
            "needs_user_confirmation": payload.get("needs_user_confirmation"),
            "self_verification_status": verification.get("status"),
            "self_verification_promotion_allowed": verification.get("promotion_allowed"),
            "self_verification_reasons": verification.get("reasons", []),
            "notes": payload.get("notes", []),
            "comparison_rejected": "a partial/fail-closed Hough segment cannot be scaled into a full post-to-post sag curve",
        }
    return {
        "comparable": False,
        "reason": "trusted_semantic_net_evidence_missing",
        "artifact_type": artifact_type,
        "coordinate_space": None,
        "segment_identity": None,
    }


def _project_pb_camera(xyz_ft: np.ndarray, segment: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    position = segment.get("position")
    orientation = segment.get("orientation")
    if not isinstance(position, Mapping) or not isinstance(orientation, Mapping):
        raise ValueError("pb camera segment lacks position/orientation")
    center = np.asarray([position[axis] for axis in ("x", "y", "z")], dtype=np.float64)
    rotation = Rotation.from_euler(
        "zyx",
        [-float(orientation["yaw"]), float(orientation["pitch"]), -float(orientation["roll"])],
    ).as_matrix()
    vectors = (rotation @ (xyz_ft - center).T).T
    camera_xyz = np.column_stack([vectors[:, 1], -vectors[:, 2], vectors[:, 0]])
    focal = width / (2.0 * math.tan(float(segment["fov"]) / 2.0))
    return np.column_stack(
        [
            focal * camera_xyz[:, 0] / camera_xyz[:, 2] + width / 2.0,
            focal * camera_xyz[:, 1] / camera_xyz[:, 2] + height / 2.0,
        ]
    )


def _ours_m_to_pb_feet(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [points[:, 0] / 0.3048 + 10.0, 22.0 - points[:, 1] / 0.3048, points[:, 2] / 0.3048]
    )


def _boundary_normal_axes(point_name: str) -> tuple[int, ...]:
    if "corner" in point_name:
        return (0, 1)
    if "sideline" in point_name:
        return (0,)
    return (1,)


def _signed_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _input_hashes(
    *,
    calibration_path: Path,
    video_path: Path | None,
    camera_motion_path: Path | None,
    net_evidence_path: Path | None,
    pb_export_path: Path | None,
) -> dict[str, Any]:
    paths = {
        "calibration": calibration_path,
        "video": video_path,
        "camera_motion": camera_motion_path,
        "net_evidence": net_evidence_path,
        "pbvision_export": pb_export_path,
    }
    return {
        role: (
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            if path is not None and path.is_file()
            else None
        )
        for role, path in paths.items()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary(values: Iterable[float], *, p90: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    result: dict[str, Any] = {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
    if p90:
        result["p90"] = float(np.percentile(array, 90))
    return result


def _absent(reason: str, **details: Any) -> dict[str, Any]:
    return {"status": MISSING, "reason": reason, **details}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _clip_id(root: Path, calibration: Mapping[str, Any]) -> str:
    source = calibration.get("clip") or calibration.get("clip_id")
    return str(source) if source else root.name
