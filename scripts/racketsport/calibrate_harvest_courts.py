#!/usr/bin/env python3
"""Freeze per-source court calibrations for the 2026-07-06 online harvest clips.

The owner-labeled CVAT task has one still frame per legal harvest source. This
tool keeps that source-level contract explicit: it parses the CVAT image export,
applies the manager-reviewed stray-drop rule, solves each usable source frame
with the existing metric camera fitter, grades the result, and maps all
prelabeled rally clips back to their source calibration coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import homography_from_planar_points  # noqa: E402
from threed.racketsport.court_calibration_metric15 import (  # noqa: E402
    METRIC15_SOURCE_TAG,
    MIN_REVIEWED_CORRESPONDENCES,
    fit_single_view_metric_camera,
    load_reviewed_court_keypoints_15pt,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME  # noqa: E402
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES  # noqa: E402


GRADE_MANUAL_BAR = "manual_bar"
GRADE_AUTO_BAR = "auto_bar"
GRADE_FAILED = "failed"
CALIBRATED_GRADES = {GRADE_MANUAL_BAR, GRADE_AUTO_BAR}

MANUAL_MEDIAN_GATE_PX = 4.8
MANUAL_P95_GATE_PX = 12.3
AUTO_P95_GATE_PX = 20.0

HELDOUT_SOURCE_IDS = frozenset({"pwxNwFfYQlQ", "vQhtz8l6VqU"})
DECLARED_SKIP_SOURCE_IDS = frozenset({"Ezz6HDNHlnk"})
TENNIS_OVERLAY_PARTIAL_SOURCE_IDS = frozenset({"_L0HVmAlCQI", "wBu8bC4OfUY"})

DEFAULT_CVAT_EXPORT_XML = Path("cvat_upload/exports/court_keypoints_20260707/annotations.xml")
DEFAULT_TASKSET_MANIFEST = Path("cvat_upload/court_keypoints_20260707/taskset_manifest.json")
DEFAULT_PACKAGE_MANIFEST = Path("cvat_upload/court_keypoints_20260707/package_manifest.json")
DEFAULT_OUT_DIR = Path("data/online_harvest_20260706/court_calibrations")
DEFAULT_PRELABELS_DIR = Path("data/online_harvest_20260706/prelabels")
DEFAULT_BASELINE_CALIBRATION_DIR = Path("data/online_harvest_20260706/court_calibrations")

ORIGINAL_NON_HELDOUT_SOURCE_IDS = (
    "73VurrTKCZ8",
    "HyUqT7zFiwk",
    "zwCtH_i1_S4",
    "_L0HVmAlCQI",
    "wBu8bC4OfUY",
    "Ezz6HDNHlnk",
)


@dataclass(frozen=True)
class CvatFramePoints:
    source_id: str
    clip_id: str
    absolute_frame_index: int
    frame_name: str
    image_size: tuple[int, int]
    points: dict[str, tuple[float, float]]
    quality_flags: list[str]

    @property
    def point_count(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class FrameSolve:
    frame: CvatFramePoints
    grade: str
    failure_reason: str | None
    calibration: dict[str, Any] | None
    reprojection: dict[str, float] | None
    residuals_by_keypoint: dict[str, float]


def parse_cvat_image_points(path: str | Path) -> list[CvatFramePoints]:
    """Parse a CVAT for images 1.1 point-label export into per-image frames."""

    xml_path = Path(path)
    root = ET.parse(xml_path).getroot()
    frames: list[CvatFramePoints] = []
    for image_el in root.findall("image"):
        frame_name = str(image_el.attrib.get("name") or "")
        source_id, clip_id, absolute_frame_index = _parse_frame_name(frame_name)
        width = int(float(image_el.attrib["width"]))
        height = int(float(image_el.attrib["height"]))
        points: dict[str, tuple[float, float]] = {}
        for point_el in image_el.findall("points"):
            label = str(point_el.attrib.get("label") or "")
            if not label:
                continue
            if label in points:
                raise ValueError(f"{frame_name}: duplicate point label {label!r}")
            points[label] = _parse_cvat_point(point_el.attrib.get("points") or "")
        frames.append(
            CvatFramePoints(
                source_id=source_id,
                clip_id=clip_id,
                absolute_frame_index=absolute_frame_index,
                frame_name=frame_name,
                image_size=(width, height),
                points=points,
                quality_flags=[],
            )
        )
    return frames


def gather_source_frames(
    frames: Sequence[CvatFramePoints],
    *,
    declared_skip_source_ids: Iterable[str] = DECLARED_SKIP_SOURCE_IDS,
    heldout_source_ids: Iterable[str] = HELDOUT_SOURCE_IDS,
    tennis_overlay_partial_source_ids: Iterable[str] = TENNIS_OVERLAY_PARTIAL_SOURCE_IDS,
) -> tuple[dict[str, list[CvatFramePoints]], list[dict[str, Any]]]:
    """Group usable labeled frames by source after owner/manager import rules."""

    declared_skip = set(declared_skip_source_ids)
    heldout = set(heldout_source_ids)
    tennis_overlay = set(tennis_overlay_partial_source_ids)
    grouped: dict[str, list[CvatFramePoints]] = {}
    dropped: list[dict[str, Any]] = []
    for frame in frames:
        if frame.source_id in heldout:
            raise ValueError(
                f"held-out source {frame.source_id!r} has a labeled court frame "
                f"{frame.frame_name}; stop and audit the taskset"
            )
        if frame.source_id in declared_skip:
            dropped.append(
                {
                    "source_id": frame.source_id,
                    "frame_name": frame.frame_name,
                    "reason": "owner_declared_skip_stray_drop",
                    "dropped_point_count": frame.point_count,
                }
            )
            continue
        flags = list(frame.quality_flags)
        if frame.source_id in tennis_overlay:
            flags.append("tennis_overlay_partial")
        grouped.setdefault(frame.source_id, []).append(replace(frame, quality_flags=flags))
    return grouped, dropped


def grade_calibration(
    *,
    median_px: float | None,
    p95_px: float | None,
    failure_reason: str | None = None,
) -> tuple[str, str | None]:
    if failure_reason is not None:
        return GRADE_FAILED, failure_reason
    if median_px is None or p95_px is None:
        return GRADE_FAILED, "solver_infeasible"
    if median_px <= MANUAL_MEDIAN_GATE_PX and p95_px <= MANUAL_P95_GATE_PX:
        return GRADE_MANUAL_BAR, None
    if p95_px <= AUTO_P95_GATE_PX:
        return GRADE_AUTO_BAR, None
    return GRADE_FAILED, "reprojection_above_auto_bar"


def solve_source_frames(
    source_id: str,
    frames: Sequence[CvatFramePoints],
    *,
    cvat_export_xml: Path,
    stray_drops: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frame_solves = [_solve_frame(frame) for frame in frames]
    successful = [solve for solve in frame_solves if solve.calibration is not None and solve.reprojection is not None]
    selected = _select_frozen_solve(successful)
    source_drops = [dict(drop) for drop in stray_drops if drop.get("source_id") == source_id]

    if selected is None:
        failure_reason = _source_failure_reason(frame_solves)
        grade, reason = grade_calibration(median_px=None, p95_px=None, failure_reason=failure_reason)
        return _failed_source_artifact(
            source_id=source_id,
            grade=grade,
            failure_reason=reason or failure_reason,
            frames=frames,
            frame_solves=frame_solves,
            cvat_export_xml=cvat_export_xml,
            stray_drops=source_drops,
        )

    median_px = float(selected.reprojection["median"])
    p95_px = float(selected.reprojection["p95"])
    grade, reason = grade_calibration(median_px=median_px, p95_px=p95_px)
    if grade == GRADE_FAILED:
        return _failed_source_artifact(
            source_id=source_id,
            grade=grade,
            failure_reason=reason or "reprojection_above_auto_bar",
            frames=frames,
            frame_solves=frame_solves,
            cvat_export_xml=cvat_export_xml,
            stray_drops=source_drops,
        )

    assert selected.calibration is not None
    artifact = dict(selected.calibration)
    artifact.update(
        {
            "artifact_type": "racketsport_harvest_source_court_calibration",
            "calibration_grade": grade,
            "source_id": source_id,
            "selection_rule": "lowest p95 reprojection, then lowest median, then highest point count, then frame name",
            "selected_frame": selected.frame.frame_name,
            "selected_absolute_frame_index": selected.frame.absolute_frame_index,
            "provenance": _provenance_payload(
                source_id=source_id,
                frames=frames,
                cvat_export_xml=cvat_export_xml,
                stray_drops=source_drops,
            ),
            "per_frame_reprojection_stats": [_frame_solve_payload(solve) for solve in frame_solves],
            "cross_frame_consistency": _cross_frame_consistency(successful),
            "quality_flags": sorted({flag for frame in frames for flag in frame.quality_flags}),
            "failure_reason": None,
        }
    )
    return artifact


def build_coverage_report(
    *,
    prelabels_dir: str | Path,
    source_grades: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for clip_dir in sorted(Path(prelabels_dir).iterdir()):
        if not clip_dir.is_dir() or not (clip_dir / "ball_track.json").is_file():
            continue
        clip_id = clip_dir.name
        source_id = _source_id_from_clip_id(clip_id)
        source_grade = source_grades.get(source_id, {})
        grade = str(source_grade.get("calibration_grade") or GRADE_FAILED)
        calibrated = grade in CALIBRATED_GRADES
        rows.append(
            {
                "clip_id": clip_id,
                "source_id": source_id,
                "prelabel_dir": clip_dir.as_posix(),
                "calibrated": calibrated,
                "grade": grade,
                "artifact_path": source_grade.get("artifact_path") if calibrated else None,
                "failure_reason": None if calibrated else source_grade.get("failure_reason", "no_source_calibration"),
            }
        )

    covered_by_grade: dict[str, int] = {}
    grade_counts: dict[str, int] = {}
    for row in rows:
        grade = str(row["grade"])
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        if row["calibrated"]:
            covered_by_grade[grade] = covered_by_grade.get(grade, 0) + 1
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_harvest_court_calibration_coverage_report",
        "summary": {
            "clip_count": len(rows),
            "covered_clip_count": sum(1 for row in rows if row["calibrated"]),
            "failed_clip_count": sum(1 for row in rows if not row["calibrated"]),
            "covered_by_grade": dict(sorted(covered_by_grade.items())),
            "clip_count_by_grade": dict(sorted(grade_counts.items())),
        },
        "clips": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.corrected_gt_root is not None:
        return _run_corrected_gt(args)

    cvat_export_xml = args.cvat_export_xml
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    taskset = _read_json(args.taskset_manifest)
    frames = parse_cvat_image_points(cvat_export_xml)
    grouped, dropped = gather_source_frames(frames)
    source_ids = sorted(set(taskset.get("non_heldout_source_ids", [])) | set(grouped) | {str(drop["source_id"]) for drop in dropped})

    source_artifacts: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        if source_id in DECLARED_SKIP_SOURCE_IDS and source_id not in grouped:
            source_drops = [dict(drop) for drop in dropped if drop.get("source_id") == source_id]
            artifact = _failed_source_artifact(
                source_id=source_id,
                grade=GRADE_FAILED,
                failure_reason="owner_declared_skip_stray_drop",
                frames=[],
                frame_solves=[],
                cvat_export_xml=cvat_export_xml,
                stray_drops=source_drops,
            )
        elif source_id not in grouped:
            artifact = _failed_source_artifact(
                source_id=source_id,
                grade=GRADE_FAILED,
                failure_reason="no_usable_labeled_frames",
                frames=[],
                frame_solves=[],
                cvat_export_xml=cvat_export_xml,
                stray_drops=[],
            )
        else:
            artifact = solve_source_frames(
                source_id,
                grouped[source_id],
                cvat_export_xml=cvat_export_xml,
                stray_drops=dropped,
            )
        artifact_path = out_dir / f"{source_id}.json"
        _write_json(artifact_path, artifact)
        source_artifacts[source_id] = artifact

    source_grades = {
        source_id: {
            "calibration_grade": artifact["calibration_grade"],
            "failure_reason": artifact.get("failure_reason"),
            "artifact_path": (out_dir / f"{source_id}.json").as_posix(),
        }
        for source_id, artifact in source_artifacts.items()
    }
    coverage = build_coverage_report(prelabels_dir=args.prelabels_dir, source_grades=source_grades)
    coverage_path = out_dir / "coverage_report.json"
    _write_json(coverage_path, coverage)

    return {
        "source_artifacts": {
            source_id: {
                "path": (out_dir / f"{source_id}.json").as_posix(),
                "calibration_grade": artifact["calibration_grade"],
                "median_px": _nested_get(artifact, ("reprojection_error_px", "median")),
                "p95_px": _nested_get(artifact, ("reprojection_error_px", "p95")),
                "failure_reason": artifact.get("failure_reason"),
            }
            for source_id, artifact in source_artifacts.items()
        },
        "coverage_report": coverage_path.as_posix(),
        "coverage_summary": coverage["summary"],
    }


def _run_corrected_gt(args: argparse.Namespace) -> dict[str, Any]:
    """Solve the corrected-r2 reviewed JSON inputs without changing legacy mode."""

    corrected_root = args.corrected_gt_root
    manifest_path, manifest_md5 = _verify_corrected_gt_manifest(corrected_root)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_paths = {
        path.parent.parent.name: path
        for path in sorted(corrected_root.glob("*/labels/court_keypoints.json"))
    }
    unexpected = sorted(set(label_paths) - set(ORIGINAL_NON_HELDOUT_SOURCE_IDS))
    if unexpected:
        raise ValueError(f"corrected GT root contains unexpected source IDs: {unexpected}")
    if set(label_paths) & HELDOUT_SOURCE_IDS:
        raise ValueError("corrected GT root contains held-out harvest source labels")

    source_artifacts: dict[str, dict[str, Any]] = {}
    for source_id in ORIGINAL_NON_HELDOUT_SOURCE_IDS:
        if source_id in label_paths:
            artifact = solve_corrected_source_labels(
                source_id,
                label_paths[source_id],
                corrected_gt_root=corrected_root,
                corrected_gt_manifest=manifest_path,
                corrected_gt_manifest_md5=manifest_md5,
            )
        else:
            artifact = _load_owner_rejected_baseline_artifact(source_id)
        artifact_path = out_dir / f"{source_id}.json"
        _write_json(artifact_path, artifact)
        source_artifacts[source_id] = artifact

    source_grades = {
        source_id: {
            "calibration_grade": artifact["calibration_grade"],
            "failure_reason": artifact.get("failure_reason"),
            "artifact_path": (out_dir / f"{source_id}.json").as_posix(),
        }
        for source_id, artifact in source_artifacts.items()
    }
    coverage = build_coverage_report(prelabels_dir=args.prelabels_dir, source_grades=source_grades)
    coverage.update(
        {
            "input_mode": "corrected_gt_root",
            "corrected_gt_root": corrected_root.as_posix(),
            "corrected_gt_manifest": manifest_path.as_posix(),
            "corrected_gt_manifest_md5": manifest_md5,
        }
    )
    coverage_path = out_dir / "coverage_report.json"
    _write_json(coverage_path, coverage)

    return {
        "input_mode": "corrected_gt_root",
        "corrected_gt_manifest": manifest_path.as_posix(),
        "corrected_gt_manifest_md5": manifest_md5,
        "source_artifacts": {
            source_id: {
                "path": (out_dir / f"{source_id}.json").as_posix(),
                "calibration_grade": artifact["calibration_grade"],
                "median_px": _nested_get(artifact, ("reprojection_error_px", "median")),
                "p95_px": _nested_get(artifact, ("reprojection_error_px", "p95")),
                "failure_reason": artifact.get("failure_reason"),
                "labeled_frame_count": len(artifact.get("per_frame_reprojection_stats", [])),
            }
            for source_id, artifact in source_artifacts.items()
        },
        "coverage_report": coverage_path.as_posix(),
        "coverage_summary": coverage["summary"],
    }


def solve_corrected_source_labels(
    source_id: str,
    label_path: Path,
    *,
    corrected_gt_root: Path,
    corrected_gt_manifest: Path,
    corrected_gt_manifest_md5: str,
) -> dict[str, Any]:
    reviewed = load_reviewed_court_keypoints_15pt(label_path)
    width, height = reviewed.source_resolution
    label_width, label_height = reviewed.label_coordinate_space
    if width <= 0 or height <= 0 or label_width <= 0 or label_height <= 0:
        raise ValueError(f"{label_path}: invalid declared image dimensions")
    scale_x, scale_y = width / label_width, height / label_height

    frame_points: list[CvatFramePoints] = []
    frame_image_paths: list[Path] = []
    frame_dir = label_path.parent / "court_keypoint_frames"
    for frame in reviewed.frames:
        frame_path = frame_dir / frame.frame
        if not frame_path.is_file():
            raise ValueError(f"{label_path}: referenced frame image not found: {frame_path}")
        absolute_index = _absolute_frame_index_from_reviewed_name(frame.frame)
        frame_points.append(
            CvatFramePoints(
                source_id=source_id,
                clip_id=source_id,
                absolute_frame_index=absolute_index,
                frame_name=frame.frame,
                image_size=(int(round(width)), int(round(height))),
                points={
                    name: (float(frame.keypoints[name][0]) * scale_x, float(frame.keypoints[name][1]) * scale_y)
                    for name in PICKLEBALL_COURT_KEYPOINT_NAMES
                },
                quality_flags=["corrected_r2_owner_reviewed"],
            )
        )
        frame_image_paths.append(frame_path)
    if not frame_points:
        raise ValueError(f"{label_path}: no complete reviewed frames")

    aggregated_points = {
        name: (
            _median([frame.points[name][0] for frame in frame_points]),
            _median([frame.points[name][1] for frame in frame_points]),
        )
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES
    }
    pooled_frame = CvatFramePoints(
        source_id=source_id,
        clip_id=source_id,
        absolute_frame_index=frame_points[0].absolute_frame_index,
        frame_name=f"{source_id}__corrected_r2_pooled",
        image_size=frame_points[0].image_size,
        points=aggregated_points,
        quality_flags=["corrected_r2_owner_reviewed", "multi_frame_median_aggregate"],
    )
    object_points = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    image_points = [list(aggregated_points[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    try:
        fit = fit_single_view_metric_camera(object_points, image_points, (float(width), float(height)))
        homography = homography_from_planar_points(object_points, image_points)
    except Exception as exc:  # noqa: BLE001 - preserved in the lane artifact
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_harvest_source_court_calibration",
            "sport": "pickleball",
            "source_id": source_id,
            "calibration_grade": GRADE_FAILED,
            "failure_reason": f"solver_infeasible: {type(exc).__name__}: {exc}",
            "frozen_calibration": None,
            "reprojection_error_px": None,
            "per_frame_reprojection_stats": [],
            "provenance": _corrected_provenance(
                source_id,
                label_path,
                frame_points,
                frame_image_paths,
                corrected_gt_root,
                corrected_gt_manifest,
                corrected_gt_manifest_md5,
            ),
        }

    projected = _project_world_points(fit, object_points)
    per_frame_stats: list[dict[str, Any]] = []
    pooled_residuals: list[float] = []
    pooled_by_name: dict[str, list[float]] = {name: [] for name in PICKLEBALL_COURT_KEYPOINT_NAMES}
    for frame in frame_points:
        residual_by_name = {
            name: math.hypot(frame.points[name][0] - projected[idx][0], frame.points[name][1] - projected[idx][1])
            for idx, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
        }
        residuals = [residual_by_name[name] for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
        pooled_residuals.extend(residuals)
        for name, residual in residual_by_name.items():
            pooled_by_name[name].append(residual)
        stats = _residual_summary(residuals)
        frame_grade, frame_reason = grade_calibration(median_px=stats["median"], p95_px=stats["p95"])
        per_frame_stats.append(
            {
                "frame_name": frame.frame_name,
                "clip_id": frame.clip_id,
                "absolute_frame_index": frame.absolute_frame_index,
                "point_count": frame.point_count,
                "points_used": list(PICKLEBALL_COURT_KEYPOINT_NAMES),
                "grade": frame_grade,
                "failure_reason": frame_reason,
                "reprojection_error_px": {"median": stats["median"], "p95": stats["p95"]},
                "residual_summary_px": stats,
                "residuals_by_keypoint_px": residual_by_name,
            }
        )
    pooled_stats = _residual_summary(pooled_residuals)
    grade, failure_reason = grade_calibration(median_px=pooled_stats["median"], p95_px=pooled_stats["p95"])
    aggregate_residual_by_name = {name: _median(values) for name, values in pooled_by_name.items()}
    calibration = _calibration_payload(
        frame=pooled_frame,
        point_names=PICKLEBALL_COURT_KEYPOINT_NAMES,
        object_points=object_points,
        image_points=image_points,
        homography=homography,
        fit=fit,
        residuals_by_keypoint=aggregate_residual_by_name,
    )
    calibration.update(
        {
            "artifact_type": "racketsport_harvest_source_court_calibration",
            "calibration_grade": grade,
            "source_id": source_id,
            "failure_reason": failure_reason,
            "frozen_calibration": None if grade == GRADE_FAILED else "self",
            "selection_rule": "median aggregate of all corrected-r2 full15 frames; frozen bars scored on all frame/keypoint residuals",
            "selected_frame": None,
            "selected_absolute_frame_index": None,
            "reference_image_path": frame_image_paths[0].as_posix(),
            "reprojection_error_px": {"median": pooled_stats["median"], "p95": pooled_stats["p95"]},
            "pooled_reprojection_stats": pooled_stats,
            "per_keypoint_residual_px": [aggregate_residual_by_name[name] for name in PICKLEBALL_COURT_KEYPOINT_NAMES],
            "per_keypoint_residual_by_name_px": aggregate_residual_by_name,
            "per_frame_reprojection_stats": per_frame_stats,
            "solved_over_frames": [frame.absolute_frame_index for frame in frame_points],
            "cross_frame_consistency": _cross_frame_label_consistency(frame_points),
            "provenance": _corrected_provenance(
                source_id,
                label_path,
                frame_points,
                frame_image_paths,
                corrected_gt_root,
                corrected_gt_manifest,
                corrected_gt_manifest_md5,
            ),
        }
    )
    if grade == GRADE_FAILED:
        calibration["frozen_calibration"] = None
    return calibration


def _verify_corrected_gt_manifest(corrected_root: Path) -> tuple[Path, str]:
    manifest_path = corrected_root.parent.parent / "gt_corpus_manifest_r2.json"
    if not manifest_path.is_file():
        raise ValueError(f"corrected GT manifest not found next to root: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if Path(str(manifest.get("corrected_gt_root"))) != corrected_root:
        raise ValueError("corrected GT manifest root does not match --corrected-gt-root")
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        path = manifest_path.parent / relative
        if not path.is_file():
            raise ValueError(f"corrected GT manifest file missing: {path}")
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != str(entry["md5"]):
            raise ValueError(f"corrected GT md5 mismatch for {path}: {digest} != {entry['md5']}")
    return manifest_path, hashlib.md5(manifest_bytes).hexdigest()


def _load_owner_rejected_baseline_artifact(source_id: str) -> dict[str, Any]:
    if source_id not in DECLARED_SKIP_SOURCE_IDS | TENNIS_OVERLAY_PARTIAL_SOURCE_IDS:
        raise ValueError(f"corrected GT is missing usable source {source_id}")
    baseline_path = DEFAULT_BASELINE_CALIBRATION_DIR / f"{source_id}.json"
    artifact = _read_json(baseline_path)
    if artifact.get("calibration_grade") != GRADE_FAILED:
        raise ValueError(f"owner-rejected baseline unexpectedly calibrated: {source_id}")
    return artifact


def _corrected_provenance(
    source_id: str,
    label_path: Path,
    frames: Sequence[CvatFramePoints],
    frame_image_paths: Sequence[Path],
    corrected_root: Path,
    manifest_path: Path,
    manifest_md5: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "input_mode": "corrected_gt_root",
        "corrected_gt_root": corrected_root.as_posix(),
        "corrected_gt_label_path": label_path.as_posix(),
        "corrected_gt_manifest": manifest_path.as_posix(),
        "corrected_gt_manifest_md5": manifest_md5,
        "frames_used": [
            {
                "frame_name": frame.frame_name,
                "absolute_frame_index": frame.absolute_frame_index,
                "image_path": image_path.as_posix(),
                "image_size": list(frame.image_size),
                "point_count": frame.point_count,
                "points_used": list(PICKLEBALL_COURT_KEYPOINT_NAMES),
                "quality_flags": list(frame.quality_flags),
            }
            for frame, image_path in zip(frames, frame_image_paths, strict=True)
        ],
        "owner_rejected_source_ids": sorted(DECLARED_SKIP_SOURCE_IDS | TENNIS_OVERLAY_PARTIAL_SOURCE_IDS),
    }


def _project_world_points(fit: Any, object_points: Sequence[Sequence[float]]) -> list[list[float]]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    rotation = np.asarray(fit.R, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    camera = np.asarray(
        [[fit.fx, 0.0, fit.cx], [0.0, fit.fy, fit.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.asarray([fit.k1, fit.k2, 0.0, 0.0], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec,
        np.asarray(fit.t, dtype=np.float64).reshape(3, 1),
        camera,
        dist,
    )
    return [[float(x), float(y)] for x, y in projected.reshape(-1, 2)]


def _residual_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("residual summary requires non-empty values")
    return {
        "count": len(values),
        "median": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(float(value) for value in values),
        "rmse": math.sqrt(sum(float(value) ** 2 for value in values) / len(values)),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 50.0)


def _absolute_frame_index_from_reviewed_name(frame_name: str) -> int:
    digits = "".join(character for character in Path(frame_name).stem if character.isdigit())
    if not digits:
        raise ValueError(f"reviewed frame name has no numeric index: {frame_name}")
    return int(digits)


def _cross_frame_label_consistency(frames: Sequence[CvatFramePoints]) -> dict[str, Any]:
    if len(frames) == 1:
        return {
            "frame_count": 1,
            "per_keypoint_frame_spread_px": {name: 0.0 for name in PICKLEBALL_COURT_KEYPOINT_NAMES},
            "median_keypoint_spread_px": 0.0,
            "p95_keypoint_spread_px": 0.0,
            "note": "single_labeled_frame_for_static_source",
        }
    spreads: dict[str, float] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        best = 0.0
        for idx, first in enumerate(frames):
            for second in frames[idx + 1 :]:
                best = max(best, math.dist(first.points[name], second.points[name]))
        spreads[name] = best
    summary = _residual_summary(list(spreads.values()))
    return {
        "frame_count": len(frames),
        "per_keypoint_frame_spread_px": spreads,
        "median_keypoint_spread_px": summary["median"],
        "p95_keypoint_spread_px": summary["p95"],
        "note": "max pairwise label displacement by keypoint before median aggregation",
    }


def _solve_frame(frame: CvatFramePoints) -> FrameSolve:
    point_names = [name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name in frame.points]
    if len(point_names) < MIN_REVIEWED_CORRESPONDENCES:
        reason = f"insufficient_points: {len(point_names)} < {MIN_REVIEWED_CORRESPONDENCES}"
        grade, failure_reason = grade_calibration(median_px=None, p95_px=None, failure_reason=reason)
        return FrameSolve(frame, grade, failure_reason, None, None, {})

    object_points = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in point_names]
    image_points = [list(frame.points[name]) for name in point_names]
    try:
        fit = fit_single_view_metric_camera(object_points, image_points, tuple(float(v) for v in frame.image_size))
        homography = homography_from_planar_points(object_points, image_points)
    except Exception as exc:  # noqa: BLE001 - recorded as source evidence
        reason = f"solver_infeasible: {type(exc).__name__}: {exc}"
        grade, failure_reason = grade_calibration(median_px=None, p95_px=None, failure_reason=reason)
        return FrameSolve(frame, grade, failure_reason, None, None, {})

    reprojection = fit.reprojection_error_px.model_dump(mode="json")
    grade, failure_reason = grade_calibration(
        median_px=float(reprojection["median"]),
        p95_px=float(reprojection["p95"]),
    )
    residuals_by_keypoint = {
        name: float(residual)
        for name, residual in zip(point_names, fit.per_point_residual_px, strict=True)
    }
    calibration = _calibration_payload(
        frame=frame,
        point_names=point_names,
        object_points=object_points,
        image_points=image_points,
        homography=homography,
        fit=fit,
        residuals_by_keypoint=residuals_by_keypoint,
    )
    return FrameSolve(frame, grade, failure_reason, calibration, reprojection, residuals_by_keypoint)


def _calibration_payload(
    *,
    frame: CvatFramePoints,
    point_names: Sequence[str],
    object_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    homography: Sequence[Sequence[float]],
    fit: Any,
    residuals_by_keypoint: Mapping[str, float],
) -> dict[str, Any]:
    camera_center_world = _camera_center_from_pose(fit.R, fit.t)
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "homography": [[float(value) for value in row] for row in homography],
        "intrinsics": {
            "fx": float(fit.fx),
            "fy": float(fit.fy),
            "cx": float(fit.cx),
            "cy": float(fit.cy),
            "dist": [float(fit.k1), float(fit.k2), 0.0, 0.0],
            "source": METRIC15_SOURCE_TAG,
        },
        "image_size": [int(frame.image_size[0]), int(frame.image_size[1])],
        "extrinsics": {
            "R": fit.R,
            "t": fit.t,
            "camera_height_m": max(abs(float(camera_center_world[2])), 1e-6),
        },
        "reprojection_error_px": fit.reprojection_error_px.model_dump(mode="json"),
        "per_keypoint_residual_px": [float(residuals_by_keypoint[name]) for name in point_names],
        "per_keypoint_residual_by_name_px": dict(residuals_by_keypoint),
        "metric_confidence": _confidence_from_reprojection(
            float(fit.reprojection_error_px.median),
            float(fit.reprojection_error_px.p95),
        ),
        "capture_quality": {
            "grade": "good" if fit.reprojection_error_px.p95 <= MANUAL_P95_GATE_PX else "warn",
            "reasons": [
                "single_view_metric_camera_fit",
                f"distortion_model={fit.distortion_model}",
                f"point_count={len(point_names)}",
                *frame.quality_flags,
            ],
        },
        "image_pts": [[float(value) for value in point] for point in image_points],
        "world_pts": [[float(value) for value in point] for point in object_points],
        "keypoint_names": list(point_names),
        "source": METRIC15_SOURCE_TAG,
        "solved_over_frames": [int(frame.absolute_frame_index)],
        "distortion_model": fit.distortion_model,
        "identifiability_notes": list(fit.identifiability_notes),
    }


def _select_frozen_solve(solves: Sequence[FrameSolve]) -> FrameSolve | None:
    if not solves:
        return None
    return min(
        solves,
        key=lambda solve: (
            float(solve.reprojection["p95"]) if solve.reprojection else math.inf,
            float(solve.reprojection["median"]) if solve.reprojection else math.inf,
            -solve.frame.point_count,
            solve.frame.frame_name,
        ),
    )


def _cross_frame_consistency(solves: Sequence[FrameSolve]) -> dict[str, Any]:
    if not solves:
        return {
            "frame_count": 0,
            "translation_l2_m_spread": None,
            "rotation_deg_spread": None,
            "focal_px_spread": None,
            "note": "no_successful_frame_solves",
        }
    calibrations = [solve.calibration for solve in solves if solve.calibration is not None]
    if len(calibrations) == 1:
        return {
            "frame_count": 1,
            "translation_l2_m_spread": 0.0,
            "rotation_deg_spread": 0.0,
            "focal_px_spread": 0.0,
            "note": "single_labeled_frame_for_static_source",
        }
    translations = [cal["extrinsics"]["t"] for cal in calibrations]
    rotations = [cal["extrinsics"]["R"] for cal in calibrations]
    focals = [float(cal["intrinsics"]["fx"]) for cal in calibrations]
    return {
        "frame_count": len(calibrations),
        "translation_l2_m_spread": _max_translation_spread(translations),
        "rotation_deg_spread": _max_rotation_spread_deg(rotations),
        "focal_px_spread": max(focals) - min(focals),
        "note": "spread_across_successful_frame_solves",
    }


def _failed_source_artifact(
    *,
    source_id: str,
    grade: str,
    failure_reason: str,
    frames: Sequence[CvatFramePoints],
    frame_solves: Sequence[FrameSolve],
    cvat_export_xml: Path,
    stray_drops: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_solves = [solve for solve in frame_solves if solve.reprojection is not None]
    best = _select_frozen_solve(candidate_solves)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_harvest_source_court_calibration",
        "sport": "pickleball",
        "source_id": source_id,
        "calibration_grade": grade,
        "failure_reason": failure_reason,
        "frozen_calibration": None,
        "selected_frame": best.frame.frame_name if best is not None else None,
        "selected_absolute_frame_index": best.frame.absolute_frame_index if best is not None else None,
        "reprojection_error_px": best.reprojection if best is not None else None,
        "selection_rule": "lowest p95 reprojection, then lowest median, then highest point count, then frame name",
        "provenance": _provenance_payload(
            source_id=source_id,
            frames=frames,
            cvat_export_xml=cvat_export_xml,
            stray_drops=stray_drops,
        ),
        "per_frame_reprojection_stats": [_frame_solve_payload(solve) for solve in frame_solves],
        "cross_frame_consistency": _cross_frame_consistency(
            [solve for solve in frame_solves if solve.calibration is not None]
        ),
        "quality_flags": sorted({flag for frame in frames for flag in frame.quality_flags}),
    }


def _provenance_payload(
    *,
    source_id: str,
    frames: Sequence[CvatFramePoints],
    cvat_export_xml: Path,
    stray_drops: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "cvat_export_xml": cvat_export_xml.as_posix(),
        "owner_review_readme": "cvat_upload/exports/court_keypoints_20260707/README.md",
        "frames_used": [
            {
                "frame_name": frame.frame_name,
                "clip_id": frame.clip_id,
                "absolute_frame_index": frame.absolute_frame_index,
                "image_size": list(frame.image_size),
                "points_used": sorted(frame.points),
                "point_count": frame.point_count,
                "quality_flags": frame.quality_flags,
            }
            for frame in frames
        ],
        "stray_drops_applied": [dict(drop) for drop in stray_drops],
    }


def _frame_solve_payload(solve: FrameSolve) -> dict[str, Any]:
    return {
        "frame_name": solve.frame.frame_name,
        "clip_id": solve.frame.clip_id,
        "absolute_frame_index": solve.frame.absolute_frame_index,
        "point_count": solve.frame.point_count,
        "points_used": sorted(solve.frame.points),
        "grade": solve.grade,
        "failure_reason": solve.failure_reason,
        "reprojection_error_px": solve.reprojection,
        "residuals_by_keypoint_px": solve.residuals_by_keypoint,
    }


def _source_failure_reason(frame_solves: Sequence[FrameSolve]) -> str:
    reasons = [solve.failure_reason for solve in frame_solves if solve.failure_reason]
    if not reasons:
        return "no_successful_frame_solves"
    return "; ".join(dict.fromkeys(reasons))


def _parse_frame_name(frame_name: str) -> tuple[str, str, int]:
    stem = Path(frame_name).stem
    marker = "__abs_"
    if marker not in stem:
        raise ValueError(f"frame name lacks __abs_ marker: {frame_name}")
    before_abs, frame_text = stem.rsplit(marker, 1)
    if "__" not in before_abs:
        raise ValueError(f"frame name lacks source/clip separator: {frame_name}")
    source_id, clip_id = before_abs.split("__", 1)
    absolute_frame_text = frame_text.split("__", 1)[0]
    return source_id, clip_id, int(absolute_frame_text)


def _parse_cvat_point(raw: str) -> tuple[float, float]:
    first_pair = raw.split(";", 1)[0]
    xy = first_pair.split(",", 1)
    if len(xy) != 2:
        raise ValueError(f"invalid CVAT point coordinate: {raw!r}")
    return float(xy[0]), float(xy[1])


def _source_id_from_clip_id(clip_id: str) -> str:
    if "_rally_" not in clip_id:
        return clip_id
    return clip_id.split("_rally_", 1)[0]


def _confidence_from_reprojection(median: float, p95: float) -> str:
    if median <= 2.0 and p95 <= 5.0:
        return "high"
    if median <= 6.0 and p95 <= 15.0:
        return "med"
    return "low"


def _camera_center_from_pose(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> list[float]:
    rotated = [sum(float(rotation[k][i]) * float(translation[k]) for k in range(3)) for i in range(3)]
    return [-value for value in rotated]


def _max_translation_spread(translations: Sequence[Sequence[float]]) -> float:
    best = 0.0
    for idx, a in enumerate(translations):
        for b in translations[idx + 1 :]:
            best = max(best, math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True))))
    return best


def _max_rotation_spread_deg(rotations: Sequence[Sequence[Sequence[float]]]) -> float:
    best = 0.0
    for idx, a in enumerate(rotations):
        for b in rotations[idx + 1 :]:
            best = max(best, _rotation_delta_deg(a, b))
    return best


def _rotation_delta_deg(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    trace = 0.0
    for i in range(3):
        for k in range(3):
            trace += float(a[i][k]) * float(b[i][k])
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _nested_get(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrected-gt-root",
        type=Path,
        default=None,
        help="Use verified per-source corrected-r2 court_keypoints.json labels instead of the legacy CVAT XML.",
    )
    parser.add_argument("--cvat-export-xml", type=Path, default=DEFAULT_CVAT_EXPORT_XML)
    parser.add_argument("--taskset-manifest", type=Path, default=DEFAULT_TASKSET_MANIFEST)
    parser.add_argument("--package-manifest", type=Path, default=DEFAULT_PACKAGE_MANIFEST)
    parser.add_argument("--prelabels-dir", type=Path, default=DEFAULT_PRELABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.corrected_gt_root is not None and not args.corrected_gt_root.is_dir():
        parser.exit(2, f"{parser.prog}: error: --corrected-gt-root not found: {args.corrected_gt_root}\n")
    if args.corrected_gt_root is None and not args.cvat_export_xml.is_file():
        parser.exit(2, f"{parser.prog}: error: --cvat-export-xml not found: {args.cvat_export_xml}\n")
    if args.corrected_gt_root is None and not args.taskset_manifest.is_file():
        parser.exit(2, f"{parser.prog}: error: --taskset-manifest not found: {args.taskset_manifest}\n")
    if args.corrected_gt_root is None and not args.package_manifest.is_file():
        parser.exit(2, f"{parser.prog}: error: --package-manifest not found: {args.package_manifest}\n")
    if not args.prelabels_dir.is_dir():
        parser.exit(2, f"{parser.prog}: error: --prelabels-dir not found: {args.prelabels_dir}\n")
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps({"coverage_report": summary["coverage_report"], "coverage_summary": summary["coverage_summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
