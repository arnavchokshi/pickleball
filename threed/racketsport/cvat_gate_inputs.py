"""Build local gate-label inputs from reviewed CVAT detector boxes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eval_guard import INTERNAL_VAL_ONLY_CLIP_IDS, STRICT_HOLDOUT_CLIP_IDS
from .schemas import BALL_VISIBILITY_WBCE_WEIGHTS, CvatVideoAnnotations, CvatVideoBox, validate_artifact_file
from .testclips import REQUIRED_LABEL_FILES


DATASET_TARGETS: dict[str, dict[str, Any]] = {
    "player": {
        "labels": ("player",),
        "target_file": "players.json",
        "trusted_for": ("player_bbox_label_check", "detector_training"),
    },
    "paddle": {
        "labels": ("paddle",),
        "target_file": "paddle_boxes.json",
        "trusted_for": ("paddle_detector_training", "racket_candidate_seed"),
        "limitations": (
            "paddle boxes are detector labels only; they are not true paddle corners or 6DoF racket_pose labels",
        ),
    },
    "ball": {
        "labels": ("ball",),
        "target_file": "ball.json",
        "trusted_for": ("ball_visibility_label_check", "detector_training"),
    },
    "combined": {
        "labels": ("player", "paddle", "ball"),
        "target_file": "combined_detector_labels.json",
        "trusted_for": ("combined_detector_training",),
    },
}


@dataclass(frozen=True)
class CvatGateClipSpec:
    clip_id: str
    reviewed_boxes_path: Path


@dataclass(frozen=True)
class Data1CvatClipSpec:
    clip_id: str
    source_video_path: Path
    cvat_export_path: Path
    reviewed_boxes_path: Path
    metadata: Mapping[str, Any]
    notes: tuple[str, ...] = ()


def build_cvat_gate_input_payloads(
    annotations: CvatVideoAnnotations,
    *,
    reviewed_boxes_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Return gate-consumable label payloads for player, paddle, ball, and combined boxes."""

    reviewed_path = Path(reviewed_boxes_path)
    payloads: dict[str, dict[str, Any]] = {}
    for dataset_name, config in DATASET_TARGETS.items():
        labels = tuple(config["labels"])
        items = [
            _item_from_box(box, clip_id=annotations.clip_id)
            for frame in annotations.frames
            for box in frame.boxes
            if box.label in labels
        ]
        label_counts = Counter(str(item["label"]) for item in items)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "human_reviewed",
            "not_ground_truth": False,
            "clip": {
                "id": annotations.clip_id,
                "frame_count": len(annotations.frames),
                "task_stop_frame": annotations.task.stop_frame,
                "source_path": annotations.source_path,
                "reviewed_boxes_path": str(reviewed_path),
            },
            "source": {
                "mode": "cvat_video_1_1",
                "reviewed_boxes_path": str(reviewed_path),
                "source_path": annotations.source_path,
                "labels": list(labels),
            },
            "trusted_for": list(config["trusted_for"]),
            "annotation": {
                "target_file": config["target_file"],
                "items": items,
            },
            "summary": {
                "item_count": len(items),
                "label_counts_by_name": {label: label_counts[label] for label in sorted(label_counts)},
                "frame_count": len(annotations.frames),
                "first_labeled_frame": min((int(item["frame_index"]) for item in items), default=None),
                "last_labeled_frame": max((int(item["frame_index"]) for item in items), default=None),
            },
        }
        if config.get("limitations"):
            payload["limitations"] = list(config["limitations"])
        payloads[dataset_name] = payload
    return payloads


def write_cvat_gate_input_package(
    *,
    clips: Sequence[CvatGateClipSpec],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write a local gate-input package derived from reviewed CVAT box artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_summaries = {
        dataset_name: {
            "target_file": config["target_file"],
            "trusted_for": list(config["trusted_for"]),
            "item_count": 0,
            "label_counts_by_name": {},
            "clips": {},
        }
        for dataset_name, config in DATASET_TARGETS.items()
    }
    for dataset_name, config in DATASET_TARGETS.items():
        if config.get("limitations"):
            dataset_summaries[dataset_name]["limitations"] = list(config["limitations"])

    clip_summaries: list[dict[str, Any]] = []
    for clip in clips:
        annotations = validate_artifact_file("cvat_video_annotations", clip.reviewed_boxes_path)
        if not isinstance(annotations, CvatVideoAnnotations):
            raise ValueError(f"reviewed boxes did not parse as CVAT video annotations: {clip.reviewed_boxes_path}")
        if annotations.clip_id != clip.clip_id:
            raise ValueError(
                f"clip id mismatch for {clip.reviewed_boxes_path}: expected {clip.clip_id}, found {annotations.clip_id}"
            )
        payloads = build_cvat_gate_input_payloads(annotations, reviewed_boxes_path=clip.reviewed_boxes_path)
        labels_dir = out / clip.clip_id / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        clip_datasets: dict[str, dict[str, Any]] = {}
        for dataset_name, payload in payloads.items():
            target_file = str(payload["annotation"]["target_file"])
            output_path = labels_dir / target_file
            _write_json(output_path, payload)
            summary = payload["summary"]
            dataset_summary = dataset_summaries[dataset_name]
            dataset_summary["item_count"] += int(summary["item_count"])
            _merge_counts(dataset_summary["label_counts_by_name"], summary["label_counts_by_name"])
            clip_entry = {
                "path": str(output_path),
                "item_count": int(summary["item_count"]),
                "label_counts_by_name": dict(summary["label_counts_by_name"]),
                "target_file": target_file,
            }
            dataset_summary["clips"][clip.clip_id] = clip_entry
            clip_datasets[dataset_name] = clip_entry
        clip_summaries.append(
            {
                "clip_id": clip.clip_id,
                "reviewed_boxes_path": str(clip.reviewed_boxes_path),
                "frame_count": len(annotations.frames),
                "datasets": clip_datasets,
            }
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_gate_input_manifest",
        "out_dir": str(out),
        "clip_count": len(clips),
        "clips": clip_summaries,
        "datasets": dataset_summaries,
        "warnings": [
            "This package contains detector-label gate inputs only.",
            "paddle_boxes.json is not canonical racket_pose.json and cannot satisfy 6DoF racket gates.",
            "Missing non-detector DATA-1 labels still block full DATA-1 readiness.",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def canonical_data1_cvat_clip_specs(
    *,
    cvat_upload_root: str | Path = Path("cvat_upload"),
    imports_root: str | Path = Path("runs/cvat_imports/2026_06_30"),
) -> list[Data1CvatClipSpec]:
    """Return the current canonical local CVAT clip plan without touching data/testclips."""

    cvat_root = Path(cvat_upload_root)
    imports = Path(imports_root)
    rows = [
        (
            "burlington_gold_0300_low_steep_corner",
            "01_burlington_gold_0300_low_steep_corner_10s.mp4",
            "01_burlington_gold_0300_low_steep_corner_cvat_for_video_1.1.zip",
            {
                "camera_height": "low",
                "camera_angle": "steep_corner",
                "play_type": "doubles",
                "environment": "outdoor",
                "frame_rate_fps": 60,
                "duration_s": 10.01,
                "racket_gt": False,
            },
            (),
        ),
        (
            "wolverine_mixed_0200_mid_steep_corner",
            "02_wolverine_mixed_0200_mid_steep_corner_10s.mp4",
            "02_wolverine_mixed_0200_mid_steep_corner_cvat_for_video_1.1.zip",
            {
                "camera_height": "mid",
                "camera_angle": "steep_corner",
                "play_type": "doubles",
                "environment": "outdoor",
                "frame_rate_fps": 30,
                "duration_s": 10.0,
                "racket_gt": False,
            },
            (),
        ),
        (
            "outdoor_webcam_iynbd_1500_long_high_baseline",
            "03_outdoor_webcam_iynbd_1500_long_high_baseline_frames_0000_1150.mp4",
            "03_outdoor_webcam_iynbd_1500_long_high_baseline_cvat_for_video_1.1.zip",
            {
                "camera_height": "high",
                "camera_angle": "shallow_baseline",
                "play_type": "doubles",
                "environment": "outdoor",
                "frame_rate_fps": 60,
                "duration_s": 19.183333,
                "racket_gt": False,
            },
            (
                "Outdoor import is capped to source frames 0..1150 and remains a strict held-out eval clip.",
            ),
        ),
        (
            "indoor_doubles_fwuks_0500_long_mid_baseline",
            "04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
            "04_indoor_doubles_fwuks_0500_long_mid_baseline_cvat_for_video_1.1.zip",
            {
                "camera_height": "mid",
                "camera_angle": "shallow_baseline",
                "play_type": "doubles",
                "environment": "indoor",
                "frame_rate_fps": 30,
                "duration_s": 30.03,
                "racket_gt": False,
            },
            (
                "Indoor export and reviewed-box import exist locally; Indoor remains a strict held-out eval clip.",
            ),
        ),
    ]
    return [
        Data1CvatClipSpec(
            clip_id=clip_id,
            source_video_path=cvat_root / video_name,
            cvat_export_path=cvat_root / "exports" / export_name,
            reviewed_boxes_path=imports / clip_id / "reviewed_boxes.json",
            metadata=metadata,
            notes=notes,
        )
        for clip_id, video_name, export_name, metadata, notes in rows
    ]


def write_data1_substitute_package(
    *,
    clips: Sequence[Data1CvatClipSpec],
    out_dir: str | Path,
    data_testclips_root: str | Path = Path("data/testclips"),
    detector_gate_inputs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a run-scoped DATA-1 bootstrap plan without promoting detector labels."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_testclips_root)
    detector_root = Path(detector_gate_inputs_root) if detector_gate_inputs_root is not None else None
    skeleton_root = out / "label_skeletons"
    registration_path = out / "canonical_testclips_registration_manifest.json"
    missing_inputs_path = out / "missing_inputs.json"
    coverage_path = out / "coverage_report.json"
    sanity_path = out / "sanity_checks.json"
    markdown_path = out / "DATA1_SUBSTITUTE_report.md"

    registration_rows: list[dict[str, Any]] = []
    clip_reports: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, Any]] = []
    skeleton_count = 0

    if not data_root.exists():
        missing_inputs.append(
            {
                "kind": "data_testclips_root",
                "path": str(data_root),
                "reason": "DATA-1 root is absent; this package only writes a registration plan.",
            }
        )

    for clip in clips:
        registration_row = _registration_row(clip)
        registration_rows.append(registration_row)
        labels_dir = skeleton_root / clip.clip_id / "labels"
        actual_data_labels_dir = data_root / clip.clip_id / "labels"
        detector_labels_dir = detector_root / clip.clip_id / "labels" if detector_root is not None else None
        label_skeletons: dict[str, str] = {}
        data1_present_labels: list[str] = []
        data1_missing_labels: list[str] = []
        detector_gate_inputs: dict[str, dict[str, Any]] = {}

        if not clip.source_video_path.is_file():
            missing_inputs.append(
                {
                    "kind": "source_video",
                    "clip_id": clip.clip_id,
                    "path": str(clip.source_video_path),
                    "reason": "source video required for DATA-1 testclip registration is missing",
                }
            )
        if not clip.cvat_export_path.is_file():
            missing_inputs.append(
                {
                    "kind": "cvat_video_export",
                    "clip_id": clip.clip_id,
                    "path": str(clip.cvat_export_path),
                    "reason": "CVAT for video 1.1 export required before detector-label import is missing",
                }
            )
        if not clip.reviewed_boxes_path.is_file():
            missing_inputs.append(
                {
                    "kind": "cvat_reviewed_boxes_import",
                    "clip_id": clip.clip_id,
                    "path": str(clip.reviewed_boxes_path),
                    "reason": "reviewed_boxes.json has not been imported from a CVAT video export",
                }
            )
        if not (data_root / clip.clip_id).is_dir():
            missing_inputs.append(
                {
                    "kind": "data1_clip_registration",
                    "clip_id": clip.clip_id,
                    "path": str(data_root / clip.clip_id),
                    "reason": "canonical clip is not registered under data/testclips",
                }
            )

        for label_file in REQUIRED_LABEL_FILES:
            actual_label = actual_data_labels_dir / label_file
            if actual_label.is_file():
                data1_present_labels.append(label_file)
            else:
                data1_missing_labels.append(label_file)
                missing_inputs.append(
                    {
                        "kind": "data1_label_file",
                        "clip_id": clip.clip_id,
                        "path": str(actual_label),
                        "required_label_file": label_file,
                        "reason": "required DATA-1 label is absent; generated skeleton is not ground truth",
                    }
                )
            detector_substitute = _detector_label_substitute(label_file, detector_labels_dir)
            detector_gate_inputs[label_file] = detector_substitute
            skeleton_path = labels_dir / label_file
            skeleton = _label_skeleton(
                clip=clip,
                label_file=label_file,
                expected_data1_path=actual_label,
                detector_substitute=detector_substitute,
            )
            _write_json(skeleton_path, skeleton)
            label_skeletons[label_file] = str(skeleton_path)
            skeleton_count += 1

        clip_reports.append(
            {
                "clip_id": clip.clip_id,
                "source_video_path": str(clip.source_video_path),
                "source_video_exists": clip.source_video_path.is_file(),
                "cvat_export_path": str(clip.cvat_export_path),
                "cvat_export_exists": clip.cvat_export_path.is_file(),
                "reviewed_boxes_path": str(clip.reviewed_boxes_path),
                "reviewed_boxes_exists": clip.reviewed_boxes_path.is_file(),
                "data1_clip_dir": str(data_root / clip.clip_id),
                "data1_clip_exists": (data_root / clip.clip_id).is_dir(),
                "data1_present_label_files": data1_present_labels,
                "data1_missing_label_files": data1_missing_labels,
                "detector_gate_inputs": detector_gate_inputs,
                "label_skeletons": label_skeletons,
                "metadata": _metadata_for_registration(clip.metadata),
                "eval_policy": _eval_policy_for_clip(clip.clip_id),
                "registration_row": registration_row,
                "notes": list(clip.notes),
            }
        )

    registration_manifest = {
        "clips": registration_rows,
    }
    missing_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_missing_inputs",
        "data_testclips_root": str(data_root),
        "missing_input_count": len(missing_inputs),
        "missing_inputs": missing_inputs,
    }
    source_video_count = sum(1 for clip in clip_reports if clip["source_video_exists"])
    cvat_export_count = sum(1 for clip in clip_reports if clip["cvat_export_exists"])
    reviewed_boxes_count = sum(1 for clip in clip_reports if clip["reviewed_boxes_exists"])
    coverage = {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_cvat_substitute_coverage_report",
        "data_testclips_root": str(data_root),
        "data_testclips_root_exists": data_root.exists(),
        "data1_ready": len(missing_inputs) == 0,
        "canonical_clip_count": len(clips),
        "source_video_count": source_video_count,
        "cvat_export_count": cvat_export_count,
        "reviewed_boxes_count": reviewed_boxes_count,
        "label_skeleton_count": skeleton_count,
        "detector_package_separate_from_data1": True,
        "clips": clip_reports,
        "missing_input_count": len(missing_inputs),
    }
    sanity = _data1_substitute_sanity(
        clip_reports=clip_reports,
        required_skeleton_count=len(clips) * len(REQUIRED_LABEL_FILES),
        actual_skeleton_count=skeleton_count,
    )
    status = "ready_to_register" if not missing_inputs else "blocked_missing_data1_inputs"
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_cvat_substitute_manifest",
        "status": status,
        "data1_ready": len(missing_inputs) == 0,
        "out_dir": str(out),
        "data_testclips_root": str(data_root),
        "data_testclips_root_exists": data_root.exists(),
        "canonical_clip_count": len(clips),
        "clips": clip_reports,
        "summary": {
            "source_video_count": source_video_count,
            "cvat_export_count": cvat_export_count,
            "reviewed_boxes_count": reviewed_boxes_count,
            "label_skeleton_count": skeleton_count,
            "missing_input_count": len(missing_inputs),
        },
        "registration_manifest": str(registration_path),
        "registration_command": (
            "python scripts/racketsport/register_testclips_manifest.py "
            f"--manifest {registration_path} --root {data_root} --symlink --continue-on-error"
        ),
        "missing_inputs_report": str(missing_inputs_path),
        "coverage_report": str(coverage_path),
        "sanity_checks": str(sanity_path),
        "markdown_report": str(markdown_path),
        "label_skeleton_root": str(skeleton_root),
        "detector_package": {
            "separate_from_data1": True,
            "detector_gate_inputs_root": str(detector_root) if detector_root is not None else None,
            "note": "CVAT detector gate inputs are referenced only as detector labels, not DATA-1 ground truth.",
        },
        "warnings": [
            "No DATA-1 promotion is claimed.",
            "Generated label skeletons are placeholders and are marked not_ground_truth=true.",
            "Any CVAT detector-label package remains separate from DATA-1 readiness.",
            "Outdoor and Indoor strict held-out eval clips are not training or validation-during-fitting inputs.",
        ],
    }

    _write_json(registration_path, registration_manifest)
    _write_json(missing_inputs_path, missing_payload)
    _write_json(coverage_path, coverage)
    _write_json(sanity_path, sanity)
    _write_json(out / "manifest.json", manifest)
    markdown_path.write_text(_render_data1_substitute_markdown(manifest, coverage, missing_payload, sanity), encoding="utf-8")
    return manifest


def _item_from_box(box: CvatVideoBox, *, clip_id: str) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in box.bbox_xyxy]
    x, y, width, height = [float(value) for value in box.bbox_xywh]
    item: dict[str, Any] = {
        "id": f"{box.label}_{box.track_id + 1}_{box.frame_index:06d}",
        "clip_id": clip_id,
        "frame": f"frame_{box.frame_index:06d}.jpg",
        "frame_index": int(box.frame_index),
        "label": box.label,
        "class_name": box.label,
        "track_id": int(box.track_id) + 1,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox": [x, y, width, height],
        "bbox_xywh": [x, y, width, height],
        "status": "accepted",
        "source": "cvat_video_1_1",
        "confidence": 1.0,
        "occluded": bool(box.occluded),
        "keyframe": bool(box.keyframe),
    }
    if box.label == "ball":
        item["xy_px"] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
        if box.visibility_level in {"full", "out_of_frame"}:
            item["visible"] = False
        else:
            item["visible"] = True
        item["visibility"] = box.visibility_level or "visible"
        if box.visibility_level is not None:
            item["visibility_level"] = box.visibility_level
            item["wbce_weight"] = BALL_VISIBILITY_WBCE_WEIGHTS[box.visibility_level]
    return item


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for label, count in source.items():
        target[label] = target.get(label, 0) + int(count)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata_for_registration(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "camera_height": metadata["camera_height"],
        "camera_angle": metadata["camera_angle"],
        "play_type": metadata["play_type"],
        "environment": metadata["environment"],
        "frame_rate_fps": int(metadata["frame_rate_fps"]),
        "duration_s": float(metadata["duration_s"]),
        "racket_gt": bool(metadata.get("racket_gt", False)),
    }


def _registration_row(clip: Data1CvatClipSpec) -> dict[str, Any]:
    return {
        "source": str(clip.source_video_path.resolve()),
        "name": clip.clip_id,
        **_metadata_for_registration(clip.metadata),
        "symlink": True,
    }


def _eval_policy_for_clip(clip_id: str) -> dict[str, Any]:
    if clip_id in STRICT_HOLDOUT_CLIP_IDS:
        return {
            "role": "strict_holdout",
            "training_allowed": False,
            "validation_during_fitting_allowed": False,
            "reason": "protected eval clip; no override exists in eval_guard.py",
        }
    if clip_id in INTERNAL_VAL_ONLY_CLIP_IDS:
        return {
            "role": "internal_val_only",
            "training_allowed": False,
            "validation_during_fitting_allowed": True,
            "reason": "internal validation only when explicitly allowed; never actual training data",
        }
    return {
        "role": "unprotected",
        "training_allowed": True,
        "validation_during_fitting_allowed": True,
        "reason": "not listed in eval_guard.py protected clip registry",
    }


def _detector_label_substitute(label_file: str, detector_labels_dir: Path | None) -> dict[str, Any]:
    detector_map = {
        "players.json": "players.json",
        "ball.json": "ball.json",
    }
    if detector_labels_dir is None:
        return {
            "present": False,
            "path": None,
            "data1_substitute": False,
            "reason": "no detector gate-input root was provided",
        }
    detector_file = detector_map.get(label_file)
    if detector_file is not None:
        path = detector_labels_dir / detector_file
        return {
            "present": path.is_file(),
            "path": str(path),
            "data1_substitute": False,
            "trusted_for": "detector_label_check_only",
        }
    if label_file == "racket_pose.json":
        paddle_path = detector_labels_dir / "paddle_boxes.json"
        return {
            "present": False,
            "path": None,
            "data1_substitute": False,
            "nearby_detector_box_path": str(paddle_path),
            "nearby_detector_box_present": paddle_path.is_file(),
            "limitation": "paddle detector boxes are not 6DoF racket_pose labels",
        }
    return {
        "present": False,
        "path": None,
        "data1_substitute": False,
        "reason": "no CVAT detector label maps to this DATA-1 label file",
    }


def _label_skeleton(
    *,
    clip: Data1CvatClipSpec,
    label_file: str,
    expected_data1_path: Path,
    detector_substitute: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_label_skeleton",
        "clip_id": clip.clip_id,
        "required_label_file": label_file,
        "status": "missing_human_review",
        "not_ground_truth": True,
        "items": [],
        "expected_data1_path": str(expected_data1_path),
        "source": {
            "mode": "data1_missing_input_skeleton",
            "source_video_path": str(clip.source_video_path),
            "cvat_export_path": str(clip.cvat_export_path),
            "reviewed_boxes_path": str(clip.reviewed_boxes_path),
        },
        "detector_label_substitute": dict(detector_substitute),
        "warnings": [
            "This file is a planning skeleton only.",
            "Do not copy this skeleton into data/testclips as reviewed DATA-1 ground truth.",
        ],
    }


def _data1_substitute_sanity(
    *,
    clip_reports: Sequence[Mapping[str, Any]],
    required_skeleton_count: int,
    actual_skeleton_count: int,
) -> dict[str, Any]:
    skeleton_paths = [
        path
        for clip in clip_reports
        for path in dict(clip["label_skeletons"]).values()
    ]
    strict_holdout_reports = [
        clip
        for clip in clip_reports
        if dict(clip.get("eval_policy", {})).get("role") == "strict_holdout"
    ]
    checks = {
        "detector_package_not_promoted": True,
        "skeletons_marked_not_ground_truth": True,
        "required_label_skeletons_written": actual_skeleton_count == required_skeleton_count,
        "registration_manifest_has_all_canonical_clips": bool(clip_reports),
        "indoor_cvat_export_status_recorded": any(
            clip["metadata"]["environment"] == "indoor" and isinstance(clip.get("cvat_export_exists"), bool)
            for clip in clip_reports
        ),
        "strict_holdouts_not_promoted": all(
            not any(
                dict(substitute).get("data1_substitute") is True
                for substitute in dict(clip["detector_gate_inputs"]).values()
            )
            for clip in strict_holdout_reports
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_cvat_substitute_sanity_checks",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "required_skeleton_count": required_skeleton_count,
        "actual_skeleton_count": actual_skeleton_count,
        "sample_skeleton_paths": skeleton_paths[:10],
    }


def _render_data1_substitute_markdown(
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    missing_payload: Mapping[str, Any],
    sanity: Mapping[str, Any],
) -> str:
    lines = [
        "# DATA-1 CVAT Substitute Bootstrap Report",
        "",
        "Source of truth: local `cvat_upload/`, imported CVAT artifacts under `runs/cvat_imports/2026_06_30/`, and the current `data/testclips` filesystem state.",
        "",
        f"Status: `{manifest['status']}`. No DATA-1 promotion is claimed.",
        "",
        "Any CVAT detector-label package remains separate from DATA-1. Generated label skeletons are planning placeholders only and are marked `not_ground_truth=true`.",
        "",
        "Outdoor and Indoor remain strict held-out eval clips; current YOLO/TrackNet training exporters fail closed if either appears in training or validation-during-fitting inputs.",
        "",
        "## Canonical Clip Coverage",
        "",
        "| clip | eval role | video | CVAT export | reviewed boxes | DATA-1 labels missing |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for clip in coverage["clips"]:
        lines.append(
            "| "
            f"`{clip['clip_id']}` | "
            f"{clip['eval_policy']['role']} | "
            f"{_markdown_bool(bool(clip['source_video_exists']))} | "
            f"{_markdown_bool(bool(clip['cvat_export_exists']))} | "
            f"{_markdown_bool(bool(clip['reviewed_boxes_exists']))} | "
            f"{len(clip['data1_missing_label_files'])} |"
        )
    lines.extend(
        [
            "",
            "## Missing Inputs",
            "",
            f"- Total missing inputs: {missing_payload['missing_input_count']}",
            f"- `data/testclips` root exists: {_markdown_bool(bool(coverage['data_testclips_root_exists']))}",
            f"- Source videos present: {coverage['source_video_count']}/{coverage['canonical_clip_count']}",
            f"- CVAT exports present: {coverage['cvat_export_count']}/{coverage['canonical_clip_count']}",
            f"- Reviewed-box imports present: {coverage['reviewed_boxes_count']}/{coverage['canonical_clip_count']}",
            "",
            "## Generated Artifacts",
            "",
            f"- Registration manifest: `{manifest['registration_manifest']}`",
            f"- Label skeleton root: `{manifest['label_skeleton_root']}`",
            f"- Missing-input JSON: `{manifest['missing_inputs_report']}`",
            f"- Coverage JSON: `{manifest['coverage_report']}`",
            f"- Sanity JSON: `{manifest['sanity_checks']}`",
            "",
            "## Sanity Checks",
            "",
        ]
    )
    for name, passed in sanity["checks"].items():
        lines.append(f"- {name}: {_markdown_bool(bool(passed))}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Register canonical clips into `data/testclips` only after reviewed DATA-1 labels exist. Keep Outdoor and Indoor excluded from training and validation-during-fitting builds, and replace skeletons with reviewed DATA-1 labels before claiming DATA-1 readiness.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "CvatGateClipSpec",
    "DATASET_TARGETS",
    "Data1CvatClipSpec",
    "build_cvat_gate_input_payloads",
    "canonical_data1_cvat_clip_specs",
    "write_data1_substitute_package",
    "write_cvat_gate_input_package",
]
