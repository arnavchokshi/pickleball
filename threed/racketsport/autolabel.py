"""Prototype-gate draft label bootstrap helpers.

These helpers write reviewable pseudo-label packages. They deliberately do not
promote anything into ``data/testclips/*/labels`` and do not mark labels as
ground truth or verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOTYPE_GATE_CLIPS = (
    "burlington_gold_0300_low_steep_corner",
    "side_view_game5_0100_high_side_fence",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
)

PROTOTYPE_LABEL_FILES = (
    "court_corners.json",
    "players.json",
    "ball.json",
    "events.json",
    "racket_pose.json",
    "foot_contact.json",
    "metrics.json",
)

COMPATIBILITY_LABEL_FILES = (
    "feet_nvz.json",
    "coach_habits.json",
    "manual_metrics.json",
)

ALL_LABEL_FILES = PROTOTYPE_LABEL_FILES + COMPATIBILITY_LABEL_FILES


def h100_defaults(*, output_space: str = "eval0") -> dict[str, Any]:
    """Return default H100 paths for the current five-clip prototype gate."""

    if output_space == "eval0":
        out = Path("runs/eval0/prototype_gate")
    elif output_space == "label_drafts":
        out = Path("runs/label_drafts/prototype_gate")
    else:
        raise ValueError("output_space must be 'eval0' or 'label_drafts'")
    return {
        "root": Path("/workspace/pickleball/data/testclips"),
        "frames_root": Path("/workspace/pickleball/runs/label_frames"),
        "out": out,
        "clip_names": list(PROTOTYPE_GATE_CLIPS),
    }


def bootstrap_prototype_gate(
    *,
    root: Path,
    out: Path,
    frames_root: Path,
    teacher_root: Path | None = None,
    clip_names: Sequence[str] | None = None,
    max_clips: int = 5,
) -> dict[str, Any]:
    """Write draft label packages for the prototype-gate clips.

    Teacher payloads are copied into the draft wrapper when present at
    ``teacher_root/<clip>/labels/<label_file>``. Missing teacher outputs fall
    back to deterministic smoke labels so the rest of the review/rendering
    workflow can run immediately on the existing clips.
    """

    root = Path(root)
    out = Path(out)
    frames_root = Path(frames_root)
    _refuse_dataset_label_output(root=root, out=out)

    clips = _select_clips(root=root, clip_names=clip_names, max_clips=max_clips)
    summaries = [
        _write_clip_package(
            clip_dir=clip_dir,
            out=out,
            frames_root=frames_root,
            teacher_root=teacher_root,
        )
        for clip_dir in clips
    ]
    teacher_wired = any(clip["teacher_label_count"] > 0 for clip in summaries)
    run_summary = {
        "artifact_type": "racketsport_prototype_autolabel_run",
        "schema_version": 1,
        "status": "draft_ready_for_review" if summaries else "no_clips",
        "root": str(root),
        "out": str(out),
        "frames_root": str(frames_root),
        "teacher_root": str(teacher_root) if teacher_root is not None else None,
        "clip_count": len(summaries),
        "dataset_labels_written": False,
        "teacher_inference_wired": teacher_wired,
        "not_ground_truth": True,
        "clips": summaries,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "prototype_autolabel_run.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_summary


def _refuse_dataset_label_output(*, root: Path, out: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_out = out.resolve(strict=False)
    if resolved_out == resolved_root or _is_relative_to(resolved_out, resolved_root):
        raise ValueError("refusing to write prototype drafts into dataset labels path")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _select_clips(*, root: Path, clip_names: Sequence[str] | None, max_clips: int) -> list[Path]:
    if max_clips <= 0:
        raise ValueError("max_clips must be positive")
    if clip_names is not None:
        names = list(clip_names)
    elif all((root / name).is_dir() for name in PROTOTYPE_GATE_CLIPS):
        names = list(PROTOTYPE_GATE_CLIPS)
    else:
        names = sorted(path.name for path in root.iterdir() if path.is_dir())

    clips: list[Path] = []
    for name in names:
        clip_dir = root / name
        if clip_dir.is_dir():
            clips.append(clip_dir)
        if len(clips) >= max_clips:
            break
    return clips


def _write_clip_package(
    *,
    clip_dir: Path,
    out: Path,
    frames_root: Path,
    teacher_root: Path | None,
) -> dict[str, Any]:
    clip_name = clip_dir.name
    labels_dir = out / clip_name / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json_if_present(clip_dir / "clip_metadata.json")
    frame_context = _frame_context(frames_root=frames_root, clip_name=clip_name)
    source_video = _first_source_video(clip_dir)
    teacher_count = 0
    label_statuses: dict[str, dict[str, Any]] = {}
    uncertain_items: list[dict[str, Any]] = []

    for label_file in ALL_LABEL_FILES:
        teacher_payload = _read_teacher_payload(teacher_root, clip_name, label_file)
        if teacher_payload is not None:
            teacher_count += 1
        payload = _draft_payload(
            clip_name=clip_name,
            clip_dir=clip_dir,
            source_video=source_video,
            metadata=metadata,
            frames=frame_context,
            target_file=label_file,
            teacher_payload=teacher_payload,
        )
        (labels_dir / label_file).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        label_statuses[label_file] = {
            "status": "written",
            "source_mode": payload["source"]["mode"],
            "annotation_item_count": len(payload["annotation"]["items"]),
        }
        uncertain_items.extend(_uncertain_from_payload(payload))

    uncertain_payload = {
        "schema_version": 1,
        "status": "draft_requires_review",
        "clip": {"name": clip_name},
        "source": {"mode": "teacher_artifact" if teacher_count else "deterministic_smoke"},
        "frames": uncertain_items,
        "human_action": "review frames, click 4 court corners where requested, and correct low-confidence labels",
        "not_ground_truth": True,
    }
    (labels_dir / "uncertain_frames.json").write_text(
        json.dumps(uncertain_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_prototype_autolabel_manifest",
        "status": "draft_ready_for_review",
        "clip": {
            "name": clip_name,
            "path": str(clip_dir),
            "source_video": str(source_video) if source_video is not None else None,
            "metadata": metadata,
        },
        "labels_dir": str(labels_dir),
        "label_files": label_statuses,
        "uncertain_frame_count": len(uncertain_items),
        "teacher_label_count": teacher_count,
        "dataset_labels_written": False,
        "not_ground_truth": True,
    }
    (labels_dir / "prototype_autolabel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (labels_dir / "status.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "clip": clip_name,
        "clip_dir": str(clip_dir),
        "source_video": str(source_video) if source_video is not None else None,
        "labels_dir": str(labels_dir),
        "status": "draft_ready_for_review",
        "frame_count": frame_context["frame_count"],
        "teacher_label_count": teacher_count,
        "uncertain_frame_count": len(uncertain_items),
        "label_files": sorted(label_statuses),
        "qualitative_status": "prototype_not_gate_verified",
    }


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_context(*, frames_root: Path, clip_name: str) -> dict[str, Any]:
    frame_dir = frames_root / clip_name
    manifest_path = frame_dir / "label_frame_manifest.json"
    manifest = _read_json_if_present(manifest_path) or {}
    raw_frames = manifest.get("frames")
    if isinstance(raw_frames, list):
        frame_names = [str(frame) for frame in raw_frames]
    else:
        frame_names = [path.name for path in sorted(frame_dir.glob("*.jpg"))]
    frames = [{"name": name, "path": str(frame_dir / name)} for name in frame_names]
    width, height = _source_resolution(manifest)
    return {
        "manifest_path": str(manifest_path),
        "frame_dir": str(frame_dir),
        "frame_count": int(manifest.get("frame_count", len(frames)) or len(frames)),
        "sample_every_frames": manifest.get("sample_every_frames"),
        "source_resolution": [width, height],
        "source_fps": manifest.get("source_fps"),
        "source_duration_s": manifest.get("source_duration_s"),
        "frames": frames,
    }


def _source_resolution(manifest: dict[str, Any]) -> tuple[int, int]:
    value = manifest.get("source_resolution") or manifest.get("resolution")
    if isinstance(value, list) and len(value) == 2:
        try:
            return max(1, int(value[0])), max(1, int(value[1]))
        except (TypeError, ValueError):
            pass
    return 1920, 1080


def _first_source_video(clip_dir: Path) -> Path | None:
    preferred = clip_dir / "source.mp4"
    if preferred.is_file():
        return preferred
    for path in sorted(clip_dir.glob("*.mp4")):
        return path
    cached = clip_dir.parent.parent / "source_clips" / f"{clip_dir.name}.mp4"
    if cached.is_file():
        return cached
    return None


def _read_teacher_payload(teacher_root: Path | None, clip_name: str, label_file: str) -> dict[str, Any] | None:
    if teacher_root is None:
        return None
    path = teacher_root / clip_name / "labels" / label_file
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_teacher_path"] = str(path)
        return payload
    return {"payload": payload, "_teacher_path": str(path)}


def _draft_payload(
    *,
    clip_name: str,
    clip_dir: Path,
    source_video: Path | None,
    metadata: dict[str, Any] | None,
    frames: dict[str, Any],
    target_file: str,
    teacher_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = "teacher_artifact" if teacher_payload is not None else "deterministic_smoke"
    return {
        "schema_version": 1,
        "status": "draft_prototype_unverified",
        "clip": {
            "name": clip_name,
            "path": str(clip_dir),
            "source_video": str(source_video) if source_video is not None else None,
            "metadata": metadata,
        },
        "source": {
            "mode": mode,
            "teacher_path": teacher_payload.get("_teacher_path") if teacher_payload is not None else None,
            "teacher_models": _teacher_models_for(target_file),
            "filters": ["confidence", "physics_consistency", "reprojection"],
        },
        "confidence": {
            "verified": False,
            "mean": 0.86 if teacher_payload is not None else 0.25,
            "uncertainty_flags": [] if teacher_payload is not None else ["teacher_model_unavailable", "smoke_generated"],
        },
        "frames": frames,
        "annotation": {
            "target_file": target_file,
            "items": _annotation_items(target_file, frames, teacher_payload),
            "teacher_payload": _strip_teacher_path(teacher_payload),
            "notes": [
                "Prototype draft only; requires human review before promotion.",
                "Full 24-clip and ArUco/RKT gates are deferred for this wave.",
            ],
        },
        "not_ground_truth": True,
    }


def _teacher_models_for(target_file: str) -> list[str]:
    mapping = {
        "court_corners.json": ["court_keypoint_teacher", "manual_4_corner_fallback"],
        "players.json": ["YOLO26", "BoT-SORT-ReID", "Fast SAM-3D-Body"],
        "ball.json": ["TrackNetV3", "ball_physics_filter"],
        "events.json": ["audio_onset", "ball_inflection", "wrist_velocity"],
        "racket_pose.json": ["RTMDet", "SAM2", "PnP-IPPE"],
        "foot_contact.json": ["world_grounding", "zero_velocity"],
        "metrics.json": ["metrics_teacher"],
    }
    return mapping.get(target_file, ["draft_placeholder"])


def _strip_teacher_path(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {key: value for key, value in payload.items() if key != "_teacher_path"}


def _annotation_items(
    target_file: str,
    frames: dict[str, Any],
    teacher_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if teacher_payload is not None:
        teacher_items = _teacher_annotation_items(teacher_payload)
        if teacher_items:
            return teacher_items
        return [
            {
                "review_id": f"{target_file.removesuffix('.json')}_teacher_payload",
                "status": "uncertain",
                "confidence": 0.86,
                "source": "teacher_artifact",
                "teacher_payload_attached": True,
            }
        ]

    frame_entries = list(_sample_frames(frames.get("frames", []), max_count=8))
    width, height = _source_resolution(frames)
    if target_file == "court_corners.json":
        return _court_corner_items(frame_entries[:1], width, height)
    if target_file == "players.json":
        return _player_items(frame_entries, width, height)
    if target_file == "ball.json":
        return _ball_items(frame_entries, width, height)
    if target_file == "events.json":
        return _event_items(frame_entries, width, height)
    if target_file == "racket_pose.json":
        return _racket_items(frame_entries, width, height)
    if target_file == "foot_contact.json":
        return _foot_contact_items(frame_entries, width, height)
    if target_file == "metrics.json":
        return [{"review_id": "metrics_smoke_summary", "status": "uncertain", "confidence": 0.2, "metrics": {}}]
    return [{"review_id": f"{target_file.removesuffix('.json')}_smoke", "status": "uncertain", "confidence": 0.2}]


def _teacher_annotation_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = payload.get("annotation")
    raw_items = annotation.get("items") if isinstance(annotation, dict) else payload.get("items")
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("review_id", f"teacher_item_{index:04d}")
        normalized.setdefault("status", "uncertain")
        normalized.setdefault("source", "teacher_artifact")
        normalized.setdefault("confidence", normalized.get("conf", 0.75))
        items.append(normalized)
    return items


def _sample_frames(frames: Iterable[Any], *, max_count: int) -> Iterable[dict[str, Any]]:
    normalized = [frame for frame in frames if isinstance(frame, dict)]
    if not normalized:
        normalized = [{"name": "frame_000001.jpg", "path": ""}]
    if len(normalized) <= max_count:
        return normalized
    step = max(1, len(normalized) // max_count)
    return normalized[::step][:max_count]


def _frame_name(frame: dict[str, Any], index: int) -> str:
    return str(frame.get("name") or f"frame_{index + 1:06d}.jpg")


def _court_corner_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    frame = frames[0] if frames else {"name": "frame_000001.jpg"}
    return [
        {
            "review_id": "court_corners_manual_seed",
            "frame": _frame_name(frame, 0),
            "status": "uncertain",
            "confidence": 0.2,
            "source": "manual_4_corner_required",
            "court_corners": {
                "far_left": [round(width * 0.38, 2), round(height * 0.12, 2)],
                "far_right": [round(width * 0.62, 2), round(height * 0.12, 2)],
                "near_right": [round(width * 0.84, 2), round(height * 0.92, 2)],
                "near_left": [round(width * 0.16, 2), round(height * 0.92, 2)],
            },
            "reasons": ["manual_4_corner_tap_needed", "smoke_generated"],
        }
    ]


def _player_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        cx = width * (0.42 + 0.06 * (index % 3))
        cy = height * 0.55
        items.append(
            {
                "review_id": f"player_smoke_{index:04d}",
                "frame": _frame_name(frame, index),
                "status": "uncertain",
                "confidence": 0.25,
                "id": "p1",
                "bbox": [round(cx - width * 0.035, 2), round(cy - height * 0.18, 2), round(width * 0.07, 2), round(height * 0.32, 2)],
                "keypoints_px": [[round(cx, 2), round(cy - height * 0.12, 2)], [round(cx, 2), round(cy, 2)], [round(cx, 2), round(cy + height * 0.16, 2)]],
            }
        )
    return items


def _ball_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    count = max(1, len(frames))
    for index, frame in enumerate(frames):
        ratio = index / max(1, count - 1)
        items.append(
            {
                "review_id": f"ball_smoke_{index:04d}",
                "frame": _frame_name(frame, index),
                "status": "uncertain",
                "confidence": 0.2,
                "xy_px": [round(width * (0.25 + 0.5 * ratio), 2), round(height * (0.42 + 0.08 * ratio), 2)],
                "reasons": ["teacher_model_unavailable", "smoke_generated"],
            }
        )
    return items


def _event_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    if not frames:
        frames = [{"name": "frame_000001.jpg"}]
    index = len(frames) // 2
    return [
        {
            "review_id": "event_smoke_contact",
            "frame": _frame_name(frames[index], index),
            "status": "uncertain",
            "confidence": 0.2,
            "type": "contact",
            "label": "contact?",
            "xy_px": [round(width * 0.5, 2), round(height * 0.5, 2)],
        }
    ]


def _racket_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    return [
        {
            "review_id": f"racket_smoke_{index:04d}",
            "frame": _frame_name(frame, index),
            "status": "uncertain",
            "confidence": 0.15,
            "player_id": "p1",
            "keypoints_px": [[round(width * 0.55, 2), round(height * 0.45, 2)], [round(width * 0.6, 2), round(height * 0.48, 2)]],
            "label": "racket?",
        }
        for index, frame in enumerate(frames[:3])
    ]


def _foot_contact_items(frames: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    return [
        {
            "review_id": f"foot_contact_smoke_{index:04d}",
            "frame": _frame_name(frame, index),
            "status": "uncertain",
            "confidence": 0.25,
            "player_id": "p1",
            "foot": "left" if index % 2 == 0 else "right",
            "xy_px": [round(width * 0.48, 2), round(height * 0.78, 2)],
        }
        for index, frame in enumerate(frames[:4])
    ]


def _uncertain_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    target = payload["annotation"]["target_file"]
    out: list[dict[str, Any]] = []
    for index, item in enumerate(payload["annotation"]["items"]):
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        confidence = item.get("confidence")
        uncertain = status == "uncertain" or (isinstance(confidence, (int, float)) and confidence < 0.7)
        if uncertain:
            frame = item.get("frame") or "frame_000001.jpg"
            out.append(
                {
                    "review_id": item.get("review_id", f"{target}_{index}"),
                    "target_file": target,
                    "frame": frame,
                    "confidence": confidence,
                    "reasons": list(item.get("reasons", [])) or ["low_confidence", "draft_prototype"],
                }
            )
    return out


__all__ = [
    "ALL_LABEL_FILES",
    "COMPATIBILITY_LABEL_FILES",
    "PROTOTYPE_GATE_CLIPS",
    "PROTOTYPE_LABEL_FILES",
    "bootstrap_prototype_gate",
    "h100_defaults",
]
