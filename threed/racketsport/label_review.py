"""Human-in-the-loop review bundle helpers for prototype draft labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from threed.racketsport.autolabel import PROTOTYPE_GATE_CLIPS

CVAT_LABELS = [
    {"name": "court_corner", "attributes": ["corner_name"]},
    {"name": "court_keypoint", "attributes": ["keypoint_name"]},
    {"name": "player_box", "attributes": ["player_id"]},
    {
        "name": "ball",
        "attributes": [
            "visibility",
            "center_convention",
            "blur_angle_deg",
            "blur_length_px",
            "blur_width_px",
            "blur_label_quality",
        ],
    },
    {"name": "event", "attributes": ["event_type"]},
    {"name": "racket_keypoint", "attributes": ["keypoint_name"]},
    {"name": "foot_contact", "attributes": ["foot"]},
]

MAX_REVIEW_ITEMS_PER_LABEL = 20


def export_review_bundle(
    *,
    drafts_root: Path,
    frames_root: Path,
    out: Path,
    confidence_threshold: float = 0.7,
) -> dict[str, Any]:
    """Copy uncertain frames and write correction templates."""

    drafts_root = Path(drafts_root)
    frames_root = Path(frames_root)
    out = Path(out)
    image_root = out / "images"
    corrections_root = out / "corrections"
    image_root.mkdir(parents=True, exist_ok=True)
    corrections_root.mkdir(parents=True, exist_ok=True)

    clip_entries: list[dict[str, Any]] = []
    missing_source_images: list[dict[str, str]] = []
    review_item_count = 0
    for clip_dir in sorted(path for path in drafts_root.iterdir() if path.is_dir()):
        labels_dir = clip_dir / "labels"
        if not labels_dir.is_dir():
            continue
        clip_items: list[dict[str, Any]] = []
        templates: dict[str, list[str]] = {}
        for label_path in sorted(labels_dir.glob("*.json")):
            if label_path.name in {"status.json", "prototype_autolabel_manifest.json", "uncertain_frames.json"}:
                continue
            payload = _read_json(label_path)
            for item in _review_items(payload, label_path.name, confidence_threshold):
                frame = str(item["frame"])
                image_path = frames_root / clip_dir.name / frame
                bundle_image = image_root / clip_dir.name / frame
                bundle_image.parent.mkdir(parents=True, exist_ok=True)
                source_image_exists = image_path.is_file()
                if source_image_exists:
                    shutil.copy2(image_path, bundle_image)
                else:
                    missing_source_images.append(
                        {
                            "clip": clip_dir.name,
                            "frame": frame,
                            "source_image_path": str(image_path),
                            "bundle_image_path": str(bundle_image),
                            "review_id": str(item["review_id"]),
                        }
                    )
                review_item = {
                    **item,
                    "clip": clip_dir.name,
                    "image_path": str(bundle_image),
                    "source_image_path": str(image_path),
                    "source_image_exists": source_image_exists,
                }
                clip_items.append(review_item)
                templates.setdefault(str(item["target_file"]), []).append(str(item["review_id"]))
        for target_file, review_ids in templates.items():
            template_path = corrections_root / clip_dir.name / target_file
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "draft_prototype_corrections",
                        "clip": clip_dir.name,
                        "target_file": target_file,
                        "review_items": review_ids,
                        "items": [],
                        "not_ground_truth": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if clip_items:
            review_item_count += len(clip_items)
            clip_entries.append({"clip": clip_dir.name, "review_items": clip_items})

    if review_item_count == 0:
        status = "no_review_items"
    elif missing_source_images:
        status = "blocked_missing_review_images"
    else:
        status = "ready_for_human_review"

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_label_review_bundle",
        "status": status,
        "drafts_root": str(drafts_root),
        "frames_root": str(frames_root),
        "out": str(out),
        "prototype_gate_clips": list(PROTOTYPE_GATE_CLIPS),
        "review_item_count": review_item_count,
        "missing_source_image_count": len(missing_source_images),
        "missing_source_images": missing_source_images,
        "clips": clip_entries,
        "not_ground_truth": True,
    }
    (out / "review_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def export_cvat_tasks(*, review_manifest: Path, out: Path) -> dict[str, Any]:
    """Write a simple CVAT-friendly task folder per clip."""

    manifest = _read_json(Path(review_manifest))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    task_count = 0
    missing_source_images: list[dict[str, str]] = []
    tasks: list[dict[str, Any]] = []
    for clip in manifest.get("clips", []):
        clip_name = str(clip["clip"])
        task_dir = out / clip_name
        image_dir = task_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        task_missing_source_images: list[dict[str, str]] = []
        for item in clip.get("review_items", []):
            src = Path(item["image_path"])
            dst = image_dir / src.name
            source_image_exists = src.is_file() and item.get("source_image_exists", True) is not False
            if source_image_exists:
                shutil.copy2(src, dst)
                images.append(
                    {
                        "file_name": src.name,
                        "frame": item["frame"],
                        "review_id": item["review_id"],
                        "target_file": item["target_file"],
                    }
                )
            else:
                missing = {
                    "clip": clip_name,
                    "frame": str(item["frame"]),
                    "image_path": str(src),
                    "review_id": str(item["review_id"]),
                }
                missing_source_images.append(missing)
                task_missing_source_images.append(missing)
        task_status = "blocked_missing_review_images" if task_missing_source_images else "ready_for_cvat_review"
        task = {
            "schema_version": 1,
            "artifact_type": "racketsport_cvat_task",
            "status": task_status,
            "task_name": f"racketsport_{clip_name}_label_review",
            "clip": clip_name,
            "labels": CVAT_LABELS,
            "images": images,
            "missing_source_image_count": len(task_missing_source_images),
            "missing_source_images": task_missing_source_images,
            "corrections_hint": "Export annotations, then convert them into the correction template under review_bundle/corrections/<clip>/<target_file>.",
            "not_ground_truth": True,
        }
        (task_dir / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        task_count += 1
        tasks.append({"clip": clip_name, "task_dir": str(task_dir), "image_count": len(images)})
    if missing_source_images:
        status = "blocked_missing_review_images"
    elif task_count == 0:
        status = "no_review_items"
    else:
        status = "ready_for_cvat_review"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_task_export",
        "status": status,
        "task_count": task_count,
        "missing_source_image_count": len(missing_source_images),
        "missing_source_images": missing_source_images,
        "tasks": tasks,
    }


def import_corrected_labels(*, drafts_root: Path, corrections_root: Path, allow_missing_drafts: bool = False) -> dict[str, Any]:
    """Merge correction template items back into draft labels without verification."""

    drafts_root = Path(drafts_root)
    corrections_root = Path(corrections_root)
    imported = 0
    correction_file_count = 0
    files: list[dict[str, Any]] = []
    skipped_corrections: list[dict[str, str]] = []
    for correction_path in sorted(corrections_root.glob("*/*.json")):
        correction_file_count += 1
        correction = _read_json(correction_path)
        clip = str(correction.get("clip") or correction_path.parent.name)
        target_file = str(correction.get("target_file") or correction_path.name)
        draft_path = drafts_root / clip / "labels" / target_file
        if not draft_path.is_file():
            skipped_corrections.append(
                {
                    "clip": clip,
                    "target_file": target_file,
                    "correction_path": str(correction_path),
                    "reason": "missing_draft_allowed" if allow_missing_drafts else "missing_draft",
                    "draft_path": str(draft_path),
                }
            )
            continue
        draft = _read_json(draft_path)
        annotation = draft.setdefault("annotation", {})
        items = annotation.setdefault("items", [])
        if not isinstance(items, list):
            items = []
            annotation["items"] = items
        corrections = [item for item in correction.get("items", []) if isinstance(item, dict)]
        correction_review = correction.get("review")
        default_item_status = "corrected_unverified"
        if (
            target_file == "court_keypoints.json"
            and isinstance(correction_review, dict)
            and correction_review.get("status") == "reviewed"
        ):
            default_item_status = "reviewed"
        for corrected in corrections:
            corrected = {**corrected, "status": corrected.get("status", default_item_status)}
            _replace_or_prepend(items, corrected)
            imported += 1
        if target_file == "court_keypoints.json" and isinstance(correction_review, dict):
            draft["review"] = dict(correction_review)
        imports = annotation.setdefault("review_imports", [])
        imports.append(
            {
                "correction_path": str(correction_path),
                "status": "corrected_unverified",
                "item_count": len(corrections),
            }
        )
        draft["status"] = draft.get("status", "draft_manual_annotation")
        draft["not_ground_truth"] = True
        draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files.append({"clip": clip, "target_file": target_file, "imported_items": len(corrections), "draft_path": str(draft_path)})
    missing_draft_count = sum(1 for item in skipped_corrections if item["reason"] == "missing_draft")
    allowed_missing_draft_count = sum(1 for item in skipped_corrections if item["reason"] == "missing_draft_allowed")
    if allowed_missing_draft_count and imported == 0:
        status = "missing_drafts_allowed"
    elif allowed_missing_draft_count:
        status = "partial_missing_drafts_allowed"
    elif missing_draft_count and imported == 0:
        status = "blocked_missing_drafts"
    elif missing_draft_count:
        status = "partial_missing_drafts"
    elif correction_file_count == 0:
        status = "no_corrections"
    elif imported == 0:
        status = "no_imported_items"
    else:
        status = "imported"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_label_review_import",
        "status": status,
        "correction_file_count": correction_file_count,
        "imported_item_count": imported,
        "missing_draft_count": missing_draft_count + allowed_missing_draft_count,
        "skipped_corrections": skipped_corrections,
        "files": files,
    }


def _review_items(payload: dict[str, Any], target_file: str, confidence_threshold: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if target_file == "uncertain_frames.json":
        for index, frame_item in enumerate(payload.get("frames", [])):
            if isinstance(frame_item, dict):
                items.append(
                    {
                        "review_id": str(frame_item.get("review_id", f"uncertain_{index:04d}")),
                        "target_file": str(frame_item.get("target_file", "court_corners.json")),
                        "frame": str(frame_item.get("frame", "frame_000001.jpg")),
                        "reason": ",".join(frame_item.get("reasons", [])) if isinstance(frame_item.get("reasons"), list) else "uncertain_frames",
                    }
                )
        return items[:MAX_REVIEW_ITEMS_PER_LABEL]

    annotation = payload.get("annotation")
    raw_items = annotation.get("items", []) if isinstance(annotation, dict) else []
    if not isinstance(raw_items, list):
        return []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence")
        is_low_conf = isinstance(confidence, (int, float)) and float(confidence) < confidence_threshold
        status = str(item.get("status", ""))
        if status != "uncertain" and not is_low_conf:
            continue
        frame = item.get("frame") or _first_frame(payload) or "frame_000001.jpg"
        reason = "status=uncertain" if status == "uncertain" else f"confidence<{confidence_threshold}"
        items.append(
            {
                "review_id": str(item.get("review_id", f"{target_file}_{index:04d}")),
                "target_file": target_file,
                "frame": str(frame),
                "reason": reason,
                "confidence": confidence,
            }
        )
    return items[:MAX_REVIEW_ITEMS_PER_LABEL]


def _first_frame(payload: dict[str, Any]) -> str | None:
    frames = payload.get("frames")
    if isinstance(frames, dict) and isinstance(frames.get("frames"), list) and frames["frames"]:
        first = frames["frames"][0]
        if isinstance(first, dict):
            return str(first.get("name") or first.get("frame") or "")
        return str(first)
    return None


def _replace_or_prepend(items: list[Any], corrected: dict[str, Any]) -> None:
    review_id = corrected.get("review_id")
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("review_id") == review_id:
            items[index] = corrected
            return
    items.insert(0, corrected)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


__all__ = ["PROTOTYPE_GATE_CLIPS", "export_cvat_tasks", "export_review_bundle", "import_corrected_labels"]
