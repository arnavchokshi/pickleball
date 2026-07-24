#!/usr/bin/env python3
"""Local, crash-resistant sequential labeler for the 2026-07 court package.

This is a review/data-entry surface, not an accuracy claim.  It deliberately
does not show model suggestions while the owner labels so the human labels stay
independent of the preview baseline being evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import mimetypes
import secrets
import socket
import sys
import threading
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.append_lock import file_lock, write_atomic


PACKAGE_ROOT = Path("cvat_upload/court_diversity_20260712")
PACKAGE_MANIFEST = PACKAGE_ROOT / "package_manifest.json"
IMPORT_REPORT = PACKAGE_ROOT / "import_report_20260712_courtsession.json"
OWNER_EXCLUSIONS = PACKAGE_ROOT / "owner_camera_policy_and_exclusions.json"
SUGGESTIONS = PACKAGE_ROOT / "model_estimated_suggestions.json"
PROGRESS_PATH = PACKAGE_ROOT / "owner_sequential_label_progress.json"
SAVE_LOCK_PATH = PACKAGE_ROOT / ".owner-sequential-label-save.lock"
MAX_SAVE_BYTES = 5_000_000
SAVE_TOKEN_PLACEHOLDER = "__COURT_DIVERSITY_SAVE_TOKEN__"

LABEL_ORDER = (
    "far_left_corner",
    "far_baseline_center",
    "far_right_corner",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
)
REQUIRED_FLOOR_ANCHORS = {
    "far_left_corner",
    "far_right_corner",
    "near_left_corner",
    "near_right_corner",
}
LABEL_HELP = {
    "far_left_corner": "Back/far baseline, left corner",
    "far_baseline_center": "Back/far baseline, center line",
    "far_right_corner": "Back/far baseline, right corner",
    "far_nvz_left": "Far kitchen line, left sideline",
    "far_nvz_center": "Far kitchen line, center line",
    "far_nvz_right": "Far kitchen line, right sideline",
    "net_left_sideline": "Top tape, above where the net meets the left court edge",
    "net_center": "Top tape at court center",
    "net_right_sideline": "Top tape, above where the net meets the right court edge",
    "near_nvz_left": "Near kitchen line, left sideline",
    "near_nvz_center": "Near kitchen line, center line",
    "near_nvz_right": "Near kitchen line, right sideline",
    "near_left_corner": "Front/near baseline, left corner",
    "near_baseline_center": "Front/near baseline, center line",
    "near_right_corner": "Front/near baseline, right corner",
}
VALID_STATUSES = {"unreviewed", "in_progress", "reviewed", "reviewed_partial", "excluded"}
VALID_EXCLUSION_REASONS = {"sideways_view", "fisheye", "camera_too_low", "bad_angle"}
EXCLUSION_REASON_ALIASES = {"bad_diagonal_angle": "bad_angle"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def parse_cvat_points(export_zip: Path) -> dict[str, dict[str, list[float]]]:
    """Read owner point annotations from one CVAT-for-images export."""
    with zipfile.ZipFile(export_zip) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith("annotations.xml")]
        if len(xml_names) != 1:
            raise ValueError(f"expected one annotations.xml in {export_zip}, found {len(xml_names)}")
        root = ElementTree.fromstring(archive.read(xml_names[0]))

    parsed: dict[str, dict[str, list[float]]] = {}
    for image in root.findall("image"):
        file_name = image.attrib.get("name", "")
        if not file_name:
            continue
        points: dict[str, list[float]] = {}
        for shape in image.findall("points"):
            label = shape.attrib.get("label", "")
            if label not in LABEL_ORDER:
                continue
            raw_xy = shape.attrib.get("points", "").split(",")
            if len(raw_xy) != 2:
                continue
            points[label] = [float(raw_xy[0]), float(raw_xy[1])]
        if points:
            parsed[file_name] = points
    return parsed


def _task_ids(root: Path) -> dict[str, int]:
    report = _read_object(root / IMPORT_REPORT)
    result: dict[str, int] = {}
    for task in report.get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("task_name"), str):
            result[task["task_name"]] = int(task["task_id"])
    return result


def _owner_exclusions(root: Path, shards: list[dict[str, Any]]) -> dict[str, list[str]]:
    path = root / OWNER_EXCLUSIONS
    if not path.is_file():
        return {}
    payload = _read_object(path)
    task_by_id = {str(task_id): name for name, task_id in _task_ids(root).items()}
    shard_by_name = {str(shard["shard_name"]): shard for shard in shards}
    exclusions: dict[str, list[str]] = {}
    for task_id, task in payload.get("tasks", {}).items():
        if not isinstance(task, dict):
            continue
        shard_name = str(task.get("task_name") or task_by_id.get(str(task_id), ""))
        shard = shard_by_name.get(shard_name)
        if not shard:
            continue
        files = shard.get("file_names", [])
        for entry in task.get("excluded_frames", []):
            if not isinstance(entry, dict):
                continue
            frame = int(entry.get("frame", -1))
            if 0 <= frame < len(files):
                reasons = [
                    EXCLUSION_REASON_ALIASES.get(str(value), str(value))
                    for value in entry.get("reasons", [])
                ]
                exclusions[str(files[frame])] = reasons
    return exclusions


def build_initial_progress(
    root: Path,
    *,
    cvat_export: Path | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package = _read_object(root / PACKAGE_MANIFEST)
    shards = package.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError("package manifest shards must be a list")
    known_files = {str(image["file_name"]): image for image in package.get("images", [])}
    imported = parse_cvat_points(cvat_export) if cvat_export and cvat_export.is_file() else {}
    exclusions = _owner_exclusions(root, shards)

    progress = existing if isinstance(existing, dict) else {}
    raw_items = progress.get("items") if isinstance(progress.get("items"), dict) else {}
    items: dict[str, Any] = {name: value for name, value in raw_items.items() if name in known_files and isinstance(value, dict)}
    for file_name in known_files:
        if file_name in items:
            continue
        if file_name in exclusions:
            items[file_name] = {
                "status": "excluded",
                "keypoints": {},
                "skipped_points": {},
                "exclusion_reasons": exclusions[file_name],
                "provenance": "owner_camera_policy_2026-07-22",
            }
        elif file_name in imported:
            items[file_name] = {
                "status": "reviewed_partial" if len(imported[file_name]) < len(LABEL_ORDER) else "reviewed",
                "keypoints": imported[file_name],
                "skipped_points": {},
                "exclusion_reasons": [],
                "provenance": f"cvat_export:{cvat_export.name}",
            }

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diversity_owner_sequential_labels",
        "authority": "owner_reviewed",
        "saved_at": str(progress.get("saved_at") or _utc_now()),
        "label_order": list(LABEL_ORDER),
        "items": items,
    }


def _frame_index(package: dict[str, Any], task_ids: dict[str, int]) -> list[dict[str, Any]]:
    images_by_name = {str(image["file_name"]): image for image in package.get("images", [])}
    frames: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(package.get("shards", [])):
        shard_name = str(shard["shard_name"])
        task_id = task_ids.get(str(shard.get("task_name") or shard_name))
        for frame_number, file_name in enumerate(shard.get("file_names", [])):
            image = images_by_name[str(file_name)]
            resolution = image.get("resolution", [1280, 720])
            frames.append(
                {
                    "global_index": len(frames),
                    "shard_index": shard_index,
                    "shard_name": shard_name,
                    "task_id": task_id,
                    "frame_number": frame_number,
                    "file_name": str(file_name),
                    "url": f"/asset?path={(PACKAGE_ROOT / 'frames' / str(file_name)).as_posix()}",
                    "width": int(resolution[0]),
                    "height": int(resolution[1]),
                    "source_id": image.get("source_id"),
                    "source_title": image.get("title"),
                }
            )
    return frames


def validate_progress(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    package = _read_object(root / PACKAGE_MANIFEST)
    images = {str(image["file_name"]): image for image in package.get("images", [])}
    raw_items = payload.get("items")
    if not isinstance(raw_items, dict) or len(raw_items) > len(images):
        raise ValueError("items must be an object containing at most the package image count")

    clean_items: dict[str, Any] = {}
    for file_name, raw in raw_items.items():
        if file_name not in images or not isinstance(raw, dict):
            raise ValueError(f"unknown or malformed image item: {file_name}")
        status = str(raw.get("status", "unreviewed"))
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status for {file_name}: {status}")
        width, height = [int(value) for value in images[file_name].get("resolution", [1280, 720])]
        raw_points = raw.get("keypoints", {})
        if not isinstance(raw_points, dict):
            raise ValueError(f"keypoints must be an object for {file_name}")
        points: dict[str, list[float]] = {}
        for label, xy in raw_points.items():
            if label not in LABEL_ORDER or not isinstance(xy, list) or len(xy) != 2:
                raise ValueError(f"invalid keypoint {label!r} for {file_name}")
            x, y = float(xy[0]), float(xy[1])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"out-of-bounds keypoint {label!r} for {file_name}")
            points[label] = [round(x, 3), round(y, 3)]
        raw_skipped = raw.get("skipped_points", {})
        if not isinstance(raw_skipped, dict):
            raise ValueError(f"skipped_points must be an object for {file_name}")
        skipped = {str(label): True for label, value in raw_skipped.items() if label in LABEL_ORDER and value}
        reasons = [str(reason) for reason in raw.get("exclusion_reasons", [])]
        if any(reason not in VALID_EXCLUSION_REASONS for reason in reasons):
            raise ValueError(f"invalid exclusion reason for {file_name}")
        clean_items[file_name] = {
            "status": status,
            "keypoints": points,
            "skipped_points": skipped,
            "exclusion_reasons": reasons,
            "provenance": str(raw.get("provenance") or "owner_sequential_ui"),
        }

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diversity_owner_sequential_labels",
        "authority": "owner_reviewed",
        "saved_at": _utc_now(),
        "label_order": list(LABEL_ORDER),
        "items": clean_items,
    }


def score_progress(root: Path, progress: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic-only comparison against the unverified classical suggestions."""
    suggestions = _read_object(root / SUGGESTIONS)
    errors: list[float] = []
    usable_rows = 0
    human_rows = 0
    excluded_rows = 0
    point_count = 0
    for file_name, item in progress.get("items", {}).items():
        if not isinstance(item, dict):
            continue
        if item.get("status") == "excluded":
            excluded_rows += 1
            continue
        points = item.get("keypoints", {})
        if not isinstance(points, dict) or not points:
            continue
        human_rows += 1
        point_count += len(points)
        if REQUIRED_FLOOR_ANCHORS.issubset(points):
            usable_rows += 1
        suggestion = suggestions.get(file_name, {})
        suggestion_points = suggestion.get("keypoints_px", {}) if isinstance(suggestion, dict) else {}
        for label, xy in points.items():
            candidate = suggestion_points.get(label, {}) if isinstance(suggestion_points, dict) else {}
            predicted = candidate.get("xy") if isinstance(candidate, dict) else None
            if isinstance(predicted, list) and len(predicted) == 2:
                errors.append(math.hypot(float(xy[0]) - float(predicted[0]), float(xy[1]) - float(predicted[1])))
    ordered = sorted(errors)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1) if ordered else 0
    return {
        "status": "diagnostic_only_do_not_promote",
        "human_labeled_frames": human_rows,
        "excluded_frames": excluded_rows,
        "strict_ingest_eligible_frames": usable_rows,
        "human_point_count": point_count,
        "compared_point_count": len(errors),
        "median_error_px": round((ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2, 3) if ordered else None,
        "p90_error_px": round(ordered[p90_index], 3) if ordered else None,
        "pck_at_5px": round(sum(value <= 5 for value in errors) / len(errors), 6) if errors else None,
        "pck_at_10px": round(sum(value <= 10 for value in errors) / len(errors), 6) if errors else None,
        "note": "Scores the existing low-confidence preview suggestions against owner clicks; it is not an independent promotion result.",
    }


def export_progress_to_cvat(
    root: Path,
    progress: dict[str, Any],
    out_dir: Path,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Losslessly adapt completed sequential labels to four CVAT 1.1 shards.

    Rejected frames remain present for exact package reconciliation but carry
    no point shapes, so the strict ingester rejects rather than trains on them.
    The separate export report retains their owner reasons.
    """
    package = _read_object(root / PACKAGE_MANIFEST)
    images = {str(image["file_name"]): image for image in package.get("images", [])}
    clean = validate_progress(root, progress)
    if isinstance(progress.get("saved_at"), str):
        clean["saved_at"] = progress["saved_at"]
    item_names = set(clean["items"])
    expected_names = set(images)
    if item_names != expected_names:
        missing = sorted(expected_names - item_names)
        extra = sorted(item_names - expected_names)
        raise ValueError(f"completed export must reconcile all package images: missing={missing[:10]}, extra={extra[:10]}")
    incomplete = sorted(
        file_name
        for file_name, item in clean["items"].items()
        if item["status"] in {"unreviewed", "in_progress"}
    )
    if incomplete:
        raise ValueError(f"cannot export {len(incomplete)} unfinished frames: {incomplete[:10]}")

    task_ids = _task_ids(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for shard in package.get("shards", []):
        shard_name = str(shard["shard_name"])
        task_name = str(shard.get("task_name") or shard_name)
        annotations = ElementTree.Element("annotations")
        ElementTree.SubElement(annotations, "version").text = "1.1"
        meta = ElementTree.SubElement(annotations, "meta")
        job = ElementTree.SubElement(meta, "job")
        ElementTree.SubElement(job, "id").text = str(task_ids.get(task_name, ""))
        ElementTree.SubElement(job, "size").text = str(len(shard.get("file_names", [])))
        ElementTree.SubElement(job, "mode").text = "annotation"
        labels_element = ElementTree.SubElement(job, "labels")
        for label in LABEL_ORDER:
            label_element = ElementTree.SubElement(labels_element, "label")
            ElementTree.SubElement(label_element, "name").text = label
            ElementTree.SubElement(label_element, "type").text = "points"

        for frame_number, file_name_value in enumerate(shard.get("file_names", [])):
            file_name = str(file_name_value)
            image = images[file_name]
            width, height = [int(value) for value in image.get("resolution", [1280, 720])]
            image_element = ElementTree.SubElement(
                annotations,
                "image",
                {"id": str(frame_number), "name": file_name, "width": str(width), "height": str(height)},
            )
            item = clean["items"][file_name]
            if item["status"] == "excluded":
                excluded.append(
                    {
                        "file_name": file_name,
                        "shard_name": shard_name,
                        "frame_number": frame_number,
                        "reasons": item["exclusion_reasons"],
                    }
                )
                continue
            for label in LABEL_ORDER:
                xy = item["keypoints"].get(label)
                if xy is None:
                    continue
                point = ElementTree.SubElement(
                    image_element,
                    "points",
                    {
                        "label": label,
                        "source": "manual",
                        "occluded": "0",
                        "points": f"{xy[0]:.3f},{xy[1]:.3f}",
                        "z_order": "0",
                    },
                )
                ElementTree.SubElement(point, "attribute", {"name": "source"}).text = "owner"

        ElementTree.indent(annotations, space="  ")
        xml_bytes = ElementTree.tostring(annotations, encoding="utf-8", xml_declaration=True)
        destination = out_dir / f"{shard_name}_annotations.zip"
        temporary = out_dir / f".{destination.name}.{secrets.token_hex(8)}.tmp"
        try:
            info = zipfile.ZipInfo("annotations.xml", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr(info, xml_bytes)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        exported.append(
            {
                "shard_name": shard_name,
                "path": str(destination),
                "image_count": len(shard.get("file_names", [])),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_sequential_court_labels_cvat_export",
        "created_at": _utc_now(),
        "source_labels": str(source_path) if source_path else str(root / PROGRESS_PATH),
        "source_saved_at": clean.get("saved_at"),
        "source_label_sha256": hashlib.sha256(
            json.dumps(progress, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reviewed_frame_count": sum(
            item["status"] in {"reviewed", "reviewed_partial"} for item in clean["items"].values()
        ),
        "excluded_frame_count": len(excluded),
        "point_count": sum(len(item["keypoints"]) for item in clean["items"].values()),
        "skipped_point_count": sum(len(item["skipped_points"]) for item in clean["items"].values()),
        "excluded_frames": excluded,
        "exports": exported,
        "handling": "Excluded frames contain no shapes and cannot become negative training examples.",
    }
    write_atomic(out_dir / "sequential_export_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _manifest(root: Path, prefill: Path | None) -> dict[str, Any]:
    package = _read_object(root / PACKAGE_MANIFEST)
    progress_path = root / PROGRESS_PATH
    existing = _read_object(progress_path) if progress_path.is_file() else None
    progress = build_initial_progress(root, cvat_export=prefill, existing=existing)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diversity_label_ui_manifest",
        "frames": _frame_index(package, _task_ids(root)),
        "labels": [
            {"name": name, "number": index + 1, "help": LABEL_HELP[name], "group": name.split("_", 1)[0]}
            for index, name in enumerate(LABEL_ORDER)
        ],
        "owner_policy": _read_object(root / OWNER_EXCLUSIONS).get("product_input_policy", {}),
        "progress": progress,
        "diagnostic": score_progress(root, progress),
    }


def _write_progress(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    clean = validate_progress(root, payload)
    path = root / PROGRESS_PATH
    with file_lock(root / SAVE_LOCK_PATH):
        write_atomic(path, json.dumps(clean, indent=2, sort_keys=True) + "\n")
    diagnostic = score_progress(root, clean)
    return {"saved": True, "saved_at": clean["saved_at"], "path": str(path), "diagnostic": diagnostic}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DinkVision · Court Label Sprint</title>
  <style>
    :root { --cream:#f4efdf; --paper:#fffaf0; --ink:#19231f; --green:#0c5b47; --deep:#07382d; --lime:#d9f25f; --orange:#ef7f4d; --blue:#4f8da8; --muted:#6b756d; --line:#cbc6b7; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--cream); font-family:"Avenir Next","Gill Sans",sans-serif; }
    button, select { font:inherit; }
    button { cursor:pointer; }
    .shell { height:100vh; min-height:0; display:grid; grid-template-rows:auto 1fr; overflow:hidden; }
    header { display:grid; grid-template-columns:1fr auto; gap:18px; align-items:center; padding:14px 20px; background:var(--deep); color:var(--paper); border-bottom:4px solid var(--lime); }
    .brand { display:flex; align-items:baseline; gap:12px; }
    .brand strong { font-family:"Rockwell","Avenir Next",serif; font-size:21px; letter-spacing:-.02em; }
    .brand span { color:#b9d4c9; font-size:12px; letter-spacing:.09em; text-transform:uppercase; }
    .save-state { text-align:right; font-size:12px; color:#d6e4de; }
    .progress-track { width:260px; height:7px; margin-top:6px; overflow:hidden; border-radius:8px; background:#355f53; }
    .progress-fill { height:100%; width:0; background:var(--lime); transition:width .2s; }
    main { min-height:0; display:grid; grid-template-columns:230px minmax(480px,1fr) 290px; }
    aside { min-height:0; overflow:auto; padding:14px; background:var(--paper); }
    .frames { border-right:1px solid var(--line); }
    .labels { border-left:1px solid var(--line); }
    h2 { margin:0 0 10px; font-family:"Rockwell","Avenir Next",serif; font-size:15px; }
    .task-block { margin-bottom:12px; }
    .task-title { display:flex; justify-content:space-between; padding:5px 4px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .frame-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:5px; }
    .frame-btn { height:34px; border:1px solid var(--line); border-radius:7px; background:#f5f0e4; color:var(--muted); font-size:11px; font-weight:700; }
    .frame-btn.done { background:#dcebd6; color:var(--green); border-color:#a9c7a8; }
    .frame-btn.excluded { background:#eadfd8; color:#86513d; border-color:#d0b4a6; text-decoration:line-through; }
    .frame-btn.partial { background:#fff0c5; color:#765a16; border-color:#dac580; }
    .frame-btn.active { outline:3px solid var(--orange); outline-offset:1px; background:var(--ink); color:white; }
    .stage { min-width:0; display:grid; grid-template-rows:auto 1fr auto; padding:14px 18px 12px; background:#e8e1d1; }
    .active-card { display:grid; grid-template-columns:auto 1fr auto; gap:12px; align-items:center; margin-bottom:12px; padding:10px 13px; border-radius:12px; background:var(--ink); color:white; box-shadow:0 5px 0 rgba(25,35,31,.15); }
    .active-number { display:grid; place-items:center; width:38px; height:38px; border-radius:50%; background:var(--lime); color:var(--ink); font-family:"Rockwell",serif; font-weight:900; }
    .active-name { font-size:17px; font-weight:800; }
    .active-help { color:#c8d2cd; font-size:12px; }
    .flow { color:var(--lime); font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .canvas-wrap { position:relative; align-self:start; max-height:calc(100vh - 205px); background:#101411; box-shadow:0 10px 28px rgba(25,35,31,.22); cursor:crosshair; overflow:hidden; }
    #frameImage { display:block; width:100%; max-height:calc(100vh - 205px); object-fit:contain; }
    #overlay { position:absolute; inset:0; width:100%; height:100%; }
    .excluded-cover { display:none; position:absolute; inset:0; place-items:center; background:rgba(25,35,31,.72); color:white; text-align:center; font-family:"Rockwell",serif; font-size:24px; }
    .excluded-cover.show { display:grid; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding-top:12px; }
    .toolbar button, .download { min-height:38px; padding:8px 12px; border:1px solid #aca596; border-radius:9px; background:var(--paper); color:var(--ink); font-weight:750; }
    .toolbar button.primary { background:var(--green); color:white; border-color:var(--green); }
    .toolbar button.warn { color:#8b3f23; border-color:#c48e76; }
    .toolbar .spacer { flex:1; }
    .meta { width:100%; color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sequence-note { margin:0 0 10px; padding:10px; border-left:4px solid var(--orange); background:#f4ead8; font-size:12px; line-height:1.4; }
    .label-btn { display:grid; grid-template-columns:27px 1fr auto; gap:8px; width:100%; align-items:center; padding:7px 8px; border:0; border-bottom:1px solid #e8e1d4; background:transparent; text-align:left; }
    .label-btn:hover { background:#f3edde; }
    .label-btn.active { background:var(--ink); color:white; border-radius:8px; }
    .label-btn .num { display:grid; place-items:center; width:24px; height:24px; border-radius:50%; background:#dfe8db; color:var(--green); font-size:11px; font-weight:900; }
    .label-btn.active .num { background:var(--lime); color:var(--ink); }
    .label-btn .label { font-size:11px; font-weight:750; overflow:hidden; text-overflow:ellipsis; }
    .label-btn .mark { color:#8a928c; font-size:11px; }
    .label-btn.set .mark { color:var(--green); }
    .label-btn.skipped .mark { color:#a36732; }
    dialog { width:min(470px,90vw); border:0; border-radius:16px; padding:0; box-shadow:0 24px 80px rgba(0,0,0,.35); }
    dialog::backdrop { background:rgba(7,30,24,.65); }
    .dialog-body { padding:22px; background:var(--paper); }
    .dialog-body h3 { margin:0 0 7px; font-family:"Rockwell",serif; }
    .reason { display:block; padding:9px 4px; border-bottom:1px solid #e6dfd2; }
    .dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:18px; }
    .dialog-actions button { padding:8px 13px; border:1px solid var(--line); border-radius:8px; background:white; }
    .dialog-actions .reject { background:#923f28; color:white; border-color:#923f28; }
    @media (max-width:1000px) { main { grid-template-columns:170px 1fr 240px; } .brand span { display:none; } }
  </style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand"><strong>DinkVision Court Sprint</strong><span>one click → next point</span></div>
    <div class="save-state"><div id="saveState">Loading protected work…</div><div class="progress-track"><div class="progress-fill" id="progressFill"></div></div></div>
  </header>
  <main>
    <aside class="frames"><h2>Frames <span id="frameCount"></span></h2><div id="frameList"></div></aside>
    <section class="stage">
      <div class="active-card"><div class="active-number" id="activeNumber">1</div><div><div class="active-name" id="activeName">Far-left corner</div><div class="active-help" id="activeHelp"></div></div><div class="flow">far → near</div></div>
      <div class="canvas-wrap" id="canvasWrap"><img id="frameImage" alt="Court frame" draggable="false"/><svg id="overlay"></svg><div class="excluded-cover" id="excludedCover"></div></div>
      <div class="toolbar">
        <button id="undo">Undo <small>U</small></button><button id="skip">Skip point <small>Space</small></button><button class="warn" id="reject">Reject frame <small>X</small></button>
        <span class="spacer"></span><button id="prevFrame">← Frame</button><button id="nextFrame">Frame →</button><button class="primary" id="saveNow">Save now</button><button id="download">Backup JSON</button>
        <div class="meta" id="frameMeta"></div>
      </div>
    </section>
    <aside class="labels"><h2>Click order</h2><p class="sequence-note"><strong>Back to front, left to right.</strong><br/>For net points: find the court/net intersection, then go vertically up to the <strong>top tape</strong>. Skip anything you cannot see—do not guess.</p><div id="labelList"></div></aside>
  </main>
</div>
<dialog id="rejectDialog"><div class="dialog-body"><h3>Reject this camera view</h3><p>Select every reason. Rejected frames stay out of training—they are not negative examples.</p>
  <label class="reason"><input type="checkbox" value="sideways_view"/> Sideways / pure side view</label>
  <label class="reason"><input type="checkbox" value="fisheye"/> Fisheye view</label>
  <label class="reason"><input type="checkbox" value="camera_too_low"/> Camera too low</label>
  <label class="reason"><input type="checkbox" value="bad_angle"/> Otherwise bad / out-of-contract angle</label>
  <div class="dialog-actions"><button id="cancelReject">Cancel</button><button class="reject" id="confirmReject">Reject frame</button></div>
</div></dialog>
<script>
const SAVE_TOKEN = "__COURT_DIVERSITY_SAVE_TOKEN__";
const STORAGE_KEY = "dinkvision-court-diversity-20260712-v1";
const LINE_GROUPS = [
  ["far_left_corner","far_baseline_center","far_right_corner"], ["far_nvz_left","far_nvz_center","far_nvz_right"],
  ["net_left_sideline","net_center","net_right_sideline"], ["near_nvz_left","near_nvz_center","near_nvz_right"],
  ["near_left_corner","near_baseline_center","near_right_corner"],
  ["far_left_corner","far_nvz_left","net_left_sideline","near_nvz_left","near_left_corner"],
  ["far_baseline_center","far_nvz_center","net_center","near_nvz_center","near_baseline_center"],
  ["far_right_corner","far_nvz_right","net_right_sideline","near_nvz_right","near_right_corner"]
];
const state = { manifest:null, data:null, frameIndex:0, labelIndex:0, undo:[], saveTimer:null, saving:false };
const $ = s => document.querySelector(s);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const frame = () => state.manifest.frames[state.frameIndex];
const label = () => state.manifest.labels[state.labelIndex];
function itemFor(create=true) {
  const name = frame().file_name;
  if (!state.data.items[name] && create) state.data.items[name] = {status:"unreviewed",keypoints:{},skipped_points:{},exclusion_reasons:[],provenance:"owner_sequential_ui"};
  return state.data.items[name];
}
function deep(value) { return JSON.parse(JSON.stringify(value)); }
function resolvedCount(item) { return Object.keys(item?.keypoints || {}).length + Object.keys(item?.skipped_points || {}).length; }
function isDone(item) { return ["reviewed","reviewed_partial","excluded"].includes(item?.status); }
function pushUndo() { state.undo.push({frameIndex:state.frameIndex,labelIndex:state.labelIndex,item:deep(itemFor())}); if (state.undo.length>100) state.undo.shift(); }
function saveLocal() { state.data.saved_at = new Date().toISOString(); localStorage.setItem(STORAGE_KEY, JSON.stringify(state.data)); }
function queueSave() { saveLocal(); $("#saveState").textContent="Browser backup saved · server save queued"; clearTimeout(state.saveTimer); state.saveTimer=setTimeout(saveNow, 350); }
async function saveNow() {
  if (state.saving) { state.saveTimer=setTimeout(saveNow, 350); return; }
  state.saving=true; $("#saveState").textContent="Saving to disk…";
  try {
    const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json","X-Court-Review-Token":SAVE_TOKEN},body:JSON.stringify(state.data)});
    const body=await response.json(); if(!response.ok) throw new Error(body.error || "save failed");
    state.data.saved_at=body.saved_at; localStorage.setItem(STORAGE_KEY,JSON.stringify(state.data));
    $("#saveState").textContent=`Saved to disk · ${new Date(body.saved_at).toLocaleTimeString()}`;
  } catch(err) { $("#saveState").textContent=`Server save failed · browser backup is safe (${err.message})`; }
  finally { state.saving=false; }
}
function firstOpenFrame() { const i=state.manifest.frames.findIndex(f => !isDone(state.data.items[f.file_name])); return i>=0?i:0; }
function nextOpenFrame(start) { for(let offset=1;offset<=state.manifest.frames.length;offset++){const i=(start+offset)%state.manifest.frames.length;if(!isDone(state.data.items[state.manifest.frames[i].file_name])) return i;} return Math.min(start+1,state.manifest.frames.length-1); }
function advance() {
  if(state.labelIndex<state.manifest.labels.length-1){state.labelIndex++;return;}
  itemFor().status="reviewed"; state.frameIndex=nextOpenFrame(state.frameIndex); state.labelIndex=0;
}
function placePoint(event) {
  const item=itemFor(); if(item.status==="excluded") return;
  const image=$("#frameImage"), rect=image.getBoundingClientRect(); if(!image.naturalWidth||!rect.width) return;
  pushUndo(); const name=label().name;
  item.keypoints[name]=[Math.max(0,Math.min(image.naturalWidth,(event.clientX-rect.left)*image.naturalWidth/rect.width)),Math.max(0,Math.min(image.naturalHeight,(event.clientY-rect.top)*image.naturalHeight/rect.height))];
  delete item.skipped_points[name]; item.status="in_progress"; advance(); queueSave(); render();
}
function skipPoint() { const item=itemFor(); if(item.status==="excluded") return; pushUndo(); const name=label().name; delete item.keypoints[name]; item.skipped_points[name]=true; item.status="in_progress"; advance(); queueSave(); render(); }
function undo() { const previous=state.undo.pop(); if(!previous)return; state.frameIndex=previous.frameIndex;state.labelIndex=previous.labelIndex;state.data.items[frame().file_name]=previous.item;queueSave();render(); }
function setFrame(index) { state.frameIndex=Math.max(0,Math.min(state.manifest.frames.length-1,index));state.labelIndex=0;render(); }
function rejectFrame(reasons) { pushUndo(); const item=itemFor();item.status="excluded";item.exclusion_reasons=reasons;item.keypoints={};item.skipped_points={};item.provenance="owner_sequential_ui";queueSave();state.frameIndex=nextOpenFrame(state.frameIndex);state.labelIndex=0;render(); }
function renderFrames() {
  const groups=[]; state.manifest.frames.forEach((f,i)=>{(groups[f.shard_index]??=[]).push({f,i});});
  $("#frameList").innerHTML=groups.map((group,g)=>`<div class="task-block"><div class="task-title"><span>Task ${group[0].f.task_id || g+1}</span><span>${group.filter(x=>isDone(state.data.items[x.f.file_name])).length}/25</span></div><div class="frame-grid">${group.map(({f,i})=>{const item=state.data.items[f.file_name];const cls=item?.status==="excluded"?"excluded":item?.status==="reviewed_partial"?"partial":isDone(item)?"done":"";return `<button class="frame-btn ${cls} ${i===state.frameIndex?"active":""}" data-frame="${i}" title="${esc(f.file_name)}">${f.frame_number+1}</button>`}).join("")}</div></div>`).join("");
  document.querySelectorAll("[data-frame]").forEach(button=>button.onclick=()=>setFrame(Number(button.dataset.frame)));
}
function renderLabels() {
  const item=itemFor(); $("#activeNumber").textContent=label().number;$("#activeName").textContent=label().name.replaceAll("_"," ");$("#activeHelp").textContent=label().help;
  $("#labelList").innerHTML=state.manifest.labels.map((entry,i)=>{const set=!!item.keypoints?.[entry.name],skipped=!!item.skipped_points?.[entry.name];return `<button class="label-btn ${i===state.labelIndex?"active":""} ${set?"set":""} ${skipped?"skipped":""}" data-label="${i}"><span class="num">${entry.number}</span><span class="label">${esc(entry.name.replaceAll("_"," "))}</span><span class="mark">${set?"●":skipped?"skip":"○"}</span></button>`}).join("");
  document.querySelectorAll("[data-label]").forEach(button=>button.onclick=()=>{state.labelIndex=Number(button.dataset.label);render();});
}
function draw() {
  const image=$("#frameImage"),svg=$("#overlay"),item=itemFor();if(!image.naturalWidth)return;svg.setAttribute("viewBox",`0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  const pts=item.keypoints||{};let lines="";for(const group of LINE_GROUPS){for(let i=0;i<group.length-1;i++){const a=pts[group[i]],b=pts[group[i+1]];if(a&&b)lines+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="rgba(217,242,95,.82)" stroke-width="2" vector-effect="non-scaling-stroke"/>`;}}
  const dots=state.manifest.labels.map((entry,i)=>{const p=pts[entry.name];if(!p)return"";const active=i===state.labelIndex;return `<g><circle cx="${p[0]}" cy="${p[1]}" r="${active?8:6}" fill="${active?"#ef7f4d":"#d9f25f"}" stroke="#19231f" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="${p[0]+9}" y="${p[1]-8}" fill="#fffaf0" stroke="#19231f" stroke-width="3" paint-order="stroke" font-size="12">${i+1}</text></g>`}).join("");svg.innerHTML=lines+dots;
}
function render() {
  renderFrames();renderLabels();const f=frame(),item=itemFor();const image=$("#frameImage");if(!image.src.endsWith(f.url)){image.src=f.url;image.onload=draw;}else draw();
  const done=state.manifest.frames.filter(f=>isDone(state.data.items[f.file_name])).length;$("#frameCount").textContent=`${done}/100`;$("#progressFill").style.width=`${done}%`;
  $("#frameMeta").textContent=`Task ${f.task_id} · frame ${f.frame_number+1}/25 · ${f.file_name} · ${resolvedCount(item)}/15 resolved`;
  const cover=$("#excludedCover");cover.classList.toggle("show",item.status==="excluded");cover.innerHTML=item.status==="excluded"?`<div>Excluded from training<br/><small>${esc((item.exclusion_reasons||[]).join(" · "))}</small></div>`:"";
}
function downloadBackup(){const blob=new Blob([JSON.stringify(state.data,null,2)+"\n"],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`court_labels_backup_${new Date().toISOString().replaceAll(":","-")}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
async function boot(){const response=await fetch("/api/manifest");state.manifest=await response.json();state.data=state.manifest.progress;try{const local=JSON.parse(localStorage.getItem(STORAGE_KEY)||"null");if(local?.items&&Date.parse(local.saved_at)>Date.parse(state.data.saved_at)){state.data=local;$("#saveState").textContent="Recovered newer browser backup · saving to disk";setTimeout(saveNow,100);}}catch(_){}state.frameIndex=firstOpenFrame();
  $("#overlay").addEventListener("click",placePoint);$("#skip").onclick=skipPoint;$("#undo").onclick=undo;$("#prevFrame").onclick=()=>setFrame(state.frameIndex-1);$("#nextFrame").onclick=()=>setFrame(state.frameIndex+1);$("#saveNow").onclick=saveNow;$("#download").onclick=downloadBackup;
  const dialog=$("#rejectDialog");$("#reject").onclick=()=>dialog.showModal();$("#cancelReject").onclick=()=>dialog.close();$("#confirmReject").onclick=()=>{const reasons=[...dialog.querySelectorAll("input:checked")].map(x=>x.value);if(!reasons.length)return;dialog.querySelectorAll("input").forEach(x=>x.checked=false);dialog.close();rejectFrame(reasons);};
  document.addEventListener("keydown",event=>{if(dialog.open)return;if(event.code==="Space"){event.preventDefault();skipPoint();}else if(event.key.toLowerCase()==="u")undo();else if(event.key.toLowerCase()==="x")dialog.showModal();else if(event.key==="ArrowLeft")setFrame(state.frameIndex-1);else if(event.key==="ArrowRight")setFrame(state.frameIndex+1);});render();if(!$("#saveState").textContent.includes("Recovered"))$("#saveState").textContent="Task 88 restored · autosave armed";}
boot().catch(error=>{document.body.innerHTML=`<pre style="padding:24px">${esc(error.stack||error.message)}</pre>`;});
</script>
</body></html>"""


class CourtDiversityHandler(BaseHTTPRequestHandler):
    server_version = "CourtDiversityLabeler/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    @property
    def root(self) -> Path:
        return self.server.repo_root  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        data = HTML.replace(SAVE_TOKEN_PLACEHOLDER, self.server.save_token).encode("utf-8")  # type: ignore[attr-defined]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
        elif parsed.path == "/api/manifest":
            self._send_json(_manifest(self.root, self.server.prefill))  # type: ignore[attr-defined]
        elif parsed.path == "/asset":
            self._serve_asset(parsed.query)
        elif parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        expected = self.server.save_token  # type: ignore[attr-defined]
        supplied = self.headers.get("X-Court-Review-Token", "")
        if not hmac.compare_digest(supplied, expected):
            self._send_json({"error": "invalid save token; reload the page"}, HTTPStatus.UNAUTHORIZED)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_SAVE_BYTES:
                raise ValueError("empty or oversized save payload")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("save payload must be a JSON object")
            result = _write_progress(self.root, payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def _serve_asset(self, query: str) -> None:
        raw = parse_qs(query).get("path", [""])[0]
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(str(candidate))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free local port found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the sequential court-diversity owner label UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--prefill-cvat", type=Path, help="Optional CVAT-for-images ZIP used to prefill saved owner work.")
    parser.add_argument("--labels", type=Path, help="Explicit sequential-label JSON for scoring or CVAT export.")
    parser.add_argument("--export-cvat-dir", type=Path, help="Write strict-ingest-compatible CVAT shard ZIPs and exit.")
    parser.add_argument("--score-only", action="store_true", help="Print the current diagnostic and exit.")
    args = parser.parse_args()
    root = ROOT
    prefill = args.prefill_cvat.expanduser().resolve() if args.prefill_cvat else None
    if prefill and not prefill.is_file():
        parser.error(f"CVAT prefill does not exist: {prefill}")
    labels_path = args.labels.expanduser().resolve() if args.labels else None
    if labels_path and not labels_path.is_file():
        parser.error(f"labels JSON does not exist: {labels_path}")
    explicit_progress = _read_object(labels_path) if labels_path else None
    if args.export_cvat_dir:
        if explicit_progress is None:
            default_labels = root / PROGRESS_PATH
            if not default_labels.is_file():
                parser.error(f"no labels available at {default_labels}; pass --labels")
            labels_path = default_labels
            explicit_progress = _read_object(default_labels)
        report = export_progress_to_cvat(
            root,
            explicit_progress,
            args.export_cvat_dir.expanduser().resolve(),
            source_path=labels_path,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    manifest = _manifest(root, prefill)
    if args.score_only:
        progress = validate_progress(root, explicit_progress) if explicit_progress is not None else manifest["progress"]
        print(json.dumps(score_progress(root, progress), indent=2, sort_keys=True))
        return 0
    port = _free_port(args.port) if args.host in {"127.0.0.1", "localhost"} else args.port
    server = ThreadingHTTPServer((args.host, port), CourtDiversityHandler)
    server.repo_root = root  # type: ignore[attr-defined]
    server.prefill = prefill  # type: ignore[attr-defined]
    server.save_token = secrets.token_urlsafe(32)  # type: ignore[attr-defined]
    print(f"Court label UI: http://{args.host}:{port}", flush=True)
    print(f"Disk autosave: {root / PROGRESS_PATH}", flush=True)
    print(json.dumps(manifest["diagnostic"], indent=2, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
