#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import shutil
import socket
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


REVIEW_ROOT = Path("runs/court_keypoint_review_20260701")
TASKS_ROOT = REVIEW_ROOT / "cvat_tasks"
LABEL_FRAMES_ROOT = REVIEW_ROOT / "label_frames"
PROGRESS_SAVE = REVIEW_ROOT / "local_court_keypoint_review_progress.json"
OUTPUT_ROOT = Path("eval_clips/ball")
MAX_SAVE_BYTES = 5_000_000
MAX_SAVE_ITEMS = 500
MAX_TEXT_CHARS = 512
KEYPOINT_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]
KEYPOINT_INDEX = {name: index for index, name in enumerate(KEYPOINT_NAMES)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json_object(path)
    except Exception:
        return None


def _asset(root: Path, path: Path) -> dict[str, Any]:
    resolved = (root / path).resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return {"exists": False, "path": str(path), "url": None}
    return {
        "exists": resolved.is_file(),
        "path": rel.as_posix(),
        "url": f"/asset?path={rel.as_posix()}" if resolved.is_file() else None,
    }


def _task_entries(root: Path) -> dict[str, dict[str, Any]]:
    tasks_dir = root / TASKS_ROOT
    if not tasks_dir.is_dir():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for task_path in sorted(tasks_dir.glob("*/task.json")):
        task = _read_json_object(task_path)
        clip = _bounded_text(task.get("clip") or task_path.parent.name, field="task.clip", max_chars=160)
        if not clip or "/" in clip or "\\" in clip or clip in {".", ".."}:
            continue
        manifest_path = root / LABEL_FRAMES_ROOT / clip / "label_frame_manifest.json"
        frame_manifest = _read_json_or_none(manifest_path) or {}
        images = _task_images(root, clip, task)
        if not images:
            continue
        source_resolution = _source_resolution(frame_manifest)
        label_coordinate_space = _label_coordinate_space(frame_manifest, source_resolution=source_resolution)
        entries[clip] = {
            "clip": clip,
            "task_path": task_path.relative_to(root),
            "images_dir": (TASKS_ROOT / clip / "images"),
            "task": task,
            "frame_manifest": frame_manifest,
            "images": images,
            "source_resolution": source_resolution,
            "label_coordinate_space": label_coordinate_space,
            "sample_every_frames": _maybe_nonnegative_int(frame_manifest.get("sample_every_frames")),
        }
    return entries


def _task_images(root: Path, clip: str, task: dict[str, Any]) -> list[dict[str, Any]]:
    raw_images = task.get("images")
    if not isinstance(raw_images, list):
        raw_images = []
    images: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_images):
        if not isinstance(item, dict):
            continue
        frame = _bounded_text(item.get("frame") or item.get("file_name"), field=f"images[{index}].frame", max_chars=160)
        file_name = _bounded_text(item.get("file_name") or frame, field=f"images[{index}].file_name", max_chars=160)
        if not frame or "/" in frame or "\\" in frame or frame in seen:
            continue
        image_path = TASKS_ROOT / clip / "images" / file_name
        images.append(
            {
                "frame": frame,
                "file_name": file_name,
                "review_id": _bounded_text(
                    item.get("review_id", f"court_keypoints_manual_15pt_{index:04d}"),
                    field=f"images[{index}].review_id",
                    max_chars=160,
                ),
                "target_file": _bounded_text(item.get("target_file", "court_keypoints.json"), field=f"images[{index}].target_file", max_chars=160),
                "asset": _asset(root, image_path),
                "url": _asset(root, image_path)["url"],
            }
        )
        seen.add(frame)
    return images


def _source_resolution(frame_manifest: dict[str, Any]) -> list[int]:
    raw = frame_manifest.get("source_resolution")
    if isinstance(raw, list) and len(raw) == 2:
        width = _positive_int_or_none(raw[0])
        height = _positive_int_or_none(raw[1])
        if width is not None and height is not None:
            return [width, height]
    return [1920, 1080]


def _label_coordinate_space(frame_manifest: dict[str, Any], *, source_resolution: list[int]) -> list[int]:
    max_width = _positive_int_or_none(frame_manifest.get("max_width"))
    source_width, source_height = source_resolution
    if max_width is not None and source_width > max_width:
        scale = max_width / float(source_width)
        return [max_width, int(round(source_height * scale))]
    return list(source_resolution)


def _manifest(root: Path) -> dict[str, Any]:
    entries = _task_entries(root)
    progress_path = root / PROGRESS_SAVE
    return {
        "schema_version": 1,
        "review_type": "court_keypoint_review",
        "repo_root": str(root),
        "progress_save_path": str(PROGRESS_SAVE),
        "output_root": str(OUTPUT_ROOT),
        "latest_progress": _read_json_or_none(progress_path) if progress_path.is_file() else None,
        "keypoints": [{"name": point.name, "index": index} for index, point in enumerate(PICKLEBALL_KEYPOINTS)],
        "clips": [_clip_manifest(entry) for entry in entries.values()],
    }


def _clip_manifest(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip": entry["clip"],
        "task_path": str(entry["task_path"]),
        "images_dir": str(entry["images_dir"]),
        "source_resolution": entry["source_resolution"],
        "label_coordinate_space": entry["label_coordinate_space"],
        "sample_every_frames": entry["sample_every_frames"],
        "output_label_path": str(OUTPUT_ROOT / entry["clip"] / "labels" / "court_keypoints.json"),
        "images": [
            {
                "frame": image["frame"],
                "file_name": image["file_name"],
                "review_id": image["review_id"],
                "url": image["url"],
                "asset": image["asset"],
            }
            for image in entry["images"]
        ],
    }


def _write_review_progress(root: Path, payload: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    entries = _task_entries(root)
    sanitized = _sanitize_progress_payload(payload, entries=entries)
    sanitized["repo_root"] = str(root)
    sanitized["server_saved_at_utc"] = now.isoformat()

    progress_path = root / PROGRESS_SAVE
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    exported: list[dict[str, str]] = []
    incomplete: dict[str, dict[str, int]] = {}
    for clip, clip_payload in sanitized.get("clips", {}).items():
        entry = entries[clip]
        completion = _clip_completion(entry, clip_payload)
        complete_images = _complete_labeled_images(entry, clip_payload)
        if complete_images:
            label_path = root / OUTPUT_ROOT / clip / "labels" / "court_keypoints.json"
            exported_frame_dir = OUTPUT_ROOT / clip / "labels" / "court_keypoint_frames"
            _copy_export_frames(root, entry, images=complete_images, exported_frame_dir=exported_frame_dir)
            label_payload = _reviewed_label_payload(
                entry,
                clip_payload,
                images=complete_images,
                reviewed_at=now,
                frame_dir=exported_frame_dir,
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(json.dumps(label_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exported.append({"clip": clip, "label_path": _rel(label_path, root)})
        if not completion["complete"]:
            incomplete[clip] = {
                "missing_frame_count": completion["missing_frame_count"],
                "missing_keypoint_count": completion["missing_keypoint_count"],
            }

    return {
        "schema_version": 1,
        "status": "saved" if not incomplete else "saved_partial",
        "progress_path": _rel(progress_path, root),
        "exported_clip_count": len(exported),
        "exported": exported,
        "incomplete": incomplete,
    }


def _complete_labeled_images(entry: dict[str, Any], clip_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_frame = {item["frame"]: item for item in clip_payload.get("items", [])}
    complete: list[dict[str, Any]] = []
    for image in entry["images"]:
        item = by_frame.get(image["frame"])
        if item is None:
            continue
        if set(item.get("keypoints", {})) == set(KEYPOINT_NAMES):
            complete.append(image)
    return complete


def _sanitize_progress_payload(payload: dict[str, Any], *, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unknown = set(payload) - {"schema_version", "review_type", "clips", "saved_from_browser_at", "repo_root", "server_saved_at_utc"}
    if unknown:
        raise ValueError(f"unexpected save fields: {', '.join(sorted(unknown))}")
    review_type = _bounded_text(payload.get("review_type", "court_keypoint_review"), field="review_type", max_chars=80)
    if review_type != "court_keypoint_review":
        raise ValueError(f"unsupported review_type: {review_type}")
    raw_clips = payload.get("clips", {})
    if not isinstance(raw_clips, dict):
        raise ValueError("clips must be a JSON object")
    out: dict[str, Any] = {
        "schema_version": _schema_version(payload.get("schema_version", 1)),
        "review_type": "court_keypoint_review",
        "clips": {},
    }
    if "saved_from_browser_at" in payload:
        out["saved_from_browser_at"] = _bounded_text(payload["saved_from_browser_at"], field="saved_from_browser_at", max_chars=160)
    for clip, raw_clip in raw_clips.items():
        clip_id = _bounded_text(clip, field="clip id", max_chars=160)
        if clip_id not in entries:
            raise ValueError(f"unknown court keypoint review clip: {clip_id}")
        if not isinstance(raw_clip, dict):
            raise ValueError(f"clips.{clip_id} must be a JSON object")
        unknown_clip = set(raw_clip) - {"items", "reviewer"}
        if unknown_clip:
            raise ValueError(f"unexpected fields for clip {clip_id}: {', '.join(sorted(unknown_clip))}")
        out["clips"][clip_id] = {
            "reviewer": _bounded_text(raw_clip.get("reviewer", "local_court_keypoint_review"), field=f"clips.{clip_id}.reviewer", max_chars=160),
            "items": _sanitize_items(raw_clip.get("items", []), entry=entries[clip_id], field=f"clips.{clip_id}.items"),
        }
    return out


def _sanitize_items(value: Any, *, entry: dict[str, Any], field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    if len(value) > MAX_SAVE_ITEMS:
        raise ValueError(f"{field} has too many items")
    allowed_frames = {image["frame"] for image in entry["images"]}
    label_w, label_h = entry["label_coordinate_space"]
    out_by_frame: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be a JSON object")
        unknown = set(item) - {"frame", "review_id", "keypoints", "status"}
        if unknown:
            raise ValueError(f"unexpected fields for {field}[{index}]: {', '.join(sorted(unknown))}")
        frame = _bounded_text(item.get("frame"), field=f"{field}[{index}].frame", max_chars=160)
        if frame not in allowed_frames:
            raise ValueError(f"unknown frame for {field}[{index}]: {frame}")
        raw_keypoints = item.get("keypoints", {})
        if not isinstance(raw_keypoints, dict):
            raise ValueError(f"{field}[{index}].keypoints must be a JSON object")
        unknown_keypoints = set(raw_keypoints) - set(KEYPOINT_NAMES)
        if unknown_keypoints:
            raise ValueError(f"unexpected keypoints for {field}[{index}]: {', '.join(sorted(unknown_keypoints))}")
        keypoints = {
            name: _point(raw_value, field=f"{field}[{index}].keypoints.{name}", max_x=label_w, max_y=label_h)
            for name, raw_value in raw_keypoints.items()
        }
        out_by_frame[frame] = {
            "frame": frame,
            "review_id": _bounded_text(item.get("review_id", ""), field=f"{field}[{index}].review_id", max_chars=160),
            "status": _bounded_text(item.get("status", "in_progress"), field=f"{field}[{index}].status", max_chars=80),
            "keypoints": keypoints,
        }
    return [out_by_frame[frame] for frame in sorted(out_by_frame, key=_frame_sort_key)]


def _clip_completion(entry: dict[str, Any], clip_payload: dict[str, Any]) -> dict[str, Any]:
    by_frame = {item["frame"]: item for item in clip_payload.get("items", [])}
    missing_frame_count = 0
    missing_keypoint_count = 0
    for image in entry["images"]:
        item = by_frame.get(image["frame"])
        if item is None:
            missing_frame_count += 1
            continue
        keypoints = item.get("keypoints", {})
        missing_keypoint_count += len(set(KEYPOINT_NAMES) - set(keypoints))
    return {
        "complete": missing_frame_count == 0 and missing_keypoint_count == 0,
        "missing_frame_count": missing_frame_count,
        "missing_keypoint_count": missing_keypoint_count,
    }


def _copy_export_frames(root: Path, entry: dict[str, Any], *, images: list[dict[str, Any]], exported_frame_dir: Path) -> None:
    out_dir = root / exported_frame_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()
    for image in images:
        source = root / entry["images_dir"] / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"missing review image for export: {source}")
        shutil.copy2(source, out_dir / image["file_name"])


def _reviewed_label_payload(
    entry: dict[str, Any],
    clip_payload: dict[str, Any],
    *,
    images: list[dict[str, Any]],
    reviewed_at: datetime,
    frame_dir: Path,
) -> dict[str, Any]:
    by_frame = {item["frame"]: item for item in clip_payload.get("items", [])}
    items: list[dict[str, Any]] = []
    for image in images:
        item = by_frame[image["frame"]]
        items.append(
            {
                "frame": image["frame"],
                "review_id": item.get("review_id") or image["review_id"],
                "status": "reviewed",
                "keypoints": {
                    name: [float(item["keypoints"][name][0]), float(item["keypoints"][name][1])]
                    for name in KEYPOINT_NAMES
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": entry["clip"],
        "review": {
            "status": "reviewed",
            "reviewer": clip_payload.get("reviewer", "local_court_keypoint_review"),
            "reviewed_at_utc": reviewed_at.isoformat(),
        },
        "frames": {
            "frame_dir": str(frame_dir),
            "frame_count": len(images),
            "available_review_frame_count": len(entry["images"]),
            "source_resolution": entry["source_resolution"],
            "label_coordinate_space": entry["label_coordinate_space"],
            "sample_every_frames": entry["sample_every_frames"],
        },
        "annotation": {"items": items},
    }


def _schema_version(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("schema_version must be an integer")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("schema_version must be an integer") from exc
    if version < 1 or version > 10:
        raise ValueError("schema_version is outside the supported range")
    return version


def _bounded_text(value: Any, *, field: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = value.strip()
    if len(text) > max_chars:
        raise ValueError(f"{field} is too long")
    return text


def _point(value: Any, *, field: str, max_x: int, max_y: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item image coordinate")
    x = _finite_nonnegative(value[0], field=f"{field}[0]")
    y = _finite_nonnegative(value[1], field=f"{field}[1]")
    if x > max_x or y > max_y:
        raise ValueError(f"{field} is outside the label image bounds")
    return [x, y]


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _maybe_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _frame_sort_key(frame: str) -> tuple[int, str]:
    stem = Path(frame).stem
    try:
        return int(stem.rsplit("_", 1)[1]), frame
    except (IndexError, ValueError):
        return 0, frame


def _copy_file_to_writer(path: Path, writer: Any, *, chunk_size: int = 1024 * 1024) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            try:
                writer.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Court Keypoint Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #171918;
      --panel: #232725;
      --panel-2: #2f3431;
      --ink: #f4f0e8;
      --muted: #aaa69b;
      --line: #4a504b;
      --accent: #e9c46a;
      --accent-2: #66d9c6;
      --danger: #ff6b6b;
      --ok: #8bd17c;
      --shadow: rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, select {
      font: inherit;
      color: var(--ink);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 38px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.active { background: var(--accent); color: #171918; border-color: var(--accent); }
    button.complete { border-color: var(--ok); }
    button.missing { border-color: var(--danger); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .app {
      display: grid;
      grid-template-columns: 280px minmax(420px, 1fr) 340px;
      min-height: 100vh;
    }
    .side, .rail {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 16px;
      overflow: auto;
    }
    .rail {
      border-left: 1px solid var(--line);
      border-right: 0;
    }
    .brand {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 760;
    }
    .tag {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .clip-list, .frame-list, .keypoint-list {
      display: grid;
      gap: 8px;
    }
    .clip-btn, .frame-btn, .keypoint-btn {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 10px;
      text-align: left;
    }
    .name {
      overflow-wrap: anywhere;
      line-height: 1.2;
      font-size: 13px;
    }
    .count {
      color: var(--muted);
      font-size: 12px;
    }
    .stage-wrap {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 100vh;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #1d201e;
    }
    .toolbar-group {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
    }
    .status strong { color: var(--ink); }
    #imageStage {
      position: relative;
      display: grid;
      place-items: center;
      padding: 16px;
      min-height: 0;
      overflow: auto;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        #121413;
      background-size: 24px 24px;
    }
    .image-box {
      position: relative;
      max-width: min(100%, 1280px);
      width: 100%;
      box-shadow: 0 22px 70px var(--shadow);
      border: 1px solid var(--line);
      background: #080908;
    }
    #frameImage {
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
    }
    #overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .footer {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      background: #1d201e;
      color: var(--muted);
      font-size: 13px;
    }
    .current {
      padding: 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 12px;
    }
    .current-title {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .current-name {
      font-size: 18px;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    .save-btn {
      grid-column: 1 / -1;
      background: var(--accent-2);
      color: #101312;
      border-color: var(--accent-2);
      font-weight: 720;
    }
    .message {
      min-height: 34px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.3;
      margin-bottom: 12px;
    }
    .progress {
      height: 8px;
      border-radius: 999px;
      background: #101312;
      overflow: hidden;
      border: 1px solid var(--line);
      margin: 8px 0 14px;
    }
    .progress span {
      display: block;
      height: 100%;
      width: 0;
      background: var(--ok);
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .side, .rail { border: 0; border-bottom: 1px solid var(--line); max-height: 38vh; }
      .stage-wrap { min-height: 62vh; }
    }
  </style>
</head>
<body>
  <div class="app" id="courtKeypointLabeler">
    <aside class="side">
      <div class="brand">
        <h1>Court Keypoints</h1>
        <div class="tag" id="totalProgress">0/0</div>
      </div>
      <div class="progress"><span id="progressBar"></span></div>
      <div class="clip-list" id="clipList"></div>
    </aside>
    <main class="stage-wrap">
      <div class="toolbar">
        <div class="toolbar-group">
          <button id="prevFrame">Prev frame</button>
          <button id="nextFrame">Next frame</button>
          <button id="copyPrev">Copy previous</button>
        </div>
        <div class="status" id="frameStatus"></div>
      </div>
      <section id="imageStage">
        <div class="image-box">
          <img id="frameImage" alt="">
          <svg id="overlay" viewBox="0 0 1280 720" preserveAspectRatio="none"></svg>
        </div>
      </section>
      <div class="footer">
        <span id="clipPath"></span>
        <span id="coordStatus"></span>
      </div>
    </main>
    <aside class="rail">
      <div class="current">
        <div class="current-title">Active point</div>
        <div class="current-name" id="activePoint">-</div>
      </div>
      <div class="actions">
        <button id="prevPoint">Prev point</button>
        <button id="nextPoint">Next point</button>
        <button id="clearPoint">Clear point</button>
        <button id="clearFrame">Clear frame</button>
        <button class="save-btn" id="saveProgress">Save progress</button>
      </div>
      <div class="message" id="message"></div>
      <div class="keypoint-list" id="keypointRail"></div>
    </aside>
  </div>
  <script>
    const LINES = [
      ["near_left_corner", "near_baseline_center"], ["near_baseline_center", "near_right_corner"],
      ["near_right_corner", "net_right_sideline"], ["net_right_sideline", "far_right_corner"],
      ["far_right_corner", "far_baseline_center"], ["far_baseline_center", "far_left_corner"],
      ["far_left_corner", "net_left_sideline"], ["net_left_sideline", "near_left_corner"],
      ["near_nvz_left", "near_nvz_center"], ["near_nvz_center", "near_nvz_right"],
      ["far_nvz_left", "far_nvz_center"], ["far_nvz_center", "far_nvz_right"],
      ["net_left_sideline", "net_center"], ["net_center", "net_right_sideline"]
    ];
    const state = { manifest: null, clipIndex: 0, frameIndex: 0, keypointIndex: 0, data: { schema_version: 1, review_type: "court_keypoint_review", clips: {} } };
    const qs = (s) => document.querySelector(s);
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

    function clips() { return state.manifest?.clips || []; }
    function keypoints() { return state.manifest?.keypoints || []; }
    function clip() { return clips()[state.clipIndex]; }
    function frame() { return clip()?.images[state.frameIndex]; }
    function clipBucket(c = clip()) {
      if (!c) return null;
      if (!state.data.clips[c.clip]) state.data.clips[c.clip] = { reviewer: "local_court_keypoint_review", items: [] };
      return state.data.clips[c.clip];
    }
    function itemFor(c = clip(), f = frame()) {
      const bucket = clipBucket(c);
      if (!bucket || !f) return null;
      let item = bucket.items.find(x => x.frame === f.frame);
      if (!item) {
        item = { frame: f.frame, review_id: f.review_id, status: "in_progress", keypoints: {} };
        bucket.items.push(item);
      }
      return item;
    }
    function existingItem(c, f) {
      return state.data.clips[c.clip]?.items?.find(x => x.frame === f.frame) || null;
    }
    function loadProgress(progress) {
      if (progress && progress.review_type === "court_keypoint_review" && progress.clips) {
        state.data.clips = progress.clips;
      }
    }
    function pointCount(c = clip(), f = frame()) {
      const item = c && f ? existingItem(c, f) : null;
      return item ? Object.keys(item.keypoints || {}).length : 0;
    }
    function clipPointCount(c) {
      return c.images.reduce((sum, image) => sum + pointCount(c, image), 0);
    }
    function totalPointCount() {
      return clips().reduce((sum, c) => sum + clipPointCount(c), 0);
    }
    function totalRequired() {
      return clips().reduce((sum, c) => sum + c.images.length * keypoints().length, 0);
    }
    function completeFrame(c, f) {
      return pointCount(c, f) === keypoints().length;
    }
    function completeClip(c) {
      return c.images.every(image => completeFrame(c, image));
    }
    function setMessage(text) {
      qs("#message").textContent = text || "";
    }
    function render() {
      if (!clip()) return;
      renderClips();
      renderKeypoints();
      renderImage();
      renderStatus();
    }
    function renderClips() {
      qs("#clipList").innerHTML = clips().map((c, i) => {
        const points = clipPointCount(c);
        const required = c.images.length * keypoints().length;
        const cls = completeClip(c) ? "complete" : points ? "" : "missing";
        return `<button class="clip-btn ${i === state.clipIndex ? "active" : ""} ${cls}" data-clip="${i}">
          <span class="name">${esc(c.clip)}</span><span class="count">${points}/${required}</span>
        </button>`;
      }).join("");
      qs("#clipList").querySelectorAll("[data-clip]").forEach(btn => {
        btn.onclick = () => { state.clipIndex = Number(btn.dataset.clip); state.frameIndex = 0; state.keypointIndex = 0; render(); };
      });
    }
    function renderKeypoints() {
      const item = itemFor();
      const active = keypoints()[state.keypointIndex]?.name || "";
      qs("#activePoint").textContent = active || "-";
      qs("#keypointRail").innerHTML = keypoints().map((kp, i) => {
        const hasPoint = !!item?.keypoints?.[kp.name];
        return `<button class="keypoint-btn ${i === state.keypointIndex ? "active" : ""} ${hasPoint ? "complete" : "missing"}" data-kp="${i}">
          <span class="name">${esc(kp.name)}</span><span class="count">${hasPoint ? "set" : "open"}</span>
        </button>`;
      }).join("");
      qs("#keypointRail").querySelectorAll("[data-kp]").forEach(btn => {
        btn.onclick = () => { state.keypointIndex = Number(btn.dataset.kp); render(); };
      });
    }
    function renderImage() {
      const img = qs("#frameImage");
      const f = frame();
      if (!f) return;
      if (!img.src.endsWith(f.url || "")) img.src = f.url;
      img.onload = drawOverlay;
      qs("#clipPath").textContent = `${clip().images_dir}/${f.file_name}`;
      drawOverlay();
    }
    function renderStatus() {
      const points = totalPointCount();
      const required = totalRequired();
      qs("#totalProgress").textContent = `${points}/${required}`;
      qs("#progressBar").style.width = required ? `${Math.round(points * 100 / required)}%` : "0";
      qs("#frameStatus").innerHTML = `<strong>${esc(clip().clip)}</strong> / ${esc(frame().frame)} / ${pointCount()}/${keypoints().length}`;
      qs("#coordStatus").textContent = `${clip().label_coordinate_space[0]}x${clip().label_coordinate_space[1]} label pixels`;
      qs("#prevFrame").disabled = state.frameIndex === 0;
      qs("#nextFrame").disabled = state.frameIndex >= clip().images.length - 1;
      qs("#prevPoint").disabled = state.keypointIndex === 0;
      qs("#nextPoint").disabled = state.keypointIndex >= keypoints().length - 1;
    }
    function drawOverlay() {
      const svg = qs("#overlay");
      const img = qs("#frameImage");
      const c = clip();
      const item = itemFor();
      if (!c || !item || !img.naturalWidth) return;
      svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
      const pts = item.keypoints || {};
      const lines = LINES.filter(([a, b]) => pts[a] && pts[b]).map(([a, b]) =>
        `<line x1="${pts[a][0]}" y1="${pts[a][1]}" x2="${pts[b][0]}" y2="${pts[b][1]}" stroke="rgba(102,217,198,.9)" stroke-width="2" vector-effect="non-scaling-stroke" />`
      ).join("");
      const circles = keypoints().map((kp, i) => {
        const p = pts[kp.name];
        if (!p) return "";
        const active = i === state.keypointIndex;
        return `<g>
          <circle cx="${p[0]}" cy="${p[1]}" r="${active ? 8 : 6}" fill="${active ? "#e9c46a" : "#66d9c6"}" stroke="#101312" stroke-width="2" vector-effect="non-scaling-stroke" />
          <text x="${p[0] + 9}" y="${p[1] - 9}" fill="#f4f0e8" stroke="#101312" stroke-width="3" paint-order="stroke" font-size="12">${i + 1}</text>
        </g>`;
      }).join("");
      svg.innerHTML = lines + circles;
    }
    function placePoint(evt) {
      const img = qs("#frameImage");
      if (!img.naturalWidth) return;
      const rect = img.getBoundingClientRect();
      const x = (evt.clientX - rect.left) * img.naturalWidth / rect.width;
      const y = (evt.clientY - rect.top) * img.naturalHeight / rect.height;
      const kp = keypoints()[state.keypointIndex]?.name;
      if (!kp) return;
      const item = itemFor();
      item.keypoints[kp] = [Math.max(0, Math.min(img.naturalWidth, x)), Math.max(0, Math.min(img.naturalHeight, y))];
      item.status = completeFrame(clip(), frame()) ? "reviewed" : "in_progress";
      if (state.keypointIndex < keypoints().length - 1) state.keypointIndex += 1;
      render();
    }
    function clearPoint() {
      const kp = keypoints()[state.keypointIndex]?.name;
      const item = itemFor();
      if (kp && item?.keypoints) delete item.keypoints[kp];
      render();
    }
    function clearFrame() {
      const item = itemFor();
      item.keypoints = {};
      item.status = "in_progress";
      render();
    }
    function copyPrevious() {
      if (state.frameIndex === 0) return;
      const prev = existingItem(clip(), clip().images[state.frameIndex - 1]);
      if (!prev?.keypoints) return;
      const item = itemFor();
      item.keypoints = JSON.parse(JSON.stringify(prev.keypoints));
      item.status = completeFrame(clip(), frame()) ? "reviewed" : "in_progress";
      render();
    }
    async function save() {
      setMessage("Saving...");
      const payload = JSON.parse(JSON.stringify(state.data));
      payload.saved_from_browser_at = new Date().toISOString();
      const res = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const body = await res.json();
      if (!res.ok) {
        setMessage(body.error || "Save failed");
        return;
      }
      const exported = body.exported_clip_count || 0;
      const partial = Object.keys(body.incomplete || {}).length;
      setMessage(`Saved. Exported ${exported} complete clip${exported === 1 ? "" : "s"}. ${partial ? partial + " clip(s) still incomplete." : ""}`);
    }
    async function boot() {
      const res = await fetch("/api/manifest");
      state.manifest = await res.json();
      loadProgress(state.manifest.latest_progress);
      if (!clips().length) {
        document.body.innerHTML = "<pre style='padding:20px'>No court keypoint review tasks found.</pre>";
        return;
      }
      qs("#overlay").addEventListener("click", placePoint);
      qs("#prevFrame").onclick = () => { if (state.frameIndex > 0) state.frameIndex--; render(); };
      qs("#nextFrame").onclick = () => { if (state.frameIndex < clip().images.length - 1) state.frameIndex++; render(); };
      qs("#prevPoint").onclick = () => { if (state.keypointIndex > 0) state.keypointIndex--; render(); };
      qs("#nextPoint").onclick = () => { if (state.keypointIndex < keypoints().length - 1) state.keypointIndex++; render(); };
      qs("#clearPoint").onclick = clearPoint;
      qs("#clearFrame").onclick = clearFrame;
      qs("#copyPrev").onclick = copyPrevious;
      qs("#saveProgress").onclick = save;
      document.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight") { if (state.keypointIndex < keypoints().length - 1) state.keypointIndex++; render(); }
        if (e.key === "ArrowLeft") { if (state.keypointIndex > 0) state.keypointIndex--; render(); }
        if (e.key === "ArrowDown") { if (state.frameIndex < clip().images.length - 1) state.frameIndex++; render(); }
        if (e.key === "ArrowUp") { if (state.frameIndex > 0) state.frameIndex--; render(); }
      });
      render();
    }
    boot().catch(err => { document.body.innerHTML = "<pre style='padding:20px'>" + esc(err.stack || err.message) + "</pre>"; });
  </script>
</body>
</html>
"""


class CourtKeypointReviewHandler(BaseHTTPRequestHandler):
    server_version = "CourtKeypointReview/1.0"

    @property
    def root(self) -> Path:
        return self.server.repo_root  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/manifest":
            self._send_json(_manifest(self.root))
            return
        if parsed.path == "/asset":
            self._serve_asset(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("request body is empty")
            if length > MAX_SAVE_BYTES:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            summary = _write_review_progress(self.root, payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(summary)

    def _serve_asset(self, query: str) -> None:
        params = parse_qs(query)
        raw = params.get("path", [""])[0]
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "path outside repo")
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        size = candidate.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        _copy_file_to_writer(candidate, self.wfile)


def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in range {preferred}-{preferred + 49}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local UI for reviewed 15-point court keypoint labels.")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    root = _repo_root()
    port = _free_port(args.port) if args.host in {"127.0.0.1", "localhost"} else args.port
    server = ThreadingHTTPServer((args.host, port), CourtKeypointReviewHandler)
    server.repo_root = root  # type: ignore[attr-defined]
    print(f"Serving court keypoint review UI at http://{args.host}:{port}")
    print(f"Repo root: {root}")
    print(f"Progress path: {root / PROGRESS_SAVE}")
    print(f"Final label root: {root / OUTPUT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping court keypoint review UI.")
    finally:
        threading.Thread(target=server.server_close).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
