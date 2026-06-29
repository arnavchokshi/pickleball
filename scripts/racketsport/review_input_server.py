#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import socket
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


CLIPS = [
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
]

RUN_ROOT = Path("runs/eval0/prototype_gate_h100_v2")
SAVE_DIR = Path("runs/review_inputs")
LATEST_SAVE = SAVE_DIR / "pickleball_cv_review_latest.json"
ACTION_MANIFEST = Path("runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review_actions.json")
MAX_SAVE_BYTES = 2_000_000
MAX_TEXT_CHARS = 4000
MAX_SAVE_LIST_ITEMS = 200
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
REQUIRED_LABEL_FILES = (
    "court_corners.json",
    "players.json",
    "feet_nvz.json",
    "ball.json",
    "events.json",
    "racket_pose.json",
    "foot_contact.json",
    "coach_habits.json",
    "manual_metrics.json",
)
REVIEWED_LABEL_STATUSES = {"accepted", "ground_truth", "human_reviewed", "reviewed", "verified"}
REQUIRED_COURT_LINE_IDS = ("near_nvz", "far_nvz", "near_centerline", "far_centerline")
REQUIRED_NET_IDS = ("top_net",)
COURT_EVIDENCE_IDS = (*REQUIRED_COURT_LINE_IDS, *REQUIRED_NET_IDS)
COURT_POINT_IDS = {f"{evidence_id}:{endpoint}" for evidence_id in COURT_EVIDENCE_IDS for endpoint in ("a", "b")}
PADDLE_CROP_TILE_SIZE = 180
TRACKING_REVIEW_ROOT = Path("runs/phase2/person_coverage_actual_source30_h100_yolo26m_fulltb3_20260628")
TRACKING_REVIEW_ITEMS = (
    {
        "clip": "outdoor_webcam_iynbd_1500_long_high_baseline",
        "title": "Outdoor widened-margin player tracks",
        "question": "Do the highlighted boxes stay on the four real players, or do they include spectators/background people?",
        "needed_answer": "Choose whether this is safe to promote to canonical player tracks.",
        "overlay_path": TRACKING_REVIEW_ROOT
        / "outdoor_webcam_iynbd_1500_long_high_baseline"
        / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock_margin2_diagnostic_overlay.mp4",
        "baseline_overlay_path": TRACKING_REVIEW_ROOT
        / "outdoor_webcam_iynbd_1500_long_high_baseline"
        / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock"
        / "track_overlay.mp4",
        "montage_path": TRACKING_REVIEW_ROOT
        / "montages"
        / "outdoor_webcam_iynbd_1500_long_high_baseline_yolo26m_margin0_vs_margin2_diagnostic_montage.jpg",
    },
    {
        "clip": "indoor_doubles_fwuks_0500_long_mid_baseline",
        "title": "Indoor widened-margin player tracks",
        "question": "Do the highlighted boxes stay on the four real players, or do they include spectators/background people?",
        "needed_answer": "Choose whether this is safe to promote to canonical player tracks.",
        "overlay_path": TRACKING_REVIEW_ROOT
        / "indoor_doubles_fwuks_0500_long_mid_baseline"
        / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock_margin4_diagnostic_overlay.mp4",
        "baseline_overlay_path": TRACKING_REVIEW_ROOT
        / "indoor_doubles_fwuks_0500_long_mid_baseline"
        / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock"
        / "track_overlay.mp4",
        "montage_path": TRACKING_REVIEW_ROOT
        / "montages"
        / "indoor_doubles_fwuks_0500_long_mid_baseline_yolo26m_margin0_vs_margin4_diagnostic_montage.jpg",
    },
)
COURT_REVIEW_RETIREMENTS = {
    "burlington_gold_0300_low_steep_corner": {
        "reason": "Retired for court review because fisheye distortion makes straight court lines appear curved.",
        "allowed_use": "player identity, ball review, non-court QA",
    }
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _asset(root: Path, path: Path) -> dict[str, Any]:
    resolved = (root / path).resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return {"exists": False, "path": str(path), "url": None}
    return {
        "exists": resolved.is_file(),
        "path": str(rel),
        "url": f"/asset?path={rel.as_posix()}" if resolved.is_file() else None,
    }


def _copy_file_to_writer(path: Path, writer: Any, *, chunk_size: int = 1024 * 1024) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            try:
                writer.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return


def _copy_file_range_to_writer(path: Path, writer: Any, start: int, end: int, *, chunk_size: int = 1024 * 1024) -> None:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            try:
                writer.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return


def _parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or size <= 0:
        return None
    if not header.startswith("bytes="):
        return None
    spec = header.removeprefix("bytes=").strip()
    if "," in spec or "-" not in spec:
        return None
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= size:
        return None
    return start, min(end, size - 1)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _label_summary(root: Path, clip: str) -> dict[str, Any]:
    labels_dir = RUN_ROOT / clip / "labels"
    out: dict[str, Any] = {
        "labels_dir": str(labels_dir),
        "files": {},
        "frame_dir": None,
        "frame_count": None,
        "sample_every_frames": None,
        "source_fps": None,
        "source_duration_s": None,
    }
    players = _read_json(root / labels_dir / "players.json")
    if isinstance(players, dict):
        frames = players.get("frames")
        if isinstance(frames, dict):
            out["frame_dir"] = frames.get("frame_dir")
            out["frame_count"] = frames.get("frame_count")
            out["sample_every_frames"] = frames.get("sample_every_frames")
            out["source_fps"] = frames.get("source_fps")
            out["source_duration_s"] = frames.get("source_duration_s")

    for name in (*REQUIRED_LABEL_FILES, "status.json"):
        path = root / labels_dir / name
        payload = _read_json(path)
        summary: dict[str, Any] = {"exists": path.is_file()}
        if isinstance(payload, dict):
            summary.update(
                {
                    "status": payload.get("status"),
                    "not_ground_truth": payload.get("not_ground_truth"),
                    "confidence": payload.get("confidence"),
                    "source": payload.get("source"),
                }
            )
            annotation = payload.get("annotation")
            if isinstance(annotation, dict) and isinstance(annotation.get("items"), list):
                summary["item_count"] = len(annotation["items"])
        out["files"][name] = summary
    return out


def _label_state(summary: dict[str, Any]) -> str:
    if not summary.get("exists"):
        return "missing"
    if summary.get("not_ground_truth") is True:
        return "draft_not_ground_truth"
    status = str(summary.get("status") or "").strip().lower()
    if status in REVIEWED_LABEL_STATUSES:
        return "reviewed"
    if "draft" in status or "unverified" in status:
        return "draft_not_ground_truth"
    if status:
        return status
    return "present_unreviewed"


def _label_intake_requirements(root: Path, clip: str) -> dict[str, dict[str, Any]]:
    label_summary = _label_summary(root, clip)
    files = label_summary.get("files", {})
    out: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_LABEL_FILES:
        summary = dict(files.get(name, {"exists": False}))
        summary["state"] = _label_state(summary)
        out[name] = summary
    return out


def _court_evidence_requirements(root: Path, clip: str) -> dict[str, Any]:
    artifact = RUN_ROOT / clip / "court_line_evidence.json"
    payload = _read_json(root / artifact)
    aggregate = payload.get("aggregate") if isinstance(payload, dict) else None
    if not isinstance(aggregate, dict):
        aggregate = {}
    missing_line_ids = _string_list(aggregate.get("missing_required_line_ids"))
    missing_net_ids = _string_list(aggregate.get("missing_required_net_ids"))
    auto_ready = aggregate.get("auto_calibration_ready") is True
    return {
        "artifact_path": str(artifact),
        "exists": (root / artifact).is_file(),
        "auto_calibration_ready": auto_ready,
        "accepted_line_ids": _string_list(aggregate.get("accepted_line_ids")),
        "missing_required_line_ids": missing_line_ids,
        "missing_required_net_ids": missing_net_ids,
        "required_line_ids": list(REQUIRED_COURT_LINE_IDS),
        "required_net_ids": list(REQUIRED_NET_IDS),
        "needs_human_review": not auto_ready or bool(missing_line_ids) or bool(missing_net_ids),
        "reasons": _string_list(aggregate.get("reasons")),
    }


def _summary_count(payload: Any, field: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get(field), int):
        return int(summary[field])
    return None


def _contact_requirements(root: Path, clip: str) -> dict[str, Any]:
    clip_root = RUN_ROOT / clip
    contact_windows_path = clip_root / "contact_windows.json"
    candidates_path = clip_root / "contact_window_candidates.json"
    review_path = clip_root / "contact_window_review.json"
    contact_windows = _read_json(root / contact_windows_path)
    candidates = _read_json(root / candidates_path)
    review = _read_json(root / review_path)

    events = contact_windows.get("events") if isinstance(contact_windows, dict) else None
    candidate_items = candidates.get("candidates") if isinstance(candidates, dict) else None
    decisions = review.get("decisions") if isinstance(review, dict) else None
    accepted_count = _summary_count(review, "accepted_count")
    pending_count = _summary_count(review, "pending_count")
    candidate_count = _summary_count(candidates, "candidate_count")

    canonical_count = len(events) if isinstance(events, list) else 0
    if candidate_count is None:
        candidate_count = len(candidate_items) if isinstance(candidate_items, list) else 0
    if accepted_count is None:
        accepted_count = (
            sum(1 for item in decisions if isinstance(item, dict) and item.get("decision") == "accepted")
            if isinstance(decisions, list)
            else 0
        )
    if pending_count is None:
        pending_count = (
            sum(1 for item in decisions if isinstance(item, dict) and item.get("decision") == "pending")
            if isinstance(decisions, list)
            else 0
        )

    return {
        "contact_windows_path": str(contact_windows_path),
        "contact_windows_exists": (root / contact_windows_path).is_file(),
        "candidate_path": str(candidates_path),
        "candidate_exists": (root / candidates_path).is_file(),
        "review_path": str(review_path),
        "review_exists": (root / review_path).is_file(),
        "canonical_contact_count": canonical_count,
        "candidate_count": candidate_count,
        "accepted_review_count": accepted_count,
        "pending_review_count": pending_count,
        "review_status": str(review.get("status", "")) if isinstance(review, dict) else "",
        "needs_human_reviewed_contacts": canonical_count == 0,
    }


def _clip_intake_requirements(root: Path, clip: str, review_actions: dict[str, Any]) -> dict[str, Any]:
    return {
        "court_evidence": _court_evidence_requirements(root, clip),
        "court_review_policy": _court_review_policy(clip),
        "contacts": _contact_requirements(root, clip),
        "labels": _label_intake_requirements(root, clip),
        "review_action_blockers": review_actions.get("blockers", []),
    }


def _court_review_policy(clip: str) -> dict[str, Any]:
    retirement = COURT_REVIEW_RETIREMENTS.get(clip)
    if not retirement:
        return {
            "court_review_enabled": True,
            "retired_for_court": False,
            "reason": "",
            "allowed_use": "court calibration, player identity, ball review",
        }
    return {
        "court_review_enabled": False,
        "retired_for_court": True,
        "reason": retirement["reason"],
        "allowed_use": retirement["allowed_use"],
    }


def _review_actions_by_clip(root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(root / ACTION_MANIFEST)
    if not isinstance(payload, dict):
        return {}
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        return {}

    by_clip: dict[str, dict[str, Any]] = {}
    packet_id = str(payload.get("packet_id", ""))
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        clip = str(raw_action.get("clip", ""))
        if not clip:
            continue
        action = _review_action_summary(raw_action)
        bucket = by_clip.setdefault(clip, _empty_review_actions(packet_id=packet_id))
        bucket["actions"].append(action)

    for bucket in by_clip.values():
        actions = bucket["actions"]
        actions.sort(key=lambda item: (PRIORITY_RANK.get(item["priority"], 9), item["category"], item["title"]))
        bucket["action_count"] = len(actions)
        bucket["high_count"] = sum(1 for action in actions if action["priority"] == "high")
        bucket["medium_count"] = sum(1 for action in actions if action["priority"] == "medium")
        categories: dict[str, int] = {}
        blockers: list[str] = []
        for action in actions:
            categories[action["category"]] = categories.get(action["category"], 0) + 1
            for blocker in action["blockers"]:
                if blocker not in blockers:
                    blockers.append(blocker)
        bucket["categories"] = dict(sorted(categories.items()))
        bucket["blockers"] = blockers
    return by_clip


def _empty_review_actions(*, packet_id: str = "") -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "manifest_path": str(ACTION_MANIFEST),
        "action_count": 0,
        "high_count": 0,
        "medium_count": 0,
        "categories": {},
        "blockers": [],
        "actions": [],
    }


def _review_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": str(action.get("category", "")),
        "priority": str(action.get("priority", "")),
        "title": str(action.get("title", "")),
        "artifact_path": str(action.get("artifact_path", "")),
        "watch_paths": _string_list(action.get("watch_paths")),
        "editable_paths": _string_list(action.get("editable_paths")),
        "details": _string_list(action.get("details")),
        "blockers": _string_list(action.get("blockers")),
        "next_commands": _string_list(action.get("next_commands")),
        "suggested_action": str(action.get("suggested_action", "")),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _visual_path(payload: dict[str, Any], visual_type: str) -> Path | None:
    visuals = payload.get("visuals")
    if not isinstance(visuals, list):
        return None
    for visual in visuals:
        if isinstance(visual, dict) and visual.get("type") == visual_type and visual.get("path"):
            return Path(str(visual["path"]))
    return None


def _paddle_corner_review(root: Path, clip: str) -> dict[str, Any]:
    artifact_path = RUN_ROOT / clip / "paddle_true_corner_review.json"
    payload = _read_json(root / artifact_path)
    if not isinstance(payload, dict):
        return {
            "exists": False,
            "artifact": _asset(root, artifact_path),
            "status": "missing_paddle_true_corner_review",
            "queue_count": 0,
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "task_items": [],
            "crop_sheet": _asset(root, RUN_ROOT / clip / "racket_candidates" / "paddle_true_corner_crop_sheet.png"),
            "candidate_overlay": _asset(root, RUN_ROOT / clip / "racket_candidates" / "racket_candidate_overlay_h264.mp4"),
            "source_video": _asset(root, RUN_ROOT / clip / "racket_candidates" / "source_0000_0030.mp4"),
            "label_artifact_type": "racketsport_paddle_true_corner_labels",
        }

    crop_summary = payload.get("crop_sheet_summary") if isinstance(payload.get("crop_sheet_summary"), dict) else {}
    crop_sheet_path = (
        Path(str(crop_summary.get("crop_sheet_path")))
        if crop_summary.get("crop_sheet_path")
        else _visual_path(payload, "true_corner_label_crop_sheet")
    )
    overlay_path = _visual_path(payload, "candidate_overlay_video")
    source_video_path = Path(str(crop_summary.get("video_path"))) if crop_summary.get("video_path") else None
    required_labels = payload.get("required_labels") if isinstance(payload.get("required_labels"), list) else []
    rendered_count = int(crop_summary.get("rendered_crop_count") or min(len(required_labels), 48))
    queue_count = min(len(required_labels), max(0, rendered_count))
    cols = min(4, max(1, queue_count))
    corner_order = _paddle_corner_order(required_labels)

    task_items: list[dict[str, Any]] = []
    for index, item in enumerate(required_labels[:queue_count]):
        if not isinstance(item, dict):
            continue
        row = index // cols
        col = index % cols
        task_items.append(
            {
                "review_id": str(item.get("review_id", f"item_{index:04d}")),
                "tile_index": index,
                "sheet_rect": [
                    col * PADDLE_CROP_TILE_SIZE,
                    row * PADDLE_CROP_TILE_SIZE,
                    (col + 1) * PADDLE_CROP_TILE_SIZE,
                    (row + 1) * PADDLE_CROP_TILE_SIZE,
                ],
                "crop_xyxy": _numeric_list(item.get("crop_xyxy")),
                "frame_index": item.get("frame_index"),
                "t": item.get("t"),
                "player_id": item.get("player_id"),
                "candidate_conf": item.get("candidate_conf"),
                "candidate_source": item.get("candidate_source"),
                "corner_order": corner_order,
            }
        )

    return {
        "exists": True,
        "artifact": _asset(root, artifact_path),
        "status": str(payload.get("status", "")),
        "trusted_for_rkt_promotion": payload.get("trusted_for_rkt_promotion") is True,
        "required_label_count": int(payload.get("required_label_count") or len(required_labels)),
        "listed_required_label_count": int(payload.get("listed_required_label_count") or len(required_labels)),
        "true_corner_label_count": int(payload.get("true_corner_label_count") or 0),
        "reference_gt_count": int(payload.get("reference_gt_count") or 0),
        "promotion_blockers": _string_list(payload.get("promotion_blockers")),
        "crop_sheet": _asset(root, crop_sheet_path or RUN_ROOT / clip / "racket_candidates" / "paddle_true_corner_crop_sheet.png"),
        "candidate_overlay": _asset(root, overlay_path or RUN_ROOT / clip / "racket_candidates" / "racket_candidate_overlay_h264.mp4"),
        "source_video": _asset(root, source_video_path or RUN_ROOT / clip / "racket_candidates" / "source_0000_0030.mp4"),
        "tile_size": PADDLE_CROP_TILE_SIZE,
        "queue_count": len(task_items),
        "corner_order": corner_order,
        "task_items": task_items,
        "label_artifact_type": "racketsport_paddle_true_corner_labels",
    }


def _paddle_corner_order(required_labels: list[Any]) -> list[str]:
    for item in required_labels:
        if not isinstance(item, dict):
            continue
        output = item.get("required_output")
        if isinstance(output, dict):
            order = output.get("corners_px_order")
            if isinstance(order, list) and order:
                return [str(value) for value in order]
    return ["top_left", "top_right", "bottom_right", "bottom_left"]


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _human_review_tasks(root: Path) -> dict[str, Any]:
    return {
        "paddle_corner_review_url": "/paddle",
        "tracking_video_review": _tracking_video_review(root),
    }


def _tracking_video_review(root: Path) -> dict[str, Any]:
    return {
        "title": "Player tracking promotion review",
        "question": "Do the widened-margin candidate videos keep boxes on the four real players without catching spectators/background?",
        "needed_answer": "For each video, choose safe_to_promote, unsafe_background_or_spectators, or unsure.",
        "choices": ["safe_to_promote", "unsafe_background_or_spectators", "unsure"],
        "items": [
            {
                "clip": str(item["clip"]),
                "title": str(item["title"]),
                "question": str(item["question"]),
                "needed_answer": str(item["needed_answer"]),
                "overlay_video": _asset(root, item["overlay_path"]),
                "baseline_overlay_video": _asset(root, item["baseline_overlay_path"]),
                "montage": _asset(root, item["montage_path"]),
            }
            for item in TRACKING_REVIEW_ITEMS
        ],
    }


def _clip_manifest(root: Path, clip: str, *, review_actions: dict[str, Any] | None = None) -> dict[str, Any]:
    clip_root = RUN_ROOT / clip
    tracknet = clip_root / "tracknet_smoke_0000_0010"
    high_detail_candidates = [
        RUN_ROOT
        / "high_detail_multiclip_20260628T052440Z"
        / clip
        / "trk1_yolo26m_botsort_reid"
        / "track_overlay_0000_0900.mp4",
        RUN_ROOT
        / "high_detail_multiclip_window_20260628T053509Z"
        / clip
        / "trk1_yolo26m_botsort_reid"
        / "track_overlay_0000_0300.mp4",
    ]
    track_overlay = next((p for p in high_detail_candidates if (root / p).is_file()), high_detail_candidates[0])
    resolved_review_actions = review_actions or _empty_review_actions()
    return {
        "id": clip,
        "label_overlay": _asset(root, clip_root / "compare" / "all_labels_overlay_h264.mp4"),
        "calibration_overlay": _asset(root, clip_root / "compare" / "calibration_overlay_h264.mp4"),
        "ball_overlay": _asset(root, tracknet / "ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"),
        "ball_track_json": _asset(root, tracknet / "ball_track_fusion_temporal_vball100_localtraj.json"),
        "track_overlay": _asset(root, track_overlay),
        "source_video": _asset(root, Path("data/testclips") / clip / "source.mp4"),
        "window_video": _asset(root, tracknet / "input_0000_0010.mp4"),
        "label_summary": _label_summary(root, clip),
        "review_actions": resolved_review_actions,
        "intake_requirements": _clip_intake_requirements(root, clip, resolved_review_actions),
        "paddle_corner_review": _paddle_corner_review(root, clip),
    }


def _manifest(root: Path) -> dict[str, Any]:
    latest = root / LATEST_SAVE
    review_actions = _review_actions_by_clip(root)
    return {
        "schema_version": 1,
        "repo_root": str(root),
        "save_path": str(LATEST_SAVE),
        "latest_save": _read_json(latest) if latest.is_file() else None,
        "action_manifest": _asset(root, ACTION_MANIFEST),
        "global_review_actions": review_actions.get("__global__", _empty_review_actions()),
        "human_review_tasks": _human_review_tasks(root),
        "clips": [_clip_manifest(root, clip, review_actions=review_actions.get(clip)) for clip in CLIPS],
        "defaults": {
            "long_clip_policy": "fixed_windows",
            "artifact_source_of_truth": "h100",
            "racket_policy": "skip_this_wave",
            "custom_windows": "outdoor: 0-10s, 300-310s, 600-610s\nindoor: 0-10s, 300-310s, 600-610s",
        },
    }


ALLOWED_TOP_LEVEL_SAVE_KEYS = {
    "schema_version",
    "review_type",
    "global",
    "clips",
    "paddle_corner_labels",
    "tracking_video_review",
    "saved_from_browser_at",
    "manifest_seen",
}
SERVER_MANAGED_SAVE_KEYS = {"repo_root", "server_saved_at_utc"}
ALLOWED_GLOBAL_KEYS = {
    "long_clip_policy",
    "custom_windows",
    "artifact_source_of_truth",
    "artifact_notes",
    "racket_policy",
    "aruco_notes",
}
ALLOWED_CLIP_SAVE_KEYS = {
    "reviewed_enough",
    "court_overlay_ok",
    "top_net",
    "source_data",
    "court_evidence",
    "players",
    "spectators_ignore",
    "ball",
    "contacts",
    "event_windows",
    "racket",
    "general_notes",
}
ALLOWED_PLAYERS = {"P1", "P2", "P3", "P4", "unknown"}
ALLOWED_PLAYER_POSITIONS = {"near-left", "near-right", "far-left", "far-right", "unclear", ""}
ALLOWED_COURT_OVERLAY_STATES = {"unsure", "yes", "no"}
ALLOWED_COURT_EVIDENCE_STATES = {"unsure", "confirmed", "not_visible", "missing"}
ALLOWED_POINT_STATUSES = {"clicked", "not_visible", "missing"}
ALLOWED_BALL_MISTAKE_KINDS = {"bad_jump", "missing_ball", "false_ball", "looks_good"}
ALLOWED_TRACKING_DECISIONS = {"safe_to_promote", "unsafe_background_or_spectators", "unsure", ""}
ALLOWED_PADDLE_STATUSES = {"pending", "in_progress", "accepted", "not_paddle", "not_visible", "ambiguous"}
ALLOWED_CORNER_NAMES = {"top_left", "top_right", "bottom_right", "bottom_left"}


def _sanitize_save_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - ALLOWED_TOP_LEVEL_SAVE_KEYS - SERVER_MANAGED_SAVE_KEYS
    if unknown:
        raise ValueError(f"unexpected save fields: {', '.join(sorted(unknown))}")

    sanitized: dict[str, Any] = {
        "schema_version": _schema_version(payload.get("schema_version", 1)),
    }

    review_type = payload.get("review_type")
    if review_type is not None:
        review_type_text = _bounded_text(review_type, field="review_type")
        if review_type_text != "pickleball_cv_blocker_review":
            raise ValueError(f"unsupported review_type: {review_type_text}")
        sanitized["review_type"] = review_type_text

    if "global" in payload:
        sanitized["global"] = _sanitize_global(payload["global"])
    if "clips" in payload:
        sanitized["clips"] = _sanitize_clips(payload["clips"])
    if "paddle_corner_labels" in payload:
        sanitized["paddle_corner_labels"] = _sanitize_paddle_corner_labels(payload["paddle_corner_labels"])
    if "tracking_video_review" in payload:
        sanitized["tracking_video_review"] = _sanitize_tracking_video_review(payload["tracking_video_review"])
    if "saved_from_browser_at" in payload:
        sanitized["saved_from_browser_at"] = _bounded_text(payload["saved_from_browser_at"], field="saved_from_browser_at", max_chars=128)
    if "manifest_seen" in payload:
        sanitized["manifest_seen"] = _sanitize_manifest_seen(payload["manifest_seen"])

    return sanitized


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


def _sanitize_global(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("global must be a JSON object")
    unknown = set(value) - ALLOWED_GLOBAL_KEYS
    if unknown:
        raise ValueError(f"unexpected global fields: {', '.join(sorted(unknown))}")
    return {key: _bounded_text(value[key], field=f"global.{key}") for key in ALLOWED_GLOBAL_KEYS if key in value}


def _sanitize_clips(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("clips must be a JSON object")
    sanitized: dict[str, dict[str, Any]] = {}
    allowed_clips = set(CLIPS)
    for clip, raw_clip in value.items():
        clip_id = _bounded_text(clip, field="clip_id", max_chars=128)
        if clip_id not in allowed_clips:
            raise ValueError(f"unknown review clip id: {clip_id}")
        if not isinstance(raw_clip, dict):
            raise ValueError(f"clips.{clip_id} must be a JSON object")
        sanitized[clip_id] = _sanitize_clip_payload(clip_id, raw_clip)
    return sanitized


def _empty_save_clip() -> dict[str, Any]:
    return {
        "reviewed_enough": False,
        "court_overlay_ok": "unsure",
        "top_net": {"left": None, "right": None, "notes": ""},
        "source_data": {"source_video_path": "", "available_label_files": "", "data_owner": "", "notes": ""},
        "court_evidence": {
            "near_nvz": "unsure",
            "far_nvz": "unsure",
            "near_centerline": "unsure",
            "far_centerline": "unsure",
            "top_net": "unsure",
            "points": {},
            "point_statuses": {},
            "notes": "",
        },
        "players": {"P1": "", "P2": "", "P3": "", "P4": ""},
        "spectators_ignore": "",
        "ball": {"mistakes": [], "notes": ""},
        "contacts": [],
        "event_windows": [],
        "racket": {"examples": [], "notes": ""},
        "general_notes": "",
    }


def _sanitize_clip_payload(clip_id: str, value: dict[str, Any]) -> dict[str, Any]:
    unknown = set(value) - ALLOWED_CLIP_SAVE_KEYS
    if unknown:
        raise ValueError(f"unexpected fields for clip {clip_id}: {', '.join(sorted(unknown))}")
    out = _empty_save_clip()
    if "reviewed_enough" in value:
        out["reviewed_enough"] = bool(value["reviewed_enough"])
    if "court_overlay_ok" in value:
        out["court_overlay_ok"] = _enum_text(value["court_overlay_ok"], ALLOWED_COURT_OVERLAY_STATES, field=f"clips.{clip_id}.court_overlay_ok")
    if "top_net" in value:
        out["top_net"] = _sanitize_top_net(value["top_net"], field=f"clips.{clip_id}.top_net")
    if "source_data" in value:
        out["source_data"] = _sanitize_text_object(
            value["source_data"],
            allowed={"source_video_path", "available_label_files", "data_owner", "notes"},
            field=f"clips.{clip_id}.source_data",
        )
    if "court_evidence" in value:
        out["court_evidence"] = _sanitize_court_evidence(value["court_evidence"], field=f"clips.{clip_id}.court_evidence")
    if "players" in value:
        out["players"] = _sanitize_players(value["players"], field=f"clips.{clip_id}.players")
    if "spectators_ignore" in value:
        out["spectators_ignore"] = _bounded_text(value["spectators_ignore"], field=f"clips.{clip_id}.spectators_ignore")
    if "ball" in value:
        out["ball"] = _sanitize_ball(value["ball"], field=f"clips.{clip_id}.ball")
    if "contacts" in value:
        out["contacts"] = _sanitize_contact_list(value["contacts"], field=f"clips.{clip_id}.contacts")
    if "event_windows" in value:
        out["event_windows"] = _sanitize_event_window_list(value["event_windows"], field=f"clips.{clip_id}.event_windows")
    if "racket" in value:
        out["racket"] = _sanitize_racket(value["racket"], field=f"clips.{clip_id}.racket")
    if "general_notes" in value:
        out["general_notes"] = _bounded_text(value["general_notes"], field=f"clips.{clip_id}.general_notes")
    return out


def _sanitize_top_net(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - {"left", "right", "notes"}
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    return {
        "left": _optional_point(value.get("left"), field=f"{field}.left"),
        "right": _optional_point(value.get("right"), field=f"{field}.right"),
        "notes": _bounded_text(value.get("notes", ""), field=f"{field}.notes"),
    }


def _sanitize_court_evidence(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    allowed = set(COURT_EVIDENCE_IDS) | {"points", "point_statuses", "notes"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    out = {
        "near_nvz": "unsure",
        "far_nvz": "unsure",
        "near_centerline": "unsure",
        "far_centerline": "unsure",
        "top_net": "unsure",
        "points": {},
        "point_statuses": {},
        "notes": "",
    }
    for evidence_id in COURT_EVIDENCE_IDS:
        if evidence_id in value:
            out[evidence_id] = _enum_text(value[evidence_id], ALLOWED_COURT_EVIDENCE_STATES, field=f"{field}.{evidence_id}")
    if "points" in value:
        out["points"] = _sanitize_court_points(value["points"], field=f"{field}.points")
    if "point_statuses" in value:
        out["point_statuses"] = _sanitize_point_statuses(value["point_statuses"], field=f"{field}.point_statuses")
    if "notes" in value:
        out["notes"] = _bounded_text(value["notes"], field=f"{field}.notes")
    return out


def _sanitize_court_points(value: Any, *, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - COURT_POINT_IDS
    if unknown:
        raise ValueError(f"unexpected point ids for {field}: {', '.join(sorted(str(item) for item in unknown))}")
    out: dict[str, dict[str, Any]] = {}
    for point_id, raw_point in value.items():
        if not isinstance(raw_point, dict):
            raise ValueError(f"{field}.{point_id} must be a JSON object")
        evidence_id, endpoint = str(point_id).split(":", 1)
        point = _point(raw_point, field=f"{field}.{point_id}")
        point.update(
            {
                "target_id": point_id,
                "evidence_id": evidence_id,
                "endpoint": endpoint,
                "label": _bounded_text(raw_point.get("label", ""), field=f"{field}.{point_id}.label", max_chars=128),
                "status": _enum_text(raw_point.get("status", "clicked"), {"clicked"}, field=f"{field}.{point_id}.status"),
            }
        )
        out[str(point_id)] = point
    return out


def _sanitize_point_statuses(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - COURT_POINT_IDS
    if unknown:
        raise ValueError(f"unexpected point ids for {field}: {', '.join(sorted(str(item) for item in unknown))}")
    return {
        str(point_id): _enum_text(status, ALLOWED_POINT_STATUSES, field=f"{field}.{point_id}")
        for point_id, status in value.items()
    }


def _sanitize_players(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - {"P1", "P2", "P3", "P4"}
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    out = {"P1": "", "P2": "", "P3": "", "P4": ""}
    for key in out:
        if key in value:
            out[key] = _enum_text(value[key], ALLOWED_PLAYER_POSITIONS, field=f"{field}.{key}")
    return out


def _sanitize_ball(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - {"mistakes", "notes"}
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    return {
        "mistakes": _sanitize_ball_mistakes(value.get("mistakes", []), field=f"{field}.mistakes"),
        "notes": _bounded_text(value.get("notes", ""), field=f"{field}.notes"),
    }


def _sanitize_ball_mistakes(value: Any, *, field: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": _enum_text(item.get("kind"), ALLOWED_BALL_MISTAKE_KINDS, field=f"{field}[{index}].kind"),
            "time_s": _finite_nonnegative(item.get("time_s"), field=f"{field}[{index}].time_s"),
            "note": _bounded_text(item.get("note", ""), field=f"{field}[{index}].note"),
        }
        for index, item in _json_object_list(value, field=field)
    ]


def _sanitize_contact_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    return [
        {
            "player": _enum_text(item.get("player"), ALLOWED_PLAYERS, field=f"{field}[{index}].player"),
            "time_s": _finite_nonnegative(item.get("time_s"), field=f"{field}[{index}].time_s"),
            "note": _bounded_text(item.get("note", ""), field=f"{field}[{index}].note"),
        }
        for index, item in _json_object_list(value, field=field)
    ]


def _sanitize_event_window_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in _json_object_list(value, field=field):
        start_s = _finite_nonnegative(item.get("start_s"), field=f"{field}[{index}].start_s")
        end_s = _finite_nonnegative(item.get("end_s"), field=f"{field}[{index}].end_s")
        if end_s < start_s:
            raise ValueError(f"{field}[{index}].end_s must be greater than or equal to start_s")
        out.append({"start_s": start_s, "end_s": end_s, "note": _bounded_text(item.get("note", ""), field=f"{field}[{index}].note")})
    return out


def _sanitize_racket(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - {"examples", "notes"}
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    return {
        "examples": [
            {
                "player": _enum_text(item.get("player"), ALLOWED_PLAYERS, field=f"{field}.examples[{index}].player"),
                "time_s": _finite_nonnegative(item.get("time_s"), field=f"{field}.examples[{index}].time_s"),
                "note": _bounded_text(item.get("note", ""), field=f"{field}.examples[{index}].note"),
            }
            for index, item in _json_object_list(value.get("examples", []), field=f"{field}.examples")
        ],
        "notes": _bounded_text(value.get("notes", ""), field=f"{field}.notes"),
    }


def _sanitize_paddle_corner_labels(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("paddle_corner_labels must be a JSON object")
    sanitized: dict[str, dict[str, Any]] = {}
    allowed_clips = set(CLIPS)
    for clip, raw_bucket in value.items():
        clip_id = _bounded_text(clip, field="paddle_corner_labels clip id", max_chars=128)
        if clip_id not in allowed_clips:
            raise ValueError(f"unknown paddle review clip id: {clip_id}")
        if not isinstance(raw_bucket, dict):
            raise ValueError(f"paddle_corner_labels.{clip_id} must be a JSON object")
        unknown = set(raw_bucket) - {"artifact_type", "labels"}
        if unknown:
            raise ValueError(f"unexpected fields for paddle_corner_labels.{clip_id}: {', '.join(sorted(unknown))}")
        artifact_type = _bounded_text(raw_bucket.get("artifact_type", "paddle_true_corner_labels"), field=f"paddle_corner_labels.{clip_id}.artifact_type")
        if artifact_type not in {"paddle_true_corner_labels", "racketsport_paddle_true_corner_labels"}:
            raise ValueError(f"unsupported paddle label artifact_type: {artifact_type}")
        raw_labels = raw_bucket.get("labels", {})
        if not isinstance(raw_labels, dict):
            raise ValueError(f"paddle_corner_labels.{clip_id}.labels must be a JSON object")
        labels: dict[str, Any] = {}
        if len(raw_labels) > MAX_SAVE_LIST_ITEMS:
            raise ValueError(f"paddle_corner_labels.{clip_id}.labels has too many items")
        for review_id, raw_label in raw_labels.items():
            safe_review_id = _bounded_text(review_id, field=f"paddle_corner_labels.{clip_id}.review_id", max_chars=128)
            if not isinstance(raw_label, dict):
                raise ValueError(f"paddle_corner_labels.{clip_id}.labels.{safe_review_id} must be a JSON object")
            labels[safe_review_id] = _sanitize_paddle_label(raw_label, review_id=safe_review_id, field=f"paddle_corner_labels.{clip_id}.labels.{safe_review_id}")
        sanitized[clip_id] = {"artifact_type": "paddle_true_corner_labels", "labels": labels}
    return sanitized


def _sanitize_paddle_label(value: dict[str, Any], *, review_id: str, field: str) -> dict[str, Any]:
    allowed = {
        "review_id",
        "player_id",
        "frame_index",
        "t",
        "crop_xyxy",
        "status",
        "evidence_type",
        "reviewer",
        "corners_px_order",
        "corners_by_name",
        "sheet_points_by_name",
        "corners_px",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    out = {
        "review_id": _bounded_text(value.get("review_id", review_id), field=f"{field}.review_id", max_chars=128),
        "status": _enum_text(value.get("status", "pending"), ALLOWED_PADDLE_STATUSES, field=f"{field}.status"),
        "evidence_type": _bounded_text(value.get("evidence_type", "true_corners"), field=f"{field}.evidence_type", max_chars=128),
        "reviewer": _bounded_text(value.get("reviewer", "local_click_review"), field=f"{field}.reviewer", max_chars=128),
        "corners_px_order": _corner_order(value.get("corners_px_order", ["top_left", "top_right", "bottom_right", "bottom_left"]), field=f"{field}.corners_px_order"),
        "corners_by_name": _corner_point_map(value.get("corners_by_name", {}), field=f"{field}.corners_by_name"),
        "sheet_points_by_name": _corner_point_map(value.get("sheet_points_by_name", {}), field=f"{field}.sheet_points_by_name"),
    }
    if "player_id" in value:
        out["player_id"] = _json_scalar(value["player_id"], field=f"{field}.player_id")
    if "frame_index" in value:
        out["frame_index"] = int(_finite_nonnegative(value["frame_index"], field=f"{field}.frame_index"))
    if "t" in value:
        out["t"] = _finite_nonnegative(value["t"], field=f"{field}.t")
    if "crop_xyxy" in value:
        out["crop_xyxy"] = _numeric_list_exact(value["crop_xyxy"], 4, field=f"{field}.crop_xyxy")
    if "corners_px" in value:
        out["corners_px"] = _point_list(value["corners_px"], field=f"{field}.corners_px")
    return out


def _sanitize_tracking_video_review(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("tracking_video_review must be a JSON object")
    allowed_clips = {str(item["clip"]) for item in TRACKING_REVIEW_ITEMS}
    sanitized: dict[str, dict[str, str]] = {}
    for clip, raw_decision in value.items():
        clip_id = _bounded_text(clip, field="tracking_video_review clip id", max_chars=128)
        if clip_id not in allowed_clips:
            raise ValueError(f"unknown tracking review clip id: {clip_id}")
        if not isinstance(raw_decision, dict):
            raise ValueError(f"tracking_video_review.{clip_id} must be a JSON object")
        unknown = set(raw_decision) - {"decision", "notes"}
        if unknown:
            raise ValueError(f"unexpected fields for tracking_video_review.{clip_id}: {', '.join(sorted(unknown))}")
        sanitized[clip_id] = {
            "decision": _enum_text(raw_decision.get("decision", ""), ALLOWED_TRACKING_DECISIONS, field=f"tracking_video_review.{clip_id}.decision"),
            "notes": _bounded_text(raw_decision.get("notes", ""), field=f"tracking_video_review.{clip_id}.notes"),
        }
    return sanitized


def _sanitize_manifest_seen(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("manifest_seen must be a JSON object")
    allowed = {"repo_root", "focused_review_page", "clips", "paddle_clips", "tracking_videos"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unexpected manifest_seen fields: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for key in ("repo_root", "focused_review_page"):
        if key in value:
            out[key] = _bounded_text(value[key], field=f"manifest_seen.{key}", max_chars=1024)
    manifest_list_fields = {
        "clips": {"id", "label_overlay", "calibration_overlay", "ball_overlay", "track_overlay"},
        "paddle_clips": {"id", "crop_sheet", "candidate_overlay"},
        "tracking_videos": {"clip", "overlay_video", "montage"},
    }
    for key, fields in manifest_list_fields.items():
        if key not in value:
            continue
        out[key] = [
            {field_name: _bounded_text(item[field_name], field=f"manifest_seen.{key}[{index}].{field_name}", max_chars=1024) for field_name in fields if field_name in item}
            for index, item in _json_object_list(value[key], field=f"manifest_seen.{key}", max_items=100)
        ]
    return out


def _sanitize_text_object(value: Any, *, allowed: set[str], field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    return {key: _bounded_text(value.get(key, ""), field=f"{field}.{key}") for key in allowed}


def _optional_point(value: Any, *, field: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be null or a JSON object")
    return _point(value, field=field)


def _point(value: dict[str, Any], *, field: str) -> dict[str, float]:
    allowed = {"x", "y", "time_s", "video_width", "video_height", "target_id", "evidence_id", "endpoint", "label", "status"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unexpected fields for {field}: {', '.join(sorted(unknown))}")
    return {
        "x": _finite_nonnegative(value.get("x"), field=f"{field}.x"),
        "y": _finite_nonnegative(value.get("y"), field=f"{field}.y"),
        "time_s": _finite_nonnegative(value.get("time_s"), field=f"{field}.time_s"),
        "video_width": _finite_positive(value.get("video_width"), field=f"{field}.video_width"),
        "video_height": _finite_positive(value.get("video_height"), field=f"{field}.video_height"),
    }


def _json_object_list(value: Any, *, field: str, max_items: int = MAX_SAVE_LIST_ITEMS) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    if len(value) > max_items:
        raise ValueError(f"{field} has too many items")
    out: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be a JSON object")
        out.append((index, item))
    return out


def _enum_text(value: Any, allowed: set[str], *, field: str) -> str:
    text = _bounded_text(value, field=field, max_chars=128)
    if text not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return text


def _bounded_text(value: Any, *, field: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) > max_chars:
        raise ValueError(f"{field} is too long")
    return value


def _finite_nonnegative(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _finite_positive(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _numeric_list_exact(value: Any, length: int, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a JSON array of {length} numbers")
    return [_finite_number(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def _corner_order(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must contain four corner names")
    out = [_enum_text(item, ALLOWED_CORNER_NAMES, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(out)) != 4:
        raise ValueError(f"{field} must not contain duplicate corner names")
    return out


def _corner_point_map(value: Any, *, field: str) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    unknown = set(value) - ALLOWED_CORNER_NAMES
    if unknown:
        raise ValueError(f"unexpected corner names for {field}: {', '.join(sorted(unknown))}")
    return {
        str(name): _numeric_list_exact(point, 2, field=f"{field}.{name}")
        for name, point in value.items()
    }


def _point_list(value: Any, *, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) > 4:
        raise ValueError(f"{field} must contain at most four points")
    return [_numeric_list_exact(point, 2, field=f"{field}[{index}]") for index, point in enumerate(value)]


def _json_scalar(value: Any, *, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _bounded_text(value, field=field, max_chars=128)
    if isinstance(value, int):
        return value
    if isinstance(value, int | float):
        return _finite_number(value, field=field)
    raise ValueError(f"{field} must be a JSON scalar")


def _write_review_input(root: Path, payload: dict[str, Any], *, now: datetime | None = None) -> tuple[Path, Path]:
    saved_at_dt = now or datetime.now(timezone.utc)
    saved_at = saved_at_dt.strftime("%Y%m%dT%H%M%SZ")
    sanitized = _sanitize_save_payload(payload)
    sanitized["server_saved_at_utc"] = saved_at_dt.isoformat()
    sanitized["repo_root"] = str(root)
    save_dir = root / SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = root / LATEST_SAVE
    timestamped = save_dir / f"pickleball_cv_review_{saved_at}.json"
    text = json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    latest.write_text(text, encoding="utf-8")
    timestamped.write_text(text, encoding="utf-8")
    return latest, timestamped


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pickleball CV Review</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdfa;
      --ink: #201b16;
      --muted: #6c6258;
      --line: #d8d0c4;
      --accent: #0f766e;
      --accent-2: #9f4f18;
      --danger: #b42318;
      --ok: #177245;
      --field: #fffcf7;
      --shadow: 0 12px 30px rgba(44, 36, 27, 0.09);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      background: var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", serif;
      line-height: 1.35;
    }
    button, input, textarea, select {
      font: inherit;
    }
    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 22px;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      background: rgba(246, 244, 238, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
      backdrop-filter: blur(8px);
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 760;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-top: 3px;
    }
    .savebar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .btn {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      min-height: 38px;
    }
    .btn:hover { border-color: #b9aa99; }
    .btn.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 720;
    }
    .btn.small {
      min-height: 30px;
      padding: 5px 9px;
      font-size: 13px;
    }
    .status {
      font-size: 13px;
      color: var(--muted);
      min-width: 260px;
      text-align: right;
    }
    .global {
      margin: 22px 0;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 8px;
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
    }
    .global-actions {
      margin: 0 0 22px;
    }
    .guide {
      margin: 22px 0 0;
      background: #23342f;
      color: #fff8ea;
      border-radius: 8px;
      padding: 14px 16px;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      box-shadow: var(--shadow);
    }
    .guide-step {
      border-left: 1px solid rgba(255,255,255,.22);
      padding-left: 10px;
      font-size: 13px;
    }
    .guide-step:first-child {
      border-left: 0;
      padding-left: 0;
    }
    .guide-step strong {
      display: block;
      font-size: 14px;
      margin-bottom: 3px;
    }
    .fieldgroup {
      display: grid;
      gap: 8px;
      align-content: start;
    }
    label.title {
      font-weight: 760;
      font-size: 14px;
    }
    .radio-row, .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      min-height: 26px;
    }
    .clip {
      margin: 24px 0 34px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .clip-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      background: #faf7f0;
    }
    .clip-title {
      font-size: 20px;
      font-weight: 780;
      overflow-wrap: anywhere;
    }
    .clip-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      justify-content: flex-end;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fffaf2;
      white-space: nowrap;
    }
    .pill.warn { color: var(--danger); border-color: #e2b7ae; }
    .pill.ok { color: var(--ok); border-color: #a9d7bd; }
    .clip-body {
      display: grid;
      grid-template-columns: minmax(360px, 1.25fr) minmax(320px, 0.85fr);
      gap: 18px;
      padding: 16px;
      align-items: start;
    }
    .media-grid {
      display: grid;
      gap: 14px;
    }
    .video-box {
      border: 1px solid var(--line);
      background: #16120f;
      border-radius: 8px;
      overflow: hidden;
    }
    .video-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      color: #fff7ec;
      background: #2a241d;
      font-size: 13px;
    }
    .video-wrap {
      position: relative;
      background: #100d0b;
    }
    video {
      display: block;
      width: 100%;
      max-height: 520px;
      background: #100d0b;
    }
    .missing {
      min-height: 150px;
      display: grid;
      place-content: center;
      padding: 24px;
      color: #f1d8c9;
      text-align: center;
      font-size: 14px;
    }
    .click-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .click-layer.active {
      pointer-events: auto;
      cursor: crosshair;
      outline: 3px solid rgba(15, 118, 110, 0.75);
      outline-offset: -3px;
    }
    .marker {
      position: absolute;
      width: 15px;
      height: 15px;
      border-radius: 999px;
      border: 2px solid white;
      background: var(--accent-2);
      transform: translate(-50%, -50%);
      box-shadow: 0 1px 8px rgba(0,0,0,.45);
    }
    .controls {
      padding: 8px 10px 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      background: #f9f4eb;
    }
    .forms {
      display: grid;
      gap: 16px;
    }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .section:first-child {
      border-top: 0;
      padding-top: 0;
    }
    .section h2 {
      margin: 0 0 10px;
      font-size: 16px;
      font-weight: 780;
    }
    .blocker-section {
      background: #fff8ea;
      border: 1px solid #decaa8;
      border-radius: 8px;
      padding: 12px;
    }
    .blocker-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-bottom: 10px;
    }
    .blocker-chip {
      background: #fffdfa;
      border: 1px solid #e2d4bc;
      border-radius: 999px;
      color: #574a3d;
      font-size: 12px;
      padding: 4px 8px;
    }
    .action-list {
      display: grid;
      gap: 8px;
    }
    .action-card {
      background: #fffdfa;
      border: 1px solid #e3d7c5;
      border-left: 4px solid #d59d54;
      border-radius: 8px;
      padding: 9px 10px;
    }
    .action-card.high {
      border-left-color: var(--danger);
    }
    .action-card.medium {
      border-left-color: #b97818;
    }
    .action-card-head {
      align-items: baseline;
      display: flex;
      gap: 8px;
      justify-content: space-between;
    }
    .action-card-title {
      font-size: 13px;
      font-weight: 780;
    }
    .action-card-meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .action-card p {
      color: #5c5147;
      font-size: 13px;
      margin: 6px 0 0;
    }
    .blocker-list {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 7px;
    }
    .blocker-tag {
      background: #fff3f0;
      border: 1px solid #efc4bb;
      border-radius: 999px;
      color: #8f241d;
      font-size: 12px;
      padding: 3px 7px;
    }
    .command-list {
      display: grid;
      gap: 4px;
      margin-top: 8px;
    }
    .command-item {
      align-items: stretch;
      display: grid;
      gap: 6px;
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .command-list code {
      background: #f7efe3;
      border: 1px solid #e5d5bf;
      border-radius: 6px;
      display: block;
      font-size: 11px;
      overflow-wrap: anywhere;
      padding: 5px 6px;
      white-space: normal;
    }
    .copy-command {
      white-space: nowrap;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .grid4 {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }
    input[type="text"], input[type="number"], textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--field);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      min-height: 38px;
    }
    textarea {
      min-height: 82px;
      resize: vertical;
    }
    .mini-label {
      display: grid;
      gap: 4px;
      font-size: 13px;
      color: var(--muted);
    }
    .list {
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }
    .item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffaf2;
      font-size: 13px;
    }
    .item button {
      border: 0;
      background: transparent;
      color: var(--danger);
      cursor: pointer;
      font-weight: 780;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    .asset-path {
      font-size: 12px;
      color: #c8b8a4;
      overflow-wrap: anywhere;
    }
    .footer-space {
      height: 80px;
    }
    @media (max-width: 1100px) {
      .global, .clip-body, .guide { grid-template-columns: 1fr; }
      .guide-step { border-left: 0; border-top: 1px solid rgba(255,255,255,.22); padding-left: 0; padding-top: 9px; }
      .guide-step:first-child { border-top: 0; padding-top: 0; }
      .grid4 { grid-template-columns: 1fr 1fr; }
      .status { text-align: left; min-width: 0; }
      .topbar { grid-template-columns: 1fr; }
      .savebar { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>Pickleball CV Review</h1>
      <div class="subtitle">One page for player IDs, ball/contact review, top-net clicks, racket decisions, and source-of-truth choices.</div>
    </div>
    <div class="savebar">
      <button class="btn" id="loadLatest">Load latest</button>
      <button class="btn primary" id="save">Save review JSON</button>
      <div class="status" id="saveStatus">Not saved yet.</div>
    </div>
  </div>
  <main class="shell">
    <section class="guide">
      <div class="guide-step"><strong>1. Players</strong>Type clothing/location for the four real players and note people to ignore.</div>
      <div class="guide-step"><strong>2. Ball</strong>Use the ball video buttons at moments where it jumps, disappears, or a paddle contact happens.</div>
      <div class="guide-step"><strong>3. Top Net</strong>On the calibration video, click the left and right ends of the top net tape.</div>
      <div class="guide-step"><strong>4. Events</strong>Use the label video time to add rally start/end windows.</div>
      <div class="guide-step"><strong>5. Save</strong>Click Save once. It writes one JSON file I can consume directly.</div>
    </section>
    <section class="global">
      <div class="fieldgroup">
        <label class="title">Long clip policy</label>
        <label class="radio-row"><input type="radio" name="longClipPolicy" value="fixed_windows"> Use fixed windows</label>
        <label class="radio-row"><input type="radio" name="longClipPolicy" value="full_sources"> Run full long clips</label>
        <label class="radio-row"><input type="radio" name="longClipPolicy" value="custom_windows"> Use custom windows below</label>
        <textarea id="customWindows"></textarea>
      </div>
      <div class="fieldgroup">
        <label class="title">Artifact source of truth</label>
        <label class="radio-row"><input type="radio" name="artifactSource" value="h100"> H100 is source of truth</label>
        <label class="radio-row"><input type="radio" name="artifactSource" value="local"> Local is source of truth</label>
        <label class="radio-row"><input type="radio" name="artifactSource" value="unsure"> Decide after sync check</label>
        <textarea id="artifactNotes" placeholder="Optional notes"></textarea>
      </div>
      <div class="fieldgroup">
        <label class="title">Racket verification</label>
        <label class="radio-row"><input type="radio" name="racketPolicy" value="skip_this_wave"> Skip racket this wave</label>
        <label class="radio-row"><input type="radio" name="racketPolicy" value="approved_examples"> Use approved visible paddle examples</label>
        <label class="radio-row"><input type="radio" name="racketPolicy" value="aruco_gt"> I can provide ArUco GT</label>
        <textarea id="arucoNotes" placeholder="ArUco dictionary, marker ID, size, mount location, valid time range"></textarea>
      </div>
    </section>
    <section id="globalActions" class="global-actions"></section>
    <div id="clips"></div>
    <div class="footer-space"></div>
  </main>

  <script>
    let manifest = null;
    let state = null;
    const clipVideos = new Map();

    const blankClip = () => ({
      reviewed_enough: false,
      court_overlay_ok: "unsure",
      top_net: { left: null, right: null, notes: "" },
      players: { P1: "", P2: "", P3: "", P4: "" },
      spectators_ignore: "",
      ball: { mistakes: [], notes: "" },
      contacts: [],
      event_windows: [],
      racket: { examples: [], notes: "" },
      general_notes: ""
    });

    function defaultState(data) {
      const clips = {};
      for (const clip of data.clips) clips[clip.id] = blankClip();
      return {
        schema_version: 1,
        review_type: "pickleball_cv_blocker_review",
        global: {
          long_clip_policy: data.defaults.long_clip_policy,
          custom_windows: data.defaults.custom_windows,
          artifact_source_of_truth: data.defaults.artifact_source_of_truth,
          artifact_notes: "",
          racket_policy: data.defaults.racket_policy,
          aruco_notes: ""
        },
        clips
      };
    }

    function qs(sel, root=document) { return root.querySelector(sel); }
    function qsa(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }
    function assetTag(asset) {
      if (!asset || !asset.exists) return '<div class="missing">Missing local asset<br><span class="asset-path">' + escapeHtml(asset?.path || "") + '</span></div>';
      return '<video controls preload="metadata" src="' + asset.url + '"></video>';
    }
    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
    }
    function fmtTime(t) {
      if (t === null || t === undefined || Number.isNaN(Number(t))) return "";
      const v = Number(t);
      const m = Math.floor(v / 60);
      const s = (v - m * 60).toFixed(2).padStart(5, "0");
      return m + ":" + s;
    }
    function videoTime(clipId, key) {
      const video = clipVideos.get(clipId + ":" + key);
      return video ? Number(video.currentTime || 0) : 0;
    }

    function setRadios(name, value) {
      qsa('input[name="' + name + '"]').forEach(input => input.checked = input.value === value);
    }
    function syncGlobalForm() {
      setRadios("longClipPolicy", state.global.long_clip_policy);
      setRadios("artifactSource", state.global.artifact_source_of_truth);
      setRadios("racketPolicy", state.global.racket_policy);
      qs("#customWindows").value = state.global.custom_windows || "";
      qs("#artifactNotes").value = state.global.artifact_notes || "";
      qs("#arucoNotes").value = state.global.aruco_notes || "";
    }
    function bindGlobalForm() {
      qsa('input[name="longClipPolicy"]').forEach(el => el.addEventListener("change", () => state.global.long_clip_policy = el.value));
      qsa('input[name="artifactSource"]').forEach(el => el.addEventListener("change", () => state.global.artifact_source_of_truth = el.value));
      qsa('input[name="racketPolicy"]').forEach(el => el.addEventListener("change", () => state.global.racket_policy = el.value));
      qs("#customWindows").addEventListener("input", e => state.global.custom_windows = e.target.value);
      qs("#artifactNotes").addEventListener("input", e => state.global.artifact_notes = e.target.value);
      qs("#arucoNotes").addEventListener("input", e => state.global.aruco_notes = e.target.value);
    }

    function renderClip(clip) {
      const data = state.clips[clip.id] || blankClip();
      const labelSummary = clip.label_summary || {};
      const frameMeta = labelSummary.frame_count ? labelSummary.frame_count + " label frames" : "label frames not synced";
      return `
      <section class="clip" data-clip="${clip.id}">
        <div class="clip-head">
          <div>
            <div class="clip-title">${escapeHtml(clip.id)}</div>
            <div class="hint">${escapeHtml(frameMeta)}${labelSummary.source_fps ? " | " + labelSummary.source_fps + " fps" : ""}</div>
          </div>
          <div class="clip-meta">
            <span class="pill ${clip.calibration_overlay.exists ? "ok" : "warn"}">calibration</span>
            <span class="pill ${clip.label_overlay.exists ? "ok" : "warn"}">labels</span>
            <span class="pill ${clip.ball_overlay.exists ? "ok" : "warn"}">ball</span>
            <span class="pill ${clip.track_overlay.exists ? "ok" : "warn"}">track overlay</span>
            <span class="pill ${clip.review_actions.action_count ? "warn" : "ok"}">actions ${clip.review_actions.action_count || 0}</span>
          </div>
        </div>
        <div class="clip-body">
          <div class="media-grid">
            ${videoBox(clip, "label", "All labels overlay", clip.label_overlay, false)}
            ${videoBox(clip, "ball", "Ball overlay", clip.ball_overlay, false)}
            ${videoBox(clip, "calibration", "Calibration overlay - click top net", clip.calibration_overlay, true)}
            ${videoBox(clip, "track", "Latest local track overlay if synced", clip.track_overlay, false)}
          </div>
          <div class="forms">
            ${actionPanel(clip)}
            <div class="section">
              <h2>Players and Background</h2>
              <div class="grid4">
                ${["P1","P2","P3","P4"].map(pid => `
                  <label class="mini-label">${pid}
                    <input type="text" data-field="players.${pid}" value="${escapeHtml(data.players[pid] || "")}" placeholder="clothing/location">
                  </label>`).join("")}
              </div>
              <label class="mini-label" style="margin-top:10px;">Spectators/background people to ignore
                <textarea data-field="spectators_ignore">${escapeHtml(data.spectators_ignore || "")}</textarea>
              </label>
            </div>
            <div class="section">
              <h2>Top Net and Court</h2>
              <div class="grid2">
                <label class="mini-label">Court overlay OK?
                  <select data-field="court_overlay_ok">
                    ${options(["unsure","yes","no"], data.court_overlay_ok)}
                  </select>
                </label>
                <label class="check-row" style="align-self:end;"><input type="checkbox" data-field="reviewed_enough" ${data.reviewed_enough ? "checked" : ""}> Clip reviewed enough</label>
              </div>
              <div class="hint" id="netStatus-${clip.id}">${topNetStatus(data)}</div>
              <label class="mini-label" style="margin-top:8px;">Top-net notes
                <textarea data-field="top_net.notes">${escapeHtml(data.top_net.notes || "")}</textarea>
              </label>
            </div>
            <div class="section">
              <h2>Ball and Contact</h2>
              <div class="controls" style="padding:0;background:transparent;">
                <button class="btn small" data-action="addBallMistake" data-kind="bad_jump">Ball jumps here</button>
                <button class="btn small" data-action="addBallMistake" data-kind="missing_ball">Ball missing here</button>
                <select data-role="contactPlayer">
                  <option value="P1">P1</option><option value="P2">P2</option><option value="P3">P3</option><option value="P4">P4</option><option value="unknown">unknown</option>
                </select>
                <button class="btn small" data-action="addContact">Add contact at ball time</button>
              </div>
              <div class="list" data-list="ball.mistakes"></div>
              <div class="list" data-list="contacts"></div>
              <label class="mini-label" style="margin-top:8px;">Ball notes
                <textarea data-field="ball.notes">${escapeHtml(data.ball.notes || "")}</textarea>
              </label>
            </div>
            <div class="section">
              <h2>Event Windows</h2>
              <div class="controls" style="padding:0;background:transparent;">
                <button class="btn small" data-action="setWindowStart">Set rally start at label time</button>
                <button class="btn small" data-action="addWindowEnd">Add rally end at label time</button>
              </div>
              <div class="list" data-list="event_windows"></div>
            </div>
            <div class="section">
              <h2>Racket Examples</h2>
              <div class="controls" style="padding:0;background:transparent;">
                <select data-role="racketPlayer">
                  <option value="P1">P1</option><option value="P2">P2</option><option value="P3">P3</option><option value="P4">P4</option><option value="unknown">unknown</option>
                </select>
                <button class="btn small" data-action="addRacketExample">Add visible paddle at label time</button>
              </div>
              <div class="list" data-list="racket.examples"></div>
              <label class="mini-label" style="margin-top:8px;">Racket notes
                <textarea data-field="racket.notes">${escapeHtml(data.racket.notes || "")}</textarea>
              </label>
            </div>
            <div class="section">
              <h2>Anything Else</h2>
              <textarea data-field="general_notes">${escapeHtml(data.general_notes || "")}</textarea>
            </div>
          </div>
        </div>
      </section>`;
    }

    function actionPanel(clip) {
      const review = clip.review_actions || {};
      const actions = Array.isArray(review.actions) ? review.actions : [];
      if (!actions.length) {
        return `<div class="section blocker-section">
          <h2>Current Blockers</h2>
          <div class="hint">No generated packet actions for this clip.</div>
        </div>`;
      }
      const categoryText = Object.entries(review.categories || {})
        .map(([key, value]) => `${escapeHtml(key)}=${escapeHtml(value)}`)
        .join(" ");
      return `<div class="section blocker-section">
        <h2>Current Blockers</h2>
        <div class="blocker-summary">
          <span class="blocker-chip">${actions.length} actions</span>
          <span class="blocker-chip">${review.high_count || 0} high</span>
          <span class="blocker-chip">${review.medium_count || 0} medium</span>
          ${categoryText ? `<span class="blocker-chip">${categoryText}</span>` : ""}
        </div>
        <div class="action-list">
          ${actions.map(actionCard).join("")}
        </div>
      </div>`;
    }

    function globalActionPanel(review) {
      const actions = Array.isArray(review?.actions) ? review.actions : [];
      if (!actions.length) return "";
      const categoryText = Object.entries(review.categories || {})
        .map(([key, value]) => `${escapeHtml(key)}=${escapeHtml(value)}`)
        .join(" ");
      return `<div class="section blocker-section">
        <h2>Global Blockers</h2>
        <div class="blocker-summary">
          <span class="blocker-chip">${actions.length} actions</span>
          <span class="blocker-chip">${review.high_count || 0} high</span>
          <span class="blocker-chip">${review.medium_count || 0} medium</span>
          ${categoryText ? `<span class="blocker-chip">${categoryText}</span>` : ""}
        </div>
        <div class="action-list">
          ${actions.map(actionCard).join("")}
        </div>
      </div>`;
    }

    function actionCard(action) {
      const blockers = Array.isArray(action.blockers) ? action.blockers : [];
      const commands = Array.isArray(action.next_commands) ? action.next_commands : [];
      return `<div class="action-card ${escapeHtml(action.priority || "")}">
        <div class="action-card-head">
          <span class="action-card-title">${escapeHtml(action.title || "Review action")}</span>
          <span class="action-card-meta">${escapeHtml(action.priority || "unknown")} / ${escapeHtml(action.category || "unknown")}</span>
        </div>
        ${action.suggested_action ? `<p>${escapeHtml(action.suggested_action)}</p>` : ""}
        ${blockers.length ? `<div class="blocker-list">${blockers.map(blocker => `<span class="blocker-tag">${escapeHtml(blocker)}</span>`).join("")}</div>` : ""}
        ${commands.length ? `<div class="command-list">${commands.map(command => `<div class="command-item"><code>${escapeHtml(command)}</code><button class="btn small copy-command" type="button" data-copy-command>Copy</button></div>`).join("")}</div>` : ""}
      </div>`;
    }

    function videoBox(clip, key, label, asset, clickable) {
      return `<div class="video-box" data-video-box="${clip.id}:${key}">
        <div class="video-label"><span>${escapeHtml(label)}</span><span class="asset-path">${escapeHtml(asset.path || "")}</span></div>
        <div class="video-wrap">
          ${assetTag(asset)}
          ${clickable ? `<div class="click-layer" data-click-layer="${clip.id}"></div>` : ""}
        </div>
        ${clickable ? `<div class="controls">
          <button class="btn small" data-action="captureNet" data-point="left">Click left top-net point</button>
          <button class="btn small" data-action="captureNet" data-point="right">Click right top-net point</button>
          <button class="btn small" data-action="clearNet">Clear points</button>
        </div>` : ""}
      </div>`;
    }

    function options(values, selected) {
      return values.map(v => `<option value="${v}" ${v === selected ? "selected" : ""}>${v}</option>`).join("");
    }

    function getPath(obj, path) {
      return path.split(".").reduce((acc, key) => acc ? acc[key] : undefined, obj);
    }
    function setPath(obj, path, value) {
      const parts = path.split(".");
      let cur = obj;
      for (const p of parts.slice(0, -1)) cur = cur[p];
      cur[parts[parts.length - 1]] = value;
    }
    function topNetStatus(data) {
      const left = data.top_net.left ? `left (${Math.round(data.top_net.left.x)}, ${Math.round(data.top_net.left.y)})` : "left missing";
      const right = data.top_net.right ? `right (${Math.round(data.top_net.right.x)}, ${Math.round(data.top_net.right.y)})` : "right missing";
      return "Top net: " + left + " | " + right;
    }

    function renderLists(section, clipId) {
      const data = state.clips[clipId];
      const listRenderers = {
        "ball.mistakes": item => `${item.kind} @ ${fmtTime(item.time_s)} ${item.note ? "- " + escapeHtml(item.note) : ""}`,
        "contacts": item => `${item.player} contact @ ${fmtTime(item.time_s)} ${item.note ? "- " + escapeHtml(item.note) : ""}`,
        "event_windows": item => `rally ${fmtTime(item.start_s)} - ${fmtTime(item.end_s)} ${item.note ? "- " + escapeHtml(item.note) : ""}`,
        "racket.examples": item => `${item.player} visible paddle @ ${fmtTime(item.time_s)} ${item.note ? "- " + escapeHtml(item.note) : ""}`
      };
      qsa("[data-list]", section).forEach(list => {
        const path = list.dataset.list;
        const arr = getPath(data, path) || [];
        list.innerHTML = arr.map((item, index) => `<div class="item"><span>${listRenderers[path](item)}</span><button data-remove="${path}" data-index="${index}">x</button></div>`).join("");
      });
    }

    function renderAll() {
      qs("#globalActions").innerHTML = globalActionPanel(manifest.global_review_actions);
      qs("#clips").innerHTML = manifest.clips.map(renderClip).join("");
      syncGlobalForm();
      bindClipForms();
      bindCommandButtons();
      updateMarkersAll();
    }

    function bindCommandButtons() {
      qsa("[data-copy-command]").forEach(btn => {
        btn.addEventListener("click", () => copyCommand(btn));
      });
    }

    async function copyCommand(btn) {
      const item = btn.closest(".command-item");
      const code = item ? qs("code", item) : null;
      const command = code ? code.textContent : "";
      if (!command) return;
      try {
        if (!navigator.clipboard) throw new Error("clipboard unavailable");
        await navigator.clipboard.writeText(command);
        markCommandCopied(btn);
      } catch {
        if (fallbackCopyCommand(command)) {
          markCommandCopied(btn);
        } else {
          selectCommandText(code);
          btn.textContent = "Selected";
          qs("#saveStatus").textContent = "Copy unavailable. Command selected.";
          window.setTimeout(() => { btn.textContent = "Copy"; }, 1400);
        }
      }
    }

    function markCommandCopied(btn) {
      btn.textContent = "Copied";
      qs("#saveStatus").textContent = "Command copied.";
      window.setTimeout(() => { btn.textContent = "Copy"; }, 1400);
    }

    function fallbackCopyCommand(command) {
      const textArea = document.createElement("textarea");
      textArea.value = command;
      textArea.setAttribute("readonly", "");
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      textArea.style.top = "0";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        return document.execCommand("copy");
      } catch {
        return false;
      } finally {
        document.body.removeChild(textArea);
      }
    }

    function selectCommandText(code) {
      if (!code || !window.getSelection || !document.createRange) return;
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }

    function bindClipForms() {
      clipVideos.clear();
      qsa(".clip").forEach(section => {
        const clipId = section.dataset.clip;
        qsa("video", section).forEach(video => {
          const box = video.closest("[data-video-box]");
          if (!box) return;
          const key = box.dataset.videoBox.split(":")[1];
          clipVideos.set(clipId + ":" + key, video);
        });

        qsa("[data-field]", section).forEach(el => {
          const path = el.dataset.field;
          const clip = state.clips[clipId];
          const handler = () => {
            const value = el.type === "checkbox" ? el.checked : el.value;
            setPath(clip, path, value);
            if (path === "top_net.notes" || path === "court_overlay_ok" || path === "reviewed_enough") {
              const status = qs("#netStatus-" + clipId);
              if (status) status.textContent = topNetStatus(clip);
            }
          };
          el.addEventListener("input", handler);
          el.addEventListener("change", handler);
        });

        qsa("[data-action]", section).forEach(btn => btn.addEventListener("click", () => handleAction(section, clipId, btn)));
        qsa("[data-click-layer]", section).forEach(layer => {
          layer.addEventListener("click", event => captureNetPoint(event, clipId, layer));
        });
        qsa("[data-remove]", section).forEach(btn => btn.addEventListener("click", () => removeListItem(section, clipId, btn)));
        renderLists(section, clipId);
      });
    }

    function handleAction(section, clipId, btn) {
      const data = state.clips[clipId];
      const action = btn.dataset.action;
      if (action === "captureNet") {
        qsa(".click-layer").forEach(layer => { layer.classList.remove("active"); layer.dataset.point = ""; });
        const layer = qs(`[data-click-layer="${clipId}"]`, section);
        if (layer) {
          layer.classList.add("active");
          layer.dataset.point = btn.dataset.point;
          qs("#saveStatus").textContent = "Click the " + btn.dataset.point + " top-net point on the calibration video.";
        }
      }
      if (action === "clearNet") {
        data.top_net.left = null;
        data.top_net.right = null;
        updateMarkers(section, clipId);
      }
      if (action === "addBallMistake") {
        const note = window.prompt("Short note for this ball issue:", "") || "";
        data.ball.mistakes.push({ kind: btn.dataset.kind, time_s: videoTime(clipId, "ball"), note });
      }
      if (action === "addContact") {
        const player = qs('[data-role="contactPlayer"]', section).value;
        const note = window.prompt("Contact note, if any:", "") || "";
        data.contacts.push({ player, time_s: videoTime(clipId, "ball"), note });
      }
      if (action === "setWindowStart") {
        data._pending_window_start = videoTime(clipId, "label");
        qs("#saveStatus").textContent = "Rally start set. Move video to rally end and click Add rally end.";
      }
      if (action === "addWindowEnd") {
        const start = data._pending_window_start ?? videoTime(clipId, "label");
        const end = videoTime(clipId, "label");
        const note = window.prompt("Window note:", "") || "";
        data.event_windows.push({ start_s: Math.min(start, end), end_s: Math.max(start, end), note });
        delete data._pending_window_start;
      }
      if (action === "addRacketExample") {
        const player = qs('[data-role="racketPlayer"]', section).value;
        const note = window.prompt("What is visible? face direction? confidence?", "") || "";
        data.racket.examples.push({ player, time_s: videoTime(clipId, "label"), note });
      }
      renderLists(section, clipId);
      updateMarkers(section, clipId);
      qsa("[data-remove]", section).forEach(remove => remove.addEventListener("click", () => removeListItem(section, clipId, remove)));
    }

    function removeListItem(section, clipId, btn) {
      const arr = getPath(state.clips[clipId], btn.dataset.remove);
      arr.splice(Number(btn.dataset.index), 1);
      renderLists(section, clipId);
      qsa("[data-remove]", section).forEach(remove => remove.addEventListener("click", () => removeListItem(section, clipId, remove)));
    }

    function captureNetPoint(event, clipId, layer) {
      const point = layer.dataset.point;
      if (!point) return;
      const video = clipVideos.get(clipId + ":calibration");
      if (!video || !video.videoWidth || !video.videoHeight) {
        qs("#saveStatus").textContent = "Video dimensions not ready. Press play/pause once, then click again.";
        return;
      }
      const rect = video.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width * video.videoWidth;
      const y = (event.clientY - rect.top) / rect.height * video.videoHeight;
      state.clips[clipId].top_net[point] = {
        x: Number(x.toFixed(2)),
        y: Number(y.toFixed(2)),
        time_s: Number(video.currentTime.toFixed(3)),
        video_width: video.videoWidth,
        video_height: video.videoHeight
      };
      layer.classList.remove("active");
      layer.dataset.point = "";
      const section = qs(`.clip[data-clip="${clipId}"]`);
      updateMarkers(section, clipId);
      qs("#saveStatus").textContent = point + " top-net point saved for " + clipId + ".";
    }

    function updateMarkersAll() {
      qsa(".clip").forEach(section => updateMarkers(section, section.dataset.clip));
    }
    function updateMarkers(section, clipId) {
      const layer = qs(`[data-click-layer="${clipId}"]`, section);
      if (!layer) return;
      const video = clipVideos.get(clipId + ":calibration");
      layer.querySelectorAll(".marker").forEach(m => m.remove());
      const data = state.clips[clipId];
      for (const key of ["left", "right"]) {
        const pt = data.top_net[key];
        if (!pt || !video || !video.videoWidth || !video.videoHeight) continue;
        const marker = document.createElement("div");
        marker.className = "marker";
        marker.title = key;
        marker.style.left = (pt.x / video.videoWidth * 100) + "%";
        marker.style.top = (pt.y / video.videoHeight * 100) + "%";
        layer.appendChild(marker);
      }
      const status = qs("#netStatus-" + clipId);
      if (status) status.textContent = topNetStatus(data);
    }

    async function save() {
      const payload = {
        ...state,
        saved_from_browser_at: new Date().toISOString(),
        manifest_seen: {
          repo_root: manifest.repo_root,
          clips: manifest.clips.map(c => ({
            id: c.id,
            label_overlay: c.label_overlay.path,
            calibration_overlay: c.calibration_overlay.path,
            ball_overlay: c.ball_overlay.path,
            track_overlay: c.track_overlay.path
          }))
        }
      };
      const response = await fetch("/api/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "save failed");
      qs("#saveStatus").textContent = "Saved: " + result.latest_path;
    }

    async function init() {
      manifest = await (await fetch("/api/manifest")).json();
      state = manifest.latest_save || defaultState(manifest);
      for (const clip of manifest.clips) {
        if (!state.clips[clip.id]) state.clips[clip.id] = blankClip();
      }
      bindGlobalForm();
      renderAll();
      qs("#save").addEventListener("click", () => save().catch(err => qs("#saveStatus").textContent = err.message));
      qs("#loadLatest").addEventListener("click", async () => {
        manifest = await (await fetch("/api/manifest")).json();
        state = manifest.latest_save || defaultState(manifest);
        renderAll();
        qs("#saveStatus").textContent = manifest.latest_save ? "Latest loaded." : "No latest save yet.";
      });
    }
    init().catch(err => {
      document.body.innerHTML = "<pre style='padding:20px'>" + escapeHtml(err.stack || err.message) + "</pre>";
    });
  </script>
</body>
</html>
"""


WIZARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pickleball Review Wizard</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --paper: #fffdfa;
      --ink: #211b14;
      --muted: #75695d;
      --line: #d8cec0;
      --soft: #f8f4ec;
      --accent: #0f766e;
      --accent-dark: #0b5751;
      --amber: #a45218;
      --red: #ad2f22;
      --green: #177245;
      --shadow: 0 14px 32px rgba(41, 33, 24, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", serif;
      line-height: 1.35;
    }
    button, input, textarea, select { font: inherit; }
    .top {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(244, 241, 234, 0.97);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .top-inner {
      max-width: 1320px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 14px; margin-top: 3px; }
    .save-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
    .status { color: var(--muted); font-size: 13px; min-width: 250px; text-align: right; }
    .btn {
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      min-height: 38px;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
    }
    .btn:hover { border-color: #b5a897; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: white; font-weight: 750; }
    .btn.primary:hover { background: var(--accent-dark); }
    .btn.small { min-height: 32px; padding: 6px 9px; font-size: 13px; }
    .btn.choice {
      min-height: 64px;
      text-align: left;
      display: grid;
      gap: 3px;
      background: var(--soft);
      width: 100%;
    }
    .btn.choice strong { font-size: 15px; }
    .btn.choice span { color: var(--muted); font-size: 13px; }
    .btn.choice.selected { border-color: var(--accent); background: #eaf5ef; box-shadow: inset 0 0 0 1px var(--accent); }
    .layout {
      max-width: 1680px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 18px;
    }
    .rail {
      position: sticky;
      top: 84px;
      align-self: start;
      display: grid;
      gap: 14px;
    }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .steps { padding: 10px; display: grid; gap: 8px; }
    .step {
      border: 1px solid transparent;
      background: transparent;
      border-radius: 8px;
      padding: 10px;
      text-align: left;
      cursor: pointer;
      display: grid;
      grid-template-columns: 26px 1fr;
      gap: 8px;
      align-items: center;
      color: var(--muted);
    }
    .step-num {
      width: 24px;
      height: 24px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: #eee5d7;
      color: var(--ink);
      font-size: 13px;
      font-weight: 750;
    }
    .step.active { background: #24362f; color: white; }
    .step.active .step-num { background: white; color: #24362f; }
    .step.done .step-num { background: var(--accent); color: white; }
    .clip-list { padding: 10px; display: grid; gap: 8px; }
    .clip-btn {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 9px;
      cursor: pointer;
      text-align: left;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .clip-btn.active { border-color: var(--accent); background: #eaf5ef; box-shadow: inset 0 0 0 1px var(--accent); }
    .clip-btn .mini { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    .main {
      min-width: 0;
      display: grid;
      gap: 16px;
    }
    .task-head {
      padding: 16px;
      display: grid;
      gap: 8px;
    }
    .task-kicker { color: var(--accent-dark); font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
    .task-title { font-size: 28px; font-weight: 820; margin: 0; }
    .task-copy { color: var(--muted); max-width: 780px; font-size: 16px; }
    .task-grid {
      display: grid;
      grid-template-columns: minmax(680px, 1fr) 380px;
      gap: 16px;
      align-items: start;
    }
    .media-card { overflow: hidden; background: #120f0c; }
    .media-title {
      padding: 10px 12px;
      background: #2a241d;
      color: #fff7e8;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
    }
    .asset-path { color: #d9c5ab; overflow-wrap: anywhere; font-size: 12px; }
    .video-wrap { position: relative; background: #100d0b; }
    video {
      display: block;
      width: 100%;
      max-height: calc(100vh - 250px);
      min-height: 420px;
      object-fit: contain;
      background: #100d0b;
    }
    .missing {
      min-height: 260px;
      display: grid;
      place-content: center;
      color: #f0d7c8;
      text-align: center;
      padding: 22px;
      background: #100d0b;
    }
    .click-layer { position: absolute; inset: 0; pointer-events: none; }
    .click-layer.active { pointer-events: auto; cursor: crosshair; outline: 5px solid rgba(15,118,110,.9); outline-offset: -5px; }
    .marker {
      position: absolute;
      width: 17px;
      height: 17px;
      border-radius: 999px;
      background: var(--amber);
      border: 2px solid white;
      transform: translate(-50%, -50%);
      box-shadow: 0 1px 8px rgba(0,0,0,.5);
    }
    .marker.saved { background: var(--accent); }
    .target-card {
      border: 1px solid #c8b38e;
      background: #fff8e8;
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 8px;
    }
    .target-kicker { color: var(--accent-dark); font-weight: 850; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .target-card h3 { margin: 0; font-size: 22px; line-height: 1.1; }
    .click-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .click-actions .btn.primary { grid-column: 1 / -1; min-height: 52px; }
    .target-list { display: grid; gap: 7px; max-height: 300px; overflow: auto; padding-right: 2px; }
    .target-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 8px;
      font-size: 13px;
    }
    .target-row.active { border-color: var(--accent); background: #eaf5ef; box-shadow: inset 0 0 0 1px var(--accent); }
    .target-row button { border: 0; background: transparent; color: var(--accent-dark); font-weight: 800; cursor: pointer; text-align: right; }
    .status-pill {
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 800;
      background: #eee5d7;
      color: #473b2d;
      white-space: nowrap;
    }
    .status-pill.good { background: #dff2e8; color: var(--green); }
    .status-pill.bad { background: #f7dfd6; color: var(--red); }
    .work {
      padding: 16px;
      display: grid;
      gap: 14px;
      align-self: start;
      position: sticky;
      top: 92px;
    }
    .callout {
      border: 1px solid #e4d3bd;
      background: #fff7e8;
      border-radius: 8px;
      padding: 12px;
      color: #4d3824;
    }
    .callout strong { display: block; margin-bottom: 4px; }
    .field { display: grid; gap: 5px; color: var(--muted); font-size: 13px; }
    input[type="text"], textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      min-height: 38px;
    }
    textarea { min-height: 86px; resize: vertical; }
    .choices { display: grid; gap: 9px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 7px; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
      padding: 6px 9px;
      cursor: pointer;
      font-size: 13px;
    }
    .chip.selected { background: #eaf5ef; border-color: var(--accent); color: var(--accent-dark); font-weight: 750; }
    .list { display: grid; gap: 7px; }
    .item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      font-size: 13px;
    }
    .item button { border: 0; background: transparent; color: var(--red); font-weight: 800; cursor: pointer; }
    .nav-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      padding: 14px 16px;
      border-top: 1px solid var(--line);
      background: #faf7f0;
    }
    .review-table { display: grid; gap: 8px; }
    .review-row {
      display: grid;
      grid-template-columns: 1.2fr repeat(6, auto);
      gap: 8px;
      align-items: center;
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      font-size: 13px;
    }
    .intake-section { display: grid; gap: 9px; }
    .intake-title { font-weight: 800; color: var(--ink); }
    .status-grid { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }
    .ok { color: var(--green); font-weight: 800; }
    .bad { color: var(--red); font-weight: 800; }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      .top-inner, .layout, .task-grid { grid-template-columns: 1fr; }
      .rail { position: static; }
      .status { text-align: left; min-width: 0; }
      .save-row { justify-content: flex-start; }
      .grid4, .grid2 { grid-template-columns: 1fr; }
      .review-row { grid-template-columns: 1fr; }
      .work { position: static; }
      video { min-height: 260px; max-height: 60vh; }
    }
  </style>
</head>
<body>
  <div class="top">
    <div class="top-inner">
      <div>
        <h1>Pickleball Review Wizard</h1>
        <div class="sub">Do one simple thing at a time. Most inputs are buttons or clicks on the video.</div>
      </div>
      <div class="save-row">
        <button class="btn" id="loadLatest">Load latest</button>
        <button class="btn primary" id="save">Save answers</button>
        <div class="status" id="saveStatus">Not saved yet.</div>
      </div>
    </div>
  </div>
  <main class="layout">
    <aside class="rail">
      <div class="panel steps" id="steps"></div>
      <div class="panel clip-list" id="clipList"></div>
    </aside>
    <section class="main">
      <div class="panel task-head">
        <div class="task-kicker" id="taskKicker"></div>
        <h2 class="task-title" id="taskTitle"></h2>
        <div class="task-copy" id="taskCopy"></div>
      </div>
      <div class="task-grid">
        <div class="panel media-card" id="mediaCard"></div>
        <div class="panel work" id="workCard"></div>
      </div>
      <div class="panel nav-row">
        <button class="btn" id="prevStep">Previous step</button>
        <div class="chips">
          <button class="btn small" id="prevClip">Previous clip</button>
          <button class="btn small" id="nextClip">Next clip</button>
        </div>
        <button class="btn primary" id="nextStep">Next step</button>
      </div>
    </section>
  </main>

  <script>
    const TASKS = [
      { id: "setup", label: "Setup", kicker: "Step 1", title: "Make three project decisions", copy: "Click the recommended choices unless you strongly want something else. This tells me how to run the next pass." },
      { id: "intake", label: "Data needed", kicker: "Step 2", title: "Source data checklist", copy: "For each clip, record the files, court evidence, and contact data you can provide. The checklist is derived from the current pipeline artifacts." },
      { id: "players", label: "Players", kicker: "Step 3", title: "Identify the four real players", copy: "Watch the label video. Use the position buttons first. Add clothing only if it is obvious." },
      { id: "ball", label: "Ball", kicker: "Step 4", title: "Mark ball mistakes and contacts", copy: "Use the ball video. Pause where the ball jumps, disappears, or hits a paddle, then click the matching button." },
      { id: "net", label: "Top net", kicker: "Step 5", title: "Click the top net tape", copy: "Use the calibration video. Click the left and right visible ends of the top edge of the net tape." },
      { id: "events", label: "Events", kicker: "Step 6", title: "Add rally windows", copy: "Use the label video. Set a start, then move to the end and add the window. One or two good windows per clip is enough." },
      { id: "racket", label: "Racket", kicker: "Step 7", title: "Decide racket scope", copy: "Most likely skip racket for this wave. Only add examples if the paddle is clearly visible." },
      { id: "review", label: "Save", kicker: "Step 8", title: "Review and save", copy: "This shows what is complete. Click Save answers when you are done." }
    ];
    const POSITIONS = ["near-left", "near-right", "far-left", "far-right", "unclear"];
    let manifest = null;
    let state = null;
    let taskIndex = 0;
    let clipIndex = 0;
    let pendingWindowStart = null;
    let activeNetPoint = null;
    let activeClickTarget = null;
    const videos = new Map();

    const blankClip = () => ({
      reviewed_enough: false,
      court_overlay_ok: "unsure",
      top_net: { left: null, right: null, notes: "" },
      source_data: { source_video_path: "", available_label_files: "", data_owner: "", notes: "" },
      court_evidence: {
        near_nvz: "unsure",
        far_nvz: "unsure",
        near_centerline: "unsure",
        far_centerline: "unsure",
        top_net: "unsure",
        points: {},
        point_statuses: {},
        notes: ""
      },
      players: { P1: "", P2: "", P3: "", P4: "" },
      spectators_ignore: "",
      ball: { mistakes: [], notes: "" },
      contacts: [],
      event_windows: [],
      racket: { examples: [], notes: "" },
      general_notes: ""
    });

    function defaultState(data) {
      const clips = {};
      for (const clip of data.clips) clips[clip.id] = blankClip();
      return {
        schema_version: 1,
        review_type: "pickleball_cv_blocker_review",
        global: {
          long_clip_policy: data.defaults.long_clip_policy,
          custom_windows: data.defaults.custom_windows,
          artifact_source_of_truth: data.defaults.artifact_source_of_truth,
          artifact_notes: "",
          racket_policy: data.defaults.racket_policy,
          aruco_notes: ""
        },
        clips
      };
    }

    function normalizeClipState(raw) {
      const base = blankClip();
      if (!raw || typeof raw !== "object") return base;
      return {
        ...base,
        ...raw,
        top_net: { ...base.top_net, ...(raw.top_net || {}) },
        source_data: { ...base.source_data, ...(raw.source_data || {}) },
        court_evidence: {
          ...base.court_evidence,
          ...(raw.court_evidence || {}),
          points: { ...base.court_evidence.points, ...((raw.court_evidence || {}).points || {}) },
          point_statuses: {
            ...base.court_evidence.point_statuses,
            ...((raw.court_evidence || {}).point_statuses || {})
          }
        },
        players: { ...base.players, ...(raw.players || {}) },
        ball: { ...base.ball, ...(raw.ball || {}) },
        contacts: Array.isArray(raw.contacts) ? raw.contacts : base.contacts,
        event_windows: Array.isArray(raw.event_windows) ? raw.event_windows : base.event_windows,
        racket: { ...base.racket, ...(raw.racket || {}) }
      };
    }

    function qs(sel, root=document) { return root.querySelector(sel); }
    function qsa(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }
    function clip() { return manifest.clips[clipIndex]; }
    function clipState() { return state.clips[clip().id]; }
    function escapeHtml(str) {
      return String(str ?? "").replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
    }
    function fmtTime(t) {
      if (t === null || t === undefined || Number.isNaN(Number(t))) return "";
      const value = Number(t);
      const m = Math.floor(value / 60);
      const s = (value - m * 60).toFixed(2).padStart(5, "0");
      return m + ":" + s;
    }
    function videoTime(key) {
      const v = videos.get(key);
      return v ? Number(v.currentTime || 0) : 0;
    }
    function assetVideo(asset, key, clickable=false) {
      if (!asset || !asset.exists) {
        return `<div class="media-title"><span>Missing local asset</span><span class="asset-path">${escapeHtml(asset?.path || "")}</span></div><div class="missing">This video is not synced locally.<br>${escapeHtml(asset?.path || "")}</div>`;
      }
      return `<div class="media-title"><span>${escapeHtml(asset.path.split("/").slice(-1)[0])}</span><span class="asset-path">${escapeHtml(asset.path)}</span></div>
        <div class="video-wrap">
          <video controls preload="metadata" data-video-key="${key}" src="${asset.url}"></video>
          ${clickable ? '<div class="click-layer" id="videoClickLayer"></div>' : ''}
        </div>`;
    }
    function chooseButton(label, sub, selected, attrs) {
      return `<button class="btn choice ${selected ? "selected" : ""}" ${attrs || ""}><strong>${escapeHtml(label)}</strong><span>${escapeHtml(sub)}</span></button>`;
    }
    function selectOption(value, label, current) {
      return `<option value="${escapeHtml(value)}" ${current === value ? "selected" : ""}>${escapeHtml(label)}</option>`;
    }

    function render() {
      const task = TASKS[taskIndex];
      qs("#taskKicker").textContent = task.kicker + " of " + TASKS.length;
      qs("#taskTitle").textContent = task.title;
      qs("#taskCopy").textContent = task.copy;
      renderSteps();
      renderClips();
      renderTask(task.id);
      bindCommon();
    }

    function renderSteps() {
      qs("#steps").innerHTML = TASKS.map((task, idx) => `
        <button class="step ${idx === taskIndex ? "active" : ""} ${taskDone(task.id) ? "done" : ""}" data-step="${idx}">
          <span class="step-num">${idx + 1}</span><span>${escapeHtml(task.label)}</span>
        </button>`).join("");
      qsa("[data-step]").forEach(btn => btn.onclick = () => { taskIndex = Number(btn.dataset.step); render(); });
    }
    function renderClips() {
      qs("#clipList").innerHTML = manifest.clips.map((c, idx) => {
        const s = state.clips[c.id];
        const doneBits = [
          hasIntakeInput(s),
          s.players.P1 && s.players.P2 && s.players.P3 && s.players.P4,
          s.top_net.left && s.top_net.right,
          s.reviewed_enough
        ].filter(Boolean).length;
        return `<button class="clip-btn ${idx === clipIndex ? "active" : ""}" data-clip-index="${idx}">
          ${escapeHtml(c.id.replaceAll("_", " "))}
          <span class="mini">${doneBits}/4 basics done</span>
        </button>`;
      }).join("");
      qsa("[data-clip-index]").forEach(btn => btn.onclick = () => { clipIndex = Number(btn.dataset.clipIndex); render(); });
    }
    function taskDone(id) {
      if (!state) return false;
      if (id === "setup") return Boolean(state.global.long_clip_policy && state.global.artifact_source_of_truth && state.global.racket_policy);
      if (id === "review") return false;
      return manifest.clips.some(c => {
        const s = state.clips[c.id];
        if (id === "players") return s.players.P1 && s.players.P2 && s.players.P3 && s.players.P4;
        if (id === "intake") return hasIntakeInput(s);
        if (id === "ball") return s.contacts.length || s.ball.mistakes.length || s.ball.notes;
        if (id === "net") return s.top_net.left && s.top_net.right;
        if (id === "events") return s.event_windows.length;
        if (id === "racket") return state.global.racket_policy === "skip_this_wave" || s.racket.examples.length || s.racket.notes;
        return false;
      });
    }

    function renderTask(id) {
      videos.clear();
      activeNetPoint = null;
      if (id === "setup") return renderSetup();
      if (id === "intake") return renderIntake();
      if (id === "players") return renderPlayers();
      if (id === "ball") return renderBall();
      if (id === "net") return renderNet();
      if (id === "events") return renderEvents();
      if (id === "racket") return renderRacket();
      return renderReview();
    }

    function renderSetup() {
      qs("#mediaCard").innerHTML = `<div class="work">
        <div class="callout"><strong>What this page saves</strong>One JSON file with your choices, clicks, and timestamps. You can save halfway and come back later.</div>
        <div class="callout"><strong>Recommended defaults</strong>Fixed windows, H100 source of truth, skip racket for this wave. These unblock the next CV run fastest.</div>
      </div>`;
      qs("#workCard").innerHTML = `
        <div class="choices">
          ${chooseButton("Use fixed windows", "Fast enough to review now. Recommended.", state.global.long_clip_policy === "fixed_windows", 'data-global-choice="long_clip_policy" data-value="fixed_windows"')}
          ${chooseButton("Run full long clips", "More complete but much slower.", state.global.long_clip_policy === "full_sources", 'data-global-choice="long_clip_policy" data-value="full_sources"')}
          ${chooseButton("Use custom windows", "Use the exact windows typed below.", state.global.long_clip_policy === "custom_windows", 'data-global-choice="long_clip_policy" data-value="custom_windows"')}
          <label class="field">Custom windows<textarea id="customWindows">${escapeHtml(state.global.custom_windows || "")}</textarea></label>
          ${chooseButton("H100 is source of truth", "Recommended. Keep model artifacts on H100.", state.global.artifact_source_of_truth === "h100", 'data-global-choice="artifact_source_of_truth" data-value="h100"')}
          ${chooseButton("Local is source of truth", "Use local artifacts as canonical.", state.global.artifact_source_of_truth === "local", 'data-global-choice="artifact_source_of_truth" data-value="local"')}
        </div>`;
      qs("#customWindows").oninput = e => state.global.custom_windows = e.target.value;
      qsa("[data-global-choice]").forEach(btn => btn.onclick = () => { state.global[btn.dataset.globalChoice] = btn.dataset.value; render(); });
    }

    function hasIntakeInput(s) {
      const source = s.source_data || {};
      const evidence = s.court_evidence || {};
      const points = evidence.points || {};
      const statuses = evidence.point_statuses || {};
      return Boolean(
        source.source_video_path ||
        source.available_label_files ||
        source.data_owner ||
        source.notes ||
        evidence.notes ||
        Object.keys(points).length ||
        Object.keys(statuses).length ||
        ["near_nvz", "far_nvz", "near_centerline", "far_centerline", "top_net"].some(key => evidence[key] && evidence[key] !== "unsure")
      );
    }

    function renderIntake() {
      const c = clip();
      const s = clipState();
      const intake = c.intake_requirements || {};
      const court = intake.court_evidence || {};
      const contacts = intake.contacts || {};
      const labels = intake.labels || {};
      const policy = intake.court_review_policy || { court_review_enabled: true };
      const missingLines = court.missing_required_line_ids || [];
      const missingNet = court.missing_required_net_ids || [];
      const blockers = intake.review_action_blockers || [];
      const targets = courtClickTargets(c);
      const current = currentClickTarget(c, s);
      qs("#mediaCard").innerHTML = assetVideo(c.calibration_overlay, "calibration", policy.court_review_enabled !== false);
      if (policy.court_review_enabled === false) {
        qs("#workCard").innerHTML = `
          <div class="target-card">
            <div class="target-kicker">Retired for court</div>
            <h3>Skip court clicks for this clip</h3>
            <div class="muted">${escapeHtml(policy.reason || "This clip is not suitable for court calibration review.")}</div>
          </div>
          <div class="callout"><strong>Allowed use</strong>${escapeHtml(policy.allowed_use || "Non-court review only.")}</div>
          <div class="intake-section">
            <div class="intake-title">Contact windows</div>
            ${contactStatusGrid(contacts)}
          </div>
          <div class="intake-section">
            <div class="intake-title">Label files</div>
            <div class="target-list">${renderLabelChecklist(labels)}</div>
          </div>`;
        bindVideos();
        return;
      }
      qs("#workCard").innerHTML = `
        <div class="target-card">
          <div class="target-kicker">Click target</div>
          <h3>${current ? escapeHtml(current.label) : "No court clicks needed"}</h3>
          <div class="muted">${current ? escapeHtml(current.prompt) : "This clip has no remaining court target in the current queue."}</div>
        </div>
        <div class="click-actions">
          <button class="btn primary" id="activateClickTarget">${current ? "Click this point on the video" : "Court queue complete"}</button>
          <button class="btn" data-point-status="not_visible">Not visible</button>
          <button class="btn" data-point-status="missing">Missing/wrong</button>
          <button class="btn" data-point-status="unsure">Unsure</button>
          <button class="btn" id="clearCurrentPoint">Clear current</button>
        </div>
        <div class="intake-section">
          <div class="intake-title">court_evidence</div>
          <div class="status-grid">
            <span>Auto-calibration ready</span>
            <span class="${court.auto_calibration_ready ? "ok" : "bad"}">${court.auto_calibration_ready ? "ready" : "blocked"}</span>
            <span>Missing required lines</span>
            <span class="${missingLines.length ? "bad" : "ok"}">${escapeHtml(missingLines.join(", ") || "none")}</span>
            <span>Missing required net cues</span>
            <span class="${missingNet.length ? "bad" : "ok"}">${escapeHtml(missingNet.join(", ") || "none")}</span>
          </div>
        </div>
        <div class="intake-section">
          <div class="intake-title">Target queue</div>
          <div class="target-list">${renderClickTargetQueue(c, s, targets, current)}</div>
        </div>
        <div class="intake-section">
          <div class="intake-title">Contact windows</div>
          ${contactStatusGrid(contacts)}
        </div>
        <div class="intake-section">
          <div class="intake-title">Label files</div>
          <div class="target-list">${renderLabelChecklist(labels)}</div>
        </div>`;
      bindVideos();
      armCurrentClickTarget(c, current);
      qsa("[data-target-id]").forEach(btn => btn.onclick = () => {
        activeClickTarget = { clipId: c.id, targetId: btn.dataset.targetId };
        renderIntake();
      });
      qsa("[data-point-status]").forEach(btn => btn.onclick = () => setCurrentPointStatus(btn.dataset.pointStatus));
      qs("#activateClickTarget").onclick = () => {
        if (!current) return;
        activeClickTarget = { clipId: c.id, targetId: current.id };
        armCurrentClickTarget(c, current);
      };
      qs("#clearCurrentPoint").onclick = () => clearCurrentPoint(current);
    }

    function contactStatusGrid(contacts) {
      return `<div class="status-grid">
        <span>Canonical contacts</span>
        <span class="${contacts.canonical_contact_count ? "ok" : "bad"}">${escapeHtml(contacts.canonical_contact_count ?? 0)}</span>
        <span>Pending review candidates</span>
        <span class="${contacts.pending_review_count ? "bad" : "ok"}">${escapeHtml(contacts.pending_review_count ?? 0)}</span>
      </div>`;
    }

    function courtClickTargets(c) {
      const intake = c.intake_requirements || {};
      const policy = intake.court_review_policy || {};
      if (policy.court_review_enabled === false) return [];
      const court = intake.court_evidence || {};
      const missing = [...(court.missing_required_line_ids || []), ...(court.missing_required_net_ids || [])];
      const fallback = [...(court.required_line_ids || []), ...(court.required_net_ids || [])];
      const evidenceIds = Array.from(new Set((missing.length ? missing : fallback).filter(Boolean)));
      return evidenceIds.flatMap(id => {
        const label = targetEvidenceLabel(id);
        return [
          { id: id + ":a", evidenceId: id, endpoint: "a", label: label + " point A", prompt: "Click one clear end or point on " + label + "." },
          { id: id + ":b", evidenceId: id, endpoint: "b", label: label + " point B", prompt: "Click a second point on the same line/cue." }
        ];
      });
    }

    function targetEvidenceLabel(id) {
      return String(id || "").replaceAll("_", " ");
    }

    function targetStatus(s, target) {
      const statuses = s.court_evidence.point_statuses || {};
      const points = s.court_evidence.points || {};
      if (points[target.id]) return "clicked";
      return statuses[target.id] || "pending";
    }

    function targetDone(s, target) {
      return targetStatus(s, target) !== "pending";
    }

    function currentClickTarget(c, s) {
      const targets = courtClickTargets(c);
      if (!targets.length) return null;
      if (activeClickTarget && activeClickTarget.clipId === c.id) {
        const active = targets.find(target => target.id === activeClickTarget.targetId);
        if (active && !targetDone(s, active)) return active;
      }
      return targets.find(target => !targetDone(s, target)) || null;
    }

    function renderClickTargetQueue(c, s, targets, current) {
      if (!targets.length) return '<div class="muted">No court click targets for this clip.</div>';
      return targets.map(target => {
        const status = targetStatus(s, target);
        const good = status === "clicked";
        const bad = status === "missing" || status === "not_visible";
        return `<div class="target-row ${current && current.id === target.id ? "active" : ""}">
          <button data-target-id="${escapeHtml(target.id)}">${escapeHtml(target.label)}</button>
          <span class="status-pill ${good ? "good" : bad ? "bad" : ""}">${escapeHtml(statusLabel(status))}</span>
        </div>`;
      }).join("");
    }

    function statusLabel(status) {
      if (status === "not_visible") return "not visible";
      if (status === "missing") return "missing/wrong";
      if (status === "clicked") return "saved";
      return status || "pending";
    }

    function armCurrentClickTarget(c, target) {
      const layer = qs("#videoClickLayer");
      if (!layer || !target) return;
      activeClickTarget = { clipId: c.id, targetId: target.id };
      layer.classList.add("active");
      layer.onclick = captureVideoPoint;
      updateClickMarkers();
      qs("#saveStatus").textContent = "Click " + target.label + " on the video.";
    }

    function setCurrentPointStatus(status) {
      const c = clip();
      const s = clipState();
      const target = currentClickTarget(c, s);
      if (!target) return;
      s.court_evidence.point_statuses[target.id] = status;
      if (s.court_evidence.points) delete s.court_evidence.points[target.id];
      if (status === "not_visible" || status === "missing") s.court_evidence[target.evidenceId] = status;
      activeClickTarget = null;
      renderIntake();
      renderSteps();
      renderClips();
    }

    function clearCurrentPoint(target) {
      const s = clipState();
      if (!target) return;
      if (s.court_evidence.points) delete s.court_evidence.points[target.id];
      if (s.court_evidence.point_statuses) delete s.court_evidence.point_statuses[target.id];
      s.court_evidence[target.evidenceId] = "unsure";
      activeClickTarget = { clipId: clip().id, targetId: target.id };
      renderIntake();
      renderSteps();
      renderClips();
    }

    function captureVideoPoint(event) {
      const c = clip();
      const s = clipState();
      const target = currentClickTarget(c, s);
      if (!target) return;
      const point = videoPointFromEvent(event, "calibration");
      if (!point) return;
      s.court_evidence.points[target.id] = {
        ...point,
        target_id: target.id,
        evidence_id: target.evidenceId,
        endpoint: target.endpoint,
        label: target.label,
        status: "clicked"
      };
      s.court_evidence.point_statuses[target.id] = "clicked";
      s.court_evidence[target.evidenceId] = "confirmed";
      activeClickTarget = null;
      qs("#saveStatus").textContent = target.label + " saved.";
      renderIntake();
      renderSteps();
      renderClips();
    }

    function videoPointFromEvent(event, videoKey) {
      const video = videos.get(videoKey);
      if (!video || !video.videoWidth || !video.videoHeight) {
        qs("#saveStatus").textContent = "Press play/pause once so the video loads, then click again.";
        return null;
      }
      const rect = video.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width * video.videoWidth;
      const y = (event.clientY - rect.top) / rect.height * video.videoHeight;
      return {
        x: Number(x.toFixed(2)),
        y: Number(y.toFixed(2)),
        time_s: Number(video.currentTime.toFixed(3)),
        video_width: video.videoWidth,
        video_height: video.videoHeight
      };
    }

    function updateClickMarkers() {
      const layer = qs("#videoClickLayer");
      const video = videos.get("calibration");
      if (!layer || !video) return;
      layer.querySelectorAll(".marker").forEach(m => m.remove());
      const points = clipState().court_evidence.points || {};
      Object.values(points).forEach(point => {
        if (!point || !video.videoWidth || !video.videoHeight) return;
        const m = document.createElement("div");
        m.className = "marker saved";
        m.title = point.label || point.target_id || "saved point";
        m.style.left = (point.x / video.videoWidth * 100) + "%";
        m.style.top = (point.y / video.videoHeight * 100) + "%";
        layer.appendChild(m);
      });
    }

    function blockerSummary(court, contacts, blockers) {
      const bits = [];
      if (court.needs_human_review) bits.push("court evidence needs human confirmation");
      if (contacts.needs_human_reviewed_contacts) bits.push("contact windows need accepted human-reviewed contacts");
      if (Array.isArray(blockers) && blockers.length) bits.push("review action blockers: " + blockers.join(", "));
      return bits.join(". ") || "No intake blockers detected for this clip.";
    }

    function renderLabelChecklist(labels) {
      const names = Object.keys(labels);
      if (!names.length) return '<div class="muted">No label manifest found.</div>';
      return names.map(name => {
        const item = labels[name] || {};
        const state = item.state || "unknown";
        const ok = state === "reviewed";
        return `<div class="item">
          <span><strong>${escapeHtml(name)}</strong><br><span class="muted">status: ${escapeHtml(item.status || "none")}</span></span>
          <span class="${ok ? "ok" : "bad"}">${escapeHtml(state)}</span>
        </div>`;
      }).join("");
    }

    function renderPlayers() {
      const c = clip();
      const s = clipState();
      qs("#mediaCard").innerHTML = assetVideo(c.label_overlay, "label");
      qs("#workCard").innerHTML = `
        <div class="callout"><strong>Your job</strong>For each real player, click the court position. Ignore exact tracking IDs.</div>
        ${["P1","P2","P3","P4"].map(pid => `
          <div class="field">
            <strong>${pid}</strong>
            <div class="chips">${POSITIONS.map(pos => `<button class="chip ${s.players[pid] === pos ? "selected" : ""}" data-player="${pid}" data-pos="${pos}">${pos}</button>`).join("")}</div>
          </div>`).join("")}
        <label class="field"><span><input type="checkbox" id="reviewedEnough" ${s.reviewed_enough ? "checked" : ""}> I reviewed this clip enough to use my answers</span></label>`;
      bindVideos();
      qsa("[data-player]").forEach(btn => btn.onclick = () => { s.players[btn.dataset.player] = btn.dataset.pos; renderPlayers(); renderSteps(); renderClips(); });
      qs("#reviewedEnough").onchange = e => { s.reviewed_enough = e.target.checked; renderSteps(); renderClips(); };
    }

    function renderBall() {
      const c = clip();
      const s = clipState();
      qs("#mediaCard").innerHTML = assetVideo(c.ball_overlay, "ball");
      qs("#workCard").innerHTML = `
        <div class="callout"><strong>Your job</strong>Pause the ball video where something happens. Then click one button.</div>
        <div class="grid2">
          <button class="btn choice" data-ball="bad_jump"><strong>Ball jumps here</strong><span>It snaps to the wrong place.</span></button>
          <button class="btn choice" data-ball="missing_ball"><strong>Ball missing here</strong><span>You can see the ball but overlay misses it.</span></button>
          <button class="btn choice" data-ball="false_ball"><strong>False ball here</strong><span>Overlay tracks a line/person/noise.</span></button>
          <button class="btn choice" data-ball="looks_good"><strong>This part looks good</strong><span>Optional positive note at current time.</span></button>
        </div>
        <div class="field"><strong>Add paddle contact at current time</strong><div class="chips">${["P1","P2","P3","P4","unknown"].map(p => `<button class="chip" data-contact="${p}">${p}</button>`).join("")}</div></div>
        <div><strong>Saved ball notes</strong><div class="list" id="ballList">${renderBallList(s)}</div></div>`;
      bindVideos();
      qsa("[data-ball]").forEach(btn => btn.onclick = () => { s.ball.mistakes.push({ kind: btn.dataset.ball, time_s: videoTime("ball"), note: "" }); renderBall(); renderSteps(); });
      qsa("[data-contact]").forEach(btn => btn.onclick = () => { s.contacts.push({ player: btn.dataset.contact, time_s: videoTime("ball"), note: "" }); renderBall(); renderSteps(); });
      bindRemovers();
    }

    function renderBallList(s) {
      const rows = [];
      s.ball.mistakes.forEach((item, i) => rows.push(`<div class="item"><span>${escapeHtml(item.kind)} @ ${fmtTime(item.time_s)}</span><button data-remove="mistake" data-index="${i}">x</button></div>`));
      s.contacts.forEach((item, i) => rows.push(`<div class="item"><span>${escapeHtml(item.player)} contact @ ${fmtTime(item.time_s)}</span><button data-remove="contact" data-index="${i}">x</button></div>`));
      return rows.join("") || '<div class="muted">Nothing marked yet.</div>';
    }

    function renderNet() {
      const c = clip();
      const s = clipState();
      qs("#mediaCard").innerHTML = assetVideo(c.calibration_overlay, "calibration", true);
      qs("#workCard").innerHTML = `
        <div class="callout"><strong>Your job</strong>Click the left and right ends of the top edge of the net tape. If you cannot see it, mark not visible.</div>
        <div class="grid2">
          <button class="btn choice" data-net-point="left"><strong>1. Click left top-net end</strong><span>${s.top_net.left ? pointText(s.top_net.left) : "not set"}</span></button>
          <button class="btn choice" data-net-point="right"><strong>2. Click right top-net end</strong><span>${s.top_net.right ? pointText(s.top_net.right) : "not set"}</span></button>
          ${chooseButton("Court overlay looks OK", "Green lines align well enough.", s.court_overlay_ok === "yes", 'data-court="yes"')}
          ${chooseButton("Court overlay is wrong", "Lines visibly miss the court.", s.court_overlay_ok === "no", 'data-court="no"')}
        </div>
        <button class="btn" id="netInvisible">Top net not visible / impossible to tell</button>
        <button class="btn" id="clearNet">Clear net clicks</button>`;
      bindVideos();
      updateNetMarkers();
      qsa("[data-net-point]").forEach(btn => btn.onclick = () => {
        activeNetPoint = btn.dataset.netPoint;
        const layer = qs("#videoClickLayer");
        if (layer) layer.classList.add("active");
        qs("#saveStatus").textContent = "Click the " + activeNetPoint + " top-net point on the video.";
      });
      qsa("[data-court]").forEach(btn => btn.onclick = () => { s.court_overlay_ok = btn.dataset.court; renderNet(); renderSteps(); });
      qs("#netInvisible").onclick = () => { s.top_net.notes = "Top net not visible / impossible to tell"; s.top_net.left = null; s.top_net.right = null; renderNet(); };
      qs("#clearNet").onclick = () => { s.top_net.left = null; s.top_net.right = null; renderNet(); renderSteps(); };
      const layer = qs("#videoClickLayer");
      if (layer) layer.onclick = captureNetPoint;
    }

    function pointText(pt) { return `(${Math.round(pt.x)}, ${Math.round(pt.y)})`; }

    function captureNetPoint(event) {
      if (!activeNetPoint) return;
      const video = videos.get("calibration");
      if (!video || !video.videoWidth || !video.videoHeight) {
        qs("#saveStatus").textContent = "Press play/pause once so the video loads, then click again.";
        return;
      }
      const rect = video.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width * video.videoWidth;
      const y = (event.clientY - rect.top) / rect.height * video.videoHeight;
      clipState().top_net[activeNetPoint] = {
        x: Number(x.toFixed(2)),
        y: Number(y.toFixed(2)),
        time_s: Number(video.currentTime.toFixed(3)),
        video_width: video.videoWidth,
        video_height: video.videoHeight
      };
      qs("#saveStatus").textContent = activeNetPoint + " top-net point saved.";
      activeNetPoint = null;
      renderNet();
      renderSteps();
      renderClips();
    }

    function updateNetMarkers() {
      const layer = qs("#videoClickLayer");
      const video = videos.get("calibration");
      if (!layer || !video) return;
      layer.querySelectorAll(".marker").forEach(m => m.remove());
      for (const key of ["left", "right"]) {
        const pt = clipState().top_net[key];
        if (!pt || !video.videoWidth || !video.videoHeight) continue;
        const m = document.createElement("div");
        m.className = "marker";
        m.style.left = (pt.x / video.videoWidth * 100) + "%";
        m.style.top = (pt.y / video.videoHeight * 100) + "%";
        layer.appendChild(m);
      }
    }

    function renderEvents() {
      const c = clip();
      const s = clipState();
      qs("#mediaCard").innerHTML = assetVideo(c.label_overlay, "label");
      qs("#workCard").innerHTML = `
        <div class="callout"><strong>Your job</strong>Pause at rally start, click start. Pause at rally end, click end. Repeat if you see another clean rally.</div>
        <div class="grid2">
          <button class="btn choice" id="setStart"><strong>Set rally start now</strong><span>${pendingWindowStart !== null ? fmtTime(pendingWindowStart) : "no start set"}</span></button>
          <button class="btn choice" id="setEnd"><strong>Add rally end now</strong><span>Creates a saved window.</span></button>
        </div>
        <div class="list" id="eventList">${renderEventList(s)}</div>`;
      bindVideos();
      qs("#setStart").onclick = () => { pendingWindowStart = videoTime("label"); renderEvents(); };
      qs("#setEnd").onclick = () => {
        const end = videoTime("label");
        const start = pendingWindowStart ?? end;
        s.event_windows.push({ start_s: Math.min(start, end), end_s: Math.max(start, end), note: "" });
        pendingWindowStart = null;
        renderEvents();
        renderSteps();
      };
      bindRemovers();
    }

    function renderEventList(s) {
      return s.event_windows.map((item, i) => `<div class="item"><span>rally ${fmtTime(item.start_s)} - ${fmtTime(item.end_s)}</span><button data-remove="window" data-index="${i}">x</button></div>`).join("") || '<div class="muted">No rally windows yet.</div>';
    }

    function renderRacket() {
      const c = clip();
      const s = clipState();
      qs("#mediaCard").innerHTML = assetVideo(c.label_overlay, "label");
      qs("#workCard").innerHTML = `
        <div class="callout"><strong>Recommended</strong>Skip racket for this wave unless you clearly see a paddle face. Bad examples do not help.</div>
        <div class="choices">
          ${chooseButton("Skip racket for this wave", "Recommended.", state.global.racket_policy === "skip_this_wave", 'data-racket-policy="skip_this_wave"')}
          ${chooseButton("I see clear paddle examples", "Add visible moments below.", state.global.racket_policy === "approved_examples", 'data-racket-policy="approved_examples"')}
          ${chooseButton("I can provide ArUco GT", "Marker dictionary, ID, size, mount, time range.", state.global.racket_policy === "aruco_gt", 'data-racket-policy="aruco_gt"')}
        </div>
        <div class="field"><strong>Add visible paddle at current time</strong><div class="chips">${["P1","P2","P3","P4","unknown"].map(p => `<button class="chip" data-racket-player="${p}">${p}</button>`).join("")}</div></div>
        <div class="list">${renderRacketList(s)}</div>`;
      bindVideos();
      qsa("[data-racket-policy]").forEach(btn => btn.onclick = () => { state.global.racket_policy = btn.dataset.racketPolicy; renderRacket(); renderSteps(); });
      qsa("[data-racket-player]").forEach(btn => btn.onclick = () => { s.racket.examples.push({ player: btn.dataset.racketPlayer, time_s: videoTime("label"), note: "" }); state.global.racket_policy = "approved_examples"; renderRacket(); renderSteps(); });
      bindRemovers();
    }

    function renderRacketList(s) {
      return s.racket.examples.map((item, i) => `<div class="item"><span>${escapeHtml(item.player)} visible paddle @ ${fmtTime(item.time_s)}</span><button data-remove="racket" data-index="${i}">x</button></div>`).join("") || '<div class="muted">No racket examples added.</div>';
    }

    function renderReview() {
      qs("#mediaCard").innerHTML = `<div class="work">
        <div class="callout"><strong>Before saving</strong>Green means enough for me to continue. Red does not mean you failed; it just tells me what remains blocked.</div>
        <div class="review-table">${manifest.clips.map(c => reviewRow(c)).join("")}</div>
      </div>`;
      qs("#workCard").innerHTML = `
        <div class="choices">
          ${chooseButton("Save answers now", "Writes runs/review_inputs/pickleball_cv_review_latest.json", false, 'id="bigSave"')}
          ${chooseButton("Load latest saved answers", "Useful if you refreshed the page.", false, 'id="bigLoad"')}
        </div>
        <div class="callout"><strong>What I will do next</strong>Parse this JSON, apply top-net corrections, choose fixed/full windows, sync artifacts, and rerun the blocked stages.</div>`;
      qs("#bigSave").onclick = save;
      qs("#bigLoad").onclick = loadLatest;
    }

    function reviewRow(c) {
      const s = state.clips[c.id];
      const intake = hasIntakeInput(s);
      const players = s.players.P1 && s.players.P2 && s.players.P3 && s.players.P4;
      const net = s.top_net.left && s.top_net.right;
      return `<div class="review-row">
        <strong>${escapeHtml(c.id.replaceAll("_", " "))}</strong>
        <span class="${intake ? "ok" : "bad"}">data</span>
        <span class="${players ? "ok" : "bad"}">players</span>
        <span class="${s.ball.mistakes.length || s.contacts.length || s.ball.notes ? "ok" : "bad"}">ball</span>
        <span class="${net ? "ok" : "bad"}">top net</span>
        <span class="${s.event_windows.length ? "ok" : "bad"}">events</span>
        <span class="${s.reviewed_enough ? "ok" : "bad"}">reviewed</span>
      </div>`;
    }

    function bindVideos() {
      videos.clear();
      qsa("[data-video-key]").forEach(v => {
        videos.set(v.dataset.videoKey, v);
        v.onloadedmetadata = () => {
          if (TASKS[taskIndex].id === "net") updateNetMarkers();
          if (TASKS[taskIndex].id === "intake") updateClickMarkers();
        };
      });
    }

    function bindCommon() {
      qs("#prevStep").onclick = () => { taskIndex = Math.max(0, taskIndex - 1); render(); };
      qs("#nextStep").onclick = () => { taskIndex = Math.min(TASKS.length - 1, taskIndex + 1); render(); };
      qs("#prevClip").onclick = () => { clipIndex = (clipIndex + manifest.clips.length - 1) % manifest.clips.length; render(); };
      qs("#nextClip").onclick = () => { clipIndex = (clipIndex + 1) % manifest.clips.length; render(); };
      qs("#save").onclick = save;
      qs("#loadLatest").onclick = loadLatest;
    }

    function bindRemovers() {
      qsa("[data-remove]").forEach(btn => btn.onclick = () => {
        const s = clipState();
        const idx = Number(btn.dataset.index);
        if (btn.dataset.remove === "mistake") s.ball.mistakes.splice(idx, 1);
        if (btn.dataset.remove === "contact") s.contacts.splice(idx, 1);
        if (btn.dataset.remove === "window") s.event_windows.splice(idx, 1);
        if (btn.dataset.remove === "racket") s.racket.examples.splice(idx, 1);
        render();
      });
    }

    async function save() {
      const payload = {
        ...state,
        saved_from_browser_at: new Date().toISOString(),
        manifest_seen: {
          repo_root: manifest.repo_root,
          clips: manifest.clips.map(c => ({
            id: c.id,
            label_overlay: c.label_overlay.path,
            calibration_overlay: c.calibration_overlay.path,
            ball_overlay: c.ball_overlay.path,
            track_overlay: c.track_overlay.path
          }))
        }
      };
      const response = await fetch("/api/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "save failed");
      qs("#saveStatus").textContent = "Saved: " + result.latest_path;
    }

    async function loadLatest() {
      manifest = await (await fetch("/api/manifest")).json();
      const defaults = defaultState(manifest);
      state = manifest.latest_save || defaults;
      state.global = { ...defaults.global, ...(state.global || {}) };
      if (!state.clips) state.clips = {};
      for (const c of manifest.clips) state.clips[c.id] = normalizeClipState(state.clips[c.id]);
      qs("#saveStatus").textContent = manifest.latest_save ? "Latest saved answers loaded." : "No saved answers yet.";
      render();
    }

    async function init() {
      manifest = await (await fetch("/api/manifest")).json();
      const defaults = defaultState(manifest);
      state = manifest.latest_save || defaults;
      state.global = { ...defaults.global, ...(state.global || {}) };
      if (!state.clips) state.clips = {};
      for (const c of manifest.clips) state.clips[c.id] = normalizeClipState(state.clips[c.id]);
      render();
    }
    init().catch(err => {
      document.body.innerHTML = "<pre style='padding:20px'>" + escapeHtml(err.stack || err.message) + "</pre>";
    });
  </script>
</body>
</html>
"""


PADDLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paddle Corner and Tracking Review</title>
  <style>
    :root {
      --bg: #f3f1eb;
      --paper: #fffdfa;
      --ink: #1f1b16;
      --muted: #6f665c;
      --line: #d7cfc4;
      --soft: #f8f4ec;
      --accent: #0f766e;
      --accent-dark: #0b5751;
      --warn: #a65318;
      --bad: #a9281c;
      --good: #177245;
      --shadow: 0 12px 28px rgba(35, 30, 23, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-serif, Georgia, "Times New Roman", serif;
      line-height: 1.35;
    }
    button, select, textarea { font: inherit; }
    .top {
      position: sticky;
      top: 0;
      z-index: 40;
      background: rgba(243, 241, 235, .97);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .top-inner {
      max-width: 1700px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 14px; margin-top: 3px; }
    .btn {
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 8px;
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .btn:hover { border-color: #b4a899; }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: white; font-weight: 760; }
    .btn.primary:hover { background: var(--accent-dark); }
    .btn.small { min-height: 32px; padding: 6px 9px; font-size: 13px; }
    .btn.choice {
      width: 100%;
      min-height: 58px;
      text-align: left;
      display: grid;
      gap: 3px;
      background: var(--soft);
    }
    .btn.choice.selected { border-color: var(--accent); background: #e8f4ee; box-shadow: inset 0 0 0 1px var(--accent); }
    .btn.choice strong { font-size: 15px; }
    .btn.choice span { color: var(--muted); font-size: 13px; }
    .save-row { display: flex; gap: 10px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .status { min-width: 260px; text-align: right; color: var(--muted); font-size: 13px; }
    .layout {
      max-width: 1700px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: 210px minmax(0, 1fr);
      gap: 18px;
    }
    .rail {
      position: sticky;
      top: 86px;
      align-self: start;
      display: grid;
      gap: 12px;
    }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .tabs, .clip-list { padding: 10px; display: grid; gap: 8px; }
    .tab, .clip-btn {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 9px;
      cursor: pointer;
      text-align: left;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .tab.active, .clip-btn.active { border-color: var(--accent); background: #e8f4ee; box-shadow: inset 0 0 0 1px var(--accent); }
    .mini { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    .main { min-width: 0; display: grid; gap: 16px; }
    .task-head { padding: 16px; display: grid; gap: 6px; }
    .kicker { color: var(--accent-dark); font-size: 13px; font-weight: 850; text-transform: uppercase; letter-spacing: .04em; }
    .title { font-size: 28px; font-weight: 840; margin: 0; }
    .copy { color: var(--muted); max-width: 920px; font-size: 16px; }
    .work-grid {
      display: grid;
      grid-template-columns: minmax(680px, 1fr) 390px;
      gap: 16px;
      align-items: start;
    }
    .media { overflow: hidden; background: #120f0c; }
    .media-stack {
      display: grid;
      gap: 14px;
      background: transparent;
      border: 0;
      box-shadow: none;
    }
    .media-head {
      padding: 10px 12px;
      background: #2a241d;
      color: #fff7e8;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
    }
    .asset-path { color: #d7c2a9; overflow-wrap: anywhere; font-size: 12px; }
    .sheet-wrap, .video-wrap, .source-wrap { position: relative; background: #100d0b; }
    .sheet {
      display: block;
      width: 100%;
      max-height: calc(100vh - 220px);
      object-fit: contain;
      background: #100d0b;
      cursor: crosshair;
    }
    .source-video {
      display: block;
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      background: #100d0b;
    }
    .tile-box {
      position: absolute;
      border: 3px solid rgba(15,118,110,.9);
      box-shadow: 0 0 0 9999px rgba(0,0,0,.32);
      pointer-events: none;
    }
    .crop-context-box {
      position: absolute;
      border: 4px solid rgba(255,214,10,.95);
      box-shadow: 0 0 0 9999px rgba(0,0,0,.18), 0 0 16px rgba(255,214,10,.9);
      pointer-events: none;
      display: none;
    }
    .crop-context-box.visible { display: block; }
    .marker {
      position: absolute;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      background: var(--warn);
      border: 2px solid white;
      transform: translate(-50%, -50%);
      box-shadow: 0 1px 8px rgba(0,0,0,.55);
      color: white;
      font-size: 10px;
      display: grid;
      place-items: center;
      pointer-events: none;
    }
    video, .montage {
      display: block;
      width: 100%;
      max-height: 560px;
      object-fit: contain;
      background: #100d0b;
    }
    .missing {
      min-height: 320px;
      display: grid;
      place-content: center;
      color: #f0d6c8;
      text-align: center;
      padding: 24px;
      background: #100d0b;
    }
    .side { padding: 16px; display: grid; gap: 14px; position: sticky; top: 88px; align-self: start; }
    .callout {
      border: 1px solid #e3d2bb;
      background: #fff7e8;
      border-radius: 8px;
      padding: 12px;
      color: #4d3825;
    }
    .callout strong { display: block; margin-bottom: 4px; }
    .target {
      border: 1px solid #c8b38e;
      background: #fff8e8;
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 7px;
    }
    .target h3 { margin: 0; font-size: 22px; line-height: 1.1; }
    .target .meta { color: var(--muted); font-size: 13px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .choices { display: grid; gap: 9px; }
    .list { display: grid; gap: 7px; max-height: 300px; overflow: auto; padding-right: 2px; }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 8px;
      font-size: 13px;
    }
    .row.active { border-color: var(--accent); background: #e8f4ee; box-shadow: inset 0 0 0 1px var(--accent); }
    .pill {
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 820;
      background: #eee5d7;
      color: #473b2d;
      white-space: nowrap;
    }
    .pill.good { background: #dff2e8; color: var(--good); }
    .pill.bad { background: #f7dfd6; color: var(--bad); }
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      min-height: 72px;
      resize: vertical;
    }
    .tracking-list { display: grid; gap: 16px; }
    .tracking-card { display: grid; grid-template-columns: minmax(420px, 1fr) 360px; gap: 14px; padding: 14px; }
    .tracking-media { display: grid; gap: 12px; }
    .muted { color: var(--muted); }
    @media (max-width: 1050px) {
      .top-inner, .layout, .work-grid, .tracking-card { grid-template-columns: 1fr; }
      .rail, .side { position: static; }
      .save-row { justify-content: flex-start; }
      .status { text-align: left; min-width: 0; }
    }
  </style>
</head>
<body>
  <div class="top">
    <div class="top-inner">
      <div>
        <h1>Paddle Corner and Tracking Review</h1>
        <div class="sub">Click actual paddle-face corners, or mark the candidate not visible/ambiguous. Then answer the tracking-video promotion question.</div>
      </div>
      <div class="save-row">
        <button class="btn" id="loadLatest">Load latest</button>
        <button class="btn primary" id="save">Save answers</button>
        <div class="status" id="saveStatus">Not saved yet.</div>
      </div>
    </div>
  </div>
  <main class="layout">
    <aside class="rail">
      <div class="panel tabs">
        <button class="tab active" data-mode="paddle">Paddle corners<span class="mini">click 4 points or skip</span></button>
        <button class="tab" data-mode="tracking">Tracking videos<span class="mini">safe / unsafe / unsure</span></button>
      </div>
      <div class="panel clip-list" id="clipList"></div>
    </aside>
    <section class="main">
      <div class="panel task-head">
        <div class="kicker" id="taskKicker">Paddle true-corner labels</div>
        <h2 class="title" id="taskTitle">Click true paddle-face corners</h2>
        <div class="copy" id="taskCopy">For each crop, click the real face corners in order: top-left, top-right, bottom-right, bottom-left. Do not click the yellow bbox corners unless they are truly the paddle face corners.</div>
      </div>
      <div id="content"></div>
    </section>
  </main>
  <script>
    let manifest = null;
    let state = null;
    let mode = "paddle";
    let clipIndex = 0;
    let itemIndexByClip = {};
    let cornerIndexByClip = {};
    const CORNER_LABELS = { top_left: "top-left", top_right: "top-right", bottom_right: "bottom-right", bottom_left: "bottom-left" };

    function qs(sel, root=document) { return root.querySelector(sel); }
    function qsa(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }
    function escapeHtml(str) {
      return String(str ?? "").replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
    }
    function clip() { return manifest.clips[clipIndex]; }
    function paddle() { return clip().paddle_corner_review || { task_items: [], corner_order: [] }; }
    function currentItemIndex() { return itemIndexByClip[clip().id] || 0; }
    function currentCornerIndex() { return cornerIndexByClip[clip().id] || 0; }
    function setCurrentItemIndex(value) { itemIndexByClip[clip().id] = Math.max(0, Math.min(value, Math.max(0, paddle().task_items.length - 1))); }
    function setCurrentCornerIndex(value) { cornerIndexByClip[clip().id] = Math.max(0, Math.min(value, Math.max(0, paddle().corner_order.length - 1))); }
    function currentItem() { return (paddle().task_items || [])[currentItemIndex()] || null; }
    function currentCorner() { return (paddle().corner_order || [])[currentCornerIndex()] || "top_left"; }
    function labelBucket(clipId) {
      state.paddle_corner_labels[clipId] ||= { artifact_type: "paddle_true_corner_labels", labels: {} };
      return state.paddle_corner_labels[clipId];
    }
    function labelEntry(clipId, item) {
      const bucket = labelBucket(clipId);
      bucket.labels[item.review_id] ||= {
        review_id: item.review_id,
        player_id: item.player_id,
        frame_index: item.frame_index,
        t: item.t,
        crop_xyxy: item.crop_xyxy,
        status: "pending",
        evidence_type: "true_corners",
        reviewer: "local_click_review",
        corners_px_order: item.corner_order,
        corners_by_name: {},
        sheet_points_by_name: {}
      };
      return bucket.labels[item.review_id];
    }
    function entryStatus(item) {
      const entry = labelBucket(clip().id).labels[item.review_id];
      if (!entry) return "pending";
      return entry.status || "pending";
    }
    function stateFromManifest(data) {
      const existing = data.latest_save && typeof data.latest_save === "object" ? data.latest_save : {};
      const next = {
        ...existing,
        schema_version: Math.max(Number(existing.schema_version || 1), 2),
        review_type: existing.review_type || "pickleball_cv_blocker_review",
        paddle_corner_labels: existing.paddle_corner_labels || {},
        tracking_video_review: existing.tracking_video_review || {}
      };
      for (const c of data.clips) {
        next.paddle_corner_labels[c.id] ||= { artifact_type: "paddle_true_corner_labels", labels: {} };
      }
      for (const item of data.human_review_tasks.tracking_video_review.items) {
        next.tracking_video_review[item.clip] ||= { decision: "", notes: "" };
      }
      return next;
    }
    function assetImage(asset, className) {
      if (!asset || !asset.exists) return `<div class="missing">Missing local asset<br><span class="asset-path">${escapeHtml(asset?.path || "")}</span></div>`;
      return `<img class="${className}" id="cropSheet" src="${asset.url}" alt="">`;
    }
    function assetVideo(asset) {
      if (!asset || !asset.exists) return `<div class="missing">Missing local video<br><span class="asset-path">${escapeHtml(asset?.path || "")}</span></div>`;
      return `<video controls preload="metadata" src="${asset.url}"></video>`;
    }
    function render() {
      renderTabs();
      if (mode === "tracking") renderTracking();
      else renderPaddle();
    }
    function renderTabs() {
      qsa("[data-mode]").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
        btn.onclick = () => { mode = btn.dataset.mode; render(); };
      });
    }
    function renderClipList() {
      qs("#clipList").innerHTML = manifest.clips.map((c, index) => {
        const p = c.paddle_corner_review || {};
        const labels = Object.values((state.paddle_corner_labels[c.id] || {}).labels || {});
        const accepted = labels.filter(item => item.status === "accepted").length;
        return `<button class="clip-btn ${index === clipIndex ? "active" : ""}" data-clip="${index}">
          ${escapeHtml(c.id.replaceAll("_", " "))}
          <span class="mini">${accepted}/${p.queue_count || 0} paddle crops saved</span>
        </button>`;
      }).join("");
      qsa("[data-clip]").forEach(btn => btn.onclick = () => { clipIndex = Number(btn.dataset.clip); renderPaddle(); });
    }
    function renderPaddle() {
      const c = clip();
      const p = paddle();
      const item = currentItem();
      const corner = currentCorner();
      qs("#taskKicker").textContent = "Paddle candidate triage";
      qs("#taskTitle").textContent = "Reject false crops before corner clicks";
      qs("#taskCopy").textContent = "These are box-derived candidates, not trusted paddle detections. First use the source-frame rectangle to decide whether the crop is actually on a visible paddle face. Only then click true paddle-face corners.";
      renderClipList();
      qs("#content").innerHTML = `
        <div class="work-grid">
          <div class="media-stack">
            <div class="panel media" id="sourceFrameContext">
              <div class="media-head">
                <span>Source frame context</span>
                <span class="asset-path">${escapeHtml(p.source_video?.path || "")}</span>
              </div>
              <div class="source-wrap">
                ${sourceVideoContext(p.source_video)}
              </div>
            </div>
            <div class="panel media">
              <div class="media-head">
                <span>Crop sheet</span>
                <span class="asset-path">${escapeHtml(p.crop_sheet?.path || "")}</span>
              </div>
              <div class="sheet-wrap">
                ${assetImage(p.crop_sheet, "sheet")}
                <div id="sheetOverlay"></div>
              </div>
            </div>
          </div>
          <div class="panel side">
            <div class="callout"><strong>First decision</strong>If the yellow rectangle is not on a paddle face, click <strong>Not a paddle</strong>. If it is a real visible paddle face, click the four true corners on the crop sheet.</div>
            ${item ? targetPanel(item, corner) : '<div class="target"><h3>No paddle crop queue</h3><div class="meta">No rendered crop-sheet items found for this clip.</div></div>'}
            <div class="grid2">
              <button class="btn" id="prevItem">Previous crop</button>
              <button class="btn" id="nextItem">Next crop</button>
              <button class="btn" id="prevCorner">Previous corner</button>
              <button class="btn" id="nextCorner">Next corner</button>
            </div>
            <div class="grid2">
              <button class="btn" id="notPaddle">Not a paddle</button>
              <button class="btn" id="notVisible">Not visible</button>
              <button class="btn" id="ambiguous">Ambiguous</button>
              <button class="btn" id="clearItem">Clear crop</button>
            </div>
            <button class="btn primary" id="saveNow">Save answers</button>
            <div class="list" id="paddleCornerQueue">${queueRows()}</div>
          </div>
        </div>`;
      bindPaddle();
      updateSourceContext();
      updateSheetOverlay();
    }
    function sourceVideoContext(asset) {
      if (!asset || !asset.exists) {
        return `<div class="missing">Missing local source video<br><span class="asset-path">${escapeHtml(asset?.path || "")}</span></div>`;
      }
      return `<video controls preload="metadata" class="source-video" id="sourceVideo" src="${asset.url}"></video><div class="crop-context-box" id="cropContextBox"></div>`;
    }
    function targetPanel(item, corner) {
      const status = entryStatus(item);
      return `<div class="target">
        <div class="kicker">Current target</div>
        <h3>Check source rectangle first</h3>
        <div class="meta">Crop ${currentItemIndex() + 1} of ${paddle().task_items.length} | review_id ${escapeHtml(item.review_id)} | frame ${escapeHtml(item.frame_index)}</div>
        <div class="meta">If it is a real visible paddle face, next click: ${escapeHtml(CORNER_LABELS[corner] || corner)} corner.</div>
        <div class="meta">Status: ${escapeHtml(statusLabel(status))}</div>
      </div>`;
    }
    function queueRows() {
      const items = paddle().task_items || [];
      if (!items.length) return '<div class="muted">No queue items.</div>';
      return items.map((item, index) => {
        const status = entryStatus(item);
        return `<div class="row ${index === currentItemIndex() ? "active" : ""}">
          <span>${index + 1}. ${escapeHtml(item.review_id)}<br><span class="muted">frame ${escapeHtml(item.frame_index)}</span></span>
          <span class="pill ${status === "accepted" ? "good" : status === "not_paddle" || status === "not_visible" || status === "ambiguous" ? "bad" : ""}">${escapeHtml(statusLabel(status))}</span>
        </div>`;
      }).join("");
    }
    function statusLabel(status) {
      if (status === "not_visible") return "not visible";
      if (status === "not_paddle") return "not a paddle";
      if (status === "accepted") return "saved";
      return status || "pending";
    }
    function bindPaddle() {
      const img = qs("#cropSheet");
      if (img) {
        img.onclick = captureCornerClick;
        img.onload = updateSheetOverlay;
      }
      const sourceVideo = qs("#sourceVideo");
      if (sourceVideo) {
        sourceVideo.onloadedmetadata = updateSourceContext;
        sourceVideo.onloadeddata = updateCropContextBox;
        sourceVideo.onseeked = updateCropContextBox;
      }
      qs("#prevItem").onclick = () => { setCurrentItemIndex(currentItemIndex() - 1); setCurrentCornerIndex(0); renderPaddle(); };
      qs("#nextItem").onclick = () => { setCurrentItemIndex(currentItemIndex() + 1); setCurrentCornerIndex(0); renderPaddle(); };
      qs("#prevCorner").onclick = () => { setCurrentCornerIndex(currentCornerIndex() - 1); renderPaddle(); };
      qs("#nextCorner").onclick = () => { setCurrentCornerIndex(currentCornerIndex() + 1); renderPaddle(); };
      qs("#notPaddle").onclick = () => markCurrentItem("not_paddle");
      qs("#notVisible").onclick = () => markCurrentItem("not_visible");
      qs("#ambiguous").onclick = () => markCurrentItem("ambiguous");
      qs("#clearItem").onclick = clearCurrentItem;
      qs("#saveNow").onclick = save;
    }
    function captureCornerClick(event) {
      const item = currentItem();
      if (!item) return;
      const img = event.currentTarget;
      if (!img.naturalWidth || !img.naturalHeight) {
        qs("#saveStatus").textContent = "Image dimensions not ready. Try again.";
        return;
      }
      const rect = img.getBoundingClientRect();
      const sheetX = (event.clientX - rect.left) / rect.width * img.naturalWidth;
      const sheetY = (event.clientY - rect.top) / rect.height * img.naturalHeight;
      const [sx1, sy1, sx2, sy2] = item.sheet_rect;
      if (sheetX < sx1 || sheetX > sx2 || sheetY < sy1 || sheetY > sy2) {
        qs("#saveStatus").textContent = "Click inside the highlighted crop tile.";
        return;
      }
      const [cx1, cy1, cx2, cy2] = item.crop_xyxy;
      const frameX = cx1 + ((sheetX - sx1) / Math.max(1, sx2 - sx1)) * (cx2 - cx1);
      const frameY = cy1 + ((sheetY - sy1) / Math.max(1, sy2 - sy1)) * (cy2 - cy1);
      const entry = labelEntry(clip().id, item);
      const corner = currentCorner();
      entry.corners_by_name[corner] = [Number(frameX.toFixed(2)), Number(frameY.toFixed(2))];
      entry.sheet_points_by_name[corner] = [Number(sheetX.toFixed(2)), Number(sheetY.toFixed(2))];
      const order = item.corner_order || paddle().corner_order;
      const complete = order.every(name => entry.corners_by_name[name]);
      if (complete) {
        entry.status = "accepted";
        entry.corners_px = order.map(name => entry.corners_by_name[name]);
      } else {
        entry.status = "in_progress";
        setCurrentCornerIndex(currentCornerIndex() + 1);
      }
      qs("#saveStatus").textContent = complete ? "Crop corners saved." : "Corner saved. Continue with the next corner.";
      renderPaddle();
    }
    function markCurrentItem(status) {
      const item = currentItem();
      if (!item) return;
      const entry = labelEntry(clip().id, item);
      entry.status = status;
      entry.corners_by_name = {};
      entry.sheet_points_by_name = {};
      delete entry.corners_px;
      setCurrentItemIndex(currentItemIndex() + 1);
      setCurrentCornerIndex(0);
      renderPaddle();
    }
    function clearCurrentItem() {
      const item = currentItem();
      if (!item) return;
      delete labelBucket(clip().id).labels[item.review_id];
      setCurrentCornerIndex(0);
      renderPaddle();
    }
    function updateSourceContext() {
      const video = qs("#sourceVideo");
      const item = currentItem();
      if (!video || !item) return;
      const targetTime = Number(item.t || 0);
      if (video.readyState >= 1 && Number.isFinite(targetTime)) {
        const boundedTime = Math.max(0, Math.min(targetTime, Number.isFinite(video.duration) ? video.duration : targetTime));
        if (Math.abs(video.currentTime - boundedTime) > 0.05) {
          video.currentTime = boundedTime;
        }
      }
      updateCropContextBox();
    }
    function updateCropContextBox() {
      const box = qs("#cropContextBox");
      const video = qs("#sourceVideo");
      const item = currentItem();
      if (!box || !video || !item || !video.videoWidth || !video.videoHeight) return;
      const [x1, y1, x2, y2] = item.crop_xyxy || [];
      if (![x1, y1, x2, y2].every(Number.isFinite)) return;
      box.style.left = (x1 / video.videoWidth * 100) + "%";
      box.style.top = (y1 / video.videoHeight * 100) + "%";
      box.style.width = ((x2 - x1) / video.videoWidth * 100) + "%";
      box.style.height = ((y2 - y1) / video.videoHeight * 100) + "%";
      box.classList.add("visible");
    }
    function updateSheetOverlay() {
      const overlay = qs("#sheetOverlay");
      const img = qs("#cropSheet");
      const item = currentItem();
      if (!overlay || !img || !item || !img.naturalWidth || !img.naturalHeight) return;
      overlay.innerHTML = "";
      const [sx1, sy1, sx2, sy2] = item.sheet_rect;
      const box = document.createElement("div");
      box.className = "tile-box";
      box.style.left = (sx1 / img.naturalWidth * 100) + "%";
      box.style.top = (sy1 / img.naturalHeight * 100) + "%";
      box.style.width = ((sx2 - sx1) / img.naturalWidth * 100) + "%";
      box.style.height = ((sy2 - sy1) / img.naturalHeight * 100) + "%";
      overlay.appendChild(box);
      const entry = labelBucket(clip().id).labels[item.review_id];
      const points = entry?.sheet_points_by_name || {};
      for (const [name, point] of Object.entries(points)) {
        const marker = document.createElement("div");
        marker.className = "marker";
        marker.textContent = String((item.corner_order || []).indexOf(name) + 1);
        marker.style.left = (point[0] / img.naturalWidth * 100) + "%";
        marker.style.top = (point[1] / img.naturalHeight * 100) + "%";
        overlay.appendChild(marker);
      }
    }
    function renderTracking() {
      qs("#clipList").innerHTML = '<div class="muted" style="padding:10px">Two tracking videos need your decision.</div>';
      qs("#taskKicker").textContent = "Player tracking review";
      qs("#taskTitle").textContent = "Decide whether widened boxes are safe";
      qs("#taskCopy").textContent = "Watch each diagnostic video. I need to know if the boxes stay on the four real players, or if they catch spectators/background.";
      const review = manifest.human_review_tasks.tracking_video_review;
      qs("#content").innerHTML = `<div class="tracking-list" id="trackingReviewVideos">${review.items.map(renderTrackingItem).join("")}</div>`;
      bindTracking();
    }
    function renderTrackingItem(item) {
      const saved = state.tracking_video_review[item.clip] || { decision: "", notes: "" };
      return `<div class="panel tracking-card" data-track-clip="${escapeHtml(item.clip)}">
        <div class="tracking-media">
          <div class="media">
            <div class="media-head"><span>${escapeHtml(item.title)}</span><span class="asset-path">${escapeHtml(item.overlay_video.path || "")}</span></div>
            ${assetVideo(item.overlay_video)}
          </div>
          <div class="media">
            <div class="media-head"><span>Montage</span><span class="asset-path">${escapeHtml(item.montage.path || "")}</span></div>
            ${item.montage.exists ? `<img class="montage" src="${item.montage.url}" alt="">` : `<div class="missing">Missing montage<br>${escapeHtml(item.montage.path || "")}</div>`}
          </div>
        </div>
        <div class="side" style="position:static">
          <div class="target">
            <div class="kicker">What I need from you</div>
            <h3>${escapeHtml(item.question)}</h3>
            <div class="meta">${escapeHtml(item.needed_answer)}</div>
          </div>
          <div class="choices">
            ${trackingChoice(item.clip, "safe_to_promote", "Safe to promote", "Boxes stay on the four real players.", saved.decision)}
            ${trackingChoice(item.clip, "unsafe_background_or_spectators", "Unsafe", "I see spectators/background/non-players getting boxed.", saved.decision)}
            ${trackingChoice(item.clip, "unsure", "Unsure", "I cannot tell from this video.", saved.decision)}
          </div>
          <textarea data-track-notes="${escapeHtml(item.clip)}" placeholder="Optional notes">${escapeHtml(saved.notes || "")}</textarea>
        </div>
      </div>`;
    }
    function trackingChoice(clipId, value, label, sub, selected) {
      return `<button class="btn choice ${selected === value ? "selected" : ""}" data-track-choice="${escapeHtml(clipId)}" data-value="${escapeHtml(value)}"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(sub)}</span></button>`;
    }
    function bindTracking() {
      qsa("[data-track-choice]").forEach(btn => btn.onclick = () => {
        const clipId = btn.dataset.trackChoice;
        state.tracking_video_review[clipId] ||= { decision: "", notes: "" };
        state.tracking_video_review[clipId].decision = btn.dataset.value;
        renderTracking();
      });
      qsa("[data-track-notes]").forEach(area => area.oninput = () => {
        const clipId = area.dataset.trackNotes;
        state.tracking_video_review[clipId] ||= { decision: "", notes: "" };
        state.tracking_video_review[clipId].notes = area.value;
      });
    }
    async function save() {
      const payload = {
        ...state,
        saved_from_browser_at: new Date().toISOString(),
        manifest_seen: {
          repo_root: manifest.repo_root,
          focused_review_page: "/paddle",
          paddle_clips: manifest.clips.map(c => ({
            id: c.id,
            crop_sheet: c.paddle_corner_review?.crop_sheet?.path,
            candidate_overlay: c.paddle_corner_review?.candidate_overlay?.path
          })),
          tracking_videos: manifest.human_review_tasks.tracking_video_review.items.map(item => ({
            clip: item.clip,
            overlay_video: item.overlay_video.path,
            montage: item.montage.path
          }))
        }
      };
      const response = await fetch("/api/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "save failed");
      qs("#saveStatus").textContent = "Saved: " + result.latest_path;
    }
    async function loadLatest() {
      manifest = await (await fetch("/api/manifest")).json();
      state = stateFromManifest(manifest);
      qs("#saveStatus").textContent = manifest.latest_save ? "Latest saved answers loaded." : "No saved answers yet.";
      render();
    }
    async function init() {
      manifest = await (await fetch("/api/manifest")).json();
      state = stateFromManifest(manifest);
      qs("#save").onclick = () => save().catch(err => qs("#saveStatus").textContent = err.message);
      qs("#loadLatest").onclick = () => loadLatest().catch(err => qs("#saveStatus").textContent = err.message);
      render();
    }
    init().catch(err => {
      document.body.innerHTML = "<pre style='padding:20px'>" + escapeHtml(err.stack || err.message) + "</pre>";
    });
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "PickleballReviewInput/1.0"

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
            self._send_text(WIZARD_HTML)
            return
        if parsed.path == "/paddle":
            self._send_text(PADDLE_HTML)
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
            latest, timestamped = _write_review_input(self.root, payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"latest_path": _rel(latest, self.root), "timestamped_path": _rel(timestamped, self.root)})

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
        byte_range = _parse_byte_range(self.headers.get("Range"), size)
        if byte_range is not None:
            start, end = byte_range
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            _copy_file_range_to_writer(candidate, self.wfile, start, end)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
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
    parser = argparse.ArgumentParser(description="Serve a one-page local UI for pickleball CV review inputs.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    root = _repo_root()
    port = _free_port(args.port) if args.host in {"127.0.0.1", "localhost"} else args.port
    server = ThreadingHTTPServer((args.host, port), ReviewHandler)
    server.repo_root = root  # type: ignore[attr-defined]

    print(f"Serving review UI at http://{args.host}:{port}")
    print(f"Repo root: {root}")
    print(f"Save path: {root / LATEST_SAVE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review UI.")
    finally:
        threading.Thread(target=server.server_close).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
