"""Build calibration artifacts from human-reviewed court-corner corrections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .court_auto_evidence import build_auto_court_line_evidence_from_frame
from .court_calibration import calibration_from_manual_taps
from .court_line_evidence import aggregate_court_line_evidence
from .court_templates import Sport
from .court_zones import build_court_zones
from .net_plane import build_net_plane
from .schemas import CaptureSidecar, CourtCalibration, NetPlane


SIDECAR_CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")


def build_calibration_from_corrections(
    *,
    drafts_root: str | Path,
    corrections_root: str | Path,
    frames_root: str | Path,
    out_root: str | Path,
    sport: Sport = "pickleball",
    net_post_height_in: float | None = None,
    net_center_height_in: float | None = None,
) -> dict[str, Any]:
    """Convert reviewed court corners into sidecars and calibration artifacts."""

    drafts_root = Path(drafts_root)
    corrections_root = Path(corrections_root)
    frames_root = Path(frames_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    clip_summaries: list[dict[str, Any]] = []
    calibrated = 0
    for correction_path in sorted(corrections_root.glob("*/court_corners.json")):
        clip = correction_path.parent.name
        out_dir = out_root / clip
        try:
            correction = _read_json(correction_path)
            item = _reviewed_corner_item(correction)
            frame_name = str(item.get("frame") or "frame_000001.jpg")
            frame_path = frames_root / clip / frame_name
            width, height = _image_size(frame_path)
            manual_taps = _manual_taps_from_reviewed_corners(item)
            sidecar = _sidecar_payload(
                clip=clip,
                frame_name=frame_name,
                manual_taps=manual_taps,
                width=width,
                height=height,
                fps=_clip_fps(drafts_root, clip),
            )
            CaptureSidecar.model_validate(sidecar)
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_json(out_dir / "capture_sidecar.json", sidecar)
            calibration = calibration_from_manual_taps(out_dir / "capture_sidecar.json", sport=sport)
            net_plane = build_net_plane(
                sport,
                post_height_in=net_post_height_in,
                center_height_in=net_center_height_in,
            )
            _write_json_artifact(out_dir / "court_calibration.json", calibration)
            _write_json_artifact(out_dir / "court_zones.json", build_court_zones(sport))
            _write_json_artifact(out_dir / "net_plane.json", net_plane)
            _write_json_artifact(
                out_dir / "court_line_evidence.json",
                _line_evidence_from_review_frame(
                    frame_path=frame_path,
                    calibration=calibration,
                    net_plane=net_plane,
                    sport=sport,
                ),
            )
        except Exception as exc:
            clip_summaries.append(
                {
                    "clip": clip,
                    "status": "blocked",
                    "correction_path": str(correction_path),
                    "notes": [str(exc)],
                }
            )
            continue

        calibrated += 1
        clip_summaries.append(
            {
                "clip": clip,
                "status": "corrected_unverified",
                "correction_path": str(correction_path),
                "frame": frame_name,
                "out_dir": str(out_dir),
                "artifacts": [
                    "capture_sidecar.json",
                    "court_calibration.json",
                    "court_zones.json",
                    "net_plane.json",
                    "court_line_evidence.json",
                ],
                "not_ground_truth": True,
            }
        )

    status = "corrected_unverified" if calibrated > 0 else "blocked"
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_corner_calibration_import",
        "status": status,
        "drafts_root": str(drafts_root),
        "corrections_root": str(corrections_root),
        "frames_root": str(frames_root),
        "out_root": str(out_root),
        "sport": sport,
        "calibrated_clip_count": calibrated,
        "clip_count": len(clip_summaries),
        "clips": clip_summaries,
        "not_ground_truth": True,
    }
    _write_json(out_root / "court_corner_calibration_summary.json", summary)
    return summary


def _reviewed_corner_item(correction: dict[str, Any]) -> dict[str, Any]:
    items = correction.get("items")
    if not isinstance(items, list):
        raise ValueError("correction items must be a list")
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("court_corners"), dict):
            return item
    raise ValueError("no reviewed court_corners item found")


def _manual_taps_from_reviewed_corners(item: dict[str, Any]) -> list[list[float]]:
    raw_corners = item.get("court_corners")
    if not isinstance(raw_corners, dict):
        raise ValueError("court_corners must be an object")
    missing = [key for key in SIDECAR_CORNER_ORDER if key not in raw_corners]
    if missing:
        raise ValueError(f"missing court corner(s): {', '.join(missing)}")

    taps: list[list[float]] = []
    for key in SIDECAR_CORNER_ORDER:
        value = raw_corners[key]
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise ValueError(f"{key} must be a 2D image point")
        taps.append([float(value[0]), float(value[1])])
    return taps


def _line_evidence_from_review_frame(
    *,
    frame_path: Path,
    calibration: CourtCalibration,
    net_plane: NetPlane,
    sport: Sport,
) -> Any:
    try:
        return build_auto_court_line_evidence_from_frame(frame_path, calibration, net_plane=net_plane, frame_index=0)
    except Exception as exc:
        evidence = aggregate_court_line_evidence(
            sport=sport,
            line_observations=[],
            net_observations=[],
            required_line_ids=("near_nvz", "far_nvz", "near_centerline", "far_centerline") if sport == "pickleball" else (),
            required_net_ids=("top_net",),
        )
        evidence.source = "review_frame_auto_detection_failed"
        evidence.aggregate.reasons.append(f"auto_detection_failed:{type(exc).__name__}")
        return evidence


def _sidecar_payload(
    *,
    clip: str,
    frame_name: str,
    manual_taps: list[list[float]],
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    focal = float(max(width, height) * 1.2)
    return {
        "schema_version": 1,
        "device_tier": "fallback",
        "device_model": f"prototype_human_review:{clip}:{frame_name}",
        "fps": fps,
        "format": "hevc",
        "resolution": [int(width), int(height)],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {
            "fx": focal,
            "fy": focal,
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "dist": [],
            "source": "estimated_from_review_frame",
        },
        "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": manual_taps,
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {
            "grade": "warn",
            "reasons": ["prototype_human_review_corners", "estimated_intrinsics", "corrected_unverified"],
        },
    }


def _clip_fps(drafts_root: Path, clip: str) -> int:
    manifest_path = drafts_root / clip / "labels" / "prototype_autolabel_manifest.json"
    if manifest_path.is_file():
        metadata = _read_json(manifest_path).get("clip", {}).get("metadata", {})
        if isinstance(metadata, dict):
            fps = metadata.get("frame_rate_fps")
            if isinstance(fps, int | float) and fps > 0:
                return int(round(float(fps)))
    return 30


def _image_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"missing review frame image: {path}")
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"unsupported image format for dimension probe: {path}")

    idx = 2
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while idx + 3 < len(data):
        while idx < len(data) and data[idx] != 0xFF:
            idx += 1
        while idx < len(data) and data[idx] == 0xFF:
            idx += 1
        if idx >= len(data):
            break
        marker = data[idx]
        idx += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if idx + 2 > len(data):
            break
        segment_length = int.from_bytes(data[idx : idx + 2], "big")
        if segment_length < 2 or idx + segment_length > len(data):
            raise ValueError(f"invalid JPEG segment while reading dimensions: {path}")
        if marker in start_of_frame_markers:
            if segment_length < 7:
                raise ValueError(f"invalid JPEG SOF segment while reading dimensions: {path}")
            height = int.from_bytes(data[idx + 3 : idx + 5], "big")
            width = int.from_bytes(data[idx + 5 : idx + 7], "big")
            return width, height
        idx += segment_length
    raise ValueError(f"could not read image dimensions: {path}")


def _write_json_artifact(path: Path, artifact: Any) -> None:
    payload = artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact
    _write_json(path, payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


__all__ = ["SIDECAR_CORNER_ORDER", "build_calibration_from_corrections"]
