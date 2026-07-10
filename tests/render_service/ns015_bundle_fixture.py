"""Real minimum-bundle fixtures for render-service status tests.

These helpers intentionally model the NS-01.5 contract. A fixture earns
``complete`` only with all mandatory artifacts and resolvable advertised URLs;
tests that omit evidence must declare ``partial`` and an explicit reason.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


TRUST_BANDS: dict[str, Any] = {
    "court": {"badge": "preview", "stage": "CAL"},
    "body": {"badge": "preview", "stage": "BODY"},
    "ball": {"badge": "low_confidence", "stage": "BALL"},
    "paddle": {"badge": "preview", "stage": "RKT"},
}


def write_minimum_bundle(
    root: Path,
    *,
    video_path: Path | None = None,
    status: str = "complete",
    missing_capabilities: list[dict[str, str]] | None = None,
    omit: str | None = None,
    missing_extra_url: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.mp4"
    if video_path is not None:
        shutil.copy2(video_path, source)
    else:
        source.write_bytes(b"fixture-video")
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    summary = {
        "status": status,
        "missing_capabilities": missing_capabilities or [],
        "trust_bands": TRUST_BANDS,
        "video": {"sha256": source_sha256, "size_bytes": len(source_bytes)},
        "stages": [{"stage": "manifest", "status": "ran", "wall_seconds": 1.0}],
    }
    _write_json(root / "PIPELINE_SUMMARY.json", summary)
    _write_json(root / "source_identity.json", summary["video"])
    _write_json(root / "capture_sidecar.json", {"schema_version": 1})
    _write_json(root / "court_calibration.json", {"coordinate_space": "encoded_pixels"})
    _write_json(root / "tracks.json", {"tracks": []})
    _write_json(root / "body_full_clip_gate.json", {"status": "preview"})
    _write_json(root / "ball_track.json", {"frames": []})
    _write_json(root / "ball_track_arc_solved.json", {"arcs": []})
    _write_json(root / "contact_windows.json", {"windows": []})
    _write_json(root / "racket_pose_estimate.json", {"poses": []})
    _write_json(root / "confidence_gated_world.json", {"players": [], "ball": {}})
    _write_json(root / "match_stats.json", {"facts": []})
    _write_json(root / "coaching_card_facts.json", {"facts": []})
    _write_json(root / "trust_bands.json", TRUST_BANDS)
    _write_json(
        root / "body_mesh_index" / "body_mesh_index.json",
        {
            "faces_url": "body_mesh_faces.json",
            "windows": [{"url": "chunks/window_000.bin.gz"}],
        },
    )
    _write_json(root / "body_mesh_index" / "body_mesh_faces.json", {"faces": []})
    chunk = root / "body_mesh_index" / "chunks" / "window_000.bin.gz"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"body-chunk")
    _write_json(root / "replay_scene.json", {"court_glb": "assets/court.glb"})
    court = root / "assets" / "court.glb"
    court.parent.mkdir(parents=True, exist_ok=True)
    court.write_bytes(b"court")

    manifest: dict[str, Any] = {
        "artifact_type": "replay_viewer_manifest",
        "clip": "clip_1",
        "video_url": "source.mp4",
        "body_mesh_index_url": "body_mesh_index/body_mesh_index.json",
        "ball_url": "ball_track.json",
        "paddle_url": "racket_pose_estimate.json",
        "virtual_world_url": "confidence_gated_world.json",
        "replay_scene_url": "replay_scene.json",
    }
    if missing_extra_url:
        manifest["label_overlays"] = [{"url": "assets/missing_overlay.png"}]

    if omit == "capture_sidecar":
        (root / "capture_sidecar.json").unlink()
    elif omit == "body":
        shutil.rmtree(root / "body_mesh_index")
        manifest["body_mesh_index_url"] = None
    elif omit == "ball":
        (root / "ball_track.json").unlink()
        manifest["ball_url"] = None
    elif omit == "paddle":
        (root / "racket_pose_estimate.json").unlink()
        manifest["paddle_url"] = None
    elif omit == "assets":
        shutil.rmtree(root / "assets")
        (root / "replay_scene.json").unlink()
        manifest["replay_scene_url"] = None
    elif omit is not None:
        raise ValueError(f"unsupported omitted capability: {omit}")

    _write_json(root / "replay_viewer_manifest.json", manifest)
    return summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
