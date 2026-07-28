#!/usr/bin/env python3
"""Owner evidence-pack renderer — lane visual_evidence_20260728.

Renders per-clip overlay MP4s (solved court lines + NVZ shading, per-player
skeleton, track box + ID, footpoint marker, per-frame kitchen decision tag,
and a top-down court minimap inset) for the six fresh split-mode runs under
runs/alwayson_fresh_wave_20260728/split/.

Reuses existing repo tooling for the skeleton layer
(threed.racketsport.skeleton_video_overlay.render_skeleton_overlay) and hand
rolls the compositing pass (court/NVZ/track/kitchen/minimap) on top, since no
single existing helper combines all of those layers.

Read-only on everything under runs/; writes only under:
  - ~/Desktop/visual_evidence_20260728/   (the evidence pack itself)
This script itself lives under runs/lanes/visual_evidence_20260728/ (this
lane's owned directory) and is committed there, not run from inside it.

VERIFIED=0. This script produces review/demo visuals, not a new measurement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/Users/arnavchokshi/Desktop/pickleball")
sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport import skeleton_video_overlay as svo  # noqa: E402
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES  # noqa: E402

DESKTOP_PACK = Path("/Users/arnavchokshi/Desktop/visual_evidence_20260728")
WORK_DIR = DESKTOP_PACK / "_work"
WAVE_DIR = REPO_ROOT / "runs" / "alwayson_fresh_wave_20260728" / "split"

# (outer split dir, inner run dir, short label)
CLIPS = [
    ("01_wolverine_indoor_diagonal_0_10", "01_wolverine_indoor_diagonal_0_10", "wolverine_indoor_diagonal", "Wolverine indoor diagonal"),
    ("02_indoor_doubles_baseline_10_20", "02_indoor_doubles_baseline_10_20", "indoor_doubles_baseline", "Indoor doubles baseline"),
    ("03_outdoor_high_baseline_0_10", "03_outdoor_high_baseline_0_10", "outdoor_high_baseline", "Outdoor high baseline"),
    ("06_replacement_indoor_straight_100_110_retry", "06_replacement_indoor_straight_100_110", "indoor_straight_replacement", "Indoor straight (replacement, retry)"),
    ("07_replacement_indoor_diagonal_100_110", "07_replacement_indoor_diagonal_100_110", "indoor_diagonal_replacement", "Indoor diagonal (replacement)"),
    ("10_replacement_outdoor_pbvision_50_60", "10_replacement_outdoor_pbvision_50_60", "outdoor_pbvision_replacement", "Outdoor PBVision (replacement)"),
]

PLAYER_COLORS = [
    (60, 220, 255),
    (80, 200, 80),
    (255, 180, 80),
    (220, 120, 255),
    (255, 255, 80),
    (80, 120, 255),
]

KITCHEN_COLOR = {
    "unknown": (140, 140, 140),
    "confirmed_outside": (60, 180, 255),
    "confirmed_inside_or_on": (0, 0, 255),
    "airborne_no_contact": (220, 120, 255),
}

# Regulation pickleball court half-extents in meters (matches court_zones.json).
COURT_X_HALF = 3.048
COURT_Y_HALF = 6.7056
MINIMAP_W, MINIMAP_H = 220, 380
MINIMAP_MARGIN = 14


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def color_for_player(player_id: int):
    return PLAYER_COLORS[(int(player_id) - 1) % len(PLAYER_COLORS)]


def project_world_xy(H: np.ndarray, xy) -> tuple[float, float]:
    v = H @ np.array([xy[0], xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def build_zone_lines(court_zones: dict, H: np.ndarray) -> dict:
    """Project each named zone polygon's edges to image pixels."""
    out = {}
    for name, poly in court_zones["zones"].items():
        pts = [project_world_xy(H, p) for p in poly]
        out[name] = pts
    return out


def build_track_index(tracks: dict) -> dict[int, list[dict]]:
    fps = float(tracks.get("fps") or 30.0)
    by_frame: dict[int, list[dict]] = {}
    for player in tracks["players"]:
        pid = player["id"]
        for fr in player["frames"]:
            fidx = fr.get("frame_idx")
            if fidx is None:
                fidx = int(round(float(fr.get("t", 0.0)) * fps))
            by_frame.setdefault(fidx, []).append(
                {
                    "id": pid,
                    "bbox": fr["bbox"],
                    "world_xy": fr.get("world_xy"),
                    "conf": fr.get("conf"),
                }
            )
    return by_frame


def build_kitchen_index(placement_refined: dict) -> dict[int, dict[int, str]]:
    fps = float(placement_refined.get("fps") or 30.0)
    by_frame: dict[int, dict[int, str]] = {}
    for player in placement_refined["players"]:
        pid = player["id"]
        for fr in player["frames"]:
            fidx = fr.get("frame_idx")
            if fidx is None:
                fidx = int(round(float(fr.get("t", 0.0)) * fps))
            kd = fr.get("kitchen_decision") or {}
            state = kd.get("court_contact_state")
            if state is None:
                continue
            by_frame.setdefault(fidx, {})[pid] = state
    return by_frame


def compute_headline_metrics(clip_dir: Path) -> dict:
    pipeline_summary = load_json(clip_dir / "PIPELINE_SUMMARY.json")
    body_gate = load_json(clip_dir / "body_full_clip_gate.json")
    body_grounding = load_json(clip_dir / "body_grounding_quality.json")
    placement_refined = load_json(clip_dir / "placement_refined.json")
    virtual_world = load_json(clip_dir / "virtual_world.json")

    wall_seconds = pipeline_summary.get("wall_seconds") or pipeline_summary.get(
        "performance", {}
    ).get("total_wall_seconds")

    grounding_metrics = body_grounding.get("grounding_metrics", {})
    max_foot_lock_slide_m = grounding_metrics.get("max_foot_lock_slide_m")
    foot_slide_pass = None
    if max_foot_lock_slide_m is not None:
        foot_slide_pass = bool(max_foot_lock_slide_m <= 0.03)

    body_samples = body_gate.get("evaluated_player_frame_count")
    players_retained = (virtual_world.get("summary") or {}).get("player_count")

    kitchen_counts: dict[str, int] = {}
    for player in placement_refined["players"]:
        for fr in player["frames"]:
            kd = fr.get("kitchen_decision") or {}
            state = str(kd.get("court_contact_state") or "unknown")
            kitchen_counts[state] = kitchen_counts.get(state, 0) + 1

    return {
        "wall_seconds": wall_seconds,
        "max_foot_lock_slide_m": max_foot_lock_slide_m,
        "foot_slide_pass_0_03m": foot_slide_pass,
        "body_samples_evaluated": body_samples,
        "players_retained": players_retained,
        "kitchen_decision_counts": kitchen_counts,
        "body_grounding_blockers": body_grounding.get("blockers", []),
    }


def draw_court_and_nvz(frame, zone_lines: dict) -> None:
    import cv2

    overlay = frame.copy()
    for zname in ("near_nvz", "far_nvz"):
        pts = np.array(zone_lines[zname], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 210, 60))
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, dst=frame)

    for zname, pts in zone_lines.items():
        pts_i = [tuple(int(round(c)) for c in p) for p in pts]
        color = (0, 220, 0) if zname == "court" else (255, 210, 60)
        thickness = 2 if zname == "court" else 1
        for a, b in zip(pts_i, pts_i[1:] + pts_i[:1]):
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def draw_tracks_and_kitchen(frame, frame_idx, track_items, kitchen_lookup) -> None:
    import cv2

    for item in track_items:
        pid = item["id"]
        color = color_for_player(pid)
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        foot = ((x1 + x2) // 2, y2)
        cv2.circle(frame, foot, 6, color, -1)
        cv2.circle(frame, foot, 7, (0, 0, 0), 1)
        label = f"P{pid}"
        cv2.putText(frame, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        state = kitchen_lookup.get(pid, "unknown")
        tag_color = KITCHEN_COLOR.get(state, (140, 140, 140))
        tag = f"NVZ: {state}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = min(frame.shape[0] - 4, y2 + th + 6)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, tag, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tag_color, 1, cv2.LINE_AA)


def minimap_xy_to_canvas(x: float, y: float) -> tuple[int, int]:
    u = (x + COURT_X_HALF) / (2 * COURT_X_HALF)
    v = (y + COURT_Y_HALF) / (2 * COURT_Y_HALF)
    px = int(round(u * (MINIMAP_W - 2 * MINIMAP_MARGIN) + MINIMAP_MARGIN))
    py = int(round((1.0 - v) * (MINIMAP_H - 2 * MINIMAP_MARGIN) + MINIMAP_MARGIN))
    return px, py


def draw_minimap(frame, zones: dict, track_items: list[dict]) -> None:
    import cv2

    h, w = frame.shape[:2]
    x0 = w - MINIMAP_W - 16
    y0 = h - MINIMAP_H - 16
    panel = np.full((MINIMAP_H, MINIMAP_W, 3), (24, 24, 24), dtype=np.uint8)

    for zname in ("near_nvz", "far_nvz"):
        pts = np.array([minimap_xy_to_canvas(*p) for p in zones["zones"][zname]], dtype=np.int32)
        cv2.fillPoly(panel, [pts], (60, 140, 200))

    for zname, poly in zones["zones"].items():
        pts = [minimap_xy_to_canvas(*p) for p in poly]
        color = (0, 200, 0) if zname == "court" else (150, 150, 150)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(panel, a, b, color, 1, cv2.LINE_AA)

    for item in track_items:
        wxy = item.get("world_xy")
        if not wxy:
            continue
        cx, cy = minimap_xy_to_canvas(wxy[0], wxy[1])
        if 0 <= cx < MINIMAP_W and 0 <= cy < MINIMAP_H:
            cv2.circle(panel, (cx, cy), 6, color_for_player(item["id"]), -1)
            cv2.circle(panel, (cx, cy), 7, (255, 255, 255), 1)

    cv2.putText(panel, "top-down court map", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    frame[y0 : y0 + MINIMAP_H, x0 : x0 + MINIMAP_W] = cv2.addWeighted(
        frame[y0 : y0 + MINIMAP_H, x0 : x0 + MINIMAP_W], 0.15, panel, 0.85, 0
    )
    cv2.rectangle(frame, (x0, y0), (x0 + MINIMAP_W, y0 + MINIMAP_H), (255, 255, 255), 1)


def composite_pass(skeleton_video: Path, clip_dir: Path, out_path: Path) -> dict:
    import cv2

    court_calibration = load_json(clip_dir / "court_calibration.json")
    H = np.array(court_calibration["homography"], dtype=float)
    court_zones = load_json(clip_dir / "court_zones.json")
    zone_lines = build_zone_lines(court_zones, H)

    tracks = load_json(clip_dir / "tracks.json")
    track_by_frame = build_track_index(tracks)

    placement_refined = load_json(clip_dir / "placement_refined.json")
    kitchen_by_frame = build_kitchen_index(placement_refined)

    cap = cv2.VideoCapture(str(skeleton_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open skeleton overlay video: {skeleton_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )

    frame_idx = 0
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_court_and_nvz(frame, zone_lines)
            track_items = track_by_frame.get(frame_idx, [])
            kitchen_lookup = kitchen_by_frame.get(frame_idx, {})
            draw_tracks_and_kitchen(frame, frame_idx, track_items, kitchen_lookup)
            draw_minimap(frame, court_zones, track_items)
            cv2.putText(
                frame,
                "review copy — VERIFIED=0, preview band, not gate-verified",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            ffmpeg.stdin.write(frame.tobytes())
            frame_idx += 1
            frame_count += 1
    finally:
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()

    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def build_patched_run_dir(clip_dir: Path, slug: str) -> Path:
    """Symlink a read-only clip run_dir into our lane's WORK_DIR, except for
    skeleton3d.json which gets a copy with joint_names remapped from generic
    ``sam3dbody_joint_###`` to their MHR70 semantic names (nose, left_shoulder,
    ...). threed.racketsport.skeleton_video_overlay.bone_pairs_for_joint_names
    matches on semantic names, so without this remap it silently draws zero
    bone connectors (joints still render as dots, just unconnected). This
    never touches the original run artifacts — only reads them.
    """
    patched = WORK_DIR / slug / "patched_run"
    patched.mkdir(parents=True, exist_ok=True)
    for item in clip_dir.iterdir():
        if item.name == "skeleton3d.json":
            continue
        link = patched / item.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(item)

    skeleton = load_json(clip_dir / "skeleton3d.json")
    joint_names = skeleton.get("joint_names") or []
    if len(joint_names) == len(MHR70_JOINT_NAMES) and all(
        n == f"sam3dbody_joint_{i:03d}" for i, n in enumerate(joint_names)
    ):
        skeleton["joint_names"] = list(MHR70_JOINT_NAMES)
    (patched / "skeleton3d.json").write_text(json.dumps(skeleton), encoding="utf-8")
    return patched


def render_clip(outer: str, inner: str, slug: str, label: str) -> dict:
    clip_dir = WAVE_DIR / outer / inner
    if not clip_dir.exists():
        raise FileNotFoundError(clip_dir)

    print(f"[{slug}] rendering skeleton layer (reused threed.racketsport.skeleton_video_overlay)...")
    patched_run_dir = build_patched_run_dir(clip_dir, slug)
    skel_out_dir = WORK_DIR / slug / "skel"
    skel_summary = svo.render_skeleton_overlay(
        run_dir=patched_run_dir,
        video_path=clip_dir / "source.mp4",
        out_dir=skel_out_dir,
    )
    skeleton_video = Path(skel_summary["overlay_path"])

    print(f"[{slug}] compositing court/NVZ/track/kitchen/minimap layers...")
    final_out = DESKTOP_PACK / f"{slug}.mp4"
    composite_summary = composite_pass(skeleton_video, clip_dir, final_out)

    print(f"[{slug}] computing headline metrics...")
    metrics = compute_headline_metrics(clip_dir)

    return {
        "slug": slug,
        "label": label,
        "clip_dir": str(clip_dir),
        "video_file": final_out.name,
        "skeleton_layer_summary": {
            "player_count": skel_summary.get("player_count"),
            "joint_count": skel_summary.get("joint_count"),
            "bone_count": skel_summary.get("bone_count"),
        },
        "composite_summary": composite_summary,
        "metrics": metrics,
    }


def main() -> None:
    DESKTOP_PACK.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for outer, inner, slug, label in CLIPS:
        try:
            result = render_clip(outer, inner, slug, label)
            results.append(result)
            print(f"[{slug}] done -> {result['video_file']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{slug}] FAILED: {exc}", file=sys.stderr)
            results.append({"slug": slug, "label": label, "error": str(exc)})

    (DESKTOP_PACK / "render_manifest.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote", DESKTOP_PACK / "render_manifest.json")


if __name__ == "__main__":
    main()
