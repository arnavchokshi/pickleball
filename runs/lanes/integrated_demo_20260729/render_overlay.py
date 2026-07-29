#!/usr/bin/env python3
"""Integration-demo overlay renderer — lane integrated_demo_20260729.

Extends `runs/lanes/visual_evidence_20260728/render_overlay.py` (read, not
modified) with the layers the owner asked for tonight: court + skeletons +
a BODY-mesh-computed indicator + the ball (band-honest) + contact/event
markers, composited over the existing court/NVZ/track/kitchen/minimap layers,
for the three full-preset co-located `--one-world` runs under
`runs/lanes/integrated_demo_20260729/<clip>/` (small/medium artifacts pulled
from night1; giant monoliths such as `one_world_v1.json`, `virtual_world.json`,
`confidence_gated_world.json`, `placement_trajectory_refined.json` were never
pulled — not needed for this render).

Ball trust-band handling (RUNBOOK "Reading 3D ball output"): `ball_track_arc_solved.json`
frames carry `band` in {anchored_measured, arc_interpolated, arc_weak, hidden}.
This renderer buckets `{anchored_measured, arc_interpolated}` as CONFIDENT (solid
marker + short trail) and `arc_weak` as WEAK (hollow/dashed marker, distinct
color) — both visually distinct per the mission brief. `hidden`-band frames get
**no ball marker at all**: never rendered as a measurement.

Read-only on everything under runs/ and eval_clips/. Writes only under:
  - runs/lanes/integrated_demo_20260729/  (this lane's own dir: work files)
  - ~/Desktop/visual_evidence_20260728/integrated/  (the delivery pack)

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

LANE_DIR = REPO_ROOT / "runs" / "lanes" / "integrated_demo_20260729"
DESKTOP_OUT = Path("/Users/arnavchokshi/Desktop/visual_evidence_20260728/integrated")
WORK_DIR = LANE_DIR / "_work"

# (run_dir under LANE_DIR, source video path, slug, label)
CLIPS = [
    (
        LANE_DIR / "wolverine_mixed_0200_mid_steep_corner",
        REPO_ROOT / "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4",
        "wolverine_mixed_0200_mid_steep_corner",
        "Wolverine mid steep corner (reused stretch demo, bodylocal_colocated_fix_20260728)",
    ),
    (
        LANE_DIR / "burlington_gold_0300_low_steep_corner",
        REPO_ROOT / "eval_clips/ball/burlington_gold_0300_low_steep_corner/source.mp4",
        "burlington_gold_0300_low_steep_corner",
        "Burlington gold low steep corner",
    ),
    (
        LANE_DIR / "outdoor_webcam_iynbd_1500_long_high_baseline",
        REPO_ROOT / "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/source.mp4",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "Outdoor webcam long high baseline",
    ),
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

BALL_CONFIDENT_BANDS = {"anchored_measured", "arc_interpolated"}
BALL_WEAK_BANDS = {"arc_weak"}
BALL_CONFIDENT_COLOR = (0, 230, 255)   # solid yellow-cyan
BALL_WEAK_COLOR = (0, 140, 255)        # orange, hollow
EVENT_FLASH_COLOR = (0, 0, 255)
EVENT_WINDOW_FRAMES = 4  # +/- frames around an event's peak frame to flash it

# Regulation pickleball court half-extents in meters (matches court_zones.json).
COURT_X_HALF = 3.048
COURT_Y_HALF = 6.7056
MINIMAP_W, MINIMAP_H = 220, 380
MINIMAP_MARGIN = 14
BALL_TRAIL_LEN = 8


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def color_for_player(player_id: int):
    return PLAYER_COLORS[(int(player_id) - 1) % len(PLAYER_COLORS)]


def project_world_xy(H: np.ndarray, xy) -> tuple[float, float]:
    v = H @ np.array([xy[0], xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def build_zone_lines(court_zones: dict, H: np.ndarray) -> dict:
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
                {"id": pid, "bbox": fr["bbox"], "world_xy": fr.get("world_xy"), "conf": fr.get("conf")}
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


def build_ball_index(ball_arc: dict) -> tuple[dict[int, dict], dict[str, int]]:
    """Index ball_track_arc_solved.json frames by frame index; also tally band counts."""
    fps = float(ball_arc.get("fps") or 30.0)
    by_frame: dict[int, dict] = {}
    counts: dict[str, int] = {}
    for i, fr in enumerate(ball_arc.get("frames", [])):
        fidx = fr.get("frame_idx")
        if fidx is None:
            fidx = int(round(float(fr.get("t", i / fps)) * fps))
        band = str(fr.get("band") or "hidden")
        counts[band] = counts.get(band, 0) + 1
        by_frame[fidx] = fr
    return by_frame, counts


def build_event_index(contact_windows: dict, fps: float) -> list[dict]:
    events = contact_windows.get("events") or []
    out = []
    for ev in events:
        frame = ev.get("frame")
        if frame is None:
            frame = int(round(float(ev.get("t", 0.0)) * fps))
        out.append(
            {
                "frame": int(frame),
                "player_id": ev.get("player_id"),
                "confidence": ev.get("confidence"),
                "type": ev.get("type", "contact"),
            }
        )
    return out


def build_mesh_index(body_mesh_index: dict | None) -> tuple[dict[int, set[int]], dict]:
    """frame_idx -> set of player_ids with real tier-1 mesh computed this frame."""
    if not body_mesh_index:
        return {}, {}
    by_frame: dict[int, set[int]] = {}
    for window in body_mesh_index.get("windows", []):
        for player in window.get("players", []):
            pid = player.get("id") or player.get("player_id")
            for fr in player.get("frames", []):
                fidx = fr.get("frame_idx")
                if fidx is None:
                    continue
                by_frame.setdefault(fidx, set()).add(pid if pid is not None else -1)
    return by_frame, body_mesh_index.get("summary", {})


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


def draw_tracks_and_kitchen(frame, track_items, kitchen_lookup, mesh_lookup) -> None:
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

        if pid in mesh_lookup:
            # tier-1 real BODY mesh was computed for this player on this frame:
            # subtle violet ring around the head/top of the bbox.
            head = ((x1 + x2) // 2, y1)
            cv2.circle(frame, head, 9, (255, 80, 220), 2, cv2.LINE_AA)

        state = kitchen_lookup.get(pid, "unknown")
        tag_color = KITCHEN_COLOR.get(state, (140, 140, 140))
        tag = f"NVZ: {state}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = min(frame.shape[0] - 4, y2 + th + 6)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, tag, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tag_color, 1, cv2.LINE_AA)


def draw_ball(frame, frame_idx, ball_by_frame, trail) -> None:
    import cv2

    fr = ball_by_frame.get(frame_idx)
    if fr is None:
        return
    band = str(fr.get("band") or "hidden")
    xy = fr.get("xy")
    if xy is None:
        return
    px, py = int(round(xy[0])), int(round(xy[1]))

    if band in BALL_CONFIDENT_BANDS:
        trail.append((px, py))
        color = BALL_CONFIDENT_COLOR
        cv2.circle(frame, (px, py), 8, color, -1)
        cv2.circle(frame, (px, py), 9, (0, 0, 0), 1)
        cv2.putText(frame, "ball", (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    elif band in BALL_WEAK_BANDS:
        trail.append((px, py))
        color = BALL_WEAK_COLOR
        cv2.circle(frame, (px, py), 8, color, 2)  # hollow = weak/depth-unvalidated
        cv2.putText(frame, "ball (weak)", (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    else:
        # hidden band: never rendered as a measurement. No marker at all.
        trail.append(None)
        return

    while len(trail) > BALL_TRAIL_LEN:
        trail.pop(0)
    pts = [p for p in trail if p is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 1, cv2.LINE_AA)


def draw_events(frame, frame_idx, events, fps) -> None:
    import cv2

    active = [ev for ev in events if abs(ev["frame"] - frame_idx) <= EVENT_WINDOW_FRAMES]
    if not active:
        return
    h, w = frame.shape[:2]
    y = 34
    for ev in active:
        conf = ev.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        text = f"CONTACT  P{ev.get('player_id')}  conf={conf_s}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = (w - tw) // 2
        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EVENT_FLASH_COLOR, 2, cv2.LINE_AA)
        y += th + 14


def minimap_xy_to_canvas(x: float, y: float) -> tuple[int, int]:
    u = (x + COURT_X_HALF) / (2 * COURT_X_HALF)
    v = (y + COURT_Y_HALF) / (2 * COURT_Y_HALF)
    px = int(round(u * (MINIMAP_W - 2 * MINIMAP_MARGIN) + MINIMAP_MARGIN))
    py = int(round((1.0 - v) * (MINIMAP_H - 2 * MINIMAP_MARGIN) + MINIMAP_MARGIN))
    return px, py


def draw_minimap(frame, zones: dict, track_items: list[dict], ball_frame: dict | None) -> None:
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

    if ball_frame is not None:
        band = str(ball_frame.get("band") or "hidden")
        wxyz = ball_frame.get("world_xyz")
        if wxyz is not None and band in (BALL_CONFIDENT_BANDS | BALL_WEAK_BANDS):
            cx, cy = minimap_xy_to_canvas(wxyz[0], wxyz[1])
            if 0 <= cx < MINIMAP_W and 0 <= cy < MINIMAP_H:
                color = BALL_CONFIDENT_COLOR if band in BALL_CONFIDENT_BANDS else BALL_WEAK_COLOR
                cv2.drawMarker(panel, (cx, cy), color, cv2.MARKER_DIAMOND, 10, 2)

    cv2.putText(panel, "top-down court map", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    frame[y0 : y0 + MINIMAP_H, x0 : x0 + MINIMAP_W] = cv2.addWeighted(
        frame[y0 : y0 + MINIMAP_H, x0 : x0 + MINIMAP_W], 0.15, panel, 0.85, 0
    )
    cv2.rectangle(frame, (x0, y0), (x0 + MINIMAP_W, y0 + MINIMAP_H), (255, 255, 255), 1)


def draw_legend(frame) -> None:
    import cv2

    h, w = frame.shape[:2]
    # Start below the reused skeleton-overlay module's own caption lines
    # (drawn at y=24/48/72 by threed.racketsport.skeleton_video_overlay).
    x0, y0 = 10, 78
    lines = [
        ("skeleton / violet ring = BODY tier-1 mesh computed this frame", (255, 80, 220)),
        ("ball, confident (anchored/interpolated)", BALL_CONFIDENT_COLOR),
        ("ball, weak (arc-solved, depth unvalidated)", BALL_WEAK_COLOR),
        ("hidden-band ball frames: no marker shown (never rendered as measurement)", (160, 160, 160)),
        ("CONTACT banner = event window from contact_windows_refined_v1.json", EVENT_FLASH_COLOR),
    ]
    line_h = 16
    box_h = line_h * len(lines) + 10
    box_w = 520
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, (text, color) in enumerate(lines):
        y = y0 + 16 + i * line_h
        cv2.circle(frame, (x0 + 10, y - 4), 4, color, -1)
        cv2.putText(frame, text, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)


def compute_headline_metrics(clip_dir: Path) -> dict:
    pipeline_summary = load_json(clip_dir / "PIPELINE_SUMMARY.json")
    ball_arc = load_json(clip_dir / "ball_track_arc_solved.json")
    contact_path = clip_dir / "contact_windows_refined_v1.json"
    if not contact_path.exists() or contact_path.stat().st_size < 10:
        contact_path = clip_dir / "contact_windows.json"
    contact_windows = load_json(contact_path) if contact_path.exists() else {"events": []}
    body_mesh_index = None
    bmi_path = clip_dir / "body_mesh_index" / "body_mesh_index.json"
    if bmi_path.exists():
        body_mesh_index = load_json(bmi_path)
    trust_bands = {}
    tb_path = clip_dir / "trust_bands.json"
    if tb_path.exists():
        trust_bands = load_json(tb_path)

    stages = pipeline_summary.get("stages") or []
    stage_table = [
        {
            "stage": s.get("stage"),
            "status": s.get("status"),
            "wall_seconds": s.get("wall_seconds"),
            "trust_badge": s.get("trust_badge"),
        }
        for s in stages
    ]
    degraded = [s["stage"] for s in stage_table if s["status"] == "degraded"]

    ball_frames = ball_arc.get("frames", [])
    band_counts: dict[str, int] = {}
    for fr in ball_frames:
        b = str(fr.get("band") or "hidden")
        band_counts[b] = band_counts.get(b, 0) + 1
    total_ball_frames = len(ball_frames) or 1
    confident = sum(band_counts.get(b, 0) for b in BALL_CONFIDENT_BANDS)
    weak = sum(band_counts.get(b, 0) for b in BALL_WEAK_BANDS)
    # Anything not explicitly confident/weak (hidden, plus any other band the
    # solver emits, e.g. arc_extrapolated) never gets a marker drawn -- see
    # draw_ball()'s else-branch, which is a strict allowlist keyed on
    # BALL_CONFIDENT_BANDS/BALL_WEAK_BANDS, not a hidden-specific check.
    # Reported as one honest "not rendered" bucket instead of mislabeling
    # non-hidden bands (e.g. arc_extrapolated) as "hidden", so the three
    # percentages always sum to 100.
    not_rendered = total_ball_frames - confident - weak

    mesh_frame_idx, mesh_summary = build_mesh_index(body_mesh_index)

    return {
        "wall_seconds": pipeline_summary.get("wall_seconds"),
        "status": pipeline_summary.get("status"),
        "pipeline_preset": pipeline_summary.get("pipeline_preset"),
        "stage_table": stage_table,
        "degraded_stages": degraded,
        "ball_band_counts": band_counts,
        "ball_total_frames": total_ball_frames,
        "ball_confident_pct": round(100.0 * confident / total_ball_frames, 1),
        "ball_weak_pct": round(100.0 * weak / total_ball_frames, 1),
        "ball_not_rendered_pct": round(100.0 * not_rendered / total_ball_frames, 1),
        "event_count": len(contact_windows.get("events") or []),
        "mesh_scheduled_frame_count": mesh_summary.get("mesh_frame_count"),
        "mesh_window_count": mesh_summary.get("window_count"),
        "mesh_distinct_video_frames_with_mesh": len(mesh_frame_idx),
        "trust_bands": trust_bands,
    }


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

    ball_arc = load_json(clip_dir / "ball_track_arc_solved.json")
    ball_by_frame, _ = build_ball_index(ball_arc)

    body_mesh_index = None
    bmi_path = clip_dir / "body_mesh_index" / "body_mesh_index.json"
    if bmi_path.exists():
        body_mesh_index = load_json(bmi_path)
    mesh_by_frame, _ = build_mesh_index(body_mesh_index)

    cap = cv2.VideoCapture(str(skeleton_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open skeleton overlay video: {skeleton_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    contact_path = clip_dir / "contact_windows_refined_v1.json"
    if not contact_path.exists() or contact_path.stat().st_size < 10:
        contact_path = clip_dir / "contact_windows.json"
    contact_windows = load_json(contact_path) if contact_path.exists() else {"events": []}
    events = build_event_index(contact_windows, fps)

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
    ball_trail: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_court_and_nvz(frame, zone_lines)
            track_items = track_by_frame.get(frame_idx, [])
            kitchen_lookup = kitchen_by_frame.get(frame_idx, {})
            mesh_lookup = mesh_by_frame.get(frame_idx, set())
            draw_tracks_and_kitchen(frame, track_items, kitchen_lookup, mesh_lookup)
            draw_ball(frame, frame_idx, ball_by_frame, ball_trail)
            draw_events(frame, frame_idx, events, fps)
            draw_minimap(frame, court_zones, track_items, ball_by_frame.get(frame_idx))
            draw_legend(frame)
            cv2.putText(
                frame,
                "review copy - VERIFIED=0, preview band, not gate-verified",
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
    """Symlink a read-only clip run_dir into WORK_DIR, except for
    skeleton3d.json which gets a copy with joint_names remapped from generic
    ``sam3dbody_joint_###`` to their MHR70 semantic names so the bone
    connectors draw. Never touches the original artifacts.
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


def render_clip(clip_dir: Path, source_video: Path, slug: str, label: str) -> dict:
    if not clip_dir.exists():
        raise FileNotFoundError(clip_dir)

    print(f"[{slug}] rendering skeleton layer (reused threed.racketsport.skeleton_video_overlay)...")
    patched_run_dir = build_patched_run_dir(clip_dir, slug)
    skel_out_dir = WORK_DIR / slug / "skel"
    skel_summary = svo.render_skeleton_overlay(
        run_dir=patched_run_dir,
        video_path=source_video,
        out_dir=skel_out_dir,
    )
    skeleton_video = Path(skel_summary["overlay_path"])

    print(f"[{slug}] compositing court/NVZ/track/kitchen/ball/events/mesh/minimap/legend layers...")
    final_out = DESKTOP_OUT / f"{slug}.mp4"
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


def render_index_html(results: list[dict]) -> str:
    rows = []
    for r in results:
        if "error" in r:
            rows.append(
                f'<div class="clip"><div class="info"><h3>{r["label"]}<span class="badge fail">FAILED</span></h3>'
                f'<p>{r["error"]}</p></div></div>'
            )
            continue
        m = r["metrics"]
        cs = r["composite_summary"]
        degraded = ", ".join(m["degraded_stages"]) or "none"
        band_counts = m["ball_band_counts"]
        band_str = ", ".join(f"{k}={v}" for k, v in sorted(band_counts.items()))
        rows.append(f"""
<div class="clip">
  <video controls muted loop preload="metadata"><source src="{r['video_file']}" type="video/mp4"></video>
  <div class="info">
    <h3>{r['label']}</h3>
    <table>
      <tr><th>Wall time (full preset, co-located, night1 A100)</th><td>{m['wall_seconds']:.1f}s</td></tr>
      <tr><th>Pipeline status</th><td>{m['status']}</td></tr>
      <tr><th>Degraded stages</th><td>{degraded}</td></tr>
      <tr><th>Ball coverage</th><td>confident {m['ball_confident_pct']}% &middot; weak {m['ball_weak_pct']}% &middot; not rendered (hidden + any other non-confident/weak band) {m['ball_not_rendered_pct']}%<br><span style="font-size:.85em;color:#666">bands: {band_str}</span></td></tr>
      <tr><th>Contact/events (contact_windows_refined_v1)</th><td>{m['event_count']}</td></tr>
      <tr><th>BODY tier-1 mesh</th><td>{m['mesh_scheduled_frame_count']} player-frames scheduled across {m['mesh_window_count']} windows &middot; {m['mesh_distinct_video_frames_with_mesh']}/{cs['frame_count']} video frames have >=1 player with real mesh</td></tr>
      <tr><th>Rendered frames</th><td>{cs['frame_count']} @ {cs['fps']:.1f}fps, {cs['width']}x{cs['height']}</td></tr>
    </table>
    <div class="caveat">preview band, VERIFIED=0 throughout</div>
  </div>
</div>""")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Pickleball integrated end-to-end demo (2026-07-29)</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;background:#faf7f0;color:#1a1a1a}}
h1{{font-size:1.5em;margin-bottom:4px}}
.sub{{color:#555;margin-top:0}}
h2{{margin-top:0;font-size:1.15em}}
.clip{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0;display:flex;gap:20px;flex-wrap:wrap}}
.clip video{{width:520px;max-width:100%;border-radius:6px;background:#000}}
.clip .info{{flex:1;min-width:260px}}
table{{border-collapse:collapse;margin:8px 0;width:100%}}
td,th{{border:1px solid #ccc;padding:5px 8px;text-align:left;font-size:.92em}}
th{{background:#f0efe8}}
.pass{{color:#1a7a1a;font-weight:600}}
.fail{{color:#a33;font-weight:600}}
.caveat{{color:#555;font-size:.85em;margin-top:10px;border-top:1px dashed #ccc;padding-top:8px}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75em;font-weight:600;background:#b80;color:#fff;margin-left:8px}}
.badge.fail{{background:#a33}}
.top-note{{background:#fff3cd;border:1px solid #e0c060;border-radius:8px;padding:10px 14px;font-size:.92em;margin-bottom:18px}}
</style></head><body>

<h1>Pickleball: integrated end-to-end demo &mdash; court + players + skeletons + BODY meshes + ball + events</h1>
<p class="sub">integrated_demo_20260729 &middot; extends bodylocal_colocated_fix_20260728's co-located full-preset pipeline + wolverine stretch demo with two more clips.
Every video below is the actual full-preset pipeline output (court, ball, one_world, ball-aware mesh scheduling all on), drawn back onto the source frames. <b>VERIFIED=0</b> throughout &mdash; this is a working product demo, not an accuracy promotion.</p>

<div class="top-note">
Each video shows, drawn live on the source footage: solved court lines + kitchen (NVZ) zone shading, each player's
skeleton (70-joint BODY inference), a violet ring on a player's head when real tier-1 BODY mesh was computed for
that player on that frame (mesh vertices are not rasterized here &mdash; this is a "mesh computed" indicator, not
a wireframe render), the ball (distinct styling for confident vs weak 3D bands; frames in the <code>hidden</code>
band get no marker at all &mdash; never rendered as a measurement, per RUNBOOK "Reading 3D ball output"), a CONTACT
banner around each event window from <code>contact_windows_refined_v1.json</code>, and a bottom-right top-down
court minimap with player and ball positions. A legend is drawn on every frame. Play any of them: nothing here
is a mockup.
</div>

{"".join(rows)}

<h2>Ball trust bands, honestly</h2>
<p style="font-size:.92em;color:#444">The BALL stage has not cleared its accuracy gate on the public domain
(0/8 milestones, see <code>trust_bands.json</code> per clip). <code>ball_track_arc_solved.json</code> frames carry a
<code>band</code>: <code>anchored_measured</code>/<code>arc_interpolated</code> (bucketed here as "confident" &mdash;
solid marker), <code>arc_weak</code> ("weak" &mdash; hollow marker, depth unvalidated), and <code>hidden</code>
(absurd/implausible position &mdash; suppressed, no marker). Reprojection error is a 2D consistency check only; it
says nothing about depth (measured: a 1.00&nbsp;m shift along the camera ray changes reprojection by 1.6e-13 px).
Treat every ball position here as a rough cue, not a verified 3D track.</p>

<p class="caveat">Preview band throughout. Nothing here has cleared an independent-data promotion gate.
See <a href="INTEGRATED_RESULTS.md">INTEGRATED_RESULTS.md</a> for the honest per-clip scorecard.</p>

</body></html>
"""


def render_results_md(results: list[dict]) -> str:
    lines = [
        "# Integrated end-to-end demo — INTEGRATED_RESULTS.md",
        "",
        "Lane `integrated_demo_20260729`. `VERIFIED=0` throughout — this is a rendered product demo,",
        "not an accuracy promotion. Full preset (`--pipeline-preset full`, the default), co-located",
        "`--body-local`, `--one-world`, `--max-players 4`, `--force`, on `pickleball-gpu-night1`",
        "(A100-40GB, GPU compute mode Default). Two clips (`burlington_gold_0300_low_steep_corner`,",
        "`outdoor_webcam_iynbd_1500_long_high_baseline`) were freshly run tonight; the third",
        "(`wolverine_mixed_0200_mid_steep_corner`) reuses the `bodylocal_colocated_fix_20260728` stretch",
        "demo bundle verbatim, not rerun.",
        "",
        "## Per-clip scorecard",
        "",
        "| Clip | Wall (s) | Status | Degraded stages | Ball confident/weak/not-rendered % | Events | Mesh player-frames (windows) | Rendered frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['label']} | — | FAILED | {r['error']} | — | — | — | — |")
            continue
        m = r["metrics"]
        cs = r["composite_summary"]
        degraded = ", ".join(m["degraded_stages"]) or "none"
        lines.append(
            f"| {r['label']} | {m['wall_seconds']:.1f} | {m['status']} | {degraded} | "
            f"{m['ball_confident_pct']}/{m['ball_weak_pct']}/{m['ball_not_rendered_pct']} | {m['event_count']} | "
            f"{m['mesh_scheduled_frame_count']} ({m['mesh_window_count']}) | {cs['frame_count']} |"
        )

    lines += [
        "",
        "## Per-clip stage tables",
        "",
    ]
    for r in results:
        if "error" in r:
            continue
        m = r["metrics"]
        lines.append(f"### {r['label']}")
        lines.append("")
        lines.append("| stage | status | wall_seconds | trust_badge |")
        lines.append("|---|---|---|---|")
        for s in m["stage_table"]:
            lines.append(f"| {s['stage']} | {s['status']} | {s['wall_seconds']} | {s['trust_badge']} |")
        lines.append("")
        lines.append(f"Ball band counts: {json.dumps(m['ball_band_counts'])}")
        lines.append("")

    lines += [
        "## Honest caveats",
        "",
        "- `VERIFIED=0` binding throughout; no artifact here has cleared an independent-data promotion gate.",
        "- BALL stage: 0/8 milestone gates pass on the public domain (see each clip's `trust_bands.json`);",
        "  every rendered ball position is a rough cue, never a verified 3D track. Depth is unvalidated for",
        "  every 3D ball frame (`depth_unvalidated: true`).",
        "- `hidden`-band ball frames are never drawn as a measurement — the renderer skips them entirely.",
        "- COURT calibration is `metric_15pt_reviewed`, grade `warn` in these clips — preview, not the",
        "  held-out PCK@5px>=0.95 gate.",
        "- BODY mesh: structural gate only (`full_clip_body_gate`); world-MPJPE accuracy has not been",
        "  measured for these runs.",
        "- Degraded stages are typed, honest reverts (e.g. `ball_arc` segment-budget kills,",
        "  `grounding_refine` restoring originals on a worsened-residual sanity check), not silent failures.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    DESKTOP_OUT.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for clip_dir, source_video, slug, label in CLIPS:
        try:
            result = render_clip(clip_dir, source_video, slug, label)
            results.append(result)
            print(f"[{slug}] done -> {result['video_file']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{slug}] FAILED: {exc}", file=sys.stderr)
            results.append({"slug": slug, "label": label, "error": str(exc)})

    (DESKTOP_OUT / "render_manifest.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (LANE_DIR / "render_manifest.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DESKTOP_OUT / "index.html").write_text(render_index_html(results), encoding="utf-8")
    (DESKTOP_OUT / "INTEGRATED_RESULTS.md").write_text(render_results_md(results), encoding="utf-8")
    print("wrote", DESKTOP_OUT / "render_manifest.json")
    print("wrote", DESKTOP_OUT / "index.html")
    print("wrote", DESKTOP_OUT / "INTEGRATED_RESULTS.md")


if __name__ == "__main__":
    main()
