#!/usr/bin/env python3
"""Extract the wave-4 freshproof gated-key checklist from a clip's process_video.py output dir.

Extends runs/lanes/w3_freshworlds_20260707/scripts/extract_checklist.py with:
ball chain census (arc segments + trail coverage), virtual_world warning strings
(missing_embedded_mesh_vertices vs missing_mesh_vertices), selected_mesh_frame_count,
and first-class decode_orientation_* keys inside camera_motion_auto.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def load(clip_dir: Path, name: str):
    p = clip_dir / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {"__error__": str(exc)}


def get(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def extract(clip_dir: Path) -> dict:
    out = {"clip_dir": str(clip_dir)}

    pipeline_summary = load(clip_dir, "PIPELINE_SUMMARY.json")
    body_grounding_quality = load(clip_dir, "body_grounding_quality.json")
    body_joint_quality = load(clip_dir, "body_joint_quality.json")
    body_full_clip_gate = load(clip_dir, "body_full_clip_gate.json")
    frame_compute_plan = load(clip_dir, "frame_compute_plan.json")
    version_stamp = load(clip_dir, "version_stamp.json")
    remote_version_verification = load(clip_dir, "remote_version_verification.json")
    remote_timing = load(clip_dir, "remote_body_dispatch_timing.json")
    virtual_world = load(clip_dir, "virtual_world.json")
    arc = load(clip_dir, "ball_track_arc_solved.json")
    if (clip_dir / "body_mesh_index" / "body_mesh_index.json").exists():
        body_mesh_index = load(clip_dir / "body_mesh_index", "body_mesh_index.json")
    else:
        body_mesh_index = load(clip_dir, "body_mesh_index.json")

    # 0. VERSION STAMP
    out["version_stamp"] = {
        "present": version_stamp is not None,
        "git_head_sha": get(version_stamp, "git_head_sha"),
        "remote_verification_verified": get(version_stamp, "remote_verification", "verified"),
        "remote_verification_expected_sha": get(version_stamp, "remote_verification", "expected_git_head_sha"),
        "remote_verification_remote_sha": get(version_stamp, "remote_verification", "remote_git_head_sha"),
        "dirty_tracked_runtime_files": get(version_stamp, "dirty_tracked_runtime_files"),
        "remote_version_verification_verified": get(remote_version_verification, "verified"),
        "remote_version_verification_checked_file_count": get(remote_version_verification, "checked_file_count"),
        "remote_version_verification_drifted": get(remote_version_verification, "drifted_files"),
    }

    # 1. SLIDE GATE (frozen bar 0.030) + companions + root jumps + body gate
    reason_counts = get(body_grounding_quality, "grounding_metrics", "candidate_phase_rejection_reason_counts", default={})
    out["slide_gate"] = {
        "grounding_metrics.max_foot_lock_slide_m": get(body_grounding_quality, "grounding_metrics", "max_foot_lock_slide_m"),
        "grounding_metrics.foot_lock_slide_p95_m": get(body_grounding_quality, "grounding_metrics", "foot_lock_slide_p95_m"),
        "foot_slide_gate_value_m": get(body_grounding_quality, "foot_slide_gate", "value_m"),
        "foot_slide_gate_threshold_m": get(body_grounding_quality, "foot_slide_gate", "threshold_m"),
        "foot_slide_gate_passed": get(body_grounding_quality, "foot_slide_gate", "passed"),
        "blockers": get(body_grounding_quality, "blockers", default=[]),
        "grounding_metrics.max_candidate_phase_slide_m": get(body_grounding_quality, "grounding_metrics", "max_candidate_phase_slide_m"),
        "candidate_phase_rejected_count": get(body_grounding_quality, "grounding_metrics", "candidate_phase_rejected_count"),
        "candidate_phase_rejection_reason_counts": reason_counts,
        "phase_slide_exceeds_lock_gate_count": (reason_counts or {}).get("phase_slide_exceeds_lock_gate", 0),
    }
    out["root_jumps"] = {
        "root_motion_temporal_jump_count": get(body_joint_quality, "summary", "root_motion_temporal_jump_count"),
        "max_root_speed_mps": get(body_joint_quality, "summary", "max_root_speed_mps"),
        "quality_blockers": get(body_joint_quality, "quality_blockers", default=[]),
    }
    out["body_full_clip_gate"] = {
        "passed": get(body_full_clip_gate, "passed"),
        "blockers": get(body_full_clip_gate, "blockers", default=[]),
    }

    # 2. CAMERA MOTION AUTO (verbatim dict incl. any decode_orientation_* first-class keys)
    out["camera_motion_auto"] = get(pipeline_summary, "camera_motion_auto")

    # 3. BALL CHAIN
    ball = {"ball_track_arc_solved_present": arc is not None}
    if isinstance(arc, dict):
        segs = arc.get("segments") or []
        frames = arc.get("frames") or []
        ball["segment_count"] = len(segs)
        ball["segment_status_census"] = dict(Counter(s.get("status") for s in segs))
        ball["frames_total"] = len(frames)
        ball["frames_world_xyz_non_null"] = sum(1 for f in frames if f.get("world_xyz") is not None)
        ball["frames_visible"] = sum(1 for f in frames if f.get("visible") is True)
        ball["kill_reasons"] = arc.get("kill_reasons")
        ball["chain_config_degraded"] = arc.get("chain_config_degraded")
    out["ball_chain"] = ball

    # 4. MESH
    mesh_policy = get(frame_compute_plan, "mesh_coverage_policy", default={}) or {}
    out["mesh"] = {
        "selected_mesh_frame_count": mesh_policy.get("selected_mesh_frame_count"),
        "mesh_fallback": mesh_policy.get("mesh_fallback"),
        "body_mesh_index_chunk_count": (
            len(body_mesh_index.get("windows", [])) if isinstance(body_mesh_index, dict) and "windows" in body_mesh_index
            else (len(body_mesh_index) if isinstance(body_mesh_index, list) else None)
        ),
        "virtual_world_warnings": get(virtual_world, "summary", "warnings"),
    }

    # 5. RUN CONTEXT
    out["run"] = {
        "pipeline_status": get(pipeline_summary, "status"),
        "wall_seconds": get(pipeline_summary, "wall_seconds"),
        "remote_transport": get(remote_timing, "transport"),
        "remote_status": get(remote_timing, "status"),
        "remote_upload_bytes": get(remote_timing, "upload_bytes"),
        "remote_download_bytes": get(remote_timing, "download_bytes"),
        "manifest_present": (clip_dir / "replay_viewer_manifest.json").exists(),
    }
    return out


if __name__ == "__main__":
    clip_dir = Path(sys.argv[1])
    print(json.dumps(extract(clip_dir), indent=2, sort_keys=True, default=str))
