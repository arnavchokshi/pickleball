#!/usr/bin/env python3
"""Trimmed, read-only field inventory for the one_world_v1 design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z"
DEMO = ROOT / "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/cpu_events_full/pbvision_11min_20260713"
TRACK_I = ROOT / "runs/lanes/trackI_placefuse_20260716/wolverine/placement_trajectory_refined.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def typed(value: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, list):
        answer["length"] = len(value)
        answer["sample"] = value[:3]
    elif isinstance(value, dict):
        answer["keys"] = sorted(value)
    else:
        answer["value"] = value
    return answer


def main() -> int:
    cal = load(RUN / "court_calibration.json")
    trust = load(RUN / "trust_bands.json")
    tracks = load(RUN / "tracks.json")
    placement = load(RUN / "placement.json")
    smpl = load(RUN / "smpl_motion.json")
    ball = load(RUN / "ball_track.json")
    arc = load(RUN / "ball_track_arc_solved.json")
    audio = load(RUN / "audio_onsets_v2.json")
    contacts = load(RUN / "contact_windows.json")
    racket = load(RUN / "racket_pose_estimate.json")
    net = load(RUN / "net_plane.json")
    zones = load(RUN / "court_zones.json")
    rallies = load(RUN / "rally_spans.json")
    world = load(RUN / "virtual_world.json")
    refined_placement = load(TRACK_I)
    demo_ball = load(DEMO / "ball_track.json")
    demo_cal = load(DEMO / "court_calibration.json")

    track_frame = tracks["players"][0]["frames"][0]
    placement_frame = placement["players"][0]["frames"][0]
    smpl_frame = smpl["players"][0]["frames"][0]
    arc_frame = arc["frames"][0]
    arc_segment = arc["segments"][0]
    contact = contacts["events"][0]
    racket_frame = racket["players"][0]["frames"][0]
    track_i_frame = refined_placement["players"][0]["frames"][0]
    track_i_refine = track_i_frame["placement_trajectory_refinement"]

    result = {
        "court_calibration.json": {
            "path": str(RUN / "court_calibration.json"),
            "schema_version": typed(cal["schema_version"]),
            "coordinate_frame": typed(cal.get("coordinate_frame")),
            "homography": typed(cal["homography"]),
            "intrinsics": typed(cal["intrinsics"]),
            "extrinsics": typed(cal["extrinsics"]),
            "metric_confidence": typed(cal.get("metric_confidence")),
            "capture_quality": typed(cal["capture_quality"]),
            "source": typed(cal.get("source")),
            "coordinate_contract": typed(cal.get("coordinate_contract")),
            "inline_trust_band": typed(cal.get("trust_band")),
            "companion_court_trust_band": trust["court"],
        },
        "tracks.json": {
            "path": str(RUN / "tracks.json"),
            "fps": typed(tracks["fps"]),
            "player_count": len(tracks["players"]),
            "player_fields": sorted(tracks["players"][0]),
            "frame_fields": sorted(track_frame),
            "frame_sample": {key: track_frame[key] for key in ("t", "bbox", "world_xy", "conf")},
            "placement_provenance": typed(tracks.get("placement_provenance")),
            "repair_markers": typed((tracks.get("placement_provenance") or {}).get("confidence_repairs")),
        },
        "placement.json": {
            "path": str(RUN / "placement.json"),
            "fps": typed(placement["fps"]),
            "world_declarations": {
                "homography_pixel_convention": placement.get("homography_pixel_convention"),
                "undistort_applied": placement["undistort_applied"],
            },
            "frame_fields": sorted(placement_frame),
            "frame_sample": {
                key: placement_frame[key]
                for key in ("frame_idx", "t", "fused_world_xy", "smoothed_world_xy", "covariance_m2", "signals")
            },
        },
        "placement_trajectory_refined.json": {
            "path": str(TRACK_I),
            "same_run_as_manager_wolverine": False,
            "top_fields": sorted(refined_placement),
            "artifact_type": refined_placement.get("artifact_type"),
            "coordinate_space": refined_placement.get("coordinate_space"),
            "world_frame": refined_placement.get("world_frame"),
            "fps": refined_placement.get("fps"),
            "preview_band": refined_placement.get("preview_band"),
            "VERIFIED": refined_placement.get("VERIFIED"),
            "player_id": refined_placement["players"][0]["id"],
            "frame_fields": sorted(track_i_frame),
            "refinement_fields": sorted(track_i_refine),
            "refinement_sample": {
                key: track_i_refine[key]
                for key in (
                    "refined_transl_world",
                    "rigid_correction_xyz_m",
                    "covariance_m2",
                    "correction_magnitude_m",
                    "provenance",
                )
            },
        },
        "smpl_motion.json": {
            "path": str(RUN / "smpl_motion.json"),
            "fps": typed(smpl["fps"]),
            "model": typed(smpl["model"]),
            "world_frame": typed(smpl["world_frame"]),
            "skeleton_stride": typed(smpl.get("skeleton_stride")),
            "player_count": len(smpl["players"]),
            "frame_fields": sorted(smpl_frame),
            "frame_sample": {
                "frame_idx": smpl_frame["frame_idx"],
                "t": smpl_frame["t"],
                "transl_world": smpl_frame["transl_world"],
                "joints_world_count": len(smpl_frame["joints_world"]),
                "left_wrist_idx9": smpl_frame["joints_world"][9],
                "right_wrist_idx10": smpl_frame["joints_world"][10],
                "left_wrist_conf": smpl_frame["joint_conf"][9],
                "right_wrist_conf": smpl_frame["joint_conf"][10],
            },
        },
        "ball_track.json": {
            "path": str(RUN / "ball_track.json"),
            "fps": typed(ball["fps"]),
            "source": typed(ball["source"]),
            "frame_fields": sorted(ball["frames"][0]),
            "frame_sample": ball["frames"][0],
            "confidence_repair_marker": typed(ball["frames"][0].get("conf_source")),
            "ball_candidates_sibling_exists": (RUN / "ball_candidates.json").exists(),
        },
        "ball_track_arc_solved.json": {
            "path": str(RUN / "ball_track_arc_solved.json"),
            "policy": arc["policy"],
            "status": arc["status"],
            "kill_reasons": arc["kill_reasons"],
            "frame_fields": sorted(arc_frame),
            "frame_sample": {key: arc_frame.get(key) for key in ("t", "world_xyz", "conf", "band", "sigma_m", "approx")},
            "anchor_fields": sorted(arc["anchors"][0]),
            "anchor_sample": arc["anchors"][0],
            "segment_fields": sorted(arc_segment),
            "segment_sample": {
                key: arc_segment.get(key)
                for key in (
                    "segment_id",
                    "frame_start",
                    "frame_end",
                    "anchors_used",
                    "reprojection_rmse_px",
                    "physical_sanity",
                    "status",
                )
            },
            "segment_budget_exceeded": "segment_budget_exceeded" in arc.get("kill_reasons", []),
            "ball_arc_render_sibling_exists": (RUN / "ball_arc_render.json").exists(),
        },
        "audio_onsets_v2.json": {
            "path": str(RUN / "audio_onsets_v2.json"),
            "frame_rate": audio["frame_rate"],
            "status": audio["status"],
            "not_gate_verified": audio["not_gate_verified"],
            "trusted_for_contact": audio["trusted_for_contact"],
            "onset_count": len(audio["onsets"]),
            "onset_fields": sorted(audio["onsets"][0]) if audio["onsets"] else [],
            "pop_band_ratio_first": audio["onsets"][0].get("features", {}).get("pop_band_ratio") if audio["onsets"] else None,
        },
        "contact_windows.json": {
            "path": str(RUN / "contact_windows.json"),
            "event_count": len(contacts["events"]),
            "event_fields": sorted(contact),
            "event_sample": contact,
            "refined_sibling_exists": (RUN / "contact_windows_refined_v1.json").exists(),
        },
        "racket_pose_estimate.json": {
            "path": str(RUN / "racket_pose_estimate.json"),
            "world_frame": racket["world_frame"],
            "translation_unit": racket["translation_unit"],
            "render_only": racket["render_only"],
            "not_for_detection_metrics": racket["not_for_detection_metrics"],
            "trust": racket["trust"],
            "frame_fields": sorted(racket_frame),
            "frame_sample": {
                key: racket_frame.get(key)
                for key in ("frame", "t", "pose_se3", "conf", "reprojection_error_px", "ambiguous", "world_frame", "translation_unit", "source")
            },
            "racket_pose_sibling_exists": (RUN / "racket_pose.json").exists(),
            "hypotheses_sibling_exists": (RUN / "racket_pose_hypotheses.json").exists(),
            "racket_candidates_sibling_exists": (RUN / "racket_candidates.json").exists(),
        },
        "net_plane.json": {
            "path": str(RUN / "net_plane.json"),
            "plane": net["plane"],
            "endpoints": net["endpoints"],
            "center_height_in": net["center_height_in"],
            "post_height_in": net["post_height_in"],
        },
        "court_zones.json": {
            "path": str(RUN / "court_zones.json"),
            "zone_names": sorted(zones["zones"]),
            "court_polygon_m": zones["zones"]["court"],
        },
        "rally_spans.json": {
            "path": str(RUN / "rally_spans.json"),
            "span_count": len(rallies["spans"]),
            "span_fields": sorted(rallies["spans"][0]),
            "span_sample": rallies["spans"][0],
            "not_ground_truth": rallies["not_ground_truth"],
        },
        "virtual_world.json": {
            "path": str(RUN / "virtual_world.json"),
            "artifact_type": world["artifact_type"],
            "world_frame": world["world_frame"],
            "fps": world["fps"],
            "summary_fields": sorted(world["summary"]),
        },
        "demo_reality": {
            "run_dir": str(DEMO),
            "ball_fields": sorted(demo_ball),
            "ball_world_xyz_count": sum(frame.get("world_xyz") is not None for frame in demo_ball["frames"]),
            "court_coordinate_frame": demo_cal.get("coordinate_frame"),
            "court_metric_confidence": demo_cal.get("metric_confidence"),
            "has_tracks": (DEMO / "tracks.json").exists(),
            "has_smpl_motion": (DEMO / "smpl_motion.json").exists(),
            "has_contact_windows": (DEMO / "contact_windows.json").exists(),
            "has_audio_onsets_v2": (DEMO / "audio_onsets_v2.json").exists(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
