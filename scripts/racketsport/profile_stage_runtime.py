#!/usr/bin/env python3
"""GLUE-3 speed-budget CLI: assemble a per-stage runtime budget table.

Standalone tool -- does not touch the orchestrator/pipeline_cli. It combines:

1. **Fresh, cheap, local measurements** (decode, calibration, world build) run
   right now against the 4 accepted eval clips -- these are CPU-only,
   sub-second operations, so there is no reason to trust a stale number.
2. **Already-recorded A100 evidence** for the expensive, GPU-bound stages
   (detection+tracking, ball TrackNetV3/WASB inference, BODY mesh, person-ReID
   global association) -- loaded from specific existing run artifacts in this
   repo (see ``EVIDENCE_SOURCES`` below), not re-measured, because
   re-measuring them requires a GPU lease, out of scope for this lane.

All numbers are normalized to seconds of compute per minute of source video
so every stage can be compared on one scale, and the top cost centers can be
ranked. Run with no arguments to regenerate the full table used by
``RUNTIME_BUDGET.md``:

    python3 scripts/racketsport/profile_stage_runtime.py --out runs/<run_dir>/profiling/runtime_budget.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.stage_runtime_budget import (
    StageCost,
    fixed_cost_amortized_per_minute_video,
    load_body_cost_model_evidence,
    load_detection_tracking_fps_evidence,
    load_offline_person_authority_evidence,
    load_tracknet_metadata_evidence,
    load_wasb_metadata_evidence,
    measure_decode_cost,
    per_frame_ms_to_seconds_per_minute_video,
    stage_cost_from_wall_clock,
)

EVAL_CLIPS: dict[str, dict[str, Any]] = {
    "burlington_gold_0300_low_steep_corner": {"fps": 60.0, "duration_s": 10.01},
    "wolverine_mixed_0200_mid_steep_corner": {"fps": 30.0, "duration_s": 10.0},
    "outdoor_webcam_iynbd_1500_long_high_baseline": {"fps": 60.0, "duration_s": 19.183333},
    "indoor_doubles_fwuks_0500_long_mid_baseline": {"fps": 30.0, "duration_s": 30.03},
}

# Evidence sources: already-recorded A100/GPU run artifacts this script reads
# (does not re-run). Paths are relative to the repo root. See RUNTIME_BUDGET.md
# for the narrative citation of each.
EVIDENCE_SOURCES = {
    "tracknetv3_a100_full_cvat": {
        "burlington_gold_0300_low_steep_corner": ROOT
        / "runs/ball_a100_full_cvat_20260630T0925Z/burlington_gold_0300_low_steep_corner/tracknet_full_a100/tracknet_metadata.json",
        "wolverine_mixed_0200_mid_steep_corner": ROOT
        / "runs/ball_a100_full_cvat_20260630T0925Z/wolverine_mixed_0200_mid_steep_corner/tracknet_full_a100/tracknet_metadata.json",
        "outdoor_webcam_iynbd_1500_long_high_baseline": ROOT
        / "runs/ball_a100_full_cvat_20260630T0925Z/outdoor_webcam_iynbd_1500_long_high_baseline/tracknet_full_a100/tracknet_metadata.json",
    },
    "tracknetv3_a100_official_heatmap_outdoor": ROOT
    / "runs/ball_goal_m1_official_tracknet_outdoor_heatstream_20260701T150210Z_a100/outdoor_webcam_iynbd_1500_long_high_baseline/official_tracknet_heatmap/tracknet_metadata.json",
    "tracknetv3_a100_committed_anchor": ROOT / "runs/gpu_pipeline_cost_model_20260701/model_assumptions.json",
    "wasb_a100_outdoor_full": ROOT
    / "runs/ball_goal_m8_wasb_heldout_20260701T162903Z_a100/outdoor_webcam_iynbd_1500_long_high_baseline/wasb_tennis_full/wasb_metadata.json",
    "detection_tracking_a100_substrate_outdoor": ROOT
    / "runs/a100_substrate_20260630T0809Z/outdoor_0000_1150_mobile_replay/run_summary.json",
    "detection_tracking_champion_recipe": ROOT
    / "runs/phase2/trk_people_id_goal_20260701T030347Z/yolo26m_base_recall_1920_eval/burlington_gold_0300_low_steep_corner/base_yolo26m_adaptivefulltb3_mindet3_b8_img1920_conf005_rolelock/metrics.json",
    "association_reid_global": {
        "burlington_gold_0300_low_steep_corner": ROOT
        / "runs/phase2/trk_offline_authority_20260701T205912Z/burlington_gold_0300_low_steep_corner/offline_authority_summary.json",
        "wolverine_mixed_0200_mid_steep_corner": ROOT
        / "runs/phase2/trk_offline_authority_20260701T205912Z/wolverine_mixed_0200_mid_steep_corner/offline_authority_summary.json",
        "outdoor_webcam_iynbd_1500_long_high_baseline": ROOT
        / "runs/phase2/trk_offline_authority_20260701T205912Z/outdoor_webcam_iynbd_1500_long_high_baseline/offline_authority_summary.json",
    },
    "body_cost_model_assumptions": ROOT / "runs/gpu_pipeline_cost_model_20260701/model_assumptions.json",
}

CAPTURE_SIDECAR_DIR = ROOT / "runs/eval0/prototype_gate_h100_v2"
EVAL_CLIP_SOURCE_DIR = ROOT / "eval_clips/ball"


def measure_decode_stage() -> list[StageCost]:
    costs = []
    for clip_id in EVAL_CLIPS:
        clip_path = EVAL_CLIP_SOURCE_DIR / clip_id / "source.mp4"
        if not clip_path.exists():
            continue
        costs.append(measure_decode_cost(clip_path, backend="cpu"))
    return costs


def measure_calibration_stage() -> list[StageCost]:
    costs = []
    for clip_id in EVAL_CLIPS:
        sidecar = CAPTURE_SIDECAR_DIR / clip_id / "capture_sidecar.json"
        if not sidecar.exists():
            continue
        out_dir = ROOT / "runs" / ".tmp_profile_calibration" / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "scripts/racketsport/calibrate.py"),
            "--sidecar",
            str(sidecar),
            "--sport",
            "pickleball",
            "--out",
            str(out_dir),
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        costs.append(
            stage_cost_from_wall_clock(
                stage="calibration",
                wall_seconds=elapsed,
                clip_seconds_processed=EVAL_CLIPS[clip_id]["duration_s"],
                basis="fresh local `calibrate.py` subprocess timing (solvePnP from capture_sidecar.json, fixed per-clip cost)",
                source=str(sidecar),
                notes=f"elapsed_s={elapsed:.3f} (includes python interpreter startup; a warm in-process call is faster)",
            )
        )
    return costs


def measure_world_build_stage() -> list[StageCost]:
    """World build is pure JSON assembly (no model inference); measure it once
    against the most complete available input set (Burlington: court
    calibration + tracks + smpl_motion + skeleton3d + ball_track)."""

    clip_dir = CAPTURE_SIDECAR_DIR / "burlington_gold_0300_low_steep_corner"
    required = {
        "--court-calibration": clip_dir / "court_calibration.json",
        "--ball-track": clip_dir / "ball_track.json",
        "--tracks": clip_dir / "tracks.json",
        "--smpl-motion": clip_dir / "smpl_motion.json",
        "--skeleton3d": clip_dir / "skeleton3d.json",
    }
    if not all(p.exists() for p in required.values()):
        return []
    out_dir = ROOT / "runs" / ".tmp_profile_world_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "virtual_world.json"
    command = [sys.executable, str(ROOT / "scripts/racketsport/build_virtual_world.py")]
    for flag, path in required.items():
        command.extend([flag, str(path)])
    command.extend(["--out", str(out_path)])
    started = time.perf_counter()
    subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return [
        stage_cost_from_wall_clock(
            stage="world_build",
            wall_seconds=elapsed,
            clip_seconds_processed=EVAL_CLIPS["burlington_gold_0300_low_steep_corner"]["duration_s"],
            basis="fresh local `build_virtual_world.py` subprocess timing (JSON merge, no model inference)",
            source=str(clip_dir),
            notes=f"elapsed_s={elapsed:.3f} out_size_bytes={out_path.stat().st_size}",
        )
    ]


def load_gpu_evidence_stage() -> list[StageCost]:
    costs: list[StageCost] = []

    for clip_id, path in EVIDENCE_SOURCES["tracknetv3_a100_full_cvat"].items():
        if path.exists():
            costs.append(load_tracknet_metadata_evidence(path, video_fps=EVAL_CLIPS[clip_id]["fps"]))

    path = EVIDENCE_SOURCES["tracknetv3_a100_official_heatmap_outdoor"]
    if path.exists():
        costs.append(load_tracknet_metadata_evidence(path, video_fps=60.0))

    path = EVIDENCE_SOURCES["tracknetv3_a100_committed_anchor"]
    if path.exists():
        with path.open() as handle:
            assumptions = json.load(handle)
        ms_per_frame = float(assumptions["tracknet_a100_seconds_per_video_frame"]) * 1000.0
        costs.append(
            StageCost(
                stage="ball_inference_tracknetv3_anchor",
                seconds_per_minute_video=per_frame_ms_to_seconds_per_minute_video(ms_per_frame, video_fps=60.0),
                basis="repo-committed A100 TrackNetV3 evidence anchor used for GPU cost-model planning",
                source=str(path),
                notes=f"ms_per_frame={ms_per_frame:.2f}",
            )
        )

    path = EVIDENCE_SOURCES["wasb_a100_outdoor_full"]
    if path.exists():
        costs.append(load_wasb_metadata_evidence(path))

    path = EVIDENCE_SOURCES["detection_tracking_a100_substrate_outdoor"]
    if path.exists():
        costs.append(
            load_detection_tracking_fps_evidence(path, video_fps=60.0, fps_key="sustained_processed_fps")
        )

    path = EVIDENCE_SOURCES["detection_tracking_champion_recipe"]
    if path.exists():
        cost = load_detection_tracking_fps_evidence(path, video_fps=60.0, fps_key="effective_fps")
        costs.append(
            StageCost(
                stage="detection_tracking_champion_recipe",
                seconds_per_minute_video=cost.seconds_per_minute_video,
                basis=cost.basis + " (2026-07-01 champion recipe candidate, one representative config)",
                source=cost.source,
                notes=cost.notes,
            )
        )

    for clip_id, path in EVIDENCE_SOURCES["association_reid_global"].items():
        if path.exists():
            costs.append(load_offline_person_authority_evidence(path, video_fps=EVAL_CLIPS[clip_id]["fps"]))

    path = EVIDENCE_SOURCES["body_cost_model_assumptions"]
    if path.exists():
        with path.open() as handle:
            assumptions = json.load(handle)
        setup_seconds = float(assumptions["body_setup_seconds"])
        for scenario_name, pframes_per_s in assumptions["scheduling_scenarios_pframes_per_video_second"].items():
            cost = load_body_cost_model_evidence(path, scenario_person_frames_per_video_second=float(pframes_per_s))
            costs.append(
                StageCost(
                    stage=f"body_mesh_a100_{scenario_name}",
                    seconds_per_minute_video=cost.seconds_per_minute_video,
                    basis=cost.basis,
                    source=cost.source,
                    notes=cost.notes,
                )
            )
        costs.append(
            StageCost(
                stage="body_mesh_a100_setup_amortized_10min_job",
                seconds_per_minute_video=fixed_cost_amortized_per_minute_video(setup_seconds, video_minutes=10.0),
                basis="A100 Fast SAM-3D-Body model setup/warmup, amortized over a 10-minute job",
                source=str(path),
                notes=f"setup_seconds={setup_seconds}",
            )
        )

    return costs


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the GLUE-3 runtime budget table.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-fresh", action="store_true", help="Skip fresh decode/calibration/world-build timing (evidence-only).")
    args = parser.parse_args()

    stages: list[StageCost] = []
    if not args.skip_fresh:
        stages.extend(measure_decode_stage())
        stages.extend(measure_calibration_stage())
        stages.extend(measure_world_build_stage())
    stages.extend(load_gpu_evidence_stage())

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_glue3_runtime_budget",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_clips": EVAL_CLIPS,
        "stages": [s.to_dict() for s in stages],
        "not_ground_truth": True,
        "notes": [
            "decode/calibration/world_build rows are fresh local CPU timings measured by this script.",
            "All other rows are loaded from already-recorded A100/GPU run evidence in this repo (not re-run here).",
            "Every row is normalized to seconds of compute per minute of source video for cross-stage comparison.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
