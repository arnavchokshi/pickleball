#!/usr/bin/env python3
"""Stage `tracks.json` + `court_calibration.json` for the 12 pre-registered ASPset-510
BODY-EXT clips (`runs/manager/heldout_eval_ledger.md` row BODY-EXT-1), so each clip has a
self-contained `--inputs` directory for `scripts/racketsport/run_body_video_smoke.py` /
`threed.racketsport.orchestrator --stage body`.

Reads real camera calibration from `raw_provenance/cameras/<subject>-<camera>.json` and
real GT joints from `labels/aspset510_<subject>_<clip>_<camera>/body_world_joints.json`
(both already present under the run dir from the prior methodology/dataset-selection pass;
this script adds no new raw downloads). See
`threed/racketsport/external_gt_aspset510_body_inputs.py` for the adapter math and
`runs/body_external_gt_20260702T040048Z/BODY_INFERENCE_INTEGRATION_TODO.md` for the plan
this implements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.external_gt_aspset510_body_inputs import (  # noqa: E402
    build_court_calibration_payload,
    build_tracks_payload,
    load_camera_calibration_raw,
)
from threed.racketsport.eval.body_gate_report import CORE_BODY_JOINT_NAMES  # noqa: E402
from threed.racketsport.schemas import CourtCalibration, Skeleton3D, Tracks  # noqa: E402

# A structurally-valid but inert placeholder for the pipeline's separate Lane A
# ("pose" stage, RTMW3D) skeleton. `--diagnostic-full-track` schedules every GT-sampled
# frame as `deep_mesh` (real Fast-SAM-3D-Body/BODY-stage compute), so Lane A's own output
# is never consulted as a prediction source for any frame in this lane (see
# `threed.racketsport.eval.body_gate_report._prediction_index`, which only falls back to
# `skeleton3d.json` when `smpl_motion.json` yields no predictions at all -- it will not
# here). This placeholder exists solely to satisfy the "pose" stage's structural
# `skeleton3d.json` contract via `threed.racketsport.body_video_smoke._InputSkeletonPoseRunner`
# (reused-as-is, not run for real), so the remote GPU host does not need a working
# `rtmpose3d`/mmpose Lane A environment for this external-GT lane. Values are a fixed,
# arbitrary small pose (NOT derived from ASPset GT in any way) with near-zero confidence
# so nothing could mistake it for a real measurement if it were ever inspected directly.
_PLACEHOLDER_JOINT_XYZ: dict[str, list[float]] = {
    "nose": [0.0, -0.7, 5.0], "left_eye": [-0.03, -0.72, 5.0], "right_eye": [0.03, -0.72, 5.0],
    "left_ear": [-0.06, -0.7, 5.0], "right_ear": [0.06, -0.7, 5.0],
    "left_shoulder": [-0.2, -0.5, 5.0], "right_shoulder": [0.2, -0.5, 5.0],
    "left_elbow": [-0.3, -0.2, 5.0], "right_elbow": [0.3, -0.2, 5.0],
    "left_wrist": [-0.35, 0.1, 5.0], "right_wrist": [0.35, 0.1, 5.0],
    "left_hip": [-0.15, 0.0, 5.0], "right_hip": [0.15, 0.0, 5.0],
    "left_knee": [-0.15, 0.5, 5.0], "right_knee": [0.15, 0.5, 5.0],
    "left_ankle": [-0.15, 1.0, 5.0], "right_ankle": [0.15, 1.0, 5.0],
}


def build_placeholder_lane_a_skeleton(*, frame_indices: list[int], fps: float, clip: str) -> dict:
    frames = [
        {
            "frame_idx": frame_idx,
            "t": round(frame_idx / fps, 9),
            "joints_world": [_PLACEHOLDER_JOINT_XYZ[name] for name in CORE_BODY_JOINT_NAMES],
            "joint_conf": [0.01] * len(CORE_BODY_JOINT_NAMES),
        }
        for frame_idx in frame_indices
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": float(fps),
        "world_frame": "relative_only",
        "source_model": "external_gt_unused_placeholder",
        "joint_names": list(CORE_BODY_JOINT_NAMES),
        "preview_only": False,
        "players": [{"id": 1, "frames": frames}],
        "provenance": {
            "lane": "A",
            "source": "external_gt_unused_placeholder_bypassed_by_diagnostic_full_track",
            "clip": clip,
            "note": (
                "fixed arbitrary pose, not derived from ASPset-510 GT; never consulted as a "
                "prediction because diagnostic_full_track schedules every frame through the "
                "real BODY (Fast-SAM-3D-Body) stage, whose smpl_motion.json is the sole "
                "prediction source body_gate_report.py's _prediction_index reads from here"
            ),
        },
    }
    Skeleton3D.model_validate(payload)
    return payload

DEFAULT_RUN_DIR = ROOT / "runs" / "body_external_gt_20260702T040048Z"

# (subject, clip, camera) -- must match runs/manager/heldout_eval_ledger.md row BODY-EXT-1
# and the label directories already built by build_external_gt_aspset510_labels.py.
CLIPS: tuple[tuple[str, str, str], ...] = (
    ("1e28", "0001", "left"),
    ("1e28", "0006", "right"),
    ("1e28", "0008", "left"),
    ("1e28", "0011", "left"),
    ("1e28", "0014", "mid"),
    ("1e28", "0018", "mid"),
    ("8a59", "0006", "mid"),
    ("8a59", "0008", "mid"),
    ("8a59", "0011", "right"),
    ("8a59", "0015", "left"),
    ("8a59", "0016", "mid"),
    ("8a59", "0018", "left"),
)


def clip_id(subject: str, clip: str, camera: str) -> str:
    return f"aspset510_{subject}_{clip}_{camera}"


def stage_one_clip(
    *,
    subject: str,
    clip: str,
    camera: str,
    run_dir: Path,
    out_root: Path,
) -> dict[str, object]:
    cam_path = run_dir / "raw_provenance" / "cameras" / f"{subject}-{camera}.json"
    label_path = run_dir / "labels" / clip_id(subject, clip, camera) / "body_world_joints.json"
    if not cam_path.is_file():
        raise FileNotFoundError(f"missing real camera calibration: {cam_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"missing real GT label file: {label_path}")

    cam_payload = json.loads(cam_path.read_text(encoding="utf-8"))
    K, R, t_m = load_camera_calibration_raw(cam_payload)

    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    samples = label_payload["samples"]
    frame_indices = [int(sample["frame_index"]) for sample in samples]
    joints = [sample["joints_world"] for sample in samples]
    all_pts = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    reference_plane_z_m = float(np.median(all_pts[:, 2]))
    reference_plane_center_xy_m = (float(np.median(all_pts[:, 0])), float(np.median(all_pts[:, 1])))

    calibration_payload = build_court_calibration_payload(
        K=K,
        R=R,
        t_m=t_m,
        source_label=f"aspset510_calibrated_multiview_mocap:{subject}-{camera}",
        reference_plane_z_m=reference_plane_z_m,
        reference_plane_center_xy_m=reference_plane_center_xy_m,
        reference_plane_half_extent_m=1.0,
    )
    # Fail loudly before writing anything if the payload is not schema-valid.
    CourtCalibration.model_validate(calibration_payload)

    tracks_payload, notes = build_tracks_payload(
        frame_indices=frame_indices,
        joints_world_m_by_frame=joints,
        K=K,
        R=R,
        t_m=t_m,
        fps=50.0,
        player_id=1,
        side="near",
        role="single",
    )
    Tracks.model_validate(tracks_payload)

    out_dir = out_root / clip_id(subject, clip, camera)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "court_calibration.json").write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "tracks.json").write_text(json.dumps(tracks_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    placeholder_skeleton = build_placeholder_lane_a_skeleton(
        frame_indices=frame_indices, fps=50.0, clip=clip_id(subject, clip, camera)
    )
    (out_dir / "skeleton3d.json").write_text(
        json.dumps(placeholder_skeleton, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "clip": clip_id(subject, clip, camera),
        "out_dir": str(out_dir),
        "gt_sample_count": len(samples),
        "tracks_frame_count": len(tracks_payload["players"][0]["frames"]),
        "reference_plane_z_m": reference_plane_z_m,
        "reference_plane_center_xy_m": list(reference_plane_center_xy_m),
        "dropped_frame_notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-root", type=Path, default=None, help="Defaults to <run-dir>/staged_inputs")
    parser.add_argument("--clips", nargs="*", default=None, help="Optional subset, e.g. 1e28_0001_left")
    args = parser.parse_args(argv)

    out_root = args.out_root or (args.run_dir / "staged_inputs")
    out_root.mkdir(parents=True, exist_ok=True)

    clips = CLIPS
    if args.clips:
        wanted = set(args.clips)
        clips = tuple(c for c in CLIPS if f"{c[0]}_{c[1]}_{c[2]}" in wanted)

    results = []
    for subject, clip, camera in clips:
        result = stage_one_clip(subject=subject, clip=clip, camera=camera, run_dir=args.run_dir, out_root=out_root)
        results.append(result)
        print(json.dumps(result, indent=2))

    summary_path = out_root / "staging_summary.json"
    summary_path.write_text(json.dumps({"clips": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
