#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.camera_motion import (  # noqa: E402
    CameraMotionParams,
    estimate_camera_motion,
    raft_small_backend_status,
    write_camera_motion_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate preview-only per-frame camera-motion transforms that map "
            "current-frame pixels into a calibration reference frame."
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video path.")
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="court_calibration.json containing the reference frame and homography.",
    )
    parser.add_argument("--tracks", type=Path, default=None, help="Optional tracks.json for padded person masks.")
    parser.add_argument("--out", required=True, type=Path, help="Output camera_motion.json path.")
    parser.add_argument(
        "--estimator",
        choices=["hardened", "legacy"],
        default="hardened",
        help="Estimator profile. hardened enables LK MAD filtering plus temporal smoothing; legacy preserves the previous LK+RANSAC behavior for ablations.",
    )
    parser.add_argument(
        "--no-person-mask",
        action="store_true",
        help="Disable track-bbox person masking for masked-vs-unmasked ablations.",
    )
    parser.add_argument(
        "--flow-backend",
        choices=["lk", "raft-small"],
        default="lk",
        help="Optical-flow backend. raft-small is flag-gated and requires already-cached torchvision weights; this CLI never downloads weights.",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=None,
        help="Optional reference frame override. Defaults to the calibration reference.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Optional directory for static-vs-motion-compensated court overlays.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for smoke runs. Capped runs are preview evidence only.",
    )
    args = parser.parse_args()

    if args.estimator == "legacy":
        params = CameraMotionParams.legacy()
    else:
        params = CameraMotionParams()
    params = CameraMotionParams(
        **{
            **params.__dict__,
            "use_person_masks": not args.no_person_mask,
            "flow_backend": args.flow_backend,
        }
    )
    raft_status = raft_small_backend_status() if args.flow_backend == "raft-small" else {"backend": "raft-small", "status": "not_requested"}
    payload = estimate_camera_motion(
        args.video,
        args.calibration,
        tracks_path=args.tracks,
        reference_frame_idx=args.reference_frame,
        max_frames=args.max_frames,
        diagnostics_dir=args.diagnostics_dir,
        params=params,
    )
    write_camera_motion_json(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out.as_posix(),
                "diagnostics_dir": args.diagnostics_dir.as_posix() if args.diagnostics_dir else None,
                "reference_frame_idx": payload["reference_frame_idx"],
                "raft_backend": raft_status["status"],
                "summary": payload["summary"],
                "verified": payload["verified"],
                "not_gate_verified": payload["not_gate_verified"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
