#!/usr/bin/env python3
"""Run real BODY (Fast SAM-3D-Body) inference against one staged ASPset-510 clip.

Thin wrapper around `threed.racketsport.body_video_smoke.run_body_video_smoke` that
injects `threed.racketsport.external_gt_precomputed_calibration_runner.
PrecomputedCalibrationRunner` for the "calibration" stage (see that module's docstring
for why the pipeline's default `ManualCalibrationRunner` cannot be used for
external-ground-truth footage) and always sets `--diagnostic-full-track` (ASPset-510 has
no ball/contact evidence for the production contact-aware tier rule to schedule against;
every GT-sampled frame must be explicitly scheduled `deep_mesh`, exactly like the
production path's diagnostic escape hatch already supports -- see
`threed.racketsport.body_video_smoke._prepare_frame_plan`'s docstring for why this must
never be read as production BODY compute-cost evidence).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_video_smoke import run_body_video_smoke  # noqa: E402
from threed.racketsport.external_gt_precomputed_calibration_runner import (  # noqa: E402
    PrecomputedCalibrationRunner,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fast-sam-repo", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument(
        "--body-detector-name",
        default="",
        help=(
            'Detector backend passed to FastSAM-3D-Body. Defaults to "" (disabled) since '
            "tracks.json already provides real bboxes (see external_gt_aspset510_body_inputs.py)."
        ),
    )
    parser.add_argument(
        "--body-fov-name",
        default="",
        help=(
            'FOV/depth-prior backend passed to FastSAM-3D-Body. Defaults to "" (runtime default) '
            "because the MoGe FOV checkpoint (models/MANIFEST.json id=moge_2_vitl_normal) is only "
            '"available_on_h100" and this lane runs on an A100 spot VM without it.'
        ),
    )
    args = parser.parse_args(argv)

    calibration_runner = PrecomputedCalibrationRunner(
        source_note=(
            f"real ASPset-510 camera calibration for clip {args.clip}, built by "
            "scripts/racketsport/build_external_gt_aspset510_body_inputs.py from "
            "raw_provenance/cameras/<subject>-<camera>.json (not manual taps, not video-derived)"
        )
    )

    try:
        report = run_body_video_smoke(
            clip=args.clip,
            inputs_dir=args.inputs,
            video_path=args.video,
            run_dir=args.out,
            tracking_mode="precomputed_tracks",
            sport="pickleball",
            max_frames=args.max_frames,
            manifest_path=args.manifest,
            max_players=1,
            court_margin_m=1000.0,
            fast_sam_repo=args.fast_sam_repo,
            body_detector_name=args.body_detector_name,
            body_fov_name=args.body_fov_name,
            diagnostic_full_track=True,
            extra_runners={"calibration": calibration_runner},
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: external-GT BODY inference failed before report write: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
