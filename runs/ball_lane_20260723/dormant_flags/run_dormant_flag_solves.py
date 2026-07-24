#!/usr/bin/env python3
"""WS2.3 dormant-flag measurement solves (measurement only, VERIFIED=0).

Re-runs ``run_default_ball_arc_chain`` over the 2026-07-05 ball_f1 original
inputs for four experiment-flag variants:

  baseline             all dormant flags off (must reproduce the archived
                       2026-07-05 three_clip_default_chain solve or the delta
                       is documented as code drift)
  joint_anchor_search  enable_joint_anchor_search=True (pinning stays off;
                       never combined with the rejected RANSAC arm)
  ukf_fallback         enable_ukf_fallback=True (sidecar only; recovery
                       policy v2 stays off -- the chain rejects combining them)
  both                 joint_anchor_search + ukf_fallback

Nothing here changes defaults or promotes anything. Outputs land under
``solves/<clip>/<variant>/`` next to this script. Input parity policy:

* ``ball_track`` / ``court_calibration`` / ``net_plane`` shas are pinned to
  the values recorded by the archived ``ball_chain_manifest.json`` files and
  verified before every solve (hard fail on mismatch).
* ``contact_windows`` / ``skeleton3d`` / ``rally_spans`` were consumed by the
  archived solves (seed pre-pass ran: seed_anchor_count 14/11/8) but their
  shas were NOT recorded by the 2026-07-05 manifests; we pass the ball_f1
  copies and record their shas here, flagged ``archived_sha_recorded=false``.
* ``frame_times`` was not provided in 2026-07-05 (absent from the archived
  manifest inputs) and is not provided here.

The script chdirs into ``--inputs-root`` and passes repo-relative paths so
the artifact ``inputs`` echo matches the archived artifact strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import shutil
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GENERATED_AT = "2026-07-24T00:00:00Z"  # pinned: byte-comparable artifacts across variants
BALL_TYPE = "outdoor"  # matches the archived physics_parameters echo
F1_BASE = "runs/lanes/ball_f1_three_clip_runs_20260705"

NET_PLANE_SHA = "8e6d67898b7bcf8023bacd3c806471ab0f5b179d1b72671a4e6d781d47f7a057"

# Pinned to the archived ball_chain_manifest.json input records (2026-07-05).
CLIPS: dict[str, dict[str, str | None]] = {
    "burlington_gold_0300_low_steep_corner": {
        "ball_track_sha256": "813f7f52ecca9383a8d866b463d214a3ce43485369a6ea18d6a01a4b5d5f4128",
        "court_calibration_sha256": "aec27ecc0f377ee930f363173032316332811ac573199a633ebe499747abed1d",
        "net_plane_sha256": NET_PLANE_SHA,
    },
    "wolverine_mixed_0200_mid_steep_corner": {
        "ball_track_sha256": "bff541ab90db60b2fe6c37b18b5895aecea982da20a5e545e4187c0a6a68e5f2",
        "court_calibration_sha256": "fb4e6f7f54d2c40e2c7b491e436261f747240945a6f0d154c4dd943e28edbacf",
        "net_plane_sha256": NET_PLANE_SHA,
    },
    "outdoor_webcam_iynbd_1500_long_high_baseline": {
        "ball_track_sha256": "d6e7e5d722a7028d1e57349ab7b16a0581d2a43fad0dda5f4109d2079eb7f086",
        "court_calibration_sha256": "bb7bb05b55af2799f49308be6e5b5e5b184af4862c1de8efe05e794f71e8dbd9",
        "net_plane_sha256": NET_PLANE_SHA,
    },
}

VARIANTS: dict[str, dict[str, bool]] = {
    "baseline": {},
    "joint_anchor_search": {"enable_joint_anchor_search": True},
    "ukf_fallback": {"enable_ukf_fallback": True},
    "both": {"enable_joint_anchor_search": True, "enable_ukf_fallback": True},
}

OPTIONAL_INPUTS = ("contact_windows", "skeleton3d", "rally_spans")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inputs-root",
        type=Path,
        required=True,
        help="Checkout holding the ball_f1 originals (read-only; e.g. the main checkout).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "solves",
        help="Output root; artifacts land in <out-dir>/<clip>/<variant>/.",
    )
    parser.add_argument("--clip", action="append", default=[], help="Clip filter (repeatable).")
    parser.add_argument("--variant", action="append", default=[], help="Variant filter (repeatable).")
    parser.add_argument("--force", action="store_true", help="Re-solve even if outputs already exist.")
    args = parser.parse_args(argv)

    inputs_root = args.inputs_root.resolve()
    out_root = args.out_dir.resolve()
    clips = args.clip or list(CLIPS)
    variants = args.variant or list(VARIANTS)
    for clip in clips:
        if clip not in CLIPS:
            raise SystemExit(f"unknown clip: {clip}")
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant}")

    # Relative input paths from inputs_root so the artifact input echo matches
    # the archived artifacts byte-for-byte where the schema allows.
    os.chdir(inputs_root)

    from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain

    for clip in clips:
        pinned = CLIPS[clip]
        clip_base = Path(F1_BASE) / clip
        ball_track = clip_base / "ball_track.json"
        calibration = clip_base / "court_calibration.json"
        net_plane = clip_base / "net_plane.json"
        for path, key in (
            (ball_track, "ball_track_sha256"),
            (calibration, "court_calibration_sha256"),
            (net_plane, "net_plane_sha256"),
        ):
            actual = _sha256(path)
            if actual != pinned[key]:
                raise SystemExit(
                    f"input sha mismatch for {clip} {path}: expected {pinned[key]}, got {actual}"
                )
        optional_paths: dict[str, Path | None] = {}
        for name in OPTIONAL_INPUTS:
            candidate = clip_base / f"{name}.json"
            optional_paths[name] = candidate if candidate.is_file() else None

        for variant in variants:
            flags = VARIANTS[variant]
            out_dir = out_root / clip / variant
            manifest_path = out_dir / "ball_chain_manifest.json"
            if manifest_path.is_file() and not args.force:
                print(f"SKIP (exists): {clip}/{variant}")
                continue
            out_rel = Path(os.path.relpath(out_dir, inputs_root))
            started = time.monotonic()
            result = run_default_ball_arc_chain(
                clip=clip,
                ball_track_path=ball_track,
                court_calibration_path=calibration,
                out_dir=out_rel,
                contact_windows_path=optional_paths["contact_windows"],
                skeleton3d_path=optional_paths["skeleton3d"],
                net_plane_path=net_plane,
                rally_spans_path=optional_paths["rally_spans"],
                frame_times_path=None,
                ball_type=BALL_TYPE,
                generated_at=GENERATED_AT,
                **flags,
            )
            wall = time.monotonic() - started
            # Sha-matched calibration copy so read-mode characterization can
            # verify it against this solve's ball_chain_manifest.json.
            shutil.copyfile(calibration, out_dir / "court_calibration.json")
            log = {
                "clip": clip,
                "variant": variant,
                "flags": {key: bool(value) for key, value in sorted(flags.items())},
                "generated_at_pinned": GENERATED_AT,
                "ball_type": BALL_TYPE,
                "inputs": {
                    "ball_track": {"path": str(ball_track), "sha256": pinned["ball_track_sha256"], "archived_sha_recorded": True},
                    "court_calibration": {"path": str(calibration), "sha256": pinned["court_calibration_sha256"], "archived_sha_recorded": True},
                    "net_plane": {"path": str(net_plane), "sha256": pinned["net_plane_sha256"], "archived_sha_recorded": True},
                    **{
                        name: (
                            {"path": str(path), "sha256": _sha256(path), "archived_sha_recorded": False}
                            if path is not None
                            else None
                        )
                        for name, path in sorted(optional_paths.items())
                    },
                    "frame_times": None,
                },
                "status": result.get("status"),
                "summary": result.get("summary"),
                "wall_seconds": round(wall, 3),
            }
            (out_dir / "solve_result.json").write_text(
                json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"DONE {clip}/{variant}: status={result.get('status')} wall={wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
