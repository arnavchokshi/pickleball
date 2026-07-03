#!/usr/bin/env python3
"""Audit pose temporal smoothing jitter before and after refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.pose_temporal import (  # noqa: E402
    DEFAULT_CORE_BODY_ONE_EURO_BETA,
    DEFAULT_CORE_BODY_ONE_EURO_MINCUTOFF,
    DEFAULT_LOW_CONFIDENCE_JOINT_THRESHOLD,
    DEFAULT_ONE_EURO_BETA,
    DEFAULT_ONE_EURO_MINCUTOFF,
    DEFAULT_WRIST_ONE_EURO_BETA,
    DEFAULT_WRIST_ONE_EURO_MINCUTOFF,
    compare_wrist_peak_timing,
    compute_pose_jitter_audit,
    refine_lane_a_skeleton3d,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run before/after pose temporal jitter audit on a skeleton3d.json.")
    parser.add_argument("--skeleton3d", type=Path, required=True, help="Input skeleton3d.json path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for audit artifacts.")
    parser.add_argument("--fps", type=float, help="Override FPS for temporal refinement.")
    parser.add_argument("--one-euro-mincutoff", type=float, default=DEFAULT_ONE_EURO_MINCUTOFF)
    parser.add_argument("--one-euro-beta", type=float, default=DEFAULT_ONE_EURO_BETA)
    parser.add_argument("--core-one-euro-mincutoff", type=float, default=DEFAULT_CORE_BODY_ONE_EURO_MINCUTOFF)
    parser.add_argument("--core-one-euro-beta", type=float, default=DEFAULT_CORE_BODY_ONE_EURO_BETA)
    parser.add_argument("--wrist-one-euro-mincutoff", type=float, default=DEFAULT_WRIST_ONE_EURO_MINCUTOFF)
    parser.add_argument("--wrist-one-euro-beta", type=float, default=DEFAULT_WRIST_ONE_EURO_BETA)
    parser.add_argument("--low-confidence-threshold", type=float, default=DEFAULT_LOW_CONFIDENCE_JOINT_THRESHOLD)
    parser.add_argument(
        "--apply-world-grounding",
        action="store_true",
        help="Re-run world grounding even if the input skeleton already has temporal_refine provenance.",
    )
    parser.add_argument("--target-core-p90-m", type=float, default=0.3)
    parser.add_argument("--max-wrist-peak-delta-frames", type=int, default=1)
    args = parser.parse_args(argv)

    skeleton = _read_json_object(args.skeleton3d)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    before = compute_pose_jitter_audit(skeleton, source_path=args.skeleton3d)
    apply_world_grounding = bool(args.apply_world_grounding or not _already_temporal_refined(skeleton))
    smoothed = refine_lane_a_skeleton3d(
        skeleton,
        fps=args.fps,
        one_euro_mincutoff=args.one_euro_mincutoff,
        one_euro_beta=args.one_euro_beta,
        core_one_euro_mincutoff=args.core_one_euro_mincutoff,
        core_one_euro_beta=args.core_one_euro_beta,
        wrist_one_euro_mincutoff=args.wrist_one_euro_mincutoff,
        wrist_one_euro_beta=args.wrist_one_euro_beta,
        low_confidence_threshold=args.low_confidence_threshold,
        apply_world_grounding=apply_world_grounding,
    )
    after = compute_pose_jitter_audit(smoothed, source_path=args.skeleton3d)
    wrist_timing = compare_wrist_peak_timing(
        skeleton,
        smoothed,
        max_allowed_delta_frames=args.max_wrist_peak_delta_frames,
    )

    core_p90 = after["group_stats"]["core_body"]["p90_frame_displacement_m"]
    core_pass = core_p90 is not None and float(core_p90) < args.target_core_p90_m
    wrist_pass = wrist_timing["status"] == "pass"
    status = "pass" if core_pass and wrist_pass else "fail"
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_pose_temporal_jitter_summary",
        "status": status,
        "source_path": str(args.skeleton3d),
        "target_core_p90_m": args.target_core_p90_m,
        "before": before,
        "after": after,
        "wrist_peak_timing": wrist_timing,
        "smoothing_flags": smoothed.get("provenance", {}).get("temporal_refine", {}).get("smoothing_flags", {}),
        "parameters": {
            "one_euro_mincutoff": args.one_euro_mincutoff,
            "one_euro_beta": args.one_euro_beta,
            "core_one_euro_mincutoff": args.core_one_euro_mincutoff,
            "core_one_euro_beta": args.core_one_euro_beta,
            "wrist_one_euro_mincutoff": args.wrist_one_euro_mincutoff,
            "wrist_one_euro_beta": args.wrist_one_euro_beta,
            "low_confidence_threshold": args.low_confidence_threshold,
            "max_wrist_peak_delta_frames": args.max_wrist_peak_delta_frames,
            "apply_world_grounding": apply_world_grounding,
        },
        "notes": [
            "Internal-val audit only; this does not promote BODY, BALL, TRK, CAL, or E2E gates.",
            "smoothing_flag is per-joint and index-aligned with joint_names/joints_world.",
        ],
    }

    _write_json(args.out_dir / "pose_jitter_before.json", before)
    _write_json(args.out_dir / "skeleton3d_pose_smooth.json", smoothed)
    _write_json(args.out_dir / "pose_jitter_after.json", after)
    _write_json(args.out_dir / "wrist_peak_timing.json", wrist_timing)
    _write_json(args.out_dir / "pose_temporal_jitter_summary.json", summary)
    _write_report(args.out_dir / "REPORT.md", summary)
    print(
        "wrote {summary_path} (status={status}, core_p90_before={before_p90}, core_p90_after={after_p90}, "
        "wrist_peak_delta={wrist_delta})".format(
            summary_path=args.out_dir / "pose_temporal_jitter_summary.json",
            status=status,
            before_p90=before["group_stats"]["core_body"]["p90_frame_displacement_m"],
            after_p90=after["group_stats"]["core_body"]["p90_frame_displacement_m"],
            wrist_delta=wrist_timing["max_abs_delta_frames"],
        )
    )
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _already_temporal_refined(payload: Mapping[str, Any]) -> bool:
    provenance = payload.get("provenance")
    return isinstance(provenance, Mapping) and isinstance(provenance.get("temporal_refine"), Mapping)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    before_core = summary["before"]["group_stats"]["core_body"]
    after_core = summary["after"]["group_stats"]["core_body"]
    wrist = summary["wrist_peak_timing"]
    lines = [
        "# POSE-SMOOTH Jitter Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Internal-val only. No BODY/BALL/TRK/CAL/E2E promotion claim.",
        "",
        "## Core Body Jitter",
        "",
        f"- Before p50/p90: {before_core['p50_frame_displacement_m']} / {before_core['p90_frame_displacement_m']} m",
        f"- After p50/p90: {after_core['p50_frame_displacement_m']} / {after_core['p90_frame_displacement_m']} m",
        f"- Target after p90: < {summary['target_core_p90_m']} m",
        "",
        "## Wrist Peak Timing",
        "",
        f"- Status: `{wrist['status']}`",
        f"- Max absolute frame delta: {wrist['max_abs_delta_frames']}",
        f"- Comparison count: {wrist['comparison_count']}",
        "",
        "## Flag Counts",
        "",
    ]
    temporal_flags = summary.get("smoothing_flags", {})
    for flag, count in sorted(temporal_flags.items()):
        lines.append(f"- {flag}: {count}")
    if not temporal_flags:
        lines.append("- none: 0")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
