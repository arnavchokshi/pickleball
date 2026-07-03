#!/usr/bin/env python3
"""Score real BODY inference output against ASPset-510 external ground truth.

Reads, per pre-registered clip (`runs/manager/heldout_eval_ledger.md` row BODY-EXT-1):
- real predictions: `<inference_root>/<clip>/smpl_motion.json` (Fast-SAM-3D-Body's raw
  70-joint MHR output, world-grounded via the real calibration built by
  `build_external_gt_aspset510_body_inputs.py`; joint order per
  `threed.racketsport.external_gt_body_prediction_schema.MHR70_JOINT_NAMES` -- see that
  module's docstring for why name-keyed, not positional, selection is required here)
- real ground truth: `<run_dir>/labels/<clip>/body_world_joints.json` (12 shared-core
  joints, real ASPset-510 `.c3d` mocap, meters)

Matches by exact `frame_index`, selects the shared 12-joint subset from the prediction by
NAME (F1 fix discipline), scores all 4 MPJPE variants
(`threed.racketsport.external_gt_alignment.score_external_gt_clip`) per clip plus a
frame-count-weighted pooled aggregate, and reports the pre-registered gate variant
(`clip_level_rigid_aligned_mpjpe`) against the existing 0.05m
`threed.racketsport.eval.body_gate_report.DEFAULT_WORLD_MPJPE_THRESHOLD_M` threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval.body_gate_report import DEFAULT_WORLD_MPJPE_THRESHOLD_M  # noqa: E402
from threed.racketsport.external_gt_alignment import per_joint_breakdown, score_external_gt_clip  # noqa: E402
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES  # noqa: E402
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES  # noqa: E402

DEFAULT_RUN_DIR = ROOT / "runs" / "body_external_gt_20260702T040048Z"
ROOT_JOINT_NAMES = ("left_hip", "right_hip")
GATE_VARIANT = "clip_level_rigid_aligned_mpjpe"
VARIANT_NAMES = ("mpjpe", "root_relative_mpjpe", "pa_mpjpe", "clip_level_rigid_aligned_mpjpe")

CLIPS: tuple[str, ...] = (
    "aspset510_1e28_0001_left",
    "aspset510_1e28_0006_right",
    "aspset510_1e28_0008_left",
    "aspset510_1e28_0011_left",
    "aspset510_1e28_0014_mid",
    "aspset510_1e28_0018_mid",
    "aspset510_8a59_0006_mid",
    "aspset510_8a59_0008_mid",
    "aspset510_8a59_0011_right",
    "aspset510_8a59_0015_left",
    "aspset510_8a59_0016_mid",
    "aspset510_8a59_0018_left",
)

_MHR70_INDEX = {name: index for index, name in enumerate(MHR70_JOINT_NAMES)}


class ScoringError(RuntimeError):
    pass


def _predicted_frame_index(smpl_motion: dict[str, Any]) -> dict[int, list[list[float]]]:
    """frame_idx -> raw 70-joint joints_world for player id 1."""

    players = smpl_motion.get("players", [])
    for player in players:
        if int(player.get("id", -1)) != 1:
            continue
        out: dict[int, list[list[float]]] = {}
        for frame in player.get("frames", []):
            joints = frame.get("joints_world")
            if not joints or len(joints) != len(MHR70_JOINT_NAMES):
                raise ScoringError(
                    f"expected {len(MHR70_JOINT_NAMES)} raw MHR joints, got "
                    f"{len(joints) if joints else 0} at frame_idx={frame.get('frame_idx')}"
                )
            out[int(frame["frame_idx"])] = joints
        return out
    raise ScoringError("smpl_motion.json has no player id=1")


def _select_shared_core_from_mhr70(raw_joints_70: list[list[float]]) -> list[list[float]]:
    return [raw_joints_70[_MHR70_INDEX[name]] for name in SHARED_CORE_JOINT_NAMES]


def score_one_clip(*, clip: str, run_dir: Path, inference_root: Path) -> dict[str, Any]:
    label_path = run_dir / "labels" / clip / "body_world_joints.json"
    smpl_path = inference_root / clip / "smpl_motion.json"
    if not label_path.is_file():
        raise ScoringError(f"missing GT label file: {label_path}")
    if not smpl_path.is_file():
        raise ScoringError(f"missing real prediction file: {smpl_path}")

    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    if label_payload.get("joint_names") != list(SHARED_CORE_JOINT_NAMES):
        raise ScoringError(f"{clip}: GT joint_names do not match SHARED_CORE_JOINT_NAMES order")

    smpl_payload = json.loads(smpl_path.read_text(encoding="utf-8"))
    predicted_by_frame = _predicted_frame_index(smpl_payload)

    matched_frame_indices: list[int] = []
    predicted_rows: list[list[list[float]]] = []
    gt_rows: list[list[list[float]]] = []
    unmatched: list[int] = []
    for sample in label_payload["samples"]:
        if sample.get("accepted") is False:
            continue
        frame_index = int(sample["frame_index"])
        raw = predicted_by_frame.get(frame_index)
        if raw is None:
            unmatched.append(frame_index)
            continue
        predicted_rows.append(_select_shared_core_from_mhr70(raw))
        gt_rows.append(sample["joints_world"])
        matched_frame_indices.append(frame_index)

    if not predicted_rows:
        raise ScoringError(f"{clip}: no matching predicted frames for any GT sample")

    predicted = np.asarray(predicted_rows, dtype=np.float64)
    gt = np.asarray(gt_rows, dtype=np.float64)

    scored = score_external_gt_clip(
        predicted_joints=predicted,
        gt_joints=gt,
        joint_names=list(SHARED_CORE_JOINT_NAMES),
        root_joint_names=ROOT_JOINT_NAMES,
        clip_id=clip,
        subject_id=label_payload.get("provenance", {}).get("subject_id", ""),
        gate_variant=GATE_VARIANT,
    )
    per_joint = per_joint_breakdown(
        predicted_joints=predicted,
        gt_joints=gt,
        joint_names=list(SHARED_CORE_JOINT_NAMES),
        root_joint_names=ROOT_JOINT_NAMES,
    )
    gate_value = scored["gate_value_m"]
    return {
        "clip": clip,
        "gt_sample_count": len(label_payload["samples"]),
        "matched_frame_count": len(matched_frame_indices),
        "unmatched_gt_frame_indices": unmatched,
        "matched_frame_indices": matched_frame_indices,
        "scored": scored,
        "per_joint_breakdown_m": per_joint,
        "gate_value_m": gate_value,
        "gate_threshold_m": DEFAULT_WORLD_MPJPE_THRESHOLD_M,
        "gate_passed": gate_value <= DEFAULT_WORLD_MPJPE_THRESHOLD_M,
    }


def pooled_variants(clip_results: list[dict[str, Any]]) -> dict[str, float]:
    """Frame-count-weighted average of each variant's per-clip mean.

    Equivalent to concatenating every matched (frame, joint) error across all clips and
    taking the grand mean, since every clip contributes the same 12-joint width and each
    clip's own alignment transform is fit independently (per the pre-registered
    methodology: one Umeyama fit *per clip*, not one fit across the whole pooled dataset).
    """

    pooled: dict[str, float] = {}
    for variant in VARIANT_NAMES:
        total_weight = sum(result["matched_frame_count"] for result in clip_results)
        if total_weight == 0:
            pooled[variant] = float("nan")
            continue
        weighted_sum = sum(
            result["scored"]["variants"][variant]["value_m"] * result["matched_frame_count"]
            for result in clip_results
        )
        pooled[variant] = weighted_sum / total_weight
    return pooled


def pooled_per_joint(clip_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    pooled: dict[str, dict[str, float]] = {}
    for joint_name in SHARED_CORE_JOINT_NAMES:
        pooled[joint_name] = {}
        for variant in VARIANT_NAMES:
            total_weight = sum(result["matched_frame_count"] for result in clip_results)
            if total_weight == 0:
                pooled[joint_name][variant] = float("nan")
                continue
            weighted_sum = sum(
                result["per_joint_breakdown_m"][joint_name][variant] * result["matched_frame_count"]
                for result in clip_results
            )
            pooled[joint_name][variant] = weighted_sum / total_weight
    return pooled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=None)
    args = parser.parse_args(argv)

    clips = args.clips or list(CLIPS)
    clip_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for clip in clips:
        try:
            clip_results.append(score_one_clip(clip=clip, run_dir=args.run_dir, inference_root=args.inference_root))
        except ScoringError as exc:
            errors.append({"clip": clip, "error": str(exc)})
            print(f"ERROR scoring {clip}: {exc}", file=sys.stderr)

    total_gt_samples = sum(r["gt_sample_count"] for r in clip_results)
    total_matched = sum(r["matched_frame_count"] for r in clip_results)
    pooled = pooled_variants(clip_results) if clip_results else {}
    pooled_joints = pooled_per_joint(clip_results) if clip_results else {}
    pooled_gate_value = pooled.get(GATE_VARIANT)
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_external_gt_body_scoring_report",
        "run_dir": str(args.run_dir),
        "inference_root": str(args.inference_root),
        "gate_variant": GATE_VARIANT,
        "gate_threshold_m": DEFAULT_WORLD_MPJPE_THRESHOLD_M,
        "root_joint_names": list(ROOT_JOINT_NAMES),
        "joint_names": list(SHARED_CORE_JOINT_NAMES),
        "total_gt_sample_count": total_gt_samples,
        "total_matched_frame_count": total_matched,
        "clip_count": len(clip_results),
        "clip_errors": errors,
        "pooled": {
            "variants_m": pooled,
            "per_joint_m": pooled_joints,
            "gate_value_m": pooled_gate_value,
            "gate_passed": (pooled_gate_value is not None and pooled_gate_value <= DEFAULT_WORLD_MPJPE_THRESHOLD_M),
        },
        "clips": clip_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
