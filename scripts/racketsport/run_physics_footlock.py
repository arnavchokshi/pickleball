#!/usr/bin/env python3
"""Measure and correct foot slide on world-frame BODY skeleton artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.foot_contact import (
    ContactMetrics,
    ContactPhase,
    ContactThresholds,
    SkeletonFrame,
    detect_contact_phases,
    measure_contact_metrics,
)
from threed.racketsport.foot_lock_solver import FootLockResult, FootLockSolverSettings, solve_foot_lock


SLIDE_GATE_MM = 3.0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _read_json(input_path)
    frames, joint_names, input_kind = _load_frames(payload)
    thresholds = ContactThresholds(
        enter_height_m=args.enter_height_m,
        exit_height_m=args.exit_height_m,
        enter_speed_mps=args.enter_speed_mps,
        exit_speed_mps=args.exit_speed_mps,
        min_confidence=args.min_confidence,
        min_phase_frames=args.min_phase_frames,
        low_foot_band_m=args.low_foot_band_m,
    )
    settings = FootLockSolverSettings(
        root_translation_weight=args.root_translation_weight,
        knee_residual_weight=args.knee_residual_weight,
        hip_residual_weight=args.hip_residual_weight,
    )

    phases = detect_contact_phases(frames, joint_names=joint_names, thresholds=thresholds)
    baseline_metrics = measure_contact_metrics(frames, phases, joint_names=joint_names)
    solve_result = solve_foot_lock(frames, phases, joint_names=joint_names, settings=settings)
    solved_metrics = measure_contact_metrics(solve_result.frames, phases, joint_names=joint_names)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    phase_payload = _phase_payload(
        clip=args.clip_name,
        source_artifact=input_path,
        input_kind=input_kind,
        generated_at=generated_at,
        thresholds=thresholds,
        phases=phases,
    )
    report_payload = _report_payload(
        clip=args.clip_name,
        source_artifact=input_path,
        input_kind=input_kind,
        generated_at=generated_at,
        thresholds=thresholds,
        settings=settings,
        baseline_metrics=baseline_metrics,
        solved_metrics=solved_metrics,
        solve_result=solve_result,
    )
    corrected_payload = _corrected_payload(
        clip=args.clip_name,
        source_artifact=input_path,
        input_kind=input_kind,
        generated_at=generated_at,
        joint_names=joint_names,
        phases=phases,
        baseline_metrics=baseline_metrics,
        solved_metrics=solved_metrics,
        solve_result=solve_result,
    )

    _write_json(out_dir / "foot_contact_phases.json", phase_payload)
    _write_json(out_dir / "foot_slide_report.json", report_payload)
    _write_json(out_dir / "physics_footlock.json", corrected_payload)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to body_world_label_packet.json or skeleton3d.json")
    parser.add_argument("--clip-name", required=True, help="Clip/run name to record in output artifacts")
    parser.add_argument("--out-dir", required=True, help="Output directory for PHYS-FOOT artifacts")
    parser.add_argument("--enter-height-m", type=float, default=0.060)
    parser.add_argument("--exit-height-m", type=float, default=0.100)
    parser.add_argument("--enter-speed-mps", type=float, default=0.75)
    parser.add_argument("--exit-speed-mps", type=float, default=1.25)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-phase-frames", type=int, default=2)
    parser.add_argument("--low-foot-band-m", type=float, default=0.025)
    parser.add_argument("--root-translation-weight", type=float, default=0.50)
    parser.add_argument("--knee-residual-weight", type=float, default=0.35)
    parser.add_argument("--hip-residual-weight", type=float, default=0.15)
    return parser.parse_args(argv)


def _load_frames(payload: dict[str, Any]) -> tuple[list[SkeletonFrame], tuple[str, ...], str]:
    if "samples" in payload:
        return _load_body_world_label_packet(payload)
    if "players" in payload:
        return _load_skeleton3d(payload)
    raise ValueError("input must contain either samples or players")


def _load_body_world_label_packet(payload: dict[str, Any]) -> tuple[list[SkeletonFrame], tuple[str, ...], str]:
    joint_names = tuple(str(name) for name in payload.get("joint_names", ()))
    frames: list[SkeletonFrame] = []
    for sample in payload.get("samples", []):
        joints = sample.get("predicted_joints_world")
        if not isinstance(joints, list):
            continue
        frames.append(
            SkeletonFrame(
                player_id=sample.get("player_id", "unknown"),
                frame_index=int(sample["frame_index"]),
                t=float(sample["t"]) if sample.get("t") is not None else None,
                joints_world=_copy_joints(joints),
                joint_conf=_copy_conf(sample.get("joint_conf")),
                source=sample,
            )
        )
    return frames, joint_names, "body_world_label_packet"


def _load_skeleton3d(payload: dict[str, Any]) -> tuple[list[SkeletonFrame], tuple[str, ...], str]:
    joint_names = tuple(str(name) for name in payload.get("joint_names", ()))
    frames: list[SkeletonFrame] = []
    for player in payload.get("players", []):
        player_id = player.get("id", player.get("player_id", "unknown"))
        for frame in player.get("frames", []):
            joints = frame.get("joints_world")
            if not isinstance(joints, list):
                continue
            frame_index = frame.get("frame_index", frame.get("frame_idx"))
            frames.append(
                SkeletonFrame(
                    player_id=player_id,
                    frame_index=int(frame_index),
                    t=float(frame["t"]) if frame.get("t") is not None else None,
                    joints_world=_copy_joints(joints),
                    joint_conf=_copy_conf(frame.get("joint_conf")),
                    source=frame,
                )
            )
    return frames, joint_names, "skeleton3d"


def _phase_payload(
    *,
    clip: str,
    source_artifact: Path,
    input_kind: str,
    generated_at: str,
    thresholds: ContactThresholds,
    phases: Sequence[ContactPhase],
) -> dict[str, Any]:
    return {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "clip": clip,
        "source_artifact": str(source_artifact),
        "source_kind": input_kind,
        "generated_at": generated_at,
        "thresholds": thresholds.to_dict(),
        "threshold_rationale": (
            "Z=0 is the court floor; enter/exit height tolerate centimeter-scale monocular foot jitter, "
            "and speed hysteresis keeps slow stance drift measurable while rejecting fast swing feet."
        ),
        "phase_count": len(phases),
        "phases": [phase.to_dict() for phase in phases],
    }


def _report_payload(
    *,
    clip: str,
    source_artifact: Path,
    input_kind: str,
    generated_at: str,
    thresholds: ContactThresholds,
    settings: FootLockSolverSettings,
    baseline_metrics: ContactMetrics,
    solved_metrics: ContactMetrics,
    solve_result: FootLockResult,
) -> dict[str, Any]:
    baseline_gate = _gate_summary(baseline_metrics)
    solved_gate = _gate_summary(solved_metrics)
    return {
        "artifact_type": "foot_slide_measurement_report",
        "schema_version": 1,
        "clip": clip,
        "source_artifact": str(source_artifact),
        "source_kind": input_kind,
        "generated_at": generated_at,
        "gate": {
            "slide_threshold_mm": SLIDE_GATE_MM,
            "requires_zero_foot_penetration": True,
            "baseline": baseline_gate,
            "solved": solved_gate,
        },
        "thresholds": thresholds.to_dict(),
        "solver": {
            "settings": settings.to_dict(),
            "method": "root_translation_plus_weighted_leg_chain_residual",
            "artifact_risk": (
                "This is deterministic foot locking, not full IK or dynamics; large corrections can be cosmetic "
                "and may show as knee/hip pops, so non-foot joint displacement is reported explicitly."
            ),
            "max_any_joint_displacement_m": solve_result.max_any_joint_displacement_m,
            "max_non_foot_joint_displacement_m": solve_result.max_non_foot_joint_displacement_m,
        },
        "baseline_metrics": baseline_metrics.to_dict(),
        "solved_metrics": solved_metrics.to_dict(),
    }


def _corrected_payload(
    *,
    clip: str,
    source_artifact: Path,
    input_kind: str,
    generated_at: str,
    joint_names: Sequence[str],
    phases: Sequence[ContactPhase],
    baseline_metrics: ContactMetrics,
    solved_metrics: ContactMetrics,
    solve_result: FootLockResult,
) -> dict[str, Any]:
    solved_gate = _gate_summary(solved_metrics)
    corrections_by_frame = {
        (correction.player_id, correction.frame_index): correction.to_dict()
        for correction in solve_result.frame_corrections
    }
    return {
        "artifact_type": "physics_footlock",
        "schema_version": 1,
        "clip": clip,
        "source_artifact": str(source_artifact),
        "source_kind": input_kind,
        "generated_at": generated_at,
        "joint_names": list(joint_names),
        "trust_band": {
            "stage": "FOOT/PHYS",
            "gate_id": "foot_slide_floor_penetration_gate",
            "gate_status": solved_gate["status"],
            "evidence_path": "foot_slide_report.json",
            "reason": solved_gate["reason"],
        },
        "not_ground_truth": True,
        "phase_count": len(phases),
        "baseline_gate": _gate_summary(baseline_metrics),
        "solved_gate": solved_gate,
        "solver": solve_result.to_dict(),
        "players": _players_payload(solve_result.frames, corrections_by_frame),
    }


def _players_payload(
    frames: Sequence[SkeletonFrame],
    corrections_by_frame: dict[tuple[str | int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str | int, list[SkeletonFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.player_id].append(frame)
    players: list[dict[str, Any]] = []
    for player_id, player_frames in grouped.items():
        sorted_frames = sorted(player_frames, key=lambda item: item.frame_index)
        players.append(
            {
                "id": player_id,
                "frames": [
                    {
                        "frame_index": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                        "foot_lock": corrections_by_frame.get((frame.player_id, frame.frame_index)),
                    }
                    for frame in sorted_frames
                ],
            }
        )
    return players


def _gate_summary(metrics: ContactMetrics) -> dict[str, Any]:
    phase_count = sum(summary.phase_count for summary in metrics.summary_by_player.values())
    max_slide_mm = max((summary.max_slide_mm for summary in metrics.summary_by_player.values()), default=0.0)
    max_penetration_mm = metrics.penetration.max_penetration_mm
    if phase_count == 0:
        status = "not_measured"
        reason = "no ground-contact phases detected"
    elif max_slide_mm <= SLIDE_GATE_MM and max_penetration_mm == 0:
        status = "pass"
        reason = f"max slide {max_slide_mm:.3f}mm <= {SLIDE_GATE_MM:.3f}mm and foot penetration is zero"
    else:
        status = "fail"
        reason = (
            f"max slide {max_slide_mm:.3f}mm against {SLIDE_GATE_MM:.3f}mm gate; "
            f"max foot penetration {max_penetration_mm:.3f}mm"
        )
    return {
        "status": status,
        "phase_count": phase_count,
        "max_slide_mm": max_slide_mm,
        "max_penetration_mm": max_penetration_mm,
        "reason": reason,
    }


def _copy_joints(joints: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1]), float(point[2])] for point in joints]


def _copy_conf(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
