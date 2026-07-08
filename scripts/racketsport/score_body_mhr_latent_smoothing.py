#!/usr/bin/env python3
"""Measure P2-2 MHR latent-smoothing acceptance keys from before/after runs.

This is a measurement harness only. It does not import or wire
``process_video.py``. The intended production input is a set of synthetic run
directories whose ``virtual_world.json`` differs only in ``joints_world``:
raw vs smoothed-decoded joints. A proxy mode exists to score the same metric
keys from direct world-joint smoothing when decoded candidates are absent; that
mode is explicitly marked non-acceptance in the output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.smooth_body_mhr_latent import sliding_window_smooth  # noqa: E402
from threed.racketsport import worldhmr  # noqa: E402
from threed.racketsport.body_grounding_quality import (  # noqa: E402
    DEFAULT_MAX_FOOT_SLIDE_M,
    build_body_grounding_quality,
)
from threed.racketsport.foot_contact import build_body_skeleton_foot_contact_phases  # noqa: E402
from threed.racketsport.pose_temporal import compare_wrist_peak_timing  # noqa: E402
from threed.racketsport.visual_quality import measure_visual_quality  # noqa: E402


DEFAULT_LAMBDAS = (0.1, 0.3, 0.6)
REPORT_NAME = "latent_smoothing_acceptance_report.json"
MARKDOWN_NAME = "latent_smoothing_acceptance_report.md"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score P2-2 latent-smoothing acceptance keys from raw and candidate "
            "synthetic run directories. Does not wire the smoother into process_video.py."
        )
    )
    parser.add_argument("--raw-run-dir", type=Path, required=True, help="Raw run directory or BODY dispatch dir.")
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        metavar="LAMBDA=DIR",
        help="Decoded candidate run dir for one lambda, e.g. 0.3=runs/lambda_0_3. Repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Lane output directory.")
    parser.add_argument("--wrist-top-k", type=int, default=5)
    parser.add_argument("--wrist-min-peak-speed-mps", type=float, default=4.0)
    parser.add_argument("--window", type=int, default=9, help="Proxy smoothing window.")
    parser.add_argument(
        "--lambda-smooth",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated lambdas for --proxy-world-joint mode.",
    )
    parser.add_argument(
        "--proxy-world-joint",
        action="store_true",
        help=(
            "Generate candidate dirs by smoothing virtual_world joints directly. "
            "This scores real metric keys but is NOT latent-decoded acceptance evidence."
        ),
    )
    parser.add_argument(
        "--mesh-divergence-report",
        type=Path,
        default=None,
        help="Optional mhr_decode gate report containing mesh_skeleton_divergence to attach as report-only context.",
    )
    return parser


def parse_candidate_run_specs(specs: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--candidate-run must be LAMBDA=DIR, got: {spec}")
        key, value = spec.split("=", 1)
        key = _lambda_key(float(key.strip()))
        path = Path(value.strip())
        if not path.exists():
            raise SystemExit(f"candidate run dir does not exist for lambda {key}: {path}")
        out[key] = path
    return dict(sorted(out.items(), key=lambda item: float(item[0])))


def evaluate_candidate_runs(
    *,
    raw_run_dir: Path,
    candidate_run_dirs: Mapping[str, Path],
    out_dir: Path,
    wrist_top_k: int = 5,
    wrist_min_peak_speed_mps: float = 4.0,
    measurement_mode: str = "decoded_candidate_run_dirs",
    mesh_divergence_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_run_dir = Path(raw_run_dir)
    candidate_run_dirs = {str(key): Path(value) for key, value in candidate_run_dirs.items()}

    raw_visual = measure_visual_quality(raw_run_dir)
    raw_world = _read_world(raw_run_dir)
    raw_gate = _predict_foot_slide_gate(raw_world)
    raw_phase = _phase_census(raw_world)
    candidate_visual = {key: measure_visual_quality(path) for key, path in candidate_run_dirs.items()}
    candidate_world = {key: _read_world(path) for key, path in candidate_run_dirs.items()}
    candidate_gate = {key: _predict_foot_slide_gate(world) for key, world in candidate_world.items()}
    candidate_phase = {key: _phase_census(world) for key, world in candidate_world.items()}
    wrist_timing = {
        key: compare_wrist_peak_timing(
            raw_world,
            world,
            top_k=wrist_top_k,
            max_allowed_delta_frames=0,
            min_peak_speed_mps=wrist_min_peak_speed_mps,
        )
        for key, world in candidate_world.items()
    }
    mesh_context = _mesh_divergence_context(mesh_divergence_report)

    player_ids = sorted(
        set(raw_visual.get("players", {}))
        | {player_id for visual in candidate_visual.values() for player_id in visual.get("players", {})}
    )
    players: dict[str, Any] = {}
    for player_id in player_ids:
        players[player_id] = _player_acceptance_row(
            player_id=player_id,
            raw_visual=raw_visual,
            candidate_visual=candidate_visual,
            raw_gate=raw_gate,
            candidate_gate=candidate_gate,
            raw_phase=raw_phase,
            candidate_phase=candidate_phase,
            wrist_timing=wrist_timing,
            mesh_context=mesh_context,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mhr_latent_smoothing_acceptance",
        "measurement_status": "measured",
        "measurement_mode": measurement_mode,
        "acceptance_boundary": _acceptance_boundary(measurement_mode),
        "raw_run_dir": str(raw_run_dir),
        "candidate_run_dirs": {key: str(path) for key, path in candidate_run_dirs.items()},
        "targets": {
            "world_jitter_effective_target_mm_per_frame2": 10.0,
            "wrist_peak_delta_frames": 0,
            "foot_slide_gate_threshold_m": DEFAULT_MAX_FOOT_SLIDE_M,
            "mesh_skeleton_divergence_report_only_target_p95_mm": 5.0,
        },
        "players": players,
        "wrist_peak_timing": wrist_timing,
        "mesh_skeleton_divergence_context": mesh_context,
        "recommendation": _recommendation(players, candidate_run_dirs.keys(), measurement_mode=measurement_mode),
    }
    _write_json(out_dir / REPORT_NAME, report)
    (out_dir / MARKDOWN_NAME).write_text(_markdown_report(report), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_run_dirs = parse_candidate_run_specs(args.candidate_run)
    mesh_report = _read_json(args.mesh_divergence_report) if args.mesh_divergence_report is not None else None
    if candidate_run_dirs:
        return evaluate_candidate_runs(
            raw_run_dir=args.raw_run_dir,
            candidate_run_dirs=candidate_run_dirs,
            out_dir=args.out_dir,
            wrist_top_k=args.wrist_top_k,
            wrist_min_peak_speed_mps=args.wrist_min_peak_speed_mps,
            mesh_divergence_report=mesh_report,
        )
    if args.proxy_world_joint:
        return evaluate_world_joint_proxy(
            source_run_dir=args.raw_run_dir,
            out_dir=args.out_dir,
            lambda_values=_parse_lambdas(args.lambda_smooth),
            window=args.window,
            wrist_top_k=args.wrist_top_k,
            wrist_min_peak_speed_mps=args.wrist_min_peak_speed_mps,
            mesh_divergence_report=mesh_report,
        )
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mhr_latent_smoothing_acceptance",
        "measurement_status": "blocked",
        "measurement_mode": "none",
        "raw_run_dir": str(args.raw_run_dir),
        "candidate_run_dirs": {},
        "blockers": ["missing_candidate_run_dirs"],
        "notes": [
            "Provide --candidate-run LAMBDA=DIR synthetic decoded run dirs, or use "
            "--proxy-world-joint for a non-acceptance world-joint proxy diagnostic."
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / REPORT_NAME, report)
    return report


def evaluate_world_joint_proxy(
    *,
    source_run_dir: Path,
    out_dir: Path,
    lambda_values: Sequence[float],
    window: int,
    wrist_top_k: int = 5,
    wrist_min_peak_speed_mps: float = 4.0,
    mesh_divergence_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    synthetic_root = out_dir / "synthetic_world_joint_proxy"
    raw_world = _world_from_run_or_skeleton(source_run_dir)
    raw_run = synthetic_root / "raw"
    _materialize_run(source_run_dir, raw_run, raw_world)
    candidate_dirs: dict[str, Path] = {}
    for value in lambda_values:
        key = _lambda_key(value)
        candidate_world = smooth_virtual_world_joints(raw_world, window=window, lambda_smooth=value)
        candidate_dir = synthetic_root / f"lambda_{key.replace('.', '_')}"
        _materialize_run(source_run_dir, candidate_dir, candidate_world)
        candidate_dirs[key] = candidate_dir
    report = evaluate_candidate_runs(
        raw_run_dir=raw_run,
        candidate_run_dirs=candidate_dirs,
        out_dir=out_dir,
        wrist_top_k=wrist_top_k,
        wrist_min_peak_speed_mps=wrist_min_peak_speed_mps,
        measurement_mode="world_joint_proxy_not_latent_decode",
        mesh_divergence_report=mesh_divergence_report,
    )
    report["proxy_inputs"] = {
        "source_run_dir": str(source_run_dir),
        "window": int(window),
        "lambda_smooth": [float(value) for value in lambda_values],
        "acceptance_warning": "This proxy changes joints_world directly; it is not smoothed-decoded MHR latent evidence.",
    }
    _write_json(out_dir / REPORT_NAME, report)
    (out_dir / MARKDOWN_NAME).write_text(_markdown_report(report), encoding="utf-8")
    return report


def smooth_virtual_world_joints(
    virtual_world: Mapping[str, Any],
    *,
    window: int,
    lambda_smooth: float,
) -> dict[str, Any]:
    smoothed = copy.deepcopy(dict(virtual_world))
    players = []
    for player in virtual_world.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        out_player = copy.deepcopy(dict(player))
        frames = [copy.deepcopy(dict(frame)) for frame in player.get("frames", []) or [] if isinstance(frame, Mapping)]
        if frames:
            joints = [_joints_array(frame) for frame in frames]
            if joints and all(item is not None and item.shape == joints[0].shape for item in joints):
                stack = np.stack([item for item in joints if item is not None], axis=0)
                flat = stack.reshape(stack.shape[0], -1)
                flat_smoothed = sliding_window_smooth(flat, window=window, lambda_smooth=lambda_smooth)
                smooth_stack = flat_smoothed.reshape(stack.shape)
                for frame, joints_world in zip(frames, smooth_stack, strict=False):
                    frame["joints_world"] = joints_world.tolist()
        out_player["frames"] = frames
        players.append(out_player)
    smoothed["players"] = players
    return smoothed


def _player_acceptance_row(
    *,
    player_id: str,
    raw_visual: Mapping[str, Any],
    candidate_visual: Mapping[str, Mapping[str, Any]],
    raw_gate: Mapping[str, Any],
    candidate_gate: Mapping[str, Mapping[str, Any]],
    raw_phase: Mapping[str, Any],
    candidate_phase: Mapping[str, Mapping[str, Any]],
    wrist_timing: Mapping[str, Mapping[str, Any]],
    mesh_context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "world_jitter_mm_per_frame2": {
            group: _variant_metric(
                raw_visual,
                candidate_visual,
                player_id=player_id,
                path=("world_jitter_mm_per_frame2", group),
            )
            for group in ("feet", "wrists", "root")
        },
        "foot_slide_mm_per_frame": {
            "stance": _variant_metric(
                raw_visual,
                candidate_visual,
                player_id=player_id,
                path=("foot_slide_mm_per_frame", "stance"),
            )
        },
        "wrist_peak_delta_frames": {
            "raw": {"status": "reference", "max_abs_delta_frames": 0},
            **{
                key: _wrist_player_summary(value, player_id=player_id)
                for key, value in wrist_timing.items()
            },
        },
        "mesh_skeleton_divergence_p95_mm": _mesh_player_context(mesh_context, player_id),
        "foot_slide_gate": {
            "raw": _gate_player_summary(raw_gate, player_id=player_id),
            **{
                key: _gate_player_summary(value, player_id=player_id)
                for key, value in candidate_gate.items()
            },
        },
        "phase_census": {
            "raw": _phase_player_summary(raw_phase, player_id=player_id),
            **{
                key: _phase_player_summary(value, player_id=player_id)
                for key, value in candidate_phase.items()
            },
        },
    }


def _variant_metric(
    raw_visual: Mapping[str, Any],
    candidate_visual: Mapping[str, Mapping[str, Any]],
    *,
    player_id: str,
    path: tuple[str, str],
) -> dict[str, Any]:
    out = {"raw": _nested(raw_visual.get("players", {}).get(player_id, {}), path)}
    for key, visual in candidate_visual.items():
        out[key] = _nested(visual.get("players", {}).get(player_id, {}), path)
    return out


def _wrist_player_summary(payload: Mapping[str, Any], *, player_id: str) -> dict[str, Any]:
    player_comparisons = [
        dict(row)
        for row in payload.get("comparisons", []) or []
        if isinstance(row, Mapping) and str(row.get("player_id")) == str(player_id)
    ]
    deltas = [
        int(row["abs_delta_frames"])
        for row in player_comparisons
        if row.get("abs_delta_frames") is not None
    ]
    max_delta = max(deltas) if deltas else None
    status = "pass" if max_delta is not None and max_delta <= int(payload.get("max_allowed_delta_frames", 0)) else "blocked"
    if max_delta is not None and max_delta > int(payload.get("max_allowed_delta_frames", 0)):
        status = "fail"
    return {
        "status": status,
        "max_abs_delta_frames": max_delta,
        "comparison_count": len(player_comparisons),
        "smart_failure_flag": bool(max_delta is not None and max_delta > 1),
    }


def _predict_foot_slide_gate(world_payload: Mapping[str, Any]) -> dict[str, Any]:
    clip = str(world_payload.get("clip") or "synthetic")
    try:
        metrics, gate_stream = worldhmr._contact_gate_stream_for_skeleton3d(  # type: ignore[attr-defined]
            world_payload,
            clip=clip,
            threshold_m=DEFAULT_MAX_FOOT_SLIDE_M,
        )
        max_slide_m = max(
            (
                float(row.get("slide_mm", 0.0)) / 1000.0
                for row in metrics.get("phase_metrics", []) or []
                if isinstance(row, Mapping)
            ),
            default=0.0,
        )
        quality = build_body_grounding_quality(
            clip=clip,
            grounding_metrics={**metrics, "max_foot_lock_slide_m": max_slide_m, "foot_lock_gate_stream": gate_stream},
        )
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        return {
            "status": "blocked",
            "threshold_m": DEFAULT_MAX_FOOT_SLIDE_M,
            "value_m": None,
            "predicted_pass": False,
            "blockers": [f"{type(exc).__name__}: {exc}"],
            "per_player": {},
        }
    per_player: dict[str, Any] = {}
    for row in metrics.get("phase_metrics", []) or []:
        if not isinstance(row, Mapping):
            continue
        player_id = str(row.get("player_id"))
        value_m = float(row.get("slide_mm", 0.0)) / 1000.0
        summary = per_player.setdefault(
            player_id,
            {
                "threshold_m": DEFAULT_MAX_FOOT_SLIDE_M,
                "value_m": 0.0,
                "predicted_pass": True,
                "phase_count": 0,
            },
        )
        summary["phase_count"] += 1
        summary["value_m"] = max(float(summary["value_m"]), value_m)
        summary["predicted_pass"] = bool(summary["value_m"] <= DEFAULT_MAX_FOOT_SLIDE_M)
    return {
        "status": quality.get("status"),
        "threshold_m": quality.get("foot_slide_gate", {}).get("threshold_m", DEFAULT_MAX_FOOT_SLIDE_M),
        "value_m": quality.get("foot_slide_gate", {}).get("value_m"),
        "predicted_pass": quality.get("foot_slide_gate", {}).get("passed") is True,
        "blockers": list(quality.get("blockers", []) or []),
        "per_player": per_player,
    }


def _gate_player_summary(payload: Mapping[str, Any], *, player_id: str) -> dict[str, Any]:
    player = payload.get("per_player", {}).get(str(player_id)) if isinstance(payload.get("per_player"), Mapping) else None
    if isinstance(player, Mapping):
        return {
            "threshold_m": float(player.get("threshold_m", DEFAULT_MAX_FOOT_SLIDE_M)),
            "value_m": float(player.get("value_m", 0.0)),
            "predicted_pass": bool(player.get("predicted_pass")),
            "phase_count": int(player.get("phase_count", 0) or 0),
        }
    return {
        "threshold_m": float(payload.get("threshold_m", DEFAULT_MAX_FOOT_SLIDE_M)),
        "value_m": _maybe_float(payload.get("value_m")),
        "predicted_pass": bool(payload.get("predicted_pass")),
        "phase_count": 0,
    }


def _phase_census(world_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_body_skeleton_foot_contact_phases(world_payload, clip=str(world_payload.get("clip") or "synthetic"))
    per_player: dict[str, Any] = {}
    for phase in payload.get("phases", []) or []:
        if not isinstance(phase, Mapping):
            continue
        row = per_player.setdefault(str(phase.get("player_id")), _empty_phase_player())
        row["confident_phase_count"] += 1
        row["phase_frame_count"] += int(phase.get("frame_count", 0) or 0)
        foot = str(phase.get("foot"))
        if foot in row["per_foot_confident_phase_count"]:
            row["per_foot_confident_phase_count"][foot] += 1
    for phase in payload.get("rejected_phases", []) or []:
        if not isinstance(phase, Mapping):
            continue
        row = per_player.setdefault(str(phase.get("player_id")), _empty_phase_player())
        row["rejected_phase_count"] += 1
        reason = str(phase.get("rejection_reason") or phase.get("reason") or "unknown")
        row["rejection_reason_counts"][reason] = int(row["rejection_reason_counts"].get(reason, 0)) + 1
        foot = str(phase.get("foot"))
        if foot in row["per_foot_rejected_phase_count"]:
            row["per_foot_rejected_phase_count"][foot] += 1
    return {
        "status": payload.get("status"),
        "summary": payload.get("summary", {}),
        "per_player": per_player,
    }


def _empty_phase_player() -> dict[str, Any]:
    return {
        "confident_phase_count": 0,
        "rejected_phase_count": 0,
        "phase_frame_count": 0,
        "rejection_reason_counts": {},
        "phase_penetrates_ground_rejections": 0,
        "per_foot_confident_phase_count": {"left": 0, "right": 0},
        "per_foot_rejected_phase_count": {"left": 0, "right": 0},
    }


def _phase_player_summary(payload: Mapping[str, Any], *, player_id: str) -> dict[str, Any]:
    row = copy.deepcopy(
        payload.get("per_player", {}).get(str(player_id), _empty_phase_player())
        if isinstance(payload.get("per_player"), Mapping)
        else _empty_phase_player()
    )
    row.setdefault("rejection_reason_counts", {})
    row["rejection_reason_counts"].setdefault("phase_penetrates_ground", 0)
    row["phase_penetrates_ground_rejections"] = int(row.get("rejection_reason_counts", {}).get("phase_penetrates_ground", 0))
    return row


def _read_world(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "virtual_world.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing virtual_world.json in {run_dir}")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"virtual_world.json did not parse as object: {path}")
    return payload


def _world_from_run_or_skeleton(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    world_path = run_dir / "virtual_world.json"
    if world_path.is_file():
        payload = _read_json(world_path)
    else:
        payload = _read_json(run_dir / "skeleton3d.json")
    if not isinstance(payload, dict):
        raise ValueError(f"run world/skeleton did not parse as object: {run_dir}")
    return _minimal_virtual_world(payload)


def _minimal_virtual_world(payload: Mapping[str, Any]) -> dict[str, Any]:
    players: list[dict[str, Any]] = []
    for player in payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        frames: list[dict[str, Any]] = []
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping) or "joints_world" not in frame:
                continue
            out_frame: dict[str, Any] = {
                "frame_idx": int(frame.get("frame_idx", len(frames))),
                "t": float(frame.get("t", len(frames) / float(payload.get("fps") or 30.0))),
                "joints_world": copy.deepcopy(frame.get("joints_world")),
            }
            if "joint_conf" in frame:
                out_frame["joint_conf"] = copy.deepcopy(frame.get("joint_conf"))
            if "smoothing_flag" in frame:
                out_frame["smoothing_flag"] = copy.deepcopy(frame.get("smoothing_flag"))
            frames.append(out_frame)
        players.append({"id": str(player.get("id")), "frames": frames})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "clip": str(payload.get("clip") or ""),
        "fps": float(payload.get("fps") or 30.0),
        "joint_names": [str(name) for name in payload.get("joint_names", [])],
        "players": players,
        "world_frame": payload.get("world_frame", "court"),
    }


def _materialize_run(source_run_dir: Path, target_run_dir: Path, virtual_world: Mapping[str, Any]) -> None:
    target_run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_run_dir / "virtual_world.json", dict(virtual_world))
    skeleton = copy.deepcopy(dict(virtual_world))
    skeleton["artifact_type"] = "racketsport_skeleton3d"
    _write_json(target_run_dir / "skeleton3d.json", skeleton)
    for name in ("placement.json", "body_joint_quality.json"):
        source = Path(source_run_dir) / name
        if source.is_file():
            shutil.copyfile(source, target_run_dir / name)
    if not (target_run_dir / "placement.json").is_file():
        _write_json(target_run_dir / "placement.json", _empty_placement(virtual_world))
    if not (target_run_dir / "body_joint_quality.json").is_file():
        _write_json(
            target_run_dir / "body_joint_quality.json",
            {"schema_version": 1, "artifact_type": "racketsport_body_joint_quality", "summary": {}},
        )


def _empty_placement(virtual_world: Mapping[str, Any]) -> dict[str, Any]:
    players = []
    for player in virtual_world.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        frames = [
            {
                "frame_idx": frame.get("frame_idx", idx),
                "t": frame.get("t", 0.0),
                "stance": False,
                "smoothed_world_xy": [0.0, 0.0],
            }
            for idx, frame in enumerate(player.get("frames", []) or [])
            if isinstance(frame, Mapping)
        ]
        players.append({"id": str(player.get("id")), "frames": frames})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": float(virtual_world.get("fps") or 30.0),
        "players": players,
        "summary": {},
    }


def _mesh_divergence_context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "status": "not_measured",
            "claim_boundary": "No mhr_decode mesh_skeleton_divergence report was supplied to this harness.",
            "per_player": {},
        }
    divergence = payload.get("mesh_skeleton_divergence")
    if not isinstance(divergence, Mapping):
        return {"status": "not_found", "per_player": {}}
    per_player = {}
    for player_id, row in (divergence.get("per_player") or {}).items():
        if isinstance(row, Mapping):
            per_player[str(player_id)] = {
                "p95_mm": _maybe_float(row.get("p95_mm_max_over_sample") or row.get("p95_mm")),
                "sample_count": int(row.get("sample_count", 0) or 0),
                "source": "supplied_mhr_decode_report",
            }
    return {
        "status": "report_only",
        "target_p95_mm": _maybe_float(divergence.get("target_p95_mm")),
        "worst_p95_mm_over_sample": _maybe_float(divergence.get("worst_p95_mm_over_sample")),
        "passed": divergence.get("passed"),
        "claim_boundary": (
            "This is attached from an external mhr_decode report. It is not recomputed per lambda "
            "unless decoded candidate runs include their own divergence report."
        ),
        "per_player": per_player,
    }


def _mesh_player_context(mesh_context: Mapping[str, Any], player_id: str) -> dict[str, Any]:
    player = mesh_context.get("per_player", {}).get(str(player_id)) if isinstance(mesh_context.get("per_player"), Mapping) else None
    if isinstance(player, Mapping):
        return dict(player)
    return {"status": mesh_context.get("status", "not_measured"), "p95_mm": None}


def _recommendation(players: Mapping[str, Any], lambda_keys: Sequence[str], *, measurement_mode: str) -> str:
    if measurement_mode != "decoded_candidate_run_dirs":
        return (
            "No lambda is wiring-ready from this run: the table is a world-joint proxy, not decoded MHR latent "
            "evidence. Next evidence needed: decoded synthetic run dirs for lambdas 0.1/0.3/0.6, 4-clip coverage, "
            "and strict GATE 1b/mesh-divergence clarification before any un-kill decision."
        )
    viable: list[str] = []
    for key in lambda_keys:
        jitter_ok = True
        wrist_ok = True
        gate_ok = True
        for player in players.values():
            feet = player["world_jitter_mm_per_frame2"]["feet"].get(str(key), {})
            wrists = player["world_jitter_mm_per_frame2"]["wrists"].get(str(key), {})
            root = player["world_jitter_mm_per_frame2"]["root"].get(str(key), {})
            jitter_ok = jitter_ok and all(float(row.get("rms", math.inf)) <= 10.0 for row in (feet, wrists, root))
            wrist_ok = wrist_ok and player["wrist_peak_delta_frames"].get(str(key), {}).get("max_abs_delta_frames") == 0
            gate_ok = gate_ok and player["foot_slide_gate"].get(str(key), {}).get("predicted_pass") is True
        if jitter_ok and wrist_ok and gate_ok:
            viable.append(str(key))
    if viable:
        return (
            f"Candidate lambda(s) {', '.join(viable)} clear this harness's decoded-run checks. "
            "Still require 4-clip coverage and strict GATE 1b/mesh-divergence evidence before process_video wiring."
        )
    return (
        "No lambda is wiring-ready from this table. Keep the smoother unwired; collect decoded candidates on all "
        "required clips and resolve strict GATE 1b/mesh-divergence before any un-kill decision."
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Latent Smoothing Acceptance Report",
        "",
        f"- status: `{report.get('measurement_status')}`",
        f"- mode: `{report.get('measurement_mode')}`",
        f"- boundary: {report.get('acceptance_boundary', '')}",
        "",
    ]
    lambda_keys = list(report.get("candidate_run_dirs", {}).keys())
    for player_id, player in sorted((report.get("players") or {}).items()):
        lines.extend([f"## Player {player_id}", ""])
        for group in ("feet", "wrists", "root"):
            metric = player["world_jitter_mm_per_frame2"][group]
            lines.append(_metric_table_line(f"world_jitter_{group}_rms_mm_f2", metric, lambda_keys, field="rms"))
        lines.append(_metric_table_line("foot_slide_stance_p95_mm_f", player["foot_slide_mm_per_frame"]["stance"], lambda_keys, field="p95"))
        wrist = player["wrist_peak_delta_frames"]
        values = ["0"] + [str(wrist.get(key, {}).get("max_abs_delta_frames")) for key in lambda_keys]
        lines.append("| wrist_peak_delta_frames | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(["## Recommendation", "", str(report.get("recommendation", "")), ""])
    return "\n".join(lines)


def _metric_table_line(label: str, metric: Mapping[str, Any], lambda_keys: Sequence[str], *, field: str) -> str:
    values = [metric.get("raw", {}).get(field)]
    values.extend(metric.get(key, {}).get(field) for key in lambda_keys)
    return "| " + label + " | " + " | ".join(_format_value(value) for value in values) + " |"


def _acceptance_boundary(mode: str) -> str:
    if mode == "decoded_candidate_run_dirs":
        return "Acceptance-capable if candidate dirs were produced by decoded MHR latent smoothing."
    if mode == "world_joint_proxy_not_latent_decode":
        return "Non-acceptance proxy: real metric keys, but joints_world was smoothed directly."
    return "No acceptance measurement."


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _joints_array(frame: Mapping[str, Any]) -> np.ndarray | None:
    joints = frame.get("joints_world")
    if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or not joints:
        return None
    return np.asarray(joints, dtype=np.float64)


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return copy.deepcopy(value)


def _parse_lambdas(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise SystemExit("--lambda-smooth must contain at least one value")
    return parsed


def _lambda_key(value: float) -> str:
    return f"{float(value):g}"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_value(value: Any) -> str:
    parsed = _maybe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "measurement_status": report.get("measurement_status"),
                "measurement_mode": report.get("measurement_mode"),
                "player_count": len(report.get("players", {}) or {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("measurement_status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
