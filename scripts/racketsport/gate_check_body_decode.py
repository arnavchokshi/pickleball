#!/usr/bin/env python3
"""Canonical GATE-1a/GATE-1b BODY decode-fidelity harness.

This supersedes the lane-local
``runs/lanes/w5_p22latent_20260707/scripts/gate_check.py`` driver without
editing that evidence in place. It reads a raw ``body_mesh.json`` monolith,
preserves the persisted SMPL-X/MHR fields consumed by
``MHRHead.mhr_forward()``, records decoder checkpoint/asset provenance, and
computes the same comparison metric names used by the wave-5/6 reports.

Local Mac dev environments normally lack ``roma`` and the SAM-3D-Body
checkpoint/assets. In that case the CLI writes a blocked report with
``MHR_RUNTIME_AVAILABLE=False`` instead of pretending a decode verdict exists.
Use ``--self-check`` to exercise field plumbing to a stub decoder boundary
without the heavy runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport import mhr_decode  # noqa: E402
from threed.racketsport.schemas import CourtCalibration  # noqa: E402


REPORT_ARTIFACT_TYPE = "racketsport_body_decode_gate_check"
GATE_1A_NAME = "gate_1a_euler_cont_euler_idempotence"
GATE_1B_NAME = "gate_1b_world_round_trip"
MESH_DIVERGENCE_NAME = "mesh_skeleton_divergence"

MHR_FORWARD_FIELD_MAPPING: list[dict[str, str]] = [
    {
        "mhr_forward_arg": "global_trans",
        "source": "zero tensor from mhr_decode.MHRDecoder._prep_batch",
        "notes": "Persisted smplx_params.transl_world is not decoded; it is used later as world-grounding track_world_xy.",
    },
    {
        "mhr_forward_arg": "global_rot",
        "source": "body_mesh.players[].frames[].smplx_params.global_orient",
        "notes": "3-dim Euler global orientation, passed as global_orient_euler to decode_euler_frame().",
    },
    {
        "mhr_forward_arg": "body_pose_params",
        "source": "body_mesh.players[].frames[].smplx_params.body_pose",
        "notes": "133-dim persisted Euler body pose; MHRHead.mhr_forward internally truncates to the first 130 channels.",
    },
    {
        "mhr_forward_arg": "hand_pose_params",
        "source": "smplx_params.left_hand_pose + smplx_params.right_hand_pose",
        "notes": "54+54 hand components, concatenated verbatim when both hands are present; otherwise mhr_decode supplies zeros.",
    },
    {
        "mhr_forward_arg": "scale_params",
        "source": "body_mesh.players[].frames[].smplx_params.scale",
        "notes": "--scale-source field passes the 28-dim field and fails if absent; --scale-source none passes None so mhr_decode uses zeros/mean scale.",
    },
    {
        "mhr_forward_arg": "shape_params",
        "source": "body_mesh.players[].frames[].smplx_params.betas",
        "notes": "45-dim shape/betas vector, passed as decode_euler_frame(shape=...).",
    },
    {
        "mhr_forward_arg": "expr_params",
        "source": "zero tensor from mhr_decode.MHRDecoder._prep_batch",
        "notes": "Persisted body_mesh frames do not carry expression params; mhr_decode supplies FACE_COMPS_DIM zeros.",
    },
]


class HarnessInputError(ValueError):
    """Input cannot produce an honest decode-fidelity measurement."""


@dataclass(frozen=True)
class BodyDecodeFrame:
    player_id: str
    frame_idx: int
    t: float
    global_orient: list[float]
    body_pose: list[float]
    shape: list[float]
    scale: list[float] | None
    left_hand_pose: list[float] | None
    right_hand_pose: list[float] | None
    hand_pose: list[float] | None
    transl_world: list[float]
    joints_world: list[Any]
    vertices_world: list[Any]

    def decode_kwargs(self, *, scale_source: str) -> dict[str, Any]:
        if scale_source == "field":
            if self.scale is None:
                raise HarnessInputError(
                    "--scale-source field requested but smplx_params.scale is absent "
                    f"or invalid for player {self.player_id} frame {self.frame_idx}"
                )
            scale: list[float] | None = self.scale
        elif scale_source == "none":
            scale = None
        else:
            raise HarnessInputError(f"unsupported --scale-source: {scale_source}")
        return {
            "global_orient_euler": self.global_orient,
            "body_pose_euler": self.body_pose,
            "shape": self.shape,
            "scale": scale,
            "hand_pose": self.hand_pose,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical BODY GATE-1b decode-fidelity harness with scale/hand_pose "
            "field plumbing and decoder checkpoint/asset provenance."
        )
    )
    parser.add_argument("--body-mesh", type=Path, default=None, help="Raw body_mesh.json monolith from a BODY run.")
    parser.add_argument("--court-calibration", type=Path, default=None, help="court_calibration.json from the same raw run.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--checkpoint", type=Path, default=Path(mhr_decode.DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--mhr-asset", type=Path, default=Path(mhr_decode.DEFAULT_MHR_ASSET_PATH))
    parser.add_argument("--device", default=None, help="Decoder device override, e.g. cuda:0.")
    parser.add_argument("--max-frames-per-player", type=int, default=40)
    parser.add_argument(
        "--scale-source",
        choices=["none", "field"],
        default="none",
        help=(
            "'field' requires smplx_params.scale on every sampled real frame and passes it to mhr_decode; "
            "'none' decodes with population-mean scale."
        ),
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a decoder-free synthetic fixture through a stub mhr_decode boundary to prove field plumbing.",
    )
    return parser


def extract_frames(body_mesh: Mapping[str, Any]) -> dict[str, list[BodyDecodeFrame]]:
    out: dict[str, list[BodyDecodeFrame]] = {}
    for player in body_mesh.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id"))
        frames: list[BodyDecodeFrame] = []
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            params = frame.get("smplx_params", {})
            if not isinstance(params, Mapping):
                continue
            global_orient = _vector_or_none(params.get("global_orient"), mhr_decode.GLOBAL_ROT_EULER_DIM)
            body_pose = _vector_or_none(params.get("body_pose"), mhr_decode.BODY_POSE_EULER_DIM)
            if global_orient is None or body_pose is None:
                continue
            joints_world = frame.get("joints_world", []) or []
            if not joints_world:
                continue
            shape = _vector_or_none(params.get("betas"), mhr_decode.SHAPE_DIM)
            if shape is None:
                raise HarnessInputError(
                    f"smplx_params.betas must be {mhr_decode.SHAPE_DIM} values for "
                    f"player {player_id} frame {frame.get('frame_idx', -1)}"
                )
            scale = _vector_or_none(params.get("scale"), mhr_decode.SCALE_DIM)
            left_hand = _vector_or_none(params.get("left_hand_pose"), mhr_decode.HAND_COMPS_DIM)
            right_hand = _vector_or_none(params.get("right_hand_pose"), mhr_decode.HAND_COMPS_DIM)
            if (params.get("left_hand_pose") is not None or params.get("right_hand_pose") is not None) and (
                left_hand is None or right_hand is None
            ):
                raise HarnessInputError(
                    "smplx_params.left_hand_pose and right_hand_pose must both be "
                    f"{mhr_decode.HAND_COMPS_DIM} values when either is present "
                    f"(player {player_id} frame {frame.get('frame_idx', -1)})"
                )
            hand_pose = left_hand + right_hand if left_hand is not None and right_hand is not None else None
            transl_world = _vector_or_none(params.get("transl_world"), 3) or [0.0, 0.0, 0.0]
            frames.append(
                BodyDecodeFrame(
                    player_id=player_id,
                    frame_idx=int(frame.get("frame_idx", -1)),
                    t=float(frame.get("t", 0.0)),
                    global_orient=global_orient,
                    body_pose=body_pose,
                    shape=shape,
                    scale=scale,
                    left_hand_pose=left_hand,
                    right_hand_pose=right_hand,
                    hand_pose=hand_pose,
                    transl_world=transl_world,
                    joints_world=joints_world,
                    vertices_world=frame.get("mesh_vertices_world", []) or [],
                )
            )
        if frames:
            out[player_id] = frames
    return out


def build_decoder_provenance(*, checkpoint_path: Path, mhr_asset_path: Path) -> dict[str, Any]:
    module_path = Path(mhr_decode.__file__).resolve()
    module_sha = _sha256_file(module_path)
    return {
        "checkpoint": _path_provenance(checkpoint_path),
        "mhr_asset": _path_provenance(mhr_asset_path),
        "mhr_decode_module": {
            "path": str(module_path),
            "sha256": module_sha,
            "version_stamp": f"mhr_decode.py:{module_sha}",
            "dimensions": {
                "global_rot_euler_dim": mhr_decode.GLOBAL_ROT_EULER_DIM,
                "body_pose_euler_dim": mhr_decode.BODY_POSE_EULER_DIM,
                "shape_dim": mhr_decode.SHAPE_DIM,
                "scale_dim": mhr_decode.SCALE_DIM,
                "hand_comps_dim_per_hand": mhr_decode.HAND_COMPS_DIM,
                "face_comps_dim": mhr_decode.FACE_COMPS_DIM,
                "num_keypoints": mhr_decode.NUM_KEYPOINTS,
            },
        },
        "mhr_runtime_available": bool(mhr_decode.MHR_RUNTIME_AVAILABLE),
        "mhr_runtime_import_error": None
        if mhr_decode.MHR_RUNTIME_IMPORT_ERROR is None
        else repr(mhr_decode.MHR_RUNTIME_IMPORT_ERROR),
        "mhr_forward_signature": "MHRHead.mhr_forward(global_trans, global_rot, body_pose_params, hand_pose_params, scale_params, shape_params, expr_params=None, return_keypoints=False, do_pcblend=True, return_joint_coords=False, return_model_params=False, return_joint_rotations=False, scale_offsets=None, vertex_offsets=None, _do_timing=False)",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        return run_self_check(args)
    if args.body_mesh is None:
        raise HarnessInputError("--body-mesh is required unless --self-check is used")
    if args.court_calibration is None:
        raise HarnessInputError("--court-calibration is required unless --self-check is used")
    if args.max_frames_per_player < 1:
        raise HarnessInputError("--max-frames-per-player must be >= 1")

    t0 = time.time()
    body_mesh = json.loads(args.body_mesh.read_text(encoding="utf-8"))
    frames_by_player = extract_frames(body_mesh)
    _validate_scale_source(frames_by_player, args.scale_source)
    provenance = build_decoder_provenance(checkpoint_path=args.checkpoint, mhr_asset_path=args.mhr_asset)

    if not mhr_decode.MHR_RUNTIME_AVAILABLE:
        report = _base_report(
            body_mesh=body_mesh,
            body_mesh_path=args.body_mesh,
            court_calibration_path=args.court_calibration,
            frames_by_player=frames_by_player,
            scale_source=args.scale_source,
            provenance=provenance,
            wall_seconds=time.time() - t0,
            measurement_status="blocked_mhr_runtime_unavailable",
            blocker=f"MHR_RUNTIME_AVAILABLE=False: {mhr_decode.MHR_RUNTIME_IMPORT_ERROR!r}",
        )
        _write_report(args.out, report)
        return report

    calibration = CourtCalibration.model_validate(json.loads(args.court_calibration.read_text(encoding="utf-8")))
    gate1a = _compute_gate_1a(frames_by_player)
    decoder = mhr_decode.MHRDecoder(
        checkpoint_path=str(args.checkpoint),
        mhr_path=str(args.mhr_asset),
        device=args.device,
    )
    gate1b_summary, divergence_summary = _compute_gate_1b_and_divergence(
        decoder=decoder,
        calibration=calibration,
        frames_by_player=frames_by_player,
        scale_source=args.scale_source,
        max_frames_per_player=args.max_frames_per_player,
    )
    report = _base_report(
        body_mesh=body_mesh,
        body_mesh_path=args.body_mesh,
        court_calibration_path=args.court_calibration,
        frames_by_player=frames_by_player,
        scale_source=args.scale_source,
        provenance=provenance,
        wall_seconds=time.time() - t0,
        measurement_status="measured",
        blocker=None,
    )
    report["gate_1a"] = gate1a
    report[GATE_1A_NAME] = gate1a
    report["gate_1b"] = gate1b_summary
    report[GATE_1B_NAME] = gate1b_summary
    report[MESH_DIVERGENCE_NAME] = divergence_summary
    _write_report(args.out, report)
    return report


def run_self_check(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    body_mesh = _self_check_body_mesh()
    frames_by_player = extract_frames(body_mesh)
    frame = next(iter(frames_by_player.values()))[0]
    decoder = _RecordingDecoder()
    kwargs = frame.decode_kwargs(scale_source="field")
    decoder.decode_euler_frame(**kwargs)
    decoder.mesh_skeleton_divergence_mm(**kwargs)
    provenance = build_decoder_provenance(checkpoint_path=args.checkpoint, mhr_asset_path=args.mhr_asset)
    first_call = decoder.decode_calls[0]
    expected_hand_pose = frame.hand_pose or []
    passed = first_call.get("scale") == frame.scale and first_call.get("hand_pose") == expected_hand_pose
    report = _base_report(
        body_mesh=body_mesh,
        body_mesh_path=Path("<self-check>"),
        court_calibration_path=Path("<self-check>"),
        frames_by_player=frames_by_player,
        scale_source="field",
        provenance=provenance,
        wall_seconds=time.time() - t0,
        measurement_status="self_check_passed" if passed else "self_check_failed",
        blocker=None if passed else "stub decoder did not receive expected scale/hand_pose fields",
    )
    report["self_check_fixture"] = {
        "scale": frame.scale,
        "hand_pose": expected_hand_pose,
    }
    report["self_check"] = {
        "passed": passed,
        "decode_call_count": len(decoder.decode_calls),
        "divergence_call_count": len(decoder.divergence_calls),
        "first_decode_call": first_call,
        "first_divergence_call": decoder.divergence_calls[0],
    }
    _write_report(args.out, report)
    return report


def _compute_gate_1a(frames_by_player: Mapping[str, Sequence[BodyDecodeFrame]]) -> dict[str, Any]:
    all_global_orient = []
    all_body_pose = []
    for frames in frames_by_player.values():
        for frame in frames:
            all_global_orient.append(frame.global_orient)
            all_body_pose.append(frame.body_pose)
    if not all_global_orient:
        raise HarnessInputError("body_mesh contains no frames with persisted global_orient/body_pose and joints_world")
    return mhr_decode.gate_1a_euler_round_trip(
        np.asarray(all_global_orient, dtype=np.float64),
        np.asarray(all_body_pose, dtype=np.float64),
    )


def _compute_gate_1b_and_divergence(
    *,
    decoder: Any,
    calibration: CourtCalibration,
    frames_by_player: Mapping[str, Sequence[BodyDecodeFrame]],
    scale_source: str,
    max_frames_per_player: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_player_gate1b: dict[str, dict[str, Any]] = {}
    per_player_divergence: dict[str, dict[str, Any]] = {}
    worst_joints_mm = 0.0
    worst_vertices_mm = 0.0
    worst_divergence_p95_mm = 0.0

    for player_id, frames in frames_by_player.items():
        sample = _sample_frames(frames, max_frames_per_player=max_frames_per_player)
        joints_errs: list[float] = []
        vertices_errs: list[float] = []
        divergences: list[float] = []
        for frame in sample:
            kwargs = frame.decode_kwargs(scale_source=scale_source)
            decoded = decoder.decode_euler_frame(**kwargs)
            regrounded = mhr_decode.ground_decoded_camera_frame(
                joints_camera=decoded["joints_camera"][0],
                vertices_camera=decoded["vertices_camera"][0] if decoded["vertices_camera"] is not None else [],
                track_world_xy=frame.transl_world[:2],
                t=frame.t,
                frame_idx=frame.frame_idx,
                player_id=int(player_id),
                calibration=calibration,
            )
            gate1b = mhr_decode.gate_1b_world_round_trip(
                decoded_joints_world=regrounded["joints_world"],
                decoded_vertices_world=regrounded["vertices_world"],
                persisted_joints_world=frame.joints_world,
                persisted_vertices_world=frame.vertices_world,
            )
            joints_errs.append(gate1b["joints_world"]["max_abs_error_mm"])
            if frame.vertices_world:
                vertices_errs.append(gate1b["vertices_world"]["max_abs_error_mm"])
            div = decoder.mesh_skeleton_divergence_mm(**kwargs)
            divergences.append(div["p95_mm"])

        joints_arr = np.asarray(joints_errs if joints_errs else [0.0], dtype=np.float64)
        vertices_arr = np.asarray(vertices_errs if vertices_errs else [0.0], dtype=np.float64)
        divergence_arr = np.asarray(divergences if divergences else [0.0], dtype=np.float64)
        per_player_gate1b[player_id] = {
            "sample_count": len(sample),
            "joints_world_max_abs_error_mm": float(joints_arr.max()),
            "joints_world_p95_abs_error_mm": float(np.percentile(joints_arr, 95)),
            "vertices_world_max_abs_error_mm": float(vertices_arr.max()) if vertices_errs else None,
            "vertices_world_p95_abs_error_mm": float(np.percentile(vertices_arr, 95)) if vertices_errs else None,
        }
        per_player_divergence[player_id] = {
            "sample_count": len(sample),
            "p95_mm_max_over_sample": float(divergence_arr.max()),
            "p95_mm_mean_over_sample": float(divergence_arr.mean()),
        }
        worst_joints_mm = max(worst_joints_mm, per_player_gate1b[player_id]["joints_world_max_abs_error_mm"])
        if vertices_errs:
            worst_vertices_mm = max(
                worst_vertices_mm,
                per_player_gate1b[player_id]["vertices_world_max_abs_error_mm"] or 0.0,
            )
        worst_divergence_p95_mm = max(
            worst_divergence_p95_mm,
            per_player_divergence[player_id]["p95_mm_max_over_sample"],
        )

    gate1b_summary = {
        "gate": GATE_1B_NAME,
        "target_max_abs_error_mm": mhr_decode.GATE_1B_MAX_ABS_ERROR_MM,
        "scale_source": scale_source,
        "worst_joints_world_max_abs_error_mm": worst_joints_mm,
        "worst_vertices_world_max_abs_error_mm": worst_vertices_mm,
        "passed": bool(
            worst_joints_mm <= mhr_decode.GATE_1B_MAX_ABS_ERROR_MM
            and worst_vertices_mm <= mhr_decode.GATE_1B_MAX_ABS_ERROR_MM
        ),
        "per_player": per_player_gate1b,
    }
    divergence_summary = {
        "target_p95_mm": mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM,
        "worst_p95_mm_over_sample": worst_divergence_p95_mm,
        "passed": bool(worst_divergence_p95_mm <= mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM),
        "per_player": per_player_divergence,
    }
    return gate1b_summary, divergence_summary


def _base_report(
    *,
    body_mesh: Mapping[str, Any],
    body_mesh_path: Path,
    court_calibration_path: Path,
    frames_by_player: Mapping[str, Sequence[BodyDecodeFrame]],
    scale_source: str,
    provenance: Mapping[str, Any],
    wall_seconds: float,
    measurement_status: str,
    blocker: str | None,
) -> dict[str, Any]:
    total_frame_count = sum(len(frames) for frames in frames_by_player.values())
    gate1a = _blocked_gate_1a(blocker)
    gate1b = _blocked_gate_1b(scale_source=scale_source, blocker=blocker)
    divergence = _blocked_mesh_divergence(blocker)
    return {
        "schema_version": 1,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "measurement_status": measurement_status,
        "blocker": blocker,
        "body_mesh_path": str(body_mesh_path),
        "court_calibration_path": str(court_calibration_path),
        "clip": str(body_mesh.get("clip", "")),
        "player_ids": sorted(frames_by_player.keys()),
        "total_real_frame_sample_count": total_frame_count,
        "field_mapping": MHR_FORWARD_FIELD_MAPPING,
        "decoder_provenance": dict(provenance),
        "gate_1a": gate1a,
        GATE_1A_NAME: gate1a,
        "gate_1b": gate1b,
        GATE_1B_NAME: gate1b,
        MESH_DIVERGENCE_NAME: divergence,
        "wall_seconds": wall_seconds,
    }


def _blocked_gate_1a(blocker: str | None) -> dict[str, Any]:
    return {
        "gate": GATE_1A_NAME,
        "target_max_abs_error_deg": mhr_decode.GATE_1A_MAX_ABS_ERROR_DEG,
        "max_abs_error_deg": None,
        "passed": None,
        "blocked_reason": blocker,
    }


def _blocked_gate_1b(*, scale_source: str, blocker: str | None) -> dict[str, Any]:
    return {
        "gate": GATE_1B_NAME,
        "target_max_abs_error_mm": mhr_decode.GATE_1B_MAX_ABS_ERROR_MM,
        "scale_source": scale_source,
        "worst_joints_world_max_abs_error_mm": None,
        "worst_vertices_world_max_abs_error_mm": None,
        "passed": None,
        "per_player": {},
        "blocked_reason": blocker,
    }


def _blocked_mesh_divergence(blocker: str | None) -> dict[str, Any]:
    return {
        "target_p95_mm": mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM,
        "worst_p95_mm_over_sample": None,
        "passed": None,
        "per_player": {},
        "blocked_reason": blocker,
    }


def _validate_scale_source(frames_by_player: Mapping[str, Sequence[BodyDecodeFrame]], scale_source: str) -> None:
    if scale_source != "field":
        return
    for frames in frames_by_player.values():
        for frame in frames:
            frame.decode_kwargs(scale_source="field")


def _sample_frames(frames: Sequence[BodyDecodeFrame], *, max_frames_per_player: int) -> list[BodyDecodeFrame]:
    stride = max(1, len(frames) // max_frames_per_player)
    return list(frames[::stride][:max_frames_per_player])


def _vector_or_none(value: Any, expected_len: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != expected_len:
        return None
    return [float(v) for v in value]


def _path_provenance(path: Path) -> dict[str, Any]:
    sha = _sha256_file(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha,
        "sha256_status": "present" if sha is not None else "missing",
    }


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _RecordingDecoder:
    def __init__(self) -> None:
        self.decode_calls: list[dict[str, Any]] = []
        self.divergence_calls: list[dict[str, Any]] = []

    def decode_euler_frame(self, **kwargs: Any) -> dict[str, Any]:
        self.decode_calls.append(_jsonable_call(kwargs))
        return {"joints_camera": np.zeros((1, 1, 3)), "vertices_camera": np.zeros((1, 1, 3))}

    def mesh_skeleton_divergence_mm(self, **kwargs: Any) -> dict[str, Any]:
        self.divergence_calls.append(_jsonable_call(kwargs))
        return {"p95_mm": 0.0}


def _jsonable_call(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in kwargs.items()
    }


def _self_check_body_mesh() -> dict[str, Any]:
    left = [1.0 + idx * 0.01 for idx in range(mhr_decode.HAND_COMPS_DIM)]
    right = [-1.0 - idx * 0.01 for idx in range(mhr_decode.HAND_COMPS_DIM)]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "self_check",
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 3,
                        "t": 0.1,
                        "smplx_params": {
                            "global_orient": [0.1, 0.2, 0.3],
                            "body_pose": [0.001 * idx for idx in range(mhr_decode.BODY_POSE_EULER_DIM)],
                            "betas": [0.01 * idx for idx in range(mhr_decode.SHAPE_DIM)],
                            "scale": [0.2 + 0.01 * idx for idx in range(mhr_decode.SCALE_DIM)],
                            "left_hand_pose": left,
                            "right_hand_pose": right,
                            "transl_world": [1.0, 2.0, 0.0],
                        },
                        "joints_world": [[0.0, 0.0, 0.0]],
                        "mesh_vertices_world": [[0.0, 0.0, 0.0]],
                    }
                ],
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except HarnessInputError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    if report["measurement_status"] == "blocked_mhr_runtime_unavailable":
        print(report["blocker"], file=sys.stderr)
    print(
        json.dumps(
            {
                "measurement_status": report["measurement_status"],
                "player_count": len(report["player_ids"]),
                "total_real_frame_sample_count": report["total_real_frame_sample_count"],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if report["measurement_status"] == "blocked_mhr_runtime_unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
