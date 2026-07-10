#!/usr/bin/env python3
"""Decompose BODY decode residual into grounding, postchain, and FK/head terms.

The instrument consumes one run directory plus the raw Fast-SAM chunk index.
It reuses ``hmr_deep.normalize_fast_sam_body_output`` and the real
``worldhmr.compute_body_skeleton_and_metrics`` entrypoint.  Stage snapshots are
captured by temporary call-time wrappers; production modules are never edited.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport import coordinates, hmr_deep, mhr_decode, worldhmr  # noqa: E402
from threed.racketsport.body_postchain import BodyPostChainConfig  # noqa: E402
from threed.racketsport.schemas import CourtCalibration  # noqa: E402


ARTIFACT_TYPE = "racketsport_body_decode_residual_attribution"
STAGE_NAMES = (
    "temporal_smoothing",
    "foot_lock",
    "root_phase_median_lock",
    "world_joint_visual_smoothing",
    "wrist_peak_restore",
)
PROVENANCE_STAMP_KEYS = frozenset(
    {
        "batch_id",
        "created_at",
        "execution_id",
        "execution_timestamp",
        "generated_at",
        "inference_execution_id",
        "inference_timestamp",
        "out_path",
        "output_path",
        "run_id",
        "timestamp",
    }
)
RUN_PROVENANCE_FILENAMES = (
    "body_stage_phase_timing.json",
    "body_serialization_timing.json",
    "remote_body_dispatch_timing.json",
    "version_stamp.json",
    "pipeline_run.json",
)
EXECUTION_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{6}T\d{6}Z)(?!\d)")
SAM3D_BATCH_ID_PATTERN = re.compile(r"batch_outputs-([A-Za-z0-9]+)")


class AttributionInputError(ValueError):
    """An input cannot support an honest attribution run."""


class IncoherentInputsError(AttributionInputError):
    """The raw inference records cannot be attributed to the scored run."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        super().__init__("raw SAM-3D records and scored BODY frames are incoherent")
        self.report = dict(report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute BODY GATE-1b residual to grounding, postchain, and FK-vs-head components."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--raw-grounded", type=Path, default=None)
    parser.add_argument("--sam3d-output-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames-per-player", type=int, default=0)
    parser.add_argument(
        "--input-coverage-threshold",
        type=float,
        default=1.0,
        help="Required raw-record coverage of selected body_mesh player-frames (default: 1.0).",
    )
    parser.add_argument(
        "--allow-incoherent-inputs",
        action="store_true",
        help="Forensic opt-out: continue while stamping incoherence evidence into every result block.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(mhr_decode.DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--mhr-asset", type=Path, default=Path(mhr_decode.DEFAULT_MHR_ASSET_PATH))
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_frames_per_player < 0:
        raise AttributionInputError("--max-frames-per-player must be >= 0")
    coverage_threshold = float(getattr(args, "input_coverage_threshold", 1.0))
    if not 0.0 <= coverage_threshold <= 1.0:
        raise AttributionInputError("--input-coverage-threshold must be between 0 and 1")
    allow_incoherent = bool(getattr(args, "allow_incoherent_inputs", False))
    run_dir = args.run_dir.resolve()
    calibration_path = (args.calibration or run_dir / "court_calibration.json").resolve()
    if not run_dir.is_dir():
        raise AttributionInputError(f"--run-dir does not exist: {run_dir}")
    if not calibration_path.is_file():
        raise AttributionInputError(f"calibration does not exist: {calibration_path}")
    index_path = args.sam3d_output_index or _discover_index(run_dir)
    if index_path is None:
        raise AttributionInputError(
            "no Fast-SAM chunk index found; pass --sam3d-output-index or place one under run-dir/fast_sam_subprocess"
        )
    index_path = index_path.resolve()
    index_payload = _read_json(index_path)
    calibration_payload = _read_json(calibration_path)
    calibration = CourtCalibration.model_validate(calibration_payload)
    raw_reference = _read_json(args.raw_grounded.resolve()) if args.raw_grounded is not None else None
    body_mesh = _optional_json(run_dir / "body_mesh.json")
    skeleton3d = _optional_json(run_dir / "skeleton3d.json")
    placement = _optional_json(run_dir / "placement.json")
    fps = _resolve_fps(raw_reference, body_mesh, skeleton3d)
    assumptions: list[str] = []
    contexts, context_source = _frame_contexts(
        raw_reference=raw_reference,
        body_mesh=body_mesh,
        skeleton3d=skeleton3d,
        placement=placement,
        fps=fps,
    )
    if context_source != "raw_grounded_reference":
        assumptions.append(
            f"Raw grounding context was unavailable; track_world_xy/t came from {context_source}."
        )
    records = load_present_raw_records(index_path, index_payload=index_payload)
    run_provenance, run_provenance_sources = _load_run_provenance(run_dir, body_mesh=body_mesh)
    expected_ids = _measurement_request_ids(
        body_mesh=body_mesh,
        records=records,
        contexts=contexts,
        max_frames_per_player=args.max_frames_per_player,
    )
    coherence = _input_coherence_evidence(
        expected_request_ids=expected_ids,
        records=records,
        index_payload=index_payload,
        run_payload=run_provenance,
        coverage_threshold=coverage_threshold,
        coverage_source=(
            "body_mesh_player_frames"
            if body_mesh is not None
            else "present_raw_records_restricted_to_available_context"
        ),
    )
    if not coherence["inputs_coherent"] and not allow_incoherent:
        failure_report = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "status": "incoherent_inputs",
            "inputs_coherent": False,
            "run_dir": str(run_dir),
            "sam3d_output_index_path": str(index_path),
            "run_provenance_sources": run_provenance_sources,
            "input_coherence": coherence,
        }
        _write_json(args.out, failure_report)
        raise IncoherentInputsError(failure_report)
    selected_ids = [request_id for request_id in expected_ids if request_id in records and request_id in contexts]
    if not selected_ids:
        raise AttributionInputError("no present raw records have matching run-frame context")
    samples, raw_records = _normalize_samples(
        selected_ids,
        records=records,
        contexts=contexts,
        calibration_payload=calibration_payload,
    )
    postchain, knobs, knob_assumptions = _load_postchain_knobs(run_dir, raw_reference)
    assumptions.extend(knob_assumptions)

    t0 = time.time()
    grounded_frames = [
        worldhmr._ground_fast_sam_sample(sample, calibration=calibration, camera_motion=None)
        for sample in samples
    ]
    grounded_snapshot = _frame_map_from_frames(grounded_frames)
    grounding = _grounding_determinism(grounded_snapshot, raw_reference)

    capture = WorldHmrStageCapture(worldhmr)
    with capture:
        computed = worldhmr.compute_body_skeleton_and_metrics(
            samples,
            calibration=calibration,
            fps=fps,
            smoothing_alpha=knobs["smoothing_alpha"],
            max_root_speed_mps=knobs["max_root_speed_mps"],
            max_track_anchor_smoothing_residual_m=knobs["max_track_anchor_smoothing_residual_m"],
            grounding_anchor_source=knobs["grounding_anchor_source"],
            smoothing_gap_carry_frames=knobs["smoothing_gap_carry_frames"],
            smoothing_residual_identity_reset_m=knobs["smoothing_residual_identity_reset_m"],
            body_postchain=postchain,
        )
    final_mesh = _frame_map_from_payload(computed.smpl_motion_view)
    final_skeleton = _frame_map_from_payload(computed.skeleton3d)
    postchain_report = _postchain_attribution(
        grounded_snapshot,
        capture.snapshots,
        final_mesh,
        postchain=postchain,
    )
    replay = _replay_validation(
        final_mesh=final_mesh,
        final_skeleton=final_skeleton,
        body_mesh=body_mesh,
        skeleton3d=skeleton3d,
        stage_snapshots=capture.snapshots,
    )
    fk_vs_head = _fk_vs_head_divergence(
        selected_ids,
        records=raw_records,
        checkpoint=args.checkpoint,
        mhr_asset=args.mhr_asset,
        device=args.device,
    )
    for result_block in (grounding, postchain_report, replay, fk_vs_head):
        result_block["inputs_coherent"] = bool(coherence["inputs_coherent"])
        result_block["input_coherence"] = coherence
    report = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "measured",
        "inputs_coherent": bool(coherence["inputs_coherent"]),
        "input_coherence": coherence,
        "run_provenance_sources": run_provenance_sources,
        "run_dir": str(run_dir),
        "calibration_path": str(calibration_path),
        "raw_grounded_path": None if args.raw_grounded is None else str(args.raw_grounded.resolve()),
        "sam3d_output_index_path": str(index_path),
        "method": {
            "grounding": (
                "hmr_deep.normalize_fast_sam_body_output -> worldhmr._ground_fast_sam_sample; "
                "pred_cam_t is applied exactly once by the production normalizer"
            ),
            "postchain": (
                "real worldhmr.compute_body_skeleton_and_metrics entrypoint with temporary call-time "
                "wrappers around stage functions; no production monkeypatch persists after the call"
            ),
            "replay": "index-aligned player/frame joint comparison on every selected present raw record",
        },
        "coordinate_spaces": {
            "raw_head": coordinates.CoordinateSpace.BODY_CAMERA_ROOT_RELATIVE_M.value,
            "normalized": coordinates.CoordinateSpace.CAMERA_M.value,
            "persisted": coordinates.CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M.value,
        },
        "input_summary": {
            "present_raw_record_count": len(records),
            "selected_raw_record_count": len(selected_ids),
            "selected_request_ids": selected_ids,
            "player_ids": sorted({request_id.split(":", 1)[1] for request_id in selected_ids}),
            "fps": fps,
            "body_mesh_present": body_mesh is not None,
            "skeleton3d_present": skeleton3d is not None,
        },
        "postchain_knobs": {**postchain.to_artifact_dict(), **knobs},
        "assumptions": assumptions,
        "grounding_determinism": grounding,
        "postchain_attribution": postchain_report,
        "replay_validation": replay,
        "fk_vs_head_divergence": fk_vs_head,
        "wall_seconds": time.time() - t0,
    }
    _write_json(args.out, report)
    return report


class WorldHmrStageCapture:
    """Temporary wrappers that snapshot real worldhmr stage outputs."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.originals: dict[str, Any] = {}
        self.snapshots: dict[str, dict[tuple[str, int], list[list[float]]]] = {}
        self._foot_frames: list[dict[str, Any]] = []

    def __enter__(self) -> WorldHmrStageCapture:
        self._wrap_frame_stage("_smooth_grounded_frames", "temporal_smoothing")
        self._wrap_frame_stage("_smooth_grounded_frames_stance_aware", "temporal_smoothing")
        self._wrap_frame_stage("_bypass_temporal_smoothing", "temporal_smoothing")
        self._wrap_foot_stage("_apply_footlock_to_player_frames")
        self._wrap_foot_stage("_bypass_footlock_for_player_frames")
        self._wrap_root_lock()
        self._wrap_peak_restore()
        self._wrap_visual_smoothing()
        return self

    def __exit__(self, *_exc: object) -> None:
        for name, original in self.originals.items():
            setattr(self.module, name, original)

    def _remember(self, name: str) -> Any:
        original = getattr(self.module, name)
        self.originals[name] = original
        return original

    def _wrap_frame_stage(self, symbol: str, stage: str) -> None:
        original = self._remember(symbol)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.snapshots[stage] = _frame_map_from_frames(result[0])
            return result

        setattr(self.module, symbol, wrapper)

    def _wrap_foot_stage(self, symbol: str) -> None:
        original = self._remember(symbol)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._foot_frames.extend(copy.deepcopy(result[0]))
            self.snapshots["foot_lock"] = _frame_map_from_frames(self._foot_frames)
            return result

        setattr(self.module, symbol, wrapper)

    def _wrap_root_lock(self) -> None:
        symbol = "_apply_root_phase_median_lock_to_payload"
        original = self._remember(symbol)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if bool(kwargs.get("translate_mesh")):
                self.snapshots["root_phase_median_lock"] = _frame_map_from_payload(result[0])
            return result

        setattr(self.module, symbol, wrapper)

    def _wrap_peak_restore(self) -> None:
        symbol = "_restore_wrist_peak_timing_windows"
        original = self._remember(symbol)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 2:
                self.snapshots["world_joint_visual_smoothing"] = _frame_map_from_payload(args[1])
            result = original(*args, **kwargs)
            if len(args) >= 2:
                self.snapshots["wrist_peak_restore"] = _frame_map_from_payload(args[1])
            return result

        setattr(self.module, symbol, wrapper)

    def _wrap_visual_smoothing(self) -> None:
        symbol = "_apply_world_joint_visual_smoothing"
        original = self._remember(symbol)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.snapshots.setdefault("world_joint_visual_smoothing", _frame_map_from_payload(result[0]))
            self.snapshots.setdefault("wrist_peak_restore", _frame_map_from_payload(result[0]))
            return result

        setattr(self.module, symbol, wrapper)


def load_present_raw_records(
    index_path: Path,
    *,
    index_payload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load only locally present buckets, ignoring dead paths in the index."""

    index = dict(index_payload) if index_payload is not None else _read_json(index_path)
    records: dict[str, dict[str, Any]] = {}
    for chunk in index.get("chunks", []) or []:
        path = index_path.parent / str(chunk.get("path", ""))
        if not path.is_file():
            continue
        if path.suffix != ".pkl":
            raise AttributionInputError(f"present chunk format is unsupported by this instrument: {path}")
        with path.open("rb") as handle:
            payload = _restore_pickle_arrays(pickle.load(handle))  # noqa: S301 - trusted local run artifact
        items = payload.get("bucket_items", [])
        raw_records = payload.get("raw_records", [])
        for item, record in zip(items, raw_records, strict=True):
            if not isinstance(item, Mapping) or not isinstance(record, Mapping) or item.get("is_padding"):
                continue
            request_id = str(item.get("request_id") or record.get("request_id"))
            if request_id in records:
                raise AttributionInputError(f"duplicate raw record for request {request_id}")
            records[request_id] = {"item": dict(item), "record": dict(record)}
    return records


def _restore_pickle_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__sam3d_pickle_ndarray__"):
            array = np.frombuffer(value["data"], dtype=value["dtype"])
            shape = value.get("shape") or []
            return array.reshape(shape).copy() if shape else array.copy()
        return {str(key): _restore_pickle_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_pickle_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_pickle_arrays(item) for item in value)
    return value


def _normalize_samples(
    request_ids: Sequence[str],
    *,
    records: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    calibration_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    image_size = calibration_payload.get("image_size") or [1920, 1080]
    width, height = int(image_size[0]), int(image_size[1])
    samples: list[dict[str, Any]] = []
    raw_records: dict[str, Mapping[str, Any]] = {}
    for request_id in request_ids:
        frame_idx_raw, player_id_raw = request_id.split(":", 1)
        packed = records[request_id]
        item = packed["item"]
        record = packed["record"]
        context = contexts[request_id]
        bbox = item.get("bbox") or record.get("bbox")
        max_x = max(float(bbox[0]), float(bbox[2]))
        max_y = max(float(bbox[1]), float(bbox[3]))
        request = hmr_deep.PlayerCropRequest(
            frame_idx=int(frame_idx_raw),
            player_id=int(player_id_raw),
            bbox_xyxy=bbox,
            image_size_px=(max(width, int(math.ceil(max_x))), max(height, int(math.ceil(max_y)))),
            track_confidence=float(context.get("confidence", 1.0)),
        )
        sample = hmr_deep.normalize_fast_sam_body_output(record, request=request)
        sample["t"] = float(context["t"])
        sample["track_world_xy"] = [float(value) for value in context["track_world_xy"]]
        samples.append(sample)
        raw_records[request_id] = record
    return samples, raw_records


def _grounding_determinism(
    grounded: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    raw_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if raw_reference is None:
        return {"status": "no_reference", "passed_1mm": None, "per_player": {}, "overall": None}
    reference = _frame_map_from_payload(raw_reference)
    stats = _delta_stats(grounded, reference)
    return {
        "status": "measured",
        **stats,
        "passed_1mm": bool(stats["overall"]["max_mm"] <= 1.0),
    }


def _postchain_attribution(
    grounded: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    snapshots: Mapping[str, Mapping[tuple[str, int], Sequence[Sequence[float]]]],
    final_mesh: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    *,
    postchain: BodyPostChainConfig,
) -> dict[str, Any]:
    enabled = {
        "temporal_smoothing": postchain.temporal_smoothing,
        "foot_lock": postchain.foot_lock,
        "root_phase_median_lock": postchain.foot_pin,
        "world_joint_visual_smoothing": postchain.world_joint_visual_smoothing,
        "wrist_peak_restore": postchain.world_joint_visual_smoothing,
    }
    previous = grounded
    stages = []
    for name in STAGE_NAMES:
        current = snapshots.get(name, previous)
        stages.append(
            {
                "stage": name,
                "enabled": bool(enabled[name]),
                "snapshot_captured": name in snapshots,
                "delta_vs_previous": _delta_stats(current, previous),
            }
        )
        previous = current
    total = _delta_stats(final_mesh, grounded)
    all_disabled = not any(enabled.values())
    return {
        "status": "measured",
        "entrypoint": "worldhmr.compute_body_skeleton_and_metrics",
        "stages": stages,
        "total_delta": total,
        "all_stages_disabled": all_disabled,
        "all_stages_disabled_identity_max_m": (
            float(total["overall"]["max_mm"]) / 1000.0 if all_disabled else None
        ),
        "skeleton_only_stages": {
            "temporal_refine_and_wrist_lock_callsite_1": postchain.temporal_smoothing,
            "foot_pin": postchain.foot_pin,
            "contact_splice": "outside worldhmr entrypoint; not replayed by this instrument",
            "wrist_lock_callsite_2": "outside worldhmr entrypoint; not replayed by this instrument",
        },
    }


def _replay_validation(
    *,
    final_mesh: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    final_skeleton: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    body_mesh: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    stage_snapshots: Mapping[str, Mapping[tuple[str, int], Sequence[Sequence[float]]]],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    measured_stats = []
    if body_mesh is None:
        targets["body_mesh"] = {"status": "absent"}
    else:
        stats = _delta_stats(final_mesh, _frame_map_from_payload(body_mesh))
        targets["body_mesh"] = {"status": "measured", **stats}
        measured_stats.append(stats)
    if skeleton3d is None:
        targets["skeleton3d"] = {"status": "absent"}
    else:
        stats = _delta_stats(final_skeleton, _frame_map_from_payload(skeleton3d))
        targets["skeleton3d"] = {"status": "measured", **stats}
        measured_stats.append(stats)
    reproduced = bool(measured_stats) and all(stats["overall"]["max_mm"] <= 1.0 for stats in measured_stats)
    first_divergent = None
    if not reproduced and measured_stats:
        reference = _frame_map_from_payload(body_mesh or skeleton3d or {})
        for name in STAGE_NAMES:
            snapshot = stage_snapshots.get(name)
            if snapshot and _delta_stats(snapshot, reference)["overall"]["max_mm"] > 1.0:
                first_divergent = name
                break
        first_divergent = first_divergent or "persisted_final_after_worldhmr_entrypoint"
    return {
        **targets,
        "chain_reproduced_1mm": reproduced,
        "first_divergent_stage": first_divergent,
    }


def _fk_vs_head_divergence(
    request_ids: Sequence[str],
    *,
    records: Mapping[str, Mapping[str, Any]],
    checkpoint: Path,
    mhr_asset: Path,
    device: str | None,
) -> dict[str, Any]:
    if not mhr_decode.MHR_RUNTIME_AVAILABLE:
        return {
            "status": "blocked_mhr_runtime_unavailable",
            "error": repr(mhr_decode.MHR_RUNTIME_IMPORT_ERROR),
            "informational": True,
        }
    try:
        decoder = mhr_decode.MHRDecoder(
            checkpoint_path=str(checkpoint),
            mhr_path=str(mhr_asset),
            device=device,
        )
        raw_errors: list[float] = []
        aligned_errors: list[float] = []
        per_player_raw: dict[str, list[float]] = defaultdict(list)
        per_player_aligned: dict[str, list[float]] = defaultdict(list)
        for request_id in request_ids:
            record = records[request_id]
            decoded = decoder.decode_euler_frame(
                global_orient_euler=record["global_rot"],
                body_pose_euler=record["body_pose_params"],
                shape=record["shape_params"],
                scale=record.get("scale_params"),
                hand_pose=record.get("hand_pose_params"),
            )
            head = np.asarray(record["pred_keypoints_3d"], dtype=np.float64)
            fk = np.asarray(decoded["joints_camera"][0], dtype=np.float64)
            n = min(len(head), len(fk))
            player_id = request_id.split(":", 1)[1]
            raw = np.linalg.norm(fk[:n] - head[:n], axis=1) * 1000.0
            aligned = np.linalg.norm(
                (fk[:n] - fk[0]) - (head[:n] - head[0]), axis=1
            ) * 1000.0
            raw_errors.extend(raw.tolist())
            aligned_errors.extend(aligned.tolist())
            per_player_raw[player_id].extend(raw.tolist())
            per_player_aligned[player_id].extend(aligned.tolist())
        return {
            "status": "measured",
            "informational": True,
            "raw": _stats(raw_errors),
            "root_aligned": _stats(aligned_errors),
            "per_player": {
                player_id: {
                    "raw": _stats(per_player_raw[player_id]),
                    "root_aligned": _stats(per_player_aligned[player_id]),
                }
                for player_id in sorted(per_player_raw)
            },
        }
    except Exception as exc:  # noqa: BLE001 - fail-soft optional GPU stage.
        return {
            "status": "blocked_mhr_runtime_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "informational": True,
        }


def _delta_stats(
    left: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    right: Mapping[tuple[str, int], Sequence[Sequence[float]]],
) -> dict[str, Any]:
    per_player_values: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(left) & set(right)):
        a = np.asarray(left[key], dtype=np.float64)
        b = np.asarray(right[key], dtype=np.float64)
        n = min(len(a), len(b))
        if n:
            per_player_values[key[0]].extend((np.linalg.norm(a[:n] - b[:n], axis=1) * 1000.0).tolist())
    if not per_player_values:
        return {"per_player": {}, "overall": _stats([]), "matched_frame_count": 0}
    all_values = [value for values in per_player_values.values() for value in values]
    return {
        "per_player": {player_id: _stats(values) for player_id, values in sorted(per_player_values.items())},
        "overall": _stats(all_values),
        "matched_frame_count": len(set(left) & set(right)),
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"sample_count": 0, "mean_mm": None, "p95_mm": None, "max_mm": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": int(array.size),
        "mean_mm": float(array.mean()),
        "p95_mm": float(np.percentile(array, 95)),
        "max_mm": float(array.max()),
    }


def _frame_map_from_frames(frames: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[list[float]]]:
    return {
        (str(frame["player_id"]), int(frame["frame_idx"])): [list(map(float, joint)) for joint in frame["joints_world"]]
        for frame in frames
        if frame.get("joints_world")
    }


def _frame_map_from_payload(payload: Mapping[str, Any]) -> dict[tuple[str, int], list[list[float]]]:
    out: dict[tuple[str, int], list[list[float]]] = {}
    for player in payload.get("players", []) or []:
        player_id = str(player.get("id"))
        for frame in player.get("frames", []) or []:
            joints = frame.get("joints_world") or []
            if joints:
                out[(player_id, int(frame.get("frame_idx", -1)))] = [list(map(float, joint)) for joint in joints]
    return out


def _frame_contexts(
    *,
    raw_reference: Mapping[str, Any] | None,
    body_mesh: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    placement: Mapping[str, Any] | None,
    fps: float,
) -> tuple[dict[str, dict[str, Any]], str]:
    if raw_reference is not None:
        return _contexts_from_payload(raw_reference, fps=fps), "raw_grounded_reference"
    contexts: dict[str, dict[str, Any]] = {}
    if placement is not None:
        for player in placement.get("players", []) or []:
            for frame in player.get("frames", []) or []:
                frame_idx = int(frame["frame_idx"])
                xy = frame.get("fused_world_xy") or frame.get("smoothed_world_xy")
                if xy:
                    contexts[f"{frame_idx}:{player['id']}"] = {
                        "t": float(frame.get("t", frame_idx / fps)),
                        "track_world_xy": xy,
                        "confidence": 1.0,
                    }
        if contexts:
            return contexts, "placement.json fused_world_xy"
    payload = body_mesh or skeleton3d
    if payload is not None:
        return _contexts_from_payload(payload, fps=fps), "persisted BODY transl_world"
    raise AttributionInputError("run has no raw reference, placement, body_mesh, or skeleton3d frame context")


def _contexts_from_payload(payload: Mapping[str, Any], *, fps: float) -> dict[str, dict[str, Any]]:
    contexts = {}
    for player in payload.get("players", []) or []:
        for frame in player.get("frames", []) or []:
            frame_idx = int(frame.get("frame_idx", -1))
            xy = frame.get("track_world_xy") or frame.get("transl_world", [])[:2]
            confidence = (frame.get("joint_conf") or [1.0])[0]
            contexts[f"{frame_idx}:{player['id']}"] = {
                "t": float(frame.get("t", frame_idx / fps)),
                "track_world_xy": xy,
                "confidence": float(confidence),
            }
    return contexts


def _load_postchain_knobs(
    run_dir: Path,
    raw_reference: Mapping[str, Any] | None,
) -> tuple[BodyPostChainConfig, dict[str, Any], list[str]]:
    assumptions = []
    grounding_quality = _optional_json(run_dir / "body_grounding_quality.json") or {}
    metrics = grounding_quality.get("grounding_metrics", {}) if isinstance(grounding_quality, Mapping) else {}
    postchain_payload = (raw_reference or {}).get("postchain") or metrics.get("body_postchain")
    if not isinstance(postchain_payload, Mapping):
        postchain_payload = {}
        assumptions.append("No body_postchain artifact knobs were found; default BodyPostChainConfig was used.")
    postchain = BodyPostChainConfig(
        temporal_smoothing=bool(postchain_payload.get("temporal_smoothing", True)),
        foot_lock=bool(postchain_payload.get("foot_lock", True)),
        foot_pin=bool(postchain_payload.get("foot_pin", True)),
        contact_splice=bool(postchain_payload.get("contact_splice", True)),
        wrist_lock=bool(postchain_payload.get("wrist_lock", True)),
        world_joint_visual_smoothing=bool(postchain_payload.get("world_joint_visual_smoothing", True)),
    )
    defaults = {
        "smoothing_alpha": 0.65,
        "max_root_speed_mps": None,
        "max_track_anchor_smoothing_residual_m": None,
        "grounding_anchor_source": None,
        "smoothing_gap_carry_frames": worldhmr.DEFAULT_SMOOTHING_GAP_CARRY_FRAMES,
        "smoothing_residual_identity_reset_m": worldhmr.DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M,
    }
    knobs = {key: metrics.get(key, value) for key, value in defaults.items()}
    for key in defaults:
        if key not in metrics:
            assumptions.append(f"{key} was not recorded; worldhmr default {defaults[key]!r} was used.")
    if (
        knobs["grounding_anchor_source"] == worldhmr.R3_GROUNDING_ANCHOR_SOURCE
        and (postchain.temporal_smoothing or postchain.foot_pin)
    ):
        assumptions.append(
            "No serialized stance_index contract is accepted by this CLI; grounding_anchor_source is recorded, "
            "but stance-aware smoothing cannot be reconstructed when enabled."
        )
    return postchain, knobs, assumptions


def _select_request_ids(request_ids: Sequence[str], *, max_frames_per_player: int) -> list[str]:
    ordered = sorted(request_ids, key=lambda value: (int(value.split(":", 1)[0]), int(value.split(":", 1)[1])))
    if max_frames_per_player == 0:
        return ordered
    selected: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for request_id in ordered:
        player_id = request_id.split(":", 1)[1]
        if counts[player_id] < max_frames_per_player:
            selected.append(request_id)
            counts[player_id] += 1
    return selected


def _measurement_request_ids(
    *,
    body_mesh: Mapping[str, Any] | None,
    records: Mapping[str, Any],
    contexts: Mapping[str, Any],
    max_frames_per_player: int,
) -> list[str]:
    """Choose the player-frames whose coherence must be established before scoring."""

    if body_mesh is not None:
        request_ids = [
            f"{frame_idx}:{player_id}"
            for player_id, frame_idx in _frame_map_from_payload(body_mesh)
        ]
    else:
        # The banked CPU fixture intentionally has no body_mesh and only two
        # locally present buckets. Its measurement domain is the explicit
        # present-record subset for which grounding context exists.
        request_ids = [request_id for request_id in records if request_id in contexts]
    return _select_request_ids(request_ids, max_frames_per_player=max_frames_per_player)


def _input_coherence_evidence(
    *,
    expected_request_ids: Sequence[str],
    records: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    run_payload: Mapping[str, Any] | None,
    coverage_threshold: float,
    coverage_source: str,
) -> dict[str, Any]:
    expected = set(expected_request_ids)
    present = set(records)
    missing = sorted(expected - present, key=_request_id_sort_key)
    covered_count = len(expected & present)
    coverage = covered_count / len(expected) if expected else 0.0
    index_stamps = _collect_provenance_stamps(index_payload)
    run_stamps = _collect_provenance_stamps(run_payload or {})
    mismatches = []
    for stamp in sorted(set(index_stamps) & set(run_stamps)):
        index_values = index_stamps[stamp]
        run_values = run_stamps[stamp]
        if set(index_values) != set(run_values):
            mismatches.append(
                {
                    "stamp": stamp,
                    "index_values": index_values,
                    "run_values": run_values,
                }
            )
    coherent = bool(expected) and coverage >= coverage_threshold and not mismatches
    return {
        "status": "coherent" if coherent else "incoherent_inputs",
        "inputs_coherent": coherent,
        "coverage_source": coverage_source,
        "coverage_threshold": coverage_threshold,
        "expected_request_id_count": len(expected),
        "present_raw_record_count": len(present),
        "covered_request_id_count": covered_count,
        "coverage_fraction": coverage,
        "missing_request_id_count": len(missing),
        "missing_request_ids_first_10": missing[:10],
        "stamp_mismatches": mismatches,
        "index_stamps": index_stamps,
        "run_stamps": run_stamps,
    }


def _collect_provenance_stamps(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    found: dict[str, set[str]] = defaultdict(set)

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in PROVENANCE_STAMP_KEYS and isinstance(item, (str, int, float)):
                    found[key].add(str(item))
                if isinstance(item, str):
                    batch_ids = SAM3D_BATCH_ID_PATTERN.findall(item)
                    if batch_ids:
                        found["sam3d_batch_id_from_path"].update(batch_ids)
                        found["sam3d_execution_timestamp_from_path"].update(
                            EXECUTION_TIMESTAMP_PATTERN.findall(item)
                        )
                if key not in {"players", "frames", "mesh_faces"}:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return {key: sorted(values) for key, values in sorted(found.items())}


def _load_run_provenance(
    run_dir: Path,
    *,
    body_mesh: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Load the small sibling artifacts that stamp the BODY inference execution."""

    payloads: dict[str, Any] = {}
    sources: list[str] = []
    if body_mesh is not None:
        payloads["body_mesh.json"] = body_mesh
        sources.append(str(run_dir / "body_mesh.json"))
    for filename in RUN_PROVENANCE_FILENAMES:
        path = run_dir / filename
        if path.is_file():
            payloads[filename] = _read_json(path)
            sources.append(str(path))
    return payloads, sources


def _request_id_sort_key(value: str) -> tuple[int, int]:
    try:
        frame_idx, player_id = value.split(":", 1)
        return int(frame_idx), int(player_id)
    except (TypeError, ValueError):
        return sys.maxsize, sys.maxsize


def _resolve_fps(*payloads: Mapping[str, Any] | None) -> float:
    for payload in payloads:
        if payload is not None and float(payload.get("fps", 0.0) or 0.0) > 0.0:
            return float(payload["fps"])
    return 30.0


def _discover_index(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "fast_sam_subprocess").glob("batch_outputs-*.json.chunks/index.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise AttributionInputError(f"multiple Fast-SAM chunk indexes found; pass one explicitly:\n{formatted}")
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionInputError(f"failed to read {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AttributionInputError(f"JSON payload must be an object: {path}")
    return payload


def _optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except IncoherentInputsError as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except AttributionInputError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "grounding_status": report["grounding_determinism"]["status"],
                "chain_reproduced_1mm": report["replay_validation"]["chain_reproduced_1mm"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
