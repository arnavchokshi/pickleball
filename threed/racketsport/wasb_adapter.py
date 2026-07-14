"""WASB-SBDT prediction adapters for schema-valid ball tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .ball_size_observations import (
    HEATMAP_THRESHOLD,
    connected_component_blob_extents,
    write_wasb_ball_size_observations,
)
from .ball_tracknet import ball_frame
from .io_decode import time_for_frame
from .schemas import BallCandidates, BallTrack


WASB_REPO_URL = "https://github.com/nttcom/WASB-SBDT"
WASB_MODEL_ZOO_URL = "https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md"
WASB_COLUMNS = ("Frame", "Visibility", "X", "Y", "Confidence")
WASB_CONFIDENCE_SEMANTICS = "WASB heatmap peak value (0..1)"
STATUS_TESTED = "TESTED-ON-REAL-DATA"
DEFAULT_WASB_VISIBLE_THRESHOLD = 0.5
DEFAULT_BALL_CANDIDATE_TOP_K = 5
DEFAULT_EMIT_SIZE_OBSERVATIONS = True
DEFAULT_EMIT_BELOW_THRESHOLD_CANDIDATES = False
DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR = 0.05
WASB_INPUT_WH = (512, 288)
WASB_FRAMES_IN = 3
WASB_FRAMES_OUT = 3
DEFAULT_WASB_INPUT_PREPROCESSING = "official"
WASB_INPUT_PREPROCESSING_MODES = ("official", "harness_v0")
NON_PROMOTABLE_INPUT_PREPROCESSING_MODES = {"harness_v0"}
WASB_IMAGENET_MEAN = (0.485, 0.456, 0.406)
WASB_IMAGENET_STD = (0.229, 0.224, 0.225)


def wasb_csv_to_ball_track(
    csv_path: str | Path,
    *,
    fps: float,
    frame_times: Any = None,
    visible_threshold: float = DEFAULT_WASB_VISIBLE_THRESHOLD,
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
) -> dict[str, Any]:
    """Convert WASB ``Frame,Visibility,X,Y,Confidence`` rows into BallTrack JSON."""

    fps = _require_positive_float(fps, "fps")
    visible_threshold = _parse_confidence(visible_threshold, "visible_threshold")
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    rows = _read_wasb_rows(Path(csv_path))
    frames = []
    for row in sorted(rows, key=lambda item: item["frame"]):
        visible = bool(row["visible"] and row["confidence"] >= visible_threshold)
        frames.append(
            ball_frame(
                t=time_for_frame(int(row["frame"]), frame_times=frame_times, fps=fps),
                xy=[row["x"], row["y"]],
                conf=float(row["confidence"]),
                visible=visible,
                approx=False,
            )
        )
    payload = {
        "schema_version": 1,
        "fps": fps,
        "source": "wasb",
        "input_preprocessing": input_preprocessing,
        "frames": frames,
        "bounces": [],
    }
    BallTrack.model_validate(payload)
    return payload


def write_ball_track_from_wasb_predictions(
    *,
    predictions_csv: str | Path,
    fps: float,
    frame_times: Any = None,
    out: str | Path,
    metadata_out: str | Path | None = None,
    source_mode: str = "wasb_csv",
    runtime: dict[str, Any] | None = None,
    visible_threshold: float = DEFAULT_WASB_VISIBLE_THRESHOLD,
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_frames: dict[int, Sequence[dict[str, Any]]] | None = None,
    candidates_out: str | Path | None = None,
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
) -> dict[str, Any]:
    """Write ``ball_track.json`` plus WASB run metadata."""

    out_path = Path(out)
    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    visible_threshold = _parse_confidence(visible_threshold, "visible_threshold")
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    payload = wasb_csv_to_ball_track(
        predictions_csv,
        fps=fps,
        frame_times=frame_times,
        visible_threshold=visible_threshold,
        input_preprocessing=input_preprocessing,
    )
    _write_json(out_path, payload)
    candidate_path: Path | None = None
    if emit_candidates:
        if candidate_frames is None:
            raise ValueError("emit_candidates requires raw WASB blob candidate_frames")
        candidate_path = Path(candidates_out) if candidates_out is not None else _ball_candidates_sidecar_path(out_path)
        _write_ball_candidates_sidecar(
            path=candidate_path,
            source="wasb",
            source_mode=source_mode,
            fps=float(fps),
            primary_output=out_path,
            max_candidates_per_frame=candidate_top_k,
            nms_radius_px=None,
            frame_ids=[int(row["frame"]) for row in _read_wasb_rows(Path(predictions_csv))],
            candidate_frames=candidate_frames,
            default_source_detector="wasb_concomp",
            provenance={
                "predictions_csv": str(predictions_csv),
                "candidate_source": "provided_candidate_frames",
                "input_preprocessing": input_preprocessing,
            },
            input_preprocessing=input_preprocessing,
        )
    visible_count = sum(1 for frame in payload["frames"] if frame["visible"])
    runtime_payload = dict(runtime or {})
    _add_runtime_metrics(runtime_payload, processed_frame_count=len(payload["frames"]), fps=float(fps))
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_wasb_ball_run",
        "status": STATUS_TESTED,
        "source_mode": source_mode,
        "predictions_csv": str(predictions_csv),
        "out": str(out_path),
        "fps": float(fps),
        "frame_count": len(payload["frames"]),
        "visible_frame_count": visible_count,
        "confidence_semantics": WASB_CONFIDENCE_SEMANTICS,
        "visible_threshold": visible_threshold,
        "input_preprocessing": input_preprocessing,
        "non_promotable_measurement_mode": input_preprocessing in NON_PROMOTABLE_INPUT_PREPROCESSING_MODES,
        "not_ground_truth": True,
        "official_repo_url": WASB_REPO_URL,
        "official_model_zoo_url": WASB_MODEL_ZOO_URL,
        "runtime": runtime_payload,
    }
    runtime_candidates_out = runtime_payload.get("candidates_out")
    if candidate_path is not None:
        metadata["candidates_out"] = str(candidate_path)
        metadata["candidate_top_k"] = candidate_top_k
    elif isinstance(runtime_candidates_out, str):
        metadata["candidates_out"] = runtime_candidates_out
        if "candidate_top_k" in runtime_payload:
            metadata["candidate_top_k"] = runtime_payload["candidate_top_k"]
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def run_official_wasb_predict(
    *,
    wasb_repo: str | Path,
    checkpoint: str | Path,
    video: str | Path,
    out_csv: str | Path,
    batch_size: int = 8,
    visible_threshold: float = DEFAULT_WASB_VISIBLE_THRESHOLD,
    video_range: tuple[int, int] | list[int] | None = None,
    max_frames: int | None = None,
    device: str = "cuda",
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidates_out: str | Path | None = None,
    candidate_fps: float | None = None,
    primary_output: str | Path | None = None,
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
    emit_size_observations: bool = DEFAULT_EMIT_SIZE_OBSERVATIONS,
    size_observations_out: str | Path | None = None,
    size_observation_fps: float | None = None,
    size_observation_frame_times: Any = None,
    emit_below_threshold_candidates: bool = DEFAULT_EMIT_BELOW_THRESHOLD_CANDIDATES,
    below_threshold_candidates_out: str | Path | None = None,
    below_threshold_candidate_floor: float = DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR,
    below_threshold_candidate_fps: float | None = None,
    below_threshold_candidate_frame_times: Any = None,
) -> dict[str, Any]:
    """Run official WASB model code on a video and write per-frame predictions CSV."""

    repo = Path(wasb_repo).resolve()
    src = repo / "src"
    if not (src / "models" / "__init__.py").is_file():
        raise FileNotFoundError(f"missing WASB-SBDT official src/models in: {repo}")
    if not (src / "detectors" / "postprocessor.py").is_file():
        raise FileNotFoundError(f"missing WASB-SBDT official detector postprocessor in: {repo}")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing WASB checkpoint: {checkpoint_path}")
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video: {video_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    visible_threshold = _parse_confidence(visible_threshold, "visible_threshold")
    if emit_below_threshold_candidates:
        below_threshold_candidate_floor = _parse_below_threshold_candidate_floor(
            below_threshold_candidate_floor,
            acceptance_threshold=visible_threshold,
        )
    else:
        # The additive sidecar is default-off. Keep every legacy invocation valid,
        # including callers whose acceptance threshold is below the sidecar floor.
        below_threshold_candidate_floor = DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    normalized_range = _normalize_video_range(video_range)

    start = time.perf_counter()
    with _wasb_repo_imports(src):
        import cv2
        import numpy as np
        import torch

        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # type: ignore[attr-defined]

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("WASB official inference requires CUDA but torch.cuda.is_available() is false")
        torch_device = torch.device(device)
        cfg = _wasb_cfg(device=device)

        from detectors.postprocessor import TracknetV2Postprocessor
        from models import build_model
        from trackers.online import OnlineTracker
        model = build_model(cfg)
        checkpoint_payload = _load_wasb_checkpoint_payload(checkpoint_path, torch=torch)
        state_dict = _checkpoint_state_dict(checkpoint_payload)
        model.load_state_dict(_strip_module_prefix(state_dict))
        model = model.to(torch_device)
        model.eval()
        postprocessor = TracknetV2Postprocessor(cfg)
        tracker = OnlineTracker(cfg)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if width <= 0 or height <= 0:
                raise ValueError(f"could not read video dimensions: {video_path}")
            start_frame, end_frame = _resolve_frame_bounds(
                fps=source_fps,
                frame_count=frame_count,
                video_range=normalized_range,
                max_frames=max_frames,
            )
            if start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            trans_input, trans_output_inv = _wasb_official_input_affines(width, height, cv2=cv2, np=np)
            postprocess_affine_inv = _preprocessing_output_affine_inv(
                input_preprocessing=input_preprocessing,
                width=width,
                height=height,
                official_affine_inv=trans_output_inv,
                np=np,
            )

            det_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
            confidence_results: dict[int, list[float]] = defaultdict(list)
            size_observation_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
            below_threshold_candidate_results: dict[int, list[dict[str, Any]]] | None = (
                defaultdict(list) if emit_below_threshold_candidates else None
            )
            pending_tensors: list[Any] = []
            pending_indices: list[list[int]] = []
            window: deque[tuple[int, Any]] = deque(maxlen=WASB_FRAMES_IN)
            processed_windows = 0
            read_frames = 0
            frame_index = start_frame

            while frame_index < end_frame:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                window.append((frame_index, frame_rgb))
                read_frames += 1
                if len(window) == WASB_FRAMES_IN:
                    frame_indices = [item[0] for item in window]
                    frame_images = [item[1] for item in window]
                    pending_tensors.append(
                        _preprocess_wasb_window(
                            frame_images,
                            trans_input,
                            cv2=cv2,
                            np=np,
                            torch=torch,
                            input_preprocessing=input_preprocessing,
                        )
                    )
                    pending_indices.append(frame_indices)
                    if len(pending_tensors) >= batch_size:
                        processed_windows += _process_wasb_batch(
                            model=model,
                            postprocessor=postprocessor,
                            tensors=pending_tensors,
                            frame_indices=pending_indices,
                            affine_inv=postprocess_affine_inv,
                            det_results=det_results,
                            confidence_results=confidence_results,
                            size_observation_results=size_observation_results,
                            below_threshold_candidate_results=below_threshold_candidate_results,
                            below_threshold_candidate_floor=below_threshold_candidate_floor,
                            acceptance_threshold=visible_threshold,
                            cv2=cv2,
                            np=np,
                            torch=torch,
                            device=torch_device,
                        )
                        pending_tensors = []
                        pending_indices = []
                frame_index += 1

            if pending_tensors:
                processed_windows += _process_wasb_batch(
                    model=model,
                    postprocessor=postprocessor,
                    tensors=pending_tensors,
                    frame_indices=pending_indices,
                    affine_inv=postprocess_affine_inv,
                    det_results=det_results,
                    confidence_results=confidence_results,
                        size_observation_results=size_observation_results,
                        below_threshold_candidate_results=below_threshold_candidate_results,
                        below_threshold_candidate_floor=below_threshold_candidate_floor,
                        acceptance_threshold=visible_threshold,
                    cv2=cv2,
                    np=np,
                    torch=torch,
                    device=torch_device,
                )
        finally:
            cap.release()

        if processed_windows <= 0:
            raise ValueError(f"WASB inference needs at least {WASB_FRAMES_IN} readable frames: {video_path}")

        rows = _track_wasb_rows(
            tracker=tracker,
            det_results=det_results,
            confidence_results=confidence_results,
            visible_threshold=visible_threshold,
        )
        candidate_sidecar_path: Path | None = None
        if emit_candidates:
            candidate_sidecar_path = Path(candidates_out) if candidates_out is not None else _ball_candidates_sidecar_path(out_csv)
            sidecar_fps = _require_positive_float(
                candidate_fps if candidate_fps is not None else source_fps,
                "candidate_fps",
            )
            _write_ball_candidates_sidecar(
                path=candidate_sidecar_path,
                source="wasb",
                source_mode="wasb_predict",
                fps=sidecar_fps,
                primary_output=Path(primary_output) if primary_output is not None else Path(out_csv),
                max_candidates_per_frame=candidate_top_k,
                nms_radius_px=None,
                frame_ids=sorted(confidence_results),
                candidate_frames=det_results,
                default_source_detector="wasb_concomp",
                provenance={
                    "video": str(video_path),
                    "wasb_repo": str(repo),
                    "wasb_checkpoint": str(checkpoint_path),
                    "input_preprocessing": input_preprocessing,
                    "postprocessor": "TracknetV2Postprocessor",
                    "blob_det_method": "concomp",
                },
                input_preprocessing=input_preprocessing,
            )
        size_observation_sidecar_path: Path | None = None
        if emit_size_observations:
            size_observation_sidecar_path = (
                Path(size_observations_out)
                if size_observations_out is not None
                else _ball_size_observations_sidecar_path(primary_output if primary_output is not None else out_csv)
            )
            observation_fps = _require_positive_float(
                size_observation_fps if size_observation_fps is not None else source_fps,
                "size_observation_fps",
            )
            write_wasb_ball_size_observations(
                path=size_observation_sidecar_path,
                fps=observation_fps,
                frame_times=size_observation_frame_times,
                source_mode="wasb_predict",
                input_preprocessing=input_preprocessing,
                primary_output=Path(primary_output) if primary_output is not None else Path(out_csv),
                frame_ids=sorted(confidence_results),
                raw_frame_observations=size_observation_results,
                provenance={
                    "video": str(video_path),
                    "wasb_repo": str(repo),
                    "wasb_checkpoint": str(checkpoint_path),
                    "postprocessor": "TracknetV2Postprocessor",
                    "blob_det_method": "concomp",
                    "component_capture_point": "raw_heatmap_before_candidate_top_k",
                    "overlap_resolution": "highest_frame_heatmap_peak_then_earliest_observation",
                },
            )
        below_threshold_sidecar_path: Path | None = None
        if emit_below_threshold_candidates:
            assert below_threshold_candidate_results is not None
            below_threshold_sidecar_path = (
                Path(below_threshold_candidates_out)
                if below_threshold_candidates_out is not None
                else _below_threshold_candidates_sidecar_path(primary_output if primary_output is not None else out_csv)
            )
            below_threshold_fps = _require_positive_float(
                below_threshold_candidate_fps if below_threshold_candidate_fps is not None else source_fps,
                "below_threshold_candidate_fps",
            )
            _write_wasb_below_threshold_candidates_sidecar(
                path=below_threshold_sidecar_path,
                fps=below_threshold_fps,
                frame_times=below_threshold_candidate_frame_times,
                source_mode="wasb_predict",
                input_preprocessing=input_preprocessing,
                primary_output=Path(primary_output) if primary_output is not None else Path(out_csv),
                frame_ids=sorted(confidence_results),
                raw_frame_observations=below_threshold_candidate_results,
                candidate_score_floor=below_threshold_candidate_floor,
                acceptance_threshold=visible_threshold,
                provenance={
                    "video": str(video_path),
                    "wasb_repo": str(repo),
                    "wasb_checkpoint": str(checkpoint_path),
                    "component_capture_point": "raw_heatmap_before_primary_acceptance",
                    "overlap_resolution": "highest_frame_heatmap_peak_then_earliest_observation",
                },
            )

    out_path = Path(out_csv)
    _write_wasb_csv(out_path, rows)
    wall_seconds = time.perf_counter() - start
    runtime = {
        "wasb_repo": str(repo),
        "wasb_repo_commit": _git_commit(repo),
        "wasb_checkpoint": checkpoint_metadata(checkpoint_path),
        "video": str(video_path),
        "source_video_fps": source_fps,
        "source_video_frame_count": frame_count,
        "source_video_size": [width, height],
        "processed_frame_count": len(rows),
        "processed_window_count": processed_windows,
        "read_frame_count": read_frames,
        "video_range_seconds": list(normalized_range) if normalized_range is not None else None,
        "max_frames": max_frames,
        "batch_size": int(batch_size),
        "device": device,
        "input_preprocessing": input_preprocessing,
        "non_promotable_measurement_mode": input_preprocessing in NON_PROMOTABLE_INPUT_PREPROCESSING_MODES,
        "wall_seconds": wall_seconds,
    }
    if emit_candidates and candidate_sidecar_path is not None:
        runtime["candidates_out"] = str(candidate_sidecar_path)
        runtime["candidate_top_k"] = candidate_top_k
    if emit_size_observations and size_observation_sidecar_path is not None:
        runtime["size_observations_out"] = str(size_observation_sidecar_path)
        runtime["size_observation_heatmap_threshold"] = HEATMAP_THRESHOLD
    if emit_below_threshold_candidates and below_threshold_sidecar_path is not None:
        runtime["below_threshold_candidates_out"] = str(below_threshold_sidecar_path)
        runtime["below_threshold_candidate_floor"] = below_threshold_candidate_floor
        runtime["below_threshold_acceptance_threshold"] = visible_threshold
    return runtime


def run_wasb_or_convert(
    *,
    out: str | Path,
    fps: float,
    frame_times: Any = None,
    metadata_out: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    video: str | Path | None = None,
    checkpoint: str | Path | None = None,
    wasb_repo: str | Path | None = None,
    prediction_csv_out: str | Path | None = None,
    batch_size: int = 8,
    visible_threshold: float = DEFAULT_WASB_VISIBLE_THRESHOLD,
    video_range: tuple[int, int] | list[int] | None = None,
    max_frames: int | None = None,
    device: str = "cuda",
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
    emit_size_observations: bool = DEFAULT_EMIT_SIZE_OBSERVATIONS,
    emit_below_threshold_candidates: bool = DEFAULT_EMIT_BELOW_THRESHOLD_CANDIDATES,
    below_threshold_candidate_floor: float = DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR,
) -> dict[str, Any]:
    """CLI-oriented entrypoint that converts CSV or runs official WASB-SBDT."""

    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    visible_threshold = _parse_confidence(visible_threshold, "visible_threshold")
    if emit_below_threshold_candidates:
        below_threshold_candidate_floor = _parse_below_threshold_candidate_floor(
            below_threshold_candidate_floor,
            acceptance_threshold=visible_threshold,
        )
    else:
        below_threshold_candidate_floor = DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    if predictions_csv is not None:
        if emit_candidates:
            raise ValueError("WASB emit_candidates requires official inference; predictions CSV has only tracked argmax rows")
        if emit_below_threshold_candidates:
            raise ValueError(
                "WASB emit_below_threshold_candidates requires official inference; "
                "predictions CSV has no raw heatmaps"
            )
        return write_ball_track_from_wasb_predictions(
            predictions_csv=predictions_csv,
            fps=fps,
            frame_times=frame_times,
            out=out,
            metadata_out=metadata_out,
            source_mode="wasb_csv",
            visible_threshold=visible_threshold,
            input_preprocessing=input_preprocessing,
        )

    if video is None or checkpoint is None or wasb_repo is None:
        raise ValueError("either --predictions-csv or --video/--checkpoint/--wasb-repo is required")
    prediction_csv_path = Path(prediction_csv_out) if prediction_csv_out is not None else _persistent_prediction_csv_path(out)
    runtime = run_official_wasb_predict(
        wasb_repo=wasb_repo,
        checkpoint=checkpoint,
        video=video,
        out_csv=prediction_csv_path,
        batch_size=batch_size,
        visible_threshold=visible_threshold,
        video_range=video_range,
        max_frames=max_frames,
        device=device,
        emit_candidates=emit_candidates,
        candidate_top_k=candidate_top_k,
        candidates_out=_ball_candidates_sidecar_path(out) if emit_candidates else None,
        candidate_fps=fps,
        primary_output=out,
        input_preprocessing=input_preprocessing,
        emit_size_observations=emit_size_observations,
        size_observations_out=_ball_size_observations_sidecar_path(out),
        size_observation_fps=fps,
        size_observation_frame_times=frame_times,
        emit_below_threshold_candidates=emit_below_threshold_candidates,
        below_threshold_candidates_out=_below_threshold_candidates_sidecar_path(out),
        below_threshold_candidate_floor=below_threshold_candidate_floor,
        below_threshold_candidate_fps=fps,
        below_threshold_candidate_frame_times=frame_times,
    )
    return write_ball_track_from_wasb_predictions(
        predictions_csv=prediction_csv_path,
        fps=fps,
        frame_times=frame_times,
        out=out,
        metadata_out=metadata_out,
        source_mode="wasb_predict",
        runtime=runtime,
        visible_threshold=visible_threshold,
        input_preprocessing=input_preprocessing,
    )


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint)}


def _read_wasb_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing WASB predictions CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in WASB_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing WASB column(s): {', '.join(missing)}")
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    "frame": _parse_nonnegative_int(row["Frame"], f"Frame/{index}"),
                    "visible": _parse_visibility(row["Visibility"], f"Visibility/{index}"),
                    "x": _parse_float(row["X"], f"X/{index}"),
                    "y": _parse_float(row["Y"], f"Y/{index}"),
                    "confidence": _parse_confidence(row["Confidence"], f"Confidence/{index}"),
                }
            )
    if not rows:
        raise ValueError(f"WASB predictions CSV is empty: {csv_path}")
    return rows


def _preprocess_wasb_window(
    frames_rgb: Sequence[Any],
    trans_input: Any,
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
) -> Any:
    mode = _normalize_input_preprocessing(input_preprocessing)
    if mode == "official":
        return _preprocess_wasb_window_official(frames_rgb, trans_input, cv2=cv2, np=np, torch=torch)
    if mode == "harness_v0":
        return _preprocess_wasb_window_harness_v0(frames_rgb, np=np, torch=torch)
    raise AssertionError(f"unhandled WASB input preprocessing mode: {mode}")


def _preprocess_wasb_window_official(
    frames_rgb: Sequence[Any],
    trans_input: Any,
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    output_wh: tuple[int, int] = WASB_INPUT_WH,
) -> Any:
    tensors = []
    mean = np.asarray(WASB_IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(WASB_IMAGENET_STD, dtype=np.float32)
    output_wh = _normalize_output_wh(output_wh)
    for frame_rgb in frames_rgb:
        warped = cv2.warpAffine(frame_rgb, trans_input, output_wh, flags=cv2.INTER_LINEAR)
        array = warped.astype(np.float32) / 255.0
        array = (array - mean) / std
        tensors.append(torch.from_numpy(array.transpose(2, 0, 1)).float())
    return torch.cat(tensors, dim=0)


def _wasb_official_input_affines(
    width: int,
    height: int,
    *,
    cv2: Any,
    np: Any,
    output_wh: tuple[int, int] = WASB_INPUT_WH,
) -> tuple[Any, Any]:
    output_wh = _normalize_output_wh(output_wh)
    trans_input = _wasb_official_input_affine(width, height, cv2=cv2, np=np, output_wh=output_wh)
    trans_output_inv = _wasb_official_input_affine(width, height, cv2=cv2, np=np, output_wh=output_wh, inv=1)
    return trans_input, trans_output_inv


def _wasb_official_input_affine(
    width: int,
    height: int,
    *,
    cv2: Any,
    np: Any,
    output_wh: tuple[int, int] = WASB_INPUT_WH,
    inv: int = 0,
) -> Any:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"WASB affine requires positive source dimensions, got width={width} height={height}")
    center = np.array([float(width) / 2.0, float(height) / 2.0], dtype=np.float32)
    scale = np.array([float(max(int(height), int(width))), float(max(int(height), int(width)))], dtype=np.float32)
    return _wasb_affine_transform(center, scale, 0.0, _normalize_output_wh(output_wh), cv2=cv2, np=np, inv=inv)


def _wasb_affine_transform(
    center: Any,
    scale: Any,
    rot: float,
    output_wh: tuple[int, int],
    *,
    cv2: Any,
    np: Any,
    inv: int = 0,
) -> Any:
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w, dst_h = _normalize_output_wh(output_wh)
    rot_rad = np.pi * float(rot) / 180.0
    src_dir = _wasb_get_dir([0, src_w * -0.5], rot_rad, np=np)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = _wasb_get_3rd_point(src[0, :], src[1, :], np=np)
    dst[2:, :] = _wasb_get_3rd_point(dst[0, :], dst[1, :], np=np)
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _wasb_affine_transform_xy(xy: Sequence[float], affine: Any, *, np: Any) -> Any:
    if len(xy) != 2:
        raise ValueError(f"xy must contain exactly two values, got {xy}")
    point = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float32)
    transformed = np.dot(affine, point)
    return transformed[:2].astype(np.float32)


def _wasb_get_3rd_point(a: Any, b: Any, *, np: Any) -> Any:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _wasb_get_dir(src_point: Sequence[float], rot_rad: float, *, np: Any) -> Any:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array(
        [
            src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs,
        ],
        dtype=np.float32,
    )


def _normalize_output_wh(output_wh: tuple[int, int]) -> tuple[int, int]:
    width, height = int(output_wh[0]), int(output_wh[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"output_wh must contain positive width,height, got {output_wh}")
    return width, height


def _preprocess_wasb_window_harness_v0(frames_rgb: Sequence[Any], *, np: Any, torch: Any) -> Any:
    from PIL import Image

    tensors = []
    for frame_rgb in frames_rgb:
        rgb = Image.fromarray(frame_rgb).convert("RGB").resize(WASB_INPUT_WH, Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.cat(tensors, dim=0)


def _preprocessing_output_affine_inv(
    *,
    input_preprocessing: str,
    width: int,
    height: int,
    official_affine_inv: Any,
    np: Any,
) -> Any:
    mode = _normalize_input_preprocessing(input_preprocessing)
    if mode == "official":
        return official_affine_inv
    if mode == "harness_v0":
        return np.asarray(
            [
                [float(width) / float(WASB_INPUT_WH[0]), 0.0, 0.0],
                [0.0, float(height) / float(WASB_INPUT_WH[1]), 0.0],
            ],
            dtype=np.float32,
        )
    raise AssertionError(f"unhandled WASB input preprocessing mode: {mode}")


def _process_wasb_batch(
    *,
    model: Any,
    postprocessor: Any,
    tensors: list[Any],
    frame_indices: list[list[int]],
    affine_inv: Any,
    det_results: dict[int, list[dict[str, Any]]],
    confidence_results: dict[int, list[float]],
    size_observation_results: dict[int, list[dict[str, Any]]],
    below_threshold_candidate_results: dict[int, list[dict[str, Any]]] | None,
    below_threshold_candidate_floor: float,
    acceptance_threshold: float,
    cv2: Any,
    np: Any,
    torch: Any,
    device: Any,
) -> int:
    batch = torch.stack(tensors, dim=0).to(device)
    affine = torch.as_tensor(affine_inv, dtype=torch.float32).unsqueeze(0).repeat(len(tensors), 1, 1)
    with torch.no_grad():
        logits_by_scale = model(batch)
    logits = logits_by_scale[0]
    heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
    pp_results = postprocessor.run({0: logits.detach().clone()}, {0: affine})
    for batch_index, indices in enumerate(frame_indices):
        for output_index, frame_index in enumerate(indices):
            frame_heatmap = heatmaps[batch_index, output_index]
            confidence_results[int(frame_index)].append(float(frame_heatmap.max()))
            scale_results = pp_results[batch_index][output_index][0]
            size_observation_results[int(frame_index)].append(
                {
                    "heatmap_peak": float(frame_heatmap.max()),
                    "blobs": connected_component_blob_extents(
                        frame_heatmap,
                        affine_inv,
                        cv2=cv2,
                        np=np,
                        threshold=HEATMAP_THRESHOLD,
                    ),
                }
            )
            if below_threshold_candidate_results is not None:
                below_threshold_candidate_results[int(frame_index)].append(
                    {
                        "heatmap_peak": float(frame_heatmap.max()),
                        "candidates": _below_threshold_heatmap_candidates(
                            frame_heatmap,
                            affine_inv,
                            candidate_score_floor=below_threshold_candidate_floor,
                            acceptance_threshold=acceptance_threshold,
                            cv2=cv2,
                            np=np,
                        ),
                    }
                )
            for xy, score in zip(scale_results["xys"], scale_results["scores"]):
                det_results[int(frame_index)].append({"xy": xy, "score": float(score)})
    return len(tensors)


def _track_wasb_rows(
    *,
    tracker: Any,
    det_results: dict[int, list[dict[str, Any]]],
    confidence_results: dict[int, list[float]],
    visible_threshold: float,
) -> list[dict[str, Any]]:
    tracker.refresh()
    rows = []
    for frame_index in sorted(confidence_results):
        result = tracker.update(det_results.get(frame_index, []))
        confidence = max(confidence_results.get(frame_index, [0.0]))
        x = float(result["x"])
        y = float(result["y"])
        visible = bool(result["visi"] and math.isfinite(x) and math.isfinite(y) and confidence >= visible_threshold)
        rows.append(
            {
                "Frame": int(frame_index),
                "Visibility": int(visible),
                "X": x if visible else 0.0,
                "Y": y if visible else 0.0,
                "Confidence": confidence,
            }
        )
    return rows


def _write_ball_candidates_sidecar(
    *,
    path: Path,
    source: str,
    source_mode: str,
    fps: float,
    primary_output: str | Path,
    max_candidates_per_frame: int,
    nms_radius_px: float | None,
    frame_ids: Sequence[int],
    candidate_frames: dict[int, Sequence[dict[str, Any]]],
    default_source_detector: str,
    provenance: dict[str, Any],
    input_preprocessing: str = DEFAULT_WASB_INPUT_PREPROCESSING,
) -> None:
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    provenance = dict(provenance)
    provenance.setdefault("input_preprocessing", input_preprocessing)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_candidates",
        "fps": float(fps),
        "source": source,
        "source_mode": source_mode,
        "input_preprocessing": input_preprocessing,
        "primary_output": str(primary_output),
        "max_candidates_per_frame": int(max_candidates_per_frame),
        "nms_radius_px": nms_radius_px,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "provenance": provenance,
        "frames": [
            {
                "frame": int(frame),
                "candidates": _topk_candidate_blobs(
                    candidate_frames.get(int(frame), []),
                    top_k=max_candidates_per_frame,
                    default_source_detector=default_source_detector,
                ),
            }
            for frame in sorted({int(frame) for frame in frame_ids})
        ],
    }
    BallCandidates.model_validate(payload)
    _write_json(path, payload)


def _topk_candidate_blobs(
    blobs: Sequence[dict[str, Any]],
    *,
    top_k: int,
    default_source_detector: str,
) -> list[dict[str, Any]]:
    candidates = []
    for blob in blobs:
        xy = blob.get("xy")
        if hasattr(xy, "tolist"):
            xy = xy.tolist()
        if not isinstance(xy, Sequence) or len(xy) != 2:
            raise ValueError("candidate xy must contain exactly two numbers")
        candidates.append(
            {
                "xy": [_parse_float(xy[0], "candidate.x"), _parse_float(xy[1], "candidate.y")],
                "score": _saturate_candidate_score(blob.get("score", 0.0), "candidate.score"),
                "source_detector": str(blob.get("source_detector") or default_source_detector),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["xy"][0]), float(item["xy"][1])))
    return candidates[:top_k]


def _below_threshold_heatmap_candidates(
    heatmap: Any,
    affine_inv: Any,
    *,
    candidate_score_floor: float,
    acceptance_threshold: float,
    cv2: Any,
    np: Any,
) -> list[dict[str, Any]]:
    floor = _parse_below_threshold_candidate_floor(
        candidate_score_floor,
        acceptance_threshold=acceptance_threshold,
    )
    blobs = connected_component_blob_extents(
        heatmap,
        affine_inv,
        cv2=cv2,
        np=np,
        threshold=math.nextafter(floor, -math.inf),
    )
    candidates = [
        {
            "xy": [float(blob["center_xy_px"][0]), float(blob["center_xy_px"][1])],
            "score": float(blob["heatmap_peak"]),
            "source_detector": "wasb_concomp_below_acceptance",
        }
        for blob in blobs
        if floor <= float(blob["heatmap_peak"]) < float(acceptance_threshold)
    ]
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["xy"][0]), float(item["xy"][1])))
    return candidates


def _write_wasb_below_threshold_candidates_sidecar(
    *,
    path: str | Path,
    fps: float,
    frame_times: Any,
    source_mode: str,
    input_preprocessing: str,
    primary_output: str | Path,
    frame_ids: Sequence[int],
    raw_frame_observations: Mapping[int, Sequence[Mapping[str, Any]]],
    candidate_score_floor: float,
    acceptance_threshold: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fps = _require_positive_float(fps, "fps")
    input_preprocessing = _normalize_input_preprocessing(input_preprocessing)
    floor = _parse_below_threshold_candidate_floor(
        candidate_score_floor,
        acceptance_threshold=acceptance_threshold,
    )
    frames: list[dict[str, Any]] = []
    candidate_count = 0
    for frame in sorted({int(value) for value in frame_ids}):
        observations = list(raw_frame_observations.get(frame, []))
        if not observations:
            raise ValueError(f"missing raw WASB below-threshold observation for frame {frame}")
        selected_index, selected = min(
            enumerate(observations),
            key=lambda item: (-float(item[1]["heatmap_peak"]), int(item[0])),
        )
        candidates = []
        for raw_candidate in selected.get("candidates", []):
            xy = raw_candidate.get("xy")
            if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) != 2:
                raise ValueError("below-threshold candidate xy must contain exactly two numbers")
            score = _saturate_candidate_score(raw_candidate.get("score"), "below_threshold_candidate.score")
            if score < floor or score >= acceptance_threshold:
                raise ValueError("below-threshold candidate score must be within [floor, acceptance_threshold)")
            candidates.append(
                {
                    "xy": [
                        _parse_float(xy[0], "below_threshold_candidate.x"),
                        _parse_float(xy[1], "below_threshold_candidate.y"),
                    ],
                    "score": score,
                    "source_detector": str(
                        raw_candidate.get("source_detector") or "wasb_concomp_below_acceptance"
                    ),
                }
            )
        candidates.sort(
            key=lambda item: (-float(item["score"]), float(item["xy"][0]), float(item["xy"][1]))
        )
        candidate_count += len(candidates)
        frames.append(
            {
                "frame": frame,
                "pts_seconds": time_for_frame(frame, frame_times=frame_times, fps=fps),
                "heatmap_observation_count": len(observations),
                "selected_heatmap_observation_index": selected_index,
                "candidates": candidates,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_wasb_below_threshold_candidates",
        "fps": fps,
        "source": "wasb",
        "source_mode": str(source_mode),
        "input_preprocessing": input_preprocessing,
        "primary_output": str(primary_output),
        "coordinate_space": "source_pixels",
        "candidate_score_floor": floor,
        "acceptance_threshold": float(acceptance_threshold),
        "threshold_interval": "floor_inclusive_acceptance_exclusive",
        "not_ground_truth": True,
        "candidate_prediction": True,
        "default_emission": False,
        "provenance": dict(provenance),
        "summary": {
            "frame_count": len(frames),
            "frame_with_candidates_count": sum(1 for item in frames if item["candidates"]),
            "candidate_count": candidate_count,
        },
        "frames": frames,
    }
    _write_json(Path(path), payload)
    return payload


def _write_wasb_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("WASB inference produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WASB_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["Frame"])):
            writer.writerow(
                {
                    "Frame": int(row["Frame"]),
                    "Visibility": int(row["Visibility"]),
                    "X": f"{float(row['X']):.6f}",
                    "Y": f"{float(row['Y']):.6f}",
                    "Confidence": f"{float(row['Confidence']):.8f}",
                }
            )


def _wasb_cfg(*, device: str) -> Any:
    return _attrdict(
        {
            "model": {
                "name": "hrnet",
                "frames_in": WASB_FRAMES_IN,
                "frames_out": WASB_FRAMES_OUT,
                "inp_height": WASB_INPUT_WH[1],
                "inp_width": WASB_INPUT_WH[0],
                "out_height": WASB_INPUT_WH[1],
                "out_width": WASB_INPUT_WH[0],
                "rgb_diff": False,
                "out_scales": [0],
                "MODEL": {
                    "EXTRA": {
                        "FINAL_CONV_KERNEL": 1,
                        "PRETRAINED_LAYERS": ["*"],
                        "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
                        "STAGE1": {
                            "NUM_MODULES": 1,
                            "NUM_BRANCHES": 1,
                            "BLOCK": "BOTTLENECK",
                            "NUM_BLOCKS": [1],
                            "NUM_CHANNELS": [32],
                            "FUSE_METHOD": "SUM",
                        },
                        "STAGE2": {
                            "NUM_MODULES": 1,
                            "NUM_BRANCHES": 2,
                            "BLOCK": "BASIC",
                            "NUM_BLOCKS": [2, 2],
                            "NUM_CHANNELS": [16, 32],
                            "FUSE_METHOD": "SUM",
                        },
                        "STAGE3": {
                            "NUM_MODULES": 1,
                            "NUM_BRANCHES": 3,
                            "BLOCK": "BASIC",
                            "NUM_BLOCKS": [2, 2, 2],
                            "NUM_CHANNELS": [16, 32, 64],
                            "FUSE_METHOD": "SUM",
                        },
                        "STAGE4": {
                            "NUM_MODULES": 1,
                            "NUM_BRANCHES": 4,
                            "BLOCK": "BASIC",
                            "NUM_BLOCKS": [2, 2, 2, 2],
                            "NUM_CHANNELS": [16, 32, 64, 128],
                            "FUSE_METHOD": "SUM",
                        },
                        "DECONV": {
                            "NUM_DECONVS": 0,
                            "KERNEL_SIZE": [],
                            "NUM_BASIC_BLOCKS": 2,
                        },
                    },
                    "INIT_WEIGHTS": True,
                },
            },
            "detector": {
                "name": "tracknetv2",
                "model_path": None,
                "step": WASB_FRAMES_IN,
                "postprocessor": {
                    "name": "tracknetv2",
                    "score_threshold": 0.5,
                    "scales": [0],
                    "blob_det_method": "concomp",
                    "use_hm_weight": True,
                },
            },
            "dataloader": {"heatmap": {"sigmas": [2.5]}},
            "runner": {"device": device, "gpus": [0]},
            "tracker": {"name": "online", "max_disp": 300},
        }
    )


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _attrdict(child) for key, child in value.items()})
    if isinstance(value, list):
        return [_attrdict(child) for child in value]
    return value


def _checkpoint_state_dict(checkpoint_payload: Any) -> dict[str, Any]:
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("WASB checkpoint must be a dictionary")
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    if "state_dict" in checkpoint_payload:
        return checkpoint_payload["state_dict"]
    raise ValueError("WASB checkpoint missing model_state_dict")


def _load_wasb_checkpoint_payload(checkpoint_path: str | Path, *, torch: Any) -> Any:
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if not _is_pathlib_weights_only_error(exc):
            raise
        safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
        if safe_globals is None:
            return torch.load(path, map_location="cpu", weights_only=False)
        with safe_globals(_torch_checkpoint_safe_globals()):
            return torch.load(path, map_location="cpu", weights_only=True)


def _is_pathlib_weights_only_error(exc: Exception) -> bool:
    message = str(exc)
    return "Weights only load failed" in message and "pathlib" in message


def _torch_checkpoint_safe_globals() -> list[type[Any]]:
    return [
        pathlib.PosixPath,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.WindowsPath,
    ]


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _resolve_frame_bounds(
    *,
    fps: float,
    frame_count: int,
    video_range: tuple[int, int] | None,
    max_frames: int | None,
) -> tuple[int, int]:
    if video_range is None:
        start_frame = 0
        end_frame = frame_count if frame_count > 0 else 10**12
    else:
        if fps <= 0:
            raise ValueError("video_range requires readable source FPS")
        start_frame = int(round(video_range[0] * fps))
        end_frame = int(round(video_range[1] * fps))
        if frame_count > 0:
            end_frame = min(end_frame, frame_count)
    if max_frames is not None:
        if max_frames < WASB_FRAMES_IN:
            raise ValueError(f"max_frames must be at least {WASB_FRAMES_IN}")
        end_frame = min(end_frame, start_frame + int(max_frames))
    if end_frame <= start_frame:
        raise ValueError("resolved video frame range is empty")
    return start_frame, end_frame


def _normalize_video_range(video_range: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if video_range is None:
        return None
    if len(video_range) != 2:
        raise ValueError("video_range must contain START_S and END_S")
    start, end = int(video_range[0]), int(video_range[1])
    if start < 0 or end <= start:
        raise ValueError("video_range must satisfy 0 <= START_S < END_S")
    return start, end


def _parse_visibility(value: object, name: str) -> bool:
    number = _parse_float(value, name)
    if number not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0 or 1")
    return number == 1.0


def _parse_confidence(value: object, name: str) -> float:
    number = _parse_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return number


def _parse_below_threshold_candidate_floor(value: object, *, acceptance_threshold: float) -> float:
    floor = _parse_confidence(value, "below_threshold_candidate_floor")
    acceptance = _parse_confidence(acceptance_threshold, "acceptance_threshold")
    if floor >= acceptance:
        raise ValueError("below_threshold_candidate_floor must be below the acceptance threshold")
    return floor


def _normalize_input_preprocessing(value: object) -> str:
    mode = str(value or DEFAULT_WASB_INPUT_PREPROCESSING)
    if mode not in WASB_INPUT_PREPROCESSING_MODES:
        choices = ", ".join(WASB_INPUT_PREPROCESSING_MODES)
        raise ValueError(f"input_preprocessing must be one of: {choices}")
    return mode


def _saturate_candidate_score(value: object, name: str) -> float:
    number = _parse_float(value, name)
    return min(1.0, max(0.0, number))


def _parse_nonnegative_int(value: object, name: str) -> int:
    number = _parse_float(value, name)
    if int(number) != number or number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _require_positive_int(value: object, name: str) -> int:
    number = _parse_nonnegative_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_positive_float(value: object, name: str) -> float:
    number = _parse_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _parse_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _persistent_prediction_csv_path(out: str | Path) -> Path:
    out_path = Path(out)
    return out_path.with_name(f"{out_path.stem}_wasb_predictions.csv")


def _ball_candidates_sidecar_path(out: str | Path) -> Path:
    return Path(out).with_name("ball_candidates.json")


def _ball_size_observations_sidecar_path(out: str | Path) -> Path:
    return Path(out).with_name("ball_size_observations.json")


def _below_threshold_candidates_sidecar_path(out: str | Path) -> Path:
    return Path(out).with_name("ball_candidates_below_threshold.json")


def _add_runtime_metrics(runtime: dict[str, Any], *, processed_frame_count: int, fps: float) -> None:
    wall = runtime.get("wall_seconds")
    if not isinstance(wall, (int, float)) or wall <= 0:
        runtime["effective_fps"] = None
        runtime["realtime_factor"] = None
        return
    effective_fps = int(processed_frame_count) / float(wall)
    runtime["effective_fps"] = effective_fps
    runtime["realtime_factor"] = effective_fps / float(fps)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _wasb_repo_imports(src: Path) -> Iterator[None]:
    src_text = str(src)
    sys.path.insert(0, src_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(src_text)
        except ValueError:
            pass
        for name, module in list(sys.modules.items()):
            module_file = getattr(module, "__file__", None)
            if not module_file:
                continue
            try:
                module_path = Path(module_file).resolve()
            except OSError:
                continue
            if module_path.is_relative_to(src):
                sys.modules.pop(name, None)


__all__ = [
    "DEFAULT_BALL_CANDIDATE_TOP_K",
    "DEFAULT_BELOW_THRESHOLD_CANDIDATE_FLOOR",
    "DEFAULT_EMIT_BELOW_THRESHOLD_CANDIDATES",
    "DEFAULT_EMIT_SIZE_OBSERVATIONS",
    "DEFAULT_WASB_VISIBLE_THRESHOLD",
    "WASB_CONFIDENCE_SEMANTICS",
    "checkpoint_metadata",
    "run_official_wasb_predict",
    "run_wasb_or_convert",
    "wasb_csv_to_ball_track",
    "write_ball_track_from_wasb_predictions",
]
