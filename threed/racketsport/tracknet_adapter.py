"""TrackNetV3 prediction adapters for schema-valid ball tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from .ball_tracknet import ball_frame
from .io_decode import time_for_frame
from .schemas import BallCandidates, BallTrack

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except Exception:  # pragma: no cover - TrackNet runtime imports torch before using this guard.
    _TorchIterableDataset = object  # type: ignore[assignment,misc]


TRACKNET_COLUMNS = ("Frame", "Visibility", "X", "Y")
TRACKNET_CONFIDENCE_COLUMNS = (*TRACKNET_COLUMNS, "Confidence")
TrackNetConfidenceMode = Literal["legacy_visibility", "heatmap_peak"]
TrackNetHeatmapEvalMode = Literal["nonoverlap", "average", "weight"]
LEGACY_CONFIDENCE_SEMANTICS = "official visibility mapped to conf 1.0/0.0"
HEATMAP_CONFIDENCE_SEMANTICS = "TrackNet heatmap peak value (0..1)"
CONFIDENCE_SEMANTICS = LEGACY_CONFIDENCE_SEMANTICS
DEFAULT_HEATMAP_VISIBLE_THRESHOLD = 0.5
DEFAULT_BALL_CANDIDATE_TOP_K = 5
DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX = 10.0


def tracknet_csv_to_ball_track(
    csv_path: str | Path,
    *,
    fps: float,
    frame_times: Any = None,
    confidence_mode: TrackNetConfidenceMode = "legacy_visibility",
    heatmap_visible_threshold: float = DEFAULT_HEATMAP_VISIBLE_THRESHOLD,
) -> dict[str, Any]:
    """Convert official TrackNetV3 ``Frame,Visibility,X,Y`` CSV into BallTrack JSON."""

    fps = _require_positive_float(fps, "fps")
    heatmap_visible_threshold = _parse_confidence(heatmap_visible_threshold, "heatmap_visible_threshold")
    rows = _read_tracknet_rows(Path(csv_path), confidence_mode=confidence_mode)
    frames = [
        ball_frame(
            t=time_for_frame(int(row["frame"]), frame_times=frame_times, fps=fps),
            xy=[row["x"], row["y"]],
            conf=_row_confidence(row, confidence_mode=confidence_mode),
            visible=_row_visible(
                row,
                confidence_mode=confidence_mode,
                heatmap_visible_threshold=heatmap_visible_threshold,
            ),
            approx=False,
        )
        for row in sorted(rows, key=lambda item: item["frame"])
    ]
    payload = {"schema_version": 1, "fps": fps, "source": "tracknet", "frames": frames, "bounces": []}
    BallTrack.model_validate(payload)
    return payload


def write_ball_track_from_csv(
    *,
    predictions_csv: str | Path,
    fps: float,
    frame_times: Any = None,
    out: str | Path,
    metadata_out: str | Path | None = None,
    source_mode: str = "tracknet_csv",
    runtime: dict[str, Any] | None = None,
    confidence_mode: TrackNetConfidenceMode = "legacy_visibility",
    heatmap_visible_threshold: float = DEFAULT_HEATMAP_VISIBLE_THRESHOLD,
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_frames: dict[int, Sequence[dict[str, Any]]] | None = None,
    candidates_out: str | Path | None = None,
    candidate_nms_radius_px: float | None = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
) -> dict[str, Any]:
    """Write ``ball_track.json`` and a sidecar run summary from official predictions."""

    out_path = Path(out)
    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    heatmap_visible_threshold = _parse_confidence(heatmap_visible_threshold, "heatmap_visible_threshold")
    payload = tracknet_csv_to_ball_track(
        predictions_csv,
        fps=fps,
        frame_times=frame_times,
        confidence_mode=confidence_mode,
        heatmap_visible_threshold=heatmap_visible_threshold,
    )
    _write_json(out_path, payload)
    candidate_path: Path | None = None
    if emit_candidates:
        if candidate_frames is None:
            raise ValueError("emit_candidates requires raw TrackNet heatmap candidate_frames")
        candidate_path = Path(candidates_out) if candidates_out is not None else _ball_candidates_sidecar_path(out_path)
        _write_ball_candidates_sidecar(
            path=candidate_path,
            source="tracknet",
            source_mode=source_mode,
            fps=float(fps),
            primary_output=out_path,
            max_candidates_per_frame=candidate_top_k,
            nms_radius_px=candidate_nms_radius_px,
            frame_ids=[int(row["frame"]) for row in _read_tracknet_rows(Path(predictions_csv), confidence_mode=confidence_mode)],
            candidate_frames=candidate_frames,
            default_source_detector="tracknet_heatmap_nms",
            provenance={"predictions_csv": str(predictions_csv), "candidate_source": "provided_candidate_frames"},
        )
    visible = sum(1 for frame in payload["frames"] if frame["visible"])
    runtime_payload = dict(runtime or {})
    _add_runtime_metrics(runtime_payload, processed_frame_count=len(payload["frames"]), fps=float(fps))
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_tracknet_ball_run",
        "source_mode": source_mode,
        "predictions_csv": str(predictions_csv),
        "out": str(out_path),
        "fps": float(fps),
        "frame_count": len(payload["frames"]),
        "visible_frame_count": visible,
        "confidence_semantics": _confidence_semantics(confidence_mode),
        "confidence_mode": confidence_mode,
        "heatmap_visible_threshold": heatmap_visible_threshold if confidence_mode == "heatmap_peak" else None,
        "runtime": runtime_payload,
        "not_ground_truth": True,
    }
    runtime_candidates_out = runtime_payload.get("candidates_out")
    if candidate_path is not None:
        metadata["candidates_out"] = str(candidate_path)
        metadata["candidate_top_k"] = candidate_top_k
        metadata["candidate_nms_radius_px"] = candidate_nms_radius_px
    elif isinstance(runtime_candidates_out, str):
        metadata["candidates_out"] = runtime_candidates_out
        if "candidate_top_k" in runtime_payload:
            metadata["candidate_top_k"] = runtime_payload["candidate_top_k"]
        if "candidate_nms_radius_px" in runtime_payload:
            metadata["candidate_nms_radius_px"] = runtime_payload["candidate_nms_radius_px"]
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def run_official_tracknet_predict(
    *,
    tracknet_repo: str | Path,
    video: str | Path,
    tracknet_file: str | Path,
    inpaintnet_file: str | Path,
    save_dir: str | Path,
    batch_size: int = 16,
    video_range: tuple[int, int] | list[int] | None = None,
    large_video: bool = True,
) -> Path:
    """Run official TrackNetV3 ``predict.py`` and return its CSV path."""

    repo = Path(tracknet_repo).resolve()
    predict_py = repo / "predict.py"
    if not predict_py.is_file():
        raise FileNotFoundError(f"missing TrackNetV3 predict.py: {predict_py}")
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video: {video_path}")
    tracknet_path = Path(tracknet_file).resolve()
    if not tracknet_path.is_file():
        raise FileNotFoundError(f"missing TrackNet checkpoint: {tracknet_path}")
    inpaintnet_path = Path(inpaintnet_file).resolve()
    if not inpaintnet_path.is_file():
        raise FileNotFoundError(f"missing InpaintNet checkpoint: {inpaintnet_path}")

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    normalized_video_range = _normalize_video_range(video_range)
    with _portable_tracknet_predict_repo(repo) as runtime_repo:
        runtime_predict_py = runtime_repo / "predict.py"
        cmd = [
            sys.executable,
            str(runtime_predict_py),
            "--video_file",
            str(video_path),
            "--tracknet_file",
            str(tracknet_path),
            "--inpaintnet_file",
            str(inpaintnet_path),
            "--save_dir",
            str(save_path),
            "--batch_size",
            str(batch_size),
        ]
        if normalized_video_range is not None:
            cmd.extend(["--video_range", f"{normalized_video_range[0]},{normalized_video_range[1]}"])
        if large_video:
            cmd.append("--large_video")
        subprocess.run(cmd, cwd=runtime_repo, check=True)
    csv_path = save_path / f"{video_path.stem}_ball.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"TrackNetV3 completed without writing predictions CSV: {csv_path}")
    return csv_path


def run_tracknet_heatmap_confidence_predict(
    *,
    tracknet_repo: str | Path,
    video: str | Path,
    tracknet_file: str | Path,
    save_dir: str | Path,
    batch_size: int = 16,
    video_range: tuple[int, int] | list[int] | None = None,
    large_video: bool = True,
    eval_mode: Literal["nonoverlap", "average", "weight"] = "weight",
    max_sample_num: int = 1800,
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_nms_radius_px: float = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
    candidates_out: str | Path | None = None,
    candidate_fps: float | None = None,
    primary_output: str | Path | None = None,
) -> Path:
    """Run TrackNet and write per-frame heatmap peak confidence values.

    The official ``predict.py`` writes only ``Frame,Visibility,X,Y`` after thresholding
    and optional InpaintNet coordinate refinement. This first-party companion pass uses
    the official model/dataset code and preserves the raw heatmap peak before thresholding
    so BallTrack ``conf`` can remain a real model probability-like signal.
    """

    repo = Path(tracknet_repo).resolve()
    if not (repo / "model.py").is_file() or not (repo / "dataset.py").is_file():
        raise FileNotFoundError(f"missing TrackNetV3 runtime files in: {repo}")
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video: {video_path}")
    tracknet_path = Path(tracknet_file).resolve()
    if not tracknet_path.is_file():
        raise FileNotFoundError(f"missing TrackNet checkpoint: {tracknet_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    candidate_nms_radius_px = _require_nonnegative_float(candidate_nms_radius_px, "candidate_nms_radius_px")
    normalized_video_range = _normalize_video_range(video_range)

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    out_csv = save_path / f"{video_path.stem}_ball_heatmap_confidence.csv"

    with _tracknet_repo_imports(repo):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
        from utils.general import HEIGHT, WIDTH, generate_frames, get_model

        device = _torch_device(torch)
        tracknet_ckpt = torch.load(tracknet_path, map_location=device)
        tracknet_seq_len = tracknet_ckpt["param_dict"]["seq_len"]
        bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]
        tracknet = get_model("TrackNet", tracknet_seq_len, bg_mode).to(device)
        tracknet.load_state_dict(tracknet_ckpt["model"])
        tracknet.eval()

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"could not read video dimensions: {video_path}")
        img_scaler = (width / WIDTH, height / HEIGHT)
        candidate_frames: dict[int, list[dict[str, Any]]] | None = {} if emit_candidates else None

        seq_len = int(tracknet_seq_len)
        if eval_mode == "nonoverlap":
            if large_video:
                dataset = Video_IterableDataset(
                    str(video_path),
                    seq_len=seq_len,
                    sliding_step=seq_len,
                    bg_mode=bg_mode,
                    max_sample_num=max_sample_num,
                    video_range=normalized_video_range,
                )
                dataset = _TrackNetVideoIterableDatasetEofGuard(dataset)
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                expected_frame_count = int(dataset.video_len)
            else:
                frame_list = generate_frames(str(video_path))
                if not frame_list:
                    raise ValueError(f"video contains no readable frames: {video_path}")
                dataset = Shuttlecock_Trajectory_Dataset(
                    seq_len=seq_len,
                    sliding_step=seq_len,
                    data_mode="heatmap",
                    bg_mode=bg_mode,
                    frame_arr=np.array(frame_list)[:, :, :, ::-1],
                    padding=True,
                )
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
                expected_frame_count = len(frame_list)

            rows = _run_nonoverlap_heatmap_confidence(
                data_loader=data_loader,
                tracknet=tracknet,
                torch=torch,
                device=device,
                img_scaler=img_scaler,
                expected_frame_count=expected_frame_count,
                candidate_frames=candidate_frames,
                candidate_top_k=candidate_top_k,
                candidate_nms_radius_px=candidate_nms_radius_px,
            )
        else:
            if large_video:
                dataset = Video_IterableDataset(
                    str(video_path),
                    seq_len=seq_len,
                    sliding_step=1,
                    bg_mode=bg_mode,
                    max_sample_num=max_sample_num,
                    video_range=normalized_video_range,
                )
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                video_len = int(dataset.video_len)
            else:
                frame_list = generate_frames(str(video_path))
                if not frame_list:
                    raise ValueError(f"video contains no readable frames: {video_path}")
                dataset = Shuttlecock_Trajectory_Dataset(
                    seq_len=seq_len,
                    sliding_step=1,
                    data_mode="heatmap",
                    bg_mode=bg_mode,
                    frame_arr=np.array(frame_list)[:, :, :, ::-1],
                )
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
                video_len = len(frame_list)
            rows = _run_overlap_heatmap_confidence(
                data_loader=data_loader,
                tracknet=tracknet,
                torch=torch,
                device=device,
                video_len=video_len,
                seq_len=seq_len,
                eval_mode=eval_mode,
                img_scaler=img_scaler,
                candidate_frames=candidate_frames,
                candidate_top_k=candidate_top_k,
                candidate_nms_radius_px=candidate_nms_radius_px,
            )
        if emit_candidates:
            assert candidate_frames is not None
            candidate_sidecar_path = Path(candidates_out) if candidates_out is not None else _ball_candidates_sidecar_path(out_csv)
            sidecar_fps = _require_positive_float(
                candidate_fps if candidate_fps is not None else source_fps,
                "candidate_fps",
            )
            _write_ball_candidates_sidecar(
                path=candidate_sidecar_path,
                source="tracknet",
                source_mode=f"tracknet_heatmap_{eval_mode}",
                fps=sidecar_fps,
                primary_output=Path(primary_output) if primary_output is not None else out_csv,
                max_candidates_per_frame=candidate_top_k,
                nms_radius_px=candidate_nms_radius_px,
                frame_ids=sorted(row["Frame"] for row in rows),
                candidate_frames=candidate_frames,
                default_source_detector="tracknet_heatmap_nms",
                provenance={
                    "video": str(video_path),
                    "tracknet_repo": str(repo),
                    "tracknet_checkpoint": str(tracknet_path),
                    "eval_mode": eval_mode,
                },
            )

    _write_confidence_csv(out_csv, rows)
    return out_csv


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint)}


def _read_tracknet_rows(csv_path: Path, *, confidence_mode: TrackNetConfidenceMode) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing TrackNet predictions CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = TRACKNET_CONFIDENCE_COLUMNS if confidence_mode == "heatmap_peak" else TRACKNET_COLUMNS
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing TrackNet column(s): {', '.join(missing)}")
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    "frame": _parse_nonnegative_int(row["Frame"], f"Frame/{index}"),
                    "visible": _parse_visibility(row["Visibility"], f"Visibility/{index}"),
                    "x": _parse_float(row["X"], f"X/{index}"),
                    "y": _parse_float(row["Y"], f"Y/{index}"),
                    "confidence": (
                        _parse_confidence(row["Confidence"], f"Confidence/{index}")
                        if confidence_mode == "heatmap_peak"
                        else None
                    ),
                }
            )
    if not rows:
        raise ValueError(f"TrackNet predictions CSV is empty: {csv_path}")
    return rows


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


def _saturate_model_confidence(value: object, name: str) -> float:
    number = _parse_float(value, name)
    return min(1.0, max(0.0, number))


def _row_confidence(row: dict[str, Any], *, confidence_mode: TrackNetConfidenceMode) -> float:
    if confidence_mode == "heatmap_peak":
        confidence = row.get("confidence")
        if confidence is None:
            raise ValueError("heatmap_peak confidence mode requires Confidence values")
        return float(confidence)
    return 1.0 if row["visible"] else 0.0


def _row_visible(
    row: dict[str, Any],
    *,
    confidence_mode: TrackNetConfidenceMode,
    heatmap_visible_threshold: float,
) -> bool:
    if confidence_mode == "heatmap_peak":
        return bool(row["visible"] and _row_confidence(row, confidence_mode=confidence_mode) >= heatmap_visible_threshold)
    return bool(row["visible"])


def _confidence_semantics(confidence_mode: TrackNetConfidenceMode) -> str:
    if confidence_mode == "heatmap_peak":
        return HEATMAP_CONFIDENCE_SEMANTICS
    return LEGACY_CONFIDENCE_SEMANTICS


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


def _require_nonnegative_float(value: object, name: str) -> float:
    number = _parse_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _parse_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_confidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("TrackNet heatmap confidence run produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACKNET_CONFIDENCE_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["Frame"])):
            writer.writerow(
                {
                    "Frame": int(row["Frame"]),
                    "Visibility": int(row["Visibility"]),
                    "X": int(row["X"]),
                    "Y": int(row["Y"]),
                    "Confidence": f"{_saturate_model_confidence(row['Confidence'], 'Confidence'):.8f}",
                }
            )


def _heatmap_prediction_rows(
    indices: Any,
    y_pred: Any,
    *,
    img_scaler: tuple[float, float],
    seen_frames: set[int],
    candidate_frames: dict[int, list[dict[str, Any]]] | None = None,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_nms_radius_px: float = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
) -> list[dict[str, Any]]:
    import numpy as np

    indices_np = indices.detach().cpu().numpy() if hasattr(indices, "detach") else np.asarray(indices)
    heatmaps = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else np.asarray(y_pred)
    rows: list[dict[str, Any]] = []
    batch_size, seq_len = indices_np.shape[0], indices_np.shape[1]
    for n in range(batch_size):
        prev_frame = -1
        for f in range(seq_len):
            frame = int(indices_np[n][f][1])
            if frame == prev_frame:
                break
            prev_frame = frame
            if frame in seen_frames:
                continue
            seen_frames.add(frame)
            heatmap = heatmaps[n][f]
            if candidate_frames is not None:
                candidate_frames[frame] = _topk_heatmap_local_maxima(
                    heatmap,
                    top_k=candidate_top_k,
                    nms_radius_px=candidate_nms_radius_px,
                    img_scaler=img_scaler,
                )
            confidence = _saturate_model_confidence(np.max(heatmap), "Confidence")
            if confidence >= 0.5:
                y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
                visibility = 1
                out_x = int(x * img_scaler[0])
                out_y = int(y * img_scaler[1])
            else:
                visibility = 0
                out_x = 0
                out_y = 0
            rows.append(
                {
                    "Frame": frame,
                    "Visibility": visibility,
                    "X": out_x,
                    "Y": out_y,
                    "Confidence": confidence,
                }
            )
    return rows


def _topk_heatmap_local_maxima(
    heatmap: Any,
    *,
    top_k: int,
    nms_radius_px: float,
    img_scaler: tuple[float, float],
) -> list[dict[str, Any]]:
    import numpy as np

    top_k = _require_positive_int(top_k, "top_k")
    nms_radius_px = _require_nonnegative_float(nms_radius_px, "nms_radius_px")
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    if img_scaler[0] <= 0.0 or img_scaler[1] <= 0.0:
        raise ValueError("img_scaler values must be positive")

    working = np.where(np.isfinite(array), array, -np.inf).copy()
    y_grid, x_grid = np.ogrid[: working.shape[0], : working.shape[1]]
    candidates: list[dict[str, Any]] = []
    for _ in range(top_k):
        flat_index = int(np.argmax(working))
        raw_score = float(working.flat[flat_index])
        if not math.isfinite(raw_score) or raw_score <= 0.0:
            break
        y, x = np.unravel_index(flat_index, working.shape)
        candidates.append(
            {
                "xy": [float(x * img_scaler[0]), float(y * img_scaler[1])],
                "score": _saturate_model_confidence(raw_score, "candidate.score"),
                "source_detector": "tracknet_heatmap_nms",
            }
        )
        if nms_radius_px <= 0.0:
            working[y, x] = -np.inf
            continue
        dx = (x_grid - x) * float(img_scaler[0])
        dy = (y_grid - y) * float(img_scaler[1])
        working[(dx * dx + dy * dy) <= nms_radius_px * nms_radius_px] = -np.inf
    return candidates


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
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_candidates",
        "fps": float(fps),
        "source": source,
        "source_mode": source_mode,
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
                "score": _saturate_model_confidence(blob.get("score", 0.0), "candidate.score"),
                "source_detector": str(blob.get("source_detector") or default_source_detector),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["xy"][0]), float(item["xy"][1])))
    return candidates[:top_k]


def _run_nonoverlap_heatmap_confidence(
    *,
    data_loader: Any,
    tracknet: Any,
    torch: Any,
    device: Any,
    img_scaler: tuple[float, float],
    expected_frame_count: int | None = None,
    candidate_frames: dict[int, list[dict[str, Any]]] | None = None,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_nms_radius_px: float = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    try:
        for indices, x in data_loader:
            x = x.float().to(device)
            with torch.no_grad():
                raw_pred = tracknet(x)
            y_pred = raw_pred.detach().cpu() if hasattr(raw_pred, "detach") else raw_pred
            rows.extend(
                _heatmap_prediction_rows(
                    indices,
                    y_pred,
                    img_scaler=img_scaler,
                    seen_frames=seen_frames,
                    candidate_frames=candidate_frames,
                    candidate_top_k=candidate_top_k,
                    candidate_nms_radius_px=candidate_nms_radius_px,
                )
            )
    except IndexError as exc:
        if not _is_tracknet_exact_end_large_video_exhaustion(exc, rows, expected_frame_count=expected_frame_count):
            raise
    _require_expected_heatmap_frame_coverage(rows, expected_frame_count=expected_frame_count)
    return rows


class _TrackNetVideoIterableDatasetEofGuard(_TorchIterableDataset):  # type: ignore[misc,valid-type]
    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self._dataset = dataset

    @property
    def video_len(self) -> Any:
        return getattr(self._dataset, "video_len", None)

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self._dataset)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except IndexError as exc:
                if str(exc) != "list index out of range":
                    raise
                return


def _is_tracknet_exact_end_large_video_exhaustion(
    exc: IndexError,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_frame_count: int | None,
) -> bool:
    if str(exc) != "list index out of range" or expected_frame_count is None or expected_frame_count <= 0:
        return False
    frames = {int(row["Frame"]) for row in rows if "Frame" in row}
    return len(frames) >= expected_frame_count and min(frames, default=-1) == 0 and max(frames, default=-1) >= expected_frame_count - 1


def _require_expected_heatmap_frame_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_frame_count: int | None,
) -> None:
    if expected_frame_count is None or expected_frame_count <= 0:
        return
    frames = {int(row["Frame"]) for row in rows if "Frame" in row}
    if len(frames) >= expected_frame_count and min(frames, default=-1) == 0 and max(frames, default=-1) >= expected_frame_count - 1:
        return
    raise ValueError(
        "TrackNet heatmap confidence produced incomplete frame coverage: "
        f"expected {expected_frame_count} frames, got {len(frames)} unique frame ids"
    )


def _run_overlap_heatmap_confidence(
    *,
    data_loader: Any,
    tracknet: Any,
    torch: Any,
    device: Any,
    video_len: int,
    seq_len: int,
    eval_mode: Literal["average", "weight"],
    img_scaler: tuple[float, float],
    candidate_frames: dict[int, list[dict[str, Any]]] | None = None,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_nms_radius_px: float = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
) -> list[dict[str, Any]]:
    from utils.general import HEIGHT, WIDTH

    num_sample = video_len - seq_len + 1
    if num_sample <= 0:
        raise ValueError(f"video must contain at least {seq_len} frames for overlap TrackNet inference")
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
    weight = _ensemble_weight(torch, seq_len, eval_mode)
    sample_count = 0
    rows: list[dict[str, Any]] = []
    seen_frames: set[int] = set()

    for indices, x in data_loader:
        x = x.float().to(device)
        batch_size = indices.shape[0]
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()
        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
        ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
        ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

        for b in range(batch_size):
            if sample_count < buffer_size:
                y_ensemble = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
            else:
                y_ensemble = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
            ensemble_i = torch.cat((ensemble_i, indices[b][0].reshape(1, 1, 2)), dim=0)
            ensemble_y_pred = torch.cat((ensemble_y_pred, y_ensemble.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
            sample_count += 1

            if sample_count == num_sample:
                y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                for f in range(1, seq_len):
                    y_ensemble = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                    ensemble_i = torch.cat((ensemble_i, indices[-1][f].reshape(1, 1, 2)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_ensemble.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

        rows.extend(
            _heatmap_prediction_rows(
                ensemble_i,
                ensemble_y_pred,
                img_scaler=img_scaler,
                seen_frames=seen_frames,
                candidate_frames=candidate_frames,
                candidate_top_k=candidate_top_k,
                candidate_nms_radius_px=candidate_nms_radius_px,
            )
        )
        y_pred_buffer = y_pred_buffer[-buffer_size:]

    return rows


def _ensemble_weight(torch: Any, seq_len: int, eval_mode: Literal["average", "weight"]) -> Any:
    if eval_mode == "average":
        return torch.ones(seq_len) / seq_len
    weight = torch.ones(seq_len)
    for index in range((seq_len + 1) // 2):
        value = index + 1
        weight[index] = value
        weight[seq_len - index - 1] = value
    return weight / weight.sum()


@contextmanager
def _tracknet_repo_imports(repo: Path) -> Iterator[None]:
    repo_text = str(repo)
    sys.path.insert(0, repo_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_text)
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
            if module_path.is_relative_to(repo):
                sys.modules.pop(name, None)


def _torch_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_video_range(video_range: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if video_range is None:
        return None
    if len(video_range) != 2:
        raise ValueError("video_range must contain START_S and END_S")
    start, end = int(video_range[0]), int(video_range[1])
    if start < 0 or end <= start:
        raise ValueError("video_range must satisfy 0 <= START_S < END_S")
    return start, end


def _persistent_prediction_csv_path(out: str | Path) -> Path:
    out_path = Path(out)
    return out_path.with_name(f"{out_path.stem}_tracknet_predictions.csv")


def _ball_candidates_sidecar_path(out: str | Path) -> Path:
    return Path(out).with_name("ball_candidates.json")


def _copy_file(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(source_path.read_bytes())


def _join_tracknet_confidence_csv(
    *,
    predictions_csv: str | Path,
    confidence_csv: str | Path,
    out: str | Path,
) -> Path:
    prediction_path = Path(predictions_csv)
    confidence_path = Path(confidence_csv)
    out_path = Path(out)
    confidence_by_frame = _read_generated_heatmap_confidences(confidence_path)
    last_confidence_frame = max(confidence_by_frame)

    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing TrackNet predictions CSV: {prediction_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        missing = [column for column in TRACKNET_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing TrackNet column(s): {', '.join(missing)}")
        with out_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=list(TRACKNET_CONFIDENCE_COLUMNS))
            writer.writeheader()
            for index, row in enumerate(reader):
                frame = _parse_nonnegative_int(row["Frame"], f"Frame/{index}")
                if frame not in confidence_by_frame:
                    if _is_terminal_hidden_zero_row(row, frame=frame, last_confidence_frame=last_confidence_frame):
                        confidence = 0.0
                    else:
                        raise ValueError(f"missing heatmap Confidence for frame {frame}")
                else:
                    confidence = confidence_by_frame[frame]
                writer.writerow(
                    {
                        "Frame": frame,
                        "Visibility": row["Visibility"],
                        "X": row["X"],
                        "Y": row["Y"],
                        "Confidence": f"{confidence:.8f}",
                    }
                )
    return out_path


def _is_terminal_hidden_zero_row(row: dict[str, str], *, frame: int, last_confidence_frame: int) -> bool:
    return (
        frame == last_confidence_frame + 1
        and not _parse_visibility(row["Visibility"], "Visibility")
        and _parse_float(row["X"], "X") == 0.0
        and _parse_float(row["Y"], "Y") == 0.0
    )


def _read_generated_heatmap_confidences(confidence_path: Path) -> dict[int, float]:
    if not confidence_path.is_file():
        raise FileNotFoundError(f"missing TrackNet heatmap confidence CSV: {confidence_path}")
    with confidence_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in TRACKNET_CONFIDENCE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing TrackNet column(s): {', '.join(missing)}")
        confidence_by_frame: dict[int, float] = {}
        for index, row in enumerate(reader):
            frame = _parse_nonnegative_int(row["Frame"], f"Frame/{index}")
            confidence_by_frame[frame] = _saturate_model_confidence(row["Confidence"], f"Confidence/{index}")
    if not confidence_by_frame:
        raise ValueError(f"TrackNet heatmap confidence CSV is empty: {confidence_path}")
    return confidence_by_frame


def _with_heatmap_confidence_csv_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_with_heatmap_confidence.csv")


@contextmanager
def _portable_tracknet_predict_repo(repo: Path) -> Iterator[Path]:
    predict_py = repo / "predict.py"
    source = predict_py.read_text(encoding="utf-8")
    if not _needs_portable_device_patch(source):
        yield repo
        return

    with tempfile.TemporaryDirectory(prefix="tracknetv3_repo_") as tmp_dir:
        runtime_repo = Path(tmp_dir) / repo.name
        shutil.copytree(
            repo,
            runtime_repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        runtime_predict_py = runtime_repo / "predict.py"
        runtime_predict_py.write_text(
            _patch_predict_for_portable_device(runtime_predict_py.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        yield runtime_repo


def _needs_portable_device_patch(source: str) -> bool:
    return ".cuda()" in source or ".cuda(" in source or "torch.load(args.tracknet_file)" in source


def _patch_predict_for_portable_device(source: str) -> str:
    patched = source
    if "DEVICE = torch.device" not in patched:
        helper = (
            "DEVICE = torch.device(\n"
            "    \"cuda\" if torch.cuda.is_available()\n"
            "    else \"mps\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()\n"
            "    else \"cpu\"\n"
            ")\n"
        )
        patched = patched.replace("import torch\n", f"import torch\n{helper}\n", 1)
    patched = patched.replace("torch.load(args.tracknet_file)", "torch.load(args.tracknet_file, map_location=DEVICE)")
    patched = patched.replace(
        "torch.load(args.inpaintnet_file)",
        "torch.load(args.inpaintnet_file, map_location=DEVICE)",
    )
    patched = patched.replace(".cuda()", ".to(DEVICE)")
    return patched


def _add_runtime_metrics(runtime: dict[str, Any], *, processed_frame_count: int, fps: float) -> None:
    runtime["processed_frame_count"] = int(processed_frame_count)
    runtime["video_seconds_processed"] = int(processed_frame_count) / float(fps)
    wall_seconds = runtime.get("wall_seconds")
    if wall_seconds is None:
        return
    wall = float(wall_seconds)
    if wall <= 0.0:
        runtime["effective_fps"] = None
        runtime["realtime_factor"] = None
        return
    effective_fps = int(processed_frame_count) / wall
    runtime["effective_fps"] = effective_fps
    runtime["realtime_factor"] = effective_fps / float(fps)


def run_tracknet_or_convert(
    *,
    out: str | Path,
    fps: float,
    frame_times: Any = None,
    metadata_out: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    video: str | Path | None = None,
    tracknet_file: str | Path | None = None,
    inpaintnet_file: str | Path | None = None,
    tracknet_repo: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    batch_size: int = 16,
    video_range: tuple[int, int] | list[int] | None = None,
    large_video: bool = False,
    confidence_mode: TrackNetConfidenceMode = "legacy_visibility",
    heatmap_visible_threshold: float = DEFAULT_HEATMAP_VISIBLE_THRESHOLD,
    heatmap_eval_mode: TrackNetHeatmapEvalMode = "weight",
    heatmap_large_video: bool | None = None,
    emit_candidates: bool = False,
    candidate_top_k: int = DEFAULT_BALL_CANDIDATE_TOP_K,
    candidate_nms_radius_px: float = DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX,
) -> dict[str, Any]:
    """CLI-oriented entrypoint that converts CSV or runs official TrackNetV3."""

    candidate_top_k = _require_positive_int(candidate_top_k, "candidate_top_k")
    candidate_nms_radius_px = _require_nonnegative_float(candidate_nms_radius_px, "candidate_nms_radius_px")
    heatmap_visible_threshold = _parse_confidence(heatmap_visible_threshold, "heatmap_visible_threshold")
    if predictions_csv is not None:
        if emit_candidates:
            raise ValueError("TrackNet emit_candidates requires heatmap inference; predictions CSV has only tracked argmax rows")
        return write_ball_track_from_csv(
            predictions_csv=predictions_csv,
            fps=fps,
            frame_times=frame_times,
            out=out,
            metadata_out=metadata_out,
            source_mode="tracknet_csv",
            confidence_mode=confidence_mode,
            heatmap_visible_threshold=heatmap_visible_threshold,
        )

    if video is None or tracknet_file is None or inpaintnet_file is None or tracknet_repo is None:
        raise ValueError("either --predictions-csv or --video/--tracknet-file/--inpaintnet-file/--tracknet-repo is required")
    if emit_candidates and confidence_mode != "heatmap_peak":
        raise ValueError("TrackNet emit_candidates requires confidence_mode='heatmap_peak' so heatmaps are available")

    predict_py = Path(tracknet_repo) / "predict.py"
    if not predict_py.is_file():
        raise FileNotFoundError(f"missing TrackNetV3 predict.py: {predict_py}")

    normalized_video_range = _normalize_video_range(video_range)
    runtime = {
        "tracknet_repo": str(tracknet_repo),
        "tracknet_checkpoint": checkpoint_metadata(tracknet_file),
        "inpaintnet_checkpoint": checkpoint_metadata(inpaintnet_file),
        "video": str(video),
        "batch_size": int(batch_size),
        "large_video": bool(large_video),
    }
    if confidence_mode == "heatmap_peak":
        runtime["heatmap_eval_mode"] = heatmap_eval_mode
        runtime["heatmap_large_video"] = bool(large_video if heatmap_large_video is None else heatmap_large_video)
    if emit_candidates:
        runtime["candidates_out"] = str(_ball_candidates_sidecar_path(out))
        runtime["candidate_top_k"] = candidate_top_k
        runtime["candidate_nms_radius_px"] = candidate_nms_radius_px
    if normalized_video_range is not None:
        runtime["video_range_seconds"] = list(normalized_video_range)
        runtime["video_range_semantics"] = (
            "official TrackNetV3 background median sampling range; does not trim prediction frames"
        )
    if prediction_dir is None:
        with tempfile.TemporaryDirectory(prefix="tracknetv3_") as tmp_dir:
            start = time.perf_counter()
            csv_path = run_official_tracknet_predict(
                tracknet_repo=tracknet_repo,
                video=video,
                tracknet_file=tracknet_file,
                inpaintnet_file=inpaintnet_file,
                save_dir=tmp_dir,
                batch_size=batch_size,
                video_range=normalized_video_range,
                large_video=large_video,
            )
            runtime["wall_seconds"] = time.perf_counter() - start
            if confidence_mode == "heatmap_peak":
                confidence_csv = run_tracknet_heatmap_confidence_predict(
                    tracknet_repo=tracknet_repo,
                    video=video,
                    tracknet_file=tracknet_file,
                    save_dir=tmp_dir,
                    batch_size=batch_size,
                    video_range=normalized_video_range,
                    large_video=bool(large_video if heatmap_large_video is None else heatmap_large_video),
                    eval_mode=heatmap_eval_mode,
                    emit_candidates=emit_candidates,
                    candidate_top_k=candidate_top_k,
                    candidate_nms_radius_px=candidate_nms_radius_px,
                    candidates_out=_ball_candidates_sidecar_path(out) if emit_candidates else None,
                    candidate_fps=fps,
                    primary_output=out,
                )
                runtime["heatmap_confidence_csv"] = str(confidence_csv)
                csv_path = _join_tracknet_confidence_csv(
                    predictions_csv=csv_path,
                    confidence_csv=confidence_csv,
                    out=_with_heatmap_confidence_csv_path(csv_path),
                )
            persistent_csv_path = _persistent_prediction_csv_path(out)
            _copy_file(csv_path, persistent_csv_path)
            return write_ball_track_from_csv(
                predictions_csv=persistent_csv_path,
                fps=fps,
                frame_times=frame_times,
                out=out,
                metadata_out=metadata_out,
                source_mode="tracknet_predict",
                runtime=runtime,
                confidence_mode=confidence_mode,
                heatmap_visible_threshold=heatmap_visible_threshold,
            )

    start = time.perf_counter()
    csv_path = run_official_tracknet_predict(
        tracknet_repo=tracknet_repo,
        video=video,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        save_dir=prediction_dir,
        batch_size=batch_size,
        video_range=normalized_video_range,
        large_video=large_video,
    )
    runtime["wall_seconds"] = time.perf_counter() - start
    if confidence_mode == "heatmap_peak":
        confidence_csv = run_tracknet_heatmap_confidence_predict(
            tracknet_repo=tracknet_repo,
            video=video,
            tracknet_file=tracknet_file,
            save_dir=prediction_dir,
            batch_size=batch_size,
            video_range=normalized_video_range,
            large_video=bool(large_video if heatmap_large_video is None else heatmap_large_video),
            eval_mode=heatmap_eval_mode,
            emit_candidates=emit_candidates,
            candidate_top_k=candidate_top_k,
            candidate_nms_radius_px=candidate_nms_radius_px,
            candidates_out=_ball_candidates_sidecar_path(out) if emit_candidates else None,
            candidate_fps=fps,
            primary_output=out,
        )
        runtime["heatmap_confidence_csv"] = str(confidence_csv)
        csv_path = _join_tracknet_confidence_csv(
            predictions_csv=csv_path,
            confidence_csv=confidence_csv,
            out=_with_heatmap_confidence_csv_path(csv_path),
        )
    return write_ball_track_from_csv(
        predictions_csv=csv_path,
        fps=fps,
        frame_times=frame_times,
        out=out,
        metadata_out=metadata_out,
        source_mode="tracknet_predict",
        runtime=runtime,
        confidence_mode=confidence_mode,
        heatmap_visible_threshold=heatmap_visible_threshold,
    )


__all__ = [
    "CONFIDENCE_SEMANTICS",
    "DEFAULT_BALL_CANDIDATE_TOP_K",
    "DEFAULT_HEATMAP_VISIBLE_THRESHOLD",
    "DEFAULT_TRACKNET_CANDIDATE_NMS_RADIUS_PX",
    "HEATMAP_CONFIDENCE_SEMANTICS",
    "LEGACY_CONFIDENCE_SEMANTICS",
    "checkpoint_metadata",
    "run_official_tracknet_predict",
    "run_tracknet_heatmap_confidence_predict",
    "run_tracknet_or_convert",
    "tracknet_csv_to_ball_track",
    "write_ball_track_from_csv",
]
