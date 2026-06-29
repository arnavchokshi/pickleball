#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.totnet_adapter import checkpoint_metadata, write_ball_track_from_totnet_predictions  # noqa: E402


def run_totnet_video(
    *,
    video: Path,
    totnet_repo: Path,
    checkpoint: Path,
    out: Path,
    predictions_out: Path,
    metadata_out: Path | None = None,
    model_id: str = "totnet_tennis_5f_288x512",
    num_frames: int = 5,
    num_channels: int = 64,
    input_height: int = 288,
    input_width: int = 512,
    batch_size: int = 8,
    confidence_threshold: float = 0.0,
    device: str | None = None,
    require_cuda: bool = False,
    min_matched_keys: int = 180,
    max_frames: int | None = None,
) -> dict[str, Any]:
    import torch

    cv2 = _cv2()
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input size must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    max_frames = _normalize_max_frames(max_frames)

    video = _require_file(video, "video")
    checkpoint = _require_file(checkpoint, "checkpoint")
    src_dir = _require_dir(totnet_repo / "src", "TOTNet src")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    build_motion_model_light = _load_totnet_builder(src_dir)

    runtime_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device but CUDA is unavailable: {runtime_device}")
    if require_cuda and runtime_device.type != "cuda":
        raise RuntimeError("TOTNet GPU eval requires CUDA; pass --device cuda:0 on a CUDA host")
    configs = SimpleNamespace(
        device=runtime_device,
        num_frames=num_frames,
        num_channels=num_channels,
        img_size=(input_height, input_width),
    )
    model = build_motion_model_light(configs).to(runtime_device)
    matched_keys = _load_checkpoint(model, checkpoint, runtime_device, min_matched_keys=min_matched_keys)
    model.eval()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not math.isfinite(fps) or fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata for {video}: fps={fps}, width={width}, height={height}")

    start = time.perf_counter()
    frames: list[dict[str, Any]] = []
    window: deque[np.ndarray] = deque(maxlen=num_frames)
    pending: list[tuple[int, np.ndarray]] = []
    timing_breakdown = {
        "decode_preprocess_seconds": 0.0,
        "host_to_device_seconds": 0.0,
        "inference_seconds": 0.0,
        "postprocess_seconds": 0.0,
    }

    def flush_pending() -> None:
        if not pending:
            return
        post_start = time.perf_counter()
        indexes = [item[0] for item in pending]
        batch_np = np.stack([item[1] for item in pending], axis=0)
        timing_breakdown["postprocess_seconds"] += time.perf_counter() - post_start
        h2d_start = time.perf_counter()
        batch = torch.from_numpy(batch_np).to(runtime_device, non_blocking=True).float()
        timing_breakdown["host_to_device_seconds"] += time.perf_counter() - h2d_start
        inference_start = time.perf_counter()
        with torch.inference_mode():
            heatmap = model(batch)
        timing_breakdown["inference_seconds"] += time.perf_counter() - inference_start
        post_start = time.perf_counter()
        probs = _flatten_totnet_heatmap(heatmap, input_height=input_height, input_width=input_width)
        flat_conf, flat_index = torch.max(probs, dim=1)
        x_pred = (flat_index % input_width).detach().cpu().numpy().astype(np.float64)
        y_pred = (flat_index // input_width).detach().cpu().numpy().astype(np.float64)
        conf = flat_conf.detach().cpu().numpy().astype(np.float64)
        scale_x = float(width) / float(input_width)
        scale_y = float(height) / float(input_height)
        for frame_index, x_value, y_value, confidence in zip(indexes, x_pred, y_pred, conf, strict=True):
            frames.append(
                {
                    "frame_index": int(frame_index),
                    "xy": [float(x_value * scale_x), float(y_value * scale_y)],
                    "confidence": float(confidence),
                    "visible": True,
                }
            )
        pending.clear()
        timing_breakdown["postprocess_seconds"] += time.perf_counter() - post_start

    frame_index = 0
    while max_frames is None or frame_index < max_frames:
        decode_start = time.perf_counter()
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        window.append(_normalize_frame(resized))
        timing_breakdown["decode_preprocess_seconds"] += time.perf_counter() - decode_start
        if len(window) < num_frames:
            frames.append({"frame_index": frame_index, "xy": None, "confidence": 0.0, "visible": False})
        else:
            post_start = time.perf_counter()
            pending.append((frame_index, np.stack(list(window), axis=0)))
            timing_breakdown["postprocess_seconds"] += time.perf_counter() - post_start
            if len(pending) >= batch_size:
                flush_pending()
        frame_index += 1
    flush_pending()
    cap.release()

    elapsed_seconds = time.perf_counter() - start
    runtime = {
        "video": str(video),
        "totnet_repo": str(totnet_repo),
        "device": str(runtime_device),
        "batch_size": batch_size,
        "video_frame_count_metadata": frame_count,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(runtime_device) if runtime_device.type == "cuda" else None,
    }
    runtime.update(
        _runtime_metrics(
            seconds=elapsed_seconds,
            decoded_frame_count=frame_index,
            source_fps=fps,
            max_frames=max_frames,
            timing_breakdown=timing_breakdown,
        )
    )

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_totnet_predictions",
        "fps": fps,
        "image_size": [width, height],
        "input_size": [input_width, input_height],
        "model": {
            "id": model_id,
            "model_family": "TOTNet",
            "num_frames": num_frames,
            "num_channels": num_channels,
            "checkpoint": checkpoint_metadata(checkpoint),
            "matched_checkpoint_keys": matched_keys,
            "min_matched_checkpoint_keys": min_matched_keys,
            "confidence_semantics": "maximum value from TOTNet heatmap output after model softmax",
        },
        "frames": sorted(frames, key=lambda item: item["frame_index"]),
        "runtime": runtime,
        "not_ground_truth": True,
    }
    _write_json(predictions_out, payload)
    metadata = write_ball_track_from_totnet_predictions(
        payload,
        out=out,
        metadata_out=metadata_out,
        confidence_threshold=confidence_threshold,
    )
    metadata["runtime"] = payload["runtime"]
    if metadata_out is not None:
        _write_json(metadata_out, metadata)
    return metadata


def _normalize_max_frames(max_frames: int | None) -> int | None:
    if max_frames is None:
        return None
    value = int(max_frames)
    if value < 1:
        raise ValueError("max_frames must be positive")
    return value


def _runtime_metrics(
    *,
    seconds: float,
    decoded_frame_count: int,
    source_fps: float,
    max_frames: int | None,
    timing_breakdown: dict[str, float] | None = None,
) -> dict[str, Any]:
    elapsed = float(seconds)
    if elapsed <= 0.0:
        effective_fps = None
        realtime_factor = None
    else:
        effective_fps = float(decoded_frame_count) / elapsed
        realtime_factor = effective_fps / float(source_fps) if source_fps > 0.0 else None
    video_seconds = float(decoded_frame_count) / float(source_fps) if source_fps > 0.0 else None
    breakdown = dict(timing_breakdown or {})
    breakdown["accounted_seconds"] = sum(float(value) for value in breakdown.values())
    return {
        "seconds": elapsed,
        "decoded_frame_count": int(decoded_frame_count),
        "effective_fps": effective_fps,
        "video_seconds_processed": video_seconds,
        "realtime_factor": realtime_factor,
        "max_frames": max_frames,
        "timing_breakdown": breakdown,
    }


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required to run TOTNet video inference") from exc
    return cv2


def _normalize_frame(rgb: np.ndarray) -> np.ndarray:
    frame = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std
    return np.transpose(frame, (2, 0, 1))


def _flatten_totnet_heatmap(heatmap: Any, *, input_height: int, input_width: int) -> Any:
    if isinstance(heatmap, (tuple, list)):
        raise ValueError("expected TOTNet heatmap tensor output, got tuple/list")
    probs = heatmap.detach().float()
    expected_size = input_height * input_width
    if probs.ndim == 4:
        if probs.shape[1] != 1:
            raise ValueError(f"expected TOTNet heatmap [B,1,H,W], got {tuple(probs.shape)}")
        probs = probs[:, 0, :, :]
    if probs.ndim == 3:
        if tuple(probs.shape[1:]) != (input_height, input_width):
            raise ValueError(f"expected TOTNet heatmap spatial shape {(input_height, input_width)}, got {tuple(probs.shape)}")
        probs = probs.reshape(probs.shape[0], expected_size)
    if probs.ndim != 2 or probs.shape[1] != expected_size:
        raise ValueError(f"expected TOTNet flattened heatmap [B,{expected_size}], got {tuple(probs.shape)}")
    return probs


def _load_checkpoint(model: Any, checkpoint: Path, device: Any, *, min_matched_keys: int) -> int:
    import torch

    if min_matched_keys < 1:
        raise ValueError("min_matched_keys must be positive")
    _install_easydict_unpickle_shim()
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint does not contain a state_dict: {checkpoint}")
    model_state = model.state_dict()
    matched = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
        if normalized_key in model_state and tuple(model_state[normalized_key].shape) == tuple(value.shape):
            matched[normalized_key] = value
    if not matched:
        raise ValueError(f"checkpoint loaded zero matching model keys: {checkpoint}")
    if len(matched) < min_matched_keys:
        raise ValueError(f"checkpoint matched only {len(matched)} model keys, expected at least {min_matched_keys}: {checkpoint}")
    model_state.update(matched)
    model.load_state_dict(model_state)
    return len(matched)


def _install_easydict_unpickle_shim() -> None:
    if "easydict" in sys.modules:
        return
    try:
        __import__("easydict")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

        def __delattr__(self, name: str) -> None:
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    EasyDict.__module__ = "easydict"
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def _load_totnet_builder(src_dir: Path) -> Any:
    module_path = src_dir / "model" / "TOTNet.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing TOTNet model file: {module_path}")
    spec = importlib.util.spec_from_file_location("totnet_public_model", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load TOTNet module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_motion_model_light


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a public TOTNet checkpoint and write BallTrack JSON.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--totnet-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, default=None)
    parser.add_argument("--model-id", default="totnet_tennis_5f_288x512")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-matched-keys", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    try:
        summary = run_totnet_video(
            video=args.video,
            totnet_repo=args.totnet_repo,
            checkpoint=args.checkpoint,
            out=args.out,
            predictions_out=args.predictions_out,
            metadata_out=args.metadata_out,
            model_id=args.model_id,
            num_frames=args.num_frames,
            num_channels=args.num_channels,
            input_height=args.input_height,
            input_width=args.input_width,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            require_cuda=args.require_cuda,
            min_matched_keys=args.min_matched_keys,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
