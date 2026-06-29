#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mobile_person_yolo_replay import _observations_from_result, _video_properties  # noqa: E402
from threed.racketsport.person_detector_oracle import (  # noqa: E402
    detections_payload_to_candidates,
    score_detector_oracle,
)
from threed.racketsport.schemas import PersonGroundTruth, validate_artifact_file  # noqa: E402
from threed.racketsport.tiled_person_detector import (  # noqa: E402
    parse_adaptive_crop_regions,
    parse_crop_regions,
    yolo_adaptive_tiled_detections_payload,
    yolo_tiled_detections_payload,
)


DEFAULT_CLIPS = (
    "task_2376761=runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376761/person_ground_truth.json",
    "task_2376765=runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376765/person_ground_truth.json",
)


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    video_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure detector-only top-N oracle recall against person GT.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clip", action="append", default=[], help="clip_id=video.mp4=person_ground_truth.json")
    parser.add_argument("--model", action="append", default=[], help="name=models/checkpoints/model.pt")
    parser.add_argument("--mode", action="append", choices=("full", "tiled", "adaptive_tiled"), default=[])
    parser.add_argument("--imgsz", action="append", type=int, default=[])
    parser.add_argument("--conf", action="append", type=float, default=[])
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--candidate-limit", action="append", type=int, default=[])
    parser.add_argument("--oracle-iou", action="append", type=float, default=[])
    parser.add_argument("--crop-regions", default="default4")
    parser.add_argument("--adaptive-crop-regions", default="adaptive_full_tb3")
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("runs/phase2/iphone_person_tracking_eval") / f"person_detector_oracle_{_timestamp()}"
    rows = run_detector_oracle(
        out_dir=out_dir,
        clips=_parse_clips(args.clip or list(DEFAULT_CLIPS), require_files=False),
        models=_parse_models(args.model or ["yolo11s=models/checkpoints/yolo11s.pt"], require_files=False),
        modes=tuple(args.mode or ["full"]),
        imgsz_values=tuple(_unique_ints(args.imgsz or [1280])),
        conf_values=tuple(_unique_floats(args.conf or [0.05])),
        detector_iou=float(args.iou),
        candidate_limits=tuple(_unique_ints(args.candidate_limit or [4, 8, 12, 20])),
        oracle_ious=tuple(_unique_floats(args.oracle_iou or [0.3, 0.5])),
        crop_regions=args.crop_regions,
        adaptive_crop_regions=args.adaptive_crop_regions,
        nms_iou=float(args.nms_iou),
        batch_size=int(args.batch_size),
        device=_resolve_device(args.device),
        max_frames=args.max_frames,
        force=bool(args.force),
    )
    _write_outputs(out_dir, rows)
    print(json.dumps({"out_dir": str(out_dir), "row_count": len(rows), "report_path": str(out_dir / "REPORT.md")}, sort_keys=True))
    return 0


def run_detector_oracle(
    *,
    out_dir: Path,
    clips: Sequence[ClipSpec],
    models: Sequence[ModelSpec],
    modes: Sequence[str],
    imgsz_values: Sequence[int],
    conf_values: Sequence[float],
    detector_iou: float,
    candidate_limits: Sequence[int],
    oracle_ious: Sequence[float],
    crop_regions: str,
    adaptive_crop_regions: str,
    nms_iou: float,
    batch_size: int,
    device: str | None,
    max_frames: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore[import-not-found]
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and ultralytics are required for detector oracle sweeps") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    max_candidate_limit = max(candidate_limits)
    for model_spec in models:
        model = YOLO(str(model_spec.path))
        for clip in clips:
            gt = validate_artifact_file("person_ground_truth", clip.ground_truth_path)
            if not isinstance(gt, PersonGroundTruth):
                raise ValueError(f"ground truth artifact did not parse as PersonGroundTruth: {clip.ground_truth_path}")
            fps, width, height, _frame_count = _video_properties(cv2, clip.video_path)
            for mode in modes:
                for imgsz in imgsz_values:
                    for conf in conf_values:
                        candidate_name = _candidate_name(model_spec.name, mode, imgsz, conf, detector_iou, crop_regions if mode == "tiled" else adaptive_crop_regions if mode == "adaptive_tiled" else None)
                        run_dir = out_dir / clip.clip_id / candidate_name
                        metrics_path = run_dir / "detector_oracle_metrics.json"
                        if metrics_path.exists() and not force:
                            rows.append(_row_from_metrics(json.loads(metrics_path.read_text(encoding="utf-8"))))
                            continue
                        started = time.perf_counter()
                        candidates_by_frame, extra = _collect_candidates(
                            model=model,
                            mode=mode,
                            video_path=clip.video_path,
                            fps=fps,
                            imgsz=imgsz,
                            conf=conf,
                            detector_iou=detector_iou,
                            device=device,
                            max_frames=max_frames,
                            max_candidate_limit=max_candidate_limit,
                            crop_regions=crop_regions,
                            adaptive_crop_regions=adaptive_crop_regions,
                            nms_iou=nms_iou,
                            batch_size=batch_size,
                        )
                        wall_time_s = time.perf_counter() - started
                        metrics = score_detector_oracle(
                            gt,
                            candidates_by_frame,
                            candidate_limits=candidate_limits,
                            iou_thresholds=oracle_ious,
                        )
                        processed_frames = len(candidates_by_frame)
                        metrics.update(
                            {
                                "candidate": candidate_name,
                                "model": model_spec.name,
                                "model_path": str(model_spec.path),
                                "mode": mode,
                                "imgsz": int(imgsz),
                                "conf": float(conf),
                                "detector_iou": float(detector_iou),
                                "device": device,
                                "video_path": str(clip.video_path),
                                "ground_truth_path": str(clip.ground_truth_path),
                                "resolution": [int(width), int(height)],
                                "fps": float(fps),
                                "processed_frame_count": processed_frames,
                                "wall_time_s": wall_time_s,
                                "processed_fps": processed_frames / wall_time_s if wall_time_s > 0.0 else 0.0,
                                "avg_detections_per_frame": _avg(len(candidates) for candidates in candidates_by_frame.values()),
                                "metrics_path": str(metrics_path),
                                **extra,
                            }
                        )
                        _write_json(metrics_path, metrics)
                        rows.append(_row_from_metrics(metrics))
    return rows


def _collect_candidates(
    *,
    model: Any,
    mode: str,
    video_path: Path,
    fps: float,
    imgsz: int,
    conf: float,
    detector_iou: float,
    device: str | None,
    max_frames: int | None,
    max_candidate_limit: int,
    crop_regions: str,
    adaptive_crop_regions: str,
    nms_iou: float,
    batch_size: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if mode == "full":
        return _collect_full_frame_candidates(
            model=model,
            video_path=video_path,
            imgsz=imgsz,
            conf=conf,
            detector_iou=detector_iou,
            device=device,
            max_frames=max_frames,
            max_candidate_limit=max_candidate_limit,
        ), {"crop_region_count": 1, "adaptive_fallback_frame_count": 0}
    if mode == "tiled":
        regions = parse_crop_regions(crop_regions)
        payload = yolo_tiled_detections_payload(
            model=model,
            video_path=video_path,
            fps=fps,
            max_frames=max_frames,
            crop_regions=regions,
            conf=conf,
            iou=detector_iou,
            imgsz=imgsz,
            device=device,
            nms_iou=nms_iou,
            batch_size=batch_size,
        )
        return detections_payload_to_candidates(payload), {"crop_region_count": len(regions), "adaptive_fallback_frame_count": 0}
    if mode == "adaptive_tiled":
        primary, fallback, min_detections = parse_adaptive_crop_regions(adaptive_crop_regions)
        payload = yolo_adaptive_tiled_detections_payload(
            model=model,
            video_path=video_path,
            fps=fps,
            max_frames=max_frames,
            primary_crop_regions=primary,
            fallback_crop_regions=fallback,
            min_detections=min_detections,
            conf=conf,
            iou=detector_iou,
            imgsz=imgsz,
            device=device,
            nms_iou=nms_iou,
            batch_size=batch_size,
        )
        return detections_payload_to_candidates(payload), {
            "crop_region_count": int(payload.get("primary_crop_region_count", len(primary))) + int(payload.get("fallback_crop_region_count", len(fallback))),
            "adaptive_fallback_frame_count": int(payload.get("fallback_frame_count", 0)),
            "adaptive_crop_eval_count": int(payload.get("crop_eval_count", 0)),
        }
    raise ValueError(f"unsupported detector oracle mode: {mode}")


def _collect_full_frame_candidates(
    *,
    model: Any,
    video_path: Path,
    imgsz: int,
    conf: float,
    detector_iou: float,
    device: str | None,
    max_frames: int | None,
    max_candidate_limit: int,
) -> dict[int, list[dict[str, Any]]]:
    iterator = model.predict(
        source=str(video_path),
        stream=True,
        classes=[0],
        conf=conf,
        iou=detector_iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    by_frame: dict[int, list[dict[str, Any]]] = {}
    try:
        for frame_index, result in enumerate(iterator):
            if max_frames is not None and frame_index >= max_frames:
                break
            by_frame[frame_index] = _observations_from_result(
                result,
                max_players=4,
                output_limit=max_candidate_limit,
                candidate_limit=max_candidate_limit,
            )
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    return by_frame


def _row_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "clip_id": metrics["clip_id"],
        "candidate": metrics["candidate"],
        "model": metrics["model"],
        "mode": metrics["mode"],
        "imgsz": metrics["imgsz"],
        "conf": metrics["conf"],
        "wall_time_s": metrics["wall_time_s"],
        "processed_fps": metrics["processed_fps"],
        "avg_detections_per_frame": metrics["avg_detections_per_frame"],
        "metrics_path": metrics["metrics_path"],
    }
    for limit, limit_metrics in metrics["candidate_limits"].items():
        for threshold_key, threshold_metrics in limit_metrics.items():
            row[f"L{limit}_{threshold_key}"] = threshold_metrics["recall"]
    return row


def _write_outputs(out_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "detector_oracle_rows.json", list(rows))
    _write_csv(out_dir / "detector_oracle.csv", rows)
    (out_dir / "REPORT.md").write_text(_render_report(rows), encoding="utf-8")


def _render_report(rows: Sequence[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: (float(row.get("L12_iou_0.50", row.get("L8_iou_0.50", 0.0))), float(row.get("L20_iou_0.50", 0.0))), reverse=True)
    lines = [
        "# Person Detector Oracle Sweep",
        "",
        "Detector-only recall: each GT person counts as hit if any top-N detector candidate overlaps it.",
        "",
        "| Rank | Clip | Candidate | Mode | L4@0.5 | L8@0.5 | L12@0.5 | L20@0.5 | FPS | Avg Det/Frame |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked[:40], start=1):
        lines.append(
            "| {rank} | `{clip}` | `{candidate}` | `{mode}` | {l4:.3f} | {l8:.3f} | {l12:.3f} | {l20:.3f} | {fps:.2f} | {det:.2f} |".format(
                rank=rank,
                clip=row["clip_id"],
                candidate=row["candidate"],
                mode=row["mode"],
                l4=float(row.get("L4_iou_0.50", 0.0)),
                l8=float(row.get("L8_iou_0.50", 0.0)),
                l12=float(row.get("L12_iou_0.50", 0.0)),
                l20=float(row.get("L20_iou_0.50", 0.0)),
                fps=float(row.get("processed_fps", 0.0)),
                det=float(row.get("avg_detections_per_frame", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _parse_clips(specs: Sequence[str], *, require_files: bool = True) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"clip spec must be clip_id=video=ground_truth: {spec}")
        clip_id, video, gt = parts
        clip = ClipSpec(clip_id=clip_id, video_path=Path(video), ground_truth_path=Path(gt))
        if require_files:
            if not clip.video_path.is_file():
                raise FileNotFoundError(f"missing video for {clip_id}: {clip.video_path}")
            if not clip.ground_truth_path.is_file():
                raise FileNotFoundError(f"missing ground truth for {clip_id}: {clip.ground_truth_path}")
        clips.append(clip)
    return clips


def _parse_models(specs: Sequence[str], *, require_files: bool = True) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for spec in specs:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        model = ModelSpec(name=_safe_token(name), path=Path(path))
        if require_files and not model.path.exists():
            raise FileNotFoundError(f"missing model for {model.name}: {model.path}")
        models.append(model)
    return models


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_name(model: str, mode: str, imgsz: int, conf: float, iou: float, crop_token: str | None) -> str:
    parts = [_safe_token(model), _safe_token(mode), f"img{imgsz}", f"conf{_number_token(conf)}", f"iou{_number_token(iou)}"]
    if crop_token:
        parts.append(_safe_token(crop_token))
    return "_".join(parts)


def _resolve_device(device: str) -> str | None:
    if device != "auto":
        return None if device in {"", "none", "default"} else device
    try:
        import torch  # type: ignore[import-not-found]

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        return None
    return None


def _unique_ints(values: Sequence[int]) -> list[int]:
    result = sorted({int(value) for value in values})
    if not result or any(value <= 0 for value in result):
        raise ValueError("integer values must be positive")
    return result


def _unique_floats(values: Sequence[float]) -> list[float]:
    result = sorted({float(value) for value in values})
    if not result:
        raise ValueError("at least one float value is required")
    return result


def _avg(values: Iterable[int | float]) -> float:
    data = [float(value) for value in values]
    return sum(data) / len(data) if data else 0.0


def _number_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "")


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("empty token")
    return token


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
