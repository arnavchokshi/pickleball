#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mobile_person_eval import score_mobile_person_tracks, write_mobile_person_metrics  # noqa: E402
from threed.racketsport.mobile_person_yolo_replay import (  # noqa: E402
    ReplayYoloCandidate,
    _closed_set_prune_frames,
    _latency_ms_from_result,
    _make_linker,
    _observations_from_result,
    _package_size_mb,
    _timing_payload,
    _tracks_payload,
    _video_properties,
    render_replay_yolo_overlay,
    run_replay_yolo_candidate,
)
from threed.racketsport.schemas import OnDevicePersonTracks, PersonGroundTruth, validate_artifact_file  # noqa: E402


DEFAULT_CLIPS = (
    "task_2376761=runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376761/person_ground_truth.json="
    "runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/court_calibration.json",
    "task_2376765=runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376765/person_ground_truth.json="
    "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/court_calibration.json",
)
DEFAULT_MODELS = (
    "yolo11n=models/checkpoints/yolo11n.pt",
    "yolo26n=models/checkpoints/yolo26n.pt",
    "yolo26s=models/checkpoints/yolo26s.pt",
    "yolo26m=models/checkpoints/yolo26m.pt",
)
DEFAULT_TRACK_MODELS = ("yolo26n=models/checkpoints/yolo26n.pt", "yolo26s=models/checkpoints/yolo26s.pt")
DEFAULT_IMGSZ = (416, 512, 640, 960)
DEFAULT_CONF = (0.05, 0.10, 0.15, 0.20)
DEFAULT_LINKERS = ("predict_iou", "predict_iou_loose", "predict_center", "predict_center_loose", "predict_role_lock")
DEFAULT_TRACK_MODES = ("track_bytetrack_loose", "track_botsort_no_reid_loose")
DEFAULT_CLOSED_SET_MODES = ("quality", "duration", "motion", "quality_cluster", "cluster_strong", "motion_cluster")


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    video_path: Path
    ground_truth_path: Path
    court_calibration_path: Path | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a laptop-local YOLO person accuracy sweep against CVAT MOT-derived person GT."
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clip", action="append", default=[], help="clip_id=video.mp4=person_ground_truth.json")
    parser.add_argument("--model", action="append", default=[], help="name=models/checkpoints/model.pt")
    parser.add_argument("--track-model", action="append", default=[], help="name=models/checkpoints/model.pt for model.track modes")
    parser.add_argument("--imgsz", action="append", type=int, default=[])
    parser.add_argument("--conf", action="append", type=float, default=[])
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--tracker", action="append", default=[])
    parser.add_argument("--detector-output-limit", action="append", type=int, default=[])
    parser.add_argument("--closed-set-mode", action="append", choices=DEFAULT_CLOSED_SET_MODES, default=[])
    parser.add_argument("--prune-mode", action="append", choices=("confidence", "court"), default=[])
    parser.add_argument("--court-margin-m", action="append", type=float, default=[])
    parser.add_argument("--bbox-expand", action="append", type=float, default=[])
    parser.add_argument("--track-mode", action="append", default=[])
    parser.add_argument("--track-imgsz", action="append", type=int, default=[])
    parser.add_argument("--track-conf", action="append", type=float, default=[])
    parser.add_argument("--include-reid-trackers", action="store_true")
    parser.add_argument("--skip-ultralytics-track", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--top-overlays", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success for exploratory sweeps with at least one successful row even if some candidates failed.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or Path("runs/phase2/iphone_person_tracking_eval") / f"local_accuracy_sweep_{_timestamp()}"
    clips = _parse_clips(args.clip or list(DEFAULT_CLIPS))
    models = _parse_models(args.model or list(DEFAULT_MODELS))
    track_models = _parse_models(args.track_model or list(DEFAULT_TRACK_MODELS))
    imgsz_values = _unique_ints(args.imgsz or list(DEFAULT_IMGSZ))
    conf_values = _unique_floats(args.conf or list(DEFAULT_CONF))
    linkers = tuple(args.tracker or list(DEFAULT_LINKERS))
    detector_output_limits = _unique_ints(args.detector_output_limit or [args.max_players])
    closed_set_modes = tuple(args.closed_set_mode or [])
    prune_modes = tuple(args.prune_mode or ["confidence"])
    court_margins = _unique_floats(args.court_margin_m or [1.25])
    bbox_expands = _unique_floats(args.bbox_expand or [1.0])
    track_modes = list(args.track_mode or list(DEFAULT_TRACK_MODES))
    if args.include_reid_trackers:
        track_modes.extend(["track_botsort_reid", "track_botsort_reid_loose"])
    track_imgsz_values = _unique_ints(args.track_imgsz or [640])
    track_conf_values = _unique_floats(args.track_conf or [0.10])
    device = _resolve_device(args.device)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing sweep to {out_dir}")
    print(
        f"device={device or 'default'} clips={len(clips)} models={len(models)} "
        f"imgsz={imgsz_values} conf={conf_values} detector_output_limits={detector_output_limits}"
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model_spec in models:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for the local person accuracy sweep") from exc
        print(f"loading detector {model_spec.name}: {model_spec.path}")
        model_load_start = time.perf_counter()
        model = YOLO(str(model_spec.path))
        model_load_ms = (time.perf_counter() - model_load_start) * 1000.0
        for clip in clips:
            for prune_mode in prune_modes:
                margin_values: Sequence[float | None] = court_margins if prune_mode == "court" else [None]
                for court_margin_m in margin_values:
                    for bbox_expand in bbox_expands:
                        for detector_output_limit in detector_output_limits:
                            for imgsz in imgsz_values:
                                for conf in conf_values:
                                    prune_token = _prune_token(prune_mode, court_margin_m)
                                    bbox_token = _bbox_token(bbox_expand)
                                    limit_token = _candidate_limit_token(detector_output_limit)
                                    base_name = _candidate_name(
                                        model_spec.name,
                                        imgsz,
                                        conf,
                                        args.iou,
                                        f"{prune_token}_{bbox_token}_{limit_token}_detect_cache",
                                    )
                                    try:
                                        detection = _load_or_collect_detections(
                                            model=model,
                                            clip=clip,
                                            model_spec=model_spec,
                                            out_dir=out_dir,
                                            base_name=base_name,
                                            imgsz=imgsz,
                                            conf=conf,
                                            iou=args.iou,
                                            device=device,
                                            max_players=args.max_players,
                                            detector_output_limit=detector_output_limit,
                                            max_frames=args.max_frames,
                                            model_load_ms=model_load_ms,
                                            prune_mode=prune_mode,
                                            court_margin_m=court_margin_m,
                                            bbox_expand=bbox_expand,
                                            force=args.force,
                                        )
                                    except Exception as exc:
                                        failure = _failure_row(
                                            candidate=base_name,
                                            clip=clip.clip_id,
                                            model=model_spec.name,
                                            imgsz=imgsz,
                                            conf=conf,
                                            iou=args.iou,
                                            tracker=f"{prune_token}_{bbox_token}_{limit_token}_detect_cache",
                                            error=exc,
                                        )
                                        failures.append(failure)
                                        print(f"FAILED {clip.clip_id} {base_name}: {exc}")
                                        continue

                                    for linker_name in linkers:
                                        candidate_name = _candidate_name(
                                            model_spec.name,
                                            imgsz,
                                            conf,
                                            args.iou,
                                            f"{prune_token}_{bbox_token}_{limit_token}_{linker_name}",
                                        )
                                        try:
                                            row = _score_cached_linker(
                                                clip=clip,
                                                out_dir=out_dir,
                                                candidate_name=candidate_name,
                                                detector=detection,
                                                linker_name=linker_name,
                                                max_players=args.max_players,
                                            )
                                            rows.append(row)
                                            print(
                                                "{clip} {candidate} IDF1={idf1:.3f} MOTA={mota:.3f} cov4={coverage:.3f} sw={switches}".format(
                                                    clip=clip.clip_id,
                                                    candidate=candidate_name,
                                                    idf1=row["idf1"],
                                                    mota=row["mota"],
                                                    coverage=row["expected_player_coverage"],
                                                    switches=row["id_switches"],
                                                )
                                            )
                                        except Exception as exc:
                                            failure = _failure_row(
                                                candidate=candidate_name,
                                                clip=clip.clip_id,
                                                model=model_spec.name,
                                                imgsz=imgsz,
                                                conf=conf,
                                                iou=args.iou,
                                                tracker=f"{prune_token}_{bbox_token}_{limit_token}_{linker_name}",
                                                error=exc,
                                            )
                                            failures.append(failure)
                                            print(f"FAILED {clip.clip_id} {candidate_name}: {exc}")

                                    for closed_set_mode in closed_set_modes:
                                        for linker_name in linkers:
                                            candidate_name = _candidate_name(
                                                model_spec.name,
                                                imgsz,
                                                conf,
                                                args.iou,
                                                f"{prune_token}_{bbox_token}_{limit_token}_{linker_name}_closed_{closed_set_mode}",
                                            )
                                            try:
                                                row = _score_cached_linker(
                                                    clip=clip,
                                                    out_dir=out_dir,
                                                    candidate_name=candidate_name,
                                                    detector=detection,
                                                    linker_name=linker_name,
                                                    max_players=args.max_players,
                                                    linker_max_players=detector_output_limit,
                                                    closed_set_mode=closed_set_mode,
                                                )
                                                rows.append(row)
                                                print(
                                                    "{clip} {candidate} IDF1={idf1:.3f} MOTA={mota:.3f} cov4={coverage:.3f} sw={switches}".format(
                                                        clip=clip.clip_id,
                                                        candidate=candidate_name,
                                                        idf1=row["idf1"],
                                                        mota=row["mota"],
                                                        coverage=row["expected_player_coverage"],
                                                        switches=row["id_switches"],
                                                    )
                                                )
                                            except Exception as exc:
                                                failure = _failure_row(
                                                    candidate=candidate_name,
                                                    clip=clip.clip_id,
                                                    model=model_spec.name,
                                                    imgsz=imgsz,
                                                    conf=conf,
                                                    iou=args.iou,
                                                    tracker=f"{prune_token}_{bbox_token}_{limit_token}_{linker_name}_closed_{closed_set_mode}",
                                                    error=exc,
                                                )
                                                failures.append(failure)
                                                print(f"FAILED {clip.clip_id} {candidate_name}: {exc}")

    if not args.skip_ultralytics_track:
        for model_spec in track_models:
            for clip in clips:
                for imgsz in track_imgsz_values:
                    for conf in track_conf_values:
                        for track_mode in track_modes:
                            candidate_name = _candidate_name(model_spec.name, imgsz, conf, args.iou, track_mode)
                            candidate_dir = out_dir / clip.clip_id / candidate_name
                            try:
                                row = _run_or_load_ultralytics_track(
                                    clip=clip,
                                    model_spec=model_spec,
                                    candidate_name=candidate_name,
                                    out_dir=candidate_dir,
                                    imgsz=imgsz,
                                    conf=conf,
                                    iou=args.iou,
                                    device=device,
                                    max_players=args.max_players,
                                    max_frames=args.max_frames,
                                    tracker=track_mode,
                                    force=args.force,
                                )
                                rows.append(row)
                                print(
                                    "{clip} {candidate} IDF1={idf1:.3f} MOTA={mota:.3f} cov4={coverage:.3f} sw={switches}".format(
                                        clip=clip.clip_id,
                                        candidate=candidate_name,
                                        idf1=row["idf1"],
                                        mota=row["mota"],
                                        coverage=row["expected_player_coverage"],
                                        switches=row["id_switches"],
                                    )
                                )
                            except Exception as exc:
                                failure = _failure_row(
                                    candidate=candidate_name,
                                    clip=clip.clip_id,
                                    model=model_spec.name,
                                    imgsz=imgsz,
                                    conf=conf,
                                    iou=args.iou,
                                    tracker=track_mode,
                                    error=exc,
                                )
                                failures.append(failure)
                                print(f"FAILED {clip.clip_id} {candidate_name}: {exc}")

    leaderboard = _aggregate_rows(rows)
    status = _sweep_status(rows, failures)
    _write_outputs(out_dir, clips=clips, rows=rows, leaderboard=leaderboard, failures=failures, args=vars(args))
    _render_top_overlays(out_dir, clips=clips, leaderboard=leaderboard, top_n=args.top_overlays)
    print(f"leaderboard: {out_dir / 'leaderboard.csv'}")
    print(f"report: {out_dir / 'REPORT.md'}")
    if status == "partial" and not args.allow_partial:
        print("sweep status=partial; rerun with --allow-partial to accept exploratory partial results")
        return 2
    return 0 if status in {"pass", "partial"} else 2


def _load_or_collect_detections(
    *,
    model: Any,
    clip: ClipSpec,
    model_spec: ModelSpec,
    out_dir: Path,
    base_name: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
    max_players: int,
    detector_output_limit: int,
    max_frames: int | None,
    model_load_ms: float,
    prune_mode: str,
    court_margin_m: float | None,
    bbox_expand: float,
    force: bool,
) -> dict[str, Any]:
    cache_dir = out_dir / "_detections" / clip.clip_id / base_name
    cache_path = cache_dir / "observations.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for local person accuracy sweeps") from exc

    fps, width, height, video_frame_count = _video_properties(cv2, clip.video_path)
    frame_limit = min(video_frame_count, max_frames) if max_frames is not None else video_frame_count
    frames: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    calibration = _load_court_calibration(clip.court_calibration_path) if prune_mode == "court" else None
    started = time.perf_counter()
    iterator = model.predict(
        source=str(clip.video_path),
        stream=True,
        classes=[0],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    try:
        for frame_index, result in enumerate(iterator):
            if max_frames is not None and frame_index >= max_frames:
                break
            observations = _observations_from_result(
                result,
                max_players=max_players,
                prune_mode=prune_mode,
                court_calibration=calibration,
                court_margin_m=1.25 if court_margin_m is None else court_margin_m,
                output_limit=detector_output_limit,
                bbox_expand=bbox_expand,
            )
            latency_ms = _latency_ms_from_result(result)
            samples.append({"frame_index": frame_index, "latency_ms": latency_ms, "processed": True})
            frames.append({"frame_index": frame_index, "observations": observations})
            if frame_index + 1 >= frame_limit:
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    wall_clock_seconds = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_mobile_person_detection_cache",
        "clip_id": clip.clip_id,
        "model": str(model_spec.path),
        "model_name": model_spec.name,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": device,
        "max_players": max_players,
        "detector_output_limit": detector_output_limit,
        "prune_mode": prune_mode,
        "court_calibration_path": str(clip.court_calibration_path) if clip.court_calibration_path is not None else None,
        "court_margin_m": court_margin_m,
        "bbox_expand": bbox_expand,
        "fps": fps,
        "resolution": [width, height],
        "video_frame_count": video_frame_count,
        "processed_frame_count": len(frames),
        "dropped_frame_count": max(0, frame_limit - len(frames)),
        "model_load_ms": model_load_ms,
        "mlpackage_size_mb": _package_size_mb(str(model_spec.path)),
        "wall_clock_seconds": wall_clock_seconds,
        "samples": samples,
        "frames": frames,
    }
    _write_json(cache_path, payload)
    return payload


def _score_cached_linker(
    *,
    clip: ClipSpec,
    out_dir: Path,
    candidate_name: str,
    detector: dict[str, Any],
    linker_name: str,
    max_players: int,
    linker_max_players: int | None = None,
    closed_set_mode: str | None = None,
) -> dict[str, Any]:
    gt = validate_artifact_file("person_ground_truth", clip.ground_truth_path)
    if not isinstance(gt, PersonGroundTruth):
        raise ValueError("ground truth artifact did not parse as PersonGroundTruth")
    candidate_dir = out_dir / clip.clip_id / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    active_linker_max = max_players if linker_max_players is None else int(linker_max_players)
    linker = _make_linker(linker_name, max_players=active_linker_max)
    frames: list[dict[str, Any]] = []
    for frame in detector["frames"]:
        detections = linker.update(frame_index=int(frame["frame_index"]), observations=list(frame["observations"]))
        frames.append({"frame_index": int(frame["frame_index"]), "detections": detections})
    if closed_set_mode is not None:
        frames = _closed_set_prune_frames(
            frames,
            max_players=max_players,
            mode=closed_set_mode,
            frame_width=float(detector["resolution"][0]),
            frame_height=float(detector["resolution"][1]),
        )

    tracks_payload = _tracks_payload(
        clip_id=gt.clip_id,
        candidate=candidate_name,
        fps=float(detector["fps"]),
        width=int(detector["resolution"][0]),
        height=int(detector["resolution"][1]),
        frames=frames,
    )
    timing_payload = _timing_payload(
        clip_id=gt.clip_id,
        candidate=candidate_name,
        wall_clock_seconds=float(detector["wall_clock_seconds"]),
        frame_count=len(frames),
        dropped_frame_count=int(detector["dropped_frame_count"]),
        model_load_ms=float(detector["model_load_ms"]),
        mlpackage_size_mb=detector.get("mlpackage_size_mb"),
        samples=list(detector["samples"]),
    )
    tracks_path = candidate_dir / "on_device_person_tracks.json"
    timing_path = candidate_dir / "timing.json"
    _write_json(tracks_path, tracks_payload)
    _write_json(timing_path, timing_payload)

    predictions = OnDevicePersonTracks.model_validate(tracks_payload)
    metrics = score_mobile_person_tracks(gt, predictions, expected_players=max_players)
    metrics_path = candidate_dir / "metrics.json"
    write_mobile_person_metrics(metrics_path, metrics)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_mobile_person_accuracy_sweep_run",
        "clip_id": gt.clip_id,
        "candidate": candidate_name,
        "model": detector["model"],
        "model_name": detector["model_name"],
        "imgsz": detector["imgsz"],
        "conf": detector["conf"],
        "iou": detector["iou"],
        "device": detector["device"],
        "tracker": linker_name,
        "linker_max_players": active_linker_max,
        "detector_output_limit": detector.get("detector_output_limit", max_players),
        "closed_set_mode": closed_set_mode,
        "prune_mode": detector.get("prune_mode"),
        "court_calibration_path": detector.get("court_calibration_path"),
        "court_margin_m": detector.get("court_margin_m"),
        "bbox_expand": detector.get("bbox_expand", 1.0),
        "video_path": str(clip.video_path),
        "ground_truth_path": str(clip.ground_truth_path),
        "tracks_path": str(tracks_path),
        "timing_path": str(timing_path),
        "metrics_path": str(metrics_path),
        "overlay_path": None,
        "metrics": metrics.model_dump(mode="json"),
        "timing": timing_payload["summary"],
        "cached_detection_runtime": True,
    }
    _write_json(candidate_dir / "run_summary.json", summary)
    return _row_from_summary(summary)


def _run_or_load_ultralytics_track(
    *,
    clip: ClipSpec,
    model_spec: ModelSpec,
    candidate_name: str,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
    max_players: int,
    max_frames: int | None,
    tracker: str,
    force: bool,
) -> dict[str, Any]:
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return _row_from_summary(json.loads(summary_path.read_text(encoding="utf-8")))
    summary = run_replay_yolo_candidate(
        video_path=clip.video_path,
        ground_truth_path=clip.ground_truth_path,
        candidate=ReplayYoloCandidate(
            name=candidate_name,
            model=str(model_spec.path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_players=max_players,
            tracker=tracker,
        ),
        out_dir=out_dir,
        max_frames=max_frames,
        render_overlay=False,
    )
    summary["model_name"] = model_spec.name
    _write_json(summary_path, summary)
    return _row_from_summary(summary)


def _row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    timing = summary["timing"]
    return {
        "candidate": summary["candidate"],
        "clip_id": summary["clip_id"],
        "model": summary.get("model_name") or _model_name_from_path(str(summary.get("model", ""))),
        "model_path": summary.get("model"),
        "imgsz": int(summary["imgsz"]),
        "conf": float(summary["conf"]),
        "iou": float(summary["iou"]),
        "tracker": summary.get("tracker", ""),
        "linker_max_players": int(summary.get("linker_max_players", summary.get("detector_output_limit", 0)) or 0),
        "detector_output_limit": int(summary.get("detector_output_limit", summary.get("max_players", 0)) or 0),
        "closed_set_mode": summary.get("closed_set_mode"),
        "prune_mode": summary.get("prune_mode", "confidence"),
        "court_margin_m": summary.get("court_margin_m"),
        "bbox_expand": float(summary.get("bbox_expand", 1.0)),
        "idf1": float(metrics["idf1"]),
        "mota": float(metrics["mota"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "expected_player_coverage": float(metrics["expected_player_coverage"]),
        "id_switches": int(metrics["id_switches"]),
        "false_positives": int(metrics["false_positives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "matches": int(metrics["matches"]),
        "gt_detections": int(metrics["gt_detections"]),
        "pred_detections": int(metrics["pred_detections"]),
        "processed_fps": float(timing.get("sustained_processed_fps", 0.0)),
        "p50_latency_ms": float(timing.get("p50_latency_ms", 0.0)),
        "p95_latency_ms": float(timing.get("p95_latency_ms", 0.0)),
        "tracks_path": summary["tracks_path"],
        "timing_path": summary["timing_path"],
        "metrics_path": summary["metrics_path"],
        "overlay_path": summary.get("overlay_path"),
        "cached_detection_runtime": bool(summary.get("cached_detection_runtime", False)),
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    leaderboard: list[dict[str, Any]] = []
    for candidate, candidate_rows in by_candidate.items():
        first = candidate_rows[0]
        detector_output_limit = int(first.get("detector_output_limit", first.get("max_players", 0)) or 0)
        linker_max_players = int(first.get("linker_max_players", detector_output_limit) or 0)
        leaderboard.append(
            {
                "candidate": candidate,
                "model": first["model"],
                "imgsz": first["imgsz"],
                "conf": first["conf"],
                "iou": first["iou"],
                "tracker": first["tracker"],
                "linker_max_players": linker_max_players,
                "detector_output_limit": detector_output_limit,
                "closed_set_mode": first.get("closed_set_mode"),
                "prune_mode": first["prune_mode"],
                "court_margin_m": first["court_margin_m"],
                "bbox_expand": first["bbox_expand"],
                "clip_count": len(candidate_rows),
                "mean_idf1": _mean(row["idf1"] for row in candidate_rows),
                "worst_idf1": min(row["idf1"] for row in candidate_rows),
                "mean_mota": _mean(row["mota"] for row in candidate_rows),
                "worst_mota": min(row["mota"] for row in candidate_rows),
                "mean_precision": _mean(row["precision"] for row in candidate_rows),
                "mean_recall": _mean(row["recall"] for row in candidate_rows),
                "mean_expected_player_coverage": _mean(row["expected_player_coverage"] for row in candidate_rows),
                "worst_expected_player_coverage": min(row["expected_player_coverage"] for row in candidate_rows),
                "total_id_switches": sum(row["id_switches"] for row in candidate_rows),
                "total_false_positives": sum(row["false_positives"] for row in candidate_rows),
                "total_false_negatives": sum(row["false_negatives"] for row in candidate_rows),
                "mean_processed_fps": _mean(row["processed_fps"] for row in candidate_rows),
                "mean_p95_latency_ms": _mean(row["p95_latency_ms"] for row in candidate_rows),
                "cached_detection_runtime": all(bool(row.get("cached_detection_runtime", False)) for row in candidate_rows),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            row["mean_idf1"],
            row["worst_idf1"],
            row["mean_expected_player_coverage"],
            row["mean_mota"],
            -row["total_id_switches"],
            -row["total_false_negatives"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank
    return leaderboard


def _write_outputs(
    out_dir: Path,
    *,
    clips: Sequence[ClipSpec],
    rows: Sequence[dict[str, Any]],
    leaderboard: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    args: dict[str, Any],
) -> None:
    status = _sweep_status(rows, failures)
    _write_json(
        out_dir / "sweep_summary.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_mobile_person_accuracy_sweep",
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "video_path": str(clip.video_path),
                    "ground_truth_path": str(clip.ground_truth_path),
                    "court_calibration_path": str(clip.court_calibration_path)
                    if clip.court_calibration_path is not None
                    else None,
                }
                for clip in clips
            ],
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in args.items()},
            "row_count": len(rows),
            "candidate_count": len(leaderboard),
            "failure_count": len(failures),
            "leaderboard_path": str(out_dir / "leaderboard.json"),
            "per_clip_rows_path": str(out_dir / "per_clip_metrics.json"),
            "failures_path": str(out_dir / "failures.json"),
        },
    )
    _write_json(out_dir / "leaderboard.json", list(leaderboard))
    _write_json(out_dir / "per_clip_metrics.json", list(rows))
    _write_json(out_dir / "failures.json", list(failures))
    _write_csv(out_dir / "leaderboard.csv", leaderboard)
    _write_csv(out_dir / "per_clip_metrics.csv", rows)
    (out_dir / "REPORT.md").write_text(_render_report(clips, leaderboard, rows, failures, status=status), encoding="utf-8")
    _try_write_chart(out_dir / "leaderboard_top20.png", leaderboard)


def _render_top_overlays(out_dir: Path, *, clips: Sequence[ClipSpec], leaderboard: Sequence[dict[str, Any]], top_n: int) -> None:
    if top_n <= 0:
        return
    top_candidates = [str(row["candidate"]) for row in leaderboard[:top_n]]
    for clip in clips:
        for candidate in top_candidates:
            tracks_path = out_dir / clip.clip_id / candidate / "on_device_person_tracks.json"
            overlay_path = out_dir / clip.clip_id / candidate / "track_overlay.mp4"
            if not tracks_path.exists() or overlay_path.exists():
                continue
            try:
                render_replay_yolo_overlay(video_path=clip.video_path, tracks_path=tracks_path, output_path=overlay_path)
                summary_path = out_dir / clip.clip_id / candidate / "run_summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["overlay_path"] = str(overlay_path)
                    _write_json(summary_path, summary)
            except Exception as exc:
                print(f"overlay failed for {clip.clip_id} {candidate}: {exc}")


def _render_report(
    clips: Sequence[ClipSpec],
    leaderboard: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    *,
    status: str,
) -> str:
    lines = [
        "# Mobile Person Local Accuracy Sweep",
        "",
        "This is a laptop-local accuracy sweep against the currently available CVAT MOT-derived annotations.",
        "It is not an iPhone speed result. Cached detector candidates reuse one detector pass across multiple ID linkers.",
        "",
        "## Summary",
        "",
        f"- Status: `{status}`",
        f"- Per-clip rows: `{len(rows)}`",
        f"- Failed candidates: `{len(failures)}`",
        "",
        "## Clips",
        "",
        "| Clip | Video | GT |",
        "| --- | --- | --- |",
    ]
    for clip in clips:
        lines.append(f"| `{clip.clip_id}` | `{clip.video_path}` | `{clip.ground_truth_path}` |")
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| Rank | Candidate | Mean IDF1 | Worst IDF1 | Mean MOTA | Mean cov4 | Switches | FN | FP | FPS |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in leaderboard[:25]:
        lines.append(
            "| {rank} | `{candidate}` | {mean_idf1:.3f} | {worst_idf1:.3f} | {mean_mota:.3f} | "
            "{coverage:.3f} | {switches} | {fn} | {fp} | {fps:.1f} |".format(
                rank=row["rank"],
                candidate=row["candidate"],
                mean_idf1=row["mean_idf1"],
                worst_idf1=row["worst_idf1"],
                mean_mota=row["mean_mota"],
                coverage=row["mean_expected_player_coverage"],
                switches=row["total_id_switches"],
                fn=row["total_false_negatives"],
                fp=row["total_false_positives"],
                fps=row["mean_processed_fps"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Clip Top Rows",
            "",
            "| Clip | Candidate | IDF1 | MOTA | cov4 | switches | FN | FP |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["clip_id"], -item["idf1"], -item["expected_player_coverage"]))[:40]:
        lines.append(
            "| `{clip}` | `{candidate}` | {idf1:.3f} | {mota:.3f} | {coverage:.3f} | {switches} | {fn} | {fp} |".format(
                clip=row["clip_id"],
                candidate=row["candidate"],
                idf1=row["idf1"],
                mota=row["mota"],
                coverage=row["expected_player_coverage"],
                switches=row["id_switches"],
                fn=row["false_negatives"],
                fp=row["false_positives"],
            )
        )
    if failures:
        lines.extend(["", "## Failures", "", "| Candidate | Clip | Error |", "| --- | --- | --- |"])
        for failure in failures[:30]:
            lines.append(f"| `{failure['candidate']}` | `{failure['clip_id']}` | `{failure['error']}` |")
    lines.append("")
    return "\n".join(lines)


def _sweep_status(rows: Sequence[dict[str, Any]], failures: Sequence[dict[str, Any]]) -> str:
    if failures and rows:
        return "partial"
    if failures:
        return "failed"
    if rows:
        return "pass"
    return "failed"


def _try_write_chart(path: Path, leaderboard: Sequence[dict[str, Any]]) -> None:
    if not leaderboard:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return
    top = list(leaderboard[:20])
    labels = [str(row["candidate"]) for row in top]
    mean = [float(row["mean_idf1"]) for row in top]
    worst = [float(row["worst_idf1"]) for row in top]
    y = list(range(len(top)))
    fig_height = max(6.0, len(top) * 0.38)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.barh([value - 0.18 for value in y], mean, height=0.32, label="Mean IDF1", color="#2f6f9f")
    ax.barh([value + 0.18 for value in y], worst, height=0.32, label="Worst IDF1", color="#d77a35")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("IDF1")
    ax.set_title("Top Mobile Person Accuracy Candidates")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_clips(specs: Sequence[str]) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 3)
        if len(parts) not in {3, 4}:
            raise ValueError(f"clip spec must be clip_id=video=ground_truth[=court_calibration]: {spec}")
        clip_id, video, gt = parts[:3]
        calibration = parts[3] if len(parts) == 4 else None
        if not clip_id:
            raise ValueError(f"clip spec missing clip_id: {spec}")
        clip = ClipSpec(
            clip_id=clip_id,
            video_path=Path(video),
            ground_truth_path=Path(gt),
            court_calibration_path=Path(calibration) if calibration else None,
        )
        if not clip.video_path.is_file():
            raise FileNotFoundError(f"missing video for {clip_id}: {clip.video_path}")
        if not clip.ground_truth_path.is_file():
            raise FileNotFoundError(f"missing ground truth for {clip_id}: {clip.ground_truth_path}")
        if clip.court_calibration_path is not None and not clip.court_calibration_path.is_file():
            raise FileNotFoundError(f"missing court calibration for {clip_id}: {clip.court_calibration_path}")
        clips.append(clip)
    return clips


def _parse_models(specs: Sequence[str]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = _model_name_from_path(path)
        model = ModelSpec(name=_safe_token(name), path=Path(path))
        if not model.path.exists():
            raise FileNotFoundError(f"missing model for {model.name}: {model.path}")
        models.append(model)
    return models


def _candidate_name(model: str, imgsz: int, conf: float, iou: float, tracker: str) -> str:
    return "_".join([_safe_token(model), f"img{imgsz}", f"conf{_number_token(conf)}", f"iou{_number_token(iou)}", _safe_token(tracker)])


def _prune_token(prune_mode: str, court_margin_m: float | None) -> str:
    if prune_mode == "confidence":
        return "confprune"
    if prune_mode == "court":
        if court_margin_m is None:
            raise ValueError("court prune token requires court_margin_m")
        return f"courtprune_m{_number_token(court_margin_m)}"
    raise ValueError(f"unsupported prune mode: {prune_mode}")


def _bbox_token(bbox_expand: float) -> str:
    return f"bboxe{_number_token(bbox_expand)}"


def _candidate_limit_token(detector_output_limit: int) -> str:
    if detector_output_limit <= 0:
        raise ValueError("detector output limit must be positive")
    return f"cand{int(detector_output_limit)}"


def _load_court_calibration(path: Path | None) -> Any:
    if path is None:
        raise ValueError("court pruning requires a per-clip court calibration path")
    from threed.racketsport.mobile_person_yolo_replay import _load_court_calibration as load

    return load(path)


def _failure_row(
    *,
    candidate: str,
    clip: str,
    model: str,
    imgsz: int,
    conf: float,
    iou: float,
    tracker: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "clip_id": clip,
        "model": model,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "tracker": tracker,
        "error": str(error),
        "error_type": type(error).__name__,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Iterable[float | int]) -> float:
    data = [float(value) for value in values]
    return sum(data) / len(data) if data else 0.0


def _unique_ints(values: Sequence[int]) -> list[int]:
    result = sorted({int(value) for value in values})
    if not result:
        raise ValueError("at least one integer value is required")
    return result


def _unique_floats(values: Sequence[float]) -> list[float]:
    result = sorted({float(value) for value in values})
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("float values must be finite")
    return result


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


def _model_name_from_path(path: str) -> str:
    return _safe_token(Path(path).stem or "model")


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
