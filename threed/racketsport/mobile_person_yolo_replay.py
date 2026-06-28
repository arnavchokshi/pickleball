"""Run YOLO person replay candidates into the mobile person tracking schema."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mobile_person_eval import score_mobile_person_tracks, write_mobile_person_metrics
from .schemas import OnDevicePersonTracks, PersonGroundTruth, validate_artifact_file


@dataclass(frozen=True)
class ReplayYoloCandidate:
    name: str
    model: str
    imgsz: int
    conf: float
    iou: float
    device: str | None = None
    max_players: int = 4


def run_replay_yolo_candidate(
    *,
    video_path: str | Path,
    ground_truth_path: str | Path,
    candidate: ReplayYoloCandidate,
    out_dir: str | Path,
    max_frames: int | None = None,
    render_overlay: bool = True,
) -> dict[str, Any]:
    """Run a YOLO detector over a replay video and score against person ground truth."""

    try:
        import cv2  # type: ignore[import-not-found]
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and ultralytics are required for YOLO replay evaluation") from exc

    gt = validate_artifact_file("person_ground_truth", Path(ground_truth_path))
    if not isinstance(gt, PersonGroundTruth):
        raise ValueError("ground truth artifact did not parse as PersonGroundTruth")

    video = Path(video_path)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    fps, width, height, video_frame_count = _video_properties(cv2, video)
    frame_limit = min(video_frame_count, max_frames) if max_frames is not None else video_frame_count

    model_load_start = time.perf_counter()
    model = YOLO(candidate.model)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0

    linker = _IoUPersonLinker(max_tracks=candidate.max_players)
    frames: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    iterator = model.predict(
        source=str(video),
        stream=True,
        classes=[0],
        conf=candidate.conf,
        iou=candidate.iou,
        imgsz=candidate.imgsz,
        device=candidate.device,
        verbose=False,
    )
    try:
        for frame_index, result in enumerate(iterator):
            if max_frames is not None and frame_index >= max_frames:
                break
            observations = _observations_from_result(result, max_players=candidate.max_players)
            detections = linker.update(frame_index=frame_index, observations=observations)
            latency_ms = _latency_ms_from_result(result)
            samples.append(
                {
                    "frame_index": frame_index,
                    "latency_ms": latency_ms,
                    "processed": True,
                }
            )
            frames.append({"frame_index": frame_index, "detections": detections})
            if frame_index + 1 >= frame_limit:
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    wall_clock_seconds = time.perf_counter() - started
    tracks_payload = _tracks_payload(
        clip_id=gt.clip_id,
        candidate=candidate.name,
        fps=fps,
        width=width,
        height=height,
        frames=frames,
    )
    timing_payload = _timing_payload(
        clip_id=gt.clip_id,
        candidate=candidate.name,
        wall_clock_seconds=wall_clock_seconds,
        frame_count=len(frames),
        dropped_frame_count=max(0, frame_limit - len(frames)),
        model_load_ms=model_load_ms,
        mlpackage_size_mb=_package_size_mb(candidate.model),
        samples=samples,
    )
    tracks_path = output / "on_device_person_tracks.json"
    timing_path = output / "timing.json"
    tracks_path.write_text(json.dumps(tracks_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timing_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    predictions = validate_artifact_file("on_device_person_tracks", tracks_path)
    if not isinstance(predictions, OnDevicePersonTracks):
        raise ValueError("prediction artifact did not parse as OnDevicePersonTracks")
    metrics = score_mobile_person_tracks(gt, predictions, expected_players=4)
    metrics_path = output / "metrics.json"
    write_mobile_person_metrics(metrics_path, metrics)

    overlay_path: Path | None = None
    if render_overlay:
        overlay_path = output / "track_overlay.mp4"
        _render_overlay(cv2, video, tracks_payload, overlay_path, max_frames=len(frames))

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_mobile_yolo_replay_run",
        "clip_id": gt.clip_id,
        "candidate": candidate.name,
        "model": candidate.model,
        "imgsz": candidate.imgsz,
        "conf": candidate.conf,
        "iou": candidate.iou,
        "device": candidate.device,
        "video_path": str(video),
        "ground_truth_path": str(ground_truth_path),
        "tracks_path": str(tracks_path),
        "timing_path": str(timing_path),
        "metrics_path": str(metrics_path),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
        "metrics": metrics.model_dump(mode="json"),
        "timing": timing_payload["summary"],
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def render_replay_yolo_overlay(
    *,
    video_path: str | Path,
    tracks_path: str | Path,
    output_path: str | Path,
    max_frames: int | None = None,
) -> Path:
    """Render an overlay from saved mobile person-track predictions."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for rendering replay overlays") from exc

    tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    frame_count = len(tracks.get("frames", []))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    out = Path(output_path)
    _render_overlay(cv2, Path(video_path), tracks, out, max_frames=frame_count)
    return out


class _IoUPersonLinker:
    def __init__(self, *, max_tracks: int = 4, iou_threshold: float = 0.3, max_age_frames: int = 10) -> None:
        self.max_tracks = max_tracks
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self._next_id = 1
        self._tracks: list[dict[str, Any]] = []

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._tracks = [track for track in self._tracks if frame_index - int(track["last_frame_index"]) <= self.max_age_frames]
        detections: list[dict[str, Any]] = []
        used_track_indexes: set[int] = set()
        for observation in observations[: self.max_tracks]:
            match_index = self._best_track_index(observation["bbox_xywh"], used_track_indexes)
            if match_index is None:
                track = {
                    "id": self._next_id,
                    "bbox_xywh": observation["bbox_xywh"],
                    "last_frame_index": frame_index,
                }
                self._next_id += 1
                self._tracks.append(track)
                match_index = len(self._tracks) - 1
            else:
                self._tracks[match_index]["bbox_xywh"] = observation["bbox_xywh"]
                self._tracks[match_index]["last_frame_index"] = frame_index
            used_track_indexes.add(match_index)
            detections.append(
                {
                    "track_id": int(self._tracks[match_index]["id"]),
                    "bbox_xywh": [float(value) for value in observation["bbox_xywh"]],
                    "confidence": float(observation["confidence"]),
                    "source": observation["source"],
                    "role": None,
                }
            )
        return sorted(detections, key=lambda item: int(item["track_id"]))

    def _best_track_index(self, bbox_xywh: list[float], used_track_indexes: set[int]) -> int | None:
        best_index: int | None = None
        best_iou = self.iou_threshold
        for index, track in enumerate(self._tracks):
            if index in used_track_indexes:
                continue
            score = _bbox_iou(bbox_xywh, track["bbox_xywh"])
            if score >= best_iou:
                best_index = index
                best_iou = score
        return best_index


def _observations_from_result(result: Any, *, max_players: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    observations: list[dict[str, Any]] = []
    for raw_box, confidence in zip(xyxy, conf, strict=False):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        observations.append(
            {
                "bbox_xywh": [x1, y1, width, height],
                "confidence": max(0.0, min(1.0, float(confidence))),
                "source": "yolo_person",
            }
        )
    observations.sort(key=lambda item: (-float(item["confidence"]), item["bbox_xywh"][0], item["bbox_xywh"][1]))
    return observations[:max_players]


def _latency_ms_from_result(result: Any) -> float:
    speed = getattr(result, "speed", None)
    if isinstance(speed, dict):
        values = [float(value) for value in speed.values() if isinstance(value, (int, float))]
        if values:
            return max(0.0, sum(values))
    return 0.0


def _tracks_payload(
    *,
    clip_id: str,
    candidate: str,
    fps: float,
    width: int,
    height: int,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    track_ids = sorted({int(det["track_id"]) for frame in frames for det in frame["detections"]})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_on_device_person_tracks",
        "clip_id": clip_id,
        "candidate": candidate,
        "device_model": None,
        "coordinate_space": "source_video_pixels",
        "resolution": [width, height],
        "fps": fps,
        "frames": frames,
        "summary": {
            "frame_count": len(frames),
            "detection_count": sum(len(frame["detections"]) for frame in frames),
            "track_ids": track_ids,
        },
    }


def _timing_payload(
    *,
    clip_id: str,
    candidate: str,
    wall_clock_seconds: float,
    frame_count: int,
    dropped_frame_count: int,
    model_load_ms: float,
    mlpackage_size_mb: float | None,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    latencies = sorted(float(sample["latency_ms"]) for sample in samples if sample["processed"])
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_on_device_person_timing",
        "clip_id": clip_id,
        "candidate": candidate,
        "mode": "replay",
        "device_model": "mac_replay_not_iphone",
        "os_version": None,
        "wall_clock_seconds": wall_clock_seconds,
        "dropped_frame_count": dropped_frame_count,
        "model_load_ms": model_load_ms,
        "mlpackage_size_mb": mlpackage_size_mb,
        "started_thermal_state": None,
        "ended_thermal_state": None,
        "samples": samples,
        "summary": {
            "processed_frame_count": frame_count,
            "dropped_frame_count": dropped_frame_count,
            "sustained_processed_fps": (frame_count / wall_clock_seconds) if wall_clock_seconds > 0 else 0.0,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
        },
    }


def _render_overlay(cv2: Any, video_path: Path, tracks_payload: dict[str, Any], output_path: Path, *, max_frames: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or tracks_payload["fps"]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or tracks_payload["resolution"][0]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or tracks_payload["resolution"][1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open overlay writer: {output_path}")
    frames = {int(frame["frame_index"]): frame["detections"] for frame in tracks_payload["frames"]}
    try:
        frame_index = 0
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            for detection in frames.get(frame_index, []):
                x, y, w, h = detection["bbox_xywh"]
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w)), int(round(y + h))
                color = _color_for_id(int(detection["track_id"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {detection['track_id']} {float(detection['confidence']):.2f}"
                cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def _video_properties(cv2: Any, video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError(f"could not read video properties: {video_path}")
    return fps, width, height, frames


def _package_size_mb(model: str) -> float | None:
    path = Path(model)
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size / 1_000_000.0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1_000_000.0


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * fraction))))
    return float(sorted_values[index])


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [(60, 220, 255), (80, 200, 80), (255, 180, 80), (220, 120, 255), (255, 255, 80), (80, 120, 255)]
    return palette[track_id % len(palette)]


__all__ = ["ReplayYoloCandidate", "render_replay_yolo_overlay", "run_replay_yolo_candidate"]
