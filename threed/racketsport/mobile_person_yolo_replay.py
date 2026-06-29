"""Run YOLO person replay candidates into the mobile person tracking schema."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

from .court_calibration import project_image_points_to_world
from .court_templates import get_court_template
from .mobile_person_eval import score_mobile_person_tracks, write_mobile_person_metrics
from .schemas import CourtCalibration, OnDevicePersonTracks, PersonGroundTruth, validate_artifact_file


@dataclass(frozen=True)
class ReplayYoloCandidate:
    name: str
    model: str
    imgsz: int
    conf: float
    iou: float
    device: str | None = None
    max_players: int = 4
    tracker: str = "predict_iou"
    tracker_config: str | None = None
    link_iou_threshold: float | None = None
    max_age_frames: int | None = None
    prune_mode: str = "confidence"
    court_calibration: str | None = None
    court_margin_m: float = 1.25
    bbox_expand: float = 1.0


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

    frames: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    court_calibration = _load_court_calibration(candidate.court_calibration) if candidate.court_calibration else None
    if candidate.tracker.startswith("track_"):
        iterator = model.track(
            source=str(video),
            stream=True,
            classes=[0],
            conf=candidate.conf,
            iou=candidate.iou,
            imgsz=candidate.imgsz,
            device=candidate.device,
            tracker=_tracker_config_path(candidate),
            persist=True,
            verbose=False,
        )
        try:
            for frame_index, result in enumerate(iterator):
                if max_frames is not None and frame_index >= max_frames:
                    break
                detections = _tracked_detections_from_result(result, max_players=candidate.max_players)
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
    else:
        linker = _make_linker(
            candidate.tracker,
            max_players=candidate.max_players,
            iou_threshold=candidate.link_iou_threshold,
            max_age_frames=candidate.max_age_frames,
        )
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
                observations = _observations_from_result(
                    result,
                    max_players=candidate.max_players,
                    prune_mode=candidate.prune_mode,
                    court_calibration=court_calibration,
                    court_margin_m=candidate.court_margin_m,
                    bbox_expand=candidate.bbox_expand,
                )
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
    metrics = score_mobile_person_tracks(gt, predictions, expected_players=candidate.max_players)
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
        "tracker": candidate.tracker,
        "tracker_config": candidate.tracker_config,
        "prune_mode": candidate.prune_mode,
        "court_calibration": candidate.court_calibration,
        "court_margin_m": candidate.court_margin_m,
        "bbox_expand": candidate.bbox_expand,
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


class _CenterDistancePersonLinker:
    def __init__(
        self,
        *,
        max_tracks: int = 4,
        distance_factor: float = 1.25,
        min_distance_px: float = 24.0,
        max_age_frames: int = 30,
    ) -> None:
        self.max_tracks = max_tracks
        self.distance_factor = distance_factor
        self.min_distance_px = min_distance_px
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
            detections.append(_detection_from_observation(int(self._tracks[match_index]["id"]), observation))
        return sorted(detections, key=lambda item: int(item["track_id"]))

    def _best_track_index(self, bbox_xywh: list[float], used_track_indexes: set[int]) -> int | None:
        best_index: int | None = None
        best_distance = float("inf")
        for index, track in enumerate(self._tracks):
            if index in used_track_indexes:
                continue
            track_bbox = track["bbox_xywh"]
            distance = _bbox_center_distance(bbox_xywh, track_bbox)
            threshold = max(
                self.min_distance_px,
                self.distance_factor * max(_bbox_diagonal(bbox_xywh), _bbox_diagonal(track_bbox)),
            )
            if distance <= threshold and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index


class _RoleLockedPersonLinker:
    def __init__(self, *, max_tracks: int = 4) -> None:
        self.max_tracks = max_tracks

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del frame_index
        ordered = _observations_in_role_order(observations[: self.max_tracks])
        detections: list[dict[str, Any]] = []
        roles = ["far_left", "far_right", "near_left", "near_right"]
        for index, observation in enumerate(ordered):
            role = roles[index] if index < len(roles) else None
            detections.append(_detection_from_observation(index + 1, observation, role=role))
        return detections


class _WideRoleLockedPersonLinker:
    def __init__(self, *, max_tracks: int = 4) -> None:
        self.max_tracks = max_tracks

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del frame_index
        selected = _select_spatially_diverse_observations(observations, self.max_tracks)
        ordered = _observations_in_role_order(selected)
        roles = ["far_left", "far_right", "near_left", "near_right"]
        detections: list[dict[str, Any]] = []
        for index, observation in enumerate(ordered):
            role = roles[index] if index < len(roles) else None
            detections.append(_detection_from_observation(index + 1, observation, role=role))
        return detections


class _StableSetPersonLinker:
    def __init__(
        self,
        *,
        max_tracks: int = 4,
        max_age_frames: int = 45,
        max_assignment_cost: float = 3.25,
    ) -> None:
        self.max_tracks = max_tracks
        self.max_age_frames = max_age_frames
        self.max_assignment_cost = max_assignment_cost
        self._tracks: list[dict[str, Any]] = []
        self._next_id = 1

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._tracks:
            return self._bootstrap(frame_index=frame_index, observations=observations)

        active_tracks = [
            track
            for track in self._tracks
            if frame_index - int(track["last_seen_frame_index"]) <= self.max_age_frames
        ]
        self._tracks = active_tracks
        assignments = _stable_track_assignments(
            active_tracks,
            observations,
            frame_index=frame_index,
            max_cost=self.max_assignment_cost,
        )
        detections: list[dict[str, Any]] = []
        used_observation_indexes = set(assignments.values())
        for track_index, observation_index in sorted(assignments.items()):
            track = active_tracks[track_index]
            observation = observations[observation_index]
            self._update_track(track, observation, frame_index)
            detections.append(_detection_from_observation(int(track["id"]), observation))

        if len(self._tracks) < self.max_tracks:
            unused = [observation for index, observation in enumerate(observations) if index not in used_observation_indexes]
            for observation in _select_spatially_diverse_observations(unused, self.max_tracks - len(self._tracks)):
                track = self._new_track(self._next_id, observation, frame_index)
                self._next_id += 1
                self._tracks.append(track)
                detections.append(_detection_from_observation(int(track["id"]), observation))

        return sorted(detections, key=lambda item: int(item["track_id"]))

    def _bootstrap(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = _observations_in_role_order(_select_spatially_diverse_observations(observations, self.max_tracks))
        self._tracks = []
        for observation in selected:
            self._tracks.append(self._new_track(self._next_id, observation, frame_index))
            self._next_id += 1
        return [_detection_from_observation(int(track["id"]), track["observation"]) for track in self._tracks]

    @staticmethod
    def _new_track(track_id: int, observation: dict[str, Any], frame_index: int) -> dict[str, Any]:
        return {
            "id": track_id,
            "bbox_xywh": [float(value) for value in observation["bbox_xywh"]],
            "velocity_xy": [0.0, 0.0],
            "last_seen_frame_index": frame_index,
            "observation": observation,
            "appearance_hsv": _appearance_vector(observation),
        }

    @staticmethod
    def _update_track(track: dict[str, Any], observation: dict[str, Any], frame_index: int) -> None:
        previous_center = _bbox_center(track["bbox_xywh"])
        current_bbox = [float(value) for value in observation["bbox_xywh"]]
        current_center = _bbox_center(current_bbox)
        frame_delta = max(1, frame_index - int(track["last_seen_frame_index"]))
        track["velocity_xy"] = [
            (current_center[0] - previous_center[0]) / frame_delta,
            (current_center[1] - previous_center[1]) / frame_delta,
        ]
        track["bbox_xywh"] = current_bbox
        track["last_seen_frame_index"] = frame_index
        track["observation"] = observation
        current_appearance = _appearance_vector(observation)
        previous_appearance = _appearance_vector(track)
        if current_appearance is not None:
            if previous_appearance is None:
                track["appearance_hsv"] = current_appearance
            else:
                track["appearance_hsv"] = [
                    0.85 * previous + 0.15 * current
                    for previous, current in zip(previous_appearance, current_appearance, strict=False)
                ]


class _WideRoleStablePersonLinker:
    def __init__(self, *, max_tracks: int = 4, max_age_frames: int = 45) -> None:
        self.max_tracks = max_tracks
        self._stable = _StableSetPersonLinker(
            max_tracks=max_tracks,
            max_age_frames=max_age_frames,
            max_assignment_cost=2.75,
        )

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._stable._tracks:
            observations = _select_spatially_diverse_observations(observations, self.max_tracks)
        return self._stable.update(frame_index=frame_index, observations=observations)


class _TemporalFillPersonLinker:
    def __init__(self, *, max_tracks: int = 4, max_fill_gap_frames: int = 8) -> None:
        self.max_tracks = max_tracks
        self.max_fill_gap_frames = max_fill_gap_frames
        self._next_id = 1
        self._tracks: list[dict[str, Any]] = []

    def update(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = _observations_in_role_order(_select_spatially_diverse_observations(observations, self.max_tracks))
        if not self._tracks:
            return self._bootstrap(frame_index=frame_index, observations=selected)

        assignments = _temporal_fill_assignments(self._tracks, selected, frame_index=frame_index)
        detections: list[dict[str, Any]] = []
        used_observations: set[int] = set()
        for track_index, observation_index in sorted(assignments.items()):
            track = self._tracks[track_index]
            observation = selected[observation_index]
            self._update_track(track, observation, frame_index)
            used_observations.add(observation_index)
            detections.append(_detection_from_observation(int(track["id"]), observation))

        for track_index, track in enumerate(self._tracks):
            if track_index in assignments:
                continue
            track["miss_count"] = int(track.get("miss_count", 0)) + 1
            if int(track["miss_count"]) > self.max_fill_gap_frames:
                continue
            predicted_bbox = _predict_track_bbox(track, frame_index)
            track["bbox_xywh"] = predicted_bbox
            track["last_seen_frame_index"] = frame_index
            detections.append(
                {
                    "track_id": int(track["id"]),
                    "bbox_xywh": predicted_bbox,
                    "confidence": 0.0,
                    "source": "temporal_fill",
                    "role": None,
                }
            )

        if len(self._tracks) < self.max_tracks:
            for index, observation in enumerate(selected):
                if index in used_observations:
                    continue
                track = self._new_track(self._next_id, observation, frame_index)
                self._next_id += 1
                self._tracks.append(track)
                detections.append(_detection_from_observation(int(track["id"]), observation))
                if len(self._tracks) >= self.max_tracks:
                    break

        return sorted(detections, key=lambda item: int(item["track_id"]))

    def _bootstrap(self, *, frame_index: int, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._tracks = []
        for observation in observations[: self.max_tracks]:
            self._tracks.append(self._new_track(self._next_id, observation, frame_index))
            self._next_id += 1
        return [_detection_from_observation(int(track["id"]), track["observation"]) for track in self._tracks]

    @staticmethod
    def _new_track(track_id: int, observation: dict[str, Any], frame_index: int) -> dict[str, Any]:
        return {
            "id": track_id,
            "bbox_xywh": [float(value) for value in observation["bbox_xywh"]],
            "velocity_xy": [0.0, 0.0],
            "last_seen_frame_index": frame_index,
            "last_detected_frame_index": frame_index,
            "miss_count": 0,
            "observation": observation,
            "appearance_hsv": _appearance_vector(observation),
        }

    @staticmethod
    def _update_track(track: dict[str, Any], observation: dict[str, Any], frame_index: int) -> None:
        previous_center = _bbox_center(track["bbox_xywh"])
        current_bbox = [float(value) for value in observation["bbox_xywh"]]
        current_center = _bbox_center(current_bbox)
        frame_delta = max(1, frame_index - int(track["last_seen_frame_index"]))
        previous_velocity = [float(value) for value in track.get("velocity_xy", [0.0, 0.0])]
        measured_velocity = [
            (current_center[0] - previous_center[0]) / frame_delta,
            (current_center[1] - previous_center[1]) / frame_delta,
        ]
        track["velocity_xy"] = [
            0.6 * previous_velocity[0] + 0.4 * measured_velocity[0],
            0.6 * previous_velocity[1] + 0.4 * measured_velocity[1],
        ]
        track["bbox_xywh"] = current_bbox
        track["last_seen_frame_index"] = frame_index
        track["last_detected_frame_index"] = frame_index
        track["miss_count"] = 0
        track["observation"] = observation
        current_appearance = _appearance_vector(observation)
        previous_appearance = _appearance_vector(track)
        if current_appearance is not None:
            if previous_appearance is None:
                track["appearance_hsv"] = current_appearance
            else:
                track["appearance_hsv"] = [
                    0.9 * previous + 0.1 * current
                    for previous, current in zip(previous_appearance, current_appearance, strict=False)
                ]


def _make_linker(
    tracker: str,
    *,
    max_players: int,
    iou_threshold: float | None = None,
    max_age_frames: int | None = None,
) -> Any:
    if tracker in {"predict_iou", "iou"}:
        return _IoUPersonLinker(
            max_tracks=max_players,
            iou_threshold=0.3 if iou_threshold is None else iou_threshold,
            max_age_frames=10 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_iou_loose", "iou_loose"}:
        return _IoUPersonLinker(
            max_tracks=max_players,
            iou_threshold=0.1 if iou_threshold is None else iou_threshold,
            max_age_frames=45 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_center", "center"}:
        return _CenterDistancePersonLinker(
            max_tracks=max_players,
            distance_factor=1.25,
            max_age_frames=30 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_center_loose", "center_loose"}:
        return _CenterDistancePersonLinker(
            max_tracks=max_players,
            distance_factor=2.0,
            max_age_frames=60 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_role_lock", "role_lock"}:
        return _RoleLockedPersonLinker(max_tracks=max_players)
    if tracker in {"predict_role_lock_wide", "role_lock_wide"}:
        return _WideRoleLockedPersonLinker(max_tracks=max_players)
    if tracker in {"predict_stable_set", "stable_set"}:
        return _StableSetPersonLinker(
            max_tracks=max_players,
            max_age_frames=60 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_role_lock_wide_stable", "role_lock_wide_stable"}:
        return _WideRoleStablePersonLinker(
            max_tracks=max_players,
            max_age_frames=60 if max_age_frames is None else max_age_frames,
        )
    if tracker in {"predict_temporal_fill", "temporal_fill"}:
        return _TemporalFillPersonLinker(max_tracks=max_players)
    raise ValueError(f"unsupported replay person tracker: {tracker}")


def _observations_from_result(
    result: Any,
    *,
    max_players: int,
    prune_mode: str = "confidence",
    court_calibration: CourtCalibration | None = None,
    court_margin_m: float = 1.25,
    candidate_limit: int = 16,
    output_limit: int | None = None,
    bbox_expand: float = 1.0,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    source_image = getattr(result, "orig_img", None)
    observations: list[dict[str, Any]] = []
    for raw_box, confidence in zip(xyxy, conf, strict=False):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        bbox_xywh = _expand_bbox_xywh([x1, y1, width, height], bbox_expand)
        observation = {
            "bbox_xywh": bbox_xywh,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "source": "yolo_person",
        }
        appearance = _appearance_hsv_from_image(source_image, [x1, y1, width, height])
        if appearance is not None:
            observation["appearance_hsv"] = appearance
        observations.append(observation)
    return _prune_observations(
        observations,
        max_players=max_players,
        prune_mode=prune_mode,
        court_calibration=court_calibration,
        court_margin_m=court_margin_m,
        candidate_limit=candidate_limit,
        output_limit=output_limit,
    )


def _prune_observations(
    observations: list[dict[str, Any]],
    *,
    max_players: int,
    prune_mode: str = "confidence",
    court_calibration: CourtCalibration | None = None,
    court_margin_m: float = 1.25,
    candidate_limit: int = 16,
    output_limit: int | None = None,
) -> list[dict[str, Any]]:
    if max_players <= 0:
        raise ValueError("max_players must be positive")
    limit = max_players if output_limit is None else int(output_limit)
    if limit <= 0:
        raise ValueError("output_limit must be positive")
    candidates = list(observations)
    candidates.sort(key=lambda item: (-float(item["confidence"]), item["bbox_xywh"][0], item["bbox_xywh"][1]))
    candidates = candidates[: max(max_players, limit, candidate_limit)]
    if prune_mode == "confidence":
        return candidates[:limit]
    if prune_mode != "court":
        raise ValueError(f"unsupported observation prune mode: {prune_mode}")
    if court_calibration is None:
        raise ValueError("court pruning requires court_calibration")
    ranked = [_with_court_prune_features(observation, court_calibration) for observation in candidates]
    ranked.sort(
        key=lambda item: (
            0 if float(item["court_outside_distance_m"]) <= court_margin_m else 1,
            -float(item["confidence"]) if float(item["court_outside_distance_m"]) <= court_margin_m else float(item["court_outside_distance_m"]),
            float(item["court_outside_distance_m"]) if float(item["court_outside_distance_m"]) <= court_margin_m else -float(item["confidence"]),
            item["bbox_xywh"][0],
            item["bbox_xywh"][1],
        )
    )
    return ranked[:limit]


def _closed_set_prune_frames(
    frames: list[dict[str, Any]],
    *,
    max_players: int,
    mode: str = "quality",
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> list[dict[str, Any]]:
    """Select the expected player set after linking a wider candidate pool.

    This mirrors the useful part of sam4dbody's closed-set pass for our mobile
    benchmark: surplus linked tracks are treated as non-player guests instead
    of allowing a high-confidence spectator box to displace a real player
    before tracking.
    """

    if max_players <= 0:
        raise ValueError("max_players must be positive")
    summaries = _closed_set_track_summaries(frames, frame_width=frame_width)
    if not summaries:
        return [{"frame_index": int(frame["frame_index"]), "detections": []} for frame in frames]
    selected_ids = _select_closed_set_track_ids(
        summaries,
        max_players=max_players,
        mode=mode,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    remap_order = _track_ids_in_role_order([summaries[track_id] for track_id in selected_ids])
    id_map = {int(track_id): index + 1 for index, track_id in enumerate(remap_order)}
    source_suffix = f"closed_set_{_safe_source_token(mode)}"
    out_frames: list[dict[str, Any]] = []
    for frame in frames:
        detections: list[dict[str, Any]] = []
        for detection in frame.get("detections", []):
            track_id = int(detection["track_id"])
            if track_id not in id_map:
                continue
            cloned = dict(detection)
            cloned["track_id"] = id_map[track_id]
            cloned["source"] = f"{cloned.get('source', 'unknown')}+{source_suffix}"
            detections.append(cloned)
        detections.sort(key=lambda item: int(item["track_id"]))
        out_frames.append({"frame_index": int(frame["frame_index"]), "detections": detections})
    return out_frames


def _closed_set_track_summaries(
    frames: list[dict[str, Any]],
    *,
    frame_width: float | None,
) -> dict[int, dict[str, Any]]:
    by_track: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for frame in frames:
        frame_index = int(frame["frame_index"])
        for detection in frame.get("detections", []):
            by_track.setdefault(int(detection["track_id"]), []).append((frame_index, detection))

    summaries: dict[int, dict[str, Any]] = {}
    for track_id, rows in by_track.items():
        rows.sort(key=lambda item: item[0])
        frame_indexes = [frame_index for frame_index, _detection in rows]
        bboxes = [[float(value) for value in detection["bbox_xywh"]] for _frame_index, detection in rows]
        confidences = [float(detection.get("confidence", 0.0)) for _frame_index, detection in rows]
        centers = [_bbox_center(bbox) for bbox in bboxes]
        areas = [max(0.0, bbox[2]) * max(0.0, bbox[3]) for bbox in bboxes]
        span = max(1, frame_indexes[-1] - frame_indexes[0] + 1)
        movement = 0.0
        for prev, current in zip(centers, centers[1:], strict=False):
            movement += math.hypot(float(current[0]) - float(prev[0]), float(current[1]) - float(prev[1]))
        summaries[int(track_id)] = {
            "track_id": int(track_id),
            "n_frames": len(frame_indexes),
            "first_frame": frame_indexes[0],
            "last_frame": frame_indexes[-1],
            "span_frames": span,
            "coverage_in_span": len(set(frame_indexes)) / span,
            "confidence_mean": sum(confidences) / len(confidences),
            "confidence_p90": _percentile_0_to_100(confidences, 90.0),
            "median_area": _percentile_0_to_100(areas, 50.0),
            "p90_area": _percentile_0_to_100(areas, 90.0),
            "edge_fraction": _edge_fraction_xywh(bboxes, frame_width),
            "median_center": [
                _percentile_0_to_100([center[0] for center in centers], 50.0),
                _percentile_0_to_100([center[1] for center in centers], 50.0),
            ],
            "movement_px": movement,
        }
    return summaries


def _select_closed_set_track_ids(
    summaries: dict[int, dict[str, Any]],
    *,
    max_players: int,
    mode: str,
    frame_width: float | None,
    frame_height: float | None,
) -> list[int]:
    if len(summaries) <= max_players:
        return sorted(summaries)
    scored = [
        (track_id, _closed_set_quality_score(summary, mode=mode))
        for track_id, summary in summaries.items()
    ]
    scored.sort(key=lambda item: (-item[1], summaries[item[0]]["first_frame"], item[0]))
    if mode in {"cluster", "quality_cluster", "cluster_strong", "motion_cluster"}:
        top_ids = [track_id for track_id, _score in scored[: min(12, len(scored))]]
        return _best_closed_set_cluster(
            top_ids,
            summaries,
            max_players=max_players,
            mode=mode,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return [track_id for track_id, _score in scored[:max_players]]


def _closed_set_quality_score(summary: dict[str, Any], *, mode: str) -> float:
    n_frames = float(summary["n_frames"])
    span = float(summary["span_frames"])
    mean_conf = float(summary["confidence_mean"])
    p90_conf = float(summary["confidence_p90"])
    coverage = float(summary["coverage_in_span"])
    edge = float(summary["edge_fraction"])
    area_term = min(40.0, math.sqrt(max(0.0, float(summary["median_area"]))) * 0.12)
    motion_term = min(80.0, math.sqrt(max(0.0, float(summary["movement_px"]))) * 2.0)
    quality = n_frames + 0.05 * span + 35.0 * mean_conf + 20.0 * p90_conf + 20.0 * coverage + area_term - 12.0 * edge
    if mode == "duration":
        return n_frames + 20.0 * coverage - 8.0 * edge
    if mode in {"motion", "motion_cluster"}:
        return quality + motion_term
    if mode == "area":
        return quality + area_term
    return quality


def _best_closed_set_cluster(
    track_ids: list[int],
    summaries: dict[int, dict[str, Any]],
    *,
    max_players: int,
    mode: str,
    frame_width: float | None,
    frame_height: float | None,
) -> list[int]:
    if len(track_ids) <= max_players:
        return sorted(track_ids)
    width = max(1.0, float(frame_width or _infer_frame_extent(summaries, axis=0)))
    height = max(1.0, float(frame_height or _infer_frame_extent(summaries, axis=1)))
    spread_weight = 120.0
    if mode == "cluster_strong":
        spread_weight = 300.0
    if mode == "motion_cluster":
        spread_weight = 180.0
    best_ids: tuple[int, ...] | None = None
    best_score = -float("inf")
    for combo in combinations(track_ids, max_players):
        centers = [summaries[track_id]["median_center"] for track_id in combo]
        xs = [float(center[0]) for center in centers]
        ys = [float(center[1]) for center in centers]
        spread_area = ((max(xs) - min(xs)) / width) * ((max(ys) - min(ys)) / height)
        edge_penalty = sum(float(summaries[track_id]["edge_fraction"]) for track_id in combo)
        score = sum(_closed_set_quality_score(summaries[track_id], mode=mode) for track_id in combo)
        score -= spread_weight * spread_area
        score -= 10.0 * edge_penalty
        if score > best_score:
            best_score = score
            best_ids = combo
    return sorted(best_ids or tuple(track_ids[:max_players]))


def _track_ids_in_role_order(summaries: list[dict[str, Any]]) -> list[int]:
    if len(summaries) <= 2:
        ordered = sorted(summaries, key=lambda item: (item["median_center"][1], item["median_center"][0]))
    else:
        by_y = sorted(summaries, key=lambda item: (item["median_center"][1], item["median_center"][0]))
        top_count = len(by_y) // 2
        top = sorted(by_y[:top_count], key=lambda item: item["median_center"][0])
        bottom = sorted(by_y[top_count:], key=lambda item: item["median_center"][0])
        ordered = top + bottom
    return [int(summary["track_id"]) for summary in ordered]


def _edge_fraction_xywh(bboxes: list[list[float]], frame_width: float | None, margin: float = 8.0) -> float:
    if not bboxes:
        return 1.0
    if not frame_width or frame_width <= 0.0:
        return sum(1 for bbox in bboxes if bbox[0] <= margin) / len(bboxes)
    width = float(frame_width)
    return sum(1 for bbox in bboxes if bbox[0] <= margin or bbox[0] + bbox[2] >= width - margin) / len(bboxes)


def _percentile_0_to_100(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _infer_frame_extent(summaries: dict[int, dict[str, Any]], *, axis: int) -> float:
    if not summaries:
        return 1.0
    centers = [float(summary["median_center"][axis]) for summary in summaries.values()]
    return max(1.0, max(centers) - min(centers))


def _safe_source_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value).lower())
    return "_".join(part for part in token.split("_") if part) or "unknown"


def _with_court_prune_features(observation: dict[str, Any], calibration: CourtCalibration) -> dict[str, Any]:
    enriched = dict(observation)
    x, y, width, height = [float(value) for value in observation["bbox_xywh"]]
    foot = [x + width / 2.0, y + height]
    foot_world_xy = project_image_points_to_world(calibration.homography, [foot])[0]
    enriched["foot_world_xy"] = [float(foot_world_xy[0]), float(foot_world_xy[1])]
    enriched["court_outside_distance_m"] = _court_outside_distance_m(calibration, enriched["foot_world_xy"])
    return enriched


def _court_outside_distance_m(calibration: CourtCalibration, world_xy: list[float]) -> float:
    template = get_court_template(calibration.sport)
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    x, y = [float(value) for value in world_xy]
    dx = max(0.0, abs(x) - half_width)
    dy = max(0.0, abs(y) - half_length)
    return (dx**2 + dy**2) ** 0.5


def _load_court_calibration(path: str | Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", Path(path))
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    return parsed


def _tracked_detections_from_result(result: Any, *, max_players: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    raw_ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else [None] * len(xyxy)
    detections: list[dict[str, Any]] = []
    for raw_box, confidence, raw_track_id in zip(xyxy, conf, raw_ids, strict=False):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        track_id = int(raw_track_id) if raw_track_id is not None else len(detections) + 1
        detections.append(
            {
                "track_id": max(1, track_id),
                "bbox_xywh": [x1, y1, width, height],
                "confidence": max(0.0, min(1.0, float(confidence))),
                "source": "ultralytics_track",
                "role": None,
            }
        )
    detections.sort(key=lambda item: (-float(item["confidence"]), int(item["track_id"])))
    return sorted(detections[:max_players], key=lambda item: int(item["track_id"]))


def _expand_bbox_xywh(bbox_xywh: list[float], scale: float) -> list[float]:
    if scale <= 0.0:
        raise ValueError("bbox expansion scale must be positive")
    if scale == 1.0:
        return [float(value) for value in bbox_xywh]
    x, y, width, height = [float(value) for value in bbox_xywh]
    center_x = x + width / 2.0
    bottom_y = y + height
    expanded_width = width * scale
    expanded_height = height * scale
    return [center_x - expanded_width / 2.0, bottom_y - expanded_height, expanded_width, expanded_height]


def _appearance_hsv_from_image(image: Any, bbox_xywh: list[float]) -> list[float] | None:
    if image is None or not hasattr(image, "shape"):
        return None
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None
    image_height, image_width = image.shape[:2]
    x, y, width, height = [float(value) for value in bbox_xywh]
    crop_x1 = int(max(0.0, min(image_width - 1, x + 0.20 * width)))
    crop_x2 = int(max(0.0, min(image_width, x + 0.80 * width)))
    crop_y1 = int(max(0.0, min(image_height - 1, y + 0.18 * height)))
    crop_y2 = int(max(0.0, min(image_height, y + 0.70 * height)))
    if crop_x2 <= crop_x1 + 1 or crop_y2 <= crop_y1 + 1:
        return None
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype("float32") * (2.0 * math.pi / 180.0)
    saturation = hsv[:, :, 1].astype("float32") / 255.0
    value = hsv[:, :, 2].astype("float32") / 255.0
    weights = saturation + 0.05
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return None
    return [
        float((weights * np.cos(hue)).sum() / weight_sum),
        float((weights * np.sin(hue)).sum() / weight_sum),
        float((weights * saturation).sum() / weight_sum),
        float((weights * value).sum() / weight_sum),
    ]


def _select_spatially_diverse_observations(observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not observations:
        return []
    candidates = list(observations)
    selected: list[dict[str, Any]] = []
    while candidates and len(selected) < limit:
        best_index = 0
        best_score = -float("inf")
        for index, candidate in enumerate(candidates):
            if any(_bbox_iou(candidate["bbox_xywh"], chosen["bbox_xywh"]) > 0.35 for chosen in selected):
                continue
            score = float(candidate.get("confidence", 0.0))
            if selected:
                min_distance = min(_bbox_center_distance(candidate["bbox_xywh"], chosen["bbox_xywh"]) for chosen in selected)
                score += min(1.0, min_distance / max(1.0, _bbox_diagonal(candidate["bbox_xywh"]) * 2.0))
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(candidates.pop(best_index))
    return selected


def _stable_track_assignments(
    tracks: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    frame_index: int,
    max_cost: float,
) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        predicted = _predict_track_bbox(track, frame_index)
        for observation_index, observation in enumerate(observations):
            cost = _stable_assignment_cost(track, predicted, observation)
            if cost <= max_cost:
                pairs.append((cost, track_index, observation_index))
    pairs.sort()
    assignments: dict[int, int] = {}
    used_observations: set[int] = set()
    for _cost, track_index, observation_index in pairs:
        if track_index in assignments or observation_index in used_observations:
            continue
        assignments[track_index] = observation_index
        used_observations.add(observation_index)
    return assignments


def _temporal_fill_assignments(
    tracks: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    frame_index: int,
) -> dict[int, int]:
    if not tracks or not observations:
        return {}
    track_count = len(tracks)
    observation_count = len(observations)
    pair_count = min(track_count, observation_count)
    best_assignments: dict[int, int] = {}
    best_cost = float("inf")

    if observation_count >= track_count:
        track_choices = (tuple(range(track_count)),)
        observation_choices = permutations(range(observation_count), pair_count)
        for track_indexes in track_choices:
            for observation_indexes in observation_choices:
                total = 0.0
                assignments: dict[int, int] = {}
                for track_index, observation_index in zip(track_indexes, observation_indexes, strict=False):
                    total += _temporal_fill_assignment_cost(
                        tracks[track_index],
                        observations[observation_index],
                        frame_index=frame_index,
                    )
                    assignments[track_index] = observation_index
                if total < best_cost:
                    best_cost = total
                    best_assignments = assignments
        return best_assignments

    for track_indexes in combinations(range(track_count), pair_count):
        for observation_indexes in permutations(range(observation_count), pair_count):
            total = 0.0
            assignments = {}
            for track_index, observation_index in zip(track_indexes, observation_indexes, strict=False):
                total += _temporal_fill_assignment_cost(
                    tracks[track_index],
                    observations[observation_index],
                    frame_index=frame_index,
                )
                assignments[track_index] = observation_index
            if total < best_cost:
                best_cost = total
                best_assignments = assignments
    return best_assignments


def _predict_track_bbox(track: dict[str, Any], frame_index: int) -> list[float]:
    bbox = [float(value) for value in track["bbox_xywh"]]
    frame_delta = max(0, frame_index - int(track["last_seen_frame_index"]))
    velocity_x, velocity_y = [float(value) for value in track.get("velocity_xy", [0.0, 0.0])]
    return [bbox[0] + velocity_x * frame_delta, bbox[1] + velocity_y * frame_delta, bbox[2], bbox[3]]


def _stable_assignment_cost(track: dict[str, Any], predicted_bbox: list[float], observation: dict[str, Any]) -> float:
    bbox = [float(value) for value in observation["bbox_xywh"]]
    distance = _bbox_center_distance(predicted_bbox, bbox)
    scale = max(32.0, _bbox_diagonal(predicted_bbox), _bbox_diagonal(bbox))
    distance_term = distance / scale
    iou_term = 1.0 - _bbox_iou(predicted_bbox, bbox)
    area_term = abs(math.log(max(1.0, bbox[2] * bbox[3]) / max(1.0, predicted_bbox[2] * predicted_bbox[3])))
    confidence_bonus = min(1.0, max(0.0, float(observation.get("confidence", 0.0))))
    appearance_distance = _appearance_distance(_appearance_vector(track), _appearance_vector(observation))
    if appearance_distance is None:
        return distance_term + 0.8 * iou_term + 0.25 * area_term - 0.2 * confidence_bonus
    return 0.35 * distance_term + 0.5 * iou_term + 0.2 * area_term + 4.0 * appearance_distance - 0.2 * confidence_bonus


def _temporal_fill_assignment_cost(track: dict[str, Any], observation: dict[str, Any], *, frame_index: int) -> float:
    predicted_bbox = _predict_track_bbox(track, frame_index)
    bbox = [float(value) for value in observation["bbox_xywh"]]
    distance = _bbox_center_distance(predicted_bbox, bbox)
    scale = max(32.0, _bbox_diagonal(predicted_bbox), _bbox_diagonal(bbox))
    distance_term = distance / scale
    iou_term = 1.0 - _bbox_iou(predicted_bbox, bbox)
    area_term = abs(math.log(max(1.0, bbox[2] * bbox[3]) / max(1.0, predicted_bbox[2] * predicted_bbox[3])))
    confidence_bonus = min(1.0, max(0.0, float(observation.get("confidence", 0.0))))
    appearance_distance = _appearance_distance(_appearance_vector(track), _appearance_vector(observation))
    appearance_term = 0.0 if appearance_distance is None else 1.0 * appearance_distance
    return 0.35 * distance_term + 0.5 * iou_term + 0.2 * area_term + appearance_term - 0.2 * confidence_bonus


def _appearance_vector(item: dict[str, Any]) -> list[float] | None:
    raw = item.get("appearance_hsv")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return None


def _appearance_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    if not a:
        return None
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=False)) / len(a))


def _detection_from_observation(track_id: int, observation: dict[str, Any], *, role: str | None = None) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "bbox_xywh": [float(value) for value in observation["bbox_xywh"]],
        "confidence": float(observation["confidence"]),
        "source": observation["source"],
        "role": role,
    }


def _observations_in_role_order(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(observations) <= 2:
        return sorted(observations, key=lambda item: (_bbox_center(item["bbox_xywh"])[1], _bbox_center(item["bbox_xywh"])[0]))
    by_y = sorted(observations, key=lambda item: (_bbox_center(item["bbox_xywh"])[1], _bbox_center(item["bbox_xywh"])[0]))
    top_count = len(by_y) // 2
    top = sorted(by_y[:top_count], key=lambda item: _bbox_center(item["bbox_xywh"])[0])
    bottom = sorted(by_y[top_count:], key=lambda item: _bbox_center(item["bbox_xywh"])[0])
    return top + bottom


def _tracker_config_path(candidate: ReplayYoloCandidate) -> str:
    if candidate.tracker_config:
        return candidate.tracker_config
    defaults = {
        "track_bytetrack": "configs/racketsport/bytetrack.yaml",
        "track_bytetrack_loose": "configs/racketsport/bytetrack_loose.yaml",
        "track_botsort_no_reid": "configs/racketsport/botsort_no_reid.yaml",
        "track_botsort_no_reid_loose": "configs/racketsport/botsort_no_reid_loose.yaml",
        "track_botsort_reid": "configs/racketsport/botsort_reid.yaml",
        "track_botsort_reid_loose": "configs/racketsport/botsort_reid_loose.yaml",
    }
    try:
        return defaults[candidate.tracker]
    except KeyError as exc:
        raise ValueError(f"missing Ultralytics tracker config for {candidate.tracker}") from exc


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
            "p50_latency_ms": _quantile_fraction(latencies, 0.50),
            "p95_latency_ms": _quantile_fraction(latencies, 0.95),
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


def _bbox_center(bbox_xywh: list[float]) -> tuple[float, float]:
    x, y, width, height = bbox_xywh
    return x + width / 2.0, y + height / 2.0


def _bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _bbox_diagonal(bbox_xywh: list[float]) -> float:
    _, _, width, height = bbox_xywh
    return (width**2 + height**2) ** 0.5


def _quantile_fraction(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * fraction))))
    return float(sorted_values[index])


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [(60, 220, 255), (80, 200, 80), (255, 180, 80), (220, 120, 255), (255, 255, 80), (80, 120, 255)]
    return palette[track_id % len(palette)]


__all__ = ["ReplayYoloCandidate", "render_replay_yolo_overlay", "run_replay_yolo_candidate"]
