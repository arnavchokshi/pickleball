"""Source-only person appearance diagnostics for tracker/ReID follow-up."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EMBEDDING_KEY_TOKENS = ("embedding", "embeddings", "reid", "appearance_vector", "feature_vector")


@dataclass(frozen=True)
class AppearanceDiagnosticConfig:
    max_samples_per_track: int = 24
    sample_stride_frames: int = 1
    crop_padding_px: int = 8
    histogram_bins: int = 8
    tile_size: int = 96

    def __post_init__(self) -> None:
        if self.max_samples_per_track <= 0:
            raise ValueError("max_samples_per_track must be positive")
        if self.sample_stride_frames <= 0:
            raise ValueError("sample_stride_frames must be positive")
        if self.crop_padding_px < 0:
            raise ValueError("crop_padding_px must be non-negative")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")


@dataclass(frozen=True)
class ReIDEmbeddingExportConfig:
    max_detections: int | None = None
    sample_stride_frames: int = 1
    crop_padding_px: int = 8
    batch_size: int = 32
    imgsz: int = 224
    embed_layer: int = 21
    device: str | None = None
    half: bool | None = None
    l2_normalize: bool = True

    def __post_init__(self) -> None:
        if self.max_detections is not None and self.max_detections <= 0:
            raise ValueError("max_detections must be positive when provided")
        if self.sample_stride_frames <= 0:
            raise ValueError("sample_stride_frames must be positive")
        if self.crop_padding_px < 0:
            raise ValueError("crop_padding_px must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive")


FeatureExtractor = Callable[[list[Any]], Sequence[Sequence[float]]]


def inspect_detection_appearance_inputs(detections_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize whether tracked detections already carry appearance signals."""

    detection_keys: dict[str, int] = {}
    embedding_like_keys: set[str] = set()
    crop_like_keys: set[str] = set()
    person_detection_count = 0
    frame_count = 0
    source_track_ids: set[int] = set()

    for frame_idx, detections in _iter_frame_detections(detections_payload):
        frame_count += 1
        for det_idx, detection in enumerate(detections):
            if not _is_person_detection(detection):
                continue
            person_detection_count += 1
            source_track_ids.add(_track_id(detection, det_idx + 1))
            for key in detection:
                detection_keys[key] = detection_keys.get(key, 0) + 1
                normalized = str(key).lower()
                if any(token in normalized for token in EMBEDDING_KEY_TOKENS):
                    embedding_like_keys.add(str(key))
                if "crop" in normalized or "image_path" in normalized:
                    crop_like_keys.add(str(key))

    has_embeddings = bool(embedding_like_keys)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_input_inspection",
        "status": "persisted_appearance_embeddings_found" if has_embeddings else "no_persisted_appearance_embeddings",
        "source_only": True,
        "uses_cvat_labels": False,
        "frame_count": frame_count,
        "person_detection_count": person_detection_count,
        "source_track_count": len(source_track_ids),
        "source_track_ids": sorted(source_track_ids),
        "detection_keys": dict(sorted(detection_keys.items())),
        "embedding_like_keys": sorted(embedding_like_keys),
        "crop_like_keys": sorted(crop_like_keys),
        "notes": [
            "Inspection reads tracked detections only; it does not read CVAT labels.",
            "Absence of embedding-like keys means true ReID embeddings are not persisted in this source pool.",
        ],
    }


def build_source_reid_embedding_export(
    *,
    video_path: str | Path,
    detections_payload: Mapping[str, Any],
    output_path: str | Path,
    model_path: str | Path,
    command_metadata: Mapping[str, Any] | None = None,
    config: ReIDEmbeddingExportConfig | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> dict[str, Any]:
    """Export learned per-detection appearance vectors for source-pixel person detections."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for ReID embedding export") from exc

    cfg = config or ReIDEmbeddingExportConfig()
    video = Path(video_path)
    model = Path(model_path)
    out = Path(output_path)
    if not model.is_file():
        raise FileNotFoundError(f"model does not exist: {model}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(detections_payload.get("fps", 0.0) or 0.0)
    selected = _select_embedding_samples(detections_payload, cfg)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for sample in selected:
        by_frame.setdefault(int(sample["frame"]), []).append(sample)

    crop_records: list[dict[str, Any]] = []
    crops: list[Any] = []
    try:
        if by_frame:
            start_frame = min(by_frame)
            end_frame = max(by_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in by_frame:
                    for sample in by_frame[frame_idx]:
                        x1, y1, x2, y2 = _clamped_bbox(sample["bbox"], video_width, video_height, cfg.crop_padding_px)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        crops.append(crop)
                        crop_records.append(
                            {
                                "frame": frame_idx,
                                "detection_index": int(sample["detection_index"]),
                                "source_track_id": int(sample["track_id"]),
                                "track_id": int(sample["track_id"]),
                                "bbox": [round(float(value), 3) for value in sample["bbox"]],
                                "crop_xyxy": [x1, y1, x2, y2],
                                "conf": round(float(sample["conf"]), 6),
                            }
                        )
                frame_idx += 1
    finally:
        cap.release()

    if not crop_records:
        raise ValueError("no valid person crops selected for ReID embedding export")

    extractor = feature_extractor or _make_ultralytics_yolo_embedder(model, cfg)
    embeddings = _extract_embeddings_in_batches(extractor, crops, batch_size=cfg.batch_size)
    if len(embeddings) != len(crop_records):
        raise ValueError("feature extractor returned a different embedding count than the selected crops")

    normalized_embeddings = _validated_embeddings(embeddings, l2_normalize=cfg.l2_normalize)
    feature_dim = len(normalized_embeddings[0])
    model_sha256 = _sha256_file(model)
    detections: list[dict[str, Any]] = []
    for record, embedding in zip(crop_records, normalized_embeddings, strict=True):
        detections.append(
            {
                **record,
                "model_path": str(model),
                "model_sha256": model_sha256,
                "feature_dim": feature_dim,
                "embedding": embedding,
            }
        )

    inspection = inspect_detection_appearance_inputs(detections_payload)
    track_sample_counts: dict[str, int] = {}
    for detection in detections:
        key = str(detection["source_track_id"])
        track_sample_counts[key] = track_sample_counts.get(key, 0) + 1

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embedding_export",
        "status": "source_only_learned_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "video_path": str(video),
        "video_width": video_width,
        "video_height": video_height,
        "video_fps": source_fps,
        "detections_frame_count": inspection["frame_count"],
        "source_person_detection_count": inspection["person_detection_count"],
        "model_path": str(model),
        "model_sha256": model_sha256,
        "feature_type": "learned_model_embedding",
        "feature_extractor": "ultralytics_yolo_embed" if feature_extractor is None else "injected_feature_extractor",
        "feature_layer": cfg.embed_layer,
        "feature_dim": feature_dim,
        "l2_normalized": cfg.l2_normalize,
        "config": asdict(cfg),
        "command_metadata": dict(command_metadata or {}),
        "detection_count": len(detections),
        "track_sample_counts": dict(sorted(track_sample_counts.items(), key=lambda item: int(item[0]))),
        "detections": detections,
        "notes": [
            "Vectors are source-only learned appearance features exported from persisted detections and source video crops.",
            "This artifact is not a labeled TRK promotion gate by itself.",
        ],
    }
    _write_json(out, payload)
    return payload


def build_source_appearance_diagnostic(
    *,
    video_path: str | Path,
    detections_payload: Mapping[str, Any],
    out_dir: str | Path,
    config: AppearanceDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Export CPU color-appearance diagnostics from source detections and video crops."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for source appearance diagnostics") from exc

    cfg = config or AppearanceDiagnosticConfig()
    video = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crop_sheet_dir = out / "crop_sheets"
    crop_sheet_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(detections_payload.get("fps", 0.0) or 0.0)

    selected = _select_samples(detections_payload, cfg)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for sample in selected:
        by_frame.setdefault(int(sample["frame"]), []).append(sample)

    samples: list[dict[str, Any]] = []
    tiles_by_track: dict[int, list[Any]] = {}
    try:
        for frame_idx in sorted(by_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            for sample in by_frame[frame_idx]:
                x1, y1, x2, y2 = _clamped_bbox(sample["bbox"], video_width, video_height, cfg.crop_padding_px)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                feature = _crop_color_feature(cv2, np, crop, bins=cfg.histogram_bins)
                track_id = int(sample["track_id"])
                sample_record = {
                    "frame": frame_idx,
                    "source_track_id": track_id,
                    "bbox": [round(float(value), 3) for value in sample["bbox"]],
                    "crop_xyxy": [x1, y1, x2, y2],
                    "conf": round(float(sample["conf"]), 6),
                    "feature": feature,
                }
                samples.append(sample_record)
                tile = cv2.resize(crop, (cfg.tile_size, cfg.tile_size), interpolation=cv2.INTER_AREA)
                _draw_tile_label(cv2, tile, f"id {track_id} f{frame_idx}")
                tiles_by_track.setdefault(track_id, []).append(tile)
    finally:
        cap.release()

    tracks = _summarize_tracks(samples)
    crop_sheets = _write_crop_sheets(cv2, np, crop_sheet_dir, tiles_by_track, cfg.tile_size)
    pairwise = _pairwise_track_distances(tracks)
    inspection = inspect_detection_appearance_inputs(detections_payload)

    feature_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_source_person_appearance_features",
        "status": "source_only_appearance_features",
        "source_only": True,
        "uses_cvat_labels": False,
        "appearance_feature_type": "cpu_color_histogram_not_reid_embedding",
        "feature_layout": {
            "segments": ["upper_crop", "lower_crop"],
            "channels": ["rgb_r", "rgb_g", "rgb_b"],
            "histogram_bins_per_channel": cfg.histogram_bins,
            "feature_length": cfg.histogram_bins * 3 * 2,
        },
        "samples": samples,
        "tracks": tracks,
        "pairwise_track_distances": pairwise,
    }
    features_path = out / "track_appearance_features.json"
    _write_json(features_path, feature_payload)

    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_source_person_reid_diagnostic",
        "status": "source_only_appearance_diagnostic",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "video_path": str(video),
        "detections_frame_count": inspection["frame_count"],
        "source_person_detection_count": inspection["person_detection_count"],
        "sample_count": len(samples),
        "sampled_source_track_count": len(tracks),
        "video_width": video_width,
        "video_height": video_height,
        "video_fps": source_fps,
        "config": asdict(cfg),
        "input_inspection": inspection,
        "appearance_feature_type": "cpu_color_histogram_not_reid_embedding",
        "feature_path": str(features_path),
        "crop_sheets": crop_sheets,
        "tracks": tracks,
        "pairwise_track_distances": pairwise,
        "blockers": [
            "no_persisted_reid_embeddings_in_tracked_detections"
            if not inspection["embedding_like_keys"]
            else "persisted_embedding_keys_exist_but_are_not_validated_for_trk_promotion",
            "cpu_color_histograms_are_diagnostic_only_and_not_a_trained_reid_model",
            "no_cvat_labels_used_for_candidate_construction_or_feature_export",
        ],
        "next_action": (
            "Use these source-only crop sheets and color descriptors to choose/validate a tracker ReID "
            "export or train detector/tracker appearance embeddings on the same source pool; then rescore "
            "with labeled IDF1/spectator/ID-switch gates."
        ),
    }
    _write_json(out / "source_appearance_diagnostics.json", report)
    return report


def _iter_frame_detections(detections_payload: Mapping[str, Any]) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")
    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, Mapping):
            raise ValueError("each frame entry must be an object")
        frame_idx = int(frame_entry.get("frame", frame_entry.get("frame_index", default_frame_idx)))
        detections = frame_entry.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("frame detections must be a list")
        dict_detections = [item for item in detections if isinstance(item, dict)]
        yield frame_idx, dict_detections


def _select_samples(detections_payload: Mapping[str, Any], config: AppearanceDiagnosticConfig) -> list[dict[str, Any]]:
    counts_by_track: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for frame_idx, detections in _iter_frame_detections(detections_payload):
        if frame_idx % config.sample_stride_frames != 0:
            continue
        for det_idx, detection in enumerate(sorted(detections, key=lambda item: float(item.get("conf", item.get("confidence", 1.0))), reverse=True)):
            if not _is_person_detection(detection):
                continue
            track_id = _track_id(detection, det_idx + 1)
            if counts_by_track.get(track_id, 0) >= config.max_samples_per_track:
                continue
            selected.append(
                {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "bbox": _bbox_xyxy(detection),
                    "conf": float(detection.get("conf", detection.get("confidence", 1.0))),
                }
            )
            counts_by_track[track_id] = counts_by_track.get(track_id, 0) + 1
    return selected


def _select_embedding_samples(detections_payload: Mapping[str, Any], config: ReIDEmbeddingExportConfig) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame_idx, detections in _iter_frame_detections(detections_payload):
        if frame_idx % config.sample_stride_frames != 0:
            continue
        for det_idx, detection in enumerate(detections):
            if not _is_person_detection(detection):
                continue
            selected.append(
                {
                    "frame": frame_idx,
                    "detection_index": det_idx,
                    "track_id": _track_id(detection, det_idx + 1),
                    "bbox": _bbox_xyxy(detection),
                    "conf": float(detection.get("conf", detection.get("confidence", 1.0))),
                }
            )
            if config.max_detections is not None and len(selected) >= config.max_detections:
                return selected
    return selected


def _make_ultralytics_yolo_embedder(model_path: Path, config: ReIDEmbeddingExportConfig) -> FeatureExtractor:
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for learned ReID embedding export") from exc

    model = YOLO(str(model_path))

    def embed(crops: list[Any]) -> Sequence[Sequence[float]]:
        kwargs: dict[str, Any] = {
            "source": crops,
            "imgsz": config.imgsz,
            "device": config.device,
            "batch": len(crops),
            "verbose": False,
            "embed": [config.embed_layer],
        }
        if config.half is not None:
            kwargs["half"] = config.half
        results = model.predict(**kwargs)
        return [_tensor_to_float_list(result) for result in results]

    return embed


def _extract_embeddings_in_batches(extractor: FeatureExtractor, crops: Sequence[Any], *, batch_size: int) -> list[Sequence[float]]:
    embeddings: list[Sequence[float]] = []
    for start in range(0, len(crops), batch_size):
        batch = list(crops[start : start + batch_size])
        embeddings.extend(extractor(batch))
    return embeddings


def _validated_embeddings(embeddings: Sequence[Sequence[float]], *, l2_normalize: bool) -> list[list[float]]:
    if not embeddings:
        raise ValueError("feature extractor returned no embeddings")
    feature_dim = len(embeddings[0])
    if feature_dim <= 0:
        raise ValueError("feature extractor returned empty embeddings")
    rows: list[list[float]] = []
    for embedding in embeddings:
        if len(embedding) != feature_dim:
            raise ValueError("all ReID embeddings must have consistent feature dimensions")
        values = [float(value) for value in embedding]
        if l2_normalize:
            norm = sum(value * value for value in values) ** 0.5
            if norm > 0.0:
                values = [value / norm for value in values]
        rows.append([round(value, 6) for value in values])
    return rows


def _tensor_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "flatten"):
        value = value.flatten()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError("embedding result is not list-like")
    return [float(item) for item in value]


def _crop_color_feature(cv2: Any, np: Any, crop: Any, *, bins: int) -> list[float]:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    height = int(rgb.shape[0])
    split = max(1, height // 2)
    parts = [rgb[:split], rgb[split:] if split < height else rgb[:split]]
    values: list[float] = []
    for part in parts:
        for channel in range(3):
            hist, _ = np.histogram(part[:, :, channel], bins=bins, range=(0, 256))
            hist = hist.astype("float64")
            total = float(hist.sum())
            if total > 0.0:
                hist = hist / total
            values.extend(round(float(value), 6) for value in hist.tolist())
    return values


def _summarize_tracks(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    import math

    by_track: dict[int, list[Mapping[str, Any]]] = {}
    for sample in samples:
        by_track.setdefault(int(sample["source_track_id"]), []).append(sample)

    summaries: dict[str, dict[str, Any]] = {}
    for track_id, track_samples in sorted(by_track.items()):
        features = [sample["feature"] for sample in track_samples if isinstance(sample.get("feature"), list)]
        mean_feature = _mean_feature(features)
        distances = [_cosine_distance(feature, mean_feature) for feature in features] if mean_feature else []
        summaries[str(track_id)] = {
            "sample_count": len(track_samples),
            "frames": [int(sample["frame"]) for sample in track_samples],
            "mean_conf": round(sum(float(sample["conf"]) for sample in track_samples) / len(track_samples), 6),
            "mean_feature": mean_feature,
            "mean_within_track_cosine_distance": round(sum(distances) / len(distances), 6) if distances else None,
            "max_within_track_cosine_distance": round(max(distances), 6) if distances else None,
            "feature_norm": round(math.sqrt(sum(value * value for value in mean_feature)), 6) if mean_feature else 0.0,
        }
    return summaries


def _pairwise_track_distances(tracks: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(tracks, key=lambda value: int(value) if value.isdigit() else value)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            distance = _cosine_distance(tracks[left].get("mean_feature", []), tracks[right].get("mean_feature", []))
            rows.append({"left_source_track_id": left, "right_source_track_id": right, "cosine_distance": round(distance, 6)})
    return rows


def _mean_feature(features: Sequence[Sequence[float]]) -> list[float]:
    if not features:
        return []
    width = len(features[0])
    totals = [0.0] * width
    count = 0
    for feature in features:
        if len(feature) != width:
            continue
        count += 1
        for index, value in enumerate(feature):
            totals[index] += float(value)
    if count == 0:
        return []
    return [round(value / count, 6) for value in totals]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    norm_a = sum(float(x) * float(x) for x in a) ** 0.5
    norm_b = sum(float(y) * float(y) for y in b) ** 0.5
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (norm_a * norm_b)))


def _write_crop_sheets(cv2: Any, np: Any, crop_sheet_dir: Path, tiles_by_track: Mapping[int, Sequence[Any]], tile_size: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    for track_id, tiles in sorted(tiles_by_track.items()):
        if not tiles:
            continue
        cols = min(6, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row = index // cols
            col = index % cols
            sheet[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = tile
        out_path = crop_sheet_dir / f"source_track_{track_id}.png"
        cv2.imwrite(str(out_path), sheet)
        paths[str(track_id)] = str(out_path)
    return paths


def _draw_tile_label(cv2: Any, tile: Any, label: str) -> None:
    cv2.rectangle(tile, (0, 0), (min(tile.shape[1] - 1, 74), 15), (0, 0, 0), -1)
    cv2.putText(tile, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


def _bbox_xyxy(detection: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four xyxy values")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detection bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _clamped_bbox(bbox: Sequence[float], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(width, int(x2 + 0.999) + padding),
        min(height, int(y2 + 0.999) + padding),
    )


def _is_person_detection(detection: Mapping[str, Any]) -> bool:
    value = detection.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


def _track_id(detection: Mapping[str, Any], fallback: int) -> int:
    value = detection.get("track_id", detection.get("player_id", detection.get("id", fallback)))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AppearanceDiagnosticConfig",
    "ReIDEmbeddingExportConfig",
    "build_source_appearance_diagnostic",
    "build_source_reid_embedding_export",
    "inspect_detection_appearance_inputs",
]
