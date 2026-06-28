"""Tiled YOLO person detection helpers for wide pickleball videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


Detection = dict[str, Any]
NormalizedCrop = tuple[float, float, float, float]

DEFAULT_CROP_REGIONS: tuple[NormalizedCrop, ...] = (
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.22, 0.68, 0.95),
    (0.32, 0.18, 1.0, 0.95),
    (0.44, 0.20, 1.0, 0.86),
)

CROP_REGION_PRESETS: dict[str, tuple[NormalizedCrop, ...]] = {
    "default4": DEFAULT_CROP_REGIONS,
    "full_lr3": (
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.20, 0.62, 0.96),
        (0.38, 0.18, 1.0, 0.96),
    ),
    "full_tb3": (
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.58),
        (0.0, 0.42, 1.0, 1.0),
    ),
    "full": ((0.0, 0.0, 1.0, 1.0),),
}

ADAPTIVE_CROP_REGION_PRESETS: dict[str, tuple[tuple[NormalizedCrop, ...], tuple[NormalizedCrop, ...], int]] = {
    "adaptive_full_tb3": (
        CROP_REGION_PRESETS["full"],
        CROP_REGION_PRESETS["full_tb3"][1:],
        4,
    ),
}


def parse_crop_regions(spec: str | Sequence[NormalizedCrop] | None) -> tuple[NormalizedCrop, ...]:
    if spec is None:
        return DEFAULT_CROP_REGIONS
    if not isinstance(spec, str):
        return tuple(tuple(float(value) for value in region) for region in spec)  # type: ignore[return-value]
    normalized = spec.strip()
    if not normalized:
        return DEFAULT_CROP_REGIONS
    if normalized in CROP_REGION_PRESETS:
        return CROP_REGION_PRESETS[normalized]
    regions: list[NormalizedCrop] = []
    for raw_region in normalized.split(";"):
        values = [float(value.strip()) for value in raw_region.split(",") if value.strip()]
        if len(values) != 4:
            raise ValueError("crop region spec must be a preset or semicolon-separated x0,y0,x1,y1 regions")
        regions.append((values[0], values[1], values[2], values[3]))
    if not regions:
        raise ValueError("crop region spec did not contain any regions")
    return tuple(regions)


def parse_adaptive_crop_regions(spec: str) -> tuple[tuple[NormalizedCrop, ...], tuple[NormalizedCrop, ...], int]:
    normalized = spec.strip()
    if normalized in ADAPTIVE_CROP_REGION_PRESETS:
        return ADAPTIVE_CROP_REGION_PRESETS[normalized]
    raise ValueError("adaptive crop region spec must be one of: " + ", ".join(sorted(ADAPTIVE_CROP_REGION_PRESETS)))


def crop_region_pixels(width: int, height: int, region: Sequence[float]) -> tuple[int, int, int, int]:
    if len(region) != 4:
        raise ValueError("crop region must contain four normalized values")
    x0_f, y0_f, x1_f, y1_f = [float(value) for value in region]
    x0 = max(0, min(width, int(round(x0_f * width))))
    y0 = max(0, min(height, int(round(y0_f * height))))
    x1 = max(0, min(width, int(round(x1_f * width))))
    y1 = max(0, min(height, int(round(y1_f * height))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop region must have positive area")
    return x0, y0, x1, y1


def offset_crop_detections(detections: Iterable[Detection], *, x0: int, y0: int) -> list[Detection]:
    offset: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        shifted = dict(detection)
        shifted["bbox"] = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
        offset.append(shifted)
    return offset


def merge_tiled_detections(detections: Iterable[Detection], *, iou_threshold: float) -> list[Detection]:
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
        bbox = _bbox(detection)
        if not _valid_person_box(bbox):
            continue
        if all(_bbox_iou(bbox, _bbox(existing)) <= iou_threshold for existing in merged):
            merged.append(detection)
    return merged


def yolo_tiled_detections_for_frame(
    model: Any,
    frame: Any,
    *,
    crop_regions: Sequence[NormalizedCrop] = DEFAULT_CROP_REGIONS,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    nms_iou: float = 0.55,
) -> list[Detection]:
    height, width = frame.shape[:2]
    detections: list[Detection] = []
    for region in crop_regions:
        x0, y0, x1, y1 = crop_region_pixels(width, height, region)
        crop = frame[y0:y1, x0:x1]
        result = model.predict(
            crop,
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        crop_detections = _detections_from_yolo_result(result)
        detections.extend(offset_crop_detections(crop_detections, x0=x0, y0=y0))
    return merge_tiled_detections(detections, iou_threshold=nms_iou)


def yolo_tiled_detections_for_frames_batched(
    *,
    model: Any,
    frames: Iterable[Any],
    fps: float,
    crop_regions: Sequence[NormalizedCrop] = DEFAULT_CROP_REGIONS,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    nms_iou: float = 0.55,
    batch_size: int = 32,
    half: bool | None = None,
) -> dict[str, Any]:
    """Run tiled person detection with crop-level batching across frames."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frame_detections: dict[int, list[Detection]] = {}
    frame_order: list[int] = []
    pending_crops: list[Any] = []
    pending_meta: list[tuple[int, int, int]] = []

    def flush_pending() -> None:
        if not pending_crops:
            return
        _predict_tiled_crop_batch(
            model=model,
            crops=pending_crops,
            meta=pending_meta,
            frame_detections=frame_detections,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            batch_size=batch_size,
            half=half,
        )
        pending_crops.clear()
        pending_meta.clear()

    for frame_index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        frame_order.append(frame_index)
        frame_detections[frame_index] = []
        for region in crop_regions:
            x0, y0, x1, y1 = crop_region_pixels(width, height, region)
            pending_crops.append(frame[y0:y1, x0:x1])
            pending_meta.append((frame_index, x0, y0))
            if len(pending_crops) >= batch_size:
                flush_pending()
    flush_pending()

    output_frames: list[dict[str, Any]] = []
    for frame_index in frame_order:
        detections = merge_tiled_detections(frame_detections[frame_index], iou_threshold=nms_iou)
        for det_index, detection in enumerate(detections, start=1):
            detection["track_id"] = det_index
        output_frames.append({"frame": frame_index, "detections": detections})
    return {"schema_version": 1, "fps": fps, "frames": output_frames}


def yolo_adaptive_tiled_detections_for_frames_batched(
    *,
    model: Any,
    frames: Iterable[Any],
    fps: float,
    primary_crop_regions: Sequence[NormalizedCrop],
    fallback_crop_regions: Sequence[NormalizedCrop],
    min_detections: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    nms_iou: float = 0.55,
    batch_size: int = 32,
    half: bool | None = None,
) -> dict[str, Any]:
    """Run full-frame detection first, then run fallback crops only on sparse frames."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if min_detections <= 0:
        raise ValueError("min_detections must be positive")
    primary_regions = tuple(primary_crop_regions)
    fallback_regions = tuple(fallback_crop_regions)
    if not primary_regions:
        raise ValueError("primary_crop_regions must not be empty")

    frame_detections: dict[int, list[Detection]] = {}
    frame_order: list[int] = []
    frame_buffer: list[tuple[int, Any]] = []
    frame_buffer_limit = max(1, batch_size // len(primary_regions))
    crop_eval_count = 0
    fallback_frame_count = 0

    def run_regions(frame_items: list[tuple[int, Any]], regions: tuple[NormalizedCrop, ...]) -> int:
        pending_crops: list[Any] = []
        pending_meta: list[tuple[int, int, int]] = []
        eval_count = 0

        def flush_pending() -> None:
            if not pending_crops:
                return
            _predict_tiled_crop_batch(
                model=model,
                crops=pending_crops,
                meta=pending_meta,
                frame_detections=frame_detections,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                batch_size=batch_size,
                half=half,
            )
            pending_crops.clear()
            pending_meta.clear()

        for frame_index, frame in frame_items:
            height, width = frame.shape[:2]
            for region in regions:
                x0, y0, x1, y1 = crop_region_pixels(width, height, region)
                pending_crops.append(frame[y0:y1, x0:x1])
                pending_meta.append((frame_index, x0, y0))
                eval_count += 1
                if len(pending_crops) >= batch_size:
                    flush_pending()
        flush_pending()
        return eval_count

    def flush_frame_buffer() -> None:
        nonlocal crop_eval_count, fallback_frame_count
        if not frame_buffer:
            return
        crop_eval_count += run_regions(frame_buffer, primary_regions)
        fallback_items: list[tuple[int, Any]] = []
        for frame_index, frame in frame_buffer:
            primary_merged = merge_tiled_detections(frame_detections[frame_index], iou_threshold=nms_iou)
            if len(primary_merged) < min_detections:
                fallback_items.append((frame_index, frame))
        if fallback_items and fallback_regions:
            fallback_frame_count += len(fallback_items)
            crop_eval_count += run_regions(fallback_items, fallback_regions)
        frame_buffer.clear()

    for frame_index, frame in enumerate(frames):
        frame_order.append(frame_index)
        frame_detections[frame_index] = []
        frame_buffer.append((frame_index, frame))
        if len(frame_buffer) >= frame_buffer_limit:
            flush_frame_buffer()
    flush_frame_buffer()

    output_frames: list[dict[str, Any]] = []
    for frame_index in frame_order:
        detections = merge_tiled_detections(frame_detections[frame_index], iou_threshold=nms_iou)
        for det_index, detection in enumerate(detections, start=1):
            detection["track_id"] = det_index
        output_frames.append({"frame": frame_index, "detections": detections})
    return {
        "schema_version": 1,
        "fps": fps,
        "frames": output_frames,
        "crop_eval_count": crop_eval_count,
        "fallback_frame_count": fallback_frame_count,
        "primary_crop_region_count": len(primary_regions),
        "fallback_crop_region_count": len(fallback_regions),
        "adaptive_min_detections": min_detections,
    }


def yolo_tiled_detections_payload(
    *,
    model: Any,
    video_path: str | Path,
    fps: float,
    max_frames: int | None,
    crop_regions: Sequence[NormalizedCrop] = DEFAULT_CROP_REGIONS,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    nms_iou: float = 0.55,
    batch_size: int = 32,
    half: bool | None = None,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for tiled person detection") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        return yolo_tiled_detections_for_frames_batched(
            model=model,
            frames=_iter_video_frames(cap, max_frames=max_frames),
            fps=fps,
            crop_regions=crop_regions,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            nms_iou=nms_iou,
            batch_size=batch_size,
            half=half,
        )
    finally:
        cap.release()


def yolo_adaptive_tiled_detections_payload(
    *,
    model: Any,
    video_path: str | Path,
    fps: float,
    max_frames: int | None,
    primary_crop_regions: Sequence[NormalizedCrop],
    fallback_crop_regions: Sequence[NormalizedCrop],
    min_detections: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    nms_iou: float = 0.55,
    batch_size: int = 32,
    half: bool | None = None,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for tiled person detection") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        return yolo_adaptive_tiled_detections_for_frames_batched(
            model=model,
            frames=_iter_video_frames(cap, max_frames=max_frames),
            fps=fps,
            primary_crop_regions=primary_crop_regions,
            fallback_crop_regions=fallback_crop_regions,
            min_detections=min_detections,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            nms_iou=nms_iou,
            batch_size=batch_size,
            half=half,
        )
    finally:
        cap.release()


def _iter_video_frames(cap: Any, *, max_frames: int | None) -> Iterator[Any]:
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            break


def _predict_tiled_crop_batch(
    *,
    model: Any,
    crops: list[Any],
    meta: list[tuple[int, int, int]],
    frame_detections: dict[int, list[Detection]],
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    batch_size: int,
    half: bool | None,
) -> None:
    kwargs: dict[str, Any] = {
        "classes": [0],
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "batch": batch_size,
        "verbose": False,
    }
    if half is not None:
        kwargs["half"] = half
    results = model.predict(list(crops), **kwargs)
    if len(results) != len(meta):
        raise RuntimeError(f"YOLO returned {len(results)} results for {len(meta)} tiled crops")
    for result, (frame_index, x0, y0) in zip(results, meta, strict=True):
        detections = _detections_from_yolo_result(result)
        frame_detections[frame_index].extend(offset_crop_detections(detections, x0=x0, y0=y0))


def _detections_from_yolo_result(result: Any) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        bbox = [float(value) for value in box.xyxy[0].tolist()]
        score = float(box.conf[0])
        detections.append({"bbox": bbox, "conf": score, "class": "person"})
    return detections


def _bbox(detection: Detection) -> tuple[float, float, float, float]:
    raw = detection.get("bbox")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four values")
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _valid_person_box(bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width >= 4.0 and height >= 12.0


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


__all__ = [
    "DEFAULT_CROP_REGIONS",
    "CROP_REGION_PRESETS",
    "ADAPTIVE_CROP_REGION_PRESETS",
    "crop_region_pixels",
    "merge_tiled_detections",
    "offset_crop_detections",
    "parse_adaptive_crop_regions",
    "parse_crop_regions",
    "yolo_adaptive_tiled_detections_for_frames_batched",
    "yolo_adaptive_tiled_detections_payload",
    "yolo_tiled_detections_for_frame",
    "yolo_tiled_detections_for_frames_batched",
    "yolo_tiled_detections_payload",
]
