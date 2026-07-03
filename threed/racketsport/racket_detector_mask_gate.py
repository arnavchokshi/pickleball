"""Real detector+SAM2 mask gate utilities for RKT step 1.

This module keeps metric/report code lightweight at import time. The actual
GroundingDINO and SAM2 runtimes are imported only by the CLI inference path.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_TYPE = "racketsport_racket_detector_mask_gate"
SCHEMA_VERSION = 1


def match_detections_to_labels(
    labels: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    """Greedily match detections to labels by descending IoU."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    pairs: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(labels):
        label_box = _bbox_xyxy(label)
        for detection_index, detection in enumerate(detections):
            iou = bbox_iou(label_box, _bbox_xyxy(detection))
            if iou >= iou_threshold:
                pairs.append((iou, label_index, detection_index))

    matched_labels: set[int] = set()
    matched_detections: set[int] = set()
    for _iou, label_index, detection_index in sorted(pairs, reverse=True):
        if label_index in matched_labels or detection_index in matched_detections:
            continue
        matched_labels.add(label_index)
        matched_detections.add(detection_index)

    label_count = len(labels)
    detection_count = len(detections)
    match_count = len(matched_labels)
    return {
        "label_count": label_count,
        "detection_count": detection_count,
        "match_count": match_count,
        "false_positive_count": max(0, detection_count - len(matched_detections)),
        "recall": _ratio(match_count, label_count),
        "precision": _ratio(match_count, detection_count),
    }


def build_detector_mask_report(
    records: Sequence[Mapping[str, Any]],
    *,
    model_sources: Mapping[str, str],
    iou_threshold: float,
    recall_gate: float,
) -> dict[str, Any]:
    """Build the gate report from per-frame detector/mask records."""

    if not 0.0 <= recall_gate <= 1.0:
        raise ValueError("recall_gate must be in [0, 1]")
    if not records:
        raise ValueError("records must not be empty")

    per_clip_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        per_clip_records[str(record.get("clip_id", "unknown"))].append(record)

    per_clip = {
        clip_id: _summarize_records(clip_records, iou_threshold=iou_threshold)
        for clip_id, clip_records in sorted(per_clip_records.items())
    }
    overall = _summarize_records(records, iou_threshold=iou_threshold)
    per_clip_pass = all(
        metrics["label_count"] > 0 and (metrics["recall"] or 0.0) >= recall_gate
        for metrics in per_clip.values()
    )
    overall_pass = overall["label_count"] > 0 and (overall["recall"] or 0.0) >= recall_gate
    mask_pass = overall["detection_count"] > 0 and (overall["mask_coverage_rate"] or 0.0) > 0.0
    status = "pass" if overall_pass and per_clip_pass and mask_pass else "fail"

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "stage": "racket_detector_mask",
        "status": status,
        "model_sources": dict(model_sources),
        "gate": {
            "detector_recall_iou_threshold": iou_threshold,
            "recall_gate_per_clip_and_overall": recall_gate,
            "mask_required": True,
        },
        "execution": {
            "runs_inference": True,
            "real_model": True,
            "uses_label_bbox_as_prediction": False,
            "claims_true_corners": False,
            "claims_6dof": False,
            "not_final_rkt_verified": True,
        },
        "metrics": overall,
        "per_clip": per_clip,
        "notes": [
            "CVAT paddle rectangles are used only as detector recall labels.",
            "This gate does not produce true paddle corners, 6DoF pose, or final RKT verification.",
        ],
    }


def load_yolo_rows(
    manifest_path: str | Path,
    *,
    max_frames_per_clip: int | None = None,
    frame_stride: int = 1,
    clip_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load balanced image/label rows from an exported YOLO paddle dataset."""

    if max_frames_per_clip is not None and max_frames_per_clip <= 0:
        raise ValueError("max_frames_per_clip must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("YOLO dataset manifest requires rows")

    selected: list[dict[str, Any]] = []
    counts_by_clip: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id", ""))
        if clip_ids is not None and clip_id not in clip_ids:
            continue
        frame_index = int(row.get("frame_index", -1))
        if frame_index < 0 or frame_index % frame_stride != 0:
            continue
        if max_frames_per_clip is not None and counts_by_clip[clip_id] >= max_frames_per_clip:
            continue
        image_path = Path(str(row.get("image_path", "")))
        label_path = Path(str(row.get("label_path", "")))
        if not image_path.is_file() or not label_path.is_file():
            continue
        selected.append(
            {
                "clip_id": clip_id,
                "frame_index": frame_index,
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )
        counts_by_clip[clip_id] += 1
    if not selected:
        raise ValueError("no YOLO rows selected for detector/mask gate")
    return selected


def labels_from_yolo_file(label_path: str | Path, *, image_width: int, image_height: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _class_id, cx, cy, width, height = [float(part) for part in parts]
        x1 = (cx - width * 0.5) * image_width
        y1 = (cy - height * 0.5) * image_height
        x2 = (cx + width * 0.5) * image_width
        y2 = (cy + height * 0.5) * image_height
        labels.append({"bbox_xyxy": [x1, y1, x2, y2]})
    return labels


def detections_from_yolo_boxes(
    *,
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float],
    classes: Sequence[float] | None = None,
    class_id: int = 0,
    label: str = "paddle",
) -> list[dict[str, Any]]:
    """Convert Ultralytics YOLO boxes into the detector gate record shape."""

    detections: list[dict[str, Any]] = []
    for index, box in enumerate(boxes_xyxy):
        if classes is not None and index < len(classes) and int(classes[index]) != class_id:
            continue
        if len(box) != 4:
            raise ValueError("YOLO boxes must be xyxy with four values")
        detections.append(
            {
                "bbox_xyxy": [float(value) for value in box],
                "score": float(scores[index]) if index < len(scores) else None,
                "text_label": label,
            }
        )
    return detections


def iter_image_tiles(
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> list[list[int]]:
    """Return xyxy tiles that cover an image, including the right/bottom edge."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        values = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    tiles: list[list[int]] = []
    for y1 in starts(image_height):
        for x1 in starts(image_width):
            tiles.append([x1, y1, min(image_width, x1 + tile_size), min(image_height, y1 + tile_size)])
    return tiles


def merge_tiled_detections(
    tiled_detections: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    """Translate tile-local detections to full-frame coordinates and NMS duplicates."""

    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    translated: list[dict[str, Any]] = []
    for tile_record in tiled_detections:
        tile_xyxy = tile_record.get("tile_xyxy")
        if not isinstance(tile_xyxy, Sequence) or isinstance(tile_xyxy, (str, bytes)) or len(tile_xyxy) != 4:
            raise ValueError("tile detection record requires tile_xyxy")
        tx1, ty1, tx2, ty2 = [float(value) for value in tile_xyxy]
        for detection in tile_record.get("detections", []):
            if not isinstance(detection, Mapping):
                continue
            x1, y1, x2, y2 = _bbox_xyxy(detection)
            full_box = [
                _clamp(x1 + tx1, 0.0, float(image_width)),
                _clamp(y1 + ty1, 0.0, float(image_height)),
                _clamp(x2 + tx1, 0.0, float(image_width)),
                _clamp(y2 + ty1, 0.0, float(image_height)),
            ]
            if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
                continue
            merged = dict(detection)
            merged["bbox_xyxy"] = full_box
            merged["tile_xyxy"] = [tx1, ty1, tx2, ty2]
            translated.append(merged)

    kept: list[dict[str, Any]] = []
    for detection in sorted(translated, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        if any(bbox_iou(detection["bbox_xyxy"], existing["bbox_xyxy"]) >= nms_iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return kept


def filter_detections_by_box_geometry(
    detections: Sequence[Mapping[str, Any]],
    *,
    min_box_area: float | None = None,
    max_box_area: float | None = None,
    min_box_aspect: float | None = None,
    max_box_aspect: float | None = None,
) -> list[dict[str, Any]]:
    """Filter detections using label-free full-frame box geometry bounds."""

    filtered: list[dict[str, Any]] = []
    for detection in detections:
        x1, y1, x2, y2 = _bbox_xyxy(detection)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        area = width * height
        aspect = width / height
        if min_box_area is not None and area < min_box_area:
            continue
        if max_box_area is not None and area > max_box_area:
            continue
        if min_box_aspect is not None and aspect < min_box_aspect:
            continue
        if max_box_aspect is not None and aspect > max_box_aspect:
            continue
        filtered.append(dict(detection))
    return filtered


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "racketsport_racket_detector_mask_records",
            "records": list(records),
        },
    )


def run_grounding_dino_sam2_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    detector_model: str,
    sam2_model: str,
    prompt: str,
    detector_device: str,
    sam2_device: str,
    box_threshold: float,
    text_threshold: float,
    mask_dir: str | Path,
    tile_size: int | None = None,
    tile_overlap: int = 0,
    tile_nms_iou: float = 0.6,
    min_box_area: float | None = None,
    max_box_area: float | None = None,
    min_box_aspect: float | None = None,
    max_box_aspect: float | None = None,
    sam2_box_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run real GroundingDINO detections and real SAM2 box-prompt masks."""

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    mask_root = Path(mask_dir)
    mask_root.mkdir(parents=True, exist_ok=True)
    detector_device_name = _resolve_device(detector_device, torch)
    sam2_device_name = _resolve_device(sam2_device, torch)

    processor = AutoProcessor.from_pretrained(detector_model)
    detector = GroundingDinoForObjectDetection.from_pretrained(detector_model).to(detector_device_name)
    detector.eval()
    sam2 = build_sam2_hf(sam2_model, device=sam2_device_name)
    predictor = SAM2ImagePredictor(sam2)

    records: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        labels = labels_from_yolo_file(row["label_path"], image_width=width, image_height=height)
        started = time.perf_counter()
        tiled = []
        tiles = (
            iter_image_tiles(image_width=width, image_height=height, tile_size=tile_size, overlap=tile_overlap)
            if tile_size is not None and tile_size > 0
            else [[0, 0, width, height]]
        )
        for tile_xyxy in tiles:
            x1, y1, x2, y2 = tile_xyxy
            tile_image = image.crop((x1, y1, x2, y2))
            tile_width, tile_height = tile_image.size
            inputs = processor(images=tile_image, text=prompt, return_tensors="pt")
            inputs = {
                key: (value.to(detector_device_name) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = detector(**inputs)
            detections = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(tile_height, tile_width)],
            )[0]
            tile_boxes = (
                detections["boxes"].detach().cpu().numpy()
                if len(detections["boxes"])
                else np.zeros((0, 4), dtype=np.float32)
            )
            tile_scores = (
                detections["scores"].detach().cpu().numpy()
                if len(detections["scores"])
                else np.zeros((0,), dtype=np.float32)
            )
            text_labels = detections.get("text_labels", detections.get("labels", []))
            tile_detections = []
            for index, box in enumerate(tile_boxes):
                tile_detections.append(
                    {
                        "bbox_xyxy": [float(value) for value in box.tolist()],
                        "score": float(tile_scores[index]) if index < len(tile_scores) else None,
                        "text_label": str(text_labels[index]) if index < len(text_labels) else prompt,
                    }
                )
            tiled.append({"tile_xyxy": tile_xyxy, "detections": tile_detections})
        frame_detections = merge_tiled_detections(
            tiled,
            image_width=width,
            image_height=height,
            nms_iou_threshold=tile_nms_iou,
        )
        frame_detections = filter_detections_by_box_geometry(
            frame_detections,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            min_box_aspect=min_box_aspect,
            max_box_aspect=max_box_aspect,
        )
        boxes = np.asarray([detection["bbox_xyxy"] for detection in frame_detections], dtype=np.float32)
        mask_infos = _predict_and_write_masks(
            predictor,
            np.asarray(image),
            boxes,
            mask_root=mask_root,
            clip_id=str(row["clip_id"]),
            frame_index=int(row["frame_index"]),
            torch=torch,
            box_batch_size=sam2_box_batch_size,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for index, detection in enumerate(frame_detections):
            detection["mask"] = mask_infos[index] if index < len(mask_infos) else {"present": False, "area_px": 0}
        records.append(
            {
                "clip_id": str(row["clip_id"]),
                "frame_index": int(row["frame_index"]),
                "image_path": str(image_path),
                "label_path": str(row["label_path"]),
                "labels": labels,
                "detections": frame_detections,
                "runtime_ms": elapsed_ms,
            }
        )
    return records


def run_yolo_sam2_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    detector_model: str,
    sam2_model: str,
    detector_device: str,
    sam2_device: str,
    imgsz: int,
    conf: float,
    nms_iou: float,
    mask_dir: str | Path,
    class_id: int = 0,
    tile_size: int | None = None,
    tile_overlap: int = 0,
    tile_nms_iou: float = 0.6,
    min_box_area: float | None = None,
    max_box_area: float | None = None,
    min_box_aspect: float | None = None,
    max_box_aspect: float | None = None,
    sam2_box_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run real Ultralytics YOLO detections and real SAM2 box-prompt masks."""

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from ultralytics import YOLO

    mask_root = Path(mask_dir)
    mask_root.mkdir(parents=True, exist_ok=True)
    detector_device_name = _resolve_ultralytics_device(detector_device, torch)
    sam2_device_name = _resolve_device(sam2_device, torch)

    detector = YOLO(detector_model)
    sam2 = build_sam2_hf(sam2_model, device=sam2_device_name)
    predictor = SAM2ImagePredictor(sam2)

    records: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        labels = labels_from_yolo_file(row["label_path"], image_width=width, image_height=height)
        started = time.perf_counter()
        tiled = []
        tiles = (
            iter_image_tiles(image_width=width, image_height=height, tile_size=tile_size, overlap=tile_overlap)
            if tile_size is not None and tile_size > 0
            else [[0, 0, width, height]]
        )
        for tile_xyxy in tiles:
            x1, y1, x2, y2 = tile_xyxy
            tile_image = image.crop((x1, y1, x2, y2))
            yolo_result = detector.predict(
                source=np.asarray(tile_image),
                imgsz=imgsz,
                conf=conf,
                iou=nms_iou,
                device=detector_device_name,
                verbose=False,
            )[0]
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.float32)
            else:
                boxes = yolo_result.boxes.xyxy.detach().cpu().numpy()
                scores = yolo_result.boxes.conf.detach().cpu().numpy()
                classes = yolo_result.boxes.cls.detach().cpu().numpy()
            tiled.append(
                {
                    "tile_xyxy": tile_xyxy,
                    "detections": detections_from_yolo_boxes(
                        boxes_xyxy=boxes,
                        scores=scores,
                        classes=classes,
                        class_id=class_id,
                    ),
                }
            )
        frame_detections = merge_tiled_detections(
            tiled,
            image_width=width,
            image_height=height,
            nms_iou_threshold=tile_nms_iou,
        )
        frame_detections = filter_detections_by_box_geometry(
            frame_detections,
            min_box_area=min_box_area,
            max_box_area=max_box_area,
            min_box_aspect=min_box_aspect,
            max_box_aspect=max_box_aspect,
        )
        selected_boxes = np.asarray([detection["bbox_xyxy"] for detection in frame_detections], dtype=np.float32)
        mask_infos = _predict_and_write_masks(
            predictor,
            np.asarray(image),
            selected_boxes,
            mask_root=mask_root,
            clip_id=str(row["clip_id"]),
            frame_index=int(row["frame_index"]),
            torch=torch,
            box_batch_size=sam2_box_batch_size,
        )
        for index, detection in enumerate(frame_detections):
            detection["mask"] = mask_infos[index] if index < len(mask_infos) else {"present": False, "area_px": 0}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            {
                "clip_id": str(row["clip_id"]),
                "frame_index": int(row["frame_index"]),
                "image_path": str(image_path),
                "label_path": str(row["label_path"]),
                "labels": labels,
                "detections": frame_detections,
                "runtime_ms": elapsed_ms,
            }
        )
    return records


def _predict_and_write_masks(
    predictor: Any,
    image_rgb: Any,
    boxes: Any,
    *,
    mask_root: Path,
    clip_id: str,
    frame_index: int,
    torch: Any,
    box_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    if len(boxes) == 0:
        return []
    import numpy as np
    from PIL import Image

    if box_batch_size is not None and box_batch_size <= 0:
        raise ValueError("box_batch_size must be positive")
    batch_size = int(box_batch_size or len(boxes))
    predictor.set_image(image_rgb)
    infos: list[dict[str, Any]] = []
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start : start + batch_size].astype("float32")
        with torch.no_grad():
            masks, scores, _logits = predictor.predict(box=batch_boxes, multimask_output=False)
        mask_array = np.asarray(masks)
        if mask_array.ndim == 4:
            mask_array = mask_array[:, 0, :, :]
        if mask_array.ndim == 2:
            mask_array = mask_array[None, :, :]
        score_array = np.asarray(scores).reshape(-1)
        for offset, mask in enumerate(mask_array):
            index = start + offset
            binary = mask.astype(bool)
            mask_path = mask_root / f"{clip_id}_{frame_index:06d}_det{index:02d}.png"
            Image.fromarray((binary.astype("uint8") * 255)).save(mask_path)
            infos.append(
                {
                    "present": bool(binary.any()),
                    "area_px": int(binary.sum()),
                    "score": float(score_array[offset]) if offset < len(score_array) else None,
                    "path": str(mask_path),
                }
            )
    return infos


def _summarize_records(records: Iterable[Mapping[str, Any]], *, iou_threshold: float) -> dict[str, Any]:
    frame_count = 0
    label_count = 0
    detection_count = 0
    match_count = 0
    false_positive_count = 0
    mask_detection_count = 0
    runtime_ms: list[float] = []
    for record in records:
        frame_count += 1
        labels = list(record.get("labels", []))
        detections = list(record.get("detections", []))
        matched = match_detections_to_labels(labels, detections, iou_threshold=iou_threshold)
        label_count += int(matched["label_count"])
        detection_count += int(matched["detection_count"])
        match_count += int(matched["match_count"])
        false_positive_count += int(matched["false_positive_count"])
        for detection in detections:
            mask = detection.get("mask") if isinstance(detection, Mapping) else None
            if isinstance(mask, Mapping) and mask.get("present") is True and int(mask.get("area_px", 0)) > 0:
                mask_detection_count += 1
        runtime = record.get("runtime_ms")
        if isinstance(runtime, (int, float)) and not isinstance(runtime, bool) and math.isfinite(float(runtime)):
            runtime_ms.append(float(runtime))
    return {
        "frame_count": frame_count,
        "label_count": label_count,
        "detection_count": detection_count,
        "match_count": match_count,
        "false_positive_count": false_positive_count,
        "recall": _ratio(match_count, label_count),
        "precision": _ratio(match_count, detection_count),
        "mask_detection_count": mask_detection_count,
        "mask_coverage_rate": _ratio(mask_detection_count, detection_count),
        "mean_runtime_ms": sum(runtime_ms) / len(runtime_ms) if runtime_ms else None,
    }


def _bbox_xyxy(item: Mapping[str, Any]) -> list[float]:
    raw = item.get("bbox_xyxy")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise ValueError("item requires bbox_xyxy with four values")
    return [float(value) for value in raw]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_ultralytics_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


__all__ = [
    "ARTIFACT_TYPE",
    "bbox_iou",
    "build_detector_mask_report",
    "detections_from_yolo_boxes",
    "filter_detections_by_box_geometry",
    "iter_image_tiles",
    "labels_from_yolo_file",
    "load_yolo_rows",
    "merge_tiled_detections",
    "match_detections_to_labels",
    "run_grounding_dino_sam2_rows",
    "run_yolo_sam2_rows",
    "write_json",
    "write_records",
]
