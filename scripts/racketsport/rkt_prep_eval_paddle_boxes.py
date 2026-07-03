#!/usr/bin/env python3
"""CPU-only paddle-box evaluator for the RKT prep lane.

This script scores paddle detector predictions at IoU 0.5 against either:

* the external-corpus validation split produced by
  ``rkt_prep_build_training_configs.py``; or
* CVAT paddle rectangles in explicit review-only mode.

It does not run a detector, train a model, or promote any RKT gate evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_TYPE = "racketsport_rkt_prep_paddle_box_eval"
SCHEMA_VERSION = 1


Box = tuple[float, float, float, float]
BoxByKey = dict[str, list[dict[str, Any]]]


def evaluate_external_split(
    *,
    split_manifest_path: str | Path,
    predictions_path: str | Path,
    split: str = "val",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score predictions against an external split manifest."""

    manifest_path = Path(split_manifest_path)
    manifest = _load_json(manifest_path)
    rows = [row for row in manifest.get("rows", []) if isinstance(row, Mapping) and row.get("split") == split]
    if not rows:
        raise ValueError(f"split manifest has no rows for split {split!r}: {manifest_path}")

    ground_truth: BoxByKey = {}
    keys: set[str] = set()
    for row in rows:
        key = _external_row_key(row)
        keys.add(key)
        label_path = _resolve_path(Path(str(row.get("label_path", ""))), base=manifest_path.parent)
        ground_truth[key] = [
            {"bbox_xyxy": box, "score": 1.0}
            for box in _load_yolo_boxes(label_path, expect_score=False)
        ]

    predictions = _load_predictions(Path(predictions_path), mode="external", keys=keys)
    metrics = score_boxes(ground_truth, predictions, iou_threshold=iou_threshold)
    return _report(
        dataset={
            "source": "external_corpus_val_split",
            "review_only": False,
            "trusted_for_training": False,
            "split_manifest": str(manifest_path),
            "split": split,
        },
        predictions_path=Path(predictions_path),
        metrics=metrics,
        iou_threshold=iou_threshold,
        notes=[
            "Scores detector predictions against the external Roboflow corpus validation split.",
            "This is an interim detector metric only; it is not face-angle or 6DoF RKT gate evidence.",
        ],
    )


def evaluate_cvat_review_boxes(
    *,
    cvat_manifest_path: str | Path,
    predictions_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score predictions against CVAT paddle rectangles in review-only mode."""

    manifest_path = Path(cvat_manifest_path)
    ground_truth = _load_cvat_paddle_boxes(manifest_path)
    predictions = _load_predictions(Path(predictions_path), mode="cvat", keys=set(ground_truth))
    metrics = score_boxes(ground_truth, predictions, iou_threshold=iou_threshold)
    return _report(
        dataset={
            "source": "cvat_paddle_rectangles",
            "review_only": True,
            "trusted_for_training": False,
            "cvat_manifest": str(manifest_path),
        },
        predictions_path=Path(predictions_path),
        metrics=metrics,
        iou_threshold=iou_threshold,
        notes=[
            "CVAT paddle rectangles are eval-clip labels allowed for scoring only.",
            "They must never be copied into a training split or used for checkpoint selection.",
            "This report is review_only and is not gate evidence for true paddle corners, face angle, or 6DoF.",
        ],
    )


def score_boxes(
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute one-class precision, recall, and AP at the requested IoU."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    total_gt = sum(len(items) for items in ground_truth.values())
    all_predictions: list[tuple[float, str, int, Box]] = []
    for key, items in predictions.items():
        for index, item in enumerate(items):
            score = float(item.get("score", 1.0))
            all_predictions.append((score, key, index, _as_xyxy(item)))
    all_predictions.sort(key=lambda item: item[0], reverse=True)

    matched: set[tuple[str, int]] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    matched_pairs: list[dict[str, Any]] = []

    for score, key, pred_index, pred_box in all_predictions:
        best_iou = 0.0
        best_gt_index: int | None = None
        for gt_index, gt in enumerate(ground_truth.get(key, [])):
            if (key, gt_index) in matched:
                continue
            iou = bbox_iou(pred_box, _as_xyxy(gt))
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched.add((key, best_gt_index))
            tp_flags.append(1)
            fp_flags.append(0)
            matched_pairs.append(
                {
                    "key": key,
                    "prediction_index": pred_index,
                    "ground_truth_index": best_gt_index,
                    "iou": best_iou,
                    "score": score,
                }
            )
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp = sum(tp_flags)
    fp = sum(fp_flags)
    fn = max(0, total_gt - tp)
    return {
        "iou_threshold": iou_threshold,
        "ground_truth_count": total_gt,
        "prediction_count": len(all_predictions),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision50": _ratio(tp, tp + fp),
        "recall50": _ratio(tp, total_gt),
        "ap50": _average_precision(tp_flags, fp_flags, total_gt),
        "map50": _average_precision(tp_flags, fp_flags, total_gt),
        "matched_pairs_sample": matched_pairs[:20],
    }


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _load_cvat_paddle_boxes(manifest_path: Path) -> BoxByKey:
    manifest = _load_json(manifest_path)
    rows: BoxByKey = defaultdict(list)
    for clip in manifest.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        paddle_info = clip.get("datasets", {}).get("paddle", {}) if isinstance(clip.get("datasets"), Mapping) else {}
        raw_path = paddle_info.get("path")
        if not raw_path:
            continue
        label_path = _resolve_path(Path(str(raw_path)), base=manifest_path.parent)
        payload = _load_json(label_path)
        items = payload.get("annotation", {}).get("items", [])
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("label") or item.get("class_name") or "").lower() != "paddle":
                continue
            clip_id = str(item.get("clip_id") or clip.get("clip_id") or "unknown")
            frame_index = int(item.get("frame_index", 0))
            rows[_cvat_key(clip_id, frame_index)].append({"bbox_xyxy": _as_xyxy(item), "score": 1.0})
    if not rows:
        raise ValueError(f"no CVAT paddle boxes found in manifest: {manifest_path}")
    return dict(rows)


def _load_predictions(path: Path, *, mode: str, keys: set[str]) -> BoxByKey:
    if path.is_dir():
        if mode != "external":
            raise ValueError("directory predictions are supported only for external YOLO split scoring")
        return _load_yolo_prediction_dir(path, keys=keys)
    payload = _load_json(path)
    return _load_json_predictions(payload, mode=mode)


def _load_yolo_prediction_dir(path: Path, *, keys: set[str]) -> BoxByKey:
    predictions: BoxByKey = {}
    for key in keys:
        label_path = path / f"{key}.txt"
        if label_path.is_file():
            predictions[key] = [
                {"bbox_xyxy": box, "score": score}
                for box, score in _load_yolo_prediction_boxes(label_path)
            ]
        else:
            predictions[key] = []
    return predictions


def _load_json_predictions(payload: Any, *, mode: str) -> BoxByKey:
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if records is None and isinstance(payload, Mapping):
        records = payload.get("predictions") or payload.get("items")
    if records is None and isinstance(payload, list):
        records = payload
    if not isinstance(records, list):
        raise ValueError("prediction JSON must contain records/predictions/items list or be a list")

    predictions: BoxByKey = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if isinstance(record.get("detections"), list):
            key = _prediction_record_key(record, mode=mode)
            for detection in record["detections"]:
                if isinstance(detection, Mapping):
                    predictions[key].append(_prediction_item(detection))
            continue
        key = _prediction_record_key(record, mode=mode)
        predictions[key].append(_prediction_item(record))
    return dict(predictions)


def _prediction_record_key(record: Mapping[str, Any], *, mode: str) -> str:
    if mode == "cvat":
        return _cvat_key(str(record.get("clip_id", "unknown")), int(record.get("frame_index", 0)))
    raw = record.get("image_id") or record.get("image_stem") or record.get("stem") or record.get("image_path")
    if raw is None:
        return _cvat_key(str(record.get("clip_id", "unknown")), int(record.get("frame_index", 0)))
    return Path(str(raw)).stem


def _prediction_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"bbox_xyxy": _as_xyxy(item), "score": float(item.get("score", item.get("confidence", 1.0)))}


def _load_yolo_boxes(path: Path, *, expect_score: bool) -> list[Box]:
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != 0:
            continue
        cx, cy, width, height = [float(value) for value in parts[1:5]]
        boxes.append(_xywh_norm_to_xyxy(cx, cy, width, height))
        if expect_score and len(parts) < 6:
            raise ValueError(f"prediction label line is missing confidence: {path}")
    return boxes


def _load_yolo_prediction_boxes(path: Path) -> list[tuple[Box, float]]:
    boxes: list[tuple[Box, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != 0:
            continue
        cx, cy, width, height = [float(value) for value in parts[1:5]]
        score = float(parts[5]) if len(parts) >= 6 else 1.0
        boxes.append((_xywh_norm_to_xyxy(cx, cy, width, height), score))
    return boxes


def _xywh_norm_to_xyxy(cx: float, cy: float, width: float, height: float) -> Box:
    return (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5)


def _as_xyxy(item: Mapping[str, Any] | Sequence[float]) -> Box:
    if isinstance(item, Mapping):
        if "bbox_xyxy" in item:
            values = item["bbox_xyxy"]
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == 4:
                return tuple(float(value) for value in values)  # type: ignore[return-value]
        if "bbox_xywh" in item:
            values = item["bbox_xywh"]
        elif "bbox" in item:
            values = item["bbox"]
        else:
            raise ValueError(f"box item lacks bbox_xyxy/bbox_xywh/bbox: {item}")
    else:
        values = item
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        raise ValueError(f"box must contain four numeric values: {values}")
    x, y, width, height = [float(value) for value in values]
    return (x, y, x + width, y + height)


def _external_row_key(row: Mapping[str, Any]) -> str:
    return Path(str(row.get("image_path") or row.get("output_image") or "")).stem


def _cvat_key(clip_id: str, frame_index: int) -> str:
    return f"{clip_id}:{int(frame_index):06d}"


def _average_precision(tp_flags: Sequence[int], fp_flags: Sequence[int], total_gt: int) -> float:
    if total_gt <= 0:
        return 0.0
    if not tp_flags:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / total_gt)
        precisions.append(_ratio(cum_tp, cum_tp + cum_fp))
    recalls.append(1.0)
    precisions.append(0.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    ap = 0.0
    for index in range(1, len(recalls)):
        delta = recalls[index] - recalls[index - 1]
        if delta > 0.0:
            ap += delta * precisions[index]
    return ap


def _ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _resolve_path(path: Path, *, base: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    for candidate in (base / path, base.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(
    *,
    dataset: dict[str, Any],
    predictions_path: Path,
    metrics: dict[str, Any],
    iou_threshold: float,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "stage": "rkt_prep_paddle_detector_eval",
        "dataset": dataset,
        "predictions_path": str(predictions_path),
        "metric_definitions": {
            "precision50": "TP / (TP + FP) using one-to-one greedy matches at IoU >= 0.5.",
            "recall50": "TP / GT using one-to-one greedy matches at IoU >= 0.5.",
            "map50": "One-class interpolated AP at IoU 0.5; same value as AP50 because only paddle class is scored.",
        },
        "iou_threshold": iou_threshold,
        "metrics": metrics,
        "notes": notes,
        "not_gate_evidence": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score RKT prep paddle detector predictions on CPU.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction directory or records JSON.")
    parser.add_argument(
        "--mode",
        choices=("external-val", "cvat-review"),
        default="external-val",
        help="Score external validation split or CVAT review-only rectangles.",
    )
    parser.add_argument("--split-manifest", type=Path, help="External split_manifest.json from rkt_prep config.")
    parser.add_argument("--split", default="val", help="Split to score from --split-manifest.")
    parser.add_argument(
        "--cvat-manifest",
        type=Path,
        default=Path("runs/cvat_imports/2026_06_30/gate_inputs/manifest.json"),
        help="CVAT gate-input manifest for --mode cvat-review.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    try:
        if args.mode == "external-val":
            if args.split_manifest is None:
                raise ValueError("--split-manifest is required for --mode external-val")
            report = evaluate_external_split(
                split_manifest_path=args.split_manifest,
                predictions_path=args.predictions,
                split=args.split,
                iou_threshold=args.iou_threshold,
            )
        else:
            report = evaluate_cvat_review_boxes(
                cvat_manifest_path=args.cvat_manifest,
                predictions_path=args.predictions,
                iou_threshold=args.iou_threshold,
            )
    except Exception as exc:
        print(f"RKT prep paddle-box eval failed: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
