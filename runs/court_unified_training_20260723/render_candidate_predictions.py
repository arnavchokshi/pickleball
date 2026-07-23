#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_court_keypoint_heatmap import (  # noqa: E402
    load_label_image,
    load_real_court_keypoint_labels,
)
from threed.racketsport.court_model_infer import infer_court_model  # noqa: E402


EDGES = (
    ("near_left_corner", "near_baseline_center"),
    ("near_baseline_center", "near_right_corner"),
    ("near_left_corner", "far_left_corner"),
    ("near_right_corner", "far_right_corner"),
    ("far_left_corner", "far_baseline_center"),
    ("far_baseline_center", "far_right_corner"),
    ("near_nvz_left", "near_nvz_center"),
    ("near_nvz_center", "near_nvz_right"),
    ("far_nvz_left", "far_nvz_center"),
    ("far_nvz_center", "far_nvz_right"),
    ("near_baseline_center", "near_nvz_center"),
    ("far_nvz_center", "far_baseline_center"),
    ("net_left_sideline", "net_center"),
    ("net_center", "net_right_sideline"),
)


def point(xy: list[float]) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def draw_lines(image: np.ndarray, points: dict[str, list[float]], color: tuple[int, int, int], width: int) -> None:
    for left, right in EDGES:
        if left in points and right in points:
            cv2.line(image, point(points[left]), point(points[right]), color, width, cv2.LINE_AA)


def make_tile(image: np.ndarray, title: str) -> np.ndarray:
    target_w, target_h = 640, 360
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h + 42, target_w, 3), dtype=np.uint8)
    canvas[42:] = resized
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--structured-best-court", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_real_court_keypoint_labels(args.real_root)
    comparison_tiles: list[np.ndarray] = []
    prediction_tiles: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        loaded = load_label_image(row, cv2=cv2, image_module=Image)
        image_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        loaded_w, loaded_h = loaded.size
        source_w, source_h = row.get("source_video_size") or [loaded_w, loaded_h]
        scale_x, scale_y = float(source_w) / loaded_w, float(source_h) / loaded_h

        result = infer_court_model(image_bgr, args.checkpoint, device=args.device)
        raw_predicted = {
            name: [float(xy[0]), float(xy[1])]
            for name, xy in result["keypoints_xy"].items()
            if not name.startswith("net_")
        }
        best_court = result.get("best_court") or {}
        selected_predictions = best_court.get("keypoints_xy") if args.structured_best_court else result["keypoints_xy"]
        predicted = {
            name: [float(xy[0]), float(xy[1])]
            for name, xy in selected_predictions.items()
            if not (args.structured_best_court and name.startswith("net_"))
        }
        truth = {
            name: [float(xy[0]) / scale_x, float(xy[1]) / scale_y]
            for name, xy in row["keypoints"].items()
            if xy is not None
        }
        errors = {
            name: math.hypot(
                predicted[name][0] * scale_x - float(row["keypoints"][name][0]),
                predicted[name][1] * scale_y - float(row["keypoints"][name][1]),
            )
            for name in truth
            if name in predicted
        }
        ordered_errors = sorted(errors.values())
        median_error = float(np.median(ordered_errors)) if ordered_errors else None
        pck5 = sum(value <= 5.0 for value in ordered_errors) / len(ordered_errors) if ordered_errors else None

        prediction_only = image_bgr.copy()
        if args.structured_best_court:
            for xy in raw_predicted.values():
                cv2.circle(prediction_only, point(xy), 3, (0, 165, 255), -1, cv2.LINE_AA)
        draw_lines(prediction_only, predicted, (40, 235, 40), 3)
        for marker_index, (name, xy) in enumerate(predicted.items(), start=1):
            cv2.circle(prediction_only, point(xy), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(prediction_only, point(xy), 5, (40, 235, 40), -1, cv2.LINE_AA)
            cv2.putText(prediction_only, str(marker_index), (point(xy)[0] + 6, point(xy)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        comparison = image_bgr.copy()
        if args.structured_best_court:
            for xy in raw_predicted.values():
                cv2.circle(comparison, point(xy), 3, (0, 165, 255), -1, cv2.LINE_AA)
            for ignored in best_court.get("ignored_observations") or []:
                if ignored.get("reason") not in {"residual_outlier", "duplicate_image_location"}:
                    continue
                xy = ignored.get("xy")
                if not isinstance(xy, list) or len(xy) != 2:
                    continue
                px, py = point(xy)
                cv2.line(comparison, (px - 5, py - 5), (px + 5, py + 5), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(comparison, (px - 5, py + 5), (px + 5, py - 5), (0, 0, 255), 2, cv2.LINE_AA)
        draw_lines(comparison, truth, (255, 255, 0), 2)
        draw_lines(comparison, predicted, (40, 235, 40), 3)
        for name, xy in truth.items():
            if name in predicted:
                cv2.line(comparison, point(xy), point(predicted[name]), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(comparison, point(xy), 6, (255, 255, 0), 2, cv2.LINE_AA)
        for xy in predicted.values():
            cv2.circle(comparison, point(xy), 5, (40, 235, 40), -1, cv2.LINE_AA)
        cv2.rectangle(comparison, (8, 8), (475, 42), (0, 0, 0), -1)
        legend = (
            "GREEN=best court  ORANGE=raw  CYAN=label  RED X=ignored"
            if args.structured_best_court
            else "GREEN=model  CYAN=label  MAGENTA=error"
        )
        cv2.putText(comparison, legend, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        stem = f"{index + 1:02d}_{row['clip']}_frame_{int(row['frame_index']):06d}"
        prediction_path = args.out_dir / f"{stem}_prediction.jpg"
        comparison_path = args.out_dir / f"{stem}_comparison.jpg"
        cv2.imwrite(str(prediction_path), prediction_only, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
        confidence_text = (
            f" | court conf {float(best_court.get('court_confidence') or 0.0):.3f}"
            if args.structured_best_court
            else ""
        )
        metric_text = f"{row['clip']} f{int(row['frame_index'])} | median {median_error:.1f}px | PCK5 {pck5:.0%}{confidence_text}"
        prediction_tiles.append(make_tile(prediction_only, metric_text))
        comparison_tiles.append(make_tile(comparison, metric_text))
        manifest_rows.append({
            "clip": row["clip"],
            "frame_index": int(row["frame_index"]),
            "prediction": str(prediction_path),
            "comparison": str(comparison_path),
            "labeled_keypoints": len(errors),
            "median_error_px_source": median_error,
            "pck_at_5px": pck5,
            "errors_px_source": errors,
            "court_confidence": best_court.get("court_confidence") if args.structured_best_court else None,
            "inlier_count": best_court.get("inlier_count") if args.structured_best_court else None,
            "ignored_observation_count": len(best_court.get("ignored_observations") or []) if args.structured_best_court else None,
        })

    for name, tiles in (("prediction_contact_sheet.jpg", prediction_tiles), ("comparison_contact_sheet.jpg", comparison_tiles)):
        rows_of_tiles = []
        for start in range(0, len(tiles), 2):
            pair = tiles[start:start + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows_of_tiles.append(np.hstack(pair))
        cv2.imwrite(str(args.out_dir / name), np.vstack(rows_of_tiles), [cv2.IMWRITE_JPEG_QUALITY, 94])

    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "court_structured_best_effort_prediction_gallery"
            if args.structured_best_court
            else "court_unet_v2_prediction_gallery"
        ),
        "checkpoint": str(args.checkpoint),
        "legend": {"model_prediction": "green", "human_label": "cyan", "error_connector": "magenta"},
        "status": "diagnostic_only" if args.structured_best_court else None,
        "promotion_allowed": False if args.structured_best_court else None,
        "rows": manifest_rows,
    }
    (args.out_dir / "prediction_gallery_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "out_dir": str(args.out_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
