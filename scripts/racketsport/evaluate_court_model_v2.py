#!/usr/bin/env python3
"""CAL-MODEL v2 real-row eval harness (2026-07-05).

Scores any `court_unet_v2` checkpoint (`threed.racketsport.court_model_infer.infer_court_model`)
against the owner's 32 reviewed CVAT court-keypoint rows (4 clips x 8 rows,
`eval_clips/ball/*/labels/court_keypoints.json`), and writes a `court_keypoint_metrics.json`
report shaped to match the fields the existing CAL evidence scanner
(`threed.racketsport.overlapping_court_calibration._neural_keypoint_checkpoint_evidence`) reads:
`checkpoint`, `gate.{value,threshold,passed}`, `after.real_keypoint_median_px`. This is read-only
inference + scoring: it never trains or mutates the checkpoint. Gate stays PCK@5px >= 0.95, scored
per-viewpoint (every clip must individually clear the threshold, not just the pooled average) --
same discipline as `scripts/racketsport/evaluate_court_keypoint_owner_gate.py`, never weakened.

Label-space rescale note (the "1280x720 label space -> correct rescale" the CAL-MODEL spec calls
out): each reviewed row's `keypoints` are stored in that row's `source_video_size` pixel space
(e.g. 1920x1080), but `load_label_image` may load a cached preview JPEG at a *different*
resolution than that (observed concretely: `eval_clips/ball/*/labels/court_keypoint_frames/*.jpg`
are saved at the row's `label_coordinate_space` resolution, 1280x720, while the clip's actual
`source.mp4` is 1920x1080). `infer_court_model` returns keypoints in whatever resolution the image
array handed to it has -- so this script always rescales the adapter's output by the ratio between
the row's declared `source_video_size` and the *actually loaded* image's own pixel size before
comparing against `row["keypoints"]`, rather than assuming the loaded image is already at source
resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_court_keypoint_heatmap import (  # noqa: E402
    load_label_image,
    load_real_court_keypoint_labels,
)
from scripts.racketsport.build_court_v31_protocol import load_protocol_partition_rows  # noqa: E402
from threed.racketsport.court_model_infer import (  # noqa: E402
    infer_court_model,
    load_court_model_checkpoint,
    make_court_model_infer_provider,
)
from threed.racketsport.court_structured_metrics import (  # noqa: E402
    evaluate_raw_vs_structured_court_outputs,
    evaluate_structured_court_outputs,
)

INDEPENDENT_REVIEWED_STATUS = "reviewed"


def _row_native_image_bgr(row: dict[str, Any]) -> tuple[Any, tuple[float, float]]:
    """Load a row's image at whatever resolution it is actually stored at, plus the (x, y) scale
    factor needed to map predictions made on that image up into `row["source_video_size"]` space
    (see module docstring)."""

    import cv2
    import numpy as np
    from PIL import Image

    image = load_label_image(row, cv2=cv2, image_module=Image)
    loaded_width, loaded_height = image.size
    source_size = row.get("source_video_size") or [loaded_width, loaded_height]
    source_width, source_height = float(source_size[0]), float(source_size[1])
    scale_to_source = (source_width / loaded_width, source_height / loaded_height)

    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    return image_bgr, scale_to_source


def predict_row_keypoints_source_px(
    row: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: str,
    heatmap_decoder: str | None = None,
    coordinate_transform: str | None = None,
    infer_provider: Any | None = None,
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] | None = None,
) -> dict[str, list[float]]:
    result, (scale_x, scale_y) = _infer_row(
        row,
        checkpoint_path,
        device=device,
        heatmap_decoder=heatmap_decoder,
        coordinate_transform=coordinate_transform,
        infer_provider=infer_provider,
        inference_cache=inference_cache,
    )
    return {
        name: [xy[0] * scale_x, xy[1] * scale_y]
        for name, xy in result["keypoints_xy"].items()
    }


def _infer_row(
    row: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: str,
    heatmap_decoder: str | None,
    coordinate_transform: str | None,
    infer_provider: Any | None,
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] | None,
) -> tuple[dict[str, Any], tuple[float, float]]:
    cache_key = id(row)
    if inference_cache is not None and cache_key in inference_cache:
        return inference_cache[cache_key]
    image_bgr, scale = _row_native_image_bgr(row)
    if infer_provider is not None:
        result = dict(infer_provider(image_bgr))
    else:
        optional: dict[str, str] = {}
        if heatmap_decoder is not None:
            optional["heatmap_decoder"] = heatmap_decoder
        if coordinate_transform is not None:
            optional["coordinate_transform"] = coordinate_transform
        result = infer_court_model(
            image_bgr,
            checkpoint_path,
            device=device,
            **optional,
        )
    value = (result, scale)
    if inference_cache is not None:
        inference_cache[cache_key] = value
    return value


def evaluate_structured_checkpoint_against_real_labels(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    *,
    device: str = "cpu",
    heatmap_decoder: str | None = None,
    coordinate_transform: str | None = None,
    infer_provider: Any | None = None,
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Score the review-only regulation-template ``best_court`` output on exact floor names."""

    records: list[dict[str, Any]] = []
    for row in rows:
        result, (scale_x, scale_y) = _infer_row(
            row,
            checkpoint_path,
            device=device,
            heatmap_decoder=heatmap_decoder,
            coordinate_transform=coordinate_transform,
            infer_provider=infer_provider,
            inference_cache=inference_cache,
        )
        best = result["best_court"]
        prediction = {
            name: [float(xy[0]) * scale_x, float(xy[1]) * scale_y]
            for name, xy in best["keypoints_xy"].items()
        }
        truth = {
            name: [float(xy[0]), float(xy[1])]
            for name, xy in row["keypoints"].items()
            if xy is not None and not name.startswith("net_")
        }
        ignored: list[dict[str, Any]] = []
        for item in best.get("ignored_observations") or []:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            semantic = normalized.get("name", normalized.get("semantic"))
            if not isinstance(semantic, str) or semantic not in truth:
                continue
            normalized["name"] = semantic
            xy = normalized.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                normalized["xy"] = [float(xy[0]) * scale_x, float(xy[1]) * scale_y]
            ignored.append(normalized)
        records.append(
            {
                "sample_id": f"{row.get('clip', 'unknown')}/frame_{int(row.get('frame_index') or 0):06d}",
                "viewpoint": str(row.get("clip") or "unknown"),
                "ground_truth": truth,
                "prediction": {
                    "keypoints": prediction,
                    "confidences": dict(best.get("point_confidence") or {}),
                    "ignored_observations": ignored,
                    "whole_court_confidence": float(best.get("court_confidence") or 0.0),
                },
            }
        )
    report = evaluate_structured_court_outputs(records)
    report.update(
        {
            "schema_version": 1,
            "artifact_type": "court_structured_best_effort_eval",
            "checkpoint": str(checkpoint_path),
            "status": "diagnostic_only",
            "promotion_allowed": False,
            "measurement_valid": False,
            "authority_state": "review_only",
            "evaluated_taxonomy": "12_canonical_floor_points_exact_name",
        }
    )
    return report


def evaluate_raw_vs_structured_checkpoint_against_real_labels(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    *,
    device: str = "cpu",
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 13,
    heatmap_decoder: str | None = None,
    coordinate_transform: str | None = None,
    infer_provider: Any | None = None,
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Run inference once per row and compare raw versus solved floor points by exact name."""

    records: list[dict[str, Any]] = []
    for row in rows:
        result, (scale_x, scale_y) = _infer_row(
            row,
            checkpoint_path,
            device=device,
            heatmap_decoder=heatmap_decoder,
            coordinate_transform=coordinate_transform,
            infer_provider=infer_provider,
            inference_cache=inference_cache,
        )
        best = result["best_court"]
        truth = {
            name: [float(xy[0]), float(xy[1])]
            for name, xy in row["keypoints"].items()
            if xy is not None and not name.startswith("net_")
        }
        raw_prediction = {
            name: [float(xy[0]) * scale_x, float(xy[1]) * scale_y]
            for name, xy in result["keypoints_xy"].items()
            if not name.startswith("net_")
        }
        structured_prediction = {
            name: [float(xy[0]) * scale_x, float(xy[1]) * scale_y]
            for name, xy in best["keypoints_xy"].items()
            if not name.startswith("net_")
        }
        clip = str(row.get("clip") or "unknown")
        visible_count = len(truth)
        visibility = str(
            row.get("visibility")
            or ("full_floor_12" if visible_count == 12 else "partial_floor")
        )
        viewpoint = str(row.get("viewpoint") or clip)
        subgroup = str(
            row.get("subgroup")
            or (
                "independent_human_reviewed"
                if row.get("label_status", INDEPENDENT_REVIEWED_STATUS)
                == INDEPENDENT_REVIEWED_STATUS
                else "non_independent_or_external"
            )
        )
        ignored: list[dict[str, Any]] = []
        for item in best.get("ignored_observations") or []:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            semantic = normalized.get("name", normalized.get("semantic"))
            if not isinstance(semantic, str) or semantic not in truth:
                continue
            normalized["name"] = semantic
            xy = normalized.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                normalized["xy"] = [float(xy[0]) * scale_x, float(xy[1]) * scale_y]
            ignored.append(normalized)
        records.append(
            {
                "sample_id": f"{clip}/frame_{int(row.get('frame_index') or 0):06d}",
                "viewpoint": viewpoint,
                "strata": {
                    "subgroup": subgroup,
                    "source": str(row.get("source") or clip),
                    "source_group": str(row.get("source_group") or clip),
                    "visibility": visibility,
                },
                "ground_truth": truth,
                "raw_prediction": {
                    "keypoints": raw_prediction,
                    "confidences": {
                        str(name): float(value)
                        for name, value in dict(result.get("keypoints_conf") or {}).items()
                        if name in truth
                    },
                },
                "structured_prediction": {
                    "keypoints": structured_prediction,
                    "confidences": {
                        str(name): float(value)
                        for name, value in dict(best.get("point_confidence") or {}).items()
                        if name in truth
                    },
                    "ignored_observations": ignored,
                    "whole_court_confidence": float(best.get("court_confidence") or 0.0),
                },
            }
        )
    report = evaluate_raw_vs_structured_court_outputs(
        records,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "status": "diagnostic_only",
            "promotion_allowed": False,
            "measurement_valid": False,
            "authority_state": "review_only",
        }
    )
    return report


def _row_errors(row: dict[str, Any], predicted: dict[str, list[float]]) -> list[float]:
    errors: list[float] = []
    for name, xy in row["keypoints"].items():
        if name not in predicted:
            continue
        px, py = predicted[name]
        errors.append(math.hypot(px - xy[0], py - xy[1]))
    return errors


def _error_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "median": _percentile(ordered, 50.0),
        "p90": _percentile(ordered, 90.0),
        "p95": _percentile(ordered, 95.0),
        "max": float(ordered[-1]),
    }


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    rank = (len(ordered_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered_values[low])
    weight = rank - low
    return float(ordered_values[low] * (1.0 - weight) + ordered_values[high] * weight)


def _pck(values: list[float], threshold_px: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value <= threshold_px) / len(values)


def evaluate_checkpoint_against_real_labels(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    *,
    device: str = "cpu",
    pck_threshold_px: float = 5.0,
    secondary_pck_threshold_px: float = 10.0,
    heatmap_decoder: str | None = None,
    coordinate_transform: str | None = None,
    infer_provider: Any | None = None,
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Score one `court_unet_v2` checkpoint against `rows` (read-only). Returns raw per-frame
    errors split into independent-human-reviewed rows (PRIMARY) and all rows including
    owner-approved static-camera copies (SECONDARY), each with a per-clip breakdown so the actual
    gate ("PCK@5px >= 0.95 per viewpoint") can be checked per clip, not just pooled."""

    all_errors: list[float] = []
    independent_errors: list[float] = []
    errors_by_clip: dict[str, list[float]] = {}
    independent_errors_by_clip: dict[str, list[float]] = {}

    for row in rows:
        predicted = predict_row_keypoints_source_px(
            row,
            checkpoint_path,
            device=device,
            heatmap_decoder=heatmap_decoder,
            coordinate_transform=coordinate_transform,
            infer_provider=infer_provider,
            inference_cache=inference_cache,
        )
        row_errors = _row_errors(row, predicted)
        clip = str(row.get("clip") or "unknown")
        all_errors.extend(row_errors)
        errors_by_clip.setdefault(clip, []).extend(row_errors)
        if row.get("label_status", INDEPENDENT_REVIEWED_STATUS) == INDEPENDENT_REVIEWED_STATUS:
            independent_errors.extend(row_errors)
            independent_errors_by_clip.setdefault(clip, []).extend(row_errors)

    def _mode_summary(errors: list[float], by_clip: dict[str, list[float]]) -> dict[str, Any]:
        return {
            "keypoint_error_summary": _error_summary(errors),
            "pck_at_2px": _pck(errors, 2.0),
            "pck_at_5px": _pck(errors, pck_threshold_px),
            "pck_at_10px": _pck(errors, secondary_pck_threshold_px),
            "per_clip": {
                clip: {
                    **_error_summary(clip_errors),
                    "pck_at_2px": _pck(clip_errors, 2.0),
                    "pck_at_5px": _pck(clip_errors, pck_threshold_px),
                    "pck_at_10px": _pck(clip_errors, secondary_pck_threshold_px),
                }
                for clip, clip_errors in sorted(by_clip.items())
            },
        }

    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_owner_gate_report_v2",
        "checkpoint": str(checkpoint_path),
        "pck_threshold_px": pck_threshold_px,
        "independent_frame_count": sum(1 for row in rows if row.get("label_status", INDEPENDENT_REVIEWED_STATUS) == INDEPENDENT_REVIEWED_STATUS),
        "all_frame_count": len(rows),
        "independent": _mode_summary(independent_errors, independent_errors_by_clip),
        "all": _mode_summary(all_errors, errors_by_clip),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--real-root",
        type=Path,
        default=ROOT / "eval_clips" / "ball",
        help="Root containing <clip>/labels/court_keypoints.json rows to score against.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pck-threshold-px", type=float, default=5.0)
    parser.add_argument("--gate-threshold", type=float, default=0.95)
    parser.add_argument(
        "--include-structured-best-court",
        action="store_true",
        help=(
            "Also score the confidence-aware floor-only regulation-template best_court output. "
            "The result remains diagnostic/review-only and cannot satisfy CAL promotion."
        ),
    )
    parser.add_argument(
        "--include-raw-vs-structured",
        action="store_true",
        help=(
            "Add paired exact-semantic floor metrics for raw heatmap points versus best_court, "
            "including PCK@2/5/10 and sample-cluster bootstrap deltas."
        ),
    )
    parser.add_argument("--paired-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=13)
    parser.add_argument("--heatmap-decoder", choices=("parabolic", "dark"))
    parser.add_argument("--coordinate-transform", choices=("legacy_stride", "udp"))
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        help="Optional frozen court v3.1 five-fold manifest; loads only the selected partition.",
    )
    parser.add_argument("--protocol-fold-index", type=int, default=0)
    parser.add_argument(
        "--protocol-partition",
        choices=("train", "validation", "test"),
        default="test",
    )
    args = parser.parse_args(argv)

    checkpoint_payload = load_court_model_checkpoint(args.checkpoint, device="cpu")
    architecture_name = str(
        checkpoint_payload.get("model_architecture")
        or checkpoint_payload.get("architecture")
        or "court_unet_v2"
    )
    decoder_name = str(
        args.heatmap_decoder or checkpoint_payload.get("heatmap_decoder") or "parabolic"
    )
    coordinate_transform = str(
        args.coordinate_transform
        or checkpoint_payload.get("coordinate_transform")
        or "legacy_stride"
    )

    rows = (
        load_protocol_partition_rows(
            args.protocol_manifest,
            fold_index=args.protocol_fold_index,
            partition=args.protocol_partition,
        )
        if args.protocol_manifest is not None
        else load_real_court_keypoint_labels(args.real_root)
    )
    infer_provider = make_court_model_infer_provider(
        checkpoint_path=args.checkpoint,
        device=args.device,
        heatmap_decoder=args.heatmap_decoder,
        coordinate_transform=args.coordinate_transform,
    )
    if infer_provider is None:
        raise ValueError(f"could not load court inference provider: {args.checkpoint}")
    inference_cache: dict[int, tuple[dict[str, Any], tuple[float, float]]] = {}
    scored = evaluate_checkpoint_against_real_labels(
        args.checkpoint,
        rows,
        device=args.device,
        pck_threshold_px=args.pck_threshold_px,
        heatmap_decoder=args.heatmap_decoder,
        coordinate_transform=args.coordinate_transform,
        infer_provider=infer_provider,
        inference_cache=inference_cache,
    )

    def _gate_passed_per_viewpoint(mode: dict[str, Any]) -> bool:
        per_clip = mode["per_clip"]
        return bool(
            per_clip
            and all(
                clip_summary["pck_at_5px"] is not None and clip_summary["pck_at_5px"] >= args.gate_threshold
                for clip_summary in per_clip.values()
            )
        )

    independent_gate_value = scored["independent"]["pck_at_5px"]
    gate_passed_pooled = {
        "independent": bool(independent_gate_value is not None and independent_gate_value >= args.gate_threshold),
        "all": bool(scored["all"]["pck_at_5px"] is not None and scored["all"]["pck_at_5px"] >= args.gate_threshold),
    }
    gate_passed_per_viewpoint = {
        "independent": _gate_passed_per_viewpoint(scored["independent"]),
        "all": _gate_passed_per_viewpoint(scored["all"]),
    }

    # Shape the report so the existing CAL evidence scanner
    # (threed.racketsport.overlapping_court_calibration._neural_keypoint_checkpoint_evidence,
    # which globs runs/**/court_keypoint_metrics.json for `checkpoint`/`gate`/`after` keys) can
    # rank this checkpoint alongside encoder_decoder_v1/local_conv_v1 evidence: the real,
    # per-viewpoint-gated PRIMARY number is the independent-reviewed-row median.
    report = {
        **scored,
        "gate_threshold": args.gate_threshold,
        "gate": {
            "metric": "real_keypoint_pck_at_5px_independent",
            "value": independent_gate_value,
            "threshold": args.gate_threshold,
            "pck_threshold_px": args.pck_threshold_px,
            "passed": gate_passed_per_viewpoint["independent"],
            "not_cal3_verified": True,
        },
        "after": {
            "real_keypoint_median_px": scored["independent"]["keypoint_error_summary"]["median"],
            "real_keypoint_mean_px": scored["independent"]["keypoint_error_summary"]["mean"],
            "real_keypoint_p90_px": scored["independent"]["keypoint_error_summary"]["p90"],
            "real_keypoint_p95_px": scored["independent"]["keypoint_error_summary"]["p95"],
            "real_keypoint_max_px": scored["independent"]["keypoint_error_summary"]["max"],
            "real_keypoint_pck_at_2px": scored["independent"]["pck_at_2px"],
            "real_keypoint_pck_at_5px": scored["independent"]["pck_at_5px"],
            "real_keypoint_pck_at_10px": scored["independent"]["pck_at_10px"],
            "real_keypoint_count": scored["independent"]["keypoint_error_summary"]["count"],
            "real_keypoint_pck_per_clip": scored["independent"]["per_clip"],
        },
        "architecture": {"name": architecture_name, "network_architecture": architecture_name},
        "postprocess": {
            "prediction_mode": "raw_heatmap_and_structured_best_court",
            "decode": decoder_name,
            "coordinate_transform": coordinate_transform,
        },
        "real_holdout_count": scored["independent_frame_count"],
        "holdout_artifacts": [],
        "gate_passed_pooled": gate_passed_pooled,
        "gate_passed_per_viewpoint": gate_passed_per_viewpoint,
        "gate_passed": gate_passed_per_viewpoint,
        "notes": [
            "independent: PRIMARY, the 4 independently human-reviewed frames (1 per clip).",
            "all: SECONDARY, all 32 rows including owner-approved reviewed_static_camera_copy rows.",
            "gate_passed_per_viewpoint requires every clip present to individually clear "
            "pck_threshold_px, not just the pooled-across-clips average (gate_passed aliases it).",
        ],
    }
    if args.protocol_manifest is not None:
        report["evaluation_protocol"] = {
            "manifest": str(args.protocol_manifest),
            "manifest_sha256": hashlib.sha256(args.protocol_manifest.read_bytes()).hexdigest(),
            "fold_index": args.protocol_fold_index,
            "partition": args.protocol_partition,
            "task88_role": "historical_development_only_not_protocol_evidence",
        }
    if args.include_structured_best_court:
        report["structured_best_court"] = evaluate_structured_checkpoint_against_real_labels(
            args.checkpoint,
            rows,
            device=args.device,
            heatmap_decoder=args.heatmap_decoder,
            coordinate_transform=args.coordinate_transform,
            infer_provider=infer_provider,
            inference_cache=inference_cache,
        )
    if args.include_raw_vs_structured:
        report["raw_vs_structured"] = evaluate_raw_vs_structured_checkpoint_against_real_labels(
            args.checkpoint,
            rows,
            device=args.device,
            bootstrap_resamples=args.paired_bootstrap_resamples,
            bootstrap_seed=args.paired_bootstrap_seed,
            heatmap_decoder=args.heatmap_decoder,
            coordinate_transform=args.coordinate_transform,
            infer_provider=infer_provider,
            inference_cache=inference_cache,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
