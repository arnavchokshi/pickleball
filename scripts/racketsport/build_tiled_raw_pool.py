#!/usr/bin/env python3
"""TRK-R2: build a raw, pre-role-lock person-detection pool from SAHI-style
overlapping native-resolution tiled YOLO inference, then assign persistent
short-term track identity with a standard (unedited, ultralytics-native)
BoT-SORT pass, so the existing `raw_pool_person_authority` association +
gate-v2 scoring chain can run on a tiled-detection pool unchanged.

Rationale (see runs/manager/codex_lanes/reports/research_trk_20260702.md
rank #1 and runs/phase2/trk_assoc_prereg_20260701T233015Z/REPORT.md): the
champion `botsort_reid_loose` pool runs YOLO26m once per full 1920x1080
frame at imgsz=1536. Far/small players end up a few dozen pixels tall,
which the research report identifies as the likely driver of the 0.42-0.45
median near-miss IoU and the sub-0.95 four-player-coverage detection
ceilings on all three clips. This script instead slices each frame into a
full-frame view plus an overlapping NxM native-resolution grid, runs the
*same* base yolo26m.pt (no training) on every crop, merges duplicate boxes
at tile seams with NMS (`tiled_person_detector.merge_tiled_detections`,
reused unedited), and only then assigns identity.

Known failure mode under test: duplicate boxes surviving at tile seams.
`--audit-path` writes a seam-duplication audit (pre/post merge box counts,
residual near-duplicate pairs) so this is measured, not assumed away.

Identity: the champion pool's persistent short-term id came from Ultralytics
BoT-SORT running *inside* `model.track()`, which only accepts one detector
call per frame -- it cannot consume externally tiled+merged boxes. This
script instead drives `ultralytics.trackers.bot_sort.BOTSORT` directly
(unedited third-party class) frame-by-frame over the merged tiled boxes, so
identity is still real motion/IoU BoT-SORT association, not a conf-rank
per-frame slot index. (The earlier `outdoor_a100_tiled_pool` diagnostic in
runs/phase2/trk_offline_authority_rawpool_20260701T222255Z/REPORT.md scored
badly -- source-only IDF1 0.5823, cov4 0.2132 -- specifically because its
per-frame track_id was a conf-rank slot index, not a persistent id; see that
REPORT.md's "champion-eval track_id" bug note. This script exists to avoid
repeating that mistake.) The tracker config used is
`configs/racketsport/botsort_no_reid_loose.yaml` (IoU/motion only, no
embedded ReID) rather than `botsort_reid_loose.yaml`: the downstream
`raw_pool_person_authority` chain already performs its own dedicated OSNet
appearance pass over these exact same boxes for the real cross-clip
identity resolution, so BoT-SORT's own embedded-ReID feature only affects
short-term fragment-boundary quality, not final identity. This is a
documented deviation from the non-tiled baseline pool and is called out in
the run report.

Camera-motion compensation (GMC/sparseOptFlow) is skipped (img=None passed
to the tracker) because all three eval clips are static tripod/webcam
mounts, not handheld/broadcast pans.

Output schema matches exactly what
`threed/racketsport/raw_pool_person_authority.run_raw_pool_authority_candidate`
already reads via `--raw-pool-dir` (unedited): `raw_tracked_detections.json`
(source-pixel), `tracked_detections.json` (calibration-scaled), and
`metrics.json` (source/calibration size counts for scale inference).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import calibration_image_size  # noqa: E402
from threed.racketsport.detection_scaling import scale_detection_payload_bboxes  # noqa: E402
from threed.racketsport.schemas import CourtCalibration, validate_artifact_file  # noqa: E402
from threed.racketsport.tiled_person_detector import (  # noqa: E402
    NormalizedCrop,
    crop_region_pixels,
    merge_tiled_detections,
    yolo_tiled_detections_for_frames_batched,
)

FULL_FRAME_REGION: NormalizedCrop = (0.0, 0.0, 1.0, 1.0)


def native_grid_regions(cols: int, rows: int, overlap: float) -> tuple[NormalizedCrop, ...]:
    """SAHI-style overlapping grid of normalized crop regions.

    ``overlap`` is the fraction of each tile's width/height that overlaps
    its neighbor (e.g. 0.22 == 22%). Tiles are sized so the grid exactly
    covers [0, 1] on each axis with uniform overlap and no gaps:
    ``tile = 1 / (1 + (n - 1) * (1 - overlap))``, ``stride = tile * (1 - overlap)``.
    """

    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    def axis_bounds(n: int) -> list[tuple[float, float]]:
        if n == 1:
            return [(0.0, 1.0)]
        tile = 1.0 / (1.0 + (n - 1) * (1.0 - overlap))
        stride = tile * (1.0 - overlap)
        return [(round(i * stride, 6), round(i * stride + tile, 6)) for i in range(n)]

    x_bounds = axis_bounds(cols)
    y_bounds = axis_bounds(rows)
    regions: list[NormalizedCrop] = []
    for y0, y1 in y_bounds:
        for x0, x1 in x_bounds:
            regions.append((x0, y0, min(1.0, x1), min(1.0, y1)))
    return tuple(regions)


def build_crop_regions(*, cols: int, rows: int, overlap: float, include_full_frame: bool) -> tuple[NormalizedCrop, ...]:
    grid = native_grid_regions(cols, rows, overlap)
    return (FULL_FRAME_REGION, *grid) if include_full_frame else grid


def _stride32_ceil(value: float) -> int:
    return int(-(-int(round(value)) // 32) * 32)


def _matched_imgsz(region_px_size: tuple[int, int]) -> list[int]:
    """Rectangular imgsz [h, w] rounded up to stride 32 that matches a crop's
    own native pixel size almost exactly (padding-only letterbox, no rescale).

    IMPORTANT perf finding (measured on this machine, Apple M1 Pro, torch
    2.5.1 MPS backend, ultralytics 8.4.22): letterboxing a tiled crop up to a
    generic square `imgsz` (e.g. 1280) that requires real upscale/downscale
    interpolation is catastrophically slow on MPS for *non-full-frame*
    aspect ratios -- 5+ s/crop measured, ~35-90x slower than a same-size
    steady-state call. A rectangular imgsz matched to the crop's own size
    (padding only, no interpolation) drops this to ~0.05-0.2s/crop after a
    one-time per-shape warmup (~10s), because every distinct (aspect,imgsz)
    pair pays a one-time Metal kernel compile the first time it is used, not
    a per-crop cost. This is why every crop *position* in the grid must
    share one exact pixel size (true by construction here) and be run as
    its own uniform-shape batch matched to that size, rather than an
    arbitrary fixed imgsz shared with the full-frame pass.
    """

    width, height = region_px_size
    return [_stride32_ceil(height), _stride32_ceil(width)]


def _iter_video_frames(video_path: Path, *, max_frames: int | None):
    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            index += 1
            if max_frames is not None and index >= max_frames:
                break
    finally:
        cap.release()


def _video_fps_size(video_path: Path) -> tuple[float, int, int]:
    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"could not determine fps/size for {video_path}")
    return fps, width, height


def audit_seam_duplication(
    *,
    pre_merge_by_frame: dict[int, list[dict[str, Any]]],
    post_merge_by_frame: dict[int, list[dict[str, Any]]],
    residual_iou_report_threshold: float = 0.3,
) -> dict[str, Any]:
    """Measure the known tiled-inference failure mode: duplicate boxes at tile seams.

    Reports pre/post merge box counts and any residual near-duplicate pairs
    (IoU > ``residual_iou_report_threshold`` between two *different* boxes
    that both survived NMS) -- i.e. duplicates the merge step failed to
    suppress. This does not assume the merge worked; it checks.
    """

    from threed.racketsport.tiled_person_detector import _bbox, _bbox_iou  # type: ignore[attr-defined]

    total_pre = sum(len(v) for v in pre_merge_by_frame.values())
    total_post = sum(len(v) for v in post_merge_by_frame.values())
    residual_pairs = 0
    residual_ious: list[float] = []
    frames_with_residual = 0
    for frame_idx, detections in post_merge_by_frame.items():
        boxes = [_bbox(d) for d in detections]
        frame_hit = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = _bbox_iou(boxes[i], boxes[j])
                if iou > residual_iou_report_threshold:
                    residual_pairs += 1
                    residual_ious.append(iou)
                    frame_hit = True
        if frame_hit:
            frames_with_residual += 1
    return {
        "artifact_type": "racketsport_tiled_seam_duplication_audit",
        "total_boxes_pre_merge": total_pre,
        "total_boxes_post_merge": total_post,
        "boxes_suppressed_by_merge": total_pre - total_post,
        "suppression_rate": (total_pre - total_post) / total_pre if total_pre else 0.0,
        "residual_iou_report_threshold": residual_iou_report_threshold,
        "residual_near_duplicate_pairs": residual_pairs,
        "residual_near_duplicate_frame_count": frames_with_residual,
        "residual_near_duplicate_iou_median": (sorted(residual_ious)[len(residual_ious) // 2] if residual_ious else None),
        "residual_near_duplicate_iou_max": max(residual_ious) if residual_ious else None,
        "frame_count": len(post_merge_by_frame),
    }


def run_botsort_pass(
    merged_frames: list[dict[str, Any]],
    *,
    fps: float,
    tracker_config_path: Path,
    frame_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Assign persistent short-term track identity with real ultralytics BoT-SORT.

    Operates directly on already-tiled-and-merged per-frame detections
    (bbox/conf/class only -- the meaningless per-frame conf-rank slot index
    written by `merge_tiled_detections` is discarded here, not reused as
    identity). Uses the standard `ultralytics.trackers.bot_sort.BOTSORT`
    class unedited; only the frame-by-frame feed loop is new code.
    """

    import numpy as np
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace

    cfg = IterableSimpleNamespace(**YAML.load(str(tracker_config_path)))
    if cfg.tracker_type != "botsort":
        raise ValueError(f"expected a botsort tracker config, got {cfg.tracker_type}")
    tracker = BOTSORT(args=cfg, frame_rate=max(1, round(fps)))

    height, width = frame_size
    output_frames: list[dict[str, Any]] = []
    for entry in merged_frames:
        frame_index = int(entry["frame"])
        detections = entry["detections"]
        if detections:
            rows = np.array(
                [[*d["bbox"], float(d.get("conf", 0.0)), 0.0] for d in detections],
                dtype=np.float64,
            )
        else:
            rows = np.zeros((0, 6), dtype=np.float64)
        boxes = Boxes(rows, (height, width))
        tracks = tracker.update(boxes, img=None, feats=None)
        out_detections: list[dict[str, Any]] = []
        for row in tracks:
            x1, y1, x2, y2, track_id, score, cls, _idx = row.tolist()
            out_detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(score),
                    "class": "person",
                    "track_id": int(track_id),
                }
            )
        output_frames.append({"frame": frame_index, "detections": out_detections})
    return output_frames


def build_tiled_raw_pool(
    *,
    clip_id: str,
    video_path: Path,
    calibration_path: Path,
    out_dir: Path,
    model_path: Path,
    tracker_config_path: Path,
    cols: int,
    rows: int,
    overlap: float,
    include_full_frame: bool,
    conf: float,
    tile_nms_iou: float,
    merge_nms_iou: float,
    device: str | None,
    batch_size: int,
    max_frames: int | None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = validate_artifact_file("court_calibration", calibration_path)
    if not isinstance(calibration, CourtCalibration):
        raise ValueError(f"{calibration_path} did not parse as CourtCalibration")

    fps, source_width, source_height = _video_fps_size(video_path)
    calibration_width, calibration_height = calibration_image_size(
        calibration, fallback_target=(source_width, source_height)
    )
    scale_x = calibration_width / source_width
    scale_y = calibration_height / source_height

    grid_regions = native_grid_regions(cols=cols, rows=rows, overlap=overlap)
    crop_regions = (FULL_FRAME_REGION, *grid_regions) if include_full_frame else grid_regions

    # Two shape-homogeneous passes, each with a rectangular imgsz matched to
    # that pass's own native crop size (see `_matched_imgsz` docstring for
    # why this is required for tractable MPS throughput). Every grid tile
    # shares one pixel size by construction, so the tile pass is one uniform
    # batch call; the full-frame region is the other.
    full_px = crop_region_pixels(source_width, source_height, FULL_FRAME_REGION)
    full_imgsz = _matched_imgsz((full_px[2] - full_px[0], full_px[3] - full_px[1]))
    tile_px = crop_region_pixels(source_width, source_height, grid_regions[0])
    tile_imgsz = _matched_imgsz((tile_px[2] - tile_px[0], tile_px[3] - tile_px[1]))

    model = YOLO(str(model_path))
    started = time.perf_counter()
    pass_payloads: list[dict[str, Any]] = []
    if include_full_frame:
        full_payload = yolo_tiled_detections_for_frames_batched(
            model=model,
            frames=_iter_video_frames(video_path, max_frames=max_frames),
            fps=fps,
            crop_regions=(FULL_FRAME_REGION,),
            conf=conf,
            iou=tile_nms_iou,
            imgsz=full_imgsz,
            device=device,
            nms_iou=merge_nms_iou,
            batch_size=batch_size,
            half=None,
        )
        pass_payloads.append(full_payload)
    tile_payload = yolo_tiled_detections_for_frames_batched(
        model=model,
        frames=_iter_video_frames(video_path, max_frames=max_frames),
        fps=fps,
        crop_regions=grid_regions,
        conf=conf,
        iou=tile_nms_iou,
        imgsz=tile_imgsz,
        device=device,
        nms_iou=merge_nms_iou,
        batch_size=batch_size,
        half=None,
    )
    pass_payloads.append(tile_payload)
    detect_wall_s = time.perf_counter() - started

    # Pre-merge baseline for the seam-duplication audit: each pass already
    # ran its own (trivial for the single-region full-frame pass; real for
    # the 6-tile grid pass) internal NMS, but boxes have *not yet* been
    # cross-checked between the full-frame view and the grid tiles, or
    # across tile seams beyond what merge_tiled_detections already did
    # per-pass. Concatenate both passes' own outputs as "pre" and run one
    # more explicit merge as "post" to measure exactly the seam-duplication
    # failure mode the mission calls out.
    pre_merge_by_frame: dict[int, list[dict[str, Any]]] = {}
    for payload in pass_payloads:
        for frame_entry in payload["frames"]:
            frame_idx = int(frame_entry["frame"])
            pre_merge_by_frame.setdefault(frame_idx, []).extend(frame_entry["detections"])
    post_merge_by_frame = {
        frame_idx: merge_tiled_detections(detections, iou_threshold=merge_nms_iou)
        for frame_idx, detections in pre_merge_by_frame.items()
    }
    seam_audit = audit_seam_duplication(pre_merge_by_frame=pre_merge_by_frame, post_merge_by_frame=post_merge_by_frame)
    seam_audit["merge_nms_iou"] = merge_nms_iou

    merged_frames = [
        {"frame": frame_idx, "detections": post_merge_by_frame[frame_idx]}
        for frame_idx in sorted(post_merge_by_frame)
    ]
    tracked_frames = run_botsort_pass(
        merged_frames,
        fps=fps,
        tracker_config_path=tracker_config_path,
        frame_size=(source_height, source_width),
    )
    raw_detections_payload = {"fps": fps, "frames": tracked_frames}
    detections_payload = scale_detection_payload_bboxes(raw_detections_payload, scale_x=scale_x, scale_y=scale_y)

    raw_path = out_dir / "raw_tracked_detections.json"
    scaled_path = out_dir / "tracked_detections.json"
    raw_path.write_text(json.dumps(raw_detections_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scaled_path.write_text(json.dumps(detections_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tracker_boxes = sum(len(f["detections"]) for f in tracked_frames)
    distinct_track_ids = {d["track_id"] for f in tracked_frames for d in f["detections"]}
    metrics = {
        "schema_version": 1,
        "artifact_type": "racketsport_tiled_raw_pool_candidate",
        "status": "ok",
        "clip": clip_id,
        "variant": (
            f"tiled_{cols}x{rows}_ov{int(round(overlap * 100))}_yolo26m_botsort_noreid_loose_"
            f"fullimg{full_imgsz[1]}x{full_imgsz[0]}_tileimg{tile_imgsz[1]}x{tile_imgsz[0]}_conf{str(conf).replace('.', '')}"
        ),
        "model": str(model_path),
        "tracker_config": str(tracker_config_path),
        "crop_regions": [list(region) for region in crop_regions],
        "crop_region_count": len(crop_regions),
        "grid": {"cols": cols, "rows": rows, "overlap": overlap, "include_full_frame": include_full_frame},
        "conf": conf,
        "tile_nms_iou": tile_nms_iou,
        "merge_nms_iou": merge_nms_iou,
        "full_imgsz_hw": full_imgsz,
        "tile_imgsz_hw": tile_imgsz,
        "device": device,
        "batch_size": batch_size,
        "half": None,
        "max_frames": max_frames,
        "source_video": str(video_path),
        "wall_time_s": round(detect_wall_s, 6),
        "detect_wall_time_s": round(detect_wall_s, 6),
        "counts": {
            "tracker_frames": len(tracked_frames),
            "tracker_boxes": tracker_boxes,
            "tracked_person_boxes": tracker_boxes,
            "untracked_person_boxes": 0,
            "tracker_non_person": 0,
            "distinct_source_track_ids": len(distinct_track_ids),
            "source_width": source_width,
            "source_height": source_height,
            "calibration_width": calibration_width,
            "calibration_height": calibration_height,
            "bbox_scale_x": round(scale_x, 6),
            "bbox_scale_y": round(scale_y, 6),
            "crop_region_count": len(crop_regions),
        },
        "seam_duplication_audit": seam_audit,
        "raw_detections_path": str(raw_path),
        "scaled_detections_path": str(scaled_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/checkpoints/yolo26m.pt"))
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("configs/racketsport/botsort_no_reid_loose.yaml"),
    )
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.22)
    parser.add_argument("--no-full-frame", action="store_true")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tile-nms-iou", type=float, default=0.6)
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    try:
        metrics = build_tiled_raw_pool(
            clip_id=args.clip_id,
            video_path=args.video,
            calibration_path=args.calibration,
            out_dir=args.out_dir,
            model_path=args.model,
            tracker_config_path=args.tracker_config,
            cols=args.cols,
            rows=args.rows,
            overlap=args.overlap,
            include_full_frame=not args.no_full_frame,
            conf=args.conf,
            tile_nms_iou=args.tile_nms_iou,
            merge_nms_iou=args.merge_nms_iou,
            device=args.device,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(metrics["counts"], indent=2, sort_keys=True))
    print(json.dumps(metrics["seam_duplication_audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
