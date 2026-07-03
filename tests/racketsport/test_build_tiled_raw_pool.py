from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "racketsport"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from threed.racketsport.tiled_person_detector import crop_region_pixels  # noqa: E402

from build_tiled_raw_pool import (  # noqa: E402
    _matched_imgsz,
    _stride32_ceil,
    audit_seam_duplication,
    build_crop_regions,
    native_grid_regions,
)


def test_native_grid_regions_covers_full_frame_with_uniform_overlap() -> None:
    regions = native_grid_regions(3, 2, 0.22)
    assert len(regions) == 6

    # Every tile has the same normalized size up to float rounding noise (<1e-5).
    widths = [x1 - x0 for x0, _y0, x1, _y1 in regions]
    heights = [y1 - y0 for _x0, y0, _x1, y1 in regions]
    assert max(widths) - min(widths) < 1e-5
    assert max(heights) - min(heights) < 1e-5

    # At real pixel resolution (what the detector actually crops), every grid
    # tile must be pixel-identical in size -- this is required for the
    # shape-homogeneous MPS batching that makes tiled inference tractable
    # (see build_tiled_raw_pool's MPS performance note): mixed shapes in one
    # batch were measured at 30-90x slower per crop on this backend.
    pixel_sizes = set()
    for region in regions:
        x0, y0, x1, y1 = crop_region_pixels(1920, 1080, region)
        pixel_sizes.add((x1 - x0, y1 - y0))
    assert len(pixel_sizes) == 1

    # Grid spans exactly [0, 1] on both axes with no gaps: first tile starts
    # at 0, last tile ends at 1.
    xs0 = sorted({round(x0, 5) for x0, _y0, _x1, _y1 in regions})
    xs1 = sorted({round(x1, 5) for _x0, _y0, x1, _y1 in regions})
    assert xs0[0] == 0.0
    assert xs1[-1] == 1.0

    # 22% overlap: adjacent tiles along the x axis overlap by ~22% of tile width.
    tile_w = sum(widths) / len(widths)
    stride = xs0[1] - xs0[0]
    overlap_frac = 1.0 - stride / tile_w
    assert abs(overlap_frac - 0.22) < 1e-4


def test_native_grid_regions_rejects_bad_params() -> None:
    import pytest

    with pytest.raises(ValueError):
        native_grid_regions(0, 2, 0.2)
    with pytest.raises(ValueError):
        native_grid_regions(2, 2, 1.0)


def test_build_crop_regions_prepends_full_frame_when_requested() -> None:
    with_full = build_crop_regions(cols=3, rows=2, overlap=0.22, include_full_frame=True)
    without_full = build_crop_regions(cols=3, rows=2, overlap=0.22, include_full_frame=False)
    assert len(with_full) == 7
    assert len(without_full) == 6
    assert with_full[0] == (0.0, 0.0, 1.0, 1.0)


def test_stride32_ceil_rounds_up_to_multiple_of_32() -> None:
    assert _stride32_ceil(607) == 608
    assert _stride32_ceil(750) == 768
    assert _stride32_ceil(1920) == 1920
    assert _stride32_ceil(1) == 32


def test_matched_imgsz_returns_height_width_order() -> None:
    # region_px_size is (width, height); matched imgsz is [h, w] (ultralytics convention).
    assert _matched_imgsz((750, 607)) == [608, 768]
    assert _matched_imgsz((1920, 1080)) == [1088, 1920]


def test_audit_seam_duplication_detects_residual_duplicate_at_tile_seam() -> None:
    # Same physical box split by two overlapping tiles produces two near-duplicate
    # detections pre-merge; a merge threshold above their IoU leaves both, which
    # the audit must flag as a residual near-duplicate pair (the known tiled
    # seam-duplication failure mode).
    pre_merge = {
        0: [
            {"bbox": [100.0, 100.0, 140.0, 220.0], "conf": 0.9, "class": "person"},
            {"bbox": [102.0, 101.0, 141.0, 219.0], "conf": 0.6, "class": "person"},
        ]
    }
    # Simulate a merge pass that (correctly) suppresses the low-conf duplicate.
    post_merge_suppressed = {0: [pre_merge[0][0]]}
    audit_ok = audit_seam_duplication(pre_merge_by_frame=pre_merge, post_merge_by_frame=post_merge_suppressed)
    assert audit_ok["total_boxes_pre_merge"] == 2
    assert audit_ok["total_boxes_post_merge"] == 1
    assert audit_ok["residual_near_duplicate_pairs"] == 0

    # Simulate a merge pass that (incorrectly) left both boxes -- this is the
    # failure mode the audit exists to catch.
    audit_failed = audit_seam_duplication(pre_merge_by_frame=pre_merge, post_merge_by_frame=pre_merge)
    assert audit_failed["total_boxes_post_merge"] == 2
    assert audit_failed["residual_near_duplicate_pairs"] == 1
    assert audit_failed["residual_near_duplicate_frame_count"] == 1
    assert audit_failed["residual_near_duplicate_iou_median"] is not None
    assert audit_failed["residual_near_duplicate_iou_median"] > 0.3
