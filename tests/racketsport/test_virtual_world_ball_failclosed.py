"""Fail-closed 3D ball emission at the arc-solved overlay boundary.

Diagnosis evidence (runs/lanes/w7_ball3ddiag_20260709/DIAGNOSIS.md): the arc
solver's ``fit_bvp_fallback`` segments with rejected 2D support (0-1 inliers,
dozens of outliers, huge reprojection error) still had their ``world_xyz``
copied into the owner-visible world — 263/300 fallback frames rendered, 0
hidden, 210 labeled ``measured``. The fix is a fail-closed boundary in
``apply_ball_track_arc_solved_overlay``: a fallback/weak segment whose own fit
statistics show the solver ignored the 2D evidence contributes
``world_xyz = null`` (the confidence gate then bands those frames hidden),
with per-segment verdicts recorded in the overlay provenance block.

The synthetic segments below mirror the measured wolverine stats: segments
akin to 2 (1 inlier / 90 outliers, max reprojection 3585 px, 23.5 m apex),
3 (0 / 52), and 6 (1 / 5) must be suppressed; the well-fit segment 1 and the
image-consistent fallback segments 4/7/9 (0 outliers, low reprojection) must
keep their positions.
"""

from __future__ import annotations

import pytest

from threed.racketsport.virtual_world import (
    apply_ball_track_arc_solved_overlay,
    ball_arc_segment_fail_closed_verdicts,
)


def _arc_frame(index: int, segment_id: int, *, fps: float = 30.0, hidden: bool = False) -> dict:
    frame = {
        "t": round(index / fps, 6),
        "band": "hidden" if hidden else "arc_weak",
        "world_xyz": None if hidden else [0.1 * segment_id, 2.0, 1.0 + 0.01 * index],
        "arc_solver": {
            "lane": "BALL-ARC-SOLVER",
            "segment_id": segment_id,
            "segment_status": "fit_bvp_fallback",
            "bvp_fallback_segment": True,
        },
    }
    return frame


def _segment(
    segment_id: int,
    status: str,
    frame_start: int,
    frame_end: int,
    *,
    inliers: int,
    outliers: int,
    max_reproj_px: float,
) -> dict:
    return {
        "segment_id": segment_id,
        "status": status,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "inlier_count": inliers,
        "outlier_count": outliers,
        "max_reprojection_error_px": max_reproj_px,
        "reprojection_rmse_px": min(max_reproj_px, 20.0),
    }


SEGMENTS = [
    # Well-fit segment: always trusted.
    _segment(0, "fit", 0, 9, inliers=14, outliers=0, max_reproj_px=2.3),
    # Wolverine segment-2 analogue: solver ignored 2D entirely.
    _segment(1, "fit_bvp_fallback", 9, 19, inliers=1, outliers=90, max_reproj_px=3585.0),
    # Wolverine segment-3 analogue: zero-inlier endpoint-only arc.
    _segment(2, "fit_bvp_fallback", 19, 29, inliers=0, outliers=52, max_reproj_px=620.5),
    # Image-consistent fallback (wolverine 4/7/9 analogue): keep.
    _segment(3, "fit_bvp_fallback", 29, 39, inliers=10, outliers=0, max_reproj_px=12.9),
    # Wolverine segment-6 analogue: minority inliers, moderate reprojection.
    _segment(4, "fit_bvp_fallback", 39, 49, inliers=1, outliers=5, max_reproj_px=183.9),
]


def _artifact(n_frames: int = 50) -> dict:
    frames = []
    for index in range(n_frames):
        segment_id = min(index // 10, len(SEGMENTS) - 1)
        frames.append(_arc_frame(index, segment_id))
    return {
        "artifact_type": "racketsport_ball_track_arc_solved",
        "status": "ran",
        "segments": [dict(seg) for seg in SEGMENTS],
        "frames": frames,
    }


def _physics_filled(n_frames: int = 50) -> dict:
    return {
        "artifact_type": "racketsport_ball_track_physics_filled",
        "frames": [
            {"t": round(index / 30.0, 6), "world_xyz": [9.0, 9.0, 9.0], "conf": 0.9}
            for index in range(n_frames)
        ],
    }


class TestSegmentVerdicts:
    def test_untrusted_segments_are_exactly_the_diagnosed_ones(self):
        verdicts = ball_arc_segment_fail_closed_verdicts(_artifact()["segments"])
        suppressed = {sid for sid, verdict in verdicts.items() if not verdict["trusted"]}
        assert suppressed == {1, 2, 4}

    def test_fit_segments_are_never_suppressed(self):
        verdicts = ball_arc_segment_fail_closed_verdicts(
            [_segment(0, "fit", 0, 9, inliers=0, outliers=99, max_reproj_px=9999.0)]
        )
        assert verdicts[0]["trusted"] is True

    def test_each_untrusted_verdict_names_a_reason(self):
        verdicts = ball_arc_segment_fail_closed_verdicts(_artifact()["segments"])
        for verdict in verdicts.values():
            if not verdict["trusted"]:
                assert verdict["reasons"], verdict

    def test_missing_stats_fail_closed_for_fallback(self):
        # A fallback segment with absent fit statistics must not be trusted.
        seg = {"segment_id": 7, "status": "fit_bvp_fallback", "frame_start": 0, "frame_end": 5}
        verdicts = ball_arc_segment_fail_closed_verdicts([seg])
        assert verdicts[7]["trusted"] is False

    def test_spatial_sanity_violation_suppresses_even_with_good_fit(self):
        # Wolverine segment-0 analogue: image-consistent fallback whose
        # trajectory starts 5 m behind the baseline.
        seg = _segment(11, "fit_bvp_fallback", 0, 12, inliers=8, outliers=4, max_reproj_px=14.7)
        seg["physical_sanity"] = {"violation": True, "violations": ["outside_court_volume"]}
        verdicts = ball_arc_segment_fail_closed_verdicts([seg])
        assert verdicts[11]["trusted"] is False
        assert "spatial_sanity_violation" in verdicts[11]["reasons"]

    def test_speed_only_sanity_violation_keeps_pixel_consistent_segment(self):
        # Wolverine segment-7/9/10 analogue: slow but matches the pixels.
        seg = _segment(12, "fit_bvp_fallback", 0, 12, inliers=10, outliers=0, max_reproj_px=12.9)
        seg["physical_sanity"] = {
            "violation": True,
            "violations": ["initial_speed_outside_plausible_range_mps"],
        }
        verdicts = ball_arc_segment_fail_closed_verdicts([seg])
        assert verdicts[12]["trusted"] is True

    def test_spatial_violation_on_fit_segment_is_still_trusted(self):
        # 'fit' segments are the solver's honest wins; depth ambiguity on a
        # true fit is the solver's problem to fix, not emission policy's.
        seg = _segment(13, "fit", 0, 12, inliers=14, outliers=0, max_reproj_px=2.3)
        seg["physical_sanity"] = {"violation": True, "violations": ["outside_court_volume"]}
        verdicts = ball_arc_segment_fail_closed_verdicts([seg])
        assert verdicts[13]["trusted"] is True


class TestOverlayFailClosed:
    def test_suppressed_segment_frames_lose_world_xyz(self):
        result = apply_ball_track_arc_solved_overlay(_physics_filled(), _artifact())
        frames = result["frames"]
        # Segment 1 analogue occupies indices 10-19, segment 2 -> 20-29, segment 4 -> 40-49.
        for index in [*range(10, 30), *range(40, 50)]:
            assert frames[index]["world_xyz"] is None, index
        # Trusted segments keep the arc-evaluated positions.
        for index in [*range(0, 10), *range(30, 40)]:
            assert frames[index]["world_xyz"] is not None, index

    def test_overlay_provenance_records_policy_and_counts(self):
        result = apply_ball_track_arc_solved_overlay(_physics_filled(), _artifact())
        overlay = result["arc_solved_overlay"]
        assert overlay["applied"] is True
        fail_closed = overlay["fail_closed"]
        assert fail_closed["enabled"] is True
        assert fail_closed["suppressed_frame_count"] == 30
        assert set(fail_closed["suppressed_segment_ids"]) == {1, 2, 4}
        assert fail_closed["min_inlier_count"] == 3
        assert fail_closed["max_reprojection_error_px"] == pytest.approx(40.0)

    def test_fail_open_escape_hatch_preserves_old_behavior(self):
        result = apply_ball_track_arc_solved_overlay(
            _physics_filled(), _artifact(), fail_closed=False
        )
        frames = result["frames"]
        assert all(frame["world_xyz"] is not None for frame in frames)
        assert result["arc_solved_overlay"]["fail_closed"]["enabled"] is False

    def test_frames_without_segment_provenance_fail_closed_when_any_segment_untrusted(self):
        artifact = _artifact()
        # Strip solver provenance from one frame inside an untrusted span.
        artifact["frames"][15] = {
            "t": artifact["frames"][15]["t"],
            "band": "arc_weak",
            "world_xyz": [1.0, 2.0, 3.0],
        }
        result = apply_ball_track_arc_solved_overlay(_physics_filled(), artifact)
        # Frame 15 falls in untrusted segment 1's span (9-19): must be suppressed
        # via the span fallback even without arc_solver.segment_id.
        assert result["frames"][15]["world_xyz"] is None

    def test_hidden_band_frames_stay_hidden(self):
        artifact = _artifact()
        artifact["frames"][5] = _arc_frame(5, 0, hidden=True)
        result = apply_ball_track_arc_solved_overlay(_physics_filled(), artifact)
        assert result["frames"][5]["world_xyz"] is None
