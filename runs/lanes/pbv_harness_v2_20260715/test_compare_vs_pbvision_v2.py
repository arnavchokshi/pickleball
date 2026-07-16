from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import compare_vs_pbvision_v2 as harness


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "NORTH_STAR_ROADMAP.md").is_file()
)
PB = ROOT / "runs/research_ball3d_20260709/pbvision_cv_export/cv_export.json"
OURS = ROOT / "runs/lanes/demo_beststack_render_20260710/after_wolv"


def args(ours: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        pb_export=PB,
        ours=ours,
        ours_ball_track=None,
        ours_calibration=None,
        frame_offset="auto",
        max_lag=90,
        fps=None,
        image_size="1920x1080",
        output=None,
    )


def test_pb_schema_and_pillars() -> None:
    scorecard = harness.build_scorecard(args(None))
    assert scorecard["policy"]["pbvision_is_not_ground_truth"] is True
    assert scorecard["pbvision"]["coverage"]["emitted_3d_count"] == 183
    assert scorecard["pbvision"]["selection_and_temporal_forensics"]["selected_counts"] == {
        "ball": 173,
        "bounce": 3,
        "net": 1,
        "shot": 6,
    }
    assert scorecard["shared_no_gt_pillars"]["pbvision_physics_reintegration"][
        "physics_plausible_segment_count"
    ] == 9
    assert scorecard["shared_no_gt_pillars"]["pbvision_court_plane_bounce"][
        "selected_bounce_z_error_m"
    ]["max"] < 1e-12


def test_raw_arc_alignment_and_determinism() -> None:
    first = harness.build_scorecard(args(OURS / "ball_track_arc_solved.json"))
    second = harness.build_scorecard(args(OURS / "ball_track_arc_solved.json"))
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["alignment"]["selected_offset"] == 0
    assert first["alignment"]["best"]["correlation"] > 0.949
    assert first["two_d_head_to_head"]["ours_only_count"] == 66
    assert first["ours"]["coverage_on_pb_rally"]["segment_status_counts"] == {
        "fit": 23,
        "fit_bvp_fallback": 229,
    }


def test_directory_world_and_plain_ball_track_are_supported() -> None:
    world = harness.build_scorecard(args(OURS))
    assert world["inputs"]["ours_resolved"].endswith("confidence_gated_world.json")
    assert world["ours"]["coverage_on_pb_rally"]["emitted_3d_count"] == 58

    track = harness.build_scorecard(args(OURS / "ball_track.json"))
    assert track["ours"]["coverage_on_pb_rally"]["emitted_3d_count"] == 0
    assert track["ours"]["coverage_on_pb_rally"]["visible_2d_count"] == 203
    assert track["head_to_head"]["paired_3d_count"] == 0


def _sample(frame: int, xyz: tuple[float, float, float]) -> harness.Sample:
    return harness.Sample(
        frame=frame,
        t=frame / 30.0,
        xyz_m=np.asarray(xyz, dtype=float),
        xy_px=None,
        confidence=1.0,
        visible=True,
        interpolated=False,
        kind="ball",
        segment=92,
    )


def test_extreme_seed_is_typed_counted_skip_and_deterministic() -> None:
    # Mirrors the full-export crash class: two frames imply vz far beyond the
    # optimizer's +/-60 m/s bound.  V2 must report it rather than crash, clip,
    # or silently omit it.
    samples = [
        _sample(4225, (0.0, 0.0, 95.0)),
        _sample(4226, (0.0, 0.1, 104.0)),
        _sample(4227, (0.0, 0.2, 113.0)),
        _sample(4228, (0.0, 0.3, 120.0)),
        _sample(4229, (0.0, 0.4, 126.0)),
    ]
    first = harness.physics_fit(samples, 92)
    second = harness.physics_fit(samples, 92)
    assert first == second
    assert first["status"] == "physics_fit_skipped"
    assert first["physics_fit_skipped_reason"] == "initial_guess_outside_optimizer_bounds"
    assert first["out_of_bounds_parameters"] == ["vz"]

    trajectory = harness.Trajectory(
        name="extreme_seed",
        fps=30.0,
        samples={sample.frame: sample for sample in samples},
        two_d_samples={},
        source_path=Path("constructed"),
    )
    pillar = harness.physics_pillar(trajectory)
    assert pillar["segment_count"] == 1
    assert pillar["fit_segment_count"] == 0
    assert pillar["physics_fit_skipped_count"] == 1
    assert pillar["physics_fit_skipped_reason_counts"] == {
        "initial_guess_outside_optimizer_bounds": 1
    }
