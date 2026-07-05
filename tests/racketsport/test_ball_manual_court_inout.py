from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_manual_court_inout import (
    apply_manual_court_inout_to_ball_track,
    manual_court_projection_from_corners,
)
from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.schemas import BallTrack, validate_artifact_file


TARGET_IMAGE_SIZE = (1000, 600)


def _court_corners_payload() -> dict[str, object]:
    return {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [100.0, 500.0],
                        "near_right": [900.0, 500.0],
                        "far_right": [700.0, 100.0],
                        "far_left": [300.0, 100.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(TARGET_IMAGE_SIZE),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }


def _ball_track_with_bounces(contact_points: list[list[float]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [100.0, 100.0], "conf": 0.8, "visible": True, "approx": False},
        ],
        "bounces": [
            {
                "t": index / 60.0,
                "frame": index,
                "world_xy": [0.0, 0.0],
                "contact_xy_img": contact,
                "p_bounce": 0.71,
                "source": "image_velocity_inflection_2d",
            }
            for index, contact in enumerate(contact_points)
        ],
    }


def _image_for_world_xy(world_xy: list[float]) -> list[float]:
    projection = manual_court_projection_from_corners(_court_corners_payload(), sport="pickleball")
    return project_planar_points(projection["homography"], [world_xy])[0]


def test_manual_court_projection_uses_four_corners_and_reports_reprojection_error() -> None:
    projection = manual_court_projection_from_corners(_court_corners_payload(), sport="pickleball")

    assert projection["status"] == "TESTED-ON-REAL-DATA"
    assert projection["corner_status"] == "corrected_unverified"
    assert projection["reprojection_error_px"]["median"] == pytest.approx(0.0)
    assert projection["reprojection_error_px"]["p95"] == pytest.approx(0.0)
    assert projection["not_ground_truth"] is True


def test_apply_manual_court_inout_projects_contact_points_and_sets_uncertainty_calls() -> None:
    template = get_court_template("pickleball")
    inside_img = _image_for_world_xy([0.0, 0.0])
    outside_img = _image_for_world_xy([template.width_m, 0.0])
    near_line_img = _image_for_world_xy([template.width_m / 2.0 + 0.02, 0.0])

    payload, summary = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([inside_img, outside_img, near_line_img]),
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        uncertainty_m=0.05,
    )

    parsed = BallTrack.model_validate(payload)
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["projected_bounce_count"] == 3
    assert parsed.bounces[0].call == "in"
    assert parsed.bounces[0].margin_m is not None and parsed.bounces[0].margin_m > 1.0
    assert parsed.bounces[0].nearest_line in {"near_baseline", "far_baseline", "left_sideline", "right_sideline"}
    assert parsed.bounces[0].region is not None
    assert parsed.bounces[0].dominant_uncertainty_term == "manual_corner_homography_projection"
    assert parsed.bounces[0].not_ground_truth is True
    assert parsed.bounces[0].render_only is True
    assert parsed.bounces[0].not_for_detection_metrics is True
    assert parsed.bounces[1].call == "out"
    assert parsed.bounces[1].margin_m is not None and parsed.bounces[1].margin_m < 0.0
    assert parsed.bounces[2].call == "too_close_to_call"
    assert parsed.bounces[2].uncertainty_m == pytest.approx(0.05)


def test_apply_manual_court_inout_cli_writes_schema_valid_track(tmp_path: Path) -> None:
    corners = tmp_path / "court_corners.json"
    track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_track_inout.json"
    summary = tmp_path / "manual_inout_summary.json"
    corners.write_text(json.dumps(_court_corners_payload()), encoding="utf-8")
    track.write_text(json.dumps(_ball_track_with_bounces([_image_for_world_xy([0.0, 0.0])])), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/apply_manual_court_inout.py",
            "--ball-track",
            str(track),
            "--court-corners",
            str(corners),
            "--target-width",
            str(TARGET_IMAGE_SIZE[0]),
            "--target-height",
            str(TARGET_IMAGE_SIZE[1]),
            "--uncertainty-m",
            "0.05",
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["projected_bounce_count"] == 1
    parsed = validate_artifact_file("ball_track", out)
    assert isinstance(parsed, BallTrack)
    assert parsed.bounces[0].call == "in"
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "TESTED-ON-REAL-DATA"


def test_apply_manual_court_inout_defaults_to_camera_geometry_uncertainty() -> None:
    """Task #14: the fixed 0.05 m radius must no longer be the default."""

    inside_img = _image_for_world_xy([0.0, 0.0])

    payload, summary = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([inside_img]),
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
    )

    parsed = BallTrack.model_validate(payload)
    bounce = parsed.bounces[0]
    assert summary["uncertainty_m"] is None
    assert summary["uncertainty_model"]["method"] == "camera_geometry_elevation_parallax_v1"
    assert summary["uncertainty_model"]["pose"]["camera_height_m"] > 0.0
    assert "physics_constants" in summary["uncertainty_model"]
    assert bounce.uncertainty_m is not None and bounce.uncertainty_m > 0.0
    assert bounce.dominant_uncertainty_term in {
        "camera_geometry_elevation_parallax",
        "calibration_reprojection",
        "ball_radius",
        "pixel_localization",
    }
    assert bounce.uncertainty_breakdown is not None
    assert bounce.uncertainty_breakdown.method == "camera_geometry_elevation_parallax_v1"
    assert bounce.uncertainty_breakdown.h_max_m is not None and bounce.uncertainty_breakdown.h_max_m > 0.0
    assert bounce.uncertainty_breakdown.camera_height_m is not None and bounce.uncertainty_breakdown.camera_height_m > 0.0


def test_apply_manual_court_inout_explicit_uncertainty_m_overrides_geometry() -> None:
    inside_img = _image_for_world_xy([0.0, 0.0])

    payload, summary = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([inside_img]),
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        uncertainty_m=0.05,
    )

    parsed = BallTrack.model_validate(payload)
    bounce = parsed.bounces[0]
    assert summary["uncertainty_m"] == pytest.approx(0.05)
    assert summary["uncertainty_model"]["method"] == "fixed_override"
    assert bounce.uncertainty_m == pytest.approx(0.05)
    assert bounce.dominant_uncertainty_term == "manual_corner_homography_projection"
    assert bounce.uncertainty_breakdown is not None
    assert bounce.uncertainty_breakdown.method == "fixed_override"


def test_apply_manual_court_inout_geometric_uncertainty_widens_on_a_steeper_far_camera() -> None:
    """Steep/low camera corners (burlington_gold_0300_low_steep_corner, real
    reviewed corners) must give a wider far-court band than a much more
    overhead-looking synthetic corner set, for a comparable court position."""

    # Burlington's real reviewed corners were tapped against a 960x540
    # preview (see threed/racketsport/ball_manual_court_inout.py's module
    # docstring); declare that pixel space explicitly and keep this test's
    # target_image_size equal to it (scale=1) so the steep-vs-overhead
    # geometry comparison below is not entangled with any rescale.
    steep_image_size = (960, 540)
    steep_corners = {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [146.0, 328.0],
                        "near_right": [555.0, 522.0],
                        "far_right": [924.0, 259.0],
                        "far_left": [688.0, 248.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(steep_image_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }
    overhead_image_size = (1000, 1000)
    overhead_corners = {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [250.0, 750.0],
                        "near_right": [750.0, 750.0],
                        "far_right": [700.0, 250.0],
                        "far_left": [300.0, 250.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(overhead_image_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }

    template = get_court_template("pickleball")
    far_corner_world = [template.width_m / 2.0 - 0.2, template.length_m / 2.0 - 0.2]

    steep_projection = manual_court_projection_from_corners(steep_corners, sport="pickleball", target_image_size=steep_image_size)
    steep_img = project_planar_points(steep_projection["homography"], [far_corner_world])[0]
    overhead_projection = manual_court_projection_from_corners(
        overhead_corners, sport="pickleball", target_image_size=overhead_image_size
    )
    overhead_img = project_planar_points(overhead_projection["homography"], [far_corner_world])[0]

    steep_payload, _ = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([steep_img]), steep_corners, target_image_size=steep_image_size
    )
    overhead_payload, _ = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([overhead_img]), overhead_corners, target_image_size=overhead_image_size
    )

    steep_uncertainty = BallTrack.model_validate(steep_payload).bounces[0].uncertainty_m
    overhead_uncertainty = BallTrack.model_validate(overhead_payload).bounces[0].uncertainty_m
    assert steep_uncertainty is not None and overhead_uncertainty is not None
    assert steep_uncertainty > overhead_uncertainty


def test_apply_manual_court_inout_cli_defaults_to_geometric_uncertainty(tmp_path: Path) -> None:
    corners = tmp_path / "court_corners.json"
    track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_track_inout.json"
    summary = tmp_path / "manual_inout_summary.json"
    corners.write_text(json.dumps(_court_corners_payload()), encoding="utf-8")
    track.write_text(json.dumps(_ball_track_with_bounces([_image_for_world_xy([0.0, 0.0])])), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/apply_manual_court_inout.py",
            "--ball-track",
            str(track),
            "--court-corners",
            str(corners),
            "--target-width",
            str(TARGET_IMAGE_SIZE[0]),
            "--target-height",
            str(TARGET_IMAGE_SIZE[1]),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_summary = json.loads(completed.stdout)
    assert stdout_summary["uncertainty_m"] is None
    assert stdout_summary["uncertainty_model"]["method"] == "camera_geometry_elevation_parallax_v1"
    parsed = validate_artifact_file("ball_track", out)
    assert isinstance(parsed, BallTrack)
    assert parsed.bounces[0].uncertainty_breakdown is not None
    assert parsed.bounces[0].uncertainty_breakdown.method == "camera_geometry_elevation_parallax_v1"


# --- Pixel-space contract regression tests (Task #20, 2026-07-02) ----------
#
# A 2026-07-02 audit found that every reviewed court_corners.json sidecar had
# been tapped against a 960x540 preview frame while the BALL tracks/bounces
# they were projected against are native 1920x1080 -- the raw corner pixels
# were silently consumed as if already native, misplacing the homography by
# 2x and flipping several reviewed "in" bounces to "out" by 0.9-2.3 m. These
# tests cover the fix: corner artifacts must declare their own pixel space
# (fail closed if they don't), and rescaling to a caller-supplied
# target_image_size must apply the declared scale exactly once (the
# "double-scaling hazard").


def test_manual_court_projection_fails_closed_when_image_size_undeclared() -> None:
    payload = _court_corners_payload()
    del payload["annotation"]["items"][0]["image_size"]  # type: ignore[index]

    with pytest.raises(ValueError, match="image_size"):
        manual_court_projection_from_corners(payload, sport="pickleball")


def test_manual_court_pose_fails_closed_when_image_size_undeclared() -> None:
    from threed.racketsport.ball_manual_court_inout import manual_court_pose_from_corners

    payload = _court_corners_payload()
    del payload["annotation"]["items"][0]["image_size"]  # type: ignore[index]

    with pytest.raises(ValueError, match="image_size"):
        manual_court_pose_from_corners(payload, sport="pickleball")


def test_apply_manual_court_inout_fails_closed_when_image_size_undeclared() -> None:
    payload = _court_corners_payload()
    del payload["annotation"]["items"][0]["image_size"]  # type: ignore[index]

    with pytest.raises(ValueError, match="image_size"):
        apply_manual_court_inout_to_ball_track(
            _ball_track_with_bounces([[500.0, 500.0]]),
            payload,
            target_image_size=(1920, 1080),
        )


def test_manual_court_projection_rescales_declared_corners_to_target_exactly_once() -> None:
    """Requesting a 2x target must scale the declared corners by 2x, not more/less.

    This is the "double-scaling hazard" guard: a naive refactor that applied
    the scale twice (e.g. once in a loader and again here) would produce 4x,
    and one that dropped it silently would produce 1x -- both wrong.
    """

    declared_size = (960, 540)
    target_size = (1920, 1080)
    corners = {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [146.0, 328.0],
                        "near_right": [555.0, 522.0],
                        "far_right": [924.0, 259.0],
                        "far_left": [688.0, 248.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(declared_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }

    unscaled = manual_court_projection_from_corners(corners, sport="pickleball")
    rescaled = manual_court_projection_from_corners(corners, sport="pickleball", target_image_size=target_size)

    assert unscaled["declared_image_size"] == [960, 540]
    assert unscaled["target_image_size"] is None
    assert unscaled["corner_pixel_scale_applied"] == [1.0, 1.0]
    assert rescaled["target_image_size"] == [1920, 1080]
    assert rescaled["corner_pixel_scale_applied"] == pytest.approx([2.0, 2.0])
    for raw_point, scaled_point in zip(unscaled["image_pts"], rescaled["image_pts"], strict=True):
        assert scaled_point == pytest.approx([raw_point[0] * 2.0, raw_point[1] * 2.0])

    # Declaring the corners already in the target space up front (a human
    # correctly re-tapping against the native frame) must reproduce exactly
    # the same rescaled points and homography as declaring the small preview
    # space and asking for a 2x rescale -- the two paths must agree.
    already_native_corners = {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        name: [value[0] * 2.0, value[1] * 2.0]
                        for name, value in corners["annotation"]["items"][0]["court_corners"].items()
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(target_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }
    already_native = manual_court_projection_from_corners(already_native_corners, sport="pickleball", target_image_size=target_size)
    assert already_native["corner_pixel_scale_applied"] == pytest.approx([1.0, 1.0])
    for expected_point, actual_point in zip(rescaled["image_pts"], already_native["image_pts"], strict=True):
        assert actual_point == pytest.approx(expected_point)
    for expected_row, actual_row in zip(rescaled["homography"], already_native["homography"], strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_manual_court_projection_rejects_non_uniform_target_aspect() -> None:
    """A target_image_size that does not share the declared corners' aspect
    ratio almost certainly means one of the two sizes is wrong (e.g. width
    and height swapped) -- refuse to silently stretch the court."""

    corners = _court_corners_payload()  # declares image_size = TARGET_IMAGE_SIZE = (1000, 600)

    with pytest.raises(ValueError, match="aspect ratio"):
        manual_court_projection_from_corners(corners, sport="pickleball", target_image_size=(600, 1000))


def test_apply_manual_court_inout_wrong_declared_scale_flips_call_vs_correct_scale() -> None:
    """Reproduces the actual 2026-07-02 incident mechanism in miniature: a
    contact point that is genuinely in-bounds (by construction, projected
    from a safely-inside world point through the *correctly* rescaled
    homography) must be called "in" when the corners are rescaled to the
    ball track's real native pixel space, but is miscalled when the corners
    are wrongly treated as already being in that native space (scale=1,
    the pre-fix bug's exact failure mode)."""

    declared_size = (960, 540)
    native_size = (1920, 1080)
    corners = {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [146.0, 328.0],
                        "near_right": [555.0, 522.0],
                        "far_right": [924.0, 259.0],
                        "far_left": [688.0, 248.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(declared_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }

    template = get_court_template("pickleball")
    safely_inside_world = [0.0, 0.0]  # dead center of the court
    correct_projection = manual_court_projection_from_corners(corners, sport="pickleball", target_image_size=native_size)
    native_contact_xy = project_planar_points(correct_projection["homography"], [safely_inside_world])[0]

    correct_payload, _ = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([native_contact_xy]),
        corners,
        target_image_size=native_size,
        uncertainty_m=0.05,
    )
    correct_call = BallTrack.model_validate(correct_payload).bounces[0]

    # The pre-fix bug: treat the native-pixel contact point as if it were
    # already in the declared (960x540) corner space, i.e. no rescale.
    buggy_payload, _ = apply_manual_court_inout_to_ball_track(
        _ball_track_with_bounces([native_contact_xy]),
        corners,
        target_image_size=declared_size,
        uncertainty_m=0.05,
    )
    buggy_call = BallTrack.model_validate(buggy_payload).bounces[0]

    assert correct_call.call == "in"
    assert correct_call.margin_m is not None and correct_call.margin_m > 0.0
    # Unscaled, the same native-pixel point projects far outside the
    # (half-sized) court model built from the un-rescaled homography.
    assert buggy_call.call == "out"
    assert buggy_call.margin_m is not None and buggy_call.margin_m < correct_call.margin_m
