from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_bounce_2d import detect_2d_bounces_from_ball_track, write_2d_bounce_ball_track
from threed.racketsport.ball_manual_court_inout import manual_court_projection_from_corners
from threed.racketsport.court_calibration import project_planar_points
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


def _image_for_world_xy(world_xy: list[float]) -> list[float]:
    projection = manual_court_projection_from_corners(_court_corners_payload(), sport="pickleball")
    return project_planar_points(projection["homography"], [world_xy])[0]


def _ball_track_payload() -> dict[str, object]:
    # Image y goes down, so a ground bounce is a local maximum in image-y.
    points = [
        _image_for_world_xy([0.0, 1.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 0.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 1.0]),
    ]
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": point, "conf": 0.9, "visible": True, "approx": False}
            for index, point in enumerate(points)
        ],
        "bounces": [],
    }


def test_detect_2d_bounces_uses_image_velocity_inflection_and_court_plane() -> None:
    payload, detector = detect_2d_bounces_from_ball_track(
        _ball_track_payload(),
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
        min_candidate_t_s=0.0,
    )

    parsed = BallTrack.model_validate(payload)
    assert detector["artifact_type"] == "racketsport_ball_bounce_2d_output"
    assert detector["status"] == "TESTED-ON-REAL-DATA"
    assert detector["accepted_bounce_count"] == 1
    assert parsed.bounces[0].frame == 2
    assert parsed.bounces[0].contact_xy_img == pytest.approx(_image_for_world_xy([0.0, 0.0]))
    assert parsed.bounces[0].world_xy == pytest.approx([0.0, 0.0])
    assert parsed.bounces[0].p_bounce is not None and parsed.bounces[0].p_bounce >= 0.5
    assert parsed.bounces[0].source == "image_velocity_inflection_court_plane_2d_v1"


def test_detect_2d_bounces_rejects_flight_apex_image_y_minimum() -> None:
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": [500.0, y], "conf": 0.9, "visible": True, "approx": False}
            for index, y in enumerate([500.0, 450.0, 400.0, 450.0, 500.0])
        ],
        "bounces": [],
    }

    parsed_payload, detector = detect_2d_bounces_from_ball_track(
        payload,
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
        min_candidate_t_s=0.0,
    )

    parsed = BallTrack.model_validate(parsed_payload)
    assert detector["accepted_bounce_count"] == 0
    assert parsed.bounces == []


def test_detect_2d_bounces_rejects_invisible_gap_fill_peak() -> None:
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": [500.0, y], "conf": conf, "visible": visible, "approx": not visible}
            for index, (y, conf, visible) in enumerate(
                [
                    (300.0, 0.9, True),
                    (400.0, 0.9, True),
                    (470.0, 0.0, False),
                    (500.0, 0.0, False),
                    (470.0, 0.0, False),
                    (300.0, 0.9, True),
                    (250.0, 0.9, True),
                ]
            )
        ],
        "bounces": [],
    }

    parsed_payload, detector = detect_2d_bounces_from_ball_track(
        payload,
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
        min_candidate_t_s=0.0,
    )

    parsed = BallTrack.model_validate(parsed_payload)
    assert detector["accepted_bounce_count"] == 0
    assert parsed.bounces == []


def test_detect_2d_bounces_accepts_visible_peak_before_dropout() -> None:
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": [500.0, y], "conf": conf, "visible": visible, "approx": not visible}
            for index, (y, conf, visible) in enumerate(
                [
                    (300.0, 0.9, True),
                    (400.0, 0.9, True),
                    (500.0, 0.9, True),
                    (430.0, 0.0, False),
                    (390.0, 0.0, False),
                    (350.0, 0.0, False),
                    (320.0, 0.0, False),
                    (300.0, 0.0, False),
                    (280.0, 0.0, False),
                    (260.0, 0.0, False),
                    (300.0, 0.9, True),
                ]
            )
        ],
        "bounces": [],
    }

    parsed_payload, detector = detect_2d_bounces_from_ball_track(
        payload,
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
        min_candidate_t_s=0.0,
    )

    parsed = BallTrack.model_validate(parsed_payload)
    assert detector["accepted_bounce_count"] == 1
    assert parsed.bounces[0].frame == 2


def test_detect_2d_bounces_rejects_default_subpixel_jitter() -> None:
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": [500.0, y], "conf": 0.9, "visible": True, "approx": False}
            for index, y in enumerate([300.0, 303.0, 300.0])
        ],
        "bounces": [],
    }

    parsed_payload, detector = detect_2d_bounces_from_ball_track(
        payload,
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
    )

    parsed = BallTrack.model_validate(parsed_payload)
    assert detector["accepted_bounce_count"] == 0
    assert parsed.bounces == []


def test_detect_2d_bounces_rejects_startup_transient_by_default() -> None:
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 60.0, "xy": [500.0, y], "conf": 0.9, "visible": True, "approx": False}
            for index, y in enumerate([300.0, 360.0, 300.0, 250.0])
        ],
        "bounces": [],
    }

    parsed_payload, detector = detect_2d_bounces_from_ball_track(
        payload,
        _court_corners_payload(),
        target_image_size=TARGET_IMAGE_SIZE,
        min_p_bounce=0.5,
    )

    parsed = BallTrack.model_validate(parsed_payload)
    assert detector["accepted_bounce_count"] == 0
    assert parsed.bounces == []


def test_write_2d_bounce_ball_track_cli_outputs_detector_artifact(tmp_path: Path) -> None:
    track = tmp_path / "ball_track.json"
    corners = tmp_path / "court_corners.json"
    out = tmp_path / "ball_track_bounces.json"
    detector = tmp_path / "bounce_2d_output.json"
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    corners.write_text(json.dumps(_court_corners_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_ball_bounce_2d.py",
            "--ball-track",
            str(track),
            "--court-corners",
            str(corners),
            "--target-width",
            str(TARGET_IMAGE_SIZE[0]),
            "--target-height",
            str(TARGET_IMAGE_SIZE[1]),
            "--min-candidate-t-s",
            "0.0",
            "--out",
            str(out),
            "--detector-out",
            str(detector),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["accepted_bounce_count"] == 1
    parsed = validate_artifact_file("ball_track", out)
    assert isinstance(parsed, BallTrack)
    assert len(parsed.bounces) == 1
    assert json.loads(detector.read_text(encoding="utf-8"))["algorithm"] == "image_velocity_inflection_court_plane_2d_v1"


def test_write_2d_bounce_ball_track_function_writes_outputs(tmp_path: Path) -> None:
    track = tmp_path / "ball_track.json"
    corners = tmp_path / "court_corners.json"
    out = tmp_path / "ball_track_bounces.json"
    detector = tmp_path / "bounce_2d_output.json"
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    corners.write_text(json.dumps(_court_corners_payload()), encoding="utf-8")

    summary = write_2d_bounce_ball_track(
        ball_track_path=track,
        court_corners_path=corners,
        out=out,
        detector_out=detector,
        target_image_size=TARGET_IMAGE_SIZE,
        min_candidate_t_s=0.0,
    )

    assert summary["accepted_bounce_count"] == 1
    assert out.is_file()
    assert detector.is_file()


def test_detect_2d_bounces_requires_target_image_size_and_surfaces_scale_metadata() -> None:
    """Task #20: this consumer shares manual_court_projection_from_corners with
    the BALL in/out path, so it must thread target_image_size through and
    surface how it rescaled the declared corners, not silently assume."""

    with pytest.raises(TypeError):
        detect_2d_bounces_from_ball_track(_ball_track_payload(), _court_corners_payload())  # type: ignore[call-arg]

    _, detector = detect_2d_bounces_from_ball_track(
        _ball_track_payload(),
        _court_corners_payload(),
        target_image_size=(2000, 1200),
        min_p_bounce=0.5,
        min_candidate_t_s=0.0,
    )

    assert detector["projection"]["declared_image_size"] == list(TARGET_IMAGE_SIZE)
    assert detector["projection"]["target_image_size"] == [2000, 1200]
    assert detector["projection"]["corner_pixel_scale_applied"] == pytest.approx([2.0, 2.0])
