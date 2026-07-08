from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from threed.racketsport.court_profile_match import (
    COURT_PROFILE_REUSE_MEDIAN_BAR_PX,
    COURT_PROFILE_REUSE_P95_BAR_PX,
    FourLineVerification,
    decide_court_profile_reuse,
    line_color_lab_from_projected_lines,
    project_outer_court_lines,
    verify_outer_court_lines,
)
from threed.racketsport.profile_registry import (
    CourtProfile,
    LabColor,
    RetentionPolicy,
    SourceTrace,
    delta_e2000,
    match_court_profile,
    rank_court_profile_matches,
    update_profile,
)
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics


ACCOUNT_ID = "owner_1"
CAMERA_FINGERPRINT = "iphone15pro-wide-1x:synthetic"


def test_delta_e2000_matches_sharma_reference_pair() -> None:
    first = LabColor(l=50.0, a=2.6772, b=-79.7751)
    second = LabColor(l=50.0, a=0.0, b=-82.7485)

    assert delta_e2000(first, second) == pytest.approx(2.0425, abs=0.0005)


def test_profile_round_trip_reuses_matching_frame_with_four_line_support(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    calibration = _synthetic_calibration()
    line_color = LabColor(l=92.0, a=-3.0, b=4.0)
    profile = _court_profile(profile_id="court_home", line_color=line_color)
    update_profile(ACCOUNT_ID, profile, profiles_root=profiles_root)

    matches = rank_court_profile_matches(
        ACCOUNT_ID,
        camera_fingerprint=CAMERA_FINGERPRINT,
        line_color_lab=LabColor(l=92.0, a=-3.0, b=4.2),
        profiles_root=profiles_root,
    )
    matched_profile, match_confidence = match_court_profile(
        ACCOUNT_ID,
        camera_fingerprint=CAMERA_FINGERPRINT,
        line_color_lab=LabColor(l=92.0, a=-3.0, b=4.2),
        profiles_root=profiles_root,
    )
    verification = verify_outer_court_lines(
        _blank_frame(),
        calibration,
        candidate_segments=list(project_outer_court_lines(calibration).values()),
    )

    decision = decide_court_profile_reuse(matches, {profile.profile_id: verification})

    assert len(matches) == 1
    assert matches[0].profile.profile_id == "court_home"
    assert matches[0].color_delta_e2000 is not None
    assert matches[0].match_confidence < 1.0
    assert matched_profile is not None
    assert matched_profile.profile_id == "court_home"
    assert match_confidence == pytest.approx(matches[0].match_confidence)
    assert verification.passed is True
    assert verification.median_px <= COURT_PROFILE_REUSE_MEDIAN_BAR_PX
    assert verification.p95_px <= COURT_PROFILE_REUSE_P95_BAR_PX
    assert verification.recovered_lines == 4
    assert decision.outcome == "reuse"
    assert decision.profile is not None
    assert decision.profile.profile_id == "court_home"
    assert decision.court_source == "profile_reuse"
    assert decision.needs_profile_refresh_offer is False


def test_missing_profile_degrades_to_generic_path() -> None:
    decision = decide_court_profile_reuse([], {})

    assert decision.outcome == "generic_path"
    assert decision.profile is None
    assert decision.court_source == "generic_path"
    assert decision.needs_profile_refresh_offer is False


def test_fingerprint_and_color_match_with_moved_camera_falls_through_without_overwrite(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    calibration = _synthetic_calibration()
    profile = _court_profile(profile_id="court_home", line_color=LabColor(l=92.0, a=-3.0, b=4.0))
    registry = update_profile(ACCOUNT_ID, profile, profiles_root=profiles_root)

    matches = rank_court_profile_matches(
        ACCOUNT_ID,
        camera_fingerprint=CAMERA_FINGERPRINT,
        line_color_lab=LabColor(l=92.0, a=-3.0, b=4.0),
        profiles_root=profiles_root,
    )
    moved_segments = [_shift_segment(segment, dx=28.0, dy=0.0) for segment in project_outer_court_lines(calibration).values()]
    verification = verify_outer_court_lines(_blank_frame(), calibration, candidate_segments=moved_segments)

    decision = decide_court_profile_reuse(matches, {profile.profile_id: verification})

    assert verification.passed is False
    assert verification.p95_px > COURT_PROFILE_REUSE_P95_BAR_PX
    assert decision.outcome == "fall_through_refresh_offer"
    assert decision.profile is None
    assert decision.court_source == "generic_path"
    assert decision.needs_profile_refresh_offer is True
    assert registry.registry_version == 2


def test_identical_color_two_court_ambiguity_uses_four_line_geometry_tiebreaker(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    line_color = LabColor(l=92.0, a=-3.0, b=4.0)
    first = _court_profile(profile_id="court_a", line_color=line_color)
    second = _court_profile(profile_id="court_b", line_color=line_color)
    update_profile(ACCOUNT_ID, first, profiles_root=profiles_root)
    update_profile(ACCOUNT_ID, second, profiles_root=profiles_root)

    calibration = _synthetic_calibration()
    matching_verification = verify_outer_court_lines(
        _blank_frame(),
        calibration,
        candidate_segments=list(project_outer_court_lines(calibration).values()),
    )
    wrong_verification = verify_outer_court_lines(
        _blank_frame(),
        calibration,
        candidate_segments=[
            _shift_segment(segment, dx=0.0, dy=32.0) for segment in project_outer_court_lines(calibration).values()
        ],
    )

    matches = rank_court_profile_matches(
        ACCOUNT_ID,
        camera_fingerprint=CAMERA_FINGERPRINT,
        line_color_lab=line_color,
        profiles_root=profiles_root,
    )
    decision = decide_court_profile_reuse(
        matches,
        {
            "court_a": wrong_verification,
            "court_b": matching_verification,
        },
    )

    assert [match.profile.profile_id for match in matches] == ["court_a", "court_b"]
    assert decision.outcome == "reuse"
    assert decision.profile is not None
    assert decision.profile.profile_id == "court_b"
    assert decision.verification == matching_verification


def test_line_color_helper_samples_projected_outer_lines_in_cielab() -> None:
    calibration = _synthetic_calibration()
    image = _blank_frame()
    line_bgr = (245, 248, 250)
    for segment in project_outer_court_lines(calibration).values():
        (x1, y1), (x2, y2) = segment
        cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), line_bgr, 5)

    sampled = line_color_lab_from_projected_lines(image, calibration)
    expected = _bgr_to_lab_color(line_bgr)

    assert sampled.l == pytest.approx(expected.l, abs=1.0)
    assert sampled.a == pytest.approx(expected.a, abs=1.0)
    assert sampled.b == pytest.approx(expected.b, abs=1.0)


def _synthetic_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[120.0 / 20.0, 0.0, 640.0], [0.0, 120.0 / 20.0, 360.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=120.0, fy=120.0, cx=640.0, cy=360.0, dist=[], source="synthetic"),
        image_size=(1280, 720),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 20.0],
            camera_height_m=20.0,
        ),
        reprojection_error_px={"median": 0.0, "p95": 0.0},
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[[621.712, 319.7664], [658.288, 319.7664], [658.288, 400.2336], [621.712, 400.2336]],
        world_pts=[
            [-3.048, -6.7056, 0.0],
            [3.048, -6.7056, 0.0],
            [3.048, 6.7056, 0.0],
            [-3.048, 6.7056, 0.0],
        ],
    )


def _court_profile(*, profile_id: str, line_color: LabColor) -> CourtProfile:
    return CourtProfile(
        schema_version=1,
        artifact_type="racketsport_court_profile",
        account_id=ACCOUNT_ID,
        profile_id=profile_id,
        display_name=profile_id.replace("_", " ").title(),
        source_trace=_source_trace(),
        retention=RetentionPolicy(
            scope="account_lifetime",
            delete_with_source_clip=True,
            delete_with_source_profile=True,
            legal_basis="owner_setup",
        ),
        frozen_calibration_ref={
            "uri": f"runs/calibration/{profile_id}/court_calibration.json",
            "artifact_type": "court_calibration",
            "source_trace": _source_trace(),
        },
        line_paint_color_lab=line_color,
        background_frame_ref={
            "uri": f"runs/backgrounds/{profile_id}/frame_000120.jpg",
            "artifact_type": "background_frame",
            "source_trace": _source_trace(),
        },
        camera_fingerprint=CAMERA_FINGERPRINT,
    )


def _source_trace() -> SourceTrace:
    return SourceTrace(
        source_clip_id="owner_empty_court_20260708",
        source_clip_ref="runs/owner_data/owner_empty_court_20260708/clip.mov",
    )


def _blank_frame() -> object:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _shift_segment(segment: tuple[tuple[float, float], tuple[float, float]], *, dx: float, dy: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((segment[0][0] + dx, segment[0][1] + dy), (segment[1][0] + dx, segment[1][1] + dy))


def _bgr_to_lab_color(bgr: tuple[int, int, int]) -> LabColor:
    pixel = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return LabColor(
        l=float(lab[0]) * 100.0 / 255.0,
        a=float(lab[1]) - 128.0,
        b=float(lab[2]) - 128.0,
    )
