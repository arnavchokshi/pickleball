from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import threed.racketsport.court_line_robustness as robustness
from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_line_keypoints import (
    DetectedCourtLineCandidate,
    DetectedCourtLineCandidates,
    _dedupe_additive_line_candidates,
    detect_court_keypoints_from_image,
    detect_court_line_candidates_from_image,
)
from threed.racketsport.court_line_robustness import (
    AssignedCourtLine,
    CourtLineHardeningConfig,
    CourtLineHardeningResult,
    FrameCourtLineEvidence,
    canonical_json_bytes,
    maybe_apply_court_line_hardening,
    pool_static_semantic_lines,
    refine_pooled_homography,
    run_court_line_hardening,
    select_regulation_template_lines,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_proposal_optimizer import (
    FLOOR_KEYPOINTS,
    RefinementConfig,
    refine_homography_with_lines,
)
from threed.racketsport.court_templates import get_court_template


def _candidate(
    candidate_id: str,
    segment: tuple[tuple[float, float], tuple[float, float]],
    *,
    support: float | None = None,
) -> DetectedCourtLineCandidate:
    (x1, y1), (x2, y2) = segment
    return DetectedCourtLineCandidate(
        candidate_id=candidate_id,
        endpoints=segment,
        support_length_px=float(support or math.hypot(x2 - x1, y2 - y1)),
        source_segment_count=1,
        angle_deg=math.degrees(math.atan2(y2 - y1, x2 - x1)),
        provider="legacy_hough",
    )


def _expected_lines() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    # A single homography, rather than independently hand-authored segments,
    # makes the fixture a real regulation-projective court.
    return robustness.projected_floor_semantic_lines(_seed_calibration())


def _roi() -> tuple[tuple[float, float], ...]:
    return ((100.0, 300.0), (300.0, 300.0), (260.0, 80.0), (140.0, 80.0))


def _seed_calibration() -> dict[str, object]:
    template = get_court_template("pickleball")
    homography = homography_from_planar_points(
        template.corners_m,
        [[100.0, 300.0], [300.0, 300.0], [260.0, 80.0], [140.0, 80.0]],
    )
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "synthetic_seed",
        "homography": homography,
        "image_size": [400, 360],
        "intrinsics": {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 200.0,
            "cy": 180.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "synthetic",
        },
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
    }


def _distorted_seed_calibration() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "synthetic_distorted_seed",
        "homography": [
            [15.0, 0.0, 200.0],
            [0.0, 15.0, 180.0],
            [0.0, 0.0, 1.0],
        ],
        "image_size": [400, 360],
        "intrinsics": {
            "fx": 150.0,
            "fy": 150.0,
            "cx": 200.0,
            "cy": 180.0,
            "dist": [-0.30, 0.10, 0.0, 0.0],
            "source": "synthetic",
        },
        "extrinsics": {
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
    }


def _seed_with_floor_points(*, source: str) -> dict[str, object]:
    seed = _seed_calibration()
    seed["source"] = source
    world_points = [point.world_xyz_m for point in PICKLEBALL_KEYPOINTS]
    image_points = project_planar_points(seed["homography"], world_points)
    seed["world_pts"] = [list(point) for point in world_points]
    seed["image_pts"] = [list(point) for point in image_points]
    point_artifact = {
        "source": source,
        "world_pts": seed["world_pts"],
        "image_pts": seed["image_pts"],
    }
    seed["point_evidence_provenance"] = {
        "authority": "automatic",
        "source": source,
        "artifact_sha256": hashlib.sha256(
            canonical_json_bytes(point_artifact)
        ).hexdigest(),
        "correspondences_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "image_pts": seed["image_pts"],
                    "world_pts": seed["world_pts"],
                }
            )
        ).hexdigest(),
    }
    return seed


def _painted_static_frames(
    seed: dict[str, object],
    *,
    vertical_shift_px: float = 0.0,
) -> list[tuple[int, np.ndarray]]:
    image = np.zeros((360, 400, 3), dtype=np.uint8)
    image[:] = (35, 105, 60)
    template = get_court_template("pickleball")
    for line_id, endpoints in template.line_segments_m.items():
        if line_id == "net":
            continue
        projected = np.asarray(
            project_planar_points(seed["homography"], endpoints),
            dtype=np.float64,
        )
        projected[:, 1] += vertical_shift_px
        cv2.line(
            image,
            tuple(int(round(value)) for value in projected[0]),
            tuple(int(round(value)) for value in projected[1]),
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    frames: list[tuple[int, np.ndarray]] = []
    for frame_index in range(8):
        frame = image.copy()
        frame[0, frame_index] = (
            frame_index + 1,
            frame_index + 2,
            frame_index + 3,
        )
        frames.append((frame_index, frame))
    return frames


def test_default_off_is_exact_passthrough_and_never_calls_hardening(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = _seed_calibration()
    golden = canonical_json_bytes(seed)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled hardening invoked the candidate path")

    monkeypatch.setattr(robustness, "run_court_line_hardening", forbidden)

    implicit = maybe_apply_court_line_hardening([], seed)
    explicit = maybe_apply_court_line_hardening(
        [],
        seed,
        config=CourtLineHardeningConfig(enabled=False),
    )

    assert implicit is seed
    assert explicit is seed
    assert canonical_json_bytes(implicit) == golden
    assert canonical_json_bytes(explicit) == golden


def test_additive_candidate_extraction_does_not_change_legacy_output_bytes() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 105, 60)
    for p1, p2 in [
        ((55, 205), (265, 205)),
        ((75, 145), (245, 145)),
        ((95, 65), (225, 65)),
        ((55, 205), (95, 65)),
        ((265, 205), (225, 65)),
    ]:
        cv2.line(image, p1, p2, (245, 245, 245), 4, cv2.LINE_AA)

    before = detect_court_keypoints_from_image(image)
    before_bytes = canonical_json_bytes(asdict(before))
    assert hashlib.sha256(before_bytes).hexdigest() == (
        "0e6709ccf1afd8e291c1fd4d2990259087772aed7a46521eed37641a6ef914de"
    )
    candidates = detect_court_line_candidates_from_image(image, provider="legacy_hough")
    after = detect_court_keypoints_from_image(image)

    assert candidates.raw_segment_count > 0
    assert canonical_json_bytes(asdict(after)) == before_bytes
    assert hashlib.sha256(before_bytes).hexdigest() == hashlib.sha256(
        canonical_json_bytes(asdict(after))
    ).hexdigest()


def test_hybrid_cross_provider_duplicates_are_corroboration_not_ambiguity() -> None:
    segment = ((100.0, 300.0), (300.0, 300.0))
    legacy = replace(
        _candidate("legacy:0", segment),
        provider="legacy_hough",
        source_candidate_ids=("legacy:0",),
    )
    paired = replace(
        _candidate("paired:0", segment),
        provider="classical_paired_edges",
        source_candidate_ids=("paired:0",),
    )

    clustered = _dedupe_additive_line_candidates(
        [legacy, paired],
        max_angle_delta_deg=1.5,
        max_normal_distance_px=2.0,
    )

    assert len(clustered) == 1
    assert clustered[0].provider == "classical_paired_edges+legacy_hough"
    assert clustered[0].source_candidate_ids == ("legacy:0", "paired:0")
    assert clustered[0].support_length_px == legacy.support_length_px


def test_hybrid_dedupe_never_merges_same_provider_or_disjoint_extents() -> None:
    same_provider = [
        _candidate("legacy:0", ((20.0, 80.0), (120.0, 80.0))),
        _candidate("legacy:1", ((20.0, 80.5), (120.0, 80.5))),
    ]
    assert len(
        _dedupe_additive_line_candidates(
            same_provider,
            max_angle_delta_deg=1.5,
            max_normal_distance_px=2.0,
        )
    ) == 2

    legacy = replace(
        _candidate("legacy:2", ((20.0, 80.0), (120.0, 80.0))),
        provider="legacy_hough",
    )
    paired = replace(
        _candidate("paired:2", ((240.0, 80.0), (340.0, 80.0))),
        provider="classical_paired_edges",
    )
    assert len(
        _dedupe_additive_line_candidates(
            [legacy, paired],
            max_angle_delta_deg=1.5,
            max_normal_distance_px=2.0,
        )
    ) == 2


def test_regulation_roi_rejects_adjacent_service_shadow_and_specular_lookalikes() -> None:
    expected = _expected_lines()
    true_candidates = [
        _candidate(f"true:{line_id}", segment)
        for line_id, segment in expected.items()
    ]
    adjacent = [
        _candidate(
            f"adjacent:{line_id}",
            ((segment[0][0] + 170.0, segment[0][1]), (segment[1][0] + 170.0, segment[1][1])),
        )
        for line_id, segment in expected.items()
    ]
    distractors = [
        _candidate("tennis_service", ((105.0, 180.0), (295.0, 180.0))),
        _candidate("net_shadow", ((105.0, 178.0), (295.0, 190.0))),
        _candidate("specular_streak", ((130.0, 285.0), (260.0, 95.0))),
        # Its midpoint is inside the court, but almost all of its own support
        # lies outside. Midpoint-only ROI logic used to accept this lookalike.
        _candidate("long_crossing_streak", ((-1000.0, 300.0), (1400.0, 300.0))),
    ]
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")

    evidence = select_regulation_template_lines(
        [*true_candidates, *adjacent, *distractors],
        expected_lines=expected,
        image_size=(600, 360),
        frame_index=7,
        frame_sha256="a" * 64,
        config=config,
        roi_polygon_px=_roi(),
        roi_source="declared_config",
        distortion_state="raw_pinhole",
        seed_calibration=_seed_calibration(),
    )

    assert evidence.status == "accepted"
    assert {assignment.line_id for assignment in evidence.assignments} == set(expected)
    assert all(assignment.candidate_id.startswith("true:") for assignment in evidence.assignments)
    selected_ids = {assignment.candidate_id for assignment in evidence.assignments}
    assert not selected_ids.intersection({candidate.candidate_id for candidate in adjacent + distractors})


def test_joint_regulation_gate_undistorts_declared_raw_geometry() -> None:
    seed = _distorted_seed_calibration()
    expected = robustness.projected_floor_semantic_lines(seed)
    config = CourtLineHardeningConfig(
        enabled=True,
        provider="legacy_hough",
    )
    roi, roi_source = robustness._resolve_roi(config, seed, expected)

    evidence = select_regulation_template_lines(
        [
            _candidate(f"true:{line_id}", segment)
            for line_id, segment in expected.items()
        ],
        expected_lines=expected,
        image_size=(400, 360),
        frame_index=9,
        frame_sha256="9" * 64,
        config=config,
        roi_polygon_px=roi,
        roi_source=roi_source,
        distortion_state="raw_distorted_with_declared_model",
        seed_calibration=seed,
    )

    assert evidence.status == "accepted"
    assert evidence.assignment_model is not None
    assert evidence.assignment_model["evaluation_coordinate_space"] == (
        "pixels_undistorted_native"
    )
    assert evidence.assignment_model["joint_p90_px"] < 0.01


def test_full_run_passes_distorted_seed_into_joint_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _distorted_seed_calibration()
    expected = robustness.projected_floor_semantic_lines(seed)
    candidates = tuple(
        _candidate(f"true:{line_id}", segment)
        for line_id, segment in expected.items()
    )
    detected = DetectedCourtLineCandidates(
        candidates=candidates,
        raw_segment_count=len(candidates),
        merged_line_count=len(candidates),
        image_size=(400, 360),
        provider="legacy_hough",
    )
    monkeypatch.setattr(
        robustness,
        "detect_court_line_candidates_from_image",
        lambda *_args, **_kwargs: detected,
    )
    frames = []
    for frame_index in range(8):
        image = np.zeros((360, 400, 3), dtype=np.uint8)
        image[0, frame_index] = frame_index + 1
        frames.append((frame_index, image))

    result = run_court_line_hardening(
        frames,
        seed,
        config=CourtLineHardeningConfig(
            enabled=True,
            provider="legacy_hough",
        ),
    )

    assert all(
        frame.status == "accepted"
        for frame in result.raw_frame_evidence
    )
    assert all(
        frame.assignment_model is not None
        and frame.assignment_model["evaluation_coordinate_space"]
        == "pixels_undistorted_native"
        for frame in result.raw_frame_evidence
    )
    assert result.pooled_evidence.distortion_state == (
        "raw_distorted_with_declared_model"
    )


def test_joint_regulation_gate_rejects_one_inconsistent_line_from_mixed_geometry() -> None:
    seed = _seed_calibration()
    expected = _expected_lines()
    candidates = []
    for line_id, segment in expected.items():
        shift = 8.0 if line_id == "far_nvz" else 0.0
        candidates.append(
            _candidate(
                f"mixed:{line_id}",
                (
                    (segment[0][0], segment[0][1] + shift),
                    (segment[1][0], segment[1][1] + shift),
                ),
            )
        )

    evidence = select_regulation_template_lines(
        candidates,
        expected_lines=expected,
        image_size=(400, 360),
        frame_index=10,
        frame_sha256="a" * 64,
        config=CourtLineHardeningConfig(
            enabled=True,
            provider="legacy_hough",
        ),
        roi_polygon_px=_roi(),
        roi_source="declared_config",
        distortion_state="raw_pinhole",
        seed_calibration=seed,
    )

    assert evidence.status == "accepted"
    assert evidence.assignment_model is not None
    assert evidence.assignment_model["joint_p90_px"] < 1e-6
    assert "far_nvz" not in {
        assignment.line_id for assignment in evidence.assignments
    }
    assert evidence.assignment_model["assignment_search"][
        "coherent_alternative_selected"
    ] is True
    assert not any(
        assignment.candidate_id == "mixed:far_nvz"
        for assignment in evidence.assignments
    )
    assert not any(
        "no_consistent_candidate" in reason
        for reason in evidence.rejection_reasons
    )


def test_joint_search_recovers_coherent_court_behind_lower_cost_lookalike() -> None:
    seed = _seed_calibration()
    expected = _expected_lines()
    shifted = {
        line_id: (
            (segment[0][0], segment[0][1] + 6.0),
            (segment[1][0], segment[1][1] + 6.0),
        )
        for line_id, segment in expected.items()
    }
    candidates = [
        _candidate(f"true:{line_id}", segment)
        for line_id, segment in shifted.items()
    ]
    candidates.append(
        _candidate("look:far_nvz", expected["far_nvz"])
    )

    evidence = select_regulation_template_lines(
        candidates,
        expected_lines=expected,
        image_size=(400, 360),
        frame_index=11,
        frame_sha256="b" * 64,
        config=CourtLineHardeningConfig(
            enabled=True,
            provider="legacy_hough",
        ),
        roi_polygon_px=_roi(),
        roi_source="declared_config",
        distortion_state="raw_pinhole",
        seed_calibration=seed,
    )

    assert evidence.status == "accepted"
    selected = {
        assignment.line_id: assignment.candidate_id
        for assignment in evidence.assignments
    }
    assert selected["far_nvz"] == "true:far_nvz"
    assert "look:far_nvz" not in selected.values()
    assert evidence.assignment_model is not None
    assert evidence.assignment_model["joint_p90_px"] < 1e-6
    assert evidence.assignment_model["assignment_search"][
        "coherent_alternative_selected"
    ] is True


def test_two_equally_valid_line_sets_abstain_instead_of_forcing_assignment() -> None:
    expected = _expected_lines()
    ambiguous: list[DetectedCourtLineCandidate] = []
    for line_id, segment in expected.items():
        ambiguous.append(_candidate(f"court_a:{line_id}", segment))
        ambiguous.append(
            _candidate(
                f"court_b:{line_id}",
                ((segment[0][0], segment[0][1] + 1.0), (segment[1][0], segment[1][1] + 1.0)),
            )
        )
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")

    evidence = select_regulation_template_lines(
        ambiguous,
        expected_lines=expected,
        image_size=(400, 360),
        frame_index=3,
        frame_sha256="b" * 64,
        config=config,
        roi_polygon_px=((70.0, 330.0), (330.0, 330.0), (300.0, 50.0), (100.0, 50.0)),
        roi_source="broad_declared_roi",
        distortion_state="raw_pinhole",
        seed_calibration=_seed_calibration(),
    )

    assert evidence.status == "abstained"
    assert len(evidence.assignments) < config.min_assigned_lines
    assert any(
        "global_assignment_ambiguous" in reason
        for reason in evidence.rejection_reasons
    )


def _assignment(
    line_id: str,
    frame_index: int,
    *,
    shift: float,
) -> AssignedCourtLine:
    expected = _expected_lines()[line_id]
    segment = (
        (expected[0][0], expected[0][1] + shift),
        (expected[1][0], expected[1][1] + shift),
    )
    return AssignedCourtLine(
        line_id=line_id,
        candidate_id=f"{line_id}:{frame_index}",
        segment=segment,
        expected_segment=expected,
        score=0.01,
        normal_distance_px=abs(shift),
        angle_delta_deg=0.0,
        overlap_fraction=1.0,
        support_length_px=math.dist(*segment),
        selection_margin=0.5,
    )


def _frame(
    frame_index: int,
    assignments: tuple[AssignedCourtLine, ...],
    *,
    coordinate_space: str = "pixels_raw_native",
) -> FrameCourtLineEvidence:
    return FrameCourtLineEvidence(
        frame_index=frame_index,
        frame_sha256=f"{frame_index:064x}",
        image_size=(400, 360),
        coordinate_space=coordinate_space,
        distortion_state="raw_pinhole",
        provider="legacy_hough",
        raw_candidates=(),
        assignments=assignments,
        status="accepted",
        rejection_reasons=(),
        roi_polygon_px=_roi(),
        roi_source="declared_config",
    )


def test_static_pool_recovers_occlusion_rejects_outlier_and_is_byte_deterministic() -> None:
    line_ids = ("far_baseline", "far_nvz", "left_sideline", "right_sideline")
    frames: list[FrameCourtLineEvidence] = []
    for frame_index, shift in enumerate(
        (0.0, 0.2, -0.1, 0.1, 0.05, -0.05, 0.08, 0.02, 40.0)
    ):
        assignments = tuple(
            _assignment(line_id, frame_index, shift=shift)
            for line_id in line_ids
            if not (line_id == "far_baseline" and frame_index == 0)
        )
        frames.append(_frame(frame_index, assignments))
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")
    raw_before = canonical_json_bytes([frame.as_dict() for frame in frames])

    first = pool_static_semantic_lines(frames, config=config)
    second = pool_static_semantic_lines(list(reversed(frames)), config=config)

    assert first.status == "accepted"
    assert first.canonical_bytes() == second.canonical_bytes()
    assert canonical_json_bytes([frame.as_dict() for frame in frames]) == raw_before
    far = next(line for line in first.lines if line.line_id == "far_baseline")
    assert 0 not in far.contributing_frame_indexes
    assert 8 in far.rejected_frame_indexes
    assert set(far.contributing_frame_indexes).issuperset({1, 2})
    assert abs(far.segment[0][1] - 80.0) < 0.25
    assert first.source_frame_hashes == tuple(
        (index, f"{index:064x}") for index in range(9)
    )


def test_static_pool_rejects_duplicate_indexes_and_coordinate_mismatch() -> None:
    assignments = tuple(
        _assignment(line_id, 0, shift=0.0)
        for line_id in ("far_baseline", "far_nvz", "left_sideline", "right_sideline")
    )
    first = _frame(0, assignments)
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")

    with pytest.raises(ValueError, match="duplicate"):
        pool_static_semantic_lines([first, first], config=config)

    second = _frame(1, assignments, coordinate_space="pixels_undistorted_native")
    with pytest.raises(ValueError, match="coordinate spaces"):
        pool_static_semantic_lines([first, second], config=config)


def test_static_pool_rejects_duplicate_frame_bytes_and_bimodal_fabrication() -> None:
    line_ids = ("far_baseline", "far_nvz", "left_sideline", "right_sideline")
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")
    first = _frame(
        0,
        tuple(_assignment(line_id, 0, shift=0.0) for line_id in line_ids),
    )
    duplicate_pixels = replace(
        _frame(
            1,
            tuple(_assignment(line_id, 1, shift=0.0) for line_id in line_ids),
        ),
        frame_sha256=first.frame_sha256,
    )
    with pytest.raises(ValueError, match="duplicate decoded frame hashes"):
        pool_static_semantic_lines([first, duplicate_pixels], config=config)

    shifts = (-10.0, -10.0, 10.0, 0.0, 10.0, 10.0, -10.0, 0.0)
    bimodal = [
        _frame(
            frame_index,
            tuple(
                _assignment(line_id, frame_index, shift=shift)
                for line_id in line_ids
            ),
        )
        for frame_index, shift in enumerate(shifts)
    ]
    pooled = pool_static_semantic_lines(bimodal, config=config)

    assert pooled.status == "abstained"
    assert not {
        line.line_id for line in pooled.lines
    }.intersection({"far_baseline", "far_nvz"})
    assert any(
        "static_signed_mad_exceeds_max" in reason
        for reason in pooled.rejection_reasons
    )


def test_static_pool_counts_centerline_halves_as_one_longitudinal_family() -> None:
    line_ids = (
        "far_baseline",
        "near_baseline",
        "near_centerline",
        "far_centerline",
    )
    frames = [
        _frame(
            frame_index,
            tuple(
                _assignment(line_id, frame_index, shift=0.0)
                for line_id in line_ids
            ),
        )
        for frame_index in range(8)
    ]

    pooled = pool_static_semantic_lines(
        frames,
        config=CourtLineHardeningConfig(
            enabled=True,
            provider="legacy_hough",
        ),
    )

    assert pooled.status == "abstained"
    assert "pooled_longitudinal_line_family_incomplete" in pooled.rejection_reasons
    assert "pooled_longitudinal_boundary_line_missing" in pooled.rejection_reasons


def test_refinement_rejects_pool_built_under_different_evidence_config() -> None:
    line_ids = ("far_baseline", "far_nvz", "left_sideline", "right_sideline")
    frames = [
        _frame(
            frame_index,
            tuple(
                _assignment(line_id, frame_index, shift=0.0)
                for line_id in line_ids
            ),
        )
        for frame_index in range(8)
    ]
    config = CourtLineHardeningConfig(enabled=True, provider="legacy_hough")
    pooled = pool_static_semantic_lines(frames, config=config)

    with pytest.raises(ValueError, match="config hash"):
        refine_pooled_homography(
            _seed_calibration(),
            replace(pooled, config_sha256="0" * 64),
            config=config,
        )


def test_optimizer_can_exercise_line_over_point_weighting_on_independent_geometry() -> None:
    template = get_court_template("pickleball")
    true_h = homography_from_planar_points(
        template.corners_m,
        [[80.0, 420.0], [560.0, 430.0], [440.0, 70.0], [170.0, 60.0]],
    )
    noise = (
        (8.0, -5.0),
        (-7.0, 6.0),
        (6.0, 7.0),
        (-8.0, -6.0),
        (5.0, -8.0),
        (-6.0, 8.0),
        (7.0, 4.0),
        (-5.0, -7.0),
        (8.0, 5.0),
        (-7.0, -4.0),
        (6.0, -6.0),
        (-5.0, 5.0),
    )
    priors: dict[str, dict[str, object]] = {}
    noisy_points: list[list[float]] = []
    world_points: list[tuple[float, float, float]] = []
    for index, point in enumerate(FLOOR_KEYPOINTS):
        projected = project_planar_points(true_h, [point.world_xyz_m])[0]
        dx, dy = noise[index]
        observed = [projected[0] + dx, projected[1] + dy]
        priors[point.name] = {"xy": observed, "confidence": 1.0}
        noisy_points.append(observed)
        world_points.append(point.world_xyz_m)
    seed_h = homography_from_planar_points(world_points, noisy_points)
    semantic_lines = {
        line_id: {
            "optimize": [project_planar_points(true_h, endpoints)],
            "heldout": [project_planar_points(true_h, endpoints)],
            "confidence": 1.0,
        }
        for line_id, endpoints in template.line_segments_m.items()
    }

    outputs = {}
    for line_weight in (0.60, 0.80):
        outputs[line_weight] = refine_homography_with_lines(
            seed_h,
            semantic_lines,
            None,
            priors,
            coordinate_space="pixels_raw_native",
            config=RefinementConfig(
                line_weight=line_weight,
                point_weight=1.0 - line_weight,
                stability_guard_enabled=False,
                max_corner_shift_px=100.0,
                heldout_p90_tolerance_px=100.0,
                heldout_max_line_family_p90_regression_px=100.0,
            ),
        )
        assert outputs[line_weight]["accepted"] is True
        assert outputs[line_weight]["telemetry"]["net_top_point_count_in_planar_fit"] == 0

    independent_world = [point.world_xyz_m for point in FLOOR_KEYPOINTS]
    truth = project_planar_points(true_h, independent_world)

    def mean_error(output: dict[str, object]) -> float:
        predicted = project_planar_points(output["homography_image_from_court"], independent_world)
        return float(np.mean([math.dist(first, second) for first, second in zip(predicted, truth, strict=True)]))

    assert mean_error(outputs[0.80]) < mean_error(outputs[0.60])


def test_weight_ablation_reuses_identical_evidence_config() -> None:
    selected = CourtLineHardeningConfig(enabled=True)
    aggressive = replace(selected, line_weight=0.80, point_weight=0.20)

    assert selected.line_weight == 0.60
    assert selected.evidence_config_dict() == aggressive.evidence_config_dict()
    assert selected.refinement_config_dict() != aggressive.refinement_config_dict()


def test_enabled_seed_guided_rebuild_is_byte_deterministic_and_preview_only() -> None:
    seed = _seed_with_floor_points(source="synthetic_automatic_detector")
    frames = _painted_static_frames(seed)
    config = CourtLineHardeningConfig(enabled=True)

    first = run_court_line_hardening(frames, seed, config=config)
    second = run_court_line_hardening(
        list(reversed(frames)),
        seed,
        config=config,
    )
    enabled_seam = maybe_apply_court_line_hardening(
        frames,
        seed,
        config=config,
    )

    assert isinstance(first, CourtLineHardeningResult)
    assert isinstance(enabled_seam, CourtLineHardeningResult)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.raw_evidence_artifact()["verified"] is False
    assert first.pooled_evidence.as_dict()["authority"] == "preview"
    assert len(first.raw_frame_evidence) == 8
    assert len(first.pooled_evidence.source_frame_evidence_hashes) == 8
    assert first.pooled_evidence.status == "accepted"
    assert len(first.pooled_evidence.lines) == 8
    assert all(
        frame.provider == "seed_guided_paired_edges"
        for frame in first.raw_frame_evidence
    )
    assert all(
        len(frame.template_samples)
        == frame.detector_metadata["raw_sample_count"]
        for frame in first.raw_frame_evidence
        if frame.detector_metadata is not None
    )
    metadata = first.raw_frame_evidence[0].detector_metadata
    assert metadata is not None
    with pytest.raises(TypeError):
        metadata["raw_sample_count"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        metadata["per_line_detected_sample_count"]["near_baseline"] = 0  # type: ignore[index]


def test_reviewed_point_priors_are_scorer_only_and_never_enter_fit() -> None:
    seed = _seed_with_floor_points(source="metric_15pt_reviewed")
    frames = _painted_static_frames(seed, vertical_shift_px=2.0)

    result = run_court_line_hardening(
        frames,
        seed,
        config=CourtLineHardeningConfig(enabled=True),
    )

    assert result.pooled_evidence.status == "accepted"
    assert result.refinement["accepted"] is False
    assert result.refinement["selection_reason"] == (
        "automatic_point_prior_provenance_required"
    )
    assert any(
        "point_prior_source_is_scorer_or_manual" in reason
        for reason in result.refinement["reject_reasons"]
    )
    assert result.candidate_calibration == seed


def test_automatic_point_prior_fit_requires_bound_artifact_provenance() -> None:
    seed = _seed_with_floor_points(source="synthetic_automatic_detector")
    seed.pop("point_evidence_provenance")

    provenance = robustness._point_prior_provenance(seed)

    assert provenance["fit_eligible"] is False
    assert "automatic_point_evidence_provenance_missing" in provenance["reasons"]

    mismatched = _seed_with_floor_points(source="synthetic_automatic_detector")
    mismatched["image_pts"][0][0] += 1.0
    mismatch_provenance = robustness._point_prior_provenance(mismatched)
    assert mismatch_provenance["fit_eligible"] is False
    assert (
        "point_evidence_correspondence_hash_mismatch"
        in mismatch_provenance["reasons"]
    )

    reordered = _seed_with_floor_points(
        source="synthetic_automatic_detector"
    )
    reordered["world_pts"][0], reordered["world_pts"][1] = (
        reordered["world_pts"][1],
        reordered["world_pts"][0],
    )
    reordered["point_evidence_provenance"]["correspondences_sha256"] = (
        hashlib.sha256(
            canonical_json_bytes(
                {
                    "image_pts": reordered["image_pts"],
                    "world_pts": reordered["world_pts"],
                }
            )
        ).hexdigest()
    )
    reordered_provenance = robustness._point_prior_provenance(reordered)
    assert reordered_provenance["fit_eligible"] is False
    assert any(
        reason.startswith("point_prior_world_order_or_value_mismatch:")
        for reason in reordered_provenance["reasons"]
    )


def test_line_only_arm_does_not_use_reviewed_points_for_hidden_initialization() -> None:
    seed = _seed_with_floor_points(source="metric_15pt_reviewed")
    frames = _painted_static_frames(seed, vertical_shift_px=2.0)

    result = run_court_line_hardening(
        frames,
        seed,
        config=CourtLineHardeningConfig(
            enabled=True,
            line_weight=1.0,
            point_weight=0.0,
        ),
    )

    assert result.pooled_evidence.status == "accepted"
    assert result.refinement["telemetry"]["provided_floor_point_count"] == 0
    assert result.refinement["telemetry"]["optimizer_point_count"] == 0
    assert result.refinement["fit_arm_role"] == (
        "diagnostic_line_only_no_point_stability"
    )
    assert result.refinement["promotion_eligible"] is False
    assert result.refinement["input_provenance"]["point_priors_sha256"] == (
        hashlib.sha256(canonical_json_bytes({})).hexdigest()
    )


def test_accepted_homography_preview_invalidates_stale_pose_fields() -> None:
    seed = _seed_with_floor_points(source="synthetic_automatic_detector")
    seed["provenance"] = {"implementation": "stale_seed_solver"}
    seed["coordinate_contract"] = {"homography_sha256": "stale"}
    seed["trust_band"] = "verified"
    seed["reprojection_error_px"] = {"median_px": 0.0}
    seed["per_keypoint_residual_px"] = [0.0] * 15
    frames = _painted_static_frames(seed, vertical_shift_px=0.2)

    result = run_court_line_hardening(
        frames,
        seed,
        config=CourtLineHardeningConfig(enabled=True),
    )

    assert result.pooled_evidence.status == "accepted"
    assert result.refinement["accepted"] is True
    assert result.candidate_calibration["authority"] == "preview"
    assert result.candidate_calibration["verified"] is False
    assert result.candidate_calibration["geometry_scope"] == (
        "planar_homography_only"
    )
    assert "extrinsics" not in result.candidate_calibration
    assert "extrinsics" in result.candidate_calibration["invalidated_seed_fields"]
    assert "coordinate_contract" not in result.candidate_calibration
    assert "coordinate_contract" in (
        result.candidate_calibration["invalidated_seed_fields"]
    )
    assert "provenance" in result.candidate_calibration["invalidated_seed_fields"]
    assert "trust_band" in result.candidate_calibration["invalidated_seed_fields"]
    assert "reprojection_error_px" not in result.candidate_calibration
    assert "per_keypoint_residual_px" not in result.candidate_calibration
    assert "reprojection_error_px" in (
        result.candidate_calibration["invalidated_seed_fields"]
    )
    assert len(
        result.candidate_calibration["floor_reprojection_keypoint_names"]
    ) == 12
    assert len(
        result.candidate_calibration["per_floor_keypoint_residual_px"]
    ) == 12
    assert all(
        "net_" not in name
        for name in result.candidate_calibration[
            "floor_reprojection_keypoint_names"
        ]
    )
    assert result.candidate_calibration["trust_band"] == "preview"
    assert result.candidate_calibration["provenance"]["implementation"] == (
        robustness.REFINEMENT_IMPLEMENTATION
    )
    assert result.refinement["telemetry"]["selected_line_search_alpha"] == 1.0
    assert result.candidate_calibration["source"] == (
        "court_line_hardening_preview"
    )


def test_optimizer_damped_selection_is_not_an_accepted_ablation() -> None:
    seed = _seed_with_floor_points(source="synthetic_automatic_detector")
    frames = _painted_static_frames(seed, vertical_shift_px=2.0)

    result = run_court_line_hardening(
        frames,
        seed,
        config=CourtLineHardeningConfig(enabled=True),
    )

    assert result.pooled_evidence.status == "accepted"
    assert result.refinement["accepted"] is False
    assert result.refinement["selection"] == "seed"
    assert result.refinement["selection_reason"] == (
        "optimizer_selected_non_preregistered_damping"
    )
    assert (
        result.refinement["rejected_optimizer_diagnostics"]["telemetry"][
            "selected_line_search_alpha"
        ]
        != 1.0
    )
    assert result.refinement["telemetry"]["selected_line_search_alpha"] == 0.0
    assert result.refinement["scores_after"] == (
        result.refinement["scores_before"]
    )
    assert result.refinement["covariance_inflation_required"] is True
    assert result.candidate_calibration == seed


def test_pool_and_refinement_are_bound_to_seed_and_template_projection() -> None:
    seed = _seed_with_floor_points(source="synthetic_automatic_detector")
    frames = _painted_static_frames(seed)
    config = CourtLineHardeningConfig(enabled=True)
    result = run_court_line_hardening(frames, seed, config=config)

    assert result.pooled_evidence.seed_calibration_sha256 == (
        result.seed_calibration_sha256
    )
    assert result.pooled_evidence.template_projection_sha256 is not None
    changed_seed = dict(seed)
    changed_seed["source"] = "different_automatic_detector"
    with pytest.raises(ValueError, match="seed hash"):
        refine_pooled_homography(
            changed_seed,
            result.pooled_evidence,
            config=config,
        )

    mismatched_frame = replace(
        result.raw_frame_evidence[0],
        seed_calibration_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="inconsistent seed calibration hashes"):
        pool_static_semantic_lines(
            [mismatched_frame, *result.raw_frame_evidence[1:]],
            config=config,
        )

    frame = result.raw_frame_evidence[1]
    sample = frame.template_samples[0]
    changed_sample = replace(
        sample,
        seed_xy=(sample.seed_xy[0] + 1.0, sample.seed_xy[1]),
    )
    changed_frame = replace(
        frame,
        template_samples=(changed_sample, *frame.template_samples[1:]),
    )
    with pytest.raises(ValueError, match="static sample basis changed"):
        pool_static_semantic_lines(
            [
                result.raw_frame_evidence[0],
                changed_frame,
                *result.raw_frame_evidence[2:],
            ],
            config=config,
        )

    mismatched_assignment = replace(
        frame.assignments[0],
        candidate_id="seed_guided:wrong_template_line",
    )
    identity_mismatch = replace(
        frame,
        assignments=(
            mismatched_assignment,
            *frame.assignments[1:],
        ),
    )
    with pytest.raises(
        ValueError,
        match="assignment candidate identity does not match",
    ):
        pool_static_semantic_lines(
            [
                result.raw_frame_evidence[0],
                identity_mismatch,
                *result.raw_frame_evidence[2:],
            ],
            config=config,
        )


def test_profile_sampling_rejects_step_larger_than_scan_radius() -> None:
    with pytest.raises(ValueError, match="profile scan radius"):
        CourtLineHardeningConfig(
            enabled=True,
            profile_scan_radius_px=1.0,
            profile_step_px=3.0,
        ).validate()


def test_paired_edge_profile_trims_remote_border_nan_but_requires_context() -> None:
    config = CourtLineHardeningConfig(enabled=True)
    offsets = np.arange(
        -config.profile_scan_radius_px,
        config.profile_scan_radius_px + config.profile_step_px * 0.5,
        config.profile_step_px,
        dtype=np.float64,
    )
    profile = np.full(offsets.shape, 35.0, dtype=np.float64)
    profile[np.abs(offsets - 0.25) <= 1.5] = 235.0

    complete = robustness._detect_paired_edge_profile(
        profile,
        offsets,
        expected_width=3.0,
        config=config,
    )
    remote_nan = profile.copy()
    remote_nan[offsets < -10.0] = np.nan
    trimmed = robustness._detect_paired_edge_profile(
        remote_nan,
        offsets,
        expected_width=3.0,
        config=config,
    )
    one_sided = profile.copy()
    one_sided[offsets < -2.0] = np.nan

    assert complete is not None
    assert trimmed is not None
    assert trimmed["signed_offset_px"] == pytest.approx(
        complete["signed_offset_px"],
        abs=1e-9,
    )
    assert (
        robustness._detect_paired_edge_profile(
            one_sided,
            offsets,
            expected_width=3.0,
            config=config,
        )
        is None
    )


def test_shadow_arm_requires_measured_failing_stratum() -> None:
    with pytest.raises(ValueError, match="measured failing stratum artifact"):
        CourtLineHardeningConfig(
            enabled=True,
            preprocessing="shadow_compensated",
        ).validate()

    with pytest.raises(ValueError, match="measured failing stratum artifact"):
        CourtLineHardeningConfig(
            enabled=True,
            provider="hybrid_paired_hough",
            preprocessing="shadow_compensated",
        ).validate()

    with pytest.raises(ValueError, match="source-image hash"):
        detect_court_line_candidates_from_image(
            np.zeros((80, 120, 3), dtype=np.uint8),
            provider="classical_paired_edges",
            preprocessing="shadow_compensated",
        )
