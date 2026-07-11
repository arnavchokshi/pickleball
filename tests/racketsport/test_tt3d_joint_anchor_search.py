from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from threed.racketsport.ball_joint_anchor_search import (
    AnchorSearchRefusal,
    BoundedPlane,
    CameraModel,
    ImageObservation,
    JointAnchorSearchConfig,
    NetConstraint,
    RefusalCode,
    intersect_ray_plane,
    pixel_ray_world,
    project_world,
    search_joint_anchor_candidates,
)


TRUE_BOUNCE_TIME_S = 0.5
TRUE_BOUNCE_XYZ = (0.0, 2.0, 0.0371)


def _camera() -> CameraModel:
    position = np.asarray((0.0, -10.0, 5.0), dtype=float)
    target = np.asarray((0.0, 2.0, 1.0), dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    translation = -rotation @ position
    return CameraModel(
        fx=900.0,
        fy=900.0,
        cx=640.0,
        cy=360.0,
        rotation_world_to_camera=tuple(tuple(float(value) for value in row) for row in rotation),
        translation_world_to_camera=tuple(float(value) for value in translation),
    )


def _court_plane() -> BoundedPlane:
    # Surface plane.  Search raises the ball-center anchor by ball_radius_m.
    return BoundedPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        u_bounds_m=(-8.0, 8.0),
        v_bounds_m=(-8.0, 8.0),
    )


def _net() -> NetConstraint:
    return NetConstraint(point=(0.0, 1.0, 0.0), normal=(0.0, 1.0, 0.0), height_m=0.8636)


def _config(**overrides: object) -> JointAnchorSearchConfig:
    base = JointAnchorSearchConfig(
        max_time_seeds=15,
        starts_per_seed=1,
        max_ranked_candidates=5,
        max_nfev=1200,
    )
    return replace(base, **overrides)


def _synthetic_observations(*, occluded: bool = False, outlier: bool = False) -> tuple[ImageObservation, ...]:
    camera = _camera()
    bounce = np.asarray(TRUE_BOUNCE_XYZ, dtype=float)
    incoming = np.asarray((2.0, 4.0, -5.0), dtype=float)
    outgoing = np.asarray((1.7, 3.4, 3.25), dtype=float)
    gravity = np.asarray((0.0, 0.0, -9.80665), dtype=float)
    observations: list[ImageObservation] = []
    for frame, pts_s in enumerate(np.linspace(0.1, 0.9, 25)):
        if occluded and 0.4 < pts_s < 0.6:
            continue
        dt = float(pts_s) - TRUE_BOUNCE_TIME_S
        velocity = incoming if dt <= 0.0 else outgoing
        world = bounce + velocity * dt + 0.5 * gravity * dt * dt
        (u, v), depth = project_world(camera, world)
        assert depth > 0.0
        if outlier and frame == 5:
            u += 140.0
            v -= 90.0
        observations.append(
            ImageObservation(
                frame=frame,
                pts_s=float(pts_s),
                u=float(u),
                v=float(v),
                confidence=0.9,
            )
        )
    return tuple(observations)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def test_ray_plane_geometry_recovers_ball_center_anchor() -> None:
    camera = _camera()
    pixel, _ = project_world(camera, TRUE_BOUNCE_XYZ)
    origin, direction = pixel_ray_world(camera, pixel)

    intersection = intersect_ray_plane(origin, direction, (0.0, 0.0, 0.0371), (0.0, 0.0, 1.0))

    assert intersection == pytest.approx(TRUE_BOUNCE_XYZ, abs=1e-9)


def test_synthetic_joint_search_recovers_known_bounce_and_emits_solver_contract() -> None:
    result = search_joint_anchor_candidates(
        _synthetic_observations(outlier=True),
        _camera(),
        _court_plane(),
        net=_net(),
        config=_config(starts_per_seed=2),
        seed=17,
    )

    best = result.candidates[0]
    assert best.bounce_time_s == pytest.approx(TRUE_BOUNCE_TIME_S, abs=0.025)
    assert _distance(best.bounce_world_xyz, TRUE_BOUNCE_XYZ) < 0.15
    assert best.net_constraint_satisfied is True
    assert best.reprojection_rmse_px < 40.0  # Includes one deliberate 166px hard outlier.
    assert _distance(best.bounce_world_xyz, (best.bounce_world_xyz[0], best.bounce_world_xyz[1], 0.0371)) < 1e-9
    assert 0.25 <= best.restitution <= 0.95
    assert 0.55 <= best.tangential_retention <= 1.0
    assert all(abs(value) <= 35.0 for value in best.incoming_velocity_mps)
    assert [anchor["kind"] for anchor in best.anchor_events] == ["contact", "bounce", "contact"]
    for anchor in best.anchor_events:
        assert set(("anchor_id", "kind", "t", "frame", "world_xyz", "sigma_m", "status", "immovable")) <= set(anchor)
        assert anchor["status"] == "candidate_hypothesis"
        assert anchor["immovable"] is False
        assert anchor["details"]["measured"] is False
    payload = result.to_json()
    assert payload["status"] == "candidates_only"
    assert payload["policy"] == {
        "candidate_only": True,
        "marks_measured": False,
        "alters_trust_bands": False,
        "alters_defaults": False,
        "protected_labels_read": False,
    }


def test_occlusion_gap_still_recovers_ranked_bounce_hypothesis() -> None:
    observations = _synthetic_observations(occluded=True)

    result = search_joint_anchor_candidates(
        observations,
        _camera(),
        _court_plane(),
        config=_config(starts_per_seed=2),
        seed=23,
    )

    best = result.candidates[0]
    assert not any(0.4 < item.pts_s < 0.6 for item in observations)
    assert best.bounce_time_s == pytest.approx(TRUE_BOUNCE_TIME_S, abs=0.035)
    assert _distance(best.bounce_world_xyz, TRUE_BOUNCE_XYZ) < 0.20
    assert best.reprojection_rmse_px < 2.0


def test_insufficient_observations_refuses_with_typed_code() -> None:
    with pytest.raises(AnchorSearchRefusal) as caught:
        search_joint_anchor_candidates(
            _synthetic_observations()[:5],
            _camera(),
            _court_plane(),
            config=_config(),
        )

    assert caught.value.code is RefusalCode.INSUFFICIENT_OBSERVATIONS
    assert caught.value.to_json()["status"] == "refused"
    assert caught.value.to_json()["details"] == {"required": 8, "actual": 5}


def test_degenerate_ray_plane_refuses_instead_of_emitting_candidate() -> None:
    camera = CameraModel(
        fx=900.0,
        fy=900.0,
        cx=640.0,
        cy=360.0,
        rotation_world_to_camera=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_world_to_camera=(0.0, 0.0, 0.0),
    )
    origin, direction = pixel_ray_world(camera, (640.0, 360.0))

    with pytest.raises(AnchorSearchRefusal) as caught:
        intersect_ray_plane(origin, direction, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert caught.value.code is RefusalCode.DEGENERATE_RAY_PLANE


def test_search_refuses_degenerate_court_plane_with_typed_code() -> None:
    degenerate = replace(_court_plane(), normal=(0.0, 0.0, 0.0))

    with pytest.raises(AnchorSearchRefusal) as caught:
        search_joint_anchor_candidates(
            _synthetic_observations(),
            _camera(),
            degenerate,
            config=_config(),
        )

    assert caught.value.code is RefusalCode.DEGENERATE_COURT_PLANE


def test_search_is_byte_deterministic_for_same_seed_and_preserves_inputs() -> None:
    observations = _synthetic_observations(occluded=True)
    before = tuple(observations)
    kwargs = {
        "net": _net(),
        "config": _config(starts_per_seed=2),
        "seed": 101,
    }

    first = search_joint_anchor_candidates(observations, _camera(), _court_plane(), **kwargs)
    second = search_joint_anchor_candidates(observations, _camera(), _court_plane(), **kwargs)

    assert tuple(observations) == before
    assert json.dumps(first.to_json(), sort_keys=True, separators=(",", ":")) == json.dumps(
        second.to_json(), sort_keys=True, separators=(",", ":")
    )
    provenance = first.candidates[0].provenance
    assert provenance["observation_digest_sha256"] == first.observation_digest_sha256
    assert provenance["camera_digest_sha256"] == first.camera_digest_sha256
    assert len(provenance["observation_refs"]) == len(observations)
    assert provenance["geometry"]["ball_center_court_plane"]["point"][2] == pytest.approx(0.0371)
    assert provenance["geometry"]["net_constraint"]["height_m"] == pytest.approx(0.8636)
    assert provenance["policy"]["candidate_only"] is True
    assert provenance["policy"]["marks_measured"] is False
