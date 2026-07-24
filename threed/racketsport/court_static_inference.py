"""Static multi-frame wrapper for confidence-aware structured court inference."""

from __future__ import annotations

from dataclasses import replace
import math
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.court_model_infer import (
    _best_court_contract,
    _infer_court_model_with_loaded_model,
    build_court_model_from_checkpoint,
    load_court_model_checkpoint,
)
from threed.racketsport.court_static_lock import (
    CourtLockArtifact,
    CourtLockCameraParameters,
    CourtLockDistortion,
    CourtLockEvidenceSummary,
    CourtLockResidual,
    FrameCourtTransform,
    StaticMotionDiagnostics,
    StaticCourtObservation,
    StaticFrameEvidence,
    diagnose_static_camera,
    pool_static_frame_evidence,
    write_court_lock,
)
from threed.racketsport.court_camera_geometry import PinholeIntrinsics
from threed.racketsport.court_net_stage import project_regulation_net_top
from threed.racketsport.court_structured_evidence import build_court_evidence_bundle
from threed.racketsport.court_structured_solver import solve_best_floor_court


def infer_static_court_model(
    frames_bgr: Sequence[Any],
    checkpoint_path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
    camera_profile: Mapping[str, Any] | None = None,
    previous_lock: Mapping[str, Any] | None = None,
    court_lock_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Pool up to eight fixed-camera frames into one review-only court lock.

    Camera motion never gets averaged into a lock: moving or ambiguous input
    returns the clearest per-frame result and explicit diagnostics instead.
    """

    import cv2

    if not frames_bgr:
        raise ValueError("frames_bgr must be non-empty")
    if frame_indices is None:
        indexes = tuple(range(len(frames_bgr)))
    else:
        indexes = tuple(int(value) for value in frame_indices)
        if len(indexes) != len(frames_bgr) or len(set(indexes)) != len(indexes):
            raise ValueError("frame_indices must be unique and match frames_bgr")
    first_shape = getattr(frames_bgr[0], "shape", None)
    if first_shape is None or len(first_shape) != 3:
        raise ValueError("frames_bgr entries must be HxWx3 images")
    source_height, source_width = int(first_shape[0]), int(first_shape[1])
    if any(getattr(frame, "shape", None) != first_shape for frame in frames_bgr):
        raise ValueError("all static frames must share one image shape")

    candidate_frame_count = len(frames_bgr)
    segment_diagnostics = _segment_camera_states(frames_bgr, frame_indices=indexes)
    selected_positions = tuple(int(value) for value in segment_diagnostics["selected_positions"])
    inference_frames = tuple(frames_bgr[position] for position in selected_positions)
    inference_indexes = tuple(indexes[position] for position in selected_positions)

    payload = load_court_model_checkpoint(checkpoint_path, device=device)
    model, keypoint_names, model_size = build_court_model_from_checkpoint(payload, device=device)
    frame_results: list[dict[str, Any]] = []
    frame_evidence: list[StaticFrameEvidence] = []
    transforms: list[FrameCourtTransform] = []
    for frame_index, frame in zip(inference_indexes, inference_frames, strict=True):
        result = _infer_court_model_with_loaded_model(
            frame,
            model=model,
            keypoint_names=keypoint_names,
            model_size=model_size,
            device=device,
        )
        frame_results.append(result)
        observations = tuple(_static_observations(result))
        score_components = result["best_court"].get("score_components") or {}
        visibility_values = [float(value) for value in result["keypoints_vis"].values()]
        visible_fraction = sum(value >= 0.5 for value in visibility_values) / max(
            len(visibility_values), 1
        )
        sharpness_raw = float(cv2.Laplacian(frame, cv2.CV_64F).var())
        sharpness = sharpness_raw / (sharpness_raw + 100.0)
        frame_evidence.append(
            StaticFrameEvidence(
                frame_index=frame_index,
                line_support=float(score_components.get("line_alignment", 0.0)),
                surface_support=float(score_components.get("surface_overlap", 0.0)),
                visible_fraction=float(visible_fraction),
                sharpness=float(min(max(sharpness, 0.0), 1.0)),
                occlusion_fraction=float(1.0 - visible_fraction),
                observations=observations,
            )
        )
        homography = result["best_court"].get("homography_image_from_court")
        if homography is not None:
            covariance = result["best_court"].get("transform_covariance")
            transforms.append(
                FrameCourtTransform(
                    frame_index=frame_index,
                    homography_image_from_court=tuple(tuple(float(v) for v in row) for row in homography),
                    confidence=float(result["best_court"].get("court_confidence", 0.0)),
                    transform_covariance=(
                        None
                        if covariance is None
                        else tuple(tuple(float(v) for v in row) for row in covariance)
                    ),
                )
            )

    pooled = pool_static_frame_evidence(frame_evidence, max_frames=8)
    transform_motion = diagnose_static_camera(transforms)
    appearance_motion = _appearance_static_diagnostics(
        inference_frames,
        frame_indices=inference_indexes,
    )
    motion = _resolve_static_motion(
        transform_motion,
        appearance_motion,
        segment_diagnostics=segment_diagnostics,
    )
    selected_results = [
        frame_results[inference_indexes.index(index)] for index in pooled.selected_frame_indices
    ]
    fallback_selected_frame_index: int | None = None
    if motion.static_lock_usable and selected_results:
        line_distances = _pooled_line_distances(selected_results)
        surface_probability = np.mean(
            [np.asarray(result["surface_mask"] == 2, dtype=np.float64) for result in selected_results],
            axis=0,
        )
        temporal_support = (
            sum(item.temporal_support for item in pooled.observations) / len(pooled.observations)
            if pooled.observations
            else 0.0
        )
        homography_candidates: list[Mapping[str, Any]] = []
        for result in selected_results:
            candidate = result["best_court"].get("homography_image_from_court")
            if candidate is not None:
                homography_candidates.append(
                    {
                        "source": "clearest_frame_point_and_line",
                        "homography": candidate,
                    }
                )
        if previous_lock is not None and previous_lock.get("homography_image_from_court") is not None:
            homography_candidates.append(
                {
                    "source": "previous_static_lock",
                    "homography": previous_lock["homography_image_from_court"],
                }
            )
        homography_candidates.append(
            {
                "source": "camera_profile_prior",
                "homography": _profile_prior_homography(source_width, source_height),
            }
        )
        bundle = build_court_evidence_bundle(
            pooled.to_solver_observations(),
            image_size=(source_width, source_height),
            line_distance_maps=line_distances,
            surface_probability=surface_probability,
            homography_candidates=homography_candidates,
            temporal_support=temporal_support,
            camera_metadata=camera_profile,
        )
        structured = solve_best_floor_court(bundle)
        best_court = _best_court_contract(
            structured,
            point_confidence_calibrator=getattr(model, "_point_confidence_calibrator", None),
            court_confidence_calibrator=getattr(model, "_court_confidence_calibrator", None),
            supported_view_probability=_mean_optional(
                [result["best_court"].get("supported_view_probability") for result in selected_results]
            ),
        )
        source = _lock_source(
            str(best_court.get("source") or ""),
            default="multi_frame_point_and_line",
        )

        # A pooled solve is preferred, but the documented fallback chain must
        # not turn a stable, supported clip into a false rejection when one of
        # its clear frames independently satisfies the exact same lock screen.
        # Camera-transition clips never reach this fallback because the motion
        # and segment checks remain part of eligibility.
        pooled_eligible, _ = _lock_eligibility(
            best_court,
            source=source,
            motion=motion,
            segment_diagnostics=segment_diagnostics,
        )
        if not pooled_eligible:
            ranked_frames = sorted(
                zip(frame_evidence, frame_results, strict=True),
                key=lambda pair: (-pair[0].quality_score, pair[0].frame_index),
            )
            for evidence, result in ranked_frames:
                candidate = result["best_court"]
                candidate_source = "clearest_frame_point_and_line"
                candidate_eligible, _ = _lock_eligibility(
                    candidate,
                    source=candidate_source,
                    motion=motion,
                    segment_diagnostics=segment_diagnostics,
                )
                if candidate_eligible:
                    best_court = candidate
                    source = candidate_source
                    fallback_selected_frame_index = int(evidence.frame_index)
                    break
    else:
        clearest = min(frame_evidence, key=lambda item: (-item.quality_score, item.frame_index))
        best_court = frame_results[inference_indexes.index(clearest.frame_index)]["best_court"]
        structured = None
        source = "clearest_frame_point_and_line"
        if best_court.get("homography_image_from_court") is None:
            prior = solve_best_floor_court(
                {},
                prior_homography=_profile_prior_homography(source_width, source_height),
            )
            best_court = _best_court_contract(
                prior,
                point_confidence_calibrator=getattr(model, "_point_confidence_calibrator", None),
                court_confidence_calibrator=getattr(model, "_court_confidence_calibrator", None),
                supported_view_probability=best_court.get("supported_view_probability"),
            )
            source = "camera_profile_prior"

    if best_court.get("homography_image_from_court") is not None:
        best_court = dict(best_court)
        intrinsics, distortion, profile_source = _camera_profile(
            camera_profile,
            image_size=(source_width, source_height),
        )
        fitted_distortion = best_court.get("distortion")
        if (
            isinstance(fitted_distortion, Mapping)
            and isinstance(fitted_distortion.get("k1"), (int, float))
        ):
            distortion = {
                "k1": float(fitted_distortion["k1"]),
                "k1_variance": float(fitted_distortion.get("k1_variance") or 0.0),
                "source": str(
                    fitted_distortion.get("source") or "joint_point_camera_refinement"
                ),
            }
        try:
            camera_parameters = best_court.get("camera_parameters")
            pose_homography = (
                camera_parameters.get("homography_undistorted_from_court")
                if isinstance(camera_parameters, Mapping)
                else None
            )
            net_stage = project_regulation_net_top(
                pose_homography or best_court["homography_image_from_court"],
                intrinsics,
                k1=distortion["k1"],
                transform_covariance=best_court.get("transform_covariance"),
                k1_variance=distortion["k1_variance"],
                floor_court_confidence=float(best_court.get("court_confidence") or 0.0),
            )
        except ValueError as exc:
            best_court["net_stage"] = {
                "status": "pose_underconstrained",
                "reason": str(exc),
                "measurement_valid": False,
                "authority_state": "review_only",
            }
        else:
            best_court["net_stage"] = net_stage
            best_court["net_keypoints_xy"] = net_stage["keypoints_xy"]
            best_court["net_point_confidence"] = net_stage["point_confidence"]
            best_court["camera_parameters"] = {
                **(dict(camera_parameters) if isinstance(camera_parameters, Mapping) else {}),
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                },
                "pose": net_stage["camera_pose"],
                "source": profile_source,
            }
            best_court["distortion"] = {
                "model": "radial_k1",
                "k1": distortion["k1"],
                "k1_variance": distortion["k1_variance"],
                "source": distortion["source"],
            }

    source = _lock_source(str(source), default="multi_frame_point_and_line")
    lock_eligible, lock_decision_reasons = _lock_eligibility(
        best_court,
        source=source,
        motion=motion,
        segment_diagnostics=segment_diagnostics,
    )
    best_court = dict(best_court)
    best_court["lock_eligible"] = lock_eligible
    best_court["lock_decision_reasons"] = list(lock_decision_reasons)

    lock_payload = None
    if court_lock_path is not None:
        stale_path = Path(court_lock_path)
        if stale_path.is_file():
            stale_path.unlink()
    if lock_eligible and best_court.get("homography_image_from_court") is not None:
        lock = _court_lock_artifact(
            best_court,
            source=source,
            pooled=pooled,
            motion=motion,
            image_size=(source_width, source_height),
            camera_profile=camera_profile,
            checkpoint_sha256=_checkpoint_sha(payload, checkpoint_path=Path(checkpoint_path)),
            lock_decision_reasons=lock_decision_reasons,
            segment_diagnostics=segment_diagnostics,
        )
        lock_payload = lock.to_dict()
        if court_lock_path is not None:
            write_court_lock(lock, court_lock_path)

    return {
        "schema_version": 1,
        "best_court": best_court,
        "court_lock": lock_payload,
        "static_motion": motion.to_dict(),
        "appearance_motion": appearance_motion,
        "candidate_frame_count": candidate_frame_count,
        "selected_frame_indices": list(pooled.selected_frame_indices),
        "segment_diagnostics": segment_diagnostics,
        "pooled_observation_count": len(pooled.observations),
        "source": source,
        "fallback_selected_frame_index": fallback_selected_frame_index,
        "frame_hypotheses": [
            {"frame_index": int(frame_index), "best_court": result["best_court"]}
            for frame_index, result in zip(inference_indexes, frame_results, strict=True)
        ],
        "frame_best_courts": [result["best_court"] for result in frame_results],
        "structured_pool_diagnostics": None if structured is None else structured.get("diagnostics"),
    }


def _segment_camera_states(
    frames_bgr: Sequence[Any],
    *,
    frame_indices: Sequence[int],
    max_selected_frames: int = 8,
    max_width: int = 640,
) -> dict[str, Any]:
    """Split sampled video frames at confident cuts, pans, tilts, or zooms.

    The estimator uses forward/backward-consistent sparse background tracks and
    RANSAC.  Independently moving people are therefore excluded as transform
    outliers before a pair can split the camera state.
    """

    import cv2

    if len(frames_bgr) != len(frame_indices) or not frames_bgr:
        raise ValueError("segment inputs must be non-empty and aligned")
    if len(frames_bgr) == 1:
        return {
            "status": "single_frame",
            "whole_clip_static_eligible": True,
            "segments": [{"segment_id": 0, "frame_indices": [int(frame_indices[0])]}],
            "pair_transforms": [],
            "transition_frame_indices": [],
            "selected_segment_id": 0,
            "selected_segment_range": [int(frame_indices[0]), int(frame_indices[0])],
            "selected_positions": [0],
            "selection_method": "single_frame",
        }

    height, width = frames_bgr[0].shape[:2]
    scale = min(1.0, float(max_width) / max(float(width), 1.0))

    def gray(frame: Any) -> np.ndarray:
        resized = frame if scale == 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    grays = [gray(frame) for frame in frames_bgr]
    pair_rows: list[dict[str, Any]] = []
    transition_positions: list[int] = []
    for position in range(1, len(grays)):
        previous = grays[position - 1]
        current = grays[position]
        features = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=1200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        row: dict[str, Any] = {
            "from_frame_index": int(frame_indices[position - 1]),
            "to_frame_index": int(frame_indices[position]),
            "transition": False,
            "reason": None,
        }
        if features is None or len(features) < 40:
            row.update({"status": "ambiguous", "reason": "too_few_background_features"})
            pair_rows.append(row)
            continue
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            features,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if forward is None or forward_status is None:
            row.update({"status": "ambiguous", "reason": "forward_flow_failed"})
            pair_rows.append(row)
            continue
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or backward_status is None:
            row.update({"status": "ambiguous", "reason": "backward_flow_failed"})
            pair_rows.append(row)
            continue
        source_all = features.reshape(-1, 2)
        forward_all = forward.reshape(-1, 2)
        backward_all = backward.reshape(-1, 2)
        keep = (
            (forward_status.reshape(-1) == 1)
            & (backward_status.reshape(-1) == 1)
            & (np.linalg.norm(backward_all - source_all, axis=1) <= 1.5)
        )
        source = source_all[keep]
        destination = forward_all[keep]
        if len(source) < 30:
            row.update(
                {
                    "status": "ambiguous",
                    "reason": "insufficient_forward_backward_consistent_tracks",
                    "track_count": int(len(source)),
                }
            )
            pair_rows.append(row)
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.0,
            maxIters=3000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 24:
            phase_shift, phase_response = cv2.phaseCorrelate(
                np.asarray(previous, dtype=np.float32),
                np.asarray(current, dtype=np.float32),
            )
            phase_translation_px = math.hypot(float(phase_shift[0]), float(phase_shift[1])) / scale
            phase_transition = bool(float(phase_response) >= 0.10 and phase_translation_px > 10.0)
            row.update(
                {
                    "status": "moving" if phase_transition else "ambiguous",
                    "transition": phase_transition,
                    "reason": (
                        "phase_correlation_detected_camera_transition"
                        if phase_transition
                        else "background_transform_underconstrained"
                    ),
                    "track_count": int(len(source)),
                    "phase_translation_px": float(phase_translation_px),
                    "phase_response": float(phase_response),
                }
            )
            if phase_transition:
                transition_positions.append(position)
            pair_rows.append(row)
            continue
        inlier_ratio = float(inliers.mean())
        a, b = float(affine[0, 0]), float(affine[1, 0])
        pair_scale = math.sqrt(a * a + b * b)
        rotation_deg = math.degrees(math.atan2(b, a))
        anchors = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [previous.shape[1] - 1.0, 0.0, 1.0],
                [0.0, previous.shape[0] - 1.0, 1.0],
                [previous.shape[1] - 1.0, previous.shape[0] - 1.0, 1.0],
                [previous.shape[1] * 0.5, previous.shape[0] * 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        transformed = (affine @ anchors.T).T
        max_drift_px = float(np.max(np.linalg.norm(transformed - anchors[:, :2], axis=1))) / scale
        translation_px = math.hypot(float(affine[0, 2]), float(affine[1, 2])) / scale
        transition_reasons: list[str] = []
        if max_drift_px > 12.0 or translation_px > 10.0:
            transition_reasons.append("pan_or_tilt")
        if abs(pair_scale - 1.0) > 0.02:
            transition_reasons.append("zoom")
        if abs(rotation_deg) > 1.0:
            transition_reasons.append("camera_rotation")
        if inlier_ratio < 0.25 and max_drift_px > 6.0:
            transition_reasons.append("possible_cut")
        row.update(
            {
                "status": "moving" if transition_reasons else "static",
                "transition": bool(transition_reasons),
                "reason": "+".join(transition_reasons) if transition_reasons else "pair_near_identity",
                "track_count": int(len(source)),
                "inlier_count": int(inliers.sum()),
                "inlier_ratio": inlier_ratio,
                "max_anchor_drift_px": max_drift_px,
                "translation_px": float(translation_px),
                "scale": float(pair_scale),
                "rotation_deg": float(rotation_deg),
            }
        )
        if transition_reasons:
            transition_positions.append(position)
        pair_rows.append(row)

    segment_position_groups: list[list[int]] = []
    start = 0
    for position in transition_positions:
        segment_position_groups.append(list(range(start, position)))
        start = position
    segment_position_groups.append(list(range(start, len(frames_bgr))))
    segment_position_groups = [group for group in segment_position_groups if group]
    ranked_segments = sorted(
        enumerate(segment_position_groups),
        key=lambda item: (-len(item[1]), int(frame_indices[item[1][0]]), item[0]),
    )
    selected_segment_id, selected_group = ranked_segments[0]
    selected_positions = _select_segment_positions(
        frames_bgr,
        selected_group,
        max_frames=max_selected_frames,
    )
    segments = [
        {
            "segment_id": int(segment_id),
            "frame_indices": [int(frame_indices[position]) for position in group],
            "sample_count": len(group),
        }
        for segment_id, group in enumerate(segment_position_groups)
    ]
    return {
        "status": "transition_detected" if transition_positions else "single_static_cluster",
        "whole_clip_static_eligible": not transition_positions,
        "segments": segments,
        "pair_transforms": pair_rows,
        "transition_frame_indices": [int(frame_indices[position]) for position in transition_positions],
        "selected_segment_id": int(selected_segment_id),
        "selected_segment_range": [
            int(frame_indices[selected_group[0]]),
            int(frame_indices[selected_group[-1]]),
        ],
        "selected_positions": [int(position) for position in selected_positions],
        "selected_frame_indices": [int(frame_indices[position]) for position in selected_positions],
        "selection_method": "sharpest_per_temporal_bin_within_largest_coherent_segment",
        "person_motion_handling": "forward_backward_tracks_plus_ransac_dynamic_outlier_rejection",
    }


def _select_segment_positions(
    frames_bgr: Sequence[Any],
    positions: Sequence[int],
    *,
    max_frames: int,
) -> tuple[int, ...]:
    import cv2

    ordered = tuple(sorted(int(value) for value in positions))
    if len(ordered) <= max_frames:
        return ordered
    selected: list[int] = []
    for raw_bin in np.array_split(np.asarray(ordered, dtype=np.int64), max_frames):
        candidates = [int(value) for value in raw_bin.tolist()]
        best = min(
            candidates,
            key=lambda position: (
                -float(cv2.Laplacian(frames_bgr[position], cv2.CV_64F).var()),
                position,
            ),
        )
        selected.append(best)
    return tuple(sorted(selected))


def _appearance_static_diagnostics(
    frames_bgr: Sequence[Any],
    *,
    frame_indices: Sequence[int],
    max_width: int = 640,
) -> dict[str, Any]:
    """Measure real image motion independently of noisy court predictions.

    The court network can jitter several pixels across a truly immutable
    tripod clip.  Sparse background flow with a RANSAC similarity transform
    distinguishes that prediction noise from camera pan/tilt/zoom.  Moving
    players are rejected as flow outliers; insufficient background support is
    explicitly ambiguous rather than accepted.
    """

    import cv2

    if len(frames_bgr) < 3 or len(frames_bgr) != len(frame_indices):
        return {
            "status": "ambiguous",
            "reason": "insufficient_frames",
            "valid_pair_count": 0,
        }
    height, width = frames_bgr[0].shape[:2]
    scale = min(1.0, float(max_width) / max(float(width), 1.0))

    def gray(frame: Any) -> np.ndarray:
        resized = (
            frame
            if scale == 1.0
            else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    reference = gray(frames_bgr[0])
    reference_points = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if reference_points is None or len(reference_points) < 40:
        return {
            "status": "ambiguous",
            "reason": "too_few_background_features",
            "reference_feature_count": 0 if reference_points is None else int(len(reference_points)),
            "valid_pair_count": 0,
        }

    anchor_points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [reference.shape[1] - 1.0, 0.0, 1.0],
            [0.0, reference.shape[0] - 1.0, 1.0],
            [reference.shape[1] - 1.0, reference.shape[0] - 1.0, 1.0],
            [reference.shape[1] * 0.5, reference.shape[0] * 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    drift_px = [0.0]
    inlier_ratios = [1.0]
    valid_indices = [int(frame_indices[0])]
    for frame_index, frame in zip(frame_indices[1:], frames_bgr[1:], strict=True):
        current = gray(frame)
        tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
            reference,
            current,
            reference_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if tracked is None or status is None:
            continue
        keep = status.reshape(-1) == 1
        source = reference_points.reshape(-1, 2)[keep]
        destination = tracked.reshape(-1, 2)[keep]
        if len(source) < 30:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.0,
            maxIters=3000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 30:
            continue
        ratio = float(inliers.mean())
        if ratio < 0.35:
            continue
        transformed = (affine @ anchor_points.T).T
        anchor_drift = np.linalg.norm(transformed - anchor_points[:, :2], axis=1)
        drift_px.append(float(np.max(anchor_drift)) / scale)
        inlier_ratios.append(ratio)
        valid_indices.append(int(frame_index))

    required = max(3, int(math.ceil(len(frames_bgr) * 0.5)))
    if len(drift_px) < required:
        return {
            "status": "ambiguous",
            "reason": "insufficient_valid_background_flow",
            "reference_feature_count": int(len(reference_points)),
            "valid_pair_count": len(drift_px),
            "required_pair_count": required,
            "valid_frame_indices": valid_indices,
        }
    drift_array = np.asarray(drift_px, dtype=np.float64)
    p50 = float(np.percentile(drift_array, 50))
    p95 = float(np.percentile(drift_array, 95))
    maximum = float(np.max(drift_array))
    mean_inlier_ratio = float(np.mean(inlier_ratios))
    if p95 <= 1.5 and maximum <= 3.0 and mean_inlier_ratio >= 0.45:
        status_name = "static"
        reason = "background_flow_near_identity"
    elif p50 >= 2.5 or p95 >= 4.0:
        status_name = "moving"
        reason = "background_flow_exceeds_static_threshold"
    else:
        status_name = "ambiguous"
        reason = "background_flow_overlaps_static_threshold"
    return {
        "status": status_name,
        "reason": reason,
        "reference_feature_count": int(len(reference_points)),
        "valid_pair_count": len(drift_px),
        "required_pair_count": required,
        "valid_frame_indices": valid_indices,
        "drift_px_p50": p50,
        "drift_px_p95": p95,
        "drift_px_max": maximum,
        "mean_inlier_ratio": mean_inlier_ratio,
    }


def _resolve_static_motion(
    transform_motion: StaticMotionDiagnostics,
    appearance_motion: Mapping[str, Any],
    *,
    segment_diagnostics: Mapping[str, Any] | None = None,
) -> StaticMotionDiagnostics:
    if segment_diagnostics is not None and not bool(
        segment_diagnostics.get("whole_clip_static_eligible", False)
    ):
        return replace(
            transform_motion,
            status="moving",
            camera_motion_suspected=True,
            static_lock_usable=False,
            reasons=(
                *transform_motion.reasons,
                "segment_first_inference_detected_camera_transition",
            ),
        )
    appearance_status = str(appearance_motion.get("status") or "ambiguous")
    if appearance_status == "ambiguous":
        p95 = appearance_motion.get("drift_px_p95")
        maximum = appearance_motion.get("drift_px_max")
        inlier_ratio = appearance_motion.get("mean_inlier_ratio")
        if (
            isinstance(p95, (int, float))
            and isinstance(maximum, (int, float))
            and isinstance(inlier_ratio, (int, float))
            and float(p95) <= 3.0
            and float(maximum) <= 3.0
            and float(inlier_ratio) >= 0.45
        ):
            return replace(
                transform_motion,
                status="static",
                camera_motion_suspected=False,
                static_lock_usable=True,
                reasons=(
                    *transform_motion.reasons,
                    "borderline_background_flow_recovered_by_static_cluster",
                ),
            )
        return transform_motion
    if appearance_status == "static":
        return replace(
            transform_motion,
            status="static",
            camera_motion_suspected=False,
            static_lock_usable=True,
            reasons=(
                *transform_motion.reasons,
                "background_flow_resolved_transform_uncertainty_as_static",
            ),
        )
    return replace(
        transform_motion,
        status="moving",
        camera_motion_suspected=True,
        static_lock_usable=False,
        reasons=(
            *transform_motion.reasons,
            "background_flow_detected_camera_motion",
        ),
    )


def _lock_eligibility(
    best_court: Mapping[str, Any],
    *,
    source: str,
    motion: StaticMotionDiagnostics,
    segment_diagnostics: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if best_court.get("homography_image_from_court") is None:
        reasons.append("missing_floor_homography")
    diagnostics = best_court.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    hard = diagnostics.get("hard_eligibility")
    hard = hard if isinstance(hard, Mapping) else {}
    if not bool(hard.get("eligible", False)):
        hard_reasons = hard.get("reasons")
        if isinstance(hard_reasons, Sequence) and not isinstance(hard_reasons, (str, bytes)):
            reasons.extend(f"hard_geometry:{value}" for value in hard_reasons)
        else:
            reasons.append("hard_geometry_not_eligible")
    if source not in {"multi_frame_point_and_line", "clearest_frame_point_and_line"}:
        reasons.append(f"source_not_lock_authorizing:{source}")
    inlier_ratio = best_court.get("inlier_ratio")
    if not isinstance(inlier_ratio, (int, float)) or float(inlier_ratio) < (2.0 / 3.0):
        reasons.append("semantic_inlier_ratio_below_two_thirds")
    residual = best_court.get("residual_stats_px")
    residual = residual if isinstance(residual, Mapping) else {}
    p90 = residual.get("p90")
    if not isinstance(p90, (int, float)) or not math.isfinite(float(p90)) or float(p90) > 10.0:
        reasons.append("actual_p90_exceeds_10px_or_missing")
    if best_court.get("supported_view") is False:
        reasons.append("unsupported_view")
    if not motion.static_lock_usable:
        reasons.append(f"camera_motion:{motion.status}")
    if not bool(segment_diagnostics.get("whole_clip_static_eligible", False)):
        reasons.append("whole_clip_contains_camera_transition")
    net_stage = best_court.get("net_stage")
    if isinstance(net_stage, Mapping) and net_stage.get("status") == "pose_underconstrained":
        reasons.append("camera_pose_underconstrained")
    return not reasons, tuple(dict.fromkeys(reasons))


def _static_observations(result: Mapping[str, Any]) -> list[StaticCourtObservation]:
    line_support = float((result["best_court"].get("score_components") or {}).get("line_alignment", 0.0))
    rows: list[StaticCourtObservation] = []
    for record in result.get("structured_observations") or []:
        primary = record["primary_peak"]
        xy = primary.get("source_xy") or record["observation_xy"]
        covariance = record["covariance_px2"]
        rows.append(
            StaticCourtObservation(
                semantic_name=str(record["keypoint_name"]),
                xy=(float(xy[0]), float(xy[1])),
                confidence=float(record["confidence"]),
                covariance_px2=(
                    (float(covariance[0][0]), float(covariance[0][1])),
                    (float(covariance[1][0]), float(covariance[1][1])),
                ),
                line_support=line_support,
            )
        )
    return rows


def _pooled_line_distances(results: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    import cv2

    by_name: dict[str, list[np.ndarray]] = {}
    for result in results:
        maps = result.get("line_distance_maps")
        if not isinstance(maps, Mapping):
            continue
        for name, value in maps.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 2 and np.isfinite(array).all():
                by_name.setdefault(str(name), []).append(array)
    if by_name:
        return {
            name: np.median(np.stack(values, axis=0), axis=0)
            for name, values in sorted(by_name.items())
        }
    distances: list[np.ndarray] = []
    for result in results:
        mask = np.asarray(result["line_family_mask"] == 1, dtype=np.uint8)
        if np.any(mask):
            distances.append(cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5).astype(np.float64))
    if not distances:
        height, width = np.asarray(results[0]["line_family_mask"]).shape
        distance = np.full((height, width), math.hypot(width, height), dtype=np.float64)
    else:
        distance = np.median(np.stack(distances, axis=0), axis=0)
    return {"pickleball_line": distance}


def _court_lock_artifact(
    best_court: Mapping[str, Any],
    *,
    source: str,
    pooled: Any,
    motion: Any,
    image_size: tuple[int, int],
    camera_profile: Mapping[str, Any] | None,
    checkpoint_sha256: str | None,
    lock_decision_reasons: Sequence[str],
    segment_diagnostics: Mapping[str, Any],
) -> CourtLockArtifact:
    width, height = image_size
    profile = dict(camera_profile or {})
    solved_camera = best_court.get("camera_parameters")
    solved_camera = solved_camera if isinstance(solved_camera, Mapping) else {}
    intrinsics_payload = solved_camera.get("intrinsics")
    if not isinstance(intrinsics_payload, Mapping):
        intrinsics_payload = profile.get("intrinsics") if isinstance(profile.get("intrinsics"), Mapping) else profile
    fx = float(intrinsics_payload.get("fx", max(width, height)))
    fy = float(intrinsics_payload.get("fy", fx))
    cx = float(intrinsics_payload.get("cx", (width - 1) * 0.5))
    cy = float(intrinsics_payload.get("cy", (height - 1) * 0.5))
    distortion_payload = best_court.get("distortion")
    if not isinstance(distortion_payload, Mapping) or not isinstance(
        distortion_payload.get("k1"), (int, float)
    ):
        distortion_payload = profile.get("distortion") if isinstance(profile.get("distortion"), Mapping) else profile
    k1 = float(distortion_payload.get("k1", 0.0))
    residual = best_court.get("residual_stats_px") or {}
    median = float(residual.get("median") or 0.0)
    p95 = float(residual.get("p95") or residual.get("p90") or median)
    pose = solved_camera.get("pose")
    pose = pose if isinstance(pose, Mapping) else {}
    rotation = pose.get("rotation_world_to_camera")
    translation = pose.get("translation_world_to_camera_m")
    return CourtLockArtifact(
        coordinate_space="pixels_raw_native",
        homography_image_from_court=tuple(
            tuple(float(value) for value in row)
            for row in best_court["homography_image_from_court"]
        ),
        camera_parameters=CourtLockCameraParameters(
            intrinsics=PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
            source=str(
                solved_camera.get("source")
                or solved_camera.get("intrinsics_source")
                or profile.get("source")
                or "bounded_image_profile_prior"
            ),
            rotation_world_to_camera=(
                None
                if rotation is None
                else tuple(tuple(float(value) for value in row) for row in rotation)
            ),
            translation_world_to_camera_m=(
                None if translation is None else tuple(float(value) for value in translation)
            ),
        ),
        distortion=CourtLockDistortion(
            k1=k1,
            source=str(distortion_payload.get("source") or "not_available_zero"),
            optimized=bool(
                distortion_payload.get("optimized", False)
                or distortion_payload.get("source") == "joint_point_camera_refinement"
            ),
            k1_variance=float(distortion_payload.get("k1_variance", 0.0)),
        ),
        transform_covariance=(
            None
            if best_court.get("transform_covariance") is None
            else tuple(
                tuple(float(value) for value in row)
                for row in best_court["transform_covariance"]
            )
        ),
        source=source,  # type: ignore[arg-type]
        evidence=CourtLockEvidenceSummary(
            candidate_frame_count=pooled.candidate_frame_count,
            selected_frame_indices=pooled.selected_frame_indices,
            pooled_semantic_count=len(pooled.observations),
        ),
        static_motion=motion,
        residual_px=CourtLockResidual(median_px=median, p95_px=max(median, p95)),
        score_components={
            str(key): float(value)
            for key, value in (best_court.get("score_components") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        checkpoint_sha256=checkpoint_sha256,
        lock_eligible=True,
        lock_decision_reasons=tuple(str(value) for value in lock_decision_reasons),
        segment_range=tuple(int(value) for value in segment_diagnostics["selected_segment_range"]),
        segment_diagnostics=dict(segment_diagnostics),
        scorer_version="confidence_aware_structured_court_v31",
        calibration_version=str(
            (best_court.get("court_confidence_calibration") or {}).get(
                "schema_version", "uncalibrated"
            )
        ),
        measurement_valid=False,
        authority_state="review_only",
        verified=False,
    )


def _checkpoint_sha(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path | None = None,
) -> str | None:
    for key in ("checkpoint_sha256", "sha256"):
        value = payload.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    if checkpoint_path is not None and checkpoint_path.is_file():
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return None


def _camera_profile(
    payload: Mapping[str, Any] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[PinholeIntrinsics, dict[str, Any], str]:
    width, height = image_size
    profile = dict(payload or {})
    intrinsics_payload = profile.get("intrinsics")
    if not isinstance(intrinsics_payload, Mapping):
        intrinsics_payload = profile
    fx = float(intrinsics_payload.get("fx", max(width, height)))
    fy = float(intrinsics_payload.get("fy", fx))
    cx = float(intrinsics_payload.get("cx", (width - 1) * 0.5))
    cy = float(intrinsics_payload.get("cy", (height - 1) * 0.5))
    distortion_payload = profile.get("distortion")
    if not isinstance(distortion_payload, Mapping):
        distortion_payload = profile
    return (
        PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        {
            "k1": float(distortion_payload.get("k1", 0.0)),
            "k1_variance": float(distortion_payload.get("k1_variance", 0.0)),
            "source": str(distortion_payload.get("source") or "not_available_zero"),
        },
        str(profile.get("source") or "bounded_image_profile_prior"),
    )


def _mean_optional(values: Sequence[Any]) -> float | None:
    parsed = [float(value) for value in values if isinstance(value, (int, float))]
    return None if not parsed else float(sum(parsed) / len(parsed))


def _profile_prior_homography(width: int, height: int) -> list[list[float]]:
    """Return a conservative centered-court initializer for last-resort optimization."""

    import cv2

    world = np.asarray(
        [
            [-3.048, -6.7056],
            [3.048, -6.7056],
            [3.048, 6.7056],
            [-3.048, 6.7056],
        ],
        dtype=np.float64,
    )
    image = np.asarray(
        [
            [0.04 * width, 0.92 * height],
            [0.96 * width, 0.92 * height],
            [0.77 * width, 0.25 * height],
            [0.23 * width, 0.25 * height],
        ],
        dtype=np.float64,
    )
    homography, _ = cv2.findHomography(world, image, method=0)
    if homography is None:
        raise ValueError("could not construct camera profile prior")
    homography /= homography[2, 2]
    return homography.tolist()


def _lock_source(selected_source: str, *, default: str) -> str:
    if selected_source.startswith("camera_profile_prior"):
        return "camera_profile_prior"
    if selected_source.startswith("previous_static_lock") or selected_source.startswith(
        "prior_homography"
    ):
        return "previous_static_lock"
    if "line_only" in selected_source:
        return "dense_line_only"
    if selected_source.startswith("clearest_frame"):
        return "clearest_frame_point_and_line"
    return default


__all__ = ["infer_static_court_model"]
