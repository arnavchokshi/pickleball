from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport import orchestrator
from threed.racketsport.body_full_clip_gate import build_body_full_clip_gate
from threed.racketsport.body_joint_quality import build_body_joint_quality
from threed.racketsport.body_mesh_index import build_body_mesh_index_from_arrays, build_body_mesh_index_from_payload
from threed.racketsport.body_mesh_readiness import build_body_mesh_readiness
from threed.racketsport.contact_splice import splice_contact_skeleton_with_body_mesh
from threed.racketsport.mesh_export import build_body_mesh_export
from threed.racketsport.pose_temporal import apply_sam3d_wrist_bone_lock
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError
from threed.racketsport.worldhmr import build_body_artifacts_from_fast_sam


def test_compact_json_writer_round_trips_nested_payload_and_terminates_with_newline(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "representative_nested_payload",
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 3,
                        "joints_world": [[0.1, 0.2, 1.0], [0.4, 0.5, 1.3]],
                        "confidence": {"band": "preview", "reasons": ["unit_test"]},
                    }
                ],
            }
        ],
        "summary": {"mesh_frame_count": 1, "player_count": 1},
    }
    out = tmp_path / "smpl_motion.json"

    timing = orchestrator._write_compact_json(out, payload)

    raw = out.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "\n " not in raw
    assert ": " not in raw
    assert ", " not in raw
    assert json.loads(raw) == payload
    assert timing["bytes"] == out.stat().st_size
    assert timing["serialization_seconds"] >= 0.0


def test_array_native_body_payload_matches_legacy_gate_and_mesh_bytes(tmp_path: Path) -> None:
    from threed.racketsport.body_array_native import build_body_array_native_artifacts_from_fast_sam

    calibration = _calibration()
    samples = _sam3d_samples()
    body_execution = _body_compute_execution()
    frame_plan = _frame_compute_plan()
    tracks = _tracks_payload()

    legacy_smpl, legacy_skeleton, legacy_grounding = build_body_artifacts_from_fast_sam(
        samples,
        calibration=calibration,
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        max_track_anchor_smoothing_residual_m=0.75,
        sam3d_wrist_bone_lock=True,
    )
    native = build_body_array_native_artifacts_from_fast_sam(
        samples,
        calibration=calibration,
        fps=30.0,
        clip="clip",
        body_compute_execution=body_execution,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        max_track_anchor_smoothing_residual_m=0.75,
        sam3d_wrist_bone_lock=True,
    )

    legacy_mesh = build_body_mesh_export(legacy_smpl, clip="clip", body_compute_execution=body_execution)
    native_mesh = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        **native.body_mesh_metadata,
        "players": native.body_mesh_players,
        "summary": native.body_mesh_summary,
    }
    legacy_final = _final_body_outputs(
        tmp_path / "legacy",
        smpl_motion=legacy_smpl,
        skeleton3d=legacy_skeleton,
        body_mesh=legacy_mesh,
        grounding_metrics=legacy_grounding,
        body_compute_execution=body_execution,
        frame_compute_plan=frame_plan,
        tracks=tracks,
    )
    native_final = _final_body_outputs(
        tmp_path / "native",
        smpl_motion=native.smpl_motion_view,
        skeleton3d=native.skeleton3d,
        body_mesh=native_mesh,
        grounding_metrics=native.grounding_metrics,
        body_compute_execution=body_execution,
        frame_compute_plan=frame_plan,
        tracks=tracks,
    )

    for name in (
        "skeleton3d.json",
        "body_full_clip_gate.json",
        "body_mesh_readiness.json",
        "contact_splice.json",
        "body_mesh_index/body_mesh_index.json",
        "body_mesh_index/body_mesh_faces.json",
        "body_mesh_index/body_mesh_chunks/window_000.bin.gz",
    ):
        assert (tmp_path / "native" / name).read_bytes() == (tmp_path / "legacy" / name).read_bytes(), name
    assert native_final["body_joint_quality"] == legacy_final["body_joint_quality"]
    assert native.grounding_metrics == legacy_grounding

    broken_players = copy.deepcopy(native.body_mesh_players)
    broken_players[0]["frames"][0]["mesh_vertices_world"] = []
    build_body_mesh_index_from_arrays(
        metadata=native.body_mesh_metadata,
        players=broken_players,
        out_dir=tmp_path / "broken" / "body_mesh_index",
    )
    assert (tmp_path / "broken" / "body_mesh_index" / "body_mesh_index.json").read_bytes() != (
        tmp_path / "legacy" / "body_mesh_index" / "body_mesh_index.json"
    ).read_bytes()


def test_array_native_body_uses_legacy_joint_compute_for_stance_wrist_finalization() -> None:
    from threed.racketsport.body_array_native import build_body_array_native_artifacts_from_fast_sam

    calibration = _calibration()
    samples = _sam3d_samples_with_stance_and_wrist_motion()
    stance_index = {(7, frame_idx): {"stance": True, "phase_id": "left_stance"} for frame_idx in range(len(samples))}

    _legacy_smpl, legacy_skeleton, _legacy_grounding = build_body_artifacts_from_fast_sam(
        samples,
        calibration=calibration,
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        max_track_anchor_smoothing_residual_m=0.75,
        sam3d_wrist_bone_lock=True,
        stance_index=stance_index,
    )
    native = build_body_array_native_artifacts_from_fast_sam(
        samples,
        calibration=calibration,
        fps=30.0,
        clip="clip",
        body_compute_execution=_body_compute_execution_for_frames(len(samples)),
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        max_track_anchor_smoothing_residual_m=0.75,
        sam3d_wrist_bone_lock=True,
        stance_index=stance_index,
    )

    asserted_indices = tuple(range(9, 21))
    arm_wrist_indices = (5, 6, 7, 8, 41, 62)
    _assert_skeleton_joint_bytes_identical(
        legacy_skeleton,
        native.skeleton3d,
        joint_indices=asserted_indices,
    )
    _assert_skeleton_joint_bytes_identical(
        legacy_skeleton,
        native.skeleton3d,
        joint_indices=arm_wrist_indices,
    )

    broken = copy.deepcopy(native.skeleton3d)
    broken["players"][0]["frames"][2]["joints_world"][10][0] += 0.0846
    with pytest.raises(AssertionError):
        _assert_skeleton_joint_bytes_identical(
            legacy_skeleton,
            broken,
            joint_indices=asserted_indices,
        )


def test_default_slim_body_runner_uses_shared_array_compute_without_writing_monoliths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "clip"
    _write_runner_inputs(run_dir)
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    calls = {"array_native": 0}
    original_array_native_builder = orchestrator.build_body_array_native_artifacts_from_fast_sam

    def fail_legacy_builder(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        raise AssertionError("legacy BODY monolith builder was called despite safe shared array-native default")

    def recording_array_native_builder(*args: Any, **kwargs: Any) -> Any:
        calls["array_native"] += 1
        return original_array_native_builder(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_body_artifacts_from_fast_sam", fail_legacy_builder)
    monkeypatch.setattr(orchestrator, "build_body_array_native_artifacts_from_fast_sam", recording_array_native_builder)

    runner = orchestrator.BodyStageRunner(
        runtime=_FakeSam3DRuntime(),
        detector_name="",
        fov_name="",
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
        write_body_monoliths=False,
    )
    result = runner.run(
        orchestrator.StageContext(
            clip="clip",
            inputs_dir=run_dir,
            run_dir=run_dir,
            sport="pickleball",
            expected_players=1,
        )
    )

    assert result.status == "ran"
    assert calls["array_native"] == 1
    assert not (run_dir / "smpl_motion.json").exists()
    assert not (run_dir / "body_mesh.json").exists()
    assert "smpl_motion.json" not in result.produced_artifacts
    assert "body_mesh.json" not in result.produced_artifacts
    readiness = json.loads((run_dir / "body_mesh_readiness.json").read_text(encoding="utf-8"))
    assert readiness["monoliths"]["status"] == "not_built"
    timing = json.loads((run_dir / "body_stage_phase_timing.json").read_text(encoding="utf-8"))
    assert timing["array_native_gate_feed_s"] >= 0.0
    assert timing["smpl_motion_payload_assembly_s"] == 0.0
    assert timing["mesh_export_payload_assembly_s"] >= 0.0
    assert timing["mesh_smpl_payload_assembly_s"] == 0.0

    from threed.racketsport.pipeline_contracts import PIPELINE_STAGE_CONTRACTS

    body_contract = next(contract for contract in PIPELINE_STAGE_CONTRACTS if contract.stage == "body")
    orchestrator._validate_contract_artifacts(body_contract, run_dir)
    (run_dir / "smpl_motion.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        orchestrator._validate_contract_artifacts(body_contract, run_dir)


def test_legacy_body_runner_opt_out_uses_legacy_joint_builder_without_writing_monoliths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "clip"
    _write_runner_inputs(run_dir)
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    calls = {"legacy": 0}
    original_legacy_builder = orchestrator.build_body_artifacts_from_fast_sam

    def recording_legacy_builder(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        calls["legacy"] += 1
        return original_legacy_builder(*args, **kwargs)

    def fail_array_native_builder(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("array-native BODY joint builder was called despite explicit legacy opt-out")

    monkeypatch.setattr(orchestrator, "build_body_artifacts_from_fast_sam", recording_legacy_builder)
    monkeypatch.setattr(orchestrator, "build_body_array_native_artifacts_from_fast_sam", fail_array_native_builder)

    runner = orchestrator.BodyStageRunner(
        runtime=_FakeSam3DRuntime(),
        detector_name="",
        fov_name="",
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
        write_body_monoliths=False,
        experimental_body_array_native=False,
    )
    result = runner.run(
        orchestrator.StageContext(
            clip="clip",
            inputs_dir=run_dir,
            run_dir=run_dir,
            sport="pickleball",
            expected_players=1,
        )
    )

    assert result.status == "ran"
    assert calls["legacy"] == 1
    assert not (run_dir / "smpl_motion.json").exists()
    assert not (run_dir / "body_mesh.json").exists()
    readiness = json.loads((run_dir / "body_mesh_readiness.json").read_text(encoding="utf-8"))
    assert readiness["monoliths"]["status"] == "not_built"
    timing = json.loads((run_dir / "body_stage_phase_timing.json").read_text(encoding="utf-8"))
    assert timing["array_native_gate_feed_s"] is None
    assert timing["smpl_motion_payload_assembly_s"] >= 0.0
    assert timing["mesh_export_payload_assembly_s"] >= 0.0
    assert timing["mesh_smpl_payload_assembly_s"] == pytest.approx(
        timing["smpl_motion_payload_assembly_s"] + timing["mesh_export_payload_assembly_s"]
    )


def _calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            t=[0.0, 0.0, 0.0],
            camera_height_m=1.5,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def _tracks_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [1.0, 2.0], "conf": 0.9},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [1.1, 2.0], "conf": 0.9},
                ],
            }
        ],
        "rally_spans": [],
    }


def _frame_compute_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 2,
        "frames": [],
        "deep_mesh_windows": [
            {
                "frame_start": 1,
                "frame_end": 1,
                "t0": 1.0 / 30.0,
                "t1": 2.0 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "body_joints",
                "target_player_ids": [7],
                "reason_counts": {"ball_aware_contact": 1},
                "max_score": 0.9,
            }
        ],
        "summary": {
            "deep_mesh_frame_count": 1,
            "by_player_target_representation": {"body_joints": 1, "world_mesh": 1},
        },
    }


def _body_compute_execution() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "mode": "adaptive",
        "fps": 30.0,
        "scheduled_frames": [
            {
                "frame_idx": 0,
                "t": 0.0,
                "target_player_ids": [7],
                "target_representation": "body_joints",
                "fallback_representation": "lane_a_skeleton",
                "reasons": ["sam3d_body_joints_all_tracked"],
            },
            {
                "frame_idx": 1,
                "t": 1.0 / 30.0,
                "target_player_ids": [7],
                "target_representation": "world_mesh",
                "fallback_representation": "body_joints",
                "source_window_index": 0,
                "window_frame_start": 1,
                "window_frame_end": 1,
                "window_t0": 1.0 / 30.0,
                "window_t1": 2.0 / 30.0,
                "window_frame_count": 1,
                "reason_counts": {"ball_aware_contact": 1},
                "reasons": ["ball_aware_contact"],
                "max_score": 0.9,
            },
        ],
        "summary": {
            "scheduled_frame_count": 2,
            "scheduled_player_frame_count": 2,
            "scheduled_by_target_representation": {"body_joints": 1, "world_mesh": 1},
            "tier1_mesh_player_frame_count": 1,
            "tier2_body_joint_player_frame_count": 1,
        },
    }


def _body_compute_execution_for_frames(frame_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "mode": "adaptive",
        "fps": 30.0,
        "scheduled_frames": [
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "target_player_ids": [7],
                "target_representation": "world_mesh" if frame_idx == frame_count - 1 else "body_joints",
                "fallback_representation": "body_joints",
                "reasons": ["unit_test"],
            }
            for frame_idx in range(frame_count)
        ],
        "summary": {
            "scheduled_frame_count": frame_count,
            "scheduled_player_frame_count": frame_count,
            "scheduled_by_target_representation": {"body_joints": max(frame_count - 1, 0), "world_mesh": 1},
            "tier1_mesh_player_frame_count": 1,
            "tier2_body_joint_player_frame_count": max(frame_count - 1, 0),
        },
    }


def _sam3d_samples() -> list[dict[str, Any]]:
    joints0 = [[0.01 * idx, 0.0, 0.02 * (idx % 5)] for idx in range(70)]
    joints1 = [[0.01 * idx, 0.02, 0.02 * (idx % 5)] for idx in range(70)]
    return [
        {
            "frame_idx": 0,
            "player_id": 7,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "joints_camera": joints0,
            "vertices_camera": [],
            "mesh_faces": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.1],
            "left_hand_pose": [0.0],
            "right_hand_pose": [0.0],
            "betas": [0.2, 0.3],
        },
        {
            "frame_idx": 1,
            "player_id": 7,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.1, 2.0],
            "joints_camera": joints1,
            "vertices_camera": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.1]],
            "mesh_faces": [[0, 1, 2]],
            "global_orient": [0.0, 0.0, 0.1],
            "body_pose": [0.2, 0.3],
            "left_hand_pose": [0.1],
            "right_hand_pose": [0.2],
            "betas": [0.2, 0.3],
        },
    ]


def _sam3d_samples_with_stance_and_wrist_motion() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for frame_idx in range(5):
        joints = [[0.01 * idx, 0.0, 0.40 + 0.005 * (idx % 7)] for idx in range(70)]
        lower_body_sawtooth = (0.04 if frame_idx % 2 else -0.04) + 0.005 * frame_idx
        for joint_idx in range(9, 21):
            joints[joint_idx] = [
                0.02 * joint_idx + lower_body_sawtooth,
                0.03 * frame_idx,
                0.01 if joint_idx in (13, 14, 15, 16, 17, 18, 19, 20) else 0.30,
            ]
        left_wrist_swing = [0.00, 0.18, -0.16, 0.20, 0.02][frame_idx]
        right_wrist_swing = [0.02, -0.14, 0.16, -0.12, 0.01][frame_idx]
        joints[5] = [-0.24, 0.02 * frame_idx, 0.82]
        joints[6] = [0.24, -0.02 * frame_idx, 0.82]
        joints[7] = [-0.38, 0.06 + 0.02 * frame_idx, 0.64]
        joints[8] = [0.38, -0.06 - 0.02 * frame_idx, 0.64]
        joints[41] = [0.54 + right_wrist_swing, -0.10, 0.46]
        joints[62] = [-0.54 + left_wrist_swing, 0.10, 0.46]
        samples.append(
            {
                "frame_idx": frame_idx,
                "player_id": 7,
                "t": frame_idx / 30.0,
                "confidence": 0.95,
                "track_world_xy": [1.0 + 0.015 * frame_idx, 2.0],
                "joints_camera": joints,
                "vertices_camera": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.1]]
                if frame_idx == 4
                else [],
                "mesh_faces": [[0, 1, 2]] if frame_idx == 4 else [],
                "global_orient": [0.0, 0.0, 0.02 * frame_idx],
                "body_pose": [0.1 * frame_idx, 0.2],
                "left_hand_pose": [0.05 * frame_idx],
                "right_hand_pose": [0.04 * frame_idx],
                "betas": [0.2, 0.3],
            }
        )
    return samples


def _assert_skeleton_joint_bytes_identical(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    joint_indices: tuple[int, ...],
) -> None:
    assert _skeleton_joint_subset_bytes(left, joint_indices=joint_indices) == _skeleton_joint_subset_bytes(
        right,
        joint_indices=joint_indices,
    )


def _skeleton_joint_subset_bytes(payload: dict[str, Any], *, joint_indices: tuple[int, ...]) -> bytes:
    subset: list[dict[str, Any]] = []
    for player in payload["players"]:
        subset.append(
            {
                "id": player["id"],
                "frames": [
                    {
                        "frame_idx": frame["frame_idx"],
                        "joints_world": [frame["joints_world"][joint_idx] for joint_idx in joint_indices],
                    }
                    for frame in player["frames"]
                ],
            }
        )
    return json.dumps(subset, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _write_runner_inputs(run_dir: Path) -> None:
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    orchestrator._write_json(run_dir / "tracks.json", _tracks_payload())
    orchestrator._write_json(run_dir / "court_calibration.json", _calibration().model_dump(mode="json"))
    orchestrator._write_json(run_dir / "frame_compute_plan.json", _frame_compute_plan())


class _FakeSam3DRuntime:
    def process_frame_batches(self, requests: list[Any], **_kwargs: Any) -> list[list[dict[str, Any]]]:
        samples = _sam3d_samples()
        outputs: list[list[dict[str, Any]]] = []
        for request in requests:
            sample = samples[int(request["request_id"].split(":")[0])]
            record: dict[str, Any] = {
                "pred_keypoints_3d": sample["joints_camera"],
                "pred_cam_t": [0.0, 0.0, 1.0],
                "confidence": 0.9,
                "global_rot": sample["global_orient"],
                "body_pose_params": sample["body_pose"],
                "shape_params": sample["betas"],
            }
            if request["target_representation"] == "world_mesh":
                record["pred_vertices"] = sample["vertices_camera"]
                record["mesh_faces"] = sample["mesh_faces"]
            outputs.append([record])
        return outputs


def _final_body_outputs(
    run_dir: Path,
    *,
    smpl_motion: dict[str, Any],
    skeleton3d: dict[str, Any],
    body_mesh: dict[str, Any],
    grounding_metrics: dict[str, Any],
    body_compute_execution: dict[str, Any],
    frame_compute_plan: dict[str, Any],
    tracks: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True)
    if "players" in body_mesh:
        build_body_mesh_index_from_payload(body_mesh, out_dir=run_dir / "body_mesh_index")
    skeleton3d_payload, contact_splice = splice_contact_skeleton_with_body_mesh(
        skeleton3d,
        body_mesh=body_mesh,
        body_compute_execution=body_compute_execution,
    )
    skeleton3d_payload = apply_sam3d_wrist_bone_lock(skeleton3d_payload)
    body_joint_quality = build_body_joint_quality(
        clip="clip",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d_payload,
        body_compute_execution=body_compute_execution,
        smpl_motion_path="/golden/smpl_motion.json",
        skeleton3d_path="/golden/skeleton3d.json",
        body_compute_execution_path="/golden/body_compute_execution.json",
    )
    full_clip_gate = build_body_full_clip_gate(
        clip="clip",
        tracks=tracks,
        body_compute_execution=body_compute_execution,
        body_joint_quality=body_joint_quality,
        contact_splice=contact_splice,
        runtime_timing={"body_wall_seconds": 1.0},
        tracks_path="/golden/tracks.json",
        body_compute_execution_path="/golden/body_compute_execution.json",
        body_joint_quality_path="/golden/body_joint_quality.json",
        contact_splice_path="/golden/contact_splice.json",
        runtime_timing_path="body_stage_wall_clock",
    )
    body_joint_quality = orchestrator._body_joint_quality_after_full_clip_gate(body_joint_quality, full_clip_gate)
    readiness = build_body_mesh_readiness(
        clip="clip",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d_payload,
        frame_compute_plan=frame_compute_plan,
        body_compute_execution=body_compute_execution,
        body_full_clip_gate=full_clip_gate,
        smpl_motion_path="/golden/smpl_motion.json",
        skeleton3d_path="/golden/skeleton3d.json",
        frame_compute_plan_path="/golden/frame_compute_plan.json",
        body_compute_execution_path="/golden/body_compute_execution.json",
        body_full_clip_gate_path="/golden/body_full_clip_gate.json",
    )
    orchestrator._write_json_artifact(run_dir / "skeleton3d.json", skeleton3d_payload)
    orchestrator._write_json_artifact(run_dir / "body_full_clip_gate.json", full_clip_gate)
    orchestrator._write_json_artifact(run_dir / "body_mesh_readiness.json", readiness)
    orchestrator._write_json_artifact(run_dir / "contact_splice.json", contact_splice)
    return {
        "body_joint_quality": body_joint_quality,
        "grounding_metrics": grounding_metrics,
    }
