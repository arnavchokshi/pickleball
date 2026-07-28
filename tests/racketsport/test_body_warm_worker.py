from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import body_warm_worker as bw
from scripts.racketsport import remote_body_dispatch as rbd


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _remote_config(**kwargs: Any) -> rbd.RemoteConfig:
    kwargs.setdefault("host", "fixture@remote")
    return rbd.RemoteConfig(**kwargs)


def test_body_warm_worker_cli_help_direct_reference() -> None:
    # Repo-policy scaffold audit (scripts/racketsport/list_scaffold_tools.py /
    # tests/racketsport/test_scaffold_tool_index.py) requires every scripts/
    # CLI to have a direct reference test that shells out to it by path.
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/body_warm_worker.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "start" in result.stdout
    assert "status" in result.stdout
    assert "stop" in result.stdout

    for mode in ("start", "status", "stop"):
        sub_result = subprocess.run(
            [sys.executable, "scripts/racketsport/body_warm_worker.py", mode, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert sub_result.returncode == 0, sub_result.stderr


def test_probe_manifest_and_socket_reports_absent() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout=json.dumps({"manifest_found": False}))

    payload = bw._probe_manifest_and_socket(_remote_config(), "/x.sock", connect_timeout_s=5.0, run=fake_run)
    assert payload["manifest_found"] is False


def test_probe_manifest_and_socket_raises_typed_error_on_ssh_failure() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(255, stderr="Connection refused")

    with pytest.raises(bw.WarmWorkerLifecycleError, match="probe SSH failed"):
        bw._probe_manifest_and_socket(_remote_config(), "/x.sock", connect_timeout_s=5.0, run=fake_run)


def test_probe_manifest_and_socket_raises_typed_error_on_non_json() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout="not json")

    with pytest.raises(bw.WarmWorkerLifecycleError, match="non-JSON probe output"):
        bw._probe_manifest_and_socket(_remote_config(), "/x.sock", connect_timeout_s=5.0, run=fake_run)


def test_stop_worker_reports_absent_when_no_manifest() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout=json.dumps({"manifest_found": False}))

    result = bw.stop_worker(_remote_config(), clip="wolverine", socket_path=None, timeout_s=5.0, run=fake_run)
    assert result["status"] == "absent"


def test_stop_worker_requires_clip_or_socket_path() -> None:
    with pytest.raises(bw.WarmWorkerLifecycleError, match="requires --clip or --socket-path"):
        bw.stop_worker(_remote_config(), clip=None, socket_path=None, timeout_s=5.0, run=lambda cmd, timeout_s: _completed(0))


def test_stop_worker_kills_recorded_pid_and_confirms_socket_gone() -> None:
    calls: list[list[str]] = []
    manifest = {"pid": 4321, "clip": "wolverine"}
    responses = iter(
        [
            {"manifest_found": True, "manifest": manifest, "socket_reachable": True},  # initial probe
            {"manifest_found": True, "manifest": manifest, "socket_reachable": False},  # after kill
        ]
    )

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[0] == "ssh" and "manifest_path = " in cmd[-1]:
            return _completed(0, stdout=json.dumps(next(responses)))
        if cmd[0] == "ssh" and cmd[-1].startswith("kill "):
            assert "4321" in cmd[-1]
            return _completed(0)
        if cmd[0] == "ssh" and cmd[-1].startswith("rm -f"):
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    result = bw.stop_worker(
        _remote_config(),
        clip="wolverine",
        socket_path="/x.sock",
        timeout_s=5.0,
        run=fake_run,
        sleep=lambda s: None,
    )
    assert result["status"] == "stopped"
    assert result["pid"] == 4321
    assert any(cmd[-1].startswith("kill ") for cmd in calls if cmd[0] == "ssh")
    assert any(cmd[-1].startswith("rm -f") for cmd in calls if cmd[0] == "ssh")


def test_stop_worker_reports_timeout_when_socket_stays_reachable() -> None:
    manifest = {"pid": 111, "clip": "wolverine"}

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and "manifest_path = " in cmd[-1]:
            return _completed(
                0, stdout=json.dumps({"manifest_found": True, "manifest": manifest, "socket_reachable": True})
            )
        if cmd[0] == "ssh" and cmd[-1].startswith("kill "):
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    result = bw.stop_worker(
        _remote_config(),
        clip="wolverine",
        socket_path="/x.sock",
        timeout_s=0.0,
        run=fake_run,
        sleep=lambda s: None,
    )
    assert result["status"] == "stop_timed_out"


def test_cmd_status_exits_zero_when_healthy(capsys: pytest.CaptureFixture[str]) -> None:
    config_for_default_path = _remote_config()
    expected_socket_path = rbd.default_warm_worker_socket_path(config_for_default_path, "wolverine")
    current_sha = rbd._git_head_sha(rbd.ROOT)
    manifest = {"clip": "wolverine", "git_head_sha": current_sha, "socket_path": expected_socket_path}

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(
            0, stdout=json.dumps({"manifest_found": True, "manifest": manifest, "socket_reachable": True})
        )

    parser = bw._build_arg_parser()
    args = parser.parse_args(["status", "--host", "fixture@remote", "--clip", "wolverine"])
    exit_code = bw.cmd_status(args, run=fake_run)
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "healthy"


def test_cmd_status_exits_one_when_absent(capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout=json.dumps({"manifest_found": False}))

    parser = bw._build_arg_parser()
    args = parser.parse_args(["status", "--host", "fixture@remote", "--clip", "wolverine"])
    exit_code = bw.cmd_status(args, run=fake_run)
    assert exit_code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "absent"


def test_cmd_stop_exits_zero_when_already_absent(capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout=json.dumps({"manifest_found": False}))

    parser = bw._build_arg_parser()
    args = parser.parse_args(["stop", "--host", "fixture@remote", "--clip", "wolverine"])
    exit_code = bw.cmd_stop(args, run=fake_run)
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "absent"


def test_find_bootstrap_requests_returns_the_newest_match() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(
            0, stdout="/remote/run/batch_requests-aaa.json\n/remote/run/batch_requests-bbb.json\n"
        )

    path = bw._find_bootstrap_requests(_remote_config(), "/remote/run", run=fake_run)
    assert path == "/remote/run/batch_requests-bbb.json"


def test_find_bootstrap_requests_raises_typed_error_when_none_found() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout="")

    with pytest.raises(bw.WarmWorkerLifecycleError, match="no batch_requests"):
        bw._find_bootstrap_requests(_remote_config(), "/remote/run", run=fake_run)


def test_resolve_checkpoint_dir_parses_manifest_lookup_output() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        assert "verify_fast_sam_manifest_assets" in cmd[-1]
        return _completed(0, stdout=json.dumps({"checkpoint_dir": "/ckpt/dir"}))

    result = bw._resolve_checkpoint_dir(
        _remote_config(), detector_name="", fov_name="", manifest_path="/repo/models/MANIFEST.json", run=fake_run
    )
    assert result == "/ckpt/dir"


def test_resolve_checkpoint_dir_raises_typed_error_on_failure() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(1, stderr="models manifest is missing required BODY model id")

    with pytest.raises(bw.WarmWorkerLifecycleError, match="could not resolve"):
        bw._resolve_checkpoint_dir(
            _remote_config(), detector_name="", fov_name="", manifest_path="/repo/models/MANIFEST.json", run=fake_run
        )


def test_poll_for_ready_returns_payload_once_available() -> None:
    calls = {"n": 0}

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] < 2:
            return _completed(0, stdout="")
        return _completed(0, stdout=json.dumps({"pid": 999, "fingerprint": "abc"}))

    payload = bw._poll_for_ready(
        _remote_config(),
        ready_path="/x.sock.ready.json",
        timeout_s=5.0,
        poll_interval_s=0.0,
        log_path="/x.sock.log",
        run=fake_run,
        sleep=lambda s: None,
    )
    assert payload["pid"] == 999


def test_poll_for_ready_raises_typed_error_on_timeout() -> None:
    def fake_run(cmd, timeout_s):  # noqa: ANN001
        return _completed(0, stdout="")

    with pytest.raises(bw.WarmWorkerLifecycleError, match="did not become ready"):
        bw._poll_for_ready(
            _remote_config(),
            ready_path="/x.sock.ready.json",
            timeout_s=0.0,
            poll_interval_s=0.0,
            log_path="/x.sock.log",
            run=fake_run,
            sleep=lambda s: None,
        )


def test_start_worker_refuses_when_already_healthy_without_replace(tmp_path: Path) -> None:
    manifest = {"pid": 123, "clip": "wolverine"}

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        if cmd[0] == "ssh" and "manifest_path = " in cmd[-1]:
            return _completed(
                0, stdout=json.dumps({"manifest_found": True, "manifest": manifest, "socket_reachable": True})
            )
        raise AssertionError(f"unexpected command: {cmd}")

    parser = bw._build_arg_parser()
    args = parser.parse_args(
        [
            "start",
            "--host",
            "fixture@remote",
            "--clip",
            "wolverine",
            "--clip-dir",
            str(tmp_path),
            "--video",
            str(tmp_path / "source.mp4"),
        ]
    )
    with pytest.raises(bw.WarmWorkerLifecycleError, match="already running"):
        bw.start_worker(args, run=fake_run)


def test_start_worker_happy_path_launches_under_gpu_lock_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cold-bootstrap dispatch itself is dispatch_body_stage()'s own,
    # already-exhaustively-tested responsibility (tests/racketsport/
    # test_remote_body_dispatch.py); stub it here so this test stays focused
    # on body_warm_worker.py's *own* new logic: refusing to double-start,
    # locating the real request payload the cold run produced, resolving the
    # checkpoint dir, launching serve under the shared lock, polling for
    # readiness, and writing the typed manifest sidecar.
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    (clip_dir / "tracks.json").write_text("{}", encoding="utf-8")
    video_path = clip_dir / "source.mp4"
    video_path.write_bytes(b"x")

    fake_cold_result = rbd.RemoteBodyDispatchResult(
        status="ran",
        remote_run_dir="/remote/repo/runs/process_video_body_dispatch/wolverine_20260728T000000Z",
    )
    dispatch_calls: list[dict[str, Any]] = []

    def fake_dispatch_body_stage(**kwargs: Any) -> rbd.RemoteBodyDispatchResult:
        dispatch_calls.append(kwargs)
        return fake_cold_result

    monkeypatch.setattr(bw.rbd, "dispatch_body_stage", fake_dispatch_body_stage)

    calls: list[list[str]] = []

    def fake_run(cmd, timeout_s):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[0] != "ssh":
            raise AssertionError(f"unexpected non-ssh command: {cmd}")
        text = cmd[-1]
        if "verify_fast_sam_manifest_assets" in text:
            return _completed(0, stdout=json.dumps({"checkpoint_dir": "/ckpt/fast_sam_3d_body_dinov3"}))
        if "manifest_path = " in text:
            return _completed(0, stdout=json.dumps({"manifest_found": False}))
        if text.startswith("find "):
            return _completed(0, stdout=f"{fake_cold_result.remote_run_dir}/fast_sam_subprocess/batch_requests-xyz.json\n")
        if text.startswith("mkdir -p"):
            return _completed(0)
        if "nohup env" in text:
            assert "setsid" in text
            assert "sam3dbody_persistent_worker.py serve" in text
            return _completed(0, stdout="55555\n")
        if text.startswith("cat >"):
            return _completed(0)
        if text.startswith("cat "):
            return _completed(
                0,
                stdout=json.dumps(
                    {
                        "pid": 44444,
                        "fingerprint": "fp123",
                        "bootstrap_timing_summary": {"model_setup_load_s": 12.0, "compile_warmup_s": 30.0},
                    }
                ),
            )
        raise AssertionError(f"unexpected ssh command: {text}")

    parser = bw._build_arg_parser()
    args = parser.parse_args(
        [
            "start",
            "--host",
            "fixture@remote",
            "--clip",
            "wolverine",
            "--clip-dir",
            str(clip_dir),
            "--video",
            str(video_path),
        ]
    )
    manifest = bw.start_worker(args, run=fake_run, sleep=lambda s: None)

    assert manifest["clip"] == "wolverine"
    assert manifest["pid"] == 44444
    assert manifest["checkpoint_dir"] == "/ckpt/fast_sam_3d_body_dinov3"
    assert manifest["bootstrap_requests_path"] == f"{fake_cold_result.remote_run_dir}/fast_sam_subprocess/batch_requests-xyz.json"
    assert manifest["bootstrap_remote_run_dir"] == fake_cold_result.remote_run_dir
    assert manifest["started_under_gpu_lock"] is True
    assert manifest["fingerprint"] == "fp123"
    assert manifest["bootstrap_timing_summary"]["compile_warmup_s"] == 30.0
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0]["clip"] == "wolverine"
    # the bootstrap cold dispatch itself must never be routed through a
    # not-yet-existent warm worker.
    assert dispatch_calls[0]["config"].warm_worker is False
    assert any("nohup env" in c[-1] and "setsid" in c[-1] for c in calls if c[0] == "ssh")
    assert any(c[-1].startswith("cat >") for c in calls if c[0] == "ssh")
