from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import run_sam3dbody_batch as batch
from scripts.racketsport import sam3dbody_persistent_worker as worker


def test_persistent_worker_cli_help_documents_serve_and_client_modes() -> None:
    # Repo-policy scaffold audit (scripts/racketsport/list_scaffold_tools.py /
    # tests/racketsport/test_scaffold_tool_index.py) requires every scripts/
    # CLI to have a direct reference test that shells out to it by path.
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/sam3dbody_persistent_worker.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "client" in result.stdout

    serve_help = subprocess.run(
        [sys.executable, "scripts/racketsport/sam3dbody_persistent_worker.py", "serve", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert serve_help.returncode == 0
    assert "--socket-path" in serve_help.stdout
    assert "--bootstrap-requests" in serve_help.stdout

    client_help = subprocess.run(
        [sys.executable, "scripts/racketsport/sam3dbody_persistent_worker.py", "client", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert client_help.returncode == 0
    assert "--socket-path" in client_help.stdout
    assert "--requests" in client_help.stdout


def _payload(
    *,
    request_count: int = 1,
    torch_compile: bool = False,
    clip_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    matrix = clip_matrix if clip_matrix is not None else [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    return {
        "schema_version": 1,
        "clip_intrinsics": {"matrix": matrix, "source": "test_fixture"},
        "optimization": {
            "sam3d_body_input_size_px": 384,
            "crop_bucket_sizes": [8],
            "torch_compile": torch_compile,
            "compile_warmup_buckets": [8] if torch_compile else [],
            "compile_warmup_passes": 1,
            "upstream_env": {},
        },
        "requests": [
            {
                "request_id": f"1:{index}",
                "image": "frame.jpg",
                "bboxes": [[0.0, 0.0, 10.0, 10.0]],
                "mask_paths": [],
                "camera_intrinsics": None,
                "sam3d_body_input_size_px": 384,
                "target_representation": "world_mesh",
            }
            for index in range(request_count)
        ],
    }


def _write_requests(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    image = tmp_path / "frame.jpg"
    if not image.exists():
        image.write_bytes(b"not-a-real-jpeg")
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FakeEstimator:
    faces = None


def _install_fake_model_bringup(monkeypatch, *, compile_warmup_status: str = "skipped") -> None:
    monkeypatch.setattr(worker, "_load_setup_sam_3d_body", lambda fast_sam_repo: (lambda **_kw: _FakeEstimator()))
    monkeypatch.setattr(worker, "_setup_estimator", lambda setup_fn, **_kw: setup_fn())
    monkeypatch.setattr(
        worker,
        "_warmup_static_clip_intrinsics",
        lambda estimator, *, clip_intrinsics, optimization, timing: {"status": compile_warmup_status},
    )
    monkeypatch.setattr(worker, "_detect_mhr_correctives_active", lambda estimator: {"status": "unknown", "active": None})
    monkeypatch.setattr(worker, "_runtime_path_errors", lambda image, fast_sam_repo, checkpoint_dir: [])


def _make_worker(tmp_path: Path, monkeypatch, **overrides: Any) -> worker.Sam3DBodyPersistentWorker:
    _install_fake_model_bringup(monkeypatch)
    bootstrap_requests = _write_requests(tmp_path, "bootstrap.json", _payload())
    kwargs: dict[str, Any] = {
        "fast_sam_repo": tmp_path / "fast_sam_repo",
        "checkpoint_dir": tmp_path / "checkpoint_dir",
        "detector_name": "",
        "detector_model": "",
        "fov_name": "",
        "bootstrap_requests_path": bootstrap_requests,
        # macOS/BSD AF_UNIX sun_path is short (~104 bytes); pytest's tmp_path
        # is often already close to that on its own, so socket paths use a
        # short /tmp name instead (regular files below still use tmp_path).
        "socket_path": f"/tmp/s3dbw_{uuid.uuid4().hex[:10]}.sock",
        "ready_path": str(tmp_path / "ready.json"),
        "idle_timeout_s": 5.0,
        "max_consecutive_job_crashes": 2,
        "accept_poll_s": 0.05,
    }
    kwargs.update(overrides)
    return worker.Sam3DBodyPersistentWorker(**kwargs)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_config_fingerprint_is_deterministic() -> None:
    code_identity = worker._hash_code_identity()
    payload = _payload()

    first = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=payload["optimization"],
        clip_intrinsics=payload["clip_intrinsics"],
        code_identity=code_identity,
    )
    second = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=dict(payload["optimization"]),
        clip_intrinsics=dict(payload["clip_intrinsics"]),
        code_identity=code_identity,
    )

    assert first == second


def test_config_fingerprint_changes_with_optimization() -> None:
    code_identity = worker._hash_code_identity()
    payload = _payload(torch_compile=False)
    other = _payload(torch_compile=True)

    first = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=payload["optimization"],
        clip_intrinsics=payload["clip_intrinsics"],
        code_identity=code_identity,
    )
    second = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=other["optimization"],
        clip_intrinsics=other["clip_intrinsics"],
        code_identity=code_identity,
    )

    assert first != second


def test_config_fingerprint_changes_with_clip_intrinsics() -> None:
    code_identity = worker._hash_code_identity()
    payload = _payload()
    other_matrix = [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]]
    other = _payload(clip_matrix=other_matrix)

    first = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=payload["optimization"],
        clip_intrinsics=payload["clip_intrinsics"],
        code_identity=code_identity,
    )
    second = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=other["optimization"],
        clip_intrinsics=other["clip_intrinsics"],
        code_identity=code_identity,
    )

    assert first != second


def test_config_fingerprint_changes_with_code_identity() -> None:
    payload = _payload()

    first = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=payload["optimization"],
        clip_intrinsics=payload["clip_intrinsics"],
        code_identity="aaaa",
    )
    second = worker.compute_config_fingerprint(
        checkpoint_dir=Path("/ckpt"),
        detector_name="",
        detector_model="",
        fov_name="",
        optimization=payload["optimization"],
        clip_intrinsics=payload["clip_intrinsics"],
        code_identity="bbbb",
    )

    assert first != second


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_computes_fingerprint_and_writes_ready_file(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch)

    ready = instance.bootstrap()

    assert instance.estimator is not None
    assert instance.fingerprint == ready["fingerprint"]
    ready_file = json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))
    assert ready_file["artifact_type"] == worker.WORKER_READY_ARTIFACT_TYPE
    assert ready_file["fingerprint"] == instance.fingerprint
    assert ready_file["pid"] == os.getpid()
    assert instance.bootstrap_timing_summary is not None


# ---------------------------------------------------------------------------
# Serve + client round trip (real Unix socket, fake inference)
# ---------------------------------------------------------------------------


def _serve_in_background(instance: worker.Sam3DBodyPersistentWorker) -> tuple[threading.Thread, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []

    def _run() -> None:
        results.append(instance.serve_forever())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # Wait for the socket file to exist before any client connects.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not Path(instance.socket_path).exists():
        time.sleep(0.01)
    return thread, results


def test_serve_and_client_round_trip_succeeds(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch)
    instance.bootstrap()

    def _fake_inference(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        Path(kwargs["out_path"]).write_text(json.dumps({"frames": []}), encoding="utf-8")
        return [], {"mode": "fake"}

    monkeypatch.setattr(worker, "_run_batch_inference_and_write", _fake_inference)

    thread, results = _serve_in_background(instance)
    try:
        job_requests = _write_requests(tmp_path, "job1.json", _payload())
        out_path = tmp_path / "job1_out.json"
        exit_code = worker.submit_job_via_client(
            socket_path=instance.socket_path,
            requests_path=job_requests,
            out_path=out_path,
            chunk_dir=None,
            chunk_format="pickle",
            no_monolithic_output=True,
            bucket_size=None,
        )
        assert exit_code == 0
        assert out_path.exists()
        assert instance.consecutive_job_crashes == 0
    finally:
        instance._should_exit = True
        thread.join(timeout=5.0)
    assert results and results[0]["reason"] in {"job_failure_limit_or_unhealthy", "idle_timeout"}


def test_second_job_reports_zero_model_setup_and_compile_cost(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # The whole point of the lever: once bootstrap has already paid
    # model_setup_load_s/compile_warmup_s, a served job's own timing summary
    # must show ~0 for both, since no model construction/compile happens
    # inside handle_job() itself.
    instance = _make_worker(tmp_path, monkeypatch)
    instance.bootstrap()
    assert instance.bootstrap_timing_summary is not None

    def _fake_inference(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        Path(kwargs["out_path"]).write_text(json.dumps({"frames": []}), encoding="utf-8")
        return [], {"mode": "fake"}

    monkeypatch.setattr(worker, "_run_batch_inference_and_write", _fake_inference)

    job_requests = _write_requests(tmp_path, "job1.json", _payload())
    response = instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "job1_out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )

    assert response["status"] == "ok"
    timing_summary = response["timing_summary"]
    assert timing_summary["model_setup_load_s"] == 0.0
    assert timing_summary["compile_warmup_s"] == 0.0


def test_job_with_different_optimization_is_refused_as_fingerprint_mismatch(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch)
    instance.bootstrap()

    inference_calls: list[Any] = []
    monkeypatch.setattr(
        worker,
        "_run_batch_inference_and_write",
        lambda **kwargs: inference_calls.append(kwargs) or ([], {}),
    )

    mismatched_requests = _write_requests(tmp_path, "job_mismatch.json", _payload(torch_compile=True))
    response = instance.handle_job(
        {
            "requests_path": str(mismatched_requests),
            "out_path": str(tmp_path / "mismatch_out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )

    assert response["status"] == "fingerprint_mismatch"
    assert response["exit_code"] == 3
    assert response["server_fingerprint"] != response["job_fingerprint"]
    assert inference_calls == []
    assert not (tmp_path / "mismatch_out.json").exists()


def test_job_with_matching_clip_but_different_clip_intrinsics_is_refused(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch)
    instance.bootstrap()
    monkeypatch.setattr(worker, "_run_batch_inference_and_write", lambda **kwargs: ([], {}))

    other_matrix = [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]]
    mismatched_requests = _write_requests(tmp_path, "job_other_clip.json", _payload(clip_matrix=other_matrix))
    response = instance.handle_job(
        {
            "requests_path": str(mismatched_requests),
            "out_path": str(tmp_path / "other_clip_out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )

    assert response["status"] == "fingerprint_mismatch"


# ---------------------------------------------------------------------------
# Crash counting / stale-context prevention / idle self-teardown
# ---------------------------------------------------------------------------


def test_worker_exits_after_max_consecutive_job_crashes(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch, max_consecutive_job_crashes=2)
    instance.bootstrap()

    def _always_raise(**_kwargs: Any) -> Any:
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(worker, "_run_batch_inference_and_write", _always_raise)

    job_requests = _write_requests(tmp_path, "job_crash.json", _payload())

    first = instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "crash1_out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )
    assert first["status"] == "error"
    assert first["consecutive_job_crashes"] == 1
    assert instance._should_exit is False
    # _write_failure_output still ran, matching run_sam3dbody_batch.py's own
    # main() failure-path parity.
    failure_payload = json.loads((tmp_path / "crash1_out.json").read_text(encoding="utf-8"))
    assert failure_payload["status"] == "failed"

    second = instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "crash2_out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )
    assert second["status"] == "error"
    assert second["consecutive_job_crashes"] == 2
    assert second.get("worker_exiting") is True
    assert instance._should_exit is True


def test_successful_job_resets_consecutive_crash_counter(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch, max_consecutive_job_crashes=2)
    instance.bootstrap()
    job_requests = _write_requests(tmp_path, "job.json", _payload())

    monkeypatch.setattr(worker, "_run_batch_inference_and_write", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "out1.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )
    assert instance.consecutive_job_crashes == 1

    def _fake_inference(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        Path(kwargs["out_path"]).write_text(json.dumps({"frames": []}), encoding="utf-8")
        return [], {"mode": "fake"}

    monkeypatch.setattr(worker, "_run_batch_inference_and_write", _fake_inference)
    ok = instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "out2.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )
    assert ok["status"] == "ok"
    assert instance.consecutive_job_crashes == 0
    assert instance._should_exit is False


def test_worker_marks_unhealthy_and_refuses_job_on_cuda_canary_failure(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch)
    instance.bootstrap()
    monkeypatch.setattr(instance, "_cuda_canary_healthy", lambda: False)
    inference_calls: list[Any] = []
    monkeypatch.setattr(worker, "_run_batch_inference_and_write", lambda **kwargs: inference_calls.append(kwargs) or ([], {}))

    job_requests = _write_requests(tmp_path, "job.json", _payload())
    response = instance.handle_job(
        {
            "requests_path": str(job_requests),
            "out_path": str(tmp_path / "out.json"),
            "chunk_dir": None,
            "chunk_format": "pickle",
            "no_monolithic_output": True,
            "bucket_size": None,
        }
    )

    assert response["status"] == "worker_unhealthy"
    assert response["exit_code"] == 4
    assert instance.healthy is False
    assert instance._should_exit is True
    assert inference_calls == []


def test_worker_idle_timeout_self_exits_and_removes_socket(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    instance = _make_worker(tmp_path, monkeypatch, idle_timeout_s=0.2, accept_poll_s=0.05)
    instance.bootstrap()

    exit_summary = instance.serve_forever()

    assert exit_summary["reason"] == "idle_timeout"
    assert not Path(instance.socket_path).exists()
    exit_file = json.loads(Path(str(tmp_path / "ready.json") + ".exit.json").read_text(encoding="utf-8"))
    assert exit_file["reason"] == "idle_timeout"


# ---------------------------------------------------------------------------
# Client-side unavailability
# ---------------------------------------------------------------------------


def test_submit_job_via_client_reports_unavailable_when_socket_missing(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    exit_code = worker.submit_job_via_client(
        socket_path=str(tmp_path / "does_not_exist.sock"),
        requests_path=tmp_path / "requests.json",
        out_path=tmp_path / "out.json",
        chunk_dir=None,
        chunk_format="pickle",
        no_monolithic_output=True,
        bucket_size=None,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "unreachable" in captured.err


# ---------------------------------------------------------------------------
# run_sam3dbody_batch.py main() delegation shim
# ---------------------------------------------------------------------------


def test_main_is_unaffected_when_persistent_worker_env_var_is_unset(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.delenv("SAM3DBODY_PERSISTENT_WORKER_SOCKET", raising=False)
    calls: list[Any] = []
    monkeypatch.setattr(worker, "submit_job_via_client", lambda **kwargs: calls.append(kwargs) or 0)

    # A missing --fast-sam-repo/--checkpoint-dir is a real, pre-existing
    # config error (EX_CONFIG) on the normal (non-delegated) path -- proof
    # that the normal path still runs its own validation untouched.
    exit_code = batch.main(
        [
            "--requests",
            str(tmp_path / "missing_requests.json"),
            "--out",
            str(tmp_path / "out.json"),
            "--fast-sam-repo",
            str(tmp_path / "missing_fast_sam_repo"),
            "--checkpoint-dir",
            str(tmp_path / "missing_checkpoint_dir"),
        ]
    )

    assert calls == []
    assert exit_code == batch.EX_CONFIG


def test_main_delegates_to_persistent_worker_client_when_env_var_set(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    socket_path = str(tmp_path / "worker.sock")
    monkeypatch.setenv("SAM3DBODY_PERSISTENT_WORKER_SOCKET", socket_path)
    calls: list[dict[str, Any]] = []

    def _fake_submit(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(worker, "submit_job_via_client", _fake_submit)

    requests_path = tmp_path / "requests.json"  # need not exist: main() forwards the path unread
    out_path = tmp_path / "out.json"
    exit_code = batch.main(
        [
            "--requests",
            str(requests_path),
            "--out",
            str(out_path),
            "--fast-sam-repo",
            str(tmp_path / "fast_sam_repo"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint_dir"),
            "--chunk-format",
            "pickle",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["socket_path"] == socket_path
    assert calls[0]["requests_path"] == requests_path
    assert calls[0]["out_path"] == out_path
