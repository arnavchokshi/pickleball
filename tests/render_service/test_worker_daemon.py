"""`server/worker/daemon.py` (INFRA-2): `run_once` against FAKE api_client +
s3_client + process_runner -- no real subprocess/GPU/S3/network, per the
lane fence -- plus the `--help`/`--check-config` CLI smoke test.
"""

import subprocess
import sys
from pathlib import Path

from server.worker.config import WorkerConfig
from server.worker.daemon import RunResult, run_once

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeApiClient:
    def __init__(self, job_payload: dict | None) -> None:
        self._job_payload = job_payload
        self._claimed = False
        self.heartbeats: list[dict] = []
        self.completed: list[dict] = []

    def claim_next_job(self) -> dict | None:
        if self._claimed or self._job_payload is None:
            return None
        self._claimed = True
        return self._job_payload

    def send_heartbeat(self, job_id: str, *, stage: str, percent: int, message: str) -> None:
        self.heartbeats.append({"job_id": job_id, "stage": stage, "percent": percent, "message": message})

    def complete_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        pipeline_stage_summary=None,
        s3_artifacts_prefix: str | None = None,
        s3_bundle_prefix: str | None = None,
    ) -> None:
        self.completed.append(
            {
                "job_id": job_id,
                "status": status,
                "error": error,
                "pipeline_stage_summary": pipeline_stage_summary,
                "s3_artifacts_prefix": s3_artifacts_prefix,
                "s3_bundle_prefix": s3_bundle_prefix,
            }
        )


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = dict(objects)
        self.uploaded: dict[str, bytes] = {}
        self.downloaded: list[tuple[str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloaded.append((bucket, key))
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self._objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploaded[key] = Path(filename).read_bytes()


def _config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        api_base_url="https://example.invalid",
        worker_bearer_token="tok",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        s3_bucket="test-bucket",
        s3_region="us-east-1",
        pipeline_python="/repo/.venv/bin/python",
        repo_dir="/repo",
        worker_id="vm-1",
        poll_wait_s=0,
        heartbeat_interval_s=9999,  # long enough it never fires mid-test
        command_timeout_s=60,
        work_dir=str(tmp_path / "worker_jobs"),
    )


def _job_payload() -> dict:
    return {
        "job_id": "job_1",
        "clip_id": "clip_1",
        "s3_raw_key": "raw/user_1/clip_1/drill.mp4",
        "s3_sidecar_key": "raw/user_1/clip_1/capture_sidecar.json",
        "video_filename": "drill.mp4",
        "max_frames": 8,
        "attempts": 1,
    }


def _s3_with_raw_inputs() -> FakeS3Client:
    return FakeS3Client(
        {
            "raw/user_1/clip_1/drill.mp4": b"video-bytes",
            "raw/user_1/clip_1/capture_sidecar.json": b"{}",
        }
    )


def test_run_once_returns_false_on_empty_queue(tmp_path: Path) -> None:
    api_client = FakeApiClient(job_payload=None)
    s3_client = FakeS3Client({})

    def process_runner(job, video_path, sidecar_path, out_dir):
        raise AssertionError("process_runner must not be called when the queue is empty")

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is False
    assert api_client.completed == []
    assert api_client.heartbeats == []


def test_run_once_success_downloads_runs_uploads_and_completes(tmp_path: Path) -> None:
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()
    calls = []

    def process_runner(job, video_path, sidecar_path, out_dir):
        calls.append((job, video_path, sidecar_path, out_dir))
        assert video_path.read_bytes() == b"video-bytes"
        assert sidecar_path is not None
        assert sidecar_path.read_text(encoding="utf-8") == "{}"
        clip_dir = out_dir / job.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "replay_viewer_manifest.json").write_text(
            '{"video_url": "/@fs//tmp/x/input/drill.mp4"}', encoding="utf-8"
        )
        (clip_dir / "PIPELINE_SUMMARY.json").write_text(
            '{"stages":[{"stage":"ingest","wall_seconds":1.0,"status":"complete"}]}', encoding="utf-8"
        )
        return RunResult(status="succeeded", pipeline_stage_summary=[{"stage": "ingest", "wall_seconds": 1.0}])

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert len(calls) == 1
    assert len(api_client.heartbeats) >= 1
    assert len(api_client.completed) == 1
    completed = api_client.completed[0]
    assert completed["job_id"] == "job_1"
    assert completed["status"] == "succeeded"
    assert completed["s3_bundle_prefix"] == "bundles/clip_1/"
    assert completed["s3_artifacts_prefix"] == "artifacts/job_1/"
    assert completed["pipeline_stage_summary"] == [{"stage": "ingest", "wall_seconds": 1.0}]

    assert any(key.startswith("bundles/clip_1/") for key in s3_client.uploaded)
    assert any(key.startswith("artifacts/job_1/") for key in s3_client.uploaded)
    assert "bundles/clip_1/source.mp4" in s3_client.uploaded
    assert s3_client.uploaded["bundles/clip_1/source.mp4"] == b"video-bytes"
    uploaded_manifest = s3_client.uploaded["bundles/clip_1/replay_viewer_manifest.json"]
    assert b"bundles/clip_1/source.mp4" in uploaded_manifest


def test_run_once_without_sidecar_key_skips_sidecar_download(tmp_path: Path) -> None:
    payload = _job_payload()
    payload["s3_sidecar_key"] = None
    api_client = FakeApiClient(job_payload=payload)
    s3_client = FakeS3Client({"raw/user_1/clip_1/drill.mp4": b"video-bytes"})

    def process_runner(job, video_path, sidecar_path, out_dir):
        assert sidecar_path is None
        clip_dir = out_dir / job.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        return RunResult(status="succeeded")

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "succeeded"


def test_run_once_failure_result_completes_with_failed_status(tmp_path: Path) -> None:
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()

    def process_runner(job, video_path, sidecar_path, out_dir):
        return RunResult(status="failed", error="process_video exit 1")

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "failed"
    assert api_client.completed[0]["error"] == "process_video exit 1"
    assert s3_client.uploaded == {}


def test_run_once_raising_process_runner_completes_with_failed_status(tmp_path: Path) -> None:
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()

    def process_runner(job, video_path, sidecar_path, out_dir):
        raise RuntimeError("gpu OOM")

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "failed"
    assert "gpu OOM" in api_client.completed[0]["error"]
    assert s3_client.uploaded == {}


def test_run_once_raw_download_failure_fails_job_without_crashing(tmp_path: Path) -> None:
    # A bad/missing raw S3 key must fail THIS job (not raise out of run_once
    # into main()'s loop, which would crash the worker and block the queue).
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = FakeS3Client({})  # raw key absent -> download_file raises KeyError

    def process_runner(job, video_path, sidecar_path, out_dir):
        raise AssertionError("process_runner must not run when input download fails")

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "failed"
    assert "input download failed" in api_client.completed[0]["error"]
    assert s3_client.uploaded == {}


def test_daemon_help_exits_zero_without_network() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "server.worker.daemon", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "worker daemon" in completed.stdout.lower() or "worker daemon" in completed.stderr.lower()


def test_daemon_check_config_exits_zero_without_network() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "server.worker.daemon", "--check-config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "worker config ok" in completed.stdout
