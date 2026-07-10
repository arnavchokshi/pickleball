"""`server/worker/daemon.py` (INFRA-2): `run_once` against FAKE api_client +
s3_client + process_runner -- no real subprocess/GPU/S3/network, per the
lane fence -- plus the `--help`/`--check-config` CLI smoke test.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import server.worker.daemon as worker_daemon
from server.worker.config import WorkerConfig
from server.worker.daemon import RunResult, run_once
from tests.render_service.ns015_bundle_fixture import write_minimum_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeApiClient:
    def __init__(self, job_payload: dict | None) -> None:
        self._job_payload = job_payload
        self._claimed = False
        self.heartbeats: list[dict] = []
        self.completed: list[dict] = []
        self.raise_after_successful_complete = False

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
        missing_capabilities=None,
        trust_bands=None,
        bundle_policy=None,
    ) -> None:
        self.completed.append(
            {
                "job_id": job_id,
                "status": status,
                "error": error,
                "pipeline_stage_summary": pipeline_stage_summary,
                "s3_artifacts_prefix": s3_artifacts_prefix,
                "s3_bundle_prefix": s3_bundle_prefix,
                "missing_capabilities": missing_capabilities,
                "trust_bands": trust_bands,
                "bundle_policy": bundle_policy,
            }
        )
        if status == "complete" and self.raise_after_successful_complete:
            raise TimeoutError("completion response lost")


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = dict(objects)
        self.uploaded: dict[str, bytes] = {}
        self.downloaded: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.fail_upload_at: int | None = None
        self.upload_attempt_count = 0
        self.return_delete_errors = False

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloaded.append((bucket, key))
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self._objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.upload_attempt_count += 1
        if self.fail_upload_at == self.upload_attempt_count:
            raise OSError("injected upload failure")
        self.uploaded[key] = Path(filename).read_bytes()

    def list_objects_v2(self, **kwargs) -> dict:
        prefix = str(kwargs["Prefix"])
        keys = sorted(key for key in self.uploaded if key.startswith(prefix))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def delete_objects(self, **kwargs) -> dict:
        if self.return_delete_errors:
            return {"Errors": [{"Key": kwargs["Delete"]["Objects"][0]["Key"], "Code": "AccessDenied"}]}
        for item in kwargs["Delete"]["Objects"]:
            key = str(item["Key"])
            self.uploaded.pop(key, None)
            self.deleted.append(key)
        return {"Deleted": [{"Key": key} for key in self.deleted]}


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
    # Original intent: a genuinely complete run downloads inputs, publishes
    # immutable generations, and reports complete without deleting old ones.
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()
    old_artifact_key = "artifacts/job_1/generations/old/obsolete.json"
    old_bundle_key = "bundles/clip_1/jobs/job_1/generations/old/obsolete.bin"
    s3_client.uploaded[old_artifact_key] = b"old-artifact"
    s3_client.uploaded[old_bundle_key] = b"old-bundle"
    calls = []

    def process_runner(job, video_path, sidecar_path, out_dir):
        calls.append((job, video_path, sidecar_path, out_dir))
        assert video_path.read_bytes() == b"video-bytes"
        assert sidecar_path is not None
        assert sidecar_path.read_text(encoding="utf-8") == "{}"
        summary = write_minimum_bundle(
            out_dir / job.clip_id,
            video_path=video_path,
        )
        return RunResult(
            status="succeeded",
            pipeline_stage_summary=summary["stages"],
            pipeline_summary=summary,
        )

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert len(calls) == 1
    assert len(api_client.heartbeats) >= 1
    assert len(api_client.completed) == 1
    completed = api_client.completed[0]
    assert completed["job_id"] == "job_1"
    assert completed["status"] == "complete"
    bundle_prefix = completed["s3_bundle_prefix"]
    artifacts_prefix = completed["s3_artifacts_prefix"]
    assert bundle_prefix.startswith("bundles/clip_1/jobs/job_1/generations/")
    assert artifacts_prefix.startswith("artifacts/job_1/generations/")
    assert completed["pipeline_stage_summary"] == [
        {"stage": "manifest", "status": "ran", "wall_seconds": 1.0}
    ]
    assert completed["missing_capabilities"] == []
    assert completed["bundle_policy"]["status"] == "complete"

    assert any(key.startswith("bundles/clip_1/") for key in s3_client.uploaded)
    assert any(key.startswith("artifacts/job_1/") for key in s3_client.uploaded)
    assert f"{bundle_prefix}source.mp4" in s3_client.uploaded
    assert s3_client.uploaded[f"{bundle_prefix}source.mp4"] == b"video-bytes"
    uploaded_manifest = s3_client.uploaded[f"{bundle_prefix}replay_viewer_manifest.json"]
    assert f"{bundle_prefix}source.mp4".encode() in uploaded_manifest
    assert f"{artifacts_prefix}clip_1/PIPELINE_SUMMARY.json" in s3_client.uploaded
    assert f"{bundle_prefix}PIPELINE_SUMMARY.json" in s3_client.uploaded
    # Superseded immutable generations are retained until a state-aware
    # lifecycle/cleanup pass; the worker never races deletion against publish.
    assert s3_client.uploaded[old_artifact_key] == b"old-artifact"
    assert s3_client.uploaded[old_bundle_key] == b"old-bundle"


def test_run_once_without_sidecar_key_skips_sidecar_download(tmp_path: Path) -> None:
    # Original intent: no sidecar key means no sidecar download. Under NS-01.5
    # that bare execution success is inspectable partial, never complete.
    payload = _job_payload()
    payload["s3_sidecar_key"] = None
    api_client = FakeApiClient(job_payload=payload)
    s3_client = FakeS3Client({"raw/user_1/clip_1/drill.mp4": b"video-bytes"})

    def process_runner(job, video_path, sidecar_path, out_dir):
        assert sidecar_path is None
        missing = [{"capability": "capture_sidecar", "reason": "sidecar key was absent"}]
        summary = write_minimum_bundle(
            out_dir / job.clip_id,
            video_path=video_path,
            status="partial",
            missing_capabilities=missing,
            omit="capture_sidecar",
        )
        return RunResult(status="succeeded", pipeline_summary=summary)

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "partial"
    assert api_client.completed[0]["missing_capabilities"] == [
        {"capability": "capture_sidecar", "reason": "sidecar key was absent"}
    ]


def test_run_once_missing_advertised_url_publishes_partial_without_ready(tmp_path: Path) -> None:
    # Original intent: a broken advertised URL must never reach ready. The new
    # contract retains an inspectable immutable bundle and reports partial.
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()

    def process_runner(job, video_path, sidecar_path, out_dir):
        summary = write_minimum_bundle(
            out_dir / job.clip_id,
            video_path=video_path,
            missing_extra_url=True,
        )
        return RunResult(status="succeeded", pipeline_summary=summary)

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    completed = api_client.completed[0]
    assert completed["status"] == "partial"
    assert completed["bundle_policy"]["missing_urls"] == ["assets/missing_overlay.png"]
    assert any(item["capability"] == "advertised_urls" for item in completed["missing_capabilities"])
    assert any(key.startswith(completed["s3_bundle_prefix"]) for key in s3_client.uploaded)


def test_run_once_failed_generation_upload_preserves_previous_live_bundle(tmp_path: Path) -> None:
    # Original intent: upload failure cannot disturb the previous immutable
    # generation. The candidate fixture is complete so it reaches upload.
    api_client = FakeApiClient(job_payload=_job_payload())
    s3_client = _s3_with_raw_inputs()
    old_key = "bundles/clip_1/jobs/job_1/generations/old/replay_viewer_manifest.json"
    s3_client.uploaded[old_key] = b'{"generation":"old"}'
    s3_client.fail_upload_at = 2

    def process_runner(job, video_path, sidecar_path, out_dir):
        summary = write_minimum_bundle(out_dir / job.clip_id, video_path=video_path)
        return RunResult(status="succeeded", pipeline_summary=summary)

    processed = run_once(api_client, s3_client, process_runner, _config(tmp_path))

    assert processed is True
    assert api_client.completed[0]["status"] == "failed"
    assert "injected upload failure" in api_client.completed[0]["error"]
    assert s3_client.uploaded == {old_key: b'{"generation":"old"}'}


def test_delete_s3_keys_rejects_per_object_errors() -> None:
    s3_client = FakeS3Client({})
    s3_client.return_delete_errors = True

    with pytest.raises(RuntimeError, match="AccessDenied"):
        worker_daemon._delete_s3_keys(s3_client, bucket="bucket", keys={"bundles/clip/file.json"})


def test_run_once_lost_completion_response_keeps_published_generation(tmp_path: Path) -> None:
    # Original intent: a lost API response after honest completion must not
    # delete the already-published immutable generation.
    api_client = FakeApiClient(job_payload=_job_payload())
    api_client.raise_after_successful_complete = True
    s3_client = _s3_with_raw_inputs()

    def process_runner(job, video_path, sidecar_path, out_dir):
        summary = write_minimum_bundle(out_dir / job.clip_id, video_path=video_path)
        return RunResult(status="succeeded", pipeline_summary=summary)

    with pytest.raises(TimeoutError, match="response lost"):
        run_once(api_client, s3_client, process_runner, _config(tmp_path))

    completed = api_client.completed[-1]
    assert completed["status"] == "complete"
    assert any(key.startswith(completed["s3_bundle_prefix"]) for key in s3_client.uploaded)
    assert any(key.startswith(completed["s3_artifacts_prefix"]) for key in s3_client.uploaded)


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
