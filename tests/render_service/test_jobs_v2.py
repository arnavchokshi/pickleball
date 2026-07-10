from pathlib import Path

import boto3
import mongomock
import requests
from fastapi.testclient import TestClient
from moto import mock_aws

from server.gpu_runner import GpuRunRequest, GpuRunResult
from server.render_app import create_app
from tests.render_service.ns015_bundle_fixture import write_minimum_bundle

BUCKET = "test-bucket"
JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
PASSWORD = "correct-horse-battery"
MB = 1024 * 1024
VIDEO_BYTES = b"fake-video-bytes"


def _accounts_env() -> dict[str, str]:
    return {
        "PICKLEBALL_JWT_SECRET": JWT_SECRET,
        "PICKLEBALL_INVITE_CODE": INVITE_CODE,
        "PICKLEBALL_S3_BUCKET": BUCKET,
    }


class CompletingRunner:
    name = "test-completing"

    def __init__(self) -> None:
        self.requests: list[GpuRunRequest] = []

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        self.requests.append(request)
        write_minimum_bundle(request.artifacts_dir, video_path=request.video_path)
        manifest = request.artifacts_dir / "replay_viewer_manifest.json"
        return GpuRunResult(
            status="complete",
            notes=["fake runner complete"],
            artifacts_dir=request.artifacts_dir,
            manifest_path=manifest,
        )


class FailingRunner:
    name = "test-failing"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        raise RuntimeError("gpu unavailable")


def _make_accounts_app(tmp_path: Path, runner=None):
    s3_client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3_client.create_bucket(Bucket=BUCKET)
    app = create_app(
        upload_root=tmp_path,
        runner=runner if runner is not None else CompletingRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=mongomock.MongoClient()["pickleball"],
        s3_client=s3_client,
        accounts_enabled=True,
        env=_accounts_env(),
    )
    return TestClient(app, base_url="https://testserver"), s3_client


def _register_and_login(client: TestClient, email: str = "jobs@example.com") -> str:
    registered = client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "invite_code": INVITE_CODE},
    )
    assert registered.status_code == 201, registered.text
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _uploaded_clip(client: TestClient, s3_client, token: str, *, with_sidecar: bool = True) -> dict:
    created = client.post(
        "/api/clips",
        json={"filename": "drill.mp4", "size_bytes": len(VIDEO_BYTES), "part_size_bytes": 5 * MB},
        headers=_auth(token),
    )
    assert created.status_code == 201, created.text
    clip = created.json()
    s3_client.put_object(Bucket=BUCKET, Key=clip["key"], Body=VIDEO_BYTES)
    if with_sidecar:
        put = requests.put(
            clip["sidecar_upload_url"], data=b"{}", headers={"Content-Type": "application/json"}
        )
        assert put.status_code == 200, put.text
    return clip


def test_flag_off_keeps_legacy_multipart_jobs_and_hides_account_routes(tmp_path: Path) -> None:
    # Original intent: flag-off preserves the legacy route surface. Its fake
    # runner now earns complete with the same minimum bundle as production.
    app = create_app(
        upload_root=tmp_path,
        runner=CompletingRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        env={},  # PICKLEBALL_ACCOUNTS_ENABLED defaults to "0"
    )
    client = TestClient(app)

    legacy = client.post(
        "/api/jobs",
        data={"clip": "drill_01"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )
    assert legacy.status_code == 202
    assert client.get(legacy.json()["links"]["status"]).json()["status"] == "complete"

    # None of the account-era routes exist with the flag off.
    assert (
        client.post(
            "/api/auth/register",
            json={"email": "a@b.com", "password": PASSWORD, "invite_code": INVITE_CODE},
        ).status_code
        == 404
    )
    assert client.post("/api/auth/login", json={"email": "a@b.com", "password": PASSWORD}).status_code == 404
    assert client.get("/api/clips").status_code == 404
    assert client.delete("/api/account").status_code == 404
    assert client.post("/api/stripe/webhook", json={}).status_code == 404

    # Legacy health shape is byte-identical: no mongo/s3/accounts fields.
    health = client.get("/api/health").json()
    assert set(health.keys()) == {"ok", "runner"}


def test_flag_on_drops_legacy_multipart_and_serves_account_routes(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make_accounts_app(tmp_path)
        token = _register_and_login(client)

        # The legacy multipart form intake is gone: the JSON-body v2 endpoint
        # rejects a multipart upload as unprocessable instead of 202-ing it.
        legacy_shaped = client.post(
            "/api/jobs",
            data={"clip": "drill_01"},
            files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
            headers=_auth(token),
        )
        assert legacy_shaped.status_code == 422

        # New routes are live.
        assert client.get("/api/clips", headers=_auth(token)).status_code == 200
        # The v2 job GET (JWT-gated) matches instead of the legacy file-store GET.
        assert client.get("/api/jobs/job_doesnotexist").status_code == 401
        health = client.get("/api/health").json()
        assert health["accounts_enabled"] is True
        assert health["mongo"]["ok"] is True
        assert health["s3"]["ok"] is True


def test_job_flow_pulls_clip_from_s3_and_completes_with_progress(tmp_path: Path) -> None:
    # Original intent: the account-era job pulls the exact S3 inputs and ends
    # ready. Complete is now backed by mandatory artifacts and valid URLs.
    with mock_aws():
        runner = CompletingRunner()
        client, s3_client = _make_accounts_app(tmp_path, runner=runner)
        token = _register_and_login(client)
        clip = _uploaded_clip(client, s3_client, token)

        accepted = client.post(
            "/api/jobs", json={"clip_id": clip["id"], "max_frames": 8}, headers=_auth(token)
        )
        assert accepted.status_code == 202, accepted.text
        job = accepted.json()
        assert job["status"] == "queued"
        assert job["clip_id"] == clip["id"]
        assert job["attempts"] == 0
        assert job["links"]["status"] == f"/api/jobs/{job['id']}"
        assert job["s3"]["video_key"] == clip["key"]

        status = client.get(job["links"]["status"], headers=_auth(token)).json()
        assert status["status"] == "complete"
        assert status["progress"]["percent"] == 100
        assert status["progress"]["stage"] == "Replay ready"
        assert status["progress"]["eta_seconds"] == 0
        assert any(
            step["id"] == "gpu_pipeline" and step["status"] == "complete"
            for step in status["progress"]["steps"]
        )
        assert status["result"]["manifest_url"].endswith("/manifest")

        request = runner.requests[0]
        assert request.clip == "drill"
        assert request.max_frames == 8
        assert request.video_path.read_bytes() == VIDEO_BYTES
        assert request.capture_sidecar_path is not None
        assert request.capture_sidecar_path.read_text(encoding="utf-8") == "{}"


def test_jobs_require_jwt_401(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make_accounts_app(tmp_path)

        assert client.post("/api/jobs", json={"clip_id": "clip_x"}).status_code == 401
        assert client.get("/api/jobs/job_x").status_code == 401


def test_jobs_are_owner_scoped_404_for_others(tmp_path: Path) -> None:
    with mock_aws():
        client, s3_client = _make_accounts_app(tmp_path)
        owner_token = _register_and_login(client, email="owner@example.com")
        other_token = _register_and_login(client, email="other@example.com")
        clip = _uploaded_clip(client, s3_client, owner_token, with_sidecar=False)

        accepted = client.post(
            "/api/jobs", json={"clip_id": clip["id"]}, headers=_auth(owner_token)
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["id"]

        assert client.get(f"/api/jobs/{job_id}", headers=_auth(owner_token)).status_code == 200
        assert client.get(f"/api/jobs/{job_id}", headers=_auth(other_token)).status_code == 404
        # Other users cannot start jobs on clips they do not own either.
        assert (
            client.post(
                "/api/jobs", json={"clip_id": clip["id"]}, headers=_auth(other_token)
            ).status_code
            == 404
        )


def test_failed_runner_marks_job_failed(tmp_path: Path) -> None:
    with mock_aws():
        client, s3_client = _make_accounts_app(tmp_path, runner=FailingRunner())
        token = _register_and_login(client)
        clip = _uploaded_clip(client, s3_client, token, with_sidecar=False)

        accepted = client.post("/api/jobs", json={"clip_id": clip["id"]}, headers=_auth(token))
        assert accepted.status_code == 202

        status = client.get(accepted.json()["links"]["status"], headers=_auth(token)).json()
        assert status["status"] == "failed"
        assert "gpu unavailable" in status["error"]
        assert status["progress"]["stage"] == "Failed"
        assert status["progress"]["eta_seconds"] is None


def test_missing_s3_object_marks_job_failed(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make_accounts_app(tmp_path)
        token = _register_and_login(client)
        created = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )
        clip = created.json()  # nothing uploaded to S3

        accepted = client.post("/api/jobs", json={"clip_id": clip["id"]}, headers=_auth(token))
        assert accepted.status_code == 202

        status = client.get(accepted.json()["links"]["status"], headers=_auth(token)).json()
        assert status["status"] == "failed"
        assert "input download failed" in status["error"]
