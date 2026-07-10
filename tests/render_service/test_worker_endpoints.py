"""Worker HTTP surface (INFRA-2), via TestClient against the full accounts
app (`create_app(accounts_enabled=True, queue_enabled=True)`): bearer auth,
next-job claim/empty-queue, heartbeat, complete, and confirming
`get_job_v2` triggers the shared stale-sweep.
"""

import time
from pathlib import Path

import boto3
import mongomock
from fastapi.testclient import TestClient
from moto import mock_aws

from server.render_app import create_app

BUCKET = "test-bucket"
JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
WORKER_TOKEN = "test-worker-bearer-token"
PASSWORD = "correct-horse-battery"
MB = 1024 * 1024


def _env() -> dict[str, str]:
    return {
        "PICKLEBALL_JWT_SECRET": JWT_SECRET,
        "PICKLEBALL_INVITE_CODE": INVITE_CODE,
        "PICKLEBALL_S3_BUCKET": BUCKET,
        "PICKLEBALL_WORKER_BEARER_TOKEN": WORKER_TOKEN,
    }


def _make_app(tmp_path: Path, *, queue_enabled: bool = True):
    s3_client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    s3_client.create_bucket(Bucket=BUCKET)
    db = mongomock.MongoClient()["pickleball"]
    app = create_app(
        upload_root=tmp_path,
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=db,
        s3_client=s3_client,
        accounts_enabled=True,
        queue_enabled=queue_enabled,
        env=_env(),
    )
    return TestClient(app, base_url="https://testserver"), db, s3_client


def _worker_auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {WORKER_TOKEN}"}


def _register_and_login(client: TestClient, email: str = "worker-tests@example.com") -> str:
    registered = client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD, "invite_code": INVITE_CODE}
    )
    assert registered.status_code == 201, registered.text
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


def _queued_job(client: TestClient, s3_client) -> tuple[dict, str]:
    token = _register_and_login(client)
    created = client.post(
        "/api/clips",
        json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert created.status_code == 201, created.text
    clip = created.json()
    s3_client.put_object(Bucket=BUCKET, Key=clip["key"], Body=b"0123456789")
    accepted = client.post(
        "/api/jobs", json={"clip_id": clip["id"]}, headers={"Authorization": f"Bearer {token}"}
    )
    assert accepted.status_code == 202, accepted.text
    return accepted.json(), token


def test_next_job_missing_bearer_403(tmp_path: Path) -> None:
    with mock_aws():
        client, _, _ = _make_app(tmp_path)

        response = client.get("/api/worker/next-job", params={"wait_s": 0})

        assert response.status_code == 403


def test_next_job_wrong_bearer_403(tmp_path: Path) -> None:
    with mock_aws():
        client, _, _ = _make_app(tmp_path)

        response = client.get(
            "/api/worker/next-job", params={"wait_s": 0}, headers={"Authorization": "Bearer wrong-token"}
        )

        assert response.status_code == 403


def test_heartbeat_and_complete_also_require_worker_bearer(tmp_path: Path) -> None:
    with mock_aws():
        client, _, _ = _make_app(tmp_path)

        assert (
            client.post("/api/worker/jobs/job_x/heartbeat", json={"stage": "x", "percent": 1}).status_code == 403
        )
        assert client.post("/api/worker/jobs/job_x/complete", json={"status": "succeeded"}).status_code == 403


def test_next_job_empty_queue_returns_204(tmp_path: Path) -> None:
    with mock_aws():
        client, _, _ = _make_app(tmp_path)

        response = client.get("/api/worker/next-job", params={"wait_s": 0}, headers=_worker_auth())

        assert response.status_code == 204


def test_next_job_claims_queued_job(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make_app(tmp_path)
        job, _user_token = _queued_job(client, s3_client)

        response = client.get(
            "/api/worker/next-job",
            params={"wait_s": 0},
            headers={**_worker_auth(), "X-Worker-Id": "vm-1"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"] == job["id"]
        assert payload["clip_id"] == job["clip_id"]
        assert payload["attempts"] == 1
        assert payload["s3_raw_key"] == job["s3"]["video_key"]
        assert payload["video_filename"] == "drill.mp4"

        doc = db.jobs.find_one({"_id": job["id"]})
        assert doc["status"] == "claimed"
        assert doc["worker_id"] == "vm-1"


def test_heartbeat_updates_progress_visible_via_get_job(tmp_path: Path) -> None:
    with mock_aws():
        client, _db, s3_client = _make_app(tmp_path)
        job, user_token = _queued_job(client, s3_client)
        claim = client.get("/api/worker/next-job", params={"wait_s": 0}, headers=_worker_auth())
        assert claim.status_code == 200

        heartbeat = client.post(
            f"/api/worker/jobs/{job['id']}/heartbeat",
            json={"stage": "Running pipeline on GPU", "percent": 42, "message": "tracking + body active"},
            headers=_worker_auth(),
        )
        assert heartbeat.status_code == 204
        assert heartbeat.content == b""

        status = client.get(f"/api/jobs/{job['id']}", headers={"Authorization": f"Bearer {user_token}"}).json()
        assert status["status"] == "running"
        assert status["progress"]["percent"] == 42
        assert status["progress"]["stage"] == "Running pipeline on GPU"


def test_heartbeat_unknown_job_404(tmp_path: Path) -> None:
    with mock_aws():
        client, _, _ = _make_app(tmp_path)

        response = client.post(
            "/api/worker/jobs/job_doesnotexist/heartbeat",
            json={"stage": "x", "percent": 1},
            headers=_worker_auth(),
        )

        assert response.status_code == 404


def test_legacy_exit_success_marks_job_partial_with_result(tmp_path: Path) -> None:
    # Original intent: worker completion persists its prefixes/stage summary.
    # NS-01.5 retires exit-success=>complete, so absent bundle-policy evidence
    # is terminal partial with an explicit missing-capabilities payload.
    with mock_aws():
        client, _db, s3_client = _make_app(tmp_path)
        job, user_token = _queued_job(client, s3_client)
        client.get("/api/worker/next-job", params={"wait_s": 0}, headers=_worker_auth())

        complete = client.post(
            f"/api/worker/jobs/{job['id']}/complete",
            json={
                "status": "succeeded",
                "pipeline_stage_summary": [{"stage": "ingest", "wall_seconds": 1.0, "status": "complete"}],
                "s3_artifacts_prefix": f"artifacts/{job['id']}/",
                "s3_bundle_prefix": f"bundles/{job['clip_id']}/",
                "missing_capabilities": [
                    {
                        "capability": "bundle_policy",
                        "reason": "legacy worker supplied exit success without minimum-bundle evidence",
                    }
                ],
            },
            headers=_worker_auth(),
        )
        assert complete.status_code == 200

        final = client.get(f"/api/jobs/{job['id']}", headers={"Authorization": f"Bearer {user_token}"}).json()
        assert final["status"] == "partial"
        assert final["progress"]["percent"] == 100
        assert final["progress"]["eta_seconds"] == 0
        assert final["progress"]["stage"] == "Partial result"
        assert final["missing_capabilities"] == [
            {
                "capability": "bundle_policy",
                "reason": "legacy worker supplied exit success without minimum-bundle evidence",
            }
        ]
        assert final["result"]["s3_bundle_prefix"] == f"bundles/{job['clip_id']}/"
        assert final["result"]["s3_artifacts_prefix"] == f"artifacts/{job['id']}/"
        assert final["result"]["pipeline_stage_summary"][0]["stage"] == "ingest"


def test_complete_failed_marks_job_failed_with_error(tmp_path: Path) -> None:
    with mock_aws():
        client, _db, s3_client = _make_app(tmp_path)
        job, user_token = _queued_job(client, s3_client)
        client.get("/api/worker/next-job", params={"wait_s": 0}, headers=_worker_auth())

        complete = client.post(
            f"/api/worker/jobs/{job['id']}/complete",
            json={"status": "failed", "error": "process_video exit 1: CUDA OOM"},
            headers=_worker_auth(),
        )
        assert complete.status_code == 200

        final = client.get(f"/api/jobs/{job['id']}", headers={"Authorization": f"Bearer {user_token}"}).json()
        assert final["status"] == "failed"
        assert final["error"] == "process_video exit 1: CUDA OOM"
        assert final["progress"]["stage"] == "Failed"
        assert final["progress"]["eta_seconds"] is None
        assert final["result"] is None


def test_get_job_triggers_stale_sweep(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make_app(tmp_path)
        job, user_token = _queued_job(client, s3_client)
        claim = client.get("/api/worker/next-job", params={"wait_s": 0}, headers=_worker_auth())
        assert claim.status_code == 200
        assert db.jobs.find_one({"_id": job["id"]})["status"] == "claimed"

        # Simulate a dead worker: push heartbeat_at into the stale past
        # directly in Mongo -- nothing else will flip status back except
        # the shared sweep that get_job_v2 must trigger.
        db.jobs.update_one({"_id": job["id"]}, {"$set": {"heartbeat_at": time.time() - 400}})

        status = client.get(
            f"/api/jobs/{job['id']}", headers={"Authorization": f"Bearer {user_token}"}
        ).json()

        assert status["status"] == "queued"  # attempts==1 < ceiling -> requeued, not failed
