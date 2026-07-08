"""INFRA-5 delete-cascade tests: `DELETE /api/account` and
`DELETE /api/clips/{id}`, via moto (S3) + mongomock (Mongo) + create_app(DI),
matching the conventions in test_clips.py / test_worker_endpoints.py.

Seeds a user with a clip (raw S3 object + sidecar), a queued job (artifacts
prefix), a completed-looking bundle prefix, an entitlement doc, and a
profile-registry doc -- then proves the cascade deletes every S3 object and
every derived Mongo doc, wrong password mutates nothing, and per-clip delete
is correctly scoped (siblings + the user doc survive).
"""

from pathlib import Path

import boto3
import mongomock
import requests
from fastapi.testclient import TestClient
from moto import mock_aws

from server.profile_store import ProfileStore
from server.render_app import create_app
from threed.racketsport.profile_registry import PlayerProfile, RetentionPolicy, SourceTrace

BUCKET = "test-bucket"
JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
WORKER_TOKEN = "test-worker-bearer-token"
PASSWORD = "correct-horse-battery"
WRONG_PASSWORD = "not-the-password-at-all"
MB = 1024 * 1024


def _env() -> dict[str, str]:
    return {
        "PICKLEBALL_JWT_SECRET": JWT_SECRET,
        "PICKLEBALL_INVITE_CODE": INVITE_CODE,
        "PICKLEBALL_S3_BUCKET": BUCKET,
        "PICKLEBALL_WORKER_BEARER_TOKEN": WORKER_TOKEN,
    }


class UnusedRunner:
    name = "test-unused"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request):  # noqa: ANN001 - never called: jobs stay queued (queue_enabled=True)
        raise AssertionError("runner must not execute in delete-cascade tests")


def _make(tmp_path: Path, *, queue_enabled: bool = True):
    s3_client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    s3_client.create_bucket(Bucket=BUCKET)
    db = mongomock.MongoClient()["pickleball"]
    app = create_app(
        upload_root=tmp_path,
        runner=UnusedRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=db,
        s3_client=s3_client,
        accounts_enabled=True,
        queue_enabled=queue_enabled,
        env=_env(),
    )
    return TestClient(app, base_url="https://testserver"), db, s3_client


def _register_and_login(client: TestClient, email: str = "cascade@example.com") -> tuple[str, str]:
    registered = client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD, "invite_code": INVITE_CODE}
    )
    assert registered.status_code == 201, registered.text
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    return registered.json()["id"], login.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _upload_and_complete_clip(client: TestClient, token: str, *, filename: str = "drill.mp4") -> dict:
    body = b"0123456789"
    created = client.post(
        "/api/clips",
        json={"filename": filename, "size_bytes": len(body), "part_size_bytes": 5 * MB},
        headers=_auth(token),
    )
    assert created.status_code == 201, created.text
    clip = created.json()

    put = requests.put(clip["part_urls"][0]["url"], data=body)
    assert put.status_code == 200, put.text
    etag = put.headers["ETag"].strip('"')

    completed = client.post(
        f"/api/clips/{clip['id']}/complete",
        json={"upload_id": clip["upload_id"], "parts": [{"part_number": 1, "etag": etag}]},
        headers=_auth(token),
    )
    assert completed.status_code == 200, completed.text
    return clip


def _queue_job_for_clip(client: TestClient, token: str, clip_id: str) -> dict:
    accepted = client.post("/api/jobs", json={"clip_id": clip_id}, headers=_auth(token))
    assert accepted.status_code == 202, accepted.text
    return accepted.json()


def _seed_entitlement(db, user_id: str) -> None:
    db.entitlements.insert_one(
        {"user_id": user_id, "stripe_customer_id": "cus_test123", "status": "active", "created_at": 0, "updated_at": 0}
    )


def _seed_profile(db, user_id: str) -> None:
    profile = PlayerProfile(
        schema_version=1,
        artifact_type="racketsport_player_profile",
        account_id=user_id,
        profile_id="player_self",
        display_name="Account Owner",
        is_account_owner=True,
        height_m=1.80,
        height_provenance="self_reported",
        handedness="right",
        cross_account_shareable=False,
        consent_status="owner",
        source_trace=SourceTrace(source_clip_id="seed_clip"),
        retention=RetentionPolicy(
            scope="account_lifetime",
            delete_with_source_clip=False,
            delete_with_source_profile=False,
            legal_basis="owner_setup",
        ),
    )
    ProfileStore(db).update(user_id, profile)


def _list_keys(s3_client, prefix: str) -> list[str]:
    listed = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in listed.get("Contents", [])]


def _full_seed(client: TestClient, db, s3_client, *, email: str = "cascade@example.com"):
    """Seeds one user with a clip (S3 raw+sidecar), a queued job (S3
    artifacts), a bundle prefix, an entitlement, and a profile doc. Returns
    (user_id, token, clip, job)."""
    user_id, token = _register_and_login(client, email=email)
    clip = _upload_and_complete_clip(client, token)
    job = _queue_job_for_clip(client, token, clip["id"])

    # Simulate worker-produced artifacts/bundle output that would exist by
    # the time a real job completes.
    s3_client.put_object(Bucket=BUCKET, Key=f"artifacts/{job['id']}/pipeline_summary.json", Body=b"{}")
    s3_client.put_object(Bucket=BUCKET, Key=f"bundles/{clip['id']}/replay_viewer_manifest.json", Body=b"{}")

    _seed_entitlement(db, user_id)
    _seed_profile(db, user_id)

    return user_id, token, clip, job


# ---------------------------------------------------------------------------
# DELETE /api/account
# ---------------------------------------------------------------------------


def test_delete_account_requires_jwt_401(tmp_path: Path) -> None:
    with mock_aws():
        client, _db, _s3 = _make(tmp_path)
        assert client.request("DELETE", "/api/account", json={"password": PASSWORD}).status_code == 401


def test_delete_account_wrong_password_403_and_nothing_mutated(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make(tmp_path)
        user_id, token, clip, job = _full_seed(client, db, s3_client)

        response = client.request(
            "DELETE", "/api/account", json={"password": WRONG_PASSWORD}, headers=_auth(token)
        )
        assert response.status_code == 403

        # Nothing mutated: S3 objects and every Mongo doc survive untouched.
        assert _list_keys(s3_client, f"raw/{user_id}/") != []
        assert _list_keys(s3_client, f"artifacts/{job['id']}/") != []
        assert _list_keys(s3_client, f"bundles/{clip['id']}/") != []

        user_doc = db.users.find_one({"_id": user_id})
        assert user_doc is not None
        assert user_doc["deleted_at"] is None
        assert db.clips.find_one({"_id": clip["id"]}) is not None
        assert db.jobs.find_one({"_id": job["id"]}) is not None
        assert db.entitlements.find_one({"user_id": user_id}) is not None
        assert ProfileStore(db).load(user_id) is not None
        refresh_doc = db.refresh_tokens.find_one({"user_id": user_id})
        assert refresh_doc is not None
        assert refresh_doc["revoked_at"] is None


def test_delete_account_correct_password_cascades_s3_and_mongo(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make(tmp_path)
        user_id, token, clip, job = _full_seed(client, db, s3_client)

        response = client.request(
            "DELETE", "/api/account", json={"password": PASSWORD}, headers=_auth(token)
        )
        assert response.status_code == 204
        assert response.content == b""

        # Every S3 prefix this user's data could live under is empty.
        assert _list_keys(s3_client, f"raw/{user_id}/") == []
        assert _list_keys(s3_client, f"artifacts/{job['id']}/") == []
        assert _list_keys(s3_client, f"bundles/{clip['id']}/") == []

        user_doc = db.users.find_one({"_id": user_id})
        assert user_doc is not None
        assert user_doc["deleted_at"] is not None
        assert db.clips.find_one({"_id": clip["id"]}) is None
        assert db.jobs.find_one({"_id": job["id"]}) is None
        assert db.entitlements.find_one({"user_id": user_id}) is None
        assert ProfileStore(db).load(user_id) is None

        # Refresh chain revoked.
        refresh_doc = db.refresh_tokens.find_one({"user_id": user_id})
        assert refresh_doc is not None
        assert refresh_doc["revoked_at"] is not None


def test_delete_account_is_idempotent(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make(tmp_path)
        user_id, token, _clip, _job = _full_seed(client, db, s3_client)

        first = client.request("DELETE", "/api/account", json={"password": PASSWORD}, headers=_auth(token))
        assert first.status_code == 204

        second = client.request("DELETE", "/api/account", json={"password": PASSWORD}, headers=_auth(token))
        assert second.status_code == 204

        assert db.users.find_one({"_id": user_id})["deleted_at"] is not None


# ---------------------------------------------------------------------------
# DELETE /api/clips/{id}
# ---------------------------------------------------------------------------


def test_delete_clip_requires_jwt_401(tmp_path: Path) -> None:
    with mock_aws():
        client, _db, _s3 = _make(tmp_path)
        assert client.delete("/api/clips/clip_doesnotexist").status_code == 401


def test_delete_clip_unknown_or_other_users_clip_404(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make(tmp_path)
        _user_a, token_a, clip_a, _job_a = _full_seed(client, db, s3_client, email="owner@example.com")
        _user_b, token_b = _register_and_login(client, email="other@example.com")

        missing = client.delete("/api/clips/clip_doesnotexist", headers=_auth(token_a))
        assert missing.status_code == 404

        not_owner = client.delete(f"/api/clips/{clip_a['id']}", headers=_auth(token_b))
        assert not_owner.status_code == 404
        # Confirms the 404 above didn't delete anything.
        assert db.clips.find_one({"_id": clip_a["id"]}) is not None


def test_delete_clip_scoped_other_clips_and_user_untouched(tmp_path: Path) -> None:
    with mock_aws():
        client, db, s3_client = _make(tmp_path)
        user_id, token = _register_and_login(client)

        clip_a = _upload_and_complete_clip(client, token, filename="clip_a.mp4")
        job_a = _queue_job_for_clip(client, token, clip_a["id"])
        s3_client.put_object(Bucket=BUCKET, Key=f"artifacts/{job_a['id']}/pipeline_summary.json", Body=b"{}")
        s3_client.put_object(Bucket=BUCKET, Key=f"bundles/{clip_a['id']}/replay_viewer_manifest.json", Body=b"{}")

        clip_b = _upload_and_complete_clip(client, token, filename="clip_b.mp4")
        job_b = _queue_job_for_clip(client, token, clip_b["id"])
        s3_client.put_object(Bucket=BUCKET, Key=f"artifacts/{job_b['id']}/pipeline_summary.json", Body=b"{}")

        response = client.delete(f"/api/clips/{clip_a['id']}", headers=_auth(token))
        assert response.status_code == 204
        assert response.content == b""

        # clip_a's data is fully gone.
        assert _list_keys(s3_client, f"raw/{user_id}/{clip_a['id']}/") == []
        assert _list_keys(s3_client, f"artifacts/{job_a['id']}/") == []
        assert _list_keys(s3_client, f"bundles/{clip_a['id']}/") == []
        assert db.clips.find_one({"_id": clip_a["id"]}) is None
        assert db.jobs.find_one({"_id": job_a["id"]}) is None

        # clip_b (sibling) and the user doc are untouched.
        assert _list_keys(s3_client, f"raw/{user_id}/{clip_b['id']}/") != []
        assert _list_keys(s3_client, f"artifacts/{job_b['id']}/") != []
        assert db.clips.find_one({"_id": clip_b["id"]}) is not None
        assert db.jobs.find_one({"_id": job_b["id"]}) is not None
        user_doc = db.users.find_one({"_id": user_id})
        assert user_doc is not None
        assert user_doc["deleted_at"] is None
