from pathlib import Path

import boto3
import mongomock
import requests
from fastapi.testclient import TestClient
from moto import mock_aws

from server.render_app import create_app

BUCKET = "test-bucket"
JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
PASSWORD = "correct-horse-battery"
MB = 1024 * 1024


def _accounts_env() -> dict[str, str]:
    return {
        "PICKLEBALL_JWT_SECRET": JWT_SECRET,
        "PICKLEBALL_INVITE_CODE": INVITE_CODE,
        "PICKLEBALL_S3_BUCKET": BUCKET,
    }


class UnusedRunner:
    name = "test-unused"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request):  # noqa: ANN001 - never called in clip tests
        raise AssertionError("runner must not execute in clip tests")


def _make(tmp_path: Path):
    s3_client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3_client.create_bucket(Bucket=BUCKET)
    app = create_app(
        upload_root=tmp_path,
        runner=UnusedRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=mongomock.MongoClient()["pickleball"],
        s3_client=s3_client,
        accounts_enabled=True,
        env=_accounts_env(),
    )
    return TestClient(app, base_url="https://testserver"), s3_client


def _register_and_login(client: TestClient, email: str = "clips@example.com") -> tuple[str, str]:
    """Returns (user_id, bearer token)."""
    registered = client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "invite_code": INVITE_CODE},
    )
    assert registered.status_code == 201, registered.text
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    return registered.json()["id"], login.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_clips_require_jwt_401(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)

        assert client.get("/api/clips").status_code == 401
        assert (
            client.post(
                "/api/clips",
                json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
            ).status_code
            == 401
        )
        assert client.get("/api/clips", headers=_auth("not-a-jwt")).status_code == 401


def test_create_clip_returns_multipart_plan_with_user_scoped_keys(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)
        user_id, token = _register_and_login(client)

        response = client.post(
            "/api/clips",
            json={"filename": "Rally Drill.mp4", "size_bytes": 10 * MB + 1, "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["id"].startswith("clip_")
        assert payload["key"] == f"raw/{user_id}/{payload['id']}/Rally_Drill.mp4"
        assert payload["upload_id"]
        assert payload["part_count"] == 3  # 10 MB + 1 byte over 5 MB parts
        assert [part["part_number"] for part in payload["part_urls"]] == [1, 2, 3]
        assert all(part["url"].startswith("http") for part in payload["part_urls"])
        assert payload["sidecar_upload_url"].startswith("http")


def test_create_clip_part_count_exact_boundary(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)
        _, token = _register_and_login(client)

        response = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10 * MB, "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )

        assert response.status_code == 201
        payload = response.json()
        # size % part_size == 0 must land exactly on the boundary: 2 parts, not 3.
        assert payload["part_count"] == 2
        assert [part["part_number"] for part in payload["part_urls"]] == [1, 2]


def test_create_clip_rejects_nonpositive_sizes(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)
        _, token = _register_and_login(client)

        zero_size = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 0, "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )
        zero_part = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 0},
            headers=_auth(token),
        )

        assert zero_size.status_code == 422
        assert zero_part.status_code == 422


def test_complete_multipart_upload_verified_via_list_objects_v2(tmp_path: Path) -> None:
    with mock_aws():
        client, s3_client = _make(tmp_path)
        _, token = _register_and_login(client)
        body = b"0123456789"

        created = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": len(body), "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )
        assert created.status_code == 201
        clip = created.json()
        assert clip["part_count"] == 1

        put = requests.put(clip["part_urls"][0]["url"], data=body)
        assert put.status_code == 200, put.text
        etag = put.headers["ETag"].strip('"')

        completed = client.post(
            f"/api/clips/{clip['id']}/complete",
            json={"upload_id": clip["upload_id"], "parts": [{"part_number": 1, "etag": etag}]},
            headers=_auth(token),
        )
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "uploaded"

        listed = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=clip["key"])
        assert listed["KeyCount"] == 1
        assert listed["Contents"][0]["Key"] == clip["key"]
        assert listed["Contents"][0]["Size"] == len(body)

        library = client.get("/api/clips", headers=_auth(token))
        assert library.status_code == 200
        assert library.json()["clips"][0]["status"] == "uploaded"


def test_sidecar_upload_url_writes_to_s3(tmp_path: Path) -> None:
    with mock_aws():
        client, s3_client = _make(tmp_path)
        user_id, token = _register_and_login(client)

        created = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
            headers=_auth(token),
        )
        clip = created.json()

        # The sidecar URL is signed for application/json, so the upload must
        # declare it (the clients do the same).
        put = requests.put(
            clip["sidecar_upload_url"], data=b"{}", headers={"Content-Type": "application/json"}
        )
        assert put.status_code == 200, put.text

        sidecar_key = f"raw/{user_id}/{clip['id']}/capture_sidecar.json"
        listed = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=sidecar_key)
        assert listed["KeyCount"] == 1


def test_clip_library_is_owner_scoped(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)
        _, token_a = _register_and_login(client, email="owner@example.com")
        _, token_b = _register_and_login(client, email="other@example.com")

        created = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
            headers=_auth(token_a),
        )
        assert created.status_code == 201

        owner_list = client.get("/api/clips", headers=_auth(token_a)).json()["clips"]
        other_list = client.get("/api/clips", headers=_auth(token_b)).json()["clips"]

        assert len(owner_list) == 1
        assert other_list == []


def test_complete_other_users_clip_404(tmp_path: Path) -> None:
    with mock_aws():
        client, _ = _make(tmp_path)
        _, token_a = _register_and_login(client, email="owner@example.com")
        _, token_b = _register_and_login(client, email="other@example.com")

        created = client.post(
            "/api/clips",
            json={"filename": "drill.mp4", "size_bytes": 10, "part_size_bytes": 5 * MB},
            headers=_auth(token_a),
        )
        clip = created.json()

        response = client.post(
            f"/api/clips/{clip['id']}/complete",
            json={"upload_id": clip["upload_id"], "parts": [{"part_number": 1, "etag": "x"}]},
            headers=_auth(token_b),
        )

        assert response.status_code == 404
