from pathlib import Path

import mongomock
from fastapi.testclient import TestClient

from server.render_app import create_app
from server.security import REFRESH_COOKIE_NAME

JWT_SECRET = "unit-test-jwt-secret-0123456789abcdef"
INVITE_CODE = "friends-of-the-court"
PASSWORD = "correct-horse-battery"


def _accounts_env() -> dict[str, str]:
    return {
        "PICKLEBALL_JWT_SECRET": JWT_SECRET,
        "PICKLEBALL_INVITE_CODE": INVITE_CODE,
        "PICKLEBALL_S3_BUCKET": "test-bucket",
    }


class UnusedRunner:
    name = "test-unused"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request):  # noqa: ANN001 - never called in auth tests
        raise AssertionError("runner must not execute in auth tests")


def _make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        upload_root=tmp_path,
        runner=UnusedRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=mongomock.MongoClient()["pickleball"],
        accounts_enabled=True,
        env=_accounts_env(),
    )
    # https base_url so the Secure refresh cookie is stored and replayed.
    return TestClient(app, base_url="https://testserver")


def _register(client: TestClient, email: str = "player@example.com") -> dict:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "invite_code": INVITE_CODE},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _login(client: TestClient, email: str = "player@example.com") -> dict:
    response = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()


def _refresh_with_cookie(client: TestClient, value: str):
    """POST /api/auth/refresh presenting exactly `value` as the refresh cookie.

    The httpx cookie jar files the response cookie under `testserver.local`,
    so re-`set()`ing the same name creates a second entry instead of replacing
    it; clearing the jar and sending an explicit Cookie header is deterministic.
    """
    client.cookies.clear()
    return client.post(
        "/api/auth/refresh", headers={"Cookie": f"{REFRESH_COOKIE_NAME}={value}"}
    )


def test_register_creates_user_and_normalizes_email(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    payload = _register(client, email="Player@Example.COM")

    assert payload["email"] == "player@example.com"
    assert payload["id"].startswith("user_")


def test_register_duplicate_email_conflicts_409(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)

    response = client.post(
        "/api/auth/register",
        json={"email": "PLAYER@example.com", "password": PASSWORD, "invite_code": INVITE_CODE},
    )

    assert response.status_code == 409


def test_register_bad_invite_code_403(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    response = client.post(
        "/api/auth/register",
        json={"email": "player@example.com", "password": PASSWORD, "invite_code": "wrong"},
    )

    assert response.status_code == 403


def test_login_wrong_password_401(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)

    response = client.post(
        "/api/auth/login", json={"email": "player@example.com", "password": "not-the-password"}
    )

    assert response.status_code == 401
    assert "access_token" not in response.json()


def test_login_sets_httponly_lax_secure_refresh_cookie(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)

    response = client.post("/api/auth/login", json={"email": "player@example.com", "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    set_cookie = response.headers["set-cookie"].lower()
    assert REFRESH_COOKIE_NAME in set_cookie
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "path=/api/auth" in set_cookie


def test_refresh_rotates_token_and_reuse_revokes_whole_chain(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)
    _login(client)
    first_token = client.cookies.get(REFRESH_COOKIE_NAME)
    assert first_token

    rotated = client.post("/api/auth/refresh")
    assert rotated.status_code == 200
    assert rotated.json()["access_token"]
    second_token = client.cookies.get(REFRESH_COOKIE_NAME)
    assert second_token and second_token != first_token

    # Replaying the rotated-away token is treated as theft: 401 + chain revoked.
    reuse = _refresh_with_cookie(client, first_token)
    assert reuse.status_code == 401

    # The still-newest token is dead too: the whole chain was revoked.
    after_revocation = _refresh_with_cookie(client, second_token)
    assert after_revocation.status_code == 401


def test_refresh_without_cookie_401(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    response = client.post("/api/auth/refresh")

    assert response.status_code == 401


def test_logout_returns_204_and_revokes_chain(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)
    _login(client)
    refresh_token = client.cookies.get(REFRESH_COOKIE_NAME)
    assert refresh_token

    response = client.post("/api/auth/logout")
    assert response.status_code == 204

    # Even replaying the pre-logout cookie fails: revoked, not just deleted.
    after_logout = _refresh_with_cookie(client, refresh_token)
    assert after_logout.status_code == 401


def test_register_rate_limited_429_at_sixth_request(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    for index in range(5):
        response = client.post(
            "/api/auth/register",
            json={
                "email": f"player{index}@example.com",
                "password": PASSWORD,
                "invite_code": INVITE_CODE,
            },
        )
        assert response.status_code == 201, f"request {index + 1}: {response.text}"

    sixth = client.post(
        "/api/auth/register",
        json={"email": "player5@example.com", "password": PASSWORD, "invite_code": INVITE_CODE},
    )
    assert sixth.status_code == 429


def test_login_rate_limited_429_at_eleventh_request(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)

    for index in range(10):
        response = client.post(
            "/api/auth/login", json={"email": "player@example.com", "password": "wrong-password"}
        )
        assert response.status_code == 401, f"request {index + 1}: {response.text}"

    eleventh = client.post(
        "/api/auth/login", json={"email": "player@example.com", "password": PASSWORD}
    )
    assert eleventh.status_code == 429


def test_account_delete_is_jwt_gated_501_stub(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    _register(client)
    access_token = _login(client)["access_token"]

    unauthenticated = client.delete("/api/account")
    assert unauthenticated.status_code == 401

    authenticated = client.delete(
        "/api/account", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert authenticated.status_code == 501


def test_stripe_webhook_stub_returns_503_while_disabled(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    response = client.post("/api/stripe/webhook", json={"type": "checkout.session.completed"})

    assert response.status_code == 503


def test_create_app_ensures_data_model_indexes_on_injected_db(tmp_path: Path) -> None:
    db = mongomock.MongoClient()["pickleball"]
    create_app(
        upload_root=tmp_path,
        runner=UnusedRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=db,
        accounts_enabled=True,
        env=_accounts_env(),
    )

    assert "users_email_unique" in db.users.index_information()
    refresh_indexes = db.refresh_tokens.index_information()
    assert "refresh_tokens_token_hash_unique" in refresh_indexes
    assert "refresh_tokens_user_id" in refresh_indexes
    assert "refresh_tokens_expires_at_ttl" in refresh_indexes
    jobs_indexes = db.jobs.index_information()
    assert "jobs_status_created_at" in jobs_indexes
    assert "jobs_worker_id_heartbeat_at" in jobs_indexes
    assert "clips_user_id_created_at" in db.clips.index_information()
    assert "entitlements_user_id_unique" in db.entitlements.index_information()
