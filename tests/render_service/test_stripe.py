"""Stripe minimal-integration tests (SDK-free via injected event verifier)."""

from pathlib import Path

import mongomock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.render_app import create_app
from server.routes.stripe_webhook import build_stripe_webhook_router

PUBLISHABLE = "pk_test_dummy_publishable"


class UnusedRunner:
    name = "test-unused"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request):  # noqa: ANN001 - never called here
        raise AssertionError("runner must not execute")


def _app(tmp_path: Path, *, stripe_enabled: bool) -> TestClient:
    app = create_app(
        upload_root=tmp_path,
        runner=UnusedRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        mongo_db=mongomock.MongoClient()["pickleball"],
        accounts_enabled=True,
        env={
            "PICKLEBALL_JWT_SECRET": "unit-test-jwt-secret-0123456789abcdef",
            "PICKLEBALL_INVITE_CODE": "friends-of-the-court",
            "PICKLEBALL_S3_BUCKET": "test-bucket",
            "PICKLEBALL_STRIPE_ENABLED": "1" if stripe_enabled else "0",
            "PICKLEBALL_STRIPE_PUBLISHABLE_KEY": PUBLISHABLE,
        },
    )
    return TestClient(app)


def test_stripe_config_503_when_disabled(tmp_path: Path) -> None:
    assert _app(tmp_path, stripe_enabled=False).get("/api/stripe/config").status_code == 503


def test_stripe_webhook_503_when_disabled(tmp_path: Path) -> None:
    assert _app(tmp_path, stripe_enabled=False).post("/api/stripe/webhook").status_code == 503


def test_stripe_config_returns_publishable_key_when_enabled(tmp_path: Path) -> None:
    response = _app(tmp_path, stripe_enabled=True).get("/api/stripe/config")
    assert response.status_code == 200
    assert response.json() == {"publishable_key": PUBLISHABLE}
    # The secret key must never surface anywhere in the config payload.
    assert "sk_" not in response.text


def _router_client(db, *, verifier) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_stripe_webhook_router(
            stripe_enabled=True,
            db=db,
            publishable_key=PUBLISHABLE,
            webhook_secret="whsec_test",
            event_verifier=verifier,
        )
    )
    return TestClient(app)


def test_webhook_checkout_completed_upserts_entitlement() -> None:
    db = mongomock.MongoClient()["pickleball"]
    db.users.insert_one({"_id": "user_1", "email": "a@b.com", "stripe_customer_id": None})

    def verifier(payload: bytes, sig: str, secret: str) -> dict:
        return {
            "id": "evt_1",
            "type": "checkout.session.completed",
            "data": {"object": {"client_reference_id": "user_1", "customer": "cus_1"}},
        }

    response = _router_client(db, verifier=verifier).post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "t=1,v1=x"}
    )
    assert response.status_code == 200
    assert response.json() == {"received": True}
    ent = db.entitlements.find_one({"user_id": "user_1"})
    assert ent is not None and ent["status"] == "active" and ent["stripe_customer_id"] == "cus_1"
    assert db.users.find_one({"_id": "user_1"})["stripe_customer_id"] == "cus_1"


def test_webhook_bad_signature_400() -> None:
    db = mongomock.MongoClient()["pickleball"]

    def verifier(payload: bytes, sig: str, secret: str) -> dict:
        raise ValueError("bad signature")

    response = _router_client(db, verifier=verifier).post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "bad"}
    )
    assert response.status_code == 400
    assert db.entitlements.find_one({}) is None
