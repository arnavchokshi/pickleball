"""Stripe routes (INFRA-1 stub → minimal integration start, 2026-07-07).

Owner started Stripe: keys wired, integration begun but intentionally minimal —
full checkout/subscription flows land at P7-3. Everything stays behind
`PICKLEBALL_STRIPE_ENABLED=0` so no charge path is live yet.

What exists now:
- `GET /api/stripe/config` — returns the PUBLISHABLE key (public by design) so web /
  mobile clients can initialize the Stripe SDK. 503 while disabled.
- `POST /api/stripe/webhook` — 503 while disabled; when enabled, verifies the Stripe
  signature and upserts an `entitlements` doc on `checkout.session.completed`. The
  event verifier is injected (defaults to `stripe.Webhook.construct_event`) so tests
  never need the real SDK or a live secret.

Still needed to go live (P7-3): the webhook signing secret (whsec_…, created with the
dashboard webhook endpoint) in `PICKLEBALL_STRIPE_WEBHOOK_SECRET`, real product/price
IDs, a checkout-session create endpoint, and the Apple-IAP-vs-web-checkout decision.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request

# (payload_bytes, sig_header, secret) -> event dict. Matches
# stripe.Webhook.construct_event's shape; injected so tests stay SDK-free.
EventVerifier = Callable[[bytes, str, str], dict[str, Any]]


def _default_event_verifier(payload: bytes, sig_header: str, secret: str) -> dict[str, Any]:
    import stripe  # lazy: only needed when Stripe is actually enabled

    return stripe.Webhook.construct_event(payload, sig_header, secret)


def build_stripe_webhook_router(
    *,
    stripe_enabled: bool,
    db: Any = None,
    publishable_key: str = "",
    webhook_secret: str = "",
    event_verifier: EventVerifier | None = None,
) -> APIRouter:
    router = APIRouter()
    verify_event = event_verifier or _default_event_verifier

    @router.get("/api/stripe/config")
    def stripe_config() -> dict[str, Any]:
        if not stripe_enabled:
            raise HTTPException(
                status_code=503,
                detail="stripe integration is disabled (PICKLEBALL_STRIPE_ENABLED=0)",
            )
        # Publishable key is safe to hand to clients; never return the secret key.
        return {"publishable_key": publishable_key}

    @router.post("/api/stripe/webhook")
    async def stripe_webhook(request: Request) -> dict[str, Any]:
        if not stripe_enabled:
            raise HTTPException(
                status_code=503,
                detail="stripe integration is disabled (PICKLEBALL_STRIPE_ENABLED=0)",
            )
        if not webhook_secret:
            raise HTTPException(
                status_code=503,
                detail="stripe webhook secret not configured (PICKLEBALL_STRIPE_WEBHOOK_SECRET)",
            )
        payload = await request.body()
        sig_header = request.headers.get("Stripe-Signature", "")
        try:
            event = verify_event(payload, sig_header, webhook_secret)
        except Exception as exc:  # noqa: BLE001 - any verification failure is a 400
            raise HTTPException(status_code=400, detail=f"invalid stripe signature: {exc}") from None

        if event.get("type") == "checkout.session.completed" and db is not None:
            session = (event.get("data") or {}).get("object") or {}
            # client_reference_id threads our user_id through Checkout; fall back
            # to the Stripe customer id if that's all the event carries.
            user_id = session.get("client_reference_id")
            customer_id = session.get("customer")
            now = time.time()
            if user_id:
                db.entitlements.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "user_id": user_id,
                            "stripe_customer_id": customer_id,
                            "status": "active",
                            "source_event": event.get("id"),
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
                if customer_id:
                    db.users.update_one(
                        {"_id": user_id},
                        {"$set": {"stripe_customer_id": customer_id, "updated_at": now}},
                    )
        # Always 200 on a verified event so Stripe stops retrying, even for
        # event types we don't handle yet.
        return {"received": True}

    return router
