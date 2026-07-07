"""Stripe webhook route (INFRA-1): feature-flagged stub.

Per the approved design (Sec 9) Stripe is scaffold-only until P7-3: the
endpoint exists so the URL can be registered with Stripe early, but it returns
503 while `PICKLEBALL_STRIPE_ENABLED=0`. Signature verification and event
handling land when Stripe goes live.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException


def build_stripe_webhook_router(*, stripe_enabled: bool) -> APIRouter:
    router = APIRouter()

    @router.post("/api/stripe/webhook")
    def stripe_webhook() -> None:
        if not stripe_enabled:
            raise HTTPException(
                status_code=503,
                detail="stripe integration is disabled (PICKLEBALL_STRIPE_ENABLED=0)",
            )
        raise HTTPException(
            status_code=503,
            detail="stripe webhook handling is scaffolded only; goes live at P7-3",
        )

    return router
