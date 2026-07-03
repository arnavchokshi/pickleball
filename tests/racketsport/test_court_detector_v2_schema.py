from __future__ import annotations

import pytest
from pydantic import ValidationError

from threed.racketsport.court_detector_v2_schema import (
    CourtDetectorV2Proposal,
    build_blocked_detector_v2_proposal,
)


def test_detector_v2_blocked_proposal_cannot_claim_promotion() -> None:
    artifact = build_blocked_detector_v2_proposal(
        clip="owner_IMG_1605_8a193402780b",
        source_frame="frame_000151.jpg",
        image_size=(1080, 1920),
        blockers=["self_verification_not_promotable"],
        needs_user_input=["near_left_corner"],
    )

    parsed = CourtDetectorV2Proposal.model_validate(artifact)

    assert parsed.promoted is False
    assert parsed.promotion_status == "needs_user_input"
    assert parsed.promotion_blockers == ["self_verification_not_promotable"]
    assert parsed.needs_user_input == ["near_left_corner"]
    assert parsed.verified is False
    assert parsed.not_cal3_verified is True


def test_detector_v2_rejects_promoted_not_verified_artifact() -> None:
    artifact = build_blocked_detector_v2_proposal(
        clip="clip",
        source_frame=None,
        image_size=(960, 540),
        blockers=[],
        needs_user_input=[],
    )
    artifact.update(
        {
            "promoted": True,
            "verified": False,
            "not_cal3_verified": True,
            "promotion_status": "promoted",
        }
    )

    with pytest.raises(ValidationError, match="promoted detector v2 proposals must be verified"):
        CourtDetectorV2Proposal.model_validate(artifact)
