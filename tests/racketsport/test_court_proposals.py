from __future__ import annotations

from threed.racketsport.court_proposals import CourtProposal, CourtProposalReport


def test_proposal_report_forces_fail_closed_flags() -> None:
    report = CourtProposalReport(
        clip="clip_a",
        image_size=(1920, 1080),
        frame_indices=[0, 10],
        proposals=[
            CourtProposal(
                proposal_id="proposal_0001",
                source="unit",
                court_keypoints={"near_left_corner": (1.0, 2.0)},
                scores={"overall": 0.25},
            )
        ],
    )

    payload = report.to_json_dict()

    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["proposals"][0]["verified"] is False
    assert payload["proposals"][0]["not_cal3_verified"] is True
    assert payload["ranking"]["abstain"] is True


def test_proposal_report_rejects_promoted_preview() -> None:
    proposal = CourtProposal(
        proposal_id="proposal_0001",
        source="unit",
        court_keypoints={},
        scores={"overall": 1.0},
        gate={"auto_usable": True, "review_usable": True, "failed": [], "warnings": []},
    )
    report = CourtProposalReport(
        clip="clip_a",
        image_size=(100, 100),
        frame_indices=[0],
        proposals=[proposal],
    )

    payload = report.to_json_dict()

    assert payload["ranking"]["abstain"] is True
    assert "not_cal3_verified" in payload["ranking"]["abstain_reasons"]
    assert payload["proposals"][0]["gate"]["auto_usable"] is False
    assert "not_verified" in payload["proposals"][0]["gate"]["failed"]
