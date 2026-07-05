from __future__ import annotations

from threed.racketsport.court_regulation_proposals import propose_regulation_courts_from_line_bank


def test_regulation_proposal_ranks_clean_pickleball_template() -> None:
    line_bank = {
        "segments": [
            {"xyxy": [20.0, 40.0, 300.0, 40.0], "detector": "unit", "length": 280.0},
            {"xyxy": [20.0, 100.0, 300.0, 100.0], "detector": "unit", "length": 280.0},
            {"xyxy": [20.0, 140.0, 300.0, 140.0], "detector": "unit", "length": 280.0},
            {"xyxy": [20.0, 200.0, 300.0, 200.0], "detector": "unit", "length": 280.0},
            {"xyxy": [20.0, 40.0, 20.0, 200.0], "detector": "unit", "length": 160.0},
            {"xyxy": [300.0, 40.0, 300.0, 200.0], "detector": "unit", "length": 160.0},
        ]
    }

    proposals = propose_regulation_courts_from_line_bank(line_bank, image_size=(320, 240))

    assert proposals
    assert proposals[0].scores["template_margin"] >= 0.0
    assert proposals[0].gate["review_usable"] is True
    assert proposals[0].gate["auto_usable"] is False
