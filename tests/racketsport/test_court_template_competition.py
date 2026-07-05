from __future__ import annotations

from threed.racketsport.court_template_competition import score_template_competition


def test_pickleball_template_wins_on_nvz_spacing() -> None:
    semantic_lines = {
        "net": {"court_y_ft": 22.0, "support": 0.9},
        "near_nvz": {"court_y_ft": 15.0, "support": 0.9},
        "far_nvz": {"court_y_ft": 29.0, "support": 0.9},
        "near_baseline": {"court_y_ft": 0.0, "support": 0.7},
        "far_baseline": {"court_y_ft": 44.0, "support": 0.7},
    }

    result = score_template_competition(semantic_lines)

    assert result["winner"] == "pickleball"
    assert result["pickleball"]["score"] > result["tennis"]["score"]
    assert result["margin"] > 0.2


def test_tennis_service_spacing_blocks_pickleball() -> None:
    semantic_lines = {
        "net": {"court_y_ft": 39.0, "support": 0.9},
        "near_service": {"court_y_ft": 18.0, "support": 0.9},
        "far_service": {"court_y_ft": 60.0, "support": 0.9},
        "near_baseline": {"court_y_ft": 0.0, "support": 0.9},
        "far_baseline": {"court_y_ft": 78.0, "support": 0.9},
    }

    result = score_template_competition(semantic_lines)

    assert result["winner"] == "tennis"
    assert "tennis_template_wins" in result["pickleball"]["reject_reasons"]
