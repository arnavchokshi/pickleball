from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from threed.racketsport.shot_rules import classify_shot, classify_shots, load_shot_rules


TABLE_PATH = Path("docs/racketsport/shot_rules_v0.json")


def test_table_covers_all_v0_classes_and_unknown() -> None:
    table = load_shot_rules(TABLE_PATH)

    assert [rule["class"] for rule in table["rules"]] == [
        "serve",
        "return",
        "lob",
        "smash",
        "volley",
        "dink",
        "drop",
        "drive",
        "unknown",
    ]
    assert all(rule["rationale"].strip() for rule in table["rules"])
    assert "shot_taxonomy.classify_shots_from_payloads" in table["reconciliation_note"]
    assert "racketsport_shots" in table["reconciliation_note"]
    assert "volley" in table["reconciliation_note"]
    assert "atp/erne/tweener" in table["reconciliation_note"]


@pytest.mark.parametrize(
    ("shot_class", "overrides", "expected_confidence"),
    [
        (
            "serve",
            {"rally_shot_index": 1, "contact_zone": "baseline", "is_serve_candidate": True},
            0.9,
        ),
        ("return", {"rally_shot_index": 2, "prev_shot_bounced": True}, 0.85),
        ("lob", {"apex_height_m": 3.2, "landing_zone": "baseline"}, 0.7),
        (
            "smash",
            {
                "contact_height_m": 1.8,
                "horizontal_speed_mps": 12.2,
                "net_crossing_height_m": 1.2,
            },
            0.7,
        ),
        ("volley", {"prev_shot_bounced": False, "contact_zone": "kitchen"}, 0.65),
        (
            "dink",
            {
                "prev_shot_bounced": True,
                "contact_zone": "kitchen",
                "apex_height_m": 1.1,
                "landing_zone": "kitchen",
            },
            0.75,
        ),
        (
            "drop",
            {
                "rally_shot_index": 3,
                "prev_shot_bounced": True,
                "landing_zone": "kitchen",
                "apex_height_m": 2.1,
            },
            0.7,
        ),
        ("drive", {"horizontal_speed_mps": 10.5, "apex_height_m": 1.4}, 0.6),
        ("unknown", {"horizontal_speed_mps": 4.0, "apex_height_m": 2.0}, 0.3),
    ],
)
def test_classify_shot_fires_each_rule(shot_class: str, overrides: dict, expected_confidence: float) -> None:
    table = load_shot_rules(TABLE_PATH)

    result = classify_shot(_features(**overrides), table)

    assert result["shot_class"] == shot_class
    assert result["rule_fired"] == shot_class
    assert result["confidence"] == pytest.approx(expected_confidence)
    assert result["rationale"]


def test_classify_shots_preserves_order() -> None:
    table = load_shot_rules(TABLE_PATH)

    results = classify_shots(
        [
            _features(rally_shot_index=1, contact_zone="baseline", is_serve_candidate=True),
            _features(horizontal_speed_mps=10.5, apex_height_m=1.4),
        ],
        table,
    )

    assert [result["shot_class"] for result in results] == ["serve", "drive"]


def test_null_arc_features_never_crash_and_can_match_arc_free_rules() -> None:
    table = load_shot_rules(TABLE_PATH)
    arcless_return = _features(
        contact_height_m=None,
        apex_height_m=None,
        landing_world_xy=None,
        landing_zone=None,
        net_crossing_height_m=None,
        horizontal_speed_mps=None,
        outgoing_dir=None,
        rally_shot_index=2,
        prev_shot_bounced=True,
    )
    arcless_mid_rally = _features(
        contact_height_m=None,
        apex_height_m=None,
        landing_world_xy=None,
        landing_zone=None,
        net_crossing_height_m=None,
        horizontal_speed_mps=None,
        outgoing_dir=None,
        rally_shot_index=5,
        prev_shot_bounced=True,
    )

    assert classify_shot(arcless_return, table)["shot_class"] == "return"
    assert classify_shot(arcless_mid_rally, table)["shot_class"] == "unknown"


def test_first_match_wins_for_higher_priority_rule() -> None:
    table = load_shot_rules(TABLE_PATH)
    serve_that_also_matches_lob = _features(
        rally_shot_index=1,
        contact_zone="baseline",
        is_serve_candidate=True,
        apex_height_m=3.4,
        landing_zone="baseline",
    )

    result = classify_shot(serve_that_also_matches_lob, table)

    assert result["shot_class"] == "serve"
    assert result["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize(("trust", "expected_confidence"), [("estimated", 0.3), ("unverified_cue", 0.18)])
def test_trust_modifiers_lower_confidence_without_exceeding_seed(trust: str, expected_confidence: float) -> None:
    table = load_shot_rules(TABLE_PATH)

    result = classify_shot(_features(horizontal_speed_mps=10.5, apex_height_m=1.4, trust=trust), table)

    assert result["shot_class"] == "drive"
    assert result["confidence"] == pytest.approx(expected_confidence)
    assert result["confidence"] <= 0.6


def test_loader_rejects_missing_rationale(tmp_path: Path) -> None:
    payload = _table_payload()
    payload["rules"][0]["rationale"] = ""

    path = tmp_path / "missing_rationale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rationale"):
        load_shot_rules(path)


def test_loader_rejects_unknown_feature_key(tmp_path: Path) -> None:
    payload = _table_payload()
    payload["rules"][0]["conditions"]["all"][0]["feature"] = "unknown_feature"

    path = tmp_path / "unknown_feature.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown feature key"):
        load_shot_rules(path)


def test_loader_rejects_bad_priorities(tmp_path: Path) -> None:
    payload = _table_payload()
    payload["rules"][1]["priority"] = payload["rules"][0]["priority"]

    path = tmp_path / "bad_priorities.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="priorities"):
        load_shot_rules(path)


def _table_payload() -> dict:
    return copy.deepcopy(load_shot_rules(TABLE_PATH))


def _features(**overrides: object) -> dict:
    features = {
        "contact_t": 1.0,
        "contact_frame": 30,
        "player_id": "p1",
        "contact_world_xy": [0.0, -3.0],
        "contact_zone": "transition",
        "contact_height_m": 1.0,
        "apex_height_m": 1.3,
        "landing_world_xy": [0.1, 4.0],
        "landing_zone": "transition",
        "net_crossing_height_m": 1.1,
        "horizontal_speed_mps": 6.0,
        "outgoing_dir": [0.0, 1.0],
        "rally_shot_index": 4,
        "is_serve_candidate": False,
        "prev_shot_bounced": True,
        "trust": "ok",
    }
    features.update(overrides)
    return features
