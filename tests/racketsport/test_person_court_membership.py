from __future__ import annotations

from threed.racketsport.person_court_membership import build_court_membership_artifact, classify_world_position


def test_court_membership_classes_use_reason_codes() -> None:
    assert classify_world_position([0.0, 0.0])["membership_class"] == "target_court_player_candidate"

    apron = classify_world_position([0.0, 7.0])
    assert apron["membership_class"] == "apron_or_boundary"
    assert "inside_apron_margin" in apron["reason_codes"]

    adjacent = classify_world_position([5.5, 1.0])
    assert adjacent["membership_class"] == "adjacent_court"
    assert "beyond_lateral_apron" in adjacent["reason_codes"]

    spectator = classify_world_position([14.0, 15.0])
    assert spectator["membership_class"] == "spectator_background"
    assert "far_from_target_court" in spectator["reason_codes"]

    unknown = classify_world_position(None)
    assert unknown["membership_class"] == "projection_unknown"
    assert "missing_projected_world_xy" in unknown["reason_codes"]


def test_court_membership_artifact_summarizes_fragments_and_preserves_uncertainty() -> None:
    observations = {
        "schema_version": 1,
        "artifact_type": "racketsport_human_observations",
        "source_only": True,
        "uses_cvat_labels": False,
        "observations": [
            {"detection_id": "1:1", "fragment_id": "frag_1", "frame_idx": 1, "projected_world_xy": [0.0, -3.0]},
            {"detection_id": "2:1", "fragment_id": "frag_1", "frame_idx": 2, "projected_world_xy": [0.2, -3.1]},
            {"detection_id": "1:2", "fragment_id": "frag_2", "frame_idx": 1, "projected_world_xy": [6.0, 3.0]},
            {"detection_id": "2:2", "fragment_id": "frag_2", "frame_idx": 2, "projected_world_xy": [6.1, 3.1]},
            {"detection_id": "1:3", "fragment_id": "frag_3", "frame_idx": 1, "projected_world_xy": None},
        ],
    }

    artifact = build_court_membership_artifact(observations)

    assert artifact["artifact_type"] == "racketsport_person_court_membership"
    assert artifact["source_only"] is True
    assert artifact["uses_cvat_labels"] is False
    by_fragment = {fragment["fragment_id"]: fragment for fragment in artifact["fragments"]}
    assert by_fragment["frag_1"]["membership_class"] == "target_court_player_candidate"
    assert by_fragment["frag_1"]["eligible_for_target_selection"] is True
    assert by_fragment["frag_2"]["membership_class"] == "adjacent_court"
    assert by_fragment["frag_2"]["eligible_for_target_selection"] is False
    assert by_fragment["frag_3"]["membership_class"] == "projection_unknown"
    assert by_fragment["frag_3"]["eligible_for_target_selection"] is False
