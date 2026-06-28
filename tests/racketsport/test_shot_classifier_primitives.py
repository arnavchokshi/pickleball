import pytest

from threed.racketsport import shot_classifier as sc


def test_taxonomy_matches_pickleball_shot_dataset_labels():
    assert sc.ALLOWED_SHOT_LABELS == (
        "serve",
        "fh_shot",
        "bh_shot",
        "fh_drive",
        "bh_drive",
        "dink",
        "lob",
        "overhead",
        "third_shot_drop",
        "reset_block",
    )


def test_valid_candidate_builds_deterministic_feature_window_metadata():
    candidate = sc.ShotCandidate(
        candidate_id="clip_001:p7:c750",
        player_id=7,
        contact_t=12.5,
        contact_frame=750,
        contact_confidence=0.92,
        audio_event_id="audio_pop_042",
        pose_track_id="pose_track_7",
        ball_event_id="ball_contact_042",
    )

    validation = sc.validate_shot_candidate(candidate)
    window = sc.build_feature_window(candidate, pre_s=0.4, post_s=0.3, fps=60.0)

    assert validation.accepted is True
    assert validation.reasons == []
    assert window == {
        "candidate_id": "clip_001:p7:c750",
        "player_id": 7,
        "center_t": 12.5,
        "start_t": 12.1,
        "end_t": 12.8,
        "center_frame": 750,
        "start_frame": 726,
        "end_frame": 768,
        "fps": 60.0,
        "sources": {
            "audio_event_id": "audio_pop_042",
            "pose_track_id": "pose_track_7",
            "ball_event_id": "ball_contact_042",
        },
    }


def test_candidate_validation_fails_closed_on_missing_or_untrusted_inputs():
    candidate = sc.ShotCandidate(
        candidate_id="bad",
        player_id=0,
        contact_t=-0.1,
        contact_frame=-3,
        contact_confidence=0.31,
        audio_event_id="",
        pose_track_id="",
        ball_event_id=None,
    )

    validation = sc.validate_shot_candidate(candidate)

    assert validation.accepted is False
    assert validation.reasons == [
        "player_id must be positive",
        "contact_t must be non-negative",
        "contact_frame must be non-negative",
        "contact_confidence below 0.50",
        "audio_event_id is required",
        "pose_track_id is required",
        "ball_event_id is required",
    ]


def test_candidate_and_prediction_reject_nonfinite_or_bool_values():
    with pytest.raises(ValueError, match="player_id must be an integer"):
        sc.ShotCandidate(
            candidate_id="bad",
            player_id=True,  # type: ignore[arg-type]
            contact_t=0.0,
            contact_frame=0,
            contact_confidence=0.9,
            audio_event_id="audio",
            pose_track_id="pose",
            ball_event_id="ball",
        )

    with pytest.raises(ValueError, match="contact_confidence must be finite"):
        sc.ShotCandidate(
            candidate_id="bad",
            player_id=1,
            contact_t=0.0,
            contact_frame=0,
            contact_confidence=float("nan"),
            audio_event_id="audio",
            pose_track_id="pose",
            ball_event_id="ball",
        )

    good_candidate = sc.ShotCandidate(
        candidate_id="clip_001:p1:c120",
        player_id=1,
        contact_t=2.0,
        contact_frame=120,
        contact_confidence=0.88,
        audio_event_id="audio_pop_001",
        pose_track_id="pose_track_1",
        ball_event_id="ball_contact_001",
    )
    with pytest.raises(ValueError, match="top2/0.confidence must be finite"):
        sc.ShotPrediction(
            candidate=good_candidate,
            label="dink",
            confidence=0.82,
            top2=(("dink", True),),  # type: ignore[arg-type]
        )


def test_prediction_gating_accepts_only_allowed_confident_labels():
    confident = sc.ShotPrediction(
        candidate=sc.ShotCandidate(
            candidate_id="clip_001:p1:c120",
            player_id=1,
            contact_t=2.0,
            contact_frame=120,
            contact_confidence=0.88,
            audio_event_id="audio_pop_001",
            pose_track_id="pose_track_1",
            ball_event_id="ball_contact_001",
        ),
        label="dink",
        confidence=0.82,
        top2=(("dink", 0.82), ("reset_block", 0.11)),
    )
    weak = sc.ShotPrediction(
        candidate=confident.candidate,
        label="smash",
        confidence=0.44,
        top2=(("smash", 0.44), ("lob", 0.2)),
    )

    assert sc.gate_prediction(confident, min_confidence=0.65) == {
        "type": "dink",
        "type_conf": 0.82,
        "gated": False,
        "gate_reasons": [],
        "top2": [
            {"type": "dink", "confidence": 0.82},
            {"type": "reset_block", "confidence": 0.11},
        ],
    }
    assert sc.gate_prediction(weak, min_confidence=0.65) == {
        "type": "unknown",
        "type_conf": 0.44,
        "gated": True,
        "gate_reasons": [
            "label must be one of bh_drive, bh_shot, dink, fh_drive, fh_shot, lob, overhead, reset_block, serve, third_shot_drop",
            "confidence below 0.65",
            "top2/0 label must be one of bh_drive, bh_shot, dink, fh_drive, fh_shot, lob, overhead, reset_block, serve, third_shot_drop",
        ],
        "original_type": "smash",
        "top2": [
            {"type": "smash", "confidence": 0.44},
            {"type": "lob", "confidence": 0.2},
        ],
    }


def test_sequence_payload_groups_by_player_sorts_by_time_and_marks_scaffold_only():
    early_candidate = sc.ShotCandidate(
        candidate_id="clip_002:p2:c60",
        player_id=2,
        contact_t=1.0,
        contact_frame=60,
        contact_confidence=0.91,
        audio_event_id="audio_pop_001",
        pose_track_id="pose_track_2",
        ball_event_id="ball_contact_001",
    )
    late_candidate = sc.ShotCandidate(
        candidate_id="clip_002:p1:c180",
        player_id=1,
        contact_t=3.0,
        contact_frame=180,
        contact_confidence=0.93,
        audio_event_id="audio_pop_003",
        pose_track_id="pose_track_1",
        ball_event_id="ball_contact_003",
    )

    payload = sc.build_shot_sequence_payload(
        clip_id="clip_002",
        predictions=[
            sc.ShotPrediction(candidate=late_candidate, label="lob", confidence=0.61),
            sc.ShotPrediction(candidate=early_candidate, label="serve", confidence=0.9),
        ],
        fps=60.0,
        min_confidence=0.7,
    )

    assert payload["schema_version"] == 1
    assert payload["clip_id"] == "clip_002"
    assert payload["classifier"] == {
        "name": "shot_classifier_cpu_scaffold",
        "scaffold_only": True,
        "model_training_complete": False,
    }
    assert [player["id"] for player in payload["players"]] == [1, 2]
    assert payload["players"][0]["shots"][0]["type"] == "unknown"
    assert payload["players"][0]["shots"][0]["original_type"] == "lob"
    assert payload["players"][0]["shots"][0]["window"]["start_frame"] == 153
    assert payload["players"][1]["shots"][0]["type"] == "serve"
    assert payload["players"][1]["shots"][0]["window"]["sources"]["pose_track_id"] == "pose_track_2"
