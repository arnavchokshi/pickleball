from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.contact_window_review import (
    build_contact_window_review_template,
    promote_reviewed_contact_windows,
    render_contact_window_review_html,
    write_contact_window_review,
)
from threed.racketsport.review_input_contact_decisions import (
    apply_review_input_contacts_to_review,
    build_contact_windows_from_review_input_contacts,
)
from threed.racketsport.frame_rating import build_frame_compute_plan
from threed.racketsport.schemas import ContactWindowReview, ContactWindows, validate_artifact_file


def _candidate_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_window_candidates",
        "clip": "clip_001",
        "fps": 60.0,
        "source_event_path": "runs/eval0/clip_001/labels/events.json",
        "not_gate_verified": True,
        "trusted_for_body": False,
        "promotion_target": "contact_windows.json",
        "candidates": [
            {
                "review_id": "event_smoke_contact",
                "type": "contact",
                "frame": 89,
                "t": 89.0 / 60.0,
                "xy_px": [960.0, 540.0],
                "source_label": "contact?",
                "source_status": "uncertain",
                "source_confidence": 0.2,
                "candidate_confidence": 0.2,
                "window": {"t0": 1.4033333333333333, "t1": 1.5633333333333332, "importance": 0.2},
            },
            {
                "review_id": "event_smoke_bounce",
                "type": "bounce",
                "frame": 120,
                "t": 2.0,
                "xy_px": [970.0, 550.0],
                "source_label": "bounce?",
                "source_status": "uncertain",
                "source_confidence": 0.9,
                "candidate_confidence": 0.9,
                "window": {"t0": 1.92, "t1": 2.08, "importance": 0.9},
            },
        ],
        "summary": {
            "candidate_count": 2,
            "rejected_item_count": 0,
            "by_type": {"bounce": 1, "contact": 1},
            "by_status": {"uncertain": 2},
            "uncertainty_flags": ["teacher_model_unavailable", "smoke_generated"],
        },
    }


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "right",
                "frames": [
                    {
                        "t": 89.0 / 60.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [1.0, -3.0],
                        "conf": 0.92,
                    }
                ],
            }
        ],
        "rally_spans": [],
    }


def test_build_contact_window_review_template_marks_all_decisions_pending() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")

    artifact = ContactWindowReview.model_validate(review)
    assert artifact.artifact_type == "racketsport_contact_window_review"
    assert artifact.status == "pending_review"
    assert artifact.clip == "clip_001"
    assert artifact.candidate_path == "contact_window_candidates.json"
    assert artifact.promotion_target == "contact_windows.json"
    assert [decision.review_id for decision in artifact.decisions] == ["event_smoke_contact", "event_smoke_bounce"]
    assert {decision.decision for decision in artifact.decisions} == {"pending"}
    assert artifact.summary.candidate_count == 2
    assert artifact.summary.pending_count == 2
    assert artifact.summary.accepted_count == 0
    assert artifact.summary.rejected_count == 0


def test_render_contact_window_review_html_shows_candidates_and_decisions() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")

    html = render_contact_window_review_html(_candidate_payload(), review)

    assert "<title>Contact Window Review - clip_001</title>" in html
    assert "What to review" in html
    assert "Visual context" in html
    assert "event_smoke_contact" in html
    assert "contact?" in html
    assert "pending" in html
    assert "contact_window_review.json" in html
    assert "contact_windows.json" in html


def test_render_contact_window_review_html_embeds_review_media() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")

    html = render_contact_window_review_html(
        _candidate_payload(),
        review,
        media_paths=[
            {"label": "Ball overlay", "path": "tracknet_smoke_0000_0010/ball_overlay.mp4"},
            {"label": "Player tracks", "path": "player_tracks/player_track_overlay_h264.mp4"},
        ],
    )

    assert "Ball overlay" in html
    assert "Player tracks" in html
    assert "tracknet_smoke_0000_0010/ball_overlay.mp4#t=1.403" in html
    assert "player_tracks/player_track_overlay_h264.mp4#t=1.403" in html


def test_promote_reviewed_contact_windows_requires_reviewer_and_reason_for_accepted_decisions() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")
    review["decisions"][0]["decision"] = "accepted"
    review["decisions"][0]["player_id"] = 7
    review["summary"] = {
        "candidate_count": 2,
        "pending_count": 1,
        "accepted_count": 1,
        "rejected_count": 0,
    }

    with pytest.raises(ValueError, match="accepted decision event_smoke_contact requires reviewer and reason"):
        promote_reviewed_contact_windows(_candidate_payload(), review)


def test_promote_reviewed_contact_windows_rejects_non_contact_acceptance() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")
    review["decisions"][1].update(
        {
            "decision": "accepted",
            "reviewer": "human",
            "reason": "This is visibly a bounce, not a paddle contact.",
        }
    )
    review["summary"] = {
        "candidate_count": 2,
        "pending_count": 1,
        "accepted_count": 1,
        "rejected_count": 0,
    }

    with pytest.raises(ValueError, match="accepted decision event_smoke_bounce has type bounce"):
        promote_reviewed_contact_windows(_candidate_payload(), review)


def test_promote_reviewed_contact_windows_writes_human_review_sources_and_triggers_deep_mesh() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")
    review["decisions"][0].update(
        {
            "decision": "accepted",
            "reviewer": "human",
            "reason": "Visible paddle-ball contact in overlay.",
            "player_id": 7,
            "confidence_override": 0.95,
            "window_override": {"t0": 1.44, "t1": 1.55, "importance": 0.95},
        }
    )
    review["decisions"][1].update(
        {
            "decision": "rejected",
            "reviewer": "human",
            "reason": "Bounce label is outside the contact moment.",
        }
    )
    review["summary"] = {
        "candidate_count": 2,
        "pending_count": 0,
        "accepted_count": 1,
        "rejected_count": 1,
    }

    contact_windows = promote_reviewed_contact_windows(_candidate_payload(), review)

    artifact = ContactWindows.model_validate(contact_windows)
    assert len(artifact.events) == 1
    event = artifact.events[0]
    assert event.type == "contact"
    assert event.frame == 89
    assert event.player_id == 7
    assert event.confidence == pytest.approx(0.95)
    assert event.sources.audio == 0.0
    assert event.sources.wrist_vel == 0.0
    assert event.sources.ball_inflection == 0.0
    assert event.sources.human_review == pytest.approx(1.0)
    assert event.window.t0 == pytest.approx(1.44)
    assert event.window.t1 == pytest.approx(1.55)
    assert event.window.importance == pytest.approx(0.95)

    plan = build_frame_compute_plan(_tracks_payload(), contact_windows=contact_windows, expected_players=1)
    assert plan["summary"]["deep_mesh_window_count"] == 1
    assert plan["frames"][0]["recommended_tier"] == "deep_mesh"
    assert plan["frames"][0]["reasons"] == ["contact_window"]


def test_apply_review_input_contacts_accepts_nearest_contact_candidate() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")
    review_input = {
        "clips": {
            "clip_001": {
                "contacts": [
                    {
                        "player": "P1",
                        "time_s": 1.49,
                        "note": "Visible paddle-ball contact in ball overlay.",
                    }
                ]
            }
        }
    }

    updated = apply_review_input_contacts_to_review(
        _candidate_payload(),
        review,
        review_input,
        clip="clip_001",
        reviewer="review-ui",
        max_delta_s=0.1,
    )

    artifact = ContactWindowReview.model_validate(updated)
    accepted = [decision for decision in artifact.decisions if decision.decision == "accepted"]
    assert len(accepted) == 1
    assert accepted[0].review_id == "event_smoke_contact"
    assert accepted[0].reviewer == "review-ui"
    assert accepted[0].player_id == 1
    assert "Visible paddle-ball contact" in accepted[0].reason
    assert artifact.summary.accepted_count == 1
    assert artifact.summary.pending_count == 1


def test_apply_review_input_contacts_fails_when_no_contact_candidate_is_nearby() -> None:
    review = build_contact_window_review_template(_candidate_payload(), candidate_path="contact_window_candidates.json")
    review_input = {"clips": {"clip_001": {"contacts": [{"player": "P1", "time_s": 4.0, "note": ""}]}}}

    with pytest.raises(ValueError, match="no contact candidate within 0.100s"):
        apply_review_input_contacts_to_review(
            _candidate_payload(),
            review,
            review_input,
            clip="clip_001",
            reviewer="review-ui",
            max_delta_s=0.1,
        )


def test_build_contact_windows_from_review_input_contacts_uses_times_without_trusting_placeholder_players() -> None:
    review_input = {
        "clips": {
            "clip_001": {
                "contacts": [
                    {"player": "P1", "time_s": 1.25, "note": ""},
                    {"player": "P1", "time_s": 2.5, "note": ""},
                ]
            }
        }
    }

    contact_windows = build_contact_windows_from_review_input_contacts(review_input, clip="clip_001", fps=60.0)

    artifact = ContactWindows.model_validate(contact_windows)
    assert [event.t for event in artifact.events] == [pytest.approx(1.25), pytest.approx(2.5)]
    assert [event.frame for event in artifact.events] == [75, 150]
    assert [event.player_id for event in artifact.events] == [None, None]
    assert artifact.events[0].sources.human_review == pytest.approx(1.0)
    assert artifact.events[0].sources.audio == pytest.approx(0.0)
    assert artifact.events[0].window.t0 == pytest.approx(1.17)
    assert artifact.events[0].window.t1 == pytest.approx(1.33)


def test_apply_review_inputs_to_contact_review_cli_writes_updated_review(tmp_path: Path) -> None:
    candidates_path = tmp_path / "contact_window_candidates.json"
    review_path = tmp_path / "contact_window_review.json"
    review_input_path = tmp_path / "review_input.json"
    out_review_path = tmp_path / "contact_window_review.updated.json"
    candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    review = build_contact_window_review_template(_candidate_payload(), candidate_path=str(candidates_path))
    write_contact_window_review(review_path, review)
    review_input_path.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_001": {
                        "contacts": [
                            {
                                "player": "P1",
                                "time_s": 1.49,
                                "note": "Visible paddle-ball contact in ball overlay.",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/apply_review_inputs_to_contact_review.py",
            "--candidates",
            str(candidates_path),
            "--review",
            str(review_path),
            "--review-input",
            str(review_input_path),
            "--clip",
            "clip_001",
            "--out-review",
            str(out_review_path),
            "--max-delta-s",
            "0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["accepted_count"] == 1
    updated = validate_artifact_file("contact_window_review", out_review_path)
    assert isinstance(updated, ContactWindowReview)
    assert updated.decisions[0].decision == "accepted"
    assert updated.decisions[0].player_id == 1


def test_build_contact_windows_from_review_inputs_cli_writes_contact_windows(tmp_path: Path) -> None:
    review_input_path = tmp_path / "review_input.json"
    contact_windows_path = tmp_path / "contact_windows.json"
    review_input_path.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_001": {
                        "contacts": [
                            {"player": "P1", "time_s": 1.25, "note": ""},
                            {"player": "P2", "time_s": 2.5, "note": ""},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_contact_windows_from_review_inputs.py",
            "--review-input",
            str(review_input_path),
            "--clip",
            "clip_001",
            "--out",
            str(contact_windows_path),
            "--fps",
            "60",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["clip"] == "clip_001"
    assert summary["event_count"] == 2
    artifact = validate_artifact_file("contact_windows", contact_windows_path)
    assert isinstance(artifact, ContactWindows)
    assert [event.frame for event in artifact.events] == [75, 150]
    assert [event.player_id for event in artifact.events] == [None, None]


def test_contact_window_review_cli_writes_template_and_promoted_contact_windows(tmp_path: Path) -> None:
    candidates_path = tmp_path / "contact_window_candidates.json"
    template_path = tmp_path / "contact_window_review.json"
    contact_windows_path = tmp_path / "contact_windows.json"
    candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/promote_contact_windows.py",
            "--candidates",
            str(candidates_path),
            "--template-out",
            str(template_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    template = validate_artifact_file("contact_window_review", template_path)
    assert isinstance(template, ContactWindowReview)
    assert template.summary.pending_count == 2

    review_payload = json.loads(template_path.read_text(encoding="utf-8"))
    review_payload["decisions"][0].update(
        {
            "decision": "accepted",
            "reviewer": "human",
            "reason": "Overlay confirms visible paddle-ball contact.",
            "player_id": 7,
        }
    )
    review_payload["summary"] = {
        "candidate_count": 2,
        "pending_count": 1,
        "accepted_count": 1,
        "rejected_count": 0,
    }
    write_contact_window_review(template_path, review_payload)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/promote_contact_windows.py",
            "--candidates",
            str(candidates_path),
            "--review",
            str(template_path),
            "--out-contact-windows",
            str(contact_windows_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["accepted_count"] == 1
    artifact = validate_artifact_file("contact_windows", contact_windows_path)
    assert isinstance(artifact, ContactWindows)
    assert artifact.events[0].sources.human_review == pytest.approx(1.0)


def test_contact_window_review_cli_refuses_empty_promotion_without_explicit_allow_empty(tmp_path: Path) -> None:
    candidates_path = tmp_path / "contact_window_candidates.json"
    template_path = tmp_path / "contact_window_review.json"
    contact_windows_path = tmp_path / "contact_windows.json"
    candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    template = build_contact_window_review_template(_candidate_payload(), candidate_path=str(candidates_path))
    write_contact_window_review(template_path, template)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/promote_contact_windows.py",
            "--candidates",
            str(candidates_path),
            "--review",
            str(template_path),
            "--out-contact-windows",
            str(contact_windows_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "no accepted contact decisions" in completed.stderr
    assert not contact_windows_path.exists()


def test_contact_window_review_render_cli_writes_html(tmp_path: Path) -> None:
    candidates_path = tmp_path / "contact_window_candidates.json"
    review_path = tmp_path / "contact_window_review.json"
    html_path = tmp_path / "contact_window_review.html"
    candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    review = build_contact_window_review_template(_candidate_payload(), candidate_path=str(candidates_path))
    write_contact_window_review(review_path, review)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_contact_window_review.py",
            "--candidates",
            str(candidates_path),
            "--review",
            str(review_path),
            "--out-html",
            str(html_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "event_smoke_contact" in html_path.read_text(encoding="utf-8")
