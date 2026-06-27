from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.eval import (
    ball_event_eval,
    body_eval,
    copy_faithfulness,
    e2e_eval,
    metric_eval,
    physics_eval,
    racket_eval,
    replay_eval,
    shot_drill_eval,
    track_eval,
)
from tests.racketsport.test_eval_ball_racket_gates import (
    _write_ball_event_artifacts,
    _write_racket_pose_artifact,
    _write_ready_clip,
)


def test_no_required_phase_gate_uses_greater_equal_zero_threshold() -> None:
    gated_groups = {
        "ball_events": ball_event_eval.BALL_EVENT_GATES,
        "physics": physics_eval.PHYSICS_GATES,
        "racket": racket_eval.RACKET_GATES,
    }

    offenders = [
        f"{group}.{metric_name}"
        for group, gates in gated_groups.items()
        for metric_name, gate in gates.items()
        if gate.op == ">=" and gate.threshold == 0
    ]

    assert offenders == []


def test_presence_and_artifact_checks_are_labeled_without_accuracy_claims() -> None:
    presence_gate_groups = {
        "tracking": track_eval.TRACK_GATES,
        "body": body_eval.BODY_GATES,
        "physics": physics_eval.PHYSICS_GATES,
        "ball_events": ball_event_eval.BALL_EVENT_GATES,
        "racket": racket_eval.RACKET_GATES,
        "metrics": metric_eval.METRIC_GATES,
        "shot_drill": shot_drill_eval.SHOT_DRILL_GATES,
        "copy": copy_faithfulness.COPY_GATES,
        "replay": replay_eval.REPLAY_GATES,
    }

    offenders = [
        f"{group}.{metric_name}:{gate.label}"
        for group, gates in presence_gate_groups.items()
        for metric_name, gate in gates.items()
        if not gate.label.startswith("presence_check.")
    ]
    assert offenders == []

    artifact_offenders = [
        f"e2e.{metric_name}:{gate.label}"
        for metric_name, gate in e2e_eval.E2E_ARTIFACT_GATES.items()
        if not gate.label.startswith("artifact_check.")
    ]
    assert artifact_offenders == []


def test_ball_event_eval_rejects_zero_contact_and_bounce_evidence(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    _write_ready_clip(labels_root, "clip_001")
    _write_ball_event_artifacts(root / "clip_001", frames=1, contacts=0, bounces=0)

    payload = ball_event_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["contact_events"].value == 0
    assert metrics["contact_events"].passed is False
    assert metrics["bounce_events"].value == 0
    assert metrics["bounce_events"].passed is False


def test_racket_eval_rejects_zero_contact_evidence(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")
    _write_racket_pose_artifact(root / "clip_001", players=1, frames=1, contacts=0)

    payload = racket_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["racket_contacts"].value == 0
    assert metrics["racket_contacts"].passed is False


def test_physics_eval_rejects_zero_contact_or_ground_force_evidence(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase4"
    _write_ready_clip(labels_root, "clip_001")
    _write_physics_without_contact_or_grf(root / "clip_001")

    payload = physics_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["foot_contact_frames"].value == 0
    assert metrics["foot_contact_frames"].passed is False
    assert metrics["grf_frames"].value == 0
    assert metrics["grf_frames"].passed is False


def _write_physics_without_contact_or_grf(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": "smplx",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 1,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "physpt",
                "frames": [
                    {
                        "t": 0.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.0, 0.0, 0.0],
                        "joints_world": [[0.0, 0.0, 0.0]],
                        "joint_conf": [0.9],
                        "foot_contact": {"left": False, "right": False},
                        "grf": None,
                    },
                ],
            }
        ],
    }
    (run_dir / "smpl_motion.json").write_text(json.dumps(payload), encoding="utf-8")
