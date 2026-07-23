from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path

import pytest

from threed.racketsport.event_head.sequence_dp import (
    SequenceDPError,
    apply_event_sequence_dp,
    selected_constraint_violations,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/racketsport/apply_event_sequence_dp.py"


def _fixture(*, events: list[tuple[int, float, float, str | None]] | None = None) -> dict:
    probabilities = [[0.98, 0.01, 0.01] for _ in range(100)]
    sides: list[str | None] = [None] * 100
    configured = events or [
        (10, 0.94, 0.05, "A"),
        (13, 0.70, 0.29, "A"),  # too close to frame 10
        (30, 0.94, 0.05, "B"),
        (33, 0.70, 0.29, "A"),  # too close to frame 30
        (42, 0.55, 0.44, "B"),  # bounded same-side penalty after frame 30
        (50, 0.94, 0.05, "A"),
        (70, 0.94, 0.05, "B"),
        (73, 0.70, 0.29, "A"),  # too close to frame 70
        (90, 0.94, 0.05, "A"),
    ]
    for frame, hit_probability, background_probability, side in configured:
        probabilities[frame] = [background_probability, hit_probability, 0.01]
        sides[frame] = side
    return {
        "schema_version": 1,
        "artifact_type": "event_head_sequence_input",
        "verified": False,
        "class_names": ["background", "HIT", "BOUNCE"],
        "ground_truth_policy": "dense_exhaustive",
        "clips": [{
            "clip_id": "synthetic_rally",
            "fps": 10.0,
            "probabilities": probabilities,
            "hit_side_by_frame": sides,
            "rally_spans": [{"rally_id": "rally_1", "start_frame": 0, "end_frame": 100}],
            "ground_truth_complete": True,
            "ground_truth": [
                {"frame": frame, "class": "HIT", "side": side}
                for frame, side in ((10, "A"), (30, "B"), (50, "A"), (70, "B"), (90, "A"))
            ],
        }],
    }


def test_default_off_is_in_memory_and_byte_for_byte_cli_identity(tmp_path: Path) -> None:
    payload = _fixture()
    before = copy.deepcopy(payload)
    result = apply_event_sequence_dp(payload)
    assert result == before
    assert result is not payload
    assert payload == before

    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload, separators=(",", ":")))
    for name, mode in (("default", []), ("explicit_off", ["--off"])):
        output = tmp_path / f"{name}.json"
        completed = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"), str(CLI), "--input", str(source),
                "--out", str(output), *mode,
            ],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert output.read_bytes() == source.read_bytes()


def test_dp_is_deterministic_and_removes_violations_without_losing_true_events() -> None:
    payload = _fixture()
    first = apply_event_sequence_dp(payload, enabled=True)
    second = apply_event_sequence_dp(payload, enabled=True)
    assert first == second
    assert payload["artifact_type"] == "event_head_sequence_input"

    rally = first["clips"][0]["sequence_dp_rallies"][0]
    assert rally["status"] == "applied"
    assert [event["frame"] for event in rally["raw_predictions"]] == [
        10, 13, 30, 33, 42, 50, 70, 73, 90,
    ]
    assert [event["frame"] for event in rally["selected_predictions"]] == [10, 30, 50, 70, 90]
    assert selected_constraint_violations(rally, fps=10.0) == []
    assert all(
        event["score_trace"]["frame"] == event["frame"]
        for event in rally["selected_predictions"]
    )

    metrics = first["sequence_dp_evaluation"]
    assert metrics["scoreable"] is True
    one_frame = metrics["tolerance_sweep"][0]
    assert one_frame["raw"]["micro"] == pytest.approx({
        "tp": 5, "fp": 4, "fn": 0, "precision": 5 / 9, "recall": 1.0, "f1": 5 / 7,
    })
    assert one_frame["dp"]["micro"] == pytest.approx({
        "tp": 5, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    })


def test_alternation_is_soft_and_missing_side_abstains() -> None:
    payload = _fixture(events=[
        (10, 0.90, 0.09, "A"),
        (40, 0.90, 0.09, "A"),  # same side remains selectable; not a veto
        (70, 0.90, 0.09, None),  # missing tracking identity abstains
    ])
    payload["clips"][0]["ground_truth"] = [
        {"frame": 10, "class": "HIT"},
        {"frame": 40, "class": "HIT"},
        {"frame": 70, "class": "HIT"},
    ]
    result = apply_event_sequence_dp(payload, enabled=True)
    selected = result["clips"][0]["sequence_dp_rallies"][0]["selected_predictions"]
    assert [event["frame"] for event in selected] == [10, 40, 70]


def test_implausible_raw_rate_fails_closed_instead_of_laundering_predictions() -> None:
    events = [(frame, 0.90, 0.09, None) for frame in range(1, 100, 9)]
    payload = _fixture(events=events)
    payload["ground_truth_policy"] = "none"
    payload["clips"][0]["ground_truth_complete"] = False
    payload["clips"][0]["ground_truth"] = []
    result = apply_event_sequence_dp(payload, enabled=True)
    rally = result["clips"][0]["sequence_dp_rallies"][0]
    assert rally["raw_rate_hz"] == pytest.approx(1.1)
    assert rally["status"] == "ineligible_raw_rate"
    assert rally["selected_predictions"] is None
    assert result["clips"][0]["typed_event_anchors"] == []
    assert result["sequence_dp_evaluation"]["scoreable"] is False


def test_enabled_mode_accepts_saved_logits_and_rejects_invalid_dense_claim() -> None:
    payload = _fixture(events=[
        (10, 0.90, 0.09, "A"), (40, 0.90, 0.09, "B"), (70, 0.90, 0.09, "A"),
    ])
    clip = payload["clips"][0]
    clip["logits"] = [[math.log(value) for value in row] for row in clip.pop("probabilities")]
    assert apply_event_sequence_dp(payload, enabled=True)["clips"][0][
        "sequence_dp_rallies"
    ][0]["status"] == "applied"

    valid = _fixture()
    valid["clips"][0]["ground_truth_complete"] = False
    with pytest.raises(SequenceDPError, match="ground_truth_complete"):
        apply_event_sequence_dp(valid, enabled=True)


def test_cli_enable_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    source.write_text(json.dumps(_fixture(), sort_keys=True) + "\n")
    for output in (first, second):
        completed = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"), str(CLI), "--input", str(source),
                "--out", str(output), "--enable",
            ],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["scoreable"] is True
    assert first.read_bytes() == second.read_bytes()
