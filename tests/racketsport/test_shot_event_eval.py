from __future__ import annotations

import json
import subprocess
import sys

import pytest

from threed.racketsport.eval.shot_event_eval import match_shot_events, score_shot_events


def _truth_event(event_id: str, t: float, label: str, player_id: int = 1) -> dict[str, object]:
    return {"id": event_id, "t": t, "frame": int(t * 60), "player_id": player_id, "shot_label": label}


def _prediction(
    event_id: str,
    t: float,
    label: str,
    confidence: float,
    *,
    top2: list[tuple[str, float]] | None = None,
    gated: bool = False,
    player_id: int = 1,
) -> dict[str, object]:
    return {
        "id": event_id,
        "t": t,
        "frame": int(t * 60),
        "player_id": player_id,
        "type": label,
        "type_conf": confidence,
        "gated": gated,
        "top2": [{"type": top_label, "confidence": top_conf} for top_label, top_conf in (top2 or [])],
    }


def test_match_shot_events_pairs_nearest_prediction_inside_tolerance_once() -> None:
    truth = [
        _truth_event("truth_a", 1.0, "serve"),
        _truth_event("truth_b", 2.0, "fh_shot"),
    ]
    predictions = [
        _prediction("pred_far", 1.4, "serve", 0.9),
        _prediction("pred_near", 1.08, "serve", 0.8),
        _prediction("pred_b", 2.02, "fh_shot", 0.7),
    ]

    matches = match_shot_events(truth, predictions, tolerance_s=0.15)

    assert [(match.truth_id, match.prediction_id, match.dt_s) for match in matches.matched] == [
        ("truth_a", "pred_near", 0.08),
        ("truth_b", "pred_b", 0.02),
    ]
    assert matches.unmatched_truth_ids == []
    assert matches.unmatched_prediction_ids == ["pred_far"]


def test_score_shot_events_reports_accuracy_top2_confusion_unknowns_and_calibration() -> None:
    truth = [
        _truth_event("truth_serve", 1.0, "serve"),
        _truth_event("truth_fh", 2.0, "fh_shot"),
        _truth_event("truth_bh", 3.0, "bh_shot"),
        _truth_event("truth_overhead", 4.0, "overhead"),
    ]
    predictions = [
        _prediction("pred_serve", 1.03, "fh_shot", 0.61, top2=[("fh_shot", 0.61), ("bh_shot", 0.45)]),
        _prediction("pred_fh", 2.02, "fh_shot", 0.82, top2=[("fh_shot", 0.82), ("serve", 0.12)]),
        _prediction("pred_bh", 3.04, "fh_shot", 0.56, top2=[("fh_shot", 0.56), ("bh_shot", 0.41)]),
        _prediction("pred_overhead", 4.01, "unknown", 0.2, gated=True),
        _prediction("pred_extra", 5.0, "dink", 0.9),
    ]

    score = score_shot_events(truth, predictions, tolerance_s=0.10)

    assert score["sample_count"] == 4
    assert score["matched_count"] == 4
    assert score["unmatched_prediction_count"] == 1
    assert score["top1_correct"] == 1
    assert score["top2_correct"] == 2
    assert score["accuracy"] == pytest.approx(0.25)
    assert score["top2_accuracy"] == pytest.approx(0.5)
    assert score["macro_f1"] == pytest.approx(0.125)
    assert score["unknown_count"] == 1
    assert score["gated_count"] == 1
    assert score["confusion"] == {
        "bh_shot": {"fh_shot": 1},
        "fh_shot": {"fh_shot": 1},
        "overhead": {"unknown": 1},
        "serve": {"fh_shot": 1},
    }
    assert score["by_label"]["fh_shot"]["precision"] == pytest.approx(1 / 3)
    assert score["by_label"]["fh_shot"]["recall"] == pytest.approx(1.0)
    assert score["by_label"]["fh_shot"]["f1"] == pytest.approx(0.5)
    assert score["calibration"]["0.50-0.70"] == {
        "count": 2,
        "correct": 0,
        "accuracy": 0.0,
        "mean_confidence": pytest.approx(0.585),
    }
    assert score["matches"][0]["truth_label"] == "serve"
    assert score["matches"][0]["predicted_label"] == "fh_shot"


def test_shot_event_eval_cli_scores_json_files(tmp_path) -> None:
    truth_path = tmp_path / "truth.json"
    predictions_path = tmp_path / "predictions.json"
    out_path = tmp_path / "shot_event_score.json"

    truth_path.write_text(
        json.dumps({"events": [_truth_event("truth_serve", 1.0, "serve")]}),
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps({"shots": [_prediction("pred_serve", 1.02, "serve", 0.9)]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.shot_event_eval",
            "--truth",
            str(truth_path),
            "--predictions",
            str(predictions_path),
            "--out",
            str(out_path),
            "--tolerance-s",
            "0.10",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["accuracy"] == 1.0
    assert payload["matches"][0]["prediction_id"] == "pred_serve"
