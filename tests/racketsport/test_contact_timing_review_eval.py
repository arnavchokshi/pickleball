from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.eval.contact_timing_review_eval import evaluate_review_alignment


def _contact_windows(times: list[float], *, fps: float = 60.0) -> dict:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": time_s,
                "frame": max(0, int(round(time_s * fps))),
                "player_id": None,
                "confidence": 1.0,
                "sources": {"audio": 0.0, "wrist_vel": 0.0, "ball_inflection": 0.0, "human_review": 1.0},
                "window": {"t0": max(0.0, time_s - 0.08), "t1": time_s + 0.08, "importance": 1.0},
            }
            for time_s in times
        ],
    }


def test_contact_timing_review_eval_reports_signed_frame_deltas_without_claiming_ball_verified(tmp_path: Path) -> None:
    review_input_path = tmp_path / "pickleball_cv_review_latest.json"
    review_input_path.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_001": {
                        "contacts": [
                            {"player": "P1", "time_s": 1.0, "note": ""},
                            {"player": "P1", "time_s": 2.0, "note": ""},
                            {"player": "P1", "time_s": 4.0, "note": ""},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "eval0"
    clip_dir = run_root / "clip_001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "contact_windows.json").write_text(
        json.dumps(_contact_windows([1.0 + 1.0 / 60.0, 2.0 - 1.0 / 60.0, 5.0])),
        encoding="utf-8",
    )

    report = evaluate_review_alignment(
        review_input_path=review_input_path,
        run_root=run_root,
        clips=["clip_001"],
        fps=60.0,
        max_match_delta_frames=2.0,
    )

    assert report["artifact_type"] == "racketsport_contact_timing_review_alignment"
    assert report["verification_scope"] == "human_review_alignment_only"
    assert report["ball_verified"] is False
    assert report["status"] == "review_alignment_needs_attention"
    assert "does not verify machine BALL contact detection" in " ".join(report["notes"])

    assert report["summary"]["reviewed_contact_count"] == 3
    assert report["summary"]["promoted_contact_count"] == 3
    assert report["summary"]["matched_contact_count"] == 2
    assert report["summary"]["missing_reviewed_contact_count"] == 1
    assert report["summary"]["extra_promoted_contact_count"] == 1
    assert report["summary"]["reviewed_contacts_within_2_frames_rate"] == pytest.approx(2.0 / 3.0)

    clip_report = report["clips"][0]
    assert [match["signed_delta_frames"] for match in clip_report["matches"]] == pytest.approx([1.0, -1.0])
    assert [match["signed_frame_index_delta"] for match in clip_report["matches"]] == [1, -1]
    assert clip_report["missing_reviewed_contacts"] == [{"time_s": 4.0, "frame": 240}]
    assert clip_report["extra_promoted_contacts"] == [{"t": 5.0, "frame": 300}]
    assert clip_report["metrics"]["mean_abs_delta_frames"] == pytest.approx(1.0)
    assert clip_report["metrics"]["p90_abs_delta_frames"] == pytest.approx(1.0)


def test_contact_timing_review_eval_cli_writes_report(tmp_path: Path) -> None:
    review_input_path = tmp_path / "pickleball_cv_review_latest.json"
    review_input_path.write_text(
        json.dumps({"clips": {"clip_001": {"contacts": [{"player": "P1", "time_s": 1.25, "note": ""}]}}}),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "eval0"
    clip_dir = run_root / "clip_001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "contact_windows.json").write_text(json.dumps(_contact_windows([1.25])), encoding="utf-8")
    out_path = tmp_path / "contact_timing_review_alignment.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_contact_timing_from_review_inputs.py",
            "--review-input",
            str(review_input_path),
            "--run-root",
            str(run_root),
            "--clip",
            "clip_001",
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "review_alignment_ok"
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["matched_contact_count"] == 1
    assert report["ball_verified"] is False


def test_contact_timing_review_eval_does_not_mark_empty_contact_sets_ok(tmp_path: Path) -> None:
    review_input_path = tmp_path / "pickleball_cv_review_latest.json"
    review_input_path.write_text(json.dumps({"clips": {"clip_001": {"contacts": []}}}), encoding="utf-8")
    run_root = tmp_path / "runs" / "eval0"
    clip_dir = run_root / "clip_001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "contact_windows.json").write_text(json.dumps(_contact_windows([])), encoding="utf-8")

    report = evaluate_review_alignment(review_input_path=review_input_path, run_root=run_root, clips=["clip_001"])

    assert report["status"] == "review_alignment_needs_attention"
    assert report["summary"]["reviewed_contact_count"] == 0
    assert report["summary"]["all_reviewed_contacts_promoted_within_tolerance"] is False
    assert report["ball_verified"] is False
