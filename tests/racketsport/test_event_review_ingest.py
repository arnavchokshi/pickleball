from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.ingest_event_review_results import ingest_results


ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    video = tmp_path / "data/online_harvest_20260706/rallies/sourceA/clip_a.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=200x100:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    rows = [
        {
            "label_id": "els_001",
            "row": 1,
            "clip_id": "clip_a",
            "source_group": "sourceA",
            "video_path": video.relative_to(tmp_path).as_posix(),
            "video_sha256": "abc",
            "anchor_pts_s": 1.0,
            "stratum": "audio_onset",
            "score_band": "low",
            "suggested_split": "train",
        },
        {
            "label_id": "els_002",
            "row": 2,
            "clip_id": "clip_a",
            "source_group": "sourceA",
            "video_path": video.relative_to(tmp_path).as_posix(),
            "video_sha256": "abc",
            "anchor_pts_s": 2.0,
            "stratum": "uniform_random",
            "score_band": None,
            "suggested_split": "train",
        },
    ]
    manifest = {
        "session_id": "fixture_session",
        "seed": 20260715,
        "generator_version": "fixture_v1",
        "generator_sha256": "def",
        "rows": rows,
    }
    manifest_path = tmp_path / "session_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results = {
        "results_schema_version": 2,
        "session_id": "fixture_session",
        "page_generator_version": "fixture_page",
        "coords": "normalized to displayed video, origin top-left",
        "dt": "seconds in SOURCE time relative to the labeled anchor PTS",
        "answers": {
            "1": {"label_id": "els_001", "decision": "paddle", "x": 0.25, "y": 0.75, "dt": -0.1},
        },
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    return manifest_path, results_path, results


def test_ingest_writes_versioned_rows_pixels_counts_and_unanswered(tmp_path: Path) -> None:
    manifest_path, results_path, _ = _fixture(tmp_path)
    out_dir = tmp_path / "reviewed"
    dataset = ingest_results(results_path, manifest_path, out_dir, root=tmp_path)
    assert dataset["dataset_schema_version"] == 2
    assert dataset["answered_count"] == 1
    assert dataset["unanswered"] == [{"row": 2, "label_id": "els_002"}]
    assert dataset["counts_by_stratum_and_decision"] == {"audio_onset": {"paddle": 1}}
    row = json.loads((out_dir / "reviewed_labels_v2.jsonl").read_text(encoding="utf-8"))
    assert row["contact"] == {
        "source_height": 100,
        "source_width": 200,
        "x_norm": 0.25,
        "x_px": 50.0,
        "y_norm": 0.75,
        "y_px": 75.0,
    }
    assert row["dt_s"] == -0.1
    assert row["corrected_contact_pts_s"] == 0.9
    assert row["review"]["reviewed_by"] == "owner"
    assert row["provenance"]["seed"] == 20260715
    assert dataset["verified"] is False


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"label_id": "els_001", "decision": "hit", "x": 0.5, "y": 0.5, "dt": 0}, "invalid decision"),
        ({"label_id": "els_001", "decision": "paddle", "x": -0.01, "y": 0.5, "dt": 0}, "outside [0,1]"),
        ({"label_id": "els_001", "decision": "ground", "x": 0.5, "y": 0.5, "dt": 0.651}, "exceeds 0.65s"),
        ({"label_id": "els_001", "decision": "other", "x": 0.5, "y": 0.5}, "requires x, y, and dt"),
        ({"label_id": "els_001", "decision": "none", "x": 0.5}, "must not carry x/y/dt"),
        ({"label_id": "unknown", "decision": "unclear"}, "does not join manifest"),
    ],
)
def test_ingest_rejects_invalid_answers(tmp_path: Path, answer: dict, message: str) -> None:
    manifest_path, results_path, results = _fixture(tmp_path)
    results["answers"] = {"1": answer}
    results_path.write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        ingest_results(results_path, manifest_path, tmp_path / "out", root=tmp_path)


def test_ingest_rejects_duplicate_manifest_rows(tmp_path: Path) -> None:
    manifest_path, results_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(manifest["rows"][1])
    duplicate["row"] = 1
    manifest["rows"].append(duplicate)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate label_id or row"):
        ingest_results(results_path, manifest_path, tmp_path / "out", root=tmp_path)


def test_ingest_event_review_results_direct_cli(tmp_path: Path) -> None:
    """Direct subprocess coverage for scripts/racketsport/ingest_event_review_results.py."""
    manifest_path, results_path, _ = _fixture(tmp_path)
    out_dir = tmp_path / "cli_out"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/ingest_event_review_results.py",
            "--results",
            str(results_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["answered_count"] == 1
    assert (out_dir / "reviewed_labels_v2.jsonl").is_file()
    assert (out_dir / "dataset_manifest.json").is_file()
