from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DISAGREE_CLI = "scripts/racketsport/export_sst_disagreements.py"


def test_sst_manifest_builds_from_one_real_local_prelabel_clip(tmp_path: Path) -> None:
    from threed.racketsport.ball_sst_dataset import build_sst_manifest

    prelabel_root = Path("data/online_harvest_20260706/prelabels")
    rally_root = Path("data/online_harvest_20260706/rallies")
    clip_id = "73VurrTKCZ8_rally_0001"
    if not (prelabel_root / clip_id / "ball_track.json").is_file():
        pytest.skip("local harvest prelabel sidecars are missing")
    if not (rally_root / "73VurrTKCZ8" / f"{clip_id}.mp4").is_file():
        pytest.skip("local harvest rally frames are missing")

    manifest_path = tmp_path / "sst_manifest.json"
    manifest = build_sst_manifest(
        prelabel_root=prelabel_root,
        rally_root=rally_root,
        out_path=manifest_path,
        clips=[clip_id],
        max_samples_per_clip=3,
        protected_eval_hashes={"synthetic_eval": ["ffffffffffffffff"]},
        expected_protected_eval_hash_count=1,
    )

    assert manifest_path.is_file()
    assert manifest["artifact_type"] == "racketsport_ball_sst_manifest"
    assert manifest["summary"]["clip_count"] == 1
    assert manifest["summary"]["sample_count"] == 3
    sample = manifest["clips"][0]["samples"][0]
    assert sample["clip_id"] == clip_id
    assert sample["frame_ref"]["video"].endswith(f"{clip_id}.mp4")
    assert isinstance(sample["teacher_xy"], list)
    assert 0.0 <= sample["score"] <= 1.0
    assert sample["weight"] == pytest.approx(sample["score"])


def test_sst_protected_hash_guard_fires_on_synthetic_collision() -> None:
    from threed.racketsport.ball_sst_dataset import assert_no_sst_protected_eval_hash_collisions
    from threed.racketsport.roboflow_corpus import ProtectedEvalHashCollisionError

    with pytest.raises(ProtectedEvalHashCollisionError, match="protected eval hash collision"):
        assert_no_sst_protected_eval_hash_collisions(
            [
                {
                    "sample_id": "clip_a:0",
                    "source_slug": "clip_a",
                    "hashes": {"dhash": "0000000000000000"},
                }
            ],
            protected_eval_hashes={"eval_clip": ["0000000000000000"]},
            threshold=0,
            expected_protected_eval_hash_count=1,
        )


def test_disagreement_emitter_outputs_ranked_fixture_queue(tmp_path: Path) -> None:
    from threed.racketsport.ball_sst_dataset import build_sst_disagreement_queue

    teacher_root = tmp_path / "teacher"
    student_root = tmp_path / "student"
    _write_ball_track(
        teacher_root / "clip_a" / "ball_track.json",
        [
            (0, True, [10.0, 10.0], 0.90),
            (1, True, [20.0, 20.0], 0.80),
            (2, False, [0.0, 0.0], 0.20),
        ],
    )
    _write_ball_track(
        student_root / "clip_a" / "ball_track.json",
        [
            (0, False, [0.0, 0.0], 0.10),
            (1, True, [55.0, 20.0], 0.70),
            (2, True, [30.0, 30.0], 0.60),
        ],
    )
    out_path = tmp_path / "queue.json"

    queue = build_sst_disagreement_queue(
        teacher_predictions=teacher_root,
        student_predictions=student_root,
        out_path=out_path,
        large_offset_px=25.0,
    )

    assert out_path.is_file()
    assert queue["artifact_type"] == "racketsport_ball_sst_disagreement_queue"
    assert queue["summary"] == {
        "clip_count": 1,
        "disagreement_count": 3,
        "teacher_only_count": 1,
        "student_only_count": 1,
        "large_offset_count": 1,
    }
    assert [item["disagreement_type"] for item in queue["queue"]] == [
        "large-offset",
        "teacher-only",
        "student-only",
    ]
    assert queue["queue"][0]["offset_px"] == pytest.approx(35.0)


def test_export_sst_disagreements_cli_help_is_indexed() -> None:
    completed = subprocess.run(
        [sys.executable, DISAGREE_CLI, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--teacher-predictions" in completed.stdout
    assert "--student-predictions" in completed.stdout
    assert "--large-offset-px" in completed.stdout


def test_scaffold_index_covers_ball_stage2_sst_clis() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    by_path = {tool["command_path"]: tool for tool in payload["tools"]}

    assert by_path[DISAGREE_CLI]["category"] == "ball"
    assert by_path[DISAGREE_CLI]["workstream"] == "BALL"
    assert by_path[DISAGREE_CLI]["direct_cli_reference_test"] == "tests/racketsport/test_ball_stage2_sst.py"


def _write_ball_track(path: Path, rows: list[tuple[int, bool, list[float], float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    fps = 30.0
    for frame_index, visible, xy, conf in rows:
        frames.append(
            {
                "t": frame_index / fps,
                "visible": visible,
                "xy": xy,
                "conf": conf,
                "approx": False,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "ball_track",
                "source": "wasb",
                "fps": fps,
                "frames": frames,
                "bounces": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
