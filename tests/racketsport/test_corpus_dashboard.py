from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.roboflow_corpus import load_protected_eval_hashes


CLI_PATH = "scripts/racketsport/corpus_dashboard.py"


def test_corpus_dashboard_cli_catches_protected_eval_hash_collision(tmp_path: Path) -> None:
    eval_hashes, _ = load_protected_eval_hashes(eval_root="eval_clips/ball")
    eval_clip, hashes = next(iter(eval_hashes.items()))
    collision_hash = int(hashes[0])
    aggregated = tmp_path / "roboflow" / "aggregated"
    _write_roboflow_index(
        aggregated,
        [
            {
                "sample_id": "synthetic:train:leaky",
                "source_slug": "synthetic/leaky",
                "split": "train",
                "bucket": "core_pickleball",
                "hashes": {"dhash": f"{collision_hash:016x}"},
                "label_kinds": ["ball"],
                "labels": {"ball": [{"visibility_level": "clear"}]},
            }
        ],
    )
    out_path = tmp_path / "dashboard.json"

    completed = _run_dashboard(
        tmp_path,
        "--roboflow-aggregated",
        str(aggregated),
        "--json",
        str(out_path),
    )

    assert completed.returncode == 1
    assert "synthetic:train:leaky" in completed.stdout
    assert eval_clip in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["leakage"]["eval_hash_collisions"] == 1


def test_corpus_dashboard_cli_catches_reserved_heldout_id_in_train_role(tmp_path: Path) -> None:
    harvest = tmp_path / "harvest"
    _write_json(
        harvest / "rally_clip_manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_online_harvest_rally_clip_manifest",
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "pwxNwFfYQlQ_rally_0001",
                    "source_id": "pwxNwFfYQlQ",
                    "source_channel": "Schwarz Pickleball",
                    "role": "train",
                    "clip_path": "data/online_harvest_20260706/rallies/pwxNwFfYQlQ/pwxNwFfYQlQ_rally_0001.mp4",
                }
            ],
        },
    )
    out_path = tmp_path / "dashboard.json"

    completed = _run_dashboard(tmp_path, "--harvest-root", str(harvest), "--json", str(out_path))

    assert completed.returncode == 1
    assert "pwxNwFfYQlQ" in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["leakage"]["heldout_id_hits"] == 1


def test_corpus_dashboard_owner_capture_absent_reports_present_false(tmp_path: Path) -> None:
    out_path = tmp_path / "dashboard.json"

    completed = _run_dashboard(tmp_path, "--json", str(out_path))

    assert completed.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    owner = payload["sources"]["owner_capture"]
    assert owner["present"] is False
    assert owner["counts"]["total"] == 0
    assert payload["leakage"]["eval_hash_collisions"] == 0
    assert payload["leakage"]["heldout_id_hits"] == 0


def test_corpus_dashboard_cli_help_is_directly_referenced() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--roboflow-aggregated" in completed.stdout
    assert "--owner-capture-manifest" in completed.stdout
    assert "--json" in completed.stdout


def test_scaffold_index_covers_corpus_dashboard_cli() -> None:
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

    assert by_path[CLI_PATH]["category"] == "dataset"
    assert by_path[CLI_PATH]["workstream"] == "DATA"
    assert by_path[CLI_PATH]["task_prefix"] == "P0-4"
    assert by_path[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_corpus_dashboard.py"


def _run_dashboard(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    missing = tmp_path / "missing"
    return subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--root",
            ".",
            "--roboflow-aggregated",
            str(missing / "roboflow"),
            "--harvest-root",
            str(missing / "harvest"),
            "--owner-capture-manifest",
            str(missing / "owner_capture" / "manifest.json"),
            "--review-label-root",
            str(missing / "review_labels"),
            "--eval-root",
            "eval_clips/ball",
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_roboflow_index(aggregated: Path, samples: list[dict]) -> None:
    aggregated.mkdir(parents=True, exist_ok=True)
    _write_json(
        aggregated / "corpus_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_roboflow_public_pretrain_corpus_index",
            "hash": {"collision_hamming_threshold": 3},
            "sample_count": len(samples),
            "samples": samples,
        },
    )
    _write_json(
        aggregated / "corpus_card.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_roboflow_public_pretrain_corpus_card",
            "corpus_index_sample_count": len(samples),
            "dedup": {"considered_sample_count": len(samples), "dedup_rate": 0.0},
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
