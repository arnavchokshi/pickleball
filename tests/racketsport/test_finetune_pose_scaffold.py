from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.finetune_pose import (
    DEFAULT_FINE_TUNE_ORDER,
    build_training_plan,
    validate_stage_order,
)


def _write_manifest(root: Path, sources: list[dict[str, object]]) -> Path:
    for source in sources:
        source_path = root / str(source["path"])
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("{}", encoding="utf-8")

    manifest = root / "pose_dataset_manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "dataset_id": "body4_sources", "sources": sources}),
        encoding="utf-8",
    )
    return manifest


def _source(source_id: str, source_type: str, split: str = "train", **metadata: object) -> dict[str, object]:
    return {
        "id": source_id,
        "source_type": source_type,
        "path": f"{source_type}/{source_id}.json",
        "split": split,
        **metadata,
    }


def test_validate_stage_order_accepts_only_canonical_body4_ladder():
    assert validate_stage_order("BEDLAM2,AthletePose3D,CalTennis,RICH,AMASS") == DEFAULT_FINE_TUNE_ORDER

    with pytest.raises(ValueError, match="must exactly match"):
        validate_stage_order("bedlam2,caltennis,athletepose3d,rich,amass")

    with pytest.raises(ValueError, match="duplicate"):
        validate_stage_order("bedlam2,athletepose3d,caltennis,rich,rich")

    with pytest.raises(ValueError, match="unknown"):
        validate_stage_order("bedlam2,athletepose3d,caltennis,rich,custom")


def test_build_training_plan_records_scaffold_only_coverage_without_training(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            _source("bedlam2_seed", "bedlam2", fps=30, frame_count=8_000_000, license="research"),
            _source("athletepose3d_train", "athletepose3d", fps=120, frame_count=1_300_000),
            _source("caltennis_train", "caltennis", fps=60, frame_count=11_000_000),
            _source("rich_contact", "rich", joint_set="SMPL-X", frame_count=90_000),
            _source("amass_prior", "amass", joint_set="SMPL", notes=["motion prior"]),
            _source("emdb_eval", "emdb_eval", split="eval", notes=["world trajectory eval placeholder"]),
        ],
    )

    plan = build_training_plan(manifest, out_dir=tmp_path / "finetuned")

    assert plan["task_id"] == "BODY-4"
    assert plan["status"] == "scaffold_only_ready_for_gpu_handoff"
    assert plan["frozen_scaffold"] is True
    assert plan["training_enabled"] is False
    assert plan["downloads_enabled"] is False
    assert plan["checkpoint_selection_enabled"] is False
    assert plan["manifest_mutation_enabled"] is False
    assert plan["fine_tune_order"] == ["bedlam2", "athletepose3d", "caltennis", "rich", "amass"]
    assert plan["coverage_gaps"] == []
    assert plan["h100_lease"]["required_for_real_fine_tune"] is True
    assert plan["h100_lease"]["mode"] == "exclusive_7g_80gb_training"
    assert [stage["source_type"] for stage in plan["stages"]] == plan["fine_tune_order"]
    assert plan["stages"][0]["source_ids"] == ["bedlam2_seed"]
    assert plan["stages"][0]["source_metadata"][0]["frame_count"] == 8_000_000
    assert plan["eval_datasets"] == [
        {"source_type": "emdb_eval", "status": "placeholder_not_measured"},
        {"source_type": "caltennis", "status": "placeholder_not_measured"},
        {"source_type": "athletepose3d", "status": "placeholder_not_measured"},
    ]


def test_cli_writes_plan_and_fails_closed_when_ladder_sources_are_missing(tmp_path):
    manifest = _write_manifest(tmp_path, [_source("bedlam2_seed", "bedlam2")])
    out_dir = tmp_path / "plans"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/finetune_pose.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    plan_path = out_dir / "pose_finetune_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert "missing fine-tune ladder source: athletepose3d" in completed.stderr
    assert plan["status"] == "blocked_missing_sources"
    assert plan["training_enabled"] is False
    assert plan["coverage_gaps"] == [
        "missing fine-tune ladder source: athletepose3d",
        "missing fine-tune ladder source: caltennis",
        "missing fine-tune ladder source: rich",
        "missing fine-tune ladder source: amass",
        "no emdb_eval entries registered for eval coverage",
    ]
