from __future__ import annotations

from pathlib import Path

from threed.racketsport.orchestrator import PoseStageRunner, StageContext


def _context(tmp_path: Path) -> StageContext:
    return StageContext(
        clip="clip_001",
        inputs_dir=tmp_path / "inputs",
        run_dir=tmp_path / "run",
        sport="pickleball",
    )


def test_pose_stage_runner_is_a_removed_stage_tombstone(tmp_path: Path) -> None:
    result = PoseStageRunner().run(_context(tmp_path))

    assert result.stage == "pose"
    assert result.status == "fail"
    assert result.real_model is False
    assert result.source_mode == "removed_legacy_pose_stage"
    assert result.produced_artifacts == ()
    assert any("run BODY" in note for note in result.notes)
    assert result.metrics["legacy_pose_stage_removed"] is True
    assert result.metrics["replacement_skeleton_source"] == "sam3d_body_joints"


def test_pose_stage_runner_tombstone_accepts_legacy_constructor_kwargs(tmp_path: Path) -> None:
    runner = PoseStageRunner(
        manifest_path=tmp_path / "MANIFEST.json",
        runtime=object(),
        motionbert_runtime=object(),
        model_id="legacy_id_ignored",
    )

    result = runner.run(_context(tmp_path))

    assert result.status == "fail"
    assert result.source_mode == "removed_legacy_pose_stage"
    assert result.produced_artifacts == ()
