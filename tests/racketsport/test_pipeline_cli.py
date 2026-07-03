from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.test_pipeline_contracts import _artifact_payload
from tests.racketsport.test_orchestrator_spine import _sidecar_payload
from threed.racketsport import pipeline_cli


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_seed_artifacts(seed_dir: Path, names: list[str]) -> None:
    for name in names:
        payload = _sidecar_payload() if name == "capture_sidecar.json" else _artifact_payload(name)
        _write_json(seed_dir / name, payload)


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test video placeholder")


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "threed.racketsport.pipeline_cli", *args],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )


def test_list_stages_reports_tier_contract_and_model() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "threed.racketsport.pipeline_cli", "--list-stages", "--json"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    stages = {stage["name"]: stage for stage in payload["stages"]}
    assert stages["capture_sidecar"]["tier"] == "on_device"
    assert stages["capture_sidecar"]["artifact"] == "capture_sidecar.json"
    assert stages["court_calibration"]["artifact"] == "court_calibration.json"
    assert stages["tracks"]["model"]
    assert stages["replay"]["tier"] == "server_offline"


def test_on_device_tier_runs_only_live_stages_from_valid_seed(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    _write_video(video)
    _write_seed_artifacts(
        seed,
        [
            "capture_sidecar.json",
            "court_calibration.json",
            "tracks.json",
            "ball_track.json",
            "contact_windows.json",
        ],
    )

    completed = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--artifact-source",
        str(seed),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--artifacts-root",
        str(tmp_path / "artifacts"),
        "--tier",
        "on_device",
        "--json",
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert [stage["name"] for stage in summary["stages"]] == [
        "capture_sidecar",
        "court_calibration",
        "tracks",
        "ball_track",
        "contact_windows",
    ]
    assert all(stage["capability_status"] == "RUNS" for stage in summary["stages"])
    assert all(stage["tier"] == "on_device" for stage in summary["stages"])
    assert not any(stage["name"] == "replay" for stage in summary["stages"])


def test_stage_range_and_skip_valid_outputs_are_resumable(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    _write_video(video)
    _write_seed_artifacts(seed, ["capture_sidecar.json", "court_calibration.json", "tracks.json"])

    first = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--artifact-source",
        str(seed),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--artifacts-root",
        str(tmp_path / "artifacts"),
        "--from",
        "court_calibration",
        "--to",
        "tracks",
        "--json",
    )
    assert first.returncode == 0, first.stderr + first.stdout

    second = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--artifact-source",
        str(seed),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--artifacts-root",
        str(tmp_path / "artifacts"),
        "--from",
        "court_calibration",
        "--to",
        "tracks",
        "--json",
    )
    assert second.returncode == 0, second.stderr + second.stdout
    summary = json.loads(second.stdout)
    assert [stage["run_status"] for stage in summary["stages"]] == ["skipped", "skipped"]
    assert summary["status"] == "RUNS"


def test_invalid_existing_artifact_fails_closed_without_reusing_it(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    artifacts_root = tmp_path / "artifacts"
    clip_dir = artifacts_root / "clip"
    _write_video(video)
    _write_seed_artifacts(seed, ["capture_sidecar.json", "court_calibration.json"])
    _write_json(clip_dir / "court_calibration.json", {"schema_version": 1, "bad": True})

    completed = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--artifact-source",
        str(seed),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--artifacts-root",
        str(artifacts_root),
        "--stage",
        "court_calibration",
        "--json",
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert summary["status"] == "FAILED"
    assert summary["stages"][0]["run_status"] == "failed"
    assert "invalid existing artifact" in summary["stages"][0]["message"]


def test_validate_pipeline_artifacts_supports_public_contracts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    artifacts_root = tmp_path / "artifacts"
    _write_video(video)
    _write_seed_artifacts(seed, ["capture_sidecar.json", "court_calibration.json", "tracks.json"])

    run = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--artifact-source",
        str(seed),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--artifacts-root",
        str(artifacts_root),
        "--from",
        "capture_sidecar",
        "--to",
        "tracks",
        "--json",
    )
    assert run.returncode == 0, run.stderr + run.stdout

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_pipeline_artifacts.py",
            "--run-dir",
            str(artifacts_root / "clip"),
            "--public-contracts",
            "--stage",
            "tracks",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["artifact_type"] == "pickleball_public_pipeline_contract_readiness"
    assert payload["status"] == "ready"
    assert [stage["stage"] for stage in payload["stages"]] == ["capture_sidecar", "court_calibration", "tracks"]


# ---------------------------------------------------------------------------
# --allow-fixture-fallback / --force behavior (guard-lane fixes)
# ---------------------------------------------------------------------------


def test_fixture_fallback_is_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale prototype-gate fixture must not be silently substituted by default."""

    video = tmp_path / "clip.mp4"
    _write_video(video)
    fixture_root = tmp_path / "fixtures"
    _write_json(fixture_root / "clip" / "capture_sidecar.json", _sidecar_payload())
    monkeypatch.setattr(pipeline_cli, "DEFAULT_SAMPLE_ARTIFACT_ROOT", fixture_root)

    summary = pipeline_cli.run_top_level_pipeline(
        video=video,
        selected_stages=["capture_sidecar"],
        inputs_root=tmp_path / "inputs",
        artifacts_root=tmp_path / "artifacts",
    )

    assert summary["status"] == "FAILED"
    assert summary["stages"][0]["run_status"] == "failed"
    assert "missing required contract artifact" in summary["stages"][0]["message"]


def test_allow_fixture_fallback_flag_enables_prototype_sample_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact same fixture is usable once the caller opts in explicitly."""

    video = tmp_path / "clip.mp4"
    _write_video(video)
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "clip"
    _write_json(fixture_dir / "capture_sidecar.json", _sidecar_payload())
    monkeypatch.setattr(pipeline_cli, "DEFAULT_SAMPLE_ARTIFACT_ROOT", fixture_root)

    summary = pipeline_cli.run_top_level_pipeline(
        video=video,
        selected_stages=["capture_sidecar"],
        inputs_root=tmp_path / "inputs",
        artifacts_root=tmp_path / "artifacts",
        allow_fixture_fallback=True,
    )

    assert summary["stages"][0]["run_status"] == "ran"
    assert str(fixture_dir) in summary["stages"][0]["message"]


def test_allow_fixture_fallback_cli_flag_forwards_to_pipeline(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    _write_video(video)
    _write_seed_artifacts(seed, ["capture_sidecar.json"])

    without_flag = _run_cli(
        tmp_path,
        "--video",
        str(video),
        "--inputs-root",
        str(tmp_path / "inputs_no_flag"),
        "--artifacts-root",
        str(tmp_path / "artifacts_no_flag"),
        "--stage",
        "capture_sidecar",
        "--json",
    )
    assert without_flag.returncode == 1
    no_flag_summary = json.loads(without_flag.stdout)
    assert no_flag_summary["status"] == "FAILED"

    help_output = _run_cli(tmp_path, "--help")
    assert "--allow-fixture-fallback" in help_output.stdout


def test_default_force_still_uses_artifact_source_copy_not_spine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    _write_video(video)
    _write_seed_artifacts(seed, ["capture_sidecar.json", "court_calibration.json"])

    spine_calls = {"count": 0}

    def fake_run_existing_spine(stage, ctx):  # noqa: ANN001
        spine_calls["count"] += 1
        raise AssertionError("spine must not run when a valid artifact source copy is available")

    monkeypatch.setattr(pipeline_cli, "_run_existing_spine", fake_run_existing_spine)

    summary = pipeline_cli.run_top_level_pipeline(
        video=video,
        selected_stages=["court_calibration"],
        inputs_root=tmp_path / "inputs",
        artifacts_root=tmp_path / "artifacts",
        artifact_source=seed,
    )

    assert spine_calls["count"] == 0
    assert summary["stages"][0]["run_status"] == "ran"
    assert "from source artifact" in summary["stages"][0]["message"]


def test_force_skips_artifact_source_copy_and_runs_real_spine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force must bypass any artifact-source copy and force real spine execution."""

    video = tmp_path / "clip.mp4"
    seed = tmp_path / "seed"
    _write_video(video)
    # A valid source artifact IS available -- proving force ignores it entirely
    # rather than opportunistically copying it.
    _write_seed_artifacts(seed, ["capture_sidecar.json", "court_calibration.json"])

    spine_calls = {"count": 0}

    def fake_run_existing_spine(stage, ctx):  # noqa: ANN001
        spine_calls["count"] += 1
        return {
            "name": stage.name,
            "artifact": stage.artifact,
            "schema": stage.schema,
            "tier": stage.tier,
            "tier_label": stage.tier_label,
            "latency_budget": stage.latency_budget,
            "model": stage.model,
            "output_timing": stage.output_timing,
            "capability_status": stage.status,
            "run_status": "ran",
            "message": "real spine executed",
            "artifacts_written": [stage.artifact],
            "log": str(ctx.artifacts_dir / "logs" / f"{stage.name}.log"),
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(pipeline_cli, "_run_existing_spine", fake_run_existing_spine)

    summary = pipeline_cli.run_top_level_pipeline(
        video=video,
        selected_stages=["court_calibration"],
        inputs_root=tmp_path / "inputs",
        artifacts_root=tmp_path / "artifacts",
        artifact_source=seed,
        force=True,
    )

    assert spine_calls["count"] == 1
    assert summary["stages"][0]["run_status"] == "ran"
    assert summary["stages"][0]["message"] == "real spine executed"
