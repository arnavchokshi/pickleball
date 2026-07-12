from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

from scripts.racketsport import process_video
from threed.racketsport.run_identity import RunIdentityStore, SourceIdentity, StageSpec, sha256_file


Builder = Callable[[Path], None]


def _source(path: Path, *, fps: float = 30.0, frame_count: int = 2) -> SourceIdentity:
    return SourceIdentity.from_path(
        path,
        timing={"fps": fps, "frame_count": frame_count, "duration_s": frame_count / fps},
    )


def _execute(
    store: RunIdentityStore,
    source: SourceIdentity,
    specs: list[StageSpec],
    builders: dict[str, Builder],
    *,
    force: bool = False,
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for spec in specs:
        decision = store.decision(spec, source, force=force)
        if decision.reusable:
            outcomes[spec.name] = "reused"
            continue
        with store.transaction(spec, source) as stage_dir:
            builders[spec.name](stage_dir)
        outcomes[spec.name] = "rebuilt"
    return outcomes


def _managed_path(store: RunIdentityStore, stage: str, relative: str) -> Path:
    manifest = store.current_manifest(stage)
    assert manifest is not None
    return Path(str(manifest["_generation_dir"])) / relative


def test_same_clip_id_with_different_video_bytes_rebuilds_every_stage_without_pixel_reuse(
    tmp_path: Path,
) -> None:
    video = tmp_path / "same_clip_id.mp4"
    video.write_bytes(b"first-video-pixels")
    store = RunIdentityStore(tmp_path / "run" / "same_clip_id")
    specs = [
        StageSpec("ingest", code={"version": 1}),
        StageSpec("tracking", dependencies=("ingest",), code={"version": 1}),
        StageSpec("world", dependencies=("tracking",), code={"version": 1}),
    ]

    current_source = _source(video)
    builders = {
        "ingest": lambda out: (out / "pixels.bin").write_bytes(video.read_bytes()),
        "tracking": lambda out: (out / "tracks.txt").write_text(current_source.sha256, encoding="utf-8"),
        "world": lambda out: (out / "world.txt").write_text(current_source.sha256, encoding="utf-8"),
    }
    first = _execute(store, current_source, specs, builders)
    first_ingest_generation = store.current_manifest("ingest")["_generation_dir"]  # type: ignore[index]

    video.write_bytes(b"second-video-pixels-are-different")
    current_source = _source(video)
    second = _execute(store, current_source, specs, builders)

    assert first == {"ingest": "rebuilt", "tracking": "rebuilt", "world": "rebuilt"}
    assert second == {"ingest": "rebuilt", "tracking": "rebuilt", "world": "rebuilt"}
    assert store.current_manifest("ingest")["_generation_dir"] != first_ingest_generation  # type: ignore[index]
    assert _managed_path(store, "ingest", "pixels.bin").read_bytes() == b"second-video-pixels-are-different"
    assert b"first-video-pixels" not in _managed_path(store, "ingest", "pixels.bin").read_bytes()


def test_config_change_rebuilds_exact_dependent_closure_and_reuses_independent_branch(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stable-video")
    source = _source(video)
    store = RunIdentityStore(tmp_path / "run")

    def specs(calibration_knob: int) -> list[StageSpec]:
        return [
            StageSpec("ingest", code={"version": 1}),
            StageSpec("calibration", dependencies=("ingest",), config={"knob": calibration_knob}),
            StageSpec("audio", dependencies=("ingest",), config={"window": 4}),
            StageSpec("tracking", dependencies=("calibration",), code={"version": 1}),
            StageSpec("world", dependencies=("tracking", "audio"), code={"version": 1}),
        ]

    builders = {
        name: (lambda stage: lambda out: (out / "artifact.txt").write_text(stage, encoding="utf-8"))(name)
        for name in ("ingest", "calibration", "audio", "tracking", "world")
    }
    assert set(_execute(store, source, specs(1), builders).values()) == {"rebuilt"}
    assert _execute(store, source, specs(2), builders) == {
        "ingest": "reused",
        "calibration": "rebuilt",
        "audio": "reused",
        "tracking": "rebuilt",
        "world": "rebuilt",
    }


@pytest.mark.parametrize("dependency_kind", ["code", "config", "model", "explicit_input"])
def test_code_model_config_and_explicit_input_changes_rebuild_only_dependent_closure(
    tmp_path: Path,
    dependency_kind: str,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stable-video")
    model = tmp_path / "checkpoint.bin"
    model.write_bytes(b"model-v1")
    explicit_input = tmp_path / "reviewed.json"
    explicit_input.write_bytes(b"input-v1")
    source = _source(video)
    store = RunIdentityStore(tmp_path / "run")

    def specs(version: int) -> list[StageSpec]:
        target_kwargs: dict[str, object] = {
            "code": {"version": version if dependency_kind == "code" else 1},
            "config": {"version": version if dependency_kind == "config" else 1},
            "models": {"checkpoint": model},
            "explicit_inputs": {"reviewed": explicit_input},
        }
        return [
            StageSpec("ingest", code={"version": 1}),
            StageSpec("independent", dependencies=("ingest",), code={"version": 1}),
            StageSpec("target", dependencies=("ingest",), **target_kwargs),  # type: ignore[arg-type]
            StageSpec("consumer", dependencies=("target",), code={"version": 1}),
        ]

    builders = {
        name: (lambda stage: lambda out: (out / "artifact.txt").write_text(stage, encoding="utf-8"))(name)
        for name in ("ingest", "independent", "target", "consumer")
    }
    _execute(store, source, specs(1), builders)
    if dependency_kind == "model":
        model.write_bytes(b"model-v2")
    elif dependency_kind == "explicit_input":
        explicit_input.write_bytes(b"input-v2")

    assert _execute(store, source, specs(2), builders) == {
        "ingest": "reused",
        "independent": "reused",
        "target": "rebuilt",
        "consumer": "rebuilt",
    }


def test_identical_dependencies_reuse_every_stage_and_force_rebuilds_every_stage(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stable-video")
    source = _source(video)
    store = RunIdentityStore(tmp_path / "run")
    specs = [
        StageSpec("ingest", code={"version": 1}),
        StageSpec("tracking", dependencies=("ingest",), code={"version": 1}),
        StageSpec("world", dependencies=("tracking",), code={"version": 1}),
    ]
    builders = {
        name: (lambda stage: lambda out: (out / "artifact.txt").write_text(stage, encoding="utf-8"))(name)
        for name in ("ingest", "tracking", "world")
    }

    assert set(_execute(store, source, specs, builders).values()) == {"rebuilt"}
    assert set(_execute(store, source, specs, builders).values()) == {"reused"}
    generations_before = {name: store.current_manifest(name)["_generation_dir"] for name in builders}  # type: ignore[index]
    assert set(_execute(store, source, specs, builders, force=True).values()) == {"rebuilt"}
    generations_after = {name: store.current_manifest(name)["_generation_dir"] for name in builders}  # type: ignore[index]
    assert generations_after != generations_before


def test_source_timing_identity_change_rebuilds_even_when_video_bytes_are_identical(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stable-video")
    store = RunIdentityStore(tmp_path / "run")
    spec = StageSpec("ingest", code={"version": 1})
    builder = {"ingest": lambda out: (out / "artifact.txt").write_text("ingest", encoding="utf-8")}

    _execute(store, _source(video, fps=30.0), [spec], builder)
    assert _execute(store, _source(video, fps=60.0), [spec], builder) == {"ingest": "rebuilt"}


def test_failed_transaction_keeps_partial_stage_dir_invisible_and_preserves_safe_current(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _source(video)
    store = RunIdentityStore(tmp_path / "run")
    spec = StageSpec("tracking", code={"version": 1})

    with store.transaction(spec, source) as stage_dir:
        (stage_dir / "complete.txt").write_text("safe", encoding="utf-8")
    safe_manifest = store.current_manifest("tracking")
    assert safe_manifest is not None
    safe_generation = safe_manifest["_generation_dir"]

    with pytest.raises(RuntimeError, match="simulated mid-stage failure"):
        with store.transaction(spec, source) as stage_dir:
            (stage_dir / "half-written.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("simulated mid-stage failure")

    current = store.current_manifest("tracking")
    assert current is not None
    assert current["_generation_dir"] == safe_generation
    assert not list(store.transactions_dir.glob("tracking-*"))
    assert store.decision(spec, source).reusable is True
    assert not (Path(str(safe_generation)) / "half-written.txt").exists()


def test_legacy_run_without_manifest_is_stale_then_rebuilds_without_adopting_legacy_artifact(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _source(video)
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "tracks.json").write_text('{"legacy": true}', encoding="utf-8")
    store = RunIdentityStore(run_dir)
    spec = StageSpec("tracking", code={"version": 1})

    decision = store.decision(spec, source)
    assert decision.reusable is False
    assert decision.reason == "unfingerprinted_stale"

    with store.transaction(spec, source) as stage_dir:
        (stage_dir / "tracks.json").write_text('{"rebuilt": true}', encoding="utf-8")
    assert store.decision(spec, source).reusable is True
    manifest = store.current_manifest("tracking")
    assert manifest is not None
    assert manifest["external_artifacts"] == []
    assert _managed_path(store, "tracking", "tracks.json").read_text(encoding="utf-8") == '{"rebuilt": true}'


def test_explicit_migration_attestation_is_hash_checked_and_separately_identified(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _source(video)
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    artifact = run_dir / "tracks.json"
    artifact.write_text('{"legacy": true}', encoding="utf-8")
    store = RunIdentityStore(run_dir)
    spec = StageSpec("tracking", code={"version": 1})

    manifest = store.attest_migration(
        spec,
        source,
        artifact=artifact,
        source_sha256=source.sha256,
        artifact_sha256=sha256_file(artifact),
        author="migration-operator@example.com",
        timestamp="2026-07-12T18:30:00Z",
    )

    attestation = manifest["metadata"]["migration_attestation"]
    assert attestation["provenance"] == "migration_attestation"
    assert attestation["source_sha256"] == source.sha256
    assert attestation["artifact_sha256"] == sha256_file(artifact)
    assert attestation["author"] == "migration-operator@example.com"
    assert attestation["timestamp"] == "2026-07-12T18:30:00Z"
    assert store.decision(spec, source).reusable is True

    artifact.write_text('{"silently_mutated": true}', encoding="utf-8")
    decision = store.decision(spec, source)
    assert decision.reusable is False
    assert decision.reason == "artifact_hash_mismatch"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_sha256", "0" * 64, "source hash"),
        ("artifact_sha256", "0" * 64, "artifact hash"),
        ("author", "", "author"),
        ("timestamp", "2026-07-12 18:30:00", "timestamp"),
    ],
)
def test_migration_attestation_rejects_untrusted_identity_fields(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _source(video)
    artifact = tmp_path / "tracks.json"
    artifact.write_text('{"legacy": true}', encoding="utf-8")
    store = RunIdentityStore(tmp_path / "run")
    kwargs = {
        "artifact": artifact,
        "source_sha256": source.sha256,
        "artifact_sha256": sha256_file(artifact),
        "author": "migration-operator@example.com",
        "timestamp": "2026-07-12T18:30:00Z",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=error):
        store.attest_migration(StageSpec("tracking"), source, **kwargs)


def test_migration_attestation_manifest_tampering_invalidates_current_pointer(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _source(video)
    artifact = tmp_path / "tracks.json"
    artifact.write_text('{"legacy": true}', encoding="utf-8")
    store = RunIdentityStore(tmp_path / "run")
    spec = StageSpec("tracking")
    manifest = store.attest_migration(
        spec,
        source,
        artifact=artifact,
        source_sha256=source.sha256,
        artifact_sha256=sha256_file(artifact),
        author="migration-operator@example.com",
        timestamp="2026-07-12T18:30:00Z",
    )
    manifest_path = Path(str(manifest["_generation_dir"])) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["migration_attestation"]["author"] = "tampered-author"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    assert store.current_manifest("tracking") is None
    decision = store.decision(spec, source)
    assert decision.reusable is False
    assert decision.reason == "unfingerprinted_stale"


def test_explicit_input_change_wins_over_existing_cached_generation(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    reviewed = tmp_path / "reviewed_tracks.json"
    reviewed.write_bytes(b"reviewed-v1")
    source = _source(video)
    store = RunIdentityStore(tmp_path / "run")

    def spec() -> StageSpec:
        return StageSpec("tracking", code={"version": 1}, explicit_inputs={"tracks": reviewed})

    builders = {"tracking": lambda out: shutil.copy2(reviewed, out / "tracks.json")}
    _execute(store, source, [spec()], builders)
    assert _managed_path(store, "tracking", "tracks.json").read_bytes() == b"reviewed-v1"

    reviewed.write_bytes(b"reviewed-v2")
    assert _execute(store, source, [spec()], builders) == {"tracking": "rebuilt"}
    assert _managed_path(store, "tracking", "tracks.json").read_bytes() == b"reviewed-v2"


def test_process_video_wrapper_replaces_same_clip_source_and_reuses_only_identical_content(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def make_video(path: Path, value: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
        for _ in range(2):
            writer.write(np.full((48, 64, 3), value, dtype="uint8"))
        writer.release()

    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    make_video(first_video, 0)
    make_video(second_video, 255)
    run_dir = tmp_path / "run"
    calls: list[str] = []

    def run_once(video: Path, *, force: bool = False) -> tuple[str, str]:
        options = process_video.PipelineOptions(
            video=video,
            clip="same_clip_id",
            run_dir=run_dir,
            court_corners=None,
            skip_ball=True,
            no_gpu=True,
            force=force,
            vite_allow_root=tmp_path,
        )
        pipeline = process_video.ProcessVideoPipeline(options)

        def probe_stage() -> process_video.StageOutcome:
            calls.append(video.name)
            (options.clip_dir / "probe.txt").write_text(video.name, encoding="utf-8")
            return process_video.StageOutcome(
                stage="probe",
                status="ran",
                wall_seconds=0.0,
                artifacts=["probe.txt"],
            )

        ingest = pipeline._run_stage_safely("ingest", pipeline._stage_ingest)
        probe = pipeline._run_stage_safely("probe", probe_stage)
        return ingest.status, probe.status

    assert run_once(first_video) == ("ran", "ran")
    first_target_identity = SourceIdentity.from_path(run_dir / "same_clip_id" / "source.mp4")
    assert run_once(second_video) == ("ran", "ran")
    second_target_identity = SourceIdentity.from_path(run_dir / "same_clip_id" / "source.mp4")
    assert second_target_identity.sha256 != first_target_identity.sha256
    assert second_target_identity.sha256 == SourceIdentity.from_path(second_video).sha256
    assert calls == ["first.mp4", "second.mp4"]

    assert run_once(second_video) == ("skipped", "skipped")
    assert calls == ["first.mp4", "second.mp4"]
    assert run_once(second_video, force=True) == ("ran", "ran")
    assert calls == ["first.mp4", "second.mp4", "second.mp4"]
