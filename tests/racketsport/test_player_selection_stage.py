from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import process_video
from threed.racketsport.run_identity import SourceIdentity
from threed.racketsport.schemas import CourtCalibration


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibration_payload() -> dict[str, Any]:
    calibration = CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "intrinsics": {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 0.0,
                "cy": 0.0,
                "dist": [],
                "source": "test",
            },
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 5.0],
                "camera_height_m": 5.0,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_pts": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
    )
    return calibration.model_dump(mode="json")


def _empty_tracks() -> dict[str, Any]:
    return {"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": []}


def _raw_pool(*, frames: int, with_embeddings: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = {
        1: (-1.5, -3.0),
        2: (1.5, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    vectors = {
        1: [1.0, 0.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0, 0.0],
        3: [0.0, 0.0, 1.0, 0.0],
        4: [0.0, 0.0, 0.0, 1.0],
    }
    raw_frames: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    for frame_idx in range(frames):
        detections: list[dict[str, Any]] = []
        for detection_index, source in enumerate(sorted(positions)):
            x, y = positions[source]
            bbox = [x - 0.25, y - 1.0, x + 0.25, y]
            detections.append(
                {"bbox": bbox, "class": "person", "conf": 0.9, "track_id": source}
            )
            if with_embeddings:
                embedding_rows.append(
                    {
                        "frame": frame_idx,
                        "source_track_id": source,
                        "detection_index": detection_index,
                        "bbox": bbox,
                        "embedding": vectors[source],
                    }
                )
        raw_frames.append({"frame": frame_idx, "detections": detections})
    return (
        {"schema_version": 1, "fps": 30.0, "frames": raw_frames},
        {
            "schema_version": 1,
            "source_only": True,
            "uses_cvat_labels": False,
            "promote_trk": False,
            "feature_dim": 4,
            "l2_normalized": True,
            "detections": embedding_rows,
        },
    )


def _pipeline(
    tmp_path: Path,
    *,
    player_selection: bool,
    explicit: bool = False,
    clip: str = "fixture",
) -> process_video.ProcessVideoPipeline:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture-video")
    options = process_video.PipelineOptions(
        video=video,
        clip=clip,
        run_dir=tmp_path / "run",
        player_selection=player_selection,
        player_selection_explicit=explicit,
        reid_model=tmp_path / "missing-reid-checkpoint.pt",
    )
    return process_video.ProcessVideoPipeline(options)


def _write_stage_inputs(
    pipeline: process_video.ProcessVideoPipeline,
    *,
    frames: int,
    embeddings: bool,
) -> bytes:
    original_tracks = json.dumps(_empty_tracks(), separators=(",", ":")).encode("utf-8") + b"\n"
    (pipeline.clip_dir / "tracks.json").write_bytes(original_tracks)
    raw_pool, embedding_payload = _raw_pool(frames=frames, with_embeddings=embeddings)
    _write_json(pipeline.clip_dir / "tracked_detections.json", raw_pool)
    _write_json(pipeline.clip_dir / "court_calibration.json", _calibration_payload())
    if embeddings:
        _write_json(
            pipeline.clip_dir / "global_association/reid_embeddings.json",
            embedding_payload,
        )
    return original_tracks


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_player_selection_off_direct_call_is_typed_and_io_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path, player_selection=False)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("OFF player_selection touched an input or output helper")

    monkeypatch.setattr(pipeline, "_player_selection_raw_pool_path", forbidden)
    monkeypatch.setattr(process_video.player_selection_cli, "stage_copy", forbidden)
    monkeypatch.setattr(process_video.player_selection_cli, "stage_text", forbidden)
    monkeypatch.setattr(process_video.player_selection_cli, "publish_output_bundle", forbidden)

    outcome = pipeline._stage_player_selection()

    assert outcome.status == "skipped"
    assert outcome.metrics["reason_code"] == "player_selection_disabled"
    assert not (pipeline.clip_dir / "tracks_preselection.json").exists()
    assert not (pipeline.clip_dir / "selection_report.json").exists()


def test_absent_flag_and_explicit_off_emit_byte_identical_whole_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture-video")
    run_dir = tmp_path / "run"
    parser = process_video.build_arg_parser()

    def options(extra: list[str]) -> process_video.PipelineOptions:
        args = parser.parse_args(
            [
                "--video",
                str(video),
                "--clip",
                "fixture",
                "--out",
                str(run_dir),
                *extra,
            ]
        )
        return process_video.build_options_from_args(args)

    def fake_run_stage_safely(
        pipeline: process_video.ProcessVideoPipeline,
        name: str,
        _fn: object,
    ) -> process_video.StageOutcome:
        _write_json(
            pipeline.clip_dir / f"{name}.fixture.json",
            {"stage": name, "fixture": True},
        )
        return process_video.StageOutcome(stage=name, status="ran", wall_seconds=0.0)

    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_run_stage_safely",
        fake_run_stage_safely,
    )
    monkeypatch.setattr(process_video.time, "monotonic", lambda: 100.0)

    absent_options = options([])
    assert absent_options.player_selection is False
    assert absent_options.player_selection_explicit is False
    absent_pipeline = process_video.ProcessVideoPipeline(absent_options)
    absent_pipeline._source_identity_cache = SourceIdentity("a" * 64, 13, {"fps": 30.0})
    absent_pipeline.run()
    absent_tree = _tree_bytes(run_dir)

    explicit_options = options(["--no-player-selection"])
    assert explicit_options.player_selection is False
    assert explicit_options.player_selection_explicit is True
    explicit_pipeline = process_video.ProcessVideoPipeline(explicit_options)
    explicit_pipeline._source_identity_cache = SourceIdentity("a" * 64, 13, {"fps": 30.0})
    explicit_pipeline.run()
    explicit_tree = _tree_bytes(run_dir)

    assert absent_tree == explicit_tree
    assert "PIPELINE_SUMMARY.json" in absent_tree
    assert "fixture/PIPELINE_SUMMARY.json" in absent_tree
    assert not any("player_selection" in name for name in absent_tree)
    root_summary = json.loads(absent_tree["PIPELINE_SUMMARY.json"])
    assert all(stage["stage"] != "player_selection" for stage in root_summary["stages"])
    assert process_video.PLAYER_SELECTION_STACK_KEY not in root_summary["best_stack"]["resolved"]
    assert process_video.PLAYER_SELECTION_STACK_KEY not in root_summary["best_stack"]["overrides"]


@pytest.mark.parametrize("frames", [0, 30], ids=["empty-input", "exactly-four"])
def test_provider_absence_warns_even_without_ambiguous_frames(
    tmp_path: Path, frames: int
) -> None:
    pipeline = _pipeline(tmp_path, player_selection=True, clip=f"fixture-{frames}")
    original_tracks = _write_stage_inputs(pipeline, frames=frames, embeddings=False)

    outcome = pipeline._stage_player_selection()

    assert outcome.status == "ran"
    assert outcome.metrics["selective_reid"]["reid_invoked"] == 0
    assert outcome.metrics["selective_reid"]["reid_unavailable"] == 1
    assert outcome.metrics["warnings"] == [
        {
            "reason_code": "player_selection_reid_unavailable",
            "count": 1,
            "provider_reason": "embedding_provider_artifact_absent",
        }
    ]
    if frames == 30:
        assert outcome.metrics["selective_reid"]["reid_skipped_unambiguous"] == 30
    assert (pipeline.clip_dir / "tracks_preselection.json").read_bytes() == original_tracks


def test_stage_preserves_preselection_bytes_and_reports_full_provenance(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, player_selection=True)
    original_tracks = _write_stage_inputs(pipeline, frames=30, embeddings=True)
    pipeline.options.reid_model.write_bytes(b"fixture-checkpoint")

    outcome = pipeline._stage_player_selection()

    assert outcome.status == "ran"
    preselection = pipeline.clip_dir / "tracks_preselection.json"
    report_path = pipeline.clip_dir / "selection_report.json"
    assert preselection.read_bytes() == original_tracks
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = [
        row
        for row in report["decisions"]
        if row.get("action") == "selection_stage_provenance"
    ]
    assert len(provenance) == 1
    assert provenance[0]["input_hashes"]["tracks_preselection_sha256"] == hashlib.sha256(
        original_tracks
    ).hexdigest()
    assert provenance[0]["config_echo"]["best_stack_revision"] == 15
    assert provenance[0]["config_echo"]["best_stack_key"] == process_video.PLAYER_SELECTION_STACK_KEY
    assert provenance[0]["config_echo"]["reid_provider_available"] is True
    assert provenance[0]["selected_tracks_sha256"] == hashlib.sha256(
        (pipeline.clip_dir / "tracks.json").read_bytes()
    ).hexdigest()
    assert outcome.metrics["warnings"] == []


def test_stage_missing_required_input_fails_loudly_without_outputs(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, player_selection=True)
    (pipeline.clip_dir / "tracks.json").write_bytes(b"original-tracks\n")

    with pytest.raises(process_video._HardStageFailure, match="missing required input"):
        pipeline._stage_player_selection()

    assert (pipeline.clip_dir / "tracks.json").read_bytes() == b"original-tracks\n"
    assert not (pipeline.clip_dir / "tracks_preselection.json").exists()
    assert not (pipeline.clip_dir / "selection_report.json").exists()


def test_selector_failure_cannot_mutate_association_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path, player_selection=True)
    original_tracks = _write_stage_inputs(pipeline, frames=0, embeddings=False)

    def fail_selector(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected selector failure")

    monkeypatch.setattr(process_video, "select_players_payload", fail_selector)
    with pytest.raises(RuntimeError, match="injected selector failure"):
        pipeline._stage_player_selection()

    assert (pipeline.clip_dir / "tracks.json").read_bytes() == original_tracks
    assert not (pipeline.clip_dir / "tracks_preselection.json").exists()
    assert not (pipeline.clip_dir / "selection_report.json").exists()


def test_authoritative_graph_routes_consumers_through_optional_selection() -> None:
    without = process_video.authoritative_stage_names(
        rally_gating=False,
        verify_viewer=False,
        player_selection=False,
    )
    with_selection = process_video.authoritative_stage_names(
        rally_gating=False,
        verify_viewer=False,
        player_selection=True,
    )

    assert "player_selection" not in without
    assert with_selection.index("tracking") + 1 == with_selection.index("player_selection")
    assert with_selection.index("player_selection") + 1 == with_selection.index("camera_motion")
    assert process_video.RUN_IDENTITY_DEPENDENCIES["player_selection"] == (
        "calibration",
        "tracking",
    )
    for consumer in ("camera_motion", "placement", "rally_gating", "events", "frames", "body", "world"):
        assert "player_selection" in process_video.RUN_IDENTITY_DEPENDENCIES[consumer]
