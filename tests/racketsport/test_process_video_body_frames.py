from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from threed.racketsport.process_video_body_frames import (
    BodyFrameMaterializationError,
    BodyFrameScheduleError,
    build_frame_schedule,
    materialize_process_video_frames,
    validate_materialized_frame_set,
)
from threed.racketsport.orchestrator import _find_body_frame_image
from threed.racketsport.schemas import Tracks


def _tracks(frame_ts: dict[int, list[float]], *, fps: float = 30.0) -> Tracks:
    """Build a Tracks object where ``frame_ts[player_id]`` is a list of ``t``
    (seconds) values that player was tracked at."""

    players = [
        {
            "id": player_id,
            "side": "near",
            "role": "left",
            "frames": [
                {"t": t, "bbox": (10.0, 10.0, 50.0, 50.0), "world_xy": (0.0, 0.0), "conf": 0.9}
                for t in ts
            ],
        }
        for player_id, ts in frame_ts.items()
    ]
    return Tracks.model_validate({"schema_version": 1, "fps": fps, "players": players, "rally_spans": []})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_tiny_clip(path: Path, *, rate: int = 10, duration_s: float = 1.0) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=96x64:rate={rate}:duration={duration_s}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg is not installed")


# ---------------------------------------------------------------------------
# build_frame_schedule: scheduling logic
# ---------------------------------------------------------------------------


def test_build_frame_schedule_default_is_tracked_frame_union() -> None:
    # player 1 tracked at frames 0,1,2 (t=0,1/30,2/30); player 2 overlaps at
    # frame 1 and extends to frame 3 -- the union should dedupe frame 1.
    tracks = _tracks({1: [0.0, 1 / 30, 2 / 30], 2: [1 / 30, 3 / 30]})

    schedule, notes = build_frame_schedule(tracks)

    assert schedule["frame_indexes"] == [0, 1, 2, 3]
    assert schedule["capped"] is False
    assert schedule["source"] == "tracks_union"
    assert [f["frame_idx"] for f in schedule["scheduled_frames"]] == [0, 1, 2, 3]
    assert any("joints-everywhere" in note for note in notes)


def test_build_frame_schedule_respects_tier_rule_when_plan_available(tmp_path: Path) -> None:
    # Tracked frames span 0..5, but only frame 10 (outside the tracked range)
    # is proposed by a stale/aggressive deep_mesh_window -- it must NOT be
    # pulled in since nothing is actually tracked there (no image would ever
    # be usable for a player crop). Frame 4 is inside both the window and the
    # tracked range, and must show up explicitly attributed to the tier rule.
    tracks = _tracks({1: [i / 30 for i in range(6)]})
    plan_path = tmp_path / "frame_compute_plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "deep_mesh_windows": [
                {"frame_start": 4, "frame_end": 4, "target_player_ids": [1]},
                {"frame_start": 10, "frame_end": 10, "target_player_ids": [1]},
            ],
        },
    )

    schedule, notes = build_frame_schedule(tracks, frame_compute_plan_path=plan_path)

    assert schedule["frame_indexes"] == [0, 1, 2, 3, 4, 5]
    assert 10 not in schedule["frame_indexes"]
    assert schedule["source"] == "tracks_union+tier_rule"
    assert any("tier rule respected" in note for note in notes)


def test_build_frame_schedule_applies_body_skeleton_stride_to_base_only(tmp_path: Path) -> None:
    tracks = _tracks({1: [i / 60 for i in range(8)]}, fps=60.0)
    plan_path = tmp_path / "frame_compute_plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "deep_mesh_windows": [
                {"frame_start": 3, "frame_end": 3, "target_player_ids": [1]},
            ],
        },
    )

    schedule, notes = build_frame_schedule(tracks, frame_compute_plan_path=plan_path, skeleton_stride=2)

    assert schedule["frame_indexes"] == [0, 2, 3, 4, 6]
    assert schedule["base_skeleton_stride"] == 2
    assert schedule["total_tracked_frame_count"] == 8
    assert schedule["base_scheduled_frame_count"] == 4
    assert schedule["event_extra_frame_count"] == 1
    assert schedule["effective_stride"] == 1.6
    assert any("skeleton_stride=2" in note for note in notes)
    assert any("tier rule respected" in note for note in notes)


def test_build_frame_schedule_ignores_missing_frame_compute_plan(tmp_path: Path) -> None:
    tracks = _tracks({1: [0.0, 1 / 30]})

    schedule, notes = build_frame_schedule(tracks, frame_compute_plan_path=tmp_path / "does_not_exist.json")

    assert schedule["source"] == "tracks_union"
    assert not any("tier rule" in note for note in notes)


def test_build_frame_schedule_applies_hard_cap_with_uniform_stride() -> None:
    tracks = _tracks({1: [i / 30 for i in range(100)]})

    schedule, notes = build_frame_schedule(tracks, max_frames=10)

    assert schedule["capped"] is True
    assert schedule["cap"] == 10
    assert len(schedule["frame_indexes"]) <= 10
    # stride sampling must span the whole range, not just the front.
    assert schedule["frame_indexes"][0] == 0
    assert schedule["frame_indexes"][-1] == 99
    assert any("hard cap applied" in note for note in notes)


def test_build_frame_schedule_uncapped_when_within_limit() -> None:
    tracks = _tracks({1: [0.0, 1 / 30, 2 / 30]})

    schedule, _notes = build_frame_schedule(tracks, max_frames=10)

    assert schedule["capped"] is False
    assert schedule["frame_indexes"] == [0, 1, 2]


def test_capped_frame_schedule_keeps_body_required_frame_for_input_assembly(tmp_path: Path) -> None:
    tracks = _tracks({1: [i / 30 for i in range(1315)]})
    plan_path = tmp_path / "frame_compute_plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "deep_mesh_windows": [
                {"frame_start": 74, "frame_end": 74, "target_player_ids": [1]},
            ],
        },
    )
    schedule, _notes = build_frame_schedule(tracks, frame_compute_plan_path=plan_path)
    body_frames = tmp_path / "body_frames"
    body_frames.mkdir()
    for frame_idx in schedule["frame_indexes"]:
        (body_frames / f"frame_{frame_idx:06d}.jpg").write_bytes(b"jpeg")

    context = SimpleNamespace(inputs_dir=tmp_path, run_dir=tmp_path, clip="cold_cap_regression")
    found = _find_body_frame_image(context, 74)

    assert found == body_frames / "frame_000074.jpg"


def test_capped_frame_schedule_contains_authoritative_body_execution_set() -> None:
    tracks = _tracks({1: [i / 30 for i in range(1315)]})
    required = {0, 74, 246, 1309}

    schedule, notes = build_frame_schedule(tracks, required_frame_indexes=required)

    assert len(schedule["frame_indexes"]) == 1200
    assert required <= set(schedule["frame_indexes"])
    assert any("authoritative BODY request" in note for note in notes)


def test_frame_schedule_fails_typed_when_body_required_set_exceeds_cap() -> None:
    tracks = _tracks({1: [i / 30 for i in range(5)]})

    with pytest.raises(BodyFrameScheduleError, match=r"exceeding materialization cap 2.*required frames=\[0, 1, 2\]"):
        build_frame_schedule(tracks, max_frames=2, required_frame_indexes={0, 1, 2})


def test_validate_materialized_frame_set_names_missing_and_stale_frames(tmp_path: Path) -> None:
    body_frames = tmp_path / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000002.jpg").write_bytes(b"jpeg")
    (body_frames / "frame_000009.jpg").write_bytes(b"stale")
    schedule = {"frame_indexes": [2, 5]}

    with pytest.raises(
        BodyFrameMaterializationError,
        match=r"missing_frames=\[5\]; unexpected_frames=\[9\]",
    ):
        validate_materialized_frame_set(out_dir=body_frames, schedule=schedule)


def test_build_frame_schedule_rejects_nonpositive_max_frames() -> None:
    tracks = _tracks({1: [0.0]})

    with pytest.raises(ValueError, match="max_frames must be positive"):
        build_frame_schedule(tracks, max_frames=0)


def test_build_frame_schedule_rejects_nonpositive_skeleton_stride() -> None:
    tracks = _tracks({1: [0.0]})

    with pytest.raises(ValueError, match="skeleton_stride must be positive"):
        build_frame_schedule(tracks, skeleton_stride=0)


# ---------------------------------------------------------------------------
# materialize_process_video_frames: end-to-end extraction + degradation
# ---------------------------------------------------------------------------


def test_materialize_process_video_frames_extracts_real_jpegs(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_tiny_clip(video, rate=10, duration_s=1.0)
    tracks_path = tmp_path / "tracks.json"
    _write_json(
        tracks_path,
        {
            "schema_version": 1,
            "fps": 10.0,
            "players": [
                {"id": 1, "side": "near", "role": "left", "frames": [{"t": 0.2, "bbox": [1, 1, 5, 5], "world_xy": [0, 0], "conf": 0.9}]},
                {"id": 2, "side": "far", "role": "right", "frames": [{"t": 0.5, "bbox": [1, 1, 5, 5], "world_xy": [0, 0], "conf": 0.9}]},
            ],
            "rally_spans": [],
        },
    )
    out_dir = tmp_path / "body_frames"

    result = materialize_process_video_frames(video_path=video, tracks_path=tracks_path, out_dir=out_dir)

    assert result["frame_count"] == 2
    assert (out_dir / "frame_000002.jpg").is_file()
    assert (out_dir / "frame_000005.jpg").is_file()
    assert result["total_bytes"] > 0
    assert result["schedule"]["source"] == "tracks_union"


def test_materialize_process_video_frames_replaces_stale_cache_with_exact_schedule(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_tiny_clip(video, rate=10, duration_s=1.0)
    tracks_path = tmp_path / "tracks.json"
    _write_json(
        tracks_path,
        {
            "schema_version": 1,
            "fps": 10.0,
            "players": [
                {
                    "id": 1,
                    "side": "near",
                    "role": "left",
                    "frames": [
                        {"t": 0.2, "bbox": [1, 1, 5, 5], "world_xy": [0, 0], "conf": 0.9},
                        {"t": 0.5, "bbox": [1, 1, 5, 5], "world_xy": [0, 0], "conf": 0.9},
                    ],
                }
            ],
            "rally_spans": [],
        },
    )
    out_dir = tmp_path / "body_frames"
    out_dir.mkdir()
    (out_dir / "frame_000002.jpg").write_bytes(b"stale-current")
    (out_dir / "frame_000003.jpg").write_bytes(b"stale-unexpected")

    result = materialize_process_video_frames(video_path=video, tracks_path=tracks_path, out_dir=out_dir)

    assert result["validation"]["equal"] is True
    assert result["validation"]["materialized_frame_indexes"] == [2, 5]
    assert not (out_dir / "frame_000003.jpg").exists()


def test_materialize_process_video_frames_missing_video_raises(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, {"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": []})

    with pytest.raises(FileNotFoundError, match="missing source video"):
        materialize_process_video_frames(video_path=tmp_path / "nope.mp4", tracks_path=tracks_path, out_dir=tmp_path / "body_frames")


def test_materialize_process_video_frames_missing_tracks_raises(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_tiny_clip(video)

    with pytest.raises(FileNotFoundError, match="missing tracks.json"):
        materialize_process_video_frames(video_path=video, tracks_path=tmp_path / "nope.json", out_dir=tmp_path / "body_frames")


def test_materialize_process_video_frames_empty_tracks_raises(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_tiny_clip(video)
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, {"schema_version": 1, "fps": 30.0, "players": [{"id": 1, "side": "near", "role": "left", "frames": []}], "rally_spans": []})

    with pytest.raises(ValueError, match="no tracked player-frames"):
        materialize_process_video_frames(video_path=video, tracks_path=tracks_path, out_dir=tmp_path / "body_frames")


def test_materialize_process_video_frames_degrades_when_video_unreadable(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"this is not a real video file")
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, {"schema_version": 1, "fps": 30.0, "players": [{"id": 1, "side": "near", "role": "left", "frames": [{"t": 0.0, "bbox": [1, 1, 5, 5], "world_xy": [0, 0], "conf": 0.9}]}], "rally_spans": []})

    with pytest.raises(RuntimeError, match="ffmpeg"):
        materialize_process_video_frames(video_path=video, tracks_path=tracks_path, out_dir=tmp_path / "body_frames")
