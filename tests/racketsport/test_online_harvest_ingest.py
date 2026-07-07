from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.online_harvest_ingest import (
    ActivityBin,
    HarvestClip,
    HarvestSource,
    assign_clip_roles,
    build_prelabel_shard_manifest,
    compare_hash_sets,
    dedupe_sources_from_manifest,
    segments_from_activity_bins,
    validate_cvat_review_task_package,
    write_cvat_review_task_package,
)


def _source(source_id: str, *, channel: str, duration_s: float = 120.0) -> HarvestSource:
    return HarvestSource(
        source_id=source_id,
        title=f"title {source_id}",
        channel=channel,
        url=f"https://example.test/{source_id}",
        video_path=Path(f"raw/{source_id}.mp4"),
        duration_s=duration_s,
        width=1920,
        height=1080,
        fps=30.0,
        bytes=1234,
        manifest_status="downloaded",
    )


def test_segments_from_activity_bins_fuses_audio_and_motion_with_motion_filter() -> None:
    bins = [
        ActivityBin(time_s=0.0, motion_score=0.00, audio_score=0.00),
        ActivityBin(time_s=1.0, motion_score=0.31, audio_score=0.05),
        ActivityBin(time_s=2.0, motion_score=0.34, audio_score=0.66),
        ActivityBin(time_s=3.0, motion_score=0.03, audio_score=0.70),
        ActivityBin(time_s=4.0, motion_score=0.00, audio_score=0.00),
        ActivityBin(time_s=9.0, motion_score=0.29, audio_score=0.10),
        ActivityBin(time_s=10.0, motion_score=0.32, audio_score=0.62),
        ActivityBin(time_s=11.0, motion_score=0.30, audio_score=0.05),
        ActivityBin(time_s=20.0, motion_score=0.02, audio_score=0.95),
    ]

    segments = segments_from_activity_bins(
        bins,
        duration_s=30.0,
        pad_s=0.5,
        max_active_gap_s=1.5,
        merge_gap_s=1.0,
        min_segment_s=1.0,
        min_motion_score=0.18,
        audio_motion_floor=0.10,
        min_motion_bins=1,
    )

    assert [(round(segment.start_s, 1), round(segment.end_s, 1)) for segment in segments] == [(0.5, 3.5), (8.5, 11.5)]
    assert set(segments[0].sources) == {"audio_onset_density", "motion_activity"}
    assert all("audio_onset_density" not in segment.sources or segment.motion_bin_count >= 1 for segment in segments)


def test_role_assignment_proposes_two_heldout_games_and_shards_exclude_them() -> None:
    sources = [
        _source("held_a", channel="Court A"),
        _source("train_a", channel="Court A"),
        _source("train_b", channel="Court B"),
        _source("held_b", channel="Court C"),
    ]
    clips = [
        HarvestClip(clip_id=f"{source.source_id}_r{idx:03d}", source=source, start_s=0.0, end_s=8.0, duration_s=8.0)
        for source in sources
        for idx in range(1, 4)
    ]

    roles = assign_clip_roles(
        clips,
        proposed_heldout_source_ids=("held_a", "held_b"),
        internal_val_modulo=3,
    )
    shard_manifest = build_prelabel_shard_manifest(clips, roles, shard_size=2)

    assert {proposal["source_id"] for proposal in roles.heldout_proposals} == {"held_a", "held_b"}
    assert all(roles.clip_roles[clip.clip_id] == "heldout_candidate_proposed" for clip in clips if clip.source.source_id in {"held_a", "held_b"})
    assert any(role == "internal_val" for clip_id, role in roles.clip_roles.items() if clip_id.startswith("train_"))
    assert all(
        item["role"] != "heldout_candidate_proposed"
        for shard in shard_manifest["shards"]
        for item in shard["items"]
    )
    assert shard_manifest["biometric_policy"]["persistent_reid_galleries_allowed"] is False
    assert shard_manifest["summary"]["excluded_heldout_candidate_clip_count"] == 6


def test_dedupe_manifest_flags_eval_and_cross_source_collisions() -> None:
    collisions = compare_hash_sets(
        left_name="harvest",
        left_hashes={"harvest_a": [0b0000, 0b1111], "harvest_b": [0b1010]},
        right_name="eval",
        right_hashes={"eval_clip": [0b0001]},
        threshold=1,
    )

    assert collisions == [
        {
            "left_group": "harvest_a",
            "left_hash": "0000000000000000",
            "right_group": "eval_clip",
            "right_hash": "0000000000000001",
            "hamming_distance": 1,
            "relation": "harvest_vs_eval",
        }
    ]
    assert dedupe_sources_from_manifest(collisions)["eval_collision_count"] == 1


def test_ingest_online_harvest_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/ingest_online_harvest.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--manifest" in completed.stdout
    assert "--harvest-root" in completed.stdout
    assert "--max-prelabel-smoke-frames" in completed.stdout
    assert "--skip-extract" in completed.stdout
    assert "--export-cvat-review-tasks" in completed.stdout
    assert "--cvat-out-root" in completed.stdout


def test_cvat_review_task_package_exports_visibility_levels_and_excludes_heldout(tmp_path: Path) -> None:
    review_subset = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_cvat_review_subset",
        "status": "selection_ready",
        "selected": [
            {
                "clip_id": "train_a_rally_0001",
                "source_id": "train_a",
                "role": "internal_val",
                "clip_path": "data/online_harvest_20260706/rallies/train_a/train_a_rally_0001.mp4",
                "start_s": 1.0,
                "end_s": 9.0,
                "frame_budget": 80,
            },
            {
                "clip_id": "train_b_rally_0002",
                "source_id": "train_b",
                "role": "train",
                "clip_path": "data/online_harvest_20260706/rallies/train_b/train_b_rally_0002.mp4",
                "start_s": 3.0,
                "end_s": 11.0,
                "frame_budget": 80,
            },
        ],
    }

    export = write_cvat_review_task_package(
        review_subset,
        out_root=tmp_path / "cvat_upload",
        heldout_source_ids=("held_a", "held_b"),
    )
    validation = validate_cvat_review_task_package(
        Path(export["manifest_path"]),
        heldout_source_ids=("held_a", "held_b"),
    )

    assert export["status"] == "ready_for_cvat_review"
    assert validation["status"] == "passed"
    assert validation["task_count"] == 2
    task_payload = json.loads(Path(export["tasks"][0]["task_definition_path"]).read_text(encoding="utf-8"))
    ball_label = next(label for label in task_payload["labels"] if label["name"] == "ball")
    assert ball_label["attribute_values"]["visibility_level"] == ["clear", "partial", "full", "out_of_frame"]
    assert {task["source_id"] for task in export["tasks"]} == {"train_a", "train_b"}


def test_cvat_review_task_package_rejects_heldout_selection(tmp_path: Path) -> None:
    review_subset = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_cvat_review_subset",
        "status": "selection_ready",
        "selected": [
            {
                "clip_id": "held_a_rally_0001",
                "source_id": "held_a",
                "role": "heldout_candidate_proposed",
                "clip_path": "data/online_harvest_20260706/rallies/held_a/held_a_rally_0001.mp4",
                "start_s": 1.0,
                "end_s": 9.0,
                "frame_budget": 80,
            },
        ],
    }

    with pytest.raises(ValueError, match="held-out"):
        write_cvat_review_task_package(
            review_subset,
            out_root=tmp_path / "cvat_upload",
            heldout_source_ids=("held_a", "held_b"),
        )
