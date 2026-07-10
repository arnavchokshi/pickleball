from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from threed.racketsport.body_compute import (
    body_frame_batches_from_execution,
    build_body_compute_execution,
)
from threed.racketsport.orchestrator import _find_body_frame_image
from threed.racketsport.process_video_body_frames import (
    DEFAULT_MAX_SCHEDULED_FRAMES,
    body_execution_frame_indexes,
    build_frame_schedule,
    materialize_process_video_frames,
    validate_materialized_frame_set,
)
from threed.racketsport.schemas import validate_artifact_file


ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = ROOT / "runs/lanes/demo_beststack_gpu_20260710/vm_pull/zwcth45s_r2"
VIDEO = ROOT / "runs/lanes/demo_beststack_gpu_20260710/zwcth45s_demo.mp4"
RUN_DIR = Path(__file__).resolve().parent / "post_fix_cold_zwcth45s_stride1"
WOLVERINE_RUN = (
    ROOT
    / "runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner"
)


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"cold repro directory already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    tracks_path = SOURCE_RUN / "tracks.json"
    plan_path = SOURCE_RUN / "frame_compute_plan.json"
    tracks = validate_artifact_file("tracks", tracks_path)
    body_execution = build_body_compute_execution(
        tracks,
        frame_plan_path=plan_path,
        max_frames=DEFAULT_MAX_SCHEDULED_FRAMES,
        include_tier2_body_joints=True,
        skeleton_stride=1,
    )
    required_frame_indexes = body_execution_frame_indexes(body_execution)
    schedule, _notes = build_frame_schedule(
        tracks,
        frame_compute_plan_path=plan_path,
        max_frames=DEFAULT_MAX_SCHEDULED_FRAMES,
        skeleton_stride=1,
        required_frame_indexes=required_frame_indexes,
    )
    materialized = materialize_process_video_frames(
        video_path=VIDEO,
        tracks_path=tracks_path,
        out_dir=RUN_DIR / "body_frames",
        frame_compute_plan_path=plan_path,
        max_frames=DEFAULT_MAX_SCHEDULED_FRAMES,
        skeleton_stride=1,
        required_frame_indexes=required_frame_indexes,
        schedule=schedule,
    )
    (RUN_DIR / "body_compute_execution.json").write_text(
        json.dumps(body_execution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized_indexes = {
        int(path.stem.removeprefix("frame_"))
        for path in (RUN_DIR / "body_frames").glob("frame_*.jpg")
    }
    requested_indexes = {
        int(frame_idx)
        for frame_idx, _requests in body_frame_batches_from_execution(tracks, body_execution)
    }
    missing = sorted(requested_indexes - materialized_indexes)
    print(f"clip=zwcth45s_demo.mp4 md5_expected=059e396317071e58478e75c55947fe6d")
    print(f"fresh_clip_id=post_fix_cold_zwcth45s_stride1")
    print(f"frames_schedule_count={len(materialized['schedule']['frame_indexes'])}")
    print(f"materialized_count={len(materialized_indexes)}")
    print(f"body_requested_count={len(requested_indexes)}")
    print(f"missing_count={len(missing)} missing_frames={missing}")
    print(f"zwcth_schedule_materialized_equal={set(schedule['frame_indexes']) == materialized_indexes}")
    print(f"zwcth_body_requested_subset_materialized={requested_indexes <= materialized_indexes}")
    context = SimpleNamespace(inputs_dir=RUN_DIR, run_dir=RUN_DIR, clip="post_fix_cold_zwcth45s_stride1")
    for frame_idx in sorted(requested_indexes):
        _find_body_frame_image(context, frame_idx)

    wolverine_tracks = validate_artifact_file("tracks", WOLVERINE_RUN / "tracks.json")
    wolverine_plan = WOLVERINE_RUN / "frame_compute_plan.json"
    wolverine_execution = build_body_compute_execution(
        wolverine_tracks,
        frame_plan_path=wolverine_plan,
        max_frames=DEFAULT_MAX_SCHEDULED_FRAMES,
        include_tier2_body_joints=True,
        skeleton_stride=2,
    )
    wolverine_required = body_execution_frame_indexes(wolverine_execution)
    wolverine_schedule, _wolverine_notes = build_frame_schedule(
        wolverine_tracks,
        frame_compute_plan_path=wolverine_plan,
        max_frames=DEFAULT_MAX_SCHEDULED_FRAMES,
        skeleton_stride=2,
        required_frame_indexes=wolverine_required,
    )
    wolverine_validation = validate_materialized_frame_set(
        out_dir=WOLVERINE_RUN / "body_frames",
        schedule=wolverine_schedule,
    )
    canonical_schedule_bytes = json.dumps(wolverine_schedule, indent=2, sort_keys=True) + "\n"
    original_schedule_bytes = (WOLVERINE_RUN / "process_video_frame_schedule.json").read_text(encoding="utf-8")
    print(f"wolverine_schedule_count={len(wolverine_schedule['frame_indexes'])}")
    print(f"wolverine_body_requested_count={len(wolverine_required)}")
    print(f"wolverine_schedule_materialized_equal={wolverine_validation['equal']}")
    print(f"wolverine_schedule_json_byte_identical={canonical_schedule_bytes == original_schedule_bytes}")


if __name__ == "__main__":
    main()
