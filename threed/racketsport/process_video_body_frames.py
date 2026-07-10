"""process_video.py's "frames" stage: decide which BODY-runtime JPEGs to
extract from the source video, then materialize them.

Both the retired ``PoseStageRunner`` and
``BodyStageRunner`` (Fast SAM-3D-Body mesh, tier-scheduled at contact
windows) read per-frame crops from a ``body_frames/`` directory
(``threed.racketsport.orchestrator._find_body_frame_image``,
``body_frames/frame_NNNNNN.jpg``) -- but nothing in
``scripts/racketsport/process_video.py`` ever populated that directory
(Task #45's "additional_finding_not_fixed"). This module is that missing
stage: it decides *which* frame indices are actually needed, then reuses
``threed.racketsport.body_frame_materialization.materialize_body_frames``
(the same ffmpeg extraction ``threed.racketsport.body_video_smoke`` already
exercises end to end) to pull them out of the source video -- no new
extraction codepath, just a new *schedule*.

Schedule precedence
--------------------
``process_video.py``'s stage order runs "frames" right after "tracking" and
before "pose"/"events"/"body" (see ``scripts/racketsport/process_video.py``'s
``ProcessVideoPipeline.run``). That means, on a true cold start,
``frame_compute_plan.json`` (the tier-rule artifact -- MESH only inside
``deep_mesh_windows``, JOINTS everywhere else, written by the *later*
"events" stage) does not exist yet when frames must be extracted for pose to
even run. Lane A pose itself is "joints everywhere" by design -- it is never
tier-scheduled, contact windows or not. So the schedule this module builds
is:

1. **Bounded default (always applied):** every frame index that carries at
   least one tracked player in ``tracks.json`` (already-computed by the time
   this stage runs). This is *not* "every frame of the source video" -- it
   is bounded by what tracking actually found (pre-rally dead air before the
   first tracked frame is skipped, for example) -- but it is also not
   "only contact-window frames": Lane A's honest joints-everywhere coverage
   genuinely needs an image for every tracked frame, and extracting a
   sparser sample would silently starve ``PoseStageRunner`` of frames it
   requires (a missing frame image raises inside its per-frame loop, failing
   the whole pose stage) rather than gracefully degrading per frame.
2. **Tier rule, when available:** if ``frame_compute_plan.json`` already
   exists on this ``clip_dir`` (a resumed/reused run whose events stage
   already ran previously), its ``deep_mesh_windows`` frame indices are
   explicitly unioned in too -- respecting the locked tier-split product
   decision (NORTH_STAR_ROADMAP.md) rather than assuming they are always a subset
   of (1). In the common case they already are a subset; this only matters
   for correctness on a resumed run.
3. **Hard cap:** ``max_frames`` (process_video.py's existing ``--max-frames``
   smoke-run cap, when given) or ``DEFAULT_MAX_SCHEDULED_FRAMES`` otherwise,
   whichever is smaller, bounds the total frame count regardless of how long
   the tracked segment is. Capping is applied as a uniform stride across the
   full sorted index range (not "keep the first N"), so temporal coverage
   degrades evenly instead of the back half of the clip silently losing all
   pose/mesh coverage. This is a real trust-affecting event -- fewer scheduled
   frames means fewer skeleton/mesh outputs than tracking actually found --
   and is always surfaced in the returned notes, never silently applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .body_frame_materialization import materialize_body_frames
from .schemas import Tracks, validate_artifact_file

ARTIFACT_TYPE = "racketsport_process_video_frame_schedule"
SCHEMA_VERSION = 1
SCHEDULE_FILENAME = "process_video_frame_schedule.json"

# Safety ceiling for the bounded-default schedule. ~1200 JPEGs at ~85KB each
# (1080p, ffmpeg's default JPEG quality -- see
# threed/racketsport/body_frame_materialization.py) is ~100MB: generous for
# the short rally clips this pipeline targets (NORTH_STAR_ROADMAP.md), while still
# bounding a pathologically long input instead of extracting its every
# tracked frame unconditionally.
DEFAULT_MAX_SCHEDULED_FRAMES = 1200


def build_frame_schedule(
    tracks: Tracks,
    *,
    frame_compute_plan_path: str | Path | None = None,
    max_frames: int | None = None,
    skeleton_stride: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    """Return a ``scheduled_frames`` execution manifest + human-readable notes.

    The manifest shape matches what
    ``body_frame_materialization.materialize_body_frames`` expects for its
    ``execution_path`` argument (``{"scheduled_frames": [{"frame_idx": ...}, ...]}``).
    """

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when given")
    if skeleton_stride <= 0:
        raise ValueError("skeleton_stride must be positive")
    cap = int(max_frames) if max_frames is not None else DEFAULT_MAX_SCHEDULED_FRAMES

    notes: list[str] = []
    pose_indexes = _tracked_frame_indexes(tracks)
    base_indexes = _stride_sample(sorted(pose_indexes), int(skeleton_stride))
    notes.append(
        f"bounded base BODY skeleton schedule: kept {len(base_indexes)}/{len(pose_indexes)} tracked frame(s) "
        f"with skeleton_stride={int(skeleton_stride)} (the prior joints-everywhere base is now cadence-controlled; "
        "events/contact-dense mesh frames are unioned separately)"
    )

    plan_path = Path(frame_compute_plan_path) if frame_compute_plan_path is not None else None
    mesh_indexes: set[int] = set()
    if plan_path is not None and plan_path.is_file():
        mesh_indexes = _mesh_window_frame_indexes(plan_path, tracked=pose_indexes)
        if mesh_indexes:
            notes.append(
                f"tier rule respected: unioned {len(mesh_indexes)} deep_mesh_windows frame(s) from {plan_path} "
                "(already a subset of the tracked-frame set in the common case; explicit here for a resumed run "
                "whose events stage already wrote frame_compute_plan.json before this stage ran)"
            )

    union_indexes = sorted(base_indexes | mesh_indexes)
    capped = len(union_indexes) > cap
    final_indexes = _uniform_sample(union_indexes, cap) if capped else union_indexes
    if capped:
        notes.append(
            f"hard cap applied: kept {len(final_indexes)} of {len(union_indexes)} scheduled frame(s) "
            f"(uniform stride across the full clip, cap={cap}) -- pose/mesh coverage for the dropped frames will "
            "be honestly absent from skeleton3d.json/smpl_motion.json rather than fabricated; re-run with a "
            "higher --max-frames for full coverage"
        )

    scheduled_frames = [{"frame_idx": idx, "t": idx / tracks.fps} for idx in final_indexes]
    source = "tracks_union" if int(skeleton_stride) == 1 else "tracks_stride"
    if mesh_indexes:
        source += "+tier_rule"
    effective_stride = round(len(pose_indexes) / len(final_indexes), 3) if final_indexes else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fps": float(tracks.fps),
        "scheduled_frames": scheduled_frames,
        "frame_indexes": final_indexes,
        "capped": capped,
        "cap": cap,
        "source": source,
        "base_skeleton_stride": int(skeleton_stride),
        "total_tracked_frame_count": len(pose_indexes),
        "base_scheduled_frame_count": len(base_indexes),
        "event_extra_frame_count": len(set(final_indexes) - base_indexes),
        "effective_stride": effective_stride,
    }
    return manifest, notes


def materialize_process_video_frames(
    *,
    video_path: str | Path,
    tracks_path: str | Path,
    out_dir: str | Path,
    frame_compute_plan_path: str | Path | None = None,
    max_frames: int | None = None,
    skeleton_stride: int = 1,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Build the frame schedule for ``tracks_path`` and materialize it into
    ``out_dir`` as ``frame_NNNNNN.jpg`` files.

    Raises on any real failure (missing video, invalid/empty tracks.json,
    ffmpeg unavailable/failing) -- callers (``process_video.py``'s "frames"
    stage) are expected to catch that and degrade to a skeleton-only bundle
    rather than crash the whole pipeline, exactly like every other stage in
    ``scripts/racketsport/process_video.py``.
    """

    video = Path(video_path)
    tracks_file = Path(tracks_path)
    out = Path(out_dir)
    if not video.is_file():
        raise FileNotFoundError(f"missing source video: {video}")
    if not tracks_file.is_file():
        raise FileNotFoundError(f"missing tracks.json: {tracks_file}")

    tracks = validate_artifact_file("tracks", tracks_file)
    if not isinstance(tracks, Tracks):
        raise ValueError(f"{tracks_file} did not validate as a tracks.json artifact")
    if not any(player.frames for player in tracks.players):
        raise ValueError(f"{tracks_file} has no tracked player-frames; nothing to schedule")

    schedule, notes = build_frame_schedule(
        tracks,
        frame_compute_plan_path=frame_compute_plan_path,
        max_frames=max_frames,
        skeleton_stride=skeleton_stride,
    )
    schedule_path = out.parent / SCHEDULE_FILENAME
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    extraction = materialize_body_frames(
        video_path=video,
        execution_path=schedule_path,
        out_dir=out,
        overwrite=overwrite,
    )
    total_bytes = sum(frame.stat().st_size for frame in out.glob("frame_*.jpg") if frame.is_file())
    return {
        "schedule": schedule,
        "schedule_path": str(schedule_path),
        "notes": notes,
        "extraction": extraction,
        "out_dir": str(out),
        "frame_count": extraction["extracted_frame_count"],
        "total_bytes": total_bytes,
    }


def _tracked_frame_indexes(tracks: Tracks) -> set[int]:
    indexes: set[int] = set()
    for player in tracks.players:
        for frame in player.frames:
            indexes.add(int(round(float(frame.t) * tracks.fps)))
    return indexes


def _stride_sample(indexes: list[int], stride: int) -> set[int]:
    if not indexes:
        return set()
    if stride <= 1:
        return set(indexes)
    anchor = indexes[0]
    return {idx for idx in indexes if (idx - anchor) % stride == 0}


def _mesh_window_frame_indexes(plan_path: Path, *, tracked: set[int]) -> set[int]:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(plan, dict):
        return set()
    indexes: set[int] = set()
    for window in plan.get("deep_mesh_windows", []) or []:
        try:
            start = int(window["frame_start"])
            end = int(window["frame_end"])
        except (KeyError, TypeError, ValueError):
            continue
        for frame_idx in range(start, end + 1):
            if frame_idx in tracked:
                indexes.add(frame_idx)
    return indexes


def _uniform_sample(indexes: list[int], cap: int) -> list[int]:
    if cap >= len(indexes):
        return list(indexes)
    if cap <= 1:
        return indexes[:1]
    step = (len(indexes) - 1) / (cap - 1)
    picked = sorted({indexes[round(i * step)] for i in range(cap)})
    # Rounding can collide and yield fewer than `cap` unique indexes -- that
    # is fine (still <= cap, still spread evenly across the full range);
    # never pad back toward the frames the stride dropped.
    return picked
