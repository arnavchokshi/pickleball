#!/usr/bin/env python3
"""Build a versioned A-3 solver-observation log from solved ball artifacts.

A-5 tooling for the ball-3D program (``runs/ball3d_lifting_plan_20260723/
PLAN.md`` §5.A + v2 reframe): persist exactly what the production solver saw
per frame — pixel observation, world ray, candidate-set summary, anchor
events, fail-closed verdict — as a frozen dataset artifact so a front-end
change can never silently shift the input distribution between experiments.

Measurement-only, ``VERIFIED=0`` binding. Walks the same artifact layout the
characterization harness consumes (``ball_track_arc_solved.json`` +
``ball_chain_manifest.json`` + ``court_calibration.json``) and emits one
``<clip>.observation_log.json`` per clip in the A-3 contract format
(``threed/racketsport/ball_metric3d_contract.py``).

Fail-closed rules:

- World rays are computed ONLY when the calibration file's sha256 matches
  the solve's recorded ``court_calibration`` input (via
  ``ball_chain_manifest.json``); otherwise every frame carries
  ``ray_status: calibration_not_sha_verified`` and no ray — recomputing rays
  against different calibration bytes would fabricate geometry.
- Solver verdicts mirror the production fail-closed overlay: a frame is
  ``accepted`` only when it carries a world position, its band is not
  ``hidden``, and its owning segment passes
  ``ball_arc_segment_fail_closed_verdicts``; frames without per-frame
  segment provenance fail closed inside any untrusted segment span.

Output bytes are deterministic given the same inputs (sorted keys, rounded
floats, no timestamps; provenance is root-relative path + sha256).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_arc_solver import pixel_ray_world  # noqa: E402
from threed.racketsport.ball_metric3d_contract import (  # noqa: E402
    RAY_STATUS_COMPUTED,
    AnchorEvent,
    CandidateSetSummary,
    SolverFrameObservation,
    SolverObservationLog,
    SourceArtifact,
    WorldRay,
    write_solver_observation_log,
)
from threed.racketsport.ball_solver_characterization import (  # noqa: E402
    ARC_SOLVED_FILENAME,
    ClipInputs,
    calibration_sha_verified,
    discover_clip_inputs,
    sha256_file,
)
from threed.racketsport.virtual_world import (  # noqa: E402
    ball_arc_segment_fail_closed_verdicts,
)


def build_solver_observation_log(
    inputs: ClipInputs, *, root: str | Path
) -> SolverObservationLog:
    """Assemble the A-3 solver-observation log for one solved clip."""

    if inputs.arc_solved is None:
        raise FileNotFoundError(
            f"{inputs.clip}: {ARC_SOLVED_FILENAME} not found in {inputs.clip_dir}"
        )
    arc_solved = _read_json_object(inputs.arc_solved)
    verified = calibration_sha_verified(inputs)
    calibration: Mapping[str, Any] | None = None
    if inputs.calibration is not None and verified is True:
        calibration = _read_json_object(inputs.calibration)

    frames = _frames(arc_solved)
    verdicts = ball_arc_segment_fail_closed_verdicts(arc_solved.get("segments"))
    untrusted_spans = _untrusted_spans(verdicts)
    anchors_by_frame = _anchors_by_frame(arc_solved.get("anchors"))

    observations = tuple(
        _frame_observation(
            frame,
            index,
            calibration=calibration,
            calibration_available=inputs.calibration is not None,
            verdicts=verdicts,
            untrusted_spans=untrusted_spans,
            anchor_events=anchors_by_frame.get(index, ()),
        )
        for index, frame in enumerate(frames)
    )
    return SolverObservationLog(
        clip=inputs.clip,
        frames=observations,
        inputs=_source_artifacts(inputs, root=Path(root)),
        calibration_sha_verified=verified,
    )


def _frame_observation(
    frame: Mapping[str, Any],
    index: int,
    *,
    calibration: Mapping[str, Any] | None,
    calibration_available: bool,
    verdicts: Mapping[int, Mapping[str, Any]],
    untrusted_spans: Sequence[tuple[int, int]],
    anchor_events: Sequence[AnchorEvent],
) -> SolverFrameObservation:
    visible = frame.get("visible") is True
    xy = frame.get("xy")
    pixel_xy: tuple[float, float] | None = None
    if visible and _is_vec(xy, 2):
        pixel_xy = (float(xy[0]), float(xy[1]))
    observation_status = "observed" if pixel_xy is not None else "missing"

    ray: WorldRay | None = None
    if pixel_xy is None:
        ray_status = "no_pixel"
    elif calibration is not None:
        origin, direction = pixel_ray_world(calibration, pixel_xy)
        ray = WorldRay(origin_m=origin, direction=direction)
        ray_status = RAY_STATUS_COMPUTED
    elif calibration_available:
        ray_status = "calibration_not_sha_verified"
    else:
        ray_status = "missing_calibration"

    solver_info = frame.get("arc_solver")
    segment_id: int | None = None
    summary = CandidateSetSummary()
    if isinstance(solver_info, Mapping):
        raw_segment = solver_info.get("segment_id")
        if isinstance(raw_segment, int) and not isinstance(raw_segment, bool):
            segment_id = raw_segment
        summary = CandidateSetSummary(
            candidate_count=None,
            selected_residual_px=_float_or_none(solver_info.get("candidate_residual_px")),
            inlier_sighting=_bool_or_none(solver_info.get("inlier_sighting")),
            outlier_pruned=_bool_or_none(solver_info.get("outlier_sighting_pruned")),
            rescued=_bool_or_none(solver_info.get("rescued")),
        )

    band = frame.get("band")
    band_text = band if isinstance(band, str) and band else None
    verdict = _solver_verdict(
        frame,
        index,
        segment_id=segment_id,
        verdicts=verdicts,
        untrusted_spans=untrusted_spans,
    )
    timestamp = frame.get("t")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError(f"frame {index}: missing numeric timestamp 't'")
    return SolverFrameObservation(
        frame_index=index,
        timestamp_s=float(timestamp),
        observation_status=observation_status,
        pixel_xy=pixel_xy,
        pixel_confidence=_float_or_none(frame.get("conf")) if visible else None,
        ray=ray,
        ray_status=ray_status,
        candidate_summary=summary,
        anchor_events=tuple(anchor_events),
        solver_verdict=verdict,
        segment_id=segment_id,
        band=band_text,
    )


def _solver_verdict(
    frame: Mapping[str, Any],
    index: int,
    *,
    segment_id: int | None,
    verdicts: Mapping[int, Mapping[str, Any]],
    untrusted_spans: Sequence[tuple[int, int]],
) -> str:
    """Mirror of the characterization accepted-frame definition (fail closed)."""

    band = str(frame.get("band") or "")
    world = frame.get("world_xyz")
    has_world = _is_vec(world, 3)
    if band == "hidden" or not has_world:
        return "hidden"
    if segment_id is not None:
        verdict = verdicts.get(segment_id)
        untrusted = verdict is None or not bool(verdict.get("trusted"))
    else:
        # No per-frame provenance: fail closed inside any untrusted span.
        untrusted = any(start <= index <= end for start, end in untrusted_spans)
    return "rejected_fail_closed" if untrusted else "accepted"


def _untrusted_spans(verdicts: Mapping[int, Mapping[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for verdict in verdicts.values():
        if verdict.get("trusted"):
            continue
        start = verdict.get("frame_start")
        end = verdict.get("frame_end")
        if isinstance(start, int) and isinstance(end, int):
            spans.append((start, end))
    return spans


def _anchors_by_frame(raw_anchors: Any) -> dict[int, tuple[AnchorEvent, ...]]:
    if not isinstance(raw_anchors, list):
        return {}
    grouped: dict[int, list[AnchorEvent]] = {}
    for anchor in raw_anchors:
        if not isinstance(anchor, Mapping):
            continue
        frame = anchor.get("frame")
        anchor_id = anchor.get("anchor_id")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            continue
        if not isinstance(anchor_id, str) or not anchor_id:
            continue
        grouped.setdefault(frame, []).append(
            AnchorEvent(
                anchor_id=anchor_id,
                kind=str(anchor.get("kind") or "unknown"),
                status=str(anchor.get("status") or "unknown"),
                source=str(anchor.get("source") or "unknown"),
            )
        )
    return {
        frame: tuple(sorted(events, key=lambda event: event.anchor_id))
        for frame, events in grouped.items()
    }


def _source_artifacts(inputs: ClipInputs, *, root: Path) -> tuple[SourceArtifact, ...]:
    named = [
        ("ball_track_arc_solved", inputs.arc_solved),
        ("ball_chain_manifest", inputs.chain_manifest),
        ("court_calibration", inputs.calibration),
    ]
    artifacts = [
        SourceArtifact(kind=kind, path=_relative_posix(path, root=root), sha256=sha256_file(path))
        for kind, path in named
        if path is not None
    ]
    return tuple(sorted(artifacts, key=lambda artifact: artifact.kind))


def _relative_posix(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _frames(arc_solved: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = arc_solved.get("frames")
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _is_vec(value: Any, length: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _parse_named_paths(entries: list[str], *, flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"{flag} expects NAME=DIR (or NAME=PATH), got: {entry!r}")
        parsed[name] = Path(value)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit versioned A-3 solver-observation logs from solved ball artifact "
            "directories (measurement only, VERIFIED=0)."
        )
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        metavar="NAME=DIR",
        required=True,
        help="Clip name and artifact directory holding ball_track_arc_solved.json et al. Repeatable.",
    )
    parser.add_argument(
        "--calibration",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional per-clip court_calibration.json override. Rays are computed only when "
            "the file's sha256 matches the clip's ball_chain_manifest.json record."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory; writes <name>.observation_log.json per clip.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Root for relativizing recorded input paths (default: repo root).",
    )
    args = parser.parse_args(argv)

    try:
        clips = _parse_named_paths(args.clip, flag="--clip")
        calibrations = _parse_named_paths(args.calibration, flag="--calibration")
        results: dict[str, Any] = {}
        for name, directory in sorted(clips.items()):
            inputs = discover_clip_inputs(
                name, directory, calibration_override=calibrations.get(name)
            )
            log = build_solver_observation_log(inputs, root=args.root)
            out_path = args.out_dir / f"{name}.observation_log.json"
            write_solver_observation_log(out_path, log)
            results[name] = {
                "out": out_path.name,
                "frame_count": len(log.frames),
                "observed_frame_count": sum(
                    1 for frame in log.frames if frame.observation_status == "observed"
                ),
                "ray_computed_frame_count": sum(
                    1 for frame in log.frames if frame.ray_status == RAY_STATUS_COMPUTED
                ),
                "accepted_frame_count": sum(
                    1 for frame in log.frames if frame.solver_verdict == "accepted"
                ),
                "calibration_sha_verified": log.calibration_sha_verified,
                "output_sha256": sha256_file(out_path),
            }
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"out_dir": str(args.out_dir), "clips": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
