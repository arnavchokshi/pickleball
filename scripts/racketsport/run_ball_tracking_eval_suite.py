#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_benchmark import BallCandidate, write_ball_tracker_benchmark  # noqa: E402
from threed.racketsport.ball_court_filter import write_filtered_ball_track  # noqa: E402
from threed.racketsport.ball_model_fusion import write_fused_ball_track  # noqa: E402
from threed.racketsport.ball_overlay import render_ball_track_overlay  # noqa: E402
from threed.racketsport.ball_temporal_filter import write_temporal_filtered_ball_track  # noqa: E402
from threed.racketsport.schemas import BallTrack  # noqa: E402
from threed.racketsport.tracknet_adapter import run_tracknet_or_convert  # noqa: E402


DEFAULT_CLIPS = (
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
)
DEFAULT_OVERLAY_CANDIDATES = (
    "pbmat_v0_motion_composite",
    "fusion_temporal_vball100_localtraj",
    "fusion_temporal_vball100_stable_veto",
)
DEFAULT_SELECTED_CANDIDATE = "pbmat_v0_motion_composite"

TemporalWriter = Callable[..., dict[str, Any]]
FusionWriter = Callable[..., dict[str, Any]]
TrackNetRunner = Callable[..., dict[str, Any]]
CourtWriter = Callable[..., dict[str, Any]]
OverlayRenderer = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class EvalSuiteConfig:
    run_root: Path
    review_root: Path
    out_root: Path
    clips: list[str] = field(default_factory=lambda: list(DEFAULT_CLIPS))
    run_tracknet: bool = False
    tracknet_repo: Path | None = None
    tracknet_file: Path | None = None
    inpaintnet_file: Path | None = None
    batch_size: int = 16
    large_video: bool = True
    include_pbmat_v0: bool = True
    render_overlays: bool = False
    overlay_candidates: tuple[str, ...] = DEFAULT_OVERLAY_CANDIDATES
    selected_candidate: str | None = None
    selected_root: Path | None = None
    hit_radius_px: float = 36.0
    teleport_px_per_frame: float = 160.0
    max_jump_gap_frames: int = 3


def run_ball_tracking_eval_suite(
    config: EvalSuiteConfig,
    *,
    tracknet_runner: TrackNetRunner = run_tracknet_or_convert,
    court_writer: CourtWriter = write_filtered_ball_track,
    temporal_writer: TemporalWriter = write_temporal_filtered_ball_track,
    fusion_writer: FusionWriter = write_fused_ball_track,
    overlay_renderer: OverlayRenderer = render_ball_track_overlay,
) -> dict[str, Any]:
    start = time.perf_counter()
    config.out_root.mkdir(parents=True, exist_ok=True)
    candidates: list[BallCandidate] = []
    timings: dict[str, Any] = {"clips": {}}
    generated: dict[str, dict[str, str]] = {}

    for clip in config.clips:
        clip_start = time.perf_counter()
        clip_dir = config.run_root / clip
        base = clip_dir / "tracknet_smoke_0000_0010"
        out_base = config.out_root / clip / "tracknet_smoke_0000_0010"
        out_base.mkdir(parents=True, exist_ok=True)
        _require_file(config.review_root / clip / "ball_points.json", f"{clip} review labels")
        video = _require_file(base / "input_0000_0010.mp4", f"{clip} source video")

        clip_candidates: dict[str, Path] = {}
        _add_existing_candidate(clip_candidates, "tracknet_raw_existing", base / "ball_track_0000_0010.json")
        _add_existing_candidate(clip_candidates, "fusion_temporal_vball100", base / "ball_track_fusion_temporal_vball100.json")
        _add_existing_candidate(
            clip_candidates,
            "fusion_temporal_vball100_localtraj",
            base / "ball_track_fusion_temporal_vball100_localtraj.json",
        )

        stage_timings: dict[str, float] = {}
        if config.run_tracknet:
            _require_tracknet_config(config)
            raw_out = out_base / "ball_track_tracknet_pretrained.json"
            fps = _track_or_video_fps(base / "ball_track_0000_0010.json", video)
            stage_timings["tracknet_pretrained_seconds"] = _timed(
                lambda: tracknet_runner(
                    out=raw_out,
                    fps=fps,
                    metadata_out=out_base / "ball_track_tracknet_pretrained_run.json",
                    video=video,
                    tracknet_file=config.tracknet_file,
                    inpaintnet_file=config.inpaintnet_file,
                    tracknet_repo=config.tracknet_repo,
                    prediction_dir=out_base / "tracknet_predictions",
                    batch_size=config.batch_size,
                    large_video=config.large_video,
                )
            )
            clip_candidates["tracknet_pretrained_raw"] = raw_out

            court_out = out_base / "ball_track_tracknet_pretrained_court_120px.json"
            stage_timings["tracknet_court_filter_seconds"] = _timed(
                lambda: court_writer(
                    ball_track_path=raw_out,
                    calibration_path=clip_dir / "court_calibration.json",
                    out_path=court_out,
                    summary_path=out_base / "ball_track_tracknet_pretrained_court_120px_summary.json",
                    target_size=None,
                    margin_px=120.0,
                )
            )
            clip_candidates["tracknet_pretrained_court_120px"] = court_out

            temporal_out = out_base / "ball_track_tracknet_pretrained_temporal_path.json"
            stage_timings["tracknet_temporal_path_seconds"] = _timed(
                lambda: temporal_writer(
                    ball_track_path=court_out,
                    out_path=temporal_out,
                    summary_path=out_base / "ball_track_tracknet_pretrained_temporal_path_summary.json",
                    mode="path",
                    max_speed_px_per_second=7200.0,
                    base_jump_px=60.0,
                    max_link_gap_frames=10,
                    max_interpolate_gap_frames=3,
                    min_chain_visible_frames=3,
                )
            )
            clip_candidates["tracknet_pretrained_temporal_path"] = temporal_out

            fusion_out = out_base / "ball_track_tracknet_pretrained_fusion_vball100.json"
            stage_timings["tracknet_fusion_vball100_seconds"] = _timed(
                lambda: fusion_writer(
                    primary_ball_track_path=court_out,
                    stable_ball_track_path=temporal_out,
                    verifier_ball_track_paths=[
                        base / "vballnet_fast" / "ball_track.json",
                        base / "vballnet_v1" / "ball_track.json",
                    ],
                    outlier_distance_px=100.0,
                    out_path=fusion_out,
                    summary_path=out_base / "ball_track_tracknet_pretrained_fusion_vball100_summary.json",
                )
            )
            clip_candidates["tracknet_pretrained_fusion_vball100"] = fusion_out

            localtraj_out = out_base / "ball_track_tracknet_pretrained_fusion_vball100_localtraj.json"
            stage_timings["tracknet_fusion_localtraj_seconds"] = _timed(
                lambda: temporal_writer(
                    ball_track_path=fusion_out,
                    out_path=localtraj_out,
                    summary_path=out_base / "ball_track_tracknet_pretrained_fusion_vball100_localtraj_summary.json",
                    mode="local_trajectory",
                    local_trajectory_window_frames=20,
                    local_trajectory_max_error_px=80.0,
                    local_trajectory_min_pair_predictions=4,
                    max_iterations=3,
                )
            )
            clip_candidates["tracknet_pretrained_fusion_vball100_localtraj"] = localtraj_out

        _generate_localtraj_candidates(
            clip_candidates=clip_candidates,
            out_base=out_base,
            temporal_writer=temporal_writer,
            timings=stage_timings,
        )
        _generate_ballistic_candidates(
            clip_candidates=clip_candidates,
            out_base=out_base,
            temporal_writer=temporal_writer,
            timings=stage_timings,
        )
        _generate_stable_veto_candidate(
            clip_candidates=clip_candidates,
            base=base,
            out_base=out_base,
            fusion_writer=fusion_writer,
            timings=stage_timings,
        )
        if config.include_pbmat_v0:
            _generate_pbmat_v0_composite_candidate(
                clip_candidates=clip_candidates,
                out_base=out_base,
                timings=stage_timings,
            )
        overlay_paths = _render_requested_overlays(
            config=config,
            clip_candidates=clip_candidates,
            video=video,
            out_base=out_base,
            overlay_renderer=overlay_renderer,
            timings=stage_timings,
        )

        for name, path in sorted(clip_candidates.items()):
            candidates.append(BallCandidate(clip=clip, name=name, path=path, category=_candidate_category(name)))
        generated[clip] = {name: str(path) for name, path in sorted(clip_candidates.items())}
        if overlay_paths:
            generated[clip]["overlays"] = json.dumps(overlay_paths, sort_keys=True)
        stage_timings["total_clip_seconds"] = time.perf_counter() - clip_start
        timings["clips"][clip] = stage_timings

    benchmark = write_ball_tracker_benchmark(
        candidates=candidates,
        review_root=config.review_root,
        out_json=config.out_root / "benchmark.json",
        out_markdown=config.out_root / "benchmark.md",
        hit_radius_px=config.hit_radius_px,
        teleport_px_per_frame=config.teleport_px_per_frame,
        max_jump_gap_frames=config.max_jump_gap_frames,
    )
    selection = _write_selected_tracks(config=config, generated=generated, benchmark=benchmark)
    timings["total_seconds"] = time.perf_counter() - start
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracking_eval_suite",
        "status": "scored_not_gate_verified",
        "run_root": str(config.run_root),
        "review_root": str(config.review_root),
        "out_root": str(config.out_root),
        "clip_count": len(config.clips),
        "clips": config.clips,
        "run_tracknet": config.run_tracknet,
        "include_pbmat_v0": config.include_pbmat_v0,
        "generated_candidates": generated,
        "selection": selection,
        "timings": timings,
        "benchmark": benchmark,
        "not_ground_truth": True,
    }
    (config.out_root / "eval_suite_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _generate_ballistic_candidates(
    *,
    clip_candidates: dict[str, Path],
    out_base: Path,
    temporal_writer: TemporalWriter,
    timings: dict[str, float],
) -> None:
    for source_name in (
        "fusion_temporal_vball100",
        "fusion_temporal_vball100_localtraj",
        "tracknet_pretrained_fusion_vball100",
        "tracknet_pretrained_fusion_vball100_localtraj",
    ):
        source_path = clip_candidates.get(source_name)
        if source_path is None:
            continue
        candidate_name = f"{source_name}_ballistic"
        out_path = out_base / f"ball_track_{candidate_name}.json"
        summary_path = out_base / f"ball_track_{candidate_name}_summary.json"
        timings[f"{candidate_name}_seconds"] = _timed(
            lambda source_path=source_path, out_path=out_path, summary_path=summary_path: temporal_writer(
                ball_track_path=source_path,
                out_path=out_path,
                summary_path=summary_path,
                mode="ballistic",
                ballistic_window_frames=24,
                ballistic_max_residual_px=60.0,
                ballistic_min_fit_points=5,
                max_iterations=2,
            )
        )
        clip_candidates[candidate_name] = out_path


def _generate_localtraj_candidates(
    *,
    clip_candidates: dict[str, Path],
    out_base: Path,
    temporal_writer: TemporalWriter,
    timings: dict[str, float],
) -> None:
    for source_name in (
        "fusion_temporal_vball100",
        "tracknet_pretrained_fusion_vball100",
    ):
        candidate_name = f"{source_name}_localtraj"
        if candidate_name in clip_candidates:
            continue
        source_path = clip_candidates.get(source_name)
        if source_path is None:
            continue
        out_path = out_base / f"ball_track_{candidate_name}.json"
        summary_path = out_base / f"ball_track_{candidate_name}_summary.json"
        timings[f"{candidate_name}_seconds"] = _timed(
            lambda source_path=source_path, out_path=out_path, summary_path=summary_path: temporal_writer(
                ball_track_path=source_path,
                out_path=out_path,
                summary_path=summary_path,
                mode="local_trajectory",
                local_trajectory_window_frames=20,
                local_trajectory_max_error_px=80.0,
                local_trajectory_min_pair_predictions=4,
                max_iterations=3,
            )
        )
        clip_candidates[candidate_name] = out_path


def _generate_pbmat_v0_composite_candidate(
    *,
    clip_candidates: dict[str, Path],
    out_base: Path,
    timings: dict[str, float],
) -> None:
    source_name = None
    for candidate in (
        "fusion_temporal_vball100_localtraj",
        "fusion_temporal_vball100_localtraj_ballistic",
        "tracknet_pretrained_fusion_vball100_localtraj",
        "tracknet_pretrained_fusion_vball100_localtraj_ballistic",
    ):
        if candidate in clip_candidates:
            source_name = candidate
            break
    if source_name is None:
        return

    source_path = clip_candidates[source_name]
    candidate_name = "pbmat_v0_motion_composite"
    out_path = out_base / f"ball_track_{candidate_name}.json"
    metadata_path = out_base / f"ball_track_{candidate_name}_summary.json"

    def write_candidate() -> None:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        BallTrack.model_validate(payload)
        _write_json(out_path, payload)
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "artifact_type": "racketsport_pbmat_v0_motion_composite_run",
                "status": "composite_not_trained_pbmat_checkpoint",
                "candidate_category": _candidate_category(candidate_name),
                "source_candidate": source_name,
                "source_ball_track": str(source_path),
                "ball_track_source": payload.get("source"),
                "out": str(out_path),
                "components": [
                    "pretrained_or_existing_tracknet_candidate",
                    "vballnet_verifier_when_available",
                    "local_trajectory_or_ballistic_temporal_filter",
                ],
                "trained_pbmat_checkpoint": False,
                "not_ground_truth": True,
                "verified": False,
            },
        )

    timings[f"{candidate_name}_seconds"] = _timed(write_candidate)
    clip_candidates[candidate_name] = out_path


def _generate_stable_veto_candidate(
    *,
    clip_candidates: dict[str, Path],
    base: Path,
    out_base: Path,
    fusion_writer: FusionWriter,
    timings: dict[str, float],
) -> None:
    required = [
        base / "ball_track_target_court_120px.json",
        base / "ball_track_target_court_temporal.json",
        base / "vballnet_fast" / "ball_track.json",
        base / "vballnet_v1" / "ball_track.json",
    ]
    if not all(path.is_file() for path in required):
        return
    candidate_name = "fusion_temporal_vball100_stable_veto"
    out_path = out_base / f"ball_track_{candidate_name}.json"
    timings[f"{candidate_name}_seconds"] = _timed(
        lambda: fusion_writer(
            primary_ball_track_path=required[0],
            stable_ball_track_path=required[1],
            verifier_ball_track_paths=required[2:],
            outlier_distance_px=100.0,
            require_stable_verifier_support=True,
            out_path=out_path,
            summary_path=out_base / f"ball_track_{candidate_name}_summary.json",
        )
    )
    clip_candidates[candidate_name] = out_path


def _render_requested_overlays(
    *,
    config: EvalSuiteConfig,
    clip_candidates: dict[str, Path],
    video: Path,
    out_base: Path,
    overlay_renderer: OverlayRenderer,
    timings: dict[str, float],
) -> dict[str, str]:
    if not config.render_overlays:
        return {}
    overlays: dict[str, str] = {}
    for candidate_name in config.overlay_candidates:
        candidate_path = clip_candidates.get(candidate_name)
        if candidate_path is None:
            continue
        out_path = out_base / f"{candidate_name}_overlay.mp4"
        timings[f"{candidate_name}_overlay_seconds"] = _timed(
            lambda candidate_path=candidate_path, out_path=out_path: overlay_renderer(
                video_path=video,
                ball_track_path=candidate_path,
                out_path=out_path,
            )
        )
        overlays[candidate_name] = str(out_path)
    return overlays


def _write_selected_tracks(
    *,
    config: EvalSuiteConfig,
    generated: dict[str, dict[str, str]],
    benchmark: dict[str, Any],
) -> dict[str, Any] | None:
    if config.selected_candidate is None:
        return None
    ranking = _candidate_ranking(benchmark=benchmark, required_clip_count=len(config.clips))
    selected_rank = ranking.get(config.selected_candidate, {})
    selected_root = config.selected_root or config.out_root / "selected_tracks"
    selected: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracking_selection",
        "status": "selected_not_gate_verified",
        "candidate": config.selected_candidate,
        "candidate_category": _candidate_category(config.selected_candidate),
        "candidate_rank": selected_rank.get("rank"),
        "candidate_score": selected_rank.get("score"),
        "candidate_clip_count": selected_rank.get("clip_count"),
        "required_clip_count": len(config.clips),
        "eligible_for_model_ranking": bool(selected_rank.get("eligible_for_model_ranking", False)),
        "better_eligible_candidates": selected_rank.get("better_eligible_candidates", []),
        "selected_root": str(selected_root),
        "clips": {},
        "not_ground_truth": True,
    }
    for clip, paths in sorted(generated.items()):
        source = paths.get(config.selected_candidate)
        clip_selection: dict[str, Any] = {
            "candidate": config.selected_candidate,
            "candidate_category": _candidate_category(config.selected_candidate),
            "candidate_score": selected_rank.get("score"),
            "candidate_rank": selected_rank.get("rank"),
        }
        if source is None:
            clip_selection["status"] = "missing"
            selected["clips"][clip] = clip_selection
            continue
        source_path = Path(source)
        _require_file(source_path, f"{clip} selected candidate")
        out_dir = selected_root / clip
        out_path = out_dir / "ball_track.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, out_path)
        _write_json(
            out_dir / "ball_track_selection.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_track_selection",
                "status": "selected_not_gate_verified",
                "clip": clip,
                "candidate": config.selected_candidate,
                "candidate_category": _candidate_category(config.selected_candidate),
                "candidate_score": selected_rank.get("score"),
                "candidate_rank": selected_rank.get("rank"),
                "eligible_for_model_ranking": bool(selected_rank.get("eligible_for_model_ranking", False)),
                "better_eligible_candidates": selected_rank.get("better_eligible_candidates", []),
                "source_ball_track": str(source_path),
                "out": str(out_path),
                "trained_pbmat_checkpoint": _trained_pbmat_checkpoint_for_candidate(config.selected_candidate),
                "not_ground_truth": True,
            },
        )
        clip_selection.update({"status": "selected", "source": str(source_path), "out": str(out_path)})
        selected["clips"][clip] = clip_selection
    _write_json(selected_root / "selection_summary.json", selected)
    return selected


def _candidate_category(name: str) -> str:
    if name == "pbmat_v0_motion_composite":
        return "composite_alias_not_trained_model"
    return "generalizable"


def _trained_pbmat_checkpoint_for_candidate(name: str) -> bool | None:
    if name == "pbmat_v0_motion_composite":
        return False
    return None


def _candidate_ranking(*, benchmark: dict[str, Any], required_clip_count: int) -> dict[str, dict[str, Any]]:
    aggregate = benchmark.get("aggregate", {})
    rows = []
    for name, row in aggregate.items():
        score = row.get("mean_quality_score")
        if score is None:
            continue
        category = str(row.get("category", _candidate_category(name)))
        complete = int(row.get("clip_count", 0)) == required_clip_count
        eligible = complete and category == "generalizable"
        rows.append(
            {
                "name": name,
                "score": float(score),
                "clip_count": int(row.get("clip_count", 0)),
                "category": category,
                "eligible_for_model_ranking": eligible,
            }
        )
    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(ranked, start=1):
        better = [
            {
                "candidate": other["name"],
                "score": other["score"],
                "category": other["category"],
            }
            for other in ranked[: index - 1]
            if other["eligible_for_model_ranking"]
        ]
        result[row["name"]] = {
            "rank": index,
            "score": row["score"],
            "clip_count": row["clip_count"],
            "category": row["category"],
            "eligible_for_model_ranking": row["eligible_for_model_ranking"],
            "better_eligible_candidates": better,
        }
    return result


def _add_existing_candidate(clip_candidates: dict[str, Path], name: str, path: Path) -> None:
    if path.is_file():
        clip_candidates[name] = path


def _require_tracknet_config(config: EvalSuiteConfig) -> None:
    if config.tracknet_repo is None:
        raise ValueError("--run-tracknet requires --tracknet-repo")
    if not config.tracknet_repo.is_dir():
        raise FileNotFoundError(f"missing tracknet_repo directory: {config.tracknet_repo}")
    _require_file(config.tracknet_repo / "predict.py", "tracknet_repo predict.py")
    for label, value in {
        "tracknet_file": config.tracknet_file,
        "inpaintnet_file": config.inpaintnet_file,
    }.items():
        if value is None:
            raise ValueError(f"--run-tracknet requires --{label.replace('_', '-')}")
        _require_file(value, label)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _track_fps(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fps = float(payload.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"cannot determine fps from {path}")
    return fps


def _track_or_video_fps(track_path: Path, video_path: Path) -> float:
    if track_path.is_file():
        return _track_fps(track_path)
    return _video_fps(video_path)


def _video_fps(video_path: Path) -> float:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"cannot read video FPS without cv2 and no TrackNet source track exists: {video_path}") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"cannot determine fps from video metadata: {video_path}")
    return fps


def _timed(action: Callable[[], Any]) -> float:
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_clips(run_root: Path, review_root: Path) -> list[str]:
    clips = [clip for clip in DEFAULT_CLIPS if (review_root / clip / "ball_points.json").is_file()]
    if clips:
        return clips
    return sorted(path.name for path in review_root.iterdir() if (path / "ball_points.json").is_file() and (run_root / path.name).is_dir())


def _parse_args() -> EvalSuiteConfig:
    parser = argparse.ArgumentParser(description="Run and benchmark ball-tracking candidates on held-out clips.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/eval0/prototype_gate_h100_v2"))
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--run-tracknet", action="store_true")
    parser.add_argument("--tracknet-repo", type=Path, default=None)
    parser.add_argument("--tracknet-file", type=Path, default=None)
    parser.add_argument("--inpaintnet-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-large-video", action="store_true")
    parser.add_argument("--no-pbmat-v0", action="store_true", help="Do not write the PB-MAT v0 motion-composite candidate.")
    parser.add_argument("--render-overlays", action="store_true")
    parser.add_argument("--overlay-candidate", action="append", default=[])
    parser.add_argument("--select-candidate", default=None, help=f"Optional candidate to copy into selected_tracks. Example: {DEFAULT_SELECTED_CANDIDATE}.")
    parser.add_argument("--selected-root", type=Path, default=None, help="Optional selected-track output root.")
    parser.add_argument("--hit-radius-px", type=float, default=36.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    args = parser.parse_args()
    review_root = args.review_root or args.run_root / "ball_click_review_30"
    clips = args.clip or _default_clips(args.run_root, review_root)
    return EvalSuiteConfig(
        run_root=args.run_root,
        review_root=review_root,
        out_root=args.out_root,
        clips=clips,
        run_tracknet=args.run_tracknet,
        tracknet_repo=args.tracknet_repo,
        tracknet_file=args.tracknet_file,
        inpaintnet_file=args.inpaintnet_file,
        batch_size=args.batch_size,
        large_video=not args.no_large_video,
        include_pbmat_v0=not args.no_pbmat_v0,
        render_overlays=args.render_overlays,
        overlay_candidates=tuple(args.overlay_candidate) if args.overlay_candidate else DEFAULT_OVERLAY_CANDIDATES,
        selected_candidate=args.select_candidate,
        selected_root=args.selected_root,
        hit_radius_px=args.hit_radius_px,
        teleport_px_per_frame=args.teleport_px_per_frame,
        max_jump_gap_frames=args.max_jump_gap_frames,
    )


def main() -> int:
    try:
        summary = run_ball_tracking_eval_suite(_parse_args())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"out_root": summary["out_root"], "timings": summary["timings"], "aggregate": summary["benchmark"]["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
