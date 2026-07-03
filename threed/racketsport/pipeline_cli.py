"""Top-level pickleball pipeline CLI.

This module is intentionally an orchestration layer. It validates and moves
contract artifacts, and delegates to the existing fail-closed spine when a
stage has no precomputed contract artifact available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from . import orchestrator
from .schemas import validate_artifact_file


CapabilityStatus = Literal["NOT IMPLEMENTED", "RUNS", "VERIFIED"]
RunStatus = Literal["ran", "skipped", "failed", "not_implemented"]
Tier = Literal["on_device", "server_offline", "borderline"]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS_ROOT = ROOT / "inputs"
DEFAULT_ARTIFACTS_ROOT = ROOT / "artifacts"
DEFAULT_SAMPLE_ARTIFACT_ROOT = ROOT / "runs" / "eval0" / "prototype_gate_h100_v2"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


@dataclass(frozen=True)
class PipelineStage:
    name: str
    artifact: str
    schema: str
    tier: Tier
    tier_label: str
    latency_budget: str
    model: str
    output_timing: str
    status: CapabilityStatus
    runner: Literal["artifact", "artifact_or_spine", "not_implemented"]
    spine_stage: str | None = None
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "artifact": self.artifact,
            "schema": self.schema,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "latency_budget": self.latency_budget,
            "model": self.model,
            "output_timing": self.output_timing,
            "status": self.status,
            "runner": self.runner,
            "spine_stage": self.spine_stage,
        }


STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        name="capture_sidecar",
        aliases=("capture", "sidecar"),
        artifact="capture_sidecar.json",
        schema="capture_sidecar",
        tier="on_device",
        tier_label="ON-DEVICE LIVE (iPhone)",
        latency_budget="capture-time metadata; written with video package",
        model="AVFoundation capture + ARKit intrinsics/pose/floor-plane metadata",
        output_timing="live",
        status="RUNS",
        runner="artifact",
    ),
    PipelineStage(
        name="court_calibration",
        aliases=("calibration", "court"),
        artifact="court_calibration.json",
        schema="court_calibration",
        tier="on_device",
        tier_label="ON-DEVICE LIVE (iPhone/ANE-adjacent setup)",
        latency_budget="<10s setup/cache target",
        model="ARKit seed + existing OpenCV/manual-tap calibration runner",
        output_timing="live cached",
        status="RUNS",
        runner="artifact_or_spine",
        spine_stage="calibration",
    ),
    PipelineStage(
        name="tracks",
        aliases=("tracking", "person_tracking"),
        artifact="tracks.json",
        schema="tracks",
        tier="on_device",
        tier_label="ON-DEVICE LIVE (iPhone/ANE)",
        latency_budget="<33ms/frame live target",
        model="CoreML YOLO26n/s + lightweight ByteTrack/BoT-SORT + court filter; server spine can invoke YOLO26m BoT-SORT-ReID",
        output_timing="live, with optional async refinement",
        status="RUNS",
        runner="artifact_or_spine",
        spine_stage="tracking",
    ),
    PipelineStage(
        name="ball_track",
        aliases=("ball",),
        artifact="ball_track.json",
        schema="ball_track",
        tier="on_device",
        tier_label="ON-DEVICE LIVE (iPhone/ANE)",
        latency_budget="<33ms/frame target; ~288p heatmap risk area",
        model="distilled/quantized CoreML heatmap tracker for live path; TrackNetV3/WASB/PB-MAT are offline candidates",
        output_timing="live preview, async authority later",
        status="RUNS",
        runner="artifact_or_spine",
        spine_stage="ball_events",
    ),
    PipelineStage(
        name="contact_windows",
        aliases=("contacts", "contact", "ball_events"),
        artifact="contact_windows.json",
        schema="contact_windows",
        tier="on_device",
        tier_label="ON-DEVICE LIVE (iPhone)",
        latency_budget="<10s after rally span; frame-local cues",
        model="on-device mic onset + wrist velocity + ball inflection fusion",
        output_timing="live cue windows",
        status="RUNS",
        runner="artifact_or_spine",
        spine_stage="ball_events",
    ),
    PipelineStage(
        name="player_ground",
        aliases=("ground", "feet"),
        artifact="player_ground.json",
        schema="player_ground",
        tier="server_offline",
        tier_label="SERVER OFFLINE (GPU/CPU postprocess)",
        latency_budget="async",
        model="existing player_grounding/foot-lock primitives after pose/body output",
        output_timing="async",
        status="NOT IMPLEMENTED",
        runner="not_implemented",
    ),
    PipelineStage(
        name="racket_pose",
        aliases=("racket", "rkt"),
        artifact="racket_pose.json",
        schema="racket_pose",
        tier="server_offline",
        tier_label="SERVER OFFLINE (GPU)",
        latency_budget="async",
        model="explicit four-corner PnP-IPPE runner now; future SAM2/GigaPose/FoundPose/FoundationPose stack",
        output_timing="async",
        status="RUNS",
        runner="artifact_or_spine",
        spine_stage="racket",
    ),
    PipelineStage(
        name="metrics",
        aliases=("racket_sport_metrics",),
        artifact="racket_sport_metrics.json",
        schema="racket_sport_metrics",
        tier="server_offline",
        tier_label="SERVER OFFLINE (GPU/CPU)",
        latency_budget="async",
        model="biomechanical metrics and confidence calibration primitives",
        output_timing="async",
        status="NOT IMPLEMENTED",
        runner="not_implemented",
    ),
    PipelineStage(
        name="replay",
        aliases=("replay_scene",),
        artifact="replay_scene.json",
        schema="replay_scene",
        tier="server_offline",
        tier_label="SERVER OFFLINE (GPU/CPU export)",
        latency_budget="async",
        model="existing CPU review GLB export; production animated GLB/USDZ still gated",
        output_timing="async",
        status="RUNS",
        runner="artifact",
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
ALIASES = {alias: stage.name for stage in STAGES for alias in stage.aliases}
ORDER = [stage.name for stage in STAGES]


@dataclass(frozen=True)
class PipelineContext:
    video: Path
    clip: str
    inputs_dir: Path
    artifacts_dir: Path
    artifact_sources: tuple[Path, ...]
    force: bool
    tracking_mode: Literal["real", "precomputed", "precomputed_tracks"]
    max_frames: int | None
    manifest: Path
    tracker_config: Path
    ball_source: Path | None


def run_top_level_pipeline(
    *,
    video: str | Path,
    selected_stages: Sequence[str] | None = None,
    tier: Tier | Literal["all"] = "all",
    inputs_root: str | Path = DEFAULT_INPUTS_ROOT,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    artifact_source: str | Path | None = None,
    allow_fixture_fallback: bool = False,
    force: bool = False,
    tracking_mode: Literal["real", "precomputed", "precomputed_tracks"] = "real",
    max_frames: int | None = None,
    manifest: str | Path = orchestrator.DEFAULT_MODEL_MANIFEST,
    tracker_config: str | Path = orchestrator.DEFAULT_BOTSORT_REID_CONFIG,
    ball_source: str | Path | None = None,
) -> dict[str, Any]:
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video suffix for {video_path}; expected one of {sorted(VIDEO_SUFFIXES)}")

    clip = _clip_id_from_video(video_path)
    inputs_dir = Path(inputs_root) / clip
    artifacts_dir = Path(artifacts_root) / clip
    inputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy_video(video_path, inputs_dir)

    sources = _artifact_sources(
        clip=clip,
        explicit=Path(artifact_source) if artifact_source else None,
        inputs_dir=inputs_dir,
        allow_fixture_fallback=allow_fixture_fallback,
    )
    ctx = PipelineContext(
        video=video_path,
        clip=clip,
        inputs_dir=inputs_dir,
        artifacts_dir=artifacts_dir,
        artifact_sources=sources,
        force=force,
        tracking_mode=tracking_mode,
        max_frames=max_frames,
        manifest=Path(manifest),
        tracker_config=Path(tracker_config),
        ball_source=Path(ball_source) if ball_source is not None else None,
    )

    names = list(selected_stages) if selected_stages is not None else list(ORDER)
    if tier != "all":
        names = [name for name in names if STAGE_BY_NAME[name].tier == tier]

    started = time.time()
    stage_results: list[dict[str, Any]] = []
    for name in names:
        result = _run_stage(STAGE_BY_NAME[name], ctx)
        stage_results.append(result)
        if result["run_status"] in {"failed", "not_implemented"}:
            break

    status = _summary_status(stage_results)
    summary = {
        "schema_version": 1,
        "artifact_type": "pickleball_pipeline_run_summary",
        "status": status,
        "clip": clip,
        "video": str(video_path),
        "tier": tier,
        "inputs_dir": str(inputs_dir),
        "artifacts_dir": str(artifacts_dir),
        "artifact_sources": [str(path) for path in sources],
        "stages": stage_results,
        "artifacts_written": [
            artifact
            for stage in stage_results
            for artifact in stage.get("artifacts_written", [])
        ],
        "elapsed_s": round(time.time() - started, 3),
    }
    _write_json(artifacts_dir / "pipeline_summary.json", summary)
    return summary


def build_public_contract_readiness(
    run_dir: str | Path,
    *,
    stage: str = "replay",
    tier: Tier | Literal["all"] = "all",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    selected = select_stage_names(None, None, stage)
    if tier != "all":
        selected = [name for name in selected if STAGE_BY_NAME[name].tier == tier]

    stage_reports: list[dict[str, Any]] = []
    for name in selected:
        stage_def = STAGE_BY_NAME[name]
        path = run_path / stage_def.artifact
        missing = [] if path.is_file() else [stage_def.artifact]
        validation_errors: list[str] = []
        if path.is_file():
            try:
                validate_artifact_file(stage_def.schema, path)
            except Exception as exc:
                validation_errors.append(f"{stage_def.artifact}: {exc}")
        status = "ready" if not missing and not validation_errors else "not_ready"
        stage_reports.append(
            {
                "stage": stage_def.name,
                "artifact": stage_def.artifact,
                "schema": stage_def.schema,
                "tier": stage_def.tier,
                "model": stage_def.model,
                "output_timing": stage_def.output_timing,
                "status": status,
                "present_artifacts": [] if missing else [stage_def.artifact],
                "missing_artifacts": missing,
                "artifact_validation_errors": validation_errors,
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "pickleball_public_pipeline_contract_readiness",
        "run_dir": str(run_path),
        "requested_stage": normalize_stage_name(stage),
        "tier": tier,
        "status": "ready" if all(item["status"] == "ready" for item in stage_reports) else "not_ready",
        "stage_order": ORDER,
        "required_artifacts": [STAGE_BY_NAME[name].artifact for name in selected],
        "missing_artifacts": [
            artifact
            for item in stage_reports
            for artifact in item["missing_artifacts"]
        ],
        "artifact_validation_errors": [
            error
            for item in stage_reports
            for error in item["artifact_validation_errors"]
        ],
        "stages": stage_reports,
    }


def _run_stage(stage: PipelineStage, ctx: PipelineContext) -> dict[str, Any]:
    started = time.time()
    out_path = ctx.artifacts_dir / stage.artifact
    log_path = ctx.artifacts_dir / "logs" / f"{stage.name}.log"
    artifacts_written: list[str] = []

    if out_path.exists() and not ctx.force:
        try:
            validate_artifact_file(stage.schema, out_path)
        except Exception as exc:
            result = _stage_result(
                stage,
                run_status="failed",
                capability_status="RUNS",
                message=f"invalid existing artifact {out_path}: {exc}",
                elapsed_s=time.time() - started,
                log_path=log_path,
            )
            _write_stage_log(log_path, result)
            return result
        result = _stage_result(
            stage,
            run_status="skipped",
            capability_status=stage.status,
            message=f"valid output already exists: {out_path}",
            elapsed_s=time.time() - started,
            log_path=log_path,
        )
        _write_stage_log(log_path, result)
        return result

    if stage.runner == "not_implemented":
        source = _find_artifact_source(ctx, stage.artifact)
        if source is None:
            result = _stage_result(
                stage,
                run_status="not_implemented",
                capability_status="NOT IMPLEMENTED",
                message=f"{stage.name} has no registered local runner and no valid source artifact was provided",
                elapsed_s=time.time() - started,
                log_path=log_path,
            )
            _write_stage_log(log_path, result)
            return result
        return _copy_contract_artifact(stage, source, out_path, started, log_path, artifacts_written)

    if stage.runner == "artifact_or_spine" and stage.spine_stage is not None and ctx.force:
        # --force means "really rerun this stage": it must not silently
        # substitute a precomputed/fixture artifact-source copy for real
        # spine execution. Go straight to the spine and skip the
        # artifact-source copy path entirely.
        forced_spine_result = _run_existing_spine(stage, ctx)
        if forced_spine_result is not None:
            return forced_spine_result

    source = _find_artifact_source(ctx, stage.artifact)
    if source is not None:
        return _copy_contract_artifact(stage, source, out_path, started, log_path, artifacts_written)

    if stage.runner == "artifact_or_spine" and stage.spine_stage is not None:
        spine_result = _run_existing_spine(stage, ctx)
        if spine_result is not None:
            return spine_result

    result = _stage_result(
        stage,
        run_status="failed",
        capability_status=stage.status,
        message=f"missing required contract artifact {stage.artifact}; checked {', '.join(str(path) for path in ctx.artifact_sources)}",
        elapsed_s=time.time() - started,
        log_path=log_path,
    )
    _write_stage_log(log_path, result)
    return result


def _copy_contract_artifact(
    stage: PipelineStage,
    source: Path,
    out_path: Path,
    started: float,
    log_path: Path,
    artifacts_written: list[str],
) -> dict[str, Any]:
    try:
        validate_artifact_file(stage.schema, source)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != out_path.resolve():
            shutil.copy2(source, out_path)
        validate_artifact_file(stage.schema, out_path)
    except Exception as exc:
        result = _stage_result(
            stage,
            run_status="failed",
            capability_status=stage.status,
            message=f"{stage.artifact} failed contract validation from {source}: {exc}",
            elapsed_s=time.time() - started,
            log_path=log_path,
        )
        _write_stage_log(log_path, result)
        return result

    artifacts_written.append(stage.artifact)
    result = _stage_result(
        stage,
        run_status="ran",
        capability_status=stage.status,
        message=f"wrote {stage.artifact} from source artifact {source}",
        elapsed_s=time.time() - started,
        log_path=log_path,
        artifacts_written=artifacts_written,
    )
    _write_stage_log(log_path, result)
    return result


def _run_existing_spine(stage: PipelineStage, ctx: PipelineContext) -> dict[str, Any] | None:
    started = time.time()
    log_path = ctx.artifacts_dir / "logs" / f"{stage.name}.log"
    try:
        _seed_spine_inputs_from_sources(ctx)
        summary = orchestrator.run_pipeline(
            clip=ctx.clip,
            inputs_dir=ctx.inputs_dir,
            run_dir=ctx.artifacts_dir,
            stage=str(stage.spine_stage),
            tracking_mode=ctx.tracking_mode,
            tracking_video=ctx.video,
            max_frames=ctx.max_frames,
            manifest_path=ctx.manifest,
            tracker_config_path=ctx.tracker_config,
            ball_source_path=ctx.ball_source,
        )
        summary_status = summary.get("status")
        if summary_status != orchestrator.PIPELINE_STATUS_PASS:
            detail = _spine_failure_detail(summary)
            result = _stage_result(
                stage,
                run_status="failed",
                capability_status=stage.status,
                message=f"existing spine stage {stage.spine_stage} completed with status {summary_status}: {detail}",
                elapsed_s=time.time() - started,
                log_path=log_path,
            )
            _write_stage_log(log_path, result)
            return result
        validate_artifact_file(stage.schema, ctx.artifacts_dir / stage.artifact)
    except Exception as exc:
        result = _stage_result(
            stage,
            run_status="failed",
            capability_status=stage.status,
            message=f"existing spine stage {stage.spine_stage} failed: {exc}",
            elapsed_s=time.time() - started,
            log_path=log_path,
        )
        _write_stage_log(log_path, result)
        return result

    run_status = "ran"
    message = f"existing spine stage {stage.spine_stage} completed with status {summary.get('status')}"
    result = _stage_result(
        stage,
        run_status=run_status,  # type: ignore[arg-type]
        capability_status=stage.status,
        message=message,
        elapsed_s=time.time() - started,
        log_path=log_path,
        artifacts_written=[stage.artifact] if run_status == "ran" else [],
    )
    _write_stage_log(log_path, result)
    return result


def _seed_spine_inputs_from_sources(ctx: PipelineContext) -> None:
    for artifact in (
        "capture_sidecar.json",
        "court_keypoints.json",
        "detections.json",
        "tracks.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_candidates.json",
    ):
        target = ctx.inputs_dir / artifact
        if target.is_file():
            continue
        source = _find_artifact_source(ctx, artifact)
        if source is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _spine_failure_detail(summary: dict[str, Any]) -> str:
    stages = summary.get("stages")
    if not isinstance(stages, list) or not stages:
        return "no stage details available"
    last = stages[-1]
    if not isinstance(last, dict):
        return "last stage details are malformed"
    notes = last.get("notes")
    if isinstance(notes, list) and notes:
        return "; ".join(str(note) for note in notes)
    stage_name = last.get("stage", "unknown")
    status = last.get("status", "unknown")
    return f"last stage {stage_name} status {status}"


def _stage_result(
    stage: PipelineStage,
    *,
    run_status: RunStatus,
    capability_status: CapabilityStatus,
    message: str,
    elapsed_s: float,
    log_path: Path,
    artifacts_written: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "name": stage.name,
        "artifact": stage.artifact,
        "schema": stage.schema,
        "tier": stage.tier,
        "tier_label": stage.tier_label,
        "latency_budget": stage.latency_budget,
        "model": stage.model,
        "output_timing": stage.output_timing,
        "capability_status": capability_status,
        "run_status": run_status,
        "message": message,
        "artifacts_written": list(artifacts_written),
        "log": str(log_path),
        "elapsed_s": round(elapsed_s, 3),
    }


def _summary_status(stage_results: Sequence[dict[str, Any]]) -> str:
    if any(stage["run_status"] == "failed" for stage in stage_results):
        return "FAILED"
    if any(stage["run_status"] == "not_implemented" for stage in stage_results):
        return "NOT IMPLEMENTED"
    if stage_results and all(stage["capability_status"] == "VERIFIED" for stage in stage_results):
        return "VERIFIED"
    return "RUNS"


def _find_artifact_source(ctx: PipelineContext, artifact: str) -> Path | None:
    for root in ctx.artifact_sources:
        candidate = root / artifact
        if candidate.is_file():
            return candidate
    return None


def _artifact_sources(
    *,
    clip: str,
    explicit: Path | None,
    inputs_dir: Path,
    allow_fixture_fallback: bool = False,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    if allow_fixture_fallback:
        # DEFAULT_SAMPLE_ARTIFACT_ROOT is a historical prototype-gate run
        # directory (frozen June-28-era sample artifacts for a fixed set of
        # clip names), not a real per-run source. Substituting it silently
        # made demo/gate output claim "RUNS" without ever running a model, so
        # it is opt-in only -- see --allow-fixture-fallback in main().
        sample = DEFAULT_SAMPLE_ARTIFACT_ROOT / clip
        if sample.is_dir():
            candidates.append(sample)
    candidates.append(inputs_dir)
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return tuple(out)


def _clip_id_from_video(video: Path) -> str:
    if video.name.startswith("source.") and video.parent.name:
        return _slug(video.parent.name)
    return _slug(video.stem)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError(f"could not derive clip id from {value!r}")
    return slug


def _link_or_copy_video(video: Path, inputs_dir: Path) -> None:
    target = inputs_dir / f"source{video.suffix.lower()}"
    if target.exists():
        return
    try:
        os.symlink(video, target)
    except OSError:
        shutil.copy2(video, target)


def _write_json(path: Path, payload: MappingPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


MappingPayload = dict[str, Any]


def _write_stage_log(path: Path, result: MappingPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"stage={result['name']}",
        f"run_status={result['run_status']}",
        f"capability_status={result['capability_status']}",
        f"artifact={result['artifact']}",
        f"tier={result['tier']}",
        f"message={result['message']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_stage_name(value: str) -> str:
    token = value.strip()
    if token in STAGE_BY_NAME:
        return token
    if token in ALIASES:
        return ALIASES[token]
    valid = ", ".join(ORDER)
    raise ValueError(f"unknown stage {value!r}; expected one of: {valid}")


def select_stage_names(stage: str | None, from_stage: str | None, to_stage: str | None) -> list[str]:
    if stage and (from_stage or to_stage):
        raise ValueError("--stage cannot be combined with --from/--to")
    if stage:
        return [normalize_stage_name(stage)]
    if from_stage or to_stage:
        start_name = normalize_stage_name(from_stage or ORDER[0])
        end_name = normalize_stage_name(to_stage or ORDER[-1])
        start = ORDER.index(start_name)
        end = ORDER.index(end_name)
        if end < start:
            raise ValueError(f"--to {end_name} comes before --from {start_name}")
        return ORDER[start : end + 1]
    return list(ORDER)


def _print_human_summary(summary: MappingPayload) -> None:
    print("PIPELINE SUMMARY")
    print(f"status: {summary['status']}")
    print(f"clip: {summary['clip']}")
    print(f"tier: {summary['tier']}")
    print(f"inputs_dir: {summary['inputs_dir']}")
    print(f"artifacts_dir: {summary['artifacts_dir']}")
    print("stages:")
    for stage in summary["stages"]:
        print(
            f"- {stage['name']}: {stage['capability_status']} "
            f"({stage['run_status']}) -> {stage['artifact']} [{stage['tier']}]"
        )
        print(f"  {stage['message']}")


def _list_stages(*, tier: str, as_json: bool) -> int:
    stages = [stage for stage in STAGES if tier == "all" or stage.tier == tier]
    payload = {"schema_version": 1, "stages": [stage.as_dict() for stage in stages]}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for stage in stages:
            print(
                f"{stage.name}\t{stage.tier}\t{stage.artifact}\t{stage.status}\t{stage.model}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the organized pickleball pipeline from a video.")
    parser.add_argument("--video", type=Path, help="Raw source video path.")
    parser.add_argument("--stage", help="Run exactly one public pipeline stage.")
    parser.add_argument("--from", dest="from_stage", help="First public stage in a range.")
    parser.add_argument("--to", dest="to_stage", help="Last public stage in a range.")
    parser.add_argument("--list-stages", action="store_true", help="List public pipeline stages and tiers.")
    parser.add_argument("--tier", choices=("all", "on_device", "server_offline", "borderline"), default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redo stages even when valid outputs already exist. For stages with a real spine "
            "runner, this also skips any artifact-source copy (explicit --artifact-source or "
            "--allow-fixture-fallback) entirely and forces real spine execution."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--artifact-source", type=Path, help="Optional directory containing pre-existing contract artifacts.")
    parser.add_argument(
        "--allow-fixture-fallback",
        action="store_true",
        help=(
            "Allow substituting historical prototype-gate sample artifacts "
            f"({DEFAULT_SAMPLE_ARTIFACT_ROOT}) for missing contract inputs when no "
            "--artifact-source is provided. Off by default: real runs should not silently "
            "reuse stale fixtures and report RUNS/VERIFIED without running anything."
        ),
    )
    parser.add_argument("--tracking-mode", choices=("real", "precomputed", "precomputed_tracks"), default="real")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--manifest", type=Path, default=orchestrator.DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--tracker-config", type=Path, default=orchestrator.DEFAULT_BOTSORT_REID_CONFIG)
    parser.add_argument("--ball-source", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.list_stages:
            return _list_stages(tier=args.tier, as_json=args.json)
        if args.video is None:
            parser.error("--video is required unless --list-stages is used")
        selected = select_stage_names(args.stage, args.from_stage, args.to_stage)
        summary = run_top_level_pipeline(
            video=args.video,
            selected_stages=selected,
            tier=args.tier,
            inputs_root=args.inputs_root,
            artifacts_root=args.artifacts_root,
            artifact_source=args.artifact_source,
            allow_fixture_fallback=args.allow_fixture_fallback,
            force=args.force,
            tracking_mode=args.tracking_mode,
            max_frames=args.max_frames,
            manifest=args.manifest,
            tracker_config=args.tracker_config,
            ball_source=args.ball_source,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)
    return 0 if summary["status"] in {"RUNS", "VERIFIED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
