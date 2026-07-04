"""Human review packet assembly for prototype and pipeline runs."""

from __future__ import annotations

import json
import os
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import ContactWindows, RacketCandidates


UTC = timezone.utc
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_human_review_packet"
GLOBAL_REVIEW_CLIP = "__global__"
REVIEW_INDEX_NAMES = (
    "label_overlay_index.json",
    "calibration_overlay_index.json",
    "player_track_overlay_index.json",
    "racket_candidate_overlay_index.json",
    "ball_click_review_index.json",
    "contact_window_candidates.json",
    "contact_window_review.json",
    "frame_compute_plan.json",
    "body_compute_execution.json",
    "body_mesh_readiness.json",
    "virtual_world.json",
    "virtual_world_paddle_preview.json",
    "virtual_world_review_index.json",
    "replay_scene.json",
    "racket_stage_diagnostics.json",
    "racket_pose_readiness.json",
    "racket_promotion_audit.json",
    "paddle_true_corner_review.json",
    "pipeline_readiness_e2e.json",
    "court_line_evidence.json",
    "ball_inflections.json",
    "audio_onsets.json",
    "wrist_velocity_peaks.json",
    "racket_model_runtime_readiness.json",
)


def build_review_packet(
    run_root: str | Path,
    *,
    packet_id: str | None = None,
    corrections_root: str | Path = "corrections/inbox",
    include_clips: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_clips: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    """Scan a run/review root and produce a compact human-review index."""

    root = Path(run_root)
    packet = packet_id or root.name or "review_packet"
    clip_filter = _ClipFilter(include_clips=include_clips, exclude_clips=exclude_clips)
    pipeline_runs = [
        summary
        for summary in (_pipeline_run_summary(path) for path in sorted(root.rglob("pipeline_run.json")))
        if clip_filter.accepts(str(summary["clip"]))
    ]
    indexed_artifacts = [
        summary
        for summary in (_review_artifact_summary(path) for path in sorted(_iter_review_index_paths(root)))
        if clip_filter.accepts(str(summary["clip"]))
    ]
    review_artifacts = sorted(
        [*indexed_artifacts, *_supplemental_review_artifacts(root, clip_filter)],
        key=lambda artifact: (str(artifact["clip"]), str(artifact["artifact_type"]), str(artifact["path"])),
    )
    corrections_template = Path(corrections_root) / f"{packet}_corrections.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "packet_id": packet,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "run_root": str(root),
        "pipeline_run_count": len(pipeline_runs),
        "pipeline_runs": pipeline_runs,
        "review_artifact_count": len(review_artifacts),
        "review_artifacts": review_artifacts,
        "corrections_manifest_template": str(corrections_template),
        "human_next_steps": _human_next_steps(corrections_template),
    }


def write_review_packet(
    packet: Mapping[str, Any],
    *,
    out_dir: str | Path,
    write_corrections_template: bool = False,
) -> dict[str, Any]:
    """Write packet JSON, Markdown, and optionally an editable corrections template."""

    packet_id = str(packet["packet_id"])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"{packet_id}.json"
    markdown_path = out_path / f"{packet_id}.md"
    html_path = out_path / f"{packet_id}.html"
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(review_packet_markdown(packet), encoding="utf-8")
    html_path.write_text(review_packet_html(packet, base_dir=out_path), encoding="utf-8")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
    }
    if write_corrections_template:
        template_path = Path(str(packet["corrections_manifest_template"]))
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            json.dumps(_corrections_template(packet), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["corrections_template_path"] = str(template_path)
    return summary


def review_packet_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Pickleball Human Review Packet",
        "",
        f"- Packet: `{packet['packet_id']}`",
        f"- Run root: `{packet['run_root']}`",
        f"- Pipeline runs: {packet['pipeline_run_count']}",
        f"- Review artifacts: {packet['review_artifact_count']}",
        f"- Corrections template: `{packet['corrections_manifest_template']}`",
        "",
        "## Pipeline Runs",
        "",
    ]
    pipeline_runs = packet.get("pipeline_runs", [])
    if not pipeline_runs:
        lines.append("- No `pipeline_run.json` files found.")
    for run in pipeline_runs:
        failed = f", failed stage `{run['failed_stage']}`" if run.get("failed_stage") else ""
        lines.append(f"- `{run['clip']}` `{run['requested_stage']}`: `{run['status']}`{failed} ({run['path']})")
        for note in run.get("notes", []):
            lines.append(f"  - {note}")

    lines.extend(["", "## Review Artifacts", ""])
    review_artifacts = packet.get("review_artifacts", [])
    if not review_artifacts:
        lines.append("- No review index files found.")
    for artifact in review_artifacts:
        lines.append(f"- `{artifact['clip']}` `{artifact['artifact_type']}`: `{artifact['status']}` ({artifact['path']})")
        for detail in artifact.get("details", []):
            lines.append(f"  - {detail}")
        for watch_path in artifact.get("watch_paths", []):
            lines.append(f"  - Watch: `{watch_path}`")
        for warning in artifact.get("warnings", []):
            lines.append(f"  - Warning: {warning}")

    lines.extend(["", "## Correction Flow", ""])
    for step in packet.get("human_next_steps", []):
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def review_packet_html(packet: Mapping[str, Any], *, base_dir: str | Path | None = None) -> str:
    """Render a dependency-free browser review page for packet artifacts."""

    packet_id = str(packet["packet_id"])
    base_path = Path(base_dir) if base_dir is not None else Path.cwd()
    pipeline_runs = packet.get("pipeline_runs", [])
    review_artifacts = packet.get("review_artifacts", [])
    title = f"Pickleball Human Review Packet - {packet_id}"

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        "<style>",
        _review_packet_css(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        f"<h1>{escape(title)}</h1>",
        '<p class="lede">Open overlays first, then edit corrections only after the artifact state looks wrong.</p>',
        '<dl class="summary-grid">',
        _summary_item("Run root", str(packet.get("run_root", ""))),
        _summary_item("Pipeline runs", str(packet.get("pipeline_run_count", 0))),
        _summary_item("Review artifacts", str(packet.get("review_artifact_count", 0))),
        _summary_item("Corrections template", str(packet.get("corrections_manifest_template", ""))),
        "</dl>",
        "</header>",
        "<section>",
        "<h2>Pipeline Runs</h2>",
    ]
    if not pipeline_runs:
        parts.append('<p class="muted">No pipeline_run.json files found.</p>')
    for run in pipeline_runs:
        if not isinstance(run, Mapping):
            continue
        parts.append('<article class="panel">')
        failed = f" failed at {run.get('failed_stage')}" if run.get("failed_stage") else ""
        parts.append(
            f"<h3>{escape(str(run.get('clip', 'unknown')))} "
            f"<span>{escape(str(run.get('status', 'unknown')))}{escape(failed)}</span></h3>"
        )
        parts.append(f'<p class="path">{escape(str(run.get("path", "")))}</p>')
        parts.extend(_html_list(run.get("notes", []), class_name="notes"))
        parts.append("</article>")

    parts.extend(["</section>", "<section>", "<h2>Review Artifacts</h2>"])
    if not review_artifacts:
        parts.append('<p class="muted">No review index files found.</p>')
    for artifact in review_artifacts:
        if isinstance(artifact, Mapping):
            parts.append(_artifact_html(artifact, base_path))

    parts.extend(["</section>", "<section>", "<h2>Correction Flow</h2>"])
    parts.extend(_html_list(packet.get("human_next_steps", []), class_name="steps"))
    parts.extend(["</section>", "</main>", "</body>", "</html>", ""])
    return "\n".join(parts)


def _artifact_html(artifact: Mapping[str, Any], base_dir: Path) -> str:
    lines = ['<article class="panel artifact">']
    artifact_type = str(artifact.get("artifact_type", "unknown"))
    status = str(artifact.get("status", "unknown"))
    lines.append(
        f"<h3>{escape(str(artifact.get('clip', 'unknown')))} "
        f"<span>{escape(artifact_type)} · {escape(status)}</span></h3>"
    )
    lines.append(_path_html(str(artifact.get("path", "")), base_dir))
    lines.extend(_html_list(artifact.get("details", []), class_name="details"))
    watch_paths = artifact.get("watch_paths", [])
    if isinstance(watch_paths, list) and watch_paths:
        lines.append('<div class="watch-grid">')
        for watch_path in watch_paths:
            lines.append(_watch_html(str(watch_path), base_dir))
        lines.append("</div>")
    lines.extend(_html_list(artifact.get("warnings", []), class_name="warnings"))
    lines.append("</article>")
    return "\n".join(lines)


def _watch_html(path: str, base_dir: Path) -> str:
    src = escape(_html_resource_path(path, base_dir), quote=True)
    label = escape(Path(path).name or path)
    if Path(path).suffix.lower() in {".mp4", ".mov", ".webm"}:
        return (
            '<figure class="watch">'
            f'<video controls preload="metadata" src="{src}"></video>'
            f"<figcaption>{label}</figcaption>"
            "</figure>"
        )
    return f'<p class="watch-link"><a href="{src}">{label}</a></p>'


def _path_html(path: str, base_dir: Path) -> str:
    if not path:
        return '<p class="path"></p>'
    src = escape(_html_resource_path(path, base_dir), quote=True)
    label = escape(path)
    return f'<p class="path"><a href="{src}">{label}</a></p>'


def _html_resource_path(path: str, base_dir: Path) -> str:
    if path.startswith(("http://", "https://", "file://")):
        return path
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists():
        return Path(os.path.relpath(candidate, base_dir)).as_posix()
    return path


def _html_list(items: Any, *, class_name: str) -> list[str]:
    if not isinstance(items, list) or not items:
        return []
    lines = [f'<ul class="{class_name}">']
    for item in items:
        lines.append(f"<li>{escape(str(item))}</li>")
    lines.append("</ul>")
    return lines


def _summary_item(label: str, value: str) -> str:
    return f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"


def _review_packet_css() -> str:
    return """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #18181b;
  background: #f7f7f4;
}
body {
  margin: 0;
}
* {
  box-sizing: border-box;
}
main {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}
header, section {
  max-width: 100%;
  margin-bottom: 28px;
}
h1, h2, h3, p {
  margin-top: 0;
}
h1, h3 {
  overflow-wrap: anywhere;
  word-break: break-word;
}
h1 {
  font-size: 30px;
  line-height: 1.15;
  margin-bottom: 8px;
}
h2 {
  font-size: 20px;
  margin-bottom: 12px;
}
h3 {
  display: block;
  font-size: 16px;
  margin-bottom: 8px;
}
h3 span {
  color: #52525b;
  display: block;
  font-size: 13px;
  font-weight: 500;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.lede, .muted {
  color: #52525b;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin: 20px 0 0;
}
.summary-grid div, .panel {
  background: #fff;
  border: 1px solid #ddd9cf;
  border-radius: 8px;
}
.summary-grid div {
  padding: 12px;
}
dt {
  color: #71717a;
  font-size: 12px;
  text-transform: uppercase;
}
dd {
  margin: 4px 0 0;
  overflow-wrap: anywhere;
}
.panel {
  margin-bottom: 10px;
  min-width: 0;
  padding: 14px;
}
.path {
  color: #71717a;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.path a {
  color: inherit;
}
.path a:hover {
  color: #18181b;
}
ul {
  margin: 10px 0 0;
  padding-left: 20px;
}
li {
  overflow-wrap: anywhere;
  word-break: break-word;
}
.warnings li {
  color: #9a3412;
}
.watch-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.watch {
  margin: 0;
}
video {
  aspect-ratio: 16 / 9;
  background: #18181b;
  border-radius: 6px;
  display: block;
  width: 100%;
}
figcaption, .watch-link {
  color: #52525b;
  font-size: 12px;
  margin-top: 6px;
  overflow-wrap: anywhere;
}
@media (max-width: 640px) {
  main {
    width: min(100% - 20px, 1180px);
    padding-top: 18px;
  }
  h1 {
    font-size: 24px;
  }
}
""".strip()


def _pipeline_run_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    stages = payload.get("stages", [])
    failed_stage = ""
    notes: list[str] = []
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_notes = [str(note) for note in stage.get("notes", []) if note]
            if stage.get("status") in {"fail", "blocked"} and not failed_stage:
                failed_stage = str(stage.get("stage", ""))
                notes.extend(stage_notes)
            elif stage_notes and payload.get("status") != "pass":
                notes.extend(stage_notes)
    return {
        "path": str(path),
        "clip": str(payload.get("clip", path.parent.name)),
        "requested_stage": str(payload.get("requested_stage", "")),
        "status": str(payload.get("status", "unknown")),
        "failed_stage": failed_stage,
        "notes": notes,
    }


def _review_artifact_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    watch_paths = _watch_paths(payload, index_path=path)
    artifact_type = _artifact_type(payload, path)
    return {
        "path": str(path),
        "clip": _artifact_clip(payload, path),
        "artifact_type": artifact_type,
        "status": _artifact_status(payload),
        "qualitative_status": str(payload.get("qualitative_status", "")),
        "watch_paths": watch_paths,
        "details": _artifact_details(payload),
        "available_layers": list(payload.get("available_layers", [])) if isinstance(payload.get("available_layers"), list) else [],
        "warnings": _artifact_warnings(payload, artifact_path=path),
        "source_path": str(payload.get("source_path", "")) if payload.get("source_path") else "",
    }


def _artifact_type(payload: Mapping[str, Any], path: Path) -> str:
    if payload.get("artifact_type"):
        return str(payload["artifact_type"])
    if path.name == "court_line_evidence.json":
        return "racketsport_court_line_evidence"
    if path.name == "replay_scene.json" and _is_replay_scene_payload(payload):
        return "racketsport_replay_scene"
    return path.stem


def _artifact_clip(payload: Mapping[str, Any], path: Path) -> str:
    if payload.get("artifact_type") == "racketsport_racket_model_runtime_readiness":
        return GLOBAL_REVIEW_CLIP
    if payload.get("clip"):
        return str(payload["clip"])
    if path.parent.name in {"compare", "player_tracks", "racket_candidates"}:
        return path.parent.parent.name
    return path.parent.name


def _artifact_status(payload: Mapping[str, Any]) -> str:
    if payload.get("status"):
        return str(payload["status"])
    if payload.get("artifact_type") == "racketsport_frame_compute_plan":
        return "planned"
    if payload.get("artifact_type") == "racketsport_body_compute_execution":
        return "scheduled"
    if payload.get("artifact_type") == "racketsport_ball_inflections":
        return "review_only"
    if payload.get("artifact_type") in {"racketsport_audio_onsets", "racketsport_wrist_velocity_peaks"}:
        return str(payload.get("status") or "review_only")
    if payload.get("artifact_type") == "racketsport_contact_window_candidates":
        return "needs_review"
    if payload.get("artifact_type") == "racketsport_contact_window_review":
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            pending = summary.get("pending_count")
            candidate_count = summary.get("candidate_count")
            if pending == candidate_count:
                return "pending_review"
            if isinstance(pending, int) and pending > 0:
                return "partially_reviewed"
            return "reviewed"
        return "pending_review"
    if payload.get("artifact_type") == "racketsport_virtual_world":
        return "assembled"
    if _is_replay_scene_payload(payload):
        return "review_only"
    if _is_court_line_evidence_payload(payload):
        aggregate = payload.get("aggregate")
        ready = aggregate.get("auto_calibration_ready") if isinstance(aggregate, Mapping) else False
        return "ready" if ready is True else "blocked"
    return "unknown"


def _artifact_details(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("artifact_type") != "racketsport_frame_compute_plan":
        if payload.get("artifact_type") == "racketsport_virtual_world":
            return _virtual_world_details(payload)
        if payload.get("artifact_type") == "racketsport_virtual_world_review":
            return _virtual_world_review_details(payload)
        if payload.get("artifact_type") == "racketsport_body_compute_execution":
            return _body_compute_execution_details(payload)
        if payload.get("artifact_type") == "racketsport_body_mesh_readiness":
            return _body_mesh_readiness_details(payload)
        if payload.get("artifact_type") == "racketsport_contact_window_candidates":
            return _contact_window_candidate_details(payload)
        if payload.get("artifact_type") == "racketsport_contact_window_review":
            return _contact_window_review_details(payload)
        if payload.get("artifact_type") == "racketsport_player_track_overlay":
            return _player_track_overlay_details(payload)
        if payload.get("artifact_type") == "racketsport_racket_candidate_overlay":
            return _racket_candidate_overlay_details(payload)
        if payload.get("artifact_type") == "racketsport_racket_stage_diagnostics":
            return _racket_stage_diagnostics_details(payload)
        if payload.get("artifact_type") == "racketsport_racket_pose_readiness":
            return _racket_pose_readiness_details(payload)
        if payload.get("artifact_type") == "racketsport_racket_promotion_audit":
            return _racket_promotion_audit_details(payload)
        if payload.get("artifact_type") == "racketsport_paddle_true_corner_review":
            return _paddle_true_corner_review_details(payload)
        if payload.get("artifact_type") == "racketsport_pipeline_artifact_readiness":
            return _pipeline_readiness_details(payload)
        if payload.get("artifact_type") == "racketsport_ball_inflections":
            return _ball_inflection_details(payload)
        if payload.get("artifact_type") == "racketsport_audio_onsets":
            return _audio_onset_details(payload)
        if payload.get("artifact_type") == "racketsport_wrist_velocity_peaks":
            return _wrist_velocity_peak_details(payload)
        if payload.get("artifact_type") == "racketsport_racket_model_runtime_readiness":
            return _racket_model_runtime_readiness_details(payload)
        if _is_replay_scene_payload(payload):
            return _replay_scene_details(payload)
        if _is_court_line_evidence_payload(payload):
            return _court_line_evidence_details(payload)
        return []
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [f"Frames planned: {payload.get('frame_count', 0)}"]
    by_tier = summary.get("by_tier")
    if isinstance(by_tier, Mapping) and by_tier:
        details.append("Tiers: " + ", ".join(f"{key}={value}" for key, value in sorted(by_tier.items())))
    by_reason = summary.get("by_reason")
    if isinstance(by_reason, Mapping) and by_reason:
        details.append("Reasons: " + ", ".join(f"{key}={value}" for key, value in sorted(by_reason.items())))
    by_player_target_representation = summary.get("by_player_target_representation")
    if isinstance(by_player_target_representation, Mapping) and by_player_target_representation:
        details.append(
            "Player targets: "
            + ", ".join(f"{key}={value}" for key, value in sorted(by_player_target_representation.items()))
        )
    deep_mesh_window_count = summary.get("deep_mesh_window_count")
    deep_mesh_frame_count = summary.get("deep_mesh_frame_count")
    if isinstance(deep_mesh_window_count, int) and deep_mesh_window_count > 0:
        frame_count = deep_mesh_frame_count if isinstance(deep_mesh_frame_count, int) else 0
        details.append(f"Deep mesh windows: {deep_mesh_window_count} ({frame_count} frames)")
    return details


def _artifact_warnings(payload: Mapping[str, Any], *, artifact_path: Path | None = None) -> list[str]:
    warnings = [str(warning) for warning in payload.get("warnings", [])] if isinstance(payload.get("warnings"), list) else []
    if payload.get("artifact_type") == "racketsport_contact_window_candidates":
        if payload.get("not_gate_verified") is True and "review_only_not_gate_verified" not in warnings:
            warnings.append("review_only_not_gate_verified")
        if payload.get("trusted_for_body") is False and "not_trusted_for_body" not in warnings:
            warnings.append("not_trusted_for_body")
    if payload.get("artifact_type") == "racketsport_contact_window_review":
        summary = payload.get("summary")
        pending_count = summary.get("pending_count") if isinstance(summary, Mapping) else 0
        if isinstance(pending_count, int) and pending_count > 0 and "pending_contact_review" not in warnings:
            warnings.append("pending_contact_review")
    if payload.get("artifact_type") == "racketsport_racket_pose_readiness":
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if payload.get("artifact_type") == "racketsport_racket_promotion_audit":
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if payload.get("artifact_type") == "racketsport_paddle_true_corner_review":
        blockers = payload.get("promotion_blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if payload.get("artifact_type") == "racketsport_pipeline_artifact_readiness":
        if payload.get("status") != "ready" and "pipeline_not_ready" not in warnings:
            warnings.append("pipeline_not_ready")
        missing = payload.get("missing_artifacts")
        if isinstance(missing, list):
            warnings.extend(f"missing:{artifact}" for artifact in missing if f"missing:{artifact}" not in warnings)
        semantic_blockers = payload.get("semantic_blockers")
        if isinstance(semantic_blockers, list):
            warnings.extend(
                f"semantic:{blocker}"
                for blocker in semantic_blockers
                if f"semantic:{blocker}" not in warnings
            )
    if payload.get("artifact_type") == "racketsport_ball_inflections":
        if payload.get("not_gate_verified") is True and "review_only_not_gate_verified" not in warnings:
            warnings.append("review_only_not_gate_verified")
        required_cues = payload.get("requires_additional_cues")
        if isinstance(required_cues, list):
            warnings.extend(
                warning
                for cue in required_cues
                for warning in _sibling_cue_warnings(str(cue), artifact_path=artifact_path)
                if warning not in warnings
            )
    if payload.get("artifact_type") in {"racketsport_audio_onsets", "racketsport_wrist_velocity_peaks"}:
        if payload.get("not_gate_verified") is True and "cue_not_gate_verified" not in warnings:
            warnings.append("cue_not_gate_verified")
        if payload.get("trusted_for_contact") is False and "not_trusted_for_contact" not in warnings:
            warnings.append("not_trusted_for_contact")
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if payload.get("artifact_type") == "racketsport_racket_model_runtime_readiness":
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if payload.get("artifact_type") == "racketsport_body_compute_execution":
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            targeted_count = summary.get("scheduled_targeted_reviewed_contact_frame_count")
            if isinstance(targeted_count, int) and targeted_count > 0:
                warning = "targeted_reviewed_contact_body_schedule"
                if warning not in warnings:
                    warnings.append(warning)
            incomplete_count = summary.get("scheduled_coverage_incomplete_frame_count")
            if isinstance(incomplete_count, int) and incomplete_count > 0:
                warning = "scheduled_with_incomplete_player_coverage"
                if warning not in warnings:
                    warnings.append(warning)
    if payload.get("artifact_type") == "racketsport_body_mesh_readiness":
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(str(blocker) for blocker in blockers if str(blocker) not in warnings)
    if _is_replay_scene_payload(payload) and "review_scene_not_accuracy_gate" not in warnings:
        warnings.append("review_scene_not_accuracy_gate")
    if _is_court_line_evidence_payload(payload):
        warnings.extend(warning for warning in _court_line_evidence_warnings(payload) if warning not in warnings)
    return warnings


def _sibling_cue_warnings(cue: str, *, artifact_path: Path | None) -> list[str]:
    filename = f"{cue}.json"
    if artifact_path is None:
        return [f"missing_cue:{cue}"]
    path = artifact_path.with_name(filename)
    if not path.is_file():
        return [f"missing_cue:{cue}"]
    try:
        payload = _read_json(path)
    except Exception:
        return [f"invalid_cue:{cue}"]
    warnings: list[str] = []
    if isinstance(payload, Mapping):
        if payload.get("status") == "blocked":
            warnings.append(f"blocked_cue:{cue}")
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            warnings.extend(f"{cue}:{blocker}" for blocker in blockers)
        if payload.get("trusted_for_contact") is False:
            warnings.append(f"not_trusted_cue:{cue}")
    return warnings


def _body_compute_execution_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Mode: {payload.get('mode', 'unknown')}",
        f"Scheduled frames: {summary.get('scheduled_frame_count', 0)}",
        f"Scheduled player-frames: {summary.get('scheduled_player_frame_count', 0)}",
        f"Skipped frames: {summary.get('skipped_frame_count', 0)}",
    ]
    scheduled_by_target = summary.get("scheduled_by_target_representation")
    if isinstance(scheduled_by_target, Mapping) and scheduled_by_target:
        details.append(
            "Scheduled targets: "
            + ", ".join(f"{key}={value}" for key, value in sorted(scheduled_by_target.items()))
        )
    scheduled_by_reason = summary.get("scheduled_by_reason")
    if isinstance(scheduled_by_reason, Mapping) and scheduled_by_reason:
        details.append(
            "Scheduled reasons: " + ", ".join(f"{key}={value}" for key, value in sorted(scheduled_by_reason.items()))
        )
    targeted_count = summary.get("scheduled_targeted_reviewed_contact_frame_count")
    if isinstance(targeted_count, int) and targeted_count > 0:
        details.append(f"Scheduled targeted reviewed-contact frames: {targeted_count}")
    incomplete_count = summary.get("scheduled_coverage_incomplete_frame_count")
    if isinstance(incomplete_count, int) and incomplete_count > 0:
        details.append(f"Scheduled incomplete-coverage frames: {incomplete_count}")
    skipped_by_tier = summary.get("skipped_by_tier")
    if isinstance(skipped_by_tier, Mapping) and skipped_by_tier:
        details.append("Skipped tiers: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped_by_tier.items())))
    skipped_by_target = summary.get("skipped_by_target_representation")
    if isinstance(skipped_by_target, Mapping) and skipped_by_target:
        details.append("Skipped targets: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped_by_target.items())))
    skipped_by_reason = summary.get("skipped_by_reason")
    if isinstance(skipped_by_reason, Mapping) and skipped_by_reason:
        details.append("Skipped reasons: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped_by_reason.items())))
    return details


def _body_mesh_readiness_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    min_vertices = summary.get("mesh_vertex_count_min", 0)
    max_vertices = summary.get("mesh_vertex_count_max", 0)
    details = [
        f"World mesh available: {str(payload.get('world_mesh_available', False)).lower()}",
        f"Trusted for BODY promotion: {str(payload.get('trusted_for_body_promotion', False)).lower()}",
        f"Players: {summary.get('player_count', 0)}",
        f"Mesh players: {summary.get('mesh_player_count', 0)}",
        f"Mesh frames: {summary.get('mesh_frame_count', 0)}",
        f"Mesh vertices/frame: {min_vertices}-{max_vertices}",
        f"Joints players: {summary.get('joints_player_count', 0)}",
        f"Joints frames: {summary.get('joints_frame_count', 0)}",
    ]
    representation_plan = payload.get("representation_plan")
    if isinstance(representation_plan, Mapping):
        details.append(f"Representation decision: {payload.get('representation_decision', 'unknown')}")
        details.append(
            "World mesh demand: "
            f"requested={representation_plan.get('requested_world_mesh_frame_count', 0)}, "
            f"scheduled={representation_plan.get('scheduled_world_mesh_frame_count', 0)}, "
            f"available={representation_plan.get('available_mesh_frame_count', 0)}"
        )
        details.append(
            "Representation targets: "
            f"lane_a_skeleton={representation_plan.get('lane_a_skeleton_target_count', 0)}, "
            f"manual_review_required={representation_plan.get('manual_review_required_target_count', 0)}, "
            f"world_mesh={representation_plan.get('requested_world_mesh_player_target_count', 0)}"
        )
    return details


def _pipeline_readiness_details(payload: Mapping[str, Any]) -> list[str]:
    details = [f"Requested stage: {payload.get('requested_stage', 'unknown')}"]
    missing = payload.get("missing_artifacts")
    if isinstance(missing, list) and missing:
        details.append("Missing artifacts: " + ", ".join(str(artifact) for artifact in missing))
    else:
        details.append("Missing artifacts: none")
    semantic_blockers = payload.get("semantic_blockers")
    if isinstance(semantic_blockers, list) and semantic_blockers:
        details.append("Semantic blockers: " + ", ".join(str(blocker) for blocker in semantic_blockers))
    stages = payload.get("stages")
    if isinstance(stages, list) and stages:
        status_counts: dict[str, int] = {}
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            status = str(stage.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        if status_counts:
            order = ("ready", "not_ready", "blocked", "unknown")
            ordered_items = [(key, status_counts[key]) for key in order if key in status_counts]
            ordered_items.extend((key, value) for key, value in sorted(status_counts.items()) if key not in order)
            details.append("Stages: " + ", ".join(f"{key}={value}" for key, value in ordered_items))
    return details


def _ball_inflection_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Candidates: {summary.get('candidate_count', 0)}",
        f"Raw candidates before suppression: {summary.get('raw_candidate_count', 0)}",
        f"Usable ball frames: {summary.get('usable_frame_count', 0)}",
        f"Source: {payload.get('source', 'unknown')}",
    ]
    required_cues = payload.get("requires_additional_cues")
    if isinstance(required_cues, list) and required_cues:
        details.append("Requires additional cues: " + ", ".join(str(cue) for cue in required_cues))
    turn = summary.get("min_turn_degrees")
    speed = summary.get("min_speed_mps")
    separation = summary.get("min_candidate_separation_s")
    if isinstance(turn, int | float) and isinstance(speed, int | float) and isinstance(separation, int | float):
        details.append(f"Thresholds: turn>={float(turn):.1f}deg, speed>={float(speed):.2f}mps, separation>={float(separation):.2f}s")
    return details


def _audio_onset_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Onsets: {summary.get('onset_count', 0)}",
        f"Raw peaks before suppression: {summary.get('raw_peak_count', 0)}",
        f"Source: {payload.get('source', 'unknown')}",
    ]
    sample_rate = payload.get("sample_rate_hz")
    details.append(f"Sample rate: {sample_rate}Hz" if isinstance(sample_rate, int) else "Sample rate: unavailable")
    threshold = summary.get("threshold_score")
    separation = summary.get("min_separation_s")
    if isinstance(threshold, int | float) and isinstance(separation, int | float):
        details.append(f"Thresholds: score>={float(threshold):.2f}, separation>={float(separation):.2f}s")
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        details.append("Blockers: " + ", ".join(str(blocker) for blocker in blockers))
    return details


def _wrist_velocity_peak_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Peaks: {summary.get('peak_count', 0)}",
        f"Raw peaks before suppression: {summary.get('raw_peak_count', 0)}",
        f"Usable wrist samples: {summary.get('usable_sample_count', 0)}",
        f"Source: {payload.get('source', 'unknown')}",
    ]
    mapping = payload.get("joint_mapping")
    if isinstance(mapping, Mapping) and mapping:
        details.append("Joint mapping: " + ", ".join(f"{key}={value}" for key, value in sorted(mapping.items())))
    else:
        details.append("Joint mapping: unavailable")
    min_speed = summary.get("min_speed_mps")
    if isinstance(min_speed, int | float):
        details.append(f"Thresholds: speed>={float(min_speed):.2f}mps")
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        details.append("Blockers: " + ", ".join(str(blocker) for blocker in blockers))
    return details


def _racket_model_runtime_readiness_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    execution = payload.get("execution")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Components: {summary.get('component_count', 0)}",
        f"Runtime-ready components: {summary.get('runtime_ready_count', 0)}",
        f"Asset ready: {str(summary.get('asset_ready', False)).lower()}",
        f"May run GPU smoke: {str(summary.get('may_run_gpu_smoke', False)).lower()}",
        f"May promote RKT: {str(summary.get('may_promote_rkt', False)).lower()}",
    ]
    if isinstance(execution, Mapping):
        details.append(
            "Execution: "
            f"cpu_only={str(execution.get('cpu_only', False)).lower()}, "
            f"uses_gpu={str(execution.get('uses_gpu', False)).lower()}, "
            f"runs_inference={str(execution.get('runs_inference', False)).lower()}, "
            f"claims_model_has_run={str(execution.get('claims_model_has_run', False)).lower()}"
        )
    return details


def _is_replay_scene_payload(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("world_frame") == "court_Z0"
        and isinstance(payload.get("court_glb"), str)
        and isinstance(payload.get("players"), list)
        and isinstance(payload.get("points"), list)
    )


def _replay_scene_details(payload: Mapping[str, Any]) -> list[str]:
    players = payload.get("players", [])
    points = payload.get("points", [])
    point_size_mb = 0.0
    if isinstance(points, list):
        for point in points:
            if isinstance(point, Mapping) and isinstance(point.get("size_mb"), int | float):
                point_size_mb += float(point["size_mb"])
    return [
        f"FPS: {payload.get('fps', 0)}",
        f"Players: {len(players) if isinstance(players, list) else 0}",
        f"Replay points: {len(points) if isinstance(points, list) else 0}",
        f"Court GLB: {payload.get('court_glb', '')}",
        f"Point GLB total MB: {round(point_size_mb, 6)}",
    ]


def _is_court_line_evidence_payload(payload: Mapping[str, Any]) -> bool:
    return (
        isinstance(payload.get("aggregate"), Mapping)
        and isinstance(payload.get("line_observations"), list)
        and isinstance(payload.get("net_observations"), list)
    )


def _court_line_evidence_details(payload: Mapping[str, Any]) -> list[str]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return []
    details = [
        f"Source: {payload.get('source', 'unknown')}",
        f"Auto calibration ready: {str(aggregate.get('auto_calibration_ready', False)).lower()}",
    ]
    accepted_line_ids = aggregate.get("accepted_line_ids")
    if isinstance(accepted_line_ids, list):
        details.append(f"Accepted lines: {len(accepted_line_ids)} ({', '.join(str(item) for item in accepted_line_ids)})")
    missing_line_ids = aggregate.get("missing_required_line_ids")
    if isinstance(missing_line_ids, list):
        details.append(
            "Missing required lines: " + (", ".join(str(item) for item in missing_line_ids) if missing_line_ids else "none")
        )
    missing_net_ids = aggregate.get("missing_required_net_ids")
    if isinstance(missing_net_ids, list):
        details.append(
            "Missing required net cues: "
            + (", ".join(str(item) for item in missing_net_ids) if missing_net_ids else "none")
        )
    mean_residual = aggregate.get("mean_residual_px")
    p95_residual = aggregate.get("p95_residual_px")
    if isinstance(mean_residual, int | float) and isinstance(p95_residual, int | float):
        details.append(f"Residual px: mean={float(mean_residual):.2f}, p95={float(p95_residual):.2f}")
    reasons = aggregate.get("reasons")
    if isinstance(reasons, list) and reasons:
        details.append("Reasons: " + ", ".join(str(reason) for reason in reasons))
    return details


def _court_line_evidence_warnings(payload: Mapping[str, Any]) -> list[str]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return ["invalid_court_line_evidence"]
    warnings: list[str] = []
    if aggregate.get("auto_calibration_ready") is not True:
        warnings.append("court_evidence_not_ready")
    missing_line_ids = aggregate.get("missing_required_line_ids")
    if isinstance(missing_line_ids, list):
        warnings.extend(f"missing_line:{line_id}" for line_id in missing_line_ids)
    missing_net_ids = aggregate.get("missing_required_net_ids")
    if isinstance(missing_net_ids, list):
        warnings.extend(f"missing_net:{net_id}" for net_id in missing_net_ids)
    return warnings


def _contact_window_candidate_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Candidates: {summary.get('candidate_count', 0)}",
        f"Rejected source items: {summary.get('rejected_item_count', 0)}",
    ]
    by_type = summary.get("by_type")
    if isinstance(by_type, Mapping) and by_type:
        details.append("Types: " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())))
    by_status = summary.get("by_status")
    if isinstance(by_status, Mapping) and by_status:
        details.append("Statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(by_status.items())))
    details.append(f"Trusted for BODY: {str(payload.get('trusted_for_body', False)).lower()}")
    details.append(f"Promotion target: {payload.get('promotion_target', 'contact_windows.json')}")
    uncertainty_flags = summary.get("uncertainty_flags")
    if isinstance(uncertainty_flags, list) and uncertainty_flags:
        details.append("Uncertainty flags: " + ", ".join(str(flag) for flag in uncertainty_flags))
    return details


def _contact_window_review_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    return [
        f"Candidates: {summary.get('candidate_count', 0)}",
        f"Pending: {summary.get('pending_count', 0)}",
        f"Accepted: {summary.get('accepted_count', 0)}",
        f"Rejected: {summary.get('rejected_count', 0)}",
        f"Promotion target: {payload.get('promotion_target', 'contact_windows.json')}",
    ]


def _virtual_world_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Players: {summary.get('player_count', 0)}",
        f"Mesh players: {summary.get('mesh_player_count', 0)}",
        f"Ball frames: {summary.get('ball_frame_count', 0)}",
    ]
    _append_count_detail(details, "Mesh player frames", summary.get("mesh_player_frame_count"))
    _append_count_detail(details, "Joint player frames", summary.get("joint_player_frame_count"))
    _append_count_detail(details, "Track-only player frames", summary.get("track_only_player_frame_count"))
    _append_count_detail(details, "Approx ball frames", summary.get("approx_ball_frame_count"))
    _append_count_detail(details, "Paddle players", summary.get("paddle_player_count"))
    details.append(f"Paddle frames: {summary.get('paddle_frame_count', 0)}")
    _append_count_detail(details, "Ambiguous paddle frames", summary.get("ambiguous_paddle_frame_count"))
    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        if "missing_paddle_pose" in {str(warning) for warning in warnings}:
            details.append("Paddle status: no racket_pose.json frames; add racket_candidates.json or run the racket stage")
        details.append("Warnings: " + ", ".join(str(warning) for warning in warnings))
    return details


def _append_count_detail(details: list[str], label: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float) and value > 0:
        details.append(f"{label}: {int(value)}")


def _virtual_world_review_details(payload: Mapping[str, Any]) -> list[str]:
    details = payload.get("details")
    if isinstance(details, list):
        return [str(detail) for detail in details]
    return []


def _player_track_overlay_details(payload: Mapping[str, Any]) -> list[str]:
    details = []
    frame_count = payload.get("frame_count")
    player_count = payload.get("player_count")
    track_frame_count = payload.get("track_frame_count")
    if isinstance(frame_count, int):
        details.append(f"Frames rendered: {frame_count}")
    if isinstance(player_count, int):
        details.append(f"Players: {player_count}")
    if isinstance(track_frame_count, int):
        details.append(f"Track frames: {track_frame_count}")
    return details


def _racket_candidate_overlay_details(payload: Mapping[str, Any]) -> list[str]:
    details = []
    frame_count = payload.get("frame_count")
    candidate_player_count = payload.get("candidate_player_count")
    candidate_frame_count = payload.get("candidate_frame_count")
    rendered_candidate_count = payload.get("rendered_candidate_count")
    unrendered_candidate_count = payload.get("unrendered_candidate_count")
    coord_scale_x = payload.get("candidate_coord_scale_x")
    coord_scale_y = payload.get("candidate_coord_scale_y")
    if isinstance(frame_count, int):
        details.append(f"Frames rendered: {frame_count}")
    if isinstance(candidate_player_count, int):
        details.append(f"Candidate players: {candidate_player_count}")
    if isinstance(candidate_frame_count, int):
        details.append(f"Candidate frames: {candidate_frame_count}")
    if (
        isinstance(coord_scale_x, int | float)
        and isinstance(coord_scale_y, int | float)
        and (float(coord_scale_x) != 1.0 or float(coord_scale_y) != 1.0)
    ):
        details.append(f"Coordinate scale: x={float(coord_scale_x):.2f} y={float(coord_scale_y):.2f}")
    if isinstance(rendered_candidate_count, int):
        details.append(f"Rendered candidates: {rendered_candidate_count}")
    if isinstance(unrendered_candidate_count, int) and unrendered_candidate_count > 0:
        details.append(f"Unrendered candidates: {unrendered_candidate_count}")
    return details


def _racket_stage_diagnostics_details(payload: Mapping[str, Any]) -> list[str]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    return [
        f"Candidate frames: {metrics.get('candidate_frame_count', 0)}",
        f"Accepted pose frames: {metrics.get('accepted_frame_count', 0)}",
        f"Rejected ambiguous: {metrics.get('rejected_ambiguous_count', 0)}",
        f"Rejected high reprojection: {metrics.get('rejected_high_reprojection_count', 0)}",
        f"Invalid candidates: {metrics.get('invalid_candidate_count', 0)}",
    ]


def _racket_pose_readiness_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Candidate frames: {summary.get('candidate_frame_count', 0)}",
        f"Box-derived frames: {summary.get('box_derived_frame_count', 0)}",
        f"True corner/reference frames: {summary.get('true_corner_frame_count', 0)}",
        f"Reference/GT frames: {summary.get('reference_gt_frame_count', 0)}",
        f"Preview pose frames: {summary.get('preview_pose_frame_count', 0)}",
        f"Promoted pose frames: {summary.get('promoted_pose_frame_count', 0)}",
    ]
    source_evidence_counts = payload.get("source_evidence_counts")
    if isinstance(source_evidence_counts, Mapping) and source_evidence_counts:
        details.append(
            "Source evidence: "
            + ", ".join(f"{key}={value}" for key, value in sorted(source_evidence_counts.items()))
        )
    source_counts = payload.get("source_counts")
    if isinstance(source_counts, Mapping) and source_counts:
        details.append("Sources: " + ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())))
    local_readiness = payload.get("local_readiness")
    if isinstance(local_readiness, Mapping) and local_readiness:
        details.append(
            "Local readiness: "
            + ", ".join(f"{key}={str(value).lower()}" for key, value in sorted(local_readiness.items()))
        )
    missing_state = payload.get("missing_label_or_asset_state")
    if isinstance(missing_state, Mapping) and missing_state:
        details.append(
            "Missing labels/assets: " + ", ".join(f"{key}={value}" for key, value in sorted(missing_state.items()))
        )
    return details


def _racket_promotion_audit_details(payload: Mapping[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    details = [
        f"Canonical racket_pose.json present: {str(payload.get('canonical_racket_pose_present', False)).lower()}",
        f"Trusted for RKT promotion: {str(payload.get('trusted_for_rkt_promotion', False)).lower()}",
        f"Candidate frames: {summary.get('candidate_frame_count', 0)}",
        f"Box-derived candidate frames: {summary.get('box_derived_candidate_frame_count', 0)}",
        f"True corner/reference frames: {summary.get('true_corner_frame_count', 0)}",
        f"Reference/GT frames: {summary.get('reference_gt_frame_count', 0)}",
        f"Preview pose frames: {summary.get('preview_pose_frame_count', 0)}",
        f"Promoted pose frames: {summary.get('promoted_pose_frame_count', 0)}",
        f"Unsafe promoted frames: {summary.get('unsafe_promoted_frame_count', 0)}",
    ]
    source_evidence_counts = payload.get("source_evidence_counts")
    if isinstance(source_evidence_counts, Mapping) and source_evidence_counts:
        details.append(
            "Source evidence: "
            + ", ".join(f"{key}={value}" for key, value in sorted(source_evidence_counts.items()))
        )
    source_counts = payload.get("source_counts")
    if isinstance(source_counts, Mapping) and source_counts:
        details.append("Candidate sources: " + ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())))
    pose_source_counts = payload.get("pose_source_counts")
    if isinstance(pose_source_counts, Mapping) and pose_source_counts:
        details.append("Promoted pose sources: " + ", ".join(f"{key}={value}" for key, value in sorted(pose_source_counts.items())))
    unsafe_sources = payload.get("unsafe_promoted_sources")
    if isinstance(unsafe_sources, Mapping) and unsafe_sources:
        details.append(
            "Unsafe promoted sources: " + ", ".join(f"{key}={value}" for key, value in sorted(unsafe_sources.items()))
        )
    return details


def _paddle_true_corner_review_details(payload: Mapping[str, Any]) -> list[str]:
    details = [
        f"Trusted for RKT promotion: {str(payload.get('trusted_for_rkt_promotion', False)).lower()}",
        f"Candidate frames: {payload.get('candidate_frame_count', 0)}",
        f"Box-derived candidate frames: {payload.get('box_derived_candidate_count', 0)}",
        f"True corner labels: {payload.get('true_corner_label_count', 0)}",
        f"Reference/GT labels: {payload.get('reference_gt_count', 0)}",
        f"Required labels: {payload.get('required_label_count', 0)}",
    ]
    listed = payload.get("listed_required_label_count")
    if isinstance(listed, int):
        details.append(f"Listed required labels: {listed}")
    source_counts = payload.get("source_counts")
    if isinstance(source_counts, Mapping) and source_counts:
        details.append("Candidate sources: " + ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())))
    return details


def _watch_paths(payload: Mapping[str, Any], *, index_path: Path) -> list[str]:
    paths: list[str] = []
    for key in ("rendered_videos", "overlay_paths", "image_paths"):
        value = payload.get(key)
        if isinstance(value, list):
            paths.extend(_resolve_watch_path(str(item), index_path=index_path) for item in value if item)
    for key in ("overlay_path", "image_path", "review_html", "html_path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            paths.append(_resolve_watch_path(value, index_path=index_path))
    visuals = payload.get("visuals")
    if isinstance(visuals, list):
        for visual in visuals:
            if not isinstance(visual, Mapping):
                continue
            value = visual.get("path")
            if isinstance(value, str) and value:
                paths.append(_resolve_watch_path(value, index_path=index_path))
    if payload.get("artifact_type") == "racketsport_contact_window_review":
        sibling_html = index_path.with_suffix(".html")
        if sibling_html.is_file():
            paths.append(str(sibling_html))
    if _is_replay_scene_payload(payload):
        court_glb = payload.get("court_glb")
        if isinstance(court_glb, str) and court_glb:
            paths.append(str(index_path.parent / court_glb))
        points = payload.get("points", [])
        if isinstance(points, list):
            for point in points:
                if not isinstance(point, Mapping):
                    continue
                glb_url = point.get("glb_url")
                if isinstance(glb_url, str) and glb_url:
                    paths.append(str(index_path.parent / glb_url))
    return paths


def _resolve_watch_path(value: str, *, index_path: Path) -> str:
    original = Path(value)
    if original.exists():
        return value
    sibling = index_path.parent / original.name
    if sibling.exists():
        return str(sibling)
    return value


def _iter_review_index_paths(root: Path) -> list[Path]:
    return [path for name in REVIEW_INDEX_NAMES for path in root.rglob(name)]


class _ClipFilter:
    def __init__(
        self,
        *,
        include_clips: list[str] | tuple[str, ...] | set[str] | None,
        exclude_clips: list[str] | tuple[str, ...] | set[str] | None,
    ) -> None:
        self.includes = {str(clip) for clip in include_clips or []}
        self.excludes = {str(clip) for clip in exclude_clips or []}

    def accepts(self, clip: str) -> bool:
        if clip == GLOBAL_REVIEW_CLIP:
            return clip not in self.excludes
        if self.includes and clip not in self.includes:
            return False
        return clip not in self.excludes


def _supplemental_review_artifacts(root: Path, clip_filter: _ClipFilter) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        clip = clip_dir.name
        if not clip_filter.accepts(clip):
            continue
        artifacts.extend(_ball_review_artifacts(root, clip))
        artifacts.extend(_contact_window_artifacts(root, clip))
        artifacts.extend(_racket_candidate_artifacts(root, clip))
    return artifacts


def _contact_window_artifacts(root: Path, clip: str) -> list[dict[str, Any]]:
    path = root / clip / "contact_windows.json"
    if not path.is_file():
        return []
    try:
        artifact = ContactWindows.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        return [
            {
                "path": str(path),
                "clip": clip,
                "artifact_type": "racketsport_contact_windows",
                "status": "invalid",
                "qualitative_status": "contact_windows_failed_schema_validation",
                "watch_paths": [],
                "details": [f"Validation error: {exc}"],
                "available_layers": ["contact_windows"],
                "warnings": ["invalid_contact_windows"],
            }
        ]

    event_count = len(artifact.events)
    human_review_count = sum(1 for event in artifact.events if event.sources.human_review is not None)
    warnings: list[str] = []
    if event_count == 0:
        warnings.append("empty_contact_windows_no_deep_mesh")
    if 0 < human_review_count < event_count:
        warnings.append("not_all_events_human_reviewed")
    return [
        {
            "path": str(path),
            "clip": clip,
            "artifact_type": "racketsport_contact_windows",
            "status": "promoted" if event_count else "empty",
            "qualitative_status": "promoted_reviewed_contact_windows" if event_count else "empty_no_contact_events",
            "watch_paths": [],
            "details": _contact_windows_details(artifact),
            "available_layers": ["contact_windows"],
            "warnings": warnings,
        }
    ]


def _contact_windows_details(artifact: ContactWindows) -> list[str]:
    details = [f"Events: {len(artifact.events)}"]
    if not artifact.events:
        return details
    by_type: dict[str, int] = {}
    frames: list[int] = []
    times: list[float] = []
    human_review_count = 0
    for event in artifact.events:
        by_type[event.type] = by_type.get(event.type, 0) + 1
        frames.append(int(event.frame))
        times.append(float(event.t))
        if event.sources.human_review is not None:
            human_review_count += 1
    details.append("Types: " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())))
    details.append(f"Human-reviewed events: {human_review_count}")
    details.append(f"Frame range: {min(frames)}-{max(frames)}")
    details.append(f"Time range: {min(times):.3f}-{max(times):.3f}s")
    return details


def _ball_review_artifacts(root: Path, clip: str) -> list[dict[str, Any]]:
    clip_root = root / clip
    artifacts: list[dict[str, Any]] = []
    ball_base = clip_root / "tracknet_smoke_0000_0010"
    ball_track = ball_base / "ball_track_fusion_temporal_vball100_localtraj.json"
    ball_overlay = ball_base / "ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"
    if ball_overlay.is_file():
        details = [f"Track JSON: {ball_track}"] if ball_track.is_file() else []
        artifacts.append(
            {
                "path": str(ball_overlay),
                "clip": clip,
                "artifact_type": "racketsport_ball_track_overlay",
                "status": "rendered",
                "qualitative_status": "strict_no_click_review_track",
                "watch_paths": [str(ball_overlay)],
                "details": details,
                "available_layers": ["ball"],
                "warnings": [],
            }
        )

    ball_review_html = root / "ball_click_review_30" / clip / "review.html"
    if ball_review_html.is_file():
        artifacts.append(
            {
                "path": str(ball_review_html),
                "clip": clip,
                "artifact_type": "racketsport_ball_click_review_html",
                "status": "ready",
                "qualitative_status": "held_out_human_benchmark_review",
                "watch_paths": [str(ball_review_html)],
                "details": ["Held-out benchmark labels only; trackers must not read ball_points.json."],
                "available_layers": ["ball"],
                "warnings": [],
            }
        )
    return artifacts


def _racket_candidate_artifacts(root: Path, clip: str) -> list[dict[str, Any]]:
    clip_root = root / clip
    candidate_path = clip_root / "racket_candidates.json"
    if not candidate_path.is_file():
        return []

    pose_path = clip_root / "racket_pose.json"
    try:
        candidates = RacketCandidates.model_validate(json.loads(candidate_path.read_text(encoding="utf-8")))
    except Exception as exc:
        return [
            {
                "path": str(candidate_path),
                "clip": clip,
                "artifact_type": "racketsport_racket_candidates",
                "status": "invalid",
                "qualitative_status": "candidate_artifact_failed_schema_validation",
                "watch_paths": [],
                "details": [f"Validation error: {exc}"],
                "available_layers": ["paddle"],
                "warnings": ["invalid_racket_candidates"],
            }
        ]

    frame_count = sum(len(player.frames) for player in candidates.players)
    sources = sorted({frame.source for player in candidates.players for frame in player.frames})
    pose_detail = f"Pose artifact: present ({pose_path})" if pose_path.is_file() else f"Pose artifact: missing ({pose_path})"
    return [
        {
            "path": str(candidate_path),
            "clip": clip,
            "artifact_type": "racketsport_racket_candidates",
            "status": "candidate_with_pose" if pose_path.is_file() else "candidate_only",
            "qualitative_status": "explicit_four_corner_candidates_not_gate_verified",
            "watch_paths": [],
            "details": [
                f"Candidate players: {len(candidates.players)}",
                f"Candidate frames: {frame_count}",
                "Candidate sources: " + (", ".join(sources) if sources else "none"),
                pose_detail,
            ],
            "available_layers": ["paddle"],
            "warnings": [] if pose_path.is_file() else ["missing_racket_pose"],
        }
    ]


def _human_next_steps(corrections_template: Path) -> list[str]:
    return [
        "For the fastest browser UI, run `python scripts/racketsport/review_input_server.py --port 8765` and open `http://127.0.0.1:8765`.",
        "Open the listed review artifacts and inspect overlays before changing any labels.",
        f"Write corrections into `{corrections_template}` using `corrections/schema.json`.",
        f"Validate corrections with `python scripts/racketsport/validate_corrections.py {corrections_template}`.",
        f"Queue accepted corrections with `python scripts/racketsport/build_corrections_queue.py --root {corrections_template.parent} --out runs/corrections_queue/corrections_queue.json`.",
    ]


def _corrections_template(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": str(packet["packet_id"]),
        "created_at": str(packet["created_at"]),
        "description": "Human corrections generated while reviewing this packet.",
        "corrections": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


__all__ = ["build_review_packet", "review_packet_markdown", "write_review_packet"]
