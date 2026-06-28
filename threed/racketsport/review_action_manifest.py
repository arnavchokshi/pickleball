"""Review action manifest extracted from human-review packets."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_review_action_manifest"
COURT_REVIEW_RETIRED_CLIPS = {"burlington_gold_0300_low_steep_corner"}


def build_review_action_manifest(packet: Mapping[str, Any], *, packet_path: str | Path | None = None) -> dict[str, Any]:
    """Build a compact list of human actions from a review packet."""

    actions: list[dict[str, Any]] = []
    artifacts = [artifact for artifact in packet.get("review_artifacts", []) if isinstance(artifact, Mapping)]
    world_review_paths = _world_review_paths_by_clip(artifacts)
    clips_with_promoted_contact_windows = _clips_with_promoted_contact_windows(artifacts)
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        action = _action_for_artifact(
            artifact,
            world_review_paths=world_review_paths,
            clips_with_promoted_contact_windows=clips_with_promoted_contact_windows,
        )
        if action is not None:
            actions.append(action)

    actions.sort(key=lambda item: (_priority_rank(str(item["priority"])), str(item["clip"]), str(item["category"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "packet_id": str(packet.get("packet_id", "review_packet")),
        "run_root": str(packet.get("run_root", "")),
        "packet_path": str(packet_path) if packet_path is not None else "",
        "summary": _summary(actions),
        "actions": actions,
    }


def write_review_action_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_review_action_manifest_html(path: str | Path, html: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


def review_action_manifest_html(manifest: Mapping[str, Any], *, base_dir: str | Path | None = None) -> str:
    """Render a dependency-free reviewer dashboard."""

    packet_id = str(manifest.get("packet_id", "review_packet"))
    base_path = Path(base_dir) if base_dir is not None else Path.cwd()
    actions = [action for action in manifest.get("actions", []) if isinstance(action, Mapping)]
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Review Actions - {escape(packet_id)}</title>",
        "<style>",
        _css(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        f"<h1>{escape(packet_id)}</h1>",
        '<p class="lede">Human review actions grouped from the packet artifacts.</p>',
        '<dl class="summary">',
        _summary_item("Actions", str(manifest.get("summary", {}).get("action_count", 0))),
        _summary_item("Run root", str(manifest.get("run_root", ""))),
        _summary_item("Packet", str(manifest.get("packet_path", ""))),
        "</dl>",
        "</header>",
        "<section>",
        "<table>",
        "<thead><tr><th>Priority</th><th>Clip</th><th>Category</th><th>Action</th><th>Evidence</th></tr></thead>",
        "<tbody>",
    ]
    for action in actions:
        parts.append(_action_row(action, base_path))
    parts.extend(["</tbody>", "</table>", "</section>", "</main>", "</body>", "</html>", ""])
    return "\n".join(parts)


def _world_review_paths_by_clip(artifacts: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_clip: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact.get("artifact_type") != "racketsport_virtual_world_review":
            continue
        clip = str(artifact.get("clip", "unknown"))
        paths = [str(path) for path in artifact.get("watch_paths", []) if path]
        if paths:
            by_clip.setdefault(clip, []).extend(paths)
    return by_clip


def _clips_with_promoted_contact_windows(artifacts: list[Mapping[str, Any]]) -> set[str]:
    clips: set[str] = set()
    for artifact in artifacts:
        if artifact.get("artifact_type") != "racketsport_contact_windows":
            continue
        if artifact.get("status") != "promoted":
            continue
        if artifact.get("warnings"):
            continue
        clips.add(str(artifact.get("clip", "unknown")))
    return clips


def _action_for_artifact(
    artifact: Mapping[str, Any],
    *,
    world_review_paths: Mapping[str, list[str]],
    clips_with_promoted_contact_windows: set[str],
) -> dict[str, Any] | None:
    artifact_type = str(artifact.get("artifact_type", ""))
    warnings = [str(warning) for warning in artifact.get("warnings", []) if warning]
    details = [str(detail) for detail in artifact.get("details", []) if detail]
    clip = str(artifact.get("clip", "unknown"))
    path = str(artifact.get("path", ""))
    source_path = str(artifact.get("source_path", ""))
    watch_paths = [str(path) for path in artifact.get("watch_paths", []) if path]

    if (
        artifact_type == "racketsport_contact_window_review"
        and "pending_contact_review" in warnings
        and clip not in clips_with_promoted_contact_windows
    ):
        return _action(
            clip=clip,
            category="contact_review",
            priority="high",
            title="Contact windows need human decisions",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Open the contact review page, edit pending decisions, then promote accepted contacts.",
        )

    if artifact_type == "racketsport_racket_pose_readiness" and warnings:
        return _action(
            clip=clip,
            category="paddle_pose",
            priority="high",
            title="Paddle pose is preview-only",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Collect true paddle keypoints/corners or CAD/reference pose evidence before RKT promotion.",
        )

    if artifact_type == "racketsport_racket_promotion_audit" and warnings:
        unsafe = "box_derived_racket_pose_promoted" in warnings
        return _action(
            clip=clip,
            category="paddle_pose",
            priority="high" if unsafe else "medium",
            title="Racket pose promotion is unsafe" if unsafe else "Racket pose promotion remains blocked",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action=(
                "Quarantine canonical racket_pose.json until it is rebuilt from true paddle evidence."
                if unsafe
                else "Keep preview paddle poses separate from canonical racket_pose.json until true pose evidence and GT evaluation exist."
            ),
        )

    if artifact_type == "racketsport_contact_windows" and warnings:
        return _action(
            clip=clip,
            category="contact_cues",
            priority="high",
            title="Contact windows are not BODY-ready",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Provide machine cue artifacts or promote reviewed contact decisions before BODY scheduling.",
        )

    if artifact_type == "racketsport_ball_inflections" and warnings:
        return _action(
            clip=clip,
            category="contact_cues",
            priority="medium",
            title="Ball inflections need cue pairing",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Pair these review-only ball trajectory cues with audio and wrist-velocity cues before contact-window promotion.",
        )

    if artifact_type == "racketsport_audio_onsets" and warnings:
        blocked = artifact.get("status") == "blocked"
        return _action(
            clip=clip,
            category="contact_cues",
            priority="medium",
            title="Audio onset cues are unavailable" if blocked else "Audio onset cues need review",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action=(
                "Rebuild audio cues from an audio-bearing clip, then keep them paired with wrist and ball cues before contact promotion."
                if blocked
                else "Review the heuristic audio peaks before treating them as contact evidence."
            ),
            next_commands=_audio_onset_next_commands(clip=clip, source_path=source_path, artifact_path=path),
        )

    if artifact_type == "racketsport_wrist_velocity_peaks" and warnings:
        return _action(
            clip=clip,
            category="contact_cues",
            priority="medium",
            title="Wrist-velocity cues are unavailable",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Rebuild wrist cues from semantic wrist joints or explicit wrist joint indices before contact promotion.",
        )

    if artifact_type == "racketsport_pipeline_artifact_readiness" and warnings:
        return _action(
            clip=clip,
            category="pipeline_readiness",
            priority="medium",
            title="E2E artifact readiness is incomplete",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Use the missing artifact list to pick the next runtime or review stage; do not treat this as an accuracy gate.",
        )

    if artifact_type == "racketsport_racket_model_runtime_readiness" and warnings:
        return _action(
            clip=clip,
            category="paddle_runtime",
            priority="medium",
            title="Paddle model/runtime readiness is blocked",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Use the blocker list to add manifest entries, approved assets, and runtime probes before any paddle 6DoF GPU smoke.",
        )

    if artifact_type == "racketsport_court_line_evidence" and warnings and clip not in COURT_REVIEW_RETIRED_CLIPS:
        return _action(
            clip=clip,
            category="court_evidence",
            priority="medium",
            title="Court-line evidence is not auto-calibration-ready",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Review the missing line/net cues before treating automatic court evidence as trusted.",
        )

    if artifact_type == "racketsport_body_mesh_readiness" and warnings:
        no_world_mesh_requested = "world_mesh_not_requested_by_current_frame_plan" in warnings
        missing_mesh = "missing_mesh_vertices" in warnings or "joints_only_no_mesh_vertices" in warnings
        if no_world_mesh_requested:
            title = "BODY frame plan has no world-mesh requests"
            suggested_action = (
                "Resolve contact-window and player-coverage blockers before running more BODY mesh."
            )
        elif missing_mesh:
            title = "BODY mesh vertices are missing"
            suggested_action = "Run the scheduled BODY stage before treating joints-only previews as mesh."
        else:
            title = "BODY mesh is not accuracy-verified"
            suggested_action = (
                "Keep the BODY mesh review-only until world-MPJPE and full-clip BODY gates pass."
            )
        return _action(
            clip=clip,
            category="body_mesh",
            priority="medium",
            title=title,
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action=suggested_action,
        )

    body_schedule_warning_keys = {
        "targeted_reviewed_contact_body_schedule",
        "scheduled_with_incomplete_player_coverage",
    }
    if artifact_type == "racketsport_body_compute_execution" and body_schedule_warning_keys.intersection(warnings):
        return _action(
            clip=clip,
            category="body_schedule",
            priority="medium",
            title="BODY schedule uses targeted reviewed contacts",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=warnings,
            suggested_action="Verify the reviewed contact/player assignment and incomplete coverage before running BODY.",
        )

    if artifact_type == "racketsport_body_compute_execution" and _detail_has_prefix(details, "Scheduled frames: 0"):
        blockers = _body_schedule_blockers(details=details, artifact_path=path)
        return _action(
            clip=clip,
            category="body_schedule",
            priority="medium",
            title="BODY has zero scheduled deep-mesh frames",
            artifact_path=path,
            watch_paths=watch_paths,
            details=details,
            blockers=blockers,
            suggested_action="Resolve contact-window review and player-coverage blockers, then rebuild frame/body compute manifests.",
        )

    if artifact_type == "racketsport_virtual_world":
        world_warnings = _world_warnings(details)
        if world_warnings:
            resolved_watch_paths = watch_paths or world_review_paths.get(clip, [])
            return _action(
                clip=clip,
                category="world_review",
                priority="medium",
                title="Virtual world has review warnings",
                artifact_path=path,
                watch_paths=resolved_watch_paths,
                details=details,
                blockers=world_warnings,
                suggested_action="Open the virtual-world review page and check whether missing mesh/paddle warnings match the source artifacts.",
            )

    return None


def _action(
    *,
    clip: str,
    category: str,
    priority: str,
    title: str,
    artifact_path: str,
    watch_paths: list[str],
    details: list[str],
    blockers: list[str],
    suggested_action: str,
    next_commands: list[str] | None = None,
) -> dict[str, Any]:
    editable_paths = _editable_paths(category=category, artifact_path=artifact_path)
    resolved_next_commands = next_commands if next_commands is not None else _next_commands(category=category, clip=clip, artifact_path=artifact_path)
    return {
        "id": f"{clip}:{category}:{Path(artifact_path).stem or 'artifact'}",
        "clip": clip,
        "category": category,
        "priority": priority,
        "title": title,
        "artifact_path": artifact_path,
        "watch_paths": watch_paths,
        "editable_paths": editable_paths,
        "details": details,
        "blockers": blockers,
        "next_commands": resolved_next_commands,
        "suggested_action": suggested_action,
    }


def _audio_onset_next_commands(*, clip: str, source_path: str, artifact_path: str) -> list[str]:
    path = Path(artifact_path)
    audio_input = Path(source_path) if source_path else path.parent / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
    frame_rate = _clip_frame_rate(path.parent)
    frame_rate_arg = f" --frame-rate {frame_rate:g}" if frame_rate is not None else ""
    return [
        f"python scripts/racketsport/build_audio_onsets.py --input {audio_input} --out {artifact_path} --clip {clip} --start-s 0 --duration-s 10 --analysis-sample-rate-hz 16000{frame_rate_arg}",
        _contact_cue_fusion_command(path.parent),
    ]


def _editable_paths(*, category: str, artifact_path: str) -> list[str]:
    path = Path(artifact_path)
    if category == "contact_review":
        return [artifact_path]
    if category == "paddle_pose":
        return [str(path.with_name("racket_candidates.json"))]
    return []


def _next_commands(*, category: str, clip: str, artifact_path: str) -> list[str]:
    path = Path(artifact_path)
    clip_dir = path.parent
    if category == "contact_review":
        candidates = clip_dir / "contact_window_candidates.json"
        review = clip_dir / "contact_window_review.json"
        contact_windows = clip_dir / "contact_windows.json"
        review_html = clip_dir / "contact_window_review.html"
        return [
            f"python scripts/racketsport/apply_review_inputs_to_contact_review.py --candidates {candidates} --review {review} --review-input runs/review_inputs/pickleball_cv_review_latest.json --clip {clip} --out-review {review}",
            f"python scripts/racketsport/promote_contact_windows.py --candidates {candidates} --review {review} --out-contact-windows {contact_windows}",
            f"python scripts/racketsport/render_contact_window_review.py --candidates {candidates} --review {review} --out-html {review_html}",
        ]
    if category == "contact_cues" and path.name == "contact_windows.json":
        candidates = clip_dir / "contact_window_candidates.json"
        review = clip_dir / "contact_window_review.json"
        contact_windows = clip_dir / "contact_windows.json"
        review_html = clip_dir / "contact_window_review.html"
        return [
            _contact_cue_fusion_command(clip_dir),
            f"python scripts/racketsport/apply_review_inputs_to_contact_review.py --candidates {candidates} --review {review} --review-input runs/review_inputs/pickleball_cv_review_latest.json --clip {clip} --out-review {review}",
            f"python scripts/racketsport/promote_contact_windows.py --candidates {candidates} --review {review} --out-contact-windows {contact_windows}",
            f"python scripts/racketsport/render_contact_window_review.py --candidates {candidates} --review {review} --out-html {review_html}",
        ]
    if category == "paddle_pose":
        candidates = clip_dir / "racket_candidates.json"
        preview = clip_dir / "racket_pose_preview.json"
        if path.name == "racket_promotion_audit.json":
            pose = clip_dir / "racket_pose.json"
            return [
                f"python scripts/racketsport/build_racket_promotion_audit.py --clip {clip} --racket-candidates {candidates} --racket-pose-preview {preview} --racket-pose {pose} --out {artifact_path}"
            ]
        readiness = clip_dir / "racket_pose_readiness.json"
        return [
            f"python scripts/racketsport/build_racket_pose_readiness.py --clip {clip} --racket-candidates {candidates} --racket-pose-preview {preview} --out {readiness}"
        ]
    if category == "body_schedule":
        tracks = clip_dir / "tracks.json"
        ball_track = clip_dir / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json"
        contact_windows = clip_dir / "contact_windows.json"
        frame_plan = clip_dir / "frame_compute_plan.json"
        execution = clip_dir / "body_compute_execution.json"
        return [
            f"python scripts/racketsport/build_frame_compute_plan.py --tracks {tracks} --ball-track {ball_track} --contact-windows {contact_windows} --expected-players 4 --out {frame_plan}",
            f"python scripts/racketsport/build_body_compute_execution.py --tracks {tracks} --frame-compute-plan {frame_plan} --out {execution}",
        ]
    if category == "body_mesh":
        smpl_motion = clip_dir / "smpl_motion.json"
        skeleton = clip_dir / "skeleton3d.json"
        frame_plan = clip_dir / "frame_compute_plan.json"
        execution = clip_dir / "body_compute_execution.json"
        readiness = clip_dir / "body_mesh_readiness.json"
        return [
            f"python scripts/racketsport/build_body_mesh_readiness.py --clip {clip} --smpl-motion {smpl_motion} --skeleton3d {skeleton} --frame-compute-plan {frame_plan} --body-compute-execution {execution} --out {readiness}"
        ]
    if category == "contact_cues" and path.name == "ball_inflections.json":
        virtual_world = clip_dir / "virtual_world.json"
        return [
            f"python scripts/racketsport/build_ball_inflections.py --virtual-world {virtual_world} --out {artifact_path}",
            _contact_cue_fusion_command(clip_dir),
        ]
    if category == "contact_cues" and path.name == "audio_onsets.json":
        video = clip_dir / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
        frame_rate = _clip_frame_rate(clip_dir)
        frame_rate_arg = f" --frame-rate {frame_rate:g}" if frame_rate is not None else ""
        return [
            f"python scripts/racketsport/build_audio_onsets.py --input {video} --out {artifact_path} --clip {clip} --start-s 0 --duration-s 10 --analysis-sample-rate-hz 16000{frame_rate_arg}",
            _contact_cue_fusion_command(clip_dir),
        ]
    if category == "contact_cues" and path.name == "wrist_velocity_peaks.json":
        skeleton = clip_dir / "skeleton3d.json"
        return [
            f"python scripts/racketsport/build_wrist_velocity_peaks.py --skeleton3d {skeleton} --out {artifact_path} --allow-missing",
            _contact_cue_fusion_command(clip_dir),
        ]
    if category == "pipeline_readiness":
        readiness = clip_dir / "pipeline_readiness_e2e.json"
        return [
            f"python scripts/racketsport/validate_pipeline_artifacts.py --run-dir {clip_dir} --stage e2e --out {readiness} || true"
        ]
    if category == "paddle_runtime":
        return [
            f"python scripts/racketsport/build_racket_model_runtime_readiness.py --manifest models/MANIFEST.json --out {artifact_path}"
        ]
    if category == "court_evidence":
        calibration = clip_dir / "court_calibration.json"
        net_plane = clip_dir / "net_plane.json"
        video = clip_dir / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
        return [
            f"python scripts/racketsport/build_court_line_evidence.py --calibration {calibration} --net-plane {net_plane} --video {video} --out {artifact_path}"
        ]
    if category == "world_review":
        virtual_world = path
        review_html = clip_dir / "virtual_world_paddle_preview.html"
        review_index = clip_dir / "virtual_world_review_index.json"
        return [
            f"python scripts/racketsport/build_virtual_world_review.py --virtual-world {virtual_world} --out-html {review_html} --index-out {review_index} --title '{clip} Paddle Preview World'"
        ]
    return []


def _summary(actions: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_category = _counts(str(action["category"]) for action in actions)
    by_priority = _counts(str(action["priority"]) for action in actions)
    return {
        "action_count": len(actions),
        "by_category": by_category,
        "by_priority": by_priority,
        "clips": sorted({str(action["clip"]) for action in actions}),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _contact_cue_fusion_command(clip_dir: Path) -> str:
    return (
        "python scripts/racketsport/build_contact_windows_from_cues.py"
        f" --audio-onsets {clip_dir / 'audio_onsets.json'}"
        f" --wrist-velocity-peaks {clip_dir / 'wrist_velocity_peaks.json'}"
        f" --ball-inflections {clip_dir / 'ball_inflections.json'}"
        f" --tracks {clip_dir / 'tracks.json'}"
        f" --out {clip_dir / 'contact_windows.json'}"
    )


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


def _detail_has_prefix(details: list[str], prefix: str) -> bool:
    return any(detail.startswith(prefix) for detail in details)


def _world_warnings(details: list[str]) -> list[str]:
    warnings: list[str] = []
    for detail in details:
        if detail.startswith("Warnings: "):
            warnings.extend(item.strip() for item in detail.removeprefix("Warnings: ").split(",") if item.strip())
    return sorted(set(warnings))


def _body_schedule_blockers(*, details: list[str], artifact_path: str) -> list[str]:
    blockers: list[str] = []
    contact_windows_path = Path(artifact_path).with_name("contact_windows.json")
    if artifact_path and not contact_windows_path.is_file():
        blockers.append("missing_promoted_contact_windows")
    elif artifact_path:
        contact_state = _contact_windows_state(contact_windows_path)
        if contact_state != "promoted":
            blockers.append(contact_state)
    if _detail_has_prefix(details, "Scheduled player-frames: 0"):
        blockers.append("no_world_mesh_player_targets")
    skipped_tiers = _detail_value(details, "Skipped tiers: ")
    if "human_review=" in skipped_tiers:
        blockers.append("player_coverage_human_review")
    return blockers


def _contact_windows_state(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_contact_windows"
    events = payload.get("events") if isinstance(payload, Mapping) else None
    if not isinstance(events, list):
        return "invalid_contact_windows"
    return "promoted" if events else "empty_contact_windows_no_deep_mesh"


def _clip_frame_rate(clip_dir: Path) -> float | None:
    for filename in ("tracks.json", "ball_track.json"):
        path = clip_dir / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fps = payload.get("fps") if isinstance(payload, Mapping) else None
        if isinstance(fps, bool) or not isinstance(fps, int | float) or fps <= 0:
            continue
        return float(fps)
    return None


def _detail_value(details: list[str], prefix: str) -> str:
    for detail in details:
        if detail.startswith(prefix):
            return detail.removeprefix(prefix)
    return ""


def _action_row(action: Mapping[str, Any], base_dir: Path) -> str:
    evidence = [_link(str(action.get("artifact_path", "")), base_dir)]
    for watch_path in action.get("watch_paths", []):
        evidence.append(_link(str(watch_path), base_dir))
    editable_paths = [str(path) for path in action.get("editable_paths", []) if path]
    next_commands = [str(command) for command in action.get("next_commands", []) if command]
    details = [escape(str(detail)) for detail in action.get("details", [])]
    blockers = [escape(str(blocker)) for blocker in action.get("blockers", [])]
    detail_html = "".join(f"<li>{detail}</li>" for detail in details)
    blocker_html = "".join(f"<li>{blocker}</li>" for blocker in blockers)
    editable_html = "".join(f"<li>{_link(path, base_dir)}</li>" for path in editable_paths)
    command_html = "".join(f"<li><code>{escape(command)}</code></li>" for command in next_commands)
    extras = ""
    if editable_html:
        extras += f'<p class="subhead">Editable files</p><ul>{editable_html}</ul>'
    if command_html:
        extras += f'<p class="subhead">Next commands</p><ul class="commands">{command_html}</ul>'
    return (
        "<tr>"
        f'<td><span class="priority {escape(str(action.get("priority", "")))}">{escape(str(action.get("priority", "")))}</span></td>'
        f"<td>{escape(str(action.get('clip', '')))}</td>"
        f"<td>{escape(str(action.get('category', '')))}</td>"
        "<td>"
        f"<strong>{escape(str(action.get('title', '')))}</strong>"
        f"<p>{escape(str(action.get('suggested_action', '')))}</p>"
        f"<ul>{detail_html}{blocker_html}</ul>"
        f"{extras}"
        "</td>"
        f"<td>{''.join(evidence)}</td>"
        "</tr>"
    )


def _link(path: str, base_dir: Path) -> str:
    if not path:
        return ""
    target = Path(path)
    href = path
    if not target.is_absolute() and (Path.cwd() / target).exists():
        href = Path(os_relpath(Path.cwd() / target, base_dir)).as_posix()
    elif target.is_absolute() and target.exists():
        href = Path(os_relpath(target, base_dir)).as_posix()
    return f'<a href="{escape(href, quote=True)}">{escape(Path(path).name or path)}</a>'


def os_relpath(path: Path, base_dir: Path) -> str:
    import os

    return os.path.relpath(path, base_dir)


def _summary_item(label: str, value: str) -> str:
    return f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"


def _css() -> str:
    return """
:root {
  color-scheme: light;
  font-family: Avenir Next, ui-sans-serif, system-ui, sans-serif;
  color: #1f2933;
  background: #f5f3ee;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
}
main {
  margin: 0 auto;
  padding: 24px 16px 44px;
  width: min(1280px, 100%);
}
h1, p {
  margin-top: 0;
}
h1 {
  font-size: 28px;
  line-height: 1.15;
  margin-bottom: 6px;
}
.lede {
  color: #5f6b7a;
}
.summary {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 18px 0;
}
.summary div {
  background: #fff;
  border: 1px solid #d8d3c8;
  border-radius: 6px;
  padding: 10px 12px;
}
dt {
  color: #667085;
  font-size: 11px;
  text-transform: uppercase;
}
dd {
  margin: 4px 0 0;
  overflow-wrap: anywhere;
}
section {
  max-width: 100%;
  overflow-x: auto;
}
table {
  background: #fff;
  border: 1px solid #d8d3c8;
  border-collapse: collapse;
  min-width: 1100px;
  width: 100%;
}
th, td {
  border-bottom: 1px solid #e7e2d8;
  padding: 11px 12px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #202124;
  color: #fff;
  font-size: 12px;
}
td {
  font-size: 13px;
}
td p {
  color: #4b5563;
  margin: 6px 0 0;
}
ul {
  margin: 8px 0 0;
  padding-left: 18px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: normal;
}
a {
  color: #0f4c81;
  display: block;
  margin-bottom: 6px;
  overflow-wrap: anywhere;
}
.subhead {
  color: #202124;
  font-size: 12px;
  font-weight: 700;
  margin: 12px 0 0;
}
.commands li {
  margin-bottom: 6px;
  overflow-wrap: anywhere;
}
.priority {
  border-radius: 999px;
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  min-width: 56px;
  padding: 4px 8px;
  text-align: center;
}
.priority.high {
  background: #fee2e2;
  color: #991b1b;
}
.priority.medium {
  background: #fef3c7;
  color: #92400e;
}
@media (max-width: 700px) {
  main {
    padding-inline: 10px;
  }
  h1 {
    font-size: 22px;
  }
}
""".strip()


__all__ = [
    "ARTIFACT_TYPE",
    "build_review_action_manifest",
    "review_action_manifest_html",
    "write_review_action_manifest",
    "write_review_action_manifest_html",
]
