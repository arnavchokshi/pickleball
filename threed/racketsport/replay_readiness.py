"""Fail-closed readiness reports for review replay artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.replay_export import audit_replay_export_manifest, inspect_glb_file, resolve_replay_glb_path
from threed.racketsport.schemas import ReplayScene, VirtualWorld, validate_artifact_file
from threed.racketsport.testclips import REQUIRED_LABEL_FILES


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_replay_readiness_report"
DEFAULT_WORLD_ARTIFACT = "virtual_world_paddle_preview.json"
DEFAULT_WORLD_HTML = "virtual_world_paddle_preview.html"
DEFAULT_REPLAY_SCENE = "replay_scene.json"


def build_replay_readiness_report(
    *,
    run_root: str | Path,
    clips: Sequence[str] | None = None,
    labels_root: str | Path | None = Path("data/testclips"),
) -> dict[str, Any]:
    """Build a fail-closed report from real run artifacts.

    ``review_visual_ready`` means inspectable JSON/HTML/GLB outputs exist.
    ``production_replay_ready`` remains false when any source is preview-only,
    approximate, missing, or not backed by accuracy labels.
    """

    root = Path(run_root)
    clip_names = list(clips) if clips is not None else _discover_clips(root)
    label_root_path = Path(labels_root) if labels_root is not None else None
    clip_reports = [_build_clip_report(root / clip, clip=clip, labels_root=label_root_path) for clip in clip_names]
    summary = {
        "clip_count": len(clip_reports),
        "review_visual_ready_clips": sum(1 for clip in clip_reports if clip["review_visual_ready"]),
        "production_replay_ready_clips": sum(1 for clip in clip_reports if clip["production_replay_ready"]),
        "metrics_gate_ready_clips": sum(1 for clip in clip_reports if clip["metrics_gate_ready"]),
        "blocked_clips": sum(1 for clip in clip_reports if clip["status"] == "blocked"),
        "missing_artifact_clips": sum(1 for clip in clip_reports if clip["status"] == "missing_artifacts"),
    }
    status = "pass" if clip_reports and all(clip["production_replay_ready"] and clip["metrics_gate_ready"] for clip in clip_reports) else "blocked"
    if not clip_reports:
        status = "not_measured"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "run_root": str(root),
        "labels_root": str(label_root_path) if label_root_path is not None else None,
        "summary": summary,
        "clips": clip_reports,
    }


def write_replay_readiness_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_replay_readiness_html(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_replay_readiness_html(payload), encoding="utf-8")


def build_replay_readiness_html(payload: Mapping[str, Any]) -> str:
    rows = "\n".join(_clip_section(clip) for clip in payload.get("clips", []) if isinstance(clip, Mapping))
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Replay Readiness</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #11130f; color: #f2f1e8; }}
body {{ margin: 0; }}
main {{ margin: 0 auto; max-width: 1180px; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 16px; }}
h2 {{ font-size: 18px; margin: 0; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
.pill {{ border: 1px solid rgba(242,241,232,0.22); border-radius: 8px; padding: 8px 10px; }}
.clip {{ border-top: 1px solid rgba(242,241,232,0.18); display: grid; gap: 16px; grid-template-columns: 270px 1fr; padding: 18px 0; }}
.viz {{ background: #18231d; border: 1px solid rgba(242,241,232,0.16); border-radius: 8px; min-height: 210px; }}
dl {{ display: grid; gap: 8px 14px; grid-template-columns: 180px 1fr; margin: 12px 0 0; }}
dt {{ color: #aaa48f; }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
.blocked {{ color: #ffbf69; }}
.pass {{ color: #8fe3b0; }}
a {{ color: #88ccff; }}
@media (max-width: 760px) {{ .clip {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
<h1>Replay Readiness</h1>
<div class="summary">
  <div class="pill">Status: {escape(str(payload.get("status", "not_measured")))}</div>
  <div class="pill">Review visuals: {escape(str(summary.get("review_visual_ready_clips", 0)))}</div>
  <div class="pill">Production replay ready: {escape(str(summary.get("production_replay_ready_clips", 0)))}</div>
  <div class="pill">Metrics gate ready: {escape(str(summary.get("metrics_gate_ready_clips", 0)))}</div>
</div>
{rows}
</main>
</body>
</html>
"""


def _build_clip_report(run_dir: Path, *, clip: str, labels_root: Path | None) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    world_path = run_dir / DEFAULT_WORLD_ARTIFACT
    world: dict[str, Any] | None = None
    if world_path.is_file():
        try:
            parsed = validate_artifact_file("virtual_world", world_path)
            if not isinstance(parsed, VirtualWorld):
                blockers.append("invalid_virtual_world")
            else:
                world = parsed.model_dump(mode="json")
        except Exception as exc:
            blockers.append("invalid_virtual_world")
            warnings.append(f"virtual world validation failed: {exc}")
    else:
        blockers.append("missing_virtual_world")

    summary = world.get("summary", {}) if isinstance(world, Mapping) else {}
    if int(summary.get("mesh_player_frame_count") or 0) <= 0:
        blockers.append("missing_body_mesh")
    if int(summary.get("approx_ball_frame_count") or 0) > 0:
        blockers.append("approximate_ball_projection")
    if int(summary.get("ambiguous_paddle_frame_count") or 0) > 0:
        blockers.append("ambiguous_paddle_pose")
    for warning in summary.get("warnings", []) if isinstance(summary.get("warnings", []), list) else []:
        warning_text = str(warning)
        if warning_text not in blockers:
            blockers.append(warning_text)

    body_status = _artifact_status(run_dir / "body_mesh_readiness.json")
    racket_status = _artifact_status(run_dir / "racket_pose_readiness.json")
    contact_status = _artifact_status(run_dir / "contact_window_review.json")
    if body_status is None:
        blockers.append("missing_body_mesh_readiness")
    elif body_status == "missing_body_output":
        blockers.append("missing_body_mesh")
    elif body_status != "gate_verified":
        blockers.append("body_mesh_needs_accuracy_gate")
    if racket_status is None:
        blockers.append("missing_racket_pose_readiness")
    elif racket_status != "gate_verified":
        blockers.append("paddle_pose_preview_only")

    glb_report = _glb_report(run_dir)
    blockers.extend(glb_report["blockers"])
    if not (run_dir / DEFAULT_WORLD_HTML).is_file():
        blockers.append("missing_virtual_world_html")

    label_blockers = _label_blockers(labels_root, clip)
    blockers.extend(label_blockers)
    blockers = sorted(set(blockers))
    review_visual_ready = (
        world is not None
        and (run_dir / DEFAULT_WORLD_HTML).is_file()
        and glb_report["valid_glb_count"] == glb_report["expected_glb_count"]
        and glb_report["expected_glb_count"] > 0
    )
    production_blockers = [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "missing_labels_root",
            "missing_label_clip",
        }
        and not blocker.startswith("missing_label_files:")
    ]
    production_replay_ready = review_visual_ready and not production_blockers
    metrics_gate_ready = production_replay_ready and not label_blockers
    visual_outputs = {
        "virtual_world_json": str(world_path) if world_path.is_file() else None,
        "virtual_world_html": str(run_dir / DEFAULT_WORLD_HTML) if (run_dir / DEFAULT_WORLD_HTML).is_file() else None,
        "replay_scene": str(run_dir / DEFAULT_REPLAY_SCENE) if (run_dir / DEFAULT_REPLAY_SCENE).is_file() else None,
        "court_glb": glb_report.get("court_glb_path"),
        "point_glbs": glb_report.get("point_glb_paths", []),
    }
    missing_artifact = any(blocker.startswith("missing_") for blocker in blockers if blocker not in {"missing_labels_root", "missing_label_clip", "missing_body_mesh"})
    return {
        "clip": clip,
        "run_dir": str(run_dir),
        "status": "missing_artifacts" if missing_artifact and not review_visual_ready else ("pass" if production_replay_ready and metrics_gate_ready else "blocked"),
        "review_visual_ready": review_visual_ready,
        "production_replay_ready": production_replay_ready,
        "metrics_gate_ready": metrics_gate_ready,
        "truth_status": {
            "body": body_status or "missing",
            "paddle_pose": racket_status or "missing",
            "contact": contact_status or "missing",
            "body_mesh_real_but_unverified": body_status == "mesh_available_needs_accuracy_gate",
            "paddle_pose_preview_only": racket_status != "gate_verified",
        },
        "counts": {
            "players": int(summary.get("player_count") or 0),
            "mesh_players": int(summary.get("mesh_player_count") or 0),
            "mesh_player_frames": int(summary.get("mesh_player_frame_count") or 0),
            "ball_frames": int(summary.get("ball_frame_count") or 0),
            "approx_ball_frames": int(summary.get("approx_ball_frame_count") or 0),
            "paddle_frames": int(summary.get("paddle_frame_count") or 0),
            "ambiguous_paddle_frames": int(summary.get("ambiguous_paddle_frame_count") or 0),
        },
        "visual_outputs": visual_outputs,
        "glb_report": glb_report,
        "blockers": blockers,
        "notes": warnings,
        "preview_svg": _clip_svg(world),
    }


def _discover_clips(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / DEFAULT_WORLD_ARTIFACT).is_file()
    ]


def _artifact_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid_json"
    if not isinstance(payload, Mapping):
        return "invalid_json"
    status = payload.get("status")
    return str(status) if status is not None else "missing_status"


def _glb_report(run_dir: Path) -> dict[str, Any]:
    scene_path = run_dir / DEFAULT_REPLAY_SCENE
    blockers: list[str] = []
    if not scene_path.is_file():
        return {
            "expected_glb_count": 0,
            "valid_glb_count": 0,
            "missing_glbs": [],
            "invalid_glbs": [],
            "largest_point_glb_mb": 0.0,
            "court_glb_path": None,
            "point_glb_paths": [],
            "artifact_class": "missing",
            "production_requirements": {},
            "blockers": ["missing_replay_scene"],
        }
    try:
        parsed = validate_artifact_file("replay_scene", scene_path)
    except Exception as exc:
        return {
            "expected_glb_count": 0,
            "valid_glb_count": 0,
            "missing_glbs": [],
            "invalid_glbs": [],
            "largest_point_glb_mb": 0.0,
            "court_glb_path": None,
            "point_glb_paths": [],
            "artifact_class": "invalid",
            "production_requirements": {},
            "blockers": [f"invalid_replay_scene:{exc}"],
        }
    if not isinstance(parsed, ReplayScene):
        return {
            "expected_glb_count": 0,
            "valid_glb_count": 0,
            "missing_glbs": [],
            "invalid_glbs": [],
            "largest_point_glb_mb": 0.0,
            "court_glb_path": None,
            "point_glb_paths": [],
            "artifact_class": "invalid",
            "production_requirements": {},
            "blockers": ["invalid_replay_scene"],
        }

    refs = [("court_glb", parsed.court_glb), *[(f"points/{index}/glb_url", point.glb_url) for index, point in enumerate(parsed.points)]]
    missing: list[str] = []
    invalid: list[str] = []
    valid = 0
    for field, ref in refs:
        try:
            path = resolve_replay_glb_path(run_dir, ref, field=field)
        except FileNotFoundError:
            missing.append(ref)
            continue
        except ValueError:
            invalid.append(ref)
            continue
        try:
            inspect_glb_file(path)
            valid += 1
        except ValueError:
            invalid.append(ref)
    if missing:
        blockers.append("missing_referenced_glb")
    if invalid:
        blockers.append("invalid_referenced_glb")
        blockers.append("invalid_replay_scene_glb_ref")
    if not parsed.players:
        blockers.append("missing_replay_players")
    if not parsed.points:
        blockers.append("missing_replay_points")
    production_audit = audit_replay_export_manifest(run_dir, parsed) if not missing and not invalid else {
        "artifact_class": "invalid" if invalid else "missing",
        "blockers": [],
        "production_requirements": {},
    }
    blockers.extend(production_audit["blockers"])
    return {
        "expected_glb_count": len(refs),
        "valid_glb_count": valid,
        "missing_glbs": missing,
        "invalid_glbs": invalid,
        "largest_point_glb_mb": max((point.size_mb for point in parsed.points), default=0.0),
        "court_glb_path": str(run_dir / parsed.court_glb),
        "point_glb_paths": [str(run_dir / point.glb_url) for point in parsed.points],
        "artifact_class": production_audit["artifact_class"],
        "production_requirements": production_audit["production_requirements"],
        "blockers": sorted(set(blockers)),
    }


def _label_blockers(labels_root: Path | None, clip: str) -> list[str]:
    if labels_root is None:
        return []
    if not labels_root.exists():
        return ["missing_labels_root"]
    labels_dir = labels_root / clip / "labels"
    if not labels_dir.exists():
        return ["missing_label_clip"]
    missing = [label for label in REQUIRED_LABEL_FILES if not (labels_dir / label).is_file()]
    return [f"missing_label_files:{','.join(missing)}"] if missing else []


def _clip_section(clip: Mapping[str, Any]) -> str:
    blockers = ", ".join(str(blocker) for blocker in clip.get("blockers", [])) or "none"
    counts = clip.get("counts", {}) if isinstance(clip.get("counts"), Mapping) else {}
    visual_outputs = clip.get("visual_outputs", {}) if isinstance(clip.get("visual_outputs"), Mapping) else {}
    status_class = "pass" if clip.get("production_replay_ready") and clip.get("metrics_gate_ready") else "blocked"
    return f"""<section class="clip">
  <div class="viz">{clip.get("preview_svg", "")}</div>
  <div>
    <h2>{escape(str(clip.get("clip", "")))}</h2>
    <dl>
      <dt>Status</dt><dd class="{status_class}">{escape(str(clip.get("status", "")))}</dd>
      <dt>Review visual ready</dt><dd>{escape(str(clip.get("review_visual_ready", False)))}</dd>
      <dt>Production replay ready</dt><dd>{escape(str(clip.get("production_replay_ready", False)))}</dd>
      <dt>Metrics gate ready</dt><dd>{escape(str(clip.get("metrics_gate_ready", False)))}</dd>
      <dt>Mesh frames</dt><dd>{escape(str(counts.get("mesh_player_frames", 0)))}</dd>
      <dt>Ball frames</dt><dd>{escape(str(counts.get("ball_frames", 0)))} ({escape(str(counts.get("approx_ball_frames", 0)))} approximate)</dd>
      <dt>Paddle frames</dt><dd>{escape(str(counts.get("paddle_frames", 0)))} ({escape(str(counts.get("ambiguous_paddle_frames", 0)))} ambiguous)</dd>
      <dt>HTML</dt><dd>{_link(visual_outputs.get("virtual_world_html"))}</dd>
      <dt>Replay scene</dt><dd>{_link(visual_outputs.get("replay_scene"))}</dd>
      <dt>Blockers</dt><dd>{escape(blockers)}</dd>
    </dl>
  </div>
</section>"""


def _link(path: Any) -> str:
    if not path:
        return "missing"
    text = escape(str(path))
    return f'<a href="{text}">{text}</a>'


def _clip_svg(world: Mapping[str, Any] | None) -> str:
    if not isinstance(world, Mapping):
        return "<svg viewBox='0 0 270 210' width='100%' height='210' role='img'><text x='20' y='105' fill='#f2f1e8'>No world payload</text></svg>"
    court = world.get("court", {}) if isinstance(world.get("court"), Mapping) else {}
    width = float(court.get("width_m") or 6.1)
    length = float(court.get("length_m") or 13.41)
    points = _sample_world_points(world)

    def tx(x: float) -> float:
        return 18 + ((x + width / 2.0) / max(width, 0.01)) * 234

    def ty(y: float) -> float:
        return 12 + ((length / 2.0 - y) / max(length, 0.01)) * 186

    lines: list[str] = ["<rect x='18' y='12' width='234' height='186' fill='#24392e' stroke='#ded7bd' stroke-width='1.5'/>"]
    segments = court.get("line_segments", {})
    if isinstance(segments, Mapping):
        for segment in segments.values():
            if isinstance(segment, list) and len(segment) >= 2:
                a = _xy(segment[0])
                b = _xy(segment[1])
                if a and b:
                    lines.append(
                        f"<line x1='{tx(a[0]):.1f}' y1='{ty(a[1]):.1f}' x2='{tx(b[0]):.1f}' y2='{ty(b[1]):.1f}' stroke='#ded7bd' stroke-width='1'/>"
                    )
    for x, y in points["mesh"]:
        lines.append(f"<circle cx='{tx(x):.1f}' cy='{ty(y):.1f}' r='1.5' fill='#8fb7ff' opacity='0.55'/>")
    for x, y in points["ball"]:
        lines.append(f"<circle cx='{tx(x):.1f}' cy='{ty(y):.1f}' r='2.2' fill='#f4d35e' opacity='0.85'/>")
    for x, y in points["paddle"]:
        lines.append(f"<rect x='{tx(x)-2:.1f}' y='{ty(y)-2:.1f}' width='4' height='4' fill='#ffb14a' opacity='0.9'/>")
    legend = "<text x='18' y='207' fill='#aaa48f' font-size='10'>blue=BODY mesh sample, yellow=ball, orange=paddle preview</text>"
    return f"<svg viewBox='0 0 270 216' width='100%' height='216' role='img'>{''.join(lines)}{legend}</svg>"


def _sample_world_points(world: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    mesh: list[tuple[float, float]] = []
    for player in world.get("players", []) if isinstance(world.get("players", []), list) else []:
        frames = player.get("frames", []) if isinstance(player, Mapping) else []
        for frame in frames[:: max(1, len(frames) // 12 or 1)] if isinstance(frames, list) else []:
            for point in frame.get("mesh_vertices_world", [])[:: max(1, len(frame.get("mesh_vertices_world", [])) // 18 or 1)]:
                xy = _xy(point)
                if xy:
                    mesh.append(xy)
    ball = [
        xy
        for xy in (_xy(frame.get("world_xyz")) for frame in (world.get("ball", {}).get("frames", []) if isinstance(world.get("ball"), Mapping) else []))
        if xy
    ][::20]
    paddle: list[tuple[float, float]] = []
    for item in world.get("paddles", []) if isinstance(world.get("paddles", []), list) else []:
        frames = item.get("frames", []) if isinstance(item, Mapping) else []
        for frame in frames[:: max(1, len(frames) // 20 or 1)] if isinstance(frames, list) else []:
            vertices = [_xy(vertex) for vertex in frame.get("mesh_vertices_world", [])]
            vertices = [vertex for vertex in vertices if vertex]
            if vertices:
                paddle.append((sum(x for x, _ in vertices) / len(vertices), sum(y for _, y in vertices) / len(vertices)))
    return {"mesh": mesh[:220], "ball": ball[:80], "paddle": paddle[:80]}


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


__all__ = [
    "ARTIFACT_TYPE",
    "build_replay_readiness_html",
    "build_replay_readiness_report",
    "write_replay_readiness_html",
    "write_replay_readiness_report",
]
