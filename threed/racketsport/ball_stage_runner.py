"""BALL StageRunner integration for the current no-click prototype track."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contact_windows import build_contact_windows_artifact
from .event_fusion import fuse_contact_windows_from_cue_payloads
from .schemas import BallTrack, ContactWindows


DEFAULT_NO_CLICK_BALL_FILENAME = "ball_track_fusion_temporal_vball100_localtraj.json"
DEFAULT_SELECTED_TRACKS_DIR = "selected_tracks"
DEFAULT_SELECTED_BALL_FILENAME = "ball_track.json"
DEFAULT_TRACKNET_SMOKE_DIR = "tracknet_smoke_0000_0010"
DEFAULT_PROTOTYPE_GATE_ROOT = Path("runs/eval0/prototype_gate_h100_v2")
DEFAULT_AUDIO_ONSETS_FILENAME = "audio_onsets.json"
DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME = "wrist_velocity_peaks.json"
DEFAULT_BALL_INFLECTIONS_FILENAME = "ball_inflections.json"


@dataclass(frozen=True)
class BallStageRun:
    stage: str
    status: str
    real_model: bool
    source_mode: str
    produced_artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "real_model": self.real_model,
            "source_mode": self.source_mode,
            "produced_artifacts": list(self.produced_artifacts),
            "notes": list(self.notes),
            "metrics": self.metrics,
        }


class BallStageRunner:
    stage = "ball_events"
    real_model = False
    source_mode = "no_click_fusion_temporal_vball100_localtraj"

    def __init__(
        self,
        *,
        source_path: str | Path | None = None,
        prototype_root: str | Path = DEFAULT_PROTOTYPE_GATE_ROOT,
    ) -> None:
        self.source_path = Path(source_path) if source_path is not None else None
        self.prototype_root = Path(prototype_root)

    def run(self, context: Any) -> BallStageRun:
        source_path = self._resolve_source_path(context)
        if source_path.name == "ball_points.json":
            raise ValueError("BALL StageRunner refuses to consume ball_points.json")

        ball_payload = _read_json(source_path)
        ball_track = BallTrack.model_validate(ball_payload)
        if ball_track.source == "tap":
            raise ValueError("BALL StageRunner refuses to consume tap/manual ball tracks")
        source_mode = _source_mode_for_path(source_path)
        selection = _selection_metadata_for_track(source_path)
        if source_mode == "selected_ball_track_prototype" and selection is None:
            raise ValueError(f"missing selected-track metadata sidecar: {source_path.parent / 'ball_track_selection.json'}")

        contact_payload, contact_notes = _contact_windows_from_cues(context, fps=ball_track.fps)
        ContactWindows.model_validate(contact_payload)

        _write_json(context.run_dir / "ball_track.json", ball_payload)
        _write_json(context.run_dir / "contact_windows.json", contact_payload)

        visible_count = sum(1 for frame in ball_track.frames if frame.visible)
        approx_count = sum(1 for frame in ball_track.frames if frame.approx)
        contact_event_count = len(contact_payload.get("events", []))
        metrics = {
            "source_ball_track": str(source_path),
            "frame_count": len(ball_track.frames),
            "visible_frame_count": visible_count,
            "invisible_frame_count": len(ball_track.frames) - visible_count,
            "approx_frame_count": approx_count,
            "bounce_count": len(ball_track.bounces),
            "contact_event_count": contact_event_count,
            "uses_human_clicks": False,
            "not_gate_verified": True,
        }
        if selection is not None:
            metrics["selection"] = selection
        status = "ran" if contact_event_count else "blocked"
        blocked_notes: tuple[str, ...] = ()
        if status == "blocked":
            blocked_notes = (
                "BALL contact windows are empty; downstream stages remain blocked until trusted cue fusion produces contacts",
            )
        return BallStageRun(
            stage=self.stage,
            status=status,
            real_model=self.real_model,
            source_mode=source_mode,
            produced_artifacts=("ball_track.json", "contact_windows.json"),
            notes=(
                _source_note_for_path(source_path),
                *contact_notes,
                *blocked_notes,
                "prototype integration only; not a BALL VERIFIED accuracy gate",
            ),
            metrics=metrics,
        )

    def _resolve_source_path(self, context: Any) -> Path:
        candidates = [self.source_path] if self.source_path is not None else _default_source_candidates(
            context,
            prototype_root=self.prototype_root,
        )
        searched: list[Path] = []
        for candidate in candidates:
            if candidate is None:
                continue
            searched.append(candidate)
            if candidate.is_file():
                return candidate

        searched_text = ", ".join(str(path) for path in searched)
        raise FileNotFoundError(
            f"missing no-click BALL source artifact: {DEFAULT_NO_CLICK_BALL_FILENAME}; "
            f"searched: {searched_text}; will not fall back to ball_points.json"
        )


def _default_source_candidates(context: Any, *, prototype_root: Path) -> list[Path]:
    filename = DEFAULT_NO_CLICK_BALL_FILENAME
    smoke_dir = DEFAULT_TRACKNET_SMOKE_DIR
    selected_dir = DEFAULT_SELECTED_TRACKS_DIR
    selected_filename = DEFAULT_SELECTED_BALL_FILENAME
    return [
        context.inputs_dir / selected_dir / context.clip / selected_filename,
        context.inputs_dir.parent / selected_dir / context.clip / selected_filename,
        context.run_dir / selected_dir / context.clip / selected_filename,
        context.run_dir.parent / selected_dir / context.clip / selected_filename,
        prototype_root / selected_dir / context.clip / selected_filename,
        context.inputs_dir / filename,
        context.inputs_dir / smoke_dir / filename,
        context.run_dir / filename,
        context.run_dir / smoke_dir / filename,
        prototype_root / context.clip / smoke_dir / filename,
    ]


def _source_mode_for_path(path: Path) -> str:
    if DEFAULT_SELECTED_TRACKS_DIR in path.parts and path.name == DEFAULT_SELECTED_BALL_FILENAME:
        return "selected_ball_track_prototype"
    return BallStageRunner.source_mode


def _source_note_for_path(path: Path) -> str:
    if _source_mode_for_path(path) == "selected_ball_track_prototype":
        return (
            "consumed eval-suite selected ball track artifact; selection may point at a composite prototype and "
            "does not prove a trained PB-MAT checkpoint"
        )
    return "consumed strict no-click TrackNet/VballNet local-trajectory prototype ball track"


def _selection_metadata_for_track(path: Path) -> dict[str, Any] | None:
    sidecar = path.parent / "ball_track_selection.json"
    if not sidecar.is_file():
        return None
    payload = _read_json(sidecar)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ball_track_selection.json payload: {sidecar}")
    if payload.get("artifact_type") != "racketsport_ball_track_selection":
        raise ValueError(f"invalid ball_track_selection.json artifact_type: {sidecar}")
    fields = (
        "status",
        "clip",
        "candidate",
        "candidate_category",
        "candidate_score",
        "candidate_rank",
        "eligible_for_model_ranking",
        "trained_pbmat_checkpoint",
        "source_ball_track",
        "out",
        "not_ground_truth",
    )
    return {field: payload[field] for field in fields if field in payload}


def _contact_windows_from_cues(context: Any, *, fps: float) -> tuple[dict[str, object], tuple[str, ...]]:
    cue_paths = {
        "audio": _first_existing(context, DEFAULT_AUDIO_ONSETS_FILENAME),
        "wrist": _first_existing(context, DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME),
        "ball": _first_existing(context, DEFAULT_BALL_INFLECTIONS_FILENAME),
    }
    missing = [name for name, path in cue_paths.items() if path is None]
    if missing:
        return (
            build_contact_windows_artifact([]),
            (f"contact_windows.json is empty because required cue artifacts are missing: {', '.join(missing)}",),
        )

    contact_payload = fuse_contact_windows_from_cue_payloads(
        fps=fps,
        audio_onsets_payload=_read_json(cue_paths["audio"]),
        wrist_velocity_peaks_payload=_read_json(cue_paths["wrist"]),
        ball_inflections_payload=_read_json(cue_paths["ball"]),
    )
    event_count = len(contact_payload.get("events", []))
    if event_count:
        note = "fused contact_windows.json from audio, wrist, and ball cue artifacts"
    else:
        note = "contact cue artifacts were present but produced zero temporally matched contact windows"
    return contact_payload, (note,)


def _first_existing(context: Any, filename: str) -> Path | None:
    for root in (context.inputs_dir, context.run_dir):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid no-click BALL source artifact JSON: {path}: {exc}") from exc


__all__ = [
    "BallStageRunner",
    "DEFAULT_AUDIO_ONSETS_FILENAME",
    "DEFAULT_BALL_INFLECTIONS_FILENAME",
    "DEFAULT_NO_CLICK_BALL_FILENAME",
    "DEFAULT_PROTOTYPE_GATE_ROOT",
    "DEFAULT_TRACKNET_SMOKE_DIR",
    "DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME",
]
