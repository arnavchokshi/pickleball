#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.validate_corrections import validate_manifest  # noqa: E402


DEFAULT_RUN_ROOT = Path("runs/eval0/prototype_gate_h100_v2")
ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def build_corrections_from_review_inputs(
    review_input: Mapping[str, Any],
    *,
    manifest_id: str,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    annotator: str = "review-ui",
) -> dict[str, Any]:
    created_at = _created_at(review_input)
    run_root_path = Path(run_root)
    corrections: list[dict[str, Any]] = []

    global_payload = _non_empty_global(review_input.get("global"))
    if global_payload:
        corrections.append(
            _correction(
                manifest_id=manifest_id,
                index=len(corrections) + 1,
                clip="global",
                artifact=Path("runs/review_inputs/pickleball_cv_review_latest.json"),
                path="/global",
                phase="phase11",
                metric="global_review_policy",
                value=global_payload,
                reason="Human review global policy from browser review UI.",
                annotator=annotator,
            )
        )

    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        clips = {}
    for clip, raw_clip_payload in sorted(clips.items()):
        if not isinstance(raw_clip_payload, Mapping):
            continue
        clip_id = str(clip)
        corrections.extend(_clip_corrections(clip_id, raw_clip_payload, run_root_path, manifest_id, annotator))

    if not corrections:
        raise ValueError("review input did not contain any exportable correction notes")

    return {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "created_at": created_at,
        "description": "Corrections exported from the local pickleball browser review UI.",
        "corrections": corrections,
    }


def _clip_corrections(
    clip: str,
    payload: Mapping[str, Any],
    run_root: Path,
    manifest_id: str,
    annotator: str,
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []

    top_net = payload.get("top_net") if isinstance(payload.get("top_net"), Mapping) else {}
    court_value = {
        "reviewed_enough": bool(payload.get("reviewed_enough", False)),
        "court_overlay_ok": payload.get("court_overlay_ok", "unsure"),
        "top_net": top_net,
    }
    if _has_review_value(court_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "court_calibration.json",
                "/human_review/top_net",
                "phase1",
                "calibration_top_net",
                court_value,
                "Human reviewed calibration overlay and top-net reference points.",
                annotator,
            )
        )

    court_evidence = payload.get("court_evidence") if isinstance(payload.get("court_evidence"), Mapping) else {}
    court_evidence_value = {
        "states": {
            str(key): value
            for key, value in court_evidence.items()
            if key not in {"points", "point_statuses", "notes"} and _has_review_value(value)
        },
        "points": court_evidence.get("points", {}),
        "point_statuses": court_evidence.get("point_statuses", {}),
        "notes": court_evidence.get("notes", ""),
    }
    if _has_review_value(court_evidence_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "court_line_evidence.json",
                "/human_review/court_evidence",
                "phase1",
                "court_line_evidence_review",
                court_evidence_value,
                "Human clicked or classified court line evidence in the browser review UI.",
                annotator,
            )
        )

    players = payload.get("players") if isinstance(payload.get("players"), Mapping) else {}
    player_value = {
        "players": {str(key): value for key, value in players.items() if value},
        "spectators_ignore": payload.get("spectators_ignore", ""),
    }
    if _has_review_value(player_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "tracks.json",
                "/human_review/player_identity",
                "phase2",
                "player_identity",
                player_value,
                "Human provided player identity or spectator ignore notes.",
                annotator,
            )
        )

    ball = payload.get("ball") if isinstance(payload.get("ball"), Mapping) else {}
    ball_value = {
        "mistakes": ball.get("mistakes", []),
        "notes": ball.get("notes", ""),
    }
    if _has_review_value(ball_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json",
                "/human_review/ball",
                "phase5",
                "ball_track_review",
                ball_value,
                "Human marked ball-track mistakes or notes.",
                annotator,
            )
        )

    event_value = {
        "contacts": payload.get("contacts", []),
        "event_windows": payload.get("event_windows", []),
    }
    if _has_review_value(event_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "contact_windows.json",
                "/human_review/events",
                "phase5",
                "contact_windows",
                event_value,
                "Human marked contact candidates or rally/event windows.",
                annotator,
            )
        )

    racket = payload.get("racket") if isinstance(payload.get("racket"), Mapping) else {}
    racket_value = {
        "examples": racket.get("examples", []),
        "notes": racket.get("notes", ""),
    }
    if _has_review_value(racket_value):
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "racket_candidates.json",
                "/human_review/racket",
                "phase6",
                "racket_candidates",
                racket_value,
                "Human marked visible paddle examples or racket notes.",
                annotator,
            )
        )

    general_notes = payload.get("general_notes", "")
    if isinstance(general_notes, str) and general_notes.strip():
        corrections.append(
            _clip_correction(
                manifest_id,
                corrections,
                clip,
                run_root / clip / "pipeline_run.json",
                "/human_review/general_notes",
                "phase11",
                "general_review",
                general_notes.strip(),
                "Human provided general clip notes.",
                annotator,
            )
        )
    return corrections


def _clip_correction(
    manifest_id: str,
    corrections: list[dict[str, Any]],
    clip: str,
    artifact: Path,
    path: str,
    phase: str,
    metric: str,
    value: Any,
    reason: str,
    annotator: str,
) -> dict[str, Any]:
    return _correction(
        manifest_id=manifest_id,
        index=len(corrections) + 1,
        clip=clip,
        artifact=artifact,
        path=path,
        phase=phase,
        metric=metric,
        value=value,
        reason=reason,
        annotator=annotator,
    )


def _correction(
    *,
    manifest_id: str,
    index: int,
    clip: str,
    artifact: Path,
    path: str,
    phase: str,
    metric: str,
    value: Any,
    reason: str,
    annotator: str,
) -> dict[str, Any]:
    return {
        "id": _correction_id(manifest_id, clip, metric, index),
        "target": {
            "artifact": artifact.as_posix(),
            "clip_id": clip,
            "phase": phase,
            "metric": metric,
            "path": path,
        },
        "operation": "append",
        "value": value,
        "reason": reason,
        "annotator": annotator,
        "status": "pending",
    }


def _correction_id(manifest_id: str, clip: str, metric: str, index: int) -> str:
    raw = f"{clip}_{metric}_{index:03d}"
    safe = ID_SAFE.sub("_", raw).strip("._-")
    if not safe or not safe[0].isalnum():
        safe = f"corr_{index:03d}_{safe}"
    return safe[:128]


def _created_at(review_input: Mapping[str, Any]) -> str:
    for key in ("server_saved_at_utc", "saved_from_browser_at"):
        value = review_input.get(key)
        if isinstance(value, str) and value:
            return value.replace("+00:00", "Z")
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _non_empty_global(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items() if _has_review_value(item)}


def _has_review_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip()) and value not in {"unsure"}
    if isinstance(value, Mapping):
        return any(_has_review_value(item) for item in value.values())
    if isinstance(value, list):
        return bool(value)
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export local browser review inputs into corrections/schema.json format.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review input JSON from review_input_server.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output corrections manifest path.")
    parser.add_argument("--manifest-id", help="Corrections manifest id. Defaults to review input stem plus _corrections.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Run root used to build target artifact paths.")
    parser.add_argument("--annotator", default="review-ui", help="Annotator label for exported corrections.")
    args = parser.parse_args(argv)

    try:
        payload = _read_json(args.review_input)
        if not isinstance(payload, Mapping):
            raise ValueError("review input must be a JSON object")
        manifest_id = args.manifest_id or f"{args.review_input.stem}_corrections"
        corrections = build_corrections_from_review_inputs(
            payload,
            manifest_id=manifest_id,
            run_root=args.run_root,
            annotator=args.annotator,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(corrections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = validate_manifest(args.out)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
