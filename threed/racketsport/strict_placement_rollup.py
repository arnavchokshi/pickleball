from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PASS = "PASS"
FAIL = "FAIL"
NOT_COMPUTABLE = "NOT_COMPUTABLE"

CLEAN = "viewable_preview_clean"
WITH_DEFECTS = "viewable_preview_with_defects"
NOT_VIEWABLE = "not_viewable"

REQUIRED_PIPELINE_STAGES = frozenset({"calibration", "tracking", "placement", "body", "world", "manifest"})
OK_STAGE_STATUSES = frozenset({"ran", "reused", "complete", "completed", "ok", "succeeded", "success"})
SMALL_INPUT_NAMES = (
    "PIPELINE_SUMMARY.json",
    "tracks.json",
    "tracks_prewrite_backup.json",
    "placement.json",
    "body_full_clip_gate.json",
    "body_grounding_quality.json",
    "trust_bands.json",
    "confidence_gate_summary.json",
    "replay_viewer_manifest.json",
)


@dataclass(frozen=True)
class StrictPlacementRollupConfig:
    side_match_threshold: float = 0.90
    net_crossing_limit: int = 0
    same_quadrant_max_fraction: float = 0.60
    min_pairwise_distance_p10_m: float = 0.50
    membership_coverage_threshold: float = 0.80


def build_strict_placement_rollup(
    clip_dir: Path | str,
    *,
    membership_path: Path | str | None = None,
    gate_table_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    config: StrictPlacementRollupConfig | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    clip_path = Path(clip_dir)
    if not clip_path.is_dir():
        raise ValueError(f"clip_dir is not a directory: {clip_path}")
    cfg = config or StrictPlacementRollupConfig()
    membership = Path(membership_path) if membership_path is not None else None
    gate_table = Path(gate_table_path) if gate_table_path is not None else None

    inputs_used = _inputs_used(clip_path, membership_path=membership, gate_table_path=gate_table)
    summary = _read_json_object(clip_path / "PIPELINE_SUMMARY.json")
    tracks = _read_json_object(clip_path / "tracks.json")
    placement = _read_json_object(clip_path / "placement.json")
    body_gate = _read_json_object(clip_path / "body_full_clip_gate.json")
    grounding = _read_json_object(clip_path / "body_grounding_quality.json")
    manifest = _read_json_object(clip_path / "replay_viewer_manifest.json")

    checks = [
        _check_pipeline_status(summary, clip_path=clip_path),
        _check_body_full_clip_gate(body_gate),
        _check_side_consistency(tracks, cfg),
        _check_quadrant_separation(tracks, cfg),
        _check_placement_sidecar_identity(placement),
        _check_foot_slide(grounding),
        _check_membership(membership, cfg),
        _check_mesh_honesty(manifest),
        _check_gate_table(gate_table),
    ]
    viewability = _viewability_state(clip_path, manifest)
    status = NOT_VIEWABLE if not viewability["viewable"] else _rollup_status(checks)
    if status == NOT_VIEWABLE:
        checks.append(
            _check(
                "viewability_artifacts",
                FAIL,
                measured=viewability,
                threshold="manifest plus world artifact",
                detail=viewability["reason"],
            )
        )

    report = {
        "clip": str(clip_path),
        "status": status,
        "checks": checks,
        "generated_at": generated_at or _now_utc(),
        "inputs_used": inputs_used,
    }
    if out_dir is not None:
        write_strict_placement_rollup(report, Path(out_dir))
    return report


def write_strict_placement_rollup(report: Mapping[str, Any], out_dir: Path | str) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "strict_rollup.json"
    md_path = out_path / "strict_rollup.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_strict_placement_rollup_md(report), encoding="utf-8")
    return json_path, md_path


def render_strict_placement_rollup_md(report: Mapping[str, Any]) -> str:
    rows = [
        "# Strict Placement Rollup",
        "",
        "Preview statuses only. Nothing here is VERIFIED in the repo's promotion sense.",
        "",
        f"clip: `{_escape_md(str(report.get('clip', '')) )}`",
        f"status: `{_escape_md(str(report.get('status', '')) )}`",
        "",
        "| check | status | measured | threshold | detail |",
        "|---|---|---|---|---|",
    ]
    for check in report.get("checks", []):
        if not isinstance(check, Mapping):
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_md(str(check.get("name", ""))),
                    _escape_md(str(check.get("status", ""))),
                    _escape_md(_compact_json(check.get("measured"))),
                    _escape_md(str(check.get("threshold", ""))),
                    _escape_md(str(check.get("detail", ""))),
                ]
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


REQUIRED_STAGE_ARTIFACTS: dict[str, str] = {
    "calibration": "court_calibration.json",
    "tracking": "tracks.json",
    "placement": "placement.json",
    "body": "smpl_motion.json",
    "world": "virtual_world.json",
    "manifest": "replay_viewer_manifest.json",
}


def _check_pipeline_status(summary: Mapping[str, Any] | None, *, clip_path: Path | None = None) -> dict[str, Any]:
    if summary is None:
        return _check(
            "pipeline_status",
            NOT_COMPUTABLE,
            measured={"summary_present": False},
            threshold="PIPELINE_SUMMARY.status in {complete, partial}; required stages present",
            detail="PIPELINE_SUMMARY.json missing or unreadable",
        )
    status = str(summary.get("status") or "").lower()
    stages = summary.get("stages")
    if status not in {"complete", "partial"}:
        return _check(
            "pipeline_status",
            FAIL,
            measured={"status": status or None},
            threshold="complete or partial",
            detail="pipeline status is outside the inspectable set",
        )
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return _check(
            "pipeline_status",
            NOT_COMPUTABLE,
            measured={"status": status, "stages_present": False},
            threshold=f"required stages: {sorted(REQUIRED_PIPELINE_STAGES)}",
            detail="stage records missing",
        )
    stage_statuses: dict[str, str | None] = {}
    for stage in stages:
        if isinstance(stage, Mapping):
            name = str(stage.get("stage") or "")
            if name:
                stage_statuses[name] = str(stage.get("status") or "").lower() or None
    def _stage_ok(stage: str) -> bool:
        if stage_statuses.get(stage) in OK_STAGE_STATUSES:
            return True
        # A "skipped" required stage is a reuse marker (no-force rerun over valid
        # artifacts); accept it only when the stage's primary artifact really exists.
        if stage_statuses.get(stage) == "skipped" and clip_path is not None:
            artifact = REQUIRED_STAGE_ARTIFACTS.get(stage)
            if artifact is not None and (clip_path / artifact).is_file():
                return True
        return False

    missing_required = sorted(stage for stage in REQUIRED_PIPELINE_STAGES if not _stage_ok(stage))
    measured = {"status": status, "required_stage_statuses": {stage: stage_statuses.get(stage) for stage in sorted(REQUIRED_PIPELINE_STAGES)}}
    if missing_required:
        return _check(
            "pipeline_status",
            FAIL,
            measured={**measured, "missing_or_blocked_required": missing_required},
            threshold=f"required stages must be in {sorted(OK_STAGE_STATUSES)}",
            detail="partial or incomplete required stage chain",
        )
    detail = "partial due only to nonrequired stages" if status == "partial" else "required stage records present"
    return _check("pipeline_status", PASS, measured=measured, threshold="required stages present", detail=detail)


def _check_body_full_clip_gate(body_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    if body_gate is None:
        return _check(
            "body_full_clip_gate",
            NOT_COMPUTABLE,
            measured={"present": False},
            threshold="passed == true",
            detail="body_full_clip_gate.json missing or unreadable",
        )
    passed = body_gate.get("passed")
    if passed is True:
        return _check("body_full_clip_gate", PASS, measured={"passed": True}, threshold="passed == true", detail="")
    if passed is False:
        return _check("body_full_clip_gate", FAIL, measured={"passed": False}, threshold="passed == true", detail="BODY full-clip gate false")
    return _check(
        "body_full_clip_gate",
        NOT_COMPUTABLE,
        measured={"passed": passed},
        threshold="passed == true",
        detail="passed field missing or not boolean",
    )


def _check_side_consistency(
    tracks: Mapping[str, Any] | None,
    config: StrictPlacementRollupConfig,
) -> dict[str, Any]:
    if tracks is None:
        return _check(
            "side_consistency",
            NOT_COMPUTABLE,
            measured={"tracks_present": False},
            threshold=f"match_fraction >= {config.side_match_threshold}; crossings <= {config.net_crossing_limit}",
            detail="tracks.json missing or unreadable",
        )
    players = _players(tracks)
    if not players:
        return _check(
            "side_consistency",
            NOT_COMPUTABLE,
            measured={"player_count": 0},
            threshold=f"match_fraction >= {config.side_match_threshold}; crossings <= {config.net_crossing_limit}",
            detail="no players in tracks.json",
        )

    per_player: list[dict[str, Any]] = []
    failures: list[str] = []
    uncomputed: list[str] = []
    for index, player in enumerate(players):
        player_id = _player_id(player, index)
        side = _player_side(player)
        expected_sign = _expected_y_sign(side)
        samples = _track_samples(player)
        if expected_sign is None or not samples:
            uncomputed.append(player_id)
            per_player.append(
                {
                    "player_id": player_id,
                    "side": side,
                    "match_fraction": None,
                    "net_crossings": None,
                    "sample_count": len(samples),
                }
            )
            continue
        matches = sum(1 for sample in samples if _sign(sample["xy"][1]) == expected_sign)
        fraction = matches / len(samples)
        crossings = _net_crossings(samples)
        if fraction < config.side_match_threshold or crossings > config.net_crossing_limit:
            failures.append(player_id)
        per_player.append(
            {
                "player_id": player_id,
                "side": side,
                "match_fraction": fraction,
                "net_crossings": crossings,
                "sample_count": len(samples),
            }
        )

    measured = {"players": per_player}
    threshold = f"match_fraction >= {config.side_match_threshold}; net_crossings <= {config.net_crossing_limit}"
    if failures:
        return _check("side_consistency", FAIL, measured=measured, threshold=threshold, detail=f"players failing side or crossing check: {', '.join(failures)}")
    if uncomputed:
        return _check("side_consistency", NOT_COMPUTABLE, measured=measured, threshold=threshold, detail=f"players not computable: {', '.join(uncomputed)}")
    return _check("side_consistency", PASS, measured=measured, threshold=threshold, detail="")


def _check_quadrant_separation(
    tracks: Mapping[str, Any] | None,
    config: StrictPlacementRollupConfig,
) -> dict[str, Any]:
    if tracks is None:
        return _check(
            "quadrant_separation",
            NOT_COMPUTABLE,
            measured={"tracks_present": False},
            threshold=f"same_x_quadrant_fraction <= {config.same_quadrant_max_fraction}; p10_distance_m >= {config.min_pairwise_distance_p10_m}",
            detail="tracks.json missing or unreadable",
        )
    players = _players(tracks)
    groups: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    unknown_side: list[str] = []
    for index, player in enumerate(players):
        player_id = _player_id(player, index)
        side = _player_side(player)
        if side is None:
            unknown_side.append(player_id)
            continue
        groups.setdefault(str(side).lower(), []).append((player_id, _track_samples(player)))
    if unknown_side:
        return _check(
            "quadrant_separation",
            NOT_COMPUTABLE,
            measured={"unknown_side_players": unknown_side},
            threshold=f"same_x_quadrant_fraction <= {config.same_quadrant_max_fraction}; p10_distance_m >= {config.min_pairwise_distance_p10_m}",
            detail="player side labels missing",
        )

    pair_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for side, entries in sorted(groups.items()):
        for i, left in enumerate(entries):
            for right in entries[i + 1 :]:
                row = _quadrant_pair_row(side, left, right)
                pair_rows.append(row)
                if row["shared_frame_count"] == 0 or row["same_x_quadrant_fraction"] is None or row["distance_p10_m"] is None:
                    continue
                if (
                    row["same_x_quadrant_fraction"] > config.same_quadrant_max_fraction
                    or row["distance_p10_m"] < config.min_pairwise_distance_p10_m
                ):
                    failures.append(f"{row['player_a']}:{row['player_b']}")
    threshold = f"same_x_quadrant_fraction <= {config.same_quadrant_max_fraction}; p10_distance_m >= {config.min_pairwise_distance_p10_m}"
    if not pair_rows or any(row["shared_frame_count"] == 0 for row in pair_rows):
        return _check("quadrant_separation", NOT_COMPUTABLE, measured={"pairs": pair_rows}, threshold=threshold, detail="same-side player pairs missing or not aligned")
    if failures:
        return _check("quadrant_separation", FAIL, measured={"pairs": pair_rows}, threshold=threshold, detail=f"same-side pair separation failed: {', '.join(failures)}")
    return _check("quadrant_separation", PASS, measured={"pairs": pair_rows}, threshold=threshold, detail="")


def _check_placement_sidecar_identity(placement: Mapping[str, Any] | None) -> dict[str, Any]:
    if placement is None:
        return _check(
            "placement_sidecar_identity",
            NOT_COMPUTABLE,
            measured={"placement_present": False},
            threshold="identity diagnostics present when sidecar inputs are used",
            detail="placement.json missing or unreadable",
        )
    provenance = placement.get("provenance") if isinstance(placement.get("provenance"), Mapping) else placement
    sidecar_used = _placement_uses_sidecar(provenance)
    diagnostics_present = _has_identity_diagnostics(provenance)
    mismatches = _integer_mismatches(provenance)
    measured = {"sidecar_used": sidecar_used, "identity_diagnostics_present": diagnostics_present, "accepted_integer_mismatch_count": len(mismatches), "accepted_integer_mismatches": mismatches}
    if sidecar_used and not diagnostics_present:
        return _check(
            "placement_sidecar_identity",
            NOT_COMPUTABLE,
            measured=measured,
            threshold="identity diagnostics present when sidecar inputs are used",
            detail="sidecar identity diagnostics absent",
        )
    detail = "sidecar not used" if not sidecar_used else "identity diagnostics present"
    if mismatches:
        detail = f"{detail}; accepted integer mismatches recorded as informational"
    return _check("placement_sidecar_identity", PASS, measured=measured, threshold="identity diagnostics present when needed", detail=detail)


def _check_foot_slide(grounding: Mapping[str, Any] | None) -> dict[str, Any]:
    pooled_threshold = 0.020
    max_threshold = 0.030
    if grounding is None:
        return _check(
            "foot_slide",
            NOT_COMPUTABLE,
            measured={"present": False},
            threshold="pooled/per-player p95 <= 0.020m; max <= 0.030m",
            detail="body_grounding_quality.json missing or unreadable",
        )
    pooled = _first_number_at(
        grounding,
        [
            ("grounding_metrics", "foot_lock_slide_p95_m"),
            ("grounding_metrics", "stance_slide_p95_m"),
            ("grounding_metrics", "pooled_stance_slide_p95_m"),
            ("foot_lock_slide_p95_m",),
            ("stance_slide_p95_m",),
        ],
    )
    max_slide = _first_number_at(
        grounding,
        [
            ("grounding_metrics", "max_foot_lock_slide_m"),
            ("grounding_metrics", "foot_slide_max_m"),
            ("foot_slide_gate", "value_m"),
            ("max_foot_lock_slide_m",),
        ],
    )
    per_player = _first_number_map_at(
        grounding,
        [
            ("grounding_metrics", "foot_lock_slide_p95_by_player_m"),
            ("grounding_metrics", "stance_slide_p95_by_player_m"),
            ("grounding_metrics", "per_player_stance_slide_p95_m"),
            ("foot_lock_slide_p95_by_player_m",),
        ],
    )
    missing = []
    if pooled is None:
        missing.append("pooled_p95")
    if per_player is None:
        missing.append("per_player_p95")
    if max_slide is None:
        missing.append("pipeline_max")
    measured = {"pooled_p95_m": pooled, "per_player_p95_m": per_player, "pipeline_max_m": max_slide}
    threshold = "pooled/per-player p95 <= 0.020m; max <= 0.030m"
    player_failures = {player: value for player, value in per_player.items() if value > pooled_threshold} if per_player else {}
    # A measurable exceedance FAILS even when other slide fields are missing —
    # missing detail must never mask a failing pipeline gate.
    if (pooled is not None and pooled > pooled_threshold) or (max_slide is not None and max_slide > max_threshold) or player_failures:
        return _check(
            "foot_slide",
            FAIL,
            measured={**measured, "per_player_failures": player_failures, "missing": missing},
            threshold=threshold,
            detail="foot slide exceeds threshold",
        )
    if missing:
        return _check("foot_slide", NOT_COMPUTABLE, measured={**measured, "missing": missing}, threshold=threshold, detail="missing stance slide fields")
    return _check("foot_slide", PASS, measured=measured, threshold=threshold, detail="")


def _check_membership(
    membership_path: Path | None,
    config: StrictPlacementRollupConfig,
) -> dict[str, Any]:
    if membership_path is None:
        return _check(
            "membership",
            NOT_COMPUTABLE,
            measured={"provided": False},
            threshold=f"on-target-court coverage >= {config.membership_coverage_threshold}",
            detail="unproven",
        )
    payload = _read_json_object(membership_path)
    if payload is None:
        return _check(
            "membership",
            NOT_COMPUTABLE,
            measured={"provided": True, "path": str(membership_path), "readable": False},
            threshold=f"on-target-court coverage >= {config.membership_coverage_threshold}",
            detail="membership artifact missing or unreadable",
        )
    records = _membership_records(payload)
    if not records:
        return _check(
            "membership",
            NOT_COMPUTABLE,
            measured={"provided": True, "players": []},
            threshold=f"on-target-court coverage >= {config.membership_coverage_threshold}",
            detail="membership rows missing",
        )
    real_records = [record for record in records if record.get("rendered_as_real") is not False]
    missing = [record for record in real_records if record.get("coverage") is None]
    failures = [record for record in real_records if record.get("coverage") is not None and record["coverage"] < config.membership_coverage_threshold]
    measured = {"players": records, "real_player_count": len(real_records)}
    threshold = f"on-target-court coverage >= {config.membership_coverage_threshold}"
    if not real_records or missing:
        return _check("membership", NOT_COMPUTABLE, measured=measured, threshold=threshold, detail="real-player membership coverage missing")
    if failures:
        return _check("membership", FAIL, measured={**measured, "failures": failures}, threshold=threshold, detail="target-court membership below threshold")
    return _check("membership", PASS, measured=measured, threshold=threshold, detail="")


def _check_mesh_honesty(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return _check(
            "mesh_honesty",
            NOT_COMPUTABLE,
            measured={"manifest_present": False},
            threshold="mesh parse marker present or skeleton_only == true",
            detail="manifest missing or unreadable",
        )
    skeleton_only = _manifest_skeleton_only(manifest)
    mesh_refs = _manifest_mesh_refs(manifest)
    mesh_parse_checked = _manifest_mesh_parse_checked(manifest)
    measured = {"skeleton_only": skeleton_only, "mesh_ref_count": len(mesh_refs), "mesh_refs": mesh_refs, "mesh_parse_marker": mesh_parse_checked}
    if skeleton_only:
        return _check("mesh_honesty", PASS, measured=measured, threshold="skeleton_only == true", detail="skeleton_only")
    if mesh_refs and mesh_parse_checked:
        return _check("mesh_honesty", PASS, measured=measured, threshold="mesh parse marker present", detail="mesh parse marker present")
    if mesh_refs:
        return _check(
            "mesh_honesty",
            NOT_COMPUTABLE,
            measured=measured,
            threshold="mesh parse marker present or skeleton_only == true",
            detail="mesh refs exist without parse marker",
        )
    return _check(
        "mesh_honesty",
        NOT_COMPUTABLE,
        measured=measured,
        threshold="mesh parse marker present or skeleton_only == true",
        detail="bundle mode undeclared",
    )


def _check_gate_table(gate_table_path: Path | None) -> dict[str, Any]:
    if gate_table_path is None:
        return _check("gate_table", PASS, measured={"provided": False}, threshold="no FAIL or NOT COMPUTABLE rows", detail="not provided")
    payload = _read_json(gate_table_path)
    if payload is None:
        return _check(
            "gate_table",
            NOT_COMPUTABLE,
            measured={"provided": True, "path": str(gate_table_path), "readable": False},
            threshold="no FAIL or NOT COMPUTABLE rows",
            detail="gate table missing or unreadable",
        )
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return _check(
            "gate_table",
            NOT_COMPUTABLE,
            measured={"provided": True, "row_count": None},
            threshold="rows list with no FAIL or NOT COMPUTABLE rows",
            detail="gate table rows missing",
        )
    blocking: list[dict[str, Any]] = []
    not_computable: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        status = _normalize_gate_status(row.get("status"))
        rendered = {"gate": row.get("gate"), "status": row.get("status")}
        if status == FAIL:
            blocking.append(rendered)
        elif status == NOT_COMPUTABLE:
            not_computable.append(rendered)
    measured = {"provided": True, "row_count": len(rows), "blocking_rows": blocking + not_computable}
    if blocking:
        return _check("gate_table", FAIL, measured=measured, threshold="no FAIL or NOT COMPUTABLE rows", detail="gate table contains FAIL rows")
    if not_computable:
        return _check("gate_table", NOT_COMPUTABLE, measured=measured, threshold="no FAIL or NOT COMPUTABLE rows", detail="gate table contains NOT COMPUTABLE rows")
    return _check("gate_table", PASS, measured=measured, threshold="no FAIL or NOT COMPUTABLE rows", detail="")


def _viewability_state(clip_dir: Path, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    manifest_path = clip_dir / "replay_viewer_manifest.json"
    if not manifest_path.is_file() or manifest is None:
        return {"viewable": False, "reason": "no manifest"}
    world_candidates = [clip_dir / "confidence_gated_world.json", clip_dir / "virtual_world.json"]
    url_value = manifest.get("virtual_world_url") if isinstance(manifest, Mapping) else None
    if isinstance(url_value, str) and url_value:
        resolved = _resolve_manifest_url(url_value, clip_dir)
        if resolved is not None:
            world_candidates.append(resolved)
    has_world = any(path.is_file() for path in world_candidates)
    if not has_world:
        return {"viewable": False, "reason": "missing world artifacts"}
    return {"viewable": True, "reason": ""}


def _rollup_status(checks: Sequence[Mapping[str, Any]]) -> str:
    return CLEAN if all(check.get("status") == PASS for check in checks) else WITH_DEFECTS


def _check(name: str, status: str, *, measured: Any, threshold: str, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "measured": measured,
        "threshold": threshold,
        "detail": detail,
    }


def _players(tracks: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = tracks.get("players")
    return [player for player in players if isinstance(player, Mapping)] if isinstance(players, Sequence) and not isinstance(players, (str, bytes)) else []


def _player_id(player: Mapping[str, Any], index: int) -> str:
    for key in ("id", "player_id", "track_id"):
        if key in player:
            return str(player[key])
    return str(index)


def _player_side(player: Mapping[str, Any]) -> str | None:
    for key in ("side", "side_label", "court_side", "player_side"):
        value = player.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _expected_y_sign(side: str | None) -> int | None:
    if side is None:
        return None
    normalized = side.lower()
    if normalized in {"near", "bottom", "home", "camera", "near_side"}:
        return -1
    if normalized in {"far", "top", "away", "far_side"}:
        return 1
    return None


def _track_samples(player: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = player.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    samples: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        xy = frame.get("world_xy")
        if not _is_xy(xy):
            continue
        samples.append({"key": _frame_key(frame, index), "xy": [float(xy[0]), float(xy[1])]})
    return sorted(samples, key=lambda sample: sample["key"])


def _frame_key(frame: Mapping[str, Any], index: int) -> tuple[str, Any]:
    for key in ("frame_idx", "frame_index", "frame"):
        value = frame.get(key)
        if isinstance(value, int):
            return key, value
    value = frame.get("t")
    if isinstance(value, (int, float)):
        return "t", round(float(value), 6)
    return "index", index


def _net_crossings(samples: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    previous: Mapping[str, Any] | None = None
    for sample in samples:
        if previous is not None:
            y0 = float(previous["xy"][1])
            y1 = float(sample["xy"][1])
            if abs(y0) > 0.3 and abs(y1) > 0.3 and _sign(y0) != _sign(y1):
                count += 1
        previous = sample
    return count


def _quadrant_pair_row(
    side: str,
    left: tuple[str, list[dict[str, Any]]],
    right: tuple[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    left_id, left_samples = left
    right_id, right_samples = right
    left_by_key = {sample["key"]: sample["xy"] for sample in left_samples}
    right_by_key = {sample["key"]: sample["xy"] for sample in right_samples}
    keys = sorted(set(left_by_key) & set(right_by_key))
    same = 0
    distances: list[float] = []
    for key in keys:
        left_xy = left_by_key[key]
        right_xy = right_by_key[key]
        if _sign(left_xy[0]) == _sign(right_xy[0]):
            same += 1
        distances.append(math.dist(left_xy, right_xy))
    same_fraction = same / len(keys) if keys else None
    distance_p10 = _percentile(distances, 10.0) if distances else None
    return {
        "side": side,
        "player_a": left_id,
        "player_b": right_id,
        "shared_frame_count": len(keys),
        "same_x_quadrant_fraction": same_fraction,
        "distance_p10_m": distance_p10,
    }


def _placement_uses_sidecar(provenance: Any) -> bool:
    if not isinstance(provenance, Mapping):
        return False
    for key in ("native2d_keypoints", "sam3d_keypoints", "capture_sidecar", "sidecar"):
        value = provenance.get(key)
        if value not in (None, "", [], {}):
            return True
    source_counts = provenance.get("source_counts")
    if isinstance(source_counts, Mapping):
        for key in ("native2d", "sam3d", "sidecar"):
            value = source_counts.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return True
    return False


def _has_identity_diagnostics(node: Any) -> bool:
    if isinstance(node, Mapping):
        for key, value in node.items():
            normalized = str(key).lower()
            if normalized in {
                "identity_diagnostics",
                "sidecar_identity_diagnostics",
                "mapping_votes",
                "dropped_counts",
                "accepted_mappings",
                "integer_match",
            } or ("identity" in normalized and "diagnostic" in normalized):
                return True
            if _has_identity_diagnostics(value):
                return True
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return any(_has_identity_diagnostics(value) for value in node)
    return False


def _integer_mismatches(node: Any, *, in_accepted_mappings: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(node, Mapping):
        accepted_context = in_accepted_mappings
        if str(node.get("status") or "").lower() == "accepted" or node.get("accepted") is True:
            accepted_context = True
        if accepted_context and node.get("integer_match") is False:
            rows.append({key: node.get(key) for key in ("player_id", "id", "sidecar_id", "track_id", "integer_match") if key in node})
        for key, value in node.items():
            rows.extend(_integer_mismatches(value, in_accepted_mappings=accepted_context or str(key).lower() == "accepted_mappings"))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for value in node:
            rows.extend(_integer_mismatches(value, in_accepted_mappings=in_accepted_mappings))
    return rows


def _membership_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = payload.get("players")
    if source is None:
        source = payload.get("per_player") or payload.get("coverage_by_player")
    records: list[dict[str, Any]] = []
    if isinstance(source, Mapping):
        iterable = []
        for player_id, value in source.items():
            if isinstance(value, Mapping):
                iterable.append({"player_id": player_id, **value})
            else:
                iterable.append({"player_id": player_id, "coverage": value})
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        iterable = [record for record in source if isinstance(record, Mapping)]
    else:
        iterable = []
    for record in iterable:
        records.append(
            {
                "player_id": record.get("player_id", record.get("id")),
                "rendered_as_real": _rendered_as_real(record),
                "coverage": _coverage_value(record),
            }
        )
    return records


def _rendered_as_real(record: Mapping[str, Any]) -> bool:
    for key in ("rendered_as_real", "rendered_real", "real_player"):
        if key in record:
            return bool(record[key])
    status = str(record.get("render_status") or record.get("rendered_as") or "").lower()
    if status in {"skeleton_only", "synthetic", "hidden", "not_real"}:
        return False
    # player_court_membership verdicts: adjacent/spectator players are excluded
    # from the rendered world by virtual_world's membership consumption.
    verdict = str(record.get("verdict") or "").lower()
    if verdict == "adjacent_or_spectator":
        return False
    return True


def _coverage_value(record: Mapping[str, Any]) -> float | None:
    for key in (
        "on_target_court_coverage",
        "target_court_coverage",
        "on_target_court_fraction",
        "coverage",
        "on_court_fraction",
        "inside_asym_frac",
    ):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _manifest_skeleton_only(manifest: Mapping[str, Any]) -> bool:
    for key, value in _walk_items(manifest):
        normalized = key.lower()
        if normalized in {"skeleton_only", "skeletononly"} and value is True:
            return True
        if normalized in {"bundle_mode", "mesh_status", "body_mesh_status"} and str(value).lower() in {"skeleton_only", "skeleton-only"}:
            return True
    return False


def _manifest_mesh_refs(manifest: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for key, value in _walk_items(manifest):
        normalized = key.lower()
        if "mesh" not in normalized:
            continue
        if "status" in normalized or "parse" in normalized or "ready" in normalized:
            continue
        if isinstance(value, str) and value:
            refs.append(key)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 0:
            refs.append(key)
    return sorted(set(refs))


def _manifest_mesh_parse_checked(manifest: Mapping[str, Any]) -> bool:
    good_states = {"parse_verified", "parse-verified", "parse_checked", "parse-checked", "checked", "available_parse_checked"}
    for key, value in _walk_items(manifest):
        normalized = key.lower()
        if "mesh" not in normalized and "parse" not in normalized:
            continue
        if isinstance(value, bool) and value and ("parse" in normalized or "checked" in normalized or "verified" in normalized):
            return True
        if isinstance(value, str) and value.lower() in good_states:
            return True
    return False


def _walk_items(node: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            items.append((str(key), value))
            items.extend(_walk_items(value))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for value in node:
            items.extend(_walk_items(value))
    return items


def _normalize_gate_status(status: Any) -> str:
    normalized = str(status or "").strip().upper().replace("_", " ").replace("-", " ")
    if normalized in {"FAIL", "FAILED"}:
        return FAIL
    if normalized in {"NOT COMPUTABLE", "NOTCOMPUTABLE", "UNKNOWN"}:
        return NOT_COMPUTABLE
    return PASS


def _inputs_used(clip_dir: Path, *, membership_path: Path | None, gate_table_path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in SMALL_INPUT_NAMES:
        path = clip_dir / name
        payload[name] = {"path": str(path), "exists": path.is_file()}
    if membership_path is not None:
        payload["membership"] = {"path": str(membership_path), "exists": membership_path.is_file()}
    else:
        payload["membership"] = {"path": None, "exists": False}
    if gate_table_path is not None:
        payload["gate_table"] = {"path": str(gate_table_path), "exists": gate_table_path.is_file()}
    else:
        payload["gate_table"] = {"path": None, "exists": False}
    return payload


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    payload = _read_json(path)
    return payload if isinstance(payload, Mapping) else None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _first_number_at(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        value = _get_path(payload, path)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _first_number_map_at(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> dict[str, float] | None:
    for path in paths:
        value = _get_path(payload, path)
        if isinstance(value, Mapping):
            parsed: dict[str, float] = {}
            for key, item in value.items():
                if isinstance(item, (int, float)):
                    parsed[str(key)] = float(item)
            if parsed:
                return parsed
    return None


def _get_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = payload
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _is_xy(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _resolve_manifest_url(value: str, clip_dir: Path) -> Path | None:
    if value.startswith("/@fs/"):
        path_text = value.removeprefix("/@fs/")
        if path_text.startswith("/"):
            return Path(path_text)
        return Path("/") / path_text
    if value.startswith("@fs/"):
        return Path("/") / value.removeprefix("@fs/")
    if value.startswith("/"):
        return Path(value)
    return clip_dir / value


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= 220 else text[:217] + "..."


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
