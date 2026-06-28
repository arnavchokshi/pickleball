from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_gate_report"
DEFAULT_WORLD_MPJPE_THRESHOLD_M = 0.15
BODY_WORLD_LABEL_FILENAMES = ("body_world_joints.json", "body_world_mpjpe.json")
FULL_CLIP_GATE_FILENAME = "body_full_clip_gate.json"
FULL_CLIP_GATE_REQUIRED_FIELDS = ("passed", "coverage", "evaluated_frame_count")
INSPECTABLE_OUTPUTS = (
    "virtual_world_paddle_preview.html",
    "virtual_world_review_index.json",
    "virtual_world.json",
)


def build_body_gate_report(
    *,
    root: str | Path,
    clips: list[str] | tuple[str, ...] | None = None,
    labels_root: str | Path | None = None,
    world_mpjpe_threshold_m: float = DEFAULT_WORLD_MPJPE_THRESHOLD_M,
) -> dict[str, Any]:
    root_path = Path(root)
    labels_path = Path(labels_root) if labels_root is not None else root_path
    clip_names = list(clips) if clips else _discover_clips(root_path)
    clip_reports = [
        _build_clip_report(
            root_path=root_path,
            labels_root=labels_path,
            clip=clip,
            world_mpjpe_threshold_m=world_mpjpe_threshold_m,
        )
        for clip in clip_names
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "root": str(root_path),
        "labels_root": str(labels_path),
        "status": _aggregate_status(clip_reports),
        "world_mpjpe_threshold_m": world_mpjpe_threshold_m,
        "summary": {
            "clip_count": len(clip_reports),
            "pass_count": sum(1 for clip in clip_reports if clip["status"] == "pass"),
            "fail_count": sum(1 for clip in clip_reports if clip["status"] == "fail"),
            "blocked_count": sum(1 for clip in clip_reports if clip["status"] == "blocked"),
        },
        "clips": clip_reports,
    }


def write_body_gate_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_body_gate_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body_gate_markdown(payload), encoding="utf-8")


def write_clip_body_gate_reports(root: str | Path, payload: Mapping[str, Any]) -> None:
    root_path = Path(root)
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        clip_name = str(clip.get("clip", ""))
        if not clip_name:
            continue
        clip_payload = {
            **dict(payload),
            "status": clip.get("status", "blocked"),
            "summary": {
                "clip_count": 1,
                "pass_count": 1 if clip.get("status") == "pass" else 0,
                "fail_count": 1 if clip.get("status") == "fail" else 0,
                "blocked_count": 1 if clip.get("status") == "blocked" else 0,
            },
            "clips": [dict(clip)],
        }
        clip_dir = root_path / clip_name
        write_body_gate_report(clip_dir / "body_gate_report.json", clip_payload)
        write_body_gate_markdown(clip_dir / "body_gate_report.md", clip_payload)


def render_body_gate_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BODY Gate Report",
        "",
        f"- status: `{payload.get('status', 'blocked')}`",
        f"- root: `{payload.get('root', '')}`",
        f"- labels_root: `{payload.get('labels_root', '')}`",
        f"- world_mpjpe_threshold_m: `{payload.get('world_mpjpe_threshold_m', DEFAULT_WORLD_MPJPE_THRESHOLD_M)}`",
        "",
        "| Clip | Status | Mesh smoke | World MPJPE | Full clip BODY | Scheduled frames | Mesh player-frames | Blockers | Inspectable outputs |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        mesh = clip.get("mesh_smoke") if isinstance(clip.get("mesh_smoke"), Mapping) else {}
        world = clip.get("world_mpjpe") if isinstance(clip.get("world_mpjpe"), Mapping) else {}
        full = clip.get("full_clip_body_gate") if isinstance(clip.get("full_clip_body_gate"), Mapping) else {}
        blockers = ", ".join(str(item) for item in clip.get("blockers", [])) or "-"
        outputs = ", ".join(str(item) for item in clip.get("inspectable_outputs", [])) or "-"
        lines.append(
            "| {clip} | {status} | {mesh_status} | {world_status} | {full_status} | {scheduled} | {mesh_frames} | {blockers} | {outputs} |".format(
                clip=clip.get("clip", ""),
                status=clip.get("status", ""),
                mesh_status=mesh.get("status", ""),
                world_status=world.get("status", ""),
                full_status=full.get("status", ""),
                scheduled=mesh.get("scheduled_frame_count", 0),
                mesh_frames=mesh.get("mesh_player_frame_count", 0),
                blockers=blockers,
                outputs=outputs,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_clip_report(
    *,
    root_path: Path,
    labels_root: Path,
    clip: str,
    world_mpjpe_threshold_m: float,
) -> dict[str, Any]:
    run_dir = root_path / clip
    labels_dir = labels_root / clip
    smpl_motion = _read_optional_json(run_dir / "smpl_motion.json")
    skeleton3d = _read_optional_json(run_dir / "skeleton3d.json")
    body_compute_execution = _read_optional_json(run_dir / "body_compute_execution.json")
    body_mesh_readiness = _read_optional_json(run_dir / "body_mesh_readiness.json")

    mesh_smoke = _mesh_smoke_status(
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution=body_compute_execution,
        body_mesh_readiness=body_mesh_readiness,
    )
    world_mpjpe = _world_mpjpe_status(
        run_dir=run_dir,
        labels_dir=labels_dir,
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        threshold_m=world_mpjpe_threshold_m,
    )
    full_clip_gate = _full_clip_gate_status(run_dir=run_dir, labels_dir=labels_dir)
    blockers = _dedupe(
        [
            *mesh_smoke.get("blockers", []),
            *world_mpjpe.get("blockers", []),
            *full_clip_gate.get("blockers", []),
        ]
    )
    status = _clip_status(mesh_smoke, world_mpjpe, full_clip_gate, blockers)
    return {
        "clip": clip,
        "run_dir": str(run_dir),
        "labels_dir": str(labels_dir),
        "status": status,
        "mesh_smoke": mesh_smoke,
        "world_mpjpe": world_mpjpe,
        "full_clip_body_gate": full_clip_gate,
        "blockers": blockers,
        "inspectable_outputs": [name for name in INSPECTABLE_OUTPUTS if (run_dir / name).is_file()],
    }


def _mesh_smoke_status(
    *,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    body_compute_execution: Mapping[str, Any] | None,
    body_mesh_readiness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    execution_summary = _mapping(body_compute_execution, "summary")
    readiness_summary = _mapping(body_mesh_readiness, "summary")
    representation_plan = _mapping(body_mesh_readiness, "representation_plan")
    scheduled_frame_count = _non_negative_int(execution_summary.get("scheduled_frame_count"))
    scheduled_player_frame_count = _non_negative_int(execution_summary.get("scheduled_player_frame_count"))
    if scheduled_frame_count == 0:
        scheduled_frame_count = _non_negative_int(representation_plan.get("scheduled_world_mesh_frame_count"))
    if scheduled_player_frame_count == 0:
        scheduled_player_frame_count = _non_negative_int(representation_plan.get("scheduled_world_mesh_player_frame_count"))
    mesh_player_frame_count = _non_negative_int(readiness_summary.get("mesh_frame_count"))
    skeleton_player_frame_count = _skeleton_player_frame_count(skeleton3d)
    smpl_player_frame_count = _smpl_player_frame_count(smpl_motion)
    blockers: list[str] = []
    notes: list[str] = []

    if body_compute_execution is None:
        blockers.append("missing_body_compute_execution")
    if body_mesh_readiness is None:
        blockers.append("missing_body_mesh_readiness")
    if scheduled_frame_count > 0 and mesh_player_frame_count > 0:
        status = "pass"
        notes.append("scheduled-frame mesh smoke available")
    elif scheduled_frame_count == 0:
        status = "not_measured"
        blockers.append("no_scheduled_body_mesh_smoke")
        notes.append("no scheduled BODY world-mesh frames")
    else:
        status = "blocked"
        blockers.append("missing_scheduled_body_mesh_output")

    if smpl_motion is None and scheduled_frame_count > 0:
        blockers.append("missing_smpl_motion_json")
    if skeleton3d is None and scheduled_frame_count > 0:
        blockers.append("missing_skeleton3d_json")

    return {
        "status": status,
        "scheduled_frame_count": scheduled_frame_count,
        "scheduled_player_frame_count": scheduled_player_frame_count,
        "mesh_player_frame_count": mesh_player_frame_count,
        "skeleton_player_frame_count": skeleton_player_frame_count,
        "smpl_player_frame_count": smpl_player_frame_count,
        "body_mesh_readiness_status": str(body_mesh_readiness.get("status", "")) if isinstance(body_mesh_readiness, Mapping) else "",
        "blockers": _dedupe(blockers),
        "notes": notes,
    }


def _world_mpjpe_status(
    *,
    run_dir: Path,
    labels_dir: Path,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    threshold_m: float,
) -> dict[str, Any]:
    label_path = _find_body_label_path(labels_dir)
    if label_path is None:
        label_import = _missing_body_label_import_status(labels_dir)
        return {
            "status": "not_measured",
            "label_path": "",
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": ["missing_world_mpjpe_gate"],
            "notes": ["no BODY world-joint labels found"],
            "label_import": label_import,
        }

    payload = _read_json(label_path)
    samples = _label_samples(payload)
    label_import = _body_label_import_status(
        labels_dir=labels_dir,
        label_path=label_path,
        payload=payload,
        accepted_sample_count=len(samples),
    )
    if label_import["status"] != "present_reviewed":
        return {
            "status": "not_measured",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": _dedupe(["missing_world_mpjpe_gate", *label_import.get("blockers", [])]),
            "notes": list(label_import.get("notes", [])),
            "label_import": label_import,
        }
    if not samples:
        return {
            "status": "not_measured",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": ["missing_world_mpjpe_gate"],
            "notes": ["BODY world-joint label file has no accepted samples"],
            "label_import": label_import,
        }

    prediction_index = _prediction_index(smpl_motion=smpl_motion, skeleton3d=skeleton3d)
    errors: list[float] = []
    unmatched_samples = 0
    for sample in samples:
        prediction = prediction_index.get((sample["frame_index"], sample["player_id"]))
        if prediction is None:
            unmatched_samples += 1
            continue
        errors.extend(_joint_errors(prediction, sample["joints_world"]))

    if not errors:
        return {
            "status": "fail",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": ["world_mpjpe_no_matching_predictions"],
            "notes": [f"{unmatched_samples} label samples had no matching BODY prediction"],
            "label_import": label_import,
        }

    mean_error = round(sum(errors) / len(errors), 6)
    passed = mean_error <= threshold_m
    return {
        "status": "pass" if passed else "fail",
        "label_path": str(label_path),
        "mean_error_m": mean_error,
        "threshold_m": threshold_m,
        "sample_count": len(samples) - unmatched_samples,
        "joint_count": len(errors),
        "blockers": [] if passed else ["world_mpjpe_gate_failed"],
        "notes": [f"{unmatched_samples} label samples had no matching BODY prediction"] if unmatched_samples else [],
        "label_import": label_import,
    }


def _full_clip_gate_status(*, run_dir: Path, labels_dir: Path) -> dict[str, Any]:
    candidate_paths = _full_clip_gate_candidate_paths(run_dir=run_dir, labels_dir=labels_dir)
    expected_paths = [str(path) for path in candidate_paths]
    path = next((candidate for candidate in candidate_paths if candidate.is_file()), candidate_paths[0])
    if not path.is_file():
        return {
            "status": "not_measured",
            "path": "",
            "expected_paths": expected_paths,
            "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
            "passed": None,
            "coverage": None,
            "evaluated_frame_count": 0,
            "blockers": ["missing_full_clip_body_gate"],
            "notes": ["no full-clip BODY gate artifact found"],
        }

    payload = _read_json(path)
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        return {
            "status": "fail",
            "path": str(path),
            "expected_paths": expected_paths,
            "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
            "artifact_type": str(payload.get("artifact_type", "")),
            "schema_version": payload.get("schema_version"),
            "passed": None,
            "coverage": payload.get("coverage"),
            "evaluated_frame_count": _non_negative_int(payload.get("evaluated_frame_count")),
            "blockers": ["invalid_full_clip_body_gate"],
            "notes": ["body_full_clip_gate.json must contain boolean passed"],
        }
    return {
        "status": "pass" if passed else "fail",
        "path": str(path),
        "expected_paths": expected_paths,
        "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
        "artifact_type": str(payload.get("artifact_type", "")),
        "schema_version": payload.get("schema_version"),
        "passed": passed,
        "coverage": payload.get("coverage"),
        "evaluated_frame_count": _non_negative_int(payload.get("evaluated_frame_count")),
        "blockers": [] if passed else ["full_clip_body_gate_failed"],
        "notes": [],
    }


def _discover_clips(root_path: Path) -> list[str]:
    if not root_path.is_dir():
        return []
    return sorted(path.name for path in root_path.iterdir() if path.is_dir() and _looks_like_clip_run_dir(path))


def _looks_like_clip_run_dir(path: Path) -> bool:
    return any(
        (path / filename).is_file()
        for filename in (
            "body_mesh_readiness.json",
            "body_compute_execution.json",
            "pipeline_readiness_e2e.json",
            "smpl_motion.json",
            "skeleton3d.json",
        )
    )


def _aggregate_status(clips: list[Mapping[str, Any]]) -> str:
    if any(clip.get("status") == "fail" for clip in clips):
        return "fail"
    if any(clip.get("status") == "blocked" for clip in clips):
        return "blocked"
    if clips and all(clip.get("status") == "pass" for clip in clips):
        return "pass"
    return "not_measured"


def _clip_status(
    mesh_smoke: Mapping[str, Any],
    world_mpjpe: Mapping[str, Any],
    full_clip_gate: Mapping[str, Any],
    blockers: list[str],
) -> str:
    if mesh_smoke.get("status") == "fail" or world_mpjpe.get("status") == "fail" or full_clip_gate.get("status") == "fail":
        return "fail"
    if blockers:
        return "blocked"
    if mesh_smoke.get("status") == "pass" and world_mpjpe.get("status") == "pass" and full_clip_gate.get("status") == "pass":
        return "pass"
    return "not_measured"


def _find_body_label_path(labels_dir: Path) -> Path | None:
    for path in _body_label_candidate_paths(labels_dir):
        if path.is_file():
            return path
    return None


def _body_label_candidate_paths(labels_dir: Path) -> list[Path]:
    labels_subdir = labels_dir / "labels"
    return [
        *[labels_dir / filename for filename in BODY_WORLD_LABEL_FILENAMES],
        *[labels_subdir / filename for filename in BODY_WORLD_LABEL_FILENAMES],
    ]


def _missing_body_label_import_status(labels_dir: Path) -> dict[str, Any]:
    return {
        "status": "missing",
        "path": "",
        "expected_paths": [str(path) for path in _body_label_candidate_paths(labels_dir)],
        "artifact_type": "",
        "payload_status": "",
        "not_ground_truth": None,
        "accepted_sample_count": 0,
        "blockers": ["missing_world_mpjpe_gate"],
        "notes": ["no BODY world-joint labels found"],
    }


def _body_label_import_status(
    *,
    labels_dir: Path,
    label_path: Path,
    payload: Mapping[str, Any],
    accepted_sample_count: int,
) -> dict[str, Any]:
    payload_status = str(payload.get("status", ""))
    not_ground_truth = payload.get("not_ground_truth") is True
    result = {
        "status": "present_reviewed",
        "path": str(label_path),
        "expected_paths": [str(path) for path in _body_label_candidate_paths(labels_dir)],
        "artifact_type": str(payload.get("artifact_type", "")),
        "payload_status": payload_status,
        "not_ground_truth": not_ground_truth,
        "accepted_sample_count": accepted_sample_count,
        "blockers": [],
        "notes": [],
    }
    if not_ground_truth:
        result["status"] = "rejected_not_ground_truth"
        result["blockers"] = ["body_world_labels_not_ground_truth"]
        result["notes"] = ["BODY world-joint labels have not_ground_truth=true; world-MPJPE not measured"]
        return result
    if any(token in payload_status.lower() for token in ("draft", "unverified", "teacher")):
        result["status"] = "rejected_unreviewed_status"
        result["blockers"] = ["body_world_labels_not_reviewed"]
        result["notes"] = [f"BODY world-joint label status is {payload_status!r}; world-MPJPE not measured"]
    return result


def _full_clip_gate_candidate_paths(*, run_dir: Path, labels_dir: Path) -> list[Path]:
    return _unique_paths(
        [
            run_dir / FULL_CLIP_GATE_FILENAME,
            labels_dir / FULL_CLIP_GATE_FILENAME,
            labels_dir / "labels" / FULL_CLIP_GATE_FILENAME,
        ]
    )


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _label_samples(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return []
    samples: list[dict[str, Any]] = []
    for item in raw_samples:
        if not isinstance(item, Mapping) or item.get("accepted", True) is False:
            continue
        frame_index = _maybe_int(item.get("frame_index"))
        player_id = _maybe_int(item.get("player_id"))
        joints = _vectors(item.get("joints_world"))
        if frame_index is None or player_id is None or not joints:
            continue
        samples.append({"frame_index": frame_index, "player_id": player_id, "joints_world": joints})
    return samples


def _prediction_index(
    *,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
    smpl_index = _players_prediction_index(smpl_motion, fps=_fps(smpl_motion))
    if smpl_index:
        return smpl_index
    return _players_prediction_index(skeleton3d, fps=_fps(smpl_motion) or 30.0)


def _players_prediction_index(payload: Mapping[str, Any] | None, *, fps: float) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return {}
    out: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = _maybe_int(player.get("id"))
        frames = player.get("frames")
        if player_id is None or not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = _vectors(frame.get("joints_world"))
            if not joints:
                continue
            frame_index = _maybe_int(frame.get("frame_index"))
            if frame_index is None:
                t = _maybe_float(frame.get("t"))
                if t is None:
                    continue
                frame_index = int(round(t * fps))
            out[(frame_index, player_id)] = joints
    return out


def _joint_errors(
    prediction: list[tuple[float, float, float]],
    label: list[tuple[float, float, float]],
) -> list[float]:
    count = min(len(prediction), len(label))
    return [_distance(prediction[index], label[index]) for index in range(count)]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _smpl_player_frame_count(payload: Mapping[str, Any] | None) -> int:
    return _player_frame_count(payload)


def _skeleton_player_frame_count(payload: Mapping[str, Any] | None) -> int:
    return _player_frame_count(payload)


def _player_frame_count(payload: Mapping[str, Any] | None) -> int:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return 0
    count = 0
    for player in players:
        frames = player.get("frames") if isinstance(player, Mapping) else None
        if isinstance(frames, list):
            count += len(frames)
    return count


def _mapping(payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    value = payload.get(key) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _read_optional_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fps(payload: Mapping[str, Any] | None) -> float:
    value = payload.get("fps") if isinstance(payload, Mapping) else None
    fps = _maybe_float(value)
    return fps if fps and fps > 0 else 30.0


def _vectors(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, list):
        return []
    vectors: list[tuple[float, float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            return []
        vector = tuple(_maybe_float(component) for component in item)
        if any(component is None for component in vector):
            return []
        vectors.append((float(vector[0]), float(vector[1]), float(vector[2])))
    return vectors


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: Any) -> int:
    number = _maybe_int(value)
    return max(0, number) if number is not None else 0


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
