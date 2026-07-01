"""M7 on-device CoreML gate report for BALL-only tracking."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import BallTrack, validate_artifact_file


M7_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M7_STATUS_SCAFFOLD = "SCAFFOLD"
M7_STATUS_NOT_STARTED = "NOT-STARTED"
MIN_ON_DEVICE_FPS = 30.0
MIN_RECALL_VS_OFFLINE = 0.85
MIN_HEATMAP_THRESHOLD = 0.50
MAX_GAP_FILL_FRAMES = 3
MIN_RALLY_START_FRAMES = 5
MIN_RALLY_END_EMPTY_S = 0.8
EXPECTED_STACK_FRAMES = 3
EXPECTED_INPUT_SIZE = (512, 288)
MAX_RECALL_CLAIM_DELTA = 0.02


def build_ball_on_device_gate_report(
    *,
    offline_ball_track_path: str | Path,
    coreml_manifest_path: str | Path | None = None,
    device_metrics_path: str | Path | None = None,
    on_device_ball_track_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M7 report for on-device BALL evidence."""

    offline_path = Path(offline_ball_track_path)
    offline_track = validate_artifact_file("ball_track", offline_path)
    if not isinstance(offline_track, BallTrack):
        raise ValueError(f"{offline_path} did not validate as BallTrack")

    coreml_manifest = _load_optional_json(coreml_manifest_path)
    device_metrics = _load_optional_json(device_metrics_path)
    on_device_track = _load_optional_ball_track(on_device_ball_track_path)

    coreml = _coreml_summary(coreml_manifest, manifest_path=coreml_manifest_path)
    metrics = _device_metrics_summary(
        device_metrics,
        offline_ball_track_path=offline_path,
        coreml_manifest_path=Path(coreml_manifest_path) if coreml_manifest_path is not None else None,
        on_device_ball_track_path=Path(on_device_ball_track_path) if on_device_ball_track_path is not None else None,
    )
    recall = _recall_summary(offline_track, on_device_track, metrics["summary"].get("recall_vs_offline"))
    rally_spans = _rally_span_summary(device_metrics)
    status = _status(coreml_manifest=coreml_manifest, device_metrics=device_metrics, on_device_track=on_device_track)

    violations: list[str] = []
    _extend_unique(violations, coreml["violations"])
    _extend_unique(violations, metrics["violations"])
    _extend_unique(violations, recall["violations"])
    _extend_unique(violations, rally_spans["violations"])
    if status != M7_STATUS_TESTED:
        _extend_unique(violations, ["m7_not_tested_on_real_device"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_on_device_gate_report",
        "milestone": "M7 On-device",
        "status": status,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_on_device_gate_failed" if violations else None,
        "offline_ball_track_path": str(offline_path),
        "offline_source": offline_track.source,
        "fps": float(offline_track.fps),
        "frame_count": len(offline_track.frames),
        "required_thresholds": {
            "min_on_device_fps": MIN_ON_DEVICE_FPS,
            "min_recall_vs_offline": MIN_RECALL_VS_OFFLINE,
            "min_heatmap_threshold": MIN_HEATMAP_THRESHOLD,
            "max_gap_fill_frames": MAX_GAP_FILL_FRAMES,
            "min_rally_start_consecutive_frames": MIN_RALLY_START_FRAMES,
            "min_rally_end_empty_s": MIN_RALLY_END_EMPTY_S,
            "expected_stack_frames": EXPECTED_STACK_FRAMES,
            "expected_input_size": list(EXPECTED_INPUT_SIZE),
            "max_recall_claim_delta": MAX_RECALL_CLAIM_DELTA,
        },
        "coreml": coreml["summary"],
        "device_metrics": metrics["summary"],
        "recall": recall["summary"],
        "rally_spans": rally_spans["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_on_device_gate_report(
    *,
    offline_ball_track_path: str | Path,
    out: str | Path,
    coreml_manifest_path: str | Path | None = None,
    device_metrics_path: str | Path | None = None,
    on_device_ball_track_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_on_device_gate_report(
        offline_ball_track_path=offline_ball_track_path,
        coreml_manifest_path=coreml_manifest_path,
        device_metrics_path=device_metrics_path,
        on_device_ball_track_path=on_device_ball_track_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _coreml_summary(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    if manifest is None:
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "model_path": None,
                "model_exists": False,
                "model_format": None,
                "quantization": None,
                "target_compute_unit": None,
                "frames_per_stack": None,
                "input_size": None,
                "heatmap_threshold": None,
                "distilled_from": None,
                "model_sha256": None,
                "model_sha256_verified": False,
                "confidence_source": None,
                "conversion_command": None,
            },
            "violations": ["missing_coreml_manifest"],
        }

    violations: list[str] = []
    artifact_type = manifest.get("artifact_type")
    model_path = manifest.get("model_path")
    model_format = manifest.get("model_format")
    quantization = manifest.get("quantization")
    target_compute_unit = manifest.get("target_compute_unit")
    frames_per_stack = _int_or_none(manifest.get("frames_per_stack"))
    input_size = _input_size(manifest.get("input_size"))
    heatmap_threshold = _finite_or_none(manifest.get("heatmap_threshold"))
    distilled_from = manifest.get("distilled_from")
    model_sha256 = manifest.get("model_sha256")
    confidence_source = manifest.get("confidence_source")
    conversion_command = manifest.get("conversion_command")
    resolved_model_path = _resolve_model_path(model_path, manifest_path=manifest_path)
    model_exists = resolved_model_path.exists() if resolved_model_path is not None else False
    model_sha256_verified = False

    if artifact_type != "racketsport_ball_on_device_coreml_manifest":
        violations.append("coreml_manifest_artifact_type_invalid")
    if model_format != "coreml":
        violations.append("model_format_not_coreml")
    if quantization != "int8":
        violations.append("model_not_int8")
    if str(target_compute_unit).upper() != "ANE":
        violations.append("target_compute_unit_not_ane")
    if frames_per_stack != EXPECTED_STACK_FRAMES:
        violations.append("frames_per_stack_not_3")
    if input_size != EXPECTED_INPUT_SIZE:
        violations.append("input_size_not_288x512_tier_a")
    if heatmap_threshold is None or heatmap_threshold < MIN_HEATMAP_THRESHOLD:
        violations.append("heatmap_threshold_below_0_50")
    if not isinstance(distilled_from, str) or not distilled_from.strip():
        violations.append("distillation_source_missing")
    if not model_exists:
        violations.append("coreml_model_path_missing")
    if not isinstance(model_sha256, str) or not model_sha256:
        violations.append("coreml_model_sha256_missing")
    elif model_exists and resolved_model_path is not None:
        actual_sha256 = _sha256_path(resolved_model_path)
        model_sha256_verified = actual_sha256 == model_sha256
        if not model_sha256_verified:
            violations.append("coreml_model_sha256_mismatch")
    if confidence_source != "heatmap_peak":
        violations.append("coreml_confidence_source_not_heatmap_peak")
    if not isinstance(conversion_command, str) or not conversion_command:
        violations.append("coreml_conversion_command_missing")

    return {
        "summary": {
            "path_present": True,
            "artifact_type": artifact_type,
            "model_path": str(model_path) if model_path is not None else None,
            "model_exists": model_exists,
            "model_format": model_format,
            "quantization": quantization,
            "target_compute_unit": target_compute_unit,
            "frames_per_stack": frames_per_stack,
            "input_size": list(input_size) if input_size is not None else None,
            "heatmap_threshold": heatmap_threshold,
            "distilled_from": distilled_from,
            "model_sha256": model_sha256,
            "model_sha256_verified": model_sha256_verified,
            "confidence_source": confidence_source,
            "conversion_command": conversion_command,
        },
        "violations": sorted(set(violations)),
    }


def _device_metrics_summary(
    metrics: Mapping[str, Any] | None,
    *,
    offline_ball_track_path: Path,
    coreml_manifest_path: Path | None,
    on_device_ball_track_path: Path | None,
) -> dict[str, Any]:
    if metrics is None:
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "tested_on_real_device": None,
                "device_name": None,
                "backend": None,
                "fps": None,
                "recall_vs_offline": None,
                "gap_fill_max_frames": None,
                "rally_start_min_consecutive_frames": None,
                "rally_end_empty_s": None,
                "offline_ball_track_path": None,
                "coreml_manifest_path": None,
                "on_device_ball_track_path": None,
                "confidence_source": None,
                "measurement_command": None,
            },
            "violations": ["missing_device_metrics"],
        }

    violations: list[str] = []
    artifact_type = metrics.get("artifact_type")
    tested_on_real_device = metrics.get("tested_on_real_device")
    device_name = metrics.get("device_name")
    backend = metrics.get("backend")
    fps = _finite_or_none(metrics.get("fps"))
    recall = _finite_or_none(metrics.get("recall_vs_offline"))
    gap_fill = _int_or_none(metrics.get("gap_fill_max_frames"))
    rally_start = _int_or_none(metrics.get("rally_start_min_consecutive_frames"))
    rally_end = _finite_or_none(metrics.get("rally_end_empty_s"))
    metrics_offline_path = metrics.get("offline_ball_track_path")
    metrics_coreml_path = metrics.get("coreml_manifest_path")
    metrics_on_device_path = metrics.get("on_device_ball_track_path")
    confidence_source = metrics.get("confidence_source")
    measurement_command = metrics.get("measurement_command")

    if artifact_type != "racketsport_ball_on_device_metrics":
        violations.append("device_metrics_artifact_type_invalid")
    if tested_on_real_device is not True:
        violations.append("device_metrics_not_from_real_device")
    if not isinstance(device_name, str) or not device_name.strip():
        violations.append("device_name_missing")
    if str(backend).lower() not in {"coreml", "ane", "coreml_ane"}:
        violations.append("device_backend_not_coreml_ane")
    if fps is None:
        violations.append("on_device_fps_missing")
    elif fps < MIN_ON_DEVICE_FPS:
        violations.append("on_device_fps_below_30")
    if recall is None:
        violations.append("reported_recall_vs_offline_missing")
    elif recall < MIN_RECALL_VS_OFFLINE:
        violations.append("reported_recall_vs_offline_below_0_85")
    if gap_fill is None:
        violations.append("gap_fill_max_frames_missing")
    elif gap_fill > MAX_GAP_FILL_FRAMES:
        violations.append("gap_fill_over_3_frames")
    if rally_start is None:
        violations.append("rally_start_rule_missing")
    elif rally_start < MIN_RALLY_START_FRAMES:
        violations.append("rally_start_rule_below_5_frames")
    if rally_end is None:
        violations.append("rally_end_rule_missing")
    elif rally_end < MIN_RALLY_END_EMPTY_S:
        violations.append("rally_end_empty_below_0_8s")
    if not isinstance(metrics_offline_path, str) or not metrics_offline_path:
        violations.append("device_metrics_offline_track_missing")
    elif not _paths_match(metrics_offline_path, offline_ball_track_path):
        violations.append("device_metrics_offline_track_mismatch")
    if not isinstance(metrics_coreml_path, str) or not metrics_coreml_path:
        violations.append("device_metrics_coreml_manifest_missing")
    elif coreml_manifest_path is None or not _paths_match(metrics_coreml_path, coreml_manifest_path):
        violations.append("device_metrics_coreml_manifest_mismatch")
    if not isinstance(metrics_on_device_path, str) or not metrics_on_device_path:
        violations.append("device_metrics_on_device_track_missing")
    elif on_device_ball_track_path is None or not _paths_match(metrics_on_device_path, on_device_ball_track_path):
        violations.append("device_metrics_on_device_track_mismatch")
    if confidence_source != "heatmap_peak":
        violations.append("device_metrics_confidence_source_not_heatmap_peak")
    if not isinstance(measurement_command, str) or not measurement_command:
        violations.append("device_metrics_measurement_command_missing")

    return {
        "summary": {
            "path_present": True,
            "artifact_type": artifact_type,
            "tested_on_real_device": tested_on_real_device,
            "device_name": device_name,
            "backend": backend,
            "fps": fps,
            "recall_vs_offline": recall,
            "gap_fill_max_frames": gap_fill,
            "rally_start_min_consecutive_frames": rally_start,
            "rally_end_empty_s": rally_end,
            "offline_ball_track_path": metrics_offline_path,
            "coreml_manifest_path": metrics_coreml_path,
            "on_device_ball_track_path": metrics_on_device_path,
            "confidence_source": confidence_source,
            "measurement_command": measurement_command,
        },
        "violations": sorted(set(violations)),
    }


def _recall_summary(
    offline: BallTrack,
    on_device: BallTrack | None,
    reported_recall: Any,
) -> dict[str, Any]:
    if on_device is None:
        return {
            "summary": {
                "on_device_track_present": False,
                "offline_visible_frame_count": sum(1 for frame in offline.frames if frame.visible),
                "on_device_visible_match_count": 0,
                "computed_recall_vs_offline": None,
                "reported_recall_vs_offline": _finite_or_none(reported_recall),
            },
            "violations": ["missing_on_device_ball_track", "on_device_recall_below_0_85"],
        }

    offline_visible = [idx for idx, frame in enumerate(offline.frames) if frame.visible]
    matches = 0
    for idx in offline_visible:
        if idx < len(on_device.frames) and on_device.frames[idx].visible:
            matches += 1
    computed = _ratio(matches, len(offline_visible))
    reported = _finite_or_none(reported_recall)
    violations: list[str] = []
    if computed is None or computed < MIN_RECALL_VS_OFFLINE:
        violations.append("on_device_recall_below_0_85")
    if reported is None:
        violations.append("reported_recall_vs_offline_missing")
    elif computed is not None and abs(reported - computed) > MAX_RECALL_CLAIM_DELTA:
        violations.append("reported_recall_mismatch")
    return {
        "summary": {
            "on_device_track_present": True,
            "on_device_source": on_device.source,
            "offline_visible_frame_count": len(offline_visible),
            "on_device_visible_match_count": matches,
            "computed_recall_vs_offline": computed,
            "reported_recall_vs_offline": reported,
        },
        "violations": sorted(set(violations)),
    }


def _rally_span_summary(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    spans = metrics.get("rally_spans") if metrics is not None else None
    if not isinstance(spans, list):
        return {
            "summary": {"span_count": 0, "spans": []},
            "violations": ["missing_rally_spans"],
        }
    violations: list[str] = []
    parsed: list[dict[str, Any]] = []
    if not spans:
        violations.append("missing_rally_spans")
    for span in spans:
        if not isinstance(span, Mapping):
            violations.append("rally_span_invalid")
            continue
        start_t = _finite_or_none(span.get("start_t"))
        end_t = _finite_or_none(span.get("end_t"))
        padding_s = _finite_or_none(span.get("padding_s"))
        if start_t is None or end_t is None or end_t <= start_t:
            violations.append("rally_span_time_invalid")
        parsed.append({"start_t": start_t, "end_t": end_t, "padding_s": padding_s})
    return {"summary": {"span_count": len(parsed), "spans": parsed}, "violations": sorted(set(violations))}


def _status(
    *,
    coreml_manifest: Mapping[str, Any] | None,
    device_metrics: Mapping[str, Any] | None,
    on_device_track: BallTrack | None,
) -> str:
    if (
        device_metrics is not None
        and device_metrics.get("tested_on_real_device") is True
        and coreml_manifest is not None
        and on_device_track is not None
    ):
        return M7_STATUS_TESTED
    if coreml_manifest is not None or device_metrics is not None or on_device_track is not None:
        return M7_STATUS_SCAFFOLD
    return M7_STATUS_NOT_STARTED


def _load_optional_ball_track(path: str | Path | None) -> BallTrack | None:
    if path is None:
        return None
    track_path = Path(path)
    if not track_path.is_file():
        return None
    artifact = validate_artifact_file("ball_track", track_path)
    if not isinstance(artifact, BallTrack):
        raise ValueError(f"{track_path} did not validate as BallTrack")
    return artifact


def _load_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{json_path} must contain a JSON object")
    return payload


def _resolve_model_path(model_path: Any, *, manifest_path: str | Path | None) -> Path | None:
    if not isinstance(model_path, str) or not model_path:
        return None
    path = Path(model_path)
    if path.is_absolute():
        return path
    if manifest_path is not None:
        manifest_parent = Path(manifest_path).parent
        candidate = manifest_parent / path
        if candidate.exists():
            return candidate
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if len(files) == 1:
            with files[0].open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        for item in files:
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_size(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    left = _int_or_none(value[0])
    right = _int_or_none(value[1])
    if left is None or right is None:
        return None
    return (left, right)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _finite_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_like(value) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _paths_match(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


__all__ = [
    "M7_STATUS_NOT_STARTED",
    "M7_STATUS_SCAFFOLD",
    "M7_STATUS_TESTED",
    "build_ball_on_device_gate_report",
    "write_ball_on_device_gate_report",
]
