from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_on_device_gate import build_ball_on_device_gate_report


def _ball_track_payload(*, source: str, visible_count: int = 10) -> dict[str, object]:
    frames = []
    for idx in range(10):
        visible = idx < visible_count
        frames.append(
            {
                "t": idx / 60.0,
                "xy": [100.0 + idx, 200.0 + idx],
                "conf": 0.86 if visible else 0.20,
                "visible": visible,
            }
        )
    return {"schema_version": 1, "fps": 60.0, "source": source, "frames": frames, "bounces": []}


def _coreml_manifest_payload(
    model_path: str,
    *,
    model_sha256: str | None = None,
    confidence_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_on_device_coreml_manifest",
        "model_name": "tracknet_nano_int8",
        "model_path": model_path,
        "model_format": "coreml",
        "quantization": "int8",
        "target_compute_unit": "ANE",
        "frames_per_stack": 3,
        "input_size": [512, 288],
        "heatmap_threshold": 0.50,
        "distilled_from": "TrackNetV3",
        "conversion_command": "python scripts/racketsport/export_ball_coreml.py --int8 --ane",
    }
    if model_sha256 is not None:
        payload["model_sha256"] = model_sha256
    if confidence_source is not None:
        payload["confidence_source"] = confidence_source
    return payload


def _device_metrics_payload(
    *,
    offline_ball_track_path: str | None = None,
    coreml_manifest_path: str | None = None,
    on_device_ball_track_path: str | None = None,
    confidence_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_on_device_metrics",
        "tested_on_real_device": True,
        "device_name": "iPhone test fixture",
        "backend": "coreml_ane",
        "fps": 34.2,
        "recall_vs_offline": 0.90,
        "gap_fill_max_frames": 3,
        "rally_start_min_consecutive_frames": 5,
        "rally_end_empty_s": 0.8,
        "rally_spans": [{"start_t": 0.0, "end_t": 0.5, "padding_s": 0.5}],
        "measurement_command": "xcodebuild test -scheme PickleballCapture -destination id=REAL_DEVICE",
    }
    if offline_ball_track_path is not None:
        payload["offline_ball_track_path"] = offline_ball_track_path
    if coreml_manifest_path is not None:
        payload["coreml_manifest_path"] = coreml_manifest_path
    if on_device_ball_track_path is not None:
        payload["on_device_ball_track_path"] = on_device_ball_track_path
    if confidence_source is not None:
        payload["confidence_source"] = confidence_source
    return payload


def test_ball_on_device_gate_passes_int8_coreml_real_device_metrics_and_recall(tmp_path: Path) -> None:
    offline = tmp_path / "offline_ball_track.json"
    on_device = tmp_path / "on_device_ball_track.json"
    model_dir = tmp_path / "models_coreml" / "tracknet_nano_int8.mlpackage"
    manifest = tmp_path / "coreml_manifest.json"
    metrics = tmp_path / "device_metrics.json"
    model_dir.mkdir(parents=True)
    model_payload = model_dir / "Data.bin"
    model_payload.write_bytes(b"fixture coreml package bytes")
    offline.write_text(json.dumps(_ball_track_payload(source="tracknet", visible_count=10)), encoding="utf-8")
    on_device.write_text(json.dumps(_ball_track_payload(source="fused", visible_count=9)), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            _coreml_manifest_payload(
                str(model_dir),
                model_sha256=hashlib.sha256(model_payload.read_bytes()).hexdigest(),
                confidence_source="heatmap_peak",
            )
        ),
        encoding="utf-8",
    )
    metrics.write_text(
        json.dumps(
            _device_metrics_payload(
                offline_ball_track_path=str(offline),
                coreml_manifest_path=str(manifest),
                on_device_ball_track_path=str(on_device),
                confidence_source="heatmap_peak",
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_on_device_gate_report(
        offline_ball_track_path=offline,
        coreml_manifest_path=manifest,
        device_metrics_path=metrics,
        on_device_ball_track_path=on_device,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M7 On-device"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["coreml"]["model_exists"] is True
    assert report["device_metrics"]["fps"] == pytest.approx(34.2)
    assert report["recall"]["computed_recall_vs_offline"] == pytest.approx(0.9)
    assert report["rally_spans"]["span_count"] == 1
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_on_device_gate_reports_not_started_without_m7_artifacts(tmp_path: Path) -> None:
    offline = tmp_path / "offline_ball_track.json"
    offline.write_text(json.dumps(_ball_track_payload(source="tracknet", visible_count=10)), encoding="utf-8")

    report = build_ball_on_device_gate_report(
        offline_ball_track_path=offline,
        coreml_manifest_path=tmp_path / "missing_coreml_manifest.json",
        device_metrics_path=tmp_path / "missing_device_metrics.json",
        on_device_ball_track_path=tmp_path / "missing_on_device_ball_track.json",
    )

    assert report["status"] == "NOT-STARTED"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_on_device_gate_failed"
    assert set(report["violations"]) >= {
        "missing_coreml_manifest",
        "missing_device_metrics",
        "missing_on_device_ball_track",
        "on_device_recall_below_0_85",
        "missing_rally_spans",
    }


def test_ball_on_device_gate_rejects_unbound_device_metrics_and_non_heatmap_confidence(tmp_path: Path) -> None:
    offline = tmp_path / "offline_ball_track.json"
    on_device = tmp_path / "on_device_ball_track.json"
    model = tmp_path / "tracknet_nano_int8.mlmodel"
    manifest = tmp_path / "coreml_manifest.json"
    metrics = tmp_path / "device_metrics.json"
    offline.write_text(json.dumps(_ball_track_payload(source="tracknet", visible_count=10)), encoding="utf-8")
    on_device.write_text(json.dumps(_ball_track_payload(source="fused", visible_count=9)), encoding="utf-8")
    model.write_bytes(b"fixture coreml model bytes")
    manifest.write_text(
        json.dumps(
            _coreml_manifest_payload(
                str(model),
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                confidence_source="visibility_binary",
            )
        ),
        encoding="utf-8",
    )
    metrics.write_text(
        json.dumps(
            _device_metrics_payload(
                offline_ball_track_path=str(tmp_path / "different_offline_ball_track.json"),
                coreml_manifest_path=str(tmp_path / "different_coreml_manifest.json"),
                on_device_ball_track_path=str(tmp_path / "different_on_device_ball_track.json"),
                confidence_source="visibility_binary",
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_on_device_gate_report(
        offline_ball_track_path=offline,
        coreml_manifest_path=manifest,
        device_metrics_path=metrics,
        on_device_ball_track_path=on_device,
    )

    assert report["gate_result"] == "fail"
    assert set(report["violations"]) >= {
        "coreml_confidence_source_not_heatmap_peak",
        "device_metrics_confidence_source_not_heatmap_peak",
        "device_metrics_offline_track_mismatch",
        "device_metrics_coreml_manifest_mismatch",
        "device_metrics_on_device_track_mismatch",
    }


def test_ball_on_device_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    offline = tmp_path / "offline_ball_track.json"
    out = tmp_path / "m7_report.json"
    offline.write_text(json.dumps(_ball_track_payload(source="tracknet", visible_count=10)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_on_device.py",
            "--offline-ball-track",
            str(offline),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["gate_result"] == "fail"
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_on_device_gate_failed"
