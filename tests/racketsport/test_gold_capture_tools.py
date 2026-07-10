from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest


PACKAGE_ROOT = Path("runs/lanes/ns021_goldcapture_20260709")


def _write_flash_clip(path: Path, *, flash_frame: int, fps: float = 120.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
    assert writer.isOpened()
    for frame_index in range(30):
        level = 235 if flash_frame <= frame_index < flash_frame + 4 else 18
        writer.write(np.full((64, 96, 3), level, dtype=np.uint8))
    writer.release()


def _write_clap_wav(path: Path, *, clap_sample: int, sample_rate: int = 48_000) -> None:
    samples = np.zeros(sample_rate, dtype=np.int16)
    samples[clap_sample : clap_sample + 48] = 28_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_make_charuco_board_direct_cli_writes_fixed_a3_pdf_and_spec(tmp_path: Path) -> None:
    pdf = tmp_path / "board.pdf"
    spec_path = tmp_path / "board.spec.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/make_charuco_board.py",
            "--output",
            str(pdf),
            "--spec-output",
            str(spec_path),
            "--dpi",
            "72",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert spec["board"] == {
        "border_bits": 1,
        "dictionary": "DICT_4X4_50",
        "marker_length_mm": 30.0,
        "marker_to_square_ratio": 0.75,
        "printed_height_mm": 280.0,
        "printed_width_mm": 200.0,
        "square_length_mm": 40.0,
        "squares_x": 5,
        "squares_y": 7,
    }
    assert spec["repo_compatibility"]["calibration_tool"] == "scripts/racketsport/calibrate_charuco_device.py"
    assert spec["output_pdf_sha256"] == summary["sha256"]
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
        assert "Page size:" in info
        assert "A3" in info


def test_verify_sync_direct_cli_passes_aligned_three_clip_fixture_and_fails_one_frame_offset(tmp_path: Path) -> None:
    aligned = [tmp_path / f"aligned_{index}.avi" for index in range(3)]
    for path in aligned:
        _write_flash_clip(path, flash_frame=12)
    pass_report = tmp_path / "pass.json"
    passed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/verify_sync.py",
            *[str(path) for path in aligned],
            "--gate-fps",
            "120",
            "--output",
            str(pass_report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(pass_report.read_text(encoding="utf-8"))
    assert json.loads(passed.stdout)["gate_pass"] is True
    assert payload["gate_pass"] is True
    assert payload["methods"]["led"]["max_pairwise_offset_frames"] == pytest.approx(0.0, abs=1e-6)
    assert all(len(item["immutable_raw_reference"]["sha256"]) == 64 for item in payload["methods"]["led"]["measurements"])

    offset = tmp_path / "offset.avi"
    _write_flash_clip(offset, flash_frame=13)
    fail_report = tmp_path / "fail.json"
    failed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/verify_sync.py",
            str(aligned[0]),
            str(offset),
            "--gate-fps",
            "120",
            "--output",
            str(fail_report),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    failed_payload = json.loads(fail_report.read_text(encoding="utf-8"))
    assert failed_payload["gate_pass"] is False
    assert failed_payload["methods"]["led"]["max_pairwise_offset_frames"] > 0.5

    if shutil.which("ffmpeg"):
        audio_a = tmp_path / "audio_a.wav"
        audio_b = tmp_path / "audio_b.wav"
        _write_clap_wav(audio_a, clap_sample=9_600)
        _write_clap_wav(audio_b, clap_sample=9_600)
        audio_report = tmp_path / "audio.json"
        audio_completed = subprocess.run(
            [
                sys.executable,
                "scripts/racketsport/gold_capture/verify_sync.py",
                str(audio_a),
                str(audio_b),
                "--method",
                "audio",
                "--gate-fps",
                "240",
                "--audio-distance-m",
                "1.0",
                "--audio-distance-m",
                "1.0",
                "--output",
                str(audio_report),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(audio_completed.stdout)["methods"]["audio"]["gate_pass"] is True


def test_check_charuco_clip_direct_cli_uses_locked_contract(tmp_path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    board_image = board.generateImage((300, 420), marginSize=0, borderBits=1)
    video = tmp_path / "charuco.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (640, 480))
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((480, 640), 255, dtype=np.uint8)
        frame[30:450, 170:470] = board_image
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    report_path = tmp_path / "check.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/check_charuco_clip.py",
            "--video",
            str(video),
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["gate_pass"] is True
    assert report["board_contract"]["square_length_m"] == 0.04
    assert report["videos"][0]["detection_frames"] >= 8
    assert report["read_only_calibration_tool_imported"] == "scripts/racketsport/calibrate_charuco_device.py"


def test_validate_label_package_direct_cli_validates_all_templates_and_rejects_candidate_gt(tmp_path: Path) -> None:
    report_path = tmp_path / "schemas.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/validate_label_package.py",
            "--schema-dir",
            str(PACKAGE_ROOT / "schemas" / "v1"),
            "--template-dir",
            str(PACKAGE_ROOT / "templates" / "v1"),
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["validated_count"] == 7
    assert report["candidate_prediction_gt_kill_rule_enforced"] is True

    schema_dir = tmp_path / "schemas"
    template_dir = tmp_path / "templates"
    shutil.copytree(PACKAGE_ROOT / "schemas" / "v1", schema_dir)
    shutil.copytree(PACKAGE_ROOT / "templates" / "v1", template_dir)
    cal_template = template_dir / "cal_points.template.json"
    payload = json.loads(cal_template.read_text(encoding="utf-8"))
    payload["independence"]["candidate_prediction_used"] = True
    cal_template.write_text(json.dumps(payload), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/validate_label_package.py",
            "--schema-dir",
            str(schema_dir),
            "--template-dir",
            str(template_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "expected constant False" in rejected.stderr


def test_rehearse_gold_capture_direct_cli_runs_every_tool_and_repo_charuco_collector(tmp_path: Path) -> None:
    output = tmp_path / "dry_run.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/gold_capture/rehearse_gold_capture.py",
            "--package-root",
            str(PACKAGE_ROOT),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["schema_validated_count"] == 7
    assert report["sync_gate_pass"] is True
    assert report["repo_charuco_compatibility"]["status"] == "pass"
    assert report["repo_charuco_compatibility"]["videos"][0]["detection_frames"] >= 8
    assert report["production_board"]["render_qa"]["status"] == "pass"
    assert report["production_board"]["render_qa"]["detected_charuco_corners"] == 24
    assert all(command["returncode"] == 0 for command in report["commands"].values())
    assert report["best_stack_delta"] == "none - GT tooling only"


def test_owner_package_has_exact_survey_fields_monocular_boundary_and_numbered_actions() -> None:
    boundary = "The product remains monocular; extra cameras, markers, and surveys are GT-only."
    sheet = PACKAGE_ROOT / "survey" / "court_net_survey_recording_sheet.csv"
    with sheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 50
    assert all(row["product_boundary"] == boundary for row in rows)
    assert all(key in rows[0] for key in ("instrument_uncertainty_m", "endpoint_uncertainty_m", "combined_uncertainty_m"))
    features = {row["feature"] for row in rows}
    assert {
        "instrument_precheck",
        "court_width_outer_edges",
        "court_length_outer_edges",
        "full_court_diagonal",
        "nvz_offset",
        "centerline_half_width",
        "line_width",
        "net_post_spacing",
        "net_top_height",
        "surface_plane",
        "camera_station",
    } <= features

    for path in (PACKAGE_ROOT / "README.md", PACKAGE_ROOT / "OWNER_HALF_DAY_CHECKLIST.md"):
        assert boundary in path.read_text(encoding="utf-8")
    checklist_lines = (PACKAGE_ROOT / "OWNER_HALF_DAY_CHECKLIST.md").read_text(encoding="utf-8").splitlines()
    action_numbers = [int(line.split(".", 1)[0]) for line in checklist_lines if line.split(".", 1)[0].isdigit()]
    assert action_numbers == list(range(1, 112))
