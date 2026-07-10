#!/usr/bin/env python3
"""Run the CPU-only gold-capture tooling rehearsal with generated sync clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sync_fixture(path: Path, *, flash_frame: int, fps: float = 120.0) -> None:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    size = (96, 64)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create synthetic fixture {path}")
    for frame_index in range(30):
        value = 235 if flash_frame <= frame_index < flash_frame + 4 else 18
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()


def _write_charuco_fixture(path: Path) -> None:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    board_image = board.generateImage((300, 420), marginSize=0, borderBits=1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (640, 480))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create ChArUco compatibility fixture {path}")
    for index in range(10):
        frame = np.full((480, 640), 255, dtype=np.uint8)
        x = 170 + (index % 3 - 1) * 3
        y = 30 + (index % 2) * 2
        frame[y : y + 420, x : x + 300] = board_image
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _production_board_qa(*, pdf: Path, temp_dir: Path) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    pdftoppm = shutil.which("pdftoppm")
    pdfinfo = shutil.which("pdfinfo")
    if pdftoppm is None or pdfinfo is None:
        return {"status": "fail", "reason": "Poppler pdftoppm/pdfinfo is required for printable QA"}
    info = subprocess.run([pdfinfo, str(pdf)], check=True, capture_output=True, text=True).stdout
    prefix = temp_dir / "production_board_render"
    subprocess.run([pdftoppm, "-singlefile", "-png", "-r", "100", str(pdf), str(prefix)], check=True)
    image = cv2.imread(str(prefix.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"status": "fail", "reason": "rendered board PNG could not be decoded"}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    _corners, ids, _marker_corners, marker_ids = detector.detectBoard(image)
    charuco_corner_count = 0 if ids is None else int(len(ids))
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    passed = "A3" in info and charuco_corner_count == 24 and marker_count == 17
    return {
        "status": "pass" if passed else "fail",
        "page_is_a3": "A3" in info,
        "render_dpi": 100,
        "render_size_px": [int(image.shape[1]), int(image.shape[0])],
        "detected_charuco_corners": charuco_corner_count,
        "detected_aruco_markers": marker_count,
    }


def rehearse(*, package_root: Path, output: Path) -> dict[str, Any]:
    schema_dir = package_root / "schemas" / "v1"
    template_dir = package_root / "templates" / "v1"
    production_board = package_root / "charuco" / "charuco_a3_5x7_40mm_square_30mm_marker.pdf"
    production_spec = package_root / "charuco" / "charuco_a3_5x7_40mm_square_30mm_marker.spec.json"
    if not production_board.is_file() or not production_spec.is_file():
        raise ValueError("production board and spec must exist before rehearsal")

    with tempfile.TemporaryDirectory(prefix="gold_capture_rehearsal_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        clips = [temp_dir / f"camera_{index}.avi" for index in range(3)]
        for clip in clips:
            _write_sync_fixture(clip, flash_frame=12)
        charuco_fixture = temp_dir / "charuco_compatibility.avi"
        _write_charuco_fixture(charuco_fixture)
        commands = {
            "board": _run(
                [
                    sys.executable,
                    "scripts/racketsport/gold_capture/make_charuco_board.py",
                    "--output",
                    str(temp_dir / "board.pdf"),
                    "--spec-output",
                    str(temp_dir / "board.spec.json"),
                    "--dpi",
                    "72",
                ]
            ),
            "sync": _run(
                [
                    sys.executable,
                    "scripts/racketsport/gold_capture/verify_sync.py",
                    *[str(clip) for clip in clips],
                    "--gate-fps",
                    "120",
                    "--output",
                    str(temp_dir / "sync_report.json"),
                ]
            ),
            "charuco_check": _run(
                [
                    sys.executable,
                    "scripts/racketsport/gold_capture/check_charuco_clip.py",
                    "--video",
                    str(charuco_fixture),
                    "--output",
                    str(temp_dir / "charuco_check.json"),
                ]
            ),
            "schemas": _run(
                [
                    sys.executable,
                    "scripts/racketsport/gold_capture/validate_label_package.py",
                    "--schema-dir",
                    str(schema_dir),
                    "--template-dir",
                    str(template_dir),
                    "--output",
                    str(temp_dir / "schema_report.json"),
                ]
            ),
        }
        production_board_qa = _production_board_qa(pdf=production_board, temp_dir=temp_dir)
        failures = [name for name, result in commands.items() if result["returncode"] != 0]
        if production_board_qa["status"] != "pass":
            failures.append("production_board_qa")
        sync_report = json.loads((temp_dir / "sync_report.json").read_text(encoding="utf-8")) if not failures else None
        schema_report = json.loads((temp_dir / "schema_report.json").read_text(encoding="utf-8")) if not failures else None
        charuco_compatibility = json.loads((temp_dir / "charuco_check.json").read_text(encoding="utf-8")) if not failures else None
        report = {
            "schema_version": 1,
            "artifact_type": "gold_capture_dry_run_report",
            "status": "pass" if not failures else "fail",
            "synthetic_fixture": {
                "camera_count": 3,
                "fps": 120.0,
                "frame_count": 30,
                "flash_frame": 12,
                "ephemeral": True,
            },
            "production_board": {
                "path": production_board.as_posix(),
                "sha256": _sha256(production_board),
                "spec_path": production_spec.as_posix(),
                "render_qa": production_board_qa,
            },
            "repo_charuco_compatibility": charuco_compatibility,
            "commands": commands,
            "sync_gate_pass": sync_report["gate_pass"] if sync_report is not None else False,
            "schema_validated_count": schema_report["validated_count"] if schema_report is not None else 0,
            "failures": failures,
            "best_stack_delta": "none - GT tooling only",
            "product_boundary": "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("runs/lanes/ns021_goldcapture_20260709"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = rehearse(package_root=args.package_root, output=args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
