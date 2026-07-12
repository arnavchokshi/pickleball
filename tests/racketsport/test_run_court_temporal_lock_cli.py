from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from threed.racketsport.court_temporal_lock import COURT_LINES_M


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts/racketsport/run_court_temporal_lock.py"


def _project(h: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
    projected = h @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    xy = np.rint(projected[:2] / projected[2]).astype(int)
    return int(xy[0]), int(xy[1])


def _fixture(tmp_path: Path, frame_count: int = 12) -> tuple[Path, Path, Path]:
    h = np.asarray([[65.0, 0.0, 320.0], [0.0, 28.0, 210.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    video = tmp_path / "court.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    assert writer.isOpened()
    for _ in range(frame_count):
        frame = np.full((480, 640, 3), (45, 100, 45), dtype=np.uint8)
        for court_a, court_b in COURT_LINES_M.values():
            cv2.line(frame, _project(h, court_a), _project(h, court_b), (245, 245, 245), 4, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    calibration = tmp_path / "court_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "homography": h.tolist(),
                "image_size": [640, 480],
                "source": "reviewed_fixture",
                "intrinsics": {"dist": [0.0, 0.0, 0.0, 0.0]},
            }
        ),
        encoding="utf-8",
    )
    motion = tmp_path / "camera_motion.json"
    motion.write_text(
        json.dumps(
            {
                "reference_frame_idx": 0,
                "summary": {"drift_px_p95": 0.0},
                "frames": [
                    {
                        "frame_idx": idx,
                        "M": np.eye(3).tolist(),
                        "compensated": True,
                        "inlier_ratio": 1.0,
                    }
                    for idx in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return video, calibration, motion


def test_cli_emits_one_immutable_artifact_with_complete_provenance(tmp_path: Path) -> None:
    video, calibration, motion = _fixture(tmp_path)
    output_dir = tmp_path / "output"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(CLI),
        "--video",
        str(video),
        "--court-calibration",
        str(calibration),
        "--camera-motion",
        str(motion),
        "--output-dir",
        str(output_dir),
        "--max-frames",
        "12",
    ]
    environment = dict(os.environ, MPLBACKEND="Agg")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    files = list(output_dir.iterdir())
    assert [path.name for path in files] == ["court_temporal_lock.json"]
    artifact = json.loads(files[0].read_text(encoding="utf-8"))
    assert artifact["summary"]["frame_count"] == 12
    assert artifact["summary"]["explicit_provenance_frame_count"] == 12
    assert all(frame["provenance"]["kind"] in {"measured", "predicted", "missing", "reset"} for frame in artifact["frames"])
    assert artifact["coordinate_contract"]["composition"] == "H_t = inv(M_t) @ H_reference"
    assert artifact["summary"]["best_stack_delta"] == "none"
    assert artifact["source"]["camera_motion_mode"] == "static_degenerate_from_drift_p95"

    repeated = subprocess.run(command, cwd=REPO_ROOT, env=environment, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite immutable artifact" in repeated.stderr


def test_cli_rejects_more_than_bounded_three_hundred_frames(tmp_path: Path) -> None:
    video, calibration, _motion = _fixture(tmp_path, frame_count=1)
    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(CLI),
            "--video",
            str(video),
            "--court-calibration",
            str(calibration),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-frames",
            "301",
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ, MPLBACKEND="Agg"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "within 1..300" in completed.stderr
