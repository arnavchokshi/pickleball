from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.racketsport.run_sam3dbody_batch import _parse_requests


def test_parse_requests_preserves_masks_and_static_intrinsics(tmp_path: Path) -> None:
    image = tmp_path / "frame_000001.jpg"
    mask = tmp_path / "frame_000001_player_7.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    payload = {
        "schema_version": 1,
        "requests": [
            {
                "request_id": "1-7",
                "image": str(image),
                "bboxes": [[92.0, 80.0, 212.0, 320.0]],
                "mask_paths": [str(mask)],
                "camera_intrinsics": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                "sam3d_body_input_size_px": 448,
            }
        ],
    }

    requests = _parse_requests(payload)

    assert requests == [
        {
            "request_id": "1-7",
            "image": image,
            "bboxes": [[92.0, 80.0, 212.0, 320.0]],
            "mask_paths": [mask],
            "camera_intrinsics": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            "sam3d_body_input_size_px": 448,
            "target_representation": "world_mesh",
        }
    ]


def test_run_sam3dbody_batch_cli_help_references_phase_c_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/run_sam3dbody_batch.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--requests" in result.stdout
    assert "mask" in result.stdout.lower()
    assert "intrinsics" in result.stdout.lower()
