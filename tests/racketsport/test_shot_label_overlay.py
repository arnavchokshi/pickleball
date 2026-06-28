import json
import subprocess
import sys

import cv2
import numpy as np


def test_render_shot_label_overlay_writes_review_video(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    assert writer.isOpened()
    for index in range(12):
        frame = np.full((180, 320, 3), 255 - index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    shots = tmp_path / "shots.json"
    shots.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_shot_classification",
                "clip_id": "clip_overlay",
                "classifier": {"name": "shot_transfer_baseline_v1", "not_gate_verified": True},
                "shots": [
                    {"id": "shot_0000", "t": 0.5, "frame": 5, "type": "dink", "type_conf": 0.82},
                ],
                "summary": {"shot_count": 1, "unknown_count": 0, "known_count": 1},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "overlay.mp4"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_shot_label_overlay.py",
            "--video",
            str(source),
            "--shots",
            str(shots),
            "--out",
            str(out),
        ],
        check=True,
    )

    assert out.is_file()
    capture = cv2.VideoCapture(str(out))
    assert capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    capture.release()
    assert frame_count == 12
    assert ok
    assert frame.mean() > 0
