from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.sam3dbody_probe import parse_bbox_arg, summarize_process_one_image_output


class FakeArray:
    def __init__(self, values, *, dtype: str = "float32", device: str | None = None):
        self._values = values
        self.shape = _shape(values)
        self.dtype = dtype
        if device is not None:
            self.device = device

    def tolist(self):
        return self._values


def _shape(value):
    if isinstance(value, list) and value and isinstance(value[0], list):
        return (len(value), len(value[0]))
    if isinstance(value, list):
        return (len(value),)
    return ()


def test_summarize_process_output_is_json_safe_probe_metadata_only():
    raw = {
        "people": [
            {
                "bbox": [10.0, 20.0, 110.0, 220.0],
                "pred_cam": [0.9, 0.1, -0.1],
                "focal_length": 1200.0,
                "vertices": FakeArray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], device="cuda:0"),
                "joints3d": FakeArray([[1.0, 2.0, 3.0]]),
            },
            {
                "boxes": {"xyxy": [30, 40, 130, 240]},
                "camera": {"translation": FakeArray([0.0, 1.0, 2.0])},
                "focal": FakeArray([800.0]),
            },
        ],
        "timings": {"inference_ms": 41.5},
    }

    payload = summarize_process_one_image_output(
        raw,
        provenance={
            "image_path": "/tmp/frame.jpg",
            "fast_sam_repo": "/opt/fast-sam-3d-body",
            "checkpoint_dir": "/models/sam-3d-body-dinov3",
            "detector_model": "yolo11n.pt",
        },
        requested_bboxes=[[10.0, 20.0, 110.0, 220.0]],
    )

    json.dumps(payload)
    assert payload["artifact_type"] == "racketsport_sam3dbody_probe"
    assert payload["status"] == "probe_only_not_verified"
    assert payload["probe_only"] is True
    assert payload["body_contract"] is False
    assert payload["verified_body_output"] is False
    assert payload["generated_artifacts"] == []
    assert "smpl_motion.json" in " ".join(payload["contract_notes"])
    assert "skeleton3d.json" in " ".join(payload["contract_notes"])
    assert payload["provenance"]["requested_bboxes"] == [[10.0, 20.0, 110.0, 220.0]]

    assert payload["person_count"] == 2
    first = payload["persons"][0]
    assert first["person_index"] == 0
    assert first["keys"] == ["bbox", "focal_length", "joints3d", "pred_cam", "vertices"]
    assert first["selected_fields"]["bbox"] == {
        "source_key": "bbox",
        "value": [10.0, 20.0, 110.0, 220.0],
    }
    assert first["selected_fields"]["camera"] == {
        "source_key": "pred_cam",
        "value": [0.9, 0.1, -0.1],
    }
    assert first["selected_fields"]["focal"] == {
        "source_key": "focal_length",
        "value": 1200.0,
    }
    assert first["array_shapes"]["vertices"] == {
        "kind": "array_like",
        "shape": [2, 3],
        "dtype": "float32",
        "device": "cuda:0",
    }
    assert first["array_shapes"]["joints3d"]["shape"] == [1, 3]

    second = payload["persons"][1]
    assert second["selected_fields"]["bbox"] == {
        "source_key": "boxes.xyxy",
        "value": [30, 40, 130, 240],
    }
    assert second["selected_fields"]["camera"]["source_key"] == "camera"
    assert second["selected_fields"]["camera"]["value"]["translation"]["shape"] == [3]
    assert second["selected_fields"]["focal"]["value"]["value"] == [800.0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,2,3,4", [1.0, 2.0, 3.0, 4.0]),
        (" 1.5, 2.5 , 3.5, 4.5 ", [1.5, 2.5, 3.5, 4.5]),
    ],
)
def test_parse_bbox_arg_accepts_xyxy_csv(text, expected):
    assert parse_bbox_arg(text) == expected


def test_parse_bbox_arg_rejects_bad_bbox():
    with pytest.raises(ValueError, match="x1,y1,x2,y2"):
        parse_bbox_arg("1,2,3")


def test_probe_cli_fails_closed_when_runtime_paths_are_missing(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    out = tmp_path / "probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_sam3dbody_probe.py",
            "--image",
            str(image),
            "--out",
            str(out),
            "--bbox",
            "1,2,3,4",
            "--fast-sam-repo",
            str(tmp_path / "missing-repo"),
            "--checkpoint-dir",
            str(tmp_path / "missing-checkpoints"),
            "--detector-model",
            "yolo11n.pt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 66
    assert "missing FastSAM-3D-Body repo" in completed.stderr
    assert not out.exists()


def test_probe_cli_uses_setup_sam3dbody_runtime_and_writes_probe_json(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    repo = tmp_path / "fast-sam"
    notebook = repo / "notebook"
    notebook.mkdir(parents=True)
    (notebook / "__init__.py").write_text("", encoding="utf-8")
    (notebook / "utils.py").write_text(
        """
class FakeEstimator:
    def process_one_image(self, img, bboxes=None, use_mask=False, hand_box_source="body_decoder"):
        return {
            "people": [
                {
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "focal_length": 900.0,
                    "process_args": {
                        "img": img,
                        "bboxes_type": type(bboxes).__name__,
                        "use_mask": use_mask,
                        "hand_box_source": hand_box_source,
                    },
                }
            ]
        }


def setup_sam_3d_body(**kwargs):
    estimator = FakeEstimator()
    estimator.setup_kwargs = kwargs
    return estimator
""",
        encoding="utf-8",
    )
    out = tmp_path / "probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_sam3dbody_probe.py",
            "--image",
            str(image),
            "--out",
            str(out),
            "--bbox",
            "1,2,3,4",
            "--fast-sam-repo",
            str(repo),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--detector-model",
            "yolo11n.pt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert str(out) in completed.stdout
    assert payload["artifact_type"] == "racketsport_sam3dbody_probe"
    assert payload["person_count"] == 1
    assert payload["verified_body_output"] is False
    assert payload["provenance"]["runtime_setup_function"] == "notebook.utils.setup_sam_3d_body"
    assert payload["provenance"]["runtime_function"].endswith("FakeEstimator.process_one_image")
    assert payload["provenance"]["requested_bboxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert payload["persons"][0]["selected_fields"]["bbox"]["value"] == [1.0, 2.0, 3.0, 4.0]
