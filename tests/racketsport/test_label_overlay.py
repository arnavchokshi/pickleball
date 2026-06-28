from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport import label_overlay
from threed.racketsport.label_overlay import PROTOTYPE_GATE_CLIPS, render_label_overlays, render_prototype_gate


class _FakeFrame:
    shape = (48, 64, 3)


class _FakeCapture:
    def __init__(self, path: str) -> None:
        self.path = path

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return {
            _FakeCv2.CAP_PROP_FPS: 30.0,
            _FakeCv2.CAP_PROP_FRAME_WIDTH: 64.0,
            _FakeCv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        }.get(prop, 0.0)

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        pass


class _FakeWriter:
    def __init__(self, cv2: "_FakeCv2", path: str, fourcc: int, fps: float, size: tuple[int, int]) -> None:
        self.cv2 = cv2
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = 0
        cv2.writers.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: _FakeFrame) -> None:
        self.frames += 1

    def release(self) -> None:
        pass


class _FakeCv2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, readable_paths: set[Path]) -> None:
        self.readable_paths = {str(path) for path in readable_paths}
        self.writers: list[_FakeWriter] = []

    def VideoCapture(self, path: str) -> _FakeCapture:
        return _FakeCapture(path)

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> _FakeWriter:
        return _FakeWriter(self, path, fourcc, fps, size)

    def VideoWriter_fourcc(self, *args: str) -> int:
        return 0

    def imread(self, path: str) -> _FakeFrame | None:
        return _FakeFrame() if path in self.readable_paths else None

    def resize(self, frame: _FakeFrame, size: tuple[int, int]) -> _FakeFrame:
        return frame


class _RecordingCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.rectangles: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
        self.lines: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
        self.circles: list[tuple[tuple[int, int], int, tuple[int, int], int]] = []
        self.text: list[str] = []

    def rectangle(
        self,
        frame: _FakeFrame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        self.rectangles.append((pt1, pt2, color, thickness))

    def line(
        self,
        frame: _FakeFrame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        line_type: int | None = None,
    ) -> None:
        self.lines.append((pt1, pt2, color, thickness))

    def circle(
        self,
        frame: _FakeFrame,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        self.circles.append((center, radius, color, thickness))

    def putText(
        self,
        frame: _FakeFrame,
        text: str,
        origin: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int | None = None,
    ) -> None:
        self.text.append(text)


def _write_frame_pack_label(labels: Path, frames_dir: Path) -> list[Path]:
    labels.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [frames_dir / f"frame_{index:06d}.jpg" for index in range(1, 4)]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"fake image")
    manifest = {
        "schema_version": 1,
        "source_fps": 60.0,
        "source_duration_s": 1.5,
        "sample_every_frames": 30,
        "frame_count": len(frame_paths),
        "frames": [frame_path.name for frame_path in frame_paths],
    }
    manifest_path = frames_dir / "label_frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (labels / "court_corners.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "draft_manual_annotation",
                "frames": {
                    "manifest_path": str(manifest_path),
                    "frame_count": len(frame_paths),
                    "frames": [{"name": frame_path.name, "path": str(frame_path)} for frame_path in frame_paths],
                },
                "annotation": {"target_file": "court_corners.json", "items": []},
            }
        ),
        encoding="utf-8",
    )
    return frame_paths


def _make_video(path: Path, *, frames: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), 30 + index * 8, dtype=np.uint8))
    writer.release()


def _write_draft(path: Path, target: str, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "status": "draft_manual_annotation", "annotation": {"target_file": target, "items": items}}),
        encoding="utf-8",
    )


def _decoded_count(path: Path) -> int:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                return count
            count += 1
    finally:
        cap.release()


def test_render_label_overlays_draws_available_layers_and_index(tmp_path: Path) -> None:
    video = tmp_path / "candidate_001.mp4"
    labels = tmp_path / "labels"
    out_root = tmp_path / "runs" / "eval0" / "prototype_gate"
    _make_video(video)
    _write_draft(labels / "court_corners.json", "court_corners.json", [{"id": "near_left", "xy_px": [8, 40]}])
    _write_draft(labels / "players.json", "players.json", [{"frame": 1, "id": "p1", "bbox": [14, 12, 18, 26]}])
    _write_draft(labels / "ball.json", "ball.json", [{"frame": 0, "xy_px": [18, 20]}, {"frame": 1, "xy_px": [28, 22]}])
    _write_draft(labels / "events.json", "events.json", [{"frame": 1, "type": "contact", "xy_px": [30, 24], "label": "hit"}])
    _write_draft(labels / "racket_pose.json", "racket_pose.json", [{"frame": 1, "keypoints_px": [[32, 20], [40, 24]]}])
    _write_draft(labels / "foot_contact.json", "foot_contact.json", [{"frame": 2, "foot": "left", "xy_px": [22, 38]}])

    summary = render_label_overlays(video_path=video, draft_label_dir=labels, output_root=out_root, clip_name="candidate_001", write_markdown=True)

    compare_dir = out_root / "candidate_001" / "compare"
    overlay_path = compare_dir / "all_labels_overlay.mp4"
    assert summary["status"] == "rendered"
    assert set(summary["available_layers"]) == {"court", "players", "ball", "events", "racket", "foot_contact"}
    assert summary["frame_count"] == 4
    assert overlay_path.stat().st_size > 0
    assert _decoded_count(overlay_path) == 4
    assert (compare_dir / "label_overlay_index.json").is_file()
    assert "all_labels_overlay.mp4" in (compare_dir / "label_overlay_index.md").read_text(encoding="utf-8")


def test_racket_overlay_draws_candidate_bbox_keypoints_and_confidence() -> None:
    fake_cv2 = _RecordingCv2()
    item = {
        "frame": 1,
        "bbox_xyxy": [10.2, 12.1, 42.8, 38.9],
        "keypoints_px": [[14.0, 20.0], [38.0, 32.0]],
        "confidence": 0.755,
        "source": "yolo26m_teacher",
    }

    label_overlay._draw_racket(fake_cv2, _FakeFrame(), 1, [item])

    assert fake_cv2.rectangles == [((10, 12), (43, 39), (255, 180, 0), 2)]
    assert fake_cv2.lines == [((14, 20), (38, 32), (255, 180, 0), 3)]
    assert [circle[0] for circle in fake_cv2.circles] == [(14, 20), (38, 32)]
    assert any("racket 0.76" in text for text in fake_cv2.text)


def test_label_payload_items_map_sampled_frames_and_scale_to_output_resolution() -> None:
    payload = {
        "frames": {"sample_every_frames": 30, "source_resolution": [1920, 1080]},
        "annotation": {
            "items": [
                {
                    "frame": "frame_000003.jpg",
                    "source": "yolo26m_teacher",
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
                    "xy_px": [20.0, 30.0],
                    "keypoints_px": [[10.0, 20.0], [40.0, 60.0]],
                    "court_corners": {"near_left": [5.0, 10.0]},
                },
                {
                    "frame": "frame_000003.jpg",
                    "review_id": "foot_contact_smoke_0000",
                    "xy_px": [960.0, 540.0],
                }
            ]
        },
    }

    items = label_overlay._payload_items_for_render(
        payload,
        render_mode="source_video",
        output_size=(1920, 1080),
        frame_pack_size=(960, 540),
    )

    assert items[0]["_render_frame_index"] == 60
    assert items[0]["bbox"] == pytest.approx([20.0, 40.0, 60.0, 80.0])
    assert items[0]["bbox_xyxy"] == pytest.approx([20.0, 40.0, 80.0, 120.0])
    assert items[0]["xy_px"] == pytest.approx([40.0, 60.0])
    assert items[0]["keypoints_px"] == [[20.0, 40.0], [80.0, 120.0]]
    assert items[0]["court_corners"]["near_left"] == pytest.approx([10.0, 20.0])
    assert items[1]["xy_px"] == pytest.approx([960.0, 540.0])

    frame_pack_items = label_overlay._payload_items_for_render(
        payload,
        render_mode="frame_pack",
        output_size=(960, 540),
        frame_pack_size=(960, 540),
    )
    assert frame_pack_items[0]["_render_frame_index"] == 2
    assert frame_pack_items[0]["xy_px"] == pytest.approx([20.0, 30.0])
    assert frame_pack_items[1]["xy_px"] == pytest.approx([480.0, 270.0])


def test_render_label_overlays_tolerates_sparse_labels_and_cli_defaults(tmp_path: Path) -> None:
    video = tmp_path / "data" / "testclips" / PROTOTYPE_GATE_CLIPS[0] / "source.mp4"
    labels = tmp_path / "runs" / "eval0" / "prototype_gate" / PROTOTYPE_GATE_CLIPS[0] / "labels"
    _make_video(video, frames=2)
    _write_draft(labels / "ball.json", "ball.json", [{"frame": 0, "xy_px": [20, 18]}])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_label_overlays.py",
            "--root",
            str(tmp_path),
            "--clip",
            PROTOTYPE_GATE_CLIPS[0],
            "--markdown",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    clip_summary = summary["clips"][0]
    assert clip_summary["available_layers"] == ["ball"]
    assert clip_summary["qualitative_status"] == "prototype_not_gate_verified"
    assert (tmp_path / "runs" / "eval0" / "prototype_gate" / PROTOTYPE_GATE_CLIPS[0] / "compare" / "all_labels_overlay.mp4").is_file()


def test_frame_pack_fallback_uses_source_sampling_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = tmp_path / "labels"
    frame_paths = _write_frame_pack_label(labels, tmp_path / "label_frames" / "candidate_001")
    fake_cv2 = _FakeCv2(set(frame_paths))
    monkeypatch.setattr("threed.racketsport.label_overlay._cv2", lambda: fake_cv2)

    summary = render_label_overlays(
        video_path=tmp_path / "candidate_001.mp4",
        draft_label_dir=labels,
        output_root=tmp_path / "runs",
        clip_name="candidate_001",
    )

    assert summary["frame_count"] == 3
    assert fake_cv2.writers[-1].fps == pytest.approx(2.0)


def test_frame_pack_resolves_moved_prototype_gate_images(tmp_path: Path) -> None:
    prototype_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2"
    clip = "candidate_001"
    labels = prototype_root / clip / "labels"
    review_images = prototype_root / "review_bundle" / "images" / clip
    labels.mkdir(parents=True)
    review_images.mkdir(parents=True)
    frame_paths = [review_images / f"frame_{index:06d}.jpg" for index in range(1, 3)]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"fake image")
    (labels / "ball.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": {
                    "manifest_path": f"runs/label_frames/{clip}/label_frame_manifest.json",
                    "sample_every_frames": 30,
                    "source_fps": 25,
                    "frames": [
                        {"name": frame_path.name, "path": f"runs/label_frames/{clip}/{frame_path.name}"}
                        for frame_path in frame_paths
                    ],
                },
                "annotation": {"items": []},
            }
        ),
        encoding="utf-8",
    )

    resolved, metadata = label_overlay._frame_pack_from_labels(labels)

    assert resolved == frame_paths
    assert metadata["sample_every_frames"] == 30


def test_frame_pack_render_uses_sparse_sample_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = tmp_path / "labels"
    frames_dir = tmp_path / "label_frames" / "candidate_001"
    frame_paths = _write_frame_pack_label(labels, frames_dir)
    keep_paths = [frame_paths[0], frame_paths[2]]
    label_path = labels / "court_corners.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    payload["frames"]["frames"] = [
        {"name": keep_paths[0].name, "path": str(keep_paths[0])},
        {"name": keep_paths[1].name, "path": str(keep_paths[1])},
    ]
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_cv2 = _FakeCv2(set(keep_paths))
    seen_indices: list[int] = []
    monkeypatch.setattr("threed.racketsport.label_overlay._cv2", lambda: fake_cv2)
    monkeypatch.setattr(label_overlay, "_draw_layers", lambda cv2, frame, frame_index, data: seen_indices.append(frame_index))

    summary = render_label_overlays(
        video_path=tmp_path / "missing_long_source.mp4",
        draft_label_dir=labels,
        output_root=tmp_path / "runs",
        clip_name="candidate_001",
        frame_pack_only=True,
    )

    assert summary["frame_count"] == 2
    assert seen_indices == [0, 2]


def test_frame_pack_only_renders_without_decoding_source_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = tmp_path / "labels"
    frame_paths = _write_frame_pack_label(labels, tmp_path / "label_frames" / "candidate_001")
    fake_cv2 = _FakeCv2(set(frame_paths))
    monkeypatch.setattr("threed.racketsport.label_overlay._cv2", lambda: fake_cv2)

    summary = render_label_overlays(
        video_path=tmp_path / "missing_long_source.mp4",
        draft_label_dir=labels,
        output_root=tmp_path / "runs",
        clip_name="candidate_001",
        frame_pack_only=True,
    )

    assert summary["frame_count"] == 3
    assert len(fake_cv2.writers) == 1
    assert fake_cv2.writers[0].fps == pytest.approx(2.0)


def test_frame_pack_resolves_repo_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "runs" / "eval0" / "prototype_gate" / "candidate_001" / "labels"
    frame_dir = tmp_path / "runs" / "label_frames" / "candidate_001"
    frame_paths = _write_frame_pack_label(labels, frame_dir)
    label_path = labels / "court_corners.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    payload["frames"]["manifest_path"] = "runs/label_frames/candidate_001/label_frame_manifest.json"
    for frame in payload["frames"]["frames"]:
        frame["path"] = f"runs/label_frames/candidate_001/{frame['name']}"
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_cv2 = _FakeCv2({Path(f"runs/label_frames/candidate_001/{path.name}") for path in frame_paths})
    monkeypatch.setattr("threed.racketsport.label_overlay._cv2", lambda: fake_cv2)

    summary = render_label_overlays(
        video_path=tmp_path / "missing_long_source.mp4",
        draft_label_dir=labels,
        output_root=tmp_path / "runs",
        clip_name="candidate_001",
        frame_pack_only=True,
    )

    assert summary["frame_count"] == 3
    assert fake_cv2.writers[0].fps == pytest.approx(2.0)


def test_prototype_gate_uses_source_clips_fallback_for_frame_pack_only_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = PROTOTYPE_GATE_CLIPS[0]
    source = tmp_path / "data" / "source_clips" / f"{clip}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source placeholder")
    labels = tmp_path / "runs" / "eval0" / "prototype_gate" / clip / "labels"
    frame_paths = _write_frame_pack_label(labels, tmp_path / "runs" / "label_frames" / clip)
    fake_cv2 = _FakeCv2(set(frame_paths))
    monkeypatch.setattr("threed.racketsport.label_overlay._cv2", lambda: fake_cv2)

    summary = render_prototype_gate(root=tmp_path, clips=[clip], frame_pack_only=True)

    clip_summary = summary["clips"][0]
    assert clip_summary["status"] == "rendered"
    assert clip_summary["video_path"] == str(source)
    assert clip_summary["frame_count"] == 3
    assert len(fake_cv2.writers) == 1
    assert fake_cv2.writers[0].fps == pytest.approx(2.0)
