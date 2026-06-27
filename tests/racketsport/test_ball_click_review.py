from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.ball_click_review import (
    build_ball_points_template,
    export_ball_click_review_bundle,
    sample_frame_indices,
    write_review_html,
)


def test_sample_frame_indices_returns_deterministic_even_spread() -> None:
    assert sample_frame_indices(total_frames=10, sample_count=4) == [0, 3, 6, 9]
    assert sample_frame_indices(total_frames=3, sample_count=30) == [0, 1, 2]


def test_build_ball_points_template_has_click_fields() -> None:
    payload = build_ball_points_template(
        clip="clip_a",
        source_video=Path("/tmp/source.mp4"),
        fps=30.0,
        frame_indices=[0, 15],
        image_paths=[Path("images/frame_000000.jpg"), Path("images/frame_000015.jpg")],
    )

    assert payload["artifact_type"] == "racketsport_ball_click_review"
    assert payload["target_file"] == "ball.json"
    assert payload["review_items"] == ["ball_frame_000000", "ball_frame_000015"]
    assert payload["coordinate_frame"] == "image_pixels_video_space"
    assert payload["items"][1]["review_id"] == "ball_frame_000015"
    assert payload["items"][1]["frame_index"] == 15
    assert payload["items"][1]["frame"] == "frame_000015.jpg"
    assert payload["items"][1]["t"] == pytest.approx(0.5)
    assert payload["items"][1]["image"] == "images/frame_000015.jpg"
    assert payload["items"][1]["ball_xy"] is None
    assert payload["items"][1]["xy_px"] is None
    assert payload["items"][1]["visibility"] is None
    assert payload["items"][1]["class_name"] == "sports ball"


def test_write_review_html_embeds_template_and_download_control(tmp_path: Path) -> None:
    payload = build_ball_points_template(
        clip="clip_a",
        source_video=Path("/tmp/source.mp4"),
        fps=30.0,
        frame_indices=[0],
        image_paths=[Path("images/frame_000000.jpg")],
    )

    html_path = tmp_path / "review.html"
    write_review_html(html_path, payload)

    html = html_path.read_text(encoding="utf-8")
    assert "const reviewData =" in html
    assert "download = \"ball_points.json\"" in html
    assert "ball_xy" in html
    assert "xy_px" in html
    assert "Mark missing" in html


def test_write_review_html_supports_keyboard_frame_navigation(tmp_path: Path) -> None:
    payload = build_ball_points_template(
        clip="clip_a",
        source_video=Path("/tmp/source.mp4"),
        fps=30.0,
        frame_indices=[0, 1],
        image_paths=[Path("images/frame_000000.jpg"), Path("images/frame_000001.jpg")],
    )

    html_path = tmp_path / "review.html"
    write_review_html(html_path, payload)

    html = html_path.read_text(encoding="utf-8")
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert 'event.key.toLowerCase() === "a"' in html
    assert 'event.key.toLowerCase() === "d"' in html
    assert "Keyboard: A/Left = previous, D/Right = next" in html


def test_export_ball_click_review_bundle_with_fake_cv2(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"fake")
    cv2 = _FakeCV2(total_frames=8, fps=24.0, width=640, height=360)

    summary = export_ball_click_review_bundle(
        video_path=video_path,
        out_dir=tmp_path / "review",
        clip="clip_a",
        sample_count=4,
        cv2_module=cv2,
    )

    out_dir = tmp_path / "review"
    payload = json.loads((out_dir / "ball_points.json").read_text(encoding="utf-8"))
    assert summary["frame_count"] == 4
    assert summary["frame_indices"] == [0, 2, 5, 7]
    assert payload["items"][2]["image"] == "images/frame_000005.jpg"
    assert (out_dir / "images" / "frame_000005.jpg").is_file()
    assert (out_dir / "review.html").is_file()


class _FakeCapture:
    def __init__(self, cv2: "_FakeCV2") -> None:
        self._cv2 = cv2
        self._position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == self._cv2.CAP_PROP_FRAME_COUNT:
            return float(self._cv2.total_frames)
        if prop == self._cv2.CAP_PROP_FPS:
            return float(self._cv2.fps)
        if prop == self._cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._cv2.width)
        if prop == self._cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._cv2.height)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == self._cv2.CAP_PROP_POS_FRAMES:
            self._position = int(value)
            return True
        return False

    def read(self) -> tuple[bool, object]:
        if self._position < 0 or self._position >= self._cv2.total_frames:
            return False, None
        return True, {"frame_index": self._position}

    def release(self) -> None:
        return None


class _FakeCV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FPS = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(self, *, total_frames: int, fps: float, width: int, height: int) -> None:
        self.total_frames = total_frames
        self.fps = fps
        self.width = width
        self.height = height

    def VideoCapture(self, _: str) -> _FakeCapture:
        return _FakeCapture(self)

    def imwrite(self, path: str, frame: object) -> bool:
        Path(path).write_bytes(f"fake-jpeg:{frame}".encode("utf-8"))
        return True
