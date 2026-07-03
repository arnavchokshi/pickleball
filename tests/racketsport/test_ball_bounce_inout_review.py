from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_bounce_inout_review import (
    build_reviewed_bounce_inout_labels,
    export_ball_bounce_inout_review_bundle,
    write_review_html,
)
from threed.racketsport.ball_bounce_simple_review_server import (
    apply_simple_review_decision,
    build_simple_review_state,
)


def _ball_track_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "fused",
        "frames": [
            {"t": 0.95, "xy": [118.0, 204.0], "conf": 0.71, "visible": True},
            {"t": 1.0, "xy": [120.0, 210.0], "conf": 0.83, "visible": True},
            {"t": 1.05, "xy": [124.0, 205.0], "conf": 0.76, "visible": True},
        ],
        "bounces": [
            {
                "t": 1.0,
                "frame": 60,
                "world_xy": [1.2, 2.4],
                "contact_xy_img": [120.0, 210.0],
                "p_bounce": 0.82,
                "margin_m": -0.18,
                "uncertainty_m": 0.06,
                "confidence": 0.75,
                "call": "out",
                "nearest_line": "left_sideline",
                "dominant_uncertainty_term": "manual_corner_homography_projection",
                "source": "image_velocity_inflection_court_plane_2d_v1",
            }
        ],
    }


def test_export_ball_bounce_inout_review_bundle_writes_fail_closed_template(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    track = tmp_path / "ball_track.json"
    video.write_bytes(b"fake")
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")

    summary = export_ball_bounce_inout_review_bundle(
        video_path=video,
        ball_track_path=track,
        out_dir=tmp_path / "review",
        clip="clip_a",
        context_frames=1,
        cv2_module=_FakeCV2(total_frames=120, fps=60.0, width=640, height=360),
    )

    review = json.loads(Path(summary["review_json"]).read_text(encoding="utf-8"))
    assert summary["artifact_type"] == "racketsport_ball_bounce_inout_review_export"
    assert summary["status"] == "needs_human_review"
    assert review["artifact_type"] == "racketsport_ball_bounce_inout_review"
    assert review["status"] == "needs_human_review"
    assert review["not_ground_truth"] is True
    assert review["items"][0]["predicted_frame"] == 60
    assert review["items"][0]["predicted_call"] == "out"
    assert review["items"][0]["reviewed_bounce_frame"] is None
    assert review["items"][0]["reviewed_call"] is None
    assert [image["frame"] for image in review["items"][0]["context_images"]] == [59, 60, 61]
    assert (tmp_path / "review" / "images" / "bounce_0000_frame_000060.jpg").is_file()


def test_build_reviewed_bounce_inout_labels_requires_human_reviewed_items() -> None:
    review = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_inout_review",
        "clip": "clip_a",
        "fps": 60.0,
        "items": [
            {
                "review_id": "bounce_0000",
                "status": "accepted",
                "reviewed_bounce_frame": 60,
                "reviewed_call": "out",
            },
            {
                "review_id": "bounce_0001",
                "status": "needs_human_review",
                "reviewed_bounce_frame": None,
                "reviewed_call": None,
            },
        ],
    }

    bounces, inout = build_reviewed_bounce_inout_labels(review)

    assert bounces["artifact_type"] == "racketsport_reviewed_ball_bounces"
    assert bounces["status"] == "partial_human_review"
    assert bounces["bounces"] == [{"frame": 60, "t": 1.0, "review_id": "bounce_0000"}]
    assert inout["artifact_type"] == "racketsport_reviewed_ball_inout"
    assert inout["status"] == "partial_human_review"
    assert inout["calls"] == [{"frame": 60, "t": 1.0, "call": "out", "review_id": "bounce_0000"}]
    assert bounces["pending_review_count"] == 1
    assert inout["pending_review_count"] == 1


def test_write_review_html_supports_editing_and_download(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_inout_review",
        "clip": "clip_a",
        "items": [
            {
                "review_id": "bounce_0000",
                "predicted_frame": 60,
                "predicted_call": "out",
                "context_images": [{"frame": 60, "image": "images/bounce_0000_frame_000060.jpg"}],
                "reviewed_bounce_frame": None,
                "reviewed_call": None,
                "status": "needs_human_review",
            }
        ],
    }

    html_path = tmp_path / "review.html"
    write_review_html(html_path, payload)

    html = html_path.read_text(encoding="utf-8")
    assert "Accept predicted" in html
    assert "Mark too close" in html
    assert "Reject candidate" in html
    assert "reviewed_bounce_frame" in html
    assert "download = \"ball_bounce_inout_review.json\"" in html
    assert "saveCurrentReview" in html


def test_ball_bounce_inout_review_clis_write_expected_files(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    track = tmp_path / "ball_track.json"
    review_dir = tmp_path / "review"
    reviewed_bounces = tmp_path / "reviewed_bounces.json"
    reviewed_inout = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_ball_bounce_inout_review.py",
            "--video",
            str(video),
            "--ball-track",
            str(track),
            "--out",
            str(review_dir),
            "--clip",
            "clip_a",
            "--context-frames",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    review_path = review_dir / "ball_bounce_inout_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["items"][0]["status"] = "accepted"
    review["items"][0]["reviewed_bounce_frame"] = 60
    review["items"][0]["reviewed_call"] = "out"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_reviewed_ball_bounce_inout.py",
            "--review",
            str(review_path),
            "--out-bounces",
            str(reviewed_bounces),
            "--out-inout",
            str(reviewed_inout),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(reviewed_bounces.read_text(encoding="utf-8"))["bounces"][0]["frame"] == 60
    assert json.loads(reviewed_inout.read_text(encoding="utf-8"))["calls"][0]["call"] == "out"


def test_simple_review_decision_saves_selected_frame_and_call(tmp_path: Path) -> None:
    review_path = tmp_path / "clip_a" / "ball_bounce_inout_review.json"
    review_path.parent.mkdir()
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_bounce_inout_review",
                "clip": "clip_a",
                "items": [
                    {
                        "review_id": "bounce_0000",
                        "status": "needs_human_review",
                        "predicted_frame": 60,
                        "reviewed_bounce_frame": None,
                        "reviewed_call": None,
                        "review_notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_simple_review_decision(
        review_path=review_path,
        clip="clip_a",
        review_id="bounce_0000",
        action="out",
        frame=61,
    )

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert result == {"accepted": 1, "rejected": 0, "pending": 0, "too_close": 0, "inout": 1, "total": 1}
    assert saved["items"][0]["status"] == "accepted"
    assert saved["items"][0]["reviewed_bounce_frame"] == 61
    assert saved["items"][0]["reviewed_call"] == "out"
    assert saved["items"][0]["review_notes"] == "simple_review: out"


def test_simple_review_decision_rejects_candidate_without_call(tmp_path: Path) -> None:
    review_path = tmp_path / "clip_a" / "ball_bounce_inout_review.json"
    review_path.parent.mkdir()
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_bounce_inout_review",
                "clip": "clip_a",
                "items": [
                    {
                        "review_id": "bounce_0000",
                        "status": "needs_human_review",
                        "predicted_frame": 60,
                        "reviewed_bounce_frame": None,
                        "reviewed_call": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = apply_simple_review_decision(
        review_path=review_path,
        clip="clip_a",
        review_id="bounce_0000",
        action="reject",
        frame=60,
    )

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert result == {"accepted": 0, "rejected": 1, "pending": 0, "too_close": 0, "inout": 0, "total": 1}
    assert saved["items"][0]["status"] == "rejected"
    assert saved["items"][0]["reviewed_bounce_frame"] is None
    assert saved["items"][0]["reviewed_call"] is None


def test_build_simple_review_state_flattens_all_clip_items(tmp_path: Path) -> None:
    root = tmp_path / "review_packets"
    review_path = root / "clip_a" / "ball_bounce_inout_review.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_bounce_inout_review",
                "clip": "clip_a",
                "items": [
                    {
                        "review_id": "bounce_0000",
                        "status": "needs_human_review",
                        "predicted_frame": 60,
                        "predicted_call": "in",
                        "context_images": [{"frame": 59, "image": "images/frame_59.jpg"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = build_simple_review_state(root=root, clips=["clip_a"])

    assert state["totals"] == {"accepted": 0, "rejected": 0, "pending": 1, "too_close": 0, "inout": 0, "total": 1}
    assert state["items"][0]["clip"] == "clip_a"
    assert state["items"][0]["review_id"] == "bounce_0000"
    assert state["items"][0]["context_images"][0]["url"] == "/media/clip_a/images/frame_59.jpg"


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


def _write_test_video(path: Path, *, size: str = "640x360", fps: int = 60) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize review CLI test video")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={fps}:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
