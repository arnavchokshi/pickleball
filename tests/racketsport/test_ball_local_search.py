from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.ball_local_search import _apply_motion_evidence_postprocess, filter_ball_track_local_search
from threed.racketsport.schemas import BallTrack, validate_artifact_file


class _FakeFrame:
    def __init__(self, width: int, height: int, *, base: int = 20) -> None:
        self.shape = (height, width, 3)
        self._pixels = [
            [[base, base, base] for _x in range(width)]
            for _y in range(height)
        ]

    def set_pixel(self, x: int, y: int, bgr: tuple[int, int, int]) -> None:
        self._pixels[y][x] = list(bgr)

    def __getitem__(self, key: tuple[int, int]) -> list[int]:
        y, x = key
        return self._pixels[y][x]


class _FakeCapture:
    def __init__(self, frames: list[_FakeFrame], *, fps: float) -> None:
        self._frames = frames
        self._fps = fps
        self._index = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, _FakeFrame | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == _FakeCV2.CAP_PROP_FPS:
            return self._fps
        if prop == _FakeCV2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1])
        if prop == _FakeCV2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0])
        return 0.0

    def release(self) -> None:
        self._opened = False


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, frames: list[_FakeFrame], *, fps: float = 30.0) -> None:
        self._frames = frames
        self._fps = fps

    def VideoCapture(self, _path: str) -> _FakeCapture:  # noqa: N802 - matches cv2 API
        return _FakeCapture(self._frames, fps=self._fps)


def _write_track(path: Path, *, off_path_conf: float = 0.2) -> None:
    frames = [
        {"t": 0 / 30.0, "xy": [10.0, 10.0], "conf": 0.95, "visible": True},
        {"t": 1 / 30.0, "xy": [20.0, 10.0], "conf": 0.95, "visible": True},
        {"t": 2 / 30.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False},
        {"t": 3 / 30.0, "xy": [100.0, 45.0], "conf": off_path_conf, "visible": True},
        {"t": 4 / 30.0, "xy": [40.0, 10.0], "conf": 0.95, "visible": True},
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "tracknet",
                "frames": frames,
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def _fake_frames() -> list[_FakeFrame]:
    frames = [_FakeFrame(120, 60) for _index in range(5)]
    frames[0].set_pixel(10, 10, (245, 245, 245))
    frames[1].set_pixel(20, 10, (245, 245, 245))
    frames[2].set_pixel(30, 10, (245, 245, 245))
    frames[4].set_pixel(40, 10, (245, 245, 245))
    return frames


def test_local_search_recovers_missing_ball_and_suppresses_weak_off_path_candidate(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    track_path = tmp_path / "ball_track.json"
    video_path.write_bytes(b"fake video")
    _write_track(track_path)

    payload, summary = filter_ball_track_local_search(
        video_path=video_path,
        ball_track_path=track_path,
        search_radius_px=5,
        min_contrast=80.0,
        max_speed_px_per_second=600.0,
        base_jump_px=6.0,
        cv2_module=_FakeCV2(_fake_frames()),
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[2].visible is True
    assert filtered.frames[2].approx is True
    assert filtered.frames[2].xy == pytest.approx([30.0, 10.0])
    assert filtered.frames[3].visible is False
    assert filtered.frames[3].conf == pytest.approx(0.0)
    assert filtered.frames[4].visible is True
    assert summary["artifact_type"] == "racketsport_ball_local_search_filter"
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["uses_human_clicks"] is False
    assert summary["recovered_count"] == 1
    assert summary["suppressed_off_path_count"] == 1
    assert summary["source_video"] == str(video_path)
    assert "clicks_path" not in summary


def test_local_search_suppresses_high_confidence_off_path_candidate_by_default(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    track_path = tmp_path / "ball_track.json"
    video_path.write_bytes(b"fake video")
    _write_track(track_path, off_path_conf=0.99)

    payload, summary = filter_ball_track_local_search(
        video_path=video_path,
        ball_track_path=track_path,
        search_radius_px=5,
        min_contrast=80.0,
        max_speed_px_per_second=600.0,
        base_jump_px=6.0,
        cv2_module=_FakeCV2(_fake_frames()),
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[2].visible is True
    assert filtered.frames[3].visible is False
    assert filtered.frames[3].conf == pytest.approx(0.0)
    assert summary["suppressed_off_path_count"] == 1
    assert summary["uses_human_clicks"] is False


def test_local_search_writer_and_cli_entrypoint_write_schema_valid_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.racketsport.filter_ball_local_search import main

    video_path = tmp_path / "clip.mp4"
    track_path = tmp_path / "ball_track.json"
    out_path = tmp_path / "ball_local_search.json"
    summary_path = tmp_path / "ball_local_search_summary.json"
    video_path.write_bytes(b"fake video")
    _write_track(track_path)

    exit_code = main(
        [
            "--video",
            str(video_path),
            "--ball-track",
            str(track_path),
            "--search-radius-px",
            "5",
            "--min-contrast",
            "80",
            "--max-speed-px-per-second",
            "600",
            "--base-jump-px",
            "6",
            "--out",
            str(out_path),
            "--summary-out",
            str(summary_path),
        ],
        cv2_module=_FakeCV2(_fake_frames()),
    )

    stdout = json.loads(capsys.readouterr().out)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout["uses_human_clicks"] is False
    assert summary["uses_human_clicks"] is False
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["recovered_count"] == 1
    assert isinstance(validate_artifact_file("ball_track", out_path), BallTrack)


def _motion_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [
            {"t": index / 30.0, "xy": [80.0, 20.0], "conf": 1.0, "visible": True, "approx": False}
            for index in range(8)
        ],
        "bounces": [],
    }


def _moving_blob_frames() -> list[object]:
    np = pytest.importorskip("numpy")

    frames = []
    for index in range(8):
        image = np.zeros((48, 120, 3), dtype=np.uint8)
        x = 24 + index
        image[20:25, x : x + 5] = 245
        frames.append(image)
    return frames


def test_motion_evidence_postprocess_replaces_stationary_false_run_with_moving_blob() -> None:
    cv2 = pytest.importorskip("cv2")
    payload = _motion_payload()

    summary = _apply_motion_evidence_postprocess(
        payload,
        source_frames=_moving_blob_frames(),
        cv2_module=cv2,
        stationary_run_min_frames=4,
        stationary_min_relocate_distance_px=20.0,
        motion_min_area_px=3,
        motion_max_area_px=80,
        motion_min_peak=40.0,
    )

    assert summary["motion_recovered_count"] >= 1
    assert payload["frames"][3]["visible"] is True
    assert payload["frames"][3]["approx"] is True
    assert payload["frames"][3]["xy"][0] < 40.0
    assert isinstance(BallTrack.model_validate(payload), BallTrack)


def test_motion_evidence_postprocess_suppresses_stale_duplicates_and_top_edge_samples() -> None:
    cv2 = pytest.importorskip("cv2")
    payload = _motion_payload()
    payload["frames"][1]["approx"] = True
    payload["frames"][2]["approx"] = True
    payload["frames"][2]["xy"] = list(payload["frames"][1]["xy"])
    payload["frames"][5]["xy"] = [44.0, 4.0]

    summary = _apply_motion_evidence_postprocess(
        payload,
        source_frames=[],
        cv2_module=cv2,
        top_edge_suppress_px=10.0,
    )

    assert payload["frames"][2]["visible"] is False
    assert payload["frames"][5]["visible"] is False
    assert summary["stale_duplicate_suppressed_count"] == 1
    assert summary["edge_suppressed_count"] == 1
    assert isinstance(BallTrack.model_validate(payload), BallTrack)


def test_motion_evidence_postprocess_relocates_approximate_sample_to_nearby_motion() -> None:
    cv2 = pytest.importorskip("cv2")
    payload = {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [12.0, 18.0], "conf": 1.0, "visible": True, "approx": False},
            {"t": 1.0 / 30.0, "xy": [54.0, 15.0], "conf": 0.4, "visible": True, "approx": True},
            {"t": 2.0 / 30.0, "xy": [30.0, 35.0], "conf": 1.0, "visible": True, "approx": False},
        ],
        "bounces": [],
    }
    frames = _moving_blob_frames()[:3]

    summary = _apply_motion_evidence_postprocess(
        payload,
        source_frames=frames,
        cv2_module=cv2,
        approximate_relocate_radius_px=60.0,
        approximate_min_relocate_distance_px=5.0,
        motion_min_area_px=3,
        motion_max_area_px=80,
        motion_min_peak=40.0,
    )

    assert summary["motion_relocated_count"] == 1
    assert payload["frames"][1]["xy"][0] < 40.0
    assert payload["frames"][1]["approx"] is True
    assert isinstance(BallTrack.model_validate(payload), BallTrack)
