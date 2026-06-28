from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.calibration_overlay import (
    NET_LINE_COLOR,
    build_calibration_overlay,
    render_calibration_image_overlay,
    render_calibration_run_overlays,
)
from threed.racketsport.net_plane import build_net_plane
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError


def _synthetic_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[20.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )


def _half_resolution_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[10.0, 0.0, 480.0], [0.0, -10.0, 270.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=480.0, cy=270.0, dist=[], source="synthetic_half"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )


def _untrusted_top_net_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 500.0], [0.0, 100.0, 300.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=300.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.35, 0.0, 1.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )


def test_overlay_emits_projected_court_lines_and_net_points():
    overlay = build_calibration_overlay(_synthetic_calibration(), net_plane=build_net_plane("pickleball"))

    assert overlay["schema_version"] == 1
    assert overlay["sport"] == "pickleball"
    assert {line["id"] for line in overlay["court_lines"]} >= {
        "near_baseline",
        "far_baseline",
        "left_sideline",
        "right_sideline",
        "near_nvz",
        "far_nvz",
        "near_centerline",
        "far_centerline",
        "net",
    }
    near_baseline = next(line for line in overlay["court_lines"] if line["id"] == "near_baseline")
    assert near_baseline["image"][0] == pytest.approx([899.04, 674.112])
    assert near_baseline["image"][1] == pytest.approx([1020.96, 674.112])
    assert set(overlay["net_points"]) == {"left_post", "right_post", "center"}
    assert overlay["net_points"]["left_post"][0] < 960.0
    assert overlay["net_points"]["right_post"][0] > 960.0
    assert overlay["net_points"]["center"] == pytest.approx([960.0, 540.0])
    assert overlay["summary"]["court_line_count"] == len(overlay["court_lines"])
    assert overlay["summary"]["net_point_count"] == 3
    assert overlay["summary"]["net_top_projection_status"] == "trusted_pnp_geometry"


def test_overlay_omits_untrusted_top_net_projection_when_pnp_disagrees_with_ground_net():
    overlay = build_calibration_overlay(_untrusted_top_net_calibration(), net_plane=build_net_plane("pickleball"))

    assert any(line["id"] == "net" for line in overlay["court_lines"])
    assert overlay["net_points"] == {}
    assert overlay["net_segments"] == []
    assert overlay["summary"]["net_point_count"] == 0
    assert overlay["summary"]["net_top_projection_status"] == "untrusted_pnp_geometry"
    assert overlay["summary"]["net_top_angle_delta_deg"] > 6.0


def test_overlay_omits_top_net_projection_for_estimated_review_frame_intrinsics():
    calibration = _synthetic_calibration()
    calibration = calibration.model_copy(
        deep=True,
        update={"intrinsics": calibration.intrinsics.model_copy(update={"source": "estimated_from_review_frame"})},
    )

    overlay = build_calibration_overlay(calibration, net_plane=build_net_plane("pickleball"))

    assert overlay["net_points"] == {}
    assert overlay["net_segments"] == []
    assert overlay["summary"]["net_top_projection_status"] == "untrusted_estimated_intrinsics"


def test_overlay_rejects_mismatched_net_plane_sport():
    with pytest.raises(ValueError, match="net plane endpoints do not match calibration sport"):
        build_calibration_overlay(_synthetic_calibration(), net_plane=build_net_plane("tennis"))


def test_render_calibration_overlay_cli_writes_svg_and_json_summary(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    net_path = tmp_path / "net_plane.json"
    svg_path = tmp_path / "calibration_overlay.svg"
    summary_path = tmp_path / "calibration_overlay.json"
    calibration_path.write_text(_synthetic_calibration().model_dump_json(), encoding="utf-8")
    net_path.write_text(build_net_plane("pickleball").model_dump_json(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(calibration_path),
            "--net-plane",
            str(net_path),
            "--out",
            str(svg_path),
            "--summary-out",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    svg = svg_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert completed.stdout == f"wrote {svg_path}\nwrote {summary_path}\n"
    assert "<svg" in svg
    assert 'data-line-id="near_baseline"' in svg
    assert 'data-line-id="near_centerline"' in svg
    assert 'data-line-id="far_centerline"' in svg
    assert 'data-net-point-id="center"' in svg
    assert summary["summary"]["court_line_count"] >= 9
    assert summary["summary"]["net_point_count"] == 3


def test_render_calibration_overlay_cli_fails_cleanly_for_missing_input(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "calibration_overlay.svg"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "missing calibration artifact" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_render_calibration_overlay_cli_fails_cleanly_for_invalid_input(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_calibration_overlay.py",
            "--calibration",
            str(calibration_path),
            "--out",
            str(tmp_path / "calibration_overlay.svg"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Field required" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_render_calibration_image_overlay_draws_projected_lines_without_real_cv2(tmp_path):
    image_path = tmp_path / "frame.jpg"
    out_path = tmp_path / "calibration_overlay_frame.jpg"
    image_path.write_bytes(b"fake image")
    fake_cv2 = _FakeCv2()

    summary = render_calibration_image_overlay(
        image_path=image_path,
        out_path=out_path,
        calibration=_synthetic_calibration(),
        net_plane=build_net_plane("pickleball"),
        cv2_module=fake_cv2,
    )

    assert summary["status"] == "rendered"
    assert summary["image_path"] == str(image_path)
    assert summary["out_path"] == str(out_path)
    assert summary["court_line_count"] >= 7
    assert {"net", "near_baseline"}.issubset(set(summary["court_line_ids"]))
    assert summary["net_point_count"] == 3
    assert out_path.read_bytes() == b"rendered"
    assert sum(1 for call in fake_cv2.calls if call["kind"] == "line") >= summary["court_line_count"]
    assert any(call["kind"] == "text" and call["text"] == "net" for call in fake_cv2.calls)
    assert any(call["kind"] == "text" and call["text"] == "near_baseline" for call in fake_cv2.calls)
    assert any(call["kind"] == "circle" for call in fake_cv2.calls)


def test_render_calibration_image_overlay_scales_calibration_to_frame_resolution(tmp_path):
    image_path = tmp_path / "frame.jpg"
    out_path = tmp_path / "calibration_overlay_frame.jpg"
    image_path.write_bytes(b"fake image")
    fake_cv2 = _FakeCv2(frame_shape=(1080, 1920, 3))

    render_calibration_image_overlay(
        image_path=image_path,
        out_path=out_path,
        calibration=_half_resolution_calibration(),
        net_plane=build_net_plane("pickleball"),
        cv2_module=fake_cv2,
    )

    first_line = next(call for call in fake_cv2.calls if call["kind"] == "line")
    assert first_line["start"] == (899, 674)
    assert first_line["end"] == (1021, 674)


def test_render_calibration_image_overlay_does_not_draw_orange_when_top_net_untrusted(tmp_path):
    image_path = tmp_path / "frame.jpg"
    out_path = tmp_path / "calibration_overlay_frame.jpg"
    image_path.write_bytes(b"fake image")
    fake_cv2 = _FakeCv2(frame_shape=(1080, 1920, 3))

    render_calibration_image_overlay(
        image_path=image_path,
        out_path=out_path,
        calibration=_untrusted_top_net_calibration(),
        net_plane=build_net_plane("pickleball"),
        cv2_module=fake_cv2,
    )

    assert not any(call["kind"] == "line" and call["color"] == NET_LINE_COLOR for call in fake_cv2.calls)
    assert not any(call["kind"] == "circle" and call["color"] == NET_LINE_COLOR for call in fake_cv2.calls)


def test_render_calibration_run_overlays_uses_reviewed_frame_and_skips_missing_clips(tmp_path):
    run_root = tmp_path / "run"
    frames_root = run_root / "review_bundle" / "images"
    clip = "clip_static"
    clip_dir = run_root / clip
    clip_dir.mkdir(parents=True)
    frames_dir = frames_root / clip
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000001.jpg").write_bytes(b"frame 1")
    (frames_dir / "frame_000002.jpg").write_bytes(b"frame 2")
    (clip_dir / "court_calibration.json").write_text(_synthetic_calibration().model_dump_json(), encoding="utf-8")
    (clip_dir / "net_plane.json").write_text(build_net_plane("pickleball").model_dump_json(), encoding="utf-8")
    (run_root / "court_corner_calibration_summary.json").write_text(
        json.dumps({"clips": [{"clip": clip, "frame": "frame_000002.jpg"}]}),
        encoding="utf-8",
    )
    fake_cv2 = _FakeCv2()

    summary = render_calibration_run_overlays(
        run_root=run_root,
        frames_root=frames_root,
        clips=[clip, "missing_clip"],
        max_video_frames=2,
        cv2_module=fake_cv2,
    )

    assert summary["status"] == "rendered"
    rendered = summary["clips"][0]
    skipped = summary["clips"][1]
    assert rendered["clip"] == clip
    assert rendered["review_frame"] == str(frames_dir / "frame_000002.jpg")
    assert rendered["video_frame_count"] == 2
    assert (clip_dir / "compare" / "calibration_overlay_frame.jpg").read_bytes() == b"rendered"
    assert (clip_dir / "compare" / "calibration_overlay.mp4").read_bytes() == b"video"
    assert skipped["status"] == "skipped"
    assert "missing calibration or net-plane artifact" in skipped["warnings"]


def test_render_calibration_video_overlay_scales_calibration_to_frame_resolution(tmp_path):
    run_root = tmp_path / "run"
    frames_root = run_root / "review_bundle" / "images"
    clip = "clip_static"
    clip_dir = run_root / clip
    clip_dir.mkdir(parents=True)
    frames_dir = frames_root / clip
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000001.jpg").write_bytes(b"frame 1")
    (clip_dir / "court_calibration.json").write_text(_half_resolution_calibration().model_dump_json(), encoding="utf-8")
    (clip_dir / "net_plane.json").write_text(build_net_plane("pickleball").model_dump_json(), encoding="utf-8")
    fake_cv2 = _FakeCv2(frame_shape=(1080, 1920, 3))

    render_calibration_run_overlays(
        run_root=run_root,
        frames_root=frames_root,
        clips=[clip],
        max_video_frames=1,
        cv2_module=fake_cv2,
    )

    video_writer_idx = next(idx for idx, call in enumerate(fake_cv2.calls) if call["kind"] == "VideoWriter")
    first_video_line = next(call for call in fake_cv2.calls[video_writer_idx:] if call["kind"] == "line")
    assert first_video_line["start"] == (899, 674)
    assert first_video_line["end"] == (1021, 674)


class _FakeFrame:
    def __init__(self, shape: tuple[int, int, int] = (48, 64, 3)) -> None:
        self.shape = shape


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self, frame_shape: tuple[int, int, int] = (48, 64, 3)) -> None:
        self.calls: list[dict[str, object]] = []
        self.frame_shape = frame_shape

    def imread(self, path: str) -> _FakeFrame:
        self.calls.append({"kind": "imread", "path": path})
        return _FakeFrame(self.frame_shape)

    def imwrite(self, path: str, frame: _FakeFrame) -> bool:
        Path(path).write_bytes(b"rendered")
        self.calls.append({"kind": "imwrite", "path": path})
        return True

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> "_FakeWriter":
        self.calls.append({"kind": "VideoWriter", "path": path, "fps": fps, "size": size})
        return _FakeWriter(path)

    def VideoWriter_fourcc(self, *args: str) -> int:
        return 1

    def line(self, frame: _FakeFrame, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int, lineType: int | None = None) -> None:
        self.calls.append({"kind": "line", "start": start, "end": end, "color": color, "thickness": thickness})

    def circle(self, frame: _FakeFrame, center: tuple[int, int], radius: int, color: tuple[int, int, int], thickness: int) -> None:
        self.calls.append({"kind": "circle", "center": center, "radius": radius, "color": color, "thickness": thickness})

    def putText(
        self,
        frame: _FakeFrame,
        text: str,
        origin: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
        lineType: int | None = None,
    ) -> None:
        self.calls.append({"kind": "text", "text": text, "origin": origin, "color": color})


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames = 0

    def isOpened(self) -> bool:
        return True

    def write(self, frame: _FakeFrame) -> None:
        self.frames += 1

    def release(self) -> None:
        self.path.write_bytes(b"video")
