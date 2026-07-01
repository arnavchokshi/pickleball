from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.body_world_label_review_overlay import build_body_world_label_review_overlays
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError


class _FakeFrame:
    def __init__(self, shape: tuple[int, int, int] = (1080, 1920, 3)) -> None:
        self.shape = shape


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self, *, frame_shape: tuple[int, int, int] = (1080, 1920, 3)) -> None:
        self.frame_shape = frame_shape
        self.calls: list[dict] = []

    def imread(self, path: str) -> _FakeFrame | None:
        if not Path(path).exists():
            return None
        self.calls.append({"kind": "read", "path": path})
        return _FakeFrame(self.frame_shape)

    def imwrite(self, path: str, frame: _FakeFrame) -> bool:
        del frame
        Path(path).write_bytes(b"rendered")
        self.calls.append({"kind": "write", "path": path})
        return True

    def rectangle(self, _frame: _FakeFrame, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int) -> None:
        self.calls.append({"kind": "rectangle", "start": start, "end": end, "color": color, "thickness": thickness})

    def circle(self, _frame: _FakeFrame, center: tuple[int, int], radius: int, color: tuple[int, int, int], thickness: int, *args) -> None:
        self.calls.append({"kind": "circle", "center": center, "radius": radius, "color": color, "thickness": thickness})

    def line(self, _frame: _FakeFrame, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int, *args) -> None:
        self.calls.append({"kind": "line", "start": start, "end": end, "color": color, "thickness": thickness})

    def putText(self, _frame: _FakeFrame, text: str, org: tuple[int, int], font: int, scale: float, color: tuple[int, int, int], thickness: int, *args) -> None:
        del font, scale, color, thickness, args
        self.calls.append({"kind": "text", "text": text, "org": org})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _queue(
    tmp_path: Path,
    *,
    image_exists: bool = True,
    track_world_xy: list[float] | None = None,
    predicted_joints_world: list[list[float]] | None = None,
) -> Path:
    image_path = tmp_path / "frames" / "frame_000010.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_exists:
        image_path.write_bytes(b"fake jpeg")
    track_world_xy = track_world_xy or [2.0, -5.5]
    predicted_joints_world = predicted_joints_world or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    path = tmp_path / "body_world_label_review_queue.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_queue",
            "status": "ready_for_review",
            "clip": "clip_001",
            "source_packet": "body_world_label_packet.json",
            "source_video": "clip.mp4",
            "sample_count": 1,
            "samples": [
                {
                    "sample_id": "frame_000010_player_7",
                    "frame_index": 10,
                    "t": 10.0 / 30.0,
                    "player_id": 7,
                    "track_world_xy": track_world_xy,
                    "joint_count": len(predicted_joints_world),
                    "predicted_joints_world": predicted_joints_world,
                    "joint_conf": [0.9] * len(predicted_joints_world),
                    "image_path": str(image_path),
                    "source_image_path": str(image_path),
                    "source_image_exists": image_exists,
                }
            ],
            "not_ground_truth": True,
        },
    )
    return path


def _tracks(
    tmp_path: Path,
    *,
    player_id: int = 7,
    bbox: list[float] | None = None,
    world_xy: list[float] | None = None,
) -> Path:
    path = tmp_path / "tracks.json"
    bbox = bbox or [900.0, 480.0, 1100.0, 650.0]
    world_xy = world_xy or [2.0, -5.5]
    _write_json(
        path,
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [
                {
                    "id": player_id,
                    "side": "near",
                    "role": "left",
                    "frames": [
                        {
                            "t": 10.0 / 30.0,
                            "bbox": bbox,
                            "world_xy": world_xy,
                            "conf": 0.93,
                        }
                    ],
                }
            ],
            "rally_spans": [],
        },
    )
    return path


def _calibration(tmp_path: Path, *, half_resolution: bool = False) -> Path:
    image_size = (960, 540) if half_resolution else (1920, 1080)
    fx = 500.0 if half_resolution else 1000.0
    cx = 480.0 if half_resolution else 960.0
    cy = 270.0 if half_resolution else 540.0
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        image_size=image_size,
        homography=[[20.0, 0.0, cx], [0.0, -20.0, cy], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=fx, fy=fx, cx=cx, cy=cy, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )
    path = tmp_path / "court_calibration.json"
    path.write_text(calibration.model_dump_json(), encoding="utf-8")
    return path


def test_body_world_label_review_overlay_draws_track_bbox_and_projected_joints(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(tmp_path),
        tracks_path=_tracks(tmp_path),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["artifact_type"] == "racketsport_body_world_label_review_overlay"
    assert manifest["status"] == "ready_for_review"
    assert manifest["rendered_count"] == 1
    assert manifest["sample_count"] == 1
    assert manifest["qualitative_status"] == "review_overlay_not_gate_verified"
    assert manifest["not_ground_truth"] is True
    assert manifest["overlays"][0]["track_bbox_status"] == "matched"
    assert manifest["overlays"][0]["projection_status"] == "projected"
    assert manifest["overlays"][0]["projection_mode"] == "homography_grounded_pnp_vertical"
    assert manifest["overlays"][0]["joint_bbox_alignment"]["status"] == "passed"
    assert manifest["overlays"][0]["track_floor_projection_alignment"]["status"] == "passed"
    assert manifest["overlays"][0]["projected_joint_count"] == 3
    assert manifest["overlays"][0]["in_frame_projected_joint_count"] == 3
    assert (tmp_path / "overlays" / "frame_000010_player_7_overlay.jpg").read_bytes() == b"rendered"
    assert (tmp_path / "overlays" / "body_world_label_review_overlay_index.json").is_file()
    assert any(call["kind"] == "rectangle" and call["start"] == (900, 480) and call["end"] == (1100, 650) for call in fake_cv2.calls)
    assert any(call["kind"] == "circle" and call["center"] == (960, 540) for call in fake_cv2.calls)
    assert any(call["kind"] == "circle" and call["center"] == (980, 540) for call in fake_cv2.calls)
    assert any(call["kind"] == "text" and "frame_000010_player_7" in call["text"] for call in fake_cv2.calls)


def test_body_world_label_review_overlay_scales_half_resolution_calibration(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2(frame_shape=(1080, 1920, 3))

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(tmp_path, track_world_xy=[1.0, -2.75]),
        tracks_path=_tracks(tmp_path, bbox=[450.0, 240.0, 550.0, 325.0], world_xy=[1.0, -2.75]),
        calibration_path=_calibration(tmp_path, half_resolution=True),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "ready_for_review"
    assert manifest["overlays"][0]["track_bbox"] == [900.0, 480.0, 1100.0, 650.0]
    assert manifest["overlays"][0]["track_bbox_scale_x"] == pytest.approx(2.0)
    assert manifest["overlays"][0]["track_bbox_scale_y"] == pytest.approx(2.0)
    assert manifest["overlays"][0]["track_floor_projection_alignment"]["status"] == "passed"
    assert any(call["kind"] == "rectangle" and call["start"] == (900, 480) and call["end"] == (1100, 650) for call in fake_cv2.calls)
    assert any(call["kind"] == "circle" and call["center"] == (1000, 540) for call in fake_cv2.calls)


def test_body_world_label_review_overlay_blocks_missing_review_frame(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(tmp_path, image_exists=False),
        tracks_path=_tracks(tmp_path),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "blocked_missing_review_frames"
    assert manifest["rendered_count"] == 0
    assert manifest["missing_frame_count"] == 1
    assert "missing_review_frame" in manifest["blockers"]
    assert not list((tmp_path / "overlays").glob("*.jpg"))


def test_body_world_label_review_overlay_reports_missing_track_bbox_without_blocking_projection(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(tmp_path),
        tracks_path=_tracks(tmp_path, player_id=8),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "ready_for_review"
    assert manifest["rendered_count"] == 1
    assert manifest["missing_track_bbox_count"] == 1
    assert manifest["overlays"][0]["track_bbox_status"] == "missing"
    assert manifest["overlays"][0]["projection_status"] == "projected"
    assert not any(call["kind"] == "rectangle" for call in fake_cv2.calls)


def test_body_world_label_review_overlay_marks_failed_joint_bbox_alignment(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(tmp_path, track_world_xy=[-40.5, 7.0]),
        tracks_path=_tracks(tmp_path, bbox=[100.0, 200.0, 200.0, 400.0], world_xy=[-40.5, 7.0]),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "rendered_alignment_failed"
    assert manifest["alignment_failed_count"] == 1
    assert "body_joint_overlay_alignment_failed" in manifest["blockers"]
    assert manifest["overlays"][0]["joint_bbox_alignment"]["status"] == "failed"
    assert "body_joint_overlay_alignment_failed" in manifest["overlays"][0]["warnings"]


def test_body_world_label_review_overlay_warns_for_low_containment_when_joint_center_is_aligned(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(
            tmp_path,
            predicted_joints_world=[
                [-9.0, -5.5, 0.0],
                [2.0, -5.5, 0.0],
                [12.0, -5.5, 0.0],
            ],
        ),
        tracks_path=_tracks(tmp_path),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "ready_for_review_with_overlay_warnings"
    assert manifest["alignment_failed_count"] == 0
    assert manifest["alignment_warning_count"] == 1
    assert not manifest["blockers"]
    alignment = manifest["overlays"][0]["joint_bbox_alignment"]
    assert alignment["status"] == "warning"
    assert alignment["containment_ratio"] < 0.35
    assert alignment["center_delta_bbox_diag"] < 0.5


def test_body_world_label_review_overlay_warns_when_joints_fit_neighboring_player_better(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()
    tracks_path = tmp_path / "tracks.json"
    _write_json(
        tracks_path,
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [
                {
                    "id": 7,
                    "side": "near",
                    "role": "left",
                    "frames": [
                        {
                            "t": 10.0 / 30.0,
                            "bbox": [1000.0, 480.0, 1250.0, 650.0],
                            "world_xy": [8.25, -5.5],
                            "conf": 0.93,
                        }
                    ],
                },
                {
                    "id": 8,
                    "side": "near",
                    "role": "right",
                    "frames": [
                        {
                            "t": 10.0 / 30.0,
                            "bbox": [1150.0, 480.0, 1350.0, 650.0],
                            "world_xy": [12.5, -5.5],
                            "conf": 0.91,
                        }
                    ],
                },
            ],
            "rally_spans": [],
        },
    )

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(
            tmp_path,
            track_world_xy=[8.25, -5.5],
            predicted_joints_world=[
                [10.0, 0.0, 0.0],
                [14.0, 0.0, 0.0],
                [18.0, 0.0, 0.0],
            ],
        ),
        tracks_path=tracks_path,
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    overlay = manifest["overlays"][0]
    assert manifest["status"] == "ready_for_review_with_overlay_warnings"
    assert manifest["competing_player_warning_count"] == 1
    assert overlay["joint_bbox_alignment"]["status"] == "passed"
    assert "body_joint_overlay_competing_player_warning" in overlay["warnings"]
    assert overlay["competing_player_alignment"]["status"] == "warning"
    assert overlay["competing_player_alignment"]["best_player_id"] == 8
    assert overlay["competing_player_alignment"]["best_player_containment_ratio"] == pytest.approx(1.0)
    assert overlay["competing_player_alignment"]["target_containment_ratio"] == pytest.approx(2.0 / 3.0, abs=0.001)


def test_body_world_label_review_overlay_uses_homography_floor_anchor_when_pnp_disagrees(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()

    manifest = build_body_world_label_review_overlays(
        queue_path=_queue(
            tmp_path,
            track_world_xy=[2.0, -5.5],
            predicted_joints_world=[[1.8, -5.5, 0.0], [2.2, -5.5, 0.0], [2.0, -5.5, 1.0]],
        ),
        tracks_path=_tracks(tmp_path, world_xy=[2.0, -5.5]),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "ready_for_review"
    assert manifest["floor_anchor_projection_failed_count"] == 0
    assert manifest["floor_anchor_projection_warning_count"] == 0
    assert "body_floor_anchor_projection_failed" not in manifest["blockers"]
    assert manifest["overlays"][0]["joint_bbox_alignment"]["status"] == "passed"
    floor_alignment = manifest["overlays"][0]["track_floor_projection_alignment"]
    assert floor_alignment["status"] == "passed"
    assert floor_alignment["bbox_footpoint"] == [1000.0, 650.0]
    assert floor_alignment["projected_track_world_xy"] == [1000.0, 650.0]
    pnp_alignment = manifest["overlays"][0]["pnp_track_floor_projection_alignment"]
    assert pnp_alignment["status"] == "failed"
    assert pnp_alignment["projected_track_world_xy"] == [1160.0, -10.0]
    assert "body_floor_anchor_projection_failed" not in manifest["overlays"][0]["warnings"]


def test_body_world_label_review_overlay_blocks_unprojectable_joint_samples(tmp_path: Path) -> None:
    fake_cv2 = _FakeCv2()
    queue = _queue(tmp_path)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["samples"][0]["predicted_joints_world"] = []
    queue.write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_body_world_label_review_overlays(
        queue_path=queue,
        tracks_path=_tracks(tmp_path),
        calibration_path=_calibration(tmp_path),
        out_dir=tmp_path / "overlays",
        cv2_module=fake_cv2,
    )

    assert manifest["status"] == "blocked_projection_failed"
    assert manifest["rendered_count"] == 1
    assert manifest["projection_failed_count"] == 1
    assert manifest["overlays"][0]["projection_status"] == "missing_predicted_joints_world"
    assert "unprojectable_body_world_joints" in manifest["blockers"]


def test_build_body_world_label_review_overlay_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_body_world_label_review_overlay.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--run-dir" in completed.stdout
    assert "--queue" in completed.stdout
    assert "--calibration" in completed.stdout
