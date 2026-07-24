from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_metric3d_contract import read_solver_observation_log

CLI_PATH = "scripts/racketsport/build_ball_observation_log.py"
REPO_ROOT = Path(__file__).resolve().parents[2]

FX = 1000.0
FY = 1000.0
CX = 640.0
CY = 360.0


def _calibration() -> dict:
    # Camera at world (0, -10, 2) looking along +y (z-up world). Rows of R are
    # the camera axes in world coordinates; projection is K (R X + t).
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "dist": []},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            "t": [0.0, 2.0, 10.0],
        },
    }


def _project(world_xyz: list[float]) -> list[float]:
    x, y, z = world_xyz
    cam = [x, -z + 2.0, y + 10.0]
    return [FX * cam[0] / cam[2] + CX, FY * cam[1] / cam[2] + CY]


WORLDS = {
    0: [0.0, -4.0, 0.5],
    1: [0.1, -3.5, 0.8],
    2: [0.2, -3.0, 1.0],
    3: [0.4, 2.0, 1.2],
    5: [0.6, 3.0, 1.1],
}


def _arc_solved_fixture() -> dict:
    frames = []
    for index in range(6):
        world = WORLDS.get(index)
        if index == 4:
            frames.append(
                {
                    "t": index / 30.0,
                    "visible": False,
                    "xy": [0.0, 0.0],
                    "conf": 0.0,
                    "band": "hidden",
                    "world_xyz": None,
                }
            )
            continue
        frame = {
            "t": index / 30.0,
            "visible": True,
            "xy": _project(world),
            "conf": 0.9,
            "band": "anchored_measured",
            "world_xyz": world,
        }
        if index <= 3:
            frame["arc_solver"] = {
                "segment_id": 0 if index <= 2 else 1,
                "segment_status": "fit" if index <= 2 else "fit_bvp_fallback",
                "inlier_sighting": index <= 2,
                "outlier_sighting_pruned": False,
                "rescued": False,
            }
        # Frame 5 intentionally has NO per-frame provenance: it must fail
        # closed inside segment 1's untrusted span.
        frames.append(frame)
    return {
        "schema_version": 1,
        "artifact_type": "ball_track_arc_solved",
        "clip_id": "synthetic_mini",
        "status": "ran",
        "kill_reasons": [],
        "frames": frames,
        "segments": [
            {
                "segment_id": 0,
                "status": "fit",
                "frame_start": 0,
                "frame_end": 2,
                "inlier_count": 3,
                "outlier_count": 0,
                "reprojection_rmse_px": 2.0,
                "max_reprojection_error_px": 5.0,
            },
            {
                "segment_id": 1,
                "status": "fit_bvp_fallback",
                "frame_start": 3,
                "frame_end": 5,
                "inlier_count": 0,
                "outlier_count": 4,
                "reprojection_rmse_px": 55.0,
                "max_reprojection_error_px": 90.0,
            },
        ],
        "anchors": [
            {
                "anchor_id": "bounce_000002",
                "kind": "bounce",
                "status": "solver_proposed",
                "source": "ball_bounce_candidates",
                "frame": 2,
                "t": 2 / 30.0,
                "world_xyz": [0.2, -3.0, 0.0371],
            }
        ],
    }


def _write_clip_dir(base: Path, *, tamper_calibration_sha: bool = False) -> Path:
    clip_dir = base / "clip_artifacts"
    clip_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = clip_dir / "court_calibration.json"
    calibration_path.write_text(json.dumps(_calibration(), indent=2), encoding="utf-8")
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    if tamper_calibration_sha:
        calibration_sha = "f" * 64
    (clip_dir / "ball_chain_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "court_calibration": {
                        "path": "court_calibration.json",
                        "sha256": calibration_sha,
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (clip_dir / "ball_track_arc_solved.json").write_text(
        json.dumps(_arc_solved_fixture(), indent=2), encoding="utf-8"
    )
    return clip_dir


def _run_cli(clip_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--clip",
            f"synthetic_mini={clip_dir}",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


def _point_to_ray_distance(point, origin, direction) -> float:
    diff = [point[i] - origin[i] for i in range(3)]
    cross = [
        diff[1] * direction[2] - diff[2] * direction[1],
        diff[2] * direction[0] - diff[0] * direction[2],
        diff[0] * direction[1] - diff[1] * direction[0],
    ]
    return math.sqrt(sum(v * v for v in cross))


def test_cli_builds_valid_observation_log(tmp_path):
    clip_dir = _write_clip_dir(tmp_path)
    out_dir = tmp_path / "out"
    result = _run_cli(clip_dir, out_dir)
    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    summary = stdout["clips"]["synthetic_mini"]
    assert summary["frame_count"] == 6
    assert summary["observed_frame_count"] == 5
    assert summary["ray_computed_frame_count"] == 5
    assert summary["accepted_frame_count"] == 3
    assert summary["calibration_sha_verified"] is True

    log = read_solver_observation_log(out_dir / "synthetic_mini.observation_log.json")
    assert log.clip == "synthetic_mini"
    assert log.calibration_sha_verified is True

    # Trusted fit segment frames are accepted; the ray passes through the
    # world point that produced the pixel (sha-verified calibration only).
    frame0 = log.frames[0]
    assert frame0.observation_status == "observed"
    assert frame0.solver_verdict == "accepted"
    assert frame0.ray_status == "computed"
    assert (
        _point_to_ray_distance(WORLDS[0], frame0.ray.origin_m, frame0.ray.direction) < 1e-3
    )

    # Untrusted fallback segment fails closed.
    assert log.frames[3].solver_verdict == "rejected_fail_closed"
    assert log.frames[3].segment_id == 1

    # Hidden frame: no pixel, no ray, verdict hidden.
    frame4 = log.frames[4]
    assert frame4.observation_status == "missing"
    assert frame4.pixel_xy is None
    assert frame4.ray is None
    assert frame4.ray_status == "no_pixel"
    assert frame4.solver_verdict == "hidden"

    # Frame without per-frame provenance inside the untrusted span fails closed.
    assert log.frames[5].segment_id is None
    assert log.frames[5].solver_verdict == "rejected_fail_closed"

    # Anchor events are attached at their frame.
    assert [event.kind for event in log.frames[2].anchor_events] == ["bounce"]
    assert log.frames[0].anchor_events == ()

    # Provenance: every recorded input carries a sha256 of the actual bytes.
    kinds = {artifact.kind: artifact for artifact in log.inputs}
    assert set(kinds) == {"ball_track_arc_solved", "ball_chain_manifest", "court_calibration"}
    arc_sha = hashlib.sha256((clip_dir / "ball_track_arc_solved.json").read_bytes()).hexdigest()
    assert kinds["ball_track_arc_solved"].sha256 == arc_sha


def test_cli_output_bytes_are_deterministic(tmp_path):
    clip_dir = _write_clip_dir(tmp_path)
    first_dir = tmp_path / "out_first"
    second_dir = tmp_path / "out_second"
    first = _run_cli(clip_dir, first_dir)
    second = _run_cli(clip_dir, second_dir)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_bytes = (first_dir / "synthetic_mini.observation_log.json").read_bytes()
    second_bytes = (second_dir / "synthetic_mini.observation_log.json").read_bytes()
    assert first_bytes == second_bytes


def test_unverified_calibration_fails_closed_to_no_rays(tmp_path):
    clip_dir = _write_clip_dir(tmp_path, tamper_calibration_sha=True)
    out_dir = tmp_path / "out"
    result = _run_cli(clip_dir, out_dir)
    assert result.returncode == 0, result.stderr
    log = read_solver_observation_log(out_dir / "synthetic_mini.observation_log.json")
    assert log.calibration_sha_verified is False
    for frame in log.frames:
        assert frame.ray is None
        if frame.pixel_xy is not None:
            assert frame.ray_status == "calibration_not_sha_verified"
        else:
            assert frame.ray_status == "no_pixel"


def test_missing_arc_solved_is_an_error(tmp_path):
    clip_dir = tmp_path / "empty_clip"
    clip_dir.mkdir()
    result = _run_cli(clip_dir, tmp_path / "out")
    assert result.returncode == 2
    assert "ball_track_arc_solved.json" in result.stderr
