from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_metric3d_contract import read_solver_observation_log
from threed.racketsport.ball_solver_characterization import (
    _frame_state as char_frame_state,
    _frames as char_frames,
    _untrusted_spans as char_untrusted_spans,
    discover_clip_inputs,
)
from threed.racketsport.virtual_world import ball_arc_segment_fail_closed_verdicts

CLI_PATH = "scripts/racketsport/build_ball_observation_log.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_ball_observation_log_under_test", REPO_ROOT / CLI_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    assert log.source_clip_id == "synthetic_mini"  # from the artifact's clip_id
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


# ---------------------------------------------------------------------------
# Verdict cross-consistency vs the characterization module
# ---------------------------------------------------------------------------
#
# The observation log deliberately re-implements (not imports) the frozen
# characterization module's per-frame accepted/suppressed/hidden policy.
# These tests run BOTH implementations over one fixture so any drift fails
# loudly. Known deliberate divergence, asserted explicitly below: when an
# untrusted segment has no integer frame span, the log fails provenance-less
# frames closed clip-wide; the characterization module drops such segments
# from its span list.


def _verdict_arc_fixture(*, include_spanless_untrusted: bool) -> dict:
    """8-frame fixture: trusted span, untrusted span, missing-verdict segment
    reference, provenance-less frames inside AND outside untrusted spans,
    and a hidden frame."""

    def _frame(index: int, *, world, segment_id=None, band="anchored_measured", visible=True):
        frame = {
            "t": index / 30.0,
            "visible": visible,
            "xy": [100.0 + index, 200.0],
            "conf": 0.9 if visible else 0.0,
            "band": band,
            "world_xyz": world,
        }
        if segment_id is not None:
            frame["arc_solver"] = {"segment_id": segment_id}
        return frame

    frames = [
        _frame(0, world=[0.0, -4.0, 0.5], segment_id=0),
        _frame(1, world=[0.1, -3.5, 0.8], segment_id=0),
        _frame(2, world=[0.2, -3.0, 1.0], segment_id=0),
        _frame(3, world=[0.4, 2.0, 1.2], segment_id=1),  # untrusted segment
        _frame(4, world=[0.5, 2.5, 1.1]),  # provenance-less, inside span 3-4
        _frame(5, world=[0.6, 3.0, 1.0], segment_id=99),  # missing verdict
        _frame(6, world=[0.7, 3.5, 0.9]),  # provenance-less, outside spans
        _frame(7, world=None, band="hidden", visible=False),
    ]
    segments = [
        {
            "segment_id": 0,
            "status": "fit",
            "frame_start": 0,
            "frame_end": 2,
            "inlier_count": 3,
            "outlier_count": 0,
            "max_reprojection_error_px": 5.0,
        },
        {
            "segment_id": 1,
            "status": "fit_bvp_fallback",
            "frame_start": 3,
            "frame_end": 4,
            "inlier_count": 0,
            "outlier_count": 4,
            "max_reprojection_error_px": 90.0,
        },
        # segment 99 is intentionally absent: frame 5's verdict is missing.
    ]
    if include_spanless_untrusted:
        segments.append(
            {
                "segment_id": 2,
                "status": "fit_bvp_fallback",
                "frame_start": None,  # unlocatable untrusted segment
                "frame_end": None,
                "inlier_count": 0,
                "outlier_count": 3,
                "max_reprojection_error_px": 80.0,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "ball_track_arc_solved",
        "clip_id": "verdict_consistency",
        "status": "ran",
        "kill_reasons": [],
        "frames": frames,
        "segments": segments,
        "anchors": [],
    }


def _characterization_verdicts(arc_solved: dict) -> list[str]:
    verdicts = ball_arc_segment_fail_closed_verdicts(arc_solved.get("segments"))
    spans = char_untrusted_spans(verdicts)
    result = []
    for index, frame in enumerate(char_frames(arc_solved)):
        state = char_frame_state(frame, index, verdicts=verdicts, untrusted_spans=spans)
        if state.hidden:
            result.append("hidden")
        elif state.accepted:
            result.append("accepted")
        else:
            result.append("rejected_fail_closed")
    return result


def _observation_log_verdicts(arc_solved: dict, tmp_path: Path) -> list[str]:
    clip_dir = tmp_path / "verdict_clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "ball_track_arc_solved.json").write_text(
        json.dumps(arc_solved, indent=2), encoding="utf-8"
    )
    module = _load_script_module()
    inputs = discover_clip_inputs("verdict_consistency", clip_dir)
    log = module.build_solver_observation_log(inputs, root=tmp_path)
    return [frame.solver_verdict for frame in log.frames]


def test_verdicts_match_characterization_module_per_frame(tmp_path):
    arc_solved = _verdict_arc_fixture(include_spanless_untrusted=False)
    char = _characterization_verdicts(arc_solved)
    log = _observation_log_verdicts(arc_solved, tmp_path)
    assert log == char
    # The fixture actually exercises every verdict class.
    assert char == [
        "accepted",
        "accepted",
        "accepted",
        "rejected_fail_closed",  # untrusted segment, own provenance
        "rejected_fail_closed",  # provenance-less inside untrusted span
        "rejected_fail_closed",  # references a segment with no verdict
        "accepted",  # provenance-less outside all untrusted spans
        "hidden",
    ]


def test_spanless_untrusted_segment_fails_closed_and_is_never_more_permissive(tmp_path):
    arc_solved = _verdict_arc_fixture(include_spanless_untrusted=True)
    char = _characterization_verdicts(arc_solved)
    log = _observation_log_verdicts(arc_solved, tmp_path)
    # Deliberate stricter-than-characterization divergence: the unlocatable
    # untrusted segment fails ALL provenance-less frames closed in the log.
    assert char[4] == "rejected_fail_closed" and log[4] == "rejected_fail_closed"
    assert char[6] == "accepted"  # characterization drops the span-less segment
    assert log[6] == "rejected_fail_closed"  # the log fails it closed
    # Everywhere else the implementations agree, and the log is NEVER more
    # permissive than the characterization module.
    for index, (char_verdict, log_verdict) in enumerate(zip(char, log)):
        if index != 6:
            assert log_verdict == char_verdict, index
        if log_verdict == "accepted":
            assert char_verdict == "accepted", index
