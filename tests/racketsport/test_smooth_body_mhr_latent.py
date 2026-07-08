from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport import smooth_body_mhr_latent as sml
from threed.racketsport import mhr_decode


def test_cli_help_references_scaffold_command_path() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/smooth_body_mhr_latent.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "lambda-smooth" in completed.stdout
    assert "lambda-foot" in completed.stdout


# ---------------------------------------------------------------------------
# Pure-numpy sliding-window ridge solver: no torch/roma dependency, must be
# robustly CPU-testable.
# ---------------------------------------------------------------------------
def test_second_difference_matrix_shape_and_zero_for_tiny_windows() -> None:
    assert sml.second_difference_matrix(1).shape == (0, 1)
    assert sml.second_difference_matrix(2).shape == (0, 2)
    d = sml.second_difference_matrix(5)
    assert d.shape == (3, 5)
    # row 0 acts on x0,x1,x2 as [1,-2,1] (discrete 2nd derivative)
    assert np.allclose(d[0], [1.0, -2.0, 1.0, 0.0, 0.0])


def test_solve_smoothing_window_returns_raw_when_lambda_zero_or_window_too_small() -> None:
    raw = np.array([[0.0], [10.0], [0.0]])
    assert np.allclose(sml.solve_smoothing_window(raw, 0.0), raw)
    tiny = np.array([[0.0], [10.0]])
    assert np.allclose(sml.solve_smoothing_window(tiny, 5.0), tiny)


def test_solve_smoothing_window_smooths_a_single_spike() -> None:
    # A single-frame spike in an otherwise-flat sequence should shrink toward
    # its neighbors as lambda_smooth grows (this is a ridge-regularized 2nd
    # difference penalty -- monotonic in lambda for this well-conditioned case).
    raw = np.array([[0.0], [0.0], [10.0], [0.0], [0.0]])
    low = sml.solve_smoothing_window(raw, 0.1)
    high = sml.solve_smoothing_window(raw, 5.0)
    assert abs(high[2, 0]) < abs(low[2, 0]) < 10.0
    # Data term always keeps SOME pull toward the raw spike.
    assert high[2, 0] > 0.0


def test_sliding_window_smooth_preserves_shape_and_handles_boundaries() -> None:
    rng = np.random.default_rng(0)
    sequence = rng.normal(size=(20, 4))
    out = sml.sliding_window_smooth(sequence, window=9, lambda_smooth=0.3)
    assert out.shape == sequence.shape
    assert np.all(np.isfinite(out))
    # lambda=0 must be a no-op (pure data term).
    identity = sml.sliding_window_smooth(sequence, window=9, lambda_smooth=0.0)
    assert np.allclose(identity, sequence)


def test_sliding_window_smooth_reduces_jitter_relative_to_raw() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 60)
    smooth_signal = np.sin(t)
    noisy = smooth_signal + rng.normal(scale=0.3, size=t.shape)
    out = sml.sliding_window_smooth(noisy[:, None], window=9, lambda_smooth=0.3)[:, 0]
    # second-difference "jitter" proxy: smaller after smoothing.
    raw_jitter = np.abs(np.diff(noisy, n=2)).mean()
    smoothed_jitter = np.abs(np.diff(out, n=2)).mean()
    assert smoothed_jitter < raw_jitter


def test_median_lock_replaces_every_row_with_the_column_median() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    locked = sml.median_lock(values)
    assert locked.shape == values.shape
    assert np.allclose(locked, np.array([[2.0, 20.0], [2.0, 20.0], [2.0, 20.0]]))


def test_median_lock_handles_empty_input() -> None:
    empty = np.zeros((0, 5))
    assert sml.median_lock(empty).shape == (0, 5)


# ---------------------------------------------------------------------------
# body_mesh.json extraction.
# ---------------------------------------------------------------------------
def _synthetic_body_mesh(*, frame_count: int = 5) -> dict:
    frames = []
    for idx in range(frame_count):
        frames.append(
            {
                "frame_idx": idx,
                "t": idx / 30.0,
                "mesh_vertices_world": [[0.0, 0.0, 0.0]],
                "smplx_params": {
                    "global_orient": [0.01 * idx, 0.0, 0.0],
                    "body_pose": [0.001 * idx] * sml.BODY_POSE_EULER_DIM,
                    "betas": [0.1] * sml.SHAPE_DIM,
                    "scale": [0.2] * sml.SCALE_DIM,
                    "transl_world": [1.0, 2.0, 0.0],
                },
            }
        )
    return {"artifact_type": "racketsport_body_mesh", "clip": "synthetic", "players": [{"id": 7, "frames": frames}]}


def test_extract_player_pose_sequences_happy_path() -> None:
    body_mesh = _synthetic_body_mesh(frame_count=6)
    extracted = sml.extract_player_pose_sequences(body_mesh)
    assert set(extracted.keys()) == {"7"}
    player = extracted["7"]
    assert player["global_orient"].shape == (6, sml.GLOBAL_ROT_EULER_DIM)
    assert player["body_pose"].shape == (6, sml.BODY_POSE_EULER_DIM)
    assert player["betas"].shape == (6, sml.SHAPE_DIM)
    assert player["scale"] is not None
    assert player["scale"].shape == (6, sml.SCALE_DIM)
    assert player["skipped_frame_count"] == 0
    assert np.allclose(player["track_world_xy"][0], [1.0, 2.0])


def test_extract_player_pose_sequences_marks_missing_scale() -> None:
    body_mesh = _synthetic_body_mesh(frame_count=3)
    for frame in body_mesh["players"][0]["frames"]:
        frame["smplx_params"].pop("scale")
    extracted = sml.extract_player_pose_sequences(body_mesh)
    assert extracted["7"]["scale"] is None


def test_extract_player_pose_sequences_skips_frames_missing_pose() -> None:
    body_mesh = _synthetic_body_mesh(frame_count=3)
    body_mesh["players"][0]["frames"][1]["smplx_params"]["body_pose"] = []
    extracted = sml.extract_player_pose_sequences(body_mesh)
    assert len(extracted["7"]["frame_idx"]) == 2
    assert extracted["7"]["skipped_frame_count"] == 1


# ---------------------------------------------------------------------------
# CLI end-to-end smoke path (--skip-decode, or the runtime is simply absent):
# numpy-only, must succeed with zero GPU/torch dependency.
# ---------------------------------------------------------------------------
def test_run_produces_smoothing_report_without_decode(tmp_path: Path) -> None:
    body_mesh = _synthetic_body_mesh(frame_count=12)
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(body_mesh), encoding="utf-8")
    out_dir = tmp_path / "out"

    parser = sml.build_arg_parser()
    args = parser.parse_args(
        [
            "--body-mesh",
            str(body_mesh_path),
            "--out-dir",
            str(out_dir),
            "--window",
            "9",
            "--lambda-smooth",
            "0.1,0.3,0.6",
            "--skip-decode",
        ]
    )
    report = sml.run(args)
    assert report["decode_available"] is False
    assert report["lambda_foot"] == 0.0
    assert set(report["lambda_smooth_sweep"]) == {0.1, 0.3, 0.6}
    assert "7" in report["players"]
    for lam_key in ("0.1", "0.3", "0.6"):
        assert lam_key in report["players"]["7"]["by_lambda"]
        assert report["players"]["7"]["by_lambda"][lam_key]["mode"] == "euler_space_smoke_path_no_runtime"
    written = json.loads((out_dir / "smoothing_report.json").read_text(encoding="utf-8"))
    assert written == report


def test_run_rejects_nonzero_lambda_foot(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(_synthetic_body_mesh()), encoding="utf-8")
    parser = sml.build_arg_parser()
    args = parser.parse_args(
        [
            "--body-mesh",
            str(body_mesh_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--lambda-foot",
            "1.0",
            "--skip-decode",
        ]
    )
    with pytest.raises(SystemExit):
        sml.run(args)


def test_run_rejects_empty_lambda_smooth(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(_synthetic_body_mesh()), encoding="utf-8")
    parser = sml.build_arg_parser()
    args = parser.parse_args(
        [
            "--body-mesh",
            str(body_mesh_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--lambda-smooth",
            "",
            "--skip-decode",
        ]
    )
    with pytest.raises(SystemExit):
        sml.run(args)


def test_cli_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(_synthetic_body_mesh(frame_count=10)), encoding="utf-8")
    out_dir = tmp_path / "out"
    exit_code = sml.main(
        [
            "--body-mesh",
            str(body_mesh_path),
            "--out-dir",
            str(out_dir),
            "--skip-decode",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["player_count"] == 1
    assert (out_dir / "smoothing_report.json").is_file()
