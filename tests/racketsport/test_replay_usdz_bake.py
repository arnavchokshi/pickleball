from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core (pxr) not installed in this environment")
from pxr import Usd, UsdGeom  # noqa: E402

from threed.racketsport.replay_usdz_bake import BodyMeshUsdzBakeError, build_animated_body_usdz  # noqa: E402


def _body_mesh(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "test_clip",
        "fps": 30.0,
        "mesh_faces": [[0, 1, 2]],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 10,
                        "t": 1.0,
                        "source_window_index": 0,
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    },
                    {
                        "frame_idx": 11,
                        "t": 1.033,
                        "source_window_index": 0,
                        "mesh_vertices_world": [[0.0, 0.0, 0.1], [1.1, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    },
                    {
                        "frame_idx": 40,
                        "t": 2.0,
                        "source_window_index": 1,
                        "mesh_vertices_world": [[0.0, 0.0, 0.5], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    },
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _single_window_body_mesh(frame_count: int = 6) -> dict:
    payload = _body_mesh()
    payload["players"][0]["frames"] = [
        {
            "frame_idx": 10 + i,
            "t": (10 + i) / 30.0,
            "source_window_index": 0,
            "mesh_vertices_world": [
                [float(i) + 0.1234, 0.0, 0.0],
                [1.0 + float(i), 0.0, 0.0],
                [0.0, 1.0 + float(i), 0.0],
            ],
        }
        for i in range(frame_count)
    ]
    return payload


class TestBuildAnimatedBodyUsdz:
    def test_writes_an_openable_usdz_package(self, tmp_path) -> None:
        out_path = tmp_path / "bake.usdz"

        summary = build_animated_body_usdz(_body_mesh(), clip="test_clip", out_path=out_path)

        assert out_path.exists()
        assert summary["out_bytes"] > 0
        assert summary["frame_range"] == [10, 40]
        stage = Usd.Stage.Open(str(out_path))
        assert stage is not None
        assert stage.GetStartTimeCode() == 10.0
        assert stage.GetEndTimeCode() == 40.0

    def test_mesh_topology_matches_mesh_faces(self, tmp_path) -> None:
        out_path = tmp_path / "bake.usdz"
        build_animated_body_usdz(_body_mesh(), clip="test_clip", out_path=out_path)

        stage = Usd.Stage.Open(str(out_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/test_clip/player7_body"))
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3]
        assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2]

    def test_points_are_time_sampled_per_scheduled_frame(self, tmp_path) -> None:
        out_path = tmp_path / "bake.usdz"
        build_animated_body_usdz(_body_mesh(), clip="test_clip", out_path=out_path)

        stage = Usd.Stage.Open(str(out_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/test_clip/player7_body"))
        points_at_10 = mesh.GetPointsAttr().Get(10)
        points_at_40 = mesh.GetPointsAttr().Get(40)

        assert tuple(points_at_10[0]) == pytest.approx((0.0, 0.0, 0.0))
        assert tuple(points_at_40[0]) == pytest.approx((0.0, 0.0, 0.5))

    def test_max_mesh_frames_subsamples_evenly_and_preserves_window_visibility(self, tmp_path) -> None:
        out_path = tmp_path / "bake.usdz"

        summary = build_animated_body_usdz(
            _single_window_body_mesh(),
            clip="test_clip",
            out_path=out_path,
            max_mesh_frames=3,
        )

        assert summary["source_mesh_frame_count"] == 6
        assert summary["baked_mesh_frame_count"] == 3
        assert summary["compression_profile"]["max_mesh_frames"] == 3
        stage = Usd.Stage.Open(str(out_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/test_clip/player7_body"))
        assert mesh.GetPointsAttr().GetTimeSamples() == [10.0, 12.0, 15.0]
        visibility = UsdGeom.Imageable(mesh).GetVisibilityAttr()
        assert visibility.Get(9) == UsdGeom.Tokens.invisible
        assert visibility.Get(10) == UsdGeom.Tokens.inherited
        assert visibility.Get(15) == UsdGeom.Tokens.inherited
        assert visibility.Get(16) == UsdGeom.Tokens.invisible

    def test_round_decimals_quantizes_authored_points_without_changing_topology(self, tmp_path) -> None:
        out_path = tmp_path / "bake.usdz"

        build_animated_body_usdz(
            _single_window_body_mesh(frame_count=2),
            clip="test_clip",
            out_path=out_path,
            round_decimals=2,
        )

        stage = Usd.Stage.Open(str(out_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/test_clip/player7_body"))
        points_at_10 = mesh.GetPointsAttr().Get(10)
        assert tuple(points_at_10[0]) == pytest.approx((0.12, 0.0, 0.0))
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3]

    def test_visibility_is_gated_to_scheduled_windows_only(self, tmp_path) -> None:
        # Frames 10-11 form one contiguous window, frame 40 is a separate window
        # (per source_window_index) far away on the timeline. Visibility must be
        # "inherited" only inside those two windows and "invisible" everywhere
        # else, including *before* the very first window (regression coverage for
        # USD's outside-sample-range hold-constant behavior).
        out_path = tmp_path / "bake.usdz"
        build_animated_body_usdz(_body_mesh(), clip="test_clip", out_path=out_path)

        stage = Usd.Stage.Open(str(out_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/test_clip/player7_body"))
        visibility = UsdGeom.Imageable(mesh).GetVisibilityAttr()

        assert visibility.Get(9) == UsdGeom.Tokens.invisible
        assert visibility.Get(10) == UsdGeom.Tokens.inherited
        assert visibility.Get(11) == UsdGeom.Tokens.inherited
        assert visibility.Get(12) == UsdGeom.Tokens.invisible
        assert visibility.Get(39) == UsdGeom.Tokens.invisible
        assert visibility.Get(40) == UsdGeom.Tokens.inherited
        assert visibility.Get(41) == UsdGeom.Tokens.invisible

    def test_bakes_one_mesh_prim_per_player(self, tmp_path) -> None:
        body_mesh = _body_mesh()
        body_mesh["players"].append(
            {
                "id": 8,
                "frames": [
                    {
                        "frame_idx": 5,
                        "t": 0.5,
                        "source_window_index": 0,
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    },
                ],
            }
        )
        out_path = tmp_path / "bake.usdz"

        summary = build_animated_body_usdz(body_mesh, clip="test_clip", out_path=out_path)

        assert {p["id"] for p in summary["players"]} == {7, 8}
        stage = Usd.Stage.Open(str(out_path))
        assert stage.GetPrimAtPath("/test_clip/player7_body").IsValid()
        assert stage.GetPrimAtPath("/test_clip/player8_body").IsValid()

    def test_rejects_empty_mesh_faces(self, tmp_path) -> None:
        with pytest.raises(BodyMeshUsdzBakeError):
            build_animated_body_usdz(_body_mesh(mesh_faces=[]), clip="test_clip", out_path=tmp_path / "bake.usdz")

    def test_rejects_players_with_no_frames(self, tmp_path) -> None:
        with pytest.raises(BodyMeshUsdzBakeError):
            build_animated_body_usdz(
                _body_mesh(players=[{"id": 7, "frames": []}]), clip="test_clip", out_path=tmp_path / "bake.usdz"
            )

    def test_rejects_frames_with_mismatched_vertex_counts(self, tmp_path) -> None:
        body_mesh = _body_mesh()
        body_mesh["players"][0]["frames"][1]["mesh_vertices_world"] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(BodyMeshUsdzBakeError):
            build_animated_body_usdz(body_mesh, clip="test_clip", out_path=tmp_path / "bake.usdz")


def test_run_build_replay_animated_usdz_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_replay_animated_usdz.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--body-mesh" in completed.stdout
    assert "--max-mesh-frames" in completed.stdout


def test_run_build_replay_animated_usdz_cli_fails_closed_on_missing_body_mesh(tmp_path: Path) -> None:
    out_path = tmp_path / "body_mesh.usdz"
    missing_body_mesh = tmp_path / "does_not_exist_body_mesh.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_animated_usdz.py",
            "--clip",
            "clip_001",
            "--body-mesh",
            str(missing_body_mesh),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not out_path.exists()


def test_run_build_replay_animated_usdz_cli_bakes_a_real_usdz_end_to_end(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(_single_window_body_mesh()), encoding="utf-8")
    out_path = tmp_path / "body_mesh.usdz"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_animated_usdz.py",
            "--clip",
            "clip_001",
            "--body-mesh",
            str(body_mesh_path),
            "--out",
            str(out_path),
            "--max-mesh-frames",
            "3",
            "--round-decimals",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert out_path.exists()
    summary = json.loads(completed.stdout)
    assert summary["players"] == [{"id": 7, "frame_count": 3, "window_count": 1, "vertex_count": 3}]
    assert summary["source_mesh_frame_count"] == 6
    assert summary["baked_mesh_frame_count"] == 3
