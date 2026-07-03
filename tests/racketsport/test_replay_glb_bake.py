from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pygltflib as gltf
import pytest

from threed.racketsport.replay_glb_bake import BodyMeshBakeError, build_animated_body_glb


def _body_mesh(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "test_clip",
        "mesh_faces": [[0, 1, 2]],
        "players": [
            {
                "id": 7,
                "frames": [
                    {"frame_idx": 10, "t": 1.0, "mesh_vertices_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
                    {"frame_idx": 11, "t": 1.1, "mesh_vertices_world": [[0.0, 0.0, 0.1], [1.1, 0.0, 0.0], [0.0, 1.0, 0.0]]},
                    {"frame_idx": 12, "t": 1.2, "mesh_vertices_world": [[0.0, 0.0, 0.2], [1.2, 0.0, 0.0], [0.0, 1.0, 0.0]]},
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _load(data: bytes) -> gltf.GLTF2:
    return gltf.GLTF2().load_from_bytes(data)


def _read_floats(document: gltf.GLTF2, accessor_index: int) -> tuple[float, ...]:
    accessor = document.accessors[accessor_index]
    buffer_view = document.bufferViews[accessor.bufferView]
    blob = document.binary_blob()
    start = buffer_view.byteOffset
    component_count = {"SCALAR": 1, "VEC3": 3}[accessor.type]
    count = accessor.count * component_count
    return struct.unpack(f"<{count}f", blob[start : start + count * 4])


class TestBuildAnimatedBodyGlb:
    def test_produces_loadable_glb_bytes(self) -> None:
        data = build_animated_body_glb(_body_mesh(), clip="test_clip")

        assert isinstance(data, bytes)
        assert data[:4] == b"glTF"  # GLB magic
        document = _load(data)
        assert document.asset.version == "2.0"

    def test_base_pose_is_the_lowest_frame_idx_frame_regardless_of_input_order(self) -> None:
        body_mesh = _body_mesh()
        body_mesh["players"][0]["frames"] = list(reversed(body_mesh["players"][0]["frames"]))

        document = _load(build_animated_body_glb(body_mesh, clip="test_clip"))
        primitive = document.meshes[0].primitives[0]
        base_position = _read_floats(document, primitive.attributes.POSITION)

        assert base_position == pytest.approx((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    def test_morph_targets_are_deltas_from_the_base_frame(self) -> None:
        document = _load(build_animated_body_glb(_body_mesh(), clip="test_clip"))
        primitive = document.meshes[0].primitives[0]

        assert len(primitive.targets) == 2  # 3 frames -> 1 base + 2 morph targets
        first_target_delta = _read_floats(document, primitive.targets[0]["POSITION"])
        # frame_idx=11 vertex 0 is [0,0,0.1] vs base [0,0,0] -> delta [0,0,0.1]
        assert first_target_delta[0:3] == pytest.approx((0.0, 0.0, 0.1))
        # vertex 1: [1.1,0,0] vs base [1,0,0] -> delta [0.1, 0, 0]
        assert first_target_delta[3:6] == pytest.approx((0.1, 0.0, 0.0))
        # vertex 2 unchanged across all frames -> delta [0, 0, 0]
        assert first_target_delta[6:9] == pytest.approx((0.0, 0.0, 0.0))

    def test_weights_animation_is_a_one_hot_step_flipbook_at_real_frame_timestamps(self) -> None:
        document = _load(build_animated_body_glb(_body_mesh(), clip="test_clip"))
        animation = document.animations[0]
        sampler = animation.samplers[0]

        assert sampler.interpolation == gltf.ANIM_STEP
        times = _read_floats(document, sampler.input)
        assert times == pytest.approx((1.0, 1.1, 1.2))  # real frame timestamps, not synthetic ones
        weights = _read_floats(document, sampler.output)
        # base keyframe (t=1.0): both targets off; then one-hot per subsequent frame.
        assert weights == pytest.approx((0.0, 0.0, 1.0, 0.0, 0.0, 1.0))

        channel = animation.channels[0]
        assert channel.target.path == "weights"
        assert channel.target.node == document.nodes.index(
            next(node for node in document.nodes if node.mesh == 0)
        )

    def test_indices_use_16_bit_component_type_when_vertex_count_fits(self) -> None:
        document = _load(build_animated_body_glb(_body_mesh(), clip="test_clip"))
        index_accessor = document.accessors[document.meshes[0].primitives[0].indices]

        assert index_accessor.componentType == gltf.UNSIGNED_SHORT

    def test_bakes_one_mesh_node_per_player(self) -> None:
        body_mesh = _body_mesh()
        body_mesh["players"].append(
            {
                "id": 8,
                "frames": [
                    {"frame_idx": 20, "t": 2.0, "mesh_vertices_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
                ],
            }
        )

        document = _load(build_animated_body_glb(body_mesh, clip="test_clip"))

        assert len(document.meshes) == 2
        assert len(document.nodes) == 2
        # player 8 has only 1 frame -> no morph targets (nothing to interpolate between).
        assert not document.meshes[1].primitives[0].targets
        animation_names = {animation.name for animation in document.animations}
        # player 7 (3 frames): flipbook + visibility gating. player 8 (1 frame): no
        # flipbook, but it still gets visibility gating so a single-frame mesh doesn't
        # render as a frozen static body for the whole clip (review finding F6).
        assert animation_names == {
            "player7_body_mesh_flipbook",
            "player7_body_mesh_visibility",
            "player8_body_mesh_visibility",
        }

    def test_visibility_scale_animation_hides_mesh_outside_contiguous_window(self) -> None:
        """Regression test for review finding F6 (2026-07-02, MEDIUM).

        A player's single contiguous window (frames 10-12) must be bracketed by
        scale=(0,0,0) keyframes immediately before and after it -- otherwise glTF's
        "hold the nearest sample" extrapolation would leave the base/rest mesh visible
        for the entire clip, not just the window it was actually baked for.
        """

        document = _load(build_animated_body_glb(_body_mesh(), clip="test_clip"))
        visibility = next(a for a in document.animations if a.name == "player7_body_mesh_visibility")
        sampler = visibility.samplers[0]

        assert sampler.interpolation == gltf.ANIM_STEP
        times = _read_floats(document, sampler.input)
        scales = _read_floats(document, sampler.output)
        fps = 30.0
        epsilon = 0.5 / fps
        assert times == pytest.approx((1.0 - epsilon, 1.0, 1.2 + epsilon))
        assert scales == pytest.approx((0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))

        channel = visibility.channels[0]
        assert channel.target.path == "scale"
        assert channel.target.node == document.nodes.index(next(node for node in document.nodes if node.mesh == 0))

    def test_visibility_scale_animation_hides_mesh_in_gap_between_non_contiguous_windows(self) -> None:
        """Regression test for review finding F6 (2026-07-02, MEDIUM).

        Two contact windows separated by an unscheduled gap (frames 10-11, then frame
        40, matching the finding's concrete scenario) must each be independently
        bracketed by invisible keyframes -- the mesh must not stay visible/frozen for
        frames 12-39, which were never computed.
        """

        body_mesh = _body_mesh(
            players=[
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
                            "t": 1.1,
                            "source_window_index": 0,
                            "mesh_vertices_world": [[0.0, 0.0, 0.1], [1.1, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        },
                        {
                            "frame_idx": 40,
                            "t": 4.0,
                            "source_window_index": 1,
                            "mesh_vertices_world": [[0.0, 0.0, 0.2], [1.2, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        },
                    ],
                }
            ]
        )

        document = _load(build_animated_body_glb(body_mesh, clip="test_clip"))
        visibility = next(a for a in document.animations if a.name == "player7_body_mesh_visibility")
        sampler = visibility.samplers[0]

        times = _read_floats(document, sampler.input)
        scales = _read_floats(document, sampler.output)
        fps = 30.0
        epsilon = 0.5 / fps
        assert times == pytest.approx((1.0 - epsilon, 1.0, 1.1 + epsilon, 4.0, 4.0 + epsilon))
        assert scales == pytest.approx(
            (
                0.0, 0.0, 0.0,  # before window 0: invisible
                1.0, 1.0, 1.0,  # window 0 start: visible
                0.0, 0.0, 0.0,  # window 0 end (+eps): invisible -- the frames 12-39 gap
                1.0, 1.0, 1.0,  # window 1 start: visible
                0.0, 0.0, 0.0,  # window 1 end (+eps): invisible
            )
        )
        # times strictly increasing -> a spec-conformant STEP-interpolation renderer
        # holds scale=(0,0,0) for the entire 1.1+eps..4.0 gap, never showing a frozen mesh.
        assert all(later > earlier for earlier, later in zip(times, times[1:]))

    def test_single_frame_player_still_gets_visibility_gating(self) -> None:
        """A mesh baked for exactly one frame must not render as a frozen static body
        for the rest of the clip (review finding F6): no morph targets exist to bake a
        flipbook animation from, but the node must still be scale-gated invisible
        outside that one instant."""

        body_mesh = _body_mesh(
            players=[
                {
                    "id": 9,
                    "frames": [
                        {"frame_idx": 20, "t": 2.0, "mesh_vertices_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
                    ],
                }
            ]
        )

        document = _load(build_animated_body_glb(body_mesh, clip="test_clip"))

        assert not any(animation.name == "player9_body_mesh_flipbook" for animation in document.animations)
        visibility = next(a for a in document.animations if a.name == "player9_body_mesh_visibility")
        times = _read_floats(document, visibility.samplers[0].input)
        scales = _read_floats(document, visibility.samplers[0].output)
        fps = 30.0
        epsilon = 0.5 / fps
        assert times == pytest.approx((2.0 - epsilon, 2.0, 2.0 + epsilon))
        assert scales == pytest.approx((0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))

    def test_rejects_empty_mesh_faces(self) -> None:
        with pytest.raises(BodyMeshBakeError):
            build_animated_body_glb(_body_mesh(mesh_faces=[]), clip="test_clip")

    def test_rejects_players_with_no_frames(self) -> None:
        with pytest.raises(BodyMeshBakeError):
            build_animated_body_glb(_body_mesh(players=[{"id": 7, "frames": []}]), clip="test_clip")

    def test_rejects_frames_with_mismatched_vertex_counts(self) -> None:
        body_mesh = _body_mesh()
        body_mesh["players"][0]["frames"][1]["mesh_vertices_world"] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(BodyMeshBakeError):
            build_animated_body_glb(body_mesh, clip="test_clip")


def test_run_build_replay_animated_glb_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_replay_animated_glb.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--body-mesh" in completed.stdout


def test_run_build_replay_animated_glb_cli_fails_closed_on_missing_body_mesh(tmp_path: Path) -> None:
    out_path = tmp_path / "body_mesh.glb"
    missing_body_mesh = tmp_path / "does_not_exist_body_mesh.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_animated_glb.py",
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


def test_run_build_replay_animated_glb_cli_bakes_a_real_glb_end_to_end(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "body_mesh.json"
    body_mesh_path.write_text(json.dumps(_body_mesh()), encoding="utf-8")
    out_path = tmp_path / "body_mesh.glb"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_animated_glb.py",
            "--clip",
            "clip_001",
            "--body-mesh",
            str(body_mesh_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert out_path.exists()
    summary = json.loads(completed.stdout)
    assert summary["players"] == [{"id": 7, "frame_count": 3}]
    assert out_path.read_bytes()[:4] == b"glTF"
