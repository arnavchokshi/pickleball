from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.strict_placement_rollup import build_strict_placement_rollup


COMMAND_PATH = "scripts/racketsport/strict_placement_rollup.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frames(x: float, y: float, count: int = 10) -> list[dict]:
    return [{"t": i / 30.0, "world_xy": [x, y]} for i in range(count)]


def _base_clip(tmp_path: Path) -> tuple[Path, Path]:
    clip = tmp_path / "clip_a"
    _write_json(
        clip / "PIPELINE_SUMMARY.json",
        {
            "status": "complete",
            "stages": [
                {"stage": stage, "status": "ran"}
                for stage in ["calibration", "tracking", "placement", "body", "world", "manifest"]
            ],
        },
    )
    _write_json(
        clip / "tracks.json",
        {
            "players": [
                {"id": 1, "side": "near", "role": "left", "frames": _frames(-2.0, -2.0)},
                {"id": 2, "side": "near", "role": "right", "frames": _frames(2.0, -2.0)},
                {"id": 3, "side": "far", "role": "left", "frames": _frames(-2.0, 2.0)},
                {"id": 4, "side": "far", "role": "right", "frames": _frames(2.0, 2.0)},
            ],
        },
    )
    _write_json(
        clip / "placement.json",
        {
            "provenance": {
                "native2d_keypoints": "keypoints_2d.json",
                "source_counts": {"native2d": 40},
                "sidecar_identity_diagnostics": {
                    "accepted_mappings": [
                        {"player_id": 1, "sidecar_id": 1, "accepted": True, "integer_match": True}
                    ],
                    "dropped_counts": {},
                    "mapping_votes": {},
                },
            }
        },
    )
    _write_json(clip / "body_full_clip_gate.json", {"passed": True})
    _write_json(
        clip / "body_grounding_quality.json",
        {
            "grounding_metrics": {
                "foot_lock_slide_p95_m": 0.010,
                "foot_lock_slide_p95_by_player_m": {"1": 0.010, "2": 0.011, "3": 0.012, "4": 0.013},
                "max_foot_lock_slide_m": 0.020,
            }
        },
    )
    _write_json(clip / "trust_bands.json", {"body": {"badge": "preview"}})
    _write_json(clip / "confidence_gate_summary.json", {"schema_version": 1})
    _write_json(clip / "confidence_gated_world.json", {"small": True})
    _write_json(clip / "replay_viewer_manifest.json", {"skeleton_only": True, "virtual_world_url": "confidence_gated_world.json"})
    membership = tmp_path / "membership.json"
    _write_json(
        membership,
        {
            "players": [
                {"player_id": 1, "rendered_as_real": True, "on_target_court_coverage": 0.95},
                {"player_id": 2, "rendered_as_real": True, "on_target_court_coverage": 0.94},
                {"player_id": 3, "rendered_as_real": True, "on_target_court_coverage": 0.93},
                {"player_id": 4, "rendered_as_real": True, "on_target_court_coverage": 0.92},
            ]
        },
    )
    return clip, membership


def _check(report: dict, name: str) -> dict:
    return {check["name"]: check for check in report["checks"]}[name]


def test_all_pass_fixture_is_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_clean"
    assert {check["status"] for check in report["checks"]} == {"PASS"}


def test_side_label_contradiction_blocks_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    tracks = json.loads((clip / "tracks.json").read_text(encoding="utf-8"))
    tracks["players"][0]["side"] = "far"
    _write_json(clip / "tracks.json", tracks)

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_with_defects"
    assert _check(report, "side_consistency")["status"] == "FAIL"


def test_same_quadrant_pair_blocks_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    tracks = json.loads((clip / "tracks.json").read_text(encoding="utf-8"))
    tracks["players"][3]["frames"] = _frames(-1.0, 2.0)
    _write_json(clip / "tracks.json", tracks)

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_with_defects"
    assert _check(report, "quadrant_separation")["status"] == "FAIL"


def test_missing_membership_is_unproven_and_never_clean(tmp_path: Path) -> None:
    clip, _membership = _base_clip(tmp_path)

    report = build_strict_placement_rollup(clip)

    assert report["status"] == "viewable_preview_with_defects"
    membership = _check(report, "membership")
    assert membership["status"] == "NOT_COMPUTABLE"
    assert membership["detail"] == "unproven"


def test_body_gate_failed_blocks_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    _write_json(clip / "body_full_clip_gate.json", {"passed": False})

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_with_defects"
    assert _check(report, "body_full_clip_gate")["status"] == "FAIL"


def test_no_manifest_is_not_viewable(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    (clip / "replay_viewer_manifest.json").unlink()

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "not_viewable"


def test_mesh_refs_without_parse_marker_block_clean_but_skeleton_only_can_be_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    _write_json(
        clip / "replay_viewer_manifest.json",
        {"body_mesh_url": "body_mesh.json", "virtual_world_url": "confidence_gated_world.json"},
    )

    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_with_defects"
    assert _check(report, "mesh_honesty")["status"] == "NOT_COMPUTABLE"

    _write_json(clip / "replay_viewer_manifest.json", {"skeleton_only": True, "virtual_world_url": "confidence_gated_world.json"})
    report = build_strict_placement_rollup(clip, membership_path=membership)

    assert report["status"] == "viewable_preview_clean"
    assert _check(report, "mesh_honesty")["status"] == "PASS"


def test_gate_table_fail_blocks_clean(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    gate_table = tmp_path / "gate_table.json"
    _write_json(gate_table, {"rows": [{"gate": "visual_side", "status": "FAIL"}]})

    report = build_strict_placement_rollup(clip, membership_path=membership, gate_table_path=gate_table)

    assert report["status"] == "viewable_preview_with_defects"
    assert _check(report, "gate_table")["status"] == "FAIL"


def test_cli_writes_json_and_md(tmp_path: Path) -> None:
    clip, membership = _base_clip(tmp_path)
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            str(clip),
            "--membership",
            str(membership),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "viewable_preview_clean"
    assert (out_dir / "strict_rollup.json").is_file()
    md = (out_dir / "strict_rollup.md").read_text(encoding="utf-8")
    assert "Preview statuses only. Nothing here is VERIFIED in the repo's promotion sense." in md
