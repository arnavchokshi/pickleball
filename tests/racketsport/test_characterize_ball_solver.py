from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_solver_characterization import (
    build_characterization_manifest,
    build_characterization_report,
    characterize_clip_payloads,
    discover_clip_inputs,
    manifest_sha256,
    render_report_markdown,
    write_characterization_outputs,
)

CLI_PATH = "scripts/racketsport/characterize_ball_solver.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

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


def _pixel_offset(xy: list[float], dx: float, dy: float) -> list[float]:
    return [xy[0] + dx, xy[1] + dy]


def _anchor(anchor_id: str, kind: str, status: str, source: str, frame: int) -> dict:
    return {
        "anchor_id": anchor_id,
        "kind": kind,
        "status": status,
        "source": source,
        "frame": frame,
        "t": frame / 30.0,
        "sigma_m": 0.18,
        "world_xyz": [0.0, 0.0, 0.0371],
    }


def _arc_solved_fixture() -> dict:
    """Three-segment synthetic solve: fit, rejected fallback, trusted fallback.

    Frames 0-3 -> segment 0 (fit, near half y<0), frames 4-6 -> segment 1
    (fallback with zero inliers, far half y>0), frames 7-8 -> segment 2
    (fallback whose own statistics pass fail-closed), frame 9 hidden.
    Per-frame residuals against the fixture calibration are exact pixel
    offsets so percentile assertions are closed-form.
    """

    residual_offsets = {
        0: (3.0, 0.0),
        1: (0.0, 4.0),
        2: (5.0, 0.0),
        3: (0.0, 12.0),
        4: (0.0, 0.0),
        5: (30.0, 0.0),
        6: (0.0, 200.0),
        7: (1.0, 0.0),
        8: (0.0, 2.0),
    }
    worlds = {
        0: [0.0, -4.0, 0.5],
        1: [0.1, -3.5, 0.8],
        2: [0.2, -3.0, 1.0],
        3: [0.3, -2.5, 0.9],
        4: [0.4, 2.0, 1.2],
        5: [0.5, 2.5, 1.4],
        6: [0.6, 3.0, 1.1],
        7: [0.7, -1.5, 0.6],
        8: [0.8, -1.0, 0.4],
    }
    segment_for_frame = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2}
    segment_status = {0: "fit", 1: "fit_bvp_fallback", 2: "fit_bvp_fallback"}
    band_for_frame = {0: "anchored_measured", 1: "arc_weak", 2: "arc_weak"}
    frames = []
    for index in range(10):
        frame: dict = {"t": index / 30.0, "conf": 0.9, "visible": True}
        if index in worlds:
            seg = segment_for_frame[index]
            projected = _project(worlds[index])
            dx, dy = residual_offsets[index]
            frame["xy"] = _pixel_offset(projected, dx, dy)
            frame["world_xyz"] = worlds[index]
            frame["band"] = band_for_frame[seg]
            frame["arc_solver"] = {
                "segment_id": seg,
                "segment_status": segment_status[seg],
                "inlier_sighting": True,
                "outlier_sighting_pruned": False,
            }
        else:
            frame["xy"] = [0.0, 0.0]
            frame["visible"] = False
            frame["world_xyz"] = None
            frame["band"] = "hidden"
        frames.append(frame)
    segments = [
        {
            "segment_id": 0,
            "status": "fit",
            "frame_start": 0,
            "frame_end": 3,
            "t0": 0.0,
            "t1": 0.1,
            "inlier_count": 4,
            "outlier_count": 0,
            "reprojection_rmse_px": 2.0,
            "max_reprojection_error_px": 12.0,
            "initial_speed_mps": 12.0,
            "net_clearance_m": None,
            "net_clearance_ok": None,
            "start_anchor": "rally_endpoint_0",
            "end_anchor": "auto_bounce_3",
            "anchors_used": [
                _anchor("rally_endpoint_0", "rally_endpoint", "rally_endpoint_weak", "ball_ray_plane_weak_endpoint_prior", 0),
                _anchor("auto_bounce_3", "bounce", "auto_bounce_candidate", "track_geometry_candidate", 3),
            ],
            "physical_sanity": {"violation": False, "violations": []},
        },
        {
            "segment_id": 1,
            "status": "fit_bvp_fallback",
            "frame_start": 4,
            "frame_end": 6,
            "t0": 4 / 30.0,
            "t1": 6 / 30.0,
            "inlier_count": 0,
            "outlier_count": 5,
            "reprojection_rmse_px": 90.0,
            "max_reprojection_error_px": 200.0,
            "initial_speed_mps": 28.0,
            "net_clearance_m": 0.4,
            "net_clearance_ok": True,
            "start_anchor": "auto_bounce_4",
            "end_anchor": "contact_6",
            "anchors_used": [
                _anchor("auto_bounce_4", "bounce", "auto_bounce_candidate", "track_geometry_candidate", 4),
                _anchor("contact_6", "contact", "contact_prior", "skeleton3d_wrist_reach_prior", 6),
            ],
            "physical_sanity": {"violation": True, "violations": ["outside_court_volume"]},
        },
        {
            "segment_id": 2,
            "status": "fit_bvp_fallback",
            "frame_start": 7,
            "frame_end": 8,
            "t0": 7 / 30.0,
            "t1": 8 / 30.0,
            "inlier_count": 4,
            "outlier_count": 1,
            "reprojection_rmse_px": 3.0,
            "max_reprojection_error_px": 20.0,
            "initial_speed_mps": 9.0,
            "net_clearance_m": None,
            "net_clearance_ok": None,
            "start_anchor": "contact_7",
            "end_anchor": "rally_endpoint_9",
            "anchors_used": [
                _anchor("contact_7", "contact", "contact_prior", "skeleton3d_wrist_reach_prior", 7),
                _anchor("rally_endpoint_9", "rally_endpoint", "rally_endpoint_weak", "ball_ray_plane_weak_endpoint_prior", 9),
            ],
            "physical_sanity": {"violation": False, "violations": []},
        },
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "synthetic_clip",
        "status": "ran",
        "kill_reasons": [],
        "net_plane_provenance": {"consumed_net_plane": True, "reason": "consumed"},
        "frames": frames,
        "segments": segments,
        "anchors": [anchor for segment in segments for anchor in segment["anchors_used"]],
        "summary": {"input_frame_count": 10},
    }


def _flight_sanity_fixture() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_flight_sanity",
        "summary": {"segment_count": 3, "failed_segment_count": 1, "demoted_frame_count": 0},
        "segments": [
            {"segment_id": 0, "frame_start": 0, "frame_end": 3, "verdict": "pass", "reasons": []},
            {"segment_id": 1, "frame_start": 4, "frame_end": 6, "verdict": "fail", "reasons": ["outside_court_volume"]},
            {"segment_id": 2, "frame_start": 7, "frame_end": 8, "verdict": "pass", "reasons": []},
        ],
    }


def _physics_filled_fixture() -> dict:
    frames = []
    for index in range(10):
        frame: dict = {"t": index / 30.0, "xy": [1.0, 1.0], "visible": True}
        if index in (4, 5, 6, 9):
            frame["source"] = "physics_interpolated"
            frame["approx"] = True
            frame["world_xyz"] = [0.0, 0.0, 1.0]
        else:
            frame["world_xyz"] = [0.0, 0.0, 0.5]
        frames.append(frame)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_physics_filled",
        "frames": frames,
    }


def _characterize(**overrides) -> dict:
    kwargs = {
        "clip": "synthetic_clip",
        "arc_solved": _arc_solved_fixture(),
        "flight_sanity": _flight_sanity_fixture(),
        "calibration": _calibration(),
        "calibration_sha_verified": True,
    }
    kwargs.update(overrides)
    return characterize_clip_payloads(**kwargs)


def _segment_by_id(result: dict, segment_id: int) -> dict:
    return next(s for s in result["segments"] if s["segment_id"] == segment_id)


# ---------------------------------------------------------------------------
# Per-clip characterization
# ---------------------------------------------------------------------------


def test_segment_verdicts_follow_fail_closed_policy() -> None:
    result = _characterize()
    seg0 = _segment_by_id(result, 0)
    seg1 = _segment_by_id(result, 1)
    seg2 = _segment_by_id(result, 2)

    assert seg0["status"] == "fit"
    assert seg0["verdict"] == "accepted"
    assert seg0["fail_closed_reasons"] == []

    assert seg1["status"] == "fit_bvp_fallback"
    assert seg1["verdict"] == "rejected_fail_closed"
    assert "insufficient_inliers" in seg1["fail_closed_reasons"]
    assert "outliers_exceed_inliers" in seg1["fail_closed_reasons"]
    assert "max_reprojection_error_above_bound" in seg1["fail_closed_reasons"]
    assert "spatial_sanity_violation" in seg1["fail_closed_reasons"]

    assert seg2["status"] == "fit_bvp_fallback"
    assert seg2["verdict"] == "accepted"
    assert seg2["fail_closed_reasons"] == []

    assert result["segment_verdict_counts"] == {"accepted": 2, "rejected_fail_closed": 1}
    assert result["segment_status_counts"] == {"fit": 1, "fit_bvp_fallback": 2}
    counts = result["fail_closed_reason_counts"]
    assert counts["insufficient_inliers"] == 1
    assert counts["spatial_sanity_violation"] == 1


def test_anchor_inventory_per_segment() -> None:
    result = _characterize()
    seg0 = _segment_by_id(result, 0)
    seg1 = _segment_by_id(result, 1)
    seg2 = _segment_by_id(result, 2)

    assert seg0["anchors"]["classes"] == ["bounce_auto", "rally_endpoint_weak"]
    assert seg0["anchors"]["metric_anchor_classes"] == ["bounce_auto"]
    assert seg0["anchors"]["metric_anchor_anatomy"] == "single_sided_metric_anchor"

    assert seg1["anchors"]["classes"] == ["bounce_auto", "contact_wrist_seed"]
    assert seg1["anchors"]["metric_anchor_anatomy"] == "double_metric_anchor"
    assert seg1["anchors"]["net_constraint_evaluated"] is True

    assert seg2["anchors"]["metric_anchor_classes"] == ["contact_wrist_seed"]
    assert seg2["anchors"]["net_constraint_evaluated"] is False

    inventory = result["anchor_inventory"]["by_metric_anchor_anatomy"]
    assert inventory["single_sided_metric_anchor"]["segments"] == 2
    assert inventory["double_metric_anchor"]["segments"] == 1
    assert inventory["double_metric_anchor"]["accepted_segments"] == 0


def test_rally_frame_coverage_and_zero_return() -> None:
    result = _characterize()
    coverage = result["coverage"]
    assert coverage["rally_frame_count"] == 10
    assert coverage["rally_frame_denominator"] == "all_input_frames"
    assert coverage["accepted_3d_frame_count"] == 6
    assert coverage["accepted_3d_coverage_fraction"] == 0.6
    assert coverage["hidden_frame_count"] == 1
    assert coverage["fail_closed_suppressed_frame_count"] == 3

    zero = result["zero_return"]
    assert zero["frame_zero_return_rate"] == 0.4
    assert zero["segments_with_zero_accepted_frames"] == 1
    assert zero["segment_zero_return_rate"] == round(1.0 / 3.0, 6)


def test_reprojection_percentiles_recomputed_with_verified_calibration() -> None:
    result = _characterize()
    seg0 = _segment_by_id(result, 0)["reprojection"]["raw_track_visible_px"]
    assert seg0["status"] == "recomputed"
    assert seg0["count"] == 4
    # Residuals 3, 4, 5, 12 -> p50 = 4.5 (linear interpolation), max = 12.
    assert seg0["p50"] == 4.5
    assert seg0["max"] == 12.0

    seg1 = _segment_by_id(result, 1)["reprojection"]["raw_track_visible_px"]
    assert seg1["count"] == 3
    assert seg1["p50"] == 30.0
    assert seg1["max"] == 200.0

    # Fit statistics from the solver artifact are echoed unchanged.
    assert _segment_by_id(result, 0)["reprojection"]["fit_rmse_px"] == 2.0
    assert _segment_by_id(result, 1)["reprojection"]["fit_max_px"] == 200.0


def test_unverified_calibration_blocks_residual_recompute() -> None:
    result = _characterize(calibration_sha_verified=False)
    seg0 = _segment_by_id(result, 0)["reprojection"]["raw_track_visible_px"]
    assert seg0["status"] == "skipped"
    assert seg0["reason"] == "calibration_not_sha_verified"
    # Camera side may still be derived, but is labelled unverified.
    assert result["court_split"]["camera_side"] == "negative_y"
    assert result["court_split"]["camera_side_source"] == "unverified_calibration"

    result_none = _characterize(calibration=None, calibration_sha_verified=None)
    seg0_none = _segment_by_id(result_none, 0)["reprojection"]["raw_track_visible_px"]
    assert seg0_none["status"] == "skipped"
    assert seg0_none["reason"] == "missing_calibration"
    assert result_none["court_split"]["camera_side"] is None


def test_far_near_court_split() -> None:
    result = _characterize()
    split = result["court_split"]
    assert split["camera_side"] == "negative_y"
    assert split["camera_side_source"] == "sha_verified_calibration"
    halves = split["halves"]
    # Near half (camera side, y<0): segments 0 and 2 -> all accepted frames.
    assert halves["y_negative"]["frames_with_world_xyz"] == 6
    assert halves["y_negative"]["accepted_frame_count"] == 6
    # Far half (y>0): segment 1 -> all suppressed.
    assert halves["y_positive"]["frames_with_world_xyz"] == 3
    assert halves["y_positive"]["accepted_frame_count"] == 0
    assert split["near_half"] == "y_negative"
    assert split["far_half"] == "y_positive"


def test_flight_sanity_joined_by_span_not_segment_id() -> None:
    sanity = _flight_sanity_fixture()
    # Shift segment ids to prove the join uses (frame_start, frame_end) spans.
    for entry in sanity["segments"]:
        entry["segment_id"] = entry["segment_id"] + 5
    result = _characterize(flight_sanity=sanity)
    assert _segment_by_id(result, 1)["flight_sanity"]["verdict"] == "fail"
    assert _segment_by_id(result, 1)["flight_sanity"]["reasons"] == ["outside_court_volume"]
    assert _segment_by_id(result, 0)["flight_sanity"]["verdict"] == "pass"


def test_physics_fill_reported_separately_never_blended() -> None:
    without_fill = _characterize()
    with_fill = _characterize(physics_filled=_physics_filled_fixture())

    fill = with_fill["physics_fill"]
    assert fill["artifact_present"] is True
    assert fill["frame_count"] == 10
    assert fill["physics_interpolated_frame_count"] == 4
    assert fill["policy"] == "render_only_not_blended_into_accepted_stats"

    # Accepted statistics must be identical with and without the fill artifact.
    assert with_fill["coverage"] == without_fill["coverage"]
    assert with_fill["zero_return"] == without_fill["zero_return"]

    assert without_fill["physics_fill"]["artifact_present"] is False


# ---------------------------------------------------------------------------
# Manifest + pooled report + determinism
# ---------------------------------------------------------------------------


def _write_clip_dir(tmp_path: Path, name: str = "synthetic_clip") -> Path:
    clip_dir = tmp_path / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "ball_track_arc_solved.json").write_text(
        json.dumps(_arc_solved_fixture(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (clip_dir / "ball_flight_sanity.json").write_text(
        json.dumps(_flight_sanity_fixture(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (clip_dir / "court_calibration.json").write_text(
        json.dumps(_calibration(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return clip_dir


def _build_outputs(tmp_path: Path) -> tuple[dict, dict]:
    clip_dir = _write_clip_dir(tmp_path)
    missing_dir = tmp_path / "missing_clip"
    missing_dir.mkdir(exist_ok=True)
    inputs = [
        discover_clip_inputs("synthetic_clip", clip_dir),
        discover_clip_inputs("missing_clip", missing_dir),
    ]
    manifest = build_characterization_manifest(inputs, root=tmp_path, label="test_lane")
    clip_results = []
    for clip_inputs in inputs:
        if clip_inputs.arc_solved is None:
            clip_results.append(
                {
                    "clip": clip_inputs.clip,
                    "skipped": "missing_artifacts",
                    "missing": ["ball_track_arc_solved.json"],
                }
            )
            continue
        clip_results.append(
            characterize_clip_payloads(
                clip=clip_inputs.clip,
                arc_solved=json.loads(clip_inputs.arc_solved.read_text(encoding="utf-8")),
                flight_sanity=json.loads(clip_inputs.flight_sanity.read_text(encoding="utf-8")),
                calibration=json.loads(clip_inputs.calibration.read_text(encoding="utf-8")),
                calibration_sha_verified=True,
            )
        )
    report = build_characterization_report(clip_results, manifest=manifest)
    return manifest, report


def test_manifest_pins_paths_and_hashes(tmp_path: Path) -> None:
    manifest, _ = _build_outputs(tmp_path)
    clip_entry = manifest["clips"]["synthetic_clip"]
    arc = clip_entry["artifacts"]["ball_track_arc_solved"]
    assert arc["path"] == "synthetic_clip/ball_track_arc_solved.json"
    assert len(arc["sha256"]) == 64
    assert manifest["clips"]["missing_clip"]["missing"] == [
        "ball_track_arc_solved.json"
    ]
    echo = manifest["solver_config_echo"]
    assert echo["fail_closed"]["min_inlier_count"] == 3
    assert echo["fail_closed"]["max_reprojection_error_px"] == 40.0
    assert "solver_a_free" in echo["default_chain_configs"]


def test_skipped_clip_listed_not_fabricated(tmp_path: Path) -> None:
    _, report = _build_outputs(tmp_path)
    skipped = next(c for c in report["clips"] if c["clip"] == "missing_clip")
    assert skipped["skipped"] == "missing_artifacts"
    assert skipped["missing"] == ["ball_track_arc_solved.json"]
    pooled = report["pooled"]
    assert pooled["clip_count"] == 1
    assert pooled["skipped_clip_count"] == 1
    # Pooled statistics come only from the non-skipped clip.
    assert pooled["coverage"]["rally_frame_count"] == 10


def test_report_bytes_are_deterministic(tmp_path: Path) -> None:
    manifest_a, report_a = _build_outputs(tmp_path)
    manifest_b, report_b = _build_outputs(tmp_path)
    bytes_a = json.dumps(report_a, indent=2, sort_keys=True).encode("utf-8")
    bytes_b = json.dumps(report_b, indent=2, sort_keys=True).encode("utf-8")
    assert bytes_a == bytes_b
    assert manifest_sha256(manifest_a) == manifest_sha256(manifest_b)
    assert report_a["manifest_sha256"] == manifest_sha256(manifest_a)

    rendered = json.dumps(report_a, sort_keys=True)
    assert str(tmp_path) not in rendered  # no absolute paths in report body
    assert "generated_at" not in rendered  # no timestamps in report body

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    write_characterization_outputs(out_dir=out_a, manifest=manifest_a, report=report_a)
    write_characterization_outputs(out_dir=out_b, manifest=manifest_b, report=report_b)
    assert (out_a / "report.json").read_bytes() == (out_b / "report.json").read_bytes()
    assert (out_a / "REPORT.md").read_bytes() == (out_b / "REPORT.md").read_bytes()


def test_markdown_report_mentions_headline_numbers(tmp_path: Path) -> None:
    _, report = _build_outputs(tmp_path)
    markdown = render_report_markdown(report)
    assert "synthetic_clip" in markdown
    assert "missing_clip" in markdown
    assert "60.0" in markdown  # accepted 3D coverage percent
    assert "VERIFIED=0" in markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_deterministic_reports(tmp_path: Path) -> None:
    clip_dir = _write_clip_dir(tmp_path)
    out_dir = tmp_path / "characterization"
    cmd = [
        sys.executable,
        str(REPO_ROOT / CLI_PATH),
        "--clip",
        f"synthetic_clip={clip_dir}",
        "--out-dir",
        str(out_dir),
        "--root",
        str(tmp_path),
        "--label",
        "test_lane",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report_path = out_dir / "report.json"
    manifest_path = out_dir / "manifest.json"
    assert report_path.is_file() and manifest_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pooled"]["coverage"]["accepted_3d_coverage_fraction"] == 0.6
    first_bytes = report_path.read_bytes()

    second = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert report_path.read_bytes() == first_bytes


def test_cli_solve_requires_ball_track_and_calibration(tmp_path: Path) -> None:
    clip_dir = _write_clip_dir(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CLI_PATH),
            "--clip",
            f"synthetic_clip={clip_dir}",
            "--out-dir",
            str(tmp_path / "out"),
            "--solve",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--solve requires" in result.stderr


def test_cli_errors_on_malformed_clip_argument(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / CLI_PATH),
            "--clip",
            "no_equals_sign",
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "NAME=DIR" in result.stderr
