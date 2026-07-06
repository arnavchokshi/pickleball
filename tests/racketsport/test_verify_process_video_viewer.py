import json
from pathlib import Path

import pytest

from scripts.racketsport.verify_process_video_viewer import (
    TRUSTED_BALL_ARC_SOLVER_STATUSES,
    assert_ball_honesty,
    assert_non_empty_entity_counts,
    read_ball_arc_solver_status,
    resolve_ball_arc_solved_path,
    screenshot_name_for_seconds,
    viewer_url_for_manifest,
    write_headless_verify_report,
)


def test_viewer_url_requires_real_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest path is required"):
        viewer_url_for_manifest(None)

    with pytest.raises(FileNotFoundError, match="manifest does not exist"):
        viewer_url_for_manifest(tmp_path / "missing_replay_viewer_manifest.json")


def test_viewer_url_adds_manifest_query_param(tmp_path: Path) -> None:
    manifest = tmp_path / "replay_viewer_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    url = viewer_url_for_manifest(manifest)

    assert url.startswith("http://127.0.0.1:5173/?manifest=/@fs")
    assert str(manifest.resolve()) in url


def test_viewer_url_accepts_custom_port_for_occupied_local_dev_port(tmp_path: Path) -> None:
    manifest = tmp_path / "replay_viewer_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    url = viewer_url_for_manifest(manifest, port=5174)

    assert url.startswith("http://127.0.0.1:5174/?manifest=/@fs")


def test_viewer_url_can_open_court_map_mode_for_second_screenshot(tmp_path: Path) -> None:
    manifest = tmp_path / "replay_viewer_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    url = viewer_url_for_manifest(manifest, view="courtmap")

    assert "manifest=/@fs" in url
    assert "&view=courtmap" in url


def test_assert_non_empty_entity_counts_rejects_silent_empty_viewer() -> None:
    # Matches the *current* status-grid render (App.tsx status-grid Metric
    # labels: Players, Contacts, Ball, Warnings, 3D FPS) -- the previous
    # label set (Mesh Frames, Solid Mesh Frames, Floor Frames, Ball Contacts,
    # Replay Points) no longer overlaps with the real page and validated
    # almost nothing.
    counts = {
        "Players": 0,
        "Contacts": 0,
        "Ball": "0/0 measured",
        "Warnings": "0 notices",
        "3D FPS": 0,
    }

    with pytest.raises(AssertionError, match="empty viewer"):
        assert_non_empty_entity_counts(counts)


def test_assert_non_empty_entity_counts_allows_explicit_empty_opt_out() -> None:
    assert_non_empty_entity_counts({"Players": 0}, allow_empty=True)


def test_assert_non_empty_entity_counts_passes_on_a_real_status_grid_snapshot() -> None:
    # Real values observed live against the wolverine_mixed_0200_mid_steep_corner
    # smoke run (runs/lanes/e2e_synergy_audit_20260705/browser_verify/probe_result.json).
    counts = {
        "Players": "4",
        "Contacts": "0",
        "Ball": "0/300 measured · 11 predicted · 289 hidden",
        "Warnings": "2 notices: 2D-only ball frames outside solved arc coverage, missing paddle pose",
        "3D FPS": "54.6",
    }

    assert_non_empty_entity_counts(counts)


def test_screenshot_name_for_seconds_is_repeatable_and_filesystem_safe() -> None:
    assert screenshot_name_for_seconds(3) == "screenshot_t3s.png"
    assert screenshot_name_for_seconds(6.25) == "screenshot_t6p25s.png"

    with pytest.raises(ValueError, match="non-negative"):
        screenshot_name_for_seconds(-1)


def test_write_headless_verify_report_records_counts_and_page_errors(tmp_path: Path) -> None:
    report_path = write_headless_verify_report(
        tmp_path,
        {
            "ok": False,
            "url": "http://127.0.0.1:5173/?manifest=/@fs/tmp/replay_viewer_manifest.json",
            "loaded_counts": {"Players": 4, "Ball Contacts": 63},
            "page_errors": ["contact_windows.events[0].sources.audio must be a number"],
        },
    )

    assert report_path == tmp_path / "headless_verify.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["loaded_counts"]["Players"] == 4
    assert payload["page_errors"] == ["contact_windows.events[0].sources.audio must be a number"]


# ---------------------------------------------------------------------------
# Ball-honesty acceptance gate (fail-closed for a self-killed ball-arc-solver)
# ---------------------------------------------------------------------------


def test_trusted_ball_arc_solver_statuses_only_allows_ran() -> None:
    assert TRUSTED_BALL_ARC_SOLVER_STATUSES == frozenset({"ran"})


def test_resolve_ball_arc_solved_path_reads_the_at_fs_manifest_url(tmp_path: Path) -> None:
    arc_path = tmp_path / "ball_track_arc_solved.json"
    arc_path.write_text(json.dumps({"status": "ran"}), encoding="utf-8")
    manifest_path = tmp_path / "replay_viewer_manifest.json"
    manifest_path.write_text(json.dumps({"ball_arc_solved_url": f"/@fs{arc_path}"}), encoding="utf-8")

    resolved = resolve_ball_arc_solved_path(manifest_path)

    assert resolved == arc_path


def test_resolve_ball_arc_solved_path_is_none_when_absent_or_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "replay_viewer_manifest.json"
    manifest_path.write_text(json.dumps({}), encoding="utf-8")
    assert resolve_ball_arc_solved_path(manifest_path) is None

    manifest_path.write_text(json.dumps({"ball_arc_solved_url": "/@fs/does/not/exist.json"}), encoding="utf-8")
    assert resolve_ball_arc_solved_path(manifest_path) is None


def test_read_ball_arc_solver_status_reads_a_real_self_killed_artifact(tmp_path: Path) -> None:
    arc_path = tmp_path / "ball_track_arc_solved.json"
    arc_path.write_text(
        json.dumps(
            {
                "status": "experimental_off",
                "kill_reasons": ["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "replay_viewer_manifest.json"
    manifest_path.write_text(json.dumps({"ball_arc_solved_url": f"/@fs{arc_path}"}), encoding="utf-8")

    status, kill_reasons = read_ball_arc_solver_status(manifest_path)

    assert status == "experimental_off"
    assert kill_reasons == ["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"]


def test_read_ball_arc_solver_status_is_none_without_a_resolvable_artifact(tmp_path: Path) -> None:
    manifest_path = tmp_path / "replay_viewer_manifest.json"
    manifest_path.write_text(json.dumps({}), encoding="utf-8")

    status, kill_reasons = read_ball_arc_solver_status(manifest_path)

    assert status is None
    assert kill_reasons == []


def test_assert_ball_honesty_fails_when_a_self_killed_solve_renders_as_measured() -> None:
    # This is the RED case that reproduces the measured 2026-07-05 defect
    # (runs/lanes/e2e_synergy_audit_20260705/browser_verify/probe_result.json):
    # status=experimental_off yet the HUD reported data-ball-state="measured".
    hud_snapshot = {"text": "ball: measured", "data_ball_state": "measured", "data_low_confidence": "false"}

    with pytest.raises(AssertionError, match="ball honesty violation"):
        assert_ball_honesty(
            hud_snapshot,
            solver_status="experimental_off",
            kill_reasons=["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"],
        )


def test_assert_ball_honesty_passes_once_the_hud_reports_the_fail_closed_state() -> None:
    # This is the GREEN case: post-fix, the HUD reports the explicit
    # fail-closed solver_off state instead of measured.
    hud_snapshot = {
        "text": "ball: solver off — physical_sanity_violation_fraction 0.400000 exceeds 0.200000",
        "data_ball_state": "solver_off",
        "data_low_confidence": "false",
    }

    assert_ball_honesty(
        hud_snapshot,
        solver_status="experimental_off",
        kill_reasons=["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"],
    )


def test_assert_ball_honesty_passes_healthy_runs_even_if_hud_reports_measured() -> None:
    hud_snapshot = {"text": "ball: measured", "data_ball_state": "measured", "data_low_confidence": "false"}

    assert_ball_honesty(hud_snapshot, solver_status="ran", kill_reasons=[])


def test_assert_ball_honesty_is_a_noop_without_a_resolvable_ball_arc_artifact() -> None:
    assert_ball_honesty(None, solver_status=None, kill_reasons=[])
    assert_ball_honesty({"data_ball_state": "measured"}, solver_status=None, kill_reasons=[])
