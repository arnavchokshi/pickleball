from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.rally_metrics import build_rally_metrics


def test_rally_metrics_split_rallies_zone_fractions_and_contact_trust(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        fps=10.0,
        players=[
            {
                "id": "p1",
                "frames": [
                    _frame(0.0, [0.0, -1.8], gate_status="physics_corrected"),
                    _frame(0.1, [0.0, -1.2], gate_status="physics_corrected"),
                    _frame(0.2, [0.0, -5.4], gate_status="physics_corrected"),
                    _frame(0.3, [0.0, 3.0], gate_status="physics_corrected"),
                    _frame(0.4, [0.0, 7.0], gate_status="physics_corrected"),
                    _frame(0.6, [0.0, 1.8], gate_status="physics_corrected"),
                    _frame(0.7, [0.0, 2.4], gate_status="physics_corrected"),
                ],
            }
        ],
        rally_spans=[
            {"id": "r0", "t0": 0.0, "t1": 0.5, "sources": ["synthetic"]},
            {"id": "r1", "t0": 0.5, "t1": 0.9, "sources": ["synthetic"]},
        ],
        contact_events=[
            {
                "type": "contact",
                "player_id": "p1",
                "frame": 2,
                "t": 0.2,
                "trust_band_note": "wrist-cue-only, unverified",
                "window": {"t0": 0.18, "t1": 0.23},
            }
        ],
    )

    result = build_rally_metrics(run_dir)

    assert result["artifact_type"] == "rally_metrics"
    assert result["rally_scope"] == "rally_spans"
    rally = result["rallies"][0]
    assert rally["id"] == "r0"
    player = rally["players"][0]
    metrics = player["metrics"]

    zone = metrics["zone_occupancy"]
    assert zone["value"] == pytest.approx(
        {
            "kitchen": 0.4,
            "transition": 0.2,
            "baseline": 0.2,
            "out_of_court": 0.2,
        }
    )
    assert zone["frames_used"] == 5
    assert zone["frames_total"] == 5
    assert zone["trust"] == "ok"

    assert metrics["kitchen_proximity_s"]["value"] == pytest.approx(0.1)
    assert metrics["contact_count"]["value"] == 1
    assert metrics["contact_count"]["trust"] == "unverified_cue"
    assert metrics["contact_positions_world"]["value"] == [
        {
            "frame": 2,
            "t": 0.2,
            "position_world_xy": [0.0, -5.4],
            "trust": "unverified_cue",
            "trust_note": "wrist-cue-only, unverified",
        }
    ]

    facts = result["coaching_card_facts"]["facts"]
    assert facts[0]["rally_id"] == "r0"
    assert facts[0]["player_id"] == "p1"
    assert facts[0]["metric"] == "contact_count"
    assert facts[0]["trust"] == "unverified_cue"


def test_gap_aware_speed_skips_missing_frame_teleports_on_clip_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        fps=10.0,
        ball_frame_count=15,
        players=[
            {
                "id": "p1",
                "frames": [
                    _frame(0.0, [0.0, 0.0]),
                    _frame(0.1, [0.1, 0.0]),
                    _frame(0.2, [0.2, 0.0]),
                    _frame(1.3, [99.0, 99.0]),
                    _frame(1.4, [99.1, 99.0]),
                ],
            }
        ],
        rally_spans=None,
    )

    result = build_rally_metrics(run_dir)

    assert result["rally_scope"] == "clip_fallback"
    metrics = result["rallies"][0]["players"][0]["metrics"]
    assert metrics["distance_covered_m"]["value"] == pytest.approx(0.3)
    assert metrics["avg_speed_mps"]["value"] == pytest.approx(1.0)
    assert metrics["p95_speed_mps"]["value"] == pytest.approx(1.0)
    assert metrics["distance_covered_m"]["frames_used"] == 5
    assert metrics["distance_covered_m"]["frames_total"] == 15
    assert metrics["distance_covered_m"]["coverage_fraction"] == pytest.approx(5 / 15)
    assert metrics["distance_covered_m"]["trust"] == "estimated"


def test_interpolated_or_predicted_frame_caps_position_metric_trust(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        fps=10.0,
        players=[
            {
                "id": "p1",
                "frames": [
                    _frame(0.0, [0.0, 0.0], gate_status="physics_corrected"),
                    _frame(0.1, [0.1, 0.0], provenance={"source": "physics_predicted"}),
                    _frame(0.2, [0.2, 0.0], gate_status="physics_corrected"),
                    _frame(0.3, [0.3, 0.0], gate_status="physics_corrected"),
                    _frame(0.4, [0.4, 0.0], gate_status="physics_corrected"),
                ],
            }
        ],
        rally_spans=[{"id": "r0", "t0": 0.0, "t1": 0.5}],
    )

    metrics = build_rally_metrics(run_dir)["rallies"][0]["players"][0]["metrics"]

    assert metrics["distance_covered_m"]["coverage_fraction"] == pytest.approx(1.0)
    assert metrics["distance_covered_m"]["trust"] == "estimated"
    assert metrics["zone_occupancy"]["trust"] == "estimated"


def test_missing_track_world_xy_fails_closed_with_exact_field_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        fps=10.0,
        players=[{"id": "bad", "frames": [{"t": 0.0, "xy": [10.0, 20.0]}]}],
        rally_spans=[{"id": "r0", "t0": 0.0, "t1": 0.1}],
    )

    with pytest.raises(ValueError, match=r"players\[bad\]\.frames\[0\]\.track_world_xy"):
        build_rally_metrics(run_dir)


def _write_run(
    run_dir: Path,
    *,
    fps: float,
    players: list[dict],
    rally_spans: list[dict] | None = None,
    contact_events: list[dict] | None = None,
    ball_frame_count: int | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    world = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": fps,
        "summary": {"ball_frame_count": ball_frame_count if ball_frame_count is not None else int(fps)},
        "players": players,
        "ball": {"frames": []},
    }
    (run_dir / "virtual_world.json").write_text(json.dumps(world, indent=2), encoding="utf-8")
    (run_dir / "court_zones.json").write_text(json.dumps({"schema_version": 1, "zones": _zones()}), encoding="utf-8")
    if rally_spans is not None:
        (run_dir / "rally_spans.json").write_text(
            json.dumps({"schema_version": 1, "spans": rally_spans}, indent=2),
            encoding="utf-8",
        )
    if contact_events is not None:
        (run_dir / "contact_windows.json").write_text(
            json.dumps({"schema_version": 1, "events": contact_events}, indent=2),
            encoding="utf-8",
        )


def _frame(
    t: float,
    xy: list[float],
    *,
    gate_status: str | None = None,
    provenance: dict | None = None,
) -> dict:
    frame = {"t": t, "track_world_xy": xy}
    if gate_status is not None:
        frame["trust_band"] = {"gate_status": gate_status}
    if provenance is not None:
        frame["confidence_provenance"] = provenance
    return frame


def _zones() -> dict[str, list[list[float]]]:
    return {
        "court": [[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0]],
        "near_nvz": [[-3.0, -2.0], [3.0, -2.0], [3.0, 0.0], [-3.0, 0.0]],
        "far_nvz": [[-3.0, 0.0], [3.0, 0.0], [3.0, 2.0], [-3.0, 2.0]],
        "near_left_service": [[-3.0, -6.0], [0.0, -6.0], [0.0, -2.0], [-3.0, -2.0]],
        "near_right_service": [[0.0, -6.0], [3.0, -6.0], [3.0, -2.0], [0.0, -2.0]],
        "far_left_service": [[-3.0, 2.0], [0.0, 2.0], [0.0, 6.0], [-3.0, 6.0]],
        "far_right_service": [[0.0, 2.0], [3.0, 2.0], [3.0, 6.0], [0.0, 6.0]],
    }
