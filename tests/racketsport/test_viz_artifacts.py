from __future__ import annotations

import json

import pytest

from threed.racketsport import viz_courtmap, viz_ghost, viz_overlay


def test_courtmap_payload_validates_heatmap_and_summarizes_priority_markers() -> None:
    payload = viz_courtmap.build_courtmap_payload(
        sport="pickleball",
        player_paths=[
            {
                "player_id": 1,
                "points": [
                    {"t": 0.0, "xy_m": [-1.0, -2.0], "confidence": 0.9},
                    {"t": 0.5, "xy_m": [0.0, -1.0], "confidence": 0.8},
                ],
            }
        ],
        heatmap_bins=[
            {"xy_m": [0.5, -0.5], "value": 3, "label": "transition"},
            {"xy_m": [-0.25, 0.25], "value": 1, "label": "nvz"},
        ],
        priority_metrics=[
            {
                "metric": "nvz_margin_ft",
                "player_id": 1,
                "t": 0.25,
                "xy_m": [0.1, -2.1],
                "value": -0.5,
                "units": "ft",
                "severity": "high",
                "confidence": 0.86,
            },
            {
                "metric": "balance_score",
                "player_id": 1,
                "t": 0.5,
                "xy_m": [0.0, -1.0],
                "value": 0.72,
                "units": "score",
                "severity": "medium",
                "confidence": 0.91,
            },
        ],
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "court_map_payload"
    assert payload["render_status"] == "cpu_payload_only"
    assert payload["world_frame"] == "court_Z0"
    assert payload["court"]["sport"] == "pickleball"
    assert payload["court"]["width_ft"] == 20.0
    assert payload["court"]["length_ft"] == 44.0
    assert payload["layers"]["heatmap"]["bins"] == [
        {"xy_m": [-0.25, 0.25], "value": 1.0, "label": "nvz"},
        {"xy_m": [0.5, -0.5], "value": 3.0, "label": "transition"},
    ]
    assert payload["priority_metric_marker_summary"] == {
        "count": 2,
        "primary": {
            "metric": "nvz_margin_ft",
            "player_id": 1,
            "t": 0.25,
            "xy_m": [0.1, -2.1],
            "value": -0.5,
            "units": "ft",
            "severity": "high",
            "confidence": 0.86,
        },
        "by_severity": {"high": 1, "medium": 1},
    }


def test_courtmap_rejects_heatmap_bins_outside_court() -> None:
    with pytest.raises(ValueError, match="heatmap_bins/0/xy_m is outside pickleball court bounds"):
        viz_courtmap.build_courtmap_payload(
            sport="pickleball",
            player_paths=[],
            heatmap_bins=[{"xy_m": [99.0, 0.0], "value": 1}],
            priority_metrics=[],
        )


def test_ghost_payload_aligns_self_vs_self_samples_at_contact() -> None:
    payload = viz_ghost.build_ghost_payload(
        baseline={
            "label": "best_rep",
            "player_id": 1,
            "contact_t": 1.0,
            "samples": [
                {"t": 0.75, "xy_m": [-0.4, -1.0], "metric_value": 0.9},
                {"t": 1.0, "xy_m": [-0.2, -0.8], "metric_value": 1.0},
            ],
        },
        comparison={
            "label": "current_rep",
            "player_id": 1,
            "contact_t": 2.0,
            "samples": [
                {"t": 1.75, "xy_m": [-0.6, -1.2], "metric_value": 0.5},
                {"t": 2.0, "xy_m": [-0.3, -1.0], "metric_value": 0.6},
            ],
        },
        metric_name="ready_position_score",
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload == {
        "schema_version": 1,
        "artifact_type": "self_vs_self_ghost_payload",
        "render_status": "cpu_payload_only",
        "alignment": {"mode": "contact_frame", "baseline_contact_t": 1.0, "comparison_contact_t": 2.0},
        "player_id": 1,
        "metric_name": "ready_position_score",
        "traces": {
            "baseline": {
                "label": "best_rep",
                "samples": [
                    {"rel_t": -0.25, "xy_m": [-0.4, -1.0], "metric_value": 0.9},
                    {"rel_t": 0.0, "xy_m": [-0.2, -0.8], "metric_value": 1.0},
                ],
            },
            "comparison": {
                "label": "current_rep",
                "samples": [
                    {"rel_t": -0.25, "xy_m": [-0.6, -1.2], "metric_value": 0.5},
                    {"rel_t": 0.0, "xy_m": [-0.3, -1.0], "metric_value": 0.6},
                ],
            },
        },
        "summary": {"contact_delta_xy_m": [-0.1, -0.2], "contact_metric_delta": -0.4},
    }


def test_ghost_rejects_cross_player_comparisons() -> None:
    with pytest.raises(ValueError, match="baseline and comparison must use the same player_id"):
        viz_ghost.build_ghost_payload(
            baseline={"label": "best", "player_id": 1, "contact_t": 1.0, "samples": []},
            comparison={"label": "current", "player_id": 2, "contact_t": 1.0, "samples": []},
            metric_name="balance",
        )


def test_overlay_metadata_packages_telestration_without_rendering() -> None:
    payload = viz_overlay.build_overlay_metadata(
        video_ref={"clip_id": "clip_001", "fps": 60.0, "duration_s": 3.0},
        elements=[
            {
                "type": "contact_marker",
                "t": 1.2,
                "frame": 72,
                "xy_px": [640, 360],
                "label": "late contact",
                "confidence": 0.88,
            },
            {
                "type": "knee_angle_arc",
                "t": 1.2,
                "frame": 72,
                "points_px": [[620, 420], [650, 470], [700, 500]],
                "value": 142.0,
                "units": "deg",
                "confidence": 0.82,
            },
        ],
        source_artifacts=["habit_report.json", "racket_pose.json"],
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload == {
        "schema_version": 1,
        "artifact_type": "video_overlay_metadata",
        "render_status": "cpu_payload_only",
        "video_ref": {"clip_id": "clip_001", "fps": 60.0, "duration_s": 3.0},
        "source_artifacts": ["habit_report.json", "racket_pose.json"],
        "element_count": 2,
        "elements": [
            {
                "type": "contact_marker",
                "t": 1.2,
                "frame": 72,
                "xy_px": [640.0, 360.0],
                "label": "late contact",
                "confidence": 0.88,
            },
            {
                "type": "knee_angle_arc",
                "t": 1.2,
                "frame": 72,
                "points_px": [[620.0, 420.0], [650.0, 470.0], [700.0, 500.0]],
                "value": 142.0,
                "units": "deg",
                "confidence": 0.82,
            },
        ],
    }


def test_overlay_rejects_unknown_element_type() -> None:
    with pytest.raises(ValueError, match="elements/0/type is unsupported"):
        viz_overlay.build_overlay_metadata(
            video_ref={"clip_id": "clip_001", "fps": 60.0, "duration_s": 3.0},
            elements=[{"type": "raw_shader", "t": 0.0, "frame": 0}],
        )
