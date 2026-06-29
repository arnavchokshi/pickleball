from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.racketsport import review_input_server


def test_review_input_manifest_includes_action_blockers_by_clip(tmp_path: Path) -> None:
    actions_path = (
        tmp_path
        / "runs"
        / "review_packets"
        / "prototype_gate_h100_v2"
        / "prototype_gate_h100_v2_review_actions.json"
    )
    actions_path.parent.mkdir(parents=True)
    actions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_review_action_manifest",
                "packet_id": "prototype_gate_h100_v2_review",
                "actions": [
                    {
                        "clip": "burlington_gold_0300_low_steep_corner",
                        "category": "court_evidence",
                        "priority": "medium",
                        "title": "Court-line evidence is not auto-calibration-ready",
                        "artifact_path": "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/court_line_evidence.json",
                        "watch_paths": [],
                        "editable_paths": [],
                        "details": ["Missing required net cues: top_net"],
                        "blockers": ["court_evidence_not_ready", "missing_net:top_net"],
                        "next_commands": ["python scripts/racketsport/build_court_line_evidence.py ..."],
                        "suggested_action": "Review the missing line/net cues.",
                    },
                    {
                        "clip": "burlington_gold_0300_low_steep_corner",
                        "category": "paddle_pose",
                        "priority": "high",
                        "title": "Paddle pose is preview-only",
                        "artifact_path": "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/racket_pose_readiness.json",
                        "watch_paths": [],
                        "editable_paths": [
                            "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/racket_candidates.json"
                        ],
                        "details": ["Candidate frames: 42"],
                        "blockers": ["box_derived_candidate_corners"],
                        "next_commands": [],
                        "suggested_action": "Collect true paddle keypoints.",
                    },
                    {
                        "clip": "wolverine_mixed_0200_mid_steep_corner",
                        "category": "pipeline_readiness",
                        "priority": "medium",
                        "title": "E2E artifact readiness is incomplete",
                        "artifact_path": "runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/pipeline_readiness_e2e.json",
                        "watch_paths": [],
                        "editable_paths": [],
                        "details": ["Missing artifacts: smpl_motion.json"],
                        "blockers": ["missing:smpl_motion.json"],
                        "next_commands": [],
                        "suggested_action": "Run the next real stage.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert hasattr(review_input_server, "_review_actions_by_clip")
    by_clip = review_input_server._review_actions_by_clip(tmp_path)

    assert by_clip["burlington_gold_0300_low_steep_corner"]["action_count"] == 2
    assert by_clip["burlington_gold_0300_low_steep_corner"]["high_count"] == 1
    assert by_clip["burlington_gold_0300_low_steep_corner"]["categories"] == {
        "court_evidence": 1,
        "paddle_pose": 1,
    }
    assert by_clip["burlington_gold_0300_low_steep_corner"]["blockers"] == [
        "box_derived_candidate_corners",
        "court_evidence_not_ready",
        "missing_net:top_net",
    ]

    manifest = review_input_server._manifest(tmp_path)
    burlington = next(clip for clip in manifest["clips"] if clip["id"] == "burlington_gold_0300_low_steep_corner")
    assert burlington["review_actions"]["packet_id"] == "prototype_gate_h100_v2_review"
    assert burlington["review_actions"]["action_count"] == 2
    assert burlington["review_actions"]["actions"][0]["priority"] == "high"
    assert burlington["review_actions"]["actions"][0]["title"] == "Paddle pose is preview-only"


def test_review_input_manifest_includes_global_actions(tmp_path: Path) -> None:
    actions_path = (
        tmp_path
        / "runs"
        / "review_packets"
        / "prototype_gate_h100_v2"
        / "prototype_gate_h100_v2_review_actions.json"
    )
    actions_path.parent.mkdir(parents=True)
    actions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_review_action_manifest",
                "packet_id": "prototype_gate_h100_v2_review",
                "actions": [
                    {
                        "clip": "__global__",
                        "category": "paddle_runtime",
                        "priority": "medium",
                        "title": "Paddle model/runtime readiness is blocked",
                        "artifact_path": "runs/eval0/prototype_gate_h100_v2/racket_model_runtime_readiness.json",
                        "watch_paths": [],
                        "editable_paths": [],
                        "details": ["Runtime-ready components: 0"],
                        "blockers": ["sam3_concept_tracker:missing_manifest_entry"],
                        "next_commands": [
                            "python scripts/racketsport/build_racket_model_runtime_readiness.py --manifest models/MANIFEST.json --out runs/eval0/prototype_gate_h100_v2/racket_model_runtime_readiness.json"
                        ],
                        "suggested_action": "Add manifest entries and assets before paddle 6DoF GPU smoke.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    by_clip = review_input_server._review_actions_by_clip(tmp_path)
    assert by_clip["__global__"]["action_count"] == 1
    assert by_clip["__global__"]["categories"] == {"paddle_runtime": 1}

    manifest = review_input_server._manifest(tmp_path)
    assert manifest["global_review_actions"]["packet_id"] == "prototype_gate_h100_v2_review"
    assert manifest["global_review_actions"]["action_count"] == 1
    assert manifest["global_review_actions"]["actions"][0]["title"] == "Paddle model/runtime readiness is blocked"
    assert 'id="globalActions"' in review_input_server.HTML


def test_review_input_manifest_exposes_per_clip_intake_requirements(tmp_path: Path) -> None:
    clip = "burlington_gold_0300_low_steep_corner"
    clip_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2" / clip
    labels_dir = clip_root / "labels"
    labels_dir.mkdir(parents=True)
    (clip_root / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_court_line_evidence",
                "aggregate": {
                    "auto_calibration_ready": False,
                    "missing_required_line_ids": ["near_nvz", "near_centerline"],
                    "missing_required_net_ids": ["top_net"],
                },
            }
        ),
        encoding="utf-8",
    )
    (clip_root / "contact_windows.json").write_text(
        json.dumps({"schema_version": 1, "artifact_type": "racketsport_contact_windows", "events": []}),
        encoding="utf-8",
    )
    (labels_dir / "players.json").write_text(
        json.dumps({"status": "human_reviewed", "not_ground_truth": False, "annotation": {"items": [{}]}}),
        encoding="utf-8",
    )
    (labels_dir / "ball.json").write_text(
        json.dumps({"status": "draft_prototype_unverified", "not_ground_truth": True}),
        encoding="utf-8",
    )

    manifest = review_input_server._manifest(tmp_path)
    clip_manifest = next(item for item in manifest["clips"] if item["id"] == clip)
    intake = clip_manifest["intake_requirements"]

    assert intake["court_evidence"]["auto_calibration_ready"] is False
    assert intake["court_evidence"]["missing_required_line_ids"] == ["near_nvz", "near_centerline"]
    assert intake["court_evidence"]["missing_required_net_ids"] == ["top_net"]
    assert intake["contacts"]["canonical_contact_count"] == 0
    assert intake["contacts"]["needs_human_reviewed_contacts"] is True
    assert intake["labels"]["players.json"]["state"] == "reviewed"
    assert intake["labels"]["ball.json"]["state"] == "draft_not_ground_truth"
    assert intake["labels"]["events.json"]["state"] == "missing"
    assert "coach_habits.json" in intake["labels"]


def test_review_input_html_exposes_copyable_next_commands() -> None:
    assert 'class="command-item"' in review_input_server.HTML
    assert 'data-copy-command' in review_input_server.HTML
    assert "navigator.clipboard.writeText" in review_input_server.HTML
    assert "document.execCommand" in review_input_server.HTML
    assert "selectCommandText" in review_input_server.HTML


def test_review_input_wizard_exposes_source_data_intake_step() -> None:
    assert 'id: "intake"' in review_input_server.WIZARD_HTML
    assert "Source data checklist" in review_input_server.WIZARD_HTML
    assert "renderIntake" in review_input_server.WIZARD_HTML
    assert "court_evidence" in review_input_server.WIZARD_HTML


def test_review_input_manifest_retires_burlington_for_court_click_review(tmp_path: Path) -> None:
    manifest = review_input_server._manifest(tmp_path)

    burlington = next(clip for clip in manifest["clips"] if clip["id"] == "burlington_gold_0300_low_steep_corner")
    wolverine = next(clip for clip in manifest["clips"] if clip["id"] == "wolverine_mixed_0200_mid_steep_corner")

    burlington_policy = burlington["intake_requirements"]["court_review_policy"]
    assert burlington_policy["court_review_enabled"] is False
    assert burlington_policy["retired_for_court"] is True
    assert "fisheye" in burlington_policy["reason"]
    assert "player" in burlington_policy["allowed_use"]
    assert wolverine["intake_requirements"]["court_review_policy"]["court_review_enabled"] is True


def test_review_input_wizard_exposes_video_first_click_target_workflow() -> None:
    html = review_input_server.WIZARD_HTML

    assert "Click target" in html
    assert "courtClickTargets" in html
    assert "captureVideoPoint" in html
    assert "data-point-status" in html
    assert "Not visible" in html
    assert "Missing/wrong" in html
    assert "Retired for court" in html
    assert "Source video, folder, Drive link" not in html
    assert "data-source-field" not in html


def test_review_input_manifest_exposes_paddle_true_corner_click_queue(tmp_path: Path) -> None:
    clip = "wolverine_mixed_0200_mid_steep_corner"
    clip_root = tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2" / clip
    crop_sheet = clip_root / "racket_candidates" / "paddle_true_corner_crop_sheet.png"
    overlay = clip_root / "racket_candidates" / "racket_candidate_overlay_h264.mp4"
    source_video = clip_root / "racket_candidates" / "source_0000_0030.mp4"
    crop_sheet.parent.mkdir(parents=True)
    crop_sheet.write_bytes(b"png")
    overlay.write_bytes(b"mp4")
    source_video.write_bytes(b"mp4")
    (clip_root / "paddle_true_corner_review.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_paddle_true_corner_review",
                "clip": clip,
                "status": "blocked_missing_true_corner_labels",
                "required_label_count": 2,
                "listed_required_label_count": 2,
                "true_corner_label_count": 0,
                "reference_gt_count": 0,
                "promotion_blockers": ["missing_reviewed_true_corner_labels"],
                "crop_sheet_summary": {
                    "status": "rendered",
                    "rendered_crop_count": 2,
                    "crop_sheet_path": str(crop_sheet.relative_to(tmp_path)),
                    "video_path": str(source_video.relative_to(tmp_path)),
                },
                "visuals": [
                    {"type": "candidate_overlay_video", "path": str(overlay.relative_to(tmp_path))},
                    {"type": "true_corner_label_crop_sheet", "path": str(crop_sheet.relative_to(tmp_path))},
                ],
                "required_labels": [
                    {
                        "review_id": "7_000120",
                        "player_id": 7,
                        "frame_index": 120,
                        "t": 4.0,
                        "crop_xyxy": [100, 200, 160, 290],
                        "required_output": {
                            "corners_px_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                            "evidence_type": "true_corners",
                            "reviewer": "required",
                        },
                    },
                    {
                        "review_id": "7_000180",
                        "player_id": 7,
                        "frame_index": 180,
                        "t": 6.0,
                        "crop_xyxy": [300, 400, 390, 460],
                        "required_output": {
                            "corners_px_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                            "evidence_type": "true_corners",
                            "reviewer": "required",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = review_input_server._manifest(tmp_path)
    clip_manifest = next(item for item in manifest["clips"] if item["id"] == clip)
    paddle = clip_manifest["paddle_corner_review"]

    assert paddle["status"] == "blocked_missing_true_corner_labels"
    assert paddle["queue_count"] == 2
    assert paddle["crop_sheet"]["exists"] is True
    assert paddle["candidate_overlay"]["exists"] is True
    assert paddle["source_video"]["exists"] is True
    assert paddle["corner_order"] == ["top_left", "top_right", "bottom_right", "bottom_left"]
    assert paddle["task_items"][0]["review_id"] == "7_000120"
    assert paddle["task_items"][0]["tile_index"] == 0
    assert paddle["task_items"][0]["sheet_rect"] == [0, 0, 180, 180]
    assert paddle["task_items"][0]["crop_xyxy"] == [100, 200, 160, 290]
    assert paddle["task_items"][1]["tile_index"] == 1
    assert paddle["task_items"][1]["sheet_rect"] == [180, 0, 360, 180]


def test_review_input_manifest_exposes_tracking_video_review_questions(tmp_path: Path) -> None:
    phase_root = tmp_path / "runs" / "phase2" / "person_coverage_actual_source30_h100_yolo26m_fulltb3_20260628"
    outdoor = phase_root / "outdoor_webcam_iynbd_1500_long_high_baseline"
    indoor = phase_root / "indoor_doubles_fwuks_0500_long_mid_baseline"
    outdoor_overlay = outdoor / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock_margin2_diagnostic_overlay.mp4"
    indoor_overlay = indoor / "yolo26m_fulltb3_actual_source30_b128_img1280_rolelock_margin4_diagnostic_overlay.mp4"
    outdoor_montage = phase_root / "montages" / "outdoor_webcam_iynbd_1500_long_high_baseline_yolo26m_margin0_vs_margin2_diagnostic_montage.jpg"
    indoor_montage = phase_root / "montages" / "indoor_doubles_fwuks_0500_long_mid_baseline_yolo26m_margin0_vs_margin4_diagnostic_montage.jpg"
    for path in (outdoor_overlay, indoor_overlay, outdoor_montage, indoor_montage):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")

    manifest = review_input_server._manifest(tmp_path)
    tracking = manifest["human_review_tasks"]["tracking_video_review"]

    assert [item["clip"] for item in tracking["items"]] == [
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
    ]
    assert tracking["items"][0]["overlay_video"]["exists"] is True
    assert tracking["items"][0]["montage"]["exists"] is True
    assert "spectators" in tracking["items"][0]["question"].lower()
    assert "promote" in tracking["items"][0]["needed_answer"].lower()
    assert tracking["choices"] == ["safe_to_promote", "unsafe_background_or_spectators", "unsure"]


def test_review_input_server_exposes_focused_paddle_review_page() -> None:
    assert "paddleCornerQueue" in review_input_server.PADDLE_HTML
    assert "trackingReviewVideos" in review_input_server.PADDLE_HTML
    assert "Not visible" in review_input_server.PADDLE_HTML
    assert "Ambiguous" in review_input_server.PADDLE_HTML
    assert "paddle_true_corner_labels" in review_input_server.PADDLE_HTML


def test_asset_streaming_ignores_client_disconnect(tmp_path: Path) -> None:
    asset = tmp_path / "clip.mp4"
    asset.write_bytes(b"abcdef")

    class DisconnectingWriter:
        def __init__(self) -> None:
            self.calls = 0

        def write(self, data: bytes) -> None:
            self.calls += 1
            raise BrokenPipeError("client closed connection")

    writer = DisconnectingWriter()

    assert hasattr(review_input_server, "_copy_file_to_writer")
    review_input_server._copy_file_to_writer(asset, writer, chunk_size=2)

    assert writer.calls == 1


def test_parse_byte_range_supports_video_seek_ranges() -> None:
    assert hasattr(review_input_server, "_parse_byte_range")

    assert review_input_server._parse_byte_range("bytes=10-19", 100) == (10, 19)
    assert review_input_server._parse_byte_range("bytes=95-", 100) == (95, 99)
    assert review_input_server._parse_byte_range("bytes=-5", 100) == (95, 99)
    assert review_input_server._parse_byte_range("items=0-5", 100) is None
    assert review_input_server._parse_byte_range("bytes=120-130", 100) is None


def test_review_input_write_uses_fixed_paths_and_server_metadata(tmp_path: Path) -> None:
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    clip = "wolverine_mixed_0200_mid_steep_corner"
    latest, timestamped = review_input_server._write_review_input(
        tmp_path,
        {
            "schema_version": 1,
            "review_type": "pickleball_cv_blocker_review",
            "repo_root": "/tmp/client-controlled",
            "server_saved_at_utc": "client-controlled",
            "clips": {
                clip: {
                    "reviewed_enough": True,
                    "court_overlay_ok": "yes",
                    "top_net": {
                        "left": {"x": 1, "y": 2, "time_s": 3, "video_width": 640, "video_height": 480},
                        "right": None,
                        "notes": "top net partly visible",
                    },
                    "players": {"P1": "near-left"},
                    "contacts": [{"player": "P1", "time_s": 1.25, "note": "clean paddle contact"}],
                }
            },
        },
        now=now,
    )

    assert latest == tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    assert timestamped == tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_20260102T030405Z.json"
    assert sorted(path.name for path in tmp_path.rglob("*.json")) == [
        "pickleball_cv_review_20260102T030405Z.json",
        "pickleball_cv_review_latest.json",
    ]

    saved = json.loads(latest.read_text(encoding="utf-8"))
    assert saved["repo_root"] == str(tmp_path)
    assert saved["server_saved_at_utc"] == "2026-01-02T03:04:05+00:00"
    assert saved["clips"][clip]["reviewed_enough"] is True
    assert saved["clips"][clip]["top_net"]["left"]["x"] == 1.0
    assert saved["clips"][clip]["players"] == {"P1": "near-left", "P2": "", "P3": "", "P4": ""}


def test_review_input_write_rejects_client_path_fields_without_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpected save fields"):
        review_input_server._write_review_input(
            tmp_path,
            {
                "schema_version": 1,
                "review_type": "pickleball_cv_blocker_review",
                "save_path": "../../evil.json",
                "filename": "/tmp/evil.json",
            },
        )

    assert not (tmp_path / "runs").exists()
    assert not (tmp_path.parent / "evil.json").exists()


def test_review_input_write_rejects_unknown_clip_ids_without_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown review clip id"):
        review_input_server._write_review_input(
            tmp_path,
            {
                "schema_version": 1,
                "review_type": "pickleball_cv_blocker_review",
                "clips": {"../../evil": {"general_notes": "do not use this as a path"}},
            },
        )

    assert not (tmp_path / "runs").exists()
