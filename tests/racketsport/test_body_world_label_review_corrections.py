from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_world_label_finalize import finalize_body_world_labels
from threed.racketsport.body_world_label_review_corrections import (
    apply_body_world_label_review_corrections,
    build_body_world_label_review_corrections_template,
    merge_body_world_label_review_decisions_into_corrections,
    run_body_world_label_review_decision_pipeline,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _template(*, label_source: str | None = "manual_3d_annotation") -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_joints_labels",
        "status": "draft_review_template",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "clip": "clip_001",
        "source_packet": "body_world_label_packet.json",
        "source_video": "source.mp4",
        "joint_names": ["pelvis", "neck"],
        "selected_sample_ids": ["frame_000001_player_7"],
        "samples": [
            {
                "sample_id": "frame_000001_player_7",
                "frame_index": 1,
                "t": 1.0 / 30.0,
                "player_id": 7,
                "accepted": False,
                "review_status": "needs_review",
                "joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
                "predicted_joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
                "joint_conf": [0.9, 0.8],
                "notes": "",
                **({"label_source": label_source} if label_source is not None else {}),
            }
        ],
    }


def _overlay_index() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_review_overlay",
        "status": "ready_for_review_with_overlay_warnings",
        "overlays": [
            {
                "sample_id": "frame_000001_player_7",
                "warnings": ["body_joint_overlay_alignment_warning"],
                "overlay_path": "overlays/frame_000001_player_7_overlay.jpg",
            }
        ],
    }


def _accepted_corrections() -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "body_clip_001_review",
        "created_at": "2026-07-01T12:00:00Z",
        "description": "Accepted BODY world-label review corrections.",
        "corrections": [
            _correction("accept_sample", "/samples/frame_000001_player_7/accepted", True),
            _correction("review_sample", "/samples/frame_000001_player_7/review_status", "reviewed"),
            _correction("mark_reviewed", "/status", "human_reviewed"),
            _correction("mark_ground_truth", "/not_ground_truth", False),
            _correction("mark_trusted", "/trusted_for_world_mpjpe", True),
            _correction(
                "resolve_overlay_warning",
                "/overlays/frame_000001_player_7/warning_review_status",
                "accepted",
                artifact="body_world_label_review_bundle/overlays/body_world_label_review_overlay_index.json",
            ),
        ],
    }


def _correction(correction_id: str, path: str, value: object, *, artifact: str | None = None) -> dict:
    return {
        "id": correction_id,
        "status": "accepted",
        "target": {
            "artifact": artifact or "body_world_label_review_bundle/body_world_joints.template.json",
            "clip_id": "clip_001",
            "phase": "phase3",
            "metric": "body_world_labels",
            "path": path,
        },
        "operation": "replace",
        "value": value,
        "reason": "Human review accepted the sample for world-MPJPE labels.",
        "annotator": "reviewer",
        "created_at": "2026-07-01T12:01:00Z",
    }


def test_apply_body_world_label_review_corrections_enables_safe_finalization(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "body_review_corrections.json"
    out_bundle = tmp_path / "reviewed" / "body_world_label_review_bundle"
    out_template = out_bundle / "body_world_joints.template.json"
    out_overlay = out_bundle / "overlays" / "body_world_label_review_overlay_index.json"
    final_labels = tmp_path / "labels" / "clip_001" / "body_world_joints.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    _write_json(corrections, _accepted_corrections())

    summary = apply_body_world_label_review_corrections(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=corrections,
        out_template_path=out_template,
        out_overlay_index_path=out_overlay,
    )
    final_report = finalize_body_world_labels(template_path=out_template, out_path=final_labels)

    assert summary["status"] == "applied"
    assert summary["applied_count"] == 6
    reviewed_template = json.loads(out_template.read_text(encoding="utf-8"))
    reviewed_overlay = json.loads(out_overlay.read_text(encoding="utf-8"))
    assert reviewed_template["status"] == "human_reviewed"
    assert reviewed_template["not_ground_truth"] is False
    assert reviewed_template["trusted_for_world_mpjpe"] is True
    assert reviewed_template["samples"][0]["accepted"] is True
    assert reviewed_overlay["overlays"][0]["warning_review_status"] == "accepted"
    assert final_report["status"] == "finalized"


def test_build_body_world_label_review_corrections_template_prefills_pending_review_steps(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    out = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())

    manifest = build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=out,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )

    assert manifest["manifest_id"] == "body_clip_001_review"
    assert manifest["correction_count"] == 9
    assert manifest["pending_correction_count"] == 9
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    corrections_by_path = {correction["target"]["path"]: correction for correction in payload["corrections"]}
    assert corrections_by_path["/samples/frame_000001_player_7/joints_world"]["value"] == [
        [0.0, 0.0, 0.1],
        [0.2, 0.0, 1.4],
    ]
    assert corrections_by_path["/samples/frame_000001_player_7/accepted"]["value"] is True
    assert corrections_by_path["/samples/frame_000001_player_7/review_status"]["value"] == "reviewed"
    assert corrections_by_path["/samples/frame_000001_player_7/notes"]["value"] == ""
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_status"]["value"] == "accepted"
    assert corrections_by_path["/status"]["status"] == "pending"
    assert all(correction["status"] == "pending" for correction in payload["corrections"])


def test_build_body_world_label_review_corrections_template_cli_writes_pending_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    out = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_world_label_review_corrections_template.py",
            "--template",
            str(template),
            "--overlay-index",
            str(overlay),
            "--out",
            str(out),
            "--manifest-id",
            "body_clip_001_review",
            "--created-at",
            "2026-07-01T12:00:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["correction_count"] == 9
    assert out.is_file()


def test_apply_body_world_label_review_corrections_cli_reports_empty_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "body_review_corrections.json"
    out_template = tmp_path / "out" / "body_world_joints.template.json"
    out_overlay = tmp_path / "out" / "overlays" / "body_world_label_review_overlay_index.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    _write_json(
        corrections,
        {
            "schema_version": 1,
            "manifest_id": "empty_body_review",
            "created_at": "2026-07-01T12:00:00Z",
            "corrections": [],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/apply_body_world_label_review_corrections.py",
            "--template",
            str(template),
            "--overlay-index",
            str(overlay),
            "--corrections",
            str(corrections),
            "--out-template",
            str(out_template),
            "--out-overlay-index",
            str(out_overlay),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["status"] == "no_accepted_corrections"
    assert summary["applied_count"] == 0
    assert not out_template.exists()


def test_merge_body_world_label_review_decisions_accepts_only_overlay_warning_corrections(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    merged = tmp_path / "corrections" / "body_clip_001_review_corrections.merged.json"
    out_template = tmp_path / "reviewed" / "body_world_joints.template.json"
    out_overlay = tmp_path / "reviewed" / "overlays" / "body_world_label_review_overlay_index.json"
    final_labels = tmp_path / "labels" / "clip_001" / "body_world_joints.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {
                        "decision": "overlay_ok",
                        "notes": "Projection stays on the intended player.",
                    }
                }
            },
        },
    )

    summary = merge_body_world_label_review_decisions_into_corrections(
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_path=merged,
    )

    assert summary["status"] == "written"
    assert summary["accepted_overlay_warning_sample_count"] == 1
    assert summary["accepted_overlay_warning_sample_ids"] == ["frame_000001_player_7"]
    payload = json.loads(merged.read_text(encoding="utf-8"))
    corrections_by_path = {correction["target"]["path"]: correction for correction in payload["corrections"]}
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_status"]["status"] == "accepted"
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_note"]["status"] == "accepted"
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_note"]["value"] == (
        "Projection stays on the intended player."
    )
    assert corrections_by_path["/samples/frame_000001_player_7/accepted"]["status"] == "pending"
    assert corrections_by_path["/samples/frame_000001_player_7/joints_world"]["status"] == "pending"
    assert corrections_by_path["/status"]["status"] == "pending"

    apply_summary = apply_body_world_label_review_corrections(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=merged,
        out_template_path=out_template,
        out_overlay_index_path=out_overlay,
    )
    final_report = finalize_body_world_labels(template_path=out_template, out_path=final_labels)
    assert apply_summary["status"] == "applied"
    assert "selected_samples_have_overlay_warnings" not in final_report["blockers"]
    assert "selected_samples_not_all_accepted" in final_report["blockers"]
    assert "template_not_reviewed" in final_report["blockers"]


def test_merge_body_world_label_review_decisions_copies_label_notes_to_template_note_correction(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    merged = tmp_path / "corrections" / "body_clip_001_review_corrections.merged.json"
    _write_json(template, _template())
    _write_json(overlay, {**_overlay_index(), "overlays": [{**_overlay_index()["overlays"][0], "warnings": []}]})
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {
                        "decision": "accept_candidate_label",
                        "notes": "systematic overlay offset, but joints align after checking source frame",
                    }
                }
            },
        },
    )

    summary = merge_body_world_label_review_decisions_into_corrections(
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_path=merged,
    )

    assert summary["status"] == "written"
    payload = json.loads(merged.read_text(encoding="utf-8"))
    corrections_by_path = {correction["target"]["path"]: correction for correction in payload["corrections"]}
    note_correction = corrections_by_path["/samples/frame_000001_player_7/notes"]
    assert note_correction["status"] == "accepted"
    assert note_correction["value"] == "systematic overlay offset, but joints align after checking source frame"


def test_merge_body_world_label_review_decisions_rejects_bad_overlay_without_accepting_corrections(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    merged = tmp_path / "corrections" / "body_clip_001_review_corrections.merged.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {
                        "decision": "wrong_player",
                        "notes": "Projected skeleton is on another player.",
                    }
                }
            },
        },
    )

    summary = merge_body_world_label_review_decisions_into_corrections(
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_path=merged,
    )

    assert summary["status"] == "written"
    assert summary["rejected_overlay_warning_sample_ids"] == ["frame_000001_player_7"]
    payload = json.loads(merged.read_text(encoding="utf-8"))
    corrections_by_path = {correction["target"]["path"]: correction for correction in payload["corrections"]}
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_status"]["status"] == "rejected"
    assert corrections_by_path["/overlays/frame_000001_player_7/warning_review_note"]["status"] == "rejected"


def test_merge_body_world_label_review_decisions_cli_writes_safe_overlay_only_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    merged = tmp_path / "corrections" / "body_clip_001_review_corrections.merged.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {"decision": "overlay_ok", "notes": ""}
                }
            },
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/merge_body_world_label_review_decisions.py",
            "--corrections",
            str(corrections),
            "--review-input",
            str(review_input),
            "--run-id",
            "body_clip_001_runtime",
            "--out",
            str(merged),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["accepted_overlay_warning_sample_count"] == 1
    assert merged.is_file()


def test_run_body_world_label_review_decision_pipeline_applies_overlay_review_but_keeps_labels_blocked(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    out_dir = tmp_path / "pipeline_out"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {"decision": "overlay_ok", "notes": "Overlay is acceptable."}
                }
            },
        },
    )

    summary = run_body_world_label_review_decision_pipeline(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_dir=out_dir,
    )

    assert summary["status"] == "blocked"
    assert summary["merge"]["accepted_overlay_warning_sample_count"] == 1
    assert summary["apply"]["status"] == "applied"
    assert summary["finalization"]["status"] == "blocked"
    assert "selected_samples_have_overlay_warnings" not in summary["finalization"]["blockers"]
    assert "selected_samples_not_all_accepted" in summary["finalization"]["blockers"]
    assert "template_not_reviewed" in summary["finalization"]["blockers"]
    assert not Path(summary["final_labels_path"]).exists()
    assert Path(summary["merged_corrections_path"]).is_file()
    reviewed_overlay = json.loads(Path(summary["reviewed_overlay_index_path"]).read_text(encoding="utf-8"))
    assert reviewed_overlay["overlays"][0]["warning_review_status"] == "accepted"


def test_run_body_world_label_review_decision_pipeline_blocks_candidate_world_label_without_independent_source(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    out_dir = tmp_path / "pipeline_out"
    _write_json(template, _template(label_source=None))
    _write_json(
        overlay,
        {
            **_overlay_index(),
            "status": "ready_for_review",
            "alignment_warning_count": 0,
            "overlays": [{**_overlay_index()["overlays"][0], "warnings": []}],
        },
    )
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {
                        "decision": "accept_candidate_label",
                        "notes": "Overlay and source frame support this candidate world label.",
                    }
                }
            },
        },
    )

    summary = run_body_world_label_review_decision_pipeline(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_dir=out_dir,
    )

    assert summary["status"] == "blocked"
    assert summary["merge"]["accepted_label_sample_ids"] == ["frame_000001_player_7"]
    assert summary["apply"]["applied_count"] == 7
    assert summary["finalization"]["status"] == "blocked"
    assert "accepted_candidate_labels_not_independent_ground_truth" in summary["finalization"]["blockers"]
    assert not Path(summary["final_labels_path"]).exists()


def test_run_body_world_label_review_decision_pipeline_writes_canonical_label_outputs(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    out_dir = tmp_path / "pipeline_out"
    final_labels = tmp_path / "run" / "labels" / "body_world_joints.json"
    finalization_report = tmp_path / "run" / "body_world_label_review_bundle" / "body_world_label_finalization.json"
    _write_json(template, _template())
    _write_json(overlay, {**_overlay_index(), "overlays": [{**_overlay_index()["overlays"][0], "warnings": []}]})
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {"decision": "accept_candidate_label", "notes": ""}
                }
            },
        },
    )

    summary = run_body_world_label_review_decision_pipeline(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_dir=out_dir,
        final_labels_path=final_labels,
        finalization_report_path=finalization_report,
    )

    assert summary["status"] == "finalized"
    assert summary["final_labels_path"] == str(final_labels)
    assert summary["finalization_report_path"] == str(finalization_report)
    assert summary["reviewed_template_path"] == str(template)
    assert summary["finalization_template_path"] == str(template)
    assert json.loads(template.read_text(encoding="utf-8"))["status"] == "human_reviewed"
    assert final_labels.is_file()
    assert json.loads(finalization_report.read_text(encoding="utf-8"))["status"] == "finalized"


def test_run_body_world_label_review_decision_pipeline_no_saved_decisions_reports_existing_blockers(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    out_dir = tmp_path / "pipeline_out"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {},
        },
    )

    summary = run_body_world_label_review_decision_pipeline(
        template_path=template,
        overlay_index_path=overlay,
        corrections_path=corrections,
        review_input_path=review_input,
        run_id="body_clip_001_runtime",
        out_dir=out_dir,
    )

    assert summary["status"] == "blocked"
    assert summary["merge"]["missing_decision_sample_count"] == 1
    assert summary["apply"]["status"] == "no_accepted_corrections"
    assert summary["finalization"]["status"] == "blocked"
    assert "selected_samples_have_overlay_warnings" in summary["finalization"]["blockers"]
    assert "selected_samples_not_all_accepted" in summary["finalization"]["blockers"]
    assert Path(summary["merged_corrections_path"]).is_file()
    assert not Path(summary["reviewed_template_path"]).exists()
    assert not Path(summary["final_labels_path"]).exists()


def test_run_body_world_label_review_decision_pipeline_cli_allows_blocked_summary(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "body_world_label_review_bundle"
    template = bundle / "body_world_joints.template.json"
    overlay = bundle / "overlays" / "body_world_label_review_overlay_index.json"
    corrections = tmp_path / "corrections" / "body_clip_001_review_corrections.json"
    review_input = tmp_path / "runs" / "review_inputs" / "pickleball_cv_review_latest.json"
    out_dir = tmp_path / "pipeline_out"
    summary_out = tmp_path / "pipeline_out" / "summary.json"
    _write_json(template, _template())
    _write_json(overlay, _overlay_index())
    build_body_world_label_review_corrections_template(
        template_path=template,
        overlay_index_path=overlay,
        out_path=corrections,
        manifest_id="body_clip_001_review",
        created_at="2026-07-01T12:00:00Z",
    )
    _write_json(
        review_input,
        {
            "schema_version": 2,
            "review_type": "pickleball_cv_blocker_review",
            "body_world_label_review": {
                "body_clip_001_runtime": {
                    "frame_000001_player_7": {"decision": "overlay_ok", "notes": ""}
                }
            },
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_body_world_label_review_decision_pipeline.py",
            "--template",
            str(template),
            "--overlay-index",
            str(overlay),
            "--corrections",
            str(corrections),
            "--review-input",
            str(review_input),
            "--run-id",
            "body_clip_001_runtime",
            "--out-dir",
            str(out_dir),
            "--summary-out",
            str(summary_out),
            "--allow-blocked",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["status"] == "blocked"
    assert summary_out.is_file()
