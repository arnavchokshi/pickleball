from __future__ import annotations

import json
import re
from pathlib import Path

from threed.racketsport.best_stack import load_best_stack_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_CONSTANT_PATTERN = re.compile(
    r"\bDEFAULT_[A-Z0-9_]*(?:MODEL|CKPT|CHECKPOINT|PROFILE|BUDGET|MODE)[A-Z0-9_]*\b"
)


def test_best_stack_manifest_integrity() -> None:
    manifest = load_best_stack_manifest()

    assert manifest.schema_version == 1
    assert manifest.revision == 15
    assert "A manifest entry is a DEFAULT selection, NEVER a VERIFIED claim" in manifest.invariants
    assert len(manifest.entries) >= 30

    required_entries = {
        "ball.wasb_checkpoint",
        "ball.wasb_repo",
        "tracking.reid_model",
        "tracking.global_association_profile",
        "tracking.eval_only_association_profiles",
        "tracking.player_selection_layer",
        "confidence.calibration_curves",
        "mesh.coverage_mode",
        "mesh.byte_budget_mib",
        "mesh.human_review_ghost_emission",
        "mesh.tier_eligibility_raise",
        "mesh.target_frame_budget",
        "body.skeleton_stride",
        "ball.detection_stride",
        "cadence.future_stage_pattern",
        "body.detector_fov",
        "camera_motion.policy",
        "server.allow_auto_court_corners_preview",
        "ball.seed_official_checkpoint",
        "ball.stage1_official_checkpoint",
        "court.court_unet_v2",
        "court.e4_fusion_default",
        "body.p22_lambda_foot_smoother",
        "body.postchain_raw_knob",
        "instrument.gate_check_body_decode",
        "body.fast_sam_3d_body_challenger_not_adopt",
        "ball.arc_solver_spin",
        "paddle.fused_estimator",
        "paddle.reflection_cone_factor",
        "input_quality.preflight",
        "stats.match_stats_v0",
    }
    assert required_entries <= set(manifest.entries)
    assert "mesh.byte_budget_policy" not in manifest.entries
    assert "ball.magnus_fit_spin_scalar" not in manifest.entries

    assert manifest.entry("mesh.byte_budget_mib").status == "WIRED_DEFAULT"
    assert any("w6_close_errand_20260708" in path for path in manifest.entry("mesh.byte_budget_mib").provenance["evidence_paths"])
    assert "21.3fps" in manifest.entry("mesh.byte_budget_mib").notes
    ghost = manifest.entry("mesh.human_review_ghost_emission")
    assert ghost.status == "WIRED_DEFAULT"
    assert ghost.value == {
        "eligible_tier": "human_review",
        "trust_badge": "preview",
        "counts_against_byte_budget": True,
        "viewer_contract": "body_mesh_index.players.frames[].trust_badge",
    }
    assert any("w7_ghostviewer_20260709/report.json" in path for path in ghost.provenance["evidence_paths"])
    assert manifest.entry("mesh.tier_eligibility_raise").status == "PENDING"
    assert manifest.entry("body.p22_lambda_foot_smoother").status == "DORMANT"
    assert "NOT-WIRING-READY" in manifest.entry("body.p22_lambda_foot_smoother").notes
    assert manifest.entry("body.p22_lambda_foot_smoother").proven_against["gate_1b_world_round_trip_mm"] == 262.35
    assert manifest.entry("body.postchain_raw_knob").status == "DORMANT"
    assert manifest.entry("instrument.gate_check_body_decode").status == "DORMANT"
    assert manifest.entry("body.fast_sam_3d_body_challenger_not_adopt").status == "DORMANT"
    assert manifest.entry("ball.arc_solver_spin").status == "DORMANT"
    assert "kill-fired" in manifest.entry("ball.arc_solver_spin").notes
    assert manifest.entry("input_quality.preflight").status == "WIRED_DEFAULT"
    assert manifest.entry("input_quality.preflight").value["mode"] == "advisory"
    assert manifest.entry("stats.match_stats_v0").status == "WIRED_DEFAULT"
    assert manifest.entry("stats.match_stats_v0").value["enabled"] is True
    assert manifest.entry("body.skeleton_stride").status == "WIRED_DEFAULT"
    assert manifest.entry("body.skeleton_stride").value == 2
    assert "owner ruling 2026-07-09" in manifest.entry("body.skeleton_stride").notes.lower()
    assert manifest.entry("body.experimental_body_array_native").status == "WIRED_DEFAULT"
    assert manifest.bool_value("body.experimental_body_array_native") is True
    assert manifest.entry("ball.detection_stride").status == "WIRED_DEFAULT"
    assert manifest.entry("ball.detection_stride").value == 1
    assert "full-rate" in manifest.entry("ball.detection_stride").notes
    assert manifest.entry("cadence.future_stage_pattern").status == "WIRED_DEFAULT"
    assert manifest.entry("cadence.future_stage_pattern").value["default_source"] == "best_stack.json"
    assert "loso_report.json" in manifest.entry("ball.wasb_checkpoint").notes
    ball_pending = manifest.entry("ball.seed_official_checkpoint")
    assert ball_pending.status == "PENDING"
    assert ball_pending.value["label"] == "A_seed_official_aug"
    assert ball_pending.value["md5"] == "cfda3c423e1f93c0db42f20e32bdae9e"
    assert ball_pending.gate["name"] == "pre_registered_heldout_eval_ledger_row_plus_owner_go"
    assert "recall >= 0.70" in ball_pending.gate["bar"]
    eval_profiles = manifest.entry("tracking.eval_only_association_profiles")
    assert eval_profiles.status == "DORMANT"
    assert eval_profiles.value["no_flag_profile"] == "tracking.global_association_profile"
    assert (
        eval_profiles.value["profiles"]["burlington_gold_0300_low_steep_corner"]["profile"]
        == "burlington_internal_val_trk10_iter5_minconf05_appw2_margin2"
    )
    assert (
        eval_profiles.value["profiles"]["outdoor_webcam_iynbd_1500_long_high_baseline"]["profile"]
        == "outdoor_preregistered_unshopped_base"
    )
    association_margin = manifest.entry("tracking.association_court_margin")
    assert association_margin.status == "WIRED_DEFAULT"
    assert association_margin.value["enabled"] is True
    assert association_margin.value["margin_m"] == 1.0
    assert association_margin.value["fallback_candidate_m"] == 0.5
    assert association_margin.gate["name"] == "trk_fresh_clip_full_gate"
    assert association_margin.proven_against["full_bar_cov4_ge_0p95"] is False
    assert association_margin.provenance["lane"] == "trk_flip_20260713"
    assert "NOT an accuracy-gate promotion" in association_margin.notes

    player_selection = manifest.entry("tracking.player_selection_layer")
    assert player_selection.status == "PENDING"
    assert player_selection.raw["trust_band"] == "preview"
    assert player_selection.value["enabled"] is False
    assert player_selection.value["do_not_promote"] is True
    assert player_selection.value["registered_constants"] == {
        "COURT_REGION_HARD_BOUND_M": 1.0,
        "EXACTLY_FOUR_HARD_CAP": 4,
    }
    assert player_selection.gate["name"] == (
        "player_selection_two_leg_reproduction_mechanics_then_untouched_judge_accuracy"
    )
    assert "Leg 1" in player_selection.gate["bar"]
    assert "Leg 2" in player_selection.gate["bar"]
    assert player_selection.provenance["commit"] == "worktree"
    assert player_selection.provenance["lane"] == "trkC_constraints_rebuild_20260722"
    assert "burned them as an accuracy judge" in player_selection.notes
    assert "VERIFIED=0" in player_selection.notes

    pending_without_gate = [
        key for key, entry in manifest.entries.items() if entry.status == "PENDING" and entry.gate is None
    ]
    assert pending_without_gate == []

    dormant_without_ruling = [
        key
        for key, entry in manifest.entries.items()
        if entry.status == "DORMANT" and "ruling" not in entry.notes.lower()
    ]
    assert dormant_without_ruling == []

    evidence_checked_entries = {
        "mesh.tier_eligibility_raise",
        "mesh.human_review_ghost_emission",
        "body.p22_lambda_foot_smoother",
        "body.postchain_raw_knob",
        "instrument.gate_check_body_decode",
        "body.fast_sam_3d_body_challenger_not_adopt",
        "ball.arc_solver_spin",
        "ball.wasb_checkpoint",
        "tracking.eval_only_association_profiles",
    }
    missing_evidence: list[str] = []
    for key in evidence_checked_entries:
        for raw_path in manifest.entry(key).provenance["evidence_paths"]:
            path = REPO_ROOT / str(raw_path).split("#", 1)[0]
            if not path.exists():
                missing_evidence.append(f"{key}:{raw_path}")
    assert missing_evidence == []


def test_best_stack_manifest_covers_selection_constants_and_model_manifest_entries() -> None:
    manifest = load_best_stack_manifest()
    covered_constants = set(manifest.raw.get("selection_constant_coverage", {}))
    constant_allowlist = set(manifest.raw.get("selection_constant_allowlist", {}))
    scanned_files = [
        REPO_ROOT / "scripts" / "racketsport" / "process_video.py",
        REPO_ROOT / "threed" / "racketsport" / "orchestrator.py",
        REPO_ROOT / "scripts" / "racketsport" / "remote_body_dispatch.py",
    ]

    missing_constants: list[str] = []
    for path in scanned_files:
        for match in SELECTION_CONSTANT_PATTERN.finditer(path.read_text(encoding="utf-8")):
            token = match.group(0)
            if token not in covered_constants and token not in constant_allowlist:
                missing_constants.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert sorted(set(missing_constants)) == []

    model_manifest = json.loads((REPO_ROOT / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    model_ids = {str(item["id"]) for item in model_manifest["models"]}
    model_coverage = set(manifest.raw.get("model_manifest_coverage", {}))
    assert sorted(model_ids - model_coverage) == []
