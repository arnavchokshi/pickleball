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
    assert manifest.revision == 5
    assert "A manifest entry is a DEFAULT selection, NEVER a VERIFIED claim" in manifest.invariants
    assert len(manifest.entries) >= 30

    required_entries = {
        "ball.wasb_checkpoint",
        "ball.wasb_repo",
        "tracking.reid_model",
        "tracking.global_association_profile",
        "tracking.eval_only_association_profiles",
        "confidence.calibration_curves",
        "mesh.coverage_mode",
        "mesh.byte_budget_mib",
        "mesh.tier_eligibility_raise",
        "mesh.target_frame_budget",
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
    }
    assert required_entries <= set(manifest.entries)
    assert "mesh.byte_budget_policy" not in manifest.entries
    assert "ball.magnus_fit_spin_scalar" not in manifest.entries

    assert manifest.entry("mesh.byte_budget_mib").status == "WIRED_DEFAULT"
    assert any("w6_close_errand_20260708" in path for path in manifest.entry("mesh.byte_budget_mib").provenance["evidence_paths"])
    assert "21.3fps" in manifest.entry("mesh.byte_budget_mib").notes
    assert manifest.entry("mesh.tier_eligibility_raise").status == "PENDING"
    assert manifest.entry("body.p22_lambda_foot_smoother").status == "DORMANT"
    assert "NOT-WIRING-READY" in manifest.entry("body.p22_lambda_foot_smoother").notes
    assert manifest.entry("body.p22_lambda_foot_smoother").proven_against["gate_1b_world_round_trip_mm"] == 262.35
    assert manifest.entry("body.postchain_raw_knob").status == "DORMANT"
    assert manifest.entry("instrument.gate_check_body_decode").status == "DORMANT"
    assert manifest.entry("body.fast_sam_3d_body_challenger_not_adopt").status == "DORMANT"
    assert manifest.entry("ball.arc_solver_spin").status == "DORMANT"
    assert "kill-fired" in manifest.entry("ball.arc_solver_spin").notes
    assert "loso_report.json" in manifest.entry("ball.wasb_checkpoint").notes
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
