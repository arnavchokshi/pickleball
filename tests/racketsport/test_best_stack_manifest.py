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
    assert manifest.revision == 2
    assert "A manifest entry is a DEFAULT selection, NEVER a VERIFIED claim" in manifest.invariants
    assert len(manifest.entries) == 28

    required_entries = {
        "ball.wasb_checkpoint",
        "ball.wasb_repo",
        "tracking.reid_model",
        "tracking.global_association_profile",
        "confidence.calibration_curves",
        "mesh.coverage_mode",
        "mesh.byte_budget_mib",
        "mesh.byte_budget_policy",
        "mesh.target_frame_budget",
        "body.detector_fov",
        "camera_motion.policy",
        "server.allow_auto_court_corners_preview",
        "ball.seed_official_checkpoint",
        "ball.stage1_official_checkpoint",
        "court.court_unet_v2",
        "court.e4_fusion_default",
        "body.postchain_raw_knob",
        "instrument.gate_check_body_decode",
        "ball.arc_solver_spin",
    }
    assert required_entries <= set(manifest.entries)
    assert "body.p22_lambda_foot_smoother" not in manifest.entries
    assert "ball.magnus_fit_spin_scalar" not in manifest.entries
    assert "mesh.tier_eligibility_raise" not in manifest.entries

    assert manifest.entry("mesh.byte_budget_policy").status == "PENDING"
    assert manifest.entry("body.postchain_raw_knob").status == "DORMANT"
    assert manifest.entry("instrument.gate_check_body_decode").status == "DORMANT"
    assert manifest.entry("ball.arc_solver_spin").status == "DORMANT"
    assert "loso_report.json" in manifest.entry("ball.wasb_checkpoint").notes

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
        "mesh.byte_budget_policy",
        "body.postchain_raw_knob",
        "instrument.gate_check_body_decode",
        "ball.arc_solver_spin",
        "ball.wasb_checkpoint",
    }
    missing_evidence: list[str] = []
    for key in evidence_checked_entries:
        for raw_path in manifest.entry(key).provenance["evidence_paths"]:
            path = REPO_ROOT / str(raw_path)
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
