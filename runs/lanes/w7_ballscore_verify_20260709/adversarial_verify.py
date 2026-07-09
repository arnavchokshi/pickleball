#!/usr/bin/env python3
"""Adversarial verifier for w7_ballretrain_20260709.

This script is intentionally read-only outside this lane. It proves or fails to
prove the six requested attack classes from local artifacts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / "runs/lanes/w7_ballscore_verify_20260709"
TARGET = ROOT / "runs/lanes/w7_ballretrain_20260709"
W6 = ROOT / "runs/lanes/w6_labelingest_20260708"
W5 = ROOT / "runs/lanes/w5_ballretrain_20260707"

EXPECTED_CORPUS_MD5 = "37a5d43ab537a15bd12d382bb882a5fe"
OFFICIAL_WASB_SHA256 = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def approx_equal(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def json_fingerprint(value: Any) -> str:
    return hashlib.md5(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_wasb_run_log(path: Path, candidate: str) -> list[dict[str, Any]]:
    """Extract the per-clip JSON summaries from a WASB scoring log."""

    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    i = 0
    marker_re = re.compile(rf"^=== \[{re.escape(candidate)}\] (?P<clip>.+) ===$")
    while i < len(lines):
        match = marker_re.match(lines[i])
        if not match:
            i += 1
            continue
        clip = match.group("clip")
        i += 1
        if i >= len(lines) or lines[i].strip() != "{":
            raise AssertionError(f"{path}: marker for {clip} is not followed by JSON")
        block: list[str] = []
        depth = 0
        while i < len(lines):
            line = lines[i]
            block.append(line)
            depth += line.count("{")
            depth -= line.count("}")
            i += 1
            if depth == 0:
                break
        payload = json.loads("\n".join(block))
        payload["_clip_marker"] = clip
        records.append(payload)
    return records


def summarize_candidate(report: dict[str, Any], candidate: str) -> dict[str, float]:
    row = report["candidates"][candidate]
    pooled = row["pooled_mixed_metrics"]
    mean = row["loso_mean_metrics"]
    return {
        "micro_label_f1_at_20px": pooled["micro_label_f1_at_20px"],
        "micro_hidden_false_positive_rate": pooled["micro_hidden_false_positive_rate"],
        "micro_precision_at_20px": pooled["micro_precision_at_20px"],
        "micro_visible_recall_at_20px": pooled["micro_visible_recall_at_20px"],
        "loso_mean_label_f1_at_20px": mean["label_f1_at_20px"],
        "loso_mean_hidden_false_positive_rate": mean["hidden_false_positive_rate"],
        "loso_mean_precision_at_20px": mean["precision_at_20px"],
        "loso_mean_visible_recall_at_20px": mean["visible_recall_at_20px"],
    }


def attack_1_circular_control() -> dict[str, Any]:
    arm0_report_path = TARGET / "vm_pull/arm0_control/gpu_rescore/loso/loso_report.json"
    arm4_report_path = TARGET / "vm_pull/arm4_score/loso_final/loso_report.json"
    w6_report_path = W6 / "gpu_rescore/loso/loso_report.json"
    arm0_log_path = TARGET / "vm_pull/logs/arm0_control.log"
    manifest_path = TARGET / "md5_manifest.txt"

    arm0 = load_json(arm0_report_path)
    arm4 = load_json(arm4_report_path)
    w6 = load_json(w6_report_path)
    arm0_records = parse_wasb_run_log(arm0_log_path, "official_tennis_control")
    md5_manifest = manifest_path.read_text(encoding="utf-8")
    expected_sources = set(arm0["candidates"]["official_tennis_control"]["sources_scored"])

    log_sources = {record["_clip_marker"] for record in arm0_records}
    runtime_ok = [
        record.get("source_mode") == "wasb_predict"
        and record.get("runtime", {}).get("device") == "cuda"
        and record.get("runtime", {}).get("wall_seconds", 0) > 0
        and record.get("runtime", {}).get("wasb_checkpoint", {}).get("sha256") == OFFICIAL_WASB_SHA256
        and "runs/lanes/w7_ballretrain_20260709/arm0_control/" in str(record.get("out", ""))
        for record in arm0_records
    ]
    raw_predictions = sorted(
        str(path.relative_to(ROOT))
        for path in (TARGET / "vm_pull/arm0_control").glob("**/ball_track.json")
    )

    arm0_control = arm0["candidates"]["official_tennis_control"]
    arm4_control = arm4["candidates"]["official_tennis_control"]
    w6_control = w6["candidates"]["official_tennis_control"]

    checks = {
        "arm0_log_has_20_control_runs": len(arm0_records) == 20,
        "arm0_log_sources_match_report": log_sources == expected_sources,
        "arm0_runtime_records_are_cuda_w7_official_anchor": all(runtime_ok),
        "arm0_loso_report_mtime_after_w6_report": arm0_report_path.stat().st_mtime > w6_report_path.stat().st_mtime,
        "arm0_report_md5_manifested": f"{md5(arm0_report_path)}  vm_pull/arm0_control/gpu_rescore/loso/loso_report.json" in md5_manifest,
        "arm0_log_md5_manifested": f"{md5(arm0_log_path)}  vm_pull/logs/arm0_control.log" in md5_manifest,
        "arm4_combined_control_row_matches_arm0_fresh_control_row": arm4_control == arm0_control,
        "w7_full_loso_report_md5_differs_from_w6_full_report_md5": md5(arm0_report_path) != md5(w6_report_path),
    }
    verdict = "CONFIRMED-VALID" if all(checks.values()) else "REFUTED"
    issues = []
    if not raw_predictions:
        issues.append(
            "Raw per-clip control ball_track.json/csv prediction files were not pulled; provenance is bounded to logs, report mtimes, and md5 manifests."
        )
    if arm0_control == w6_control:
        issues.append(
            "The official_tennis_control candidate object is byte-identical to w6, which is expected for deterministic control scoring but still means copied-vs-deterministic-equal cannot be distinguished from JSON alone."
        )

    return {
        "id": 1,
        "name": "CIRCULAR CONTROL / fresh control provenance",
        "verdict": verdict,
        "checks": checks,
        "evidence": {
            "arm0_report": str(arm0_report_path.relative_to(ROOT)),
            "arm0_log": str(arm0_log_path.relative_to(ROOT)),
            "arm4_report": str(arm4_report_path.relative_to(ROOT)),
            "w6_report": str(w6_report_path.relative_to(ROOT)),
            "arm0_report_mtime_utc": iso_mtime(arm0_report_path),
            "w6_report_mtime_utc": iso_mtime(w6_report_path),
            "arm0_control_fingerprint": json_fingerprint(arm0_control),
            "arm4_control_fingerprint": json_fingerprint(arm4_control),
            "w6_control_fingerprint": json_fingerprint(w6_control),
            "logged_control_clip_count": len(arm0_records),
            "raw_prediction_files_pulled": raw_predictions,
            "control_metrics": summarize_candidate(arm0, "official_tennis_control"),
        },
        "issues": issues,
    }


def attack_2_checkpoint_provenance() -> dict[str, Any]:
    a_summary_path = TARGET / "vm_pull/arm3_finetunes/A_seed_official_aug/summary.json"
    c_summary_path = TARGET / "vm_pull/arm3_finetunes/C_stage1_official_aug/summary.json"
    a_ckpt_path = TARGET / "vm_pull/arm3_finetunes/A_seed_official_aug/checkpoints/latest.pt"
    c_ckpt_path = TARGET / "vm_pull/arm3_finetunes/C_stage1_official_aug/checkpoints/latest.pt"
    seed_base_path = W5 / "vm_pull/seed_official/checkpoints/latest.pt"
    stage1_base_path = W5 / "vm_pull/stage1_official/checkpoints/latest.pt"
    official_base_path = ROOT / "models/checkpoints/wasb/wasb_tennis_best.pth.tar"

    a = load_json(a_summary_path)
    c = load_json(c_summary_path)
    probe100 = load_json(TARGET / "vm_pull/arm2_probe/probe100/summary.json")
    probe300 = load_json(TARGET / "vm_pull/arm2_probe/probe300/summary.json")
    delta_rate = (probe300["recipe"]["steps"] - probe100["recipe"]["steps"]) / (
        probe300["runtime"]["wall_seconds"] - probe100["runtime"]["wall_seconds"]
    )
    budget_steps = min(12000, math.floor(45 * 60 * delta_rate))

    file_hashes = {
        "A_md5": md5(a_ckpt_path),
        "C_md5": md5(c_ckpt_path),
        "A_sha256_file": sha256(a_ckpt_path),
        "C_sha256_file": sha256(c_ckpt_path),
        "seed_base_md5": md5(seed_base_path),
        "stage1_base_md5": md5(stage1_base_path),
        "official_anchor_md5": md5(official_base_path),
    }
    checks = {
        "A_checkpoint_file_exists": a_ckpt_path.is_file(),
        "C_checkpoint_file_exists": c_ckpt_path.is_file(),
        "A_C_checkpoint_md5_distinct": file_hashes["A_md5"] != file_hashes["C_md5"],
        "A_checkpoint_distinct_from_seed_base": file_hashes["A_md5"] != file_hashes["seed_base_md5"],
        "C_checkpoint_distinct_from_stage1_base": file_hashes["C_md5"] != file_hashes["stage1_base_md5"],
        "A_file_sha256_matches_scoring_log_sha": file_hashes["A_sha256_file"] == "630a4f4206b114ce9a09a6153297bfeb8074623cf061f0d4f985dfa810ede5a9",
        "C_file_sha256_matches_scoring_log_sha": file_hashes["C_sha256_file"] == "e6f1219a827a28bb029fa379b78e3844a580dbd340b75879e6b620f29d877e22",
        "A_state_distinct_from_loaded_base_state": a["checkpoint"]["state_sha256"] != a["model"]["init_summary"]["loaded_state_sha256"],
        "C_state_distinct_from_loaded_base_state": c["checkpoint"]["state_sha256"] != c["model"]["init_summary"]["loaded_state_sha256"],
        "A_loss_curve_exists_and_has_2372_values": a["loss"]["count"] == 2372 and len(a["loss"]["values"]) == 2372,
        "C_loss_curve_exists_and_has_2372_values": c["loss"]["count"] == 2372 and len(c["loss"]["values"]) == 2372,
        "A_step_is_budget_2372": a["checkpoint"]["step"] == 2372 and a["recipe"]["steps"] == 2372,
        "C_step_is_budget_2372": c["checkpoint"]["step"] == 2372 and c["recipe"]["steps"] == 2372,
        "budget_formula_recomputes_2372": budget_steps == 2372,
        "A_round_trip_state_sha_match": bool(a["checkpoint"]["round_trip_state_sha256_match"]),
        "C_round_trip_state_sha_match": bool(c["checkpoint"]["round_trip_state_sha256_match"]),
        "A_key_diff_empty": a["model"]["init_summary"]["missing_keys"] == [] and a["model"]["init_summary"]["unexpected_keys"] == [],
        "C_key_diff_empty": c["model"]["init_summary"]["missing_keys"] == [] and c["model"]["init_summary"]["unexpected_keys"] == [],
    }
    return {
        "id": 2,
        "name": "CHECKPOINT PROVENANCE / distinct A-C models",
        "verdict": "CONFIRMED-VALID" if all(checks.values()) else "REFUTED",
        "checks": checks,
        "evidence": {
            "A_summary": str(a_summary_path.relative_to(ROOT)),
            "C_summary": str(c_summary_path.relative_to(ROOT)),
            "hashes": file_hashes,
            "A_loss_first_last": [a["loss"]["first"], a["loss"]["last"]],
            "C_loss_first_last": [c["loss"]["first"], c["loss"]["last"]],
            "delta_rate_steps_per_s": delta_rate,
            "budget_steps": budget_steps,
        },
        "issues": [],
    }


def attack_3_loso_integrity() -> dict[str, Any]:
    corpus_manifest_path = W6 / "corpus_md5_manifest.json"
    fold_manifest_path = W6 / "loso_fold_manifest.json"
    report_path = TARGET / "vm_pull/arm4_score/loso_final/loso_report.json"
    corpus = load_json(corpus_manifest_path)
    folds = load_json(fold_manifest_path)
    report = load_json(report_path)

    all_corpus_clips = set(corpus["counts"]["per_clip"])
    all_report_sources: dict[str, set[str]] = {
        name: set(row["sources_scored"]) for name, row in report["candidates"].items()
    }
    excluded_totals: dict[str, dict[str, int]] = {}
    for name, row in report["candidates"].items():
        pooled = row["pooled_mixed_metrics"]
        excluded_totals[name] = {
            key: int(value)
            for key, value in pooled.items()
            if key.startswith("total_excluded_")
        }

    recomputed_disjoint: list[dict[str, Any]] = []
    for fold in folds["folds"]:
        train = set(fold["train_row_keys"])
        val = set(fold["val_row_keys"])
        recomputed_disjoint.append(
            {
                "source_id": fold["source_id"],
                "source_class": fold["source_class"],
                "is_outdoor_fold": bool(fold.get("is_outdoor_fold")),
                "train_row_count": len(train),
                "val_row_count": len(val),
                "intersection_count": len(train & val),
            }
        )
    outdoor_sources = [row["source_id"] for row in recomputed_disjoint if row["is_outdoor_fold"]]
    checks = {
        "corpus_manifest_md5_matches_expected_1121": md5(corpus_manifest_path) == EXPECTED_CORPUS_MD5
        and corpus["counts"]["totals"]["reviewed_row_count"] == 1121,
        "source_fold_manifest_all_disjoint": folds["fold_disjointness"]["all_disjoint"] is True
        and all(row["intersection_count"] == 0 for row in recomputed_disjoint),
        "source_fold_count_is_6": len(folds["folds"]) == 6,
        "outdoor_source_folds_present": len(outdoor_sources) >= 1,
        "w7_harness_scores_all_20_clips_for_every_candidate": all(
            sources == all_corpus_clips for sources in all_report_sources.values()
        ),
        "w7_harness_fold_count_20_for_every_candidate": all(
            row["fold_count"] == 20 for row in report["candidates"].values()
        ),
        "no_excluded_frames_or_labels_in_scored_report": all(
            all(value == 0 for value in totals.values()) for totals in excluded_totals.values()
        ),
    }
    return {
        "id": 3,
        "name": "LoSO INTEGRITY / corpus md5, disjoint folds, OUTDOOR scored",
        "verdict": "CONFIRMED-VALID" if all(checks.values()) else "REFUTED",
        "checks": checks,
        "evidence": {
            "corpus_manifest": str(corpus_manifest_path.relative_to(ROOT)),
            "fold_manifest": str(fold_manifest_path.relative_to(ROOT)),
            "scored_report": str(report_path.relative_to(ROOT)),
            "corpus_manifest_md5": md5(corpus_manifest_path),
            "corpus_reviewed_row_count": corpus["counts"]["totals"]["reviewed_row_count"],
            "source_fold_disjointness": recomputed_disjoint,
            "outdoor_source_folds": outdoor_sources,
            "scored_clip_count_per_candidate": {
                name: len(sources) for name, sources in all_report_sources.items()
            },
            "excluded_totals": excluded_totals,
        },
        "issues": [
            "W7 scoring uses the known 20 per-clip folds rather than the 6 source-grouped folds; this is protocol-consistent with W6 but not true source-grouped LoSO."
        ],
    }


def attack_4_contract_evidence() -> dict[str, Any]:
    report_path = TARGET / "REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    claim_present = "test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine PASSED" in report_text
    search_terms = ("pytest", "test_ball_stage2_training", "tensor_and_label_geometry", "20 passed")
    evidence_hits: list[dict[str, str]] = []
    for path in TARGET.rglob("*"):
        if not path.is_file() or path == report_path:
            continue
        if path.suffix not in {".log", ".txt", ".md", ".json", ".sh"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            continue
        matched = [term for term in search_terms if term in text]
        if matched:
            evidence_hits.append({"path": str(path.relative_to(ROOT)), "matched_terms": ",".join(matched)})
    expected_w5_contract_artifact = W5 / "vm_contract_tests.log"
    checks = {
        "report_claims_contract_pytest_passed": claim_present,
        "w7_has_non_report_contract_execution_artifact": bool(evidence_hits),
        "w5_precedent_contract_log_exists": expected_w5_contract_artifact.is_file(),
    }
    verdict = "CONFIRMED-VALID" if checks["w7_has_non_report_contract_execution_artifact"] else "UNVERIFIABLE"
    return {
        "id": 4,
        "name": "CONTRACT CHECK / executed tensor assertion evidence",
        "verdict": verdict,
        "checks": checks,
        "evidence": {
            "report": str(report_path.relative_to(ROOT)),
            "non_report_contract_artifact_hits": evidence_hits,
            "w5_contract_log_precedent": str(expected_w5_contract_artifact.relative_to(ROOT)),
        },
        "issues": [
            "The lane report claims the tensor-contract pytest passed, but no pulled non-report stdout/log artifact containing that assertion or '20 passed' was found under the w7 lane."
        ]
        if verdict == "UNVERIFIABLE"
        else [],
    }


def attack_5_metric_keys() -> dict[str, Any]:
    report_path = TARGET / "vm_pull/arm4_score/loso_final/loso_report.json"
    report = load_json(report_path)
    expected = {
        "official_tennis_control": (0.361111, 0.599089),
        "A_seed_official_aug": (0.615172, 0.250569),
        "C_stage1_official_aug": (0.612075, 0.259681),
    }
    extracted: dict[str, dict[str, float]] = {}
    checks: dict[str, bool] = {}
    for candidate, (expected_f1, expected_hfp) in expected.items():
        pooled = report["candidates"][candidate]["pooled_mixed_metrics"]
        extracted[candidate] = {
            "pooled_mixed_metrics.micro_label_f1_at_20px": pooled["micro_label_f1_at_20px"],
            "pooled_mixed_metrics.micro_hidden_false_positive_rate": pooled[
                "micro_hidden_false_positive_rate"
            ],
        }
        checks[f"{candidate}_uses_exact_micro_f1_key"] = approx_equal(
            pooled["micro_label_f1_at_20px"], expected_f1, 0.0000005
        )
        checks[f"{candidate}_uses_exact_micro_hfp_key"] = approx_equal(
            pooled["micro_hidden_false_positive_rate"], expected_hfp, 0.0000005
        )
    checks["harness_key_container_is_pooled_mixed_metrics"] = all(
        "pooled_mixed_metrics" in row for row in report["candidates"].values()
    )
    return {
        "id": 5,
        "name": "METRIC KEYS / exact harness statistics",
        "verdict": "CONFIRMED-VALID" if all(checks.values()) else "REFUTED",
        "checks": checks,
        "evidence": {
            "scored_report": str(report_path.relative_to(ROOT)),
            "extracted_exact_keys": extracted,
        },
        "issues": [
            "REPORT.md prose says 'pooled micro_*', while the machine key is pooled_mixed_metrics.micro_*; the numeric values are from the exact machine keys."
        ],
    }


def attack_6_486_protocol() -> dict[str, Any]:
    w5_path = W5 / "vm_pull/loso/loso_report.json"
    w7_path = TARGET / "vm_pull/arm1_486anomaly/gpu_rescore/loso/loso_report.json"
    w5 = load_json(w5_path)
    w7 = load_json(w7_path)
    candidates = ["official_tennis_control", "seed_official"]
    expected_sources = {
        "burlington_gold_0300_low_steep_corner",
        "wolverine_mixed_0200_mid_steep_corner",
    }
    checks: dict[str, bool] = {
        "w7_arm1_candidates_are_control_and_seed": set(w7["candidates"]) == set(candidates),
        "w7_arm1_sources_match_w5_for_control_and_seed": all(
            set(w7["candidates"][candidate]["sources_scored"]) == expected_sources
            and set(w5["candidates"][candidate]["sources_scored"]) == expected_sources
            for candidate in candidates
        ),
        "w7_arm1_reviewed_frame_count_900_for_control_and_seed": all(
            w7["candidates"][candidate]["pooled_mixed_metrics"]["total_reviewed_frame_count"] == 900
            for candidate in candidates
        ),
        "w5_reviewed_frame_count_900_for_control_and_seed": all(
            w5["candidates"][candidate]["pooled_mixed_metrics"]["total_reviewed_frame_count"] == 900
            for candidate in candidates
        ),
        "w7_control_loso_mean_beats_seed_same_direction_as_w5": (
            w7["candidates"]["official_tennis_control"]["loso_mean_metrics"]["label_f1_at_20px"]
            > w7["candidates"]["seed_official"]["loso_mean_metrics"]["label_f1_at_20px"]
            and w5["candidates"]["official_tennis_control"]["loso_mean_metrics"]["label_f1_at_20px"]
            > w5["candidates"]["seed_official"]["loso_mean_metrics"]["label_f1_at_20px"]
        ),
        "w7_no_excluded_frames": all(
            all(
                int(value) == 0
                for key, value in w7["candidates"][candidate]["pooled_mixed_metrics"].items()
                if key.startswith("total_excluded_")
            )
            for candidate in candidates
        ),
    }
    diffs = {}
    for candidate in candidates:
        diffs[candidate] = {
            "w5_loso_mean_f1": w5["candidates"][candidate]["loso_mean_metrics"]["label_f1_at_20px"],
            "w7_loso_mean_f1": w7["candidates"][candidate]["loso_mean_metrics"]["label_f1_at_20px"],
            "abs_diff": abs(
                w5["candidates"][candidate]["loso_mean_metrics"]["label_f1_at_20px"]
                - w7["candidates"][candidate]["loso_mean_metrics"]["label_f1_at_20px"]
            ),
        }
    checks["w7_w5_loso_mean_f1_drift_under_0_003"] = all(row["abs_diff"] < 0.003 for row in diffs.values())
    return {
        "id": 6,
        "name": "486-ANOMALY ARM / W5 protocol identity",
        "verdict": "CONFIRMED-VALID" if all(checks.values()) else "REFUTED",
        "checks": checks,
        "evidence": {
            "w5_report": str(w5_path.relative_to(ROOT)),
            "w7_arm1_report": str(w7_path.relative_to(ROOT)),
            "sources": sorted(expected_sources),
            "diffs": diffs,
            "w7_metrics": {
                candidate: summarize_candidate(w7, candidate) for candidate in candidates
            },
        },
        "issues": [
            "W7 ARM1 intentionally scored only the anomaly pair (control and seed_official), not W5's stage1_official third row."
        ],
    }


def run() -> dict[str, Any]:
    attacks = [
        attack_1_circular_control(),
        attack_2_checkpoint_provenance(),
        attack_3_loso_integrity(),
        attack_4_contract_evidence(),
        attack_5_metric_keys(),
        attack_6_486_protocol(),
    ]
    counts = {
        "CONFIRMED-VALID": sum(1 for attack in attacks if attack["verdict"] == "CONFIRMED-VALID"),
        "REFUTED": sum(1 for attack in attacks if attack["verdict"] == "REFUTED"),
        "UNVERIFIABLE": sum(1 for attack in attacks if attack["verdict"] == "UNVERIFIABLE"),
    }
    return {
        "artifact_type": "w7_ballscore_verify_adversarial_proof_results",
        "schema_version": 1,
        "target_lane": "runs/lanes/w7_ballretrain_20260709",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attacks": attacks,
        "verdict_counts": counts,
        "overall": "PARTIAL" if counts["UNVERIFIABLE"] or counts["REFUTED"] else "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=LANE / "proof_results.json",
        help="Path to write proof results JSON.",
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "overall": result["overall"], "verdict_counts": result["verdict_counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
