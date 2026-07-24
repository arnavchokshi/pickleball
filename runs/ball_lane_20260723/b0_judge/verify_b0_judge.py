#!/usr/bin/env python3
"""WS1.0 (B0) clean-judge verification for lane ball_lane_20260723.

Measurement-only. Reads the immutable ball_b0_split_20260721 artifacts (main
checkout + worktree tracked copy), recomputes counts/hashes, and emits
verification.json next to this script. It never writes outside its own lane
directory and never modifies any prior lane artifact.

Run:
  /Users/arnavchokshi/Desktop/pickleball/.venv/bin/python \
    runs/ball_lane_20260723/b0_judge/verify_b0_judge.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path

MAIN = Path("/Users/arnavchokshi/Desktop/pickleball")
WORKTREE = Path("/Users/arnavchokshi/Desktop/pickleball/.claude/worktrees/ball-lane-20260723")
LANE_MAIN = MAIN / "runs/lanes/ball_b0_split_20260721"
LANE_WT = WORKTREE / "runs/lanes/ball_b0_split_20260721"
OUT_DIR = WORKTREE / "runs/ball_lane_20260723/b0_judge"
SCORER = WORKTREE / "scripts/racketsport/ball_loso_validation.py"

HOLDOUT_FAMILIES = {"HyUqT7zFiwk", "Ezz6HDNHlnk"}
TRAIN_FAMILIES = {"73VurrTKCZ8", "_L0HVmAlCQI", "wBu8bC4OfUY", "zwCtH_i1_S4"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frozen_constants_from_scorer() -> tuple[dict, dict]:
    """Extract FROZEN_B0_ARTIFACT_SHA256 / FROZEN_B0_SOURCE_VIDEO_SHA256 from
    the committed scorer via AST (no import side effects)."""
    tree = ast.parse(SCORER.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "FROZEN_B0_ARTIFACT_SHA256",
                "FROZEN_B0_SOURCE_VIDEO_SHA256",
            ):
                found[target.id] = ast.literal_eval(node.value)
    return found["FROZEN_B0_ARTIFACT_SHA256"], found["FROZEN_B0_SOURCE_VIDEO_SHA256"]


def check(name: str, ok: bool, detail):
    return {"check": name, "verdict": "PASS" if ok else "FAIL", "detail": detail}


def main() -> int:
    checks: list[dict] = []
    frozen_artifacts, frozen_videos = frozen_constants_from_scorer()

    # V1: frozen artifact hashes -- main copy and worktree tracked copy.
    for label, lane in (("main_checkout", LANE_MAIN), ("worktree_tracked", LANE_WT)):
        actual = {name: sha256_file(lane / "split" / name) for name in frozen_artifacts}
        checks.append(
            check(
                f"frozen_split_artifact_sha256[{label}]",
                actual == frozen_artifacts,
                {"expected_source": "scripts/racketsport/ball_loso_validation.py FROZEN_B0_ARTIFACT_SHA256",
                 "actual": actual},
            )
        )
    fix2_hashes = {
        name: sha256_file(LANE_MAIN / "split_fix2" / name)
        for name in ("report.json", "train.jsonl", "validation.jsonl", "lineage_rows.jsonl")
    }

    # Load rows (round 1 = frozen judge; fix2 = provenance-hardened rebuild).
    split1 = {n: read_jsonl(LANE_MAIN / "split" / f"{n}.jsonl") for n in ("train", "validation", "lineage_rows")}
    split2 = {n: read_jsonl(LANE_MAIN / "split_fix2" / f"{n}.jsonl") for n in ("train", "validation", "lineage_rows")}
    report1 = json.loads((LANE_MAIN / "split/report.json").read_text())
    report2 = json.loads((LANE_MAIN / "split_fix2/report.json").read_text())

    # V2: row counts.
    for label, split, report in (("split", split1, report1), ("split_fix2", split2, report2)):
        counts = {
            "train": len(split["train"]),
            "validation": len(split["validation"]),
            "lineage_rows": len(split["lineage_rows"]),
            "scratch_train": sum(1 for r in split["train"] if r["lineage_class"] == "scratch"),
            "old_train": sum(1 for r in split["train"] if r["lineage_class"] != "scratch"),
        }
        expected = {"train": 2249, "validation": 167, "lineage_rows": 3376, "scratch_train": 183, "old_train": 2066}
        report_counts = report["split_counts"]
        ok = counts == expected and report_counts == {
            "old_train": 2066, "scratch_train": 183, "train": 2249, "validation": 167}
        checks.append(check(f"row_counts[{label}]", ok, {"actual": counts, "report_split_counts": report_counts}))

    # V3: validation purity (scratch-only, no prelabels) for both rounds.
    for label, split in (("split", split1), ("split_fix2", split2)):
        rows = split["validation"]
        bad = [
            r["row_key"]
            for r in rows
            if r["lineage_class"] != "scratch"
            or r.get("original_prelabel") is not None
            or r.get("lineage_origin") != "scratch_no_prelabel_package"
            or r.get("evaluation_eligible") is not True
            or r.get("split") != "validation"
            or r.get("teacher_derived") is True
        ]
        per_source = {}
        for r in rows:
            per_source[r["parent_source_id"]] = per_source.get(r["parent_source_id"], 0) + 1
        venue = {}
        for r in rows:
            venue.setdefault(r["parent_source_id"], set()).add(str(r.get("source_class")))
        ok = not bad and per_source == {"HyUqT7zFiwk": 100, "Ezz6HDNHlnk": 67}
        checks.append(
            check(
                f"validation_scratch_only[{label}]",
                ok,
                {"violations": bad, "per_source_counts": per_source,
                 "source_class_by_family": {k: sorted(v) for k, v in venue.items()}},
            )
        )

    # V4: confirmed_prelabel weight policy in train.
    for label, split in (("split", split1), ("split_fix2", split2)):
        by_class: dict[str, dict] = {}
        bad_weight = []
        for r in split["train"]:
            cls = r["lineage_class"]
            entry = by_class.setdefault(cls, {"rows": 0, "weights": set()})
            entry["rows"] += 1
            entry["weights"].add(r["training_weight"])
            expected_weight = 0.25 if cls == "confirmed_prelabel" else 1.0
            if r["training_weight"] != expected_weight:
                bad_weight.append(r["row_key"])
        ok = not bad_weight and set(by_class) == {"confirmed_prelabel", "corrected_prelabel", "scratch"}
        checks.append(
            check(
                f"train_weight_policy[{label}]",
                ok,
                {"by_class": {k: {"rows": v["rows"], "weights": sorted(v["weights"])} for k, v in by_class.items()},
                 "weight_violations": bad_weight},
            )
        )

    # V5: train/validation disjointness on every identity axis present.
    for label, split in (("split", split1), ("split_fix2", split2)):
        train, val = split["train"], split["validation"]
        axes = {
            "parent_source_id": (
                {r["parent_source_id"] for r in train},
                {r["parent_source_id"] for r in val},
            ),
            "clip_id": ({r["clip_id"] for r in train}, {r["clip_id"] for r in val}),
            "row_key": ({r["row_key"] for r in train}, {r["row_key"] for r in val}),
            "image_name": (
                {r.get("image_name") for r in train} - {None},
                {r.get("image_name") for r in val} - {None},
            ),
        }
        inter = {axis: sorted(a & b) for axis, (a, b) in axes.items()}
        ok = (
            not any(inter.values())
            and axes["parent_source_id"][0] == TRAIN_FAMILIES
            and axes["parent_source_id"][1] == HOLDOUT_FAMILIES
        )
        checks.append(
            check(
                f"train_validation_disjoint[{label}]",
                ok,
                {"intersections": inter,
                 "train_families": sorted(axes["parent_source_id"][0]),
                 "validation_families": sorted(axes["parent_source_id"][1])},
            )
        )

    # V6: 960 historical (non-scratch) HyU/Ezz rows exist and none reach train.
    for label, split in (("split", split1), ("split_fix2", split2)):
        lineage = split["lineage_rows"]
        historical_holdout = [
            r for r in lineage
            if r.get("parent_source_id") in HOLDOUT_FAMILIES and r.get("lineage_class") != "scratch"
        ]
        train_keys = {r["row_key"] for r in split["train"]}
        leaked = sorted({r["row_key"] for r in historical_holdout} & train_keys)
        ok = len(historical_holdout) == 960 and not leaked
        checks.append(
            check(
                f"historical_holdout_excluded[{label}]",
                ok,
                {"historical_non_scratch_HyU_Ezz_rows": len(historical_holdout),
                 "expected": 960, "leaked_into_train": leaked},
            )
        )

    # V7: cross-round judge identity (round-1 frozen judge vs fix2 rebuild).
    def judge_map(rows):
        return {
            r["row_key"]: {
                "clip_id": r["clip_id"],
                "frame_index": r["frame_index"],
                "parent_source_id": r["parent_source_id"],
                "final_label": r["final_label"],
                "ground_truth": r["ground_truth"],
                "evaluation_eligible": r["evaluation_eligible"],
                "image_name": r.get("image_name"),
                "image_md5": r.get("image_md5"),
            }
            for r in rows
        }

    j1, j2 = judge_map(split1["validation"]), judge_map(split2["validation"])
    key_delta = {"only_in_split": sorted(set(j1) - set(j2)), "only_in_fix2": sorted(set(j2) - set(j1))}
    field_deltas = [k for k in j1 if k in j2 and j1[k] != j2[k]]
    checks.append(
        check(
            "judge_identity_across_rounds[validation]",
            not key_delta["only_in_split"] and not key_delta["only_in_fix2"] and not field_deltas,
            {"row_key_delta": key_delta, "rows_with_field_differences": field_deltas},
        )
    )
    # Train-side cross-round comparison (informational: fix2 reclassified rows are reported, not hidden).
    t1 = {r["row_key"]: (r["lineage_class"], r["training_weight"]) for r in split1["train"]}
    t2 = {r["row_key"]: (r["lineage_class"], r["training_weight"]) for r in split2["train"]}
    train_diffs = {k: {"split": t1[k], "split_fix2": t2[k]} for k in t1 if k in t2 and t1[k] != t2[k]}
    checks.append(
        {
            "check": "train_cross_round_differences[informational]",
            "verdict": "INFO",
            "detail": {
                "row_key_set_equal": set(t1) == set(t2),
                "rows_with_class_or_weight_change": train_diffs,
            },
        }
    )

    # V8: image byte binding -- zip sha256 + all 167 validation member digests.
    zip_path = MAIN / "cvat_upload/w7_audit_stratum_20260709/w7_audit_stratum_uniform350_images.zip"
    expected_zip_sha = report2["checks"]["scratch_materialized_image_bytes"]["image_zip_sha256"]
    actual_zip_sha = sha256_file(zip_path)
    contract = report2["input_contract"]["image_zip_entry_sha256"]
    member_mismatches = []
    md5_mismatches = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for r in split2["validation"]:
            member = r["image_zip_member"]
            if member not in names:
                member_mismatches.append({"row_key": r["row_key"], "reason": "member_absent"})
                continue
            data = zf.read(member)
            digest = hashlib.sha256(data).hexdigest()
            if digest != r["image_sha256"] or digest != contract.get(member):
                member_mismatches.append(
                    {"row_key": r["row_key"], "member": member, "computed": digest,
                     "row_image_sha256": r["image_sha256"], "contract": contract.get(member)}
                )
            if hashlib.md5(data).hexdigest() != r["image_md5"]:
                md5_mismatches.append(r["row_key"])
    checks.append(
        check(
            "validation_image_byte_binding",
            actual_zip_sha == expected_zip_sha and not member_mismatches and not md5_mismatches,
            {"zip_path": str(zip_path), "zip_sha256_expected": expected_zip_sha,
             "zip_sha256_actual": actual_zip_sha, "members_checked": len(split2["validation"]),
             "sha256_mismatches": member_mismatches, "md5_mismatches": md5_mismatches},
        )
    )

    # V9: input-contract manifest digests.
    ic1 = report1["input_contract"]
    sampling_path = Path(ic1["scratch_sampling_manifest"])
    package_path = MAIN / ic1["scratch_package"]
    export_path = MAIN / ic1["scratch_export"]
    digests = {
        "scratch_sampling_manifest_md5": (ic1["scratch_sampling_manifest_md5"], md5_file(sampling_path)),
        "scratch_package_sha256": (ic1["scratch_package_sha256"], sha256_file(package_path)),
        "scratch_export_sha256": (ic1["scratch_export_sha256"], sha256_file(export_path)),
    }
    checks.append(
        check(
            "input_contract_manifest_digests",
            all(exp == act for exp, act in digests.values()),
            {k: {"expected": exp, "actual": act} for k, (exp, act) in digests.items()},
        )
    )

    # V10: judge source-video identity constants vs on-disk media (main checkout).
    video_results = {}
    for clip, expected_sha in frozen_videos.items():
        parent = clip.split("_rally_")[0]
        path = MAIN / f"data/online_harvest_20260706/rallies/{parent}/{clip}.mp4"
        actual = sha256_file(path) if path.exists() else None
        video_results[clip] = {"path": str(path), "expected": expected_sha, "actual": actual,
                               "match": actual == expected_sha}
    checks.append(
        check(
            "frozen_judge_source_videos_local",
            all(v["match"] for v in video_results.values()),
            video_results,
        )
    )

    # V11: protected-collision guard -- recorded evidence only (not re-executed here).
    guard = report2["protected_collision_guard"]
    checks.append(
        {
            "check": "protected_collision_guard[reported_not_reexecuted]",
            "verdict": "PASS_AS_REPORTED" if guard.get("collision_count") == 0 else "FAIL",
            "detail": {
                "protected_frame_count": report2["checks"]["protected_collision_count"]["protected_frame_count"],
                "collision_count": guard.get("collision_count"),
                "hash_type": guard.get("hash_type"),
                "collision_hamming_threshold": guard.get("collision_hamming_threshold"),
                "independent_review": "runs/lanes/ball_b0_split_20260721_review/review.json protected_frame_purity PASS (2953 frames, 0 exact, 0 dhash<=3)",
            },
        }
    )

    # V12: ledger family scan -- where do the holdout families appear as train pools?
    ledger = json.loads((WORKTREE / "runs/manager/data_ledger.json").read_text())
    family_table = []
    for asset in ledger["assets"]:
        part = asset.get("partitions", {})
        hits_train = sorted(HOLDOUT_FAMILIES & set(part.get("train", [])))
        hits_val = sorted(HOLDOUT_FAMILIES & set(part.get("val", [])))
        in_sources = sorted(HOLDOUT_FAMILIES & set(asset.get("source_lineage", {}).get("original_sources", [])))
        if hits_train or hits_val or in_sources:
            family_table.append(
                {"asset_id": asset["asset_id"], "state": asset["state"],
                 "holdout_families_in_train_partition": hits_train,
                 "holdout_families_in_val_partition": hits_val,
                 "holdout_families_in_original_sources": in_sources}
            )
    checks.append({"check": "ledger_holdout_family_scan", "verdict": "INFO", "detail": family_table})

    result = {
        "artifact_type": "racketsport_ball_b0_judge_verification",
        "lane": "ball_lane_20260723/b0_judge",
        "verified_flag": 0,
        "language": "measurement-only; no promotion claims",
        "frozen_judge": {
            "path": "runs/lanes/ball_b0_split_20260721/split/validation.jsonl",
            "row_count": len(split1["validation"]),
            "sha256": sha256_file(LANE_MAIN / "split/validation.jsonl"),
            "identity_authority": "FROZEN_B0_ARTIFACT_SHA256 in scripts/racketsport/ball_loso_validation.py (commit 4c27023)",
        },
        "split_fix2_hashes": fix2_hashes,
        "checks": checks,
        "overall": "PASS" if all(c["verdict"] in ("PASS", "PASS_AS_REPORTED", "INFO") for c in checks) else "FAIL",
    }
    out = OUT_DIR / "verification.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "out": str(out),
                      "fail_checks": [c["check"] for c in checks if c["verdict"] == "FAIL"]}, indent=2))
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
