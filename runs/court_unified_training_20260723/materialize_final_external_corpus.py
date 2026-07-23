#!/usr/bin/env python3
"""Materialize the owner-approved external court corpus with floor-only supervision.

This dated run helper never starts training and never mutates source corpora. Images are
referenced by relative symlink; label JSON is copied into a new corpus only after every external
``net_*`` channel has been set to JSON null. The canonical owner act is a deterministic,
hash-pinned compilation of the current nine-source review and the earlier pb.vision approvals.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
OUTPUT_ROOT = RUN_DIR / "final_external_corpus"
ROBOFLOW_OUTPUT = OUTPUT_ROOT / "roboflow_train"
PBVISION_OUTPUT = OUTPUT_ROOT / "pbvision_train"
ROBOFLOW_SOURCE = ROOT / "runs/lanes/court_data2b_20260709/real_court_corpus_partial"
ROBOFLOW_STATS = ROOT / "runs/lanes/court_data2b_20260709/corpus_stats.json"
PBVISION_SOURCE = ROOT / "data/court_real_pbvision_20260722"
PBVISION_FRAMES = ROOT / "cvat_upload/court_diversity_followup_20260723/frames"

ADMISSION_SUMMARY = RUN_DIR / "external_source_admission_final.json"
CURRENT_REVIEW_RESULT = RUN_DIR / "review_pack/results/final_ca86b21250fa.json"
PRIOR_PBV_OWNER_ACT = ROOT / "runs/lanes/court_owner_pack_20260722/results/batch_01_answers_final.json"
CANONICAL_OWNER_ACT = RUN_DIR / "external_training_owner_act_final.json"

NET_KEYS = ("net_left_sideline", "net_center", "net_right_sideline")
ROBOFLOW_DATASETS = (
    "chetan-rajagiri-9abfm__pickleball-court-v2__v1",
    "n-do-tran__pickleball-court-p3chl__v4",
    "necromancer__pickleball-court-vbmkq__v2",
    "nigh-workspace__pickleball-court-vhpgp__v11",
    "pickleball-ball-detection__pickleball-court-keypoints-syncz__v6",
    "ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3",
    "stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1",
)
PRIOR_APPROVED_PBV = (
    "143sf3gdwxsa",
    "98z43hspqz13",
    "td2szayjwtrj",
    "tqjlrcntpjvt",
)
NEWLY_APPROVED_PBV = ("st0epgnab7dr", "xkadsq9bli3h")
ALL_TRAIN_PBV = PRIOR_APPROVED_PBV + NEWLY_APPROVED_PBV


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))


def relative_repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def make_relative_symlink(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def assert_owner_evidence() -> dict[str, str]:
    admission = json.loads(ADMISSION_SUMMARY.read_text(encoding="utf-8"))
    review = json.loads(CURRENT_REVIEW_RESULT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_PBV_OWNER_ACT.read_text(encoding="utf-8"))

    expected_review_ids = {f"roboflow::{name}" for name in ROBOFLOW_DATASETS} | {
        f"pbvision::{name}" for name in NEWLY_APPROVED_PBV
    }
    if set(review.get("answers", {})) != expected_review_ids:
        raise ValueError("current owner review does not cover the exact nine requested sources")
    if any(answer.get("choice") != "KEEP" for answer in review["answers"].values()):
        raise ValueError("current owner review contains a non-KEEP decision")
    if admission.get("owner_decision", {}).get("external_net_keypoints") != "NULL_BEFORE_TRAINING":
        raise ValueError("owner admission does not require external net-keypoint nulling")
    prior_decisions = prior.get("final_decisions", {})
    if any(prior_decisions.get(video_id) != "APPROVE" for video_id in PRIOR_APPROVED_PBV):
        raise ValueError("prior pb.vision act does not approve the exact four carried-forward videos")
    return {
        "admission_summary_sha256": sha256_path(ADMISSION_SUMMARY),
        "current_review_sha256": sha256_path(CURRENT_REVIEW_RESULT),
        "prior_pbvision_owner_act_sha256": sha256_path(PRIOR_PBV_OWNER_ACT),
    }


def build_canonical_owner_act(evidence_hashes: dict[str, str]) -> str:
    final_decisions = {
        **{f"roboflow::{dataset}": "APPROVE" for dataset in ROBOFLOW_DATASETS},
        **{f"pbvision::{video_id}": "APPROVE" for video_id in ALL_TRAIN_PBV},
    }
    payload = {
        "artifact_type": "racketsport_court_external_training_owner_act",
        "schema_version": 1,
        "completed_at": "2026-07-23T07:12:44Z",
        "status": "OWNER_APPROVED_CONDITIONAL_FLOOR_ONLY",
        "final_decisions": final_decisions,
        "training_condition": {
            "court_floor_keypoints": "TRAIN",
            "external_net_keypoints": "NULL_ALL_THREE_CHANNELS",
            "required_net_semantic": "top_of_net",
            "reason": "Reviewed external net markers use bottom-of-net semantics and must not supervise top-of-net channels.",
        },
        "scopes": {
            "roboflow_dataset_ids": list(ROBOFLOW_DATASETS),
            "pbvision_video_ids": list(ALL_TRAIN_PBV),
            "row_scope": "only rows materialized in the hash-pinned final external corpus",
        },
        "evidence": {
            "current_nine_source_review": {
                "path": relative_repo_path(CURRENT_REVIEW_RESULT),
                "sha256": evidence_hashes["current_review_sha256"],
                "covers": [
                    *[f"roboflow::{dataset}" for dataset in ROBOFLOW_DATASETS],
                    *[f"pbvision::{video_id}" for video_id in NEWLY_APPROVED_PBV],
                ],
            },
            "admission_summary": {
                "path": relative_repo_path(ADMISSION_SUMMARY),
                "sha256": evidence_hashes["admission_summary_sha256"],
            },
            "incorporated_prior_pbvision_approvals": {
                "path": relative_repo_path(PRIOR_PBV_OWNER_ACT),
                "sha256": evidence_hashes["prior_pbvision_owner_act_sha256"],
                "video_ids": list(PRIOR_APPROVED_PBV),
            },
        },
        "training_started": False,
        "promotion_status": "not_promoted",
    }
    write_json(CANONICAL_OWNER_ACT, payload)
    return sha256_path(CANONICAL_OWNER_ACT)


def owner_binding(*, act_sha256: str, scope_type: str, scope_id: str) -> dict[str, Any]:
    return {
        "decision": "APPROVE",
        "path": relative_repo_path(CANONICAL_OWNER_ACT),
        "sha256": act_sha256,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "training_condition": "FLOOR_ONLY_ALL_NET_CHANNELS_NULL",
    }


def sanitize_item(item: dict[str, Any]) -> tuple[int, int]:
    original_nonnull = 0
    changed_values = 0
    for field in ("keypoints", "keypoint_confidence", "keypoint_spread_normalized"):
        values = item.get(field)
        if not isinstance(values, dict):
            continue
        for name in NET_KEYS:
            if values.get(name) is not None:
                original_nonnull += 1
                changed_values += 1
            values[name] = None
    item["status"] = "reviewed_external_dataset"
    item["pseudo_label_status"] = "OWNER_APPROVED"
    return original_nonnull, changed_values


def source_frame_dir(payload: dict[str, Any], label_path: Path) -> Path:
    frame_dir = Path(payload["frames"]["frame_dir"])
    if frame_dir.is_absolute():
        return frame_dir
    if payload["frames"].get("path_base") == "corpus_root":
        return label_path.parents[2] / frame_dir
    return label_path.parent / frame_dir


def materialize_roboflow(act_sha256: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for source_label in sorted(ROBOFLOW_SOURCE.glob("*/labels/court_keypoints.json")):
        payload = json.loads(source_label.read_text(encoding="utf-8"))
        items = payload["annotation"]["items"]
        dataset_ids = {item["provenance"]["dataset"] for item in items}
        if len(dataset_ids) != 1:
            raise ValueError(f"mixed dataset ids in {source_label}")
        dataset = next(iter(dataset_ids))
        if dataset not in ROBOFLOW_DATASETS:
            continue

        clip = source_label.parents[1].name
        payload["clip"] = clip
        source_frames = source_frame_dir(payload, source_label)
        output_frames = ROBOFLOW_OUTPUT / clip / "frames"
        payload["frames"]["frame_dir"] = f"{clip}/frames"
        payload["frames"]["path_base"] = "corpus_root"
        payload["status"] = "OWNER_APPROVED"
        payload["training_eligibility"] = {
            "queued": True,
            "reason": "owner kept this dataset for court-floor supervision with all external net channels nulled",
            "owner_adjudication": owner_binding(
                act_sha256=act_sha256, scope_type="roboflow_dataset", scope_id=dataset
            ),
        }
        payload["materialization"] = {
            "policy": "FLOOR_ONLY_ALL_NET_CHANNELS_NULL",
            "source_label_path": relative_repo_path(source_label),
            "source_label_sha256": sha256_path(source_label),
        }
        for item in items:
            original_nonnull, changed_values = sanitize_item(item)
            counters["rows"] += 1
            counters["rows_with_original_nonnull_net"] += int(original_nonnull > 0)
            counters["original_nonnull_net_values"] += original_nonnull
            counters["nulled_values"] += changed_values
            source_image = source_frames / item["frame"]
            source_sha = sha256_path(source_image)
            expected_sha = item["provenance"]["sha256"]
            if source_sha != expected_sha:
                raise ValueError(f"source image hash mismatch for {source_image}")
            destination = output_frames / item["frame"]
            make_relative_symlink(source_image, destination)
            rows.append({
                "kind": "roboflow",
                "scope_id": dataset,
                "clip": clip,
                "frame": item["frame"],
                "image": relative_repo_path(destination),
                "image_sha256": source_sha,
                "source_label": relative_repo_path(source_label),
            })
        output_label = ROBOFLOW_OUTPUT / clip / "labels/court_keypoints.json"
        write_json(output_label, payload)
        counters["label_files"] += 1
    return rows, dict(counters)


def materialize_pbvision(act_sha256: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for video_id in ALL_TRAIN_PBV:
        source_label = PBVISION_SOURCE / video_id / "labels/court_keypoints.json"
        payload = json.loads(source_label.read_text(encoding="utf-8"))
        items = payload["annotation"]["items"]
        if len(items) != 1 or payload.get("clip") != video_id:
            raise ValueError(f"unexpected pb.vision payload shape for {video_id}")
        source_images = list(PBVISION_FRAMES.glob(f"pbv_{video_id}_f*.png"))
        if len(source_images) != 1:
            raise ValueError(f"expected exactly one extracted frame for {video_id}")
        source_image = source_images[0]
        item = items[0]
        output_name = f"{Path(item['frame']).stem}.png"
        item["frame"] = output_name
        original_nonnull, changed_values = sanitize_item(item)
        counters["rows"] += 1
        counters["rows_with_original_nonnull_net"] += int(original_nonnull > 0)
        counters["original_nonnull_net_values"] += original_nonnull
        counters["nulled_values"] += changed_values

        payload["frames"]["frame_dir"] = f"{video_id}/frames"
        payload["frames"]["path_base"] = "corpus_root"
        payload["frames"]["available_review_frame_count"] = 1
        payload["status"] = "OWNER_APPROVED"
        payload["training_eligibility"] = {
            "queued": True,
            "reason": "owner approved this pb.vision teacher row for court-floor supervision with all net channels nulled",
            "owner_adjudication": owner_binding(
                act_sha256=act_sha256, scope_type="pbvision_video", scope_id=video_id
            ),
        }
        payload["materialization"] = {
            "policy": "FLOOR_ONLY_ALL_NET_CHANNELS_NULL",
            "source_label_path": relative_repo_path(source_label),
            "source_label_sha256": sha256_path(source_label),
            "source_frame_path": relative_repo_path(source_image),
        }
        destination = PBVISION_OUTPUT / video_id / "frames" / output_name
        make_relative_symlink(source_image, destination)
        output_label = PBVISION_OUTPUT / video_id / "labels/court_keypoints.json"
        write_json(output_label, payload)
        rows.append({
            "kind": "pbvision",
            "scope_id": video_id,
            "clip": video_id,
            "frame": output_name,
            "image": relative_repo_path(destination),
            "image_sha256": sha256_path(source_image),
            "source_label": relative_repo_path(source_label),
        })
        counters["label_files"] += 1
    return rows, dict(counters)


def verify_materialization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    net_values = 0
    image_symlinks = 0
    broken_symlinks = 0
    label_rows = 0
    source_counts: Counter[str] = Counter()
    for corpus_root in (ROBOFLOW_OUTPUT, PBVISION_OUTPUT):
        for label_path in sorted(corpus_root.glob("*/labels/court_keypoints.json")):
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            source_counts[payload["training_eligibility"]["owner_adjudication"]["scope_id"]] += len(
                payload["annotation"]["items"]
            )
            for item in payload["annotation"]["items"]:
                label_rows += 1
                for field in ("keypoints", "keypoint_confidence", "keypoint_spread_normalized"):
                    values = item.get(field)
                    if isinstance(values, dict):
                        net_values += sum(values.get(name) is not None for name in NET_KEYS)
                frame = corpus_root / payload["frames"]["frame_dir"] / item["frame"]
                image_symlinks += int(frame.is_symlink())
                broken_symlinks += int(frame.is_symlink() and not frame.is_file())
    if label_rows != len(rows):
        raise ValueError(f"manifest/label row mismatch: {len(rows)} vs {label_rows}")
    if net_values:
        raise ValueError(f"materialized corpus still has {net_values} non-null external net values")
    if image_symlinks != len(rows) or broken_symlinks:
        raise ValueError("materialized images are not exact, live symlink references")
    return {
        "label_rows": label_rows,
        "non_null_external_net_values": net_values,
        "image_symlinks": image_symlinks,
        "broken_image_symlinks": broken_symlinks,
        "rows_by_owner_scope": dict(sorted(source_counts.items())),
    }


def main() -> None:
    evidence_hashes = assert_owner_evidence()
    act_sha256 = build_canonical_owner_act(evidence_hashes)
    if OUTPUT_ROOT.exists():
        if OUTPUT_ROOT.parent != RUN_DIR:
            raise RuntimeError("refusing to replace an unexpected output root")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    roboflow_rows, roboflow_counts = materialize_roboflow(act_sha256)
    pbvision_rows, pbvision_counts = materialize_pbvision(act_sha256)
    rows = sorted(roboflow_rows + pbvision_rows, key=lambda row: (row["kind"], row["clip"], row["frame"]))
    if len(roboflow_rows) != 2833:
        raise ValueError(f"expected 2,833 Roboflow rows, found {len(roboflow_rows)}")
    if len(pbvision_rows) != 6:
        raise ValueError(f"expected 6 pb.vision train rows, found {len(pbvision_rows)}")
    if roboflow_counts.get("rows_with_original_nonnull_net") != 443:
        raise ValueError("expected 443 Roboflow rows to contain bottom-of-net values before nulling")
    if roboflow_counts.get("original_nonnull_net_values") != 921:
        raise ValueError("expected 921 bottom-of-net values before nulling")

    verification = verify_materialization(rows)
    row_digest = hashlib.sha256(
        json.dumps(
            [(row["kind"], row["scope_id"], row["clip"], row["frame"], row["image_sha256"]) for row in rows],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    label_hashes = {
        relative_repo_path(path): sha256_path(path)
        for corpus_root in (ROBOFLOW_OUTPUT, PBVISION_OUTPUT)
        for path in sorted(corpus_root.glob("*/labels/court_keypoints.json"))
    }
    manifest = {
        "artifact_type": "racketsport_court_owner_approved_external_training_corpus",
        "schema_version": 1,
        "status": "MATERIALIZED_FLOOR_ONLY_NOT_TRAINED",
        "training_started": False,
        "promotion_status": "not_promoted",
        "roots": {
            "roboflow_train": relative_repo_path(ROBOFLOW_OUTPUT),
            "pbvision_train": relative_repo_path(PBVISION_OUTPUT),
        },
        "canonical_owner_act": {
            "path": relative_repo_path(CANONICAL_OWNER_ACT),
            "sha256": act_sha256,
        },
        "policy": {
            "court_floor_keypoints": "TRAIN",
            "external_net_keypoints": "ALL_THREE_CHANNELS_NULL",
            "required_net_semantic": "top_of_net",
        },
        "counts": {
            "total_rows": len(rows),
            "roboflow_rows": len(roboflow_rows),
            "pbvision_rows": len(pbvision_rows),
            "roboflow_dataset_count": len(ROBOFLOW_DATASETS),
            "pbvision_video_count": len(ALL_TRAIN_PBV),
            "roboflow_rows_with_net_values_removed": roboflow_counts["rows_with_original_nonnull_net"],
            "roboflow_net_values_removed": roboflow_counts["original_nonnull_net_values"],
        },
        "row_digest_sha256": row_digest,
        "label_file_sha256": label_hashes,
        "rows": rows,
        "verification": verification,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    manifest_sha256 = sha256_path(OUTPUT_ROOT / "manifest.json")
    report = {
        "artifact_type": "racketsport_court_external_corpus_materialization_report",
        "schema_version": 1,
        "status": "PASS",
        "checks": {
            "exact_roboflow_rows_2833": len(roboflow_rows) == 2833,
            "exact_pbvision_rows_6": len(pbvision_rows) == 6,
            "all_external_net_channels_null": verification["non_null_external_net_values"] == 0,
            "all_images_symlinked_and_resolve": verification["image_symlinks"] == len(rows)
            and verification["broken_image_symlinks"] == 0,
            "owner_scope_count_13": len(verification["rows_by_owner_scope"]) == 13,
            "training_not_started": True,
        },
        "counts": manifest["counts"],
        "verification": verification,
        "immutable": {
            "canonical_owner_act_sha256": act_sha256,
            "row_digest_sha256": row_digest,
            "manifest_sha256": manifest_sha256,
        },
    }
    write_json(OUTPUT_ROOT / "verification_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
