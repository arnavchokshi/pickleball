#!/usr/bin/env python3
"""Validate and ingest the owner person-box export with provenance and audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CLASSES = {"player", "off_court_person"}
SOURCES = {"proposal_confirmed", "proposal_adjusted", "proposal_deleted", "drawn"}
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iou(left: dict[str, float], right: dict[str, float]) -> float:
    x1, y1 = max(left["x1"], right["x1"]), max(left["y1"], right["y1"])
    x2, y2 = min(left["x2"], right["x2"]), min(left["y2"], right["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left["x2"] - left["x1"]) * max(0.0, left["y2"] - left["y1"])
    right_area = max(0.0, right["x2"] - right["x1"]) * max(0.0, right["y2"] - right["y1"])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _match_iou(owner: list[dict[str, Any]], proposals: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for owner_index, owner_box in enumerate(owner):
        for proposal_index, proposal_box in enumerate(proposals):
            pairs.append((_iou(owner_box, proposal_box), owner_index, proposal_index))
    used_owner: set[int] = set()
    used_proposals: set[int] = set()
    matches: list[float] = []
    for iou, owner_index, proposal_index in sorted(pairs, reverse=True):
        if iou < threshold:
            break
        if owner_index in used_owner or proposal_index in used_proposals:
            continue
        used_owner.add(owner_index)
        used_proposals.add(proposal_index)
        matches.append(iou)
    return {
        "owner_count": len(owner),
        "proposal_count": len(proposals),
        "matched_count_iou_gte_0_5": len(matches),
        "precision_vs_withheld": round(len(matches) / len(owner), 6) if owner else (1.0 if not proposals else 0.0),
        "recall_vs_withheld": round(len(matches) / len(proposals), 6) if proposals else (1.0 if not owner else 0.0),
        "mean_matched_iou": round(sum(matches) / len(matches), 6) if matches else None,
        "box_count_delta": len(owner) - len(proposals),
    }


def _validate_box(box: Any, *, frame_id: str, width: float, height: float) -> dict[str, Any]:
    if not isinstance(box, dict):
        raise ValueError(f"box must be an object for {frame_id}")
    required = {"x1", "y1", "x2", "y2", "class", "source"}
    if not required.issubset(box):
        raise ValueError(f"box missing required fields for {frame_id}: {sorted(required - set(box))}")
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"box coordinates must be numeric for {frame_id}") from exc
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"box coordinates outside image or non-positive for {frame_id}")
    class_name, source = box["class"], box["source"]
    if class_name not in CLASSES:
        raise ValueError(f"invalid class for {frame_id}: {class_name!r}")
    if source not in SOURCES:
        raise ValueError(f"invalid source for {frame_id}: {source!r}")
    deleted = bool(box.get("deleted", False))
    if deleted != (source == "proposal_deleted"):
        raise ValueError(f"deleted flag/source mismatch for {frame_id}")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": class_name, "source": source, "deleted": deleted}


def validate_export(export: dict[str, Any], manifest: dict[str, Any], *, allow_partial: bool) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if export.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"export schema_version must be {SCHEMA_VERSION}")
    if export.get("session_id") != manifest.get("session_id"):
        raise ValueError("export session_id does not match pack manifest")
    frames = export.get("frames")
    if not isinstance(frames, list):
        raise ValueError("export frames must be an array")
    manifest_frames = manifest.get("frames")
    if not isinstance(manifest_frames, list):
        raise ValueError("pack manifest frames must be an array")
    by_id = {str(frame["frame_id"]): frame for frame in manifest_frames}
    if len(by_id) != len(manifest_frames):
        raise ValueError("pack manifest contains duplicate frame ids")
    seen: set[str] = set()
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    width, height = float(manifest["image"]["width"]), float(manifest["image"]["height"])
    for answer in frames:
        if not isinstance(answer, dict):
            raise ValueError("export frame entry must be an object")
        frame_id = str(answer.get("frame_id", ""))
        if frame_id not in by_id:
            raise ValueError(f"export frame does not join manifest: {frame_id!r}")
        if frame_id in seen:
            raise ValueError(f"duplicate export frame: {frame_id}")
        seen.add(frame_id)
        boxes_raw = answer.get("boxes")
        if not isinstance(boxes_raw, list):
            raise ValueError(f"boxes must be an array for {frame_id}")
        boxes = [_validate_box(box, frame_id=frame_id, width=width, height=height) for box in boxes_raw]
        active = [box for box in boxes if not box["deleted"]]
        empty = answer.get("empty_confirmed")
        if not isinstance(empty, bool):
            raise ValueError(f"empty_confirmed must be boolean for {frame_id}")
        if empty and active:
            raise ValueError(f"empty-confirmed frame carries active boxes: {frame_id}")
        if not empty and not active:
            raise ValueError(f"frame is unanswered: {frame_id}")
        ms_spent = answer.get("ms_spent")
        if not isinstance(ms_spent, (int, float)) or isinstance(ms_spent, bool) or not math.isfinite(float(ms_spent)) or ms_spent < 0:
            raise ValueError(f"ms_spent must be finite and nonnegative for {frame_id}")
        clean = {"frame_id": frame_id, "boxes": boxes, "empty_confirmed": empty, "ms_spent": round(float(ms_spent))}
        validated.append((by_id[frame_id], clean))
    missing = sorted(set(by_id) - seen)
    if missing and not allow_partial:
        raise ValueError(f"partial export refused: {len(missing)} of {len(by_id)} frames missing; pass --allow-partial to override")
    return validated


def build_outputs(export_path: Path, manifest_path: Path, *, allow_partial: bool) -> dict[str, Any]:
    export = json.loads(export_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_export(export, manifest, allow_partial=allow_partial)
    from threed.racketsport.eval_guard import PROTECTED_EVAL_CLIP_IDS, assert_not_training_on_eval_clip

    guard = assert_not_training_on_eval_clip([row[0]["clip_id"] for row in validated])
    if guard["status"] != "clean":
        raise AssertionError("owner person-label dataset unexpectedly intersects protected eval")

    labels: list[dict[str, Any]] = []
    review_events: list[dict[str, Any]] = []
    empties: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    stratum_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    per_session_frames: Counter[str] = Counter()
    scratch_metrics: list[dict[str, Any]] = []
    elapsed_ms = 0
    session_to_split = manifest["split"]["session_to_split"]
    for frame, answer in validated:
        split = session_to_split[frame["session_id"]]
        per_session_frames[frame["session_id"]] += 1
        elapsed_ms += int(answer["ms_spent"])
        active = [box for box in answer["boxes"] if not box["deleted"]]
        for box_index, box in enumerate(answer["boxes"]):
            source_counts[box["source"]] += 1
            stratum_counts[frame["stratum"]][box["source"]] += 1
            event = {
                "frame_id": frame["frame_id"],
                "box_index": box_index,
                "class": box["class"],
                "source": box["source"],
                "deleted": box["deleted"],
                "clip_id": frame["clip_id"],
                "pts_s": frame["timestamp_s"],
                "session_id": frame["session_id"],
                "split": split,
                "reviewer": "owner",
            }
            review_events.append(event)
            if box["deleted"]:
                continue
            class_counts[box["class"]] += 1
            stratum_counts[frame["stratum"]][box["class"]] += 1
            labels.append(
                {
                    "label_id": f"{frame['frame_id']}:{box_index}",
                    "frame_id": frame["frame_id"],
                    "image_file": f"frames/{frame['filename']}",
                    "bbox_xyxy": [box["x1"], box["y1"], box["x2"], box["y2"]],
                    "class": box["class"],
                    "split": split,
                    "provenance": {
                        "clip_id": frame["clip_id"],
                        "video_path": frame["video_path"],
                        "video_sha256": frame["video_sha256"],
                        "pts_s": frame["timestamp_s"],
                        "session_id": frame["session_id"],
                        "reviewer": "owner",
                        "source": box["source"],
                        "pack_manifest_sha256": _sha256(manifest_path),
                        "owner_export_sha256": _sha256(export_path),
                    },
                }
            )
        if answer["empty_confirmed"]:
            empties.append(
                {
                    "frame_id": frame["frame_id"], "image_file": f"frames/{frame['filename']}",
                    "clip_id": frame["clip_id"], "pts_s": frame["timestamp_s"],
                    "session_id": frame["session_id"], "split": split, "reviewer": "owner",
                }
            )
            stratum_counts[frame["stratum"]]["empty_confirmed"] += 1
        if frame["scratch"]:
            scratch_metrics.append({"frame_id": frame["frame_id"], **_match_iou(active, frame["proposals"])})

    selected_sessions = set(per_session_frames)
    train_sessions = {session for session in selected_sessions if session_to_split[session] == "train"}
    val_sessions = {session for session in selected_sessions if session_to_split[session] == "validation"}
    if train_sessions & val_sessions:
        raise AssertionError("session split overlap")
    split_manifest = {
        "schema_version": 1,
        "policy": "whole-session holdout",
        "session_to_split": {session: session_to_split[session] for session in sorted(selected_sessions)},
        "train_sessions": sorted(train_sessions),
        "validation_sessions": sorted(val_sessions),
        "session_disjoint": not bool(train_sessions & val_sessions),
        "frames_by_session": dict(sorted(per_session_frames.items())),
    }
    scratch_summary = {
        "frames": len(scratch_metrics),
        "per_frame": scratch_metrics,
        "mean_owner_box_count": round(sum(row["owner_count"] for row in scratch_metrics) / len(scratch_metrics), 6) if scratch_metrics else None,
        "mean_withheld_proposal_count": round(sum(row["proposal_count"] for row in scratch_metrics) / len(scratch_metrics), 6) if scratch_metrics else None,
        "mean_count_delta": round(sum(row["box_count_delta"] for row in scratch_metrics) / len(scratch_metrics), 6) if scratch_metrics else None,
        "micro_iou_match_precision": round(sum(row["matched_count_iou_gte_0_5"] for row in scratch_metrics) / sum(row["owner_count"] for row in scratch_metrics), 6) if sum(row["owner_count"] for row in scratch_metrics) else None,
        "micro_iou_match_recall": round(sum(row["matched_count_iou_gte_0_5"] for row in scratch_metrics) / sum(row["proposal_count"] for row in scratch_metrics), 6) if sum(row["proposal_count"] for row in scratch_metrics) else None,
    }
    audit = {
        "schema_version": 1,
        "artifact_type": "owner_person_labels_audit",
        "frames_ingested": len(validated),
        "frames_expected": len(manifest["frames"]),
        "partial": len(validated) != len(manifest["frames"]),
        "active_label_count": len(labels),
        "labels_by_class": dict(sorted(class_counts.items())),
        "counts_by_stratum": {key: dict(sorted(value.items())) for key, value in sorted(stratum_counts.items())},
        "proposal_review_sources": dict(sorted(source_counts.items())),
        "empty_confirmed_count": len(empties),
        "review_ms_total": elapsed_ms,
        "scratch_vs_withheld_proposals": scratch_summary,
        "eval_protected_disjointness": {
            "assertion": True,
            "owner_clip_ids_checked": sorted({frame["clip_id"] for frame, _answer in validated}),
            "protected_registry": list(PROTECTED_EVAL_CLIP_IDS),
            "guard": guard,
        },
        "session_disjointness_assertion": split_manifest["session_disjoint"],
        "verified": False,
    }
    dataset_manifest = {
        "schema_version": 1,
        "artifact_type": "owner_reviewed_person_boxes",
        "session_id": manifest["session_id"],
        "reviewer": "owner",
        "ingested_at": manifest.get("created_at", "2026-07-16T00:00:00Z"),
        "files": {"labels": "labels.jsonl", "review_events": "review_events.jsonl", "empty_frames": "empty_frames.jsonl", "audit": "audit.json", "split": "split_manifest.json"},
        "counts": {"frames": len(validated), "active_labels": len(labels), "deleted_proposals": source_counts["proposal_deleted"], "empty_frames": len(empties)},
        "provenance": {"pack_manifest_sha256": _sha256(manifest_path), "owner_export_sha256": _sha256(export_path)},
        "honest_limits": ["Owner-reviewed data-channel output only; no model or product promotion.", "VERIFIED=0 remains binding."],
        "verified": False,
    }
    return {"labels": labels, "review_events": review_events, "empties": empties, "split": split_manifest, "audit": audit, "dataset_manifest": dataset_manifest}


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (temp / "labels.jsonl").write_text(_jsonl(payload["labels"]), encoding="utf-8")
        (temp / "review_events.jsonl").write_text(_jsonl(payload["review_events"]), encoding="utf-8")
        (temp / "empty_frames.jsonl").write_text(_jsonl(payload["empties"]), encoding="utf-8")
        _write_json(temp / "split_manifest.json", payload["split"])
        _write_json(temp / "audit.json", payload["audit"])
        _write_json(temp / "dataset_manifest.json", payload["dataset_manifest"])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        temp.replace(out_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_outputs(args.export.resolve(), args.manifest.resolve(), allow_partial=args.allow_partial)
    if not args.dry_run:
        write_outputs(payload, args.out_dir)
    print(json.dumps({"dry_run": args.dry_run, "would_write": str(args.out_dir), "audit": payload["audit"], "split": payload["split"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
