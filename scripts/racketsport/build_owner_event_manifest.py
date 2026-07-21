#!/usr/bin/env python3
"""Build the frozen 102-row owner event-head manifest from raw answers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    CURRENT_MANIFEST_CLASSES,
    sha256_file,
    validate_current_manifest,
)


DEFAULT_RESULTS = ROOT / "data/event_labels_owner_20260719/results_batch1_102rows.json"
DEFAULT_SESSION = ROOT / "runs/lanes/owner_event_labels_20260715/session_manifest.json"
DEFAULT_PROTECTED = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
DEFAULT_OUT = ROOT / "runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json"
WINDOW_FRAMES = 64
EXPECTED_ROWS = 102
EXPECTED_TRAIN_ROWS = 61
EXPECTED_VAL_ROWS = 41
EXPECTED_TRAIN_GROUPS = {"73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY"}
EXPECTED_VAL_GROUPS = {"HyUqT7zFiwk", "zwCtH_i1_S4"}
DECISION_TO_CLASS = {"paddle": "HIT", "ground": "BOUNCE"}


class OwnerManifestBuildError(ValueError):
    """Raised when the frozen owner inputs cannot produce the declared manifest."""


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise OwnerManifestBuildError(f"{role} is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OwnerManifestBuildError(f"{role} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OwnerManifestBuildError(f"{role} must be a JSON object")
    return payload


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _finite(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OwnerManifestBuildError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise OwnerManifestBuildError(f"{field} must be finite")
    return parsed


def _video_metadata(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OwnerManifestBuildError(f"could not open owner media: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frames < WINDOW_FRAMES:
        raise OwnerManifestBuildError(
            f"owner media lacks {WINDOW_FRAMES} decodable frames: {path}"
        )
    return fps, frames


def _protected_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != 50:
        raise OwnerManifestBuildError("protected inventory must contain exactly 50 labels")
    entries: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        try:
            source = label["source"]
            frame = label["anchor"]["frame"]
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise OwnerManifestBuildError(
                    f"protected label {index} anchor.frame must be a nonnegative integer"
                )
            entries.append({
                "video_sha256": str(source["video_sha256"]),
                "clip_id": str(source["clip_id"]),
                "video_path": str(_repo_path(str(source["video_path"])).resolve()),
                "frame": frame,
            })
        except (KeyError, TypeError) as exc:
            raise OwnerManifestBuildError(
                f"protected label {index} lacks content identity or anchor time"
            ) from exc
    return entries


def _same_protected_media(row: dict[str, Any], seed: dict[str, Any]) -> bool:
    resolved = str(_repo_path(str(row["video_path"])).resolve())
    return (
        row["video_sha256"] == seed["video_sha256"]
        or row["clip_id"] == seed["clip_id"]
        or resolved == seed["video_path"]
    )


def _safe_training_window_start(
    *,
    initial_start: int,
    center_frame: int,
    total_frames: int,
    media_identity: dict[str, Any],
    protected: list[dict[str, Any]],
) -> int:
    protected_frames = {
        int(seed["frame"])
        for seed in protected
        if _same_protected_media(media_identity, seed)
    }
    if not protected_frames:
        return initial_start

    minimum_start = max(0, center_frame - WINDOW_FRAMES + 1)
    maximum_start = min(center_frame, total_frames - WINDOW_FRAMES)
    candidates = [
        start
        for start in range(minimum_start, maximum_start + 1)
        if not any(start <= frame < start + WINDOW_FRAMES for frame in protected_frames)
    ]
    if not candidates:
        raise OwnerManifestBuildError(
            "PROTECTED_SEED_WINDOW_EXHAUSTED: no 64-frame training window can "
            f"retain owner event {media_identity['label_id']} without protected overlap"
        )
    return min(candidates, key=lambda start: (abs(start - initial_start), start))


def _assert_zero_protected_overlap(
    rows: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> dict[str, Any]:
    overlaps: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "train":
            continue
        start = int(row["source_start_frame"])
        end = start + int(row["num_frames"])
        for seed in protected:
            if not _same_protected_media(row, seed):
                continue
            protected_frame = int(seed["frame"])
            if start <= protected_frame < end:
                overlaps.append({
                    "label_id": row["label_id"],
                    "clip_id": row["clip_id"],
                    "training_window": [start, end],
                    "protected_frame": protected_frame,
                })
    if overlaps:
        raise OwnerManifestBuildError(f"PROTECTED_SEED_WINDOW_OVERLAP: {overlaps}")
    return {
        "status": "pass",
        "overlap_rows": 0,
        "policy": "no_protected_frame_in_half_open_training_window",
        "window_frames": WINDOW_FRAMES,
        "interval_semantics": "[source_start_frame, source_start_frame + num_frames)",
        "checked_training_windows": sum(row["split"] == "train" for row in rows),
        "identity_fields": ["video_sha256", "clip_id", "resolved_video_path"],
    }


def build_owner_manifest(
    results_path: Path = DEFAULT_RESULTS,
    session_path: Path = DEFAULT_SESSION,
    protected_path: Path = DEFAULT_PROTECTED,
) -> dict[str, Any]:
    results = _read_json(results_path, role="owner results")
    session = _read_json(session_path, role="owner session manifest")
    protected_payload = _read_json(protected_path, role="protected inventory")
    protected = _protected_entries(protected_payload)
    if results.get("results_schema_version") != 2:
        raise OwnerManifestBuildError("owner results_schema_version must be 2")
    if results.get("session_id") != session.get("session_id"):
        raise OwnerManifestBuildError("owner results session_id does not match session manifest")
    answers = results.get("answers")
    session_rows = session.get("rows")
    if not isinstance(answers, dict) or len(answers) != EXPECTED_ROWS:
        raise OwnerManifestBuildError(f"owner results must contain exactly {EXPECTED_ROWS} answers")
    if not isinstance(session_rows, list):
        raise OwnerManifestBuildError("session manifest rows must be an array")
    rows_by_label: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        label_id = str(row.get("label_id", ""))
        if not label_id or label_id in rows_by_label:
            raise OwnerManifestBuildError("session manifest has missing or duplicate label_id")
        rows_by_label[label_id] = row
    source_split = session.get("source_split", {})
    if set(source_split.get("train_groups", [])) != EXPECTED_TRAIN_GROUPS:
        raise OwnerManifestBuildError("session manifest train source groups changed")
    if set(source_split.get("validation_groups", [])) != EXPECTED_VAL_GROUPS:
        raise OwnerManifestBuildError("session manifest validation source groups changed")

    metadata_cache: dict[str, tuple[float, int]] = {}
    hash_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    answer_field_counts: Counter[str] = Counter()
    split_decisions: dict[str, Counter[str]] = defaultdict(Counter)
    protected_adjusted_windows = 0
    for answer_key in sorted(answers, key=lambda value: int(value)):
        answer = answers[answer_key]
        if not isinstance(answer, dict):
            raise OwnerManifestBuildError(f"answer {answer_key} must be an object")
        label_id = str(answer.get("label_id", ""))
        session_row = rows_by_label.get(label_id)
        if session_row is None or int(session_row["row"]) != int(answer_key):
            raise OwnerManifestBuildError(f"answer {answer_key} does not join the frozen session")
        decision = str(answer.get("decision", ""))
        if decision not in {"paddle", "ground", "other", "none"}:
            raise OwnerManifestBuildError(f"answer {answer_key} has invalid decision {decision!r}")
        decision_counts[decision] += 1
        typed = decision != "none"
        if typed:
            missing = [field for field in ("x", "y", "dt") if field not in answer]
            if missing:
                raise OwnerManifestBuildError(
                    f"typed answer {answer_key} is missing raw answer fields: {missing}"
                )
            x = _finite(answer["x"], field=f"answers[{answer_key}].x")
            y = _finite(answer["y"], field=f"answers[{answer_key}].y")
            dt = _finite(answer["dt"], field=f"answers[{answer_key}].dt")
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise OwnerManifestBuildError(f"answer {answer_key} coordinates are not normalized")
            answer_field_counts.update(("coords", "dt"))
        else:
            x = y = dt = None

        declared_split = str(session_row["suggested_split"])
        split = "val" if declared_split == "validation" else declared_split
        source_group = str(session_row["source_group"])
        expected_split = "train" if source_group in EXPECTED_TRAIN_GROUPS else "val"
        if split != expected_split:
            raise OwnerManifestBuildError(
                f"source-held split changed for {label_id}: {split} != {expected_split}"
            )
        split_decisions[split][decision] += 1

        video_path = str(session_row["video_path"])
        media_path = _repo_path(video_path)
        if not media_path.is_file():
            raise OwnerManifestBuildError(f"owner media is absent: {video_path}")
        if video_path not in metadata_cache:
            metadata_cache[video_path] = _video_metadata(media_path)
            hash_cache[video_path] = sha256_file(media_path)
        decoded_fps, total_frames = metadata_cache[video_path]
        fps = _finite(session_row["source_fps"], field=f"session row {answer_key} source_fps")
        if not math.isclose(decoded_fps, fps, rel_tol=1e-4, abs_tol=1e-3):
            raise OwnerManifestBuildError(
                f"session/video fps mismatch for {label_id}: {fps} != {decoded_fps}"
            )
        declared_sha = str(session_row["video_sha256"])
        if hash_cache[video_path] != declared_sha:
            raise OwnerManifestBuildError(f"session/video SHA-256 mismatch for {label_id}")

        source_anchor = _finite(
            session_row["anchor_pts_s"], field=f"session row {answer_key} anchor_pts_s"
        )
        review_anchor = source_anchor + (dt or 0.0)
        center_frame = min(max(0, round(review_anchor * fps)), total_frames - 1)
        initial_start_frame = min(
            max(0, center_frame - WINDOW_FRAMES // 2), total_frames - WINDOW_FRAMES
        )
        media_identity = {
            "label_id": label_id,
            "clip_id": str(session_row["clip_id"]),
            "video_path": video_path,
            "video_sha256": declared_sha,
        }
        start_frame = (
            _safe_training_window_start(
                initial_start=initial_start_frame,
                center_frame=center_frame,
                total_frames=total_frames,
                media_identity=media_identity,
                protected=protected,
            )
            if split == "train"
            else initial_start_frame
        )
        protected_adjusted_windows += int(start_frame != initial_start_frame)
        event_class = DECISION_TO_CLASS.get(decision)
        events = (
            [{"frame": center_frame - start_frame, "class": event_class}]
            if event_class else []
        )
        review = {"decision": decision}
        if typed:
            review.update({"x": x, "y": y, "dt": dt})
        rows.append({
            "source": "owner_reviewed",
            "video": label_id,
            "source_video": source_group,
            "video_path": video_path,
            "video_sha256": declared_sha,
            "media_present": True,
            "split": split,
            "fps": fps,
            "source_start_frame": start_frame,
            "num_frames": WINDOW_FRAMES,
            "event_counts": {
                "HIT": int(event_class == "HIT"),
                "BOUNCE": int(event_class == "BOUNCE"),
                "background": int(event_class is None),
            },
            "inventory_event_count": len(events),
            "events": events,
            "loss_validity_mask": [True, True, True],
            "license_id": "OWNER_REVIEWED_HARVEST_INTERNAL",
            "license_posture": "OWNER_REVIEWED_INTERNAL",
            "label_id": label_id,
            "clip_id": str(session_row["clip_id"]),
            "stratum": str(session_row["stratum"]),
            "review_anchor_pts_s": round(review_anchor, 9),
            "source_anchor_pts_s": source_anchor,
            "review": review,
            "target_mapping": (
                event_class if event_class else
                "out_of_taxonomy_background" if decision == "other" else
                "hard_negative_background"
            ),
        })

    if len(rows) != EXPECTED_ROWS:
        raise OwnerManifestBuildError(f"built {len(rows)} rows, expected {EXPECTED_ROWS}")
    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != Counter({"train": EXPECTED_TRAIN_ROWS, "val": EXPECTED_VAL_ROWS}):
        raise OwnerManifestBuildError(f"frozen owner split changed: {dict(split_counts)}")
    expected_decisions = Counter({"paddle": 38, "ground": 21, "other": 1, "none": 42})
    if decision_counts != expected_decisions:
        raise OwnerManifestBuildError(f"owner answer decisions changed: {dict(decision_counts)}")
    if answer_field_counts != Counter({"coords": 60, "dt": 60}):
        raise OwnerManifestBuildError(
            f"raw typed-answer bookkeeping changed: {dict(answer_field_counts)}"
        )

    protected_check = _assert_zero_protected_overlap(rows, protected)
    protected_check["adjusted_training_windows"] = protected_adjusted_windows
    manifest = {
        "schema_version": 1,
        "artifact_type": "event_head_owner_reviewed_dataset_manifest",
        "verified": False,
        "teacher_derived": False,
        "ground_truth": True,
        "session_id": str(results["session_id"]),
        "config": {
            "window_frames": WINDOW_FRAMES,
            "split_unit": "original_source_video_id",
            "train_source_groups": sorted(EXPECTED_TRAIN_GROUPS),
            "validation_source_groups": sorted(EXPECTED_VAL_GROUPS),
            "protected_exclusion_policy": (
                "no_protected_frame_in_half_open_training_window"
            ),
        },
        "classes": CURRENT_MANIFEST_CLASSES,
        "image_size": 224,
        "decode_policy": "on_the_fly_no_frame_cache",
        "license_posture": "OWNER_REVIEWED_INTERNAL",
        "inputs": {
            "results": str(results_path.relative_to(ROOT)),
            "results_sha256": sha256_file(results_path),
            "session_manifest": str(session_path.relative_to(ROOT)),
            "session_manifest_sha256": sha256_file(session_path),
            "protected_inventory_sha256": sha256_file(protected_path),
        },
        "totals": {
            "rows": EXPECTED_ROWS,
            "train_rows": split_counts["train"],
            "val_rows": split_counts["val"],
            "target_events": decision_counts["paddle"] + decision_counts["ground"],
            "HIT": decision_counts["paddle"],
            "BOUNCE": decision_counts["ground"],
            "hard_negative_answers": decision_counts["none"],
            "out_of_taxonomy_answers": decision_counts["other"],
            "typed_answers": sum(decision_counts[name] for name in ("paddle", "ground", "other")),
            "answers_with_coordinates": answer_field_counts["coords"],
            "answers_with_dt": answer_field_counts["dt"],
            "legacy_provenance_counts": {
                "coordinates": 46,
                "dt": 57,
                "status": "stale_summary_not_used; raw answers contain x/y/dt on all 60 typed answers",
            },
            "decision_counts": dict(sorted(decision_counts.items())),
            "split_decision_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(split_decisions.items())
            },
        },
        "protected_seed_check": protected_check,
        "rows": rows,
    }
    validate_current_manifest(manifest)
    return manifest


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--protected-inventory", type=Path, default=DEFAULT_PROTECTED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        manifest = build_owner_manifest(
            args.results, args.session_manifest, args.protected_inventory
        )
        payload = _json_bytes(manifest)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
    except (OwnerManifestBuildError, OSError, ValueError) as exc:
        parser.exit(2, f"owner event manifest build rejected: {exc}\n")
    print(json.dumps({
        "out": str(args.out),
        "sha256": sha256_file(args.out),
        "rows": len(manifest["rows"]),
        "train_rows": manifest["totals"]["train_rows"],
        "val_rows": manifest["totals"]["val_rows"],
        "protected_overlap_rows": manifest["protected_seed_check"]["overlap_rows"],
        "verified": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
