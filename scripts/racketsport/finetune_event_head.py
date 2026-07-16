#!/usr/bin/env python3
"""Fine-tune event head on owner-reviewed schema-v2 labels with hard guards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    EventWindowDataset, WindowSpec, rows_jsonl, sha256_file, HIT, BOUNCE,
)
from threed.racketsport.event_head.model import checkpoint_payload, load_checkpoint, masked_cross_entropy

SEED = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"


class FineTuneInputError(ValueError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _contains_bootstrap(value: Any, *, key: str = "") -> bool:
    if isinstance(value, dict):
        for child_key, child in value.items():
            normalized = child_key.lower()
            if normalized in {"tier", "label_tier", "bootstrap_tier"}:
                return True
            if _contains_bootstrap(child, key=normalized):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_bootstrap(item, key=key) for item in value)
    text = str(value).lower()
    return "event_bootstrap_v0" in text or "data/event_bootstrap_20260713" in text


def _protected_anchors(seed_path: Path = SEED) -> dict[str, list[float]]:
    protected: dict[str, list[float]] = defaultdict(list)
    for label in json.loads(seed_path.read_text())["labels"]:
        video_path = Path(label["source"]["video_path"])
        protected[video_path.name].append(float(label["anchor"]["pts_s"]))
        protected[str(video_path.resolve())].append(float(label["anchor"]["pts_s"]))
    return protected


def validate_inputs(reviewed: Path, manifest: Path, *, seed_path: Path = SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not reviewed.is_file():
        raise FineTuneInputError(
            f"reviewed labels are absent: {reviewed}; run ingest_event_review_results.py first", 2
        )
    if not manifest.is_file():
        raise FineTuneInputError(f"dataset manifest is absent: {manifest}", 2)
    manifest_data = json.loads(manifest.read_text())
    if manifest_data.get("schema_version") != 2:
        raise FineTuneInputError("dataset manifest must declare schema_version 2", 20)
    rows = list(rows_jsonl(reviewed))
    if not rows or any(row.get("schema_version", 2) != 2 for row in rows):
        raise FineTuneInputError("reviewed rows must follow schema_version 2", 20)
    ids = [row.get("label_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise FineTuneInputError("missing or duplicate label_ids", 23)
    if _contains_bootstrap(manifest_data) or any(_contains_bootstrap(row) for row in rows):
        raise FineTuneInputError("FORBIDDEN_BOOTSTRAP_PROVENANCE: rejected Tier-A/B label source", 21)
    protected = _protected_anchors(seed_path)
    for row in rows:
        video_path = Path(row.get("video_path", ""))
        anchors = protected.get(video_path.name, []) + protected.get(str(video_path.resolve()), [])
        anchor = row.get("corrected_contact_pts_s")
        if anchor is None:
            anchor = row.get("anchor_pts_s")
        if anchor is not None and any(abs(float(anchor) - item) <= 0.75 for item in anchors):
            raise FineTuneInputError(
                f"PROTECTED_SEED_OVERLAP: {row['label_id']} is within 0.75s of eval seed", 22
            )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("suggested_split")
        if split not in {"train", "val", "test"}:
            raise FineTuneInputError(f"invalid suggested_split for {row['label_id']}: {split}", 20)
        group_splits[str(row.get("source_group"))].add(split)
    leaking = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leaking:
        raise FineTuneInputError(f"SOURCE_SPLIT_LEAKAGE: {leaking}", 24)
    return rows, manifest_data


def _window_specs(rows: list[dict[str, Any]], *, frames: int) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for row in rows:
        if row["suggested_split"] != "train" or row["decision"] == "other":
            continue
        video = Path(row["video_path"])
        if not video.is_file():
            raise FineTuneInputError(f"reviewed video is absent: {video}", 25)
        capture = cv2.VideoCapture(str(video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        timestamp = row.get("corrected_contact_pts_s")
        if timestamp is None:
            timestamp = row["anchor_pts_s"]
        center = min(max(0, round(float(timestamp) * fps)), max(0, total - 1))
        start = min(max(0, center - frames // 2), max(0, total - frames))
        events: tuple[tuple[int, int], ...] = ()
        if row["decision"] in {"paddle", "ground"}:
            class_id = HIT if row["decision"] == "paddle" else BOUNCE
            events = ((center - start, class_id),)
        specs.append(WindowSpec(
            video_path=video, start_frame=start, num_frames=frames, fps=fps,
            events=events, validity_mask=(True, True, True), source="owner_reviewed_v2",
            license_posture="OWNER_REVIEWED_INTERNAL",
        ))
    if not specs:
        raise FineTuneInputError("no trainable contact or hard-negative rows", 25)
    return specs


def run_finetune(
    *, reviewed: Path, manifest: Path, pretrain: Path, out: Path,
    steps: int, image_size: int, window_frames: int,
) -> dict[str, Any]:
    rows, manifest_data = validate_inputs(reviewed, manifest)
    if not pretrain.is_file():
        raise FineTuneInputError(f"pretrain checkpoint is absent: {pretrain}", 2)
    model, pretrain_payload = load_checkpoint(pretrain)
    dataset = EventWindowDataset(_window_specs(rows, frames=window_frames), image_size=image_size)
    samples = [dataset[index] for index in range(len(dataset))]
    frames = torch.stack([sample["frames"] for sample in samples])
    targets = torch.stack([sample["targets"] for sample in samples])
    masks = torch.stack([sample["validity_mask"] for sample in samples])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(frames), targets, masks)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("fine-tune produced non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "event_head_finetuned.pt"
    input_provenance = {
        "reviewed_sha256": sha256_file(reviewed), "manifest_sha256": sha256_file(manifest),
        "pretrain_sha256": sha256_file(pretrain),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    torch.save(checkpoint_payload(
        model, license_posture=pretrain_payload.get("license_posture", "RD_ONLY"),
        license_reason=pretrain_payload.get("license_reason", "inherits pretrain posture"),
        fine_tune_provenance=input_provenance,
    ), checkpoint)
    counts = Counter(row["decision"] for row in rows)
    result = {
        "schema_version": 1, "artifact_type": "event_head_finetune_manifest",
        "verified": False, "reviewed_schema_version": manifest_data["schema_version"],
        "config": {"steps": steps, "image_size": image_size, "window_frames": window_frames,
                   "classes": ["background", "HIT", "BOUNCE"]},
        "decision_counts": dict(sorted(counts.items())), "other_excluded_from_typed": counts["other"],
        "hard_negative_count": counts["none"] + counts["unclear"], "losses": losses,
        "checkpoint": str(checkpoint), "provenance": input_provenance,
        "license_posture": pretrain_payload.get("license_posture", "RD_ONLY"),
    }
    (out / "finetune_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-frames", type=int, default=9)
    args = parser.parse_args()
    try:
        result = run_finetune(**vars(args))
    except FineTuneInputError as exc:
        parser.exit(exc.exit_code, f"fine-tune input rejected: {exc}\n")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(30, f"fine-tune failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
