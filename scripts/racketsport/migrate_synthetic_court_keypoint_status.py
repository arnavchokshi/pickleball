#!/usr/bin/env python3
"""CAL-R2 provenance-fix migration: rewrite an already-generated synthetic court-keypoint
corpus's item status from the old ``reviewed_static_camera_copy`` enum workaround to the
dedicated ``synthetic`` status now accepted by ``train_court_keypoint_heatmap.py``.

Why this exists: the 2,000-image domain-randomized corpus at
``runs/training_corpora_20260701/court_synthetic/`` was generated before the trainer had a
distinct ``synthetic`` item status, so ``scripts/racketsport/generate_synthetic_court_keypoints.py``
labeled every row ``reviewed_static_camera_copy`` -- a status whose whole purpose is to mark
owner-approved COPIES of an independent REAL human review on the same static camera. Loading the
synthetic corpus alongside real external-corpus tiers therefore silently inflated
``labels_static_camera_copy_frame_count`` (and any downstream "human verification" reporting
built on it) by up to 2,000 synthetic rows. This script migrates already-written corpus files in
place (no re-rendering) to the corrected status, matching the generator fix landed in the same
change (``SYNTHETIC_ITEM_STATUS`` in ``generate_synthetic_court_keypoints.py``).

Idempotent: rows already carrying ``status: synthetic`` are left untouched and counted as
"already migrated," so this is safe to re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OLD_STATUS = "reviewed_static_camera_copy"
NEW_STATUS = "synthetic"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_label_file(path: Path) -> str:
    """Rewrite one ``court_keypoints.json`` label file's synthetic item(s) from
    ``reviewed_static_camera_copy`` to ``synthetic`` in place. Returns one of
    ``"migrated"``, ``"already_migrated"``, or ``"skipped_not_synthetic"`` (a defensive guard:
    a row is only ever migrated if its own provenance already self-identifies as synthetic, so
    this can never touch a real owner-approved static-camera-copy row that happens to share a
    root with synthetic data)."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("annotation", {}).get("items", [])
    if not items:
        raise ValueError(f"{path}: no annotation.items to migrate")

    changed = False
    already = True
    for item in items:
        provenance = item.get("provenance", {})
        is_synthetic_row = bool(provenance.get("synthetic"))
        status = item.get("status")
        if status == NEW_STATUS:
            continue
        already = False
        if status != OLD_STATUS or not is_synthetic_row:
            raise ValueError(
                f"{path}: refusing to migrate a non-synthetic-provenance row "
                f"(status={status!r}, provenance.synthetic={provenance.get('synthetic')!r}) -- "
                "this migration only ever touches rows that already self-identify as synthetic."
            )
        item["status"] = NEW_STATUS
        changed = True

    review = payload.get("review")
    if isinstance(review, dict) and changed:
        if review.get("static_camera_copy_count"):
            review["static_camera_copy_count"] = 0
        review.setdefault("synthetic_count", 0)
        if not review["synthetic_count"]:
            review["synthetic_count"] = sum(
                1 for item in items if item.get("status") == NEW_STATUS and bool(item.get("provenance", {}).get("synthetic"))
            )

    if changed:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return "migrated"
    return "already_migrated" if already else "skipped_not_synthetic"


def migrate_synthetic_corpus(root: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    """Migrate every ``<sample>/labels/court_keypoints.json`` under ``root``, then refresh the
    per-sample ``label_sha256`` in ``manifest.json`` (label file content changes, so its hash
    must be recomputed -- leaving the manifest's old hash in place after an in-place content
    edit would itself be a silent integrity bug)."""

    manifest_path = manifest_path or (root / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError(f"{manifest_path}: no samples to migrate")

    counts = {"migrated": 0, "already_migrated": 0, "skipped_not_synthetic": 0}
    sha_updates = 0
    for sample in samples:
        label_path = root / sample["label_path"]
        result = migrate_label_file(label_path)
        counts[result] += 1
        new_sha256 = sha256_file(label_path)
        if sample.get("label_sha256") != new_sha256:
            sample["label_sha256"] = new_sha256
            sha_updates += 1

    if counts["migrated"] or sha_updates:
        manifest_notes = manifest.setdefault("schema_notes", [])
        migration_note = (
            "CAL-R2 provenance-fix migration (scripts/racketsport/migrate_synthetic_court_keypoint_status.py) "
            "rewrote item.status from reviewed_static_camera_copy to synthetic in place; label_sha256 values "
            "were refreshed to match."
        )
        if migration_note not in manifest_notes:
            manifest_notes.append(migration_note)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "label_sha256_updates": sha_updates,
        **counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "runs" / "training_corpora_20260701" / "court_synthetic",
        help="Synthetic court-keypoint corpus root (default: the CAL-R2 2,000-image corpus).",
    )
    args = parser.parse_args(argv)
    try:
        report = migrate_synthetic_corpus(args.root)
    except Exception as exc:  # noqa: BLE001 - surfaced to the CLI caller, not swallowed
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
