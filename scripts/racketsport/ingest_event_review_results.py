#!/usr/bin/env python3
"""Validate and ingest owner event-review results into reviewed labels v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
DECISIONS = {"paddle", "ground", "other", "none", "unclear"}
CONTACT_DECISIONS = {"paddle", "ground", "other"}
NONCONTACT_DECISIONS = {"none", "unclear"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolution(path: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _answers(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    if payload.get("results_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"results_schema_version must be {SCHEMA_VERSION}")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("answers must be an object keyed by presentation row")
    parsed: list[tuple[int, dict[str, Any]]] = []
    for key, value in answers.items():
        try:
            row = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"answer key is not an integer row: {key!r}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"answer {key} must be an object")
        parsed.append((row, value))
    return sorted(parsed)


def ingest_results(results_path: Path, manifest_path: Path, out_dir: Path, *, root: Path | None = None) -> dict[str, Any]:
    results_path = results_path.resolve()
    manifest_path = manifest_path.resolve()
    root = (root or Path.cwd()).resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if results.get("session_id") != manifest.get("session_id"):
        raise ValueError("results session_id does not match manifest")

    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manifest rows must be an array")
    by_label: dict[str, dict[str, Any]] = {}
    by_row: dict[int, dict[str, Any]] = {}
    for row in rows:
        label_id = str(row["label_id"])
        presentation_row = int(row["row"])
        if label_id in by_label or presentation_row in by_row:
            raise ValueError("manifest contains duplicate label_id or row")
        by_label[label_id] = row
        by_row[presentation_row] = row

    seen_labels: set[str] = set()
    seen_rows: set[int] = set()
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for presentation_row, answer in _answers(results):
        label_id = answer.get("label_id")
        if not isinstance(label_id, str) or label_id not in by_label:
            raise ValueError(f"label_id does not join manifest: {label_id!r}")
        if presentation_row not in by_row or by_row[presentation_row]["label_id"] != label_id:
            raise ValueError(f"row/label_id join mismatch at row {presentation_row}")
        if label_id in seen_labels or presentation_row in seen_rows:
            raise ValueError(f"duplicate answer for {label_id}")
        seen_labels.add(label_id)
        seen_rows.add(presentation_row)

        decision = answer.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"invalid decision for {label_id}: {decision!r}")
        coordinate_keys = {"x", "y", "dt"}
        present = coordinate_keys & set(answer)
        if decision in CONTACT_DECISIONS:
            if present != coordinate_keys:
                raise ValueError(f"contact decision {label_id} requires x, y, and dt")
            try:
                x, y, dt = float(answer["x"]), float(answer["y"]), float(answer["dt"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"contact coordinates must be numeric for {label_id}") from exc
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"normalized contact coordinates outside [0,1] for {label_id}")
            if abs(dt) > 0.65:
                raise ValueError(f"|dt| exceeds 0.65s for {label_id}")
        elif present:
            raise ValueError(f"{decision} answer {label_id} must not carry x/y/dt")
        validated.append((by_label[label_id], answer))

    manifest_sha = _sha256(manifest_path)
    results_sha = _sha256(results_path)
    ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    resolution_cache: dict[str, tuple[int, int]] = {}
    output_rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for manifest_row, answer in sorted(validated, key=lambda item: int(item[0]["row"])):
        decision = str(answer["decision"])
        contact = None
        dt_s = None
        corrected = None
        if decision in CONTACT_DECISIONS:
            video_path = str(manifest_row["video_path"])
            if video_path not in resolution_cache:
                resolution_cache[video_path] = _resolution(root / video_path)
            width, height = resolution_cache[video_path]
            x, y, dt_s = float(answer["x"]), float(answer["y"]), float(answer["dt"])
            contact = {
                "x_norm": x,
                "y_norm": y,
                "x_px": round(x * width, 4),
                "y_px": round(y * height, 4),
                "source_width": width,
                "source_height": height,
            }
            corrected = round(float(manifest_row["anchor_pts_s"]) + dt_s, 9)
        output_rows.append(
            {
                "label_id": manifest_row["label_id"],
                "clip_id": manifest_row["clip_id"],
                "source_group": manifest_row["source_group"],
                "video_path": manifest_row["video_path"],
                "video_sha256": manifest_row["video_sha256"],
                "anchor_pts_s": manifest_row["anchor_pts_s"],
                "stratum": manifest_row["stratum"],
                "score_band": manifest_row.get("score_band"),
                "decision": decision,
                "contact": contact,
                "dt_s": dt_s,
                "corrected_contact_pts_s": corrected,
                "suggested_split": manifest_row["suggested_split"],
                "review": {
                    "session_id": manifest["session_id"],
                    "reviewed_by": "owner",
                    "ingested_at": ingested_at,
                },
                "provenance": {
                    "seed": manifest["seed"],
                    "generator_version": manifest["generator_version"],
                    "generator_sha256": manifest["generator_sha256"],
                    "manifest_sha256": manifest_sha,
                    "results_sha256": results_sha,
                },
            }
        )
        counts[str(manifest_row["stratum"])][decision] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "reviewed_labels_v2.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    unanswered = [
        {"row": int(row["row"]), "label_id": row["label_id"]}
        for row in sorted(rows, key=lambda item: int(item["row"]))
        if row["label_id"] not in seen_labels
    ]
    dataset_manifest = {
        "dataset_schema_version": SCHEMA_VERSION,
        "artifact_type": "owner_reviewed_event_labels",
        "session_id": manifest["session_id"],
        "answered_count": len(output_rows),
        "unanswered_count": len(unanswered),
        "unanswered": unanswered,
        "counts_by_stratum_and_decision": {
            stratum: dict(sorted(decisions.items())) for stratum, decisions in sorted(counts.items())
        },
        "files": {"labels": labels_path.name},
        "provenance": {"manifest_sha256": manifest_sha, "results_sha256": results_sha},
        "honest_limits": [
            "These are owner-reviewed bootstrap-era labels from the 2026-07-15 review channel; they are not a VERIFIED promotion.",
            "Suggested splits preserve original-source disjointness but do not create an independent promotion test by themselves.",
            "VERIFIED=0 remains binding until the named source-disjoint event gates pass.",
        ],
        "verified": False,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used to resolve source videos")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = ingest_results(args.results, args.manifest, args.out_dir, root=args.root)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
