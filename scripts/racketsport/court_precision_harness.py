#!/usr/bin/env python3
"""Score existing court-calibration runs with GT-free diagnostic metrics M1-M5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_precision_metrics import (  # noqa: E402
    SCORER_VERSION,
    SCORER_VERSION_POLICY,
    render_m1_overlay,
    score_court_precision_run,
    source_provenance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="Existing run directory; repeat for baselines.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics, table, and overlays.")
    parser.add_argument("--sample-count", type=int, default=12, help="Evenly spaced source frames scored for M1 (default: 12).")
    parser.add_argument("--search-window-px", type=int, default=14, help="Bounded perpendicular M1 search radius (default: 14).")
    parser.add_argument("--overlay-count", type=int, default=4, help="Number of global worst-M1 PNGs to render (default: 4).")
    parser.add_argument("--pb-export", type=Path, help="PB Vision cv_export.json; applied only to Wolverine for M4.")
    parser.add_argument(
        "--net-evidence",
        action="append",
        default=[],
        metavar="CLIP=PATH",
        help="Optional existing net solver/evidence artifact for a clip; repeat as needed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.sample_count <= 0:
            raise ValueError("--sample-count must be positive")
        if args.search_window_px <= 0:
            raise ValueError("--search-window-px must be positive")
        if args.overlay_count < 0:
            raise ValueError("--overlay-count cannot be negative")
        net_evidence = _parse_net_evidence(args.net_evidence)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        clips: list[dict[str, Any]] = []
        for run_dir in args.run_dir:
            clip_hint = run_dir.name
            evidence_path = _matching_evidence(net_evidence, clip_hint, run_dir)
            result = score_court_precision_run(
                run_dir,
                sample_count=args.sample_count,
                search_window_px=args.search_window_px,
                pb_export_path=args.pb_export,
                net_evidence_path=evidence_path,
            )
            clip_dir = args.out_dir / _safe_name(result["clip"])
            clip_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = clip_dir / "court_precision_metrics_v2.json"
            _write_json(metrics_path, result)
            result["artifact_path"] = str(metrics_path)
            clips.append(result)

        overlays = _write_worst_overlays(clips, args.out_dir / "overlays_v2", args.overlay_count)
        table = {
            "schema_version": 1,
            "artifact_type": "court_precision_baseline_table",
            "scorer_version": SCORER_VERSION,
            "diagnostic_only": True,
            "promotion_gate": False,
            "best_stack_delta": "none",
            "policy": {
                "ground_truth_free": True,
                "input_run_calibration_overwritten": False,
                "diagnostic_observation_bootstrap_refits": True,
                "protected_outdoor_indoor_labels_read": False,
                "metrics_are_diagnostics_never_promotion_gates": True,
                "scorer_version_bump": SCORER_VERSION_POLICY,
            },
            "freeze_contract": {
                "scorer_version": SCORER_VERSION,
                "scorer_version_policy": SCORER_VERSION_POLICY,
                "per_clip_contracts": [clip["freeze_contract"] for clip in clips],
            },
            "provenance": {
                "imported_current_worktree_state": source_provenance(
                    [
                        "threed/racketsport/court_precision_metrics.py",
                        "threed/racketsport/court_calibration.py",
                        "threed/racketsport/court_templates.py",
                    ]
                ),
                "foreign_read_only_not_imported": source_provenance(
                    ["threed/racketsport/court_calibration_metric15.py"]
                ),
                "candidate_evidence_api_imported": False,
                "pbvision_protocol_reference_read_not_imported": (
                    "runs/research_pbv_reveng_20260712/compare_vs_pbvision.py" if args.pb_export else None
                ),
            },
            "clip_count": len(clips),
            "clips": clips,
            "overlays": overlays,
        }
        table_path = args.out_dir / "baseline_table_v2.json"
        _write_json(table_path, table)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"baseline_table": str(table_path), "clip_count": len(clips), "overlay_count": len(overlays)}, sort_keys=True))
    return 0


def _write_worst_overlays(clips: list[dict[str, Any]], out_dir: Path, count: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for clip in clips:
        m1 = clip["metrics"]["M1"]
        if m1.get("status") != "present":
            continue
        for row in m1.get("per_frame", []):
            residual = row.get("residual_px")
            if row.get("status") == "present" and isinstance(residual, dict) and residual.get("median") is not None:
                candidates.append((float(residual["median"]), clip, row))
    candidates.sort(key=lambda item: (item[0], -float(item[2].get("evidence_coverage_fraction", 0.0))), reverse=True)
    selected: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    if count > 0:
        for clip in clips:
            clip_candidates = [item for item in candidates if item[1] is clip]
            if clip_candidates:
                selected.append(clip_candidates[0])
    selected_keys = {(str(item[1]["run_dir"]), int(item[2]["frame_index"])) for item in selected}
    for item in candidates:
        key = (str(item[1]["run_dir"]), int(item[2]["frame_index"]))
        if len(selected) >= count:
            break
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)
    selected.sort(key=lambda item: (item[0], -float(item[2].get("evidence_coverage_fraction", 0.0))), reverse=True)
    overlays: list[dict[str, Any]] = []
    for _residual, clip, row in selected[:count]:
        frame_index = int(row["frame_index"])
        out_path = out_dir / f"{_safe_name(clip['clip'])}_frame_{frame_index:06d}.png"
        calibration = json.loads(Path(clip["inputs"]["calibration"]).read_text(encoding="utf-8"))
        render_m1_overlay(
            Path(clip["inputs"]["video"]),
            calibration,
            frame_index,
            out_path,
            title=str(clip["clip"]),
        )
        overlays.append(
            {
                "path": str(out_path),
                "clip": clip["clip"],
                "frame_index": frame_index,
                "reported_m1_median_residual_px": row["residual_px"]["median"],
                "reported_m1_evidence_coverage_fraction": row["evidence_coverage_fraction"],
                "selection": "global_worst_m1_frame",
            }
        )
    return overlays


def _parse_net_evidence(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--net-evidence must be CLIP=PATH, got: {value}")
        clip, raw_path = value.split("=", 1)
        if not clip or not raw_path:
            raise ValueError(f"--net-evidence must be CLIP=PATH, got: {value}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"net evidence path does not exist: {path}")
        result[clip.lower()] = path
    return result


def _matching_evidence(evidence: dict[str, Path], clip_hint: str, run_dir: Path) -> Path | None:
    haystack = f"{clip_hint} {run_dir}".lower()
    matches = [path for key, path in evidence.items() if key in haystack or key in clip_hint.lower()]
    if len(matches) > 1:
        raise ValueError(f"multiple --net-evidence entries match {run_dir}")
    return matches[0] if matches else None


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
