#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.validate_pose_dataset import FINE_TUNE_LADDER, validate_manifest


SCHEMA_VERSION = 1
DEFAULT_FINE_TUNE_ORDER = tuple(FINE_TUNE_LADDER)
PLAN_FILENAME = "pose_finetune_plan.json"
EVAL_DATASETS = ("emdb_eval", "caltennis", "athletepose3d")
H100_LEASE = {
    "required_for_real_fine_tune": True,
    "mode": "exclusive_7g_80gb_training",
    "note": "Scaffold does not acquire GPU lease; real BODY-4 fine-tune must serialize through the GPU queue.",
}


def validate_stage_order(order: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    stages = DEFAULT_FINE_TUNE_ORDER if order is None else _normalize_order(order)
    unknown = [stage for stage in stages if stage not in DEFAULT_FINE_TUNE_ORDER]
    if unknown:
        raise ValueError(f"unknown fine-tune stage(s): {', '.join(unknown)}")

    duplicates = _duplicates(stages)
    if duplicates:
        raise ValueError(f"duplicate fine-tune stage(s): {', '.join(duplicates)}")

    missing = [stage for stage in DEFAULT_FINE_TUNE_ORDER if stage not in stages]
    if missing:
        raise ValueError(f"missing fine-tune stage(s): {', '.join(missing)}")

    if tuple(stages) != DEFAULT_FINE_TUNE_ORDER:
        expected = " -> ".join(DEFAULT_FINE_TUNE_ORDER)
        received = " -> ".join(stages)
        raise ValueError(f"fine-tune order must exactly match {expected}; received {received}")
    return tuple(stages)


def build_training_plan(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    order: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    fine_tune_order = validate_stage_order(order)
    manifest = Path(manifest_path)
    coverage = validate_manifest(manifest)
    payload = _read_json(manifest)
    sources_by_type = _sources_by_type(payload["sources"])
    gaps = list(coverage["coverage_summary"]["gaps"])

    status = "blocked_missing_sources" if gaps else "scaffold_only_ready_for_gpu_handoff"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "BODY-4",
        "status": status,
        "scaffold_scope": "CPU-only fine-tune planning scaffold; no training, downloads, checkpoint selection, or manifest mutation.",
        "frozen_scaffold": True,
        "training_enabled": False,
        "downloads_enabled": False,
        "checkpoint_selection_enabled": False,
        "manifest_mutation_enabled": False,
        "manifest": str(manifest),
        "output_dir": str(Path(out_dir)),
        "fine_tune_order": list(fine_tune_order),
        "source_type_counts": coverage["source_type_counts"],
        "coverage_gaps": gaps,
        "h100_lease": H100_LEASE,
        "stages": [_stage_plan(stage, sources_by_type.get(stage, [])) for stage in fine_tune_order],
        "eval_datasets": [{"source_type": source_type, "status": "placeholder_not_measured"} for source_type in EVAL_DATASETS],
    }


def write_training_plan(plan: dict[str, Any], out_dir: str | Path) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / PLAN_FILENAME
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


def _normalize_order(order: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(order, str):
        raw_stages = order.split(",")
    else:
        raw_stages = order
    return tuple(stage.strip().lower() for stage in raw_stages if stage.strip())


def _duplicates(stages: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for stage in stages:
        if stage in seen and stage not in duplicates:
            duplicates.append(stage)
        seen.add(stage)
    return duplicates


def _sources_by_type(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        grouped[source["source_type"]].append(source)
    return grouped


def _stage_plan(source_type: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    source_metadata = [_source_metadata(source) for source in sources]
    return {
        "source_type": source_type,
        "status": "planned_not_started" if sources else "blocked_missing_source",
        "source_count": len(sources),
        "source_ids": [source["id"] for source in sources],
        "source_metadata": source_metadata,
    }


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    fields = ("id", "source_type", "path", "split", "fps", "frame_count", "joint_set", "license", "notes")
    return {field: source[field] for field in fields if field in source}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a CPU-only BODY-4 fine-tune plan scaffold from a pose dataset manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Pose dataset manifest JSON to plan from.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for pose_finetune_plan.json.")
    parser.add_argument(
        "--order",
        default=",".join(DEFAULT_FINE_TUNE_ORDER),
        help="Fine-tune order; must match bedlam2,athletepose3d,caltennis,rich,amass.",
    )
    args = parser.parse_args(argv)

    try:
        plan = build_training_plan(args.manifest, out_dir=args.out, order=args.order)
    except ValueError as exc:
        print("ERROR: BODY-4 fine-tune scaffold planning failed:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 2

    plan_path = write_training_plan(plan, args.out)
    print(json.dumps({"plan": str(plan_path), "status": plan["status"]}, indent=2, sort_keys=True))
    if plan["coverage_gaps"]:
        for gap in plan["coverage_gaps"]:
            print(f"- {gap}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
