#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path("scripts/racketsport")
TEST_ROOT = Path("tests/racketsport")
SCHEMA_ROOT = Path("docs/racketsport")
INDEXED_GLOBS = ("scripts/racketsport/*.py", "scripts/racketsport/*.sh", "scripts/*.py", "scripts/*.sh")

PREFIXES = (
    "aggregate_",
    "audit_",
    "benchmark_",
    "build_",
    "check_",
    "extract_",
    "init_",
    "materialize_",
    "register_",
    "render_",
    "report_",
    "smoke_",
    "summarize_",
    "validate_",
)

RELATED_TEST_OVERRIDES = {
    "audit_label_drafts": "test_label_draft_audit.py",
    "benchmark_person_trackers": "test_person_tracking_benchmark.py",
    "benchmark_decode": "test_decode_benchmark_summary.py",
    "benchmark_sam3dbody": "test_benchmark_sam3dbody.py",
    "build_contact_windows_from_review_inputs": "test_contact_window_review.py",
    "build_eval0_index": "test_eval0_index.py",
    "build_paddle_true_corner_review": "test_racket_candidate_generation.py",
    "build_report_artifacts": "test_report_artifacts.py",
    "build_serving_manifest": "test_serving_manifest.py",
    "build_variant_comparison": "test_variant_comparison.py",
    "calibrate": "test_court_calibration.py",
    "check_eval_regression": "test_eval_regression.py",
    "build_corrections_queue": "test_corrections.py",
    "export_ball_click_review": "test_ball_click_review.py",
    "export_cvat_tasks": "test_label_review_flow.py",
    "export_review_frames": "test_label_review_flow.py",
    "extract_label_frames": "test_label_workdir.py",
    "filter_ball_local_search": "test_ball_local_search.py",
    "finetune_pose": "test_finetune_pose_scaffold.py",
    "ingest_testclips": "test_io_decode.py",
    "init_label_workdir": "test_label_workdir.py",
    "manifest_report": "test_manifest_report.py",
    "materialize_body_frames": "test_body_frame_materialization.py",
    "materialize_seed_manifest": "test_seed_manifest_materializer.py",
    "prepare_tracknetv3_finetune_dataset": "test_pretraining_dataset_prep.py",
    "render_calibration_overlay": "test_calibration_overlay.py",
    "report_testclip_coverage": "test_testclip_coverage_report.py",
    "run_ball_tracking_eval_suite": "test_ball_tracking_eval_suite.py",
    "run_mobile_person_accuracy_sweep": "test_mobile_person_accuracy_sweep.py",
    "run_mobile_person_yolo_replay": "test_mobile_person_yolo_replay.py",
    "run_totnet_ball": "test_totnet_runner_runtime.py",
    "run_yolo26_teacher": "test_yolo26_teacher_filters.py",
    "smoke_models": "test_smoke_models.py",
    "summarize_decode_benchmarks": "test_decode_benchmark_summary.py",
    "track": "test_track_cli.py",
    "train_court_keypoint_heatmap": "test_pretraining_dataset_prep.py",
    "train_tenniset_shot_baseline": "test_tenniset_shot_baseline.py",
    "validate_ball_audio_dataset": "test_ball_audio_dataset.py",
    "validate_corrections": "test_corrections.py",
    "validate_pipeline_artifacts": "test_pipeline_contracts.py",
    "validate_pose_dataset": "test_pose_dataset.py",
    "validate_racket_dataset": "test_racket_dataset.py",
    "validate_shot_dataset": "test_shot_dataset.py",
    "validate_testclips": "test_testclips.py",
}

SCHEMA_OVERRIDES = {
    "audit_label_drafts": "label_draft_audit_schema.json",
    "build_eval0_index": "eval0_index_schema.json",
    "build_report_artifacts": "report_artifacts_schema.json",
    "build_serving_manifest": "serving_manifest_schema.json",
    "build_variant_comparison": "eval0_index_schema.json",
    "list_scaffold_tools": "scaffold_tool_index_schema.json",
    "summarize_eval_runs": "eval_summary_schema.json",
    "validate_ball_audio_dataset": "ball_audio_dataset_schema.json",
    "validate_pipeline_artifacts": "pipeline_contracts_schema.json",
    "validate_pose_dataset": "pose_dataset_schema.json",
    "validate_racket_dataset": "racket_dataset_schema.json",
    "validate_shot_dataset": "shot_dataset_schema.json",
}

TASK_HINTS = {
    "aggregate_roboflow_corpus": ("DATA", "P1-0"),
    "audit_label_drafts": ("DATA", "DATA-1"),
    "benchmark_decode": ("EVAL", "EVAL-0"),
    "benchmark_sam3dbody": ("EVAL", "EVAL-0"),
    "build_corrections_queue": ("RPT", "RPT-1"),
    "build_eval0_index": ("EVAL", "EVAL-0"),
    "build_report_artifacts": ("RPT", "RPT-1"),
    "build_serving_manifest": ("RPT", "RPT-1"),
    "build_variant_comparison": ("EVAL", "EVAL-0"),
    "calibrate": ("CAL", "CAL-2"),
    "check_eval_regression": ("EVAL", "EVAL-1"),
    "corpus_dashboard": ("DATA", "P0-4"),
    "court_precision_harness": ("CAL", "CAL-2"),
    "doctor": ("ENV", "ENV-2"),
    "extract_label_frames": ("DATA", "DATA-1"),
    "finetune_pose": ("BODY", "BODY-4"),
    "generate_flight_corpus": ("BALL", "P0-7"),
    "ingest_testclips": ("DATA", "DATA-1"),
    "init_label_workdir": ("DATA", "DATA-1"),
    "manifest_report": ("RPT", "RPT-1"),
    "materialize_seed_manifest": ("DATA", "DATA-1"),
    "register_testclip": ("DATA", "DATA-1"),
    "register_testclips_manifest": ("DATA", "DATA-1"),
    "render_calibration_overlay": ("CAL", "CAL-2"),
    "report_testclip_coverage": ("DATA", "DATA-1"),
    "smoke_models": ("ENV", "ENV-2"),
    "smoke_mujoco_mjx": ("ENV", "ENV-1"),
    "summarize_decode_benchmarks": ("EVAL", "EVAL-0"),
    "summarize_eval_runs": ("EVAL", "EVAL-1"),
    "track": ("TRK", "TRK-1"),
    "validate_ball_audio_dataset": ("DATA", "DATA-3"),
    "validate_corrections": ("RPT", "RPT-1"),
    "validate_pipeline_artifacts": ("EVAL", "EVAL-4"),
    "validate_pose_dataset": ("DATA", "DATA-2"),
    "validate_racket_dataset": ("DATA", "DATA-4"),
    "validate_reference_ranges": ("COACH", "P6-3"),
    "validate_shot_dataset": ("DATA", "DATA-5"),
    "validate_testclips": ("DATA", "DATA-1"),
}


def build_scaffold_tool_index(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _validate_root(root)
    tests_root = root / TEST_ROOT
    schemas_root = root / SCHEMA_ROOT

    script_paths = _script_paths(root)
    tools = [
        _tool_entry(path, root=root, tests_root=tests_root, schemas_root=schemas_root)
        for path in script_paths
    ]
    category_counts = Counter(tool["category"] for tool in tools)

    return {
        "schema_version": 3,
        "artifact_type": "racketsport_scaffold_tool_index",
        "scripts_root": SCRIPT_ROOT.as_posix(),
        "tests_root": TEST_ROOT.as_posix(),
        "schema_root": SCHEMA_ROOT.as_posix(),
        "scope": {
            "indexed_globs": list(INDEXED_GLOBS),
            "excluded_globs": [],
            "repo_wide_hygiene_report": False,
        },
        "execution": {
            "cpu_only": True,
            "runs_scaffold_commands": False,
            "uses_gpu": False,
            "downloads": False,
            "mutates_repo": False,
            "claims_build_or_eval_status": False,
        },
        "summary": {
            "tool_count": len(tools),
            "with_related_tests": sum(1 for tool in tools if tool["related_test"] is not None),
            "missing_related_tests": sum(1 for tool in tools if tool["related_test"] is None),
            "with_direct_cli_reference_tests": sum(
                1 for tool in tools if tool["direct_cli_reference_test"] is not None
            ),
            "missing_direct_cli_reference_tests": sum(
                1 for tool in tools if tool["direct_cli_reference_test"] is None
            ),
            "with_matching_json_schema_files": sum(1 for tool in tools if tool["matching_schema"] is not None),
            "missing_matching_json_schema_files": sum(1 for tool in tools if tool["matching_schema"] is None),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "tools": tools,
    }


def _validate_root(root: Path) -> None:
    if not root.exists():
        raise ValueError(f"root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    scripts_root = root / SCRIPT_ROOT
    if not scripts_root.is_dir():
        raise ValueError(f"scripts root does not exist: {scripts_root}")


def _script_paths(root: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in INDEXED_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                paths[path.relative_to(root).as_posix()] = path
    return [paths[key] for key in sorted(paths)]


def _tool_entry(path: Path, *, root: Path, tests_root: Path, schemas_root: Path) -> dict[str, Any]:
    stem = path.stem
    workstream, task_prefix = TASK_HINTS.get(stem, _guess_task(stem))
    command_path = _relative_posix(path, root=root)
    return {
        "command_path": command_path,
        "stem": stem,
        "category": _category(stem),
        "workstream": workstream,
        "task_prefix": task_prefix,
        "related_test": _related_test(stem, command_path=command_path, tests_root=tests_root, root=root),
        "direct_cli_reference_test": _direct_cli_reference_test(command_path, tests_root=tests_root, root=root),
        "matching_schema": _matching_schema(stem, schemas_root=schemas_root, root=root),
    }


def _category(stem: str) -> str:
    normalized = stem.replace("-", "_")
    if (
        normalized in {"doctor", "gpu_train_lock", "gpu_cold_start", "setup_env"}
        or normalized.startswith("install_")
        or normalized.startswith("smoke_")
    ):
        return "env"
    if normalized == "process_video":
        return "pipeline"
    if (
        "confidence" in normalized
        or "runtime" in normalized
        or "diagnostic" in normalized
        or "stats" in normalized
        or normalized.startswith("monitor_")
    ):
        return "report"
    if "download_checkpoint" in normalized:
        return "model"
    if "pipeline_artifacts" in normalized:
        return "eval"
    if "decode" in normalized:
        return "decode"
    if "serving" in normalized:
        return "serving"
    if normalized == "generate_flight_corpus":
        return "physics"
    if "replay" in normalized or "scrubber" in normalized or "viewer" in normalized:
        return "replay"
    if "eval" in normalized or "variant_comparison" in normalized or "benchmark" in normalized or "sweep" in normalized or normalized.startswith("measure_"):
        return "eval"
    if (
        "calibrat" in normalized
        or normalized == "calibrate"
        or "camera_motion" in normalized
        or "court_line" in normalized
        or "line_family" in normalized
        or "court_keypoint" in normalized
        or "court_detector" in normalized
        or "court_proposal" in normalized
        or "court_precision" in normalized
        or "court_temporal" in normalized
        or "net_anchor_court" in normalized
    ):
        return "calibration"
    if (
        "dataset" in normalized
        or "corpus" in normalized
        or "roboflow" in normalized
        or "testclip" in normalized
        or "seed_manifest" in normalized
        or "tiled_raw_pool" in normalized
        or normalized.startswith("ingest_")
    ):
        return "dataset"
    if (
        "label" in normalized
        or "cvat" in normalized
        or "teacher" in normalized
        or "review_frames" in normalized
        or "review_input" in normalized
        or "event_review" in normalized
    ):
        return "label"
    if "shot" in normalized or "tenniset" in normalized:
        return "shot"
    if "contact" in normalized or "audio" in normalized or "wrist" in normalized or "rally" in normalized:
        return "contact"
    if (
        "ball" in normalized
        or "sst" in normalized
        or "totnet" in normalized
        or "pbmat" in normalized
        or "tracknet" in normalized
        or "inout" in normalized
    ):
        return "ball"
    if "racket" in normalized or "paddle" in normalized or normalized.startswith("rkt_"):
        return "racket"
    if (
        "body" in normalized
        or "person" in normalized
        or "player_track" in normalized
        or "frame_compute" in normalized
        or "hmr" in normalized
        or "pose" in normalized
        or "skeleton" in normalized
        or "player_scale" in normalized
        or "player_court_membership" in normalized
        or "vn_trajectories" in normalized
    ):
        return "body"
    if (
        "physics" in normalized
        or "flight" in normalized
        or "foot" in normalized
        or "placement" in normalized
        or "virtual_world" in normalized
        or "mujoco" in normalized
    ):
        return "physics"
    if (
        "model" in normalized
        or "coreml" in normalized
        or "sam3dbody" in normalized
        or "finetune" in normalized
        or normalized.startswith("train_")
    ):
        return "model"
    if (
        "report" in normalized
        or "corrections" in normalized
        or "reference_ranges" in normalized
        or "readiness" in normalized
        or "promotion_audit" in normalized
        or "review_packet" in normalized
        or "review_action_manifest" in normalized
        or "scaffold" in normalized
    ):
        return "report"
    if normalized.startswith("audit_"):
        return "report"
    if "track" in normalized:
        return "tracking"
    return "unknown"


def _guess_task(stem: str) -> tuple[str | None, str | None]:
    category = _category(stem)
    if category == "dataset":
        return "DATA", None
    if category == "label":
        return "DATA", None
    if category == "eval":
        return "EVAL", None
    if category == "env":
        return "ENV", None
    if category == "model":
        return "BODY", None
    if category == "body":
        return "BODY", None
    if category == "ball":
        return "BALL", None
    if category == "racket":
        return "RKT", None
    if category == "contact":
        return "BALL", None
    if category == "shot":
        return "SHOT", None
    if category == "physics":
        return "E2E", None
    if category == "report":
        return "RPT", None
    if category == "serving":
        return "RPT", None
    if category == "replay":
        return "RPL", None
    if category == "calibration":
        return "CAL", None
    if category == "tracking":
        return "TRK", None
    if category == "pipeline":
        return "E2E", None
    return None, None


def _related_test(stem: str, *, command_path: str, tests_root: Path, root: Path) -> str | None:
    candidates = []
    override = RELATED_TEST_OVERRIDES.get(stem)
    if override is not None:
        candidates.append(override)
    candidates.extend(f"test_{candidate}.py" for candidate in _stem_candidates(stem))
    direct_match = _first_existing(candidates, directory=tests_root, root=root)
    if direct_match is not None:
        return direct_match
    return _first_test_referencing(command_path, tests_root=tests_root, root=root)


def _direct_cli_reference_test(command_path: str, *, tests_root: Path, root: Path) -> str | None:
    return _first_test_referencing(command_path, tests_root=tests_root, root=root)


def _matching_schema(stem: str, *, schemas_root: Path, root: Path) -> str | None:
    candidates = []
    override = SCHEMA_OVERRIDES.get(stem)
    if override is not None:
        candidates.append(override)
    candidates.extend(f"{candidate}_schema.json" for candidate in _stem_candidates(stem))
    return _first_existing(candidates, directory=schemas_root, root=root)


def _stem_candidates(stem: str) -> list[str]:
    candidates = [stem]
    for prefix in PREFIXES:
        if stem.startswith(prefix):
            candidates.append(stem.removeprefix(prefix))
    if stem.endswith("_benchmarks"):
        candidates.append(stem.removesuffix("s"))
    if stem.endswith("_artifacts"):
        candidates.append(stem)
        candidates.append(stem.removesuffix("_artifacts"))
    if stem.endswith("_manifest"):
        candidates.append(stem.removesuffix("_manifest"))
    return _dedupe(candidates)


def _first_existing(names: list[str], *, directory: Path, root: Path) -> str | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return _relative_posix(path, root=root)
    return None


def _first_test_referencing(command_path: str, *, tests_root: Path, root: Path) -> str | None:
    if not tests_root.is_dir():
        return None
    for path in sorted(tests_root.glob("test_*.py")):
        if path.name == "test_scaffold_tool_index.py" and command_path != "scripts/racketsport/list_scaffold_tools.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if command_path in text:
            return _relative_posix(path, root=root)
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _relative_posix(path: Path, *, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List racketsport scaffold CLIs and their local test/schema coverage gaps."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root to inspect.")
    args = parser.parse_args()

    try:
        report = build_scaffold_tool_index(args.root)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
