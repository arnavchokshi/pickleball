from __future__ import annotations

from collections import Counter
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_DOCS = [
    "README.md",
    "AGENTS.md",
    "MASTER_PLAN.md",
    "RUNBOOK.md",
    "CAPABILITIES.md",
    "BUILD_CHECKLIST.md",
    "TECH_STACK.md",
    "BALL_TRACKING_PIPELINE.md",
    "TIER_MAP.md",
]

ALLOWED_MARKDOWN_DOCS = set(CANONICAL_DOCS) | {
    "corrections/README.md",
    "cvat_upload/CVAT_LABELING_INSTRUCTIONS.md",
    "cvat_upload/exports/README.md",
    "eval_clips/ball/README.md",
    "ios/README.md",
    "serving/triton/racketsport_ensemble/README.md",
    "web/replay/README.md",
}

REMOVED_NARRATIVE_DOCS = [
    "ACCURACY_AND_TRAINING.md",
    "IMPLEMENTATION_PHASES.md",
    "SWAY_BODY_PICKLEBALL_MVP.md",
    "PRODUCT_ROADMAP.md",
    "COURT_POSITIONING_PIPELINE_SPEC.md",
    "pickleball-joint-pipeline-spec.md",
    "CHECKPOINT_20260702.md",
    "OWNER_CHECKIN_20260702.md",
]

REQUIRED_CAPABILITY_COLUMNS = [
    "stage",
    "named tech (registry)",
    "actually invoked?",
    "correct variant+weight?",
    "wired into spine?",
    "gate type (accuracy/presence/none)",
    "gate run on real labels?",
    "honest status",
]

EXPECTED_STATUS_COUNTS = Counter(
    {
        "IN-PROGRESS": 3,
        "INTERNAL-VAL DONE": 1,
        "SCAFFOLD": 3,
        "SCAFFOLD/PREVIEW": 1,
        "SCOPED PASS": 2,
        "SCAFFOLD/SCOPED PASS": 1,
    }
)

ALLOWED_LARGE_TRACKED_FILES = {
    "cvat_upload/04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
    "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
    "models_coreml/yolo26m_img416_int8/yolo26m.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "models_coreml/yolo26s_img416_int8/yolo26s.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
}

ALLOWED_DUPLICATE_TRACKED_BLOBS = {
    frozenset(
        {
            "cvat_upload/04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
            "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
        }
    )
}

ALLOWED_LARGE_UNTRACKED_SOURCE_FILES = {
    "ios/Replay/Sources/PickleballReplay/Resources/RealityReplayFixture/body_mesh_animated_budget53.usdz",
    "ios/Replay/Sources/PickleballReplay/Resources/WorldFixture/virtual_world.json",
    "tests/racketsport/fixtures/solid_mesh_real_window_000/body_mesh_faces.json",
}


def test_canonical_doc_set_exists_and_obsolete_root_docs_are_removed() -> None:
    for relpath in CANONICAL_DOCS:
        assert (ROOT / relpath).is_file(), relpath

    for relpath in REMOVED_NARRATIVE_DOCS:
        assert not (ROOT / relpath).exists(), relpath

    assert not list((ROOT / "docs" / "superpowers").rglob("*.md"))
    assert not list((ROOT / "docs" / "racketsport").rglob("*.md"))


def test_markdown_doc_inventory_stays_small_and_explicit() -> None:
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv_yolo_coreml",
        "runs",
        "third_party",
        "node_modules",
    }
    markdown_docs = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.md")
        if ignored_parts.isdisjoint(path.relative_to(ROOT).parts)
    }

    assert markdown_docs == ALLOWED_MARKDOWN_DOCS


def test_readme_points_agents_to_small_canonical_stack() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for relpath in CANONICAL_DOCS[1:-1]:
        assert relpath in text

    for relpath in REMOVED_NARRATIVE_DOCS:
        assert relpath not in text

    assert "`VERIFIED=0`" in text
    assert "scripts/racketsport/process_video.py" in text
    assert "docs/racketsport/` | JSON schemas and manifests only" in text
    assert "## Storage Policy" in text
    assert "Do not add new long-lived research/status Markdown under `docs/racketsport/`" in text


def test_agents_doc_gives_future_agents_a_code_navigation_map() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "## Agent Navigation" in text
    for relpath in (
        "README.md",
        "MASTER_PLAN.md",
        "RUNBOOK.md",
        "CAPABILITIES.md",
        "BUILD_CHECKLIST.md",
        "TECH_STACK.md",
        "scripts/racketsport/process_video.py",
        "scripts/racketsport/",
        "threed/racketsport/",
        "tests/racketsport/",
        "ios/",
        "web/replay/",
        "models/MANIFEST.json",
        "configs/",
        "docs/racketsport/",
        "runs/",
    ):
        assert relpath in text

    for command in (
        "scripts/racketsport/list_scaffold_tools.py --root .",
        "scripts/racketsport/audit_dead_code.py --root .",
        "scripts/racketsport/audit_storage_policy.py --root . --json",
    ):
        assert command in text

    assert "JSON schemas/manifests only" in text
    assert "Generated evidence only" in text


def test_storage_policy_keeps_large_tracked_artifacts_explicit() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    large_tracked = {
        path
        for path in _git_tracked_paths()
        if (ROOT / path).is_file() and (ROOT / path).stat().st_size > 5 * 1024 * 1024
    }

    assert large_tracked == ALLOWED_LARGE_TRACKED_FILES
    assert _duplicate_tracked_blob_groups(large_tracked) == ALLOWED_DUPLICATE_TRACKED_BLOBS
    assert "`models_coreml/`" in text
    assert "Indoor CVAT/eval video mirror" in text


def test_storage_policy_audit_classifies_large_worktree_artifacts() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    _remove_generated_artifacts()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/audit_storage_policy.py",
            "--root",
            ".",
            "--json",
            "--ignore-generated-artifacts",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    report = json.loads(completed.stdout)

    assert report["unknown_large_tracked_files"] == []
    assert report["unknown_large_untracked_source_files"] == []
    assert report["check_generated_artifacts"] is False
    assert report["generated_artifacts"] == []
    assert set(report["allowed_large_tracked_files"]) == ALLOWED_LARGE_TRACKED_FILES
    assert set(report["allowed_large_untracked_source_files"]) == ALLOWED_LARGE_UNTRACKED_SOURCE_FILES
    assert "scripts/racketsport/audit_storage_policy.py" in text
    assert "active large source/fixture exceptions" in text
    assert "generated cache/build leftovers" in text


def test_python_verification_commands_use_repo_virtualenv() -> None:
    for relpath in ("README.md", "TECH_STACK.md", "RUNBOOK.md"):
        text = (ROOT / relpath).read_text(encoding="utf-8")
        assert ".venv/bin/python -m pytest" in text, relpath
        assert "python3 -m pytest" not in text, relpath


def test_master_plan_preserves_goal_and_no_overclaim_boundary() -> None:
    text = (ROOT / "MASTER_PLAN.md").read_text(encoding="utf-8")

    assert "Build the best single-camera pickleball video-to-3D analysis pipeline" in text
    assert "## What Exists So Far" in text
    assert "`runs/` is generated evidence" in text
    assert "`VERIFIED=0`" in text
    assert "SCAFFOLD/SCOPED PASS, not VERIFIED" in text
    assert "Do not call a stage `VERIFIED` from smoke tests" in text
    assert "overlapping or multipurpose court paint" in text
    assert "scripts/racketsport/evaluate_overlapping_court_calibration.py" in text
    assert "LM-optimized mean residual: 0.414584 ft" in text
    assert "`opencv_hsv_paint_hough`: 0.0000" in text
    assert "`opencv_hsv_paint_net_crop_hough`: 0.0250" in text
    assert "`AGENTS.md`" in text
    assert "Candidate predictions, copied labels, and box-derived paddle candidates" not in text


def test_runbook_documents_current_process_video_entrypoint() -> None:
    text = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    compact_text = " ".join(text.split())

    assert "scripts/racketsport/process_video.py" in text
    assert "`./run_pipeline`" not in text
    assert "`threed/racketsport/pipeline_cli.py`" in text
    assert "you are copying old sample artifacts, not running models" in compact_text

    expected_order = [
        "**ingest**",
        "**calibration**",
        "**tracking**",
        "**rally_gating**",
        "**frames**",
        "**ball**",
        "**events**",
        "**body**",
        "**grounding**",
        "**world**",
        "**confidence**",
        "**manifest**",
        "**verify**",
    ]
    last_index = -1
    for marker in expected_order:
        index = text.index(marker)
        assert index > last_index
        last_index = index

    for flag in (
        "--court-calibration",
        "--tracks",
        "--ball-track",
        "--no-gpu",
        "--body-local",
        "--verify-viewer",
        "--allow-auto-court-corners-preview",
    ):
        assert flag in text

    for command in (
        "scripts/racketsport/audit_dead_code.py",
        "scripts/racketsport/audit_storage_policy.py",
        "scripts/racketsport/list_scaffold_tools.py",
        "tests/racketsport/test_storage_policy_audit.py",
    ):
        assert command in text


def test_runbook_documents_remote_body_runtime_flags_from_help() -> None:
    text = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/process_video.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    help_text = completed.stdout

    remote_body_flags = (
        "--remote-host",
        "--remote-ssh-key",
        "--remote-repo",
        "--remote-python",
        "--remote-fast-sam-python",
        "--remote-fast-sam-root",
        "--remote-lock-wait-timeout-s",
        "--remote-command-timeout-s",
        "--sam3d-body-input-size-px",
        "--sam3d-crop-bucket-sizes",
        "--no-sam3d-torch-compile",
        "--sam3d-compile-warmup-buckets",
        "--serialize-tier2-mesh-vertices",
    )
    for flag in remote_body_flags:
        assert flag in help_text
        assert flag in text

    compact_text = " ".join(text.split())
    assert "remote dispatch is the default BODY path unless `--body-local` or `--no-gpu` is set" in compact_text
    assert "Shared GPU lock wait and remote command timeout are separate budgets" in compact_text
    assert "not promotion evidence by themselves" in compact_text


def test_capabilities_matrix_is_compact_and_no_row_is_verified() -> None:
    text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")

    assert "`VERIFIED=0`" in text
    assert "This section is the single source of truth for live/server placement" in text
    assert _first_markdown_table_headers(text) == REQUIRED_CAPABILITY_COLUMNS

    rows = _capability_rows(text)
    assert {
        "calibration",
        "tracking",
        "ball",
        "body",
        "foot/physics",
        "racket",
        "metrics",
        "shot/drill",
        "replay",
        "e2e",
    } == set(rows)
    assert rows["e2e"]["honest status"] == "SCAFFOLD/SCOPED PASS, not VERIFIED"
    assert all("VERIFIED" not in row["honest status"].replace("not VERIFIED", "") for row in rows.values())


def test_build_checklist_board_and_count_summary_match() -> None:
    text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    rows = _status_board_rows(text)

    assert "recent handoff" not in text.lower()
    assert "runs/sam3d_stall_schema_fix_20260703T0802Z/REPORT.md" not in text
    assert "chronological narratives" in text
    assert len(rows) == 11
    assert [row["ID"] for row in rows] == [
        "DOCS-1",
        "CAL-1",
        "TRK-1",
        "BALL-1",
        "BODY-1",
        "PHYS-1",
        "RKT-1",
        "IOS-1",
        "RPL-1",
        "E2E-1",
        "DATA-1",
    ]
    docs_row = next(row for row in rows if row["ID"] == "DOCS-1")
    assert docs_row["Status"] == "IN-PROGRESS"
    assert "full cleanup proof is still incomplete" in docs_row["Current blocker"]
    assert "Keep docs small" in docs_row["Next useful action"]

    parsed_counts = Counter(row["Status"] for row in rows)
    assert parsed_counts == EXPECTED_STATUS_COUNTS
    assert _visible_status_count_table(text) == EXPECTED_STATUS_COUNTS
    assert "No row is `VERIFIED`." in text


def test_tech_stack_maps_code_surfaces_without_old_companion_docs() -> None:
    text = (ROOT / "TECH_STACK.md").read_text(encoding="utf-8")

    for required in (
        "`models/MANIFEST.json`",
        "`scripts/racketsport/process_video.py`",
        "`scripts/racketsport/list_scaffold_tools.py`",
        "`threed/racketsport/`",
        "`ios/`",
        "`web/replay/`",
    ):
        assert required in text

    assert "category/workstream map for every checked-in CLI" in text

    for relpath in REMOVED_NARRATIVE_DOCS:
        assert relpath not in text


def test_ball_tracking_doc_keeps_code_referenced_section_anchors() -> None:
    text = (ROOT / "BALL_TRACKING_PIPELINE.md").read_text(encoding="utf-8")

    for heading in (
        "## 5. Runtime Policy",
        "### 5.1 Rally Spans",
        "### 5.6 Bounce Uncertainty",
        "## 6. Constants And Targets",
        "## 9. Acceptance Gates",
    ):
        assert heading in text

    assert "BALL is not verified" in text
    assert "too_close_to_call" in text


def test_cvat_exports_readme_does_not_recommend_stale_holdout_dataset_scope() -> None:
    text = (ROOT / "cvat_upload" / "exports" / "README.md").read_text(encoding="utf-8")

    assert "videos 1-3" not in text
    assert "Outdoor and Indoor are strict held-out eval clips" in text
    assert "current YOLO/TrackNet training exporters fail closed" in text
    assert "training/export artifacts" not in text
    assert "reviewed import/eval artifacts" in text


def _first_markdown_table_headers(text: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith("| stage |"):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]
    return []


def _capability_rows(text: str) -> dict[str, dict[str, str]]:
    headers = _first_markdown_table_headers(text)
    rows: dict[str, dict[str, str]] = {}
    in_matrix = False
    for line in text.splitlines():
        if line.startswith("| stage |"):
            in_matrix = True
            continue
        if in_matrix and line.startswith("|---"):
            continue
        if in_matrix and not line.startswith("|"):
            break
        if not in_matrix:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        rows[row["stage"]] = row
    return rows


def _status_board_rows(text: str) -> list[dict[str, str]]:
    headers: list[str] = []
    rows: list[dict[str, str]] = []
    in_board = False
    for line in text.splitlines():
        if line.startswith("| ID |"):
            in_board = True
            headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
            continue
        if in_board and line.startswith("|---"):
            continue
        if in_board and not line.startswith("|"):
            break
        if not in_board:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _visible_status_count_table(text: str) -> Counter[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "| status | count |":
            continue
        table_counts: Counter[str] = Counter()
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) != 2:
                break
            table_counts[cells[0]] = int(cells[1])
        return table_counts
    return Counter()


def _git_tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _duplicate_tracked_blob_groups(paths: set[str]) -> set[frozenset[str]]:
    completed = subprocess.run(
        ["git", "ls-files", "-s", *sorted(paths)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    by_blob: dict[str, set[str]] = {}
    for line in completed.stdout.splitlines():
        _mode, blob, _stage, path = line.split(maxsplit=3)
        by_blob.setdefault(blob, set()).add(path)
    return {frozenset(group) for group in by_blob.values() if len(group) > 1}


def _remove_generated_artifacts() -> None:
    ignored_roots = {
        ".git",
        ".venv",
        ".venv_yolo_coreml",
        "node_modules",
        "runs",
        "third_party",
    }
    generated_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".build",
    }
    for path in sorted(ROOT.rglob("*"), reverse=True):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in ignored_roots for part in rel_parts):
            continue
        if path.is_dir() and path.name in generated_dirs:
            shutil.rmtree(path)
        elif path.is_file() and (path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}):
            path.unlink()
