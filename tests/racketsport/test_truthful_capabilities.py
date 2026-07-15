from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.racketsport import audit_storage_policy


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_DOCS = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "NORTH_STAR_ROADMAP.md",
    "RUNBOOK.md",
    "BALL_TRACKING_PIPELINE.md",
]

ALLOWED_MARKDOWN_DOCS = set(CANONICAL_DOCS) | {
    "corrections/README.md",
    "cvat_upload/CVAT_LABELING_INSTRUCTIONS.md",
    "cvat_upload/OWNER_SESSION_20260708.md",
    "cvat_upload/OWNER_SESSION_W6_20260708.md",
    "cvat_upload/court_keypoints_20260707/OWNER_COURT_KP_GUIDE.md",
    "cvat_upload/exports/README.md",
    "cvat_upload/exports/w7_labels_20260709/README_INGEST_QUEUE_20260709.md",
    "cvat_upload/w7_audit_stratum_20260709/TASK_README.md",
    "cvat_upload/court_diversity_20260712/OWNER_GUIDE.md",
    "cvat_upload/exports/court_keypoints_20260707/PARTIAL_EXPORT_NOTES_20260709.md",
    "cvat_upload/exports/w6_labelpack_20260708/SESSION_NOTES_20260709.md",
    "cvat_upload/exports/court_keypoints_20260707/README.md",
    "cvat_upload/exports/harvest_review_20260707/Ezz6HDNHlnk_rally_0004/MANAGER_NOTE.md",
    "cvat_upload/exports/harvest_review_20260707/HyUqT7zFiwk_rally_0001/MANAGER_NOTE.md",
    "cvat_upload/exports/harvest_review_20260707/README.md",
    "cvat_upload/exports/harvest_review_20260707/_L0HVmAlCQI_rally_0001/MANAGER_NOTE.md",
    "cvat_upload/exports/harvest_review_20260707/zwCtH_i1_S4_rally_0001/MANAGER_NOTE.md",
    "eval_clips/ball/README.md",
    "ios/README.md",
    "serving/triton/racketsport_ensemble/README.md",
    "web/replay/README.md",
}

GENERATED_MARKDOWN_ARTIFACTS = {
    "data/roboflow_universe_20260706/aggregated/corpus_card.md",
}

REMOVED_NARRATIVE_DOCS = [
    "MASTER_PLAN.md",
    "BUILD_CHECKLIST.md",
    "CAPABILITIES.md",
    "TECH_STACK.md",
    "TIER_MAP.md",
    "TECH_BLUEPRINTS.md",
    "EDGE_PLAYBOOK.md",
    "FABLE_OPERATING_MANUAL.md",
    "OVERLAPPING_COURT_CALIBRATION_GOAL.md",
    "RACKET_6DOF_GOAL.md",
    "OWNER_CHECKIN_20260707.md",
    "OWNER_CHECKIN_20260708.md",
    "OWNER_CHECKIN_20260709.md",
    "docs/specs/2026-07-07-product-infra-design.md",
    "JOINT_DETECTION_AND_PLACEMENT_HANDOFF.md",
    "JOINT_PLACEMENT_HANDOFF_20260704.md",
    "RESET_HANDOFF_20260705.md",
    "PIPELINE_STATUS.md",
    "OWNER_CHECKIN_20260703.md",
    "OWNER_CHECKIN_20260705.md",
    "OWNER_CHECKIN_20260706.md",
    "ACCURACY_AND_TRAINING.md",
    "IMPLEMENTATION_PHASES.md",
    "SWAY_BODY_PICKLEBALL_MVP.md",
    "PRODUCT_ROADMAP.md",
    "COURT_POSITIONING_PIPELINE_SPEC.md",
    "pickleball-joint-pipeline-spec.md",
    "CHECKPOINT_20260702.md",
    "OWNER_CHECKIN_20260702.md",
]

ARCHIVED_DOCS = {
    "NORTH_STAR_ROADMAP_PRE_CONSOLIDATION.md",
    "MASTER_PLAN.md",
    "BUILD_CHECKLIST.md",
    "CAPABILITIES.md",
    "TECH_STACK.md",
    "TIER_MAP.md",
    "TECH_BLUEPRINTS.md",
    "EDGE_PLAYBOOK.md",
    "FABLE_OPERATING_MANUAL.md",
    "OVERLAPPING_COURT_CALIBRATION_GOAL.md",
    "RACKET_6DOF_GOAL.md",
    "OWNER_CHECKIN_20260707.md",
    "OWNER_CHECKIN_20260708.md",
    "OWNER_CHECKIN_20260709.md",
    "PRODUCT_INFRA_DESIGN_20260707.md",
    "INDEX.md",
}

ALLOWED_LARGE_TRACKED_FILES = {
    "cvat_upload/04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
    "runs/lanes/w7_ballretrain2_20260709/vm_pull/arm_finetunes/E3k_matched_seed_official_aug/checkpoints/latest.pt",
    "runs/lanes/w7_ballretrain2_20260709/vm_pull/arm_finetunes/E3k_seed_official_aug/checkpoints/latest.pt",
    "cvat_upload/court_keypoints_20260707/packages/court_keypoints_metric15_20260707_6frames.zip",
    "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
    "models_coreml/yolo26m_img416_int8/yolo26m.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "models_coreml/yolo26s_img416_int8/yolo26s.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
}

W6_LABELPACK_IMAGE_ZIPS = {
    f"cvat_upload/w6_labelpack_20260708/packages/ball_session_{index:02d}_640f_w6_images.zip"
    for index in range(1, 68)
} | {
    "cvat_upload/w6_labelpack_20260708/packages/ball_session_68_350f_w6_images.zip",
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
    "cvat_upload/w7_audit_stratum_20260709/w7_audit_stratum_uniform350_images.zip",
    "data/event_bootstrap_20260713/audio_onsets_v0/HyUqT7zFiwk_rally_0001.json",
    "data/event_bootstrap_20260713/audio_onsets_v0/pbvision_11min_20260713.json",
    "data/event_bootstrap_20260713/contact_windows_v0.jsonl",
    "data/event_bootstrap_20260713/negative_windows_v0.jsonl",
    "data/event_public_20260713/openttgames/markup/game_1.zip",
    "data/event_public_20260713/openttgames/markup/game_2.zip",
    "data/event_public_20260713/openttgames/markup/game_3.zip",
    "data/event_public_20260713/openttgames/markup/game_4.zip",
    "data/event_public_20260713/openttgames/markup/game_5.zip",
    "data/event_public_20260713/openttgames/markup/test_4.zip",
    "data/event_public_20260713/openttgames/markup/test_5.zip",
    "data/event_public_20260713/openttgames/markup/test_6.zip",
    "data/event_public_20260713/openttgames/videos/game_4.mp4",
    "data/event_public_20260713/openttgames/videos/test_2.mp4",
    "data/event_public_20260713/padeltracker100/labels.zip",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalF_1_ball.json",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalF_1_homography.json",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalF_1_pose.json",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalM_1_ball.json",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalM_1_homography.json",
    "data/event_public_20260713/padeltracker100/labels_extracted/2022_BCN_FinalM_1_pose.json",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00170.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00171.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00172.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00173.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00174.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00175.mp4",
    "data/event_public_20260713/shuttlecock_hitting_zenodo/sample_clips/00176.mp4",
    "data/event_public_20260713/squash_audio_figshare/audio1_targeted_shots_part1.wav",
    "cvat_upload/court_diversity_20260712/packages/court_diversity_20260712_shard1.zip",
    "cvat_upload/court_diversity_20260712/packages/court_diversity_20260712_shard2.zip",
    "cvat_upload/court_diversity_20260712/packages/court_diversity_20260712_shard3.zip",
    "cvat_upload/court_diversity_20260712/packages/court_diversity_20260712_shard4.zip",
    "cvat_upload/court_diversity_20260712/frames/jr_60WVlG4c__seg1_f005.png",
    "cvat_upload/court_diversity_20260712/frames/jr_60WVlG4c__seg1_f008.png",
    "cvat_upload/court_diversity_20260712/frames/jr_60WVlG4c__seg2_f005.png",
    "cvat_upload/court_diversity_20260712/frames/jr_60WVlG4c__seg2_f007.png",
    "cvat_upload/court_diversity_20260712/frames/ltIxlS0QJhg__seg1_f011.png",
    "cvat_upload/court_diversity_20260712/frames/ltIxlS0QJhg__seg1_f012.png",
    "cvat_upload/court_diversity_20260712/frames/ltIxlS0QJhg__seg2_f004.png",
    "cvat_upload/court_diversity_20260712/frames/ltIxlS0QJhg__seg2_f012.png",
    # Local-only owner labeling packages regenerated from the wave-5 labelpack lane.
    "cvat_upload/w5_labelpack_20260708/packages/ball_session_01_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip",
    "cvat_upload/w5_labelpack_20260708/packages/ball_session_02_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip",
    "cvat_upload/w5_labelpack_20260708/packages/ball_session_03_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip",
    "cvat_upload/w5_labelpack_20260708/packages/ball_session_04_640f_73VurrTKCZ8_Ezz6HDNHlnk_images.zip",
    "cvat_upload/w5_labelpack_20260708/packages/court_kp_relabel_HyUqT7zFiwk_zwCtH_i1_S4_images.zip",
    "ios/Replay/Sources/PickleballReplay/Resources/RealityReplayFixture/body_mesh_animated_budget53.usdz",
    # pb.vision 11-min export dropped 2026-07-13 (reference diagnostic only, never training/GT).
    "data/pbvision_11min_20260713/cv_export.json",
    "data/pbvision_11min_20260713/source_video.mp4",
    "ios/Replay/Sources/PickleballReplay/Resources/WorldFixture/virtual_world.json",
    "tests/racketsport/fixtures/solid_mesh_real_window_000/body_mesh_faces.json",
} | W6_LABELPACK_IMAGE_ZIPS


def test_canonical_doc_set_exists_and_obsolete_root_docs_are_removed() -> None:
    for relpath in CANONICAL_DOCS:
        assert (ROOT / relpath).is_file(), relpath

    for relpath in REMOVED_NARRATIVE_DOCS:
        assert not (ROOT / relpath).exists(), relpath

    archive_root = ROOT / "runs" / "archive" / "root_docs_20260709"
    assert {path.name for path in archive_root.glob("*.md")} == ARCHIVED_DOCS
    archive_index = (archive_root / "INDEX.md").read_text(encoding="utf-8")
    assert "immutable historical context" in archive_index
    assert "NORTH_STAR_ROADMAP.md" in archive_index

    assert not list((ROOT / "docs" / "superpowers").rglob("*.md"))
    assert not list((ROOT / "docs" / "racketsport").rglob("*.md"))


def test_markdown_doc_inventory_stays_small_and_explicit() -> None:
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".claude",
        ".venv",
        ".venv_yolo_coreml",
        "models",
        "runs",
        "third_party",
        "node_modules",
        "data",
    }
    markdown_docs = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.md")
        if ignored_parts.isdisjoint(path.relative_to(ROOT).parts)
        and path.relative_to(ROOT).as_posix() not in GENERATED_MARKDOWN_ARTIFACTS
    }

    assert markdown_docs == ALLOWED_MARKDOWN_DOCS


def test_readme_points_agents_to_small_canonical_stack() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for relpath in (
        "NORTH_STAR_ROADMAP.md",
        "AGENTS.md",
        "RUNBOOK.md",
        "BALL_TRACKING_PIPELINE.md",
    ):
        assert relpath in text

    for relpath in REMOVED_NARRATIVE_DOCS:
        if "/" not in relpath:
            assert f"`{relpath}`" not in text

    assert "`VERIFIED=0`" in text
    assert "sole product vision" in text
    assert "runs/archive/root_docs_20260709/" in text
    assert "scripts/racketsport/process_video.py" in text
    assert "docs/racketsport/` | JSON schemas and manifests only" in text
    assert "## Storage Policy" in text
    assert "Do not add new long-lived research/status Markdown under `docs/racketsport/`" in text


def test_retired_authority_names_only_survive_as_archived_evidence_paths() -> None:
    retired_authorities = (
        "MASTER_PLAN.md",
        "BUILD_CHECKLIST.md",
        "CAPABILITIES.md",
        "TECH_STACK.md",
        "TIER_MAP.md",
        "TECH_BLUEPRINTS.md",
        "EDGE_PLAYBOOK.md",
        "FABLE_OPERATING_MANUAL.md",
    )
    scan_roots = [
        *(ROOT / relpath for relpath in CANONICAL_DOCS),
        ROOT / "configs" / "racketsport",
        ROOT / "docs" / "racketsport",
        ROOT / "ios",
        ROOT / "scripts" / "racketsport",
        ROOT / "server",
        ROOT / "serving",
        ROOT / "spikes" / "vn_trajectories",
        ROOT / "threed" / "racketsport",
        ROOT / "web" / "replay" / "src",
    ]
    text_suffixes = {".json", ".md", ".py", ".swift", ".toml", ".ts", ".tsx", ".yaml", ".yml"}
    ignored_parts = {".build", "node_modules"}

    paths: list[Path] = []
    for root in scan_roots:
        paths.extend([root] if root.is_file() else root.rglob("*"))

    for path in paths:
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        if ignored_parts.intersection(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        for retired in retired_authorities:
            active_text = text.replace(
                f"runs/archive/root_docs_20260709/{retired}",
                "",
            )
            assert retired not in active_text, f"{path.relative_to(ROOT)}: {retired}"


def test_agents_doc_gives_future_agents_a_code_navigation_map() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "## Agent Navigation" in text
    for relpath in (
        "README.md",
        "NORTH_STAR_ROADMAP.md",
        "RUNBOOK.md",
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
        "runs/archive/root_docs_20260709/INDEX.md",
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
    assert "only product/current-truth/future-plan authority" in text


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
    original_allowed_large_untracked = audit_storage_policy.ALLOWED_LARGE_UNTRACKED_SOURCE_FILES
    try:
        audit_storage_policy.ALLOWED_LARGE_UNTRACKED_SOURCE_FILES = ALLOWED_LARGE_UNTRACKED_SOURCE_FILES
        report = audit_storage_policy.build_storage_report(ROOT, check_generated_artifacts=False)
    finally:
        audit_storage_policy.ALLOWED_LARGE_UNTRACKED_SOURCE_FILES = original_allowed_large_untracked

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
    for relpath in ("README.md", "RUNBOOK.md"):
        text = (ROOT / relpath).read_text(encoding="utf-8")
        assert ".venv/bin/python -m pytest" in text, relpath
        assert "python3 -m pytest" not in text, relpath


def test_north_star_is_the_single_product_and_execution_authority() -> None:
    text = (ROOT / "NORTH_STAR_ROADMAP.md").read_text(encoding="utf-8")

    assert text.startswith("# DinkVision North Star")
    assert "This is the sole authority" in text
    assert "`VERIFIED=0`" in text
    assert "## 1. The product" in text
    assert "## 2. Current truth" in text
    assert "## 3. Target CV architecture and data reuse" in text
    assert "## 4. Ordered execution program" in text
    assert "## 5. Active queue for the next agents" in text
    assert "## 6. Standing rules" in text
    assert "## 7. Evidence and history" in text
    assert len(text.splitlines()) <= 500
    assert "runs/CV_SOTA_RESEARCH_20260709.md" in text

    for research_direction in (
        "encoded PTS",
        "existing TOTNet adapter",
        "both IPPE poses",
        "GEM-X",
        "render-only appearance",
    ):
        assert research_direction in text, research_direction

    for phase in ("NS-01", "NS-02", "NS-03", "NS-04", "NS-05", "NS-06", "NS-07"):
        assert phase in text

    for gate in (
        "F1@20 ≥0.90",
        "recall@20 ≥0.75",
        "hFP ≤0.05",
        "IDF1 ≥0.85",
        "zero far-off-court FP",
        "world-MPJPE ≤50mm",
        "`grounding_metrics.max_foot_lock_slide_m` ≤0.03",
        "face-angle p90 ≤5°",
        "contact-point p90 ≤3cm",
        "shot macro-F1 ≥0.65",
        "top-2 accuracy ≥0.85",
        "Usefulness ≥8/10",
        "fabrication 0/300",
        "≤2× source-video duration",
    ):
        assert gate in text, gate

    for retired in (
        "MASTER_PLAN.md",
        "BUILD_CHECKLIST.md",
        "CAPABILITIES.md",
        "TECH_STACK.md",
        "TIER_MAP.md",
        "TECH_BLUEPRINTS.md",
        "EDGE_PLAYBOOK.md",
        "FABLE_OPERATING_MANUAL.md",
    ):
        assert f"`{retired}`" not in text


def test_north_star_orders_correctness_before_accuracy_and_productization() -> None:
    text = (ROOT / "NORTH_STAR_ROADMAP.md").read_text(encoding="utf-8")
    ordered_markers = (
        "### NS-01 — Make the real product route correct",
        "### NS-02 — Build independent truth and reset evaluation",
        "### NS-03 — Improve components in parallel",
        "### NS-04 — Join the lanes into one world",
        "### NS-05 — Turn the world into a useful product",
        "### NS-06 — Optimize speed, cost, and reliability",
        "### NS-07 — Launch safely and prove repeatability",
    )
    offsets = [text.index(marker) for marker in ordered_markers]
    assert offsets == sorted(offsets)
    assert "Do not start another broad model search" in text
    assert "Parallel now" in text
    assert "In-flight background lanes may finish and save their reports" in text


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
        "**input_quality**",
        "**tracking**",
        "**camera_motion**",
        "**placement**",
        "**rally_gating**",
        "**ball**",
        "**ball_arc**",
        "**events**",
        "**ball_fill**",
        "**frames**",
        "**body**",
        "**placement_refine**",
        "**grounding_refine**",
        "**paddle_pose**",
        "**world**",
        "**confidence_gate**",
        "**match_stats**",
        "**coaching_facts**",
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
