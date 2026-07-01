from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEMOTED_TASKS = {
    "IOS-1",
    "IOS-2",
    "IOS-3",
    "IOS-4",
    "IOS-5",
    "IOS-6",
    "BODY-1",
    "BODY-2",
    "BODY-3",
    "BODY-4",
    "FOOT-1",
    "FOOT-2",
    "BALL-1",
    "BALL-2",
    "BALL-3",
    "BALL-4",
    "RKT-1",
    "MET-1",
    "SHOT-1",
    "SHOT-2",
    "RPT-2",
    "RPT-3",
    "RPL-1",
    "RPL-2",
    "DATA-3",
    "DATA-4",
    "DATA-5",
    "EVAL-2",
    "EVAL-4",
}

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


def test_build_checklist_demotes_model_and_algorithm_scaffolds() -> None:
    text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    statuses = _checklist_statuses(text)

    missing = sorted(DEMOTED_TASKS.difference(statuses))
    assert missing == []
    assert {task_id: statuses[task_id] for task_id in sorted(DEMOTED_TASKS)} == {
        task_id: "SCAFFOLD" for task_id in sorted(DEMOTED_TASKS)
    }

    counts = Counter(statuses.values())
    assert counts == Counter(
        {
            "DONE": 10,
            "SCAFFOLD": 30,
            "PROTOTYPE-GATE": 2,
            "IN-PROGRESS": 2,
        }
    )
    assert "VERIFIED" not in counts
    assert "SCAFFOLD" in text
    assert "exact model/variant/weight is actually invoked through a registered `StageRunner`" in text


def test_build_checklist_does_not_repeat_superseded_ball_hard_negative_gap() -> None:
    text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")

    assert "local dataset builders do not consume the hard-negative plan" not in text
    assert "hard-negative materializer" in text
    assert "diagnostic-only rejected" in text


def test_build_checklist_does_not_repeat_superseded_trk_embedding_consumer_gap() -> None:
    text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")

    assert "no current source-selection command consumes them" not in text
    assert "selector/CLI now consumes" in text
    assert "runs/phase2/trk_embedding_source_selection_20260630T223103Z/" in text


def test_ball_local_search_postprocess_rejection_is_documented() -> None:
    checklist_text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    capabilities_text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")

    for text in (checklist_text, capabilities_text):
        assert "runs/ball_local_search_postprocess_20260630T225735Z/" in text
        assert "dense local-search postprocess regresses to F1@20 0.487" in text
        assert "hard-negative local-search postprocess regresses to F1@20 0.502" in text
        assert "local-search postprocess is diagnostic-only and rejected" in text


def test_ball_physics3d_stage_hook_is_documented_without_promotion() -> None:
    checklist_text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    capabilities_text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")

    for text in (checklist_text, capabilities_text):
        assert "`BallStageRunner(ball_physics3d=True)`" in text
        assert "`ball_physics3d_summary.json`" in text
        assert "insufficient_world_xyz_samples" in text
        assert "sample_count=0" in text
        assert "not a BALL/PHYSICS verification gate" in text


def test_active_ball_docs_qualify_physics_targets_and_historical_benchmarks() -> None:
    accuracy_text = (ROOT / "ACCURACY_AND_TRAINING.md").read_text(encoding="utf-8")
    phases_text = (ROOT / "IMPLEMENTATION_PHASES.md").read_text(encoding="utf-8")
    runbook_text = (ROOT / "docs" / "racketsport" / "prototype_gate_h100_v2_usage.md").read_text(encoding="utf-8")
    tech_text = (ROOT / "TECH_STACK.md").read_text(encoding="utf-8")
    wave4_text = (
        ROOT / "runs" / "manager_goal_continuation_20260630_wave4" / "MANAGER_GOAL_CONTINUATION_WAVE4_BLOCKER_ADDENDUM.md"
    ).read_text(encoding="utf-8")

    assert "Target, not current evidence: the current repo has not proven this FP reduction" in accuracy_text
    assert "Current implemented subset:" in phases_text
    assert "`BallStageRunner(ball_physics3d=True)`" in phases_text
    assert "historical pre-A100 CVAT ball benchmark" in runbook_text
    assert "current full-CVAT A100 rerun evaluates all 1524 visible labels" in runbook_text
    assert "manager_goal_continuation_wave4_blocker_addendum.json" in wave4_text


def test_active_body_docs_reflect_latest_reset_bound_bboxscaled_diagnostics_without_promotion() -> None:
    checklist_text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    capabilities_text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")
    phases_text = (ROOT / "IMPLEMENTATION_PHASES.md").read_text(encoding="utf-8")
    runbook_text = (ROOT / "docs" / "racketsport" / "prototype_gate_h100_v2_usage.md").read_text(encoding="utf-8")
    tech_text = (ROOT / "TECH_STACK.md").read_text(encoding="utf-8")
    wave4_text = (
        ROOT / "runs" / "manager_goal_continuation_20260630_wave4" / "MANAGER_GOAL_CONTINUATION_WAVE4_BLOCKER_ADDENDUM.md"
    ).read_text(encoding="utf-8")

    required = (
        "a100_body_video_smoke_burlington_bboxscaled_resetcap075_runtime_20260701T001500Z",
        "a100_body_video_smoke_wolverine_bboxscaled_resetcap075_runtime_20260701T002500Z",
        "0.974265",
        "0.964497",
        "0 selected overlay alignment failures",
        "max track-anchor residual",
        "missing_world_mpjpe_gate",
    )
    for text in (checklist_text, capabilities_text, phases_text, runbook_text, wave4_text):
        for phrase in required:
            assert phrase in text
        assert any(
            phrase in text
            for phrase in (
                "no BODY promotion",
                "not BODY promotion",
                "not BODY verification",
                "does not promote BODY",
                "diagnostic/review evidence, not promotion",
                "before promotion",
                "before using diagnostic BODY runs for canonical promotion",
            )
        )

    for text in (checklist_text, capabilities_text, phases_text, runbook_text):
        assert "body_gate_report_resetcap075.json" in text
        assert "body_world_label_packet.json" in text
        assert "body_joint_quality_from_packet.json" in text
        assert "body_joint_overlay_warning_review_required" in text
        assert "body_world_label_finalization_blocked" in text
        assert "packet quality" in text.lower()
        assert "compact" in text
        assert "blocked_finalization" in text

    for text in (checklist_text, capabilities_text, phases_text, runbook_text, tech_text):
        assert "BODY label review" in text or "body_label_review" in text
        assert "27 / 20 selected" in text or "27 Burlington / 20 Wolverine selected" in text
        assert "0 accepted" in text
        assert "selected_samples_have_overlay_warnings" in text or "selected overlay-warning samples" in text

    for text in (checklist_text, capabilities_text):
        assert "warning_samples" in text
        assert "frame_000047_player_2" in text
        assert "competing_player_warning_count" in text
        assert "not a proven wrong-track assignment" in text


def test_active_handoff_docs_do_not_repeat_superseded_ball_and_body_next_steps() -> None:
    wave4_text = (
        ROOT / "runs" / "manager_goal_continuation_20260630_wave4" / "MANAGER_GOAL_CONTINUATION_WAVE4_BLOCKER_ADDENDUM.md"
    ).read_text(encoding="utf-8")
    runbook_text = (ROOT / "docs" / "racketsport" / "prototype_gate_h100_v2_usage.md").read_text(encoding="utf-8")
    capabilities_text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")

    for text in (wave4_text, runbook_text):
        assert "Add a hard-negative materializer first" not in text
        assert "existing local scripts do not consume the hard-negative plan" not in text
        assert "hard-negative-only and local-search postprocess paths are rejected" in text

    assert "Current synced A100 reset-bound bbox-scaled BODY diagnostics produced review-only Burlington/Wolverine outputs" in wave4_text
    assert "0 selected overlay alignment failures" in wave4_text
    assert "selected overlays still fail alignment" not in wave4_text
    assert "false by default; true only when explicitly configured with TrackNetV3/TOTNet runtime paths" in capabilities_text


def test_capabilities_matrix_records_truth_columns_and_current_spine_limit() -> None:
    capabilities_path = ROOT / "CAPABILITIES.md"
    assert capabilities_path.is_file()

    text = capabilities_path.read_text(encoding="utf-8")
    headers = _first_markdown_table_headers(text)
    assert headers == REQUIRED_CAPABILITY_COLUMNS

    rows = _capability_rows(text)
    assert {"calibration", "tracking", "body", "ball", "racket", "metrics", "replay", "e2e"}.issubset(rows)
    assert "real YOLO26m BoT-SORT-ReID runner is registered" in rows["tracking"]["actually invoked?"]
    assert "precomputed detections remain explicit manual mode" in rows["tracking"]["actually invoked?"]
    assert "consumed by a scored weak-cost diagnostic selector" in rows["tracking"]["actually invoked?"]
    assert "no candidate passes TRK gates" in rows["tracking"]["actually invoked?"]
    assert rows["tracking"]["wired into spine?"] == "yes, default real runner plus explicit precomputed runner"
    assert "scheduled H100 BODY spine runs passed on Burlington, Wolverine, and Indoor" in rows["body"]["actually invoked?"]
    assert "registered `BodyStageRunner` writes BODY contracts with world joints and mesh vertices only from runtime outputs" in rows["body"][
        "wired into spine?"
    ]
    assert "schedule manifests are review/coverage only" in rows["body"]["wired into spine?"]
    assert rows["body"]["gate run on real labels?"] == "no"
    assert rows["tracking"]["gate run on real labels?"] == "no; draft/not_ground_truth labels are refused"
    assert rows["ball"]["gate type (accuracy/presence/none)"] == (
        "presence_check plus optional reviewed-label ball F1 check; held-out prototype benchmark is not an acceptance gate"
    )
    assert rows["e2e"]["honest status"] == "SCAFFOLD/BLOCKED, not PROTOTYPE-GATE"


def test_checklist_ids_are_unique_and_only_operational_count_table_matches_rows() -> None:
    checklist_text = (ROOT / "BUILD_CHECKLIST.md").read_text(encoding="utf-8")
    capabilities_text = (ROOT / "CAPABILITIES.md").read_text(encoding="utf-8")
    rows = _checklist_rows(checklist_text)
    task_ids = [task_id for task_id, _status in rows]

    assert len(task_ids) == 44
    assert sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1) == []

    parsed_counts = Counter(status for _task_id, status in rows)
    assert _visible_status_count_table(checklist_text) == parsed_counts
    assert _visible_status_count_table(capabilities_text) == Counter()


def _checklist_statuses(text: str) -> dict[str, str]:
    return dict(_checklist_rows(text))


def _checklist_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    in_checklist = False
    for line in text.splitlines():
        if line.startswith("### ENV "):
            in_checklist = True
        if line.startswith("## 4. Phase-gate summary"):
            break
        if not in_checklist:
            continue
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        task_id = cells[1].strip("*")
        if "-" not in task_id or task_id == "ID":
            continue
        status = cells[7].strip("*")
        rows.append((task_id, status))
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
