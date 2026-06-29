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


def test_capabilities_matrix_records_truth_columns_and_current_spine_limit() -> None:
    capabilities_path = ROOT / "CAPABILITIES.md"
    assert capabilities_path.is_file()

    text = capabilities_path.read_text(encoding="utf-8")
    headers = _first_markdown_table_headers(text)
    assert headers == REQUIRED_CAPABILITY_COLUMNS

    rows = _capability_rows(text)
    assert {"calibration", "tracking", "body", "ball", "racket", "metrics", "replay", "e2e"}.issubset(rows)
    assert rows["tracking"]["actually invoked?"] == (
        "real YOLO26m BoT-SORT-ReID runner is registered; precomputed detections remain explicit manual mode"
    )
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
