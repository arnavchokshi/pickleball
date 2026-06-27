from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEMOTED_TASKS = {
    "TRK-1",
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
    "RPL-1",
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
            "DONE": 23,
            "SCAFFOLD": 17,
            "PROTOTYPE-GATE": 3,
            "IN-PROGRESS": 1,
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
    assert rows["tracking"]["actually invoked?"] == "precomputed detections only"
    assert rows["tracking"]["wired into spine?"] == "yes, scaffold runner"
    assert rows["body"]["wired into spine?"] == "no"
    assert rows["ball"]["actually invoked?"] == "smoke/probe only"
    assert rows["e2e"]["honest status"] == "BLOCKED past tracking"


def _checklist_statuses(text: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        task_id = cells[1].strip("*")
        if "-" not in task_id or task_id == "ID":
            continue
        statuses[task_id] = cells[7].strip("*")
    return statuses


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
