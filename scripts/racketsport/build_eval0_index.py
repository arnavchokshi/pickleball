#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


INDEX_NAME = "eval0_index.json"
MARKDOWN_NAME = "eval0_index.md"


def build_eval0_index(root: Path) -> dict[str, Any]:
    eval0_root = root / "runs" / "eval0"
    entries = [_summarize_variant_selection(path, root=root) for path in sorted(eval0_root.rglob("variant_selection.json"))]
    status_counts = Counter(entry["approval_status"] for entry in entries)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_eval0_index",
        "eval0_root": _relative_posix(eval0_root, root=root),
        "run_count": len(entries),
        "approval_status_counts": dict(sorted(status_counts.items())),
        "runs": entries,
    }


def write_eval0_index(root: Path, *, markdown: bool = False) -> dict[str, Any]:
    index = build_eval0_index(root)
    eval0_root = root / "runs" / "eval0"
    eval0_root.mkdir(parents=True, exist_ok=True)
    (eval0_root / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown:
        (eval0_root / MARKDOWN_NAME).write_text(render_markdown(index), encoding="utf-8")
    return index


def render_markdown(index: dict[str, Any]) -> str:
    rows = [
        "# EVAL-0 Variant Selection Index",
        "",
        f"- run_count: {index['run_count']}",
        "- manifest_update: models/MANIFEST.json is not read or modified",
        "",
        "| Stage | Status | Selected Candidate | Candidates | Overlays | Metric Paths | Artifact |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for entry in index["runs"]:
        metric_paths = ", ".join(_md_code(path) for path in entry["metric_paths"]) or "n/a"
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_text(entry["stage"]),
                    f"approval_status: {_md_text(entry['approval_status'])}",
                    _md_text(entry["selected_candidate"] if entry["selected_candidate"] is not None else "n/a"),
                    str(entry["candidate_count"]),
                    str(entry["overlay_count"]),
                    metric_paths,
                    _md_code(entry["variant_selection_path"]),
                ]
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _summarize_variant_selection(path: Path, *, root: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    candidates = _candidates(payload, path)
    candidate_count = _candidate_count(payload, candidates, path)
    metric_paths = _unique_paths(_collect_paths(payload, candidates, names={"metric_path", "metrics_path", "metric_paths"}))
    overlay_count = len(_unique_paths(_collect_paths(payload, candidates, names={"overlay_path", "overlay_paths"})))

    return {
        "variant_selection_path": _relative_posix(path, root=root),
        "run_dir": _relative_posix(path.parent, root=root),
        "stage": _stage(payload, candidates, path),
        "approval_status": _required_str(payload, "approval_status", path),
        "selected_candidate": _selected_candidate(payload.get("selected_candidate"), path),
        "candidate_count": candidate_count,
        "overlay_count": overlay_count,
        "metric_paths": metric_paths,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: variant_selection must be a JSON object")
    return payload


def _candidates(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    value = payload.get("candidates", [])
    if not isinstance(value, list):
        raise ValueError(f"{path}: candidates must be a list")
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            raise ValueError(f"{path}: candidates[{index}] must be an object")
        candidates.append(candidate)
    return candidates


def _candidate_count(payload: dict[str, Any], candidates: list[dict[str, Any]], path: Path) -> int:
    value = payload.get("candidate_count", len(candidates))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: candidate_count must be a non-negative integer")
    if "candidate_count" in payload and value != len(candidates):
        raise ValueError(f"{path}: candidate_count does not match candidates length")
    return value


def _stage(payload: dict[str, Any], candidates: list[dict[str, Any]], path: Path) -> str:
    value = payload.get("stage")
    if isinstance(value, str) and value.strip():
        return value.strip()

    stages = {candidate.get("stage").strip() for candidate in candidates if isinstance(candidate.get("stage"), str) and candidate.get("stage").strip()}
    if len(stages) == 1:
        return next(iter(stages))
    if len(stages) > 1:
        return "mixed"
    raise ValueError(f"{path}: stage is required at top level or on candidates")


def _required_str(payload: dict[str, Any], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _selected_candidate(value: Any, path: Path) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        selected = value.strip()
        return selected or None
    if isinstance(value, dict):
        for field in ("variant_id", "id", "candidate_id"):
            candidate_id = value.get(field)
            if isinstance(candidate_id, str) and candidate_id.strip():
                return candidate_id.strip()
    raise ValueError(f"{path}: selected_candidate must be null, a string, or an object with variant_id")


def _collect_paths(payload: dict[str, Any], candidates: list[dict[str, Any]], *, names: set[str]) -> list[str]:
    paths: list[str] = []
    for source in [payload, *candidates]:
        for name in names:
            _append_path_values(paths, source.get(name))
    return paths


def _append_path_values(paths: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            paths.append(value.strip())
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                paths.append(item.strip())


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _relative_posix(path: Path, *, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _md_code(value: Any) -> str:
    escaped = str(value).replace("`", "\\`").replace("|", "\\|")
    return f"`{escaped}`"


def _md_text(value: Any) -> str:
    return str(value).replace("|", "\\|")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only index over EVAL-0 variant selection artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository or artifact root containing runs/eval0.")
    parser.add_argument("--markdown", action="store_true", help="Also write runs/eval0/eval0_index.md.")
    args = parser.parse_args()

    try:
        write_eval0_index(args.root, markdown=args.markdown)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
