#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPORT_NAME = "variant_selection.md"
SUMMARY_NAME = "variant_selection.json"
PENDING_STATUS = "pending"
AUTO_FINALIZED_STATUS = "auto_finalized_obvious"


@dataclass(frozen=True)
class VariantCandidate:
    variant_id: str
    stage: str
    clip_id: str
    overlay_path: str
    accuracy_metric: Any
    latency_ms: float | int | None
    vram_gb: float | int | None
    notes: list[str]


def load_candidates(path: Path) -> list[VariantCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of candidate objects")
    if not payload:
        raise ValueError(f"{path} must contain at least one candidate object")

    candidates: list[VariantCandidate] = []
    for index, candidate in enumerate(payload):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate[{index}] must be an object")
        candidates.append(_parse_candidate(candidate, index=index))
    return candidates


def build_summary(
    candidates: list[VariantCandidate],
    *,
    approval_status: str = PENDING_STATUS,
    finalized_reason: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_variant_selection_comparison",
        "approval_status": approval_status,
        "candidate_count": len(candidates),
        "manifest_update": "not_written",
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    if finalized_reason is not None:
        summary["finalized_reason"] = finalized_reason
    return summary


def write_comparison_artifacts(
    *,
    candidates_path: Path,
    out_dir: Path,
    auto_finalized_obvious: bool = False,
    finalized_reason: str | None = None,
) -> dict[str, Any]:
    candidates = load_candidates(candidates_path)
    approval_status = AUTO_FINALIZED_STATUS if auto_finalized_obvious else PENDING_STATUS
    summary = build_summary(candidates, approval_status=approval_status, finalized_reason=finalized_reason)
    markdown = render_markdown(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / REPORT_NAME).write_text(markdown, encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# Variant Selection Comparison",
        "",
        f"- approval_status: {summary['approval_status']}",
        f"- candidate_count: {summary['candidate_count']}",
        "- manifest_update: not_written",
        "",
        "This report does not approve or lock any model variant.",
        "It does not update models/MANIFEST.json.",
        "",
    ]

    finalized_reason = summary.get("finalized_reason")
    if finalized_reason:
        rows.extend(["## Auto-finalized Reason", "", str(finalized_reason), ""])

    rows.extend(
        [
            "## Candidates",
            "",
            "| Variant | Stage | Clip | Overlay | Accuracy | Latency ms | VRAM GB | Notes |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for candidate in summary["candidates"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_code(candidate["variant_id"]),
                    _md_text(candidate["stage"]),
                    _md_code(candidate["clip_id"]),
                    _md_code(candidate["overlay_path"]),
                    _md_metric(candidate["accuracy_metric"]),
                    _md_metric(candidate["latency_ms"]),
                    _md_metric(candidate["vram_gb"]),
                    _md_text("; ".join(candidate["notes"])),
                ]
            )
            + " |"
        )
    rows.append("")
    return "\n".join(rows)


def _parse_candidate(candidate: dict[str, Any], *, index: int) -> VariantCandidate:
    return VariantCandidate(
        variant_id=_required_str(candidate, "variant_id", index=index),
        stage=_required_str(candidate, "stage", index=index),
        clip_id=_required_str(candidate, "clip_id", index=index),
        overlay_path=_safe_relative_path(_required_str(candidate, "overlay_path", index=index), index=index),
        accuracy_metric=_required_value(candidate, "accuracy_metric", index=index),
        latency_ms=_optional_number(candidate.get("latency_ms"), field="latency_ms", index=index),
        vram_gb=_optional_number(candidate.get("vram_gb"), field="vram_gb", index=index),
        notes=_notes(candidate.get("notes")),
    )


def _required_value(candidate: dict[str, Any], field: str, *, index: int) -> Any:
    if field not in candidate:
        raise ValueError(f"candidate[{index}].{field} is required")
    return candidate[field]


def _required_str(candidate: dict[str, Any], field: str, *, index: int) -> str:
    value = _required_value(candidate, field, index=index)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate[{index}].{field} must be a non-empty string")
    return value.strip()


def _optional_number(value: Any, *, field: str, index: int) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate[{index}].{field} must be a number when provided")
    return value


def _notes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _safe_relative_path(value: str, *, index: int) -> str:
    if "\x00" in value or "\\" in value or ":" in value:
        raise ValueError(f"candidate[{index}].overlay_path is an unsafe relative path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"candidate[{index}].overlay_path is an unsafe relative path: {value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"candidate[{index}].overlay_path is an unsafe relative path: {value}")
    return path.as_posix()


def _md_code(value: Any) -> str:
    escaped = str(value).replace("`", "\\`").replace("|", "\\|")
    return f"`{escaped}`"


def _md_text(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _md_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return _md_text(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an EVAL-0 variant comparison report without approving or locking a model variant."
    )
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list of candidate run artifacts/metrics.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for variant_selection.md/json outputs.")
    parser.add_argument(
        "--auto-finalized-obvious",
        action="store_true",
        help="Mark report status auto_finalized_obvious. Requires --finalized-reason.",
    )
    parser.add_argument("--finalized-reason", help="Required reason when --auto-finalized-obvious is set.")
    args = parser.parse_args()

    if args.auto_finalized_obvious and not args.finalized_reason:
        parser.error("--finalized-reason is required when --auto-finalized-obvious is set.")
    if args.finalized_reason is not None and not args.finalized_reason.strip():
        parser.error("--finalized-reason must be non-empty.")
    if not args.auto_finalized_obvious and args.finalized_reason:
        parser.error("--finalized-reason may only be used with --auto-finalized-obvious.")

    try:
        write_comparison_artifacts(
            candidates_path=args.candidates,
            out_dir=args.out_dir,
            auto_finalized_obvious=args.auto_finalized_obvious,
            finalized_reason=args.finalized_reason.strip() if args.finalized_reason else None,
        )
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
