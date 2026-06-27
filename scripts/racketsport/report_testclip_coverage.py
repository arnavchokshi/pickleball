#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.testclips import TestClipManifest, build_testclip_manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_pack(frames_root: Path | None, clip_name: str) -> dict[str, Any]:
    if frames_root is None:
        return {
            "manifest_path": None,
            "present": False,
            "frame_count": 0,
            "sample_every_frames": None,
        }

    manifest_path = frames_root / clip_name / "label_frame_manifest.json"
    if not manifest_path.is_file():
        return {
            "manifest_path": str(manifest_path),
            "present": False,
            "frame_count": 0,
            "sample_every_frames": None,
        }

    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"frame manifest has non-list frames field: {manifest_path}")

    return {
        "manifest_path": str(manifest_path),
        "present": True,
        "frame_count": int(manifest.get("frame_count", len(frames))),
        "sample_every_frames": manifest.get("sample_every_frames"),
    }


def _clip_metadata(clip: TestClipManifest) -> dict[str, Any] | None:
    if clip.metadata is None:
        return None
    return clip.metadata.model_dump(mode="json")


def _clip_report(clip: TestClipManifest, frames_root: Path | None) -> dict[str, Any]:
    frame_pack = _frame_pack(frames_root, clip.name)
    return {
        "name": clip.name,
        "path": str(clip.path),
        "ready": clip.is_ready,
        "metadata_present": clip.metadata_present,
        "metadata_errors": clip.metadata_errors,
        "metadata": _clip_metadata(clip),
        "present_label_files": clip.present_label_files,
        "missing_label_files": clip.missing_label_files,
        "missing_label_count": len(clip.missing_label_files),
        "frame_pack": frame_pack,
    }


def build_coverage_report(*, root: Path, frames_root: Path | None = None) -> dict[str, Any]:
    manifest = build_testclip_manifest(root)
    clip_reports = [_clip_report(clip, frames_root) for clip in manifest.clips]
    total_frames = sum(clip["frame_pack"]["frame_count"] for clip in clip_reports)
    clips_with_frame_packs = sum(1 for clip in clip_reports if clip["frame_pack"]["present"])

    return {
        "schema_version": 1,
        "root": str(root),
        "root_exists": manifest.root_exists,
        "frames_root": str(frames_root) if frames_root is not None else None,
        "ready": manifest.dataset_ready,
        "counts": {
            "total_clips": manifest.total_clips,
            "metadata_ready_clips": manifest.metadata_ready_clips,
            "ready_clips": manifest.ready_clips,
            "not_ready_clips": manifest.not_ready_clips,
        },
        "label_readiness": {
            "ready": manifest.is_ready,
            "required_label_files": list(manifest.required_label_files),
            "label_file_counts": manifest.label_file_counts,
            "ready_clips": manifest.ready_clips,
            "not_ready_clips": manifest.not_ready_clips,
        },
        "matrix": {
            "ready": manifest.meets_dataset_matrix,
            "coverage_counts": manifest.coverage_counts,
            "missing_coverage": manifest.coverage_gaps,
        },
        "frame_packs": {
            "clips_with_frame_packs": clips_with_frame_packs,
            "clips_missing_frame_packs": manifest.total_clips - clips_with_frame_packs,
            "total_frames": total_frames,
        },
        "clips": {clip["name"]: clip for clip in clip_reports},
    }


def _markdown_bool(value: bool) -> str:
    return "true" if value else "false"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# DATA-1 Test Clip Coverage Triage",
        "",
        "## Current Counts",
        "",
        f"- Root: `{report['root']}`",
        f"- Frames root: `{report['frames_root']}`",
        f"- Ready: {_markdown_bool(report['ready'])}",
        f"- Total clips: {report['counts']['total_clips']}",
        f"- Metadata-ready clips: {report['counts']['metadata_ready_clips']}",
        f"- Label-ready clips: {report['label_readiness']['ready_clips']}",
        f"- Not label-ready clips: {report['label_readiness']['not_ready_clips']}",
        f"- Frame-pack clips: {report['frame_packs']['clips_with_frame_packs']}",
        f"- Frame-pack frames: {report['frame_packs']['total_frames']}",
        "",
        "## Missing Matrix Coverage",
        "",
    ]

    gaps = report["matrix"]["missing_coverage"]
    if gaps:
        lines.extend(f"- {gap}" for gap in gaps)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Label Readiness",
            "",
            "| Label file | Clips present |",
            "| --- | ---: |",
        ]
    )
    for label, count in report["label_readiness"]["label_file_counts"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Frame Packs",
            "",
            "| Clip | Ready | Missing labels | Frames |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for clip in report["clips"].values():
        lines.append(
            "| "
            f"{clip['name']} | "
            f"{_markdown_bool(clip['ready'])} | "
            f"{clip['missing_label_count']} | "
            f"{clip['frame_pack']['frame_count']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report DATA-1 test-clip coverage and label triage gaps.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--frames-root", type=Path, help="Optional output root from extract_label_frames.py.")
    parser.add_argument("--markdown-out", type=Path, help="Optional path to write a Markdown triage report.")
    args = parser.parse_args()

    try:
        report = build_coverage_report(root=args.root, frames_root=args.frames_root)
    except ValueError as exc:
        parser.error(str(exc))

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown_report(report), encoding="utf-8")
        report = report | {"markdown_report": str(args.markdown_out)}

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
