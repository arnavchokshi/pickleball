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

from threed.racketsport.testclips import REQUIRED_LABEL_FILES, TestClipManifest, build_testclip_manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _frame_context(frames_root: Path | None, clip_name: str) -> dict[str, Any]:
    if frames_root is None:
        return {
            "manifest_path": None,
            "frame_count": 0,
            "frames": [],
        }

    clip_frames_dir = frames_root / clip_name
    manifest_path = clip_frames_dir / "label_frame_manifest.json"
    if not manifest_path.is_file():
        return {
            "manifest_path": str(manifest_path),
            "frame_count": 0,
            "frames": [],
        }

    manifest = _read_json(manifest_path)
    frame_names = manifest.get("frames", [])
    if not isinstance(frame_names, list):
        raise ValueError(f"frame manifest has non-list frames field: {manifest_path}")

    frames = [{"name": str(name), "path": str(clip_frames_dir / str(name))} for name in frame_names]
    return {
        "manifest_path": str(manifest_path),
        "frame_count": int(manifest.get("frame_count", len(frames))),
        "sample_every_frames": manifest.get("sample_every_frames"),
        "frames": frames,
    }


def _clip_metadata(clip: TestClipManifest) -> dict[str, Any] | None:
    if clip.metadata is None:
        return None
    return clip.metadata.model_dump(mode="json")


def _draft_template(*, clip: TestClipManifest, label_file: str, frames: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "draft_manual_annotation",
        "clip": {
            "name": clip.name,
            "path": str(clip.path),
            "metadata_path": str(clip.metadata_path),
            "metadata": _clip_metadata(clip),
            "metadata_errors": clip.metadata_errors,
        },
        "frames": frames,
        "annotation": {
            "target_file": label_file,
            "items": [],
            "notes": [],
        },
    }


def _write_clip_drafts(*, clip: TestClipManifest, out: Path, frames_root: Path | None) -> dict[str, Any]:
    draft_labels_dir = out / clip.name / "labels"
    if _resolved(draft_labels_dir) == _resolved(clip.labels_dir):
        raise ValueError(f"refusing to write draft labels into dataset labels path: {clip.labels_dir}")

    draft_labels_dir.mkdir(parents=True, exist_ok=True)
    frames = _frame_context(frames_root, clip.name)
    written: list[str] = []
    for label_file in REQUIRED_LABEL_FILES:
        draft_path = draft_labels_dir / label_file
        draft_path.write_text(
            json.dumps(_draft_template(clip=clip, label_file=label_file, frames=frames), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        written.append(str(draft_path))

    return {
        "clip": clip.name,
        "clip_dir": str(clip.path),
        "dataset_labels_dir": str(clip.labels_dir),
        "draft_labels_dir": str(draft_labels_dir),
        "metadata_present": clip.metadata_present,
        "metadata_errors": clip.metadata_errors,
        "frame_count": frames["frame_count"],
        "written_label_files": written,
    }


def init_label_workdir(*, root: Path, out: Path, frames_root: Path | None = None) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(root)
    manifest = build_testclip_manifest(root)
    clips = [_write_clip_drafts(clip=clip, out=out, frames_root=frames_root) for clip in manifest.clips]
    return {
        "schema_version": 1,
        "root": str(root),
        "out": str(out),
        "frames_root": str(frames_root) if frames_root is not None else None,
        "clip_count": len(clips),
        "draft_ready": True,
        "dataset_labels_written": False,
        "clips": clips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create draft DATA-1 label workdirs outside dataset labels.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/label_drafts"))
    parser.add_argument("--frames-root", type=Path, help="Optional output root from extract_label_frames.py.")
    args = parser.parse_args()

    summary = init_label_workdir(root=args.root, out=args.out, frames_root=args.frames_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
