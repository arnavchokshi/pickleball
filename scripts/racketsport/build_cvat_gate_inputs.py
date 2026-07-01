#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.cvat_gate_inputs import (  # noqa: E402
    CvatGateClipSpec,
    canonical_data1_cvat_clip_specs,
    write_cvat_gate_input_package,
    write_data1_substitute_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local gate-input label payloads from reviewed CVAT boxes.")
    parser.add_argument("--clip", action="append", default=[], help="clip_id=reviewed_boxes.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--data1-substitute-out-dir", type=Path)
    parser.add_argument("--cvat-upload-root", type=Path, default=Path("cvat_upload"))
    parser.add_argument("--imports-root", type=Path, default=Path("runs/cvat_imports/2026_06_30"))
    parser.add_argument("--data-testclips-root", type=Path, default=Path("data/testclips"))
    args = parser.parse_args()

    try:
        if args.data1_substitute_out_dir is not None:
            manifest = write_data1_substitute_package(
                clips=canonical_data1_cvat_clip_specs(
                    cvat_upload_root=args.cvat_upload_root,
                    imports_root=args.imports_root,
                ),
                out_dir=args.data1_substitute_out_dir,
                data_testclips_root=args.data_testclips_root,
                detector_gate_inputs_root=args.imports_root / "gate_inputs",
            )
            print(json.dumps(_compact_data1_manifest(manifest), sort_keys=True))
            return 0
        if not args.clip or args.out_dir is None:
            raise ValueError("--clip and --out-dir are required unless --data1-substitute-out-dir is set")
        manifest = write_cvat_gate_input_package(clips=_parse_clips(args.clip), out_dir=args.out_dir)
    except Exception as exc:
        print(f"CVAT gate-input build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_compact_manifest(manifest), sort_keys=True))
    return 0


def _parse_clips(specs: Sequence[str]) -> list[CvatGateClipSpec]:
    clips: list[CvatGateClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"clip spec must be clip_id=reviewed_boxes: {spec}")
        clip_id, reviewed = parts
        path = Path(reviewed)
        if not path.is_file():
            raise FileNotFoundError(f"missing reviewed boxes for {clip_id}: {path}")
        clips.append(CvatGateClipSpec(clip_id=clip_id, reviewed_boxes_path=path))
    return clips


def _compact_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_type": manifest.get("artifact_type"),
        "clip_count": manifest.get("clip_count"),
        "out_dir": manifest.get("out_dir"),
        "datasets": {
            name: {
                "item_count": dataset.get("item_count"),
                "label_counts_by_name": dataset.get("label_counts_by_name"),
                "target_file": dataset.get("target_file"),
            }
            for name, dataset in (manifest.get("datasets") or {}).items()  # type: ignore[union-attr]
            if isinstance(dataset, dict)
        },
    }


def _compact_data1_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_type": manifest.get("artifact_type"),
        "canonical_clip_count": manifest.get("canonical_clip_count"),
        "data1_ready": manifest.get("data1_ready"),
        "status": manifest.get("status"),
        "summary": manifest.get("summary"),
        "missing_inputs_report": manifest.get("missing_inputs_report"),
        "markdown_report": manifest.get("markdown_report"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
