#!/usr/bin/env python3
"""Build external-only RKT paddle detector/segmenter training configs.

The generated artifacts are CPU prep only. They define deterministic train/val
splits inside the external Roboflow corpus and write an executable remote A100
command packet, but they do not connect to the VM or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval_guard import assert_not_training_on_eval_clip


ARTIFACT_TYPE = "racketsport_rkt_prep_training_configs"
SCHEMA_VERSION = 1
SHARED_BROADCAST_GROUP = "carvana_mesa_johns_staksrud_shared_broadcast"
DEFAULT_MANIFEST = Path("runs/training_corpora_20260701/rkt/manifest.json")
DEFAULT_REMOTE = "arnavchokshi@34.126.67.233"
DEFAULT_REMOTE_REPO = "/home/arnavchokshi/pickleball_git"
DEFAULT_DET_MODEL = "models/checkpoints/yolo26s.pt"
DEFAULT_SEG_MODEL = "yolo26s-seg.pt"


def build_training_configs(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    out_dir: str | Path,
    repo_root: str | Path = Path("."),
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote: str = DEFAULT_REMOTE,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """Write split manifests, YOLO data yamls, and COMMANDS.sh."""

    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    repo_root = Path(repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus") != "RKT":
        raise ValueError(f"expected RKT manifest, got {manifest.get('corpus')!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    det_dir = _resolve_manifest_dir(
        manifest.get("yolo_paddle_detector_corpus", {}).get("output_dir"),
        manifest_path=manifest_path,
    )
    seg_dir = _resolve_manifest_dir(
        manifest.get("yolo_paddle_seg_corpus", {}).get("output_dir"),
        manifest_path=manifest_path,
    )

    det = _write_split_artifacts(
        kind="det",
        task="detect",
        corpus_dir=det_dir,
        out_dir=out_dir / "det",
        repo_root=repo_root,
        remote_repo=remote_repo,
        val_fraction=val_fraction,
        label_count_field="n_paddle_boxes",
    )
    seg = _write_split_artifacts(
        kind="seg",
        task="segment",
        corpus_dir=seg_dir,
        out_dir=out_dir / "seg",
        repo_root=repo_root,
        remote_repo=remote_repo,
        val_fraction=val_fraction,
        label_count_field="n_paddle_polygons",
    )

    commands_path = _write_commands(
        out_dir=out_dir,
        repo_root=repo_root,
        remote=remote,
        remote_repo=remote_repo,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir),
        "policy": {
            "manifest_policy": manifest.get("policy", ""),
            "training_source": "external Roboflow corpus only",
            "eval_clips_used_for_training": False,
            "eval_guard": {
                "det_train": det["eval_guard"]["train"],
                "det_val": det["eval_guard"]["val"],
                "seg_train": seg["eval_guard"]["train"],
                "seg_val": seg["eval_guard"]["val"],
            },
        },
        "det": det,
        "seg": seg,
        "commands_sh": str(commands_path),
        "not_gate_evidence": True,
        "notes": [
            "This prep artifact defines A100-ready commands only; no GPU command was executed.",
            "Detector and segmenter validation splits are inside external data only.",
            "CVAT paddle rectangles remain review/eval-only and are not referenced by these training configs.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _write_split_artifacts(
    *,
    kind: str,
    task: str,
    corpus_dir: Path,
    out_dir: Path,
    repo_root: Path,
    remote_repo: str,
    val_fraction: float,
    label_count_field: str,
) -> dict[str, Any]:
    rows = _load_per_image_rows(corpus_dir)
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        image_path = corpus_dir / str(row["output_image"])
        label_path = corpus_dir / str(row["output_label"])
        if not image_path.is_file():
            raise FileNotFoundError(f"missing corpus image: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"missing corpus label: {label_path}")
        image_rel = _repo_relative(image_path, repo_root)
        label_rel = _repo_relative(label_path, repo_root)
        label_count = _count_label_lines(label_path)
        prepared_rows.append(
            {
                "image_path": image_rel,
                "label_path": label_rel,
                "remote_image_path": _remote_path(remote_repo, image_rel),
                "remote_label_path": _remote_path(remote_repo, label_rel),
                "source_filename": str(row.get("source_filename", "")),
                "source_group": _source_group(row),
                "label_count": label_count,
                label_count_field: int(row.get(label_count_field, label_count) or 0),
            }
        )

    split_by_group = _assign_splits(prepared_rows, val_fraction=val_fraction)
    for row in prepared_rows:
        row["split"] = split_by_group[row["source_group"]]

    train_rows = [row for row in prepared_rows if row["split"] == "train"]
    val_rows = [row for row in prepared_rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError(f"{kind} split must have non-empty train and val rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text(
        "".join(f"{row['remote_image_path']}\n" for row in train_rows),
        encoding="utf-8",
    )
    (out_dir / "val.txt").write_text(
        "".join(f"{row['remote_image_path']}\n" for row in val_rows),
        encoding="utf-8",
    )
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {remote_repo}",
                f"train: {_remote_path(remote_repo, _repo_relative(out_dir / 'train.txt', repo_root))}",
                f"val: {_remote_path(remote_repo, _repo_relative(out_dir / 'val.txt', repo_root))}",
                "nc: 1",
                "names:",
                "  0: paddle",
                "",
            ]
        ),
        encoding="utf-8",
    )

    guard_train = assert_not_training_on_eval_clip(
        [row["image_path"] for row in train_rows] + [row["label_path"] for row in train_rows],
        allow_internal_val=False,
    )
    guard_val = assert_not_training_on_eval_clip(
        [row["image_path"] for row in val_rows] + [row["label_path"] for row in val_rows],
        allow_internal_val=False,
    )

    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_rkt_prep_split_manifest",
        "kind": kind,
        "task": task,
        "corpus_dir": str(corpus_dir),
        "data_yaml": str(data_yaml),
        "train_list": str(out_dir / "train.txt"),
        "val_list": str(out_dir / "val.txt"),
        "split_policy": {
            "method": "deterministic_source_group_hash",
            "val_fraction": val_fraction,
            "source_group_field": "source_group",
            "known_shared_camera_source_group": SHARED_BROADCAST_GROUP,
        },
        "totals": _totals(prepared_rows),
        "splits": {
            "train": _totals(train_rows),
            "val": _totals(val_rows),
        },
        "eval_guard": {
            "train": guard_train,
            "val": guard_val,
        },
        "rows": sorted(prepared_rows, key=lambda item: item["image_path"]),
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "kind": kind,
        "task": task,
        "corpus_dir": str(corpus_dir),
        "data_yaml": str(data_yaml),
        "split_manifest": str(out_dir / "split_manifest.json"),
        "train_list": str(out_dir / "train.txt"),
        "val_list": str(out_dir / "val.txt"),
        "totals": split_manifest["totals"],
        "splits": split_manifest["splits"],
        "eval_guard": split_manifest["eval_guard"],
        "known_shared_camera_source_group": SHARED_BROADCAST_GROUP,
    }


def _assign_splits(rows: Sequence[Mapping[str, Any]], *, val_fraction: float) -> dict[str, str]:
    group_counts: dict[str, int] = {}
    for row in rows:
        group = str(row["source_group"])
        group_counts[group] = group_counts.get(group, 0) + 1
    groups = sorted(group_counts, key=lambda group: _stable_hash(group))
    total = len(rows)
    target_val = max(1, round(total * val_fraction))
    eligible_groups = [group for group in groups if group_counts[group] <= target_val]
    if not eligible_groups:
        eligible_groups = [min(groups, key=lambda group: group_counts[group])]
    val_groups: set[str] = set()
    val_count = 0
    for group in eligible_groups:
        if val_count >= target_val and val_groups:
            break
        val_groups.add(group)
        val_count += group_counts[group]
    if len(val_groups) == len(groups) and len(groups) > 1:
        val_groups.remove(groups[-1])
    return {group: ("val" if group in val_groups else "train") for group in groups}


def _source_group(row: Mapping[str, Any]) -> str:
    raw = _strip_roboflow_suffix(str(row.get("source_filename") or row.get("source_stem") or row.get("output_image", "")))
    dataset = str(row.get("dataset", "pickleball_seg" if str(row.get("output_image", "")).startswith("images/0") else "unknown"))
    lowered = raw.lower()
    if (
        lowered.startswith("championship-ben-johns-vs-federico-staksrud-at-the-carvana-mesa-arizona-open")
        or lowered.startswith("singles-ben-j-1")
        or lowered.startswith("singles-ben-j-2")
        or lowered.startswith("output_frame_file4_")
    ):
        return SHARED_BROADCAST_GROUP
    if "_mp4-" in raw:
        return f"{dataset}:{raw.split('_mp4-', 1)[0]}"
    return f"{dataset}:{raw}"


def _strip_roboflow_suffix(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_jpg.rf.", 1)[0]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_per_image_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    path = corpus_dir / "per_image_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"per-image manifest must be a non-empty list: {path}")
    for row in rows:
        if not isinstance(row, Mapping) or "output_image" not in row or "output_label" not in row:
            raise ValueError(f"invalid per-image row in {path}: {row!r}")
    return [dict(row) for row in rows]


def _write_commands(
    *,
    out_dir: Path,
    repo_root: Path,
    remote: str,
    remote_repo: str,
) -> Path:
    run_dir_rel = _repo_relative(out_dir, repo_root)
    commands = out_dir / "COMMANDS.sh"
    text = f"""#!/usr/bin/env bash
set -euo pipefail

# CPU-safe launcher for the remote A100. This file is syntax-checked locally by
# the RKT-PREP lane; do not run it until the GPU queue owner grants a slot.
SSH_KEY="${{RKT_PREP_SSH_KEY:-$HOME/.ssh/google_compute_engine}}"
REMOTE="${{RKT_PREP_REMOTE:-{remote}}}"
REMOTE_REPO="${{RKT_PREP_REMOTE_REPO:-{remote_repo}}}"
RUN_DIR="{run_dir_rel}"
DET_MODEL="${{RKT_DET_MODEL:-{DEFAULT_DET_MODEL}}}"
SEG_MODEL="${{RKT_SEG_MODEL:-{DEFAULT_SEG_MODEL}}}"

ssh -i "$SSH_KEY" "$REMOTE" \\
  "REMOTE_REPO='$REMOTE_REPO' RUN_DIR='$RUN_DIR' DET_MODEL='$DET_MODEL' SEG_MODEL='$SEG_MODEL' bash -s" <<'REMOTE_RKT_PREP'
set -euo pipefail

cd "$REMOTE_REPO"
git fetch origin main
git checkout main
git pull --ff-only origin main

test -f scripts/gpu-train-lock.sh
test -f "$RUN_DIR/det/data.yaml"
test -f "$RUN_DIR/seg/data.yaml"
python -m py_compile \\
  scripts/racketsport/rkt_prep_build_training_configs.py \\
  scripts/racketsport/rkt_prep_eval_paddle_boxes.py
bash -n "$RUN_DIR/COMMANDS.sh"

scripts/gpu-train-lock.sh bash -lc '
set -euo pipefail
cd "$REMOTE_REPO"
python - <<PY
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ultralytics") else 1)
PY
yolo detect train \\
  model="$DET_MODEL" \\
  data="$RUN_DIR/det/data.yaml" \\
  imgsz=1280 \\
  epochs=100 \\
  batch=-1 \\
  project="$RUN_DIR/a100" \\
  name=det_yolo26_external_split \\
  exist_ok=True
yolo segment train \\
  model="$SEG_MODEL" \\
  data="$RUN_DIR/seg/data.yaml" \\
  imgsz=1280 \\
  epochs=100 \\
  batch=-1 \\
  project="$RUN_DIR/a100" \\
  name=seg_yolo_external_split \\
  exist_ok=True
'
REMOTE_RKT_PREP
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    commands.write_text(text, encoding="utf-8")
    commands.chmod(commands.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return commands


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "image_count": len(rows),
        "label_count": sum(int(row.get("label_count", 0)) for row in rows),
        "source_group_count": len({str(row.get("source_group", "")) for row in rows}),
    }


def _count_label_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _resolve_manifest_dir(raw: Any, *, manifest_path: Path) -> Path:
    if not raw:
        raise ValueError(f"manifest missing output_dir near {manifest_path}")
    path = Path(str(raw))
    if path.is_absolute() or path.exists():
        return path
    for candidate in (manifest_path.parent / path, ROOT / path):
        if candidate.exists():
            return candidate
    return path


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _remote_path(remote_repo: str, repo_rel: str) -> str:
    return f"{remote_repo.rstrip('/')}/{repo_rel.lstrip('/')}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build RKT external-only training configs and A100 commands.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    try:
        summary = build_training_configs(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            remote_repo=args.remote_repo,
            remote=args.remote,
            val_fraction=args.val_fraction,
        )
    except Exception as exc:
        print(f"RKT prep training config build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
