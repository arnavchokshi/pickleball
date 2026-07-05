"""Convert TrackNet-layout BALL datasets into WASB-SBDT layout."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .eval_guard import assert_not_training_on_eval_clip


ARTIFACT_TYPE = "racketsport_ball_wasb_dataset"
TRACKNET_MANIFEST_JSON = "ball_tracknet_cvat_dataset_manifest.json"
MANIFEST_JSON = "ball_wasb_dataset_manifest.json"
MANIFEST_MD = "ball_wasb_dataset_manifest.md"
DATASET_CONFIG_YAML = "pickleball.yaml"
BLURBALL_DATASET_CONFIG_YAML = "pickleball_blurball.yaml"
BLURBALL_CSV_FILENAME = "Label.csv"
BLURBALL_FRAME_DIGITS = 5
BLURBALL_VISIBLE_FLAGS = (1,)
WASB_FRAME_DIRNAME = "videos"
WASB_CSV_DIRNAME = "pickleball_ball_annotation"
WASB_EXT = ".png"
TRACKNET_COLUMNS = ("Frame", "Visibility", "X", "Y")
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class TrackNetWasbRow:
    frame: int
    visibility: int
    x: float
    y: float


@dataclass(frozen=True)
class TrackNetWasbSample:
    split: str
    clip: str
    source_match: str
    rally_id: str
    csv_path: Path
    frame_dir: Path
    rows: tuple[TrackNetWasbRow, ...]


def build_ball_wasb_dataset(
    *,
    tracknet_root: str | Path,
    out_dir: str | Path,
    allow_internal_val: bool = False,
) -> dict[str, Any]:
    """Build a WASB-SBDT pickleball dataset from a materialized TrackNet root.

    The input must be the output contract from
    ``build_ball_tracknet_cvat_dataset.py``: a manifest plus split/match CSVs,
    frame PNG directories, and ``median.npz`` files. The converter preserves
    TrackNet PNG frames and writes a pickleball dataset class/config that use
    ``ext: '.png'``; no JPEG conversion is needed for WASB's image loader.
    """

    tracknet_base = Path(tracknet_root)
    out = Path(out_dir)
    manifest_path = tracknet_base / TRACKNET_MANIFEST_JSON
    source_manifest = _load_tracknet_manifest(manifest_path)
    guard_summary = _guard_tracknet_manifest(source_manifest, allow_internal_val=allow_internal_val)
    samples = _collect_tracknet_samples(source_manifest, tracknet_base)
    _validate_samples(samples)
    _require_empty_or_missing_out_dir(out)

    out.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    label_counts = {
        "sample_count": 0,
        "frame_count": 0,
        "visible_frame_count": 0,
        "hidden_frame_count": 0,
    }
    matches_by_split: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}
    for sample in samples:
        written = _write_sample(out, sample)
        split_rows.setdefault(sample.split, []).append(written)
        matches_by_split.setdefault(sample.split, [])
        if written["wasb_match"] not in matches_by_split[sample.split]:
            matches_by_split[sample.split].append(written["wasb_match"])
        label_counts["sample_count"] += 1
        label_counts["frame_count"] += len(sample.rows)
        label_counts["visible_frame_count"] += sum(1 for row in sample.rows if row.visibility == 1)
        label_counts["hidden_frame_count"] += sum(1 for row in sample.rows if row.visibility == 0)

    dataset_config = _dataset_config(out, matches_by_split)
    config_path = out / DATASET_CONFIG_YAML
    config_path.write_text(_render_dataset_yaml(dataset_config), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "wasb_dataset_prepared",
        "ball_verified": False,
        "tracknet_root": str(tracknet_base),
        "source_manifest": str(manifest_path),
        "out_dir": str(out),
        "frame_dirname": WASB_FRAME_DIRNAME,
        "csv_dirname": WASB_CSV_DIRNAME,
        "image_extension": WASB_EXT,
        "annotation_format": {
            "file_extension": ".txt",
            "columns": ["x", "y"],
            "delimiter": "space",
            "has_header": False,
        },
        "frame_copy_policy": "preserve_png_no_jpeg_conversion",
        "frame_copy_policy_detail": (
            "The additive pickleball WASB/BlurBall dataset classes read cfg['dataset']['ext']; "
            "the generated config sets ext='.png', so TrackNet PNG frames are copied without JPEG conversion."
        ),
        "hidden_frame_policy": (
            "TrackNet Visibility=0 rows are preserved as WASB no-ball rows with x=0,y=0. "
            "The vendored volleyball-style loader marks rows with x==0 and y==0 as not visible "
            "while still including the frame in training sequences."
        ),
        "blurball_layout": {
            "root_dir": str(out),
            "csv_filename": BLURBALL_CSV_FILENAME,
            "label_columns": ["file name", "visibility", "x-coordinate", "y-coordinate"],
            "image_extension": WASB_EXT,
            "frame_digits": BLURBALL_FRAME_DIGITS,
            "visible_flags": list(BLURBALL_VISIBLE_FLAGS),
            "matches_by_split": {split: list(matches) for split, matches in matches_by_split.items()},
            "layout": "<root>/<match>/<clip>/Label.csv plus zero-padded PNG frames in the same clip directory",
            "recovered_vm_script_alignment": (
                "Matches the recovered VM script's Label.csv column names and zero-padded frame filenames. "
                "Divergence: this general converter preserves the TrackNet manifest split/match/rally ids "
                "instead of creating a contiguous intra-clip train/val split from explicit --pair inputs."
            ),
        },
        "blurball_dataset_config_path": str(out / BLURBALL_DATASET_CONFIG_YAML),
        "blurball_dataset_config": _blurball_dataset_config(out, matches_by_split),
        "eval_guard": guard_summary,
        "dataset_config_path": str(config_path),
        "dataset_config": dataset_config,
        "label_counts": label_counts,
        "splits": split_rows,
        "limitations": [
            "This artifact converts labels and frames only; it does not train a model or verify BALL quality.",
            "Split boundaries are preserved in the generated match ids and config sections; no re-splitting is performed.",
            "Outdoor/Indoor strict holdouts are rejected by eval_guard before any output path is created.",
        ],
    }
    manifest_json = out / MANIFEST_JSON
    manifest_md = out / MANIFEST_MD
    (out / BLURBALL_DATASET_CONFIG_YAML).write_text(
        _render_blurball_dataset_yaml(manifest["blurball_dataset_config"]),
        encoding="utf-8",
    )
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_md.write_text(render_ball_wasb_dataset_markdown(manifest), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_json)
    manifest["manifest_md"] = str(manifest_md)
    return manifest


def render_ball_wasb_dataset_markdown(manifest: Mapping[str, Any]) -> str:
    counts = manifest.get("label_counts", {})
    lines = [
        "# BALL WASB Dataset",
        "",
        f"Status: `{manifest.get('status')}`",
        "",
        "BALL is not verified by this artifact. It only converts a guarded TrackNet-layout label/frame dataset into WASB-SBDT layout.",
        "",
        "## Counts",
        "",
        f"- Samples: {counts.get('sample_count', 0)}",
        f"- Frames: {counts.get('frame_count', 0)}",
        f"- Visible frames: {counts.get('visible_frame_count', 0)}",
        f"- Hidden/no-ball frames: {counts.get('hidden_frame_count', 0)}",
        "",
        "## Layout",
        "",
        f"- Frames: `{manifest.get('frame_dirname')}/<split_match>/<rally>/*.png`",
        f"- Annotations: `{manifest.get('csv_dirname')}/<split_match>/<rally>.txt`",
        "- Annotation rows have no header and contain `x y` space-separated values.",
        "- PNG frames are preserved. No PNG-to-JPEG conversion is performed because the additive pickleball dataset class reads `ext: '.png'` from config.",
        "",
        "## BlurBall Layout",
        "",
        f"- Clips: `<root>/<match>/<clip>/{BLURBALL_CSV_FILENAME}` plus zero-padded PNG frames in the same directory.",
        "- Label rows use `file name,visibility,x-coordinate,y-coordinate`, matching the recovered VM conversion script.",
        "- Divergence from the recovered VM script: this converter preserves source manifest splits instead of making an intra-clip contiguous train/val split.",
        "",
        "## Hidden Frames",
        "",
        "- visibility-0 TrackNet rows are written as 0.000 0.000.",
        "- The vendored volleyball-style WASB loader treats `x == 0 and y == 0` as `is_visible=False` and still includes those frames in sequences.",
        "",
        "## Splits",
        "",
        "| Split | Clip | WASB match | Rally | Frames | Visible | Hidden |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split, rows in manifest.get("splits", {}).items():
        for row in rows:
            lines.append(
                "| {split} | {clip} | {match} | {rally} | {frames} | {visible} | {hidden} |".format(
                    split=split,
                    clip=row.get("clip", ""),
                    match=row.get("wasb_match", ""),
                    rally=row.get("wasb_clip", ""),
                    frames=row.get("frame_count", 0),
                    visible=row.get("visible_frame_count", 0),
                    hidden=row.get("hidden_frame_count", 0),
                )
            )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This is a dataset-glue artifact, not a checkpoint.",
            "- Do not claim BALL promotion from loader success or trainer compatibility.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_tracknet_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing TrackNet dataset manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "racketsport_ball_tracknet_cvat_dataset":
        raise ValueError(f"unexpected TrackNet manifest artifact_type in {path}: {payload.get('artifact_type')!r}")
    if payload.get("tracknet_columns") not in (None, list(TRACKNET_COLUMNS), tuple(TRACKNET_COLUMNS)):
        raise ValueError(f"TrackNet manifest columns must be {TRACKNET_COLUMNS}: {payload.get('tracknet_columns')!r}")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("TrackNet manifest requires non-empty splits object")
    return payload


def _guard_tracknet_manifest(manifest: Mapping[str, Any], *, allow_internal_val: bool) -> dict[str, Any]:
    all_strings_guard = assert_not_training_on_eval_clip([manifest], allow_internal_val=allow_internal_val)
    train_rows = []
    for split, rows in _iter_manifest_split_rows(manifest):
        if split == "train":
            train_rows.extend(rows)
    train_guard = assert_not_training_on_eval_clip(train_rows, allow_internal_val=False)
    return {
        "all_manifest_strings": all_strings_guard,
        "train_rows": train_guard,
        "policy": (
            "all clip ids and paths in the TrackNet manifest are checked before output creation; "
            "train rows are always checked with allow_internal_val=False"
        ),
    }


def _collect_tracknet_samples(manifest: Mapping[str, Any], tracknet_root: Path) -> list[TrackNetWasbSample]:
    samples: list[TrackNetWasbSample] = []
    for split, rows in _iter_manifest_split_rows(manifest):
        for row in rows:
            clip = _required_str(row, "clip")
            source_match = _required_str(row, "match")
            rally_id = _required_str(row, "rally_id")
            csv_path = _resolve_manifest_path(row.get("csv"), tracknet_root)
            frame_dir = _resolve_manifest_path(row.get("frame_dir"), tracknet_root)
            samples.append(
                TrackNetWasbSample(
                    split=split,
                    clip=clip,
                    source_match=source_match,
                    rally_id=rally_id,
                    csv_path=csv_path,
                    frame_dir=frame_dir,
                    rows=tuple(_read_tracknet_rows(csv_path)),
                )
            )
    return samples


def _iter_manifest_split_rows(manifest: Mapping[str, Any]) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("TrackNet manifest splits must be an object")
    seen = set()
    ordered = [split for split in SPLIT_ORDER if split in splits]
    ordered.extend(sorted(str(split) for split in splits.keys() if str(split) not in set(ordered)))
    for split in ordered:
        raw_rows = splits.get(split)
        if split in seen:
            continue
        seen.add(split)
        if not isinstance(raw_rows, list):
            raise ValueError(f"TrackNet manifest split {split!r} must be a list")
        normalized_rows: list[Mapping[str, Any]] = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"TrackNet manifest split {split!r} row {index} must be an object")
            row_split = row.get("split", split)
            if str(row_split) != split:
                raise ValueError(f"TrackNet manifest split mismatch for row {index}: key={split!r} row={row_split!r}")
            normalized_rows.append(row)
        yield split, normalized_rows


def _resolve_manifest_path(value: object, tracknet_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("TrackNet manifest row requires path string")
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return tracknet_root / path


def _read_tracknet_rows(csv_path: Path) -> list[TrackNetWasbRow]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing TrackNet CSV: {csv_path}")
    rows: list[TrackNetWasbRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(TRACKNET_COLUMNS):
            raise ValueError(f"TrackNet CSV columns must be {TRACKNET_COLUMNS}: {csv_path}")
        for index, row in enumerate(reader):
            frame = _int_field(row, "Frame", csv_path, index)
            visibility = _int_field(row, "Visibility", csv_path, index)
            if visibility not in (0, 1):
                raise ValueError(f"TrackNet Visibility must be 0 or 1 in {csv_path} row {index}")
            rows.append(
                TrackNetWasbRow(
                    frame=frame,
                    visibility=visibility,
                    x=_float_field(row, "X", csv_path, index),
                    y=_float_field(row, "Y", csv_path, index),
                )
            )
    if not rows:
        raise ValueError(f"TrackNet CSV has no label rows: {csv_path}")
    return rows


def _validate_samples(samples: Sequence[TrackNetWasbSample]) -> None:
    if not samples:
        raise ValueError("TrackNet manifest did not describe any samples")
    seen_targets: set[tuple[str, str, str]] = set()
    for sample in samples:
        if sample.split != _safe_component(sample.split):
            raise ValueError(f"split contains unsafe path characters: {sample.split!r}")
        if sample.source_match != _safe_component(sample.source_match):
            raise ValueError(f"match contains unsafe path characters: {sample.source_match!r}")
        if sample.rally_id != _safe_component(sample.rally_id):
            raise ValueError(f"rally_id contains unsafe path characters: {sample.rally_id!r}")
        if not sample.frame_dir.is_dir():
            raise FileNotFoundError(f"missing TrackNet frame directory: {sample.frame_dir}")
        if not (sample.frame_dir / "median.npz").is_file():
            raise FileNotFoundError(f"missing TrackNet median.npz: {sample.frame_dir / 'median.npz'}")
        frames = [row.frame for row in sample.rows]
        expected = list(range(len(sample.rows)))
        if frames != expected:
            raise ValueError(f"TrackNet frames must be dense 0..N-1 in {sample.csv_path}: {frames[:5]}...")
        missing_frames = [row.frame for row in sample.rows if not (sample.frame_dir / f"{row.frame}.png").is_file()]
        if missing_frames:
            raise FileNotFoundError(f"missing TrackNet PNG frames in {sample.frame_dir}: {missing_frames[:10]}")
        target_key = (sample.split, _wasb_match(sample), sample.rally_id)
        if target_key in seen_targets:
            raise ValueError(f"duplicate WASB target sample: {target_key}")
        seen_targets.add(target_key)


def _write_sample(out: Path, sample: TrackNetWasbSample) -> dict[str, Any]:
    wasb_match = _wasb_match(sample)
    wasb_clip = sample.rally_id
    frame_out = out / WASB_FRAME_DIRNAME / wasb_match / wasb_clip
    csv_out = out / WASB_CSV_DIRNAME / wasb_match / f"{wasb_clip}.txt"
    blurball_clip_out = out / wasb_match / wasb_clip
    blurball_csv_out = blurball_clip_out / BLURBALL_CSV_FILENAME
    frame_out.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    blurball_clip_out.mkdir(parents=True, exist_ok=True)
    blurball_frame_names: list[str] = []
    for new_index, row in enumerate(sample.rows):
        source_frame = sample.frame_dir / f"{row.frame}.png"
        shutil.copyfile(source_frame, frame_out / f"{row.frame}{WASB_EXT}")
        blurball_frame_name = f"{new_index:0{BLURBALL_FRAME_DIGITS}d}{WASB_EXT}"
        shutil.copyfile(source_frame, blurball_clip_out / blurball_frame_name)
        blurball_frame_names.append(blurball_frame_name)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        for row in sample.rows:
            if row.visibility == 0:
                handle.write("0.000 0.000\n")
            else:
                handle.write(f"{row.x:.3f} {row.y:.3f}\n")
    with blurball_csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file name", "visibility", "x-coordinate", "y-coordinate"])
        for frame_name, row in zip(blurball_frame_names, sample.rows):
            writer.writerow([frame_name, row.visibility, f"{row.x:.3f}", f"{row.y:.3f}"])
    visible = sum(1 for row in sample.rows if row.visibility == 1)
    hidden = len(sample.rows) - visible
    return {
        "split": sample.split,
        "clip": sample.clip,
        "source_match": sample.source_match,
        "source_csv": str(sample.csv_path),
        "source_frame_dir": str(sample.frame_dir),
        "wasb_match": wasb_match,
        "wasb_clip": wasb_clip,
        "wasb_frame_dir": str(frame_out),
        "wasb_annotation": str(csv_out),
        "blurball_clip_dir": str(blurball_clip_out),
        "blurball_label_csv": str(blurball_csv_out),
        "frame_count": len(sample.rows),
        "visible_frame_count": visible,
        "hidden_frame_count": hidden,
    }


def _dataset_config(out: Path, matches_by_split: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "pickleball",
        "root_dir": str(out),
        "frame_dirname": WASB_FRAME_DIRNAME,
        "csv_dirname": WASB_CSV_DIRNAME,
        "ext": WASB_EXT,
    }
    for split in SPLIT_ORDER:
        config[split] = {"matches": list(matches_by_split.get(split, [])), "num_clip_ratio": 1.0}
    for split in sorted(set(matches_by_split) - set(SPLIT_ORDER)):
        config[split] = {"matches": list(matches_by_split.get(split, [])), "num_clip_ratio": 1.0}
    return config


def _blurball_dataset_config(out: Path, matches_by_split: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    test_matches = list(matches_by_split.get("val", [])) + list(matches_by_split.get("test", []))
    return {
        "name": "pickleball",
        "root_dir": str(out),
        "csv_filename": BLURBALL_CSV_FILENAME,
        "ext": WASB_EXT,
        "visible_flags": list(BLURBALL_VISIBLE_FLAGS),
        "train": {
            "matches": list(matches_by_split.get("train", [])),
            "num_clip_ratio": 1.0,
            "refine_npz_path": None,
        },
        "test": {
            "matches": test_matches,
            "num_clip_ratio": 1.0,
            "refine_npz_path": None,
        },
    }


def _render_dataset_yaml(config: Mapping[str, Any]) -> str:
    lines = [
        "# Generated by scripts/racketsport/build_ball_wasb_dataset.py",
        "# This config points at a converted pickleball WASB-SBDT dataset root.",
    ]
    for key in ("name", "root_dir", "frame_dirname", "csv_dirname", "ext"):
        lines.append(f"{key}: {_yaml_scalar(config[key])}")
    for split in [key for key in SPLIT_ORDER if key in config] + [
        key for key in sorted(config) if key not in {*SPLIT_ORDER, "name", "root_dir", "frame_dirname", "csv_dirname", "ext"}
    ]:
        value = config[split]
        if not isinstance(value, Mapping):
            continue
        matches = value.get("matches", [])
        lines.append(f"{split}:")
        lines.append(f"  matches: [{', '.join(_yaml_scalar(item) for item in matches)}]")
        lines.append(f"  num_clip_ratio: {float(value.get('num_clip_ratio', 1.0)):.1f}")
    return "\n".join(lines) + "\n"


def _render_blurball_dataset_yaml(config: Mapping[str, Any]) -> str:
    lines = [
        "# Generated by scripts/racketsport/build_ball_wasb_dataset.py",
        "# This config points at the BlurBall-compatible pickleball layout in the converted dataset root.",
    ]
    for key in ("name", "root_dir", "csv_filename", "ext"):
        lines.append(f"{key}: {_yaml_scalar(config[key])}")
    visible_flags = config.get("visible_flags", [])
    lines.append(f"visible_flags: [{', '.join(str(int(flag)) for flag in visible_flags)}]")
    for split in ("train", "test"):
        value = config[split]
        lines.append(f"{split}:")
        lines.append(f"  matches: [{', '.join(_yaml_scalar(item) for item in value.get('matches', []))}]")
        lines.append(f"  num_clip_ratio: {float(value.get('num_clip_ratio', 1.0)):.1f}")
        refine_npz_path = value.get("refine_npz_path")
        lines.append(f"  refine_npz_path: {_yaml_scalar(refine_npz_path) if refine_npz_path else ''}")
    return "\n".join(lines) + "\n"


def _require_empty_or_missing_out_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"out_dir must be missing or empty to avoid stale WASB dataset files: {out}")


def _wasb_match(sample: TrackNetWasbSample) -> str:
    return f"{sample.split}_{sample.source_match}"


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"TrackNet manifest row requires non-empty {key}")
    return value


def _safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _int_field(row: Mapping[str, str], key: str, csv_path: Path, index: int) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {key!r} in {csv_path} row {index}") from exc


def _float_field(row: Mapping[str, str], key: str, csv_path: Path, index: int) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid float field {key!r} in {csv_path} row {index}") from exc


def _yaml_scalar(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    if all(char.isalnum() or char in "._/-" for char in text):
        return text
    return "'" + text.replace("'", "''") + "'"


__all__ = [
    "ARTIFACT_TYPE",
    "BLURBALL_CSV_FILENAME",
    "BLURBALL_DATASET_CONFIG_YAML",
    "BLURBALL_FRAME_DIGITS",
    "BLURBALL_VISIBLE_FLAGS",
    "DATASET_CONFIG_YAML",
    "MANIFEST_JSON",
    "MANIFEST_MD",
    "TRACKNET_MANIFEST_JSON",
    "WASB_CSV_DIRNAME",
    "WASB_EXT",
    "WASB_FRAME_DIRNAME",
    "TrackNetWasbRow",
    "TrackNetWasbSample",
    "build_ball_wasb_dataset",
    "render_ball_wasb_dataset_markdown",
]
