from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
HELD_OUT_SOURCE_IDS = {"pwxNwFfYQlQ", "vQhtz8l6VqU"}
CANONICAL_KEYPOINTS = (
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
    "far_right_corner",
    "far_baseline_center",
    "far_left_corner",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
)
COURT_TERMS = ("court", "keypoint", "line", "net", "kitchen", "surface")
NET_KEYPOINTS = {"net_left_sideline", "net_center", "net_right_sideline"}
PLANAR_KEYPOINTS = set(CANONICAL_KEYPOINTS) - NET_KEYPOINTS
EXTERNAL_DATASET_STATUS = "reviewed_external_dataset"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def manifest_entries(dataset_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(dataset_root / "manifest.json")
    entries: dict[str, dict[str, Any]] = {}
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict) or entry.get("status") != "downloaded":
            continue
        local_path = entry.get("local_path")
        if isinstance(local_path, str) and local_path:
            entries[Path(local_path).name] = entry
    return entries


def is_court_related(name: str, entry: dict[str, Any]) -> bool:
    if entry.get("content_category_guess") == "2_court":
        return True
    text = " ".join([name, str(entry.get("project", "")), *[str(v) for v in entry.get("classes", [])]]).lower()
    return any(term in text for term in COURT_TERMS)


def split_payloads(dataset_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = dataset_dir / split / "_annotations.coco.json"
        if path.is_file():
            result[split] = (path, read_json(path))
    return result


def category_schema(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for payload in payloads:
        categories = {int(cat["id"]): cat for cat in payload.get("categories", []) if "id" in cat}
        counts = Counter(int(ann.get("category_id", -1)) for ann in payload.get("annotations", []))
        for cat_id, cat in categories.items():
            keypoints = tuple(str(name) for name in cat.get("keypoints", []))
            key = (str(cat.get("name", "")), keypoints)
            row = merged.setdefault(
                key,
                {
                    "name": key[0],
                    "keypoint_names": list(keypoints),
                    "keypoint_count": len(keypoints),
                    "skeleton": cat.get("skeleton", []),
                    "annotation_count": 0,
                },
            )
            row["annotation_count"] += counts[cat_id]
    return sorted(merged.values(), key=lambda row: (row["name"], row["keypoint_names"]))


def task_type(payloads: Iterable[dict[str, Any]]) -> str:
    annotations = [ann for payload in payloads for ann in payload.get("annotations", [])]
    if any(ann.get("keypoints") for ann in annotations):
        return "keypoints"
    if any(ann.get("segmentation") for ann in annotations):
        return "segmentation"
    if any(ann.get("bbox") for ann in annotations):
        return "bbox"
    return "unknown"


def resolution_stats(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sizes = [
        (int(image["width"]), int(image["height"]))
        for payload in payloads
        for image in payload.get("images", [])
        if isinstance(image.get("width"), int) and isinstance(image.get("height"), int)
    ]
    if not sizes:
        return {"count": 0, "distinct": [], "width": None, "height": None}
    widths = sorted(width for width, _ in sizes)
    heights = sorted(height for _, height in sizes)
    return {
        "count": len(sizes),
        "distinct": [{"width": width, "height": height, "count": count} for (width, height), count in Counter(sizes).most_common()],
        "width": {"min": widths[0], "median": statistics.median(widths), "max": widths[-1]},
        "height": {"min": heights[0], "median": statistics.median(heights), "max": heights[-1]},
    }


def split_image_records(dataset_dir: Path, payloads: dict[str, tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, (annotation_path, payload) in payloads.items():
        for image in payload.get("images", []):
            path = annotation_path.parent / str(image.get("file_name", ""))
            if path.is_file():
                rows.append({"split": split, "path": path, "image": image, "payload": payload})
    return rows


def commercial_license(license_name: str) -> tuple[bool, str]:
    normalized = license_name.upper().replace("CREATIVE COMMONS ", "")
    if "NC" in normalized or "NONCOMMERCIAL" in normalized:
        return False, "research-only; excluded from default corpus"
    if normalized in {"CC BY 4.0", "PUBLIC DOMAIN", "MIT"}:
        return True, "commercial use allowed subject to the recorded license and attribution terms"
    return False, "unknown commercial terms; quarantined by default"


def annotation_index(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    images = {int(image["id"]): image for image in payload.get("images", [])}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        annotations[int(ann.get("image_id", -1))].append(ann)
    categories = {int(cat["id"]): cat for cat in payload.get("categories", []) if "id" in cat}
    return images, annotations, categories


def draw_sample(
    image_path: Path,
    annotations: list[dict[str, Any]],
    categories: dict[int, dict[str, Any]],
    mapping: dict[str, str],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1, round(max(image.size) / 700))
    for ann in annotations:
        category = categories.get(int(ann.get("category_id", -1)), {})
        names = [str(name) for name in category.get("keypoints", [])]
        points = ann.get("keypoints", [])
        for index in range(min(len(names), len(points) // 3)):
            x, y, visibility = points[index * 3 : index * 3 + 3]
            if float(visibility) <= 0:
                continue
            source_name = names[index]
            canonical = mapping.get(source_name)
            label = canonical or source_name
            radius = 4 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ff3344", outline="white", width=scale)
            draw.text((x + radius + scale, y - radius), label, fill="#ffe45c", stroke_width=scale, stroke_fill="black")
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if isinstance(polygon, list) and len(polygon) >= 6:
                    draw.line([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)], fill="#ff3344", width=2 * scale, joint="curve")
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and not points and not segmentation:
            x, y, width, height = [float(value) for value in bbox]
            draw.rectangle((x, y, x + width, y + height), outline="#ff3344", width=2 * scale)
    image.thumbnail((1400, 1000))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def render_dataset_samples(
    dataset_dir: Path,
    payloads: dict[str, tuple[Path, dict[str, Any]]],
    mapping_row: dict[str, Any],
    output_dir: Path,
    limit: int = 5,
) -> list[str]:
    rendered: list[str] = []
    mapping = {str(key): str(value) for key, value in mapping_row.get("keypoint_mapping", {}).items()}
    for split, (annotation_path, payload) in payloads.items():
        images, annotations, categories = annotation_index(payload)
        for image_id in sorted(images):
            image_path = annotation_path.parent / str(images[image_id].get("file_name", ""))
            if not image_path.is_file():
                continue
            output_path = output_dir / dataset_dir.name / f"sample_{len(rendered) + 1:02d}.jpg"
            draw_sample(image_path, annotations.get(image_id, []), categories, mapping, output_path)
            rendered.append(str(output_path))
            if len(rendered) == limit:
                return rendered
    return rendered


def sample_overlap_report(dataset_images: dict[str, list[Path]], sample_size: int) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    for dataset, paths in dataset_images.items():
        if len(paths) <= sample_size:
            sampled = paths
        else:
            step = (len(paths) - 1) / float(sample_size - 1)
            sampled = [paths[round(index * step)] for index in range(sample_size)]
        for path in sampled:
            try:
                by_hash[sha256(path)].append({"dataset": dataset, "path": str(path)})
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
    duplicates = [rows for rows in by_hash.values() if len({row["dataset"] for row in rows}) > 1]
    return {
        "sample_size_per_dataset": sample_size,
        "hashes_checked": len(by_hash),
        "cross_dataset_exact_duplicate_groups": duplicates,
        "errors": errors,
    }


def audit_datasets(dataset_root: Path, mapping_payload: dict[str, Any], lane_dir: Path, sample_size: int) -> dict[str, Any]:
    entries = manifest_entries(dataset_root)
    mapping_rows = mapping_payload.get("datasets", {})
    audited: list[dict[str, Any]] = []
    dataset_images: dict[str, list[Path]] = {}
    for name, entry in sorted(entries.items()):
        if not is_court_related(name, entry):
            continue
        dataset_dir = dataset_root / name
        payloads = split_payloads(dataset_dir)
        payload_values = [payload for _, payload in payloads.values()]
        mapping_row = mapping_rows.get(name, {})
        images = split_image_records(dataset_dir, payloads)
        dataset_images[name] = [row["path"] for row in images]
        overlays = render_dataset_samples(dataset_dir, payloads, mapping_row, lane_dir / "mapping_overlays")
        schema = category_schema(payload_values)
        audited.append(
            {
                "dir": name,
                "slug": entry.get("slug"),
                "task_type": task_type(payload_values),
                "category_count": len(schema),
                "annotation_schema": schema,
                "image_counts_per_split": {
                    split: len(payload.get("images", [])) for split, (_, payload) in payloads.items()
                },
                "resolution_stats": resolution_stats(payload_values),
                "license_as_recorded": entry.get("license_as_recorded", ""),
                "viewpoint_character": mapping_row.get("viewpoint_character", ["unclassified"]),
                "viewpoint_sample_size": len(overlays),
                "mapping_overlay_paths": overlays,
                "usability_verdict": mapping_row.get("verdict", "unusable_ambiguous"),
                "mapping_confidence": mapping_row.get("mapping_confidence", "none"),
                "mapping_rationale": mapping_row.get("mapping_rationale", "No explicit mapping table was supplied."),
                "canonical_keypoints_mapped": sorted(set(mapping_row.get("keypoint_mapping", {}).values())),
                "source_group": mapping_row.get("source_group", name),
                "human_annotated": bool(mapping_row.get("human_annotated", False)),
                "include_default": bool(mapping_row.get("include_default", False)),
            }
        )
    report = {
        "schema_version": 1,
        "dataset_manifest_downloaded_count": len(entries),
        "filesystem_dataset_dir_count": len([path for path in dataset_root.iterdir() if path.is_dir() and path.name != "aggregated"]),
        "search_terms": list(COURT_TERMS),
        "court_related_dir_count": len(audited),
        "datasets": audited,
        "sample_overlap_check": sample_overlap_report(dataset_images, sample_size),
    }
    write_json(lane_dir / "court_dataset_audit.json", report)
    lines = [
        "# Roboflow real court dataset audit",
        "",
        f"Scanned {len(entries)} locally downloaded manifest datasets and found {len(audited)} court/theme-related directories.",
        "",
        "| Dataset | Task | Split images | Resolution | License | Viewpoint | Verdict | Mapped | Confidence |",
        "|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for row in audited:
        split_counts = ", ".join(f"{key}:{value}" for key, value in row["image_counts_per_split"].items())
        res = row["resolution_stats"]
        resolution = "n/a" if not res["count"] else f"{res['width']['min']}-{res['width']['max']}x{res['height']['min']}-{res['height']['max']}"
        lines.append(
            f"| `{row['dir']}` | {row['task_type']} | {split_counts} | {resolution} | {row['license_as_recorded']} | "
            f"{', '.join(row['viewpoint_character'])} | {row['usability_verdict']} | {len(row['canonical_keypoints_mapped'])} | {row['mapping_confidence']} |"
        )
    (lane_dir / "COURT_DATASET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def visible_keypoints(ann: dict[str, Any], category: dict[str, Any]) -> dict[str, list[float]]:
    names = [str(name) for name in category.get("keypoints", [])]
    values = ann.get("keypoints", [])
    result: dict[str, list[float]] = {}
    for index, name in enumerate(names):
        if index * 3 + 2 >= len(values):
            break
        x, y, visibility = values[index * 3 : index * 3 + 3]
        if float(visibility) > 0 and math.isfinite(float(x)) and math.isfinite(float(y)):
            result[name] = [float(x), float(y)]
    return result


def guard_hashes(roots: list[Path]) -> tuple[set[str], int]:
    hashes: set[str] = set()
    count = 0
    for root in roots:
        for path in image_files(root):
            hashes.add(sha256(path))
            count += 1
    return hashes, count


def reused_guard_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(path)
    required = {"guard_image_files_hashed", "guard_unique_hashes", "guard_hash_matches_included"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"reused guard report is missing required fields: {', '.join(missing)}")
    if int(payload["guard_hash_matches_included"]) != 0:
        raise ValueError("reused guard report is not green")
    return {
        "artifact": str(path),
        "artifact_sha256": sha256(path),
        "guard_image_files_hashed": int(payload["guard_image_files_hashed"]),
        "guard_unique_hashes": int(payload["guard_unique_hashes"]),
        "guard_hash_matches_included": int(payload["guard_hash_matches_included"]),
    }


def git_index_image_oids(prefixes: list[str]) -> tuple[set[str], int]:
    """Read protected image blob identities from Git's index without opening protected files."""
    if not prefixes:
        return set(), 0
    completed = subprocess.run(
        ["git", "ls-files", "-s", "-z", "--", *prefixes],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    oids: set[str] = set()
    count = 0
    for record in completed.stdout.split("\0"):
        if not record:
            continue
        metadata, path_text = record.split("\t", 1)
        if Path(path_text).suffix.lower() not in IMAGE_SUFFIXES or "/labels/" not in f"/{path_text}":
            continue
        fields = metadata.split()
        if len(fields) < 2:
            raise ValueError(f"unexpected git index record for protected image: {record!r}")
        oids.add(fields[1])
        count += 1
    return oids, count


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_is_partial_planar(mapping_row: dict[str, Any]) -> bool:
    mapped = {str(value) for value in mapping_row.get("keypoint_mapping", {}).values()}
    verdict = str(mapping_row.get("verdict", ""))
    return verdict == "direct-map" or (verdict == "partial-map" and PLANAR_KEYPOINTS <= mapped)


def build_corpus(
    dataset_root: Path,
    mapping_payload: dict[str, Any],
    output_root: Path,
    guard_roots: list[Path],
    *,
    partial_rows: bool = False,
    reused_guard_report: Path | None = None,
    guard_git_index_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    if tuple(mapping_payload.get("canonical_keypoints", [])) != CANONICAL_KEYPOINTS:
        raise ValueError("mapping table canonical_keypoints does not match the trainer's ordered 15-keypoint contract")
    entries = manifest_entries(dataset_root)
    protected_hashes, protected_count = guard_hashes(guard_roots)
    protected_git_oids, protected_git_count = git_index_image_oids(guard_git_index_prefixes or [])
    inherited_guard = reused_guard_evidence(reused_guard_report)
    observed_guard_count = protected_count + protected_git_count
    observed_guard_unique = len(protected_hashes) + len(protected_git_oids)
    if inherited_guard is not None:
        if observed_guard_count != inherited_guard["guard_image_files_hashed"]:
            raise ValueError(
                "reconstructed guard file count does not match reused parent evidence: "
                f"{observed_guard_count} != {inherited_guard['guard_image_files_hashed']}"
            )
        if observed_guard_unique != inherited_guard["guard_unique_hashes"]:
            raise ValueError(
                "reconstructed guard unique count does not match reused parent evidence: "
                f"{observed_guard_unique} != {inherited_guard['guard_unique_hashes']}"
            )
    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    seen_hashes: dict[str, dict[str, str]] = {}
    rows_by_dataset: Counter[str] = Counter()
    rows_by_split: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    source_groups: dict[str, str] = {}
    provenance_rows: list[dict[str, Any]] = []
    coverage_histogram: Counter[str] = Counter()
    for name, mapping_row in sorted(mapping_payload.get("datasets", {}).items()):
        if partial_rows:
            if not mapping_is_partial_planar(mapping_row):
                continue
        elif not mapping_row.get("include_default", False):
            continue
        entry = entries.get(name)
        if entry is None:
            exclusions["missing_manifest_entry"] += 1
            continue
        license_name = str(entry.get("license_as_recorded", ""))
        commercial, _ = commercial_license(license_name)
        if not commercial:
            exclusions["noncommercial_or_unknown_license"] += 1
            continue
        mapping = {str(key): str(value) for key, value in mapping_row.get("keypoint_mapping", {}).items()}
        if not partial_rows and not mapping_row.get("human_annotated", False):
            exclusions["not_documented_human_annotated"] += 1
            continue
        if not partial_rows and (set(mapping.values()) != set(CANONICAL_KEYPOINTS) or len(mapping) != len(CANONICAL_KEYPOINTS)):
            exclusions["not_full_15_direct_mapping"] += 1
            continue
        if partial_rows and not PLANAR_KEYPOINTS <= set(mapping.values()):
            exclusions["not_partial_planar_mapping"] += 1
            continue
        source_groups[name] = str(mapping_row.get("source_group", name))
        dataset_dir = dataset_root / name
        for split, (annotation_path, payload) in split_payloads(dataset_dir).items():
            images, annotations, categories = annotation_index(payload)
            clip_name = f"{name}__{split}"
            frame_dir = staging / clip_name / "labels" / "court_keypoint_frames"
            items: list[dict[str, Any]] = []
            for image_id in sorted(images):
                image = images[image_id]
                original = annotation_path.parent / str(image.get("file_name", ""))
                if not original.is_file():
                    exclusions["missing_image"] += 1
                    continue
                if any(source_id in str(original) for source_id in HELD_OUT_SOURCE_IDS):
                    exclusions["held_out_source_id"] += 1
                    continue
                candidates = [ann for ann in annotations.get(image_id, []) if ann.get("keypoints")]
                if not candidates:
                    exclusions["no_keypoint_annotation"] += 1
                    continue
                ann = max(candidates, key=lambda row: len(row.get("keypoints", [])))
                category = categories.get(int(ann.get("category_id", -1)), {})
                source_points = visible_keypoints(ann, category)
                if any(source_name not in source_points for source_name in mapping):
                    exclusions["incomplete_visibility"] += 1
                    continue
                canonical: dict[str, list[float] | None] = {name: None for name in CANONICAL_KEYPOINTS}
                for source_name, canonical_name in mapping.items():
                    canonical[canonical_name] = source_points[source_name]
                if set(canonical) != set(CANONICAL_KEYPOINTS):
                    exclusions["not_full_15_after_mapping"] += 1
                    continue
                digest = sha256(original)
                if digest in protected_hashes or git_blob_sha1(original) in protected_git_oids:
                    exclusions["protected_image_hash"] += 1
                    continue
                if digest in seen_hashes:
                    exclusions["exact_duplicate"] += 1
                    if len(duplicate_examples) < 25:
                        duplicate_examples.append({"dropped": str(original), "kept": seen_hashes[digest], "sha256": digest})
                    continue
                frame_name = f"frame_{len(items) + 1:06d}{original.suffix.lower()}"
                link_path = frame_dir / frame_name
                link_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(original.resolve(), link_path)
                provenance = {
                    "dataset": name,
                    "split": split,
                    "original_image": str(original),
                    "license": license_name,
                    "source_group": source_groups[name],
                    "sha256": digest,
                }
                status = EXTERNAL_DATASET_STATUS if partial_rows else "reviewed"
                items.append({"frame": frame_name, "status": status, "keypoints": canonical, "provenance": provenance})
                seen_hashes[digest] = provenance
                provenance_rows.append(provenance)
                rows_by_dataset[name] += 1
                rows_by_split[split] += 1
                coverage_histogram[str(sum(value is not None for value in canonical.values()))] += 1
            if items:
                write_json(
                    staging / clip_name / "labels" / "court_keypoints.json",
                    {
                        "schema_version": 1,
                        "annotation": {"items": items},
                        "frames": {
                            "frame_dir": str((output_root / clip_name / "labels" / "court_keypoint_frames").resolve())
                        },
                        "review": {
                            "status": "reviewed",
                            "reviewer": "upstream_roboflow_human_annotations",
                            "note": (
                                "Every item contains all 15 canonical names. Available COCO keypoints are mapped directly; "
                                "unavailable channels are JSON null and are not guessed."
                                if partial_rows
                                else "Each included row has all 15 canonical points directly mapped from a COCO keypoint category. "
                                "The dataset README states that the images are annotated; mapping-table human_annotated=true records the reviewed choice."
                            ),
                        },
                    },
                )
    if output_root.exists():
        shutil.rmtree(output_root)
    staging.replace(output_root)
    report = {
        "schema_version": 1,
        "corpus_root": str(output_root),
        "row_count": sum(rows_by_dataset.values()),
        "dataset_count": len(rows_by_dataset),
        "rows_by_dataset": dict(sorted(rows_by_dataset.items())),
        "rows_by_original_split": dict(sorted(rows_by_split.items())),
        "keypoint_coverage_histogram": dict(sorted(coverage_histogram.items())),
        "source_groups": source_groups,
        "near_duplicate_policy_note": (
            "Same-court/different-frame near duplicates are retained because they are expected temporal/domain samples. "
            "Training must group by source_groups/dataset and must not randomly split frames across train and validation."
        ),
        "excluded_counts": dict(sorted(exclusions.items())),
        "duplicate_examples": duplicate_examples,
        "guard_image_files_hashed": observed_guard_count,
        "guard_unique_hashes": observed_guard_unique,
        "guard_hash_matches_included": 0,
        "guard_evidence_reused": inherited_guard,
        "guard_reconstruction": {
            "git_index_protected_image_files": protected_git_count,
            "git_index_unique_blob_oids": len(protected_git_oids),
            "direct_hashed_nonprotected_image_files": protected_count,
            "direct_hashed_nonprotected_unique_sha256": len(protected_hashes),
            "protected_eval_label_files_opened": 0,
        },
        "held_out_source_ids": sorted(HELD_OUT_SOURCE_IDS),
        "provenance_rows": provenance_rows,
    }
    return report


def draw_emitted_sample(image_path: Path, keypoints: dict[str, Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1, round(max(image.size) / 700))
    null_names = []
    for name in CANONICAL_KEYPOINTS:
        value = keypoints[name]
        if value is None:
            null_names.append(name)
            continue
        x, y = float(value[0]), float(value[1])
        radius = 4 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ff3344", outline="white", width=scale)
        draw.text((x + radius + scale, y - radius), name, fill="#ffe45c", stroke_width=scale, stroke_fill="black")
    banner = f"EMITTED labeled={15 - len(null_names)}/15 | null={','.join(null_names) if null_names else 'none'}"
    draw.rectangle((0, 0, image.width, 18 * scale), fill="black")
    draw.text((4 * scale, 3 * scale), banner, fill="white")
    image.thumbnail((1400, 1000))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def render_emitted_corpus_overlays(output_root: Path, lane_dir: Path, *, limit: int = 5) -> dict[str, list[str]]:
    rendered: dict[str, list[str]] = defaultdict(list)
    for label_path in sorted(output_root.glob("*/labels/court_keypoints.json")):
        payload = read_json(label_path)
        frame_dir = label_path.parent / "court_keypoint_frames"
        for item in payload["annotation"]["items"]:
            provenance = item.get("provenance", {})
            dataset = str(provenance.get("dataset", "unknown"))
            if len(rendered[dataset]) >= limit:
                continue
            image_path = frame_dir / str(item["frame"])
            output_path = lane_dir / "emission_overlays" / dataset / f"sample_{len(rendered[dataset]) + 1:02d}.jpg"
            draw_emitted_sample(image_path, item["keypoints"], output_path)
            rendered[dataset].append(str(output_path))
    short = sorted(dataset for dataset, paths in rendered.items() if len(paths) != limit)
    if short:
        raise ValueError(f"fewer than {limit} emitted overlays for datasets: {', '.join(short)}")
    return dict(sorted(rendered.items()))


def proposed_dataset_split(included: set[str], mapping_payload: dict[str, Any]) -> dict[str, Any]:
    preferred_val = {
        "testworkspace-i8nb1__pickle-court-keypoints__v2",
        "xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10",
    }
    val = preferred_val & included
    if len(val) < 2:
        val = set(sorted(included)[-2:]) if len(included) >= 3 else set()
    train = included - val
    return {
        "schema_version": 1,
        "method": "by_dataset",
        "train_datasets": sorted(train),
        "val_datasets": sorted(val),
        "rationale": (
            "Hold out the complete xuann/testworkspace source family together to avoid family leakage. "
            "Its combined audit tags span broadcast, elevated, low, and steep viewpoints, while the remaining datasets retain "
            "elevated, steep, and broadcast training coverage. No dataset is split across train and validation."
        ),
        "source_groups": {
            name: str(mapping_payload["datasets"][name].get("source_group", name)) for name in sorted(included)
        },
    }


def write_loader_contract_proof(output_root: Path, lane_dir: Path) -> dict[str, Any]:
    from scripts.racketsport.train_court_keypoint_heatmap import (
        _label_status_counts,
        court_keypoint_label_rows,
        load_real_court_keypoint_labels,
    )

    rows = load_real_court_keypoint_labels(output_root)
    histogram = Counter(str(len(row["keypoints"])) for row in rows)
    status_counts = Counter(str(row["label_status"]) for row in rows)
    partial_payload = None
    partial_item = None
    partial_label_path = None
    twelve_point_items: list[tuple[Path, dict[str, Any]]] = []
    for label_path in sorted(output_root.glob("*/labels/court_keypoints.json")):
        payload = read_json(label_path)
        for item in payload["annotation"]["items"]:
            labeled_count = sum(value is not None for value in item["keypoints"].values())
            if labeled_count == 12:
                twelve_point_items.append((label_path, item))
            if partial_payload is None and labeled_count < 15:
                partial_payload, partial_item, partial_label_path = payload, item, label_path
    if partial_payload is None or partial_item is None or partial_label_path is None:
        raise ValueError("partial-row loader proof requires at least one emitted partial item")
    before = json.loads(json.dumps(partial_item["keypoints"]))
    roundtrip_rows = court_keypoint_label_rows(partial_payload, clip_root=partial_label_path.parent.parent)
    item_index = partial_payload["annotation"]["items"].index(partial_item)
    roundtrip = roundtrip_rows[item_index]
    after = partial_item["keypoints"]
    null_names = sorted(name for name, value in before.items() if value is None)
    if before != after or any(name in roundtrip["keypoints"] for name in null_names):
        raise AssertionError("partial-row nulls did not survive court_keypoint_label_rows")
    rng = random.Random(20260709)
    samples = rng.sample(twelve_point_items, 5)
    spot_proof = []
    for label_path, item in samples:
        net_values = {name: item["keypoints"][name] for name in sorted(NET_KEYPOINTS)}
        if any(value is not None for value in net_values.values()):
            raise AssertionError("12-point row contains a fabricated net coordinate")
        spot_proof.append(
            {
                "dataset": item["provenance"]["dataset"],
                "label_file": str(label_path),
                "frame": item["frame"],
                "net_channels": net_values,
            }
        )
    proof = {
        "schema_version": 1,
        "corpus_root": str(output_root),
        "loaded_rows": len(rows),
        "labeled_keypoint_histogram": dict(sorted(histogram.items())),
        "label_status_counts": dict(sorted(status_counts.items())),
        "training_summary_label_buckets": _label_status_counts(rows),
        "schema_errors": 0,
        "partial_roundtrip": {
            "label_file": str(partial_label_path),
            "frame": partial_item["frame"],
            "raw_null_names_before": null_names,
            "raw_null_names_after": sorted(name for name, value in after.items() if value is None),
            "loader_omits_null_channels": all(name not in roundtrip["keypoints"] for name in null_names),
            "loader_labeled_keypoint_count": len(roundtrip["keypoints"]),
        },
        "random_12_point_net_null_spot_proof": spot_proof,
    }
    write_json(lane_dir / "loader_contract_proof.json", proof)
    return proof


def write_license_card(audit: dict[str, Any], lane_dir: Path) -> dict[str, Any]:
    rows = []
    for dataset in audit["datasets"]:
        license_name = str(dataset["license_as_recorded"])
        commercial, verdict = commercial_license(license_name)
        rows.append(
            {
                "dataset": dataset["dir"],
                "license_as_recorded": license_name,
                "attribution": f"Dataset: {dataset['slug']} via Roboflow Universe; licensed {license_name}. Preserve the source project attribution and license notice.",
                "commercial_use": commercial,
                "verdict": verdict,
                "included_in_default_corpus": bool(dataset["include_default"] and commercial),
            }
        )
    payload = {"schema_version": 1, "datasets": rows, "quarantined_noncommercial": [row["dataset"] for row in rows if not row["commercial_use"]]}
    write_json(lane_dir / "license_card.json", payload)
    lines = ["# License card", "", "| Dataset | Recorded license | Commercial use | Default corpus | Attribution |", "|---|---|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | {row['license_as_recorded']} | {'yes' if row['commercial_use'] else 'no'} | "
            f"{'yes' if row['included_in_default_corpus'] else 'no'} | {row['attribution']} |"
        )
    (lane_dir / "LICENSE_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_stats(
    report: dict[str, Any],
    audit: dict[str, Any],
    lane_dir: Path,
    *,
    mapping_payload: dict[str, Any],
    partial_rows: bool = False,
) -> None:
    included = set(report["rows_by_dataset"])
    viewpoints = Counter(
        viewpoint
        for dataset in audit["datasets"]
        if dataset["dir"] in included
        for viewpoint in dataset["viewpoint_character"]
    )
    report["viewpoint_distribution_by_included_dataset"] = dict(sorted(viewpoints.items()))
    if partial_rows and len(included) >= 3:
        report["split_proposal"] = proposed_dataset_split(included, mapping_payload)
        write_json(lane_dir / "split_proposal.json", report["split_proposal"])
    elif len(included) >= 3:
        proposed_val = sorted(included)[-2:]
        report["split_proposal"] = {"method": "by_dataset", "train_datasets": sorted(included - set(proposed_val)), "val_datasets": proposed_val}
    else:
        report["split_proposal"] = {
            "method": "blocked_by_dataset_count",
            "train_datasets": [],
            "val_datasets": [],
            "note": "The exact loader contract admits fewer than three fully mapped datasets, so a leakage-safe two-dataset validation holdout is not yet possible.",
        }
    write_json(lane_dir / "corpus_stats.json", report)
    lines = [
        "# Unified real-court corpus statistics",
        "",
        f"- Rows: {report['row_count']}",
        f"- Distinct datasets: {report['dataset_count']}",
        f"- Labeled-keypoint histogram: {json.dumps(report['keypoint_coverage_histogram'], sort_keys=True)}",
        f"- Protected/harvest guard images hashed: {report['guard_image_files_hashed']}",
        f"- Protected hash matches included: {report['guard_hash_matches_included']}",
        f"- Best-stack delta: none (data preparation only).",
        "",
        "## Rows by dataset",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in report["rows_by_dataset"].items())
    lines.extend(["", "## Split proposal", "", json.dumps(report["split_proposal"], sort_keys=True)])
    (lane_dir / "STATS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local Roboflow court datasets and build a deduplicated real court-keypoint corpus.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping-table", type=Path, required=True)
    parser.add_argument("--lane-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--guard-image-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--guard-git-index-prefix",
        action="append",
        default=[],
        help="Use tracked protected-image blob IDs from Git's index without opening those protected files.",
    )
    parser.add_argument(
        "--reuse-guard-report",
        type=Path,
        help="Reuse a prior lane's immutable eval/harvest hash-guard evidence without opening protected label files.",
    )
    parser.add_argument(
        "--partial-rows",
        action="store_true",
        help="Emit direct and partial-planar external rows using exact JSON nulls for unmapped canonical channels.",
    )
    parser.add_argument("--overlap-sample-size", type=int, default=20)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mapping_payload = read_json(args.mapping_table)
    args.lane_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_datasets(args.dataset_root, mapping_payload, args.lane_dir, args.overlap_sample_size)
    write_license_card(audit, args.lane_dir)
    if args.audit_only:
        print(json.dumps({"audit_dataset_count": audit["court_related_dir_count"], "audit_path": str(args.lane_dir / "court_dataset_audit.json")}, sort_keys=True))
        return 0
    output_root = args.output_root or args.lane_dir / "real_court_corpus"
    report = build_corpus(
        args.dataset_root,
        mapping_payload,
        output_root,
        args.guard_image_root,
        partial_rows=args.partial_rows,
        reused_guard_report=args.reuse_guard_report,
        guard_git_index_prefixes=args.guard_git_index_prefix,
    )
    if args.partial_rows:
        report["emission_overlays"] = render_emitted_corpus_overlays(output_root, args.lane_dir)
        report["loader_contract_proof"] = write_loader_contract_proof(output_root, args.lane_dir)
    write_stats(report, audit, args.lane_dir, mapping_payload=mapping_payload, partial_rows=args.partial_rows)
    print(json.dumps({"audit_dataset_count": audit["court_related_dir_count"], "corpus_rows": report["row_count"], "dataset_count": report["dataset_count"], "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
