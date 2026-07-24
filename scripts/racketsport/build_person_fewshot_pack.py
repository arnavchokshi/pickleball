#!/usr/bin/env python3
"""Build and materialize the owner-verified PERSON few-shot review pack.

Planning is CPU-only and content-blind.  It samples a fixed 32-frame uniform
temporal grid per venue after trimming the first/last three percent.  The
materialize command decodes only a named, SHA-pinned local video, runs the
stock YOLO26m person teacher, and emits offline HTML plus a CVAT-images package.

Teacher output is always written separately from the empty owner-verified
label file.  This command intentionally has no mode that can promote a
pre-label to ground truth; a future reviewed-label import tool must do that.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport import build_person_mixed_pseudo_pack as mixed_pack  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260722
FRAMES_PER_VENUE = 32
MAX_PACK_FRAMES = 480
TEMPORAL_TRIM_FRACTION = 0.03
PROMOTION_JUDGE_FRACTION = 0.30
MIN_PROMOTION_JUDGE_VENUES = 3
SECONDARY_VENUE_LIMIT = 5
TEACHER_CONFIDENCE = 0.25
TEACHER_ID = "yolo26m"
TEACHER_RELATIVE_PATH = Path("models/checkpoints/yolo26m.pt")
MODEL_MANIFEST_RELATIVE_PATH = Path("models/MANIFEST.json")
PERSON_P1_MANIFEST_RELATIVE_PATH = Path(
    "runs/lanes/person_p1_roboflow_20260721/roboflow_person/dataset_manifest.json"
)
PERSON_MIXED_MANIFEST_RELATIVE_PATH = Path(
    "runs/lanes/person_mixed_20260722/pack_manifest.json"
)
DEFAULT_OUT_DIR = Path("runs/lanes/trkC_fewshot_pack_20260722")
PARTITION_FINETUNE = "FINETUNE_MATERIAL"
PARTITION_JUDGE = "PROMOTION_JUDGE"
PARTITIONS = (PARTITION_FINETUNE, PARTITION_JUDGE)
REVIEW_ANSWER_CONTRACT = {
    "1": {
        "answer": "ALL_BOXES_CORRECT",
        "ground_truth": True,
        "verified_by_owner": True,
    },
    "2": {
        "answer": "BOX_WRONG",
        "ground_truth": False,
        "verified_by_owner": False,
    },
    "3": {
        "answer": "PERSON_MISSED",
        "ground_truth": False,
        "verified_by_owner": False,
    },
    "4": {
        "answer": "UNSURE",
        "ground_truth": False,
        "verified_by_owner": False,
    },
}
BEST_STACK_DELTA = (
    "(c) none — data-pack tooling; the fine-tune lane that consumes this pack will carry the manifest delta."
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v"})

# Re-export the canonical refusal registry. This lane wraps the person_mixed
# guard so a normalization budget exhaustion fails closed at this boundary.
COMPARE_ONLY_PBVISION_IDS = mixed_pack.COMPARE_ONLY_PBVISION_IDS
COMPARE_ONLY_MEDIA_SHA256 = mixed_pack.COMPARE_ONLY_MEDIA_SHA256
PROTECTED_CLIP_IDS = mixed_pack.PROTECTED_CLIP_IDS
PROTECTED_MEDIA_SHA256 = mixed_pack.PROTECTED_MEDIA_SHA256
IYNBD_DERIVATIVE_TOKEN = mixed_pack.IYNBD_DERIVATIVE_TOKEN
PBVISION_MEDIA_SHA256_BY_ID = mixed_pack.PBVISION_MEDIA_SHA256_BY_ID
QuarantinedSourceError = mixed_pack.QuarantinedSourceError
MAX_CANONICAL_DECODE_PASSES = mixed_pack.MAX_CANONICAL_DECODE_PASSES


class VerifiedLabelsOverwriteError(RuntimeError):
    """Raised before a regeneration could erase owner-verified labels."""


class VenueAliasCollisionError(ValueError):
    """Raised when a known physical-venue alias crosses partition sides."""


def _canonical_guard_scalar(value: Any) -> str:
    """Decode to a fixpoint within the registered budget or fail closed."""

    text = str(value)
    for _ in range(MAX_CANONICAL_DECODE_PASSES):
        decoded = unicodedata.normalize("NFKC", unquote(text, errors="replace"))
        if decoded == text:
            return decoded.casefold()
        text = decoded
    one_more = unicodedata.normalize("NFKC", unquote(text, errors="replace"))
    if one_more != text:
        raise QuarantinedSourceError(
            "refused source identity: CANONICAL_DECODE_BUDGET_EXHAUSTED"
        )
    return text.casefold()


def _assert_guard_fixpoint(value: Any) -> None:
    for scalar in mixed_pack._guard_scalars(value):
        _canonical_guard_scalar(scalar)


def quarantine_reason(source: Mapping[str, Any]) -> str | None:
    try:
        _assert_guard_fixpoint(source)
    except QuarantinedSourceError:
        return "CANONICAL_DECODE_BUDGET_EXHAUSTED"
    return mixed_pack.quarantine_reason(source)


def assert_source_allowed(source: Mapping[str, Any]) -> None:
    reason = quarantine_reason(source)
    if reason is not None:
        source_id = source.get("source_id") or source.get("video_id") or source.get("id") or "unknown"
        raise QuarantinedSourceError(f"refused source {source_id}: {reason}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _json_for_html_script(value: Any) -> str:
    """Serialize JSON without allowing HTML parser script termination."""

    return (
        json.dumps(value, separators=(",", ":"), sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_bytes(path, _jsonl_bytes(rows))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(row)
    return rows


def _resolve_manifest_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _teacher_manifest_path(pack_manifest: Mapping[str, Any]) -> Path:
    teacher = pack_manifest.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("pack manifest lacks teacher identity")
    value = teacher.get("model_manifest_path") or teacher.get("manifest_path")
    if not value:
        raise ValueError("pack manifest teacher lacks model manifest path")
    return _resolve_manifest_path(Path(str(value)))


def _expected_teacher_checkpoint_sha256(model_manifest_path: Path) -> str:
    manifest = _load_json(model_manifest_path)
    models = manifest.get("models") if isinstance(manifest, Mapping) else None
    if not isinstance(models, list):
        raise ValueError("model manifest must contain a models list")
    matches = [row for row in models if row.get("id") == TEACHER_ID]
    if len(matches) != 1:
        raise ValueError(f"model manifest must contain exactly one {TEACHER_ID} entry")
    expected = str(matches[0].get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"model manifest {TEACHER_ID} entry lacks a valid SHA-256")
    return expected


def load_person_family_alias_taxonomy(
    *,
    person_p1_manifest_path: Path = ROOT / PERSON_P1_MANIFEST_RELATIVE_PATH,
    person_mixed_manifest_path: Path = ROOT / PERSON_MIXED_MANIFEST_RELATIVE_PATH,
) -> dict[str, str]:
    """Map source/family names to the aliases known by P1 and person_mixed."""

    aliases: dict[str, str] = {}

    def register(token: Any, alias: str) -> None:
        key = str(token or "").strip()
        if not key:
            return
        existing = aliases.get(key)
        if existing is not None and existing != alias:
            raise ValueError(f"person taxonomy maps {key!r} to conflicting aliases")
        aliases[key] = alias

    p1_manifest = _load_json(person_p1_manifest_path)
    p1_rows = p1_manifest.get("rows") if isinstance(p1_manifest, Mapping) else None
    if not isinstance(p1_rows, list):
        raise ValueError("person_p1 manifest must contain a rows list")
    for row in p1_rows:
        if not isinstance(row, Mapping) or not row.get("family_id"):
            raise ValueError("person_p1 manifest row lacks family_id")
        alias = f"person_p1:{row['family_id']}"
        register(row.get("family_id"), alias)
        register(row.get("source"), alias)

    mixed_manifest = _load_json(person_mixed_manifest_path)
    count_tables = mixed_manifest.get("count_tables") if isinstance(mixed_manifest, Mapping) else None
    mixed_sources = count_tables.get("per_source") if isinstance(count_tables, Mapping) else None
    if not isinstance(mixed_sources, list):
        raise ValueError("person_mixed manifest must contain count_tables.per_source")
    for row in mixed_sources:
        if not isinstance(row, Mapping) or not row.get("venue_source_family_id"):
            raise ValueError("person_mixed source lacks venue_source_family_id")
        alias = f"person_mixed:{row['venue_source_family_id']}"
        register(row.get("source_id"), alias)
        register(row.get("source_family_id"), alias)
        register(row.get("venue_source_family_id"), alias)
    return aliases


def validate_partition_venue_aliases(
    sources: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
    *,
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    """Refuse a known family/venue alias that occurs on both partition sides."""

    partition_by_source = {
        str(row["source_id"]): str(row["partition"])
        for row in assignment["assignments"]
    }
    alias_partitions: dict[str, set[str]] = defaultdict(set)
    alias_sources: dict[str, set[str]] = defaultdict(set)
    known_source_count = 0
    for source in sources:
        source_id = str(source["source_id"])
        tokens = (
            source_id,
            source.get("source_family_id"),
            source.get("family_id"),
            source.get("venue_source_family_id"),
            source.get("venue_family_id"),
            source.get("venue_id"),
        )
        known = {aliases[str(token)] for token in tokens if str(token or "") in aliases}
        if len(known) > 1:
            raise ValueError(f"source {source_id} maps to conflicting known venue aliases: {sorted(known)}")
        if not known:
            continue
        known_source_count += 1
        alias = next(iter(known))
        alias_partitions[alias].add(partition_by_source[source_id])
        alias_sources[alias].add(source_id)
    collisions = {
        alias: sorted(alias_sources[alias])
        for alias, partitions in alias_partitions.items()
        if len(partitions) > 1
    }
    if collisions:
        alias = sorted(collisions)[0]
        raise VenueAliasCollisionError(
            f"physical-venue alias crosses partition sides: {alias} sources={collisions[alias]}"
        )
    return {
        "known_source_count": known_source_count,
        "known_alias_count": len(alias_partitions),
        "cross_partition_alias_collisions": 0,
    }


def _nonempty_verified_label_paths(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    backup_root = out_dir / "verified_label_backups"
    return sorted(
        path
        for path in out_dir.rglob("verified_labels.jsonl")
        if path.is_file()
        and path.stat().st_size > 0
        and backup_root not in path.parents
    )


def _guard_verified_labels(
    out_dir: Path,
    *,
    force_with_backup: bool,
) -> list[Path]:
    protected = _nonempty_verified_label_paths(out_dir)
    if not protected:
        return []
    relative = [path.relative_to(out_dir).as_posix() for path in protected]
    if not force_with_backup:
        raise VerifiedLabelsOverwriteError(
            f"refusing to overwrite nonempty verified labels: {relative}; "
            "use --force-with-backup to preserve them first"
        )
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = out_dir / "verified_label_backups" / timestamp
    backups: list[Path] = []
    for source in protected:
        destination = backup_root / source.relative_to(out_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.read_bytes() != source.read_bytes():
            raise IOError(f"verified-label backup verification failed: {destination}")
        backups.append(destination)
    return backups


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seeded_hash(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError(f"cannot create a safe slug for {value!r}")
    return slug


def _repo_reference(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def verify_teacher_checkpoint(
    model_manifest_path: Path,
    teacher_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    manifest = _load_json(model_manifest_path)
    models = manifest.get("models") if isinstance(manifest, Mapping) else None
    if not isinstance(models, list):
        raise ValueError("model manifest must contain a models list")
    matches = [row for row in models if row.get("id") == TEACHER_ID]
    if len(matches) != 1:
        raise ValueError(f"model manifest must contain exactly one {TEACHER_ID} entry")
    expected_sha = str(matches[0].get("sha256") or "").lower()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint is absent: {teacher_path}")
    actual_sha = sha256_file(teacher_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"teacher checkpoint SHA-256 mismatch: manifest={expected_sha}, actual={actual_sha}"
        )
    return {
        "model_id": TEACHER_ID,
        "checkpoint_path": _repo_reference(teacher_path, repo_root),
        "checkpoint_sha256": actual_sha,
        "model_manifest_path": _repo_reference(model_manifest_path, repo_root),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "class_filter": {"class_id": 0, "class_name": "person"},
        "confidence_min": TEACHER_CONFIDENCE,
        "nms": "ultralytics_defaults_no_override",
    }


def sample_frame_indices(
    frame_count: int,
    *,
    sample_count: int = FRAMES_PER_VENUE,
    trim_fraction: float = TEMPORAL_TRIM_FRACTION,
) -> list[int]:
    """Return a content-blind uniform grid in the untrimmed middle interval."""

    if frame_count <= 0 or sample_count <= 0:
        raise ValueError("frame_count and sample_count must be positive")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    first = int(math.ceil(frame_count * trim_fraction))
    stop_exclusive = int(math.floor(frame_count * (1.0 - trim_fraction)))
    last = stop_exclusive - 1
    usable = last - first + 1
    if usable < sample_count:
        raise ValueError(
            f"source has only {usable} usable frames after trim; {sample_count} are required"
        )
    if sample_count == 1:
        return [first + usable // 2]
    indices = [
        first + round(position * (usable - 1) / (sample_count - 1))
        for position in range(sample_count)
    ]
    if len(indices) != len(set(indices)):
        raise AssertionError("uniform sampler produced duplicate frame indices")
    if indices[0] < first or indices[-1] > last:
        raise AssertionError("uniform sampler escaped the trimmed interval")
    return indices


def select_secondary_harvest_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    limit: int = SECONDARY_VENUE_LIMIT,
) -> list[dict[str, Any]]:
    """Choose at most one source per existing person_mixed venue family."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        assert_source_allowed(source)
        family = str(source.get("venue_source_family_id") or "")
        if not family:
            raise ValueError("harvest source lacks venue_source_family_id")
        grouped[family].append(source)
    ranked_families = sorted(
        grouped,
        key=lambda family: (_seeded_hash(seed, "secondary-venue", family), family),
    )
    selected: list[dict[str, Any]] = []
    for family in ranked_families[:limit]:
        candidates = sorted(
            grouped[family],
            key=lambda row: (
                _seeded_hash(seed, "secondary-source", str(row["source_id"])),
                str(row["source_id"]),
            ),
        )
        selected.append(dict(candidates[0]))
    return sorted(selected, key=lambda row: str(row["source_id"]))


def _with_venue_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(source)
    venue_family = str(result.get("venue_source_family_id") or result["source_family_id"])
    result["venue_id"] = venue_family
    result["venue_family_id"] = venue_family
    result["family_id"] = str(result["source_family_id"])
    result["requested_frame_count"] = FRAMES_PER_VENUE
    result["teacher_derived"] = True
    result["ground_truth"] = False
    result["verified_by_owner"] = False
    result["training_eligible"] = False
    result["production_eligible"] = False
    result["do_not_promote"] = True
    if result.get("source_pool") == "pbvision":
        result["usage_posture"] = "RD_ONLY_competitor_processed_internal_use"
        result["redistribution_allowed"] = False
    else:
        result["usage_posture"] = "internal_only_existing_online_harvest"
        result["redistribution_allowed"] = False
    assert_source_allowed(result)
    return result


def build_partition_assignment(
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    venue_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    venue_ids = [str(source["venue_id"]) for source in sources]
    if len(venue_ids) != len(set(venue_ids)):
        duplicates = sorted(venue for venue, count in Counter(venue_ids).items() if count > 1)
        raise ValueError(f"venue IDs must be unique before partitioning: {duplicates}")
    ranked = sorted(
        sources,
        key=lambda source: (
            _seeded_hash(seed, "partition-venue", str(source["venue_id"])),
            str(source["venue_id"]),
        ),
    )
    judge_count = max(
        MIN_PROMOTION_JUDGE_VENUES,
        int(math.ceil(len(ranked) * PROMOTION_JUDGE_FRACTION)),
    )
    if judge_count >= len(ranked):
        raise ValueError("partition needs at least one fine-tune venue after reserving judge venues")
    judge_ids = {str(source["venue_id"]) for source in ranked[:judge_count]}
    assignments = []
    for source in sorted(sources, key=lambda row: str(row["venue_id"])):
        venue_id = str(source["venue_id"])
        assignments.append(
            {
                "venue_id": venue_id,
                "source_id": str(source["source_id"]),
                "source_pool": str(source["source_pool"]),
                "family_id": str(source["family_id"]),
                "venue_family_id": str(source["venue_family_id"]),
                "hash_rank_key": _seeded_hash(seed, "partition-venue", venue_id),
                "partition": PARTITION_JUDGE if venue_id in judge_ids else PARTITION_FINETUNE,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_fewshot_venue_partition_assignment",
        "seed": seed,
        "method": "sha256(seed:partition-venue:venue_id) ascending; first ceil(30%) are PROMOTION_JUDGE",
        "content_peeking": False,
        "rough_ratio_target": {PARTITION_FINETUNE: 0.70, PARTITION_JUDGE: 0.30},
        "counts": dict(sorted(Counter(row["partition"] for row in assignments).items())),
        "assignments": assignments,
        "venue_disjoint": True,
        "promotion_judge_never_trainable": True,
    }
    if venue_aliases is not None:
        validate_partition_venue_aliases(sources, result, aliases=venue_aliases)
    return result


def build_decode_rows(
    sources: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    partition_by_venue = {
        str(row["venue_id"]): str(row["partition"])
        for row in assignment["assignments"]
    }
    rows: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda row: str(row["venue_id"])):
        venue_id = str(source["venue_id"])
        partition = partition_by_venue[venue_id]
        frame_count = int(source["expected_frame_count"])
        fps = float(source["fps_inventory"])
        indices = sample_frame_indices(frame_count)
        source_slug = _slug(str(source["source_id"]))
        venue_slug = _slug(venue_id)
        for ordinal, frame_index in enumerate(indices, start=1):
            output_name = f"{source_slug}__f{frame_index:09d}.jpg"
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "video_id": str(source["source_id"]),
                    "source_id": str(source["source_id"]),
                    "source_pool": str(source["source_pool"]),
                    "source_family_id": str(source["source_family_id"]),
                    "venue_id": venue_id,
                    "venue_family_id": str(source["venue_family_id"]),
                    "partition": partition,
                    "media_sha256": str(source["expected_media_sha256"]),
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 6),
                    "sample_ordinal": ordinal,
                    "output_name": output_name,
                    "output_relpath": (
                        f"materialized/{venue_slug}/frames/{output_name}"
                    ),
                    "sampling": "uniform_temporal_stride_middle_94_percent",
                    "teacher_derived": True,
                    "ground_truth": False,
                    "verified_by_owner": False,
                    "training_eligible": False,
                    "production_eligible": False,
                    "do_not_promote": True,
                }
            )
    if len(rows) > MAX_PACK_FRAMES:
        raise ValueError(f"decode plan has {len(rows)} rows; cap is {MAX_PACK_FRAMES}")
    counts = Counter(str(row["venue_id"]) for row in rows)
    if any(count != FRAMES_PER_VENUE for count in counts.values()):
        raise AssertionError("every venue must contribute exactly 32 frames")
    validate_prelabel_rows(rows, allow_missing_teacher_box_fields=True)
    return rows


def validate_prelabel_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_missing_teacher_box_fields: bool = False,
    model_manifest_path: Path = ROOT / MODEL_MANIFEST_RELATIVE_PATH,
) -> None:
    expected_checkpoint_sha256 = (
        "" if allow_missing_teacher_box_fields else _expected_teacher_checkpoint_sha256(model_manifest_path)
    )
    for index, row in enumerate(rows):
        if row.get("teacher_derived") is not True:
            raise ValueError(f"pre-label row {index} must be teacher_derived=true")
        if row.get("ground_truth") is not False:
            raise ValueError(f"pre-label row {index} must be ground_truth=false")
        if row.get("verified_by_owner") is not False:
            raise ValueError(f"pre-label row {index} must be verified_by_owner=false")
        if row.get("partition") not in PARTITIONS:
            raise ValueError(f"pre-label row {index} lacks a valid partition")
        if not allow_missing_teacher_box_fields:
            if "teacher_conf" not in row or "teacher_checkpoint_sha256" not in row:
                raise ValueError(f"pre-label box row {index} lacks teacher identity/confidence")
            confidence_value = row["teacher_conf"]
            if isinstance(confidence_value, bool):
                raise ValueError(f"pre-label box row {index} has invalid teacher confidence")
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pre-label box row {index} has invalid teacher confidence"
                ) from exc
            if not math.isfinite(confidence) or confidence < TEACHER_CONFIDENCE:
                raise ValueError(
                    f"pre-label box row {index} confidence {confidence!r} is below "
                    f"the registered {TEACHER_CONFIDENCE} floor"
                )
            if confidence > 1.0:
                raise ValueError(
                    f"pre-label box row {index} confidence {confidence!r} exceeds "
                    "the registered 1.0 ceiling"
                )
            checkpoint_sha256 = str(row["teacher_checkpoint_sha256"]).lower()
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise ValueError(
                    f"pre-label box row {index} checkpoint SHA does not match "
                    f"{model_manifest_path}"
                )


def _gallery_local_candidates(
    source: Mapping[str, Any],
    *,
    gallery_root: Path,
    eval_root: Path,
) -> list[Path]:
    source_id = str(source["source_id"])
    candidates: list[Path] = []
    source_dir = gallery_root / source_id
    if source_dir.is_dir():
        candidates.extend(
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    if eval_root.is_dir():
        source_token = mixed_pack._normalise_guard_text(source_id)
        for path in eval_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if source_token in mixed_pack._normalise_guard_text(path.as_posix()):
                candidates.append(path)
    return sorted(set(candidates))


def inspect_local_media(
    sources: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    gallery_root: Path,
    eval_root: Path,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for source in sources:
        expected_sha = str(source["expected_media_sha256"])
        candidates: list[Path] = []
        if source["source_pool"] == "pbvision":
            candidates = _gallery_local_candidates(
                source,
                gallery_root=gallery_root,
                eval_root=eval_root,
            )
        else:
            expected_path = Path(str(source["expected_media_location"]))
            if not expected_path.is_absolute():
                expected_path = repo_root / expected_path
            if expected_path.is_file():
                candidates = [expected_path]
        matching: list[str] = []
        mismatches: list[dict[str, str]] = []
        for candidate in candidates:
            actual_sha = sha256_file(candidate)
            identity = {
                "source_id": source["source_id"],
                "media_sha256": actual_sha,
                "path": candidate.as_posix(),
            }
            assert_source_allowed(identity)
            if actual_sha == expected_sha:
                matching.append(_repo_reference(candidate, repo_root))
            else:
                mismatches.append(
                    {"path": _repo_reference(candidate, repo_root), "actual_sha256": actual_sha}
                )
        observations.append(
            {
                "source_id": str(source["source_id"]),
                "venue_id": str(source["venue_id"]),
                "source_pool": str(source["source_pool"]),
                "expected_media_sha256": expected_sha,
                "local_media_present": bool(matching),
                "matching_paths": matching,
                "sha_mismatches": mismatches,
            }
        )
    return observations


def _ledger_rows(
    sources: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    partition_by_venue = {
        str(row["venue_id"]): str(row["partition"])
        for row in assignment["assignments"]
    }
    rows = []
    for source in sorted(sources, key=lambda row: str(row["venue_id"])):
        venue_id = str(source["venue_id"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "data_steward_ledger_row",
                "consumer_lane": "trkC_fewshot_pack_20260722",
                "source_id": str(source["source_id"]),
                "source_pool": str(source["source_pool"]),
                "source_family_id": str(source["source_family_id"]),
                "venue_id": venue_id,
                "venue_family_id": str(source["venue_family_id"]),
                "partition": partition_by_venue[venue_id],
                "media_sha256": str(source["expected_media_sha256"]),
                "planned_frames": FRAMES_PER_VENUE,
                "sampling": "content_blind_uniform_middle_94_percent",
                "usage_posture": str(source["usage_posture"]),
                "redistribution_allowed": False,
                "teacher_derived": True,
                "ground_truth": False,
                "verified_by_owner": False,
                "protected_eval": False,
                "production_eligible": False,
                "do_not_promote": True,
            }
        )
    return rows


def _assignment_markdown(assignment: Mapping[str, Any]) -> str:
    lines = [
        "| Partition | Venue | Source | Family |",
        "|---|---|---|---|",
    ]
    for row in sorted(
        assignment["assignments"],
        key=lambda item: (str(item["partition"]), str(item["venue_id"])),
    ):
        lines.append(
            f"| {row['partition']} | `{row['venue_id']}` | `{row['source_id']}` | `{row['family_id']}` |"
        )
    return "\n".join(lines)


def _vm_materialize_commands(
    assignment: Mapping[str, Any],
    missing_gallery: Sequence[Mapping[str, Any]],
) -> str:
    partition_by_source = {
        str(row["source_id"]): str(row["partition"])
        for row in assignment["assignments"]
    }
    lines = [
        "# VM materialize commands",
        "",
        "The plan artifacts must be present at the same lane-relative path on the VM. No command fetches media.",
        "Run from `/home/arnavchokshi/pickleball_git`:",
        "",
    ]
    for row in sorted(missing_gallery, key=lambda item: str(item["source_id"])):
        source_id = str(row["source_id"])
        lines.extend(
            [
                f"## {source_id} — {partition_by_source[source_id]}",
                "",
                "```bash",
                ".venv/bin/python scripts/racketsport/build_person_fewshot_pack.py materialize "
                "--out-dir runs/lanes/trkC_fewshot_pack_20260722 "
                f"--source-id {source_id} "
                f"--media /home/arnavchokshi/pbv_gallery/{source_id}/max.mp4 "
                "--device cpu",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _owner_ask_markdown(
    manifest: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> str:
    planned = int(manifest["counts"]["planned_frames"])
    materialized = int(manifest["counts"].get("materialized_frames", 0))
    min_minutes = planned * 10 / 60
    max_minutes = planned * 15 / 60
    materialized_min = materialized * 10 / 60
    materialized_max = materialized * 15 / 60
    html_status = (
        "Open `review/START_HERE.html` in a browser."
        if materialized
        else "Open `review/START_HERE.html` to see the plan-only venue queue."
    )
    return f"""# PERSON few-shot owner ask

## Primary path: offline HTML

{html_status} This is the primary review surface. It is dependency-free, resumes from browser
local storage, and keeps the two partitions and every venue visibly separated.

For every frame, verify **all people**: players and spectators are both class `person`.

- Press **1** when all boxes are correct and no person is missed.
- Press **2** when any shown box is wrong.
- Press **3** when any person is missed.
- Press **4** when unsure. Every answer auto-advances to the next frame.
- Use **Export results.json** before closing or moving to another browser. The page saves and
  resumes locally, but the downloaded `results.json` is the one durable handoff.

Answers 2, 3, and 4 are explicitly non-verifying: they can never yield
`ground_truth=true` or `verified_by_owner=true` in the import contract. The downstream selection
layer, not this label task, separates players from spectators.

Teacher boxes are suggestions only. `prelabels_teacher.jsonl` is structurally marked
`teacher_derived=true`, `ground_truth=false`, `verified_by_owner=false`. The separate
`verified_labels.jsonl` is empty. Exporting `results.json` does not mutate those files. A future
reviewed-label import tool must create verified rows and enforce the answer contract.

## Secondary path: CVAT fallback

If detailed geometric edits are easier in CVAT, use the per-venue jobs under `cvat_upload/`.
Each job contains `images/`, CVAT Images 1.1 `annotations.xml`, `task_manifest.json`, and a ZIP.
Do not merge venues or partitions into one task.

## Time arithmetic

- Full planned pack: {planned} frames × 10–15 seconds/frame = {planned * 10:,}–{planned * 15:,} seconds = **{min_minutes:.0f}–{max_minutes:.0f} minutes**.
- Owner budget: **90 minutes**. The low estimate fits by {90 - min_minutes:.0f} minutes; the high estimate exceeds it by {max_minutes - 90:.0f} minutes.
- Materialized now: {materialized} frames = **{materialized_min:.1f}–{materialized_max:.1f} minutes**.

## Binding venue-disjoint partition

This assignment was produced before any owner review artifact, from the registered seed
`{assignment['seed']}` and venue-ID hash order only. A venue may never cross partitions.
`PROMOTION_JUDGE` labels are never trainable and may never tune a threshold or model.

Alias guard: {manifest['venue_alias_risk']['statement']}

{_assignment_markdown(assignment)}

## Cross-signal

- CONSUMES: stock YOLO26m detector; pb.vision gallery inventory and venue metadata; person_p1/person_mixed family taxonomy; no court calibration.
- FEEDS: exposure-matched frozen-backbone RF-DETR vs YOLO26 fine-tune arms using only `FINETUNE_MATERIAL`; untouched selection/detector promotion accuracy using only `PROMOTION_JUDGE`; self-training-v2 anchor diversity; data-steward ledger rows.

## Best-stack delta

{BEST_STACK_DELTA}
"""


def write_plan(
    *,
    out_dir: Path,
    sources: Sequence[Mapping[str, Any]],
    teacher: Mapping[str, Any],
    local_media: Sequence[Mapping[str, Any]],
    seed: int = DEFAULT_SEED,
    person_p1_manifest_path: Path = ROOT / PERSON_P1_MANIFEST_RELATIVE_PATH,
    person_mixed_manifest_path: Path = ROOT / PERSON_MIXED_MANIFEST_RELATIVE_PATH,
    force_with_backup: bool = False,
) -> dict[str, Any]:
    _guard_verified_labels(out_dir, force_with_backup=force_with_backup)
    normalized = [_with_venue_identity(source) for source in sources]
    if len(normalized) * FRAMES_PER_VENUE > MAX_PACK_FRAMES:
        raise ValueError("selected venues exceed the registered 480-frame cap")
    if len(normalized) < 10:
        raise ValueError("registered pack requires at least 10 venue families")
    venue_aliases = load_person_family_alias_taxonomy(
        person_p1_manifest_path=person_p1_manifest_path,
        person_mixed_manifest_path=person_mixed_manifest_path,
    )
    assignment = build_partition_assignment(
        normalized,
        seed=seed,
        venue_aliases=venue_aliases,
    )
    alias_check = validate_partition_venue_aliases(
        normalized,
        assignment,
        aliases=venue_aliases,
    )
    assignment_bytes = _json_bytes(assignment)

    # Amendment 1 ordering: the pre-registered partition is the first artifact.
    _write_bytes(out_dir / "partition_assignment.json", assignment_bytes)

    decode_rows = build_decode_rows(normalized, assignment)
    ledger_rows = _ledger_rows(normalized, assignment)
    local_by_source = {str(row["source_id"]): dict(row) for row in local_media}
    source_rows = []
    for source in sorted(normalized, key=lambda row: str(row["venue_id"])):
        row = dict(source)
        row["partition"] = next(
            item["partition"]
            for item in assignment["assignments"]
            if item["venue_id"] == source["venue_id"]
        )
        row["local_media"] = local_by_source.get(str(source["source_id"]), {})
        source_rows.append(row)

    missing_gallery = [
        row
        for row in local_media
        if row.get("source_pool") == "pbvision" and not row.get("local_media_present")
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_fewshot_owner_review_pack",
        "lane": "trkC_fewshot_pack_20260722",
        "program_authority": "runs/research_sota_20260722/PROGRAM.md Track C action 3",
        "verified": False,
        "verified_by_owner": False,
        "ground_truth": False,
        "training_eligible": False,
        "production_eligible": False,
        "do_not_promote": True,
        "protocol": {
            "seed": seed,
            "sampling": "32-frame uniform temporal stride per venue, first/last 3% excluded",
            "content_based_selection": False,
            "frames_per_venue": FRAMES_PER_VENUE,
            "max_pack_frames": MAX_PACK_FRAMES,
            "secondary_harvest_venue_limit": SECONDARY_VENUE_LIMIT,
            "partition_method": assignment["method"],
        },
        "counts": {
            "venue_families": len(normalized),
            "planned_frames": len(decode_rows),
            "materialized_venues": 0,
            "materialized_frames": 0,
            "teacher_boxes": 0,
            "verified_label_rows": 0,
        },
        "partition_counts": assignment["counts"],
        "partition_assignment": {
            "path": "partition_assignment.json",
            "sha256": _sha256_bytes(assignment_bytes),
            "venue_disjoint": True,
            "promotion_judge_never_trainable": True,
        },
        "teacher": dict(teacher),
        "sources": source_rows,
        "local_media_audit": list(local_media),
        "missing_gallery_venues": [str(row["venue_id"]) for row in missing_gallery],
        "labels": {
            "prelabels": {
                "path": "prelabels_teacher.jsonl",
                "teacher_derived": True,
                "ground_truth": False,
                "verified_by_owner": False,
            },
            "verified_labels": {
                "path": "verified_labels.jsonl",
                "row_count": 0,
                "write_authority": "future_reviewed_label_import_tool_only",
                "verified_by_owner": False,
            },
            "pseudo_never_verified_invariant": True,
        },
        "artifacts": {
            "decode_plan": "decode_plan.jsonl",
            "data_steward_ledger": "data_steward_ledger_rows.jsonl",
            "owner_ask": "OWNER_ASK.md",
            "vm_materialize_commands": "VM_MATERIALIZE_COMMANDS.md",
            "review_html": None,
            "cvat_upload_root": None,
        },
        "materialized_venues": [],
        "venue_alias_risk": {
            "statement": (
                "Partitioning is name-based. Aliases known by the person_p1/person_mixed "
                "family taxonomy are refused across sides, but physical venue identity is "
                "not fully provable from these manifests."
            ),
            "person_p1_manifest": _repo_reference(person_p1_manifest_path, ROOT),
            "person_mixed_manifest": _repo_reference(person_mixed_manifest_path, ROOT),
            "known_source_count": alias_check["known_source_count"],
            "known_alias_count": alias_check["known_alias_count"],
            "cross_partition_alias_collisions": 0,
        },
        "cross_signal": {
            "consumes": [
                "stock YOLO26m person detector",
                "pb.vision gallery inventory and venue metadata",
                "person_p1/person_mixed family taxonomy",
                "raw frames; court calibration not required",
            ],
            "feeds": [
                "exposure-matched frozen-backbone RF-DETR vs YOLO26 fine-tune arms",
                "untouched PROMOTION_JUDGE selection/detector promotion accuracy",
                "self-training v2 anchor diversity",
                "data-steward ledger rows",
            ],
        },
        "best_stack_delta": BEST_STACK_DELTA,
    }
    _write_jsonl(out_dir / "decode_plan.jsonl", decode_rows)
    _write_jsonl(out_dir / "data_steward_ledger_rows.jsonl", ledger_rows)
    _write_bytes(out_dir / "prelabels_teacher.jsonl", b"")
    _write_bytes(out_dir / "verified_labels.jsonl", b"")
    _write_json(out_dir / "pack_manifest.json", manifest)
    _write_bytes(out_dir / "OWNER_ASK.md", _owner_ask_markdown(manifest, assignment).encode("utf-8"))
    _write_bytes(
        out_dir / "VM_MATERIALIZE_COMMANDS.md",
        _vm_materialize_commands(assignment, missing_gallery).encode("utf-8"),
    )
    return manifest


def build_pack(
    *,
    repo_root: Path,
    out_dir: Path,
    gallery_manifest: Path,
    harvest_manifest: Path,
    model_manifest: Path,
    checkpoint: Path,
    gallery_root: Path,
    eval_root: Path,
    seed: int = DEFAULT_SEED,
    force_with_backup: bool = False,
) -> dict[str, Any]:
    if (out_dir / "pack_manifest.json").exists():
        raise FileExistsError(f"refusing to replace existing pack manifest: {out_dir}")
    pbvision, refusals = mixed_pack.load_pbvision_sources(gallery_manifest)
    if {row["source_id"] for row in refusals} != set(COMPARE_ONLY_PBVISION_IDS):
        raise AssertionError("the complete compare-only registry was not refused")
    harvest = mixed_pack.load_harvest_sources(harvest_manifest, repo_root)
    secondary = select_secondary_harvest_sources(harvest, seed=seed)
    sources = [_with_venue_identity(row) for row in pbvision + secondary]
    teacher = verify_teacher_checkpoint(model_manifest, checkpoint, repo_root=repo_root)
    local_media = inspect_local_media(
        sources,
        repo_root=repo_root,
        gallery_root=gallery_root,
        eval_root=eval_root,
    )
    manifest = write_plan(
        out_dir=out_dir,
        sources=sources,
        teacher=teacher,
        local_media=local_media,
        seed=seed,
        person_p1_manifest_path=repo_root / PERSON_P1_MANIFEST_RELATIVE_PATH,
        person_mixed_manifest_path=repo_root / PERSON_MIXED_MANIFEST_RELATIVE_PATH,
        force_with_backup=force_with_backup,
    )
    manifest["quarantine_refusals"] = refusals
    manifest["counts"]["compare_only_refusals"] = len(refusals)
    _write_json(out_dir / "pack_manifest.json", manifest)
    return manifest


PredictionFn = Callable[[Sequence[Path], Path, float, str], Sequence[Sequence[Mapping[str, Any]]]]


def _run_yolo_predictions(
    image_paths: Sequence[Path],
    checkpoint: Path,
    confidence: float,
    device: str,
) -> list[list[dict[str, Any]]]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in image_paths],
        classes=[0],
        conf=confidence,
        device=device,
        verbose=False,
        save=False,
    )
    predictions: list[list[dict[str, Any]]] = []
    for result in results:
        frame_predictions: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for coordinates, score, class_id in zip(xyxy, confs, classes, strict=True):
                if int(class_id) != 0:
                    raise AssertionError("person-only inference returned a non-person class")
                frame_predictions.append(
                    {"bbox_xyxy": [float(value) for value in coordinates], "confidence": float(score)}
                )
        predictions.append(frame_predictions)
    return predictions


def _verify_partition_pin(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    assignment_path = out_dir / str(manifest["partition_assignment"]["path"])
    actual_sha = sha256_file(assignment_path)
    expected_sha = str(manifest["partition_assignment"]["sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            f"partition assignment SHA-256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    assignment = _load_json(assignment_path)
    if assignment.get("venue_disjoint") is not True:
        raise ValueError("partition assignment is not venue-disjoint")
    return assignment


def _decode_frames(
    *,
    media_path: Path,
    plan_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    import cv2

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open media: {media_path}")
    decoded: list[dict[str, Any]] = []
    try:
        for row in sorted(plan_rows, key=lambda item: int(item["sample_ordinal"])):
            frame_index = int(row["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"decode failed at frame {frame_index} in {media_path}")
            output_path = out_dir / str(row["output_relpath"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write frame JPEG: {output_path}")
            height, width = frame.shape[:2]
            decoded.append(
                {
                    **dict(row),
                    "image_width": int(width),
                    "image_height": int(height),
                    "frame_md5": md5_file(output_path),
                    "image_path": str(row["output_relpath"]),
                    "prelabel_box_count": 0,
                }
            )
    finally:
        capture.release()
    return decoded


def _cvat_annotations_xml(
    frames: Sequence[Mapping[str, Any]],
    prelabels: Sequence[Mapping[str, Any]],
    *,
    job_name: str,
) -> bytes:
    boxes_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prelabels:
        boxes_by_image[str(row["output_name"])].append(row)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = job_name
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "person"
    ET.SubElement(label, "color").text = "#f2d13d"
    for image_id, frame in enumerate(sorted(frames, key=lambda row: int(row["sample_ordinal"]))):
        name = str(frame["output_name"])
        image_node = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": f"images/{name}",
                "width": str(frame["image_width"]),
                "height": str(frame["image_height"]),
            },
        )
        for box_index, box in enumerate(boxes_by_image.get(name, []), start=1):
            x1, y1, x2, y2 = box["bbox_xyxy"]
            box_node = ET.SubElement(
                image_node,
                "box",
                {
                    "label": "person",
                    "source": "auto",
                    "occluded": "0",
                    "xtl": f"{float(x1):.3f}",
                    "ytl": f"{float(y1):.3f}",
                    "xbr": f"{float(x2):.3f}",
                    "ybr": f"{float(y2):.3f}",
                    "z_order": "0",
                },
            )
            attribute = ET.SubElement(box_node, "attribute", {"name": "teacher_conf"})
            attribute.text = f"{float(box['teacher_conf']):.6f}"
            box_id = ET.SubElement(box_node, "attribute", {"name": "prelabel_box_id"})
            box_id.text = str(box.get("box_id", box_index))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _write_cvat_job(
    *,
    out_dir: Path,
    frames: Sequence[Mapping[str, Any]],
    prelabels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = frames[0]
    partition = str(first["partition"])
    venue_id = str(first["venue_id"])
    venue_slug = _slug(venue_id)
    job_dir = out_dir / "cvat_upload" / partition / venue_slug
    if job_dir.exists():
        shutil.rmtree(job_dir)
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        source_path = out_dir / str(frame["image_path"])
        shutil.copy2(source_path, images_dir / str(frame["output_name"]))
    xml_bytes = _cvat_annotations_xml(frames, prelabels, job_name=venue_id)
    _write_bytes(job_dir / "annotations.xml", xml_bytes)
    task_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "person_fewshot_cvat_images_job",
        "cvat_format": "CVAT Images 1.1",
        "venue_id": venue_id,
        "partition": partition,
        "job_count": 1,
        "image_count": len(frames),
        "prelabel_box_count": len(prelabels),
        "class_names": ["person"],
        "teacher_derived": True,
        "ground_truth": False,
        "verified_by_owner": False,
        "media_local_only": True,
    }
    _write_json(job_dir / "task_manifest.json", task_manifest)
    zip_path = job_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(job_dir / "annotations.xml", "annotations.xml")
        archive.write(job_dir / "task_manifest.json", "task_manifest.json")
        for image_path in sorted(images_dir.iterdir()):
            archive.write(image_path, f"images/{image_path.name}")
    return {
        "venue_id": venue_id,
        "partition": partition,
        "job_dir": job_dir.relative_to(out_dir).as_posix(),
        "zip_path": zip_path.relative_to(out_dir).as_posix(),
        "zip_md5": md5_file(zip_path),
        "image_count": len(frames),
        "prelabel_box_count": len(prelabels),
    }


def _review_html(
    frames: Sequence[Mapping[str, Any]],
    prelabels: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
    decode_rows: Sequence[Mapping[str, Any]],
) -> str:
    boxes_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for box in prelabels:
        boxes_by_image[str(box["output_name"])].append(box)
    partition_order = {partition: index for index, partition in enumerate(PARTITIONS)}
    ordered_frames = sorted(
        frames,
        key=lambda row: (
            partition_order[str(row["partition"])],
            str(row["venue_id"]),
            int(row["sample_ordinal"]),
        ),
    )
    items: list[dict[str, Any]] = []
    for frame in ordered_frames:
        image_name = str(frame["output_name"])
        width = float(frame["image_width"])
        height = float(frame["image_height"])
        boxes = []
        for index, box in enumerate(boxes_by_image.get(image_name, []), start=1):
            x1, y1, x2, y2 = (float(value) for value in box["bbox_xyxy"])
            boxes.append(
                {
                    "box": index,
                    "confidence": round(float(box["teacher_conf"]), 4),
                    "left_pct": round(100 * x1 / width, 5),
                    "top_pct": round(100 * y1 / height, 5),
                    "width_pct": round(100 * (x2 - x1) / width, 5),
                    "height_pct": round(100 * (y2 - y1) / height, 5),
                }
            )
        items.append(
            {
                "frame_id": image_name,
                "venue": str(frame["venue_id"]),
                "partition": str(frame["partition"]),
                "sample_ordinal": int(frame["sample_ordinal"]),
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": float(frame["timestamp_s"]),
                "image": "../" + str(frame["image_path"]),
                "boxes": boxes,
            }
        )

    planned_by_venue = Counter(str(row["venue_id"]) for row in decode_rows)
    materialized_by_venue = Counter(str(row["venue_id"]) for row in frames)
    venue_cards: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}
    for row in sorted(
        assignment["assignments"],
        key=lambda value: (
            partition_order[str(value["partition"])],
            str(value["venue_id"]),
        ),
    ):
        venue = str(row["venue_id"])
        partition = str(row["partition"])
        planned = planned_by_venue[venue]
        materialized = materialized_by_venue[venue]
        status = "ready" if materialized == planned and planned else "pending"
        status_text = f"READY · {materialized} frames" if status == "ready" else f"PENDING · {planned} planned"
        venue_cards[partition].append(
            f'<article class="venue-card {status}" data-status="{status}">'
            f'<div><small>VENUE</small><strong>{html.escape(venue)}</strong></div>'
            f'<b>{status_text}</b></article>'
        )
    plan_sections = []
    for partition in PARTITIONS:
        label = partition.replace("_", " ")
        warning = (
            "TRAINABLE ONLY AFTER OWNER IMPORT"
            if partition == PARTITION_FINETUNE
            else "PROMOTION_JUDGE · NEVER TRAIN OR TUNE"
        )
        plan_sections.append(
            f'<section class="plan-partition {partition.lower()}">'
            f'<header><small>VENUE-DISJOINT PARTITION</small><h2>{label}</h2><p>{warning}</p></header>'
            f'<div class="venue-grid">{"".join(venue_cards[partition])}</div></section>'
        )

    assignment_sha = _sha256_bytes(_json_bytes(assignment))[:12]
    contract_json = _json_for_html_script(REVIEW_ANSWER_CONTRACT)
    items_json = _json_for_html_script(items)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Start Here · PERSON box review</title>
<style>
:root{{--ink:#13231d;--cream:#f3eddb;--paper:#fffaf0;--court:#255d45;--ball:#f0d43b;--red:#a33f35;--blue:#2d6778;--muted:#6d746c;--edge:#c8c0aa}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:linear-gradient(128deg,#e8dfca,#f8f3e6 48%,#dce8da);color:var(--ink);font-family:"Avenir Next Condensed","Gill Sans",sans-serif}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;background-image:repeating-linear-gradient(0deg,transparent 0 23px,#17332618 24px),linear-gradient(90deg,transparent 49.85%,#17332620 50%,transparent 50.15%)}}
.shell{{width:min(1120px,96vw);margin:auto;padding:18px 0 48px}}.topbar{{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;min-height:52px;padding:8px 14px;background:#13231df2;color:var(--cream);border-bottom:5px solid var(--ball)}}
#progress{{font:800 12px ui-monospace,SFMono-Regular,monospace;letter-spacing:.08em;text-transform:uppercase}}#export{{margin-left:auto;background:var(--ball);color:var(--ink)}}
.panel{{margin-top:18px;background:#fffaf0ed;border:1px solid var(--edge);box-shadow:10px 12px 0 #13231d18;padding:clamp(18px,4vw,38px)}}.kicker,.partition-stripe small,.venue-head small,.plan-partition small{{font:800 11px ui-monospace,SFMono-Regular,monospace;letter-spacing:.17em;text-transform:uppercase}}
h1{{font:700 clamp(42px,7vw,82px)/.9 "Iowan Old Style","Palatino Linotype",serif;margin:10px 0 18px;max-width:880px}}h2{{font:700 clamp(27px,4vw,44px)/1 "Iowan Old Style","Palatino Linotype",serif;margin:5px 0}}p{{font-size:17px;line-height:1.45}}kbd{{border:1px solid #9b998f;border-bottom-width:3px;border-radius:5px;background:white;padding:2px 7px;font:800 14px ui-monospace,monospace}}
.answer-key{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}}.answer-key div{{padding:12px;border:1px solid var(--ink);background:var(--paper)}}.answer-key b{{display:block;font-size:15px;margin-top:7px}}button{{border:1px solid #ffffff73;padding:13px 15px;background:var(--ink);color:white;font:800 14px ui-monospace,SFMono-Regular,monospace;letter-spacing:.03em;cursor:pointer;box-shadow:0 4px 0 #07100c}}button:hover{{filter:brightness(1.13);transform:translateY(-1px)}}button:active{{transform:translateY(2px);box-shadow:0 2px 0 #07100c}}
#start{{font-size:18px;background:var(--court)}}.resume{{border-left:8px solid var(--ball);padding:10px 14px;background:#ece5cf;font:800 13px ui-monospace,monospace}}.plan-partition{{margin-top:28px;padding-top:20px;border-top:8px double var(--ink)}}.plan-partition>header{{display:grid;grid-template-columns:1fr auto;align-items:end;border-left:10px solid var(--court);padding-left:14px}}.promotion_judge>header{{border-color:var(--red)}}.plan-partition>header small,.plan-partition>header h2{{grid-column:1}}.plan-partition>header p{{grid-column:2;grid-row:1/3;margin:0;font-weight:800;color:var(--red)}}
.venue-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;margin-top:14px}}.venue-card{{display:flex;justify-content:space-between;gap:10px;min-height:86px;padding:12px;background:white;border:1px solid var(--edge)}}.venue-card strong{{display:block;overflow-wrap:anywhere;margin-top:5px}}.venue-card b{{align-self:start;white-space:nowrap;font:800 11px ui-monospace,monospace;color:var(--muted)}}.venue-card.ready{{border:2px solid var(--court)}}.venue-card.ready b{{color:var(--court)}}
#reviewScreen{{padding:0;overflow:hidden}}.partition-stripe{{display:flex;justify-content:space-between;align-items:center;padding:13px 18px;background:var(--court);color:white}}.partition-stripe.judge{{background:var(--red)}}.partition-stripe b{{font:800 14px ui-monospace,monospace}}.venue-head{{display:flex;justify-content:space-between;gap:18px;padding:16px 18px;background:#e6dfca;border-bottom:1px solid var(--edge)}}#venueName{{font:700 18px "Iowan Old Style",serif;overflow-wrap:anywhere}}#frameMeta{{white-space:nowrap;font:700 12px ui-monospace,monospace;color:var(--muted)}}
.image-wrap{{position:relative;background:#06100c;line-height:0;overflow:hidden}}#frameImage{{display:block;width:100%;height:auto}}#overlays{{position:absolute;inset:0}}.box{{position:absolute;border:3px solid var(--ball);box-shadow:0 0 0 1px #000b}}.box span{{position:absolute;top:-1px;left:-1px;background:var(--ball);color:#000;padding:8px 5px 3px;font:800 10px ui-monospace,monospace;line-height:1}}
.question{{padding:17px 18px 8px;text-align:center;font:700 clamp(23px,4vw,36px) "Iowan Old Style",serif}}.buttons{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;padding:8px 18px 18px}}.buttons button[data-key="1"]{{background:var(--court)}}.buttons button[data-key="2"]{{background:var(--blue)}}.buttons button[data-key="3"]{{background:var(--red)}}.buttons button[data-key="4"]{{background:#656b66}}.secondary{{display:flex;justify-content:center;padding:0 18px 18px}}.hidden{{display:none!important}}#doneScreen{{text-align:center}}#doneScreen h1{{margin-inline:auto}}
@media(max-width:760px){{.answer-key,.buttons{{grid-template-columns:1fr 1fr}}.plan-partition>header{{display:block}}.plan-partition>header p{{margin-top:10px}}.venue-head{{display:block}}#frameMeta{{margin-top:7px}}.topbar{{align-items:flex-start}}}}
</style></head><body>
<div class="topbar"><div id="progress">Saved locally · 0 of {len(items)} answered</div><button type="button" id="export">Export results.json</button></div>
<main class="shell">
<section id="startScreen" class="panel"><div class="kicker">OWNER REVIEW · VERIFY-NOT-DRAW · PARTITION PIN {assignment_sha}</div><h1>Are all people boxed correctly?</h1><p>Review every yellow teacher box and the full frame. Players and spectators both count as people. Number keys answer and advance immediately.</p>
<div class="answer-key"><div><kbd>1</kbd><b>All boxes correct</b></div><div><kbd>2</kbd><b>A box is wrong</b></div><div><kbd>3</kbd><b>A person is missed</b></div><div><kbd>4</kbd><b>Unsure</b></div></div>
<p class="resume" id="resumeIndicator">No saved answers yet. Start at frame 1.</p><button type="button" id="start">Begin {len(items)} materialized frames</button>
<div id="venuePlan">{"".join(plan_sections)}</div></section>
<section id="reviewScreen" class="panel hidden"><div class="partition-stripe" id="partitionStripe"><small>PARTITION</small><b id="partitionName"></b></div><div class="venue-head"><div><small>VENUE</small><div id="venueName"></div></div><div id="frameMeta"></div></div>
<div class="image-wrap"><img id="frameImage" alt=""><div id="overlays" aria-hidden="true"></div></div><div class="question">Are all people boxed correctly?</div><div class="buttons"><button type="button" data-key="1">1 · ALL CORRECT</button><button type="button" data-key="2">2 · BOX WRONG</button><button type="button" data-key="3">3 · PERSON MISSED</button><button type="button" data-key="4">4 · UNSURE</button></div><div class="secondary"><button type="button" id="back">Go back one</button></div></section>
<section id="doneScreen" class="panel hidden"><div class="kicker">MATERIALIZED QUEUE COMPLETE</div><h1>Answers are saved locally.</h1><p id="summary"></p><p>Use the single <b>Export results.json</b> button above for the durable handoff.</p></section>
</main>
<script id="answer-contract" type="application/json">{contract_json}</script>
<script id="review-items" type="application/json">{items_json}</script>
<script>
const ITEMS=JSON.parse(document.getElementById("review-items").textContent);
const ANSWER_CONTRACT=JSON.parse(document.getElementById("answer-contract").textContent);
const STORAGE_KEY="trkC_fewshot_pack_20260722_"+"{assignment_sha}"+"_answers_v2";
const $=id=>document.getElementById(id);let answers={{}};try{{answers=JSON.parse(localStorage.getItem(STORAGE_KEY)||"{{}}")}}catch(error){{answers={{}}}}
let index=0;function show(id){{["startScreen","reviewScreen","doneScreen"].forEach(name=>$(name).classList.toggle("hidden",name!==id))}}
function firstOpen(){{const found=ITEMS.findIndex(item=>!answers[item.frame_id]);return found<0?ITEMS.length:found}}
function updateProgress(){{const count=ITEMS.filter(item=>answers[item.frame_id]).length;$("progress").textContent="Saved locally · "+count+" of "+ITEMS.length+" answered";const next=firstOpen();$("resumeIndicator").textContent=count?"Resume saved progress at frame "+Math.min(next+1,ITEMS.length)+". "+count+" answers are stored in this browser.":"No saved answers yet. Start at frame 1.";$("start").textContent=count?"Resume review":"Begin "+ITEMS.length+" materialized frames"}}
function persist(){{localStorage.setItem(STORAGE_KEY,JSON.stringify(answers));updateProgress()}}
function render(){{if(index>=ITEMS.length){{finish();return}}const item=ITEMS[index];show("reviewScreen");$("partitionName").textContent=item.partition;$("partitionStripe").classList.toggle("judge",item.partition==="PROMOTION_JUDGE");$("venueName").textContent=item.venue;$("frameMeta").textContent="#"+String(item.sample_ordinal).padStart(2,"0")+" · frame "+item.frame_index+" · "+item.timestamp_s.toFixed(2)+"s · "+item.boxes.length+" boxes";$("frameImage").src=item.image;$("frameImage").alt=item.frame_id;$("overlays").innerHTML=item.boxes.map(box=>'<span class="box" style="left:'+box.left_pct+'%;top:'+box.top_pct+'%;width:'+box.width_pct+'%;height:'+box.height_pct+'%"><span>'+box.box+'</span></span>').join("");$("back").style.visibility=index?"visible":"hidden";updateProgress()}}
function commit(key){{const item=ITEMS[index],rule=ANSWER_CONTRACT[key];answers[item.frame_id]={{frame_id:item.frame_id,venue:item.venue,partition:item.partition,answer:rule.answer,timestamp:new Date().toISOString()}};persist();index+=1;render()}}
function finish(){{show("doneScreen");const counts={{}};Object.values(answers).forEach(row=>counts[row.answer]=(counts[row.answer]||0)+1);$("summary").textContent=Object.entries(counts).map(([answer,count])=>answer+" "+count).join(" · ")||"No answers yet.";updateProgress()}}
$("start").onclick=()=>{{index=firstOpen();render()}};document.querySelectorAll("[data-key]").forEach(button=>button.onclick=()=>commit(button.dataset.key));$("back").onclick=()=>{{if(index>0){{index-=1;delete answers[ITEMS[index].frame_id];persist();render()}}}};
$("export").onclick=()=>{{const results=ITEMS.filter(item=>answers[item.frame_id]).map(item=>answers[item.frame_id]);const blob=new Blob([JSON.stringify(results,null,2)+"\\n"],{{type:"application/json"}}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="results.json";link.click();URL.revokeObjectURL(link.href)}};
document.addEventListener("keydown",event=>{{if($("reviewScreen").classList.contains("hidden")||event.repeat)return;if(ANSWER_CONTRACT[event.key]){{event.preventDefault();commit(event.key)}}}});updateProgress();
</script></body></html>"""


def emit_owner_review(
    out_dir: Path,
    *,
    model_manifest_path: Path = ROOT / MODEL_MANIFEST_RELATIVE_PATH,
) -> dict[str, Any]:
    """Regenerate only the offline owner page; accepted plan artifacts stay untouched."""

    manifest = _load_json(out_dir / "pack_manifest.json")
    assignment = _verify_partition_pin(out_dir, manifest)
    decode_rows = _load_jsonl(out_dir / "decode_plan.jsonl")
    venue_dirs = sorted((out_dir / "materialized").glob("*/frames.jsonl"))
    frames: list[dict[str, Any]] = []
    prelabels: list[dict[str, Any]] = []
    for frames_path in venue_dirs:
        frames.extend(_load_jsonl(frames_path))
        prelabels.extend(_load_jsonl(frames_path.with_name("prelabels_teacher.jsonl")))
    validate_prelabel_rows(
        prelabels,
        model_manifest_path=model_manifest_path,
    )
    planned_by_venue = Counter(str(row["venue_id"]) for row in decode_rows)
    materialized_by_venue = Counter(str(row["venue_id"]) for row in frames)
    for venue, count in materialized_by_venue.items():
        if count != planned_by_venue[venue]:
            raise ValueError(
                f"materialized venue {venue} has {count} frames; plan requires {planned_by_venue[venue]}"
            )
    for frame in frames:
        image_path = out_dir / str(frame["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"review image is absent: {image_path}")
    review_html = _review_html(frames, prelabels, assignment, decode_rows)
    review_dir = out_dir / "review"
    _write_bytes(review_dir / "START_HERE.html", review_html.encode("utf-8"))
    redirect = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" '
        'content="0; url=START_HERE.html"><title>Open owner review</title>'
        '<p><a href="START_HERE.html">Open START_HERE.html</a></p>\n'
    )
    _write_bytes(review_dir / "index.html", redirect.encode("utf-8"))
    return {
        "review_html": "review/START_HERE.html",
        "index_redirect": "review/index.html",
        "materialized_venues": len(materialized_by_venue),
        "materialized_frames": len(frames),
        "pending_venues": len(assignment["assignments"]) - len(materialized_by_venue),
        "teacher_boxes": len(prelabels),
    }


def validate_cvat_package(out_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _load_json(out_dir / "pack_manifest.json")
    assignment = _verify_partition_pin(out_dir, manifest)
    assigned = {str(row["venue_id"]): str(row["partition"]) for row in assignment["assignments"]}
    for partition in PARTITIONS:
        if not (out_dir / "cvat_upload" / partition).is_dir():
            errors.append(f"missing visibly separate CVAT partition directory {partition}")
    materialized = manifest.get("materialized_venues") or []
    for venue in materialized:
        venue_id = str(venue["venue_id"])
        partition = str(venue["partition"])
        if assigned.get(venue_id) != partition:
            errors.append(f"{venue_id} is packaged under the wrong partition")
            continue
        job_dir = out_dir / str(venue["cvat_job_dir"])
        zip_path = out_dir / str(venue["cvat_zip_path"])
        required = [job_dir / "images", job_dir / "annotations.xml", job_dir / "task_manifest.json", zip_path]
        for path in required:
            if not path.exists():
                errors.append(f"missing CVAT job artifact {path.relative_to(out_dir)}")
        if (job_dir / "annotations.xml").is_file():
            root = ET.parse(job_dir / "annotations.xml").getroot()
            images = root.findall("image")
            if len(images) != int(venue["frame_count"]):
                errors.append(f"{venue_id} XML image count differs from frame count")
    fine = {venue for venue, partition in assigned.items() if partition == PARTITION_FINETUNE}
    judge = {venue for venue, partition in assigned.items() if partition == PARTITION_JUDGE}
    if fine & judge:
        errors.append("partition assignment is not venue-disjoint")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "materialized_job_count": len(materialized),
        "partition_directories": list(PARTITIONS),
        "venue_disjoint": not bool(fine & judge),
    }


def _regenerate_review_artifacts(
    out_dir: Path,
    *,
    force_with_backup: bool = False,
    verified_labels_already_guarded: bool = False,
    model_manifest_path: Path = ROOT / MODEL_MANIFEST_RELATIVE_PATH,
) -> dict[str, Any]:
    if not verified_labels_already_guarded:
        _guard_verified_labels(out_dir, force_with_backup=force_with_backup)
    manifest = _load_json(out_dir / "pack_manifest.json")
    assignment = _verify_partition_pin(out_dir, manifest)
    venue_dirs = sorted((out_dir / "materialized").glob("*/frames.jsonl"))
    all_frames: list[dict[str, Any]] = []
    all_prelabels: list[dict[str, Any]] = []
    for frames_path in venue_dirs:
        all_frames.extend(_load_jsonl(frames_path))
        all_prelabels.extend(_load_jsonl(frames_path.with_name("prelabels_teacher.jsonl")))
    validate_prelabel_rows(
        all_prelabels,
        model_manifest_path=model_manifest_path,
    )
    _write_jsonl(out_dir / "prelabels_teacher.jsonl", all_prelabels)
    _write_bytes(out_dir / "verified_labels.jsonl", b"")
    for partition in PARTITIONS:
        (out_dir / "cvat_upload" / partition).mkdir(parents=True, exist_ok=True)
    frames_by_venue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boxes_by_venue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in all_frames:
        frames_by_venue[str(frame["venue_id"])].append(frame)
    for box in all_prelabels:
        boxes_by_venue[str(box["venue_id"])].append(box)
    materialized_rows: list[dict[str, Any]] = []
    for venue_id in sorted(frames_by_venue):
        frames = frames_by_venue[venue_id]
        package = _write_cvat_job(
            out_dir=out_dir,
            frames=frames,
            prelabels=boxes_by_venue[venue_id],
        )
        media_path = out_dir / "materialized" / _slug(venue_id) / "media_identity.json"
        media_identity = _load_json(media_path)
        materialized_rows.append(
            {
                "venue_id": venue_id,
                "source_id": str(frames[0]["source_id"]),
                "partition": str(frames[0]["partition"]),
                "frame_count": len(frames),
                "teacher_box_count": len(boxes_by_venue[venue_id]),
                "frame_manifest": (
                    Path("materialized") / _slug(venue_id) / "frames.jsonl"
                ).as_posix(),
                "prelabels": (
                    Path("materialized") / _slug(venue_id) / "prelabels_teacher.jsonl"
                ).as_posix(),
                "verified_labels": (
                    Path("materialized") / _slug(venue_id) / "verified_labels.jsonl"
                ).as_posix(),
                "frame_md5_by_name": {
                    str(frame["output_name"]): str(frame["frame_md5"])
                    for frame in frames
                },
                "media_sha256": str(media_identity["media_sha256"]),
                "media_md5": str(media_identity["media_md5"]),
                "cvat_job_dir": package["job_dir"],
                "cvat_zip_path": package["zip_path"],
                "cvat_zip_md5": package["zip_md5"],
                "teacher_derived": True,
                "ground_truth": False,
                "verified_by_owner": False,
                "training_eligible": False,
                "media_local_only": True,
            }
        )
    review_summary = emit_owner_review(
        out_dir,
        model_manifest_path=model_manifest_path,
    )
    manifest["materialized_venues"] = materialized_rows
    manifest["counts"]["materialized_venues"] = len(materialized_rows)
    manifest["counts"]["materialized_frames"] = len(all_frames)
    manifest["counts"]["teacher_boxes"] = len(all_prelabels)
    manifest["counts"]["verified_label_rows"] = 0
    manifest["labels"]["verified_labels"]["row_count"] = 0
    manifest["artifacts"]["review_html"] = review_summary["review_html"]
    manifest["artifacts"]["cvat_upload_root"] = "cvat_upload"
    _write_json(out_dir / "pack_manifest.json", manifest)
    _write_bytes(out_dir / "OWNER_ASK.md", _owner_ask_markdown(manifest, assignment).encode("utf-8"))
    validation = validate_cvat_package(out_dir)
    _write_json(out_dir / "cvat_package_validation.json", validation)
    if validation["status"] != "PASS":
        raise ValueError(f"CVAT package validation failed: {validation['errors']}")
    return manifest


def materialize_source(
    *,
    out_dir: Path,
    source_id: str,
    media_path: Path,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
    predictor: PredictionFn | None = None,
    overwrite: bool = False,
    force_with_backup: bool = False,
    model_manifest_path: Path = ROOT / MODEL_MANIFEST_RELATIVE_PATH,
) -> dict[str, Any]:
    manifest = _load_json(out_dir / "pack_manifest.json")
    verified_label_backups = _guard_verified_labels(
        out_dir,
        force_with_backup=force_with_backup,
    )
    assignment = _verify_partition_pin(out_dir, manifest)
    sources = [row for row in manifest["sources"] if str(row["source_id"]) == source_id]
    if len(sources) != 1:
        raise ValueError(f"source {source_id!r} is not uniquely present in the pack")
    source = sources[0]
    assert_source_allowed(source)
    if not media_path.is_file():
        raise FileNotFoundError(f"local media is absent: {media_path}")
    actual_sha = sha256_file(media_path)
    expected_sha = str(source["expected_media_sha256"])
    assert_source_allowed(
        {"source_id": source_id, "media_sha256": actual_sha, "path": media_path.as_posix()}
    )
    if actual_sha != expected_sha:
        raise ValueError(
            f"media SHA-256 mismatch for {source_id}: expected={expected_sha}, actual={actual_sha}"
        )
    venue_id = str(source["venue_id"])
    venue_dir = out_dir / "materialized" / _slug(venue_id)
    if venue_dir.exists():
        if not overwrite:
            raise FileExistsError(f"venue is already materialized: {venue_dir}")
        shutil.rmtree(venue_dir)
    plan_rows = [
        row
        for row in _load_jsonl(out_dir / "decode_plan.jsonl")
        if str(row["source_id"]) == source_id
    ]
    if len(plan_rows) != FRAMES_PER_VENUE:
        raise ValueError(f"source {source_id} must have exactly {FRAMES_PER_VENUE} decode rows")
    assigned_partition = next(
        row["partition"] for row in assignment["assignments"] if row["venue_id"] == venue_id
    )
    if any(row.get("partition") != assigned_partition for row in plan_rows):
        raise ValueError("decode rows disagree with the pinned venue partition")
    frames = _decode_frames(media_path=media_path, plan_rows=plan_rows, out_dir=out_dir)

    checkpoint_ref = checkpoint_path or Path(str(manifest["teacher"]["checkpoint_path"]))
    if not checkpoint_ref.is_absolute():
        checkpoint_ref = ROOT / checkpoint_ref
    checkpoint_sha = sha256_file(checkpoint_ref)
    if checkpoint_sha != str(manifest["teacher"]["checkpoint_sha256"]):
        raise ValueError("materialize checkpoint does not match the plan's pinned YOLO26m SHA-256")
    image_paths = [out_dir / str(frame["image_path"]) for frame in frames]
    prediction_fn = predictor or _run_yolo_predictions
    predictions = prediction_fn(image_paths, checkpoint_ref, TEACHER_CONFIDENCE, device)
    if len(predictions) != len(frames):
        raise ValueError("predictor result count differs from decoded frame count")
    prelabels: list[dict[str, Any]] = []
    for frame, frame_predictions in zip(frames, predictions, strict=True):
        frame["prelabel_box_count"] = len(frame_predictions)
        for box_id, prediction in enumerate(frame_predictions, start=1):
            coordinates = [float(value) for value in prediction["bbox_xyxy"]]
            if len(coordinates) != 4:
                raise ValueError("teacher bbox_xyxy must contain four numbers")
            x1, y1, x2, y2 = coordinates
            if not (0 <= x1 < x2 <= float(frame["image_width"])):
                raise ValueError(f"teacher box has invalid x coordinates: {coordinates}")
            if not (0 <= y1 < y2 <= float(frame["image_height"])):
                raise ValueError(f"teacher box has invalid y coordinates: {coordinates}")
            confidence = float(prediction["confidence"])
            if confidence < TEACHER_CONFIDENCE:
                raise ValueError("predictor returned a box below the registered confidence floor")
            prelabels.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_id": source_id,
                    "source_pool": str(source["source_pool"]),
                    "source_family_id": str(source["source_family_id"]),
                    "venue_id": venue_id,
                    "venue_family_id": str(source["venue_family_id"]),
                    "partition": str(frame["partition"]),
                    "frame_index": int(frame["frame_index"]),
                    "timestamp_s": float(frame["timestamp_s"]),
                    "sample_ordinal": int(frame["sample_ordinal"]),
                    "output_name": str(frame["output_name"]),
                    "image_path": str(frame["image_path"]),
                    "image_width": int(frame["image_width"]),
                    "image_height": int(frame["image_height"]),
                    "box_id": box_id,
                    "class_id": 0,
                    "class_name": "person",
                    "bbox_xyxy": [round(value, 3) for value in coordinates],
                    "teacher_conf": round(confidence, 8),
                    "teacher_model_id": TEACHER_ID,
                    "teacher_checkpoint_sha256": checkpoint_sha,
                    "teacher_derived": True,
                    "ground_truth": False,
                    "verified_by_owner": False,
                    "label_state": "PRELABEL_ONLY",
                    "training_eligible": False,
                    "production_eligible": False,
                    "do_not_promote": True,
                }
            )
    validate_prelabel_rows(
        prelabels,
        model_manifest_path=model_manifest_path,
    )
    _write_jsonl(venue_dir / "frames.jsonl", frames)
    _write_jsonl(venue_dir / "prelabels_teacher.jsonl", prelabels)
    _write_bytes(venue_dir / "verified_labels.jsonl", b"")
    _write_json(
        venue_dir / "media_identity.json",
        {
            "source_id": source_id,
            "venue_id": venue_id,
            "partition": assigned_partition,
            "media_sha256": actual_sha,
            "media_md5": md5_file(media_path),
            "media_path_local_only": str(media_path.resolve()),
            "redistribution_allowed": False,
        },
    )
    updated = _regenerate_review_artifacts(
        out_dir,
        force_with_backup=force_with_backup,
        verified_labels_already_guarded=True,
        model_manifest_path=model_manifest_path,
    )
    return {
        "source_id": source_id,
        "venue_id": venue_id,
        "partition": assigned_partition,
        "decoded_frames": len(frames),
        "teacher_boxes": len(prelabels),
        "review_html": updated["artifacts"]["review_html"],
        "cvat_validation": _load_json(out_dir / "cvat_package_validation.json"),
        "verified_by_owner": False,
        "verified_label_backups": [
            path.relative_to(out_dir).as_posix() for path in verified_label_backups
        ],
    }


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/materialize the quarantined, venue-disjoint PERSON few-shot owner pack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write the deterministic content-blind decode plan.")
    plan.add_argument("--repo-root", type=Path, default=ROOT)
    plan.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    plan.add_argument(
        "--gallery-manifest",
        type=Path,
        default=Path("data/pbvision_gallery_20260719/MANIFEST.json"),
    )
    plan.add_argument(
        "--harvest-manifest",
        type=Path,
        default=Path("data/online_harvest_20260706/manifest.json"),
    )
    plan.add_argument("--model-manifest", type=Path, default=Path("models/MANIFEST.json"))
    plan.add_argument("--checkpoint", type=Path, default=TEACHER_RELATIVE_PATH)
    plan.add_argument(
        "--gallery-root", type=Path, default=Path("data/pbvision_gallery_20260719")
    )
    plan.add_argument("--eval-root", type=Path, default=Path("eval_clips"))
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument(
        "--force-with-backup",
        action="store_true",
        help="Back up any nonempty verified_labels.jsonl before a permitted overwrite.",
    )

    materialize = subparsers.add_parser(
        "materialize", help="Decode and pre-label one SHA-pinned local venue."
    )
    materialize.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    materialize.add_argument("--source-id", required=True)
    materialize.add_argument("--media", type=Path, required=True)
    materialize.add_argument("--checkpoint", type=Path, default=None)
    materialize.add_argument("--device", default="cpu")
    materialize.add_argument("--overwrite", action="store_true")
    materialize.add_argument(
        "--force-with-backup",
        action="store_true",
        help="Back up any nonempty verified_labels.jsonl before regeneration.",
    )

    validate = subparsers.add_parser("validate", help="Validate the materialized CVAT layout.")
    validate.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    emit_review = subparsers.add_parser(
        "emit-review",
        help="Regenerate START_HERE.html without mutating accepted plan or manifest artifacts.",
    )
    emit_review.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        repo_root = args.repo_root.resolve()
        out_dir = _resolve_repo_path(repo_root, args.out_dir)
        result = build_pack(
            repo_root=repo_root,
            out_dir=out_dir,
            gallery_manifest=_resolve_repo_path(repo_root, args.gallery_manifest),
            harvest_manifest=_resolve_repo_path(repo_root, args.harvest_manifest),
            model_manifest=_resolve_repo_path(repo_root, args.model_manifest),
            checkpoint=_resolve_repo_path(repo_root, args.checkpoint),
            gallery_root=_resolve_repo_path(repo_root, args.gallery_root),
            eval_root=_resolve_repo_path(repo_root, args.eval_root),
            seed=args.seed,
            force_with_backup=args.force_with_backup,
        )
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "venue_families": result["counts"]["venue_families"],
                    "planned_frames": result["counts"]["planned_frames"],
                    "partition_counts": result["partition_counts"],
                    "missing_gallery_venues": result["missing_gallery_venues"],
                    "verified_by_owner": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "materialize":
        result = materialize_source(
            out_dir=args.out_dir,
            source_id=args.source_id,
            media_path=args.media.expanduser(),
            checkpoint_path=args.checkpoint,
            device=args.device,
            overwrite=args.overwrite,
            force_with_backup=args.force_with_backup,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "emit-review":
        result = emit_owner_review(args.out_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = validate_cvat_package(args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
