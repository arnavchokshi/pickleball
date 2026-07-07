#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.roboflow_corpus import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    ProtectedEvalHashCollisionError,
    assert_no_protected_eval_hash_collisions,
    load_protected_eval_hashes,
)


ARTIFACT_TYPE = "racketsport_corpus_dashboard"
RESERVED_HELDOUT_IDS = ("pwxNwFfYQlQ", "vQhtz8l6VqU")
TRAIN_VAL_ROLES = {"train", "internal_val", "val", "valid", "validation", "train_val"}


def build_dashboard(
    *,
    root: Path,
    roboflow_aggregated: Path,
    harvest_root: Path,
    owner_capture_manifest: Path,
    review_label_root: Path,
    eval_root: Path,
    eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
    hash_harvest_videos: bool = True,
) -> dict[str, Any]:
    eval_hashes, eval_hash_source = load_protected_eval_hashes(
        eval_root=eval_root,
        eval_sample_every_s=eval_sample_every_s,
    )
    eval_hash_count = sum(len(values) for values in eval_hashes.values())
    leakage: dict[str, Any] = {
        "eval_hash_collisions": 0,
        "heldout_id_hits": 0,
        "eval_hash_hits": [],
        "heldout_id_hit_details": [],
    }
    sources: dict[str, dict[str, Any]] = {}

    sources["roboflow"] = _load_roboflow_source(roboflow_aggregated, eval_hashes, leakage)
    sources["harvest"] = _load_harvest_source(
        harvest_root,
        eval_hashes,
        leakage,
        hash_videos=hash_harvest_videos,
        eval_sample_every_s=eval_sample_every_s,
    )
    sources["owner_capture"] = _load_owner_capture_source(owner_capture_manifest, eval_hashes, leakage)
    sources["human_review_labels"] = _load_human_review_source(review_label_root)

    leakage["eval_hash_collisions"] = len(leakage["eval_hash_hits"])
    leakage["heldout_id_hits"] = len(leakage["heldout_id_hit_details"])

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": str(root),
        "eval_hash_check": {
            "eval_root": str(eval_root),
            "hash_source": eval_hash_source,
            "eval_hash_count": eval_hash_count,
            "eval_hash_counts": {key: len(values) for key, values in sorted(eval_hashes.items())},
            "sample_every_s": eval_sample_every_s,
        },
        "reserved_heldout_ids": list(RESERVED_HELDOUT_IDS),
        "sources": sources,
        "rollup": _rollup_sources(sources),
        "leakage": leakage,
    }


def _load_roboflow_source(
    aggregated: Path,
    eval_hashes: Mapping[str, Sequence[int]],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    source = _empty_source("roboflow", aggregated, unit="samples")
    index_path = aggregated / "corpus_index.json"
    card_path = aggregated / "corpus_card.json"
    if not index_path.is_file():
        return source

    source["present"] = True
    payload = _read_json(index_path)
    samples = _as_list(payload.get("samples"))
    roles: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    label_kinds: Counter[str] = Counter()
    source_slugs: set[str] = set()
    sequence_ids: set[str] = set()
    hash_samples: list[Mapping[str, Any]] = []
    threshold = int(payload.get("hash", {}).get("collision_hamming_threshold", DEFAULT_DEDUP_THRESHOLD))

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        role = _normalize_role(sample.get("role", sample.get("split")))
        roles[role] += 1
        visibility[_sample_visibility(sample)] += 1
        for label_kind in _as_list(sample.get("label_kinds")):
            label_kinds[str(label_kind)] += 1
        if sample.get("source_slug"):
            source_slugs.add(str(sample["source_slug"]))
        temporal = sample.get("temporal")
        if isinstance(temporal, Mapping) and temporal.get("sequence_id"):
            sequence_ids.add(str(temporal["sequence_id"]))
        if isinstance(sample.get("hashes"), Mapping) and sample["hashes"].get("dhash"):
            hash_samples.append(sample)
        leakage["heldout_id_hit_details"].extend(
            _reserved_id_hits_for_record(
                sample,
                source="roboflow",
                path=index_path,
                record_id=str(sample.get("sample_id", "<unknown>")),
            )
        )

    source["counts"] = {
        "total": len(samples),
        "role_breakdown": _counter_dict(roles),
        "visibility_breakdown": _counter_dict(visibility),
        "label_kind_breakdown": _counter_dict(label_kinds),
    }
    source["diversity"] = {
        "distinct_sessions": len(source_slugs),
        "distinct_sequence_ids": len(sequence_ids),
        "distinct_courts": 0,
        "distinct_channels": 0,
        "unavailable": ["courts", "channels"],
    }
    if card_path.is_file():
        card = _read_json(card_path)
        source["dedup"] = _roboflow_dedup(card, kept_count=len(samples))
        source["recorded_corpus_index_sample_count"] = card.get("corpus_index_sample_count")
    source["leakage_check"] = _eval_hash_check_for_samples(
        hash_samples,
        eval_hashes=eval_hashes,
        threshold=threshold,
        source="roboflow",
        leakage=leakage,
    )
    return source


def _load_harvest_source(
    harvest_root: Path,
    eval_hashes: Mapping[str, Sequence[int]],
    leakage: dict[str, Any],
    *,
    hash_videos: bool,
    eval_sample_every_s: float,
) -> dict[str, Any]:
    source = _empty_source("harvest", harvest_root, unit="clips")
    card_path = harvest_root / "corpus_card.json"
    rally_path = harvest_root / "rally_clip_manifest.json"
    shard_path = harvest_root / "prelabel_shard_manifest.json"
    dedup_path = harvest_root / "dedup_report.json"
    if not any(path.is_file() for path in (card_path, rally_path, shard_path, dedup_path)):
        return source

    source["present"] = True
    clips: list[Mapping[str, Any]] = []
    if rally_path.is_file():
        rally = _read_json(rally_path)
        clips = [item for item in _as_list(rally.get("clips")) if isinstance(item, Mapping)]
    roles = Counter(_normalize_role(clip.get("role")) for clip in clips)
    sessions = {str(clip.get("source_id")) for clip in clips if clip.get("source_id")}
    channels = {str(clip.get("source_channel")) for clip in clips if clip.get("source_channel")}

    source["counts"] = {
        "total": len(clips),
        "role_breakdown": _counter_dict(roles),
        "visibility_breakdown": {"unlabeled": len(clips)} if clips else {},
    }
    source["diversity"] = {
        "distinct_sessions": len(sessions),
        "distinct_courts": 0,
        "distinct_channels": len(channels),
        "unavailable": ["courts"],
    }

    if card_path.is_file():
        card = _read_json(card_path)
        summary = card.get("summary") if isinstance(card.get("summary"), Mapping) else {}
        after = summary.get("role_split_after_ruling") if isinstance(summary, Mapping) else None
        if isinstance(after, Mapping):
            source["recorded_role_split_after_ruling"] = {str(k): int(v) for k, v in after.items()}
    if dedup_path.is_file():
        source["dedup"] = _harvest_dedup(_read_json(dedup_path))

    for clip in clips:
        leakage["heldout_id_hit_details"].extend(
            _reserved_id_hits_for_record(
                clip,
                source="harvest.rally_clip_manifest",
                path=rally_path,
                record_id=str(clip.get("clip_id", "<unknown>")),
            )
        )

    shard_items = _harvest_shard_items(shard_path)
    for item in shard_items:
        leakage["heldout_id_hit_details"].extend(
            _reserved_id_hits_for_record(
                item,
                source="harvest.prelabel_shard_manifest",
                path=shard_path,
                record_id=str(item.get("clip_id", "<unknown>")),
            )
        )

    hash_samples: list[Mapping[str, Any]] = []
    missing_hash_videos: list[str] = []
    if hash_videos:
        for clip in clips:
            if _normalize_role(clip.get("role")) not in TRAIN_VAL_ROLES:
                continue
            clip_path = Path(str(clip.get("clip_path", "")))
            if not clip_path.is_file():
                missing_hash_videos.append(str(clip_path))
                continue
            hash_samples.extend(
                _video_hash_samples(
                    clip_path,
                    sample_id_prefix=str(clip.get("clip_id", clip_path.stem)),
                    source_slug=str(clip.get("source_id", clip_path.parent.name)),
                    sample_every_s=eval_sample_every_s,
                )
            )
    source["leakage_check"] = _eval_hash_check_for_samples(
        hash_samples,
        eval_hashes=eval_hashes,
        threshold=DEFAULT_DEDUP_THRESHOLD,
        source="harvest_video_hashes",
        leakage=leakage,
    )
    source["leakage_check"]["missing_video_count"] = len(missing_hash_videos)
    source["leakage_check"]["missing_videos"] = missing_hash_videos[:20]
    source["prelabel_shard_item_count"] = len(shard_items)
    return source


def _load_owner_capture_source(
    manifest_path: Path,
    eval_hashes: Mapping[str, Sequence[int]],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    source = _empty_source("owner_capture", manifest_path, unit="samples")
    if not manifest_path.is_file():
        return source

    source["present"] = True
    manifest = _read_json(manifest_path)
    samples = [item for item in _as_list(manifest.get("samples")) if isinstance(item, Mapping)]
    roles: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    sessions: set[str] = set()
    courts: set[str] = set()
    hash_samples: list[Mapping[str, Any]] = []
    for sample in samples:
        role = _owner_sample_role(sample)
        roles[role] += 1
        visibility[_sample_visibility(sample)] += 1
        if sample.get("capture_id"):
            sessions.add(str(sample["capture_id"]))
        if sample.get("court_id"):
            courts.add(str(sample["court_id"]))
        if isinstance(sample.get("hashes"), Mapping) and sample["hashes"].get("dhash"):
            hash_samples.append(sample)
        leakage["heldout_id_hit_details"].extend(
            _reserved_id_hits_for_record(
                sample,
                source="owner_capture",
                path=manifest_path,
                record_id=str(sample.get("capture_id", sample.get("sample_id", "<unknown>"))),
            )
        )
    source["counts"] = {
        "total": len(samples),
        "role_breakdown": _counter_dict(roles),
        "visibility_breakdown": _counter_dict(visibility),
    }
    source["diversity"] = {
        "distinct_sessions": len(sessions),
        "distinct_courts": len(courts),
        "distinct_channels": 1 if samples else 0,
    }
    source["leakage_check"] = _eval_hash_check_for_samples(
        hash_samples,
        eval_hashes=eval_hashes,
        threshold=DEFAULT_DEDUP_THRESHOLD,
        source="owner_capture",
        leakage=leakage,
    )
    return source


def _load_human_review_source(review_root: Path) -> dict[str, Any]:
    source = _empty_source("human_review_labels", review_root, unit="boxes")
    if not review_root.is_dir():
        return source
    source["present"] = True
    visibility: Counter[str] = Counter()
    boxes_by_clip: Counter[str] = Counter()
    parse_errors: list[str] = []
    for xml_path in sorted(review_root.glob("*/annotations.xml")):
        try:
            clip_counts = _parse_cvat_ball_boxes(xml_path)
        except ET.ParseError as exc:
            parse_errors.append(f"{xml_path}: {exc}")
            continue
        clip_id = xml_path.parent.name
        for level, count in clip_counts.items():
            visibility[level] += count
            boxes_by_clip[clip_id] += count
    total = sum(visibility.values())
    source["counts"] = {
        "total": total,
        "role_breakdown": {"human_verified": total} if total else {},
        "visibility_breakdown": _counter_dict(visibility),
    }
    source["diversity"] = {
        "distinct_sessions": len(boxes_by_clip),
        "distinct_courts": 0,
        "distinct_channels": 0,
        "unavailable": ["courts", "channels"],
    }
    source["parse_status"] = "parsed" if not parse_errors else "unparsed"
    source["parse_errors"] = parse_errors
    source["boxes_by_clip"] = _counter_dict(boxes_by_clip)
    return source


def _eval_hash_check_for_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    eval_hashes: Mapping[str, Sequence[int]],
    threshold: int,
    source: str,
    leakage: dict[str, Any],
) -> dict[str, Any]:
    checked = 0
    before = len(leakage["eval_hash_hits"])
    for sample in samples:
        if not isinstance(sample.get("hashes"), Mapping) or not sample["hashes"].get("dhash"):
            continue
        checked += 1
        try:
            assert_no_protected_eval_hash_collisions([sample], eval_hashes=eval_hashes, threshold=threshold)
        except ProtectedEvalHashCollisionError as exc:
            leakage["eval_hash_hits"].append(
                _collision_hit_from_error(sample, source=source, message=str(exc))
            )
    return {
        "sample_hashes_checked": checked,
        "eval_hash_collisions": len(leakage["eval_hash_hits"]) - before,
        "collision_hamming_threshold": threshold,
    }


def _collision_hit_from_error(sample: Mapping[str, Any], *, source: str, message: str) -> dict[str, Any]:
    match = re.search(r" vs (?P<clip>\S+) hamming=(?P<hamming>\d+)", message)
    hit = {
        "source": source,
        "sample_id": str(sample.get("sample_id", "<unknown>")),
        "source_slug": str(sample.get("source_slug", "")),
        "message": message,
    }
    if match:
        hit["eval_clip"] = match.group("clip")
        hit["hamming_distance"] = int(match.group("hamming"))
    hashes = sample.get("hashes")
    if isinstance(hashes, Mapping) and hashes.get("dhash"):
        hit["dhash"] = str(hashes["dhash"])
    return hit


def _reserved_id_hits_for_record(
    record: Mapping[str, Any],
    *,
    source: str,
    path: Path,
    record_id: str,
) -> list[dict[str, Any]]:
    role = _normalize_role(record.get("role", record.get("split", record.get("eval_role"))))
    if role not in TRAIN_VAL_ROLES:
        return []
    text = json.dumps(record, sort_keys=True, default=str)
    hits = []
    for heldout_id in RESERVED_HELDOUT_IDS:
        if heldout_id in text:
            hits.append(
                {
                    "source": source,
                    "path": str(path),
                    "record_id": record_id,
                    "role": role,
                    "heldout_id": heldout_id,
                }
            )
    return hits


def _video_hash_samples(
    video_path: Path,
    *,
    sample_id_prefix: str,
    source_slug: str,
    sample_every_s: float,
) -> list[dict[str, Any]]:
    from threed.racketsport.online_harvest_ingest import perceptual_hash_video

    samples: list[dict[str, Any]] = []
    for hash_index, dhash in enumerate(perceptual_hash_video(video_path, sample_every_s=sample_every_s)):
        samples.append(
            {
                "sample_id": f"{sample_id_prefix}:video_dhash:{hash_index}",
                "source_slug": source_slug,
                "hashes": {"dhash": f"{int(dhash):016x}"},
            }
        )
    return samples


def _harvest_shard_items(shard_path: Path) -> list[Mapping[str, Any]]:
    if not shard_path.is_file():
        return []
    shard_manifest = _read_json(shard_path)
    items: list[Mapping[str, Any]] = []
    for shard in _as_list(shard_manifest.get("shards")):
        if not isinstance(shard, Mapping):
            continue
        items.extend(item for item in _as_list(shard.get("items")) if isinstance(item, Mapping))
    return items


def _parse_cvat_ball_boxes(xml_path: Path) -> Counter[str]:
    root = ET.parse(xml_path).getroot()
    counts: Counter[str] = Counter()
    for track in root.findall("track"):
        if track.attrib.get("label") != "ball":
            continue
        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue
            counts[_box_visibility(box)] += 1
    for image in root.findall("image"):
        for box in image.findall("box"):
            if box.attrib.get("label") == "ball":
                counts[_box_visibility(box)] += 1
    return counts


def _box_visibility(box: ET.Element) -> str:
    for attr in box.findall("attribute"):
        if attr.attrib.get("name") == "visibility_level" and attr.text:
            return attr.text.strip() or "unlabeled"
    return "unlabeled"


def _owner_sample_role(sample: Mapping[str, Any]) -> str:
    if sample.get("role") or sample.get("split"):
        return _normalize_role(sample.get("role", sample.get("split")))
    if sample.get("review_status") == "reviewed":
        return "train"
    return "held_out"


def _sample_visibility(sample: Mapping[str, Any]) -> str:
    if sample.get("visibility_level"):
        return str(sample["visibility_level"])
    labels = sample.get("labels")
    if isinstance(labels, Mapping):
        levels: list[str] = []
        for label_list in labels.values():
            for label in _as_list(label_list):
                if isinstance(label, Mapping) and label.get("visibility_level"):
                    levels.append(str(label["visibility_level"]))
        if levels:
            return levels[0] if len(set(levels)) == 1 else "mixed"
    return "unlabeled"


def _normalize_role(value: Any) -> str:
    text = str(value or "unlabeled").strip().lower()
    if text in {"valid", "validation", "val"}:
        return "internal_val"
    if text in {"test", "eval", "evaluation"}:
        return "eval"
    if text in {"heldout", "held_out", "heldout_candidate_proposed", "held_out_candidate_proposed"}:
        return "held_out"
    return text


def _roboflow_dedup(card: Mapping[str, Any], *, kept_count: int) -> dict[str, Any]:
    dedup = card.get("dedup")
    if not isinstance(dedup, Mapping):
        return {"considered": None, "removed": None, "rate": None}
    considered = _maybe_int(dedup.get("considered_sample_count"))
    removed = considered - kept_count if considered is not None else None
    return {
        "considered": considered,
        "removed": removed,
        "rate": dedup.get("dedup_rate"),
        "collision_hamming_threshold": dedup.get("collision_hamming_threshold"),
    }


def _harvest_dedup(report: Mapping[str, Any]) -> dict[str, Any]:
    hash_counts = report.get("harvest_hash_counts")
    considered = sum(int(value) for value in hash_counts.values()) if isinstance(hash_counts, Mapping) else None
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    return {
        "considered": considered,
        "removed": None,
        "rate": None,
        "collision_count": summary.get("collision_count"),
        "cross_source_collision_count": summary.get("cross_source_collision_count"),
        "eval_collision_count": summary.get("eval_collision_count"),
    }


def _rollup_sources(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    total = 0
    present = 0
    for source in sources.values():
        if source.get("present"):
            present += 1
        counts = source.get("counts") if isinstance(source.get("counts"), Mapping) else {}
        total += int(counts.get("total", 0) or 0)
        roles.update({str(k): int(v) for k, v in dict(counts.get("role_breakdown", {})).items()})
        visibility.update({str(k): int(v) for k, v in dict(counts.get("visibility_breakdown", {})).items()})
    return {
        "present_source_count": present,
        "total": total,
        "role_breakdown": _counter_dict(roles),
        "visibility_breakdown": _counter_dict(visibility),
    }


def _empty_source(name: str, path: Path, *, unit: str) -> dict[str, Any]:
    return {
        "present": False,
        "name": name,
        "path": str(path),
        "unit": unit,
        "counts": {"total": 0, "role_breakdown": {}, "visibility_breakdown": {}},
        "diversity": {"distinct_sessions": 0, "distinct_courts": 0, "distinct_channels": 0},
        "dedup": {"considered": 0, "removed": 0, "rate": None},
        "leakage_check": {"sample_hashes_checked": 0, "eval_hash_collisions": 0},
    }


def _print_human(report: Mapping[str, Any]) -> None:
    print("Corpus Dashboard")
    print(f"eval_hash_count: {report['eval_hash_check']['eval_hash_count']}")
    print("")
    print("SOURCE                 PRESENT  TOTAL   ROLES                              VISIBILITY")
    for name, source in report["sources"].items():
        counts = source["counts"]
        roles = _compact_dict(counts.get("role_breakdown", {}))
        visibility = _compact_dict(counts.get("visibility_breakdown", {}))
        print(f"{name:<22} {str(source['present']):<7} {int(counts.get('total', 0)):>6}  {roles:<34} {visibility}")
    print("")
    print("LEAKAGE")
    leakage = report["leakage"]
    print(f"eval_hash_collisions: {leakage['eval_hash_collisions']}")
    for hit in leakage["eval_hash_hits"]:
        print(
            "  eval_hash_hit: "
            f"{hit.get('source')} {hit.get('sample_id')} vs {hit.get('eval_clip', '<unknown>')} "
            f"hamming={hit.get('hamming_distance', '<unknown>')}"
        )
    print(f"heldout_id_hits: {leakage['heldout_id_hits']}")
    for hit in leakage["heldout_id_hit_details"]:
        print(
            "  heldout_id_hit: "
            f"{hit['heldout_id']} role={hit['role']} source={hit['source']} "
            f"record={hit['record_id']} path={hit['path']}"
        )


def _compact_dict(value: Mapping[str, Any]) -> str:
    if not value:
        return "{}"
    return ",".join(f"{key}={value[key]}" for key in sorted(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize registered labeled corpora and re-run leakage checks."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--roboflow-aggregated",
        type=Path,
        default=Path("data/roboflow_universe_20260706/aggregated"),
        help="Roboflow aggregated corpus directory.",
    )
    parser.add_argument(
        "--harvest-root",
        type=Path,
        default=Path("runs/lanes/p01b_harvest_ingest_20260706"),
        help="Harvest ingest lane artifact directory.",
    )
    parser.add_argument(
        "--owner-capture-manifest",
        type=Path,
        default=Path("runs/training_corpora_20260702/owner_capture/manifest.json"),
        help="Owner-capture corpus manifest; absence is reported, not an error.",
    )
    parser.add_argument(
        "--review-label-root",
        type=Path,
        default=Path("cvat_upload/exports/harvest_review_20260707"),
        help="Optional human-verified harvest review CVAT export root.",
    )
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"), help="Protected eval clip root.")
    parser.add_argument(
        "--eval-sample-every-s",
        type=float,
        default=DEFAULT_EVAL_SAMPLE_EVERY_S,
        help="Video dHash sampling interval used by the protected-eval helper.",
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable dashboard JSON to this path.")
    parser.add_argument(
        "--no-harvest-video-hash",
        action="store_true",
        help="Skip re-hashing harvest train/internal_val videos; string leakage checks still run.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    report = build_dashboard(
        root=root,
        roboflow_aggregated=_resolve(root, args.roboflow_aggregated),
        harvest_root=_resolve(root, args.harvest_root),
        owner_capture_manifest=_resolve(root, args.owner_capture_manifest),
        review_label_root=_resolve(root, args.review_label_root),
        eval_root=_resolve(root, args.eval_root),
        eval_sample_every_s=args.eval_sample_every_s,
        hash_harvest_videos=not args.no_harvest_video_hash,
    )
    if args.json is not None:
        _write_json(_resolve(root, args.json), report)
    _print_human(report)
    leakage = report["leakage"]
    return 1 if leakage["eval_hash_collisions"] or leakage["heldout_id_hits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
