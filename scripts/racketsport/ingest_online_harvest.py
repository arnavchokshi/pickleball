#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.online_harvest_ingest import (  # noqa: E402
    build_corpus_card,
    build_cvat_status,
    build_dedup_report,
    build_dispatch_doc,
    build_prelabel_shard_manifest,
    clips_to_manifest,
    four_level_visibility_schema_available,
    load_harvest_sources,
    process_source_to_clips,
    select_review_subset,
    validate_cvat_review_task_package,
    write_clip_provenance,
    write_corpus_card_md,
    write_cvat_review_task_package,
    write_prelabel_cpu_smoke,
    _write_json,
    assign_clip_roles,
)


DEFAULT_HARVEST_ROOT = Path("data/online_harvest_20260706")
DEFAULT_LANE_ROOT = Path("runs/lanes/p01b_harvest_ingest_20260706")
DEFAULT_HELDOUT_SOURCE_IDS = ("pwxNwFfYQlQ", "vQhtz8l6VqU")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest online harvest videos into rally clips and prelabel-ready manifests.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_HARVEST_ROOT / "manifest.json")
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST_ROOT)
    parser.add_argument("--data-out-root", type=Path, default=DEFAULT_HARVEST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_LANE_ROOT)
    parser.add_argument(
        "--heldout-source-id",
        action="append",
        default=None,
        help="Source id to propose as held-out candidate. Provide exactly two; defaults to lane-selected diverse games.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Write manifests/provenance without ffmpeg clip extraction.")
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--dedup-sample-every-s", type=float, default=2.0)
    parser.add_argument("--dedup-threshold", type=int, default=3)
    parser.add_argument("--max-prelabel-smoke-frames", type=int, default=50)
    parser.add_argument("--pad-s", type=float, default=1.5)
    parser.add_argument("--max-active-gap-s", type=float, default=2.0)
    parser.add_argument("--merge-gap-s", type=float, default=2.5)
    parser.add_argument("--min-segment-s", type=float, default=4.0)
    parser.add_argument("--min-motion-score", type=float, default=0.20)
    parser.add_argument("--audio-score-threshold", type=float, default=0.55)
    parser.add_argument("--audio-motion-floor", type=float, default=0.08)
    parser.add_argument("--min-motion-bins", type=int, default=2)
    parser.add_argument(
        "--export-cvat-review-tasks",
        action="store_true",
        help="Write CVAT task definitions for the selected non-heldout review subset.",
    )
    parser.add_argument(
        "--cvat-out-root",
        type=Path,
        default=None,
        help="Output root for --export-cvat-review-tasks; defaults to <out-root>/cvat_upload.",
    )
    args = parser.parse_args()

    heldout_source_ids = tuple(args.heldout_source_id or DEFAULT_HELDOUT_SOURCE_IDS)
    try:
        sources = load_harvest_sources(args.manifest, harvest_root=args.harvest_root)
        if len(sources) != 8:
            raise ValueError(f"expected 8 downloaded harvest videos; found {len(sources)}")

        segment_kwargs = {
            "pad_s": args.pad_s,
            "max_active_gap_s": args.max_active_gap_s,
            "merge_gap_s": args.merge_gap_s,
            "min_segment_s": args.min_segment_s,
            "min_motion_score": args.min_motion_score,
            "audio_score_threshold": args.audio_score_threshold,
            "audio_motion_floor": args.audio_motion_floor,
            "min_motion_bins": args.min_motion_bins,
        }
        all_clips = []
        source_infos = {}
        for source in sources:
            clips, info = process_source_to_clips(
                source,
                data_out_root=args.data_out_root,
                lane_out_root=args.out_root,
                skip_extract=args.skip_extract,
                segment_kwargs=segment_kwargs,
            )
            all_clips.extend(clips)
            source_infos[source.source_id] = info

        roles = assign_clip_roles(
            all_clips,
            proposed_heldout_source_ids=heldout_source_ids,
        )

        for source in sources:
            source_clips = [clip for clip in all_clips if clip.source.source_id == source.source_id]
            source_segments = source_infos[source.source_id]["segments"]
            for clip, segment in zip(source_clips, source_segments, strict=True):
                write_clip_provenance(
                    clip.provenance_path,
                    clip=clip,
                    segment=segment,
                    role=roles.clip_roles[clip.clip_id],
                    source_sha256=source_infos[source.source_id]["source_sha256"],
                )

        dedup_report = build_dedup_report(
            harvest_sources=sources,
            hash_sample_every_s=args.dedup_sample_every_s,
            threshold=args.dedup_threshold,
        )
        review_subset = select_review_subset(all_clips, roles.clip_roles)
        schema_available = four_level_visibility_schema_available()
        cvat_task_export = None
        cvat_task_validation = None
        cvat_out_root = args.cvat_out_root or (args.out_root / "cvat_upload")
        if args.export_cvat_review_tasks:
            if not schema_available:
                raise RuntimeError("cannot export CVAT review tasks before the 4-level visibility schema is available")
            cvat_task_export = write_cvat_review_task_package(
                review_subset,
                out_root=cvat_out_root,
                heldout_source_ids=heldout_source_ids,
            )
            cvat_task_validation = validate_cvat_review_task_package(
                cvat_task_export["manifest_path"],
                heldout_source_ids=heldout_source_ids,
            )
            validation_path = cvat_out_root / "cvat_task_validation.json"
            cvat_task_validation = {**cvat_task_validation, "validation_path": str(validation_path)}
            _write_json(validation_path, cvat_task_validation)
            if cvat_task_validation["status"] != "passed":
                raise RuntimeError(f"CVAT task validation failed: {cvat_task_validation['errors']}")
        cvat_status = build_cvat_status(
            review_subset,
            schema_available=schema_available,
            task_export=cvat_task_export,
            task_validation=cvat_task_validation,
        )
        shard_manifest = build_prelabel_shard_manifest(all_clips, roles, shard_size=args.shard_size)
        eligible_for_smoke = [clip for clip in all_clips if roles.clip_roles[clip.clip_id] != "heldout_candidate_proposed"]
        if not eligible_for_smoke:
            raise ValueError("no non-heldout rally clips available for prelabel smoke")
        smoke_report = write_prelabel_cpu_smoke(
            clip=eligible_for_smoke[0],
            out_dir=args.out_root / "prelabel_cpu_smoke",
            max_frames=args.max_prelabel_smoke_frames,
        )
        if smoke_report["status"] != "passed":
            raise RuntimeError(f"prelabel CPU smoke failed: {smoke_report['stderr']}")

        clip_manifest = clips_to_manifest(all_clips, roles.clip_roles)
        corpus_card = build_corpus_card(
            sources=sources,
            clips=all_clips,
            roles=roles,
            dedup_report=dedup_report,
            heldout_proposals=roles.heldout_proposals,
            cvat_status=cvat_status,
        )

        args.out_root.mkdir(parents=True, exist_ok=True)
        _write_json(args.out_root / "rally_clip_manifest.json", clip_manifest)
        _write_json(args.out_root / "dedup_report.json", dedup_report)
        _write_json(args.out_root / "prelabel_shard_manifest.json", shard_manifest)
        _write_json(args.out_root / "cvat_review_selection.json", review_subset)
        _write_json(args.out_root / "cvat_export_status.json", cvat_status)
        _write_json(args.out_root / "corpus_card.json", corpus_card)
        write_corpus_card_md(args.out_root / "corpus_card.md", corpus_card)
        build_dispatch_doc(
            args.out_root / "prelabel_dispatch.md",
            args.out_root / "prelabel_shard_manifest.json",
            args.out_root / "prelabel_cpu_smoke" / "prelabel_cpu_smoke.json",
        )

        result = {
            "status": "ok",
            "sources": len(sources),
            "clips": len(all_clips),
            "rally_clip_manifest": str(args.out_root / "rally_clip_manifest.json"),
            "corpus_card_json": str(args.out_root / "corpus_card.json"),
            "corpus_card_md": str(args.out_root / "corpus_card.md"),
            "dedup_report": str(args.out_root / "dedup_report.json"),
            "prelabel_shard_manifest": str(args.out_root / "prelabel_shard_manifest.json"),
            "prelabel_cpu_smoke": str(args.out_root / "prelabel_cpu_smoke" / "prelabel_cpu_smoke.json"),
            "cvat_export_status": cvat_status,
            "heldout_proposals": roles.heldout_proposals,
        }
    except Exception as exc:
        print(f"online harvest ingest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
