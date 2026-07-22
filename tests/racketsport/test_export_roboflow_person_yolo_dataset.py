from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.racketsport.export_roboflow_person_yolo_dataset as export_cli


def _write_image(path: Path, seed: int, *, width: int = 96, height: int = 64) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    cv2.line(image, (0, seed % height), (width - 1, (seed * 7) % height), (255, 255, 255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _protected_bundle(path: Path) -> None:
    clips = []
    for clip_number, clip_id in enumerate(export_cli.PROTECTED_CLIP_IDS):
        video = path / clip_id / "source.avi"
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (96, 64),
        )
        assert writer.isOpened()
        for frame_number in range(3):
            rng = np.random.default_rng(900_000 + clip_number * 10 + frame_number)
            frame = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"{clip_number}:{frame_number}",
                (8, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
        writer.release()
        clips.append(
            {
                "clip": clip_id,
                "source_video": f"{clip_id}/source.avi",
                "source_sha256": export_cli.file_sha256(video),
                "width": 96,
                "height": 64,
                "frame_count": 3,
            }
        )
    _write_image(path / "additional_conservative_still.jpg", 999_999)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "pickleball_local_eval_clip_bundle",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )


def _dataset(
    tmp_path: Path,
    source_counts: dict[str, int],
    *,
    buckets: dict[str, str] | None = None,
    licenses: dict[str, str] | None = None,
    notes: dict[str, str] | None = None,
    temporal_prefixes: dict[str, str] | None = None,
) -> tuple[Path, Path, list[dict]]:
    root = tmp_path / "roboflow"
    index = root / "aggregated" / "subset_indexes" / "person_index.json"
    protected = tmp_path / "protected"
    _protected_bundle(protected)
    samples: list[dict] = []
    seed = 1
    for source, count in source_counts.items():
        for position in range(count):
            image = root / "pixels" / export_cli._safe_token(source) / f"{position:04d}.jpg"
            _write_image(image, seed)
            samples.append(
                {
                    "bucket": (buckets or {}).get(source, "core_pickleball"),
                    "height": 64,
                    "image_path": str(image),
                    "labels": [
                        {
                            "annotation_id": seed,
                            "bbox_xywh": [-2, 8, 30, 40],
                            "original_category": "player",
                        }
                    ],
                    "sample_id": f"{export_cli._safe_token(source)}:{position}",
                    "source_slug": source,
                    "temporal": {
                        "sequence_id": f"{source}:sequence",
                        "filename_prefix": (temporal_prefixes or {}).get(
                            source, export_cli._safe_token(source)
                        ),
                        "frame_number": position,
                    },
                    "width": 96,
                }
            )
            seed += 1
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_roboflow_subset_index",
                "label_kind": "person",
                "sample_count": len(samples),
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    entries = []
    for source in source_counts:
        entries.append(
            {
                "slug": source,
                "license_as_recorded": (licenses or {}).get(source, "CC BY 4.0"),
                "url": f"https://example.invalid/{source}",
                "note": (notes or {}).get(source, ""),
            }
        )
    (root / "manifest.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    (root / "aggregated" / "corpus_card.json").write_text(
        json.dumps({"fork_duplicate_mappings": []}), encoding="utf-8"
    )
    return index, protected, samples


def _write_review(
    path: Path,
    samples: list[dict],
    *,
    per_source: int,
    bad_sources: set[str] | None = None,
    leave_last_pending: bool = False,
) -> None:
    selection = export_cli._select_audit_samples(
        samples,
        per_source=per_source,
        seed=export_cli.DEFAULT_SEED,
    )
    rows = [sample for source_rows in selection.values() for sample in source_rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(export_cli.REVIEW_COLUMNS)
        for index, sample in enumerate(rows):
            bad = str(sample["source_slug"]) in (bad_sources or set())
            pending = leave_last_pending and index == len(rows) - 1
            writer.writerow(
                [
                    sample["source_slug"],
                    sample["sample_id"],
                    len(sample["labels"]),
                    "" if pending else 0 if bad else len(sample["labels"]),
                    "" if pending else 1,
                    "" if pending else 0 if bad else 1,
                    "" if pending else "yes",
                    "",
                ]
            )


def test_export_builds_guarded_whole_source_dataset_and_pending_human_pack(tmp_path: Path) -> None:
    counts = {"fork/a": 2, "fork/b": 2, "train/c": 2, "heldout/val": 2, "heldout/test": 2}
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        notes={"fork/a": "possible fork/mirror of fork/b"},
    )
    out = tmp_path / "out"

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(export_cli.REQUIRED_NC_EXCLUSION,),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=3,
        protected_root=protected,
        out_dir=out,
        min_train_images=1,
        min_train_source_groups=2,
    )

    assert result["objective_result"] == "PARTIAL"
    assert result["verdict"] == "PENDING_HUMAN_REVIEW"
    assert result["retention"]["pre_quarantine"]["verdict"] == "PROVISIONAL_PENDING_HUMAN_REVIEW"
    assert result["retention"]["pre_quarantine"]["meets_numeric_threshold_provisionally"] is True
    assert result["retention"]["post_quarantine"] is None
    assert result["annotation_quality"]["status"] == "PENDING_HUMAN_REVIEW"
    assert result["annotation_quality"]["quality_targets_pass"] is False
    assert result["training_ready_gate"]["status"] == "PENDING"
    assert result["training_ready_gate"]["p2_disposition"] == "NO_ATTEMPT_PENDING_PREREQ"
    assert result["protected_collision_check"]["collision_pair_count"] == 0
    assert result["cross_split_content_audit"]["status"] == "PASS"
    assert result["cross_split_content_audit"]["mandatory_production_check"] is True
    assert result["cross_split_content_audit"]["exhaustive_pair_count"] == 28
    assert result["cross_split_content_audit"]["verified_leak_count"] == 0
    assert result["protected_collision_check"]["exhaustive_pair_count"] == 130
    probe = result["protected_collision_check"]["protected_inventory"]["robustness_probe"]
    assert probe["protected_video_probe_frame_count"] == 12
    assert all(row == {"detected": 12, "total": 12, "detection_rate": 1.0} for row in probe["classes"].values())
    assert result["materialization"]["mode"] == "hardlink_with_copy_fallback"
    assert sum(row["images"] for row in result["split_counts"]) == 10
    assert next(row for row in result["split_counts"] if row["split"] == "train")["images"] == 6
    paired = next(row for row in result["fork_families"] if len(row["members"]) == 2)
    assert paired["members"] == ["fork/a", "fork/b"]
    assert paired["split"] == "train"
    assert paired["images"] == 4
    assert all(row["staged_unique_samples"] == 2 for row in result["audit_sample_counts"])
    assert all(row["shortfall"] == 1 for row in result["audit_sample_counts"])
    assert Path(result["audit_pack"]["page"]).is_file()
    assert "does not certify precision or recall" in Path(result["audit_pack"]["page"]).read_text()
    assert Path(result["audit_pack"]["review_template"]).is_file()
    train_images = list((out / "images" / "train").iterdir())
    assert len(train_images) == 6
    assert all(not path.is_symlink() and path.is_file() for path in train_images)
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 6
    assert not (out / "data.yaml").exists()
    assert (out / "train_family_balanced.txt").is_file()
    label = next((out / "labels" / "train").glob("*.txt")).read_text().strip()
    assert label == "0 0.145833 0.437500 0.291667 0.625000"


def test_named_holdout_assignment_propagates_to_every_fork_family_member(tmp_path: Path) -> None:
    counts = {
        "train/a": 1,
        "workspace/validation-fork": 1,
        "workspace/validation-original": 1,
        "heldout/test": 1,
    }
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        notes={"workspace/validation-fork": "LIKELY DUPLICATE of validation-original"},
    )

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="workspace/validation-fork",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=1,
        protected_root=protected,
        out_dir=tmp_path / "out",
        min_train_images=1,
        min_train_source_groups=1,
    )

    by_source = {row["source"]: row for row in result["source_counts"]}
    assert by_source["workspace/validation-fork"]["split"] == "val"
    assert by_source["workspace/validation-original"]["split"] == "val"
    family = next(
        row for row in result["fork_families"] if "workspace/validation-fork" in row["members"]
    )
    assert family["members"] == [
        "workspace/validation-fork",
        "workspace/validation-original",
    ]
    assert family["split"] == "val"
    assert family["images"] == 2


def test_exact_us_open_family_is_one_validation_family_and_lineage_audits_clean(
    tmp_path: Path,
) -> None:
    exact_family = {
        "pickle-es3fs/pickleball-video": 1,
        "nigh-workspace/pickleball-player-object-detection-cc2sw": 2,
        "pickleball-od8al/pickleball-seg": 1,
        "pickleball-od8al/pickleball-tsgju": 2,
        "pickleball-od8al/pickleball-version2": 3,
    }
    counts = {"train/a": 2, **exact_family, "heldout/test": 2}
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        temporal_prefixes={source: "output_frame" for source in exact_family},
    )

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="pickleball-od8al/pickleball-version2",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=1,
        protected_root=protected,
        out_dir=tmp_path / "out",
        min_train_images=1,
        min_train_source_groups=1,
    )

    by_source = {row["source"]: row for row in result["source_counts"]}
    assert {by_source[source]["split"] for source in exact_family} == {"val"}
    family = next(
        row
        for row in result["fork_families"]
        if "pickleball-od8al/pickleball-tsgju" in row["members"]
    )
    assert family["members"] == sorted(exact_family)
    assert family["split"] == "val"
    mapped = next(
        row
        for row in result["original_video_family_map"]["families"]
        if "pickleball-od8al/pickleball-tsgju" in row["members"]
    )
    assert mapped["family_id"] == "family:pickleball-od8al/pickleball-seg"
    assert mapped["original_video_family_id"] == "original_footage_component:od8al_validation_r2"
    assert "holdout-side-wins" in result["original_video_family_map"]["policy"]
    assert result["temporal_lineage_audit"]["status"] == "PASS"
    prefix_row = next(
        row
        for row in result["temporal_lineage_audit"]["temporal_prefix_overlap_rows"]
        if row["channel"] == "pickleball-od8al" and row["lineage"] == "output_frame"
    )
    assert prefix_row["sources"] == sorted(
        source for source in exact_family if source.startswith("pickleball-od8al/")
    )
    assert prefix_row["splits"] == ["val"]


def test_content_level_scan_refuses_cross_workspace_rename(tmp_path: Path) -> None:
    counts = {"train/workspace-a": 1, "validation/workspace-b": 1, "heldout/test": 1}
    index, protected, samples = _dataset(tmp_path, counts)
    train_sample = next(row for row in samples if row["source_slug"] == "train/workspace-a")
    val_sample = next(row for row in samples if row["source_slug"] == "validation/workspace-b")
    shutil.copy2(train_sample["image_path"], val_sample["image_path"])
    out = tmp_path / "out"

    with pytest.raises(export_cli.CrossSplitContentLeakError, match="content-level final-split scan"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="validation/workspace-b",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=out,
            min_train_images=1,
            min_train_source_groups=1,
        )

    report = json.loads((out / "cross_split_content_audit.json").read_text())
    assert report["status"] == "FAIL"
    assert report["phash_candidate_pair_count"] >= 1
    assert report["verified_leak_count"] >= 1
    leak = next(
        row
        for row in report["verified_leaks"]
        if row["left"]["source"] == "train/workspace-a"
        and row["right"]["source"] == "validation/workspace-b"
    )
    assert leak["phash_hamming_distances"] == [0, 0, 0]
    assert leak["verification"]["ssim"] == 1.0
    assert leak["verdict"] == "VERIFIED_CROSS_SPLIT_LEAK"
    assert json.loads((out / "refusal.json").read_text())["verdict"] == "CROSS_SPLIT_CONTENT_LEAK"
    assert not (out / "images").exists()


def test_temporal_prefix_overlap_fails_closed_if_sources_cross_splits(tmp_path: Path) -> None:
    _, _, samples = _dataset(
        tmp_path,
        {"same-channel/a": 1, "same-channel/b": 1, "heldout/test": 1},
        temporal_prefixes={"same-channel/a": "output_frame", "same-channel/b": "output_frame"},
    )
    with pytest.raises(ValueError, match="cross-split temporal/filename-lineage overlap"):
        export_cli._audit_cross_split_lineage(
            samples,
            split_for_source={
                "same-channel/a": "train",
                "same-channel/b": "val",
                "heldout/test": "test",
            },
            family_for_source={source: f"family:{source}" for source in {
                "same-channel/a", "same-channel/b", "heldout/test"
            }},
        )


def test_export_never_admits_adjacent_bucket_or_nc_source(tmp_path: Path) -> None:
    nc = export_cli.REQUIRED_NC_EXCLUSION
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1, "tennis/source": 2, nc: 2}
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        buckets={"tennis/source": "adjacent_sport_aux"},
        licenses={nc: "BY-NC-SA 4.0"},
    )

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=1,
        protected_root=protected,
        out_dir=tmp_path / "out",
        min_train_images=1,
        min_train_source_groups=1,
    )

    assert result["retained_images"] == 3
    assert result["exclusions"]["by_bucket"] == [
        {"bucket": "adjacent_sport_aux", "images": 2, "boxes": 2}
    ]
    assert result["exclusions"]["by_source"] == [{"source": nc, "images": 2, "boxes": 2}]
    manifest = json.loads((tmp_path / "out" / "dataset_manifest.json").read_text())
    retained_sources = {row["source"] for row in manifest["rows"]}
    assert nc not in retained_sources
    assert "tennis/source" not in retained_sources


def test_adjacent_bucket_request_is_refused_before_creating_outputs(tmp_path: Path) -> None:
    index, protected, _ = _dataset(
        tmp_path,
        {"train/a": 1, "heldout/val": 1, "heldout/test": 1},
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="hard-refuses bucket"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="adjacent_sport_aux",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=out,
        )
    assert not out.exists()


def test_export_reports_and_excludes_impossible_box_without_dropping_valid_image(tmp_path: Path) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, samples = _dataset(tmp_path, counts)
    payload = json.loads(index.read_text())
    payload["samples"][0]["labels"].append(
        {"annotation_id": 999, "bbox_xywh": [20, 64, 10, 0.25], "original_category": "player"}
    )
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=1,
        protected_root=protected,
        out_dir=tmp_path / "out",
        min_train_images=1,
        min_train_source_groups=1,
    )

    assert result["eligible_index_images"] == 3
    assert result["eligible_index_boxes"] == 4
    assert result["retained_images"] == 3
    assert result["retained_boxes"] == 3
    assert result["exclusions"]["invalid_annotation_count"] == 1
    assert result["exclusions"]["dropped_image_count"] == 0
    train = next(row for row in result["source_counts"] if row["source"] == "train/a")
    assert train["indexed_boxes"] == 2
    assert train["boxes"] == 1
    assert train["invalid_boxes_excluded"] == 1


def test_export_refuses_non_cc_by_selected_source(tmp_path: Path) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        licenses={"train/a": "CC BY-NC 4.0"},
    )

    with pytest.raises(ValueError, match="is not CC BY 4.0"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=tmp_path / "out",
            min_train_images=1,
            min_train_source_groups=1,
        )


def test_export_refuses_named_val_and_test_in_same_fork_family(tmp_path: Path) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, _ = _dataset(
        tmp_path,
        counts,
        notes={"heldout/val": "possible duplicate fork/mirror of heldout/test"},
    )

    with pytest.raises(ValueError, match="belong to one fork family"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=tmp_path / "out",
            min_train_images=1,
            min_train_source_groups=1,
        )


def test_exhaustive_multiscale_phash_collision_refuses_before_yolo_materialization(tmp_path: Path) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, samples = _dataset(tmp_path, counts)
    protected_image = protected / "additional_conservative_still.jpg"
    protected_image.write_bytes(Path(samples[0]["image_path"]).read_bytes())
    out = tmp_path / "out"

    with pytest.raises(export_cli.ProtectedCollisionError, match="protected-frame collision"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=out,
            min_train_images=1,
            min_train_source_groups=1,
        )

    report = json.loads((out / "protected_collision_report.json").read_text())
    assert report["exhaustive_pair_count"] == 39
    assert report["descriptor_comparison_count"] == 39 * len(export_cli.PROTECTED_TRANSFORMS)
    assert report["collision_pair_count"] == 1
    assert report["collision_image_count"] == 1
    assert report["collisions"][0]["hamming_distances"] == [0, 0, 0]
    assert not (out / "images").exists()


@pytest.mark.parametrize("transform", export_cli.ROBUSTNESS_PROBE_TRANSFORMS)
def test_transform_aware_collision_refuses_protected_derivatives(
    tmp_path: Path, transform: str
) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, samples = _dataset(tmp_path, counts)
    video = protected / export_cli.PROTECTED_CLIP_IDS[0] / "source.avi"
    capture = cv2.VideoCapture(str(video))
    ok, protected_frame = capture.read()
    capture.release()
    assert ok
    derivative = export_cli._apply_probe_transform(protected_frame, transform)
    derivative_path = tmp_path / f"candidate_{transform}.png"
    assert cv2.imwrite(str(derivative_path), derivative)
    payload = json.loads(index.read_text())
    payload["samples"][0]["image_path"] = str(derivative_path)
    payload["samples"][0]["width"] = derivative.shape[1]
    payload["samples"][0]["height"] = derivative.shape[0]
    index.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"

    with pytest.raises(export_cli.ProtectedCollisionError):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=out,
            min_train_images=1,
            min_train_source_groups=1,
        )

    report = json.loads((out / "protected_collision_report.json").read_text())
    collision = next(row for row in report["collisions"] if row["sample_id"] == samples[0]["sample_id"])
    assert collision["matched_protected_transform"] == transform


def test_protected_manifest_hash_and_inventory_are_binding(tmp_path: Path) -> None:
    index, protected, _ = _dataset(
        tmp_path,
        {"train/a": 1, "heldout/val": 1, "heldout/test": 1},
    )
    manifest_path = protected / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"][0]["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_cli.export_roboflow_person_yolo_dataset(
            index_path=index,
            bucket="core_pickleball",
            exclude_sources=(),
            val_source="heldout/val",
            test_source="heldout/test",
            group_forks=True,
            source_balanced=True,
            audit_samples_per_source=1,
            protected_root=protected,
            out_dir=tmp_path / "out",
        )


def test_completed_review_quarantines_below_90_source_and_recomputes_training_gate(
    tmp_path: Path,
) -> None:
    counts = {"train/good": 3, "train/bad": 2, "heldout/val": 2, "heldout/test": 2}
    index, protected, samples = _dataset(tmp_path, counts)
    review = tmp_path / "completed_review.csv"
    _write_review(review, samples, per_source=3, bad_sources={"train/bad"})
    out = tmp_path / "out"

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=3,
        protected_root=protected,
        review_csv=review,
        out_dir=out,
        min_train_images=3,
        min_train_source_groups=1,
    )

    assert result["annotation_quality"]["status"] == "COMPLETE"
    assert result["annotation_quality"]["quarantined_sources"] == ["train/bad"]
    assert result["annotation_quality"]["quality_targets_pass"] is True
    assert result["retention"]["pre_quarantine"]["status"] == "PROVISIONAL_PENDING_HUMAN_REVIEW"
    assert result["retention"]["pre_quarantine"]["verdict"] != "PASS"
    assert result["retention"]["post_quarantine"]["status"] == "MEASURED_POST_QUARANTINE"
    assert result["retention"]["post_quarantine"]["train_images"] == 3
    assert result["retention"]["post_quarantine"]["verdict"] == "PASS"
    assert result["training_ready_gate"]["status"] == "PASS"
    assert result["training_ready_gate"]["p2_disposition"] == "READY"
    assert result["objective_result"] == "PASS"
    assert (out / "data.yaml").is_file()
    assert result["audit_pack"]["status"] == "COMPLETE"
    assert result["audit_pack"]["review_template"] is None
    completed_review = Path(result["audit_pack"]["completed_review_csv"])
    assert completed_review.read_bytes() == review.read_bytes()
    bundled_audit = json.loads(Path(result["audit_pack"]["manifest"]).read_text())
    assert bundled_audit["status"] == "COMPLETE"
    assert bundled_audit["pending_row_count"] == 0
    assert bundled_audit["completed_review_csv_sha256"] == export_cli.file_sha256(completed_review)
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert "train/bad" not in {row["source"] for row in manifest["rows"]}


def test_partial_review_stays_pending_and_withholds_p2_yaml(tmp_path: Path) -> None:
    counts = {"train/a": 2, "heldout/val": 2, "heldout/test": 2}
    index, protected, samples = _dataset(tmp_path, counts)
    review = tmp_path / "partial_review.csv"
    _write_review(review, samples, per_source=2, leave_last_pending=True)
    out = tmp_path / "out"

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=2,
        protected_root=protected,
        review_csv=review,
        out_dir=out,
        min_train_images=1,
        min_train_source_groups=1,
    )

    assert result["annotation_quality"]["status"] == "PENDING_HUMAN_REVIEW"
    assert result["annotation_quality"]["pending_row_count"] == 1
    assert result["training_ready_gate"]["status"] == "PENDING"
    assert result["objective_result"] == "PARTIAL"
    assert result["retention"]["post_quarantine"] is None
    assert not (out / "data.yaml").exists()


def test_audit_sampling_deduplicates_roboflow_derivatives_by_original_frame(tmp_path: Path) -> None:
    counts = {"train/a": 18, "heldout/val": 1, "heldout/test": 1}
    index, protected, samples = _dataset(tmp_path, counts)
    payload = json.loads(index.read_text())
    train_rows = [row for row in payload["samples"] if row["source_slug"] == "train/a"]
    for position, row in enumerate(train_rows[:10]):
        old_path = Path(row["image_path"])
        derivative = old_path.parent / f"original_{position // 2}_jpg.rf.{position:032x}.jpg"
        shutil.copy2(old_path, derivative)
        row["image_path"] = str(derivative)
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=15,
        protected_root=protected,
        out_dir=tmp_path / "out",
        min_train_images=1,
        min_train_source_groups=1,
    )

    audit = next(row for row in result["audit_sample_counts"] if row["source"] == "train/a")
    assert audit["available_images"] == 18
    assert audit["available_original_frame_identities"] == 13
    assert audit["staged_unique_samples"] == 13
    assert audit["shortfall"] == 2


def test_family_balanced_list_is_deterministic_consumed_by_p2_and_relocatable(
    tmp_path: Path,
) -> None:
    counts = {
        "train/small": 2,
        "train/medium": 5,
        "train/large": 20,
        "heldout/val": 2,
        "heldout/test": 2,
    }
    index, protected, samples = _dataset(tmp_path, counts)
    review = tmp_path / "completed_review.csv"
    _write_review(review, samples, per_source=20)
    out = tmp_path / "out"

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=20,
        protected_root=protected,
        review_csv=review,
        out_dir=out,
        min_train_images=1,
        min_train_source_groups=3,
    )

    balance = result["source_balance"]
    assert balance["target_entries_per_family"] == 5
    assert [row["sampled_entries"] for row in balance["families"]] == [5, 5, 5]
    small = next(row for row in balance["families"] if row["family_id"] == "family:train/small")
    large = next(row for row in balance["families"] if row["family_id"] == "family:train/large")
    assert small["maximum_image_repetitions"] == 3
    assert large["omitted_unique_images"] == 15
    yaml_text = (out / "data.yaml").read_text()
    assert "train: train_family_balanced.txt" in yaml_text
    assert "path:" not in yaml_text
    first_list = (out / "train_family_balanced.txt").read_bytes()

    relocated = tmp_path / "relocated" / "dataset"
    relocated.parent.mkdir(parents=True)
    shutil.move(str(out), relocated)
    from ultralytics.data.utils import check_det_dataset

    loaded = check_det_dataset(str(relocated / "data.yaml"), autodownload=False)
    assert Path(loaded["train"]) == (relocated / "train_family_balanced.txt").resolve()
    train_paths = [
        relocated / line.removeprefix("./")
        for line in (relocated / "train_family_balanced.txt").read_text().splitlines()
    ]
    assert len(train_paths) == 15
    assert all(path.is_file() and not path.is_symlink() for path in train_paths)
    assert first_list == (relocated / "train_family_balanced.txt").read_bytes()


def test_retention_negative_is_reported_without_massaging_pool(tmp_path: Path) -> None:
    counts = {"train/a": 1, "heldout/val": 1, "heldout/test": 1}
    index, protected, _ = _dataset(tmp_path, counts)

    result = export_cli.export_roboflow_person_yolo_dataset(
        index_path=index,
        bucket="core_pickleball",
        exclude_sources=(),
        val_source="heldout/val",
        test_source="heldout/test",
        group_forks=True,
        source_balanced=True,
        audit_samples_per_source=1,
        protected_root=protected,
        out_dir=tmp_path / "out",
    )

    assert result["objective_result"] == "PARTIAL"
    assert result["verdict"] == "PERSON_RF_POOL_TOO_THIN"
    assert result["retention"]["pre_quarantine"]["verdict"] == "PERSON_RF_POOL_TOO_THIN"
    assert result["retention"]["pre_quarantine"]["meets_numeric_threshold_provisionally"] is False
    assert result["retention"]["pre_quarantine"]["permanently_closes_training_for_export"] is True
    assert result["retention"]["pre_quarantine"]["p2_disposition"] == "NO_ATTEMPT_PREREQ"
    assert result["training_ready_gate"]["status"] == "FAIL"
    assert result["training_ready_gate"]["permanently_closed_for_export"] is True
    assert result["training_ready_gate"]["p2_disposition"] == "NO_ATTEMPT_PREREQ"
    assert "PERSON_RF_POOL_TOO_THIN" in result["training_ready_gate"]["blockers"]
    assert not (tmp_path / "out" / "data.yaml").exists()
    assert result["retained_images"] == 3


def test_cli_forwards_exact_lane_contract(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return {
            "objective_result": "PASS",
            "verdict": "TRAINING_READY",
            "retained_source_count": 14,
            "retained_images": 15312,
            "retained_boxes": 47044,
            "split_counts": [],
            "protected_collision_check": {"collision_pair_count": 0},
            "retention": {"post_quarantine": {"verdict": "PASS"}},
            "training_ready_gate": {"status": "PASS"},
            "audit_pack": {"path": "audit"},
            "data_yaml": str(tmp_path / "out" / "data.yaml"),
            "dataset_manifest": str(tmp_path / "out" / "dataset_manifest.json"),
        }

    monkeypatch.setattr(export_cli, "export_roboflow_person_yolo_dataset", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/export_roboflow_person_yolo_dataset.py",
            "--index",
            "data/roboflow_universe_20260706/aggregated/subset_indexes/person_index.json",
            "--bucket",
            "core_pickleball",
            "--exclude-source",
            export_cli.REQUIRED_NC_EXCLUSION,
            "--val-source",
            "pickleball-od8al/pickleball-version2",
            "--test-source",
            "hemel/pickleball-cedmo",
            "--group-forks",
            "--source-balanced",
            "--audit-samples-per-source",
            "15",
            "--protected-root",
            "eval_clips/ball",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert export_cli.main() == 0
    assert captured["group_forks"] is True
    assert captured["source_balanced"] is True
    assert captured["audit_samples_per_source"] == 15
    assert captured["review_csv"] is None
    assert captured["expected_index_sha256"] == export_cli.EXPECTED_PERSON_INDEX_SHA256
    assert captured["exclude_sources"] == (export_cli.REQUIRED_NC_EXCLUSION,)
    assert json.loads(capsys.readouterr().out)["collision_count"] == 0


def test_cli_error_has_no_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(export_cli, "export_roboflow_person_yolo_dataset", lambda **_: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/export_roboflow_person_yolo_dataset.py",
            "--index",
            "index.json",
            "--bucket",
            "core_pickleball",
            "--val-source",
            "val",
            "--test-source",
            "test",
            "--group-forks",
            "--source-balanced",
            "--audit-samples-per-source",
            "15",
            "--protected-root",
            "protected",
            "--out",
            "out",
        ],
    )

    assert export_cli.main() == 1
    captured = capsys.readouterr()
    assert "Roboflow PERSON export failed: bad" in captured.err
    assert "Traceback" not in captured.err


def test_cli_path_is_directly_referenced_for_scaffold_coverage() -> None:
    assert Path("scripts/racketsport/export_roboflow_person_yolo_dataset.py").is_file()
