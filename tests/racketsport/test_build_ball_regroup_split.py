from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest


CLI_PATH = Path("scripts/racketsport/build_ball_regroup_split.py")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    cv2.line(image, (seed % 31, 0), (63, (seed * 7) % 47), (255, 255, 255), 2)
    return image


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _reviewed_payload(
    clip_id: str,
    labels: dict[int, tuple[float, float, float, float] | None],
) -> dict[str, object]:
    frame_count = max(labels) + 1
    frames: list[dict[str, object]] = []
    for frame_index in range(frame_count):
        bbox = labels.get(frame_index)
        boxes: list[dict[str, object]] = []
        visibility: dict[str, str] = {}
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            boxes.append(
                {
                    "track_id": 1,
                    "label": "ball",
                    "frame_index": frame_index,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "keyframe": True,
                    "occluded": False,
                    "source": "manual",
                    "visibility_level": "clear",
                }
            )
            visibility = {"ball": "clear"}
        elif frame_index in labels:
            visibility = {"ball": "none"}
        frames.append(
            {
                "frame_index": frame_index,
                "boxes": boxes,
                "visibility_levels_by_label": visibility,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": clip_id,
        "source_format": "cvat_images_1_1",
        "source_path": "fixture.zip",
        "reviewed_frame_indices": sorted(labels),
        "reviewed_frame_indices_source": "explicit",
        "task": {
            "name": clip_id,
            "size": frame_count,
            "mode": "annotation",
            "start_frame": 0,
            "stop_frame": frame_count - 1,
            "original_size": [64, 48],
            "source": f"{clip_id}.mp4",
        },
        "frames": frames,
        "tracks": [],
        "summary": {"frame_count": frame_count},
    }


def _selection_row(
    source_id: str,
    frame_index: int,
    *,
    x: float,
    y: float,
) -> dict[str, object]:
    clip_id = f"{source_id}_rally_0001"
    return {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "source_id": source_id,
        "disagreement_type": "teacher-only",
        "teacher": {"visible": True, "score": 0.9, "xy": [x, y]},
        "student": {"visible": False, "score": 0.1, "xy": [0, 0]},
    }


def _write_scratch_export(
    path: Path,
    rows: list[dict[str, object]],
    *,
    boxes_by_name: dict[str, tuple[float, float, float, float] | None],
    box_source: str = "manual",
    task_id: int | None = 87,
) -> None:
    image_xml: list[str] = []
    for image_id, row in enumerate(rows):
        image_name = str(row["image_name"])
        bbox = boxes_by_name[image_name]
        box_xml = ""
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            box_xml = (
                f'<box label="ball" source="{box_source}" occluded="0" xtl="{x1}" ytl="{y1}" '
                f'xbr="{x2}" ybr="{y2}" z_order="0">'
                '<attribute name="visibility_level">clear</attribute></box>'
            )
        image_xml.append(
            f'<image id="{image_id}" name="{image_name}" width="64" height="48">'
            f"{box_xml}</image>"
        )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<annotations><version>1.1</version><meta><job>"
        + ("" if task_id is None else f"<id>{task_id}</id>")
        + f"<size>{len(rows)}</size><mode>annotation</mode>"
        "</job></meta>"
        + "".join(image_xml)
        + "</annotations>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations.xml", xml)


def _build_fixture(tmp_path: Path) -> dict[str, object]:
    from scripts.racketsport.build_ball_regroup_split import _lineage_inputs_digest

    reviewed_root = tmp_path / "reviewed"
    train_clip = "train_src_rally_0001"
    holdout_clip = "hold_src_rally_0001"
    _write_json(
        reviewed_root / train_clip / "reviewed_boxes.json",
        _reviewed_payload(
            train_clip,
            {
                0: (12.0, 12.0, 28.0, 28.0),  # unchanged 16px prelabel
                1: (30.0, 20.0, 44.0, 34.0),  # corrected prelabel
            },
        ),
    )
    _write_json(
        reviewed_root / holdout_clip / "reviewed_boxes.json",
        _reviewed_payload(holdout_clip, {0: (12.0, 12.0, 28.0, 28.0)}),
    )

    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "sessions": [
                {
                    "session_id": "fixture_session",
                    "frames": [
                        _selection_row("train_src", 0, x=20.0, y=20.0),
                        _selection_row("train_src", 1, x=20.0, y=20.0),
                        _selection_row("hold_src", 0, x=20.0, y=20.0),
                    ],
                }
            ]
        },
    )

    scratch_rows: list[dict[str, object]] = []
    scratch_images: dict[str, bytes] = {}
    specs = [
        ("hold_src", "hold_src_rally_0001", 7, 101),
        ("scratch_train", "scratch_train_rally_0001", 8, 202),
        ("scratch_train", "scratch_train_rally_0001", 9, 303),
    ]
    for ordinal, (source_id, clip_id, frame_index, seed) in enumerate(specs):
        image_name = f"{source_id}__{clip_id}__abs_{frame_index:06d}.png"
        image_bytes = _encode_png(_image(seed))
        scratch_images[image_name] = image_bytes
        scratch_rows.append(
            {
                "sample_ordinal": ordinal,
                "source_id": source_id,
                "source_class": f"class_{source_id}",
                "rally_id": clip_id,
                "clip_id": clip_id,
                "frame_index": frame_index,
                "row_key": f"{clip_id}:{frame_index:06d}",
                "image_name": image_name,
                "image_zip_member": image_name,
                "image_md5": hashlib.md5(image_bytes).hexdigest(),
                "provenance_class": "scratch",
            }
        )

    images_zip = tmp_path / "scratch_images.zip"
    with zipfile.ZipFile(images_zip, "w") as archive:
        for name, payload in scratch_images.items():
            archive.writestr(name, payload)
    image_zip_sha256 = hashlib.sha256(images_zip.read_bytes()).hexdigest()
    sampling_manifest = tmp_path / "sampling_manifest.json"
    _write_json(
        sampling_manifest,
        {
            "artifact_type": "fixture_scratch_sampling",
            "labeling_mode": "scratch",
            "provenance_class": "scratch",
            "frames": scratch_rows,
            "per_source_distribution": {"hold_src": 1, "scratch_train": 2},
        },
    )
    package_manifest = tmp_path / "package_manifest.json"
    _write_json(
        package_manifest,
        {
            "artifact_type": "w7_audit_stratum_package_manifest",
            "task_name": "fixture_scratch_task",
            "labeling_mode": "scratch",
            "prelabels_present": False,
            "sampling_manifest": str(sampling_manifest),
            "sampling_manifest_md5": hashlib.md5(sampling_manifest.read_bytes()).hexdigest(),
            "ball_sessions": [
                {
                    "session_id": "audit_stratum_uniform3",
                    "task_name": "fixture_scratch_task",
                    "frame_count": 3,
                    "image_zip": str(images_zip),
                    "prelabels_present": False,
                    "provenance_class": "scratch",
                    "source_counts": {"hold_src": 1, "scratch_train": 2},
                }
            ],
        },
    )
    export_zip = tmp_path / "scratch_annotations.zip"
    _write_scratch_export(
        export_zip,
        scratch_rows,
        boxes_by_name={
            str(scratch_rows[0]["image_name"]): (10, 10, 18, 18),
            str(scratch_rows[1]["image_name"]): None,
            str(scratch_rows[2]["image_name"]): (20, 20, 28, 28),
        },
    )
    task_fingerprint = "a" * 64
    import_ledger = tmp_path / "import_report.json"
    _write_json(
        import_ledger,
        {
            "schema_version": 2,
            "status": "imported",
            "tasks": [
                {
                    "frame_count": 3,
                    "image_zip": str(images_zip.resolve()),
                    "kind": "ball",
                    "prelabel_zip": None,
                    "status": "imported",
                    "task_fingerprint": task_fingerprint,
                    "task_id": 87,
                    "task_name": "fixture_scratch_task",
                }
            ],
        },
    )
    export_sha256 = hashlib.sha256(export_zip.read_bytes()).hexdigest()
    negative_attestation = tmp_path / "owner_attestation.json"
    _write_json(
        negative_attestation,
        {
            "task_id": 87,
            "export_sha256": export_sha256,
            "statement": "I inspected every frame; boxless images contain no visible ball.",
            "attested_by": "fixture-owner",
            "attested_utc": "2026-07-21T20:30:00Z",
            "all_frames_inspected": True,
            "boxless_means_no_ball": True,
        },
    )
    lineage_sha256, _ = _lineage_inputs_digest(
        selection_manifests=[selection],
        legacy_reviewed_root=None,
        legacy_prelabel_root=None,
    )

    protected_video = tmp_path / "protected.mp4"
    _write_video(protected_video, [_image(901), _image(902)])
    protected_addition = tmp_path / "protected_addition.png"
    assert cv2.imwrite(str(protected_addition), _image(903))
    return {
        "reviewed_root": reviewed_root,
        "selection": selection,
        "package_manifest": package_manifest,
        "images_zip": images_zip,
        "export_zip": export_zip,
        "export_sha256": export_sha256,
        "negative_attestation": negative_attestation,
        "import_ledger": import_ledger,
        "import_ledger_sha256": hashlib.sha256(import_ledger.read_bytes()).hexdigest(),
        "image_zip_sha256": image_zip_sha256,
        "task_fingerprint": task_fingerprint,
        "lineage_sha256": lineage_sha256,
        "scratch_rows": scratch_rows,
        "scratch_images": scratch_images,
        "protected_video": protected_video,
        "protected_addition": protected_addition,
    }


def _build(
    paths: dict[str, object], out: Path, *, production_mode: bool = False
) -> dict[str, object]:
    from scripts.racketsport.build_ball_regroup_split import build_ball_regroup_split

    return build_ball_regroup_split(
        reviewed_root=Path(paths["reviewed_root"]),
        scratch_package=Path(paths["package_manifest"]),
        scratch_export=Path(paths["export_zip"]),
        negative_attestation=(
            None
            if paths.get("negative_attestation") is None
            else Path(paths["negative_attestation"])
        ),
        scratch_import_ledger=Path(paths["import_ledger"]),
        holdout_sources=["hold_src"],
        out=out,
        selection_manifests=[Path(paths["selection"])],
        legacy_reviewed_root=None,
        legacy_prelabel_root=None,
        protected_videos=[Path(paths["protected_video"])],
        protected_additions=[Path(paths["protected_addition"])],
        collision_hamming_threshold=0,
        expected_reviewed_count=3,
        expected_scratch_count=3,
        expected_old_train_count=2,
        expected_scratch_train_count=2,
        expected_validation_count=1,
        expected_holdout_counts={"hold_src": 1},
        confirmed_prelabel_weight=0.25,
        expected_reviewed_report_sha256=None,
        expected_scratch_package_sha256=None,
        expected_scratch_export_sha256=str(paths["export_sha256"]),
        expected_scratch_import_ledger_sha256=str(paths["import_ledger_sha256"]),
        expected_image_zip_sha256=paths.get("image_zip_sha256"),
        expected_task_fingerprint=str(paths["task_fingerprint"]),
        expected_lineage_inputs_sha256=str(paths["lineage_sha256"]),
        production_mode=production_mode,
    )


def test_splitter_reconciles_lineage_and_holds_out_whole_parent_source(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    out = tmp_path / "out"

    report = _build(paths, out)

    assert report["verdict"] == "BALL_FIXTURE_CLEAN"
    assert report["production_eligible"] is False
    assert report["checks"]["scratch_package_reconciled"]["after"] == "3/3"
    assert report["checks"]["historical_scratch_row_intersection"]["count"] == 0
    assert report["checks"]["train_validation_source_intersection"]["count"] == 0
    assert report["checks"]["protected_collision_count"]["count"] == 0
    assert report["split_counts"] == {
        "old_train": 2,
        "scratch_train": 2,
        "train": 4,
        "validation": 1,
    }
    assert report["lineage_counts"]["totals"] == {
        "confirmed_prelabel": 2,
        "corrected_prelabel": 1,
        "scratch": 3,
    }
    assert report["input_contract"]["image_zip_sha256"] == paths["image_zip_sha256"]
    assert len(report["input_contract"]["image_zip_entry_sha256"]) == 3
    assert report["input_contract"]["job_id"] == 87
    assert report["input_contract"]["job_id_binding"] == "EXPORT_XML_ONLY"
    assert report["residual_assumptions"] == [
        "local CVAT rendered the imported staged bytes for task 87; "
        "no independent job-id binding exists in the historical import ledger"
    ]
    assert report["checks"]["scratch_materialized_image_bytes"] == {
        "verdict": "PASS",
        "digest": "sha256",
        "image_count": 3,
        "image_zip_sha256": paths["image_zip_sha256"],
    }
    assert set(report["evaluation_metrics_by_source"]) == {"hold_src"}

    train_rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in (out / "validation.jsonl").read_text().splitlines()
    ]
    assert {row["source_id"] for row in train_rows} == {"train_src", "scratch_train"}
    assert {row["source_id"] for row in validation_rows} == {"hold_src"}
    assert {row["lineage_class"] for row in validation_rows} == {"scratch"}
    assert all(row["lineage_class"] != "confirmed_prelabel" for row in validation_rows)
    confirmed = [row for row in train_rows if row["lineage_class"] == "confirmed_prelabel"]
    assert len(confirmed) == 1
    assert confirmed[0]["training_weight"] == 0.25
    assert confirmed[0]["ground_truth"] is False
    negatives = [row for row in train_rows if not row["final_label"]["ball_present"]]
    assert len(negatives) == 1
    assert negatives[0]["negative_attestation_status"] == "OWNER_ATTESTED_NEGATIVE"
    assert negatives[0]["ground_truth"] is True
    scratch_materialized = [
        row for row in [*train_rows, *validation_rows] if row["lineage_class"] == "scratch"
    ]
    assert all(
        row["image_sha256"]
        == report["input_contract"]["image_zip_entry_sha256"][row["image_zip_member"]]
        for row in scratch_materialized
    )

    before = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    rerun = _build(paths, out)
    assert rerun == report
    assert {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_splitter_refuses_absent_or_wrong_image_zip_sha256(
    tmp_path: Path, digest: str | None
) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    paths["image_zip_sha256"] = digest

    with pytest.raises(BallNoCleanJudge, match="scratch image ZIP.*SHA-256"):
        _build(paths, tmp_path / "out")


def test_materialized_image_verification_refuses_tampered_zip_entry_byte(
    tmp_path: Path,
) -> None:
    from scripts.racketsport.build_ball_regroup_split import (
        BallNoCleanJudge,
        _load_scratch_package,
        _verify_materialized_scratch_image_bytes,
    )

    paths = _build_fixture(tmp_path)
    package_rows, contract, _ = _load_scratch_package(
        Path(paths["package_manifest"]),
        expected_scratch_count=3,
        expected_scratch_package_sha256=None,
        expected_image_zip_sha256=str(paths["image_zip_sha256"]),
    )
    images_zip = Path(paths["images_zip"])
    with zipfile.ZipFile(images_zip) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    tampered_member = str(package_rows[0]["image_zip_member"])
    tampered = bytearray(members[tampered_member])
    tampered[-1] ^= 0x01
    members[tampered_member] = bytes(tampered)
    replacement = tmp_path / "replacement.zip"
    with zipfile.ZipFile(replacement, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    replacement.replace(images_zip)
    tampered_zip_sha256 = hashlib.sha256(images_zip.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="image byte SHA-256 mismatch"):
        _verify_materialized_scratch_image_bytes(
            [{**row, "lineage_class": "scratch"} for row in package_rows],
            expected_image_zip=images_zip,
            expected_image_zip_sha256=tampered_zip_sha256,
            expected_entry_sha256=dict(contract["image_zip_entry_sha256"]),
        )


def test_splitter_refuses_missing_negative_attestation(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    paths["negative_attestation"] = None

    with pytest.raises(
        BallNoCleanJudge,
        match=r"AWAITING_ATTESTATION: .*UNATTESTED_NEGATIVE.*evaluation_eligible=false",
    ):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_negative_attestation_for_different_export(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    attestation_path = Path(paths["negative_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["export_sha256"] = "0" * 64
    _write_json(attestation_path, attestation)

    with pytest.raises(BallNoCleanJudge, match="attestation export_sha256 mismatch"):
        _build(paths, tmp_path / "out")


@pytest.mark.parametrize("field", ["all_frames_inspected", "boxless_means_no_ball"])
def test_splitter_refuses_false_negative_attestation_flags(
    tmp_path: Path, field: str
) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    attestation_path = Path(paths["negative_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation[field] = False
    _write_json(attestation_path, attestation)

    with pytest.raises(BallNoCleanJudge, match=rf"{field} must be true"):
        _build(paths, tmp_path / "out")


def test_production_mode_refuses_noncanonical_protected_set(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import (
        BallNoCleanJudge,
        _assert_no_protected_collisions,
    )

    paths = _build_fixture(tmp_path)

    with pytest.raises(BallNoCleanJudge, match="exact canonical four protected videos"):
        _assert_no_protected_collisions(
            [],
            protected_videos=[Path(paths["protected_video"])],
            protected_additions=[Path(paths["protected_addition"])],
            threshold=0,
            production_mode=True,
        )


def test_splitter_refuses_tampered_pinned_lineage_input(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    selection_path = Path(paths["selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["tampered"] = True
    _write_json(selection_path, selection)

    with pytest.raises(BallNoCleanJudge, match="lineage inputs SHA-256 mismatch"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_auto_sourced_scratch_export(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    export_path = Path(paths["export_zip"])
    _write_scratch_export(
        export_path,
        list(paths["scratch_rows"]),
        boxes_by_name={
            str(paths["scratch_rows"][0]["image_name"]): (10, 10, 18, 18),
            str(paths["scratch_rows"][1]["image_name"]): None,
            str(paths["scratch_rows"][2]["image_name"]): (20, 20, 28, 28),
        },
        box_source="auto",
    )
    export_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
    paths["export_sha256"] = export_sha
    attestation_path = Path(paths["negative_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["export_sha256"] = export_sha
    _write_json(attestation_path, attestation)

    with pytest.raises(BallNoCleanJudge, match="box source must be manual"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_nonnull_task_prelabel_zip(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    ledger_path = Path(paths["import_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["tasks"][0]["prelabel_zip"] = "automatic_labels.zip"
    _write_json(ledger_path, ledger)
    paths["import_ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="prelabel_zip must be null or absent"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_wrong_task_fingerprint(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    ledger_path = Path(paths["import_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["tasks"][0]["task_fingerprint"] = "b" * 64
    _write_json(ledger_path, ledger)
    paths["import_ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="task fingerprint mismatch"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_wrong_task_identity(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    ledger_path = Path(paths["import_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["tasks"][0]["task_id"] = 88
    _write_json(ledger_path, ledger)
    paths["import_ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="task_id must be 87"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_injected_wrong_image_zip_sha256(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    ledger_path = Path(paths["import_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["tasks"][0]["image_zip_sha256"] = "0" * 64
    _write_json(ledger_path, ledger)
    paths["import_ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="image_zip_sha256 mismatch"):
        _build(paths, tmp_path / "out")


def test_splitter_refuses_injected_wrong_job_id(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    ledger_path = Path(paths["import_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["tasks"][0]["job_id"] = 999
    _write_json(ledger_path, ledger)
    paths["import_ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

    with pytest.raises(BallNoCleanJudge, match="job_id 999 != 87"):
        _build(paths, tmp_path / "out")


def test_splitter_declares_job_id_unavailable_when_all_artifacts_omit_it(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    export_path = Path(paths["export_zip"])
    _write_scratch_export(
        export_path,
        list(paths["scratch_rows"]),
        boxes_by_name={
            str(paths["scratch_rows"][0]["image_name"]): (10, 10, 18, 18),
            str(paths["scratch_rows"][1]["image_name"]): None,
            str(paths["scratch_rows"][2]["image_name"]): (20, 20, 28, 28),
        },
        task_id=None,
    )
    export_sha256 = hashlib.sha256(export_path.read_bytes()).hexdigest()
    paths["export_sha256"] = export_sha256
    attestation_path = Path(paths["negative_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["export_sha256"] = export_sha256
    _write_json(attestation_path, attestation)

    report = _build(paths, tmp_path / "out")

    assert report["input_contract"]["job_id"] is None
    assert report["input_contract"]["job_id_binding"] == "UNAVAILABLE_IN_ARTIFACTS"
    assert report["residual_assumptions"]


def test_splitter_refuses_collision_with_any_protected_frame(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    colliding = tmp_path / "colliding_protected.png"
    first_image = next(iter(paths["scratch_images"].values()))
    colliding.write_bytes(first_image)
    paths["protected_addition"] = colliding

    with pytest.raises(BallNoCleanJudge, match="protected image collision"):
        _build(paths, tmp_path / "out")


def test_collision_guard_hashes_every_protected_video_frame(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import (
        BallNoCleanJudge,
        _assert_no_protected_collisions,
        _hash_video_all_frames,
    )

    paths = _build_fixture(tmp_path)
    protected_video = Path(paths["protected_video"])
    protected_frames = _hash_video_all_frames(protected_video)
    assert len(protected_frames) == 2

    with pytest.raises(BallNoCleanJudge, match="protected image collision"):
        _assert_no_protected_collisions(
            [protected_frames[-1]],
            protected_videos=[protected_video],
            protected_additions=[],
            threshold=0,
        )


def test_direct_cli_failure_contains_required_verdict(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    missing_export = tmp_path / "missing.zip"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--reviewed-root",
            str(paths["reviewed_root"]),
            "--scratch-package",
            str(paths["package_manifest"]),
            "--scratch-export",
            str(missing_export),
            "--holdout-source",
            "hold_src",
            "--out",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "BALL_NO_CLEAN_JUDGE" in completed.stderr


def test_splitter_refuses_non_scratch_package_row_provenance(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import BallNoCleanJudge

    paths = _build_fixture(tmp_path)
    package_path = Path(paths["package_manifest"])
    package = json.loads(package_path.read_text(encoding="utf-8"))
    sampling_path = Path(package["sampling_manifest"])
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    sampling["frames"][0]["provenance_class"] = "confirmed_prelabel"
    _write_json(sampling_path, sampling)
    package["sampling_manifest_md5"] = hashlib.md5(sampling_path.read_bytes()).hexdigest()
    _write_json(package_path, package)

    with pytest.raises(BallNoCleanJudge, match="provenance_class must be scratch"):
        _build(paths, tmp_path / "out")


def test_legacy_lineage_uses_export_directory_for_project_task_identity(tmp_path: Path) -> None:
    from scripts.racketsport.build_ball_regroup_split import _legacy_reviewed_keys

    clip_id = "source_with_prefix_rally_0001"
    annotations = tmp_path / clip_id / "annotations.xml"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(
        """<annotations><version>1.1</version><meta><project><tasks><task>
        <name>inconsistent_display_prefix_source_with_prefix_rally_0001</name>
        <size>3</size><start_frame>0</start_frame><stop_frame>8</stop_frame>
        <frame_filter>step=4</frame_filter>
        </task></tasks></project></meta></annotations>""",
        encoding="utf-8",
    )

    assert _legacy_reviewed_keys(tmp_path) == {
        f"{clip_id}:000000",
        f"{clip_id}:000004",
        f"{clip_id}:000008",
    }


def test_absent_legacy_model_record_does_not_confirm_out_of_frame_label() -> None:
    from scripts.racketsport.build_ball_regroup_split import _legacy_label_confirmed

    original_absence = {
        "ball_present": False,
        "xy": None,
        "confidence": 0.0,
        "proposal_source": "wasb",
    }
    assert _legacy_label_confirmed(
        {
            "ball_present": False,
            "bbox_xyxy": None,
            "visibility_level": "none",
        },
        original_absence,
    )
    assert not _legacy_label_confirmed(
        {
            "ball_present": False,
            "bbox_xyxy": None,
            "visibility_level": "out_of_frame",
        },
        original_absence,
    )


def test_direct_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--holdout-source" in completed.stdout
    assert "--negative-attestation" in completed.stdout
