from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import pytest
from PIL import Image

from scripts.racketsport import ingest_cvat_court_images as ingest
from scripts.racketsport.train_court_model_v2 import load_real_training_rows, real_row_to_sample_arrays


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/racketsport/ingest_cvat_court_images.py"
PINNED_MANIFEST_SHA256 = "c0243e9146152c5c46b5d0aebca9d571bfd39b6e90b34227d4024d09eabcdd7e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def _write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_video(path: Path, frames_rgb: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError(f"could not create synthetic protected video: {path}")
    try:
        for frame_rgb in frames_rgb:
            writer.write(np.ascontiguousarray(frame_rgb[:, :, ::-1]))
    finally:
        writer.release()


def _first_video_frame_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open synthetic protected video: {path}")
    try:
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not decode synthetic protected video: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _phash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image))


def _point(image: ET.Element, label: str, x: float, y: float) -> None:
    shape = ET.SubElement(
        image,
        "points",
        {
            "label": label,
            "source": "manual",
            "occluded": "0",
            "points": f"{x:.6f},{y:.6f}",
            "z_order": "0",
        },
    )
    attribute = ET.SubElement(shape, "attribute", {"name": "source"})
    attribute.text = "owner"


def _write_shard_zip(path: Path, image_elements: list[ET.Element]) -> None:
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    for image in image_elements:
        annotations.append(image)
    xml_bytes = ET.tostring(annotations, encoding="utf-8", xml_declaration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.xml", xml_bytes)


def _fixture(
    tmp_path: Path,
    *,
    invalid_train_rows: int = 0,
    invalid_holdout_rows: int = 0,
    train_source_count: int = 15,
    shared_holdout_family: bool = False,
    denied_row_count: int = 3,
) -> dict[str, Any]:
    package = tmp_path / "package"
    frames = package / "frames"
    protected_root = tmp_path / "protected"
    protected_a = protected_root / "eval_a" / "frame_000001.png"
    protected_b = protected_root / "eval_b" / "frame_000002.png"
    _write_image(protected_a, _noise(9001))
    _write_image(protected_b, _noise(9002))
    protected_video = protected_root / "eval_video" / "source.avi"
    _write_video(protected_video, [_noise(9003), _noise(9004)])
    protected_video_collision_pixels = _first_video_frame_rgb(protected_video)

    train_sources = [f"train_src_{index:02d}" for index in range(train_source_count)]
    holdout_sources = list(ingest.HOLDOUT_SOURCE_IDS)
    denied_source = "IYnbdRs1Jdk"
    family_collision_source = train_sources[-1] if shared_holdout_family else None
    image_specs: list[tuple[str, str, np.ndarray, str]] = []
    seed = 1
    for source_id in train_sources:
        kind = "family_poison" if source_id == family_collision_source else "regular"
        for row_index in range(4):
            image_specs.append(
                (source_id, f"{source_id}__seg1_f{row_index + 1:03d}.png", _noise(seed), kind)
            )
            seed += 1
    collision_name = f"{train_sources[0]}__seg2_f099.png"
    image_specs.append((train_sources[0], collision_name, protected_video_collision_pixels, "phash_poison"))
    for source_id in holdout_sources:
        for row_index in range(2):
            image_specs.append(
                (source_id, f"{source_id}__seg2_f{row_index + 1:03d}.png", _noise(seed), "regular")
            )
            seed += 1
    for row_index in range(denied_row_count):
        image_specs.append(
            (denied_source, f"{denied_source}__seg1_f{row_index + 1:03d}.png", _noise(seed), "source_poison")
        )
        seed += 1

    train_regular = [name for source, name, _, kind in image_specs if source in train_sources and kind == "regular"]
    holdout_regular = [name for source, name, _, kind in image_specs if source in holdout_sources and kind == "regular"]
    invalid_names = set(train_regular[:invalid_train_rows]) | set(holdout_regular[:invalid_holdout_rows])

    manifest_images: list[dict[str, Any]] = []
    elements_by_name: dict[str, ET.Element] = {}
    for image_id, (source_id, file_name, pixels, kind) in enumerate(image_specs):
        image_path = frames / file_name
        _write_image(image_path, pixels)
        if source_id == family_collision_source:
            channel = "PPA Tour"
            venue_group = "inferred:PPA Tour"
        elif source_id == holdout_sources[1]:
            channel = "PPA Tour"
            venue_group = "inferred:PPA Tour"
        else:
            channel = f"channel:{source_id}"
            venue_group = f"venue:{source_id}"
        manifest_images.append(
            {
                "file_name": file_name,
                "source_id": source_id,
                "source_video_url": f"https://example.test/watch?v={source_id}",
                "channel": channel,
                "venue_group": venue_group,
                "strata": {"indoor_outdoor": "indoor" if source_id in holdout_sources[4:] else "outdoor"},
                "resolution": [64, 48],
                "frame_sha256": _sha256(image_path),
                "phash64_hex": _phash(image_path),
            }
        )
        image = ET.Element(
            "image",
            {"id": str(image_id), "name": file_name, "width": "64", "height": "48"},
        )
        elements_by_name[file_name] = image
        if kind in {"source_poison", "phash_poison", "family_poison"}:
            ET.SubElement(image, "points", {"label": "POISON_LABEL", "points": "not,a,point"})
            continue
        _point(image, "near_left_corner", 5.0, 42.0)
        _point(image, "near_right_corner", 59.0, 42.0)
        _point(image, "far_left_corner", 19.0, 8.0)
        if file_name not in invalid_names:
            _point(image, "far_right_corner", 45.0, 8.0)
        _point(image, "net_center", 32.0, 24.0)

    shard_names = [f"court_diversity_20260712_shard{index}" for index in range(1, 5)]
    shard_file_names = [list(elements_by_name)[index::4] for index in range(4)]
    shards = [
        {
            "shard_name": shard_name,
            "task_name": shard_name,
            "image_count": len(file_names),
            "file_names": file_names,
        }
        for shard_name, file_names in zip(shard_names, shard_file_names)
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": ingest.SYNTHETIC_MANIFEST_ARTIFACT_TYPE,
        "image_count": len(manifest_images),
        "distinct_source_video_count": len({row["source_id"] for row in manifest_images}),
        "shards": shards,
        "images": manifest_images,
    }
    manifest_path = package / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    export_dir = tmp_path / "exports"
    for shard_name, file_names in zip(shard_names, shard_file_names):
        _write_shard_zip(
            export_dir / f"{shard_name}_annotations.zip",
            [elements_by_name[file_name] for file_name in file_names],
        )
    return {
        "manifest": manifest_path,
        "manifest_payload": manifest,
        "export": export_dir,
        "protected_root": protected_root,
        "protected_paths": [protected_a, protected_b],
        "protected_video": protected_video,
        "out": tmp_path / "out",
        "collision_name": collision_name,
        "train_sources": train_sources,
        "holdout_sources": holdout_sources,
        "denied_source": denied_source,
        "family_collision_source": family_collision_source,
        "expected_shards": {
            shard_name: frozenset(file_names) for shard_name, file_names in zip(shard_names, shard_file_names)
        },
    }


def _run(
    fixture: dict[str, Any],
    *,
    deny_sources: list[str] | None = None,
    phash_max_distance: int = 3,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--package-manifest",
        str(fixture["manifest"]),
        "--cvat-export",
        str(fixture["export"]),
        "--protected-root",
        str(fixture["protected_root"]),
        "--out",
        str(fixture["out"]),
        "--phash-max-distance",
        str(phash_max_distance),
        "--synthetic-fixture",
    ]
    for source_id in deny_sources or [fixture["denied_source"]]:
        command.extend(["--deny-source", source_id])
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


def _loaded_fixture_state(fixture: dict[str, Any]) -> tuple[list[ingest.ManifestImage], dict[str, ingest.SourceAssignment]]:
    payload, _, images = ingest.load_package_manifest(fixture["manifest"], synthetic_fixture=True)
    assignments, _ = ingest.build_source_assignments(images, deny_sources=set(ingest.PERMANENT_DENY_SOURCE_IDS))
    expected_shards = ingest._expected_export_shards(payload, images=images, required=False)
    assert expected_shards is not None
    prelabel = {
        image.file_name: assignments[image.source_id].state
        for image in images
        if assignments[image.source_id].state
        in {ingest.STATE_DENIED_PERMANENT_SOURCE, ingest.STATE_QUARANTINED_FAMILY_COLLISION}
    }
    usable, _, _ = ingest.read_cvat_export(
        fixture["export"],
        images=images,
        prelabel_excluded_states=prelabel,
        phash_denied_names={fixture["collision_name"]},
        assignments=assignments,
        expected_shards=expected_shards,
    )
    return images, assignments


def _geometry_image(*, width: int = 64, height: int = 48) -> ingest.ManifestImage:
    return ingest.ManifestImage(
        file_name="geometry__seg_f001.png",
        source_id="geometry",
        path=Path("unused.png"),
        width=width,
        height=height,
        frame_sha256="0" * 64,
        declared_phash64_hex=None,
        channel="geometry-channel",
        venue_group="geometry-venue",
        indoor_outdoor="outdoor",
        source_video_url=None,
    )


def _parse_geometry(
    points: dict[str, tuple[float, float]],
    *,
    width: int = 64,
    height: int = 48,
) -> dict[str, list[float] | None]:
    element = ET.Element(
        "image",
        {"name": "geometry__seg_f001.png", "width": str(width), "height": str(height)},
    )
    for label, (x, y) in points.items():
        _point(element, label, x, y)
    return ingest.parse_reviewed_keypoints(element, _geometry_image(width=width, height=height))


VALID_CORNERS = {
    "near_left_corner": (5.0, 42.0),
    "near_right_corner": (59.0, 42.0),
    "far_left_corner": (19.0, 8.0),
    "far_right_corner": (45.0, 8.0),
}


def test_contract_literals_and_manifest_mode_are_independently_pinned() -> None:
    assert ingest.PRODUCTION_MANIFEST_SHA256 == PINNED_MANIFEST_SHA256
    assert ingest.ORGANIZATIONAL_FAMILY_ALIAS_MAP == {"PPA Tour Asia": "PPA Tour"}
    alias_bytes = json.dumps(
        ingest.ORGANIZATIONAL_FAMILY_ALIAS_MAP,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(alias_bytes).hexdigest() == ingest.ORGANIZATIONAL_FAMILY_ALIAS_MAP_SHA256
    production = {"artifact_type": "racketsport_court_diversity_20260712_package_manifest"}
    ingest._validate_manifest_mode(production, PINNED_MANIFEST_SHA256, synthetic_fixture=False)
    with pytest.raises(ValueError, match="artifact_type mismatch"):
        ingest._validate_manifest_mode({"artifact_type": "altered"}, PINNED_MANIFEST_SHA256, synthetic_fixture=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest._validate_manifest_mode(production, "0" * 64, synthetic_fixture=False)
    with pytest.raises(ValueError, match="--synthetic-fixture requires"):
        ingest._validate_manifest_mode(production, PINNED_MANIFEST_SHA256, synthetic_fixture=True)


def test_real_pinned_manifest_shards_and_ppa_family_ruling_are_accepted_today(tmp_path: Path) -> None:
    manifest_path = ROOT / "cvat_upload/court_diversity_20260712/package_manifest.json"
    payload, manifest_sha256, images = ingest.load_package_manifest(manifest_path)
    assert manifest_sha256 == PINNED_MANIFEST_SHA256
    assert "shards" in payload
    assert "task_shards" not in payload
    expected_shards = ingest._expected_export_shards(payload, images=images, required=True)
    assert expected_shards is not None
    assert len(expected_shards) == 4
    assert sum(len(names) for names in expected_shards.values()) == 100

    assignments, _ = ingest.build_source_assignments(
        images,
        deny_sources=set(ingest.PERMANENT_DENY_SOURCE_IDS),
    )
    row_counts = {
        split: sum(1 for image in images if assignments[image.source_id].split == split)
        for split in ("train", "holdout", "quarantined", "denied")
    }
    assert row_counts == {"train": 66, "holdout": 27, "quarantined": 4, "denied": 3}
    train_family_count = len(
        {
            assignments[image.source_id].source_family_key
            for image in images
            if assignments[image.source_id].split == "train"
        }
    )
    assert train_family_count == 18
    assert train_family_count >= ingest.MIN_TRAIN_SOURCE_GROUPS
    assert len({assignments[source_id].source_family_key for source_id in ingest.HOLDOUT_SOURCE_IDS}) == 7
    assert tuple(ingest.HOLDOUT_SOURCE_IDS) == (
        "1or-bXVM80M",
        "4qSoA-jwpVM",
        "C5YUQlqZqBY",
        "q3575jnmjJQ",
        "A9H6EWfXht0",
        "Se7M6ZKaC4Y",
        "a_HzWrwK6vM",
        "wv3aPJrDwK4",
    )
    ppa_family = assignments["4qSoA-jwpVM"].source_family_key
    assert assignments["Se7M6ZKaC4Y"].source_family_key == ppa_family
    assert assignments["3sC53GlvW_s"].source_family_key == ppa_family
    assert assignments["3sC53GlvW_s"].state == ingest.STATE_QUARANTINED_FAMILY_COLLISION

    # Exercise the production CLI with the real pinned manifest. It must advance
    # past manifest/shard validation and fail only at the deliberately absent
    # protected-root seam (the real four CVAT export ZIPs are not present yet).
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--package-manifest",
            str(manifest_path),
            "--cvat-export",
            str(tmp_path / "missing-export"),
            "--deny-source",
            "IYnbdRs1Jdk",
            "--protected-root",
            str(tmp_path / "missing-protected-root"),
            "--out",
            str(ROOT / "runs/lanes/court_c0_ingest_20260721" / f".pytest-manifest-probe-{tmp_path.name}"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    cli_error = json.loads(completed.stderr)
    assert "protected root does not exist" in cli_error["error"]
    assert "task_shards" not in cli_error["error"]


def test_wrong_valid_and_extra_deny_sets_fail_before_any_input_read(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for deny_sources in ([fixture["train_sources"][0]], [fixture["denied_source"], fixture["train_sources"][0]]):
        completed = _run(fixture, deny_sources=deny_sources)
        assert completed.returncode == 1
        error = json.loads(completed.stderr)
        assert "must be exactly the permanent deny set" in error["error"]
        assert not fixture["out"].exists()


@pytest.mark.parametrize("denied_row_count", [2, 4])
def test_manifest_requires_exactly_three_permanent_deny_rows(tmp_path: Path, denied_row_count: int) -> None:
    fixture = _fixture(tmp_path, denied_row_count=denied_row_count)
    completed = _run(fixture)
    assert completed.returncode == 1
    error = json.loads(completed.stderr)
    assert "must contain exactly 3 rows" in error["error"]
    assert not fixture["out"].exists()


def test_production_phash_floor_cannot_be_weakened_and_distances_one_through_three_hit() -> None:
    with pytest.raises(ValueError, match="cannot weaken"):
        ingest._validate_runtime_contract(
            deny_sources=set(ingest.PERMANENT_DENY_SOURCE_IDS),
            phash_max_distance=2,
            synthetic_fixture=False,
        )
    base = imagehash.hex_to_hash("0000000000000000")
    for distance in (1, 2, 3):
        protected = imagehash.hex_to_hash(f"{(1 << distance) - 1:016x}")
        assert int(base - protected) == distance
        assert ingest._phash_is_protected(base, protected, max_distance=3)
    distance_four = imagehash.hex_to_hash("000000000000000f")
    assert not ingest._phash_is_protected(base, distance_four, max_distance=3)


def test_fixture_passes_quarantine_family_gate_portable_c1_handoff_and_image_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    fixture = _fixture(tmp_path)
    repo_parent = Path(
        tempfile.mkdtemp(
            prefix=".pytest-c1-repo-root-",
            dir=ROOT / "runs/lanes/court_c0_ingest_20260721",
        )
    )
    request.addfinalizer(lambda: shutil.rmtree(repo_parent, ignore_errors=True))
    fixture["out"] = repo_parent / "real_court_diversity"
    monkeypatch.chdir(ROOT)
    completed = _run(fixture)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)

    assert report["verdict"] == ingest.SUFFICIENT_VERDICT
    assert report["counts"] == {
        "cvat_present": 80,
        "reviewed": 76,
        "usable": 76,
        "protected_denied": 4,
        "train": 60,
        "holdout": 16,
        "rejected": 0,
        "label_rejected": 0,
        "source_denied": 3,
        "phash_denied": 1,
        "family_quarantined": 0,
    }
    assert report["gate"]["passed"] is True
    assert report["gate"]["observed"]["train_family_group_count"] == 15
    assert report["gate"]["observed"]["holdout_family_group_count"] == 8
    assert report["gate"]["observed"]["holdout_rows_by_source"] == {
        source_id: 2 for source_id in ingest.HOLDOUT_SOURCE_IDS
    }
    phash = report["phash_guard"]
    assert phash["candidate_image_count"] == 77
    assert phash["protected_image_file_count"] == 2
    assert phash["protected_video_count"] == 1
    assert phash["protected_video_frame_count"] == 2
    assert phash["protected_frame_count"] == 4
    assert phash["comparison_count"] == 308
    assert phash["hit_file_names"] == [fixture["collision_name"]]
    assert any("source.avi#frame=0" in hit["protected_frame"] for hit in phash["hits"])

    split = json.loads((fixture["out"] / "source_split.json").read_text(encoding="utf-8"))
    assert split["partition_unit"] == "connected_source_channel_venue_family"
    assert split["frame_random_split"] is False
    expected_corpus_root = fixture["out"].relative_to(ROOT).as_posix()
    assert split["path_base"] == "repo_root"
    assert split["corpus_root"] == expected_corpus_root
    assert split["real_root"] == f"{expected_corpus_root}/train"
    assert set(split["train_datasets"]) == set(fixture["train_sources"])
    assert split["holdout_datasets"] == list(ingest.HOLDOUT_SOURCE_IDS)
    assert split["denied_source_ids"] == [fixture["denied_source"]]
    assert split["quarantined_family_collision_source_ids"] == []
    assert all(group["pre_label_inspection_assignment"] for group in split["groups"])

    raw_label_path = next((fixture["out"] / "train").glob("*/labels/court_keypoints.json"))
    raw_label = json.loads(raw_label_path.read_text(encoding="utf-8"))
    raw_item = raw_label["annotation"]["items"][0]
    assert set(raw_item["keypoints"]) == set(ingest.CANONICAL_KEYPOINT_NAMES)
    assert raw_item["keypoints"]["net_center"] == [32.0, 24.0]
    assert raw_item["keypoints"]["near_baseline_center"] is None
    assert raw_item["status"] == "reviewed"
    assert raw_label["frames"]["path_base"] == "corpus_root"
    assert not Path(raw_label["frames"]["frame_dir"]).is_absolute()
    expected_frame_dir = Path(raw_label_path.parent.parent.name) / "labels" / "court_keypoint_frames"
    assert raw_label["frames"]["frame_dir"] == expected_frame_dir.as_posix()
    assert (fixture["out"] / "train" / expected_frame_dir).is_dir()

    protected_shas = {_sha256(path) for path in fixture["protected_paths"]}
    staged_images = list(fixture["out"].glob("*/*/labels/court_keypoint_frames/*"))
    assert len(staged_images) == 76
    assert protected_shas.isdisjoint({_sha256(path) for path in staged_images})

    relocated_parent = Path(
        tempfile.mkdtemp(
            prefix=".pytest-c1-relocated-root-",
            dir=ROOT / "runs/lanes/court_c0_ingest_20260721",
        )
    )
    request.addfinalizer(lambda: shutil.rmtree(relocated_parent, ignore_errors=True))
    relocated = relocated_parent / "real_court_diversity_copy"
    shutil.copytree(fixture["out"], relocated)

    assert Path.cwd() == ROOT
    real_rows = load_real_training_rows(
        [relocated / "train"],
        split_proposal=relocated / "source_split.json",
    )
    assert len(real_rows) == 60
    assert {row["clip"] for row in real_rows} == set(fixture["train_sources"])
    assert all(row["image_path"] and Path(row["image_path"]).is_file() for row in real_rows)
    resolved_paths = [Path(row["image_path"]).resolve() for row in real_rows]
    assert all(path.is_relative_to(relocated.resolve()) for path in resolved_paths)
    assert not any(path.is_relative_to(fixture["out"].resolve()) for path in resolved_paths)
    arrays = real_row_to_sample_arrays(real_rows[0], model_width=64, model_height=48, sigma_px=1.5)
    assert int(arrays["keypoint_supervision_mask"].sum()) == 5


def test_shared_holdout_family_is_quarantined_before_label_read_and_gate_still_passes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_source_count=16, shared_holdout_family=True)
    completed = _run(fixture)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["counts"]["cvat_present"] == 84
    assert report["counts"]["family_quarantined"] == 4
    assert report["counts"]["rejected"] == 4
    assert report["counts"]["train"] == 60
    assert report["gate"]["observed"]["train_family_group_count"] == 15
    assert report["family_collision_quarantine"] == {
        "state": ingest.STATE_QUARANTINED_FAMILY_COLLISION,
        "source_ids": [fixture["family_collision_source"]],
        "row_count": 4,
    }
    split = json.loads((fixture["out"] / "source_split.json").read_text(encoding="utf-8"))
    collision_group = next(
        group for group in split["groups"] if group["source_id"] == fixture["family_collision_source"]
    )
    holdout_group = next(group for group in split["groups"] if group["source_id"] == ingest.HOLDOUT_SOURCE_IDS[1])
    assert collision_group["state"] == ingest.STATE_QUARANTINED_FAMILY_COLLISION
    assert collision_group["split"] == "quarantined"
    assert collision_group["source_family_key"] == holdout_group["source_family_key"]
    assert set(collision_group["family_source_ids"]) == {
        fixture["family_collision_source"],
        ingest.HOLDOUT_SOURCE_IDS[1],
    }


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (
            {
                "near_left_corner": (5.0, 42.0),
                "near_right_corner": (5.0, 42.0),
                "far_left_corner": (19.0, 8.0),
                "far_right_corner": (45.0, 8.0),
            },
            "duplicate floor anchors",
        ),
        (
            {
                "near_left_corner": (5.0, 20.0),
                "near_right_corner": (20.0, 20.0),
                "far_left_corner": (50.0, 20.0),
                "far_right_corner": (35.0, 20.0),
            },
            "collinear",
        ),
        (
            {
                "near_left_corner": (5.0, 42.0),
                "near_right_corner": (59.0, 42.0),
                "far_left_corner": (45.0, 8.0),
                "far_right_corner": (19.0, 8.0),
            },
            "crossed",
        ),
        (
            {
                "near_left_corner": (5.0, 20.0),
                "near_right_corner": (59.0, 20.0),
                "far_left_corner": (5.0, 19.99),
                "far_right_corner": (59.0, 19.99),
            },
            "near-zero-area",
        ),
        (
            {
                "near_left_corner": (5.0, 42.0),
                "near_right_corner": (59.0, 42.0),
                "far_left_corner": (19.0, 8.0),
                "far_right_corner": (32.0, 30.0),
            },
            "invertible",
        ),
    ],
)
def test_geometric_validity_rejects_unusable_four_corner_layouts(
    points: dict[str, tuple[float, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_geometry(points)


@pytest.mark.parametrize(
    ("width", "height", "points", "message"),
    [
        # Reviewer R2 probe verbatim: 96px^2 used to clear the old 92.16px^2 floor.
        (
            1280,
            720,
            {
                "near_left_corner": (40.0, 400.0),
                "near_right_corner": (1240.0, 400.0),
                "far_left_corner": (40.0, 399.92),
                "far_right_corner": (1240.0, 399.92),
            },
            "near-zero-area",
        ),
        # Reviewer R2 probe verbatim: large area but a 0.001px near baseline.
        (
            1280,
            720,
            {
                "near_left_corner": (10.0, 700.0),
                "near_right_corner": (10.001, 700.0),
                "far_left_corner": (100.0, 20.0),
                "far_right_corner": (1180.0, 20.0),
            },
            "subpixel/short floor edge",
        ),
        # Same normalized sliver at 2x resolution.
        (
            2560,
            1440,
            {
                "near_left_corner": (80.0, 800.0),
                "near_right_corner": (2480.0, 800.0),
                "far_left_corner": (80.0, 799.84),
                "far_right_corner": (2480.0, 799.84),
            },
            "near-zero-area",
        ),
        # A 9.7px square also cleared the old resolution-scaled area threshold.
        (
            1280,
            720,
            {
                "near_left_corner": (100.0, 109.7),
                "near_right_corner": (109.7, 109.7),
                "far_left_corner": (100.0, 100.0),
                "far_right_corner": (109.7, 100.0),
            },
            "near-zero-area",
        ),
        # Edges and area clear their floors, but normalized homography solve is unstable.
        (
            1280,
            720,
            {
                "near_left_corner": (10.0, 700.0),
                "near_right_corner": (12.0, 700.0),
                "far_left_corner": (100.0, 20.0),
                "far_right_corner": (1180.0, 20.0),
            },
            "ill-conditioned normalized court homography",
        ),
    ],
)
def test_scale_aware_geometry_rejects_reviewer_subpixel_slivers_and_variants(
    width: int,
    height: int,
    points: dict[str, tuple[float, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_geometry(points, width=width, height=height)


def test_geometric_validity_accepts_finite_invertible_court() -> None:
    parsed = _parse_geometry(VALID_CORNERS)
    assert parsed["near_left_corner"] == [5.0, 42.0]
    assert parsed["far_right_corner"] == [45.0, 8.0]


def _gate_rows(train_family_count: int) -> list[ingest.ReviewedRow]:
    rows: list[ingest.ReviewedRow] = []
    for index in range(60):
        family_index = index % train_family_count
        source_id = f"train_{family_index:02d}"
        image = ingest.ManifestImage(
            file_name=f"{source_id}__f{index:03d}.png",
            source_id=source_id,
            path=Path("unused.png"),
            width=64,
            height=48,
            frame_sha256="0" * 64,
            declared_phash64_hex=None,
            channel=f"channel:{source_id}",
            venue_group=f"venue:{source_id}",
            indoor_outdoor="outdoor",
            source_video_url=None,
        )
        rows.append(ingest.ReviewedRow(image=image, keypoints={}, source_family_key=f"family:{family_index:02d}"))
    for source_index, source_id in enumerate(ingest.HOLDOUT_SOURCE_IDS):
        for row_index in range(2):
            image = ingest.ManifestImage(
                file_name=f"{source_id}__f{row_index:03d}.png",
                source_id=source_id,
                path=Path("unused.png"),
                width=64,
                height=48,
                frame_sha256="0" * 64,
                declared_phash64_hex=None,
                channel=f"channel:{source_id}",
                venue_group=f"venue:{source_id}",
                indoor_outdoor="outdoor",
                source_video_url=None,
            )
            rows.append(
                ingest.ReviewedRow(image=image, keypoints={}, source_family_key=f"holdout-family:{source_index}")
            )
    return rows


def test_gate_counts_families_and_pins_fourteen_vs_fifteen_boundary() -> None:
    fourteen = ingest.evaluate_gate(_gate_rows(14))
    assert fourteen["passed"] is False
    assert "usable train family groups 14 < 15" in fourteen["failure_reasons"]
    fifteen = ingest.evaluate_gate(_gate_rows(15))
    assert fifteen["passed"] is True
    assert fifteen["observed"]["train_family_group_count"] == 15


def test_gate_fails_nonzero_when_usable_train_rows_drop_below_sixty(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, invalid_train_rows=1)
    completed = _run(fixture)
    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    assert report["counts"]["train"] == 59
    assert report["counts"]["label_rejected"] == 1
    assert "usable train rows 59 < 60" in report["gate"]["failure_reasons"]
    assert (fixture["out"] / "ingest_report.json").is_file()


def test_gate_fails_nonzero_when_any_frozen_holdout_group_has_fewer_than_two_rows(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, invalid_holdout_rows=1)
    completed = _run(fixture)
    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    first_holdout = ingest.HOLDOUT_SOURCE_IDS[0]
    assert report["counts"]["holdout"] == 15
    assert report["gate"]["observed"]["holdout_rows_by_source"][first_holdout] == 1
    assert any(first_holdout in reason for reason in report["gate"]["failure_reasons"])


def test_missing_duplicate_and_incomplete_four_shard_exports_fail_closed(tmp_path: Path) -> None:
    missing_fixture = _fixture(tmp_path / "missing")
    missing_zip = sorted(missing_fixture["export"].glob("*.zip"))[0]
    missing_zip.unlink()
    with pytest.raises(ValueError, match="shard set mismatch"):
        ingest._xml_documents(missing_fixture["export"], expected_shards=missing_fixture["expected_shards"])

    incomplete_fixture = _fixture(tmp_path / "incomplete")
    first_zip = sorted(incomplete_fixture["export"].glob("*.zip"))[0]
    with zipfile.ZipFile(first_zip) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))
    root.remove(root.findall("image")[0])
    with zipfile.ZipFile(first_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True))
    with pytest.raises(ValueError, match="shard image reconciliation failed"):
        _loaded_fixture_state(incomplete_fixture)

    duplicate_fixture = _fixture(tmp_path / "duplicate")
    duplicate_zip = sorted(duplicate_fixture["export"].glob("*.zip"))[0]
    with zipfile.ZipFile(duplicate_zip) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))
    root.append(ET.fromstring(ET.tostring(root.findall("image")[0])))
    with zipfile.ZipFile(duplicate_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True))
    with pytest.raises(ValueError, match="duplicate images"):
        _loaded_fixture_state(duplicate_fixture)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["out"].mkdir()
    marker = fixture["out"] / "user-owned.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    completed = _run(fixture)
    assert completed.returncode == 1
    error = json.loads(completed.stderr)
    assert "refusing to overwrite" in error["error"]
    assert marker.read_text(encoding="utf-8") == "preserve\n"
