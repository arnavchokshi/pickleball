from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "racketsport" / "build_person_mixed_pseudo_pack.py"
SCRIPT_REPO_RELATIVE = Path("scripts/racketsport/build_person_mixed_pseudo_pack.py")
SPEC = importlib.util.spec_from_file_location("build_person_mixed_pseudo_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pack_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pack_cli
SPEC.loader.exec_module(pack_cli)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _percent_encoded_alias(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in value.encode("utf-8"))


def _fullwidth_alias(value: str) -> str:
    return "".join(
        chr(ord(character) + 0xFEE0) if 0x21 <= ord(character) <= 0x7E else character
        for character in value
    )


def _make_closed_p1(repo_root: Path) -> Path:
    root = repo_root / "closed_p1"
    rows = []
    train_entries = []
    for index in range(7):
        image = f"images/train/family_{index}.jpg"
        label = f"labels/train/family_{index}.txt"
        (root / image).parent.mkdir(parents=True, exist_ok=True)
        (root / label).parent.mkdir(parents=True, exist_ok=True)
        (root / image).write_bytes(f"train-image-{index}".encode())
        (root / label).write_text(f"0 0.5 0.5 0.2 0.4 # {index}\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": f"train:{index}",
                "family_id": f"family:train-{index}",
                "source": f"source/train-{index}",
                "image": image,
                "label": label,
                "split": "train",
            }
        )
        train_entries.append(f"./{image}")

    for split, family_id, name in (
        ("val", pack_cli.OD8AL_FAMILY, "od8al"),
        ("test", pack_cli.HEMEL_FAMILY, "hemel"),
    ):
        image = f"images/{split}/{name}.jpg"
        label = f"labels/{split}/{name}.txt"
        (root / image).parent.mkdir(parents=True, exist_ok=True)
        (root / label).parent.mkdir(parents=True, exist_ok=True)
        (root / image).write_bytes(f"{name}-human-image".encode())
        (root / label).write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": f"{split}:{name}",
                "family_id": family_id,
                "source": f"source/{name}",
                "image": image,
                "label": label,
                "split": split,
            }
        )

    _write_json(root / "dataset_manifest.json", {"schema_version": 2, "rows": rows})
    # Order and repetition are deliberate: the builder must preserve the exact
    # closed list bytes and exposure order, not reconstruct it from families.
    ordered = [train_entries[3], train_entries[0], *train_entries, train_entries[3]]
    (root / "train_family_balanced.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return root


def _make_source_inventories(repo_root: Path) -> tuple[Path, Path, dict[str, str]]:
    pb_manifest = repo_root / "data" / "pbvision" / "MANIFEST.json"
    pb_ids = sorted(pack_cli.PBVISION_TRAIN_IDS | pack_cli.COMPARE_ONLY_PBVISION_IDS)
    pb_videos = [
        {
            "video_id": source_id,
            "title": source_id,
            "duration_s": 2.0,
            "resolution": "640x360@30fps",
            "video_sha256": pack_cli.PBVISION_MEDIA_SHA256_BY_ID[source_id],
            "video_location": f"vm:/media/{source_id}.mp4",
        }
        for source_id in pb_ids
    ]
    _write_json(pb_manifest, {"videos": pb_videos})

    harvest_manifest = repo_root / "data" / "harvest" / "manifest.json"
    harvest_rows = []
    expected_shas = {}
    for source_id in sorted(pack_cli.HARVEST_TRAIN_IDS):
        content = f"harvest-media:{source_id}".encode()
        media = harvest_manifest.parent / "raw" / f"{source_id}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(content)
        expected_shas[source_id] = _digest(content)
        harvest_rows.append(
            {
                "id": source_id,
                "title": source_id,
                "channel": f"channel:{source_id}",
                "duration_s": 2.0,
                "fps": 30,
                "file": f"raw/{source_id}.mp4",
                "license_field": None,
                "status": "downloaded",
            }
        )
    _write_json(harvest_manifest, harvest_rows)
    return pb_manifest, harvest_manifest, expected_shas


def _make_teacher(repo_root: Path) -> tuple[Path, Path]:
    teacher = repo_root / "models" / "checkpoints" / "yolo26m.pt"
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"stock-yolo26m-fixture")
    manifest = repo_root / "models" / "MANIFEST.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "models": [
                {
                    "id": "yolo26m",
                    "local_path": "/remote/yolo26m.pt",
                    "sha256": _digest(teacher.read_bytes()),
                }
            ],
        },
    )
    return manifest, teacher


def _fixture_closed_registry(p1: Path) -> dict[str, str]:
    fidelity = pack_cli._inspect_closed_p1(p1)["fidelity"]
    return {key: fidelity[key] for key in pack_cli.FROZEN_FIDELITY_HASH_KEYS}


def _build_fixture(repo_root: Path, out: Path) -> tuple[dict, Path]:
    p1 = _make_closed_p1(repo_root)
    expected_fidelity = _fixture_closed_registry(p1)
    pb_manifest, harvest_manifest, harvest_shas = _make_source_inventories(repo_root)
    model_manifest, teacher = _make_teacher(repo_root)
    result = pack_cli.build_pack(
        repo_root=repo_root,
        p1_root=p1,
        pbvision_manifest_path=pb_manifest,
        harvest_manifest_path=harvest_manifest,
        model_manifest_path=model_manifest,
        teacher_path=teacher,
        out=out,
        protocol=pack_cli.Protocol(
            target_pseudo_frames=72,
            max_frames_per_source=4,
            max_family_fraction=0.15,
            cli_protocol_override=True,
        ),
        expected_harvest_sha256=harvest_shas,
        expected_closed_p1_fidelity=expected_fidelity,
        closed_p1_registry_id="pytest.synthetic_closed_p1",
    )
    return result, p1


def _materialize_fixture_final_lists(
    repo_root: Path,
    out: Path,
    p1: Path,
    *,
    collide_with_validation: bool = False,
) -> tuple[Path, Path]:
    decode_rows = [
        json.loads(line) for line in (out / "decode_plan.jsonl").read_text().splitlines()
    ]
    pseudo_entries: list[str] = []
    validation_bytes = (p1 / "images/val/od8al.jpg").read_bytes()
    for index, _ in enumerate(decode_rows):
        image = out / "materialized" / f"pseudo_{index:05d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(
            validation_bytes
            if collide_with_validation and index == 0
            else f"materialized-pseudo-frame-{index}".encode()
        )
        pseudo_entries.append(image.relative_to(repo_root).as_posix())

    pseudo_list = out / "pseudo_train.txt"
    pseudo_list.write_text("\n".join(pseudo_entries) + "\n", encoding="utf-8")
    p1_data = pack_cli.load_closed_p1(
        p1,
        expected_fidelity=_fixture_closed_registry(p1),
        registry_id="pytest.synthetic_closed_p1",
    )
    anchor_refs = [
        (p1 / str(row["image"])).relative_to(repo_root).as_posix()
        for row in p1_data["anchor_records"]
    ]
    mixed_entries: list[str] = []
    for index, pseudo_entry in enumerate(pseudo_entries):
        mixed_entries.extend([anchor_refs[index % len(anchor_refs)], pseudo_entry])
    mixed_list = out / "mixed_train.txt"
    mixed_list.write_text("\n".join(mixed_entries) + "\n", encoding="utf-8")
    return pseudo_list, mixed_list


@pytest.mark.parametrize(
    "source",
    [
        *({"source_id": source_id} for source_id in pack_cli.PROTECTED_CLIP_IDS),
        *({"source_id": source_id} for source_id in pack_cli.COMPARE_ONLY_PBVISION_IDS),
        {"source_id": "crop-IYnbdRs1Jdk-v3-horizontal-flip"},
        {"source_id": "renamed", "lineage": {"parent": "IYnbdRs1Jdk/frame_0020"}},
        {"source_id": "renamed", "expected_media_sha256": next(iter(pack_cli.PROTECTED_MEDIA_SHA256))},
        {"source_id": "derived-83gyqyc10y8f-crop"},
    ],
)
def test_binding_quarantines_refuse_protected_compare_and_iynbd_derivatives(source: dict) -> None:
    with pytest.raises(pack_cli.QuarantinedSourceError, match="refused source"):
        pack_cli.assert_source_allowed(source)


@pytest.mark.parametrize(
    "identity",
    [
        *pack_cli.PROTECTED_CLIP_IDS,
        *sorted(pack_cli.COMPARE_ONLY_PBVISION_IDS),
        pack_cli.IYNBD_DERIVATIVE_TOKEN,
    ],
)
def test_percent_encoded_quarantine_aliases_are_refused(identity: str) -> None:
    with pytest.raises(pack_cli.QuarantinedSourceError, match="refused source"):
        pack_cli.assert_source_allowed(
            {"source_id": "permitted-looking", "path": _percent_encoded_alias(identity)}
        )


@pytest.mark.parametrize(
    "identity",
    [
        pack_cli.PROTECTED_CLIP_IDS[0].upper(),
        _fullwidth_alias(sorted(pack_cli.COMPARE_ONLY_PBVISION_IDS)[0]),
        _fullwidth_alias(pack_cli.IYNBD_DERIVATIVE_TOKEN),
    ],
)
def test_case_and_unicode_form_quarantine_aliases_are_refused(identity: str) -> None:
    with pytest.raises(pack_cli.QuarantinedSourceError, match="refused source"):
        pack_cli.assert_source_allowed({"source_id": identity})


def test_compare_only_media_sha_is_refused_under_permitted_id() -> None:
    with pytest.raises(pack_cli.QuarantinedSourceError, match="COMPARE_ONLY_MEDIA_SHA256"):
        pack_cli.assert_source_allowed(
            {
                "source_id": sorted(pack_cli.PBVISION_TRAIN_IDS)[0],
                "expected_media_sha256": sorted(pack_cli.COMPARE_ONLY_MEDIA_SHA256)[0],
            }
        )


def test_pbvision_manifest_requires_committed_id_to_sha_binding(tmp_path: Path) -> None:
    pb_manifest, _, _ = _make_source_inventories(tmp_path)
    manifest = json.loads(pb_manifest.read_text(encoding="utf-8"))
    permitted = sorted(pack_cli.PBVISION_TRAIN_IDS)[0]
    row = next(value for value in manifest["videos"] if value["video_id"] == permitted)
    row["video_sha256"] = sorted(pack_cli.COMPARE_ONLY_MEDIA_SHA256)[0]
    _write_json(pb_manifest, manifest)
    with pytest.raises(pack_cli.QuarantinedSourceError, match="COMPARE_ONLY_MEDIA_SHA256"):
        pack_cli.load_pbvision_sources(pb_manifest)


def test_ball_judge_holdouts_are_person_train_eligible_but_role_stamped() -> None:
    for source_id in ("HyUqT7zFiwk", "Ezz6HDNHlnk"):
        pack_cli.assert_source_allowed({"source_id": source_id})
        assert pack_cli.CROSS_COMPONENT_HOLDOUT_ROLES[source_id] == ("BALL_judge_holdout",)


def test_uniform_sampling_and_caps_are_deterministic() -> None:
    sources = [
        {
            "source_id": f"source-{index:02d}",
            "source_family_id": f"family-{index:02d}",
            "expected_frame_count": 20_000,
        }
        for index in range(18)
    ]
    protocol = pack_cli.Protocol(target_pseudo_frames=10_000)
    first = pack_cli.allocate_source_counts(sources, protocol)
    second = pack_cli.allocate_source_counts(list(reversed(sources)), protocol)
    assert first == second
    assert sum(first.values()) == 7200
    assert set(first.values()) == {400}
    assert max(first.values()) / sum(first.values()) <= 0.15
    indices = pack_cli.uniform_frame_indices(20_000, 400)
    assert indices == pack_cli.uniform_frame_indices(20_000, 400)
    assert len(indices) == len(set(indices)) == 400
    assert (indices[0], indices[-1]) == (0, 19_999)


def test_shared_venue_family_cap_applies_across_multiple_source_videos() -> None:
    sources = [
        {
            "source_id": f"source-{index:02d}",
            "source_family_id": f"source-family-{index:02d}",
            "venue_source_family_id": (
                "shared-venue" if index < 3 else f"venue-family-{index:02d}"
            ),
            "expected_frame_count": 20_000,
        }
        for index in range(18)
    ]
    protocol = pack_cli.Protocol(target_pseudo_frames=7200)
    counts = pack_cli.allocate_source_counts(sources, protocol)
    planned = sum(counts.values())
    shared_count = sum(counts[f"source-{index:02d}"] for index in range(3))
    assert 6000 <= planned <= 7200
    assert shared_count / planned <= 0.15
    assert max(counts.values()) <= 400


def test_pack_is_byte_deterministic_and_gpu_gated(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "lane"
    result, _ = _build_fixture(tmp_path, out)
    first = {path.name: path.read_bytes() for path in sorted(out.iterdir())}
    result_again, _ = _build_fixture(tmp_path, out)
    second = {path.name: path.read_bytes() for path in sorted(out.iterdir())}

    assert result == result_again
    assert first == second
    assert result["status"] == "GPU_TEACHER_INFERENCE_REQUIRED"
    assert result["production_eligible"] is False
    assert "CLI_PROTOCOL_OVERRIDE" in result["production_ineligibility_reasons"]
    assert not (out / "data.yaml").exists()
    assert not (out / "mixed_train.txt").exists()
    assert (out / "data.yaml.template").is_file()


def test_pseudo_rows_carry_teacher_and_holdout_provenance(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "lane"
    result, _ = _build_fixture(tmp_path, out)
    rows = [json.loads(line) for line in (out / "decode_plan.jsonl").read_text().splitlines()]
    assert len(rows) == 72
    assert {row["source_id"] for row in rows} == (
        pack_cli.PBVISION_TRAIN_IDS | pack_cli.HARVEST_TRAIN_IDS
    )
    assert all(row["split"] == "train" for row in rows)
    assert all(row["teacher_derived"] is True and row["ground_truth"] is False for row in rows)
    assert all("teacher_conf" in row and row["teacher_conf"] is None for row in rows)
    assert all(row["teacher_confidence_min"] == 0.60 for row in rows)
    assert all(row["teacher_checkpoint_sha256"] == result["teacher"]["checkpoint_sha256"] for row in rows)
    assert all(row["blind_spot_caveat"] == pack_cli.BLIND_SPOT_CAVEAT for row in rows)
    holdout_rows = [row for row in rows if row["source_id"] in {"HyUqT7zFiwk", "Ezz6HDNHlnk"}]
    assert holdout_rows
    assert all(row["cross_component_holdout_roles"] == ["BALL_judge_holdout"] for row in holdout_rows)


def test_pseudo_can_never_enter_validation_structurally() -> None:
    pseudo = [
        {
            "sample_id": "pseudo:1",
            "split": "train",
            "teacher_derived": True,
            "ground_truth": False,
        }
    ]
    human = [
        {
            "sample_id": "human:1",
            "split": "val",
            "teacher_derived": False,
            "ground_truth": True,
        }
    ]
    pack_cli.assert_no_pseudo_in_validation(pseudo, human)
    with pytest.raises(ValueError, match="human-only"):
        pack_cli.assert_no_pseudo_in_validation(
            pseudo,
            [{**human[0], "teacher_derived": True, "ground_truth": False}],
        )
    with pytest.raises(ValueError, match="overlap"):
        pack_cli.assert_no_pseudo_in_validation(
            pseudo,
            [{**human[0], "sample_id": "pseudo:1"}],
        )
    shared_content_sha = _digest(b"same-decoded-frame-bytes")
    with pytest.raises(ValueError, match="content identity overlap"):
        pack_cli.assert_no_pseudo_in_validation(
            [{**pseudo[0], "image_sha256": shared_content_sha}],
            [{**human[0], "image_sha256": shared_content_sha}],
        )


@pytest.mark.parametrize(
    "injected",
    [
        {"teacher_derived": True},
        {"ground_truth": False},
        {"label_origin": "teacher_pseudo_label"},
        {"teacher_checkpoint_sha256": "a" * 64},
    ],
)
def test_validation_rows_refuse_conflicting_provenance_instead_of_rewriting(
    injected: dict,
) -> None:
    row = {
        "sample_id": "val:human",
        "family_id": pack_cli.OD8AL_FAMILY,
        "source": "human-source",
        "image": "images/val/human.jpg",
        "label": "labels/val/human.txt",
        **injected,
    }
    with pytest.raises(pack_cli.FidelityError, match="human-only validation provenance conflict"):
        pack_cli._validation_rows([row], "val")


def test_closed_p1_manifest_rejects_injected_teacher_provenance_before_hashing(
    tmp_path: Path,
) -> None:
    p1 = _make_closed_p1(tmp_path)
    manifest_path = p1 / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_row = next(row for row in manifest["rows"] if row["split"] == "val")
    validation_row.update({"teacher_derived": True, "ground_truth": False})
    _write_json(manifest_path, manifest)
    with pytest.raises(pack_cli.FidelityError, match="human-only validation provenance conflict"):
        pack_cli.load_closed_p1(p1)


def test_closed_anchor_and_human_validation_byte_fidelity(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "lane"
    result, p1 = _build_fixture(tmp_path, out)
    fidelity = result["anchor_and_validation_fidelity"]
    assert fidelity["counts"] == {
        "closed_train_all": 7,
        "anchor_train_exposures": 10,
        "human_val_od8al": 1,
        "human_val_hemel": 1,
    }
    assert (out / "anchor_train_closed_byte_copy.txt").read_bytes() == (
        p1 / "train_family_balanced.txt"
    ).read_bytes()
    assert fidelity["closed_manifest_sha256"] == pack_cli.sha256_file(p1 / "dataset_manifest.json")
    assert fidelity["closed_train_list_sha256"] == pack_cli.sha256_file(
        p1 / "train_family_balanced.txt"
    )
    assert fidelity["direct_closed_file_references"] is True
    assert fidelity["images_or_labels_reencoded"] is False
    assert result["validation"]["human_only"] is True
    assert result["validation"]["pseudo_rows"] == 0
    assert fidelity["closed_hash_binding_passed"] is True
    assert fidelity["closed_hash_registry_id"] == "pytest.synthetic_closed_p1"


@pytest.mark.parametrize("drift_kind", ["manifest", "train_list", "row_bytes"])
def test_closed_p1_six_hash_registry_refuses_drift(tmp_path: Path, drift_kind: str) -> None:
    p1 = _make_closed_p1(tmp_path)
    registry = _fixture_closed_registry(p1)
    if drift_kind == "manifest":
        manifest_path = p1 / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["non_authoritative_replacement_note"] = "drift"
        _write_json(manifest_path, manifest)
    elif drift_kind == "train_list":
        train_list = p1 / "train_family_balanced.txt"
        train_list.write_text(
            train_list.read_text(encoding="utf-8") + "./images/train/family_0.jpg\n",
            encoding="utf-8",
        )
    else:
        (p1 / "images/train/family_0.jpg").write_bytes(b"replacement-image-bytes")

    with pytest.raises(pack_cli.FidelityError, match="closed P1 fidelity drift"):
        pack_cli.load_closed_p1(
            p1,
            expected_fidelity=registry,
            registry_id="pytest.synthetic_closed_p1",
        )


def test_interleave_is_exact_one_to_one_and_anchor_order_is_closed_order(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "lane"
    result, _ = _build_fixture(tmp_path, out)
    rows = [json.loads(line) for line in (out / "interleave_plan.jsonl").read_text().splitlines()]
    assert len(rows) == 144
    assert [row["position"] for row in rows] == list(range(144))
    assert [row["shard"] for row in rows[:8]] == ["anchor", "pseudo"] * 4
    assert rows[0]["sample_id"] == "train:3"
    assert rows[2]["sample_id"] == "train:0"
    assert result["shards"]["mixed_train"]["anchor_pseudo_exposure_ratio"] == "1:1"
    assert result["shards"]["mixed_train"]["anchor_exposures"] == 72
    assert result["shards"]["mixed_train"]["pseudo_exposures"] == 72


def test_executable_final_list_validator_passes_exact_materialization(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "lane"
    result, p1 = _build_fixture(tmp_path, out)
    pseudo_list, mixed_list = _materialize_fixture_final_lists(tmp_path, out, p1)
    report_path = out / "final_list_validation.json"
    validation = pack_cli.validate_final_lists(
        repo_root=tmp_path,
        p1_root=p1,
        out=out,
        pseudo_train_list_path=pseudo_list,
        mixed_train_list_path=mixed_list,
        report_path=report_path,
        expected_closed_p1_fidelity=_fixture_closed_registry(p1),
        closed_p1_registry_id="pytest.synthetic_closed_p1",
    )
    assert validation["status"] == "PASS"
    assert validation["pseudo_rows"] == 72
    assert validation["mixed_exposures"] == 144
    assert validation["human_validation_rows"] == 2
    assert validation["content_identity_overlap"] == 0
    assert report_path.is_file()
    contract = result["gpu_materialization_gate"]["final_list_validator"]
    assert contract["executable"] is True
    assert contract["must_pass_before_data_yaml"] is True
    assert contract["path"] == pack_cli.FINAL_LIST_VALIDATOR_RELATIVE_PATH.as_posix()
    assert "--validate-final-lists" in contract["command"]


def test_executable_final_list_validator_refuses_decoded_content_collision(
    tmp_path: Path,
) -> None:
    out = tmp_path / "runs" / "lane"
    _, p1 = _build_fixture(tmp_path, out)
    pseudo_list, mixed_list = _materialize_fixture_final_lists(
        tmp_path,
        out,
        p1,
        collide_with_validation=True,
    )
    with pytest.raises(ValueError, match="content identity overlap"):
        pack_cli.validate_final_lists(
            repo_root=tmp_path,
            p1_root=p1,
            out=out,
            pseudo_train_list_path=pseudo_list,
            mixed_train_list_path=mixed_list,
            expected_closed_p1_fidelity=_fixture_closed_registry(p1),
            closed_p1_registry_id="pytest.synthetic_closed_p1",
        )


def test_manifest_records_verbatim_blind_spot_and_experiment_bars(tmp_path: Path) -> None:
    result, _ = _build_fixture(tmp_path, tmp_path / "out")
    assert result["blind_spot_caveat"] == "teacher misses become background holes in pseudo frames"
    assert result["experiment_bars"]["aggregate_required"] == {
        "heldout_family_macro_F1_delta": ">0",
        "heldout_family_macro_mAP50_delta": ">0",
    }
    assert result["experiment_bars"]["per_family_required"][pack_cli.OD8AL_FAMILY] == {
        "F1_delta": ">=0",
        "mAP50_delta": ">=0",
    }
    assert result["experiment_bars"]["per_family_required"][pack_cli.HEMEL_FAMILY] == {
        "F1_delta": ">=0",
        "mAP50_delta": ">=0",
    }
    assert result["experiment_bars"]["selection_or_promotion_allowed"] is False


def test_teacher_checkpoint_must_match_manifest_sha(tmp_path: Path) -> None:
    manifest, teacher = _make_teacher(tmp_path)
    teacher.write_bytes(b"changed-after-manifest")
    with pytest.raises(ValueError, match="teacher checkpoint SHA-256 mismatch"):
        pack_cli.verify_teacher_checkpoint(manifest, teacher)


def test_any_protocol_cli_argument_marks_override() -> None:
    args = argparse.Namespace(
        target_pseudo_frames=None,
        max_frames_per_source=None,
        max_family_fraction=None,
        teacher_confidence=0.60,
        seed=None,
    )
    protocol = pack_cli._protocol_from_args(args)
    assert protocol.cli_protocol_override is True
    assert protocol.teacher_confidence == 0.60


def test_cli_help_is_directly_invocable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_REPO_RELATIVE), "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CPU decode/interleave plan" in result.stdout
