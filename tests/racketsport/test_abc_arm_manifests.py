from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.racketsport.build_abc_arm_manifests import (
    ABCMaterializationError,
    _assert_b_c_parity,
    build_vm_needs,
    materialize_arms,
    write_materializations,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/build_abc_arm_manifests.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _fixture_inputs(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Path]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video_id = "synthetic_train_clip"
    media = tmp_path / "synthetic.mp4"
    media.write_bytes(b"synthetic-media-pixels-for-hash-binding")
    media_sha = _sha256(media)
    frame_times = _write_json(tmp_path / "frame_times.json", {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_times",
        "source_video_sha256": media_sha,
        "fps": 10.0,
        "frame_count": 100,
        "duration_s": 10.0,
        "frames": [
            {"frame": frame, "pts_s": frame / 10.0}
            for frame in range(100)
        ],
    })
    frame_times_sha = _sha256(frame_times)
    binding_sha = _canonical_sha({
        "source_video_sha256": media_sha,
        "frame_times_sha256": frame_times_sha,
    })
    audio = _write_json(tmp_path / "audio_onsets_v2.json", {
        "schema_version": 1,
        "artifact_type": "racketsport_audio_onsets_v2",
        "clip": video_id,
        "source_video_sha256": media_sha,
        "frame_times_sha256": frame_times_sha,
        "onsets": [
            {"cue_id": "audio_e0", "corrected_time_s": 1.01},
            {"cue_id": "audio_e1", "corrected_time_s": 2.01},
        ],
    })
    ball = _write_json(tmp_path / "ball_inflections.json", {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_inflections",
        "video_id": video_id,
        "source_video_sha256": media_sha,
        "frame_times_sha256": frame_times_sha,
        "candidates": [
            {"cue_id": "ball_e0", "time_s": 1.02},
        ],
    })
    unknown = [False] * 100
    for frame in (10, 20, 30, 40):
        unknown[frame] = True
    events = [
        {
            "event_id": "e0",
            "class": "HIT",
            "frame": 10,
            "source_pts_s": 1.0,
            "teacher_confidence": 0.9,
            "needs_agreement_pass": True,
            "filter_decision": "pending_independent_agreement",
            "rally_source_start_frame": 0,
            "rally_source_end_frame_exclusive": 50,
        },
        {
            "event_id": "e1",
            "class": "BOUNCE",
            "frame": 20,
            "source_pts_s": 2.0,
            "teacher_confidence": 0.8,
            "needs_agreement_pass": True,
            "filter_decision": "pending_independent_agreement",
            "rally_source_start_frame": 0,
            "rally_source_end_frame_exclusive": 50,
        },
        {
            "event_id": "e_low",
            "class": "HIT",
            "frame": 30,
            "source_pts_s": 3.0,
            "teacher_confidence": 0.2,
            "needs_agreement_pass": False,
            "filter_decision": "rejected_low_teacher_confidence",
            "rally_source_start_frame": 0,
            "rally_source_end_frame_exclusive": 50,
        },
        {
            "event_id": "e_zero",
            "class": "BOUNCE",
            "frame": 40,
            "source_pts_s": 4.0,
            "teacher_confidence": 0.7,
            "needs_agreement_pass": True,
            "filter_decision": "pending_independent_agreement",
            "rally_source_start_frame": 0,
            "rally_source_end_frame_exclusive": 50,
        },
    ]
    row = {
        "source": "pbvision_teacher_predictions",
        "video": video_id,
        "source_video": video_id,
        "video_path": str(media),
        "media_present": True,
        "split": "train",
        "fps": 10.0,
        "source_start_frame": 0,
        "num_frames": 100,
        "event_counts": {"HIT": 2, "BOUNCE": 2, "background": 0},
        "inventory_event_count": 4,
        "events": events,
        "loss_validity_mask": [True, True, True],
        "unknown_frame_mask": unknown,
        "sample_weight": 0.0,
        "agreement_count": 0,
        "needs_agreement_pass": True,
        "training_eligible": False,
        "source_video_sha256": media_sha,
        "parent_identity": f"pbvision:{video_id}:sha256:{media_sha}",
        "source_lineage_key": hashlib.sha256(video_id.encode()).hexdigest(),
        "timebase_conversion": {
            "needs_pts_verify": False,
            "frame_times_sha256": frame_times_sha,
            "pts_media_binding": {
                "binding_schema_version": 1,
                "status": "sha256_bound",
                "source_video_sha256": media_sha,
                "media_path": str(media),
                "media_sha256_verified_from_file": True,
                "frame_times_path": str(frame_times),
                "frame_times_sha256": frame_times_sha,
                "frame_times_declares_media_sha256": True,
                "binding_sha256": binding_sha,
            },
        },
        "license_id": "synthetic_fixture",
        "license_posture": "pbvision_signed_full_usage",
    }
    manifest = {
        "schema_version": 2,
        "artifact_type": "event_head_pbvision_teacher_staging_dataset_manifest",
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        "ground_truth": False,
        "config": {"window_frames": 64},
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "permanent_compare_only_denylist": [
            "83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"
        ],
        "provenance": {
            "sources": [
                {
                    "video_id": video_id,
                    "compare_only": False,
                    "source_video_sha256": media_sha,
                },
                {
                    "video_id": "83gyqyc10y8f",
                    "compare_only": True,
                    "source_video_sha256": "1" * 64,
                },
                {
                    "video_id": "iottnc0h3ekn",
                    "compare_only": True,
                    "source_video_sha256": "2" * 64,
                },
                {
                    "video_id": "o4dee9dn0ccr",
                    "compare_only": True,
                    "source_video_sha256": "3" * 64,
                },
            ]
        },
        "rows": [row],
    }
    teacher = _write_json(tmp_path / "teacher_manifest.json", manifest)
    return teacher, {
        "media": {video_id: media},
        "frame_times": {video_id: frame_times},
        "audio": {video_id: audio},
        "ball": {video_id: ball},
    }


def _materialize(
    teacher: Path, paths: dict[str, dict[str, Path]], *, seed: int = 20260720
):
    return materialize_arms(
        teacher,
        media_paths=paths["media"],
        frame_times_paths=paths["frame_times"],
        audio_paths=paths["audio"],
        ball_paths=paths["ball"],
        seed=seed,
    )


def test_synthetic_agreement_weights_unknown_masks_and_sha_bindings(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, decisions, bindings = _materialize(teacher, paths)

    assert b_manifest["verified"] is c_manifest["verified"] is False
    assert b_manifest["ground_truth"] is c_manifest["ground_truth"] is False
    assert b_manifest["totals"] == {
        "rows": 2,
        "HIT": 1,
        "BOUNCE": 1,
        "sample_weight": 0.75,
    }
    weights = {item["event_id"]: item["pseudo_weight"] for item in decisions}
    assert weights == {"e0": 0.5, "e1": 0.25, "e_low": 0.0, "e_zero": 0.0}
    assert {row["sample_weight"] for row in b_manifest["rows"]} == {0.25, 0.5}
    for row in b_manifest["rows"]:
        assert row["split"] == "train"
        assert row["num_frames"] == len(row["unknown_frame_mask"]) == 64
        focal = row["events"][0]
        assert row["unknown_frame_mask"][focal["frame"]] is False
        assert row["agreement_count"] == len(focal["independent_agreements"])

    binding = bindings[0]
    assert binding["media"]["sha256"] == _sha256(paths["media"]["synthetic_train_clip"])
    assert binding["frame_times"]["sha256"] == _sha256(
        paths["frame_times"]["synthetic_train_clip"]
    )
    assert binding["audio_onsets"]["sha256"] == _sha256(
        paths["audio"]["synthetic_train_clip"]
    )
    assert binding["ball_velocity_kinks"]["sha256"] == _sha256(
        paths["ball"]["synthetic_train_clip"]
    )
    assert len(binding["audio_onsets"]["dependency_binding_sha256"]) == 64
    assert len(binding["ball_velocity_kinks"]["dependency_binding_sha256"]) == 64


def test_c_placebo_preserves_pixel_rows_classes_weights_and_agreement_metadata(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, _, _ = _materialize(teacher, paths, seed=77)

    assert c_manifest["placebo"]["seed"] == 77
    assert c_manifest["placebo"]["source_arm_b_manifest_sha256"] == hashlib.sha256(
        (json.dumps(b_manifest, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    for b_row, c_row in zip(b_manifest["rows"], c_manifest["rows"]):
        assert c_row["video_path"] == b_row["video_path"]
        assert c_row["source_start_frame"] == b_row["source_start_frame"]
        assert c_row["num_frames"] == b_row["num_frames"]
        assert c_row["sample_weight"] == b_row["sample_weight"]
        assert c_row["events"][0]["class"] == b_row["events"][0]["class"]
        assert c_row["events"][0]["independent_agreements"] == (
            b_row["events"][0]["independent_agreements"]
        )
        assert c_row["events"][0]["source_frame"] != b_row["events"][0]["source_frame"]
        assert sum(not value for value in c_row["unknown_frame_mask"]) == sum(
            not value for value in b_row["unknown_frame_mask"]
        )
        assert c_row["events"][0]["rally_source_start_frame"] <= (
            c_row["events"][0]["source_frame"]
        ) < c_row["events"][0]["rally_source_end_frame_exclusive"]


def test_parity_guard_rejects_reviewer_64_vs_63_mask_probe(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, _, _ = _materialize(teacher, paths, seed=77)
    b_row = b_manifest["rows"][0]
    c_row = c_manifest["rows"][0]
    b_row["unknown_frame_mask"] = [False] * 64
    c_row["unknown_frame_mask"] = [False] * 63 + [True]

    with pytest.raises(ABCMaterializationError, match="64 != 63"):
        _assert_b_c_parity([b_row], [c_row])


def test_materialization_and_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    first = _materialize(teacher, paths, seed=123)
    second = _materialize(teacher, paths, seed=123)
    assert first == second

    teacher_payload = json.loads(teacher.read_text())
    needs = build_vm_needs(
        teacher_payload,
        teacher_manifest_path=teacher,
        media_paths=paths["media"],
        frame_times_paths=paths["frame_times"],
        audio_paths=paths["audio"],
        ball_paths=paths["ball"],
    )
    out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
    hashes_a = write_materializations(
        out_a,
        needs=needs,
        b_manifest=first[0],
        c_manifest=first[1],
        decisions=first[2],
        input_bindings=first[3],
    )
    hashes_b = write_materializations(
        out_b,
        needs=needs,
        b_manifest=second[0],
        c_manifest=second[1],
        decisions=second[2],
        input_bindings=second[3],
    )
    assert hashes_a == hashes_b
    assert {path.name for path in out_a.iterdir()} == {
        "VM_ABC_NEEDS.json",
        "arm_b_manifest.json",
        "arm_c_manifest.json",
        "agreement_decisions.jsonl",
        "input_bindings.json",
    }
    for name in hashes_a:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_needs_only_cli_emits_per_train_clip_requirements_without_scoring(
    tmp_path: Path,
) -> None:
    teacher, _ = _fixture_inputs(tmp_path)
    output = tmp_path / "needs"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--teacher-manifest", str(teacher),
            "--output-dir", str(output),
            "--needs-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["needs_only"] is True
    assert summary["verified"] is False
    needs = json.loads((output / "VM_ABC_NEEDS.json").read_text())
    assert needs["no_scoring"] is True
    assert needs["required_train_clips"] == 1
    required = needs["clips"][0]["required_artifacts"]
    assert set(required) == {
        "media", "frame_times", "audio_onsets", "ball_velocity_kinks"
    }
    assert all(item["provided"] is False for item in required.values())
    assert not (output / "VM_ABC_RUN.md").exists()


def test_materializer_never_overwrites_existing_vm_runbook(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    output = tmp_path / "needs"
    output.mkdir()
    runbook = output / "VM_ABC_RUN.md"
    sentinel = b"operator-owned frozen runbook\n"
    runbook.write_bytes(sentinel)
    video_id = "synthetic_train_clip"

    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--teacher-manifest", str(teacher),
            "--output-dir", str(output),
            "--media", f"{video_id}={paths['media'][video_id]}",
            "--frame-times", f"{video_id}={paths['frame_times'][video_id]}",
            "--audio-onsets", f"{video_id}={paths['audio'][video_id]}",
            "--ball-velocity-kinks", f"{video_id}={paths['ball'][video_id]}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert runbook.read_bytes() == sentinel


def test_full_cli_materializes_b_and_c_from_explicit_artifact_paths(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    output = tmp_path / "cli_full"
    video_id = "synthetic_train_clip"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--teacher-manifest", str(teacher),
            "--output-dir", str(output),
            "--seed", "99",
            "--media", f"{video_id}={paths['media'][video_id]}",
            "--frame-times", f"{video_id}={paths['frame_times'][video_id]}",
            "--audio-onsets", f"{video_id}={paths['audio'][video_id]}",
            "--ball-velocity-kinks", f"{video_id}={paths['ball'][video_id]}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["arm_b_rows"] == summary["arm_c_rows"] == 2
    assert summary["verified"] is False
    assert summary["scoring_performed"] is False
    assert json.loads((output / "arm_b_manifest.json").read_text())["arm"] == "B"
    assert json.loads((output / "arm_c_manifest.json").read_text())["arm"] == "C"


def test_pts_binding_or_media_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    payload = json.loads(teacher.read_text())
    payload["rows"][0]["timebase_conversion"]["pts_media_binding"][
        "binding_sha256"
    ] = "0" * 64
    _write_json(teacher, payload)

    with pytest.raises(ABCMaterializationError, match="not SHA-bound"):
        _materialize(teacher, paths)

    teacher, paths = _fixture_inputs(tmp_path / "second")
    paths["media"]["synthetic_train_clip"].write_bytes(b"different-media")
    with pytest.raises(ABCMaterializationError, match="media SHA-256 mismatch"):
        _materialize(teacher, paths)

    teacher, paths = _fixture_inputs(tmp_path / "renamed_holdout")
    payload = json.loads(teacher.read_text())
    payload["rows"][0]["source_video_sha256"] = "1" * 64
    _write_json(teacher, payload)
    with pytest.raises(ABCMaterializationError, match="media SHA reached train rows"):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    ("artifact", "missing_field"),
    (("audio", "source_video_sha256"), ("ball", "frame_times_sha256")),
)
def test_agreement_artifacts_require_media_and_pts_hashes(
    tmp_path: Path, artifact: str, missing_field: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    artifact_path = paths[artifact]["synthetic_train_clip"]
    payload = json.loads(artifact_path.read_text())
    del payload[missing_field]
    _write_json(artifact_path, payload)

    with pytest.raises(ABCMaterializationError, match="unbound agreement inputs"):
        _materialize(teacher, paths)
