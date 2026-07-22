from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path

import pytest

from scripts.racketsport import build_abc_arm_manifests as abc_builder

from threed.racketsport.ball_inflections import (
    build_ball_inflections_from_ball_track_file,
)

from scripts.racketsport.build_abc_arm_manifests import (
    ABCMaterializationError,
    _assert_b_c_parity,
    _match_family,
    _validate_and_index_events,
    _validated_emitted_agreement,
    build_parser,
    build_vm_needs,
    materialize_arms,
    write_materializations,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/build_abc_arm_manifests.py"
PINNED_DECISIONS = (
    ROOT
    / "runs/lanes/abc_experiment_20260721/vm_pull/abc_out/agreement_decisions.jsonl"
)
PINNED_DECISIONS_SHA256 = (
    "3a3463565e57a5cd909eaad01f2ddf6fa66f23468396f7162a94c85f8b1bf4f1"
)


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
        "artifact_type": "racketsport_audio_onsets",
        "detector_version": "audio_onset_pop_v2",
        "source": "video_audio_pop_v2",
        "clip": video_id,
        "media_sha256": media_sha,
        "source_video_sha256": media_sha,
        "frame_times_sha256": frame_times_sha,
        "pts_source": {
            "path": str(frame_times),
            "sha256": frame_times_sha,
            "source_video_sha256": media_sha,
        },
        "onsets": [
            {
                "cue_id": "audio_e0",
                "corrected_time_s": 1.01,
                "source": "audio_pop_v2",
            },
            {
                "cue_id": "audio_e1",
                "corrected_time_s": 2.01,
                "source": "audio_pop_v2",
            },
        ],
    })
    track_points = {
        9: [0.0, 0.0],
        10: [10.0, 0.0],
        11: [10.0, 10.0],
        64: [100.0, 100.0],
        65: [110.0, 100.0],
        66: [110.0, 110.0],
    }
    ball_track = _write_json(tmp_path / "ball_track.json", {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track",
        "source": "wasb",
        "source_video_sha256": media_sha,
        "frame_times_sha256": frame_times_sha,
        "fps": 10.0,
        "frames": [
            (
                {
                    "frame": frame,
                    "visible": True,
                    "xy": track_points[frame],
                    "conf": 1.0,
                }
                if frame in track_points
                else {"frame": frame, "visible": False}
            )
            for frame in range(100)
        ],
    })
    ball = _write_json(
        tmp_path / "ball_inflections.json",
        build_ball_inflections_from_ball_track_file(
            ball_track, frame_times_path=frame_times
        ),
    )
    unknown = [False] * 100
    for frame in (10, 20, 30, 40, 65):
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
        {
            "event_id": "e_kink",
            "class": "HIT",
            "frame": 65,
            "source_pts_s": 6.5,
            "teacher_confidence": 0.85,
            "needs_agreement_pass": True,
            "filter_decision": "pending_independent_agreement",
            "rally_source_start_frame": 0,
            "rally_source_end_frame_exclusive": 100,
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
        "event_counts": {"HIT": 3, "BOUNCE": 2, "background": 0},
        "inventory_event_count": 5,
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
        "ball_track": {video_id: ball_track},
    }


def _rewrite_frame_times_and_rebind(
    teacher: Path,
    paths: dict[str, dict[str, Path]],
    frame_times_payload: dict[str, object],
) -> None:
    video_id = "synthetic_train_clip"
    frame_times_path = paths["frame_times"][video_id]
    _write_json(frame_times_path, frame_times_payload)
    frame_times_sha = _sha256(frame_times_path)

    ball_track_path = paths["ball_track"][video_id]
    ball_track = json.loads(ball_track_path.read_text())
    ball_track["frame_times_sha256"] = frame_times_sha
    _write_json(ball_track_path, ball_track)

    for artifact_name in ("audio", "ball"):
        artifact_path = paths[artifact_name][video_id]
        artifact = json.loads(artifact_path.read_text())
        artifact["frame_times_sha256"] = frame_times_sha
        artifact["pts_source"]["path"] = str(frame_times_path)
        artifact["pts_source"]["sha256"] = frame_times_sha
        if artifact_name == "ball":
            artifact["ball_track_source"]["sha256"] = _sha256(ball_track_path)
        _write_json(artifact_path, artifact)

    teacher_payload = json.loads(teacher.read_text())
    row = teacher_payload["rows"][0]
    row["timebase_conversion"]["frame_times_sha256"] = frame_times_sha
    binding = row["timebase_conversion"]["pts_media_binding"]
    binding["frame_times_sha256"] = frame_times_sha
    binding["binding_sha256"] = _canonical_sha({
        "source_video_sha256": row["source_video_sha256"],
        "frame_times_sha256": frame_times_sha,
    })
    _write_json(teacher, teacher_payload)


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


def _audio_null_block(
    manifest: dict[str, object], bindings: list[dict[str, object]]
) -> dict[str, object]:
    video_id = "synthetic_train_clip"
    block = manifest["metadata"]["audio_time_shift_null"][video_id]
    assert block == bindings[0]["audio_time_shift_null"]
    assert set(block) == {
        "eligible_event_count",
        "observed_match_count",
        "minimum_observed_match_count",
        "support_satisfied",
        "pts_origin_s",
        "circular_period_s",
        "observed_match_rate",
        "shift_offsets_s",
        "unique_shift_count",
        "null_match_rates",
        "null_max_rate",
        "beats_null",
    }
    assert len(block["shift_offsets_s"]) == 20
    assert len(set(block["shift_offsets_s"])) == 20
    assert block["unique_shift_count"] == 20
    assert block["minimum_observed_match_count"] == 2
    assert len(block["null_match_rates"]) == 20
    assert all(abs(offset) >= 1.0 for offset in block["shift_offsets_s"])
    assert all(0.0 <= rate <= 1.0 for rate in block["null_match_rates"])
    assert block["null_max_rate"] == max(block["null_match_rates"])
    assert isinstance(block["beats_null"], bool)
    return block


def test_synthetic_agreement_weights_unknown_masks_and_sha_bindings(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, decisions, bindings = _materialize(teacher, paths)

    assert b_manifest["verified"] is c_manifest["verified"] is False
    assert b_manifest["ground_truth"] is c_manifest["ground_truth"] is False
    assert b_manifest["totals"] == {
        "rows": 2,
        "HIT": 2,
        "BOUNCE": 0,
        "sample_weight": 0.75,
    }
    by_event = {item["event_id"]: item for item in decisions}
    assert {event_id: item["pseudo_weight"] for event_id, item in by_event.items()} == {
        "e0": 0.5,
        "e1": 0.0,
        "e_kink": 0.25,
        "e_low": 0.0,
        "e_zero": 0.0,
    }
    assert by_event["e1"]["accepted_into_arm_b"] is False
    assert by_event["e1"]["rejection_reason"] == "audio_only_no_physical_cue"
    assert by_event["e1"]["recorded_agreement_count"] == 1
    assert by_event["e1"]["agreement_count"] == 0
    assert by_event["e1"]["audio_weight_eligible"] is False
    assert by_event["e_kink"]["accepted_into_arm_b"] is True
    assert by_event["e_kink"]["recorded_agreement_count"] == 1
    assert by_event["e_kink"]["agreement_count"] == 1
    assert by_event["e_kink"]["audio_weight_eligible"] is False
    assert by_event["e0"]["accepted_into_arm_b"] is True
    assert by_event["e0"]["recorded_agreement_count"] == 2
    assert by_event["e0"]["agreement_count"] == 2
    assert by_event["e0"]["audio_weight_eligible"] is True
    assert all(isinstance(item["audio_weight_eligible"], bool) for item in decisions)
    for decision in decisions:
        for agreement in decision["independent_agreements"]:
            assert agreement["source_event_index"] == decision["source_event_index"]
            assert agreement["matched_event_source_pts_s"] == decision["source_pts_s"]
            assert agreement["absolute_delta_s"] == pytest.approx(
                abs(agreement["cue_time_s"] - decision["source_pts_s"])
            )
    assert {row["sample_weight"] for row in b_manifest["rows"]} == {0.25, 0.5}
    for row in b_manifest["rows"]:
        assert row["split"] == "train"
        assert row["num_frames"] == len(row["unknown_frame_mask"]) == 64
        focal = row["events"][0]
        assert row["unknown_frame_mask"][focal["frame"]] is False
        assert row["recorded_agreement_count"] == len(
            focal["independent_agreements"]
        )
        assert row["agreement_count"] <= row["recorded_agreement_count"]

    block = _audio_null_block(b_manifest, bindings)
    assert block["observed_match_rate"] == 0.5
    assert block["null_max_rate"] <= 0.25
    assert block["beats_null"] is True
    assert c_manifest["metadata"]["audio_time_shift_null"] == (
        b_manifest["metadata"]["audio_time_shift_null"]
    )

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
    assert binding["audio_onsets"]["artifact_contract"] == {
        "schema_version": 1,
        "artifact_type": "racketsport_audio_onsets",
        "detector_version": "audio_onset_pop_v2",
        "source": "video_audio_pop_v2",
    }
    assert binding["ball_velocity_kinks"]["artifact_contract"] == {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_inflections",
        "source": "ball_track_image_motion",
        "world_frame": "image_xy",
    }
    assert binding["ball_velocity_kinks"]["ball_track_source"]["path"] == str(
        paths["ball_track"]["synthetic_train_clip"]
    )
    assert binding["ball_velocity_kinks"]["ball_track_source"]["sha256"] == (
        _sha256(paths["ball_track"]["synthetic_train_clip"])
    )
    assert binding["ball_velocity_kinks"]["ball_track_source"][
        "declared_source"
    ] == "wasb"
    assert binding["ball_velocity_kinks"]["derivation_validation"]["status"] == (
        "exact_upstream_rebuild_match"
    )
    assert binding["frame_times"]["declared_media_sha256"] == (
        binding["frame_times"]["staged_media_sha256"]
    )
    assert binding["frame_times"]["duration_s"] == 10.0
    assert binding["frame_times"]["validated_terminal_frame_duration_s"] == (
        pytest.approx(0.1)
    )
    assert binding["frame_times"]["circular_period_s"] == pytest.approx(10.0)
    assert binding["frame_times"]["pts_origin_s"] == 0.0


def test_both_agreements_keep_audio_recorded_but_weight_inert_when_null_fails(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    audio_path = paths["audio"]["synthetic_train_clip"]
    audio = json.loads(audio_path.read_text())
    audio["onsets"] = [
        {
            "cue_id": f"dense_audio_{index:03d}",
            "corrected_time_s": round(0.01 + 0.05 * index, 9),
            "source": "audio_pop_v2",
        }
        for index in range(200)
    ]
    _write_json(audio_path, audio)

    b_manifest, c_manifest, decisions, bindings = _materialize(teacher, paths)
    block = _audio_null_block(b_manifest, bindings)
    assert block["observed_match_rate"] == 1.0
    assert block["null_max_rate"] == 1.0
    assert block["beats_null"] is False

    by_event = {item["event_id"]: item for item in decisions}
    both = by_event["e0"]
    assert both["accepted_into_arm_b"] is True
    assert {item["family"] for item in both["independent_agreements"]} == {
        "audio_onset",
        "ball_velocity_kink",
    }
    assert both["recorded_agreement_count"] == 2
    assert both["agreement_count"] == 1
    assert both["audio_weight_eligible"] is False
    assert both["pseudo_weight"] == 0.25

    b_row = next(row for row in b_manifest["rows"] if row["focal_event_id"] == "e0")
    c_row = next(row for row in c_manifest["rows"] if row["focal_event_id"] == "e0")
    assert b_row["recorded_agreement_count"] == 2
    assert b_row["agreement_count"] == 1
    assert b_row["sample_weight"] == c_row["sample_weight"] == 0.25
    assert len(b_manifest["rows"]) == len(c_manifest["rows"]) == 2


def test_c_placebo_preserves_pixel_rows_classes_weights_and_agreement_metadata(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, _, _ = _materialize(teacher, paths, seed=77)

    assert c_manifest["placebo"]["seed"] == 77
    assert c_manifest["placebo"]["source_arm_b_manifest_sha256"] == hashlib.sha256(
        (json.dumps(b_manifest, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert len(b_manifest["rows"]) == len(c_manifest["rows"]) == 2
    assert {row["focal_event_id"] for row in b_manifest["rows"]} == {
        "e0", "e_kink"
    }
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

    with pytest.raises(ABCMaterializationError, match="exact UNKNOWN-mask"):
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
        "materialization_complete.json",
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
    ("artifact", "missing_fields", "error"),
    (
        (
            "audio",
            ("source_video_sha256", "media_sha256"),
            "must declare source_video_sha256/media_sha256",
        ),
        ("ball", ("frame_times_sha256",), "frame_times_sha256"),
    ),
)
def test_agreement_artifacts_require_media_and_pts_hashes(
    tmp_path: Path, artifact: str, missing_fields: tuple[str, ...], error: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    artifact_path = paths[artifact]["synthetic_train_clip"]
    payload = json.loads(artifact_path.read_text())
    for missing_field in missing_fields:
        del payload[missing_field]
    _write_json(artifact_path, payload)

    with pytest.raises(ABCMaterializationError, match=error):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    "attack",
    (
        "exotic_artifact",
        "pbvision_derived",
        "audio_disguised_as_ball",
        "wrong_world_frame",
        "case_variant_family",
        "case_variant_candidate_family",
    ),
)
def test_ball_cli_slot_cannot_spoof_artifact_family(
    tmp_path: Path, attack: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    ball_path = paths["ball"]["synthetic_train_clip"]
    ball = json.loads(ball_path.read_text())
    if attack == "exotic_artifact":
        ball["artifact_type"] = "exotic_audio_disguised_as_ball"
    elif attack == "pbvision_derived":
        ball["source"] = "pbvision_teacher_predictions"
    elif attack == "audio_disguised_as_ball":
        ball = json.loads(paths["audio"]["synthetic_train_clip"].read_text())
    elif attack == "wrong_world_frame":
        ball["world_frame"] = "court_Z0"
    elif attack == "case_variant_family":
        ball["family"] = "AuDiO_OnSeT"
    else:
        ball["candidates"][0]["family"] = "AuDiO_OnSeT"
    _write_json(ball_path, ball)

    with pytest.raises(
        ABCMaterializationError,
        match=(
            "artifact contract|family|candidates do not match|"
            "audio-derived provenance"
        ),
    ):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_type", "racketsport_audio_onsets_v2"),
        ("detector_version", "audio_onset_pop_v1"),
        ("source", "VIDEO_AUDIO_POP_V2"),
    ),
)
def test_audio_contract_is_exact_and_case_sensitive(
    tmp_path: Path, field: str, value: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    audio_path = paths["audio"]["synthetic_train_clip"]
    audio = json.loads(audio_path.read_text())
    audio[field] = value
    _write_json(audio_path, audio)

    with pytest.raises(ABCMaterializationError, match="artifact contract mismatch"):
        _materialize(teacher, paths)


@pytest.mark.parametrize("level", ("artifact", "cue"))
def test_audio_family_case_variant_is_rejected(tmp_path: Path, level: str) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    audio_path = paths["audio"]["synthetic_train_clip"]
    audio = json.loads(audio_path.read_text())
    if level == "artifact":
        audio["family"] = "AuDiO_OnSeT"
    else:
        audio["onsets"][0]["family"] = "AuDiO_OnSeT"
    _write_json(audio_path, audio)

    with pytest.raises(
        ABCMaterializationError, match="declares incompatible family"
    ):
        _materialize(teacher, paths)


@pytest.mark.parametrize("artifact", ("frame_times", "audio", "ball"))
@pytest.mark.parametrize("schema_version", (True, 1.0))
def test_schema_v1_contract_rejects_boolean_and_float_aliases(
    tmp_path: Path, artifact: str, schema_version: object
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    artifact_path = paths[artifact]["synthetic_train_clip"]
    payload = json.loads(artifact_path.read_text())
    payload["schema_version"] = schema_version
    if artifact == "frame_times":
        _rewrite_frame_times_and_rebind(teacher, paths, payload)
    else:
        _write_json(artifact_path, payload)

    with pytest.raises(
        ABCMaterializationError,
        match="schema-v1 racketsport_frame_times|artifact contract mismatch",
    ):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("missing_ball_track", "ball_track_source provenance"),
        ("wrong_ball_track_sha", "ball_track_source SHA mismatch"),
        ("missing_ball_pts", "lacks pts_source identity"),
        ("wrong_ball_pts_path", "pts_source.path"),
        ("wrong_ball_pts_sha", "pts_source SHA mismatch"),
        ("wrong_ball_pts_media", "pts_source media SHA mismatch"),
        ("missing_audio_pts", "lacks pts_source identity"),
        ("wrong_candidate_source", "pb.vision-derived provenance"),
        ("missing_candidate_xy", "candidates do not match"),
    ),
)
def test_agreement_upstream_and_candidate_provenance_is_authenticated(
    tmp_path: Path, attack: str, error: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    artifact_name = "audio" if attack == "missing_audio_pts" else "ball"
    artifact_path = paths[artifact_name]["synthetic_train_clip"]
    artifact = json.loads(artifact_path.read_text())
    if attack == "missing_ball_track":
        artifact.pop("ball_track_source")
    elif attack == "wrong_ball_track_sha":
        artifact["ball_track_source"]["sha256"] = "0" * 64
    elif attack in {"missing_ball_pts", "missing_audio_pts"}:
        artifact.pop("pts_source")
    elif attack == "wrong_ball_pts_path":
        artifact["pts_source"]["path"] = str(tmp_path / "other_frame_times.json")
    elif attack == "wrong_ball_pts_sha":
        artifact["pts_source"]["sha256"] = "0" * 64
    elif attack == "wrong_ball_pts_media":
        artifact["pts_source"]["source_video_sha256"] = "0" * 64
    elif attack == "wrong_candidate_source":
        artifact["candidates"][0]["source"] = "pbvision_teacher_predictions"
    else:
        artifact["candidates"][0].pop("ball_image_xy")
    _write_json(artifact_path, artifact)

    with pytest.raises(ABCMaterializationError, match=error):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("fabricated_candidate", "candidates do not match"),
        ("empty_authenticated_track", "derivation summary"),
        ("pbvision_upstream", "pb.vision-derived ball tracks"),
        ("exotic_upstream", "source must be exactly 'wasb'"),
        ("audio_upstream", "source must be exactly 'wasb'"),
        ("case_variant_wasb", "source must be exactly 'wasb'"),
        ("nested_pbvision_upstream", "pb.vision-derived ball tracks"),
        ("nested_audio_upstream", "audio-derived provenance"),
        ("nested_video_audio_upstream", "audio-derived provenance"),
        ("nested_exotic_upstream", "contradictory nested source provenance"),
        ("root_pbvision_provenance", "pb.vision-derived provenance"),
        ("root_audio_provenance", "audio-derived provenance"),
        ("binding_pbvision_provenance", "pb.vision-derived provenance"),
        ("structured_path_pbvision", "pb.vision-derived provenance"),
        ("structured_path_audio", "audio-derived provenance"),
        ("missing_upstream_media", "must declare source_video_sha256/media_sha256"),
        ("wrong_upstream_media", "media SHA mismatch"),
        ("missing_upstream_pts", "lowercase 64-character SHA-256"),
        ("wrong_upstream_pts", "frame-times SHA mismatch"),
        ("boolean_schema_upstream", "schema-v1 WASB track"),
        ("float_schema_upstream", "schema-v1 WASB track"),
    ),
)
def test_ball_agreement_must_rebuild_from_exact_wasb_upstream_track(
    tmp_path: Path, attack: str, error: str
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    video_id = "synthetic_train_clip"
    ball_path = paths["ball"][video_id]
    ball_track_path = paths["ball_track"][video_id]
    ball = json.loads(ball_path.read_text())
    if attack == "fabricated_candidate":
        ball["candidates"].append({
            "time_s": 2.0,
            "frame": 20,
            "ball_image_xy": [9_999.0, -9_999.0],
            "source": "ball_track_image_motion",
        })
    elif attack == "root_pbvision_provenance":
        ball["provenance"] = {"source": "pbvision_teacher_predictions"}
    elif attack == "root_audio_provenance":
        ball["provenance"] = {"source": "video_audio_pop_v2"}
    elif attack == "binding_pbvision_provenance":
        ball["ball_track_source"]["source"] = "pbvision_teacher_predictions"
    elif attack == "structured_path_pbvision":
        ball["provenance"] = {
            "path": {"source": "pbvision_teacher_predictions"},
        }
    elif attack == "structured_path_audio":
        ball["provenance"] = {
            "path": {"source": "video_audio_pop_v2"},
        }
    else:
        ball_track = json.loads(ball_track_path.read_text())
        if attack == "empty_authenticated_track":
            ball_track["frames"] = []
        elif attack == "pbvision_upstream":
            ball_track["source"] = "pbvision_teacher_predictions"
        elif attack == "exotic_upstream":
            ball_track["source"] = "exotic_audio_disguised_as_ball"
        elif attack == "audio_upstream":
            ball_track["source"] = "audio_onsets_v2"
        elif attack == "case_variant_wasb":
            ball_track["source"] = "WASB"
        elif attack == "nested_pbvision_upstream":
            ball_track["provenance"] = {
                "source": "pbvision_teacher_predictions",
            }
        elif attack == "nested_audio_upstream":
            ball_track["provenance"] = {"source": "audio_pop_v2"}
        elif attack == "nested_video_audio_upstream":
            ball_track["provenance"] = {"source": "video_audio_pop_v2"}
        elif attack == "nested_exotic_upstream":
            ball_track["provenance"] = {
                "source": "exotic_sensor_disguised_as_wasb",
            }
        elif attack == "missing_upstream_media":
            ball_track.pop("source_video_sha256")
        elif attack == "wrong_upstream_media":
            ball_track["source_video_sha256"] = "0" * 64
        elif attack == "missing_upstream_pts":
            ball_track.pop("frame_times_sha256")
        elif attack == "wrong_upstream_pts":
            ball_track["frame_times_sha256"] = "0" * 64
        elif attack == "boolean_schema_upstream":
            ball_track["schema_version"] = True
        elif attack == "float_schema_upstream":
            ball_track["schema_version"] = 1.0
        else:
            raise AssertionError(f"unhandled attack fixture: {attack}")
        _write_json(ball_track_path, ball_track)
        ball["ball_track_source"]["sha256"] = _sha256(ball_track_path)
    _write_json(ball_path, ball)

    with pytest.raises(ABCMaterializationError, match=error):
        _materialize(teacher, paths)


@pytest.mark.parametrize(
    ("swap_target", "error"),
    (
        ("ball_track", "ball_track_source changed during materialization"),
        ("frame_times", "frame-times artifact changed during materialization"),
    ),
)
def test_ball_derivation_uses_verified_snapshots_and_refuses_path_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
    error: str,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    video_id = "synthetic_train_clip"
    target_path = paths[swap_target][video_id]
    original_builder = abc_builder.build_ball_inflections_from_ball_track_file
    builder_calls = 0

    def swap_original_then_build(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal builder_calls
        builder_calls += 1
        swapped = json.loads(target_path.read_text())
        swapped["concurrent_swap_marker"] = "different_authenticated_bytes"
        _write_json(target_path, swapped)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        abc_builder,
        "build_ball_inflections_from_ball_track_file",
        swap_original_then_build,
    )
    with pytest.raises(ABCMaterializationError, match=error):
        _materialize(teacher, paths)
    assert builder_calls == 1


def test_frame_times_requires_declared_staged_media_identity(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times.pop("source_video_sha256")
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)

    with pytest.raises(ABCMaterializationError, match="must declare staged media SHA"):
        _materialize(teacher, paths)


def test_frame_times_rejects_conflicting_media_identity(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["media_sha256"] = "0" * 64
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)

    with pytest.raises(ABCMaterializationError, match="wrong media SHA"):
        _materialize(teacher, paths)


def test_frame_times_requires_exact_artifact_type_and_frame_count(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["frame_count"] = 101
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)

    with pytest.raises(ABCMaterializationError, match="count mismatch"):
        _materialize(teacher, paths)

    teacher, paths = _fixture_inputs(tmp_path / "artifact_type")
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["artifact_type"] = "exotic_frame_times"
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)
    with pytest.raises(ABCMaterializationError, match="racketsport_frame_times"):
        _materialize(teacher, paths)


def test_inflated_declared_duration_cannot_game_audio_null(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["duration_s"] = 1_000.0
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)

    with pytest.raises(ABCMaterializationError, match="duration is inconsistent"):
        _materialize(teacher, paths)


@pytest.mark.parametrize("declared_duration", (True, "10.0"))
def test_frame_duration_rejects_json_type_coercion(
    tmp_path: Path, declared_duration: object
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["duration_s"] = declared_duration
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)

    with pytest.raises(ABCMaterializationError, match="duration_s must be numeric"):
        _materialize(teacher, paths)


def test_audio_null_period_is_pts_derived_not_declared_duration(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path / "baseline")
    baseline_manifest, _, _, baseline_bindings = _materialize(teacher, paths)
    baseline_null = _audio_null_block(baseline_manifest, baseline_bindings)

    teacher, paths = _fixture_inputs(tmp_path / "rounding_only")
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["duration_s"] = 10.00005
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)
    rounded_manifest, _, _, rounded_bindings = _materialize(teacher, paths)
    rounded_null = _audio_null_block(rounded_manifest, rounded_bindings)
    assert rounded_null == baseline_null
    assert rounded_bindings[0]["frame_times"]["declared_duration_s"] == 10.00005
    assert rounded_bindings[0]["frame_times"]["circular_period_s"] == (
        pytest.approx(10.0)
    )

    teacher, paths = _fixture_inputs(tmp_path / "reviewer_repro")
    frame_times_path = paths["frame_times"]["synthetic_train_clip"]
    frame_times = json.loads(frame_times_path.read_text())
    frame_times["duration_s"] = 9.96
    _rewrite_frame_times_and_rebind(teacher, paths, frame_times)
    with pytest.raises(ABCMaterializationError, match="duration is inconsistent"):
        _materialize(teacher, paths)


def test_duplicate_and_string_coercion_event_ids_fail_before_matching(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    payload = json.loads(teacher.read_text())
    duplicate = dict(payload["rows"][0]["events"][0])
    duplicate.update({"frame": 80, "source_pts_s": 8.0})
    payload["rows"][0]["events"].append(duplicate)
    _write_json(teacher, payload)
    with pytest.raises(ABCMaterializationError, match="duplicate event_id"):
        _materialize(teacher, paths)

    teacher, paths = _fixture_inputs(tmp_path / "coercion")
    payload = json.loads(teacher.read_text())
    payload["rows"][0]["events"][0]["event_id"] = 1
    payload["rows"][0]["events"][1]["event_id"] = "1"
    _write_json(teacher, payload)
    with pytest.raises(ABCMaterializationError, match="string-coercion event_id collision"):
        _materialize(teacher, paths)

    teacher, paths = _fixture_inputs(tmp_path / "empty")
    payload = json.loads(teacher.read_text())
    payload["rows"][0]["events"][0]["event_id"] = ""
    _write_json(teacher, payload)
    with pytest.raises(ABCMaterializationError, match="nonempty string"):
        _materialize(teacher, paths)


def test_internal_event_index_prevents_distant_alias_lookup() -> None:
    events = _validate_and_index_events(
        [
            {"event_id": "near", "source_pts_s": 1.0},
            {"event_id": "distant", "source_pts_s": 8.0},
        ],
        video_id="alias_probe",
    )
    matches = _match_family(
        events,
        [{
            "stable_id": "one_kink",
            "time_s": 1.0,
            "source_index": 0,
            "cue_provenance": {
                "source": "ball_track_image_motion",
                "world_frame": "image_xy",
            },
        }],
        family="ball_velocity_kink",
        max_delta_s=0.035,
    )
    assert set(matches) == {0}
    assert 1 not in matches


def test_emitted_agreement_delta_is_rechecked_against_current_event() -> None:
    with pytest.raises(ABCMaterializationError, match="no longer belongs"):
        _validated_emitted_agreement(
            {
                "family": "ball_velocity_kink",
                "cue_stable_id": "aliased",
                "cue_time_s": 1.0,
                "absolute_delta_s": 0.0,
                "cue_provenance": {},
            },
            event_index=1,
            event={"event_id": "distant", "source_pts_s": 8.0},
            max_delta_s=0.035,
        )


def test_singleton_audio_match_is_explicitly_weight_ineligible(
    tmp_path: Path,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    audio_path = paths["audio"]["synthetic_train_clip"]
    audio = json.loads(audio_path.read_text())
    audio["onsets"] = audio["onsets"][:1]
    _write_json(audio_path, audio)

    b_manifest, _, decisions, bindings = _materialize(teacher, paths)
    block = _audio_null_block(b_manifest, bindings)
    assert block["observed_match_count"] == 1
    assert block["support_satisfied"] is False
    assert block["beats_null"] is False
    e0 = next(item for item in decisions if item["event_id"] == "e0")
    assert e0["audio_weight_eligible"] is False
    assert e0["pseudo_weight"] == 0.25


def test_c_parity_rejects_non_placebo_field_drift(tmp_path: Path) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    b_manifest, c_manifest, _, _ = _materialize(teacher, paths)
    c_manifest["rows"][0]["license_id"] = "tampered"

    with pytest.raises(ABCMaterializationError, match="non-placebo field"):
        _assert_b_c_parity(b_manifest["rows"], c_manifest["rows"])


def test_cli_flag_contract_is_unchanged() -> None:
    flags = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option != "-h" and option != "--help"
    }
    assert flags == {
        "--teacher-manifest",
        "--output-dir",
        "--seed",
        "--max-delta-s",
        "--needs-only",
        "--media",
        "--frame-times",
        "--audio-onsets",
        "--ball-velocity-kinks",
    }


def test_publication_requires_fresh_managed_output_and_completion_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    materialized = _materialize(teacher, paths)
    needs = build_vm_needs(
        json.loads(teacher.read_text()),
        teacher_manifest_path=teacher,
        media_paths=paths["media"],
        frame_times_paths=paths["frame_times"],
        audio_paths=paths["audio"],
        ball_paths=paths["ball"],
    )
    output = tmp_path / "publication"
    replace_destinations: list[str] = []
    original_replace = abc_builder.os.replace

    def recording_replace(source: object, destination: object) -> None:
        replace_destinations.append(Path(destination).name)
        original_replace(source, destination)

    monkeypatch.setattr(abc_builder.os, "replace", recording_replace)
    hashes = write_materializations(
        output,
        needs=needs,
        b_manifest=materialized[0],
        c_manifest=materialized[1],
        decisions=materialized[2],
        input_bindings=materialized[3],
    )
    completion = json.loads((output / "materialization_complete.json").read_text())
    assert completion["complete"] is True
    assert completion["mode"] == "full"
    assert completion["artifact_sha256"] == {
        name: digest
        for name, digest in hashes.items()
        if name != "materialization_complete.json"
    }
    assert replace_destinations[-1] == "materialization_complete.json"
    assert replace_destinations.count("materialization_complete.json") == 1
    with pytest.raises(ABCMaterializationError, match="fresh output directory"):
        write_materializations(output, needs=needs)


def test_concurrent_publication_refuses_second_writer_and_keeps_hashes_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher, paths = _fixture_inputs(tmp_path)
    needs = build_vm_needs(
        json.loads(teacher.read_text()),
        teacher_manifest_path=teacher,
        media_paths=paths["media"],
        frame_times_paths=paths["frame_times"],
        audio_paths=paths["audio"],
        ball_paths=paths["ball"],
    )
    output = tmp_path / "concurrent_publication"
    first_writer_entered_replace = threading.Event()
    release_first_writer = threading.Event()
    original_replace = abc_builder.os.replace

    def gated_replace(source: object, destination: object) -> None:
        if (
            threading.current_thread().name == "abc-writer-a"
            and Path(destination).name == "VM_ABC_NEEDS.json"
            and not first_writer_entered_replace.is_set()
        ):
            first_writer_entered_replace.set()
            if not release_first_writer.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release first ABC writer")
        original_replace(source, destination)

    monkeypatch.setattr(abc_builder.os, "replace", gated_replace)
    results: list[dict[str, str]] = []
    errors: list[BaseException] = []

    def first_writer() -> None:
        try:
            results.append(write_materializations(output, needs=needs))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=first_writer, name="abc-writer-a")
    writer.start()
    entered = first_writer_entered_replace.wait(timeout=5.0)
    try:
        assert entered, "first writer never reached publication replace"
        with pytest.raises(ABCMaterializationError, match="already in progress"):
            write_materializations(output, needs=needs)
    finally:
        release_first_writer.set()
        writer.join(timeout=5.0)

    assert not writer.is_alive()
    assert errors == []
    assert len(results) == 1
    assert not (output / ".abc_materialization.lock").exists()
    assert {
        name: _sha256(output / name)
        for name in results[0]
    } == results[0]


def test_pulled_real_decisions_recount_requires_physical_cue() -> None:
    if not PINNED_DECISIONS.is_file():
        pytest.fail("pinned VM decision evidence is required for ABC preflight")
    assert _sha256(PINNED_DECISIONS) == PINNED_DECISIONS_SHA256
    decisions = [
        json.loads(line)
        for line in PINNED_DECISIONS.read_text().splitlines()
        if line.strip()
    ]
    assert len(decisions) == 2_192

    def families(item: dict[str, object]) -> set[str]:
        return {
            agreement["family"]
            for agreement in item["independent_agreements"]
        }

    corrected_eligible = sum(
        "ball_velocity_kink" in families(item) for item in decisions
    )
    kink_only = sum(
        item["accepted_into_arm_b"] is True
        and families(item) == {"ball_velocity_kink"}
        for item in decisions
    )
    both = sum(
        item["accepted_into_arm_b"] is True
        and families(item) == {"audio_onset", "ball_velocity_kink"}
        for item in decisions
    )
    old_audio_only = sum(
        item["accepted_into_arm_b"] is True
        and families(item) == {"audio_onset"}
        for item in decisions
    )
    kink_bearing_old_rejected = sum(
        "ball_velocity_kink" in families(item)
        and item["accepted_into_arm_b"] is not True
        for item in decisions
    )
    assert corrected_eligible == 1_189
    assert kink_only == 773
    assert both == 416
    assert old_audio_only == 292
    assert kink_bearing_old_rejected == 0
