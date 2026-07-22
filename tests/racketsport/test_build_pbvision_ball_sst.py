from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


CLI = "scripts/racketsport/build_pbvision_ball_sst.py"


def _observation(frame: int, xy: tuple[float, float], confidence: float = 0.95):
    from scripts.racketsport.build_pbvision_ball_sst import TeacherObservation

    return TeacherObservation(
        teacher_frame_index=frame,
        teacher_time_s=frame / 30.0,
        xy_px=xy,
        confidence=confidence,
    )


def test_eligibility_agree_disagree_low_conf_and_absent_teacher() -> None:
    from scripts.racketsport.build_pbvision_ball_sst import (
        WasbObservation,
        eligibility_decision,
    )

    current = _observation(10, (100.0, 80.0))
    agreed = eligibility_decision(
        current,
        teacher_observations={10: current},
        wasb=WasbObservation(10, (112.0, 84.0), 0.96, True),
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )
    assert agreed[0] is True and agreed[1] == "frozen_wasb_spatial"

    disagreed = eligibility_decision(
        current,
        teacher_observations={10: current},
        wasb=WasbObservation(10, (200.0, 180.0), 0.99, True),
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )
    assert disagreed[0] is False
    assert disagreed[2]["rejection"] == "high_confidence_wasb_disagreement"

    low = _observation(10, (100.0, 80.0), confidence=0.899)
    low_result = eligibility_decision(
        low,
        teacher_observations={10: low},
        wasb=WasbObservation(10, (100.0, 80.0), 0.99, True),
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )
    assert low_result == (False, None, {"rejection": "teacher_low_confidence"})

    absent = eligibility_decision(
        None,
        teacher_observations={},
        wasb=WasbObservation(10, (100.0, 80.0), 0.99, True),
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )
    assert absent == (False, None, {"rejection": "teacher_absent_ignored_not_negative"})

    malformed_teacher = _observation(10, (100.0, 80.0), confidence=1.01)
    assert eligibility_decision(
        malformed_teacher,
        teacher_observations={10: malformed_teacher},
        wasb=WasbObservation(10, (100.0, 80.0), 0.99, True),
        width=640,
        height=360,
    ) == (False, None, {"rejection": "teacher_confidence_not_probability"})

    assert eligibility_decision(
        current,
        teacher_observations={10: current},
        wasb=WasbObservation(10, (100.0, 80.0), 1.01, True),
        width=640,
        height=360,
    ) == (False, None, {"rejection": "wasb_confidence_not_probability"})


def test_same_teacher_temporal_smoothness_is_never_independent() -> None:
    from scripts.racketsport.build_pbvision_ball_sst import eligibility_decision

    observations = {
        9: _observation(9, (90.0, 80.0)),
        10: _observation(10, (100.0, 80.0)),
        11: _observation(11, (110.0, 80.0)),
    }
    result = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )

    assert result == (False, None, {"rejection": "no_independent_wasb_agreement"})


def test_temporal_bridge_requires_two_preregistered_frozen_wasb_anchors() -> None:
    from scripts.racketsport.build_pbvision_ball_sst import WasbObservation, eligibility_decision

    observations = {
        9: _observation(9, (90.0, 80.0)),
        10: _observation(10, (100.0, 80.0)),
        11: _observation(11, (110.0, 80.0)),
    }
    wasb = {
        9: WasbObservation(9, (91.0, 80.0), 0.97, True),
        11: WasbObservation(11, (109.0, 80.0), 0.98, True),
    }
    result = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations=wasb,
    )

    assert result[0] is True
    assert result[1] == "frozen_wasb_temporal_bridge_v2"
    assert result[2]["prior_anchor"]["source_frame_index"] == 9
    assert result[2]["following_anchor"]["source_frame_index"] == 11
    assert result[2]["interpolation_residual_px"] == 0.0
    assert result[2]["current_wasb"] == {"status": "absent", "present": False}

    missing_following = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations={9: wasb[9]},
    )
    assert missing_following[0] is False


@pytest.mark.parametrize(
    ("following_anchor_frame", "accepted"),
    [
        (11, True),  # anchors 8/11 -> interior 9-10 -> total teacher-only gap 2
        (12, False),  # reviewer exact: anchors 8/12 -> interior 9-11 -> total gap 3
    ],
)
def test_temporal_bridge_bounds_total_teacher_only_gap_not_each_side(
    following_anchor_frame: int,
    accepted: bool,
) -> None:
    from scripts.racketsport.build_pbvision_ball_sst import WasbObservation, eligibility_decision

    observations = {
        frame: _observation(frame, (float(frame * 10), 80.0))
        for frame in range(8, following_anchor_frame + 1)
    }
    wasb = {
        8: WasbObservation(8, observations[8].xy_px, 0.99, True),
        following_anchor_frame: WasbObservation(
            following_anchor_frame,
            observations[following_anchor_frame].xy_px,
            0.99,
            True,
        ),
    }

    result = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations=wasb,
    )

    assert result[0] is accepted
    if accepted:
        assert result[2]["teacher_only_gap_length_source_frames"] == 2
        assert [row["source_frame_index"] for row in result[2]["intermediate_frames"]] == [9, 10]
    else:
        assert result == (False, None, {"rejection": "no_independent_wasb_agreement"})


@pytest.mark.parametrize(
    ("prior_anchor_frame", "following_anchor_frame", "contradictory_frame"),
    [
        (8, 11, 9),
        (9, 12, 11),
    ],
)
def test_temporal_search_cannot_skip_outward_past_contradictory_wasb(
    prior_anchor_frame: int,
    following_anchor_frame: int,
    contradictory_frame: int,
) -> None:
    from scripts.racketsport.build_pbvision_ball_sst import WasbObservation, eligibility_decision

    observations = {
        frame: _observation(frame, (float(frame * 10), 80.0))
        for frame in range(prior_anchor_frame, following_anchor_frame + 1)
    }
    wasb = {
        prior_anchor_frame: WasbObservation(
            prior_anchor_frame, observations[prior_anchor_frame].xy_px, 0.99, True
        ),
        following_anchor_frame: WasbObservation(
            following_anchor_frame, observations[following_anchor_frame].xy_px, 0.99, True
        ),
        contradictory_frame: WasbObservation(contradictory_frame, (400.0, 250.0), 0.99, True),
    }

    result = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations=wasb,
    )

    assert result == (False, None, {"rejection": "no_independent_wasb_agreement"})


def test_temporal_evidence_binds_sample_frame_dimensions_radius_and_current_wasb() -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    observations = {
        9: _observation(9, (90.0, 80.0)),
        10: _observation(10, (100.0, 80.0)),
        11: _observation(11, (110.0, 80.0)),
    }
    wasb = {
        9: builder.WasbObservation(9, (90.0, 80.0), 0.99, True),
        11: builder.WasbObservation(11, (110.0, 80.0), 0.99, True),
    }
    accepted, reason, evidence = builder.eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=None,
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations=wasb,
    )
    assert accepted is True
    builder._validate_agreement_evidence(
        sample_id="143sf3gdwxsa:10",
        reason=reason,
        evidence=evidence,
        teacher_xy=(100.0, 80.0),
        score=0.95,
        frame_index=10,
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
    )

    wrong_frame = copy.deepcopy(evidence)
    wrong_frame["current_source_frame_index"] = 100
    with pytest.raises(builder.BallSstBuildError, match="temporal current frame mismatch"):
        builder._validate_agreement_evidence(
            sample_id="143sf3gdwxsa:10",
            reason=reason,
            evidence=wrong_frame,
            teacher_xy=(100.0, 80.0),
            score=0.95,
            frame_index=10,
            width=640,
            height=360,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
        )

    wrong_radius = copy.deepcopy(evidence)
    wrong_radius["anchor_agreement_radius_px"] = 999.0
    with pytest.raises(builder.BallSstBuildError, match="temporal anchor radius mismatch"):
        builder._validate_agreement_evidence(
            sample_id="143sf3gdwxsa:10",
            reason=reason,
            evidence=wrong_radius,
            teacher_xy=(100.0, 80.0),
            score=0.95,
            frame_index=10,
            width=640,
            height=360,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
        )

    fabricated_gap = copy.deepcopy(evidence)
    fabricated_gap["current_wasb"] = {"status": "absent", "present": True}
    with pytest.raises(builder.BallSstBuildError, match="absent current WASB is malformed"):
        builder._validate_agreement_evidence(
            sample_id="143sf3gdwxsa:10",
            reason=reason,
            evidence=fabricated_gap,
            teacher_xy=(100.0, 80.0),
            score=0.95,
            frame_index=10,
            width=640,
            height=360,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
        )

    with pytest.raises(builder.BallSstBuildError, match="image dimensions mismatch"):
        builder._validate_agreement_evidence(
            sample_id="143sf3gdwxsa:10",
            reason=reason,
            evidence=evidence,
            teacher_xy=(100.0, 80.0),
            score=0.95,
            frame_index=10,
            width=320,
            height=180,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
        )


def test_all_wasb_absent_cannot_create_rows_across_frozen_sources(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    observations = {
        frame: _observation(frame, (100.0 + frame, 80.0))
        for frame in range(1, 8)
    }
    pts = tuple(index / 30.0 for index in range(12))
    for video_id in builder.TRAIN_IDS:
        rows = builder.build_source_samples(
            video_id=video_id,
            video_path=tmp_path / "media" / video_id / "max.mp4",
            teacher_observations=observations,
            wasb_observations={},
            pts_s=pts,
            width=640,
            height=360,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
            pseudo_weight=0.25,
            dependency_hashes={
                "source_video_sha256": builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id]
            },
        )
        assert rows == []


@pytest.mark.parametrize(
    ("teacher_xy", "wasb_xy", "rejection"),
    [
        ((-1.0, 80.0), (0.0, 80.0), "teacher_out_of_image_bounds"),
        ((100.0, 80.0), (640.0, 80.0), "wasb_out_of_image_bounds"),
    ],
)
def test_spatial_agreement_refuses_out_of_bounds_coordinates(
    teacher_xy: tuple[float, float],
    wasb_xy: tuple[float, float],
    rejection: str,
) -> None:
    from scripts.racketsport.build_pbvision_ball_sst import WasbObservation, eligibility_decision

    current = _observation(10, teacher_xy)
    result = eligibility_decision(
        current,
        teacher_observations={10: current},
        wasb=WasbObservation(10, wasb_xy, 0.99, True),
        width=640,
        height=360,
    )
    assert result == (False, None, {"rejection": rejection})


def test_high_confidence_current_wasb_disagreement_cannot_use_temporal_bridge() -> None:
    from scripts.racketsport.build_pbvision_ball_sst import WasbObservation, eligibility_decision

    observations = {
        9: _observation(9, (90.0, 80.0)),
        10: _observation(10, (100.0, 80.0)),
        11: _observation(11, (110.0, 80.0)),
    }
    wasb = {
        9: WasbObservation(9, (90.0, 80.0), 0.99, True),
        10: WasbObservation(10, (300.0, 200.0), 0.99, True),
        11: WasbObservation(11, (110.0, 80.0), 0.99, True),
    }
    result = eligibility_decision(
        observations[10],
        teacher_observations=observations,
        wasb=wasb[10],
        width=640,
        height=360,
        source_frame_index=10,
        teacher_by_source_frame=observations,
        wasb_observations=wasb,
    )
    assert result == (False, None, {"rejection": "high_confidence_wasb_disagreement"})


def test_compare_only_id_refuses_before_any_source_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    reads: list[Path] = []

    def forbidden_read(self: Path, *args, **kwargs):  # noqa: ANN002, ANN003
        reads.append(self)
        raise AssertionError("source read should be unreachable")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(builder.BallSstBuildError, match="structurally unreadable"):
        builder._read_source_json(tmp_path, "83gyqyc10y8f", "cv_export.json")
    assert reads == []


def test_missing_media_cli_emits_machine_readable_refusal(tmp_path: Path) -> None:
    split = Path("runs/lanes/pbv_pickleball_corpus_20260720/manifest.json")
    out = tmp_path / "refusal.json"

    completed = subprocess.run(
        [
            sys.executable,
            CLI,
            "--gallery-root",
            "data/pbvision_gallery_20260719",
            "--media-root",
            "data/pbv_replay_20260720",
            "--split-manifest",
            str(split),
            "--wasb-checkpoint",
            "models/checkpoints/wasb/wasb_tennis_best.pth.tar",
            "--teacher-confidence-min",
            "0.90",
            "--agreement-radius-px",
            "20",
            "--pseudo-weight",
            "0.25",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 3
    cli_payload = json.loads(completed.stdout)
    refusal = json.loads(out.read_text(encoding="utf-8"))
    assert cli_payload["verdict"] == "MISSING_MEDIA"
    assert refusal["artifact_type"] == "racketsport_pbvision_ball_sst_build_refusal"
    assert refusal["accepted_windows"] == 0
    assert refusal["accepted_sources"] == 0
    assert refusal["holdout_rows_present"] == 0
    assert refusal["decode_status"] == "not_attempted"
    assert refusal["decode_failures"] is None
    assert refusal["missing_media_count"] == 6
    assert refusal["staged_media_count"] == 1
    assert [row["video_id"] for row in refusal["missing_media"]] == list(_train_ids()[:-1])
    assert "stage the 6 missing" in refusal["next"]
    assert "1 of seven already present" in refusal["next"]


def test_cli_policy_override_is_diagnostic_not_preregistration(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    builder_identity, wasb_identity, dependencies = _identity_context(builder)
    clips = _empty_clips(builder, tmp_path, dependencies)
    manifest = builder.assemble_sst_manifest(
        clips=clips,
        gallery_root=tmp_path / "gallery",
        media_root=tmp_path / "media",
        split_manifest=tmp_path / "split.json",
        split_manifest_sha256=dependencies["split_manifest_sha256"],
        wasb_checkpoint=Path(wasb_identity["checkpoint_path"]),
        wasb_checkpoint_sha256=builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
        wasb_identity=wasb_identity,
        builder_identity=builder_identity,
        teacher_confidence_min=0.10,
        agreement_radius_px=999.0,
        pseudo_weight=0.50,
        policy_overrides=[
            "teacher_confidence_min",
            "agreement_radius_px",
            "pseudo_weight",
        ],
        decode_failures=0,
    )

    assert manifest["production_eligible"] is False
    assert manifest["gate"]["verdict"] == "NON_PRODUCTION_MANIFEST"
    assert manifest["preregistration"]["teacher_confidence_min"] == 0.90
    assert manifest["preregistration"]["agreement_radius_px"] == 20.0
    assert manifest["preregistration"]["pseudo_weight"] == 0.25
    assert manifest["requested_parameters"] == {
        "teacher_confidence_min": 0.10,
        "agreement_radius_px": 999.0,
        "pseudo_weight": 0.50,
    }
    authority = manifest["preregistration"]["teacher_input_authority"]
    assert manifest["teacher_input_authority"] == authority
    assert authority["authority_id"] == builder.PRODUCTION_GALLERY_AUTHORITY_ID
    assert authority["canonical_gallery_relative_path"] == str(
        builder.FROZEN_GALLERY_RELATIVE_PATH
    )
    assert set(authority["expected_sha256_by_source"]) == set(builder.TRAIN_IDS)


def test_production_gallery_authority_pins_all_current_canonical_teacher_inputs() -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    canonical_gallery = (builder.ROOT / builder.FROZEN_GALLERY_RELATIVE_PATH).resolve(
        strict=True
    )
    assert set(builder.PRODUCTION_GALLERY_ARTIFACT_SHA256) == set(builder.TRAIN_IDS)
    for video_id in builder.TRAIN_IDS:
        expected = builder.PRODUCTION_GALLERY_ARTIFACT_SHA256[video_id]
        assert set(expected) == set(builder.PRODUCTION_GALLERY_ARTIFACT_FILENAMES)
        for filename in builder.PRODUCTION_GALLERY_ARTIFACT_FILENAMES:
            assert builder._sha256_file(canonical_gallery / video_id / filename) == expected[
                filename
            ]


def test_frozen_policy_with_zero_independent_rows_is_honest_insufficiency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    monkeypatch.setattr(
        builder,
        "_verify_production_artifacts",
        lambda **_kwargs: {
            "verified": True,
            "status": "passed",
            "reason": "unit fixture identity boundary",
            "verified_clip_count": 7,
            "verified_sample_count": 0,
        },
    )
    builder_identity, wasb_identity, dependencies = _identity_context(builder)
    manifest = builder.assemble_sst_manifest(
        clips=_empty_clips(builder, tmp_path, dependencies),
        gallery_root=tmp_path / "gallery",
        media_root=tmp_path / "media",
        split_manifest=tmp_path / "split.json",
        split_manifest_sha256=dependencies["split_manifest_sha256"],
        wasb_checkpoint=Path(wasb_identity["checkpoint_path"]),
        wasb_checkpoint_sha256=builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
        wasb_identity=wasb_identity,
        builder_identity=builder_identity,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        policy_overrides=[],
        decode_failures=0,
    )
    assert manifest["production_eligible"] is True
    assert manifest["gate"]["verdict"] == "PBV_BALL_INSUFFICIENT_AGREEMENT"
    assert manifest["accepted_windows"] == 0
    assert manifest["accepted_sources"] == 0


def test_production_identity_pins_checkpoint_path_sha_and_wasb_commit(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    identity = builder._resolve_wasb_identity(
        checkpoint_path=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        repo_path=Path("third_party/WASB-SBDT"),
        require_production_identity=True,
    )
    assert identity["checkpoint_sha256"] == builder.PRODUCTION_WASB_CHECKPOINT_SHA256
    assert identity["repo_commit"] == builder.PRODUCTION_WASB_REPO_COMMIT
    assert identity["production_identity_verified"] is True

    arbitrary = tmp_path / "arbitrary.pth.tar"
    arbitrary.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(builder.BallSstBuildError, match="production mode requires"):
        builder._resolve_wasb_identity(
            checkpoint_path=arbitrary,
            repo_path=Path("third_party/WASB-SBDT"),
            require_production_identity=True,
        )

    aliased = tmp_path / "same-bytes-through-alias.pth.tar"
    aliased.symlink_to(Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar").resolve())
    with pytest.raises(builder.BallSstBuildError, match="production mode requires"):
        builder._resolve_wasb_identity(
            checkpoint_path=aliased,
            repo_path=Path("third_party/WASB-SBDT"),
            require_production_identity=True,
        )


def test_production_artifact_verifier_refuses_reauthored_split(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    split = tmp_path / "manifest.json"
    split.write_text(json.dumps(_frozen_split_payload()), encoding="utf-8")
    identity = builder._resolve_wasb_identity(
        checkpoint_path=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        repo_path=Path("third_party/WASB-SBDT"),
        require_production_identity=True,
    )
    verification = builder._verify_production_artifacts(
        clips=[],
        gallery_root=Path("data/pbvision_gallery_20260719"),
        media_root=Path("data/pbv_replay_20260720"),
        split_manifest=split,
        split_manifest_sha256=builder._sha256_file(split),
        wasb_checkpoint=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        wasb_identity=identity,
        builder_identity=builder._builder_identity(),
    )
    assert verification["verified"] is False
    assert "split_manifest must be canonical" in verification["reason"]


def test_production_artifact_verifier_rederives_rows_from_hashed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    video_id = builder.TRAIN_IDS[0]
    gallery = tmp_path / "gallery"
    media_root = tmp_path / "media"
    source_dir = gallery / video_id
    source_dir.mkdir(parents=True)
    source_media = media_root / video_id / "max.mp4"
    source_media.parent.mkdir(parents=True)
    source_media.write_bytes(b"bound-media-fixture")
    monkeypatch.setitem(
        builder.EXPECTED_SOURCE_VIDEO_SHA256,
        video_id,
        builder._sha256_file(source_media),
    )
    monkeypatch.setattr(builder, "FROZEN_GALLERY_RELATIVE_PATH", gallery)

    cv_export = {
        "camera": {"fps": 30.0},
        "sessions": [
            {
                "rallies": [
                    {
                        "frame_index": 0,
                        "frames": [
                            {},
                            {
                                "actions": {
                                    "ball": {
                                        "u": 0.5,
                                        "v": 0.5,
                                        "confidence": 0.95,
                                    }
                                }
                            },
                            {},
                            {},
                            {},
                        ],
                    }
                ]
            }
        ],
    }
    metadata = {"metadata": {"width": 64, "height": 48, "fps": 30.0}}
    provenance = {
        "video_id": video_id,
        "source_video_url": f"https://storage.googleapis.com/pbv-pro/{video_id}/max.mp4",
    }
    cv_export_path = source_dir / "cv_export.json"
    metadata_path = source_dir / "api_get_metadata.json"
    provenance_path = source_dir / "video_provenance.json"
    builder._write_json(cv_export_path, cv_export)
    builder._write_json(metadata_path, metadata)
    builder._write_json(provenance_path, provenance)
    monkeypatch.setitem(
        builder.PRODUCTION_GALLERY_ARTIFACT_SHA256,
        video_id,
        {
            "cv_export.json": builder._sha256_file(cv_export_path),
            "api_get_metadata.json": builder._sha256_file(metadata_path),
            "video_provenance.json": builder._sha256_file(provenance_path),
        },
    )

    split = tmp_path / "split.json"
    builder._write_json(split, _frozen_split_payload())
    monkeypatch.setattr(builder, "FROZEN_SPLIT_RELATIVE_PATH", split)
    monkeypatch.setattr(builder, "FROZEN_SPLIT_SHA256", builder._sha256_file(split))

    timing = builder.MediaTiming(
        fps=30.0,
        duration_s=5.0 / 30.0,
        pts_s=tuple(index / 30.0 for index in range(5)),
        width=64,
        height=48,
    )
    monkeypatch.setattr(builder, "probe_media_pts", lambda *_args, **_kwargs: timing)
    dependency_dir = tmp_path / "dependencies" / video_id
    frame_times_path = dependency_dir / "frame_times.json"
    wasb_track_path = dependency_dir / "wasb_ball_track.json"
    wasb_metadata_path = dependency_dir / "wasb_ball_track_metadata.json"
    wasb_csv_path = dependency_dir / "wasb_predictions.csv"
    builder._write_json(
        frame_times_path,
        builder._frame_times_payload(
            timing,
            media_sha256=builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id],
        ),
    )
    retained_csv_bytes = (
        "Frame,Visibility,X,Y,Confidence\n"
        "0,0,0.0,0.0,0.0\n"
        "1,1,32.0,24.0,0.96\n"
        "2,0,0.0,0.0,0.0\n"
        "3,0,0.0,0.0,0.0\n"
        "4,0,0.0,0.0,0.0\n"
    ).encode("utf-8")
    wasb_csv_path.write_bytes(retained_csv_bytes)
    wasb_track = builder.wasb_csv_to_ball_track(
        wasb_csv_path,
        fps=timing.fps,
        frame_times=frame_times_path,
        visible_threshold=0.90,
        input_preprocessing="official",
    )
    builder._write_json(wasb_track_path, wasb_track)

    wasb_identity = builder._resolve_wasb_identity(
        checkpoint_path=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        repo_path=Path("third_party/WASB-SBDT"),
        require_production_identity=True,
    )
    builder_identity = builder._builder_identity()
    wasb_bindings = {
        "source_video_sha256": builder._sha256_file(source_media),
        "frame_times_sha256": builder._sha256_file(frame_times_path),
        "wasb_predictions_csv_sha256": builder._sha256_file(wasb_csv_path),
        "wasb_ball_track_sha256": builder._sha256_file(wasb_track_path),
        "wasb_checkpoint_sha256": wasb_identity["checkpoint_sha256"],
        "wasb_repo_commit": wasb_identity["repo_commit"],
        "wasb_adapter_code_sha256": builder_identity["wasb_adapter_code_sha256"],
    }
    wall_seconds = 1.0
    wasb_runtime = {
        "wasb_repo": wasb_identity["repo_path"],
        "wasb_repo_commit": wasb_identity["repo_commit"],
        "wasb_checkpoint": {
            "path": wasb_identity["checkpoint_path"],
            "sha256": wasb_identity["checkpoint_sha256"],
        },
        "video": str(source_media),
        "source_video_fps": 30.0,
        "source_video_frame_count": 5,
        "source_video_size": [64, 48],
        "processed_frame_count": 5,
        "processed_window_count": 3,
        "read_frame_count": 5,
        "video_range_seconds": None,
        "max_frames": None,
        "batch_size": 1,
        "device": "cpu",
        "input_preprocessing": "official",
        "non_promotable_measurement_mode": False,
        "wall_seconds": wall_seconds,
        "effective_fps": 5.0 / wall_seconds,
        "realtime_factor": (5.0 / wall_seconds) / 30.0,
    }
    wasb_metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_wasb_ball_run",
        "status": builder.STATUS_TESTED,
        "source_mode": "wasb_predict",
        "predictions_csv": str(wasb_csv_path),
        "out": str(wasb_track_path),
        "fps": 30.0,
        "frame_count": 5,
        "visible_frame_count": 1,
        "confidence_semantics": builder.WASB_CONFIDENCE_SEMANTICS,
        "visible_threshold": 0.90,
        "input_preprocessing": "official",
        "non_promotable_measurement_mode": False,
        "not_ground_truth": True,
        "official_repo_url": builder.WASB_REPO_URL,
        "official_model_zoo_url": builder.WASB_MODEL_ZOO_URL,
        "runtime": wasb_runtime,
        "builder_bindings": wasb_bindings,
    }
    builder._write_json(wasb_metadata_path, wasb_metadata)
    dependencies = {
        "split_manifest_sha256": builder._sha256_file(split),
        "pbvision_cv_export_sha256": builder._sha256_file(cv_export_path),
        "pbvision_metadata_sha256": builder._sha256_file(metadata_path),
        "pbvision_provenance_sha256": builder._sha256_file(provenance_path),
        "source_video_sha256": builder._sha256_file(source_media),
        "frame_times_sha256": builder._sha256_file(frame_times_path),
        "wasb_checkpoint_sha256": wasb_identity["checkpoint_sha256"],
        "wasb_repo_commit": wasb_identity["repo_commit"],
        "models_manifest_sha256": wasb_identity["models_manifest_sha256"],
        "builder_code_sha256": builder_identity["builder_code_sha256"],
        "wasb_adapter_code_sha256": builder_identity["wasb_adapter_code_sha256"],
        "wasb_ball_track_sha256": builder._sha256_file(wasb_track_path),
        "wasb_metadata_sha256": builder._sha256_file(wasb_metadata_path),
        "wasb_predictions_csv_sha256": builder._sha256_file(wasb_csv_path),
    }
    samples = builder.build_source_samples(
        video_id=video_id,
        video_path=source_media,
        teacher_observations=builder.extract_teacher_observations(
            cv_export,
            width=64,
            height=48,
            teacher_fps=30.0,
        ),
        wasb_observations=builder.extract_wasb_observations(
            wasb_track,
            pts_s=timing.pts_s,
            fps=timing.fps,
            width=timing.width,
            height=timing.height,
            visible_threshold=0.90,
        )[0],
        pts_s=timing.pts_s,
        width=64,
        height=48,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        dependency_hashes=dependencies,
    )
    assert len(samples) == 1
    clip = {
        "clip_id": video_id,
        "canonical_source_id": video_id,
        "split": "train",
        "teacher_derived": True,
        "ground_truth": False,
        "rally_video": str(source_media),
        "source_video_sha256": dependencies["source_video_sha256"],
        "source_width": 64,
        "source_height": 48,
        "fps": 30.0,
        "sample_count": 1,
        "samples": samples,
        "dependencies": {
            **dependencies,
            "frame_times_path": str(frame_times_path),
            "wasb_ball_track": str(wasb_track_path),
            "wasb_metadata_path": str(wasb_metadata_path),
            "wasb_predictions_csv_path": str(wasb_csv_path),
            "wasb_runtime": wasb_metadata,
        },
    }
    verifier_args = {
        "gallery_root": gallery,
        "media_root": media_root,
        "split_manifest": split,
        "split_manifest_sha256": dependencies["split_manifest_sha256"],
        "wasb_checkpoint": Path(wasb_identity["checkpoint_path"]),
        "wasb_identity": wasb_identity,
        "builder_identity": builder_identity,
    }
    replay_state = {"matches_retained": True}
    replay_calls: list[dict[str, object]] = []

    def fake_official_replay(**kwargs: object) -> dict[str, object]:
        replay_calls.append(dict(kwargs))
        assert Path(kwargs["video"]).resolve() == source_media.resolve()
        assert Path(kwargs["checkpoint"]).resolve() == Path(
            wasb_identity["checkpoint_path"]
        ).resolve()
        assert Path(kwargs["wasb_repo"]).resolve() == Path(
            wasb_identity["repo_path"]
        ).resolve()
        assert Path(kwargs["frame_times"]).resolve() == frame_times_path.resolve()
        assert kwargs["fps"] == 30.0
        assert kwargs["visible_threshold"] == 0.90
        assert kwargs["batch_size"] == 1
        assert kwargs["device"] == "cpu"
        assert kwargs["input_preprocessing"] == "official"
        assert kwargs["emit_size_observations"] is False
        assert kwargs["emit_below_threshold_candidates"] is False
        replay_csv = Path(kwargs["prediction_csv_out"])
        if replay_state["matches_retained"]:
            # This is a separately materialized inference result, not a copy/read of
            # the retained dependency path that the verifier is authenticating.
            replay_csv.write_bytes(
                b"Frame,Visibility,X,Y,Confidence\n"
                b"0,0,0.0,0.0,0.0\n"
                b"1,1,32.0,24.0,0.96\n"
                b"2,0,0.0,0.0,0.0\n"
                b"3,0,0.0,0.0,0.0\n"
                b"4,0,0.0,0.0,0.0\n"
            )
        else:
            replay_csv.write_bytes(
                b"Frame,Visibility,X,Y,Confidence\n"
                b"0,0,0.0,0.0,0.0\n"
                b"1,1,31.0,24.0,0.96\n"
                b"2,0,0.0,0.0,0.0\n"
                b"3,0,0.0,0.0,0.0\n"
                b"4,0,0.0,0.0,0.0\n"
            )
        return {"source_mode": "wasb_predict"}

    monkeypatch.setattr(builder, "run_wasb_or_convert", fake_official_replay)
    verified = builder._verify_production_artifacts(clips=[clip], **verifier_args)
    assert verified["verified"] is True
    assert verified["official_wasb_replay_verified"] is True
    assert verified["official_wasb_replay_clip_count"] == 1
    assert verified["replayed_prediction_sha256_by_clip"] == {
        video_id: dependencies["wasb_predictions_csv_sha256"]
    }
    assert verified["pbvision_gallery_authority_id"] == (
        builder.PRODUCTION_GALLERY_AUTHORITY_ID
    )
    assert verified["verified_pbvision_gallery_sha256_by_source"] == {
        video_id: builder.PRODUCTION_GALLERY_ARTIFACT_SHA256[video_id]
    }
    assert len(replay_calls) == 1

    # A semantically valid teacher edit remains structurally reproducible when its
    # manifest dependencies are rehashed, but it must not redefine the independent
    # preregistered teacher-input authority.
    edited_cv_export = copy.deepcopy(cv_export)
    edited_cv_export["sessions"][0]["rallies"][0]["frames"][1]["actions"]["ball"][
        "confidence"
    ] = 0.96
    builder._write_json(cv_export_path, edited_cv_export)
    edited_cv_export_sha256 = builder._sha256_file(cv_export_path)
    rehashed_teacher_edit = copy.deepcopy(clip)
    rehashed_teacher_edit["dependencies"][
        "pbvision_cv_export_sha256"
    ] = edited_cv_export_sha256
    rehashed_sample = rehashed_teacher_edit["samples"][0]
    rehashed_sample["score"] = 0.96
    rehashed_sample["agreement"]["teacher_confidence"] = 0.96
    rehashed_sample["dependency_hashes"][
        "pbvision_cv_export_sha256"
    ] = edited_cv_export_sha256
    builder._validate_manifest_clips(
        [rehashed_teacher_edit],
        media_root=media_root,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        split_manifest_sha256=dependencies["split_manifest_sha256"],
        wasb_checkpoint_sha256=dependencies["wasb_checkpoint_sha256"],
        wasb_repo_commit=dependencies["wasb_repo_commit"],
        models_manifest_sha256=dependencies["models_manifest_sha256"],
        builder_code_sha256=dependencies["builder_code_sha256"],
        wasb_adapter_code_sha256=dependencies["wasb_adapter_code_sha256"],
    )
    authority_refusal = builder._verify_production_artifacts(
        clips=[rehashed_teacher_edit], **verifier_args
    )
    assert authority_refusal["verified"] is False
    assert "gallery authority SHA mismatch" in authority_refusal["reason"]
    assert "cannot redefine the preregistered authority" in authority_refusal["reason"]
    builder._write_json(cv_export_path, cv_export)

    replay_state["matches_retained"] = False
    replay_mismatch = builder._verify_production_artifacts(
        clips=[clip], **verifier_args
    )
    assert replay_mismatch["verified"] is False
    assert "differs from the pinned official inference replay" in replay_mismatch["reason"]
    replay_state["matches_retained"] = True

    forged = copy.deepcopy(clip)
    forged_sample = forged["samples"][0]
    forged_sample["teacher_xy"] = [20.0, 20.0]
    forged_sample["agreement"]["teacher_xy"] = [20.0, 20.0]
    forged_sample["agreement"]["wasb_xy"] = [20.0, 20.0]
    forged_sample["agreement"]["distance_px"] = 0.0
    builder._validate_manifest_clips(
        [forged],
        media_root=media_root,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        split_manifest_sha256=dependencies["split_manifest_sha256"],
        wasb_checkpoint_sha256=dependencies["wasb_checkpoint_sha256"],
        wasb_repo_commit=dependencies["wasb_repo_commit"],
        models_manifest_sha256=dependencies["models_manifest_sha256"],
        builder_code_sha256=dependencies["builder_code_sha256"],
        wasb_adapter_code_sha256=dependencies["wasb_adapter_code_sha256"],
    )
    refused = builder._verify_production_artifacts(clips=[forged], **verifier_args)
    assert refused["verified"] is False
    assert "do not reproduce from hashed teacher/WASB evidence" in refused["reason"]

    builder._write_json(
        wasb_track_path,
        {
            "frames": [
                {
                    "frame": 1,
                    "xy": [32.0, 24.0],
                    "conf": 0.96,
                    "visible": True,
                }
            ]
        },
    )
    synthetic_track_refusal = builder._verify_production_artifacts(
        clips=[clip], **verifier_args
    )
    assert synthetic_track_refusal["verified"] is False
    assert "does not regenerate from the bound prediction CSV" in synthetic_track_refusal["reason"]

    builder._write_json(wasb_track_path, wasb_track)
    builder._write_json(wasb_metadata_path, {"fixture": True})
    synthetic_metadata_refusal = builder._verify_production_artifacts(
        clips=[clip], **verifier_args
    )
    assert synthetic_metadata_refusal["verified"] is False
    assert "official builder-bound schema" in synthetic_metadata_refusal["reason"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"source": "synthetic"}, "source must be wasb"),
        ({"input_preprocessing": "harness_v0"}, "official input preprocessing"),
        ({"schema_version": "1"}, "schema_version must be integer 1"),
    ],
)
def test_wasb_track_refuses_nonofficial_top_level_identity(
    mutation: dict[str, object], reason: str
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    payload = _official_wasb_track_fixture()
    payload.update(mutation)
    with pytest.raises(builder.BallSstBuildError, match=reason):
        builder.extract_wasb_observations(
            payload,
            pts_s=(0.0, 1.0 / 30.0),
            fps=30.0,
            width=64,
            height=48,
        )


@pytest.mark.parametrize(
    ("frame_patch", "reason"),
    [
        ({"t": 99.0}, "timestamp differs from bound PTS"),
        ({"visible": "false"}, "visible must be a strict boolean"),
        ({"conf": 1.01}, r"conf must be in \[0, 1\]"),
        ({"xy": [64.0, 20.0]}, "visible point is out of bounds"),
    ],
)
def test_wasb_track_refuses_fabricated_frame_semantics(
    frame_patch: dict[str, object], reason: str
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    payload = _official_wasb_track_fixture()
    payload["frames"][1].update(frame_patch)
    with pytest.raises(builder.BallSstBuildError, match=reason):
        builder.extract_wasb_observations(
            payload,
            pts_s=(0.0, 1.0 / 30.0),
            fps=30.0,
            width=64,
            height=48,
        )


def _official_wasb_track_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "wasb",
        "input_preprocessing": "official",
        "frames": [
            {"t": 0.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False, "approx": False},
            {
                "t": 1.0 / 30.0,
                "xy": [32.0, 24.0],
                "conf": 0.96,
                "visible": True,
                "approx": False,
            },
        ],
        "bounces": [],
    }


def test_split_conflicting_video_source_alias_refuses() -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    payload = _frozen_split_payload()
    payload["rows"][0]["source_video"] = "83gyqyc10y8f"
    with pytest.raises(builder.BallSstBuildError, match="compare-only alias"):
        builder._validate_frozen_split(payload, Path("fixture.json"))


def test_media_identity_refuses_hash_substitution_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    video_id = builder.TRAIN_IDS[0]
    media_root = tmp_path / "media"
    canonical = media_root / video_id / "max.mp4"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical-test-bytes")
    monkeypatch.setitem(
        builder.EXPECTED_SOURCE_VIDEO_SHA256,
        video_id,
        builder._sha256_file(canonical),
    )
    assert builder._validate_media_identity(
        canonical, video_id=video_id, media_root=media_root.resolve()
    ) == builder._sha256_file(canonical)

    canonical.write_bytes(b"copied-compare-derivative")
    with pytest.raises(builder.BallSstBuildError, match="renamed, copied, or compare-derived"):
        builder._validate_media_identity(canonical, video_id=video_id, media_root=media_root.resolve())

    canonical.unlink()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"canonical-test-bytes")
    canonical.symlink_to(outside)
    with pytest.raises(builder.BallSstBuildError, match="non-symlink canonical path"):
        builder._validate_media_identity(canonical, video_id=video_id, media_root=media_root.resolve())


def test_assemble_revalidates_claimed_counts_and_cannot_pass_fabrication(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    builder_identity, wasb_identity, dependencies = _identity_context(builder)
    clips = _empty_clips(builder, tmp_path, dependencies)
    clips[0]["sample_count"] = 1_000
    with pytest.raises(builder.BallSstBuildError, match="sample_count does not match"):
        builder.assemble_sst_manifest(
            clips=clips,
            gallery_root=tmp_path / "gallery",
            media_root=tmp_path / "media",
            split_manifest=tmp_path / "split.json",
            split_manifest_sha256=dependencies["split_manifest_sha256"],
            wasb_checkpoint=Path(wasb_identity["checkpoint_path"]),
            wasb_checkpoint_sha256=builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
            wasb_identity=wasb_identity,
            builder_identity=builder_identity,
            teacher_confidence_min=0.90,
            agreement_radius_px=20.0,
            pseudo_weight=0.25,
            policy_overrides=[],
            decode_failures=0,
        )


def test_builder_identity_refuses_dirty_wasb_adapter_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    monkeypatch.setattr(builder, "_git_blob_sha256", lambda *_args, **_kwargs: "0" * 64)
    with pytest.raises(builder.BallSstBuildError, match="pinned HEAD blob"):
        builder._builder_identity()


def test_emitted_sample_manifest_satisfies_stage2_schema(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder
    from threed.racketsport.ball_sst_dataset import iter_sst_manifest_samples, load_sst_manifest

    video_id = _train_ids()[0]
    builder_identity = builder._builder_identity()
    wasb_identity = builder._resolve_wasb_identity(
        checkpoint_path=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        repo_path=Path("third_party/WASB-SBDT"),
        require_production_identity=True,
    )
    dependency_hashes = {
        "split_manifest_sha256": "1" * 64,
        "pbvision_cv_export_sha256": "2" * 64,
        "pbvision_metadata_sha256": "3" * 64,
        "pbvision_provenance_sha256": "4" * 64,
        "source_video_sha256": builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id],
        "frame_times_sha256": "6" * 64,
        "wasb_checkpoint_sha256": builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
        "wasb_repo_commit": builder.PRODUCTION_WASB_REPO_COMMIT,
        "models_manifest_sha256": wasb_identity["models_manifest_sha256"],
        "builder_code_sha256": builder_identity["builder_code_sha256"],
        "wasb_adapter_code_sha256": builder_identity["wasb_adapter_code_sha256"],
        "wasb_ball_track_sha256": "8" * 64,
        "wasb_metadata_sha256": "9" * 64,
        "wasb_predictions_csv_sha256": "a" * 64,
    }
    video_path = tmp_path / "media" / video_id / "max.mp4"
    samples = builder.build_source_samples(
        video_id=video_id,
        video_path=video_path,
        teacher_observations={10: _observation(10, (100.0, 80.0))},
        wasb_observations={
            10: builder.WasbObservation(10, (102.0, 80.0), 0.97, True)
        },
        pts_s=tuple(index / 30.0 for index in range(30)),
        width=640,
        height=360,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        dependency_hashes=dependency_hashes,
    )
    clip = {
        "clip_id": video_id,
        "canonical_source_id": video_id,
        "split": "train",
        "teacher_derived": True,
        "ground_truth": False,
        "rally_video": str(video_path),
        "source_video_sha256": builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id],
        "source_width": 640,
        "source_height": 360,
        "sample_count": len(samples),
        "samples": samples,
        "dependencies": dependency_hashes,
    }
    manifest = builder.assemble_sst_manifest(
        clips=[clip],
        gallery_root=tmp_path / "gallery",
        media_root=tmp_path / "media",
        split_manifest=tmp_path / "split.json",
        split_manifest_sha256="1" * 64,
        wasb_checkpoint=Path(wasb_identity["checkpoint_path"]),
        wasb_checkpoint_sha256=builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
        wasb_identity=wasb_identity,
        builder_identity=builder_identity,
        teacher_confidence_min=0.90,
        agreement_radius_px=20.0,
        pseudo_weight=0.25,
        policy_overrides=[],
        decode_failures=0,
    )
    out = tmp_path / "sst.json"
    out.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_sst_manifest(out)["artifact_type"] == "racketsport_ball_sst_manifest"
    parsed_samples = iter_sst_manifest_samples(out)
    assert len(parsed_samples) == 1
    row = parsed_samples[0]
    assert row["teacher_derived"] is True
    assert row["ground_truth"] is False
    assert row["agreement_reason"] == "frozen_wasb_spatial"
    assert row["weight"] == 0.25
    assert row["dependency_hashes"] == dependency_hashes
    assert manifest["holdout_rows_present"] == 0
    assert manifest["production_eligible"] is False  # fixture lacks the other six source entries
    assert manifest["gate"]["verdict"] == "NON_PRODUCTION_MANIFEST"
    assert manifest["preregistration"]["teacher_confidence_min"] == 0.90
    assert manifest["preregistration"]["builder_code_sha256"] == builder_identity["builder_code_sha256"]


def _train_ids() -> tuple[str, ...]:
    from scripts.racketsport.build_pbvision_ball_sst import TRAIN_IDS

    return TRAIN_IDS


def _frozen_split_payload() -> dict[str, object]:
    from scripts.racketsport.build_pbvision_ball_sst import (
        TEACHER_TEST_ONLY_IDS,
        TEACHER_VAL_ONLY_IDS,
        TRAIN_IDS,
    )

    rows = [
        {"video": video_id, "source_video": video_id, "split": "train"}
        for video_id in TRAIN_IDS
    ]
    rows.extend(
        {"video": video_id, "source_video": video_id, "split": "val"}
        for video_id in TEACHER_VAL_ONLY_IDS
    )
    rows.extend(
        {"video": video_id, "source_video": video_id, "split": "test"}
        for video_id in TEACHER_TEST_ONLY_IDS
    )
    return {"schema_version": 1, "rows": rows}


def _identity_context(builder):  # noqa: ANN001, ANN202
    builder_identity = builder._builder_identity()
    wasb_identity = builder._resolve_wasb_identity(
        checkpoint_path=Path("models/checkpoints/wasb/wasb_tennis_best.pth.tar"),
        repo_path=Path("third_party/WASB-SBDT"),
        require_production_identity=True,
    )
    dependencies = {
        "split_manifest_sha256": "1" * 64,
        "pbvision_cv_export_sha256": "2" * 64,
        "pbvision_metadata_sha256": "3" * 64,
        "pbvision_provenance_sha256": "4" * 64,
        "source_video_sha256": "5" * 64,
        "frame_times_sha256": "6" * 64,
        "wasb_checkpoint_sha256": builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
        "wasb_repo_commit": builder.PRODUCTION_WASB_REPO_COMMIT,
        "models_manifest_sha256": wasb_identity["models_manifest_sha256"],
        "builder_code_sha256": builder_identity["builder_code_sha256"],
        "wasb_adapter_code_sha256": builder_identity["wasb_adapter_code_sha256"],
        "wasb_ball_track_sha256": "8" * 64,
        "wasb_metadata_sha256": "9" * 64,
        "wasb_predictions_csv_sha256": "a" * 64,
    }
    return builder_identity, wasb_identity, dependencies


def _empty_clips(builder, tmp_path: Path, dependencies):  # noqa: ANN001, ANN202
    clips = []
    for video_id in builder.TRAIN_IDS:
        clip_dependencies = dict(dependencies)
        clip_dependencies["source_video_sha256"] = builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id]
        clips.append(
            {
                "clip_id": video_id,
                "canonical_source_id": video_id,
                "split": "train",
                "teacher_derived": True,
                "ground_truth": False,
                "rally_video": str(tmp_path / "media" / video_id / "max.mp4"),
                "source_video_sha256": builder.EXPECTED_SOURCE_VIDEO_SHA256[video_id],
                "source_width": 640,
                "source_height": 360,
                "sample_count": 0,
                "samples": [],
                "dependencies": clip_dependencies,
            }
        )
    return clips
