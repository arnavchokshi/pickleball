from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


CLI = "scripts/racketsport/build_pbvision_ball_sst.py"
PREPATCH_BUILDER_REVISION = "4c27023f686dd61200cf0394a8d900510596c8b0"
PREPATCH_BUILDER_SHA256 = "cad2b8907e88dd5ebb30c013d4269be1a404409797d19ac7533d46669228fb57"


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


def test_resume_off_manifest_is_byte_identical_to_prepatch_builder(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    head_builder = _load_prepatch_builder(tmp_path)
    fixture_root = tmp_path / "fixture"
    with pytest.MonkeyPatch.context() as patch:
        head_kwargs, _ = _configure_resume_build_fixture(head_builder, patch, fixture_root)
        head_builder.build_pbvision_ball_sst(**head_kwargs)
        baseline_bytes = Path(head_kwargs["out"]).read_bytes()
        baseline_dependencies = {
            path.relative_to(fixture_root / "sst_dependencies").as_posix(): path.read_bytes()
            for path in sorted((fixture_root / "sst_dependencies").rglob("*"))
            if path.is_file()
        }

    Path(head_kwargs["out"]).unlink()
    shutil.rmtree(fixture_root / "sst_dependencies")
    with pytest.MonkeyPatch.context() as patch:
        current_kwargs, _ = _configure_resume_build_fixture(builder, patch, fixture_root)
        builder.build_pbvision_ball_sst(**current_kwargs, resume_dependencies=False)
        patched_bytes = Path(current_kwargs["out"]).read_bytes()
        patched_dependencies = {
            path.relative_to(fixture_root / "sst_dependencies").as_posix(): path.read_bytes()
            for path in sorted((fixture_root / "sst_dependencies").rglob("*"))
            if path.is_file()
        }

    assert patched_bytes == baseline_bytes
    assert patched_dependencies == baseline_dependencies
    assert b"dependency_reused" not in patched_bytes
    assert b"dependencies_reused_count" not in patched_bytes


def test_real_production_dependency_copies_fail_closed_on_unbound_fifth_source() -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    dependency_root = Path(
        "runs/lanes/ball_b2_seed1_20260722/vm_pull/"
        "ball_data_regroup_20260722_partial/pbv_ball_sst_dependencies"
    )
    assert sorted(path.name for path in dependency_root.iterdir()) == [
        "143sf3gdwxsa",
        "98z43hspqz13",
        "bewqc0glhgpq",
        "st0epgnab7dr",
        "td2szayjwtrj",
    ]
    adapter_sha256 = builder._sha256_file(Path("threed/racketsport/wasb_adapter.py"))
    reusable_ids: list[str] = []
    rejected_ids: list[str] = []
    for source_dir in dependency_root.iterdir():
        metadata_path = source_dir / "wasb_ball_track_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_bindings = {
            "source_video_sha256": builder.EXPECTED_SOURCE_VIDEO_SHA256[
                source_dir.name
            ],
            "frame_times_sha256": builder._sha256_file(
                source_dir / "frame_times.json"
            ),
            "wasb_predictions_csv_sha256": builder._sha256_file(
                source_dir / "wasb_predictions.csv"
            ),
            "wasb_ball_track_sha256": builder._sha256_file(
                source_dir / "wasb_ball_track.json"
            ),
            "wasb_checkpoint_sha256": builder.PRODUCTION_WASB_CHECKPOINT_SHA256,
            "wasb_repo_commit": builder.PRODUCTION_WASB_REPO_COMMIT,
            "wasb_adapter_code_sha256": adapter_sha256,
        }
        if not builder._dependency_builder_bindings_match(
            metadata,
            expected_bindings,
        ):
            rejected_ids.append(source_dir.name)
        else:
            reusable_ids.append(source_dir.name)
            assert "builder_code_sha256" not in metadata["builder_bindings"]

    assert sorted(reusable_ids) == [
        "143sf3gdwxsa",
        "98z43hspqz13",
        "bewqc0glhgpq",
        "st0epgnab7dr",
    ]
    assert rejected_ids == ["td2szayjwtrj"]
    assert "builder_bindings" not in json.loads(
        (dependency_root / rejected_ids[0] / "wasb_ball_track_metadata.json").read_text(
            encoding="utf-8"
        )
    )


def test_valid_resume_skips_inference_and_preserves_manifest_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    metadata_validation_calls: list[dict[str, object]] = []
    original_validator = builder._validate_wasb_run_metadata

    def recording_validator(payload, **validator_kwargs):  # noqa: ANN001, ANN202
        metadata_validation_calls.append(dict(payload))
        return original_validator(payload, **validator_kwargs)

    patch.setattr(builder, "_validate_wasb_run_metadata", recording_validator)
    try:
        fresh = builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        assert len(calls) == 1
        assert len(metadata_validation_calls) == 1
        assert fresh["dependencies_reused_count"] == 0
        assert fresh["clips"][0]["dependency_reused"] is False
        calls.clear()
        metadata_validation_calls.clear()

        def forbidden_inference(**_kwargs):  # noqa: ANN202
            raise AssertionError("valid dependencies must skip run_wasb_or_convert")

        original_runner = builder.run_wasb_or_convert
        builder.run_wasb_or_convert = forbidden_inference
        try:
            resumed = builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        finally:
            builder.run_wasb_or_convert = original_runner

        assert calls == []
        assert len(metadata_validation_calls) == 1
        assert resumed["dependencies_reused_count"] == 1
        assert resumed["clips"][0]["dependency_reused"] is True
        assert _manifest_without_reuse_telemetry(resumed) == _manifest_without_reuse_telemetry(
            fresh
        )
        stdout_rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert stdout_rows[-1] == {"reused": True, "video_id": builder.TRAIN_IDS[0]}
    finally:
        patch.undo()


def test_resume_dependency_identity_change_mid_consumption_fails_without_output(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        out_path = Path(kwargs["out"])
        out_path.unlink()
        dependency_dir = (
            out_path.parent / "sst_dependencies" / builder.TRAIN_IDS[0]
        )
        predictions_csv = dependency_dir / "wasb_predictions.csv"
        predictions_before = predictions_csv.read_bytes()
        original_validator = builder._validate_wasb_predictions_csv
        identity_changed = False

        def change_identity_after_validation(path, **validator_kwargs):  # noqa: ANN001, ANN202
            nonlocal identity_changed
            result = original_validator(path, **validator_kwargs)
            if not identity_changed:
                text = predictions_csv.read_text(encoding="utf-8")
                predictions_csv.write_text(
                    text.replace("0.0", "0.00", 1),
                    encoding="utf-8",
                )
                identity_changed = True
            return result

        patch.setattr(
            builder,
            "_validate_wasb_predictions_csv",
            change_identity_after_validation,
        )

        manifest = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )

        assert identity_changed is True
        assert calls == []
        assert out_path.exists()
        assert manifest["dependencies_reused_count"] == 1
        assert predictions_csv.read_bytes() == predictions_before
    finally:
        patch.undo()


def test_resume_source_video_identity_change_mid_consumption_fails_without_output(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        out_path = Path(kwargs["out"])
        out_path.unlink()
        source_video = (
            Path(kwargs["media_root"]) / builder.TRAIN_IDS[0] / "max.mp4"
        )
        original_extractor = builder.extract_wasb_observations
        identity_changed = False

        def change_source_after_extraction(payload, **extract_kwargs):  # noqa: ANN001, ANN202
            nonlocal identity_changed
            result = original_extractor(payload, **extract_kwargs)
            if not identity_changed:
                source_video.write_bytes(source_video.read_bytes() + b"\x00")
                identity_changed = True
            return result

        patch.setattr(
            builder,
            "extract_wasb_observations",
            change_source_after_extraction,
        )

        manifest = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )

        assert identity_changed is True
        assert calls == []
        assert out_path.exists()
        assert manifest["dependencies_reused_count"] == 1
        assert (
            manifest["clips"][0]["source_video_sha256"]
            != builder._sha256_file(source_video)
        )
    finally:
        patch.undo()


def test_resume_checkpoint_identity_change_mid_consumption_fails_without_output(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    checkpoint_path = Path(kwargs["wasb_checkpoint"])
    checked_checkpoint_sha256 = builder._sha256_file(checkpoint_path)
    resolved_identity = builder._resolve_wasb_identity()
    resolved_identity["checkpoint_sha256"] = checked_checkpoint_sha256
    resolved_identity["expected_checkpoint_sha256"] = checked_checkpoint_sha256
    patch.setattr(
        builder,
        "PRODUCTION_WASB_CHECKPOINT_SHA256",
        checked_checkpoint_sha256,
    )
    patch.setattr(
        builder,
        "_resolve_wasb_identity",
        lambda **_kwargs: dict(resolved_identity),
    )
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        out_path = Path(kwargs["out"])
        out_path.unlink()
        metadata_path = (
            out_path.parent
            / "sst_dependencies"
            / builder.TRAIN_IDS[0]
            / "wasb_ball_track_metadata.json"
        )
        metadata_before = metadata_path.read_bytes()
        original_validator = builder._validate_wasb_run_metadata
        identity_changed = False

        def change_checkpoint_after_runtime_validation(
            payload, **validator_kwargs  # noqa: ANN001
        ):  # noqa: ANN202
            nonlocal identity_changed
            result = original_validator(payload, **validator_kwargs)
            if not identity_changed:
                checkpoint_path.write_bytes(
                    checkpoint_path.read_bytes() + b" changed"
                )
                identity_changed = True
            return result

        patch.setattr(
            builder,
            "_validate_wasb_run_metadata",
            change_checkpoint_after_runtime_validation,
        )

        manifest = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )

        assert identity_changed is True
        assert calls == []
        assert out_path.exists()
        assert manifest["dependencies_reused_count"] == 1
        assert metadata_path.read_bytes() == metadata_before
        assert builder._sha256_file(checkpoint_path) != checked_checkpoint_sha256
    finally:
        patch.undo()


def test_resume_wasb_repo_commit_change_mid_consumption_fails_without_output(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    resolved_identity = builder._resolve_wasb_identity()
    checked_repo_commit = str(resolved_identity["repo_commit"])
    changed_repo_commit = "0" * 40
    assert changed_repo_commit != checked_repo_commit
    repo_commit = checked_repo_commit
    patch.setattr(
        builder,
        "_resolve_wasb_identity",
        lambda **_kwargs: dict(resolved_identity),
    )

    def fake_wasb_git_output(_repo, *args):  # noqa: ANN001, ANN202
        if args == ("rev-parse", "HEAD"):
            return repo_commit
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(f"unexpected git identity request: {args}")

    patch.setattr(builder, "_git_output", fake_wasb_git_output)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        out_path = Path(kwargs["out"])
        out_path.unlink()
        metadata_path = (
            out_path.parent
            / "sst_dependencies"
            / builder.TRAIN_IDS[0]
            / "wasb_ball_track_metadata.json"
        )
        metadata_before = metadata_path.read_bytes()
        original_validator = builder._validate_wasb_run_metadata
        identity_changed = False

        def change_repo_commit_after_runtime_validation(
            payload, **validator_kwargs  # noqa: ANN001
        ):  # noqa: ANN202
            nonlocal identity_changed, repo_commit
            result = original_validator(payload, **validator_kwargs)
            if not identity_changed:
                repo_commit = changed_repo_commit
                identity_changed = True
            return result

        patch.setattr(
            builder,
            "_validate_wasb_run_metadata",
            change_repo_commit_after_runtime_validation,
        )

        manifest = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )

        assert identity_changed is True
        assert calls == []
        assert out_path.exists()
        assert manifest["dependencies_reused_count"] == 1
        assert metadata_path.read_bytes() == metadata_before
    finally:
        patch.undo()


@pytest.mark.parametrize(
    ("binding_key", "target_path"),
    [
        ("builder_code_sha256", "builder"),
        ("wasb_adapter_code_sha256", "adapter"),
    ],
)
def test_resume_code_identity_change_precedes_metadata_output(
    tmp_path: Path,
    binding_key: str,
    target_path: str,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        out_path = Path(kwargs["out"])
        out_path.unlink()
        metadata_path = (
            out_path.parent
            / "sst_dependencies"
            / builder.TRAIN_IDS[0]
            / "wasb_ball_track_metadata.json"
        )
        metadata_before = metadata_path.read_bytes()
        original_validator = builder._validate_wasb_run_metadata
        original_sha256_file = builder._sha256_file
        identity_changed = False
        identity_path = (
            Path(builder.__file__).resolve(strict=True)
            if target_path == "builder"
            else (
                Path(builder.ROOT) / builder.WASB_ADAPTER_RELATIVE_PATH
            ).resolve(strict=True)
        )

        def change_code_identity_after_runtime_validation(
            payload, **validator_kwargs  # noqa: ANN001
        ):  # noqa: ANN202
            nonlocal identity_changed
            result = original_validator(payload, **validator_kwargs)
            identity_changed = True
            return result

        def divergent_sha256_file(path):  # noqa: ANN001, ANN202
            observed = original_sha256_file(Path(path))
            if identity_changed and Path(path).resolve(strict=True) == identity_path:
                return "0" * 64
            return observed

        patch.setattr(
            builder,
            "_validate_wasb_run_metadata",
            change_code_identity_after_runtime_validation,
        )
        patch.setattr(builder, "_sha256_file", divergent_sha256_file)

        manifest = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )

        assert identity_changed is True
        assert calls == []
        assert out_path.exists()
        assert manifest["dependencies_reused_count"] == 1
        assert metadata_path.read_bytes() == metadata_before
    finally:
        patch.undo()


def test_identity_stability_checks_precede_each_output_boundary(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, _calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    events: list[tuple[str, Path | None]] = []
    original_snapshot = builder._create_immutable_dependency_snapshot
    original_probe = builder.probe_media_pts
    original_publish = builder._publish_snapshot_file

    def recording_snapshot(**snapshot_kwargs):  # noqa: ANN202
        snapshot = original_snapshot(**snapshot_kwargs)
        events.append(("snapshot_verified", snapshot.root))
        return snapshot

    def recording_probe(path, **probe_kwargs):  # noqa: ANN001, ANN202
        resolved = Path(path).resolve(strict=True)
        events.append(("probe", resolved))
        assert resolved.stat().st_mode & 0o222 == 0
        return original_probe(path, **probe_kwargs)

    def recording_publish(source, destination):  # noqa: ANN001, ANN202
        events.append(("publish", Path(destination)))
        return original_publish(Path(source), Path(destination))

    patch.setattr(builder, "_create_immutable_dependency_snapshot", recording_snapshot)
    patch.setattr(builder, "probe_media_pts", recording_probe)
    patch.setattr(builder, "_publish_snapshot_file", recording_publish)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)

        snapshot_index = next(
            index for index, event in enumerate(events) if event[0] == "snapshot_verified"
        )
        probe_indices = [
            index for index, event in enumerate(events) if event[0] == "probe"
        ]
        publish_indices = [
            index for index, event in enumerate(events) if event[0] == "publish"
        ]
        assert probe_indices and publish_indices
        assert snapshot_index < min(probe_indices)
        assert max(probe_indices) < min(publish_indices)
        source_video = (
            Path(kwargs["media_root"]) / builder.TRAIN_IDS[0] / "max.mp4"
        ).resolve(strict=True)
        assert all(
            event[1] != source_video
            for event in events
            if event[0] == "probe"
        )
    finally:
        patch.undo()


@pytest.mark.parametrize(
    "tamper_case",
    [
        "predictions_csv_bytes",
        "ball_track_bytes",
        "checkpoint_binding",
        "repo_commit_binding",
        "deleted_dependency",
        "missing_builder_bindings",
    ],
)
def test_resume_tampering_falls_through_to_recompute(
    tmp_path: Path,
    tamper_case: str,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(builder, patch, tmp_path)
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        calls.clear()
        dependency_dir = Path(kwargs["out"]).parent / "sst_dependencies" / builder.TRAIN_IDS[0]
        _tamper_resume_dependency(dependency_dir, tamper_case)

        rebuilt = builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)

        assert len(calls) == 1
        assert rebuilt["dependencies_reused_count"] == 0
        assert rebuilt["clips"][0]["dependency_reused"] is False
    finally:
        patch.undo()


def test_mixed_resume_preserves_all_fresh_manifest_data(tmp_path: Path) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    video_ids = builder.TRAIN_IDS[:2]
    patch = pytest.MonkeyPatch()
    kwargs, calls = _configure_resume_build_fixture(
        builder,
        patch,
        tmp_path,
        video_ids=video_ids,
    )
    try:
        fresh = builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        assert len(calls) == 2
        calls.clear()
        dependency_root = Path(kwargs["out"]).parent / "sst_dependencies"
        (dependency_root / video_ids[1] / "wasb_predictions.csv").write_bytes(b"tampered")

        mixed = builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)

        assert len(calls) == 1
        assert calls[0].parent.name == video_ids[1]
        assert mixed["dependencies_reused_count"] == 1
        assert [clip["dependency_reused"] for clip in mixed["clips"]] == [True, False]
        assert _manifest_without_reuse_telemetry(mixed) == _manifest_without_reuse_telemetry(
            fresh
        )
    finally:
        patch.undo()


def test_original_source_mutation_after_snapshot_verify_is_byte_identical(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, inference_calls = _configure_resume_build_fixture(
        builder,
        patch,
        tmp_path,
    )
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        inference_calls.clear()
        unmutated = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )
        unmutated_bytes = Path(kwargs["out"]).read_bytes()
        inference_calls.clear()

        source_video = (
            Path(kwargs["media_root"]) / builder.TRAIN_IDS[0] / "max.mp4"
        ).resolve(strict=True)
        original_probe = builder.probe_media_pts
        consumed_paths: list[Path] = []
        source_mutated = False

        def mutate_original_after_snapshot_probe(
            path, **probe_kwargs  # noqa: ANN001
        ):  # noqa: ANN202
            nonlocal source_mutated
            consumed_paths.append(Path(path).resolve(strict=True))
            result = original_probe(path, **probe_kwargs)
            if not source_mutated:
                source_video.write_bytes(source_video.read_bytes() + b" changed")
                source_mutated = True
            return result

        patch.setattr(
            builder,
            "probe_media_pts",
            mutate_original_after_snapshot_probe,
        )

        mutated = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )
        mutated_bytes = Path(kwargs["out"]).read_bytes()

        assert source_mutated is True
        assert consumed_paths and all(path != source_video for path in consumed_paths)
        assert inference_calls == []
        assert mutated == unmutated
        assert mutated_bytes == unmutated_bytes
    finally:
        patch.undo()


def test_original_source_deletion_after_snapshot_verify_is_byte_identical(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    patch = pytest.MonkeyPatch()
    kwargs, inference_calls = _configure_resume_build_fixture(
        builder,
        patch,
        tmp_path,
    )
    try:
        builder.build_pbvision_ball_sst(**kwargs, resume_dependencies=True)
        inference_calls.clear()
        baseline = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )
        baseline_manifest_bytes = Path(kwargs["out"]).read_bytes()
        dependency_root = Path(kwargs["out"]).parent / "sst_dependencies"
        baseline_dependency_bytes = {
            path.relative_to(dependency_root).as_posix(): path.read_bytes()
            for path in sorted(dependency_root.rglob("*"))
            if path.is_file()
        }
        inference_calls.clear()

        video_id = builder.TRAIN_IDS[0]
        source_video = (
            Path(kwargs["media_root"]) / video_id / "max.mp4"
        ).resolve(strict=True)
        checkpoint = Path(kwargs["wasb_checkpoint"]).resolve(strict=True)
        wasb_repo = Path(kwargs["wasb_repo"]).resolve(strict=True)
        original_dependency_files = {
            source_video,
            checkpoint,
            Path(kwargs["split_manifest"]).resolve(strict=True),
            *(
                (Path(kwargs["gallery_root"]) / video_id / filename).resolve(
                    strict=True
                )
                for filename in builder.PRODUCTION_GALLERY_ARTIFACT_FILENAMES
            ),
            *(
                path.resolve(strict=True)
                for path in dependency_root.rglob("*")
                if path.is_file()
            ),
        }
        original_snapshot = builder._create_immutable_dependency_snapshot
        original_resolve = Path.resolve
        original_stat = Path.stat
        original_open = Path.open
        snapshot_verified = False
        post_snapshot_original_accesses: list[tuple[str, str]] = []

        def is_original_dependency(path: Path) -> bool:
            candidate = path if path.is_absolute() else Path.cwd() / path
            return (
                candidate in original_dependency_files
                or candidate == wasb_repo
                or candidate.is_relative_to(wasb_repo)
            )

        def record_snapshot_and_delete_source(**snapshot_kwargs):  # noqa: ANN202
            nonlocal snapshot_verified
            snapshot = original_snapshot(**snapshot_kwargs)
            snapshot_verified = True
            source_video.unlink()
            return snapshot

        def record_resolve(self, strict=False):  # noqa: ANN001, ANN202
            if snapshot_verified and is_original_dependency(self):
                post_snapshot_original_accesses.append(("resolve", str(self)))
            return original_resolve(self, strict=strict)

        def record_stat(self, *args, **stat_kwargs):  # noqa: ANN001, ANN202
            if snapshot_verified and is_original_dependency(self):
                post_snapshot_original_accesses.append(("stat", str(self)))
            return original_stat(self, *args, **stat_kwargs)

        def record_open(self, *args, **open_kwargs):  # noqa: ANN001, ANN202
            if snapshot_verified and is_original_dependency(self):
                post_snapshot_original_accesses.append(("open", str(self)))
            return original_open(self, *args, **open_kwargs)

        patch.setattr(
            builder,
            "_create_immutable_dependency_snapshot",
            record_snapshot_and_delete_source,
        )
        patch.setattr(Path, "resolve", record_resolve)
        patch.setattr(Path, "stat", record_stat)
        patch.setattr(Path, "open", record_open)

        deleted_source = builder.build_pbvision_ball_sst(
            **kwargs,
            resume_dependencies=True,
        )
        snapshot_verified = False
        deleted_manifest_bytes = Path(kwargs["out"]).read_bytes()
        deleted_dependency_bytes = {
            path.relative_to(dependency_root).as_posix(): path.read_bytes()
            for path in sorted(dependency_root.rglob("*"))
            if path.is_file()
        }

        assert inference_calls == []
        assert deleted_source == baseline
        assert deleted_manifest_bytes == baseline_manifest_bytes
        assert deleted_dependency_bytes == baseline_dependency_bytes
        assert post_snapshot_original_accesses == []
        assert not source_video.exists()
    finally:
        patch.undo()


def test_non_snapshot_dependency_consumption_is_typed_error(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder

    snapshot_root = tmp_path / "immutable_snapshot"
    snapshot_root.mkdir()
    outside = tmp_path / "original_source.mp4"
    outside.write_bytes(b"original")

    with pytest.raises(
        builder.BallSstSnapshotPathError,
        match="non-snapshot dependency path",
    ):
        builder._require_snapshot_path(
            snapshot_root,
            outside,
            context="source PTS probe",
        )


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


def _load_prepatch_builder(tmp_path: Path):  # noqa: ANN202
    source = subprocess.run(
        ["git", "show", f"{PREPATCH_BUILDER_REVISION}:{CLI}"],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(source).hexdigest() == PREPATCH_BUILDER_SHA256
    module_path = tmp_path / "head_builder" / CLI
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(source)
    module_name = "_head_build_pbvision_ball_sst_fixture"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _configure_resume_build_fixture(
    builder,  # noqa: ANN001
    patch: pytest.MonkeyPatch,
    root: Path,
    *,
    video_ids: tuple[str, ...] | None = None,
):  # noqa: ANN202
    repository_root = Path.cwd().resolve(strict=True)
    selected_ids = video_ids or tuple(builder.TRAIN_IDS[:1])
    gallery = root / "gallery"
    media_root = root / "media"
    checkpoint = root / "checkpoint.pth.tar"
    repo = root / "wasb_repo"
    split = root / "split.json"
    models_manifest = root / "models_manifest.json"
    out = root / "sst.json"
    identity_file = root / "stable_builder_identity.py"
    root.mkdir(parents=True, exist_ok=True)
    repo.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixture checkpoint")
    split.write_text("{}\n", encoding="utf-8")
    models_manifest.write_text('{"models": []}\n', encoding="utf-8")
    identity_file.write_text("# stable fixture identity\n", encoding="utf-8")

    media_hashes: dict[str, str] = {}
    gallery_hashes: dict[str, dict[str, str]] = {}
    for video_id in selected_ids:
        source_media = media_root / video_id / "max.mp4"
        source_media.parent.mkdir(parents=True, exist_ok=True)
        source_media.write_bytes(f"fixture media {video_id}".encode("utf-8"))
        media_hashes[video_id] = builder._sha256_file(source_media)

        source_gallery = gallery / video_id
        source_gallery.mkdir(parents=True, exist_ok=True)
        payloads = {
            "cv_export.json": {"camera": {"fps": 30.0}, "sessions": []},
            "api_get_metadata.json": {
                "metadata": {"width": 64, "height": 48, "fps": 30.0}
            },
            "video_provenance.json": {
                "video_id": video_id,
                "source_video_url": f"https://storage.googleapis.com/pbv-pro/{video_id}/max.mp4",
            },
        }
        hashes: dict[str, str] = {}
        for filename, payload in payloads.items():
            path = source_gallery / filename
            builder._write_json(path, payload)
            hashes[filename] = builder._sha256_file(path)
        gallery_hashes[video_id] = hashes

    adapter_path = repository_root / "threed/racketsport/wasb_adapter.py"
    builder_identity = {
        "builder_path": CLI,
        "builder_code_sha256": builder._sha256_file(identity_file),
        "builder_git_commit": "d" * 40,
        "wasb_adapter_path": "threed/racketsport/wasb_adapter.py",
        "wasb_adapter_code_sha256": builder._sha256_file(adapter_path),
        "wasb_adapter_git_commit": "d" * 40,
    }
    wasb_identity = {
        "manifest_model_id": builder.PRODUCTION_WASB_MODEL_ID,
        "models_manifest_path": str(models_manifest.resolve(strict=True)),
        "models_manifest_sha256": builder._sha256_file(models_manifest),
        "checkpoint_path": str(checkpoint.resolve(strict=True)),
        "checkpoint_sha256": builder._sha256_file(checkpoint),
        "expected_checkpoint_sha256": builder._sha256_file(checkpoint),
        "repo_path": str(repo.resolve(strict=True)),
        "repo_commit": builder.PRODUCTION_WASB_REPO_COMMIT,
        "expected_repo_commit": builder.PRODUCTION_WASB_REPO_COMMIT,
        "repo_clean": True,
        "production_identity_verified": True,
    }
    authority = {
        "authority_id": "resume_unit_fixture",
        "canonical_gallery_relative_path": str(gallery),
        "artifact_filenames": list(builder.PRODUCTION_GALLERY_ARTIFACT_FILENAMES),
        "expected_sha256_by_source": gallery_hashes,
    }
    timing = builder.MediaTiming(
        fps=30.0,
        duration_s=0.1,
        pts_s=(0.0, 1.0 / 30.0, 2.0 / 30.0),
        width=64,
        height=48,
    )
    calls: list[Path] = []

    def fake_run_wasb_or_convert(**call_kwargs):  # noqa: ANN202
        calls.append(Path(call_kwargs["video"]))
        csv_path = Path(call_kwargs["prediction_csv_out"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "Frame,Visibility,X,Y,Confidence\n"
            "0,0,0.0,0.0,0.0\n"
            "1,0,0.0,0.0,0.0\n"
            "2,0,0.0,0.0,0.0\n",
            encoding="utf-8",
        )
        track = builder.wasb_csv_to_ball_track(
            csv_path,
            fps=call_kwargs["fps"],
            frame_times=Path(call_kwargs["frame_times"]),
            visible_threshold=call_kwargs["visible_threshold"],
            input_preprocessing="official",
        )
        builder._write_json(Path(call_kwargs["out"]), track)
        frame_count = len(timing.pts_s)
        wall_seconds = 1.0
        effective_fps = frame_count / wall_seconds
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_wasb_ball_run",
            "status": builder.STATUS_TESTED,
            "source_mode": "wasb_predict",
            "predictions_csv": str(csv_path),
            "out": str(call_kwargs["out"]),
            "fps": call_kwargs["fps"],
            "frame_count": frame_count,
            "visible_frame_count": 0,
            "confidence_semantics": builder.WASB_CONFIDENCE_SEMANTICS,
            "visible_threshold": call_kwargs["visible_threshold"],
            "input_preprocessing": "official",
            "non_promotable_measurement_mode": False,
            "not_ground_truth": True,
            "official_repo_url": builder.WASB_REPO_URL,
            "official_model_zoo_url": builder.WASB_MODEL_ZOO_URL,
            "runtime": {
                "wasb_repo": str(call_kwargs["wasb_repo"]),
                "wasb_repo_commit": wasb_identity["repo_commit"],
                "wasb_checkpoint": {
                    "path": str(call_kwargs["checkpoint"]),
                    "sha256": wasb_identity["checkpoint_sha256"],
                },
                "video": str(call_kwargs["video"]),
                "source_video_fps": call_kwargs["fps"],
                "source_video_frame_count": frame_count,
                "source_video_size": [timing.width, timing.height],
                "processed_frame_count": frame_count,
                "processed_window_count": frame_count - 2,
                "read_frame_count": frame_count,
                "video_range_seconds": None,
                "max_frames": None,
                "batch_size": call_kwargs["batch_size"],
                "device": call_kwargs["device"],
                "input_preprocessing": "official",
                "non_promotable_measurement_mode": False,
                "wall_seconds": wall_seconds,
                "effective_fps": effective_fps,
                "realtime_factor": effective_fps / call_kwargs["fps"],
            },
        }

    patch.setattr(builder, "__file__", str(identity_file))
    patch.setattr(builder, "ROOT", repository_root)
    patch.setattr(builder, "TRAIN_IDS", selected_ids)
    patch.setattr(builder, "TEACHER_VAL_ONLY_IDS", ())
    patch.setattr(builder, "TEACHER_TEST_ONLY_IDS", ())
    patch.setattr(builder, "ALL_NONTRAIN_IDS", frozenset())
    patch.setattr(builder, "EXPECTED_SOURCE_VIDEO_SHA256", media_hashes)
    patch.setattr(builder, "PRODUCTION_GALLERY_ARTIFACT_SHA256", gallery_hashes)
    patch.setattr(builder, "_validate_frozen_split", lambda *_args, **_kwargs: None)
    patch.setattr(
        builder,
        "_canonical_gallery_root",
        lambda path, **_kwargs: Path(path).resolve(strict=True),
    )
    patch.setattr(
        builder,
        "_validate_production_gallery_inventory",
        lambda _gallery: gallery_hashes,
    )
    patch.setattr(builder, "_builder_identity", lambda: dict(builder_identity))
    patch.setattr(builder, "_resolve_wasb_identity", lambda **_kwargs: dict(wasb_identity))

    original_git_output = builder._git_output

    def fixture_git_output(git_repo, *args):  # noqa: ANN001, ANN202
        resolved_git_repo = Path(git_repo).resolve()
        if (
            resolved_git_repo == repo.resolve()
            or resolved_git_repo.name == "wasb_repo"
        ):
            if args == ("rev-parse", "HEAD"):
                return wasb_identity["repo_commit"]
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return ""
            raise AssertionError(f"unexpected fixture WASB git request: {args}")
        return original_git_output(Path(git_repo), *args)

    patch.setattr(builder, "_git_output", fixture_git_output)
    patch.setattr(builder, "_production_gallery_authority_payload", lambda: dict(authority))
    patch.setattr(builder, "probe_media_pts", lambda *_args, **_kwargs: timing)
    patch.setattr(builder, "run_wasb_or_convert", fake_run_wasb_or_convert)
    patch.setattr(
        builder,
        "_verify_production_artifacts",
        lambda **_kwargs: {
            "verified": True,
            "status": "passed",
            "reason": "resume unit fixture",
            "verified_clip_count": len(selected_ids),
            "verified_sample_count": 0,
        },
    )
    return (
        {
            "gallery_root": gallery,
            "media_root": media_root,
            "split_manifest": split,
            "wasb_checkpoint": checkpoint,
            "wasb_repo": repo,
            "teacher_confidence_min": 0.90,
            "agreement_radius_px": 20.0,
            "pseudo_weight": 0.25,
            "out": out,
            "device": "cpu",
            "wasb_batch_size": 1,
        },
        calls,
    )


def _tamper_resume_dependency(dependency_dir: Path, tamper_case: str) -> None:
    metadata_path = dependency_dir / "wasb_ball_track_metadata.json"
    if tamper_case == "predictions_csv_bytes":
        path = dependency_dir / "wasb_predictions.csv"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif tamper_case == "ball_track_bytes":
        path = dependency_dir / "wasb_ball_track.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper_case == "deleted_dependency":
        (dependency_dir / "wasb_predictions.csv").unlink()
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tamper_case == "checkpoint_binding":
            metadata["builder_bindings"]["wasb_checkpoint_sha256"] = "0" * 64
        elif tamper_case == "repo_commit_binding":
            metadata["builder_bindings"]["wasb_repo_commit"] = "0" * 40
        elif tamper_case == "missing_builder_bindings":
            metadata.pop("builder_bindings")
        else:
            raise AssertionError(f"unknown tamper case: {tamper_case}")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _manifest_without_reuse_telemetry(payload: dict[str, object]) -> bytes:
    normalized = copy.deepcopy(payload)
    normalized.pop("dependencies_reused_count", None)
    for clip in normalized["clips"]:
        clip.pop("dependency_reused", None)
    return (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
