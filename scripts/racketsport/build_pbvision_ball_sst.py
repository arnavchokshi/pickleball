#!/usr/bin/env python3
"""Build quarantined, agreement-gated pb.vision BALL pseudo supervision.

Production eligibility is preregistered structurally in this module.  A production
row uses confidence ``0.90``, a ``20 px`` agreement radius, and pseudo weight ``0.25``.
CLI values remain available for explicit experiments, but any mismatch makes the
whole manifest non-production and the manifest gate refuses it.

There are exactly two acceptance paths:

* direct, in-bounds agreement between pb.vision and the pinned frozen-WASB result; or
* ``frozen_wasb_temporal_bridge_v2`` for a teacher-only current frame.  This path
  requires an in-bounds pb.vision/WASB agreement anchor both before and after the
  current source frame, each no more than two source frames away.  The current
  teacher point must be within 20 px of the source-frame interpolation of the two
  frozen-WASB anchor positions.  A high-confidence current-frame WASB disagreement
  is a refusal, not a gap.  Same-teacher smoothness never supplies independence.

Teacher absence is ignored and never creates a negative row.  The builder addresses
only seven frozen training sources and binds each to its canonical ``<id>/max.mp4``
path and preregistered SHA-256.  Compare-only IDs and conflicting aliases are refused
before a gallery or media source can be read.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.wasb_adapter import (  # noqa: E402
    STATUS_TESTED,
    WASB_CONFIDENCE_SEMANTICS,
    WASB_MODEL_ZOO_URL,
    WASB_REPO_URL,
    run_wasb_or_convert,
    wasb_csv_to_ball_track,
)


ARTIFACT_TYPE = "racketsport_ball_sst_manifest"
REFUSAL_ARTIFACT_TYPE = "racketsport_pbvision_ball_sst_build_refusal"
PRODUCTION_POLICY_ID = "pbv_ball_sst_production_v2"
PRODUCTION_TEACHER_CONFIDENCE_MIN = 0.90
PRODUCTION_AGREEMENT_RADIUS_PX = 20.0
PRODUCTION_PSEUDO_WEIGHT = 0.25
PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES = 2
PRODUCTION_WASB_MODEL_ID = "wasb_tennis_bmvc2023"
PRODUCTION_WASB_CHECKPOINT_RELATIVE_PATH = Path(
    "models/checkpoints/wasb/wasb_tennis_best.pth.tar"
)
PRODUCTION_WASB_CHECKPOINT_SHA256 = (
    "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
)
PRODUCTION_WASB_REPO_RELATIVE_PATH = Path("third_party/WASB-SBDT")
PRODUCTION_WASB_REPO_COMMIT = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
WASB_ADAPTER_RELATIVE_PATH = Path("threed/racketsport/wasb_adapter.py")
FROZEN_GALLERY_RELATIVE_PATH = Path("data/pbvision_gallery_20260719")
FROZEN_SPLIT_RELATIVE_PATH = Path("runs/lanes/pbv_pickleball_corpus_20260720/manifest.json")
FROZEN_SPLIT_SHA256 = "cf8f251827688c7923e35ce93b06b66c014ba9192b9d18f4ecbd2a256195451b"
PRODUCTION_GALLERY_AUTHORITY_ID = "pbvision_gallery_20260719_teacher_inputs_sha256_v1"
PRODUCTION_GALLERY_ARTIFACT_FILENAMES = (
    "cv_export.json",
    "api_get_metadata.json",
    "video_provenance.json",
)
TRAIN_IDS = (
    "143sf3gdwxsa",
    "98z43hspqz13",
    "bewqc0glhgpq",
    "st0epgnab7dr",
    "td2szayjwtrj",
    "tqjlrcntpjvt",
    "xkadsq9bli3h",
)
TEACHER_VAL_ONLY_IDS = ("pldtjpw3h0jw", "utasf5hnozwz")
TEACHER_TEST_ONLY_IDS = ("0tmdeghtfvjx",)
COMPARE_ONLY_IDS = ("83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr")
ALL_NONTRAIN_IDS = frozenset((*TEACHER_VAL_ONLY_IDS, *TEACHER_TEST_ONLY_IDS, *COMPARE_ONLY_IDS))
ACCEPTED_WINDOW_TARGET = 1_000
ACCEPTED_SOURCE_TARGET = 5

# These hashes were captured from the staged canonical GCS ``<id>/max.mp4`` objects
# before B1.  They are code constants on purpose: a renamed/copy-derived file is not
# allowed to redefine source identity after the experiment starts.
EXPECTED_SOURCE_VIDEO_SHA256: dict[str, str] = {
    "0tmdeghtfvjx": "8b007124fa949defff85b11f70de5bf4c4c0e43ba64c085c7eded18f0041dfd1",
    "143sf3gdwxsa": "03fbdc2b056c1b1ed665c71994c06bc485f385b44a2fee892338360c666f845c",
    "98z43hspqz13": "006eb7d0e7e7c5c351ea72b88c946a452660adb24eff87e77d12419b7330b11f",
    "bewqc0glhgpq": "e6b73a38535aea5d3644c3a94091b3c5d261b6c2b60e5d80a21514ad502b69cf",
    "pldtjpw3h0jw": "4d55d822c0b0bbaedbf27e16301b035beef2542df0b11b416b87f898ba8ff59c",
    "st0epgnab7dr": "2803b4a18c97e3d3165cdbacbe7bcbe6c4b0c273820aa6840b7e731aea98ff04",
    "td2szayjwtrj": "9594260561b334937a1dfb62c1450315fdcb1ee3e1ece304961416c7d15a2d79",
    "tqjlrcntpjvt": "176cb66c13e2fa481839815c1dc41c063b2a0cc17758e75dd9c7f39627f31490",
    "utasf5hnozwz": "614580f5b3a2f634a76f5483e10b8c1f7919fd5affbd3ee86532a539f3f58197",
    "xkadsq9bli3h": "5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181",
}

# Independent preregistered authority for the teacher inputs.  Per-manifest
# dependency hashes are observations and therefore cannot be allowed to redefine
# teacher identity after an edit.  These values were captured from the canonical
# gallery before B1 and are intentionally fixed in source alongside the media
# identities above.
PRODUCTION_GALLERY_ARTIFACT_SHA256: dict[str, dict[str, str]] = {
    "143sf3gdwxsa": {
        "cv_export.json": "a9b8ab2fd6a9511d2202f6a6b211ea572303ff65d88131db18d71b1e95bf43f9",
        "api_get_metadata.json": "1bf2d462897edf7e3459648c7f4f99f211229f20bfad8ed0306aa0c8e2211fbb",
        "video_provenance.json": "5b613052e0173ecc7542e06ae9e3d3c15691e039219369fcb98b9cd1b0ab9a48",
    },
    "98z43hspqz13": {
        "cv_export.json": "e2c679a40a76d031eb3a60ae79aba5903d545efd7c9d9a49d08d2eb9bcab3658",
        "api_get_metadata.json": "f5a649d4f7391c25aec864d47cc51bf4bda80cde5348d127ca64c9d33eca6ed0",
        "video_provenance.json": "b4af3459c7874e76fdba762e589d0cd6369b59029f2f0a5620522f6f5f083392",
    },
    "bewqc0glhgpq": {
        "cv_export.json": "93ccab69ae6520ac509a9f71ce5290d240b7c3115832866c2470eda6eb66a6c9",
        "api_get_metadata.json": "9069ae843ca7740a15c7e03304570488782891f92da9c3a28423b4e4697f40bc",
        "video_provenance.json": "19293a2370d2972e34f58e27207d08977da93b8043f1fc2763bd44a10da2f8ee",
    },
    "st0epgnab7dr": {
        "cv_export.json": "5cf81d9aaaf1c4f3271f60449a7564fcc25f18cca71fc823b5534e9bcd864296",
        "api_get_metadata.json": "819099065af01bd14c1cfab1a46bdbd7982a33d595878c25da9da0401841a4d6",
        "video_provenance.json": "8d144223916599c81fff00d02068fb6c599c366abe91b792b7c3b09046277ca7",
    },
    "td2szayjwtrj": {
        "cv_export.json": "5e28ef09b5f94e39c0698e221b49bf321ba8741daf0daec702ddd7f5121d26ab",
        "api_get_metadata.json": "d8147609eaefc2609198eaf330e3b8a8c89f89dc6829f64c6a02eafb9dff9b27",
        "video_provenance.json": "c09de45f87e9603524294b0f9941b7cae80f2b2e71376f31f63b3e6f6eb35f4a",
    },
    "tqjlrcntpjvt": {
        "cv_export.json": "fc908c56ccec7f64a049a77099f9cf5b9b9e4400197ce3a4a91f3be88420a0ed",
        "api_get_metadata.json": "c7e6ebf5db688a48ad2bd13ee91ae96070e1768c5aadb5cbe2f18d3523f78157",
        "video_provenance.json": "54687a9c9585d34d3fc2c2534bfb7dfc0aa92fdb2ee3196fbeda13a0c42e66f5",
    },
    "xkadsq9bli3h": {
        "cv_export.json": "80b10d494a338d2bede65a136f78e117df76b32c0a4634531435ea78759bfc13",
        "api_get_metadata.json": "03e14d18add52c66497fc5ca10e71cb161d4e8cbc7c6604c8ee5721e3d9d56c7",
        "video_provenance.json": "05fa760ab2f38670c93726af156f96974dfce7d88468dad4a1c8b956d6711dac",
    },
}

TEMPORAL_GEOMETRY_POLICY: dict[str, Any] = {
    "policy_id": "frozen_wasb_temporal_bridge_v2",
    "independent_verifier": "pinned_frozen_wasb",
    "teacher_confidence_min": PRODUCTION_TEACHER_CONFIDENCE_MIN,
    "wasb_confidence_min": PRODUCTION_TEACHER_CONFIDENCE_MIN,
    "anchor_agreement_radius_px": PRODUCTION_AGREEMENT_RADIUS_PX,
    "interpolation_residual_max_px": PRODUCTION_AGREEMENT_RADIUS_PX,
    "max_gap_source_frames": PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
    "gap_length_semantics": "total_consecutive_teacher_only_interior_frames",
    "requires_bracketing_wasb_agreement_anchors": True,
    "requires_every_interior_frame_teacher_only": True,
    "contradictory_high_confidence_wasb_is_search_barrier": True,
    "requires_current_wasb_absent_invisible_or_below_threshold": True,
    "requires_current_wasb_status_evidence": True,
    "image_bounds_required_for_teacher_and_wasb_points": True,
    "same_teacher_self_agreement_eligible": False,
}

SAMPLE_SHA_DEPENDENCY_KEYS = frozenset(
    {
        "split_manifest_sha256",
        "pbvision_cv_export_sha256",
        "pbvision_metadata_sha256",
        "pbvision_provenance_sha256",
        "source_video_sha256",
        "frame_times_sha256",
        "wasb_checkpoint_sha256",
        "models_manifest_sha256",
        "builder_code_sha256",
        "wasb_ball_track_sha256",
        "wasb_metadata_sha256",
        "wasb_predictions_csv_sha256",
        "wasb_adapter_code_sha256",
    }
)
SAMPLE_DEPENDENCY_KEYS = SAMPLE_SHA_DEPENDENCY_KEYS | {"wasb_repo_commit"}
WASB_BUILDER_BINDING_KEYS = frozenset(
    {
        "source_video_sha256",
        "frame_times_sha256",
        "wasb_predictions_csv_sha256",
        "wasb_ball_track_sha256",
        "wasb_checkpoint_sha256",
        "wasb_repo_commit",
        "wasb_adapter_code_sha256",
    }
)


class BallSstBuildError(ValueError):
    """Invalid or unsafe B1 input."""


class BallSstDependencyStabilityError(BallSstBuildError):
    """A captured dependency identity diverged during materialization."""


class BallSstSnapshotPathError(BallSstDependencyStabilityError):
    """A downstream consumer was given a path outside its immutable snapshot."""


class BallSstBuildRefusal(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(str(payload.get("verdict") or "BUILD_REFUSED"))
        self.payload = dict(payload)


@dataclass(frozen=True)
class TeacherObservation:
    teacher_frame_index: int
    teacher_time_s: float
    xy_px: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class WasbObservation:
    frame_index: int
    xy_px: tuple[float, float]
    confidence: float
    visible: bool


@dataclass(frozen=True)
class MediaTiming:
    fps: float
    duration_s: float
    pts_s: tuple[float, ...]
    width: int
    height: int


@dataclass(frozen=True)
class ReusableDependencyIdentity:
    metadata: Mapping[str, Any]
    expected_bindings: Mapping[str, str]


@dataclass(frozen=True)
class SnapshotArtifactPaths:
    frame_times: Path
    ball_track: Path
    metadata: Path
    predictions_csv: Path


@dataclass(frozen=True)
class SnapshotDeclarationIdentities:
    gallery_root: str
    media_root: str
    split_manifest: str
    source_video_by_id: Mapping[str, str]
    checkpoint: str
    wasb_repo: str


@dataclass(frozen=True)
class ImmutableDependencySnapshot:
    root: Path
    split_manifest: Path
    gallery_root: Path
    media_root: Path
    checkpoint: Path
    wasb_repo: Path
    models_manifest: Path
    builder_code: Path
    wasb_adapter_code: Path
    reused_artifacts: Mapping[str, SnapshotArtifactPaths]
    declarations: SnapshotDeclarationIdentities


@dataclass(frozen=True)
class SnapshotVerificationContext:
    inputs: ImmutableDependencySnapshot
    artifacts: Mapping[str, SnapshotArtifactPaths]
    publication_artifacts_by_video: Mapping[str, SnapshotArtifactPaths]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_pbvision_ball_sst(
            gallery_root=args.gallery_root,
            media_root=args.media_root,
            split_manifest=args.split_manifest,
            wasb_checkpoint=args.wasb_checkpoint,
            wasb_repo=args.wasb_repo,
            teacher_confidence_min=args.teacher_confidence_min,
            agreement_radius_px=args.agreement_radius_px,
            pseudo_weight=args.pseudo_weight,
            out=args.out,
            device=args.device,
            wasb_batch_size=args.wasb_batch_size,
            resume_dependencies=args.resume_dependencies,
        )
    except BallSstBuildRefusal as exc:
        _write_json(args.out, exc.payload)
        print(json.dumps(_cli_report(exc.payload, out=args.out), indent=2, sort_keys=True))
        missing = exc.payload.get("missing_media") or []
        if missing:
            print(
                "missing pb.vision media: "
                + ", ".join(str(row.get("video_id")) for row in missing if isinstance(row, Mapping)),
                file=sys.stderr,
            )
        return 3
    except Exception as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(_cli_report(manifest, out=args.out), indent=2, sort_keys=True))
    return 0 if manifest["gate"]["verdict"] == "PASS" else 4


def build_pbvision_ball_sst(
    *,
    gallery_root: str | Path,
    media_root: str | Path,
    split_manifest: str | Path,
    wasb_checkpoint: str | Path,
    wasb_repo: str | Path,
    teacher_confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
    out: str | Path,
    device: str = "cuda",
    wasb_batch_size: int = 8,
    resume_dependencies: bool = False,
) -> dict[str, Any]:
    gallery = Path(gallery_root)
    media = Path(media_root)
    split_path = Path(split_manifest)
    out_path = Path(out)
    confidence_min = _probability(teacher_confidence_min, "teacher_confidence_min")
    radius = _positive_float(agreement_radius_px, "agreement_radius_px")
    weight = _probability(pseudo_weight, "pseudo_weight")
    if wasb_batch_size <= 0:
        raise BallSstBuildError("wasb_batch_size must be positive")

    policy_overrides = _production_policy_overrides(
        teacher_confidence_min=confidence_min,
        agreement_radius_px=radius,
        pseudo_weight=weight,
    )
    production_policy_selected = not policy_overrides
    builder_identity = _builder_identity()

    split_sha256 = _sha256_file(split_path)
    wasb_identity = _resolve_wasb_identity(
        checkpoint_path=Path(wasb_checkpoint),
        repo_path=Path(wasb_repo),
        require_production_identity=production_policy_selected,
    )
    media_paths = {video_id: _discover_media(media, video_id) for video_id in TRAIN_IDS}
    media_hashes: dict[str, str] = {}
    canonical_media_root: Path | None = None
    if any(path is not None for path in media_paths.values()):
        canonical_media_root = _canonical_root(media, field="media_root")
        for video_id, source_media in media_paths.items():
            if source_media is not None:
                media_hashes[video_id] = _validate_media_identity(
                    source_media,
                    video_id=video_id,
                    media_root=canonical_media_root,
                )
    missing_media = [
        {
            "video_id": video_id,
            "expected_source_video_sha256": EXPECTED_SOURCE_VIDEO_SHA256[video_id],
            "searched_path": str(_canonical_media_path(media, video_id)),
        }
        for video_id, path in media_paths.items()
        if path is None
    ]
    if missing_media:
        raise BallSstBuildRefusal(
            _missing_media_refusal(
                gallery_root=gallery,
                media_root=media,
                split_manifest=split_path,
                split_manifest_sha256=split_sha256,
                wasb_checkpoint=Path(wasb_checkpoint),
                confidence_min=confidence_min,
                agreement_radius_px=radius,
                pseudo_weight=weight,
                policy_overrides=policy_overrides,
                builder_identity=builder_identity,
                wasb_identity=wasb_identity,
                missing_media=missing_media,
            )
        )

    checkpoint_sha256 = str(wasb_identity["checkpoint_sha256"])
    gallery = _canonical_gallery_root(gallery, require_production_identity=production_policy_selected)
    media = canonical_media_root or _canonical_root(media, field="media_root")
    production_gallery_hashes_by_source = (
        _validate_production_gallery_inventory(gallery)
        if production_policy_selected
        else {}
    )

    dependency_root = out_path.parent / f"{out_path.stem}_dependencies"
    snapshot = _create_immutable_dependency_snapshot(
        out_path=out_path,
        gallery_root=gallery,
        media_root=media,
        split_manifest=split_path,
        split_manifest_sha256=split_sha256,
        dependency_root=dependency_root,
        builder_identity=builder_identity,
        wasb_identity=wasb_identity,
        media_sha256_by_video=media_hashes,
        gallery_sha256_by_video=production_gallery_hashes_by_source,
        resume_dependencies=resume_dependencies,
    )
    declaration_gallery = Path(snapshot.declarations.gallery_root)
    declaration_media = Path(snapshot.declarations.media_root)
    declaration_split = Path(snapshot.declarations.split_manifest)
    declaration_checkpoint = Path(snapshot.declarations.checkpoint)
    declaration_wasb_repo = Path(snapshot.declarations.wasb_repo)
    materialization_root = Path(
        tempfile.mkdtemp(
            prefix=f".{out_path.stem}_materialization_",
            dir=str(out_path.parent),
        )
    )
    try:
        snapshot_split = _require_snapshot_path(
            snapshot.root,
            snapshot.split_manifest,
            context="frozen split validation",
        )
        split_payload = _read_json(snapshot_split)
        _validate_frozen_split(split_payload, snapshot_split)

        snapshot_checkpoint = _require_snapshot_path(
            snapshot.root,
            snapshot.checkpoint,
            context="WASB checkpoint consumption",
        )
        snapshot_repo = _require_snapshot_path(
            snapshot.root,
            snapshot.wasb_repo,
            context="WASB repository consumption",
        )
        clips: list[dict[str, Any]] = []
        artifacts_by_video: dict[str, SnapshotArtifactPaths] = {}
        publication_artifacts_by_video: dict[str, SnapshotArtifactPaths] = {}
        decode_failures = 0
        for video_id in TRAIN_IDS:
            source_declaration = Path(
                snapshot.declarations.source_video_by_id[video_id]
            )
            snapshot_source = _require_snapshot_path(
                snapshot.root,
                _canonical_media_path(snapshot.media_root, video_id),
                context=f"{video_id} source-video consumption",
            )
            cv_export_path = _require_snapshot_path(
                snapshot.root,
                _source_file(snapshot.gallery_root, video_id, "cv_export.json"),
                context=f"{video_id} cv_export consumption",
            )
            metadata_path = _require_snapshot_path(
                snapshot.root,
                _source_file(
                    snapshot.gallery_root,
                    video_id,
                    "api_get_metadata.json",
                ),
                context=f"{video_id} metadata consumption",
            )
            provenance_path = _require_snapshot_path(
                snapshot.root,
                _source_file(
                    snapshot.gallery_root,
                    video_id,
                    "video_provenance.json",
                ),
                context=f"{video_id} provenance consumption",
            )
            cv_export = _read_json(cv_export_path)
            metadata = _read_json(metadata_path)
            provenance = _read_json(provenance_path)
            _validate_gallery_provenance(provenance, video_id=video_id)
            gallery_hashes = production_gallery_hashes_by_source.get(video_id) or {
                "cv_export.json": _sha256_file(cv_export_path),
                "api_get_metadata.json": _sha256_file(metadata_path),
                "video_provenance.json": _sha256_file(provenance_path),
            }
            source_width, source_height = _source_dimensions(metadata, video_id)
            teacher_fps = _teacher_fps(cv_export, metadata, video_id)
            teacher = extract_teacher_observations(
                cv_export,
                width=source_width,
                height=source_height,
                teacher_fps=teacher_fps,
            )

            media_sha256 = media_hashes[video_id]
            timing = probe_media_pts(snapshot_source, video_id=video_id)
            if (timing.width, timing.height) != (source_width, source_height):
                raise BallSstBuildError(
                    f"{video_id} gallery dimensions {(source_width, source_height)} differ from "
                    f"decoded media dimensions {(timing.width, timing.height)}"
                )

            source_dir = dependency_root / video_id
            public_artifacts = SnapshotArtifactPaths(
                frame_times=source_dir / "frame_times.json",
                ball_track=source_dir / "wasb_ball_track.json",
                metadata=source_dir / "wasb_ball_track_metadata.json",
                predictions_csv=source_dir / "wasb_predictions.csv",
            )
            publication_artifacts_by_video[video_id] = public_artifacts
            work_dir = materialization_root / video_id
            work_dir.mkdir(parents=True, exist_ok=True)
            work_frame_times = work_dir / "frame_times.json"
            frame_times_payload = _frame_times_payload(
                timing,
                media_sha256=media_sha256,
            )
            _write_json(work_frame_times, frame_times_payload)
            sealed_frame_times_root = materialization_root / "sealed_frame_times" / video_id
            sealed_frame_times = _seal_snapshot_file(
                work_frame_times,
                sealed_frame_times_root / "frame_times.json",
            )
            _seal_snapshot_directory(sealed_frame_times_root)
            sealed_frame_times = _require_snapshot_path(
                sealed_frame_times_root,
                sealed_frame_times,
                context=f"{video_id} generated frame-times consumption",
            )
            frame_times_sha256 = _sha256_file(sealed_frame_times)

            snapshot_reused = snapshot.reused_artifacts.get(video_id)
            reusable_identity = (
                _dependency_artifacts_reusable(
                    frame_times=_require_snapshot_path(
                        snapshot.root,
                        snapshot_reused.frame_times,
                        context=f"{video_id} reused frame-times consumption",
                    ),
                    ball_track=_require_snapshot_path(
                        snapshot.root,
                        snapshot_reused.ball_track,
                        context=f"{video_id} reused ball-track consumption",
                    ),
                    metadata=_require_snapshot_path(
                        snapshot.root,
                        snapshot_reused.metadata,
                        context=f"{video_id} reused metadata consumption",
                    ),
                    predictions_csv=_require_snapshot_path(
                        snapshot.root,
                        snapshot_reused.predictions_csv,
                        context=f"{video_id} reused prediction-CSV consumption",
                    ),
                    source_video=snapshot_source,
                    source_video_sha256=media_sha256,
                    expected_frame_times_sha256=frame_times_sha256,
                    wasb_checkpoint_sha256=checkpoint_sha256,
                    wasb_repo_commit=str(wasb_identity["repo_commit"]),
                    wasb_adapter_code_sha256=str(
                        builder_identity["wasb_adapter_code_sha256"]
                    ),
                )
                if resume_dependencies and snapshot_reused is not None
                else None
            )
            reused = reusable_identity is not None
            if reused:
                assert snapshot_reused is not None
                artifact_paths = snapshot_reused
                run_summary = dict(reusable_identity.metadata)
                run_summary.pop("builder_bindings", None)
            else:
                work_artifacts = SnapshotArtifactPaths(
                    frame_times=sealed_frame_times,
                    ball_track=work_dir / "wasb_ball_track.json",
                    metadata=work_dir / "wasb_ball_track_metadata.json",
                    predictions_csv=work_dir / "wasb_predictions.csv",
                )
                run_summary = run_wasb_or_convert(
                    out=work_artifacts.ball_track,
                    fps=timing.fps,
                    frame_times=sealed_frame_times,
                    metadata_out=work_artifacts.metadata,
                    video=snapshot_source,
                    checkpoint=snapshot_checkpoint,
                    wasb_repo=snapshot_repo,
                    prediction_csv_out=work_artifacts.predictions_csv,
                    batch_size=wasb_batch_size,
                    visible_threshold=confidence_min,
                    device=device,
                    input_preprocessing="official",
                    emit_size_observations=False,
                    emit_below_threshold_candidates=False,
                )
                sealed_outputs_root = materialization_root / "sealed_outputs" / video_id
                sealed_ball_track = _seal_snapshot_file(
                    work_artifacts.ball_track,
                    sealed_outputs_root / "wasb_ball_track.json",
                )
                sealed_predictions_csv = _seal_snapshot_file(
                    work_artifacts.predictions_csv,
                    sealed_outputs_root / "wasb_predictions.csv",
                )
                _seal_snapshot_directory(sealed_outputs_root)
                artifact_paths = SnapshotArtifactPaths(
                    frame_times=sealed_frame_times,
                    ball_track=_require_snapshot_path(
                        sealed_outputs_root,
                        sealed_ball_track,
                        context=f"{video_id} generated ball-track consumption",
                    ),
                    metadata=work_artifacts.metadata,
                    predictions_csv=_require_snapshot_path(
                        sealed_outputs_root,
                        sealed_predictions_csv,
                        context=f"{video_id} generated prediction-CSV consumption",
                    ),
                )
            if resume_dependencies:
                print(json.dumps({"reused": reused, "video_id": video_id}, sort_keys=True))

            run_summary = _normalize_wasb_run_summary_paths(
                run_summary,
                predictions_csv=public_artifacts.predictions_csv,
                ball_track=public_artifacts.ball_track,
                source_video=source_declaration,
                checkpoint=declaration_checkpoint,
                wasb_repo=declaration_wasb_repo,
            )
            actual_frame_times = (
                artifact_paths.frame_times if reused else sealed_frame_times
            )
            wasb_payload = _read_json(artifact_paths.ball_track)
            wasb_rows, wasb_frame_count = extract_wasb_observations(
                wasb_payload,
                pts_s=timing.pts_s,
                fps=timing.fps,
                width=source_width,
                height=source_height,
                visible_threshold=confidence_min,
            )
            _validate_wasb_predictions_csv(
                artifact_paths.predictions_csv,
                pts_s=timing.pts_s,
                width=source_width,
                height=source_height,
            )
            regenerated_track = wasb_csv_to_ball_track(
                artifact_paths.predictions_csv,
                fps=timing.fps,
                frame_times=actual_frame_times,
                visible_threshold=confidence_min,
                input_preprocessing="official",
            )
            if regenerated_track != wasb_payload:
                raise BallSstBuildError(
                    f"{video_id} WASB ball track does not reproduce from its prediction CSV"
                )
            decode_failures += max(0, len(timing.pts_s) - wasb_frame_count)
            wasb_bindings = (
                dict(reusable_identity.expected_bindings)
                if reusable_identity is not None
                else {
                    "source_video_sha256": media_sha256,
                    "frame_times_sha256": _sha256_file(actual_frame_times),
                    "wasb_predictions_csv_sha256": _sha256_file(
                        artifact_paths.predictions_csv
                    ),
                    "wasb_ball_track_sha256": _sha256_file(
                        artifact_paths.ball_track
                    ),
                    "wasb_checkpoint_sha256": checkpoint_sha256,
                    "wasb_repo_commit": str(wasb_identity["repo_commit"]),
                    "wasb_adapter_code_sha256": str(
                        builder_identity["wasb_adapter_code_sha256"]
                    ),
                }
            )
            run_summary = {**run_summary, "builder_bindings": wasb_bindings}
            _validate_wasb_run_metadata(
                run_summary,
                predictions_csv=artifact_paths.predictions_csv,
                ball_track=artifact_paths.ball_track,
                source_video=snapshot_source,
                checkpoint=snapshot_checkpoint,
                wasb_repo=snapshot_repo,
                timing=timing,
                visible_threshold=confidence_min,
                expected_bindings=wasb_bindings,
                declared_predictions_csv=public_artifacts.predictions_csv,
                declared_ball_track=public_artifacts.ball_track,
                declared_source_video=snapshot.declarations.source_video_by_id[
                    video_id
                ],
                declared_checkpoint=snapshot.declarations.checkpoint,
                declared_wasb_repo=snapshot.declarations.wasb_repo,
            )
            work_metadata = work_dir / "published_wasb_ball_track_metadata.json"
            _write_json(work_metadata, run_summary)
            sealed_metadata_root = materialization_root / "sealed_metadata" / video_id
            sealed_metadata = _seal_snapshot_file(
                work_metadata,
                sealed_metadata_root / "wasb_ball_track_metadata.json",
            )
            _seal_snapshot_directory(sealed_metadata_root)
            sealed_metadata = _require_snapshot_path(
                sealed_metadata_root,
                sealed_metadata,
                context=f"{video_id} generated metadata consumption",
            )
            artifact_paths = SnapshotArtifactPaths(
                frame_times=actual_frame_times,
                ball_track=artifact_paths.ball_track,
                metadata=sealed_metadata,
                predictions_csv=artifact_paths.predictions_csv,
            )
            artifacts_by_video[video_id] = artifact_paths

            dependency_hashes = {
                "split_manifest_sha256": split_sha256,
                "pbvision_cv_export_sha256": gallery_hashes["cv_export.json"],
                "pbvision_metadata_sha256": gallery_hashes["api_get_metadata.json"],
                "pbvision_provenance_sha256": gallery_hashes[
                    "video_provenance.json"
                ],
                "source_video_sha256": media_sha256,
                "frame_times_sha256": _sha256_file(artifact_paths.frame_times),
                "wasb_checkpoint_sha256": checkpoint_sha256,
                "wasb_repo_commit": str(wasb_identity["repo_commit"]),
                "models_manifest_sha256": str(
                    wasb_identity["models_manifest_sha256"]
                ),
                "builder_code_sha256": str(builder_identity["builder_code_sha256"]),
                "wasb_adapter_code_sha256": str(
                    builder_identity["wasb_adapter_code_sha256"]
                ),
                "wasb_ball_track_sha256": _sha256_file(
                    artifact_paths.ball_track
                ),
                "wasb_metadata_sha256": _sha256_file(artifact_paths.metadata),
                "wasb_predictions_csv_sha256": _sha256_file(
                    artifact_paths.predictions_csv
                ),
            }
            samples = build_source_samples(
                video_id=video_id,
                video_path=snapshot_source,
                teacher_observations=teacher,
                wasb_observations=wasb_rows,
                pts_s=timing.pts_s,
                width=source_width,
                height=source_height,
                teacher_confidence_min=confidence_min,
                agreement_radius_px=radius,
                pseudo_weight=weight,
                dependency_hashes=dependency_hashes,
            )
            _normalize_sample_video_paths(
                samples,
                source_media=source_declaration,
            )
            clip = {
                "clip_id": video_id,
                "canonical_source_id": video_id,
                "split": "train",
                "teacher_derived": True,
                "ground_truth": False,
                "rally_video": str(source_declaration),
                "source_video_sha256": media_sha256,
                "source_width": source_width,
                "source_height": source_height,
                "fps": timing.fps,
                "sample_count": len(samples),
                "samples": samples,
                "dependencies": {
                    **dependency_hashes,
                    "frame_times_path": str(public_artifacts.frame_times),
                    "wasb_ball_track": str(public_artifacts.ball_track),
                    "wasb_metadata_path": str(public_artifacts.metadata),
                    "wasb_predictions_csv_path": str(
                        public_artifacts.predictions_csv
                    ),
                    "wasb_runtime": run_summary,
                },
            }
            if resume_dependencies:
                clip["dependency_reused"] = reused
            clips.append(clip)

        verification_snapshot = SnapshotVerificationContext(
            inputs=snapshot,
            artifacts=artifacts_by_video,
            publication_artifacts_by_video=publication_artifacts_by_video,
        )
        manifest = assemble_sst_manifest(
            clips=clips,
            gallery_root=declaration_gallery,
            media_root=declaration_media,
            split_manifest=declaration_split,
            split_manifest_sha256=split_sha256,
            wasb_checkpoint=declaration_checkpoint,
            wasb_checkpoint_sha256=checkpoint_sha256,
            wasb_identity=wasb_identity,
            builder_identity=builder_identity,
            teacher_confidence_min=confidence_min,
            agreement_radius_px=radius,
            pseudo_weight=weight,
            policy_overrides=policy_overrides,
            decode_failures=decode_failures,
            dependencies_reused_count=(
                sum(clip["dependency_reused"] is True for clip in clips)
                if resume_dependencies
                else None
            ),
            verification_snapshot=verification_snapshot,
        )
        work_manifest = materialization_root / "sst.json"
        _write_json(work_manifest, manifest)
        sealed_manifest_root = materialization_root / "sealed_manifest"
        sealed_manifest = _seal_snapshot_file(
            work_manifest,
            sealed_manifest_root / out_path.name,
        )
        _seal_snapshot_directory(sealed_manifest_root)
        sealed_manifest = _require_snapshot_path(
            sealed_manifest_root,
            sealed_manifest,
            context="final SST publication",
        )
        for video_id in TRAIN_IDS:
            _publish_snapshot_file(
                artifacts_by_video[video_id].frame_times,
                publication_artifacts_by_video[video_id].frame_times,
            )
            _publish_snapshot_file(
                artifacts_by_video[video_id].ball_track,
                publication_artifacts_by_video[video_id].ball_track,
            )
            _publish_snapshot_file(
                artifacts_by_video[video_id].metadata,
                publication_artifacts_by_video[video_id].metadata,
            )
            _publish_snapshot_file(
                artifacts_by_video[video_id].predictions_csv,
                publication_artifacts_by_video[video_id].predictions_csv,
            )
        _publish_snapshot_file(sealed_manifest, out_path)
        return manifest
    finally:
        _remove_private_tree(materialization_root)
        _remove_private_tree(snapshot.root)


def extract_teacher_observations(
    payload: Mapping[str, Any],
    *,
    width: int,
    height: int,
    teacher_fps: float,
) -> dict[int, TeacherObservation]:
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise BallSstBuildError("pb.vision cv_export sessions must be a list")
    observations: dict[int, TeacherObservation] = {}
    for session in sessions:
        if not isinstance(session, Mapping):
            raise BallSstBuildError("pb.vision session must be an object")
        rallies = session.get("rallies")
        if not isinstance(rallies, list):
            raise BallSstBuildError("pb.vision session rallies must be a list")
        for rally in rallies:
            if not isinstance(rally, Mapping) or not isinstance(rally.get("frames"), list):
                raise BallSstBuildError("pb.vision rally requires frame_index and frames")
            start = int(rally.get("frame_index"))
            for offset, frame in enumerate(rally["frames"]):
                if not isinstance(frame, Mapping):
                    continue
                actions = frame.get("actions")
                ball = actions.get("ball") if isinstance(actions, Mapping) else None
                if not isinstance(ball, Mapping):
                    continue
                confidence = _probability(ball.get("confidence"), "teacher confidence")
                u = _probability(ball.get("u"), "teacher u")
                v = _probability(ball.get("v"), "teacher v")
                frame_index = start + offset
                observation = TeacherObservation(
                    teacher_frame_index=frame_index,
                    teacher_time_s=frame_index / teacher_fps,
                    xy_px=(u * width, v * height),
                    confidence=confidence,
                )
                previous = observations.get(frame_index)
                if previous is None or observation.confidence > previous.confidence:
                    observations[frame_index] = observation
    return observations


def extract_wasb_observations(
    payload: Mapping[str, Any],
    *,
    pts_s: Sequence[float],
    fps: float,
    width: int,
    height: int,
    visible_threshold: float = PRODUCTION_TEACHER_CONFIDENCE_MIN,
) -> tuple[dict[int, WasbObservation], int]:
    expected_top_keys = {
        "schema_version",
        "fps",
        "source",
        "input_preprocessing",
        "frames",
        "bounces",
    }
    if set(payload) != expected_top_keys:
        raise BallSstBuildError("WASB ball track top-level schema is not the official adapter schema")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise BallSstBuildError("WASB ball track schema_version must be integer 1")
    if payload.get("source") != "wasb":
        raise BallSstBuildError("WASB ball track source must be wasb")
    if payload.get("input_preprocessing") != "official":
        raise BallSstBuildError("WASB ball track requires official input preprocessing")
    track_fps = _positive_float(payload.get("fps"), "WASB ball track fps")
    if not math.isclose(track_fps, fps, rel_tol=0.0, abs_tol=1e-12):
        raise BallSstBuildError("WASB ball track fps differs from bound media timing")
    if payload.get("bounces") != []:
        raise BallSstBuildError("raw WASB ball track bounces must be an empty list")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise BallSstBuildError("WASB ball track requires frames list")
    if len(frames) != len(pts_s):
        raise BallSstBuildError("WASB ball track frame count differs from bound PTS")
    observations: dict[int, WasbObservation] = {}
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise BallSstBuildError("WASB ball track frames must be objects")
        if set(frame) != {"t", "xy", "conf", "visible", "approx"}:
            raise BallSstBuildError(
                f"WASB frame {frame_index} schema is not the official adapter frame schema"
            )
        t = _finite_float(frame.get("t"), f"WASB frame {frame_index} t")
        if not math.isclose(t, float(pts_s[frame_index]), rel_tol=0.0, abs_tol=1e-9):
            raise BallSstBuildError(f"WASB frame {frame_index} timestamp differs from bound PTS")
        if frame.get("approx") is not False:
            raise BallSstBuildError(f"WASB frame {frame_index} must be a non-approximate raw output")
        visible = frame.get("visible")
        if not isinstance(visible, bool):
            raise BallSstBuildError(f"WASB frame {frame_index} visible must be a strict boolean")
        xy = _xy_value(frame.get("xy"), f"WASB frame {frame_index} xy")
        confidence = _probability(frame.get("conf"), f"WASB frame {frame_index} conf")
        if visible:
            if confidence < visible_threshold:
                raise BallSstBuildError(
                    f"WASB frame {frame_index} visible confidence is below the run threshold"
                )
            if not _inside_image(xy, width=width, height=height):
                raise BallSstBuildError(f"WASB frame {frame_index} visible point is out of bounds")
        elif xy != (0.0, 0.0):
            raise BallSstBuildError(
                f"WASB frame {frame_index} invisible official output must use zero coordinates"
            )
        observations[frame_index] = WasbObservation(
            frame_index=frame_index,
            xy_px=xy,
            confidence=confidence,
            visible=visible,
        )
    return observations, len(frames)


def _validate_wasb_predictions_csv(
    path: Path,
    *,
    pts_s: Sequence[float],
    width: int,
    height: int,
    visible_threshold: float = PRODUCTION_TEACHER_CONFIDENCE_MIN,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing WASB prediction CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["Frame", "Visibility", "X", "Y", "Confidence"]
        if reader.fieldnames != expected_header:
            raise BallSstBuildError("WASB prediction CSV header is not the official schema")
        rows = list(reader)
    if len(rows) != len(pts_s):
        raise BallSstBuildError("WASB prediction CSV row count differs from bound PTS")
    for index, row in enumerate(rows):
        frame_text = row.get("Frame")
        if frame_text != str(index):
            raise BallSstBuildError(
                f"WASB prediction CSV Frame/{index} must be the contiguous list index"
            )
        visibility_text = row.get("Visibility")
        if visibility_text not in {"0", "1"}:
            raise BallSstBuildError(
                f"WASB prediction CSV Visibility/{index} must be exactly 0 or 1"
            )
        x = _csv_finite_float(row.get("X"), f"WASB prediction CSV X/{index}")
        y = _csv_finite_float(row.get("Y"), f"WASB prediction CSV Y/{index}")
        confidence = _csv_finite_float(
            row.get("Confidence"), f"WASB prediction CSV Confidence/{index}"
        )
        if not 0.0 <= confidence <= 1.0:
            raise BallSstBuildError(
                f"WASB prediction CSV Confidence/{index} must be in [0, 1]"
            )
        visible = visibility_text == "1"
        if visible:
            if confidence < visible_threshold:
                raise BallSstBuildError(
                    f"WASB prediction CSV visible row {index} is below the run threshold"
                )
            if not _inside_image((x, y), width=width, height=height):
                raise BallSstBuildError(
                    f"WASB prediction CSV visible row {index} is out of bounds"
                )
        elif (x, y) != (0.0, 0.0):
            raise BallSstBuildError(
                f"WASB prediction CSV invisible row {index} must use zero coordinates"
            )


def _create_immutable_dependency_snapshot(
    *,
    out_path: Path,
    gallery_root: Path,
    media_root: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    dependency_root: Path,
    builder_identity: Mapping[str, Any],
    wasb_identity: Mapping[str, Any],
    media_sha256_by_video: Mapping[str, str],
    gallery_sha256_by_video: Mapping[str, Mapping[str, str]],
    resume_dependencies: bool,
) -> ImmutableDependencySnapshot:
    """Copy every filesystem dependency, verify the copies, then seal them read-only."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        tempfile.mkdtemp(
            prefix=f".{out_path.stem}_immutable_inputs_",
            dir=str(out_path.parent),
        )
    )
    try:
        snapshot_split = _copy_verified_snapshot_file(
            split_manifest.resolve(strict=True),
            snapshot_root / "split" / split_manifest.name,
            expected_sha256=split_manifest_sha256,
            context="split manifest",
        )
        snapshot_gallery = snapshot_root / "gallery"
        snapshot_media = snapshot_root / "media"
        source_video_declarations: dict[str, str] = {}
        for video_id in TRAIN_IDS:
            source_media = _canonical_media_path(
                media_root,
                video_id,
            ).resolve(strict=True)
            source_video_declarations[video_id] = str(source_media)
            _copy_verified_snapshot_file(
                source_media,
                _canonical_media_path(snapshot_media, video_id),
                expected_sha256=str(media_sha256_by_video[video_id]),
                context=f"{video_id} source video",
            )
            for filename in PRODUCTION_GALLERY_ARTIFACT_FILENAMES:
                source_gallery = _source_file(gallery_root, video_id, filename)
                expected_gallery_sha256 = (
                    gallery_sha256_by_video.get(video_id, {}).get(filename)
                    or _sha256_file(source_gallery)
                )
                _copy_verified_snapshot_file(
                    source_gallery,
                    snapshot_gallery / video_id / filename,
                    expected_sha256=str(expected_gallery_sha256),
                    context=f"{video_id} {filename}",
                )

        original_checkpoint = Path(
            str(wasb_identity["checkpoint_path"])
        ).resolve(strict=True)
        snapshot_checkpoint = _copy_verified_snapshot_file(
            original_checkpoint,
            snapshot_root / "checkpoint" / original_checkpoint.name,
            expected_sha256=str(wasb_identity["checkpoint_sha256"]),
            context="WASB checkpoint",
        )
        original_models_manifest = Path(
            str(wasb_identity["models_manifest_path"])
        ).resolve(strict=True)
        snapshot_models_manifest = _copy_verified_snapshot_file(
            original_models_manifest,
            snapshot_root / "models" / "MANIFEST.json",
            expected_sha256=str(wasb_identity["models_manifest_sha256"]),
            context="models manifest",
        )
        original_builder = Path(__file__).resolve(strict=True)
        snapshot_builder = _copy_verified_snapshot_file(
            original_builder,
            snapshot_root / "code" / "build_pbvision_ball_sst.py",
            expected_sha256=str(builder_identity["builder_code_sha256"]),
            context="builder code",
        )
        original_adapter = (
            ROOT / WASB_ADAPTER_RELATIVE_PATH
        ).resolve(strict=True)
        snapshot_adapter = _copy_verified_snapshot_file(
            original_adapter,
            snapshot_root / "code" / "wasb_adapter.py",
            expected_sha256=str(builder_identity["wasb_adapter_code_sha256"]),
            context="WASB adapter code",
        )

        original_repo = Path(str(wasb_identity["repo_path"])).resolve(strict=True)
        original_repo_commit = _git_output(original_repo, "rev-parse", "HEAD")
        original_repo_status = _git_output(
            original_repo,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        original_repo_tree_sha256 = _directory_tree_sha256(
            original_repo,
            exclude_top_level={".git"},
        )
        snapshot_repo = snapshot_root / "wasb_repo"
        shutil.copytree(original_repo, snapshot_repo, symlinks=False)
        snapshot_repo_commit = _git_output(snapshot_repo, "rev-parse", "HEAD")
        snapshot_repo_status = _git_output(
            snapshot_repo,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        snapshot_repo_tree_sha256 = _directory_tree_sha256(
            snapshot_repo,
            exclude_top_level={".git"},
        )
        if (
            original_repo_commit != str(wasb_identity["repo_commit"])
            or snapshot_repo_commit != original_repo_commit
            or original_repo_status != snapshot_repo_status
            or (original_repo_status == "") is not bool(wasb_identity["repo_clean"])
            or snapshot_repo_tree_sha256 != original_repo_tree_sha256
        ):
            raise BallSstDependencyStabilityError(
                "immutable snapshot verification failed: WASB repository state"
            )

        reused_artifacts: dict[str, SnapshotArtifactPaths] = {}
        if resume_dependencies:
            for video_id in TRAIN_IDS:
                original_dir = dependency_root / video_id
                original_artifacts = SnapshotArtifactPaths(
                    frame_times=original_dir / "frame_times.json",
                    ball_track=original_dir / "wasb_ball_track.json",
                    metadata=original_dir / "wasb_ball_track_metadata.json",
                    predictions_csv=original_dir / "wasb_predictions.csv",
                )
                snapshot_dir = snapshot_root / "reused" / video_id
                copied: dict[str, Path] = {}
                for field_name, source in (
                    ("frame_times", original_artifacts.frame_times),
                    ("ball_track", original_artifacts.ball_track),
                    ("metadata", original_artifacts.metadata),
                    ("predictions_csv", original_artifacts.predictions_csv),
                ):
                    if not source.is_file():
                        continue
                    copied[field_name] = _copy_verified_snapshot_file(
                        source.resolve(strict=True),
                        snapshot_dir / source.name,
                        expected_sha256=_sha256_file(source),
                        context=f"{video_id} persisted {source.name}",
                    )
                if set(copied) == {
                    "frame_times",
                    "ball_track",
                    "metadata",
                    "predictions_csv",
                }:
                    reused_artifacts[video_id] = SnapshotArtifactPaths(**copied)

        _seal_snapshot_directory(snapshot_root)
        return ImmutableDependencySnapshot(
            root=snapshot_root,
            split_manifest=snapshot_split,
            gallery_root=snapshot_gallery,
            media_root=snapshot_media,
            checkpoint=snapshot_checkpoint,
            wasb_repo=snapshot_repo,
            models_manifest=snapshot_models_manifest,
            builder_code=snapshot_builder,
            wasb_adapter_code=snapshot_adapter,
            reused_artifacts=reused_artifacts,
            declarations=SnapshotDeclarationIdentities(
                gallery_root=str(gallery_root),
                media_root=str(media_root),
                split_manifest=str(split_manifest),
                source_video_by_id=source_video_declarations,
                checkpoint=str(original_checkpoint),
                wasb_repo=str(original_repo),
            ),
        )
    except Exception:
        _remove_private_tree(snapshot_root)
        raise


def _copy_verified_snapshot_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    context: str,
) -> Path:
    if source.is_symlink() or not source.is_file():
        raise BallSstDependencyStabilityError(
            f"immutable snapshot verification failed: {context} is not a regular file"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or _sha256_file(destination) != expected_sha256:
        raise BallSstDependencyStabilityError(
            f"immutable snapshot verification failed: {context} SHA-256"
        )
    return destination.resolve(strict=True)


def _directory_tree_sha256(
    root: Path,
    *,
    exclude_top_level: set[str],
) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in exclude_top_level:
            continue
        if path.is_symlink():
            raise BallSstDependencyStabilityError(
                f"immutable snapshot repository contains symlink: {relative}"
            )
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(_sha256_file(path).encode("ascii"))
        elif path.is_dir():
            digest.update(b"directory")
        else:
            raise BallSstDependencyStabilityError(
                f"immutable snapshot repository contains unsupported entry: {relative}"
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _seal_snapshot_file(source: Path, destination: Path) -> Path:
    expected_sha256 = _sha256_file(source)
    sealed = _copy_verified_snapshot_file(
        source.resolve(strict=True),
        destination,
        expected_sha256=expected_sha256,
        context=destination.name,
    )
    executable = bool(sealed.stat().st_mode & 0o111)
    sealed.chmod(0o555 if executable else 0o444)
    return sealed


def _seal_snapshot_directory(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            raise BallSstDependencyStabilityError(
                f"immutable snapshot contains symlink: {path}"
            )
        if path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o555 if executable else 0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _require_snapshot_path(
    snapshot_root: Path,
    path: Path,
    *,
    context: str,
) -> Path:
    try:
        root = snapshot_root.resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BallSstSnapshotPathError(
            f"{context}: non-snapshot dependency path is unreadable"
        ) from exc
    if resolved != root and not resolved.is_relative_to(root):
        raise BallSstSnapshotPathError(
            f"{context}: non-snapshot dependency path {resolved}"
        )
    current = resolved
    while True:
        if current.stat().st_mode & 0o222:
            raise BallSstSnapshotPathError(
                f"{context}: snapshot dependency path is writable: {current}"
            )
        if current == root:
            break
        current = current.parent
    return resolved


def _normalize_wasb_run_summary_paths(
    payload: Mapping[str, Any],
    *,
    predictions_csv: Path,
    ball_track: Path,
    source_video: Path,
    checkpoint: Path,
    wasb_repo: Path,
) -> dict[str, Any]:
    normalized = dict(payload)
    runtime_value = normalized.get("runtime")
    if not isinstance(runtime_value, Mapping):
        raise BallSstBuildError("WASB run metadata requires runtime object")
    runtime = dict(runtime_value)
    checkpoint_value = runtime.get("wasb_checkpoint")
    if not isinstance(checkpoint_value, Mapping):
        raise BallSstBuildError("WASB runtime checkpoint binding is malformed")
    checkpoint_binding = dict(checkpoint_value)
    normalized["predictions_csv"] = str(predictions_csv)
    normalized["out"] = str(ball_track)
    runtime["video"] = str(source_video)
    runtime["wasb_repo"] = str(wasb_repo)
    checkpoint_binding["path"] = str(checkpoint)
    runtime["wasb_checkpoint"] = checkpoint_binding
    normalized["runtime"] = runtime
    return normalized


def _normalize_sample_video_paths(
    samples: Sequence[dict[str, Any]],
    *,
    source_media: Path,
) -> None:
    for sample in samples:
        frame_ref = sample.get("frame_ref")
        if not isinstance(frame_ref, dict):
            raise BallSstBuildError("built sample requires frame_ref object")
        frame_ref["video"] = str(source_media)


def _publish_snapshot_file(source: Path, destination: Path) -> None:
    if source.stat().st_mode & 0o222:
        raise BallSstSnapshotPathError(
            f"publication source is not an immutable snapshot file: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        dir=str(destination.parent),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        if _sha256_file(temporary) != _sha256_file(source):
            raise BallSstDependencyStabilityError(
                f"publication copy verification failed: {destination}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_private_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif not path.is_symlink():
                path.chmod(0o600)
        except FileNotFoundError:
            pass
    root.chmod(0o700)
    shutil.rmtree(root)


def _dependency_artifacts_reusable(
    *,
    frame_times: Path,
    ball_track: Path,
    metadata: Path,
    predictions_csv: Path,
    source_video: Path,
    source_video_sha256: str,
    expected_frame_times_sha256: str,
    wasb_checkpoint_sha256: str,
    wasb_repo_commit: str,
    wasb_adapter_code_sha256: str,
) -> ReusableDependencyIdentity | None:
    """Accept reuse only from one already-sealed immutable dependency snapshot."""

    dependency_paths = (frame_times, ball_track, metadata, predictions_csv)
    paths_by_identity = {
        "source_video_sha256": source_video,
        "frame_times_sha256": frame_times,
        "wasb_predictions_csv_sha256": predictions_csv,
        "wasb_ball_track_sha256": ball_track,
        "wasb_metadata_sha256": metadata,
    }
    try:
        if any(not path.is_file() or path.stat().st_size <= 0 for path in dependency_paths):
            return None
        checked_sha256 = {
            key: _sha256_file(path) for key, path in paths_by_identity.items()
        }
        payload = _read_json(metadata)
        expected_bindings = {
            "source_video_sha256": source_video_sha256,
            "frame_times_sha256": checked_sha256["frame_times_sha256"],
            "wasb_predictions_csv_sha256": checked_sha256[
                "wasb_predictions_csv_sha256"
            ],
            "wasb_ball_track_sha256": checked_sha256["wasb_ball_track_sha256"],
            "wasb_checkpoint_sha256": wasb_checkpoint_sha256,
            "wasb_repo_commit": wasb_repo_commit,
            "wasb_adapter_code_sha256": wasb_adapter_code_sha256,
        }
        if checked_sha256["source_video_sha256"] != source_video_sha256:
            return None
        if checked_sha256["frame_times_sha256"] != expected_frame_times_sha256:
            return None
        if not _dependency_builder_bindings_match(payload, expected_bindings):
            return None
        return ReusableDependencyIdentity(
            metadata=dict(payload),
            expected_bindings=expected_bindings,
        )
    except Exception:
        return None


def _dependency_builder_bindings_match(
    metadata: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> bool:
    bindings = metadata.get("builder_bindings")
    return (
        isinstance(bindings, Mapping)
        and WASB_BUILDER_BINDING_KEYS.issubset(bindings)
        and all(bindings.get(key) == value for key, value in expected_bindings.items())
    )


def _validate_wasb_run_metadata(
    payload: Mapping[str, Any],
    *,
    predictions_csv: Path,
    ball_track: Path,
    source_video: Path,
    checkpoint: Path,
    wasb_repo: Path,
    timing: MediaTiming,
    visible_threshold: float,
    expected_bindings: Mapping[str, str],
    declared_predictions_csv: str | Path | None = None,
    declared_ball_track: str | Path | None = None,
    declared_source_video: str | Path | None = None,
    declared_checkpoint: str | Path | None = None,
    declared_wasb_repo: str | Path | None = None,
) -> None:
    expected_top_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "source_mode",
        "predictions_csv",
        "out",
        "fps",
        "frame_count",
        "visible_frame_count",
        "confidence_semantics",
        "visible_threshold",
        "input_preprocessing",
        "non_promotable_measurement_mode",
        "not_ground_truth",
        "official_repo_url",
        "official_model_zoo_url",
        "runtime",
        "builder_bindings",
    }
    if set(payload) != expected_top_keys:
        raise BallSstBuildError("WASB run metadata schema is not the official builder-bound schema")
    if isinstance(payload.get("schema_version"), bool) or payload.get("schema_version") != 1:
        raise BallSstBuildError("WASB run metadata schema_version must be integer 1")
    expected_scalars = {
        "artifact_type": "racketsport_wasb_ball_run",
        "status": STATUS_TESTED,
        "source_mode": "wasb_predict",
        "confidence_semantics": WASB_CONFIDENCE_SEMANTICS,
        "input_preprocessing": "official",
        "non_promotable_measurement_mode": False,
        "not_ground_truth": True,
        "official_repo_url": WASB_REPO_URL,
        "official_model_zoo_url": WASB_MODEL_ZOO_URL,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            raise BallSstBuildError(f"WASB run metadata {key} is not production-authentic")
    _require_consumed_or_recorded_path_identity(
        payload.get("predictions_csv"),
        consumed=predictions_csv,
        recorded=declared_predictions_csv,
        field="WASB predictions_csv",
    )
    _require_consumed_or_recorded_path_identity(
        payload.get("out"),
        consumed=ball_track,
        recorded=declared_ball_track,
        field="WASB out",
    )
    metadata_fps = _positive_float(payload.get("fps"), "WASB metadata fps")
    if not math.isclose(metadata_fps, timing.fps, rel_tol=0.0, abs_tol=1e-12):
        raise BallSstBuildError("WASB run metadata fps differs from bound timing")
    metadata_threshold = _probability(
        payload.get("visible_threshold"), "WASB metadata visible_threshold"
    )
    if metadata_threshold != visible_threshold:
        raise BallSstBuildError("WASB run metadata visible threshold mismatch")
    frame_count = _strict_nonnegative_int(payload.get("frame_count"), "WASB metadata frame_count")
    if frame_count != len(timing.pts_s):
        raise BallSstBuildError("WASB run metadata frame count differs from bound PTS")
    visible_frame_count = _strict_nonnegative_int(
        payload.get("visible_frame_count"), "WASB metadata visible_frame_count"
    )
    track_payload = _read_json(ball_track)
    actual_visible_count = sum(
        1
        for frame in track_payload.get("frames", [])
        if isinstance(frame, Mapping) and frame.get("visible") is True
    )
    if visible_frame_count != actual_visible_count:
        raise BallSstBuildError("WASB run metadata visible count differs from ball track")
    if payload.get("builder_bindings") != dict(expected_bindings):
        raise BallSstBuildError("WASB run metadata builder bindings mismatch")

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BallSstBuildError("WASB run metadata requires runtime object")
    expected_runtime_keys = {
        "wasb_repo",
        "wasb_repo_commit",
        "wasb_checkpoint",
        "video",
        "source_video_fps",
        "source_video_frame_count",
        "source_video_size",
        "processed_frame_count",
        "processed_window_count",
        "read_frame_count",
        "video_range_seconds",
        "max_frames",
        "batch_size",
        "device",
        "input_preprocessing",
        "non_promotable_measurement_mode",
        "wall_seconds",
        "effective_fps",
        "realtime_factor",
    }
    if set(runtime) != expected_runtime_keys:
        raise BallSstBuildError("WASB runtime schema is not the bounded official-inference schema")
    _require_consumed_or_recorded_path_identity(
        runtime.get("wasb_repo"),
        consumed=wasb_repo,
        recorded=declared_wasb_repo,
        field="WASB runtime repo",
    )
    _require_consumed_or_recorded_path_identity(
        runtime.get("video"),
        consumed=source_video,
        recorded=declared_source_video,
        field="WASB runtime source video",
    )
    if runtime.get("wasb_repo_commit") != expected_bindings.get("wasb_repo_commit"):
        raise BallSstBuildError("WASB runtime repo commit mismatch")
    checkpoint_binding = runtime.get("wasb_checkpoint")
    if not isinstance(checkpoint_binding, Mapping) or set(checkpoint_binding) != {"path", "sha256"}:
        raise BallSstBuildError("WASB runtime checkpoint binding is malformed")
    _require_consumed_or_recorded_path_identity(
        checkpoint_binding.get("path"),
        consumed=checkpoint,
        recorded=declared_checkpoint,
        field="WASB runtime checkpoint",
    )
    if checkpoint_binding.get("sha256") != expected_bindings.get("wasb_checkpoint_sha256"):
        raise BallSstBuildError("WASB runtime checkpoint SHA mismatch")
    runtime_fps = _positive_float(runtime.get("source_video_fps"), "WASB runtime source fps")
    if not math.isclose(runtime_fps, timing.fps, rel_tol=0.0, abs_tol=1e-6):
        raise BallSstBuildError("WASB runtime source fps differs from bound timing")
    expected_counts = {
        "source_video_frame_count": len(timing.pts_s),
        "processed_frame_count": len(timing.pts_s),
        "processed_window_count": len(timing.pts_s) - 2,
        "read_frame_count": len(timing.pts_s),
    }
    for key, expected in expected_counts.items():
        if _strict_nonnegative_int(runtime.get(key), f"WASB runtime {key}") != expected:
            raise BallSstBuildError(f"WASB runtime {key} mismatch")
    if runtime.get("source_video_size") != [timing.width, timing.height]:
        raise BallSstBuildError("WASB runtime source-video dimensions mismatch")
    if runtime.get("video_range_seconds") is not None or runtime.get("max_frames") is not None:
        raise BallSstBuildError("production WASB runtime must cover the full source video")
    _strict_positive_int(runtime.get("batch_size"), "WASB runtime batch_size")
    if runtime.get("device") not in {"cpu", "cuda"}:
        raise BallSstBuildError("WASB runtime device is invalid")
    if runtime.get("input_preprocessing") != "official":
        raise BallSstBuildError("WASB runtime did not use official preprocessing")
    if runtime.get("non_promotable_measurement_mode") is not False:
        raise BallSstBuildError("WASB runtime is marked non-promotable")
    wall_seconds = _positive_float(runtime.get("wall_seconds"), "WASB runtime wall_seconds")
    effective_fps = _positive_float(runtime.get("effective_fps"), "WASB runtime effective_fps")
    realtime_factor = _positive_float(
        runtime.get("realtime_factor"), "WASB runtime realtime_factor"
    )
    expected_effective_fps = len(timing.pts_s) / wall_seconds
    if not math.isclose(effective_fps, expected_effective_fps, rel_tol=1e-12, abs_tol=1e-12):
        raise BallSstBuildError("WASB runtime effective_fps is not reproducible")
    if not math.isclose(
        realtime_factor,
        expected_effective_fps / timing.fps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BallSstBuildError("WASB runtime realtime_factor is not reproducible")


def _require_path_identity(value: Any, expected: Path, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise BallSstBuildError(f"{field} must be a path string")
    if Path(value).resolve(strict=False) != expected.resolve(strict=False):
        raise BallSstBuildError(f"{field} path mismatch")


def _require_consumed_or_recorded_path_identity(
    value: Any,
    *,
    consumed: Path,
    recorded: str | Path | None,
    field: str,
) -> None:
    if recorded is None:
        _require_path_identity(value, consumed, field)
        return
    if not isinstance(value, str) or not value:
        raise BallSstBuildError(f"{field} must be a path string")
    if value != str(recorded):
        raise BallSstBuildError(f"{field} path mismatch")


def _csv_finite_float(value: Any, field: str) -> float:
    if not isinstance(value, str) or not value:
        raise BallSstBuildError(f"{field} must be a numeric string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise BallSstBuildError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise BallSstBuildError(f"{field} must be finite")
    return parsed


def eligibility_decision(
    current: TeacherObservation | None,
    *,
    teacher_observations: Mapping[int, TeacherObservation],
    wasb: WasbObservation | None,
    width: int,
    height: int,
    teacher_confidence_min: float = PRODUCTION_TEACHER_CONFIDENCE_MIN,
    agreement_radius_px: float = PRODUCTION_AGREEMENT_RADIUS_PX,
    source_frame_index: int | None = None,
    teacher_by_source_frame: Mapping[int, TeacherObservation] | None = None,
    wasb_observations: Mapping[int, WasbObservation] | None = None,
    temporal_max_gap_source_frames: int = PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Return agreement eligibility without treating teacher self-agreement as independent.

    ``teacher_observations`` remains in the signature for API compatibility and audit
    visibility, but it is deliberately not used to establish temporal independence.
    Only ``wasb_observations`` can provide the two bridge anchors.
    """

    if current is None:
        return False, None, {"rejection": "teacher_absent_ignored_not_negative"}
    if not 0.0 <= current.confidence <= 1.0:
        return False, None, {"rejection": "teacher_confidence_not_probability"}
    if current.confidence < teacher_confidence_min:
        return False, None, {"rejection": "teacher_low_confidence"}
    if not _inside_image(current.xy_px, width=width, height=height):
        return False, None, {"rejection": "teacher_out_of_image_bounds"}
    if wasb is not None and not 0.0 <= wasb.confidence <= 1.0:
        return False, None, {"rejection": "wasb_confidence_not_probability"}
    if wasb is not None and wasb.visible and wasb.confidence >= teacher_confidence_min:
        if not _inside_image(wasb.xy_px, width=width, height=height):
            return False, None, {"rejection": "wasb_out_of_image_bounds"}
        evidence_frame_index = (
            wasb.frame_index if source_frame_index is None else source_frame_index
        )
        if wasb.frame_index != evidence_frame_index:
            return False, None, {"rejection": "wasb_source_frame_mismatch"}
        distance = _distance(current.xy_px, wasb.xy_px)
        if distance <= agreement_radius_px:
            return True, "frozen_wasb_spatial", {
                "policy_id": "frozen_wasb_spatial_v2",
                "source_frame_index": evidence_frame_index,
                "teacher_xy": [float(current.xy_px[0]), float(current.xy_px[1])],
                "wasb_xy": [float(wasb.xy_px[0]), float(wasb.xy_px[1])],
                "teacher_confidence": float(current.confidence),
                "wasb_confidence": float(wasb.confidence),
                "distance_px": distance,
                "agreement_radius_px": agreement_radius_px,
                "image_width": int(width),
                "image_height": int(height),
                "all_points_in_bounds": True,
            }
        return False, None, {"rejection": "high_confidence_wasb_disagreement"}

    temporal = _temporal_wasb_bridge_evidence(
        current,
        source_frame_index=source_frame_index,
        teacher_by_source_frame=teacher_by_source_frame,
        wasb_observations=wasb_observations,
        current_wasb=wasb,
        width=width,
        height=height,
        confidence_min=teacher_confidence_min,
        agreement_radius_px=agreement_radius_px,
        max_gap_source_frames=temporal_max_gap_source_frames,
    )
    if temporal is not None:
        return True, str(TEMPORAL_GEOMETRY_POLICY["policy_id"]), temporal
    return False, None, {"rejection": "no_independent_wasb_agreement"}


def build_source_samples(
    *,
    video_id: str,
    video_path: Path,
    teacher_observations: Mapping[int, TeacherObservation],
    wasb_observations: Mapping[int, WasbObservation],
    pts_s: Sequence[float],
    width: int,
    height: int,
    teacher_confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
    dependency_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    _assert_training_source_id(video_id)
    teacher_by_source_frame: dict[int, TeacherObservation] = {}
    for observation in teacher_observations.values():
        source_frame = _nearest_pts_frame(observation.teacher_time_s, pts_s)
        previous = teacher_by_source_frame.get(source_frame)
        if previous is None or observation.confidence > previous.confidence:
            teacher_by_source_frame[source_frame] = observation
    accepted_by_frame: dict[int, dict[str, Any]] = {}
    for teacher_frame in sorted(teacher_observations):
        current = teacher_observations[teacher_frame]
        if current.confidence < teacher_confidence_min:
            continue
        source_frame = _nearest_pts_frame(current.teacher_time_s, pts_s)
        if source_frame <= 0 or source_frame >= len(pts_s) - 1:
            continue
        accepted, reason, evidence = eligibility_decision(
            current,
            teacher_observations=teacher_observations,
            wasb=wasb_observations.get(source_frame),
            width=width,
            height=height,
            teacher_confidence_min=teacher_confidence_min,
            agreement_radius_px=agreement_radius_px,
            source_frame_index=source_frame,
            teacher_by_source_frame=teacher_by_source_frame,
            wasb_observations=wasb_observations,
        )
        if not accepted or reason is None:
            continue
        row = {
            "sample_id": f"{video_id}:{source_frame}",
            "clip_id": video_id,
            "canonical_source_id": video_id,
            "frame_index": source_frame,
            "teacher_frame_index": current.teacher_frame_index,
            "t": float(pts_s[source_frame]),
            "frame_ref": {
                "video": str(video_path),
                "frame_index": source_frame,
                "t": float(pts_s[source_frame]),
                "source_video_sha256": str(dependency_hashes["source_video_sha256"]),
            },
            "source_video_sha256": str(dependency_hashes["source_video_sha256"]),
            "teacher_xy": [float(current.xy_px[0]), float(current.xy_px[1])],
            "score": float(current.confidence),
            "weight": float(pseudo_weight),
            "weight_policy": "fixed_low_weight_pbvision_teacher",
            "teacher_source": "pbvision_actions_ball",
            "teacher_derived": True,
            "ground_truth": False,
            "ball_present": True,
            "agreement_reason": reason,
            "agreement": evidence,
            "dependency_hashes": dict(dependency_hashes),
        }
        previous = accepted_by_frame.get(source_frame)
        if previous is None or row["score"] > previous["score"]:
            accepted_by_frame[source_frame] = row
    return [accepted_by_frame[index] for index in sorted(accepted_by_frame)]


def assemble_sst_manifest(
    *,
    clips: Sequence[Mapping[str, Any]],
    gallery_root: Path,
    media_root: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    wasb_checkpoint: Path,
    wasb_checkpoint_sha256: str,
    wasb_identity: Mapping[str, Any],
    builder_identity: Mapping[str, str],
    teacher_confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
    policy_overrides: Sequence[str],
    decode_failures: int,
    dependencies_reused_count: int | None = None,
    verification_snapshot: SnapshotVerificationContext | None = None,
) -> dict[str, Any]:
    computed_policy_overrides = _production_policy_overrides(
        teacher_confidence_min=teacher_confidence_min,
        agreement_radius_px=agreement_radius_px,
        pseudo_weight=pseudo_weight,
    )
    declared_policy_overrides = list(policy_overrides)
    policy_declaration_matches = declared_policy_overrides == computed_policy_overrides
    clip_rows = [dict(clip) for clip in clips]
    if dependencies_reused_count is not None:
        reused_count = _strict_nonnegative_int(
            dependencies_reused_count, "dependencies_reused_count"
        )
        reuse_values = [clip.get("dependency_reused") for clip in clip_rows]
        if any(type(value) is not bool for value in reuse_values):
            raise BallSstBuildError(
                "resume-enabled clips require boolean dependency_reused telemetry"
            )
        if reused_count != sum(value is True for value in reuse_values):
            raise BallSstBuildError(
                "dependencies_reused_count differs from per-clip reuse telemetry"
            )
    samples = _validate_manifest_clips(
        clip_rows,
        media_root=media_root,
        teacher_confidence_min=teacher_confidence_min,
        agreement_radius_px=agreement_radius_px,
        pseudo_weight=pseudo_weight,
        split_manifest_sha256=split_manifest_sha256,
        wasb_checkpoint_sha256=wasb_checkpoint_sha256,
        wasb_repo_commit=str(wasb_identity.get("repo_commit") or ""),
        models_manifest_sha256=str(wasb_identity.get("models_manifest_sha256") or ""),
        builder_code_sha256=str(builder_identity.get("builder_code_sha256") or ""),
        wasb_adapter_code_sha256=str(
            builder_identity.get("wasb_adapter_code_sha256") or ""
        ),
        recorded_source_video_by_id=(
            verification_snapshot.inputs.declarations.source_video_by_id
            if verification_snapshot is not None
            else None
        ),
    )
    holdout_rows = [
        sample
        for sample in samples
        if str(sample.get("clip_id")) in ALL_NONTRAIN_IDS
        or str(sample.get("clip_id")) not in TRAIN_IDS
    ]
    accepted_windows = len(samples)
    accepted_sources = len({str(sample["canonical_source_id"]) for sample in samples})
    complete_source_inventory = (
        len(clip_rows) == len(TRAIN_IDS)
        and {str(clip.get("clip_id")) for clip in clip_rows} == set(TRAIN_IDS)
    )
    artifact_verification: dict[str, Any] = {
        "verified": False,
        "status": "not_attempted",
        "reason": "production structural prerequisites did not pass",
    }
    structural_prerequisites = (
        not computed_policy_overrides
        and policy_declaration_matches
        and str(wasb_checkpoint_sha256) == PRODUCTION_WASB_CHECKPOINT_SHA256
        and complete_source_inventory
    )
    if structural_prerequisites:
        artifact_verification = _verify_production_artifacts(
            clips=clip_rows,
            gallery_root=gallery_root,
            media_root=media_root,
            split_manifest=split_manifest,
            split_manifest_sha256=split_manifest_sha256,
            wasb_checkpoint=wasb_checkpoint,
            wasb_identity=wasb_identity,
            builder_identity=builder_identity,
            verification_snapshot=verification_snapshot,
        )
    production_eligible = (
        structural_prerequisites and artifact_verification.get("verified") is True
    )
    production_ineligibility_reasons: list[str] = []
    if computed_policy_overrides:
        production_ineligibility_reasons.append(
            "policy overrides: " + ",".join(computed_policy_overrides)
        )
    if not policy_declaration_matches:
        production_ineligibility_reasons.append("caller policy override declaration mismatch")
    if not complete_source_inventory:
        production_ineligibility_reasons.append("incomplete canonical seven-source inventory")
    if artifact_verification.get("verified") is not True:
        production_ineligibility_reasons.append(
            "artifact verification: " + str(artifact_verification.get("reason"))
        )
    passed = (
        production_eligible
        and accepted_windows >= ACCEPTED_WINDOW_TARGET
        and accepted_sources >= ACCEPTED_SOURCE_TARGET
        and not holdout_rows
        and decode_failures == 0
    )
    if not production_eligible:
        verdict = "NON_PRODUCTION_MANIFEST"
    elif passed:
        verdict = "PASS"
    else:
        verdict = "PBV_BALL_INSUFFICIENT_AGREEMENT"
    teacher_input_authority = _production_gallery_authority_payload()
    preregistration = {
        "policy_id": PRODUCTION_POLICY_ID,
        "teacher_confidence_min": PRODUCTION_TEACHER_CONFIDENCE_MIN,
        "agreement_radius_px": PRODUCTION_AGREEMENT_RADIUS_PX,
        "pseudo_weight": PRODUCTION_PSEUDO_WEIGHT,
        "temporal_max_gap_source_frames": PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
        "temporal_geometry": dict(TEMPORAL_GEOMETRY_POLICY),
        "canonical_media_relative_path": "<video_id>/max.mp4",
        "expected_source_video_sha256": {
            video_id: EXPECTED_SOURCE_VIDEO_SHA256[video_id] for video_id in TRAIN_IDS
        },
        "teacher_input_authority": teacher_input_authority,
        "builder_code_sha256": str(builder_identity["builder_code_sha256"]),
        "builder_git_commit": str(builder_identity["builder_git_commit"]),
        "wasb_adapter_code_sha256": str(
            builder_identity["wasb_adapter_code_sha256"]
        ),
        "wasb_adapter_git_commit": str(
            builder_identity["wasb_adapter_git_commit"]
        ),
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "production_eligible": production_eligible,
        "production_policy_selected": not computed_policy_overrides,
        "policy_override_fields": computed_policy_overrides,
        "declared_policy_override_fields": declared_policy_overrides,
        "policy_override_declaration_matches": policy_declaration_matches,
        "production_ineligibility_reasons": production_ineligibility_reasons,
        "artifact_verification": artifact_verification,
        "ball_verified": False,
        "promotion_claimed": False,
        "protected_eval_clips_touched": False,
        "teacher_derived": True,
        "ground_truth": False,
        "teacher": "pbvision_actions_ball_agreement_gated_by_pinned_frozen_wasb",
        "teacher_input_authority": teacher_input_authority,
        "weight_policy": "fixed pseudo weight after eligibility; trainer applies a separate per-step loss cap",
        "gallery_root": str(gallery_root),
        "media_root": str(media_root),
        "split_manifest": str(split_manifest),
        "wasb_checkpoint": str(wasb_checkpoint),
        "source_policy": {
            "train_ids": list(TRAIN_IDS),
            "teacher_validation_only_ids": list(TEACHER_VAL_ONLY_IDS),
            "teacher_test_only_ids": list(TEACHER_TEST_ONLY_IDS),
            "compare_only_ids": list(COMPARE_ONLY_IDS),
            "compare_only_read_policy": "hard_refusal_before_source_path_construction",
            "teacher_absence_policy": "ignored_never_negative",
            "positive_rows_only": True,
            "protected_eval_policy": "never read; protected clips remain evaluation-only",
        },
        "preregistration": preregistration,
        "preregistered_parameters": preregistration,
        "requested_parameters": {
            "teacher_confidence_min": teacher_confidence_min,
            "agreement_radius_px": agreement_radius_px,
            "pseudo_weight": pseudo_weight,
        },
        "builder_identity": dict(builder_identity),
        "wasb_identity": dict(wasb_identity),
        "dependency_hashes": {
            "split_manifest_sha256": split_manifest_sha256,
            "models_manifest_sha256": str(wasb_identity["models_manifest_sha256"]),
            "builder_code_sha256": str(builder_identity["builder_code_sha256"]),
            "wasb_adapter_code_sha256": str(
                builder_identity["wasb_adapter_code_sha256"]
            ),
            "wasb_checkpoint_sha256": wasb_checkpoint_sha256,
            "wasb_repo_commit": str(wasb_identity["repo_commit"]),
        },
        "accepted_windows": accepted_windows,
        "accepted_sources": accepted_sources,
        "holdout_rows_present": len(holdout_rows),
        "decode_status": "completed",
        "decode_failures": int(decode_failures),
        "summary": {
            "clip_count": len(clip_rows),
            "sample_count": accepted_windows,
            "accepted_windows": accepted_windows,
            "accepted_sources": accepted_sources,
            "holdout_rows_present": len(holdout_rows),
            "decode_failures": int(decode_failures),
            "production_eligible": production_eligible,
            "complete_source_inventory": complete_source_inventory,
            "artifacts_verified": artifact_verification.get("verified") is True,
            "agreement_reason_counts": _counts(str(sample["agreement_reason"]) for sample in samples),
        },
        "gate": {
            "verdict": verdict,
            "production_eligible": {"after": production_eligible, "target": True},
            "artifacts_verified": {
                "after": artifact_verification.get("verified") is True,
                "target": True,
            },
            "accepted_windows": {"after": accepted_windows, "target": ACCEPTED_WINDOW_TARGET},
            "accepted_sources": {"after": accepted_sources, "target": ACCEPTED_SOURCE_TARGET},
            "holdout_rows_present": {"after": len(holdout_rows), "target": 0},
            "decode_failures": {"after": int(decode_failures), "target": 0},
        },
        "clips": clip_rows,
    }
    if dependencies_reused_count is not None:
        manifest["dependencies_reused_count"] = dependencies_reused_count
    return manifest


def _validate_manifest_clips(
    clips: Sequence[Mapping[str, Any]],
    *,
    media_root: Path,
    teacher_confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
    split_manifest_sha256: str,
    wasb_checkpoint_sha256: str,
    wasb_repo_commit: str,
    models_manifest_sha256: str,
    builder_code_sha256: str,
    wasb_adapter_code_sha256: str,
    recorded_source_video_by_id: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild the manifest counts from authenticated rows; never trust caller counts."""

    required_dependencies = {
        "split_manifest_sha256": split_manifest_sha256,
        "wasb_checkpoint_sha256": wasb_checkpoint_sha256,
        "wasb_repo_commit": wasb_repo_commit,
        "models_manifest_sha256": models_manifest_sha256,
        "builder_code_sha256": builder_code_sha256,
        "wasb_adapter_code_sha256": wasb_adapter_code_sha256,
    }
    sha_dependency_keys = SAMPLE_SHA_DEPENDENCY_KEYS
    all_required_keys = SAMPLE_DEPENDENCY_KEYS
    seen_clips: set[str] = set()
    seen_samples: set[str] = set()
    validated: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(clips):
        clip_id = str(clip.get("clip_id") or "")
        _assert_training_source_id(clip_id)
        if clip_id in seen_clips:
            raise BallSstBuildError(f"duplicate SST clip_id: {clip_id}")
        seen_clips.add(clip_id)
        if clip.get("canonical_source_id") != clip_id:
            raise BallSstBuildError(f"clip {clip_id} canonical_source_id mismatch")
        if clip.get("split") != "train":
            raise BallSstBuildError(f"clip {clip_id} must remain train-only")
        if clip.get("teacher_derived") is not True or clip.get("ground_truth") is not False:
            raise BallSstBuildError(f"clip {clip_id} authority flags are invalid")
        recorded_source_video = (
            recorded_source_video_by_id.get(clip_id)
            if recorded_source_video_by_id is not None
            else None
        )
        if recorded_source_video is None:
            expected_media = _canonical_media_path(
                media_root,
                clip_id,
            ).resolve(strict=False)
            rally_video = Path(
                str(clip.get("rally_video") or "")
            ).resolve(strict=False)
            if rally_video != expected_media:
                raise BallSstBuildError(
                    f"clip {clip_id} rally_video is not its canonical max.mp4"
                )
        else:
            _require_consumed_or_recorded_path_identity(
                clip.get("rally_video"),
                consumed=Path(recorded_source_video),
                recorded=recorded_source_video,
                field=f"clip {clip_id} rally_video",
            )
        expected_media_sha = EXPECTED_SOURCE_VIDEO_SHA256[clip_id]
        if clip.get("source_video_sha256") != expected_media_sha:
            raise BallSstBuildError(f"clip {clip_id} source-video SHA is not preregistered")
        source_width = _strict_positive_int(
            clip.get("source_width"), f"clip {clip_id} source_width"
        )
        source_height = _strict_positive_int(
            clip.get("source_height"), f"clip {clip_id} source_height"
        )
        samples = clip.get("samples")
        if not isinstance(samples, list):
            raise BallSstBuildError(f"clip {clip_id} samples must be a list")
        sample_count = clip.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count != len(samples):
            raise BallSstBuildError(f"clip {clip_id} sample_count does not match rows")
        clip_dependencies = clip.get("dependencies")
        if not isinstance(clip_dependencies, Mapping):
            raise BallSstBuildError(f"clip {clip_id} requires dependency bindings")
        for key in all_required_keys:
            if key not in clip_dependencies:
                raise BallSstBuildError(f"clip {clip_id} missing dependency {key}")
        for key, expected in required_dependencies.items():
            if clip_dependencies.get(key) != expected:
                raise BallSstBuildError(f"clip {clip_id} dependency {key} mismatch")
        if clip_dependencies.get("source_video_sha256") != expected_media_sha:
            raise BallSstBuildError(f"clip {clip_id} dependency source-video SHA mismatch")
        for key in sha_dependency_keys:
            _require_sha256(clip_dependencies.get(key), f"clip {clip_id} {key}")
        _require_git_commit(clip_dependencies.get("wasb_repo_commit"), f"clip {clip_id} wasb_repo_commit")

        for sample_index, sample_value in enumerate(samples):
            if not isinstance(sample_value, Mapping):
                raise BallSstBuildError(f"clip {clip_id} sample {sample_index} must be an object")
            sample = dict(sample_value)
            frame_index = sample.get("frame_index")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                raise BallSstBuildError(f"clip {clip_id} sample frame_index must be non-negative int")
            sample_id = str(sample.get("sample_id") or "")
            if sample_id != f"{clip_id}:{frame_index}" or sample_id in seen_samples:
                raise BallSstBuildError(f"invalid or duplicate SST sample_id: {sample_id}")
            seen_samples.add(sample_id)
            if sample.get("clip_id") != clip_id or sample.get("canonical_source_id") != clip_id:
                raise BallSstBuildError(f"sample {sample_id} canonical source mismatch")
            if sample.get("teacher_derived") is not True or sample.get("ground_truth") is not False:
                raise BallSstBuildError(f"sample {sample_id} authority flags are invalid")
            if sample.get("ball_present") is not True:
                raise BallSstBuildError(f"sample {sample_id} must be an explicit positive")
            if sample.get("teacher_source") != "pbvision_actions_ball":
                raise BallSstBuildError(f"sample {sample_id} teacher source is invalid")
            score = _probability(sample.get("score"), f"sample {sample_id} score")
            if score < teacher_confidence_min:
                raise BallSstBuildError(f"sample {sample_id} is below the active confidence threshold")
            if _finite_float(sample.get("weight"), f"sample {sample_id} weight") != pseudo_weight:
                raise BallSstBuildError(f"sample {sample_id} pseudo weight mismatch")
            teacher_xy = _xy_value(sample.get("teacher_xy"), f"sample {sample_id} teacher_xy")
            if sample.get("source_video_sha256") != expected_media_sha:
                raise BallSstBuildError(f"sample {sample_id} source-video SHA mismatch")
            frame_ref = sample.get("frame_ref")
            if not isinstance(frame_ref, Mapping):
                raise BallSstBuildError(f"sample {sample_id} frame_ref must be an object")
            if frame_ref.get("frame_index") != frame_index:
                raise BallSstBuildError(f"sample {sample_id} frame_ref index mismatch")
            if recorded_source_video is None:
                if (
                    Path(
                        str(frame_ref.get("video") or "")
                    ).resolve(strict=False)
                    != expected_media
                ):
                    raise BallSstBuildError(
                        f"sample {sample_id} frame_ref is not canonical media"
                    )
            else:
                _require_consumed_or_recorded_path_identity(
                    frame_ref.get("video"),
                    consumed=Path(recorded_source_video),
                    recorded=recorded_source_video,
                    field=f"sample {sample_id} frame_ref",
                )
            if frame_ref.get("source_video_sha256") != expected_media_sha:
                raise BallSstBuildError(f"sample {sample_id} frame_ref media SHA mismatch")
            dependencies = sample.get("dependency_hashes")
            if not isinstance(dependencies, Mapping):
                raise BallSstBuildError(f"sample {sample_id} requires dependency_hashes")
            for key in all_required_keys:
                if dependencies.get(key) != clip_dependencies.get(key):
                    raise BallSstBuildError(f"sample {sample_id} dependency {key} mismatch")
            _validate_agreement_evidence(
                sample_id=sample_id,
                reason=sample.get("agreement_reason"),
                evidence=sample.get("agreement"),
                teacher_xy=teacher_xy,
                score=score,
                frame_index=frame_index,
                width=source_width,
                height=source_height,
                teacher_confidence_min=teacher_confidence_min,
                agreement_radius_px=agreement_radius_px,
            )
            validated.append(sample)
    return validated


def _validate_agreement_evidence(
    *,
    sample_id: str,
    reason: Any,
    evidence: Any,
    teacher_xy: tuple[float, float],
    score: float,
    frame_index: int,
    width: int,
    height: int,
    teacher_confidence_min: float,
    agreement_radius_px: float,
) -> None:
    if not isinstance(evidence, Mapping):
        raise BallSstBuildError(f"sample {sample_id} agreement evidence must be an object")
    evidence_width = _strict_positive_int(
        evidence.get("image_width"), f"sample {sample_id} image_width"
    )
    evidence_height = _strict_positive_int(
        evidence.get("image_height"), f"sample {sample_id} image_height"
    )
    if (evidence_width, evidence_height) != (width, height):
        raise BallSstBuildError(f"sample {sample_id} agreement image dimensions mismatch")
    if evidence.get("all_points_in_bounds") is not True:
        raise BallSstBuildError(f"sample {sample_id} does not attest in-bounds evidence")
    if reason == "frozen_wasb_spatial":
        if evidence.get("policy_id") != "frozen_wasb_spatial_v2":
            raise BallSstBuildError(f"sample {sample_id} spatial policy id mismatch")
        if evidence.get("source_frame_index") != frame_index:
            raise BallSstBuildError(f"sample {sample_id} spatial source frame mismatch")
        evidence_teacher = _xy_value(evidence.get("teacher_xy"), f"sample {sample_id} evidence teacher_xy")
        wasb_xy = _xy_value(evidence.get("wasb_xy"), f"sample {sample_id} evidence wasb_xy")
        if evidence_teacher != teacher_xy:
            raise BallSstBuildError(f"sample {sample_id} evidence teacher coordinate mismatch")
        teacher_conf = _probability(evidence.get("teacher_confidence"), f"sample {sample_id} teacher confidence")
        wasb_conf = _probability(evidence.get("wasb_confidence"), f"sample {sample_id} WASB confidence")
        if teacher_conf != score or teacher_conf < teacher_confidence_min or wasb_conf < teacher_confidence_min:
            raise BallSstBuildError(f"sample {sample_id} spatial confidence mismatch")
        if not _inside_image(teacher_xy, width=width, height=height) or not _inside_image(wasb_xy, width=width, height=height):
            raise BallSstBuildError(f"sample {sample_id} spatial evidence is out of bounds")
        distance = _distance(teacher_xy, wasb_xy)
        if not math.isclose(
            _finite_float(evidence.get("distance_px"), f"sample {sample_id} distance"),
            distance,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise BallSstBuildError(f"sample {sample_id} spatial distance is not reproducible")
        if evidence.get("agreement_radius_px") != agreement_radius_px or distance > agreement_radius_px:
            raise BallSstBuildError(f"sample {sample_id} spatial radius mismatch")
        return
    if reason != TEMPORAL_GEOMETRY_POLICY["policy_id"]:
        raise BallSstBuildError(f"sample {sample_id} has unsupported agreement reason {reason!r}")
    if evidence.get("policy_id") != TEMPORAL_GEOMETRY_POLICY["policy_id"]:
        raise BallSstBuildError(f"sample {sample_id} temporal policy id mismatch")
    if evidence.get("independent_verifier") != "pinned_frozen_wasb":
        raise BallSstBuildError(f"sample {sample_id} temporal verifier is not independent WASB")
    current_frame = _strict_nonnegative_int(
        evidence.get("current_source_frame_index"), f"sample {sample_id} current frame"
    )
    if current_frame != frame_index:
        raise BallSstBuildError(f"sample {sample_id} temporal current frame mismatch")
    current_teacher = _xy_value(
        evidence.get("current_teacher_xy"), f"sample {sample_id} current teacher_xy"
    )
    current_conf = _probability(
        evidence.get("current_teacher_confidence"), f"sample {sample_id} current confidence"
    )
    if current_teacher != teacher_xy or current_conf != score or current_conf < teacher_confidence_min:
        raise BallSstBuildError(f"sample {sample_id} temporal current-teacher mismatch")
    if not _inside_image(current_teacher, width=width, height=height):
        raise BallSstBuildError(f"sample {sample_id} temporal teacher is out of bounds")
    max_gap = evidence.get("max_gap_source_frames")
    if max_gap != PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES:
        raise BallSstBuildError(f"sample {sample_id} temporal max gap is not preregistered")
    if evidence.get("anchor_agreement_radius_px") != agreement_radius_px:
        raise BallSstBuildError(f"sample {sample_id} temporal anchor radius mismatch")
    if evidence.get("interpolation_residual_max_px") != agreement_radius_px:
        raise BallSstBuildError(f"sample {sample_id} temporal residual radius mismatch")
    _validate_current_wasb_gap_evidence(
        sample_id=sample_id,
        evidence=evidence.get("current_wasb"),
        frame_index=current_frame,
        width=width,
        height=height,
        confidence_min=teacher_confidence_min,
    )
    anchors: list[tuple[str, int, tuple[float, float]]] = []
    for key in ("prior_anchor", "following_anchor"):
        anchor = evidence.get(key)
        if not isinstance(anchor, Mapping):
            raise BallSstBuildError(f"sample {sample_id} missing {key}")
        frame = _strict_nonnegative_int(anchor.get("source_frame_index"), f"sample {sample_id} {key} frame")
        anchor_teacher = _xy_value(anchor.get("teacher_xy"), f"sample {sample_id} {key} teacher_xy")
        anchor_wasb = _xy_value(anchor.get("wasb_xy"), f"sample {sample_id} {key} wasb_xy")
        teacher_conf = _probability(anchor.get("teacher_confidence"), f"sample {sample_id} {key} teacher confidence")
        wasb_conf = _probability(anchor.get("wasb_confidence"), f"sample {sample_id} {key} WASB confidence")
        if teacher_conf < teacher_confidence_min or wasb_conf < teacher_confidence_min:
            raise BallSstBuildError(f"sample {sample_id} {key} is below confidence threshold")
        if not _inside_image(anchor_teacher, width=width, height=height) or not _inside_image(anchor_wasb, width=width, height=height):
            raise BallSstBuildError(f"sample {sample_id} {key} is out of bounds")
        distance = _distance(anchor_teacher, anchor_wasb)
        if not math.isclose(
            _finite_float(anchor.get("distance_px"), f"sample {sample_id} {key} distance"),
            distance,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or distance > agreement_radius_px:
            raise BallSstBuildError(f"sample {sample_id} {key} lacks spatial agreement")
        anchors.append((key, frame, anchor_wasb))
    prior_frame = anchors[0][1]
    following_frame = anchors[1][1]
    if not prior_frame < current_frame < following_frame:
        raise BallSstBuildError(f"sample {sample_id} temporal anchors do not bridge the short gap")
    gap_length = following_frame - prior_frame - 1
    if gap_length > max_gap:
        raise BallSstBuildError(f"sample {sample_id} temporal bridge exceeds total teacher-only gap")
    if evidence.get("gap_length_semantics") != "total_consecutive_teacher_only_interior_frames":
        raise BallSstBuildError(f"sample {sample_id} temporal gap semantics mismatch")
    if evidence.get("teacher_only_gap_length_source_frames") != gap_length:
        raise BallSstBuildError(f"sample {sample_id} temporal total gap length mismatch")
    intermediate = evidence.get("intermediate_frames")
    if not isinstance(intermediate, list) or len(intermediate) != gap_length:
        raise BallSstBuildError(f"sample {sample_id} temporal intermediate-frame evidence mismatch")
    expected_frames = list(range(prior_frame + 1, following_frame))
    for expected_frame, item in zip(expected_frames, intermediate):
        if not isinstance(item, Mapping) or item.get("source_frame_index") != expected_frame:
            raise BallSstBuildError(f"sample {sample_id} temporal intermediate-frame sequence mismatch")
        intermediate_teacher = _xy_value(
            item.get("teacher_xy"), f"sample {sample_id} intermediate teacher_xy"
        )
        intermediate_confidence = _probability(
            item.get("teacher_confidence"), f"sample {sample_id} intermediate teacher confidence"
        )
        if intermediate_confidence < teacher_confidence_min:
            raise BallSstBuildError(f"sample {sample_id} intermediate teacher is below threshold")
        if not _inside_image(intermediate_teacher, width=width, height=height):
            raise BallSstBuildError(f"sample {sample_id} intermediate teacher is out of bounds")
        _validate_current_wasb_gap_evidence(
            sample_id=sample_id,
            evidence=item.get("wasb"),
            frame_index=expected_frame,
            width=width,
            height=height,
            confidence_min=teacher_confidence_min,
        )
        if expected_frame == current_frame:
            if intermediate_teacher != current_teacher or intermediate_confidence != current_conf:
                raise BallSstBuildError(f"sample {sample_id} current intermediate teacher mismatch")
            if item.get("wasb") != evidence.get("current_wasb"):
                raise BallSstBuildError(f"sample {sample_id} current intermediate WASB mismatch")
    alpha = (current_frame - prior_frame) / float(following_frame - prior_frame)
    expected_interpolation = (
        anchors[0][2][0] + alpha * (anchors[1][2][0] - anchors[0][2][0]),
        anchors[0][2][1] + alpha * (anchors[1][2][1] - anchors[0][2][1]),
    )
    emitted_interpolation = _xy_value(
        evidence.get("interpolated_wasb_xy"), f"sample {sample_id} interpolation"
    )
    if not _inside_image(emitted_interpolation, width=width, height=height):
        raise BallSstBuildError(f"sample {sample_id} temporal interpolation is out of bounds")
    if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(expected_interpolation, emitted_interpolation)):
        raise BallSstBuildError(f"sample {sample_id} temporal interpolation is not reproducible")
    residual = _distance(current_teacher, expected_interpolation)
    if not math.isclose(
        _finite_float(evidence.get("interpolation_residual_px"), f"sample {sample_id} residual"),
        residual,
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or residual > agreement_radius_px:
        raise BallSstBuildError(f"sample {sample_id} temporal residual exceeds policy")


def _validate_current_wasb_gap_evidence(
    *,
    sample_id: str,
    evidence: Any,
    frame_index: int,
    width: int,
    height: int,
    confidence_min: float,
) -> None:
    if not isinstance(evidence, Mapping):
        raise BallSstBuildError(f"sample {sample_id} requires current-WASB gap evidence")
    status = evidence.get("status")
    present = evidence.get("present")
    if status == "absent":
        if present is not False:
            raise BallSstBuildError(f"sample {sample_id} absent current WASB is malformed")
        return
    if status not in {"not_visible", "below_confidence_threshold"} or present is not True:
        raise BallSstBuildError(f"sample {sample_id} current WASB gap status is invalid")
    if evidence.get("frame_index") != frame_index:
        raise BallSstBuildError(f"sample {sample_id} current WASB frame mismatch")
    xy = _xy_value(evidence.get("xy"), f"sample {sample_id} current WASB xy")
    if not _inside_image(xy, width=width, height=height):
        raise BallSstBuildError(f"sample {sample_id} current WASB is out of bounds")
    confidence = _probability(
        evidence.get("confidence"), f"sample {sample_id} current WASB confidence"
    )
    visible = evidence.get("visible")
    if not isinstance(visible, bool):
        raise BallSstBuildError(f"sample {sample_id} current WASB visible must be boolean")
    if status == "not_visible" and visible:
        raise BallSstBuildError(f"sample {sample_id} current WASB visibility contradicts status")
    if status == "below_confidence_threshold" and (
        not visible or confidence >= confidence_min
    ):
        raise BallSstBuildError(f"sample {sample_id} current WASB confidence contradicts status")


def _verify_production_artifacts(
    *,
    clips: Sequence[Mapping[str, Any]],
    gallery_root: Path,
    media_root: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    wasb_checkpoint: Path,
    wasb_identity: Mapping[str, Any],
    builder_identity: Mapping[str, str],
    verification_snapshot: SnapshotVerificationContext | None = None,
) -> dict[str, Any]:
    """Recompute every production identity and every accepted row from bound artifacts."""

    try:
        return _verify_production_artifacts_or_raise(
            clips=clips,
            gallery_root=gallery_root,
            media_root=media_root,
            split_manifest=split_manifest,
            split_manifest_sha256=split_manifest_sha256,
            wasb_checkpoint=wasb_checkpoint,
            wasb_identity=wasb_identity,
            builder_identity=builder_identity,
            verification_snapshot=verification_snapshot,
        )
    except (
        BallSstBuildError,
        FileNotFoundError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "verified": False,
            "status": "failed",
            "reason": str(exc),
        }


def _verify_production_artifacts_or_raise(
    *,
    clips: Sequence[Mapping[str, Any]],
    gallery_root: Path,
    media_root: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    wasb_checkpoint: Path,
    wasb_identity: Mapping[str, Any],
    builder_identity: Mapping[str, str],
    verification_snapshot: SnapshotVerificationContext | None = None,
) -> dict[str, Any]:
    if verification_snapshot is None:
        canonical_gallery = _canonical_gallery_root(
            gallery_root,
            require_production_identity=True,
        )
        canonical_media = _canonical_root(media_root, field="media_root")
        verification_split = split_manifest
    else:
        canonical_gallery = _require_snapshot_path(
            verification_snapshot.inputs.root,
            verification_snapshot.inputs.gallery_root,
            context="production gallery verification",
        )
        canonical_media = _require_snapshot_path(
            verification_snapshot.inputs.root,
            verification_snapshot.inputs.media_root,
            context="production media verification",
        )
        verification_split = _require_snapshot_path(
            verification_snapshot.inputs.root,
            verification_snapshot.inputs.split_manifest,
            context="production split verification",
        )
    verified_gallery_hashes_by_source: dict[str, dict[str, str]] = {}
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        _assert_training_source_id(clip_id)
        verified_gallery_hashes_by_source[clip_id] = (
            _validate_production_gallery_artifact_identity(
                canonical_gallery,
                video_id=clip_id,
            )
        )

    split_resolved = verification_split.resolve(strict=True)
    if verification_snapshot is None:
        expected_split = (ROOT / FROZEN_SPLIT_RELATIVE_PATH).resolve(strict=True)
        if split_manifest.is_symlink() or split_manifest.absolute() != split_resolved:
            raise BallSstBuildError(
                "production split_manifest must be a non-symlink canonical path"
            )
        if split_resolved != expected_split:
            raise BallSstBuildError(
                f"production split_manifest must be canonical {expected_split}"
            )
    observed_split_sha = _sha256_file(split_resolved)
    if (
        observed_split_sha != FROZEN_SPLIT_SHA256
        or split_manifest_sha256 != FROZEN_SPLIT_SHA256
    ):
        raise BallSstBuildError("production split_manifest SHA differs from the frozen identity")
    _validate_frozen_split(_read_json(split_resolved), split_resolved)

    current_builder_identity = (
        _builder_identity()
        if verification_snapshot is None
        else {
            **builder_identity,
            "builder_code_sha256": _sha256_file(
                _require_snapshot_path(
                    verification_snapshot.inputs.root,
                    verification_snapshot.inputs.builder_code,
                    context="builder-code verification",
                )
            ),
            "wasb_adapter_code_sha256": _sha256_file(
                _require_snapshot_path(
                    verification_snapshot.inputs.root,
                    verification_snapshot.inputs.wasb_adapter_code,
                    context="adapter-code verification",
                )
            ),
        }
    )
    if any(
        builder_identity.get(key) != current_builder_identity[key]
        for key in (
            "builder_path",
            "builder_code_sha256",
            "builder_git_commit",
            "wasb_adapter_path",
            "wasb_adapter_code_sha256",
            "wasb_adapter_git_commit",
        )
    ):
        raise BallSstBuildError("builder identity changed or was fabricated")

    if verification_snapshot is None:
        repo_path = Path(str(wasb_identity.get("repo_path") or ""))
        current_wasb_identity = _resolve_wasb_identity(
            checkpoint_path=wasb_checkpoint,
            repo_path=repo_path,
            require_production_identity=True,
        )
    else:
        current_wasb_identity = {
            **wasb_identity,
            "models_manifest_path": str(
                verification_snapshot.inputs.models_manifest
            ),
            "checkpoint_path": str(verification_snapshot.inputs.checkpoint),
            "repo_path": str(verification_snapshot.inputs.wasb_repo),
        }
    identity_keys = (
        (
            "manifest_model_id",
            "models_manifest_path",
            "models_manifest_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "repo_path",
            "repo_commit",
            "repo_clean",
            "production_identity_verified",
        )
        if verification_snapshot is None
        else (
            "manifest_model_id",
            "models_manifest_sha256",
            "checkpoint_sha256",
            "repo_commit",
            "repo_clean",
            "production_identity_verified",
        )
    )
    if any(wasb_identity.get(key) != current_wasb_identity[key] for key in identity_keys):
        raise BallSstBuildError("WASB identity changed or was fabricated")
    if verification_snapshot is not None and (
        _sha256_file(verification_snapshot.inputs.models_manifest)
        != str(wasb_identity["models_manifest_sha256"])
        or _sha256_file(verification_snapshot.inputs.checkpoint)
        != str(wasb_identity["checkpoint_sha256"])
    ):
        raise BallSstBuildError("WASB snapshot bytes changed after verification")

    verified_sample_count = 0
    replayed_prediction_sha256_by_clip: dict[str, str] = {}
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        _assert_training_source_id(clip_id)
        source_media = _canonical_media_path(canonical_media, clip_id)
        if verification_snapshot is not None:
            source_media = _require_snapshot_path(
                verification_snapshot.inputs.root,
                source_media,
                context=f"{clip_id} verification source-video consumption",
            )
        media_sha = _validate_media_identity(
            source_media,
            video_id=clip_id,
            media_root=canonical_media,
        )

        cv_export_path = _source_file(canonical_gallery, clip_id, "cv_export.json")
        metadata_path = _source_file(canonical_gallery, clip_id, "api_get_metadata.json")
        provenance_path = _source_file(canonical_gallery, clip_id, "video_provenance.json")
        cv_export = _read_json(cv_export_path)
        metadata = _read_json(metadata_path)
        provenance = _read_json(provenance_path)
        _validate_gallery_provenance(provenance, video_id=clip_id)
        source_width, source_height = _source_dimensions(metadata, clip_id)
        if (
            clip.get("source_width") != source_width
            or clip.get("source_height") != source_height
        ):
            raise BallSstBuildError(f"clip {clip_id} source dimensions changed")

        timing = probe_media_pts(source_media, video_id=clip_id)
        if (timing.width, timing.height) != (source_width, source_height):
            raise BallSstBuildError(
                f"clip {clip_id} decoded dimensions differ from canonical gallery metadata"
            )
        clip_fps = _positive_float(clip.get("fps"), f"clip {clip_id} fps")
        if not math.isclose(clip_fps, timing.fps, rel_tol=0.0, abs_tol=1e-12):
            raise BallSstBuildError(f"clip {clip_id} fps differs from decoded media")

        dependencies = clip.get("dependencies")
        if not isinstance(dependencies, Mapping):
            raise BallSstBuildError(f"clip {clip_id} requires dependency bindings")
        if verification_snapshot is None:
            frame_times_path = _bound_dependency_file(
                dependencies.get("frame_times_path"),
                f"clip {clip_id} frame_times_path",
            )
            wasb_track_path = _bound_dependency_file(
                dependencies.get("wasb_ball_track"),
                f"clip {clip_id} wasb_ball_track",
            )
            wasb_metadata_path = _bound_dependency_file(
                dependencies.get("wasb_metadata_path"),
                f"clip {clip_id} wasb_metadata_path",
            )
            wasb_predictions_csv_path = _bound_dependency_file(
                dependencies.get("wasb_predictions_csv_path"),
                f"clip {clip_id} wasb_predictions_csv_path",
            )
        else:
            snapshot_artifacts = verification_snapshot.artifacts[clip_id]
            frame_times_path = _require_snapshot_path(
                snapshot_artifacts.frame_times.parent,
                snapshot_artifacts.frame_times,
                context=f"{clip_id} verification frame-times consumption",
            )
            wasb_track_path = _require_snapshot_path(
                snapshot_artifacts.ball_track.parent,
                snapshot_artifacts.ball_track,
                context=f"{clip_id} verification ball-track consumption",
            )
            wasb_metadata_path = _require_snapshot_path(
                snapshot_artifacts.metadata.parent,
                snapshot_artifacts.metadata,
                context=f"{clip_id} verification metadata consumption",
            )
            wasb_predictions_csv_path = _require_snapshot_path(
                snapshot_artifacts.predictions_csv.parent,
                snapshot_artifacts.predictions_csv,
                context=f"{clip_id} verification prediction-CSV consumption",
            )
        expected_frame_times = _frame_times_payload(timing, media_sha256=media_sha)
        if _read_json(frame_times_path) != expected_frame_times:
            raise BallSstBuildError(f"clip {clip_id} frame-times payload is not reproducible")

        _validate_wasb_predictions_csv(
            wasb_predictions_csv_path,
            pts_s=timing.pts_s,
            width=source_width,
            height=source_height,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
        )
        track_payload = _read_json(wasb_track_path)
        regenerated_track = wasb_csv_to_ball_track(
            wasb_predictions_csv_path,
            fps=timing.fps,
            frame_times=frame_times_path,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
            input_preprocessing="official",
        )
        if track_payload != regenerated_track:
            raise BallSstBuildError(
                f"clip {clip_id} WASB track does not regenerate from the bound prediction CSV"
            )
        wasb_observations, wasb_frame_count = extract_wasb_observations(
            track_payload,
            pts_s=timing.pts_s,
            fps=timing.fps,
            width=source_width,
            height=source_height,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
        )
        if wasb_frame_count != len(timing.pts_s):
            raise BallSstBuildError(f"clip {clip_id} WASB output did not cover every decoded frame")

        expected_wasb_bindings = {
            "source_video_sha256": media_sha,
            "frame_times_sha256": _sha256_file(frame_times_path),
            "wasb_predictions_csv_sha256": _sha256_file(wasb_predictions_csv_path),
            "wasb_ball_track_sha256": _sha256_file(wasb_track_path),
            "wasb_checkpoint_sha256": current_wasb_identity["checkpoint_sha256"],
            "wasb_repo_commit": current_wasb_identity["repo_commit"],
            "wasb_adapter_code_sha256": current_builder_identity[
                "wasb_adapter_code_sha256"
            ],
        }
        wasb_metadata_payload = _read_json(wasb_metadata_path)
        _validate_wasb_run_metadata(
            wasb_metadata_payload,
            predictions_csv=wasb_predictions_csv_path,
            ball_track=wasb_track_path,
            source_video=source_media,
            checkpoint=Path(current_wasb_identity["checkpoint_path"]),
            wasb_repo=Path(current_wasb_identity["repo_path"]),
            timing=timing,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
            expected_bindings=expected_wasb_bindings,
            declared_predictions_csv=(
                verification_snapshot.publication_artifacts_by_video[
                    clip_id
                ].predictions_csv
                if verification_snapshot is not None
                else None
            ),
            declared_ball_track=(
                verification_snapshot.publication_artifacts_by_video[
                    clip_id
                ].ball_track
                if verification_snapshot is not None
                else None
            ),
            declared_source_video=(
                verification_snapshot.inputs.declarations.source_video_by_id[
                    clip_id
                ]
                if verification_snapshot is not None
                else None
            ),
            declared_checkpoint=(
                verification_snapshot.inputs.declarations.checkpoint
                if verification_snapshot is not None
                else None
            ),
            declared_wasb_repo=(
                verification_snapshot.inputs.declarations.wasb_repo
                if verification_snapshot is not None
                else None
            ),
        )
        if dependencies.get("wasb_runtime") != wasb_metadata_payload:
            raise BallSstBuildError(f"clip {clip_id} embedded WASB runtime metadata mismatch")

        replayed_prediction_sha256_by_clip[clip_id] = (
            _replay_pinned_official_wasb_or_raise(
                clip_id=clip_id,
                retained_predictions_csv=wasb_predictions_csv_path,
                frame_times=frame_times_path,
                source_video=source_media,
                checkpoint=Path(current_wasb_identity["checkpoint_path"]),
                wasb_repo=Path(current_wasb_identity["repo_path"]),
                timing=timing,
                runtime=wasb_metadata_payload["runtime"],
            )
        )

        actual_dependencies = {
            "split_manifest_sha256": observed_split_sha,
            "pbvision_cv_export_sha256": verified_gallery_hashes_by_source[clip_id][
                "cv_export.json"
            ],
            "pbvision_metadata_sha256": verified_gallery_hashes_by_source[clip_id][
                "api_get_metadata.json"
            ],
            "pbvision_provenance_sha256": verified_gallery_hashes_by_source[clip_id][
                "video_provenance.json"
            ],
            "source_video_sha256": media_sha,
            "frame_times_sha256": _sha256_file(frame_times_path),
            "wasb_checkpoint_sha256": current_wasb_identity["checkpoint_sha256"],
            "wasb_repo_commit": current_wasb_identity["repo_commit"],
            "models_manifest_sha256": current_wasb_identity["models_manifest_sha256"],
            "builder_code_sha256": current_builder_identity["builder_code_sha256"],
            "wasb_adapter_code_sha256": current_builder_identity[
                "wasb_adapter_code_sha256"
            ],
            "wasb_ball_track_sha256": _sha256_file(wasb_track_path),
            "wasb_metadata_sha256": _sha256_file(wasb_metadata_path),
            "wasb_predictions_csv_sha256": _sha256_file(wasb_predictions_csv_path),
        }
        for key in SAMPLE_DEPENDENCY_KEYS:
            if dependencies.get(key) != actual_dependencies[key]:
                raise BallSstBuildError(f"clip {clip_id} dependency {key} failed rehash")

        teacher_fps = _teacher_fps(cv_export, metadata, clip_id)
        teacher_observations = extract_teacher_observations(
            cv_export,
            width=source_width,
            height=source_height,
            teacher_fps=teacher_fps,
        )
        expected_samples = build_source_samples(
            video_id=clip_id,
            video_path=source_media,
            teacher_observations=teacher_observations,
            wasb_observations=wasb_observations,
            pts_s=timing.pts_s,
            width=source_width,
            height=source_height,
            teacher_confidence_min=PRODUCTION_TEACHER_CONFIDENCE_MIN,
            agreement_radius_px=PRODUCTION_AGREEMENT_RADIUS_PX,
            pseudo_weight=PRODUCTION_PSEUDO_WEIGHT,
            dependency_hashes=actual_dependencies,
        )
        if verification_snapshot is not None:
            _normalize_sample_video_paths(
                expected_samples,
                source_media=Path(
                    verification_snapshot.inputs.declarations.source_video_by_id[
                        clip_id
                    ]
                ),
            )
        actual_samples = clip.get("samples")
        if not isinstance(actual_samples, list) or actual_samples != expected_samples:
            raise BallSstBuildError(
                f"clip {clip_id} accepted samples do not reproduce from hashed teacher/WASB evidence"
            )
        verified_sample_count += len(expected_samples)

    return {
        "verified": True,
        "status": "passed",
        "reason": (
            "canonical artifacts rehashed, pinned official WASB inference replay matched, "
            "and accepted rows reproduced"
        ),
        "verified_clip_count": len(clips),
        "verified_sample_count": verified_sample_count,
        "official_wasb_replay_verified": True,
        "official_wasb_replay_clip_count": len(replayed_prediction_sha256_by_clip),
        "replayed_prediction_sha256_by_clip": replayed_prediction_sha256_by_clip,
        "pbvision_gallery_authority_id": PRODUCTION_GALLERY_AUTHORITY_ID,
        "verified_pbvision_gallery_sha256_by_source": verified_gallery_hashes_by_source,
        "split_manifest_sha256": observed_split_sha,
        "builder_code_sha256": current_builder_identity["builder_code_sha256"],
        "wasb_adapter_code_sha256": current_builder_identity[
            "wasb_adapter_code_sha256"
        ],
        "wasb_checkpoint_sha256": current_wasb_identity["checkpoint_sha256"],
        "wasb_repo_commit": current_wasb_identity["repo_commit"],
    }


def _replay_pinned_official_wasb_or_raise(
    *,
    clip_id: str,
    retained_predictions_csv: Path,
    frame_times: Path,
    source_video: Path,
    checkpoint: Path,
    wasb_repo: Path,
    timing: MediaTiming,
    runtime: Mapping[str, Any],
) -> str:
    """Replay pinned official inference and authenticate the retained prediction bytes."""

    batch_size = _strict_positive_int(
        runtime.get("batch_size"), f"clip {clip_id} WASB replay batch_size"
    )
    device = runtime.get("device")
    if device not in {"cpu", "cuda"}:
        raise BallSstBuildError(f"clip {clip_id} WASB replay device is invalid")
    retained_bytes = retained_predictions_csv.read_bytes()
    retained_sha256 = _sha256_file(retained_predictions_csv)

    with tempfile.TemporaryDirectory(prefix=f"pbv-wasb-replay-{clip_id}-") as directory:
        replay_root = Path(directory)
        replay_csv = replay_root / "wasb_predictions.csv"
        replay_track = replay_root / "wasb_ball_track.json"
        replay_metadata = replay_root / "wasb_ball_track_metadata.json"
        run_wasb_or_convert(
            out=replay_track,
            fps=timing.fps,
            frame_times=frame_times,
            metadata_out=replay_metadata,
            video=source_video,
            checkpoint=checkpoint,
            wasb_repo=wasb_repo,
            prediction_csv_out=replay_csv,
            batch_size=batch_size,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
            device=str(device),
            input_preprocessing="official",
            emit_size_observations=False,
            emit_below_threshold_candidates=False,
        )
        _validate_wasb_predictions_csv(
            replay_csv,
            pts_s=timing.pts_s,
            width=timing.width,
            height=timing.height,
            visible_threshold=PRODUCTION_TEACHER_CONFIDENCE_MIN,
        )
        replay_bytes = replay_csv.read_bytes()
        replay_sha256 = _sha256_file(replay_csv)
        if replay_bytes != retained_bytes or replay_sha256 != retained_sha256:
            raise BallSstBuildError(
                f"clip {clip_id} retained WASB prediction CSV differs from the pinned "
                "official inference replay"
            )
    return replay_sha256


def _bound_dependency_file(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BallSstBuildError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {field}: {path}")
    resolved = path.resolve(strict=True)
    if path.is_symlink() or path.absolute() != resolved:
        raise BallSstBuildError(f"{field} must be a non-symlink canonical file")
    return resolved


def probe_media_pts(media_path: Path, *, video_id: str) -> MediaTiming:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration,width,height:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise BallSstBuildError(f"ffprobe is required to SHA-bind PTS for {media_path}") from exc
    if completed.returncode != 0:
        raise BallSstBuildError(f"ffprobe failed for {video_id}: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        numerator, denominator = stream["avg_frame_rate"].split("/", 1)
        fps = float(numerator) / float(denominator)
        duration = float(stream["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        pts = tuple(float(frame["best_effort_timestamp_time"]) for frame in payload["frames"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise BallSstBuildError(f"ffprobe metadata malformed for {video_id}") from exc
    if not pts or pts[0] < 0.0 or any(right <= left for left, right in zip(pts, pts[1:])):
        raise BallSstBuildError(f"encoded PTS must be non-negative and strictly increasing: {video_id}")
    if width <= 0 or height <= 0:
        raise BallSstBuildError(f"decoded media dimensions must be positive: {video_id}")
    return MediaTiming(
        fps=fps,
        duration_s=duration,
        pts_s=pts,
        width=width,
        height=height,
    )


def _temporal_wasb_bridge_evidence(
    current: TeacherObservation,
    *,
    source_frame_index: int | None,
    teacher_by_source_frame: Mapping[int, TeacherObservation] | None,
    wasb_observations: Mapping[int, WasbObservation] | None,
    current_wasb: WasbObservation | None,
    width: int,
    height: int,
    confidence_min: float,
    agreement_radius_px: float,
    max_gap_source_frames: int,
) -> dict[str, Any] | None:
    """Require two independent frozen-WASB anchors around a teacher-only gap."""

    if (
        source_frame_index is None
        or teacher_by_source_frame is None
        or wasb_observations is None
        or max_gap_source_frames <= 0
    ):
        return None

    mapped_current_wasb = wasb_observations.get(source_frame_index)
    if current_wasb is None:
        current_wasb = mapped_current_wasb
    elif mapped_current_wasb is not None and current_wasb != mapped_current_wasb:
        return None
    current_wasb_evidence = _current_wasb_gap_evidence(
        current_wasb,
        source_frame_index=source_frame_index,
        confidence_min=confidence_min,
        width=width,
        height=height,
    )
    if current_wasb_evidence is None:
        return None

    def teacher_path_observation(frame_index: int) -> TeacherObservation | None:
        teacher = teacher_by_source_frame.get(frame_index)
        if teacher is None:
            return None
        if (
            not 0.0 <= teacher.confidence <= 1.0
            or teacher.confidence < confidence_min
            or not _inside_image(teacher.xy_px, width=width, height=height)
        ):
            return None
        return teacher

    def anchor(frame_index: int) -> dict[str, Any] | None:
        teacher = teacher_path_observation(frame_index)
        wasb = wasb_observations.get(frame_index)
        if teacher is None or wasb is None:
            return None
        if (
            not 0.0 <= wasb.confidence <= 1.0
            or not wasb.visible
            or wasb.confidence < confidence_min
            or not _inside_image(wasb.xy_px, width=width, height=height)
        ):
            return None
        distance = _distance(teacher.xy_px, wasb.xy_px)
        if distance > agreement_radius_px:
            return None
        return {
            "source_frame_index": frame_index,
            "teacher_xy": [float(teacher.xy_px[0]), float(teacher.xy_px[1])],
            "teacher_confidence": float(teacher.confidence),
            "wasb_xy": [float(wasb.xy_px[0]), float(wasb.xy_px[1])],
            "wasb_confidence": float(wasb.confidence),
            "distance_px": distance,
        }

    def nearest_anchor(step: int) -> dict[str, Any] | None:
        for distance_from_current in range(1, max_gap_source_frames + 1):
            frame_index = source_frame_index + step * distance_from_current
            teacher = teacher_path_observation(frame_index)
            if teacher is None:
                return None
            observation = wasb_observations.get(frame_index)
            if observation is not None and observation.visible and observation.confidence >= confidence_min:
                # A high-confidence disagreement is contradictory evidence, not a
                # transparent hole through which the search may continue outward.
                return anchor(frame_index)
            if _current_wasb_gap_evidence(
                observation,
                source_frame_index=frame_index,
                confidence_min=confidence_min,
                width=width,
                height=height,
            ) is None:
                return None
        return None

    prior_anchor = nearest_anchor(-1)
    following_anchor = nearest_anchor(1)
    if prior_anchor is None or following_anchor is None:
        return None

    prior_frame = int(prior_anchor["source_frame_index"])
    following_frame = int(following_anchor["source_frame_index"])
    teacher_only_gap_length = following_frame - prior_frame - 1
    if teacher_only_gap_length > max_gap_source_frames:
        return None
    intermediate_frames: list[dict[str, Any]] = []
    for frame_index in range(prior_frame + 1, following_frame):
        teacher = teacher_path_observation(frame_index)
        if teacher is None:
            return None
        gap_evidence = _current_wasb_gap_evidence(
            wasb_observations.get(frame_index),
            source_frame_index=frame_index,
            confidence_min=confidence_min,
            width=width,
            height=height,
        )
        if gap_evidence is None:
            return None
        intermediate_frames.append(
            {
                "source_frame_index": frame_index,
                "teacher_xy": [float(teacher.xy_px[0]), float(teacher.xy_px[1])],
                "teacher_confidence": float(teacher.confidence),
                "wasb": gap_evidence,
            }
        )
    alpha = (source_frame_index - prior_frame) / float(following_frame - prior_frame)
    prior_wasb = prior_anchor["wasb_xy"]
    following_wasb = following_anchor["wasb_xy"]
    interpolated = (
        float(prior_wasb[0]) + alpha * (float(following_wasb[0]) - float(prior_wasb[0])),
        float(prior_wasb[1]) + alpha * (float(following_wasb[1]) - float(prior_wasb[1])),
    )
    if not _inside_image(interpolated, width=width, height=height):
        return None
    residual = _distance(current.xy_px, interpolated)
    if residual > agreement_radius_px:
        return None
    return {
        "policy_id": TEMPORAL_GEOMETRY_POLICY["policy_id"],
        "independent_verifier": "pinned_frozen_wasb",
        "current_source_frame_index": source_frame_index,
        "current_teacher_xy": [float(current.xy_px[0]), float(current.xy_px[1])],
        "current_teacher_confidence": float(current.confidence),
        "current_wasb": current_wasb_evidence,
        "prior_anchor": prior_anchor,
        "following_anchor": following_anchor,
        "intermediate_frames": intermediate_frames,
        "interpolated_wasb_xy": [float(interpolated[0]), float(interpolated[1])],
        "interpolation_residual_px": residual,
        "max_gap_source_frames": max_gap_source_frames,
        "gap_length_semantics": "total_consecutive_teacher_only_interior_frames",
        "teacher_only_gap_length_source_frames": teacher_only_gap_length,
        "anchor_agreement_radius_px": agreement_radius_px,
        "interpolation_residual_max_px": agreement_radius_px,
        "image_width": int(width),
        "image_height": int(height),
        "all_points_in_bounds": True,
    }


def _current_wasb_gap_evidence(
    observation: WasbObservation | None,
    *,
    source_frame_index: int,
    confidence_min: float,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Describe why the current frame is a WASB gap, or refuse a usable detection."""

    if observation is None:
        return {"status": "absent", "present": False}
    if observation.frame_index != source_frame_index:
        return None
    if not _inside_image(observation.xy_px, width=width, height=height):
        return None
    if observation.visible and observation.confidence >= confidence_min:
        return None
    status = "not_visible" if not observation.visible else "below_confidence_threshold"
    return {
        "status": status,
        "present": True,
        "frame_index": observation.frame_index,
        "xy": [float(observation.xy_px[0]), float(observation.xy_px[1])],
        "confidence": float(observation.confidence),
        "visible": bool(observation.visible),
    }


def _production_policy_overrides(
    *,
    teacher_confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
) -> list[str]:
    values = {
        "teacher_confidence_min": (
            teacher_confidence_min,
            PRODUCTION_TEACHER_CONFIDENCE_MIN,
        ),
        "agreement_radius_px": (agreement_radius_px, PRODUCTION_AGREEMENT_RADIUS_PX),
        "pseudo_weight": (pseudo_weight, PRODUCTION_PSEUDO_WEIGHT),
    }
    return [key for key, (requested, frozen) in values.items() if requested != frozen]


def _builder_identity() -> dict[str, str]:
    builder_path = Path(__file__).resolve(strict=True)
    wasb_adapter_path = (ROOT / WASB_ADAPTER_RELATIVE_PATH).resolve(strict=True)
    git_commit = _git_output(ROOT, "rev-parse", "HEAD")
    adapter_code_sha256 = _sha256_file(wasb_adapter_path)
    adapter_head_sha256 = _git_blob_sha256(
        ROOT,
        revision=git_commit,
        relative_path=WASB_ADAPTER_RELATIVE_PATH,
    )
    if adapter_code_sha256 != adapter_head_sha256:
        raise BallSstBuildError(
            "WASB adapter working bytes differ from the pinned HEAD blob"
        )
    return {
        "builder_path": str(builder_path.relative_to(ROOT)),
        "builder_code_sha256": _sha256_file(builder_path),
        "builder_git_commit": git_commit,
        "wasb_adapter_path": str(wasb_adapter_path.relative_to(ROOT)),
        "wasb_adapter_code_sha256": adapter_code_sha256,
        "wasb_adapter_git_commit": git_commit,
    }


def _resolve_wasb_identity(
    *,
    checkpoint_path: Path,
    repo_path: Path,
    require_production_identity: bool,
) -> dict[str, Any]:
    models_manifest_path = (ROOT / "models/MANIFEST.json").resolve(strict=True)
    models_manifest = _read_json(models_manifest_path)
    models = models_manifest.get("models")
    if not isinstance(models, list):
        raise BallSstBuildError("models/MANIFEST.json requires models list")
    matches = [row for row in models if isinstance(row, Mapping) and row.get("id") == PRODUCTION_WASB_MODEL_ID]
    if len(matches) != 1:
        raise BallSstBuildError(f"models/MANIFEST.json must contain one {PRODUCTION_WASB_MODEL_ID}")
    manifest_identity = matches[0]
    if manifest_identity.get("local_path") != str(PRODUCTION_WASB_CHECKPOINT_RELATIVE_PATH):
        raise BallSstBuildError("models/MANIFEST.json WASB checkpoint path differs from frozen code identity")
    if manifest_identity.get("sha256") != PRODUCTION_WASB_CHECKPOINT_SHA256:
        raise BallSstBuildError("models/MANIFEST.json WASB checkpoint SHA differs from frozen code identity")
    if manifest_identity.get("repo_commit") != PRODUCTION_WASB_REPO_COMMIT:
        raise BallSstBuildError("models/MANIFEST.json WASB repo commit differs from frozen code identity")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing WASB checkpoint: {checkpoint_path}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"missing WASB repo: {repo_path}")
    checkpoint_resolved = checkpoint_path.resolve(strict=True)
    repo_resolved = repo_path.resolve(strict=True)
    checkpoint_sha256 = _sha256_file(checkpoint_resolved)
    repo_commit = _git_output(repo_resolved, "rev-parse", "HEAD")
    repo_status = _git_output(repo_resolved, "status", "--porcelain", "--untracked-files=all")
    expected_checkpoint = (ROOT / PRODUCTION_WASB_CHECKPOINT_RELATIVE_PATH).resolve(strict=True)
    expected_repo = (ROOT / PRODUCTION_WASB_REPO_RELATIVE_PATH).resolve(strict=True)
    production_identity_verified = (
        checkpoint_path.absolute() == expected_checkpoint
        and checkpoint_resolved == expected_checkpoint
        and checkpoint_sha256 == PRODUCTION_WASB_CHECKPOINT_SHA256
        and repo_path.absolute() == expected_repo
        and repo_resolved == expected_repo
        and repo_commit == PRODUCTION_WASB_REPO_COMMIT
        and repo_status == ""
    )
    if require_production_identity and not production_identity_verified:
        raise BallSstBuildError(
            "production mode requires the canonical models/MANIFEST.json WASB checkpoint SHA, "
            "canonical WASB repo path, pinned repo commit, and a clean repo"
        )
    return {
        "manifest_model_id": PRODUCTION_WASB_MODEL_ID,
        "models_manifest_path": str(models_manifest_path),
        "models_manifest_sha256": _sha256_file(models_manifest_path),
        "checkpoint_path": str(checkpoint_resolved),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": PRODUCTION_WASB_CHECKPOINT_SHA256,
        "repo_path": str(repo_resolved),
        "repo_commit": repo_commit,
        "expected_repo_commit": PRODUCTION_WASB_REPO_COMMIT,
        "repo_clean": repo_status == "",
        "production_identity_verified": production_identity_verified,
    }


def _canonical_root(path: Path, *, field: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {field}: {path}")
    resolved = path.resolve(strict=True)
    if path.is_symlink():
        raise BallSstBuildError(f"{field} must not be a symlink: {path}")
    return resolved


def _canonical_gallery_root(path: Path, *, require_production_identity: bool) -> Path:
    resolved = _canonical_root(path, field="gallery_root")
    expected = (ROOT / FROZEN_GALLERY_RELATIVE_PATH).resolve(strict=True)
    if require_production_identity and resolved != expected:
        raise BallSstBuildError(f"production gallery_root must be canonical {expected}")
    return resolved


def _production_gallery_authority_payload() -> dict[str, Any]:
    _validate_production_gallery_authority_constants()
    return {
        "authority_id": PRODUCTION_GALLERY_AUTHORITY_ID,
        "canonical_gallery_relative_path": FROZEN_GALLERY_RELATIVE_PATH.as_posix(),
        "artifact_filenames": list(PRODUCTION_GALLERY_ARTIFACT_FILENAMES),
        "expected_sha256_by_source": {
            video_id: dict(PRODUCTION_GALLERY_ARTIFACT_SHA256[video_id])
            for video_id in TRAIN_IDS
        },
    }


def _validate_production_gallery_authority_constants() -> None:
    if set(PRODUCTION_GALLERY_ARTIFACT_SHA256) != set(TRAIN_IDS):
        raise BallSstBuildError(
            "production pb.vision gallery SHA authority must name exactly the seven train IDs"
        )
    expected_filenames = set(PRODUCTION_GALLERY_ARTIFACT_FILENAMES)
    for video_id in TRAIN_IDS:
        hashes = PRODUCTION_GALLERY_ARTIFACT_SHA256.get(video_id)
        if not isinstance(hashes, Mapping) or set(hashes) != expected_filenames:
            raise BallSstBuildError(
                f"production pb.vision gallery SHA authority is incomplete for {video_id}"
            )
        for filename in PRODUCTION_GALLERY_ARTIFACT_FILENAMES:
            _require_sha256(
                hashes.get(filename),
                f"production pb.vision gallery authority {video_id}/{filename}",
            )


def _validate_production_gallery_artifact_identity(
    gallery_root: Path,
    *,
    video_id: str,
) -> dict[str, str]:
    """Authenticate one teacher source against code-pinned preregistration."""

    _validate_production_gallery_authority_constants()
    _assert_training_source_id(video_id)
    expected_hashes = PRODUCTION_GALLERY_ARTIFACT_SHA256[video_id]
    observed_hashes: dict[str, str] = {}
    for filename in PRODUCTION_GALLERY_ARTIFACT_FILENAMES:
        source_path = _source_file(gallery_root, video_id, filename)
        observed_sha256 = _sha256_file(source_path)
        expected_sha256 = expected_hashes[filename]
        if observed_sha256 != expected_sha256:
            raise BallSstBuildError(
                f"production pb.vision gallery authority SHA mismatch for "
                f"{video_id}/{filename}: expected {expected_sha256}, "
                f"observed {observed_sha256}; rehashing an edited teacher input "
                "cannot redefine the preregistered authority"
            )
        observed_hashes[filename] = observed_sha256
    return observed_hashes


def _validate_production_gallery_inventory(
    gallery_root: Path,
) -> dict[str, dict[str, str]]:
    """Authenticate all seven teacher sources before any teacher extraction."""

    return {
        video_id: _validate_production_gallery_artifact_identity(
            gallery_root,
            video_id=video_id,
        )
        for video_id in TRAIN_IDS
    }


def _validate_media_identity(source_path: Path, *, video_id: str, media_root: Path) -> str:
    expected_path = _canonical_media_path(media_root, video_id)
    resolved = source_path.resolve(strict=True)
    if resolved != expected_path or source_path.is_symlink() or source_path.parent.is_symlink():
        raise BallSstBuildError(
            f"{video_id} media must be the non-symlink canonical path {expected_path}"
        )
    if not resolved.is_relative_to(media_root):
        raise BallSstBuildError(f"{video_id} media escaped media_root")
    observed_sha = _sha256_file(resolved)
    expected_sha = EXPECTED_SOURCE_VIDEO_SHA256[video_id]
    if observed_sha != expected_sha:
        raise BallSstBuildError(
            f"{video_id} source-video SHA mismatch: expected {expected_sha}, observed {observed_sha}; "
            "renamed, copied, or compare-derived media is refused"
        )
    return observed_sha


def _validate_gallery_provenance(payload: Mapping[str, Any], *, video_id: str) -> None:
    if payload.get("video_id") != video_id:
        raise BallSstBuildError(f"gallery provenance video_id mismatch for {video_id}")
    expected_url = f"https://storage.googleapis.com/pbv-pro/{video_id}/max.mp4"
    if payload.get("source_video_url") != expected_url:
        raise BallSstBuildError(f"gallery provenance source_video_url mismatch for {video_id}")


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BallSstBuildError(
            f"git {' '.join(args)} failed for {repo}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_blob_sha256(repo: Path, *, revision: str, relative_path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative_path.as_posix()}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BallSstBuildError(
            "git show failed for pinned WASB adapter blob: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _validate_frozen_split(payload: Mapping[str, Any], path: Path) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise BallSstBuildError(f"split manifest requires rows list: {path}")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise BallSstBuildError(f"split manifest rows must be objects: {path}")
        video = row.get("video")
        source_video = row.get("source_video")
        aliases = {
            key: value
            for key, value in {
                "video": video,
                "source_video": source_video,
                "video_id": row.get("video_id"),
                "source_id": row.get("source_id"),
                "clip_id": row.get("clip_id"),
                "parent_source_id": row.get("parent_source_id"),
            }.items()
            if value is not None
        }
        if any(str(value) in COMPARE_ONLY_IDS for value in aliases.values()):
            raise BallSstBuildError(
                f"split manifest names compare-only alias {aliases}; refusing before any source read"
            )
        if not isinstance(video, str) or not video or not isinstance(source_video, str) or not source_video:
            raise BallSstBuildError("split rows require explicit video and source_video aliases")
        if video != source_video or any(str(value) != video for value in aliases.values()):
            raise BallSstBuildError(f"split manifest has conflicting source aliases: {aliases}")
        video_id = video
        if video_id in by_id:
            raise BallSstBuildError(f"split manifest repeats source id {video_id}")
        declared_sha_values = {
            str(row[key])
            for key in ("source_video_sha256", "media_sha256")
            if row.get(key) is not None
        }
        if declared_sha_values and declared_sha_values != {EXPECTED_SOURCE_VIDEO_SHA256.get(video_id)}:
            raise BallSstBuildError(f"split manifest source-video SHA alias mismatch for {video_id}")
        if row.get("video_path") is not None or row.get("media_path") is not None:
            raise BallSstBuildError(
                f"split manifest must not redirect canonical media with a path alias: {video_id}"
            )
        by_id[video_id] = row
    expected = frozenset((*TRAIN_IDS, *TEACHER_VAL_ONLY_IDS, *TEACHER_TEST_ONLY_IDS))
    if frozenset(by_id) != expected:
        raise BallSstBuildError(
            f"split manifest source ids differ from frozen ten: missing={sorted(expected - frozenset(by_id))} "
            f"unexpected={sorted(frozenset(by_id) - expected)}"
        )
    expected_splits = {
        **{video_id: "train" for video_id in TRAIN_IDS},
        **{video_id: "val" for video_id in TEACHER_VAL_ONLY_IDS},
        **{video_id: "test" for video_id in TEACHER_TEST_ONLY_IDS},
    }
    wrong = {
        video_id: row.get("split")
        for video_id, row in by_id.items()
        if row.get("split") != expected_splits[video_id]
    }
    if wrong:
        raise BallSstBuildError(f"split manifest violates frozen source roles: {wrong}")


def _assert_training_source_id(video_id: str) -> None:
    if video_id in COMPARE_ONLY_IDS:
        raise BallSstBuildError(
            f"compare-only id {video_id} is structurally unreadable: refusal precedes path construction"
        )
    if video_id not in TRAIN_IDS:
        raise BallSstBuildError(f"source id {video_id} is not one of the seven frozen training ids")


def _source_file(gallery_root: Path, video_id: str, filename: str) -> Path:
    _assert_training_source_id(video_id)
    expected = gallery_root / video_id / filename
    if not expected.is_file():
        raise FileNotFoundError(f"missing canonical pb.vision source file: {expected}")
    resolved = expected.resolve(strict=True)
    if resolved != expected:
        raise BallSstBuildError(f"pb.vision source file must not be a symlink/alias: {expected}")
    if not resolved.is_relative_to(gallery_root):
        raise BallSstBuildError(f"pb.vision source file escaped frozen gallery root: {expected}")
    return resolved


def _read_source_json(gallery_root: Path, video_id: str, filename: str) -> dict[str, Any]:
    return _read_json(_source_file(gallery_root, video_id, filename))


def _discover_media(media_root: Path, video_id: str) -> Path | None:
    _assert_training_source_id(video_id)
    path = _canonical_media_path(media_root, video_id)
    return path if path.is_file() else None


def _canonical_media_path(media_root: Path, video_id: str) -> Path:
    _assert_training_source_id(video_id)
    return media_root / video_id / "max.mp4"


def _missing_media_refusal(
    *,
    gallery_root: Path,
    media_root: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    wasb_checkpoint: Path,
    confidence_min: float,
    agreement_radius_px: float,
    pseudo_weight: float,
    policy_overrides: Sequence[str],
    builder_identity: Mapping[str, str],
    wasb_identity: Mapping[str, Any],
    missing_media: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing_count = len(missing_media)
    staged_count = len(TRAIN_IDS) - missing_count
    teacher_input_authority = _production_gallery_authority_payload()
    preregistration = {
        "policy_id": PRODUCTION_POLICY_ID,
        "teacher_confidence_min": PRODUCTION_TEACHER_CONFIDENCE_MIN,
        "agreement_radius_px": PRODUCTION_AGREEMENT_RADIUS_PX,
        "pseudo_weight": PRODUCTION_PSEUDO_WEIGHT,
        "temporal_max_gap_source_frames": PRODUCTION_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
        "temporal_geometry": dict(TEMPORAL_GEOMETRY_POLICY),
        "canonical_media_relative_path": "<video_id>/max.mp4",
        "expected_source_video_sha256": {
            video_id: EXPECTED_SOURCE_VIDEO_SHA256[video_id] for video_id in TRAIN_IDS
        },
        "teacher_input_authority": teacher_input_authority,
        "builder_code_sha256": str(builder_identity["builder_code_sha256"]),
        "builder_git_commit": str(builder_identity["builder_git_commit"]),
        "wasb_adapter_code_sha256": str(
            builder_identity["wasb_adapter_code_sha256"]
        ),
        "wasb_adapter_git_commit": str(
            builder_identity["wasb_adapter_git_commit"]
        ),
    }
    return {
        "schema_version": 1,
        "artifact_type": REFUSAL_ARTIFACT_TYPE,
        "objective_result": "BLOCKED",
        "verdict": "MISSING_MEDIA",
        "production_eligible": False,
        "production_policy_selected": not policy_overrides,
        "policy_override_fields": list(policy_overrides),
        "ball_verified": False,
        "promotion_claimed": False,
        "protected_eval_clips_touched": False,
        "teacher_input_authority": teacher_input_authority,
        "gallery_root": str(gallery_root),
        "media_root": str(media_root),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": split_manifest_sha256,
        "wasb_checkpoint": str(wasb_checkpoint),
        "required_train_ids": list(TRAIN_IDS),
        "missing_media": [dict(row) for row in missing_media],
        "missing_media_count": missing_count,
        "staged_media_count": staged_count,
        "accepted_windows": 0,
        "accepted_sources": 0,
        "holdout_rows_present": 0,
        "decode_status": "not_attempted",
        "decode_failures": None,
        "preregistration": preregistration,
        "preregistered_parameters": preregistration,
        "requested_parameters": {
            "teacher_confidence_min": confidence_min,
            "agreement_radius_px": agreement_radius_px,
            "pseudo_weight": pseudo_weight,
        },
        "builder_identity": dict(builder_identity),
        "wasb_identity": dict(wasb_identity),
        "next": (
            f"stage the {missing_count} missing public GCS media objects on the GPU VM "
            f"({staged_count} of seven already present) and rerun unchanged"
        ),
    }


def _frame_times_payload(timing: MediaTiming, *, media_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_times",
        "source_video_sha256": media_sha256,
        "fps": timing.fps,
        "duration_s": timing.duration_s,
        "width": timing.width,
        "height": timing.height,
        "frame_count": len(timing.pts_s),
        "frames": [
            {"frame": frame_index, "pts_s": pts}
            for frame_index, pts in enumerate(timing.pts_s)
        ],
    }


def _source_dimensions(metadata_payload: Mapping[str, Any], video_id: str) -> tuple[int, int]:
    metadata = metadata_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise BallSstBuildError(f"{video_id} metadata object missing")
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        raise BallSstBuildError(f"{video_id} source dimensions must be positive")
    return width, height


def _teacher_fps(
    cv_export: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
    video_id: str,
) -> float:
    camera = cv_export.get("camera")
    metadata = metadata_payload.get("metadata")
    value = camera.get("fps") if isinstance(camera, Mapping) else None
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("fps")
    return _positive_float(value, f"{video_id} teacher fps")


def _nearest_pts_frame(timestamp_s: float, pts_s: Sequence[float]) -> int:
    insertion = bisect.bisect_left(pts_s, timestamp_s)
    if insertion <= 0:
        return 0
    if insertion >= len(pts_s):
        return len(pts_s) - 1
    before = insertion - 1
    return before if abs(pts_s[before] - timestamp_s) <= abs(pts_s[insertion] - timestamp_s) else insertion


def _cli_report(payload: Mapping[str, Any], *, out: Path) -> dict[str, Any]:
    gate = payload.get("gate") if isinstance(payload.get("gate"), Mapping) else {}
    verdict = gate.get("verdict") or payload.get("verdict")
    return {
        "objective_result": (
            "PASS" if verdict == "PASS" else payload.get("objective_result", "PARTIAL")
        ),
        "verdict": verdict,
        "out": str(out),
        "accepted_windows": int(payload.get("accepted_windows") or 0),
        "accepted_sources": int(payload.get("accepted_sources") or 0),
        "holdout_rows_present": int(payload.get("holdout_rows_present") or 0),
        "production_eligible": payload.get("production_eligible") is True,
        "decode_status": payload.get("decode_status", "completed"),
        "decode_failures": payload.get("decode_failures"),
        "missing_media": payload.get("missing_media") or [],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build agreement-gated pb.vision BALL SST positives.")
    parser.add_argument("--gallery-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--wasb-checkpoint", type=Path, required=True)
    parser.add_argument("--wasb-repo", type=Path, default=Path("third_party/WASB-SBDT"))
    parser.add_argument(
        "--teacher-confidence-min", type=float, default=PRODUCTION_TEACHER_CONFIDENCE_MIN
    )
    parser.add_argument(
        "--agreement-radius-px", type=float, default=PRODUCTION_AGREEMENT_RADIUS_PX
    )
    parser.add_argument("--pseudo-weight", type=float, default=PRODUCTION_PSEUDO_WEIGHT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wasb-batch-size", type=int, default=8)
    parser.add_argument(
        "--resume-dependencies",
        action="store_true",
        help="reuse only fully hash-bound per-video WASB dependencies; default is fresh inference",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _inside_image(xy: Sequence[float], *, width: int, height: int) -> bool:
    return 0.0 <= float(xy[0]) < float(width) and 0.0 <= float(xy[1]) < float(height)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _xy_value(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise BallSstBuildError(f"{field} must be [x, y]")
    return (_finite_float(value[0], f"{field} x"), _finite_float(value[1], f"{field} y"))


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BallSstBuildError(f"{field} must be a non-negative integer")
    return value


def _strict_positive_int(value: Any, field: str) -> int:
    parsed = _strict_nonnegative_int(value, field)
    if parsed <= 0:
        raise BallSstBuildError(f"{field} must be positive")
    return parsed


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise BallSstBuildError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _require_git_commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise BallSstBuildError(f"{field} must be a lowercase 40-character git commit")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BallSstBuildError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BallSstBuildError(f"{field} must be finite")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed <= 0.0:
        raise BallSstBuildError(f"{field} must be positive")
    return parsed


def _probability(value: Any, field: str) -> float:
    parsed = _finite_float(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise BallSstBuildError(f"{field} must be in [0, 1]")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BallSstBuildError(f"JSON input must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
