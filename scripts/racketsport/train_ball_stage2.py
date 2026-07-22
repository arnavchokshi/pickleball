#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_ball_pretrain import (  # noqa: E402
    MODEL_FAMILIES,
    _collate_batch,
    _device,
    _loader_generator,
    _parse_image_path_rewrites,
    _parse_image_size,
    _parse_protected_eval_hashes,
    _primary_logits,
    _seed_loader_worker,
    _seed_training_process,
    atomic_torch_save,
    build_model,
    checkpoint_round_trip_summary,
    load_model_weights,
    state_dict_sha256,
    train_one_batch,
)
from threed.racketsport.ball_sst_dataset import (  # noqa: E402
    build_sst_manifest,
)
from threed.racketsport.ball_tracknet_cvat_dataset import TrackNetCvatLabel  # noqa: E402
from threed.racketsport.cvat_video import import_cvat_video_zip  # noqa: E402
from threed.racketsport.roboflow_corpus import (  # noqa: E402
    DEFAULT_BALL_PRETRAIN_FRAMES_IN,
    DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
    DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    DEFAULT_PROTECTED_EVAL_HASH_COUNT,
)
from threed.racketsport.schemas import (  # noqa: E402
    BALL_VISIBILITY_WBCE_WEIGHTS,
    BallVisibilityLevel,
    CvatVideoAnnotations,
    CvatVideoFrame,
    validate_artifact_file,
)
from threed.racketsport.wasb_adapter import (  # noqa: E402
    STATUS_TESTED,
    WASB_CONFIDENCE_SEMANTICS,
    WASB_MODEL_ZOO_URL,
    WASB_REPO_URL,
    _preprocess_wasb_window_official,
    _wasb_affine_transform_xy,
    _wasb_official_input_affine,
    run_wasb_or_convert,
    wasb_csv_to_ball_track,
)


ARTIFACT_TYPE = "racketsport_ball_stage2_run"
DEFAULT_CVAT_EXPORT_ROOT = Path("cvat_upload/exports/harvest_review_20260707")
DEFAULT_RALLY_ROOT = Path("data/online_harvest_20260706/rallies")
DEFAULT_PRELABEL_ROOT = Path("data/online_harvest_20260706/prelabels")

# B0 is a frozen data artifact, not a query against the mutable reviewed corpus.
# These hashes pin the accepted artifact named in EXACT_PLAN B0 and in the review.
DEFAULT_B0_SPLIT_ROOT = Path("runs/lanes/ball_b0_split_20260721/split")
B0_REPORT_SHA256 = "122e65913d54df6be6c3e5c6ca91229fc17d207674edc7a31ea705bafd6eb3a3"
B0_TRAIN_SHA256 = "b92218d47816e01893a687c6414bdaa5220f02be6d3b1c25b684128d12ee9c20"
B0_VALIDATION_SHA256 = "39a07ed6d5211cbdc2ccc8a3f1f73b298a1ed262a6cae1f8a6190e5aa1533429"
B0_TRAIN_SOURCE_IDS = frozenset(("73VurrTKCZ8", "_L0HVmAlCQI", "wBu8bC4OfUY", "zwCtH_i1_S4"))
B0_JUDGE_PARENT_IDS = frozenset(("HyUqT7zFiwk", "Ezz6HDNHlnk"))
B0_EXPECTED_TRAIN_ROWS = 2_249
B0_EXPECTED_VALIDATION_ROWS = 167
B0_EXPECTED_EXCLUDED_JUDGE_REVIEWED_ROWS = 960
B0_SOURCE_MEDIA_SHA256 = {
    "73VurrTKCZ8/73VurrTKCZ8_rally_0001.mp4": "a0a840080578a44f50d6a764433e198836f670d5dc3c0bfe15c04e024477f009",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0002.mp4": "2423295f4ce8c3f222d1e0357d16293e190cce162ef3ea773d133110fb9a14a4",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0003.mp4": "ba3fcdf8aa2646c49c2248418a06a1499a4836bf3debfdf5a0fe127bec10013b",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0004.mp4": "3fc78f7ffcfd456a82a28d74e57850dda2fe5e38bfa8777ba73755fabb29ed83",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0005.mp4": "2738583ba81790b4a39b042deab385e1533f9e8c4e6545e57959c3940fc0431a",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0006.mp4": "29e799a73041b8edd100e2c51b07ca5222954d00f0bb38327487e795603186f0",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0007.mp4": "a7f4fc9ea5caeeb632bbdbfdcc5c708653f159779e463df9bc059535cc67a195",
    "73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4": "64930671feec04de10423d1e6430c7ee758cf05b4ee44665912bf0e999aa16d2",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0001.mp4": "ab45db7a7b6c6c37d1f0cafb0d2cfeaf7d8b0f874cc068ff2d2ea23dfdd553ae",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0002.mp4": "8270775518e623a0301dc771394816bd7ebbbe14f42228dd40f7ee2b9c4ef29d",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0003.mp4": "3fe77ab6d48850c330583d7145e98bc5366b295a88afd6eeb0d1684a45fed6a9",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0004.mp4": "73cbedc6a385113eb10427925f2c133300e5d377a9ea06e60fe507edd36c2399",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0005.mp4": "95b30ee556bafd517d5c42be744368649c67b4ee3e0493cc62380be2d6b4b932",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0006.mp4": "0cdad235b9435865a1733cba98ac752dd9e15787e8841092f2a75f30c6838cae",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0007.mp4": "bb39439492055d874a38632a338153c4dc43e0ee827a1dc6bf9ca287f674634b",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0008.mp4": "84515b2d8a69ee7ab45ca9f0c5eb5d21c9556abe63ca7798fc1fb528a870b5dc",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0009.mp4": "767e9bfe079543a6ccfd9502ee51ab9bcd3d751281e694144aa9f1e77a09a09e",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0010.mp4": "0ba24c485e290f2be39fa85eb5388af19d268a73593b0587d4dc034ce02e41e5",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0011.mp4": "075259cc011ff425078413d28455f39ee1de9c6f8e749986a974229f100fa916",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0012.mp4": "99c7a2284ce20abb624c91ffb3ab4f9c2b3e612a9fa030ad02e619af73aee1ce",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0013.mp4": "004c3739d13383895179e1468c61f9a844d928360453164aa2193e9208df7785",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0014.mp4": "7125d71ccfdee033c5b8c0684e6bc5ef5eb10935d05d4417a8fbcde76511cfbe",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0015.mp4": "8abd2659695947078d06b6759f92ba4b876b4791683f0af338342fd843010613",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0016.mp4": "cce5fecad2268b51141739089c45c2f99e8ac6726c0232d5816e1672b8fea066",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0017.mp4": "71d3bc332e69707ad082e7ad440f0aa27b4ab992fa63a033308a7a02f49544c8",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0018.mp4": "ef149cb6f939956a8ed41d005e352109c8cf720e5752a0455dba836396a23cc6",
    "_L0HVmAlCQI/_L0HVmAlCQI_rally_0019.mp4": "756064522f2fabb773a3d6ad3db0f3a463585808d83971a00ed90e3c6b36ffe0",
    "wBu8bC4OfUY/wBu8bC4OfUY_rally_0001.mp4": "2e33a1007163d337b3d00ecf3b3c3fa5996793648d0e917a1c754efbc530eb0c",
    "wBu8bC4OfUY/wBu8bC4OfUY_rally_0002.mp4": "aea9c5e89c26bb11e1f08ad2bd2c9fe953494ce530adf8b50500d926dd8eec24",
    "wBu8bC4OfUY/wBu8bC4OfUY_rally_0003.mp4": "dc5c93403b610f252a57e1a06ec0c0af5bd76803c20602c0a204700b0b51cee1",
    "zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4": "7a66de6134a583d67faa3b0a1269abb461caaed14b4a327d1f316b5249cd86f1",
}
# The held-out parent media are pinned by the live B0 judge scorer.  The generic
# CVAT path duplicates the content authority deliberately so a renamed copy or
# a caller-provided alias cannot turn held-out bytes into an apparently new
# training source. Ezz rally 0003 is included as parent-level protection even
# though it has no row in the frozen 167-row validation JSONL.
B0_JUDGE_SOURCE_MEDIA_SHA256 = {
    "Ezz6HDNHlnk_rally_0001": "582ffbb02098bb8e59afff74ffa25fb178be8ecc7782889b06a0e0aa64bef844",
    "Ezz6HDNHlnk_rally_0002": "6ed769d5464d89fee54e29603605cdcac53c3ade29aa1b306f4b1eca50228650",
    "Ezz6HDNHlnk_rally_0003": "1f045df5c99ec40614522c350a5ce8fbf671809a99d08d4eab52f2443cf315c2",
    "Ezz6HDNHlnk_rally_0004": "05fe6312108825bdd51a812dac6dd5a2450f2ee1e20aca62197569b233e906a2",
    "Ezz6HDNHlnk_rally_0005": "e622b1646920f43639d4c076e5e7cc5a10484d83526d5478c18c3836b221819d",
    "Ezz6HDNHlnk_rally_0006": "4f30bc394415f9e7d4de6cf0cc20e493646c266eb83c998f562a0b830e35a355",
    "Ezz6HDNHlnk_rally_0007": "77d02862ffd890e1f2935f56ed66c64bf39eb984f8b7c125723b83db8883b0f8",
    "Ezz6HDNHlnk_rally_0008": "88e095ca7260226f65d9b24ba2257590ba558a28c777d380daba94390e163c9a",
    "HyUqT7zFiwk_rally_0001": "056f1710d864bf9f5847c896cab8842d34b94da661fa4fd62b59d9ae1219eae3",
}
B0_CANONICAL_MEDIA_IDENTITY_BY_SHA256 = {
    sha256: (Path(relative_path).stem, Path(relative_path).parts[0])
    for relative_path, sha256 in B0_SOURCE_MEDIA_SHA256.items()
}
B0_CANONICAL_MEDIA_IDENTITY_BY_SHA256.update(
    {
        sha256: (clip_id, clip_id.split("_rally_", 1)[0])
        for clip_id, sha256 in B0_JUDGE_SOURCE_MEDIA_SHA256.items()
    }
)
if len(B0_CANONICAL_MEDIA_IDENTITY_BY_SHA256) != len(B0_SOURCE_MEDIA_SHA256) + len(
    B0_JUDGE_SOURCE_MEDIA_SHA256
):
    raise RuntimeError("frozen B0 media SHA authority contains a content-identity collision")

# EXACT_PLAN B1/B2 production policy. These values are duplicated deliberately:
# the consumer must not trust a producer module to validate its own output.
SST_PRODUCTION_POLICY_ID = "pbv_ball_sst_production_v2"
SST_TEACHER_CONFIDENCE_MIN = 0.90
SST_AGREEMENT_RADIUS_PX = 20.0
SST_PSEUDO_WEIGHT = 0.25
SST_TEMPORAL_MAX_GAP_SOURCE_FRAMES = 2
SST_MIN_ACCEPTED_WINDOWS = 1_000
SST_MIN_ACCEPTED_SOURCES = 5
SST_TRAIN_SOURCE_IDS = frozenset(
    (
        "143sf3gdwxsa",
        "98z43hspqz13",
        "bewqc0glhgpq",
        "st0epgnab7dr",
        "td2szayjwtrj",
        "tqjlrcntpjvt",
        "xkadsq9bli3h",
    )
)
SST_EXPECTED_SOURCE_VIDEO_SHA256 = {
    "143sf3gdwxsa": "03fbdc2b056c1b1ed665c71994c06bc485f385b44a2fee892338360c666f845c",
    "98z43hspqz13": "006eb7d0e7e7c5c351ea72b88c946a452660adb24eff87e77d12419b7330b11f",
    "bewqc0glhgpq": "e6b73a38535aea5d3644c3a94091b3c5d261b6c2b60e5d80a21514ad502b69cf",
    "st0epgnab7dr": "2803b4a18c97e3d3165cdbacbe7bcbe6c4b0c273820aa6840b7e731aea98ff04",
    "td2szayjwtrj": "9594260561b334937a1dfb62c1450315fdcb1ee3e1ece304961416c7d15a2d79",
    "tqjlrcntpjvt": "176cb66c13e2fa481839815c1dc41c063b2a0cc17758e75dd9c7f39627f31490",
    "xkadsq9bli3h": "5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181",
}
SST_SPATIAL_REASON = "frozen_wasb_spatial"
SST_TEMPORAL_REASON = "frozen_wasb_temporal_bridge_v2"
SST_TEMPORAL_GEOMETRY_POLICY: dict[str, Any] = {
    "policy_id": SST_TEMPORAL_REASON,
    "independent_verifier": "pinned_frozen_wasb",
    "teacher_confidence_min": SST_TEACHER_CONFIDENCE_MIN,
    "wasb_confidence_min": SST_TEACHER_CONFIDENCE_MIN,
    "anchor_agreement_radius_px": SST_AGREEMENT_RADIUS_PX,
    "interpolation_residual_max_px": SST_AGREEMENT_RADIUS_PX,
    "max_gap_source_frames": SST_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
    "gap_length_semantics": "total_consecutive_teacher_only_interior_frames",
    "requires_bracketing_wasb_agreement_anchors": True,
    "requires_every_interior_frame_teacher_only": True,
    "contradictory_high_confidence_wasb_is_search_barrier": True,
    "requires_current_wasb_absent_invisible_or_below_threshold": True,
    "requires_current_wasb_status_evidence": True,
    "image_bounds_required_for_teacher_and_wasb_points": True,
    "same_teacher_self_agreement_eligible": False,
}
SST_REQUIRED_ROW_DEPENDENCY_HASHES = frozenset(
    (
        "builder_code_sha256",
        "frame_times_sha256",
        "models_manifest_sha256",
        "pbvision_cv_export_sha256",
        "pbvision_metadata_sha256",
        "pbvision_provenance_sha256",
        "source_video_sha256",
        "split_manifest_sha256",
        "wasb_ball_track_sha256",
        "wasb_predictions_csv_sha256",
        "wasb_adapter_code_sha256",
        "wasb_checkpoint_sha256",
        "wasb_metadata_sha256",
        "wasb_repo_commit",
    )
)
SST_WASB_MODEL_ID = "wasb_tennis_bmvc2023"
SST_WASB_ADAPTER_PATH = Path("threed/racketsport/wasb_adapter.py")
SST_FROZEN_GALLERY_ROOT = Path("data/pbvision_gallery_20260719")
SST_PBVISION_GALLERY_FILENAMES = (
    "api_get_metadata.json",
    "cv_export.json",
    "video_provenance.json",
)
SST_PBVISION_GALLERY_AUTHORITY_ID = "pbvision_gallery_20260719_teacher_inputs_sha256_v1"
# Independent trainer-side authority for the exact 21 PBVision inputs used by B1.
# The manifest's own dependency hashes are not an authority because an attacker can
# rewrite both an ignored gallery file and every hash that points at it.
SST_FROZEN_GALLERY_BUNDLE_SHA256 = "c4d3c632cf5250b5f1b243ce7a7b49fb9eed4eab76dd1db26baacd5c0ad5dad2"
SST_FROZEN_SPLIT_MANIFEST = Path("runs/lanes/pbv_pickleball_corpus_20260720/manifest.json")
SST_FROZEN_SPLIT_SHA256 = "cf8f251827688c7923e35ce93b06b66c014ba9192b9d18f4ecbd2a256195451b"
SST_WASB_REPO_ROOT = Path("third_party/WASB-SBDT")
SST_GLOBAL_DEPENDENCY_KEYS = frozenset(
    (
        "split_manifest_sha256",
        "models_manifest_sha256",
        "builder_code_sha256",
        "wasb_adapter_code_sha256",
        "wasb_checkpoint_sha256",
        "wasb_repo_commit",
    )
)
B2_HUMAN_BATCH_SIZE = 8
B2_SST_BATCH_SIZE = 8
B2_SST_LOSS_CAP = 0.25
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PINNED_HEAD_PRODUCTION_PARITY_CONFIG: dict[str, Any] = {
    "model_family": "wasb_hrnet",
    "image_size": list(DEFAULT_BALL_PRETRAIN_IMAGE_SIZE),
    "frames_in": 3,
    "output_channels": 3,
    "steps": 2_372,
    "batch_size": B2_HUMAN_BATCH_SIZE,
    "learning_rate": 5e-4,
    "weight_decay": 5e-5,
    "heatmap_radius_px": float(DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX),
    "occluded_prob": 0.0,
    "seed": 20260721,
    "device": "cuda",
    "b0_split_root": str(DEFAULT_B0_SPLIT_ROOT),
    "init_checkpoint": "models/checkpoints/wasb/wasb_tennis_best.pth.tar",
    "wasb_repo": str(SST_WASB_REPO_ROOT),
    "rally_root": str(DEFAULT_RALLY_ROOT),
    "image_root_rewrite": [],
    "num_workers": 0,
}


@dataclass(frozen=True)
class Stage2SampleRecord:
    sample_id: str
    source_kind: str
    clip_id: str
    parent_source_id: str
    video_path: Path
    source_video_sha256: str
    frame_index: int
    source_width: int
    source_height: int
    ball_present: bool
    source_xy_px: tuple[float, float]
    visibility_level: BallVisibilityLevel | None
    wbce_weight: float
    source_path: str


class CvatBallStage2Dataset:
    def __init__(
        self,
        records: Sequence[Stage2SampleRecord],
        *,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
    ) -> None:
        _validate_dataset_shape(image_size=image_size, frames_in=frames_in, heatmap_radius_px=heatmap_radius_px)
        self.records = tuple(records)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.frames_in = int(frames_in)
        self.heatmap_radius_px = float(heatmap_radius_px)
        self.image_path_rewrites = _parse_image_path_rewrites(image_path_rewrites)
        self.summary = _dataset_summary("cvat_owner_sparse", self.records, self.image_size, self.frames_in, self.heatmap_radius_px)
        self.summary["dataset_provenance"] = _dataset_provenance(self.records)

    @classmethod
    def from_export_root(
        cls,
        cvat_export_root: str | Path,
        *,
        rally_root: str | Path = DEFAULT_RALLY_ROOT,
        video_paths: Mapping[str, str | Path] | None = None,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
        max_samples: int | None = None,
    ) -> "CvatBallStage2Dataset":
        root = Path(cvat_export_root)
        if not root.is_dir():
            raise FileNotFoundError(f"missing CVAT export root: {root}")
        normalized_videos = {str(clip): Path(path) for clip, path in (video_paths or {}).items()}
        normalized_rewrites = _parse_image_path_rewrites(image_path_rewrites)
        records: list[Stage2SampleRecord] = []
        media_sha_cache: dict[Path, str] = {}
        for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            annotations = load_cvat_annotations_from_export_clip(clip_dir)
            video_path = normalized_videos.get(annotations.clip_id) or _resolve_rally_video(Path(rally_root), annotations.clip_id)
            decoded_video_path = _rewrite_path(video_path, normalized_rewrites)
            if not decoded_video_path.is_file():
                raise FileNotFoundError(
                    f"missing CVAT source video for {annotations.clip_id}: {decoded_video_path}"
                )
            if decoded_video_path not in media_sha_cache:
                media_sha_cache[decoded_video_path] = _sha256_file(decoded_video_path)
            media_sha256 = media_sha_cache[decoded_video_path]
            declared_parent = _parent_source_id(annotations.clip_id)
            canonical_identity = _canonical_b0_media_identity(media_sha256)
            if canonical_identity is not None:
                canonical_clip, canonical_parent = canonical_identity
                if canonical_parent in B0_JUDGE_PARENT_IDS:
                    raise ValueError(
                        "generic CVAT input resolves by content SHA to frozen B0 judge media: "
                        f"declared_clip={annotations.clip_id} canonical_clip={canonical_clip} "
                        f"canonical_parent={canonical_parent}"
                    )
                if annotations.clip_id != canonical_clip or declared_parent != canonical_parent:
                    raise ValueError(
                        "generic CVAT clip/media alias disagrees with canonical content identity: "
                        f"declared=({annotations.clip_id},{declared_parent}) "
                        f"canonical=({canonical_clip},{canonical_parent})"
                    )
            if declared_parent in B0_JUDGE_PARENT_IDS:
                raise ValueError(f"generic CVAT input cannot consume frozen B0 judge parent {declared_parent}")
            labels = sparse_tracknet_labels_from_annotations(annotations)
            for label in labels:
                if max_samples is not None and len(records) >= max_samples:
                    break
                records.append(
                    _record_from_cvat_label(
                        annotations,
                        label,
                        video_path=video_path,
                        source_video_sha256=media_sha256,
                        source_path=str(clip_dir),
                    )
                )
            if max_samples is not None and len(records) >= max_samples:
                break
        if not records:
            raise ValueError(f"no sparse reviewed CVAT training rows found under {root}")
        return cls(
            records,
            image_size=image_size,
            frames_in=frames_in,
            heatmap_radius_px=heatmap_radius_px,
            image_path_rewrites=image_path_rewrites,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _record_to_item(
            self.records[index],
            image_size=self.image_size,
            frames_in=self.frames_in,
            heatmap_radius_px=self.heatmap_radius_px,
            image_path_rewrites=self.image_path_rewrites,
        )


class B0BallStage2Dataset(CvatBallStage2Dataset):
    """The exact accepted B0 train split, including its lineage weights."""

    @classmethod
    def from_split_root(
        cls,
        split_root: str | Path,
        *,
        rally_root: str | Path = DEFAULT_RALLY_ROOT,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
    ) -> "B0BallStage2Dataset":
        rows, split_summary = load_and_validate_b0_split(split_root)
        normalized_rewrites = _parse_image_path_rewrites(image_path_rewrites)
        if normalized_rewrites:
            raise ValueError("frozen B0 training refuses --image-root-rewrite media substitution")
        media_identity = _validate_b0_source_media(rows, rally_root=rally_root)
        rally = (ROOT / DEFAULT_RALLY_ROOT).resolve(strict=True)
        size_cache: dict[Path, tuple[int, int]] = {}
        records: list[Stage2SampleRecord] = []
        for row in rows:
            source_id = str(row["source_id"])
            clip_id = str(row["clip_id"])
            video_path = rally / source_id / f"{clip_id}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"missing frozen B0 source video: {video_path}")
            if video_path not in size_cache:
                size_cache[video_path] = _video_size(video_path)
            source_width, source_height = size_cache[video_path]
            label = row["final_label"]
            ball_present = bool(label["ball_present"])
            source_xy = (0.0, 0.0)
            if ball_present:
                x1, y1, x2, y2 = _validated_bbox(
                    label.get("bbox_xyxy"),
                    field=f"B0 row {row['row_key']} final_label.bbox_xyxy",
                    width=source_width,
                    height=source_height,
                )
                source_xy = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            visibility_raw = str(label.get("visibility_level") or "none")
            visibility: BallVisibilityLevel | None = (
                visibility_raw if visibility_raw in {"clear", "partial", "full", "out_of_frame"} else None
            )
            records.append(
                Stage2SampleRecord(
                    sample_id=f"b0:{row['row_key']}",
                    source_kind=f"b0_{row['lineage_class']}",
                    clip_id=clip_id,
                    parent_source_id=source_id,
                    video_path=video_path,
                    source_video_sha256=media_identity[f"{source_id}/{clip_id}.mp4"],
                    frame_index=int(row["frame_index"]),
                    source_width=source_width,
                    source_height=source_height,
                    ball_present=ball_present,
                    source_xy_px=source_xy,
                    visibility_level=visibility,
                    wbce_weight=float(row["training_weight"]),
                    source_path=str(Path(split_root) / "train.jsonl"),
                )
            )
        dataset = cls(
            records,
            image_size=image_size,
            frames_in=frames_in,
            heatmap_radius_px=heatmap_radius_px,
            image_path_rewrites={},
        )
        dataset.summary.update(split_summary)
        dataset.summary["source_kind"] = "frozen_b0_parent_source_split"
        dataset.summary["lineage_weight_policy"] = {
            "scratch": 1.0,
            "corrected_prelabel": 1.0,
            "confirmed_prelabel": 0.25,
        }
        dataset.summary["source_media_sha256"] = media_identity
        return dataset


class SstBallStage2Dataset(CvatBallStage2Dataset):
    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
        max_samples: int | None = None,
    ) -> "SstBallStage2Dataset":
        if max_samples is not None:
            raise ValueError("production SST manifests cannot be truncated with --max-sst-samples")
        manifest, samples = load_and_validate_production_sst_manifest(manifest_path)
        records: list[Stage2SampleRecord] = []
        size_cache: dict[Path, tuple[int, int]] = {}
        for sample in samples:
            frame_ref = sample.get("frame_ref")
            assert isinstance(frame_ref, Mapping)  # validated at the trust boundary
            video_path = Path(str(frame_ref.get("video") or ""))
            if video_path not in size_cache:
                size_cache[video_path] = _video_size(video_path)
            source_width, source_height = size_cache[video_path]
            xy = sample.get("teacher_xy")
            assert isinstance(xy, Sequence)
            records.append(
                Stage2SampleRecord(
                    sample_id=str(sample.get("sample_id") or f"{sample.get('clip_id')}:{sample.get('frame_index')}"),
                    source_kind="sst_pseudo_label",
                    clip_id=str(sample["clip_id"]),
                    parent_source_id=str(sample["canonical_source_id"]),
                    video_path=video_path,
                    source_video_sha256=str(sample["source_video_sha256"]),
                    frame_index=int(sample["frame_index"]),
                    source_width=source_width,
                    source_height=source_height,
                    ball_present=True,
                    source_xy_px=(float(xy[0]), float(xy[1])),
                    visibility_level=None,
                    wbce_weight=SST_PSEUDO_WEIGHT,
                    source_path=str(manifest_path),
                )
            )
        if not records:
            raise ValueError(f"SST manifest contains no student samples: {manifest_path}")
        dataset = cls(
            records,
            image_size=image_size,
            frames_in=frames_in,
            heatmap_radius_px=heatmap_radius_px,
            image_path_rewrites=image_path_rewrites,
        )
        dataset.summary["source_kind"] = "sst_pseudo_label"
        dataset.summary["weight_policy"] = "frozen fixed pseudo weight 0.25"
        dataset.summary["manifest_trust_boundary"] = dict(manifest["trainer_validation"])
        return dataset


class CombinedStage2Dataset:
    def __init__(self, datasets: Sequence[Any]) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("at least one stage-2 data source is required")
        self.offsets: list[tuple[int, int, Any]] = []
        total = 0
        for dataset in self.datasets:
            length = len(dataset)
            self.offsets.append((total, total + length, dataset))
            total += length
        self.summary = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_stage2_combined_dataset_summary",
            "source_count": len(self.datasets),
            "selected_sample_count": total,
            "sources": [getattr(dataset, "summary", {}) for dataset in self.datasets],
            "dataset_provenance": _dataset_provenance(
                [record for dataset in self.datasets for record in getattr(dataset, "records", ())]
            ),
        }

    def __len__(self) -> int:
        return self.offsets[-1][1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        for start, end, dataset in self.offsets:
            if start <= index < end:
                return dataset[index - start]
        raise IndexError(index)


def load_and_validate_b0_split(split_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the content-pinned B0 train artifact and independently revalidate it."""

    expected_root = (ROOT / DEFAULT_B0_SPLIT_ROOT).resolve(strict=True)
    root = _required_canonical_directory(
        split_root,
        "B0 split_root",
        expected=expected_root,
    )
    report_path = root / "report.json"
    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"
    expected_hashes = {
        report_path: B0_REPORT_SHA256,
        train_path: B0_TRAIN_SHA256,
        validation_path: B0_VALIDATION_SHA256,
    }
    for path, expected_sha in expected_hashes.items():
        canonical_path = _required_canonical_file(
            path,
            f"accepted B0 artifact {path.name}",
            expected=path,
        )
        actual_sha = _sha256_file(canonical_path)
        if actual_sha != expected_sha:
            raise ValueError(f"accepted B0 artifact SHA mismatch for {path}: expected={expected_sha} actual={actual_sha}")

    report = _read_json_object(report_path)
    if report.get("artifact_type") != "racketsport_ball_regroup_parent_source_split":
        raise ValueError("B0 report artifact_type mismatch")
    if report.get("verdict") != "BALL_CLEAN_JUDGE":
        raise ValueError(f"B0 report is not accepted: verdict={report.get('verdict')}")
    if report.get("split_semantics") != "parent_source":
        raise ValueError("B0 report must use parent_source split semantics")
    counts = _required_mapping(report.get("split_counts"), "B0 report split_counts")
    if int(counts.get("train", -1)) != B0_EXPECTED_TRAIN_ROWS or int(counts.get("validation", -1)) != B0_EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"B0 report count mismatch: {dict(counts)}")
    if set(report.get("train_sources") or []) != B0_TRAIN_SOURCE_IDS:
        raise ValueError("B0 report train_sources are not the frozen four source parents")
    if set(report.get("validation_sources") or []) != B0_JUDGE_PARENT_IDS:
        raise ValueError("B0 report validation_sources are not the frozen HyU/Ezz judge parents")
    input_contract = _required_mapping(report.get("input_contract"), "B0 report input_contract")
    reviewed_per_source = _required_mapping(
        input_contract.get("reviewed_per_source"),
        "B0 report input_contract.reviewed_per_source",
    )
    excluded_judge_reviewed_rows = sum(
        _nonnegative_int(
            reviewed_per_source.get(source_id),
            f"B0 reviewed row count for {source_id}",
        )
        for source_id in B0_JUDGE_PARENT_IDS
    )
    if excluded_judge_reviewed_rows != B0_EXPECTED_EXCLUDED_JUDGE_REVIEWED_ROWS:
        raise ValueError(
            "B0 report does not bind the exact 960 excluded judge-parent reviewed rows: "
            f"actual={excluded_judge_reviewed_rows}"
        )
    checks = _required_mapping(report.get("checks"), "B0 report checks")
    for name, check in checks.items():
        if not isinstance(check, Mapping) or check.get("verdict") != "PASS":
            raise ValueError(f"B0 report check is not PASS: {name}")
    weight_policy = _required_mapping(report.get("weight_policy"), "B0 report weight_policy")
    expected_weights = {"scratch": 1.0, "corrected_prelabel": 1.0, "confirmed_prelabel": 0.25}
    for lineage, expected_weight in expected_weights.items():
        if not _exact_number(weight_policy.get(lineage), expected_weight):
            raise ValueError(f"B0 report {lineage} weight is not frozen at {expected_weight}")

    train_rows = _read_jsonl_objects(train_path)
    validation_rows = _read_jsonl_objects(validation_path)
    if len(train_rows) != B0_EXPECTED_TRAIN_ROWS or len(validation_rows) != B0_EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"B0 JSONL counts mismatch: train={len(train_rows)} validation={len(validation_rows)}")
    train_keys: set[str] = set()
    for row in train_rows:
        _validate_b0_row(row, expected_split="train", allowed_sources=B0_TRAIN_SOURCE_IDS)
        key = str(row["row_key"])
        if key in train_keys:
            raise ValueError(f"duplicate B0 training row_key: {key}")
        train_keys.add(key)
    validation_keys: set[str] = set()
    for row in validation_rows:
        _validate_b0_row(row, expected_split="validation", allowed_sources=B0_JUDGE_PARENT_IDS)
        if row.get("lineage_class") != "scratch" or row.get("ground_truth") is not True:
            raise ValueError(f"B0 judge row is not scratch ground truth: {row.get('row_key')}")
        key = str(row["row_key"])
        if key in validation_keys:
            raise ValueError(f"duplicate B0 validation row_key: {key}")
        validation_keys.add(key)
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"B0 train/validation row leakage: {sorted(overlap)[:5]}")
    if any(_row_mentions_source(row, source) for row in train_rows for source in B0_JUDGE_PARENT_IDS):
        raise ValueError("B0 HyU/Ezz judge-parent row reached the frozen training artifact")
    lineage_counts: dict[str, int] = {}
    for row in train_rows:
        lineage = str(row["lineage_class"])
        lineage_counts[lineage] = lineage_counts.get(lineage, 0) + 1
    return train_rows, {
        "b0_report_sha256": B0_REPORT_SHA256,
        "b0_train_sha256": B0_TRAIN_SHA256,
        "b0_validation_sha256": B0_VALIDATION_SHA256,
        "b0_train_row_count": len(train_rows),
        "b0_validation_row_count": len(validation_rows),
        "b0_train_sources": sorted(B0_TRAIN_SOURCE_IDS),
        "b0_judge_parent_exclusion": sorted(B0_JUDGE_PARENT_IDS),
        "b0_excluded_judge_reviewed_row_count": excluded_judge_reviewed_rows,
        "b0_lineage_counts": dict(sorted(lineage_counts.items())),
    }


def _validate_b0_source_media(
    rows: Sequence[Mapping[str, Any]],
    *,
    rally_root: str | Path,
) -> dict[str, str]:
    """Bind every decoded B0 training clip to its frozen canonical path and bytes."""

    root = _required_canonical_directory(
        rally_root,
        "B0 rally_root",
        expected=ROOT / DEFAULT_RALLY_ROOT,
    )
    required_relative_paths = {
        f"{row['source_id']}/{row['clip_id']}.mp4"
        for row in rows
    }
    if required_relative_paths != set(B0_SOURCE_MEDIA_SHA256):
        raise ValueError(
            "B0 source-media inventory differs from the frozen 31 clips: "
            f"missing={sorted(set(B0_SOURCE_MEDIA_SHA256) - required_relative_paths)} "
            f"unexpected={sorted(required_relative_paths - set(B0_SOURCE_MEDIA_SHA256))}"
        )
    observed: dict[str, str] = {}
    for relative_path in sorted(required_relative_paths):
        expected_path = root / relative_path
        media_path = _required_canonical_file(
            expected_path,
            f"B0 source media {relative_path}",
            expected=expected_path,
        )
        actual_sha = _sha256_file(media_path)
        expected_sha = B0_SOURCE_MEDIA_SHA256[relative_path]
        if actual_sha != expected_sha:
            raise ValueError(
                f"B0 source media SHA mismatch for {relative_path}: "
                f"expected={expected_sha} actual={actual_sha}"
            )
        observed[relative_path] = actual_sha
    return observed


def _validate_b0_row(
    row: Mapping[str, Any],
    *,
    expected_split: str,
    allowed_sources: frozenset[str],
) -> None:
    source_id = str(row.get("source_id") or "")
    parent_source_id = str(row.get("parent_source_id") or "")
    clip_id = str(row.get("clip_id") or "")
    if source_id not in allowed_sources or parent_source_id != source_id:
        raise ValueError(f"B0 row has noncanonical source/parent identity: {row.get('row_key')}")
    if not clip_id.startswith(f"{source_id}_rally_"):
        raise ValueError(f"B0 row clip does not belong to source parent: {row.get('row_key')}")
    if row.get("split") != expected_split:
        raise ValueError(f"B0 row split mismatch: {row.get('row_key')}")
    frame_index = _nonnegative_int(row.get("frame_index"), f"B0 row {row.get('row_key')} frame_index")
    expected_key = f"{clip_id}:{frame_index:06d}"
    if row.get("row_key") != expected_key:
        raise ValueError(f"B0 row_key mismatch: expected={expected_key} actual={row.get('row_key')}")
    if expected_split == "train" and row.get("evaluation_eligible") is not False:
        raise ValueError(f"B0 training row marked evaluation eligible: {expected_key}")
    lineage = str(row.get("lineage_class") or "")
    expected_lineage = {
        "scratch": (1.0, False, True),
        "corrected_prelabel": (1.0, False, True),
        "confirmed_prelabel": (0.25, True, False),
    }
    if lineage not in expected_lineage:
        raise ValueError(f"B0 row has invalid lineage_class: {expected_key}")
    expected_weight, expected_teacher, expected_ground_truth = expected_lineage[lineage]
    if not _exact_number(row.get("training_weight"), expected_weight):
        raise ValueError(f"B0 row lineage weight mismatch: {expected_key}")
    if row.get("teacher_derived") is not expected_teacher or row.get("ground_truth") is not expected_ground_truth:
        raise ValueError(f"B0 row authority mismatch: {expected_key}")
    label = _required_mapping(row.get("final_label"), f"B0 row {expected_key} final_label")
    if not isinstance(label.get("ball_present"), bool):
        raise ValueError(f"B0 row {expected_key} ball_present must be boolean")
    if bool(label["ball_present"]):
        _validated_bbox(label.get("bbox_xyxy"), field=f"B0 row {expected_key} bbox")
    elif label.get("bbox_xyxy") is not None:
        raise ValueError(f"B0 negative row {expected_key} must not carry a bbox")
    for forbidden in B0_JUDGE_PARENT_IDS if expected_split == "train" else ():
        if _row_mentions_source(row, forbidden):
            raise ValueError(f"B0 judge-parent {forbidden} reached a training row: {expected_key}")


def load_and_validate_production_sst_manifest(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Revalidate a B1 manifest without trusting the builder's verdict or counts."""

    path = Path(manifest_path)
    manifest = _read_json_object(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported SST manifest schema_version: {manifest.get('schema_version')}")
    if manifest.get("artifact_type") != "racketsport_ball_sst_manifest":
        raise ValueError(f"unexpected SST manifest artifact_type: {manifest.get('artifact_type')}")
    gate = _required_mapping(manifest.get("gate"), "SST gate")
    if gate.get("verdict") != "PASS":
        raise ValueError(f"SST gate verdict must be PASS, got {gate.get('verdict')}")
    if manifest.get("production_eligible") is not True:
        raise ValueError("SST manifest is not production_eligible (CLI policy overrides are forbidden)")
    if manifest.get("production_policy_selected") is not True or manifest.get("policy_override_fields") != []:
        raise ValueError("SST manifest was built with policy overrides")
    if manifest.get("declared_policy_override_fields") != [] or manifest.get("policy_override_declaration_matches") is not True:
        raise ValueError("SST manifest override declaration is not the frozen production policy")
    if manifest.get("teacher_derived") is not True or manifest.get("ground_truth") is not False:
        raise ValueError("SST manifest must be teacher_derived=true and ground_truth=false")
    if manifest.get("protected_eval_clips_touched") is not False:
        raise ValueError("SST manifest touched protected evaluation clips")
    if manifest.get("decode_status") != "completed":
        raise ValueError(f"SST decoding was not completed: {manifest.get('decode_status')}")
    artifact_verification = _required_mapping(
        manifest.get("artifact_verification"),
        "SST artifact_verification",
    )
    if artifact_verification.get("verified") is not True or artifact_verification.get("status") != "passed":
        raise ValueError("SST producer did not pass canonical artifact verification")

    expected_wasb = _expected_wasb_identity()
    preregistration = _required_mapping(manifest.get("preregistration"), "SST preregistration")
    exact_preregistered = {
        "policy_id": SST_PRODUCTION_POLICY_ID,
        "teacher_confidence_min": SST_TEACHER_CONFIDENCE_MIN,
        "agreement_radius_px": SST_AGREEMENT_RADIUS_PX,
        "pseudo_weight": SST_PSEUDO_WEIGHT,
        "temporal_max_gap_source_frames": SST_TEMPORAL_MAX_GAP_SOURCE_FRAMES,
    }
    for key, expected in exact_preregistered.items():
        actual = preregistration.get(key)
        if isinstance(expected, float):
            matches = _exact_number(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"SST preregistration {key} mismatch: expected={expected!r} actual={actual!r}")
    if preregistration.get("temporal_geometry") != SST_TEMPORAL_GEOMETRY_POLICY:
        raise ValueError("SST temporal geometry preregistration is not the frozen independent-WASB policy")
    if preregistration.get("expected_source_video_sha256") != SST_EXPECTED_SOURCE_VIDEO_SHA256:
        raise ValueError("SST preregistration source-video SHA map is not the frozen seven-source identity")
    if manifest.get("preregistered_parameters") != preregistration:
        raise ValueError("SST preregistration aliases disagree")
    requested_parameters = _required_mapping(manifest.get("requested_parameters"), "SST requested_parameters")
    for key in ("teacher_confidence_min", "agreement_radius_px", "pseudo_weight"):
        if not _exact_number(requested_parameters.get(key), float(exact_preregistered[key])):
            raise ValueError(f"SST requested parameter {key} is not the frozen production value")
    builder_path = ROOT / "scripts/racketsport/build_pbvision_ball_sst.py"
    builder_sha = _sha256_file(builder_path)
    adapter_path = (ROOT / SST_WASB_ADAPTER_PATH).resolve(strict=True)
    adapter_sha = _sha256_file(adapter_path)
    current_commit = _git_commit("HEAD")
    adapter_head_bytes = subprocess.run(
        ["git", "show", f"{current_commit}:{SST_WASB_ADAPTER_PATH.as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    adapter_head_sha = hashlib.sha256(adapter_head_bytes).hexdigest()
    if adapter_sha != adapter_head_sha:
        raise ValueError("SST WASB adapter working bytes differ from the pinned HEAD blob")
    if preregistration.get("builder_code_sha256") != builder_sha:
        raise ValueError("SST builder code identity does not match this working tree")
    if preregistration.get("builder_git_commit") != current_commit:
        raise ValueError("SST builder git commit does not match pinned HEAD")
    if preregistration.get("wasb_adapter_code_sha256") != adapter_sha:
        raise ValueError("SST WASB adapter code identity does not match this working tree")
    if preregistration.get("wasb_adapter_git_commit") != current_commit:
        raise ValueError("SST WASB adapter git commit does not match pinned HEAD")
    builder_identity = _required_mapping(manifest.get("builder_identity"), "SST builder_identity")
    expected_builder_identity = {
        "builder_path": "scripts/racketsport/build_pbvision_ball_sst.py",
        "builder_code_sha256": builder_sha,
        "builder_git_commit": current_commit,
        "wasb_adapter_path": str(SST_WASB_ADAPTER_PATH),
        "wasb_adapter_code_sha256": adapter_sha,
        "wasb_adapter_git_commit": current_commit,
    }
    if dict(builder_identity) != expected_builder_identity:
        raise ValueError("SST builder identity changed or was fabricated")

    gallery_root = _required_canonical_directory(
        manifest.get("gallery_root"),
        "SST gallery_root",
        expected=ROOT / SST_FROZEN_GALLERY_ROOT,
    )
    gallery_bundle = _validate_frozen_pbvision_gallery(gallery_root)
    _validate_manifest_teacher_input_authority(
        preregistration.get("teacher_input_authority"),
        gallery_bundle=gallery_bundle,
    )
    media_root = _required_canonical_directory(manifest.get("media_root"), "SST media_root")
    split_manifest_path = _required_canonical_file(
        manifest.get("split_manifest"),
        "SST split_manifest",
        expected=ROOT / SST_FROZEN_SPLIT_MANIFEST,
    )
    actual_split_sha = _sha256_file(split_manifest_path)
    if actual_split_sha != SST_FROZEN_SPLIT_SHA256:
        raise ValueError("SST split manifest does not match the frozen production SHA")
    checkpoint_path = _required_canonical_file(
        manifest.get("wasb_checkpoint"),
        "SST wasb_checkpoint",
        expected=ROOT / expected_wasb["checkpoint_path"],
    )
    if _sha256_file(checkpoint_path) != expected_wasb["checkpoint_sha256"]:
        raise ValueError("SST WASB checkpoint bytes do not match models/MANIFEST.json")

    dependencies = _required_mapping(manifest.get("dependency_hashes"), "SST dependency_hashes")
    expected_dependencies = {
        "split_manifest_sha256": actual_split_sha,
        "models_manifest_sha256": expected_wasb["models_manifest_sha256"],
        "builder_code_sha256": builder_sha,
        "wasb_adapter_code_sha256": adapter_sha,
        "wasb_checkpoint_sha256": expected_wasb["checkpoint_sha256"],
        "wasb_repo_commit": expected_wasb["repo_commit"],
    }
    for key, expected in expected_dependencies.items():
        if dependencies.get(key) != expected:
            raise ValueError(f"SST top-level dependency mismatch for {key}")
    wasb_identity = _required_mapping(manifest.get("wasb_identity"), "SST wasb_identity")
    if wasb_identity.get("manifest_model_id") != SST_WASB_MODEL_ID:
        raise ValueError("SST WASB manifest model id mismatch")
    if wasb_identity.get("checkpoint_sha256") != expected_wasb["checkpoint_sha256"]:
        raise ValueError("SST WASB checkpoint SHA is not models/MANIFEST.json")
    if wasb_identity.get("repo_commit") != expected_wasb["repo_commit"]:
        raise ValueError("SST WASB repository commit is not models/MANIFEST.json")
    if wasb_identity.get("repo_clean") is not True or wasb_identity.get("production_identity_verified") is not True:
        raise ValueError("SST WASB repository/checkpoint identity was not production verified")
    models_manifest_path = _required_canonical_file(
        wasb_identity.get("models_manifest_path"),
        "SST models_manifest_path",
        expected=ROOT / "models/MANIFEST.json",
    )
    if _sha256_file(models_manifest_path) != expected_wasb["models_manifest_sha256"]:
        raise ValueError("SST models manifest bytes changed")
    identity_checkpoint_path = _required_canonical_file(
        wasb_identity.get("checkpoint_path"),
        "SST wasb_identity.checkpoint_path",
        expected=checkpoint_path,
    )
    if identity_checkpoint_path != checkpoint_path:
        raise ValueError("SST checkpoint path aliases disagree")
    repo_identity = _required_clean_git_repo(
        wasb_identity.get("repo_path"),
        "SST wasb_identity.repo_path",
        expected=ROOT / SST_WASB_REPO_ROOT,
        expected_commit=expected_wasb["repo_commit"],
    )
    if wasb_identity.get("models_manifest_sha256") != expected_wasb["models_manifest_sha256"]:
        raise ValueError("SST WASB identity models-manifest SHA mismatch")
    if wasb_identity.get("expected_checkpoint_sha256") != expected_wasb["checkpoint_sha256"]:
        raise ValueError("SST WASB expected checkpoint SHA mismatch")
    if wasb_identity.get("expected_repo_commit") != expected_wasb["repo_commit"]:
        raise ValueError("SST WASB expected repository commit mismatch")

    source_policy = _required_mapping(manifest.get("source_policy"), "SST source_policy")
    if set(source_policy.get("train_ids") or []) != SST_TRAIN_SOURCE_IDS:
        raise ValueError("SST source policy does not contain the canonical seven train IDs")
    if source_policy.get("positive_rows_only") is not True or source_policy.get("teacher_absence_policy") != "ignored_never_negative":
        raise ValueError("SST source policy does not preserve positive-only teacher semantics")

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ValueError("SST manifest requires clips list")
    samples: list[dict[str, Any]] = []
    clip_ids: set[str] = set()
    media_sha_cache: dict[Path, str] = {}
    sample_keys: set[tuple[str, int]] = set()
    dependency_root_path = path.parent / f"{path.stem}_dependencies"
    dependency_root = _required_canonical_directory(
        str(dependency_root_path),
        "SST dependency_root",
        expected=dependency_root_path,
    )
    for clip_index, clip_raw in enumerate(clips):
        clip = _required_mapping(clip_raw, f"SST clip {clip_index}")
        clip_id = str(clip.get("clip_id") or "")
        if clip_id not in SST_TRAIN_SOURCE_IDS or clip_id in clip_ids:
            raise ValueError(f"SST clip has duplicate/noncanonical source ID: {clip_id}")
        clip_ids.add(clip_id)
        if clip.get("split") != "train" or clip.get("teacher_derived") is not True or clip.get("ground_truth") is not False:
            raise ValueError(f"SST clip authority mismatch: {clip_id}")
        if clip.get("canonical_source_id") != clip_id:
            raise ValueError(f"SST clip canonical source mismatch: {clip_id}")
        if clip.get("source_video_sha256") != SST_EXPECTED_SOURCE_VIDEO_SHA256[clip_id]:
            raise ValueError(f"SST clip source-video SHA mismatch: {clip_id}")
        expected_clip_media = (media_root / clip_id / "max.mp4").resolve(strict=True)
        resolved_clip_media = _required_canonical_file(
            clip.get("rally_video"),
            f"SST clip {clip_id} rally_video",
            expected=expected_clip_media,
        )
        if not resolved_clip_media.is_relative_to(media_root):
            raise ValueError(f"SST clip media path is not canonical: {clip_id}")
        if resolved_clip_media not in media_sha_cache:
            media_sha_cache[resolved_clip_media] = _sha256_file(resolved_clip_media)
        if media_sha_cache[resolved_clip_media] != SST_EXPECTED_SOURCE_VIDEO_SHA256[clip_id]:
            raise ValueError(f"SST clip media bytes do not match preregistration: {clip_id}")
        clip_dependencies = _required_mapping(clip.get("dependencies"), f"SST clip {clip_id} dependencies")
        for key in SST_REQUIRED_ROW_DEPENDENCY_HASHES:
            if key not in clip_dependencies:
                raise ValueError(f"SST clip {clip_id} missing dependency {key}")
        for key in SST_GLOBAL_DEPENDENCY_KEYS:
            if clip_dependencies.get(key) != expected_dependencies[key]:
                raise ValueError(f"SST clip {clip_id} global dependency mismatch for {key}")
        if clip_dependencies.get("source_video_sha256") != media_sha_cache[resolved_clip_media]:
            raise ValueError(f"SST clip {clip_id} dependency source-video SHA mismatch")
        source_width = _positive_int(clip.get("source_width"), f"SST clip {clip_id} source_width")
        source_height = _positive_int(clip.get("source_height"), f"SST clip {clip_id} source_height")
        decoded_width, decoded_height = _video_size(resolved_clip_media)
        if (source_width, source_height) != (decoded_width, decoded_height):
            raise ValueError(f"SST clip {clip_id} decoded dimensions do not match manifest")
        frame_times, wasb_frames, teacher_by_source_frame = _rehash_sst_clip_dependencies(
            manifest_path=path,
            dependency_root=dependency_root,
            gallery_root=gallery_root,
            clip_id=clip_id,
            clip_dependencies=clip_dependencies,
            expected_dependencies=expected_dependencies,
            source_video_sha256=media_sha_cache[resolved_clip_media],
            source_video=resolved_clip_media,
            source_width=source_width,
            source_height=source_height,
            checkpoint_path=checkpoint_path,
            wasb_repo=repo_identity["path"],
            adapter_sha256=adapter_sha,
        )
        if not math.isclose(
            _finite_float(clip.get("fps"), f"SST clip {clip_id} fps"),
            frame_times["fps"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"SST clip {clip_id} fps does not match frame-times artifact")
        clip_samples = clip.get("samples")
        if not isinstance(clip_samples, list) or int(clip.get("sample_count", -1)) != len(clip_samples):
            raise ValueError(f"SST clip sample_count mismatch: {clip_id}")
        for sample_index, sample_raw in enumerate(clip_samples):
            sample = dict(_required_mapping(sample_raw, f"SST sample {clip_id}:{sample_index}"))
            _validate_sst_sample(
                sample,
                clip_id=clip_id,
                builder_sha=builder_sha,
                expected_wasb=expected_wasb,
                top_dependencies=dependencies,
                clip_dependencies=clip_dependencies,
                media_sha_cache=media_sha_cache,
                resolved_clip_media=resolved_clip_media,
                source_width=source_width,
                source_height=source_height,
                frame_times=frame_times,
                wasb_frames=wasb_frames,
                teacher_by_source_frame=teacher_by_source_frame,
            )
            key = (clip_id, int(sample["frame_index"]))
            if key in sample_keys:
                raise ValueError(f"duplicate SST source window: {clip_id}:{key[1]}")
            sample_keys.add(key)
            samples.append(sample)

    if clip_ids != SST_TRAIN_SOURCE_IDS or len(clips) != len(SST_TRAIN_SOURCE_IDS):
        raise ValueError("SST manifest must retain the complete canonical seven-source inventory")
    accepted_sources = len({str(sample["clip_id"]) for sample in samples})
    if len(samples) < SST_MIN_ACCEPTED_WINDOWS or accepted_sources < SST_MIN_ACCEPTED_SOURCES:
        raise ValueError(f"SST minimum gate failed: windows={len(samples)} sources={accepted_sources}")
    if int(manifest.get("accepted_windows", -1)) != len(samples) or int(manifest.get("accepted_sources", -1)) != accepted_sources:
        raise ValueError("SST top-level accepted counts do not match independently counted rows")
    if int(manifest.get("holdout_rows_present", -1)) != 0 or int(manifest.get("decode_failures", -1)) != 0:
        raise ValueError("SST manifest has holdout rows or decode failures")
    _validate_gate_count(gate, "accepted_windows", len(samples), SST_MIN_ACCEPTED_WINDOWS)
    _validate_gate_count(gate, "accepted_sources", accepted_sources, SST_MIN_ACCEPTED_SOURCES)
    _validate_gate_count(gate, "holdout_rows_present", 0, 0)
    _validate_gate_count(gate, "decode_failures", 0, 0)
    _validate_gate_count(gate, "production_eligible", 1, 1)
    _validate_gate_count(gate, "artifacts_verified", 1, 1)
    if int(artifact_verification.get("verified_clip_count", -1)) != len(clips):
        raise ValueError("SST artifact verification clip count mismatch")
    if int(artifact_verification.get("verified_sample_count", -1)) != len(samples):
        raise ValueError("SST artifact verification sample count mismatch")
    expected_replay_sha_by_clip = {
        str(clip["clip_id"]): str(_required_mapping(clip.get("dependencies"), "SST clip dependencies")["wasb_predictions_csv_sha256"])
        for clip in clips
    }
    if artifact_verification.get("official_wasb_replay_verified") is not True:
        raise ValueError("SST producer did not verify official WASB inference replay")
    if int(artifact_verification.get("official_wasb_replay_clip_count", -1)) != len(clips):
        raise ValueError("SST producer replay clip count mismatch")
    if artifact_verification.get("replayed_prediction_sha256_by_clip") != expected_replay_sha_by_clip:
        raise ValueError("SST producer replayed prediction SHA map mismatch")
    if artifact_verification.get("pbvision_gallery_authority_id") != SST_PBVISION_GALLERY_AUTHORITY_ID:
        raise ValueError("SST producer PBVision gallery authority id mismatch")
    if (
        artifact_verification.get("verified_pbvision_gallery_sha256_by_source")
        != gallery_bundle["sha256_by_source"]
    ):
        raise ValueError("SST producer PBVision gallery verification map mismatch")
    for key, expected in {
        "split_manifest_sha256": actual_split_sha,
        "builder_code_sha256": builder_sha,
        "wasb_adapter_code_sha256": adapter_sha,
        "wasb_checkpoint_sha256": expected_wasb["checkpoint_sha256"],
        "wasb_repo_commit": repo_identity["commit"],
    }.items():
        if artifact_verification.get(key) != expected:
            raise ValueError(f"SST artifact verification identity mismatch for {key}")
    manifest["trainer_validation"] = {
        "verdict": "PASS",
        "manifest_sha256": _sha256_file(path),
        "accepted_windows_recounted": len(samples),
        "accepted_sources_recounted": accepted_sources,
        "canonical_source_ids": sorted({str(sample["clip_id"]) for sample in samples}),
        "media_sha256_recomputed": len(media_sha_cache),
        "builder_code_sha256_recomputed": builder_sha,
        "wasb_adapter_code_sha256_recomputed": adapter_sha,
        "wasb_checkpoint_sha256_from_models_manifest": expected_wasb["checkpoint_sha256"],
        "wasb_repo_commit_from_models_manifest": expected_wasb["repo_commit"],
        "dependency_files_rehashed": len(clips) * 7 + 4,
        "pbvision_gallery_bundle_sha256": gallery_bundle["bundle_sha256"],
        "pbvision_gallery_file_count": gallery_bundle["file_count"],
        "wasb_official_inference_replays": len(clips),
        "agreement_rows_cross_checked_against_wasb_track": len(samples),
    }
    return manifest, samples


def _required_canonical_directory(value: Any, field: str, *, expected: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"missing {field}: {path}")
    resolved = path.resolve(strict=True)
    _require_lexical_canonical_path(path, resolved=resolved, field=field)
    if expected is not None and resolved != expected.resolve(strict=True):
        raise ValueError(f"{field} is not the frozen canonical path: {resolved}")
    return resolved


def _required_canonical_file(value: Any, field: str, *, expected: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {field}: {path}")
    resolved = path.resolve(strict=True)
    _require_lexical_canonical_path(path, resolved=resolved, field=field)
    if expected is not None and resolved != Path(expected).resolve(strict=True):
        raise ValueError(f"{field} is not the canonical bound file: expected={expected} actual={resolved}")
    return resolved


def _require_lexical_canonical_path(path: Path, *, resolved: Path, field: str) -> None:
    lexical_absolute = path if path.is_absolute() else Path.cwd() / path
    if lexical_absolute != resolved:
        raise ValueError(f"{field} must use its direct canonical path, not an alias: {path}")
    for component in (lexical_absolute, *lexical_absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{field} contains a symlink path component: {component}")


def _required_clean_git_repo(
    value: Any,
    field: str,
    *,
    expected: Path,
    expected_commit: str,
) -> dict[str, Any]:
    repo = _required_canonical_directory(value, field, expected=expected)
    commit_result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip()
    if commit != expected_commit:
        raise ValueError(f"{field} commit mismatch: expected={expected_commit} actual={commit}")
    if status_result.stdout.strip():
        raise ValueError(f"{field} must be clean for production SST use")
    return {"path": str(repo), "commit": commit, "clean": True}


def _compute_pbvision_gallery_bundle(gallery_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    sha256_by_source: dict[str, dict[str, str]] = {}
    for source_id in sorted(SST_TRAIN_SOURCE_IDS):
        sha256_by_source[source_id] = {}
        for filename in SST_PBVISION_GALLERY_FILENAMES:
            relative_path = f"{source_id}/{filename}"
            expected_path = gallery_root / relative_path
            path = _required_canonical_file(
                expected_path,
                f"SST frozen PBVision gallery {relative_path}",
                expected=expected_path,
            )
            sha256 = _sha256_file(path)
            rows.append({"path": relative_path, "sha256": sha256})
            sha256_by_source[source_id][filename] = sha256
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(rows),
        "files": rows,
        "sha256_by_source": sha256_by_source,
    }


def _validate_frozen_pbvision_gallery(gallery_root: Path) -> dict[str, Any]:
    identity = _compute_pbvision_gallery_bundle(gallery_root)
    if identity["bundle_sha256"] != SST_FROZEN_GALLERY_BUNDLE_SHA256:
        raise ValueError(
            "SST frozen PBVision gallery bundle SHA mismatch: "
            f"expected={SST_FROZEN_GALLERY_BUNDLE_SHA256} "
            f"actual={identity['bundle_sha256']}"
        )
    return identity


def _validate_manifest_teacher_input_authority(
    value: Any,
    *,
    gallery_bundle: Mapping[str, Any],
) -> None:
    authority = _required_mapping(value, "SST preregistration teacher_input_authority")
    if authority.get("authority_id") != SST_PBVISION_GALLERY_AUTHORITY_ID:
        raise ValueError("SST teacher-input authority id is not the frozen production authority")
    if authority.get("canonical_gallery_relative_path") != SST_FROZEN_GALLERY_ROOT.as_posix():
        raise ValueError("SST teacher-input authority gallery path mismatch")
    filenames = authority.get("artifact_filenames")
    if (
        not isinstance(filenames, list)
        or set(filenames) != set(SST_PBVISION_GALLERY_FILENAMES)
        or len(filenames) != len(SST_PBVISION_GALLERY_FILENAMES)
    ):
        raise ValueError("SST teacher-input authority artifact inventory mismatch")
    if authority.get("expected_sha256_by_source") != gallery_bundle["sha256_by_source"]:
        raise ValueError("SST teacher-input authority SHA map mismatch")


def _rehash_sst_clip_dependencies(
    *,
    manifest_path: Path,
    dependency_root: Path,
    gallery_root: Path,
    clip_id: str,
    clip_dependencies: Mapping[str, Any],
    expected_dependencies: Mapping[str, str],
    source_video_sha256: str,
    source_video: Path,
    source_width: int,
    source_height: int,
    checkpoint_path: Path,
    wasb_repo: str | Path,
    adapter_sha256: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[int, dict[str, Any]]]:
    del manifest_path
    gallery_paths = {
        "pbvision_cv_export_sha256": gallery_root / clip_id / "cv_export.json",
        "pbvision_metadata_sha256": gallery_root / clip_id / "api_get_metadata.json",
        "pbvision_provenance_sha256": gallery_root / clip_id / "video_provenance.json",
    }
    resolved_gallery = {
        key: _required_canonical_file(path, f"SST clip {clip_id} {path.name}", expected=path)
        for key, path in gallery_paths.items()
    }
    metadata_payload = _read_json_object(resolved_gallery["pbvision_metadata_sha256"])
    metadata = _required_mapping(metadata_payload.get("metadata"), f"SST clip {clip_id} gallery metadata")
    if (
        _positive_int(metadata.get("width"), f"SST clip {clip_id} metadata width"),
        _positive_int(metadata.get("height"), f"SST clip {clip_id} metadata height"),
    ) != (source_width, source_height):
        raise ValueError(f"SST clip {clip_id} gallery dimensions do not match media")
    provenance = _read_json_object(resolved_gallery["pbvision_provenance_sha256"])
    if provenance.get("video_id") != clip_id or provenance.get("source_video_url") != (
        f"https://storage.googleapis.com/pbv-pro/{clip_id}/max.mp4"
    ):
        raise ValueError(f"SST clip {clip_id} gallery provenance identity mismatch")
    cv_export = _read_json_object(resolved_gallery["pbvision_cv_export_sha256"])

    clip_dependency_root_path = dependency_root / clip_id
    clip_dependency_root = _required_canonical_directory(
        str(clip_dependency_root_path),
        f"SST clip {clip_id} dependency root",
        expected=clip_dependency_root_path,
    )
    expected_files = {
        "frame_times_path": clip_dependency_root / "frame_times.json",
        "wasb_ball_track": clip_dependency_root / "wasb_ball_track.json",
        "wasb_metadata_path": clip_dependency_root / "wasb_ball_track_metadata.json",
        "wasb_predictions_csv_path": clip_dependency_root / "wasb_predictions.csv",
    }
    resolved_files = {
        key: _required_canonical_file(
            clip_dependencies.get(key),
            f"SST clip {clip_id} {key}",
            expected=expected,
        )
        for key, expected in expected_files.items()
    }
    frame_times = _validate_sst_frame_times(
        _read_json_object(resolved_files["frame_times_path"]),
        clip_id=clip_id,
        source_video_sha256=source_video_sha256,
        source_width=source_width,
        source_height=source_height,
    )
    encoded_timing = _probe_canonical_media_timing(source_video, clip_id=clip_id)
    _validate_sst_frame_times_against_media(
        frame_times,
        encoded_timing=encoded_timing,
        clip_id=clip_id,
    )
    _validate_sst_wasb_predictions_csv(
        resolved_files["wasb_predictions_csv_path"],
        pts_s=frame_times["pts_s"],
        width=source_width,
        height=source_height,
        clip_id=clip_id,
    )
    wasb_payload = _read_json_object(resolved_files["wasb_ball_track"])
    regenerated_track = wasb_csv_to_ball_track(
        resolved_files["wasb_predictions_csv_path"],
        fps=frame_times["fps"],
        frame_times=resolved_files["frame_times_path"],
        visible_threshold=SST_TEACHER_CONFIDENCE_MIN,
        input_preprocessing="official",
    )
    if wasb_payload != regenerated_track:
        raise ValueError(f"SST clip {clip_id} WASB track does not regenerate from its bound prediction CSV")
    wasb_frames = _validate_sst_wasb_track(
        wasb_payload,
        pts_s=frame_times["pts_s"],
        fps=frame_times["fps"],
        width=source_width,
        height=source_height,
        clip_id=clip_id,
    )
    expected_wasb_bindings = {
        "source_video_sha256": source_video_sha256,
        "frame_times_sha256": _sha256_file(resolved_files["frame_times_path"]),
        "wasb_predictions_csv_sha256": _sha256_file(resolved_files["wasb_predictions_csv_path"]),
        "wasb_ball_track_sha256": _sha256_file(resolved_files["wasb_ball_track"]),
        "wasb_checkpoint_sha256": expected_dependencies["wasb_checkpoint_sha256"],
        "wasb_repo_commit": expected_dependencies["wasb_repo_commit"],
        "wasb_adapter_code_sha256": adapter_sha256,
    }
    wasb_metadata = _read_json_object(resolved_files["wasb_metadata_path"])
    runtime = _validate_sst_wasb_run_metadata(
        wasb_metadata,
        predictions_csv=resolved_files["wasb_predictions_csv_path"],
        ball_track=resolved_files["wasb_ball_track"],
        source_video=source_video,
        checkpoint=checkpoint_path,
        wasb_repo=Path(wasb_repo),
        frame_times=frame_times,
        source_width=source_width,
        source_height=source_height,
        expected_bindings=expected_wasb_bindings,
        clip_id=clip_id,
    )
    if clip_dependencies.get("wasb_runtime") != wasb_metadata:
        raise ValueError(f"SST clip {clip_id} embedded WASB runtime metadata mismatch")
    replayed_csv = _run_official_wasb_replay(
        source_video=source_video,
        checkpoint=checkpoint_path,
        wasb_repo=Path(wasb_repo),
        frame_times_path=resolved_files["frame_times_path"],
        fps=frame_times["fps"],
        batch_size=_positive_int(runtime.get("batch_size"), f"SST clip {clip_id} replay batch_size"),
        device=str(runtime.get("device")),
    )
    if replayed_csv != resolved_files["wasb_predictions_csv_path"].read_bytes():
        raise ValueError(f"SST clip {clip_id} official WASB inference replay differs from the bound prediction CSV")
    teacher_by_source_frame = _extract_bound_teacher_observations(
        cv_export,
        metadata_payload=metadata_payload,
        pts_s=frame_times["pts_s"],
        width=source_width,
        height=source_height,
        clip_id=clip_id,
    )

    actual_dependencies = {
        **dict(expected_dependencies),
        **{key: _sha256_file(path) for key, path in resolved_gallery.items()},
        "source_video_sha256": source_video_sha256,
        "frame_times_sha256": _sha256_file(resolved_files["frame_times_path"]),
        "wasb_ball_track_sha256": _sha256_file(resolved_files["wasb_ball_track"]),
        "wasb_metadata_sha256": _sha256_file(resolved_files["wasb_metadata_path"]),
        "wasb_predictions_csv_sha256": _sha256_file(resolved_files["wasb_predictions_csv_path"]),
        "wasb_adapter_code_sha256": adapter_sha256,
    }
    for key in SST_REQUIRED_ROW_DEPENDENCY_HASHES:
        if clip_dependencies.get(key) != actual_dependencies[key]:
            raise ValueError(f"SST clip {clip_id} dependency {key} failed canonical rehash")
    return frame_times, wasb_frames, teacher_by_source_frame


def _validate_sst_wasb_predictions_csv(
    path: Path,
    *,
    pts_s: Sequence[float],
    width: int,
    height: int,
    clip_id: str,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Frame", "Visibility", "X", "Y", "Confidence"]:
            raise ValueError(f"SST clip {clip_id} WASB prediction CSV header is not official")
        rows = list(reader)
    if len(rows) != len(pts_s):
        raise ValueError(f"SST clip {clip_id} WASB prediction CSV does not cover every PTS frame")
    for frame_index, row in enumerate(rows):
        if row.get("Frame") != str(frame_index):
            raise ValueError(f"SST clip {clip_id} WASB prediction CSV frame indices are not contiguous")
        visibility = row.get("Visibility")
        if visibility not in {"0", "1"}:
            raise ValueError(f"SST clip {clip_id} WASB prediction CSV visibility must be exactly 0 or 1")
        x = _csv_finite_float(row.get("X"), f"SST clip {clip_id} WASB CSV X/{frame_index}")
        y = _csv_finite_float(row.get("Y"), f"SST clip {clip_id} WASB CSV Y/{frame_index}")
        confidence = _csv_finite_float(
            row.get("Confidence"),
            f"SST clip {clip_id} WASB CSV confidence/{frame_index}",
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"SST clip {clip_id} WASB CSV confidence must be in [0, 1]")
        if visibility == "1":
            if confidence < SST_TEACHER_CONFIDENCE_MIN:
                raise ValueError(f"SST clip {clip_id} WASB CSV visible row is below the frozen threshold")
            if not _inside_xy((x, y), width=width, height=height):
                raise ValueError(f"SST clip {clip_id} WASB CSV visible row is out of bounds")
        elif (x, y) != (0.0, 0.0):
            raise ValueError(f"SST clip {clip_id} WASB CSV invisible row must use zero coordinates")


def _validate_sst_wasb_track(
    payload: Mapping[str, Any],
    *,
    pts_s: Sequence[float],
    fps: float,
    width: int,
    height: int,
    clip_id: str,
) -> list[Mapping[str, Any]]:
    expected_keys = {"schema_version", "fps", "source", "input_preprocessing", "frames", "bounces"}
    if set(payload) != expected_keys or payload.get("schema_version") != 1:
        raise ValueError(f"SST clip {clip_id} WASB track schema is not the official adapter schema")
    if payload.get("source") != "wasb" or payload.get("input_preprocessing") != "official":
        raise ValueError(f"SST clip {clip_id} WASB track is not official WASB output")
    if payload.get("bounces") != []:
        raise ValueError(f"SST clip {clip_id} raw WASB track must not contain bounce postprocessing")
    if not math.isclose(_finite_float(payload.get("fps"), f"SST clip {clip_id} WASB fps"), fps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"SST clip {clip_id} WASB track fps mismatch")
    frames_raw = payload.get("frames")
    if not isinstance(frames_raw, list) or len(frames_raw) != len(pts_s):
        raise ValueError(f"SST clip {clip_id} WASB track does not cover every bound PTS frame")
    frames: list[Mapping[str, Any]] = []
    for frame_index, frame_raw in enumerate(frames_raw):
        frame = _required_mapping(frame_raw, f"SST clip {clip_id} WASB frame {frame_index}")
        if set(frame) != {"t", "xy", "conf", "visible", "approx"}:
            raise ValueError(f"SST clip {clip_id} WASB frame schema is not official")
        if not math.isclose(
            _finite_float(frame.get("t"), f"SST clip {clip_id} WASB t/{frame_index}"),
            float(pts_s[frame_index]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"SST clip {clip_id} WASB frame timestamp differs from bound PTS")
        if frame.get("approx") is not False or not isinstance(frame.get("visible"), bool):
            raise ValueError(f"SST clip {clip_id} WASB frame flags are not raw strict booleans")
        xy = _validated_xy(frame.get("xy"), f"SST clip {clip_id} WASB xy/{frame_index}")
        confidence = _finite_float(frame.get("conf"), f"SST clip {clip_id} WASB conf/{frame_index}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"SST clip {clip_id} WASB confidence must be in [0, 1]")
        if frame["visible"] is True:
            if confidence < SST_TEACHER_CONFIDENCE_MIN or not _inside_xy(xy, width=width, height=height):
                raise ValueError(f"SST clip {clip_id} WASB visible frame is below threshold or out of bounds")
        elif xy != (0.0, 0.0):
            raise ValueError(f"SST clip {clip_id} WASB invisible frame must use zero coordinates")
        frames.append(frame)
    return frames


def _validate_sst_wasb_run_metadata(
    payload: Mapping[str, Any],
    *,
    predictions_csv: Path,
    ball_track: Path,
    source_video: Path,
    checkpoint: Path,
    wasb_repo: Path,
    frame_times: Mapping[str, Any],
    source_width: int,
    source_height: int,
    expected_bindings: Mapping[str, str],
    clip_id: str,
) -> Mapping[str, Any]:
    expected_top_keys = {
        "schema_version", "artifact_type", "status", "source_mode", "predictions_csv", "out",
        "fps", "frame_count", "visible_frame_count", "confidence_semantics", "visible_threshold",
        "input_preprocessing", "non_promotable_measurement_mode", "not_ground_truth",
        "official_repo_url", "official_model_zoo_url", "runtime", "builder_bindings",
    }
    if set(payload) != expected_top_keys or payload.get("schema_version") != 1:
        raise ValueError(f"SST clip {clip_id} WASB run metadata schema is not builder-bound")
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
            raise ValueError(f"SST clip {clip_id} WASB metadata {key} is not production-authentic")
    _require_recorded_path_identity(payload.get("predictions_csv"), predictions_csv, f"SST clip {clip_id} predictions_csv")
    _require_recorded_path_identity(payload.get("out"), ball_track, f"SST clip {clip_id} WASB out")
    fps = _finite_float(frame_times.get("fps"), f"SST clip {clip_id} frame-times fps")
    if not math.isclose(_finite_float(payload.get("fps"), f"SST clip {clip_id} WASB metadata fps"), fps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"SST clip {clip_id} WASB metadata fps mismatch")
    if not _exact_number(payload.get("visible_threshold"), SST_TEACHER_CONFIDENCE_MIN):
        raise ValueError(f"SST clip {clip_id} WASB metadata threshold is not frozen")
    frame_count = _nonnegative_int(payload.get("frame_count"), f"SST clip {clip_id} metadata frame_count")
    if frame_count != int(frame_times["frame_count"]):
        raise ValueError(f"SST clip {clip_id} WASB metadata frame count mismatch")
    visible_count = _nonnegative_int(payload.get("visible_frame_count"), f"SST clip {clip_id} visible count")
    track = _read_json_object(ball_track)
    if visible_count != sum(1 for frame in track["frames"] if frame.get("visible") is True):
        raise ValueError(f"SST clip {clip_id} WASB metadata visible count mismatch")
    if payload.get("builder_bindings") != dict(expected_bindings):
        raise ValueError(f"SST clip {clip_id} WASB metadata builder bindings mismatch")
    runtime = _required_mapping(payload.get("runtime"), f"SST clip {clip_id} WASB runtime")
    expected_runtime_keys = {
        "wasb_repo", "wasb_repo_commit", "wasb_checkpoint", "video", "source_video_fps",
        "source_video_frame_count", "source_video_size", "processed_frame_count",
        "processed_window_count", "read_frame_count", "video_range_seconds", "max_frames",
        "batch_size", "device", "input_preprocessing", "non_promotable_measurement_mode",
        "wall_seconds", "effective_fps", "realtime_factor",
    }
    if set(runtime) != expected_runtime_keys:
        raise ValueError(f"SST clip {clip_id} WASB runtime schema is not bounded official inference")
    _require_recorded_path_identity(runtime.get("wasb_repo"), wasb_repo, f"SST clip {clip_id} WASB runtime repo")
    _require_recorded_path_identity(runtime.get("video"), source_video, f"SST clip {clip_id} WASB runtime video")
    if runtime.get("wasb_repo_commit") != expected_bindings["wasb_repo_commit"]:
        raise ValueError(f"SST clip {clip_id} WASB runtime repo commit mismatch")
    checkpoint_binding = _required_mapping(runtime.get("wasb_checkpoint"), f"SST clip {clip_id} runtime checkpoint")
    if set(checkpoint_binding) != {"path", "sha256"}:
        raise ValueError(f"SST clip {clip_id} WASB runtime checkpoint binding malformed")
    _require_recorded_path_identity(checkpoint_binding.get("path"), checkpoint, f"SST clip {clip_id} runtime checkpoint")
    if checkpoint_binding.get("sha256") != expected_bindings["wasb_checkpoint_sha256"]:
        raise ValueError(f"SST clip {clip_id} WASB runtime checkpoint SHA mismatch")
    if not math.isclose(_finite_float(runtime.get("source_video_fps"), f"SST clip {clip_id} source fps"), fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"SST clip {clip_id} WASB runtime source fps mismatch")
    expected_counts = {
        "source_video_frame_count": frame_count,
        "processed_frame_count": frame_count,
        "processed_window_count": frame_count - 2,
        "read_frame_count": frame_count,
    }
    for key, expected in expected_counts.items():
        if _nonnegative_int(runtime.get(key), f"SST clip {clip_id} runtime {key}") != expected:
            raise ValueError(f"SST clip {clip_id} WASB runtime {key} mismatch")
    if runtime.get("source_video_size") != [source_width, source_height]:
        raise ValueError(f"SST clip {clip_id} WASB runtime source size mismatch")
    if runtime.get("video_range_seconds") is not None or runtime.get("max_frames") is not None:
        raise ValueError(f"SST clip {clip_id} WASB runtime did not cover the full video")
    _positive_int(runtime.get("batch_size"), f"SST clip {clip_id} runtime batch_size")
    if runtime.get("device") not in {"cpu", "cuda"}:
        raise ValueError(f"SST clip {clip_id} WASB runtime device invalid")
    if runtime.get("input_preprocessing") != "official" or runtime.get("non_promotable_measurement_mode") is not False:
        raise ValueError(f"SST clip {clip_id} WASB runtime is not promotable official preprocessing")
    wall_seconds = _finite_float(runtime.get("wall_seconds"), f"SST clip {clip_id} wall_seconds")
    effective_fps = _finite_float(runtime.get("effective_fps"), f"SST clip {clip_id} effective_fps")
    realtime_factor = _finite_float(runtime.get("realtime_factor"), f"SST clip {clip_id} realtime_factor")
    if wall_seconds <= 0.0 or effective_fps <= 0.0 or realtime_factor <= 0.0:
        raise ValueError(f"SST clip {clip_id} WASB runtime rates must be positive")
    expected_effective_fps = frame_count / wall_seconds
    if not math.isclose(effective_fps, expected_effective_fps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"SST clip {clip_id} WASB runtime effective_fps mismatch")
    if not math.isclose(realtime_factor, expected_effective_fps / fps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"SST clip {clip_id} WASB runtime realtime_factor mismatch")
    return runtime


def _run_official_wasb_replay(
    *,
    source_video: Path,
    checkpoint: Path,
    wasb_repo: Path,
    frame_times_path: Path,
    fps: float,
    batch_size: int,
    device: str,
) -> bytes:
    """Replay pinned official inference; production callers cannot bypass this check."""

    with tempfile.TemporaryDirectory(prefix="ball_sst_wasb_replay_") as temporary:
        root = Path(temporary)
        replay_csv = root / "wasb_predictions.csv"
        run_wasb_or_convert(
            out=root / "wasb_ball_track.json",
            fps=fps,
            frame_times=frame_times_path,
            metadata_out=root / "wasb_ball_track_metadata.json",
            video=source_video,
            checkpoint=checkpoint,
            wasb_repo=wasb_repo,
            prediction_csv_out=replay_csv,
            batch_size=batch_size,
            visible_threshold=SST_TEACHER_CONFIDENCE_MIN,
            device=device,
            input_preprocessing="official",
            emit_size_observations=False,
            emit_below_threshold_candidates=False,
        )
        return replay_csv.read_bytes()


def _require_recorded_path_identity(value: Any, expected: Path, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a canonical path string")
    recorded = Path(value)
    resolved = recorded.resolve(strict=True)
    _require_lexical_canonical_path(recorded, resolved=resolved, field=field)
    if resolved != expected.resolve(strict=True):
        raise ValueError(f"{field} path mismatch: expected={expected} actual={resolved}")


def _csv_finite_float(value: Any, field: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a numeric string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _validate_sst_frame_times(
    payload: Mapping[str, Any],
    *,
    clip_id: str,
    source_video_sha256: str,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    if payload.get("artifact_type") != "racketsport_frame_times" or payload.get("schema_version") != 1:
        raise ValueError(f"SST clip {clip_id} frame-times schema mismatch")
    if payload.get("source_video_sha256") != source_video_sha256:
        raise ValueError(f"SST clip {clip_id} frame-times media SHA mismatch")
    if (
        _positive_int(payload.get("width"), f"SST clip {clip_id} frame-times width"),
        _positive_int(payload.get("height"), f"SST clip {clip_id} frame-times height"),
    ) != (source_width, source_height):
        raise ValueError(f"SST clip {clip_id} frame-times dimensions mismatch")
    fps = _finite_float(payload.get("fps"), f"SST clip {clip_id} frame-times fps")
    if fps <= 0.0:
        raise ValueError(f"SST clip {clip_id} frame-times fps must be positive")
    duration_s = _finite_float(payload.get("duration_s"), f"SST clip {clip_id} frame-times duration_s")
    if duration_s <= 0.0:
        raise ValueError(f"SST clip {clip_id} frame-times duration must be positive")
    frames = payload.get("frames")
    if not isinstance(frames, list) or _positive_int(payload.get("frame_count"), f"SST clip {clip_id} frame_count") != len(frames):
        raise ValueError(f"SST clip {clip_id} frame-times count mismatch")
    pts_s: list[float] = []
    for frame_index, row_raw in enumerate(frames):
        row = _required_mapping(row_raw, f"SST clip {clip_id} frame-times row {frame_index}")
        if int(row.get("frame", -1)) != frame_index:
            raise ValueError(f"SST clip {clip_id} frame-times index mismatch")
        pts_s.append(_finite_float(row.get("pts_s"), f"SST clip {clip_id} PTS {frame_index}"))
    if pts_s[0] < 0.0 or any(right <= left for left, right in zip(pts_s, pts_s[1:])):
        raise ValueError(f"SST clip {clip_id} PTS must be nonnegative and strictly increasing")
    return {
        "fps": fps,
        "duration_s": duration_s,
        "width": source_width,
        "height": source_height,
        "frame_count": len(pts_s),
        "pts_s": pts_s,
    }


def _probe_canonical_media_timing(source_video: Path, *, clip_id: str) -> dict[str, Any]:
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
        str(source_video),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe is required to verify canonical frame PTS for SST clip {clip_id}") from exc
    if completed.returncode != 0:
        raise ValueError(f"ffprobe failed for SST clip {clip_id}: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
        duration_s = float(stream["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        pts_s = [float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]]
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"ffprobe timing metadata is malformed for SST clip {clip_id}") from exc
    if (
        not math.isfinite(fps)
        or not math.isfinite(duration_s)
        or fps <= 0.0
        or duration_s <= 0.0
        or width <= 0
        or height <= 0
        or not pts_s
        or any(not math.isfinite(value) for value in pts_s)
        or pts_s[0] < 0.0
        or any(right <= left for left, right in zip(pts_s, pts_s[1:]))
    ):
        raise ValueError(f"ffprobe timing metadata is invalid for SST clip {clip_id}")
    return {
        "fps": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "frame_count": len(pts_s),
        "pts_s": pts_s,
    }


def _validate_sst_frame_times_against_media(
    frame_times: Mapping[str, Any],
    *,
    encoded_timing: Mapping[str, Any],
    clip_id: str,
) -> None:
    scalar_fields = ("fps", "duration_s", "width", "height", "frame_count")
    for field in scalar_fields:
        if frame_times.get(field) != encoded_timing.get(field):
            raise ValueError(
                f"SST clip {clip_id} frame-times {field} differs from canonical encoded media"
            )
    if frame_times.get("pts_s") != encoded_timing.get("pts_s"):
        raise ValueError(f"SST clip {clip_id} frame PTS differ from canonical encoded media")


def _extract_bound_teacher_observations(
    payload: Mapping[str, Any],
    *,
    metadata_payload: Mapping[str, Any],
    pts_s: Sequence[float],
    width: int,
    height: int,
    clip_id: str,
) -> dict[int, dict[str, Any]]:
    camera = payload.get("camera")
    metadata = _required_mapping(metadata_payload.get("metadata"), f"SST clip {clip_id} teacher metadata")
    fps_value = camera.get("fps") if isinstance(camera, Mapping) else None
    if fps_value is None:
        fps_value = metadata.get("fps")
    teacher_fps = _finite_float(fps_value, f"SST clip {clip_id} teacher fps")
    if teacher_fps <= 0.0:
        raise ValueError(f"SST clip {clip_id} teacher fps must be positive")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError(f"SST clip {clip_id} cv_export sessions must be a list")
    by_source_frame: dict[int, dict[str, Any]] = {}
    for session_index, session_raw in enumerate(sessions):
        session = _required_mapping(session_raw, f"SST clip {clip_id} session {session_index}")
        rallies = session.get("rallies")
        if not isinstance(rallies, list):
            raise ValueError(f"SST clip {clip_id} session rallies must be a list")
        for rally_index, rally_raw in enumerate(rallies):
            rally = _required_mapping(rally_raw, f"SST clip {clip_id} rally {rally_index}")
            teacher_start = _nonnegative_int(
                rally.get("frame_index"),
                f"SST clip {clip_id} rally teacher frame_index",
            )
            frames = rally.get("frames")
            if not isinstance(frames, list):
                raise ValueError(f"SST clip {clip_id} rally frames must be a list")
            for offset, frame_raw in enumerate(frames):
                if not isinstance(frame_raw, Mapping):
                    continue
                actions = frame_raw.get("actions")
                ball = actions.get("ball") if isinstance(actions, Mapping) else None
                if not isinstance(ball, Mapping):
                    continue
                teacher_frame_index = teacher_start + offset
                confidence = _finite_float(ball.get("confidence"), f"SST clip {clip_id} teacher confidence")
                u = _finite_float(ball.get("u"), f"SST clip {clip_id} teacher u")
                v = _finite_float(ball.get("v"), f"SST clip {clip_id} teacher v")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"SST clip {clip_id} teacher confidence must be in [0, 1]")
                if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
                    raise ValueError(f"SST clip {clip_id} teacher u/v must be in [0, 1]")
                xy = (
                    u * width,
                    v * height,
                )
                teacher_time_s = teacher_frame_index / teacher_fps
                insertion = bisect.bisect_left(pts_s, teacher_time_s)
                if insertion <= 0:
                    source_frame_index = 0
                elif insertion >= len(pts_s):
                    source_frame_index = len(pts_s) - 1
                else:
                    before = insertion - 1
                    source_frame_index = (
                        before
                        if abs(float(pts_s[before]) - teacher_time_s)
                        <= abs(float(pts_s[insertion]) - teacher_time_s)
                        else insertion
                    )
                candidate = {
                    "teacher_frame_index": teacher_frame_index,
                    "teacher_time_s": teacher_time_s,
                    "xy": xy,
                    "confidence": confidence,
                }
                previous = by_source_frame.get(source_frame_index)
                if previous is None or confidence > float(previous["confidence"]):
                    by_source_frame[source_frame_index] = candidate
    return by_source_frame


def _validate_sst_sample(
    sample: Mapping[str, Any],
    *,
    clip_id: str,
    builder_sha: str,
    expected_wasb: Mapping[str, str],
    top_dependencies: Mapping[str, Any],
    clip_dependencies: Mapping[str, Any],
    media_sha_cache: dict[Path, str],
    resolved_clip_media: Path,
    source_width: int,
    source_height: int,
    frame_times: Mapping[str, Any],
    wasb_frames: Sequence[Mapping[str, Any]],
    teacher_by_source_frame: Mapping[int, Mapping[str, Any]],
) -> None:
    canonical_id = str(sample.get("canonical_source_id") or "")
    if str(sample.get("clip_id") or "") != clip_id or canonical_id != clip_id:
        raise ValueError(f"SST row has conflicting source aliases: {sample.get('sample_id')}")
    if sample.get("teacher_derived") is not True or sample.get("ground_truth") is not False:
        raise ValueError(f"SST row authority mismatch: {sample.get('sample_id')}")
    if sample.get("ball_present") is not True:
        raise ValueError(f"SST row must be an explicit positive: {sample.get('sample_id')}")
    if sample.get("teacher_source") != "pbvision_actions_ball":
        raise ValueError(f"SST row teacher source mismatch: {sample.get('sample_id')}")
    frame_index = _nonnegative_int(sample.get("frame_index"), f"SST {sample.get('sample_id')} frame_index")
    pts_s = frame_times.get("pts_s")
    if not isinstance(pts_s, list) or frame_index >= len(pts_s):
        raise ValueError(f"SST row frame is outside the bound PTS artifact: {sample.get('sample_id')}")
    if sample.get("sample_id") != f"{clip_id}:{frame_index}":
        raise ValueError(f"SST sample_id mismatch: {sample.get('sample_id')}")
    if not _exact_number(sample.get("weight"), SST_PSEUDO_WEIGHT):
        raise ValueError(f"SST row weight is not frozen at {SST_PSEUDO_WEIGHT}: {sample.get('sample_id')}")
    score = _finite_float(sample.get("score"), f"SST {sample.get('sample_id')} score")
    if not SST_TEACHER_CONFIDENCE_MIN <= score <= 1.0:
        raise ValueError(f"SST row teacher confidence is below frozen threshold: {sample.get('sample_id')}")
    teacher_xy = _validated_xy(sample.get("teacher_xy"), f"SST {sample.get('sample_id')} teacher_xy")
    bound_teacher = teacher_by_source_frame.get(frame_index)
    if bound_teacher is None:
        raise ValueError(f"SST row has no teacher observation in hashed cv_export: {sample.get('sample_id')}")
    if (
        int(sample.get("teacher_frame_index", -1)) != int(bound_teacher["teacher_frame_index"])
        or not _xy_close(teacher_xy, bound_teacher["xy"])
        or not math.isclose(score, float(bound_teacher["confidence"]), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"SST row teacher evidence does not match hashed cv_export: {sample.get('sample_id')}")
    frame_ref = _required_mapping(sample.get("frame_ref"), f"SST {sample.get('sample_id')} frame_ref")
    if int(frame_ref.get("frame_index", -1)) != frame_index:
        raise ValueError(f"SST frame_ref index mismatch: {sample.get('sample_id')}")
    resolved_video = _required_canonical_file(
        frame_ref.get("video"),
        f"SST {sample.get('sample_id')} frame_ref.video",
        expected=resolved_clip_media,
    )
    expected_pts = float(pts_s[frame_index])
    for field, value in (("sample.t", sample.get("t")), ("frame_ref.t", frame_ref.get("t"))):
        if not math.isclose(
            _finite_float(value, f"SST {sample.get('sample_id')} {field}"),
            expected_pts,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"SST row time does not match bound PTS: {sample.get('sample_id')}")

    row_dependencies = _required_mapping(sample.get("dependency_hashes"), f"SST {sample.get('sample_id')} dependency_hashes")
    missing = SST_REQUIRED_ROW_DEPENDENCY_HASHES - set(row_dependencies)
    if missing:
        raise ValueError(f"SST row missing dependency hashes {sorted(missing)}: {sample.get('sample_id')}")
    for key in SST_REQUIRED_ROW_DEPENDENCY_HASHES - {"wasb_repo_commit"}:
        _require_sha256(row_dependencies.get(key), f"SST {sample.get('sample_id')} {key}")
    for key in SST_REQUIRED_ROW_DEPENDENCY_HASHES:
        if row_dependencies.get(key) != clip_dependencies.get(key):
            raise ValueError(f"SST row/clip dependency mismatch for {key}: {sample.get('sample_id')}")
    if row_dependencies.get("builder_code_sha256") != builder_sha:
        raise ValueError(f"SST row builder code SHA mismatch: {sample.get('sample_id')}")
    if row_dependencies.get("wasb_checkpoint_sha256") != expected_wasb["checkpoint_sha256"]:
        raise ValueError(f"SST row WASB checkpoint SHA mismatch: {sample.get('sample_id')}")
    if row_dependencies.get("wasb_repo_commit") != expected_wasb["repo_commit"]:
        raise ValueError(f"SST row WASB repo commit mismatch: {sample.get('sample_id')}")
    if row_dependencies.get("split_manifest_sha256") != top_dependencies.get("split_manifest_sha256"):
        raise ValueError(f"SST row split manifest SHA mismatch: {sample.get('sample_id')}")
    if resolved_video not in media_sha_cache:
        media_sha_cache[resolved_video] = _sha256_file(resolved_video)
    actual_media_sha = media_sha_cache[resolved_video]
    expected_media_sha = SST_EXPECTED_SOURCE_VIDEO_SHA256[clip_id]
    if actual_media_sha != expected_media_sha:
        raise ValueError(f"SST source media bytes are not the preregistered video: {sample.get('sample_id')}")
    if sample.get("source_video_sha256") != expected_media_sha:
        raise ValueError(f"SST row source-video SHA field mismatch: {sample.get('sample_id')}")
    if row_dependencies.get("source_video_sha256") != actual_media_sha:
        raise ValueError(f"SST row source media SHA mismatch: {sample.get('sample_id')}")
    if frame_ref.get("source_video_sha256") != actual_media_sha:
        raise ValueError(f"SST frame_ref source media SHA mismatch: {sample.get('sample_id')}")

    reason = sample.get("agreement_reason")
    evidence = _required_mapping(sample.get("agreement"), f"SST {sample.get('sample_id')} agreement")
    width = _positive_int(evidence.get("image_width"), f"SST {sample.get('sample_id')} image_width")
    height = _positive_int(evidence.get("image_height"), f"SST {sample.get('sample_id')} image_height")
    if (width, height) != (source_width, source_height):
        raise ValueError(f"SST agreement dimensions do not match decoded media: {sample.get('sample_id')}")
    if evidence.get("all_points_in_bounds") is not True or not _inside_xy(teacher_xy, width=width, height=height):
        raise ValueError(f"SST agreement evidence is out of image bounds: {sample.get('sample_id')}")
    if reason == SST_SPATIAL_REASON:
        _validate_sst_spatial_evidence(
            sample,
            evidence,
            teacher_xy=teacher_xy,
            frame_index=frame_index,
            width=width,
            height=height,
            wasb_frames=wasb_frames,
        )
    elif reason == SST_TEMPORAL_REASON:
        _validate_sst_temporal_evidence(
            sample,
            evidence,
            teacher_xy=teacher_xy,
            frame_index=frame_index,
            width=width,
            height=height,
            wasb_frames=wasb_frames,
            teacher_by_source_frame=teacher_by_source_frame,
        )
    else:
        raise ValueError(f"SST row has non-independent agreement reason {reason!r}: {sample.get('sample_id')}")


def _validate_sst_spatial_evidence(
    sample: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    teacher_xy: tuple[float, float],
    frame_index: int,
    width: int,
    height: int,
    wasb_frames: Sequence[Mapping[str, Any]],
) -> None:
    if evidence.get("policy_id") != "frozen_wasb_spatial_v2":
        raise ValueError(f"SST spatial policy id mismatch: {sample.get('sample_id')}")
    if int(evidence.get("source_frame_index", -1)) != frame_index:
        raise ValueError(f"SST spatial source frame mismatch: {sample.get('sample_id')}")
    evidence_teacher = _validated_xy(evidence.get("teacher_xy"), "SST spatial teacher_xy")
    wasb_xy = _validated_xy(evidence.get("wasb_xy"), "SST spatial wasb_xy")
    if evidence_teacher != teacher_xy or not _inside_xy(wasb_xy, width=width, height=height):
        raise ValueError(f"SST spatial points are not bound/in-bounds: {sample.get('sample_id')}")
    teacher_confidence = _finite_float(evidence.get("teacher_confidence"), "SST spatial teacher_confidence")
    if teacher_confidence != float(sample["score"]) or teacher_confidence < SST_TEACHER_CONFIDENCE_MIN:
        raise ValueError(f"SST spatial teacher confidence too low: {sample.get('sample_id')}")
    wasb_confidence = _finite_float(evidence.get("wasb_confidence"), "SST spatial wasb_confidence")
    if wasb_confidence < SST_TEACHER_CONFIDENCE_MIN:
        raise ValueError(f"SST spatial WASB confidence too low: {sample.get('sample_id')}")
    actual_wasb = _wasb_frame_observation(wasb_frames, frame_index, sample_id=str(sample.get("sample_id")))
    if (
        actual_wasb["visible"] is not True
        or not _xy_close(wasb_xy, actual_wasb["xy"])
        or not math.isclose(wasb_confidence, actual_wasb["confidence"], rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"SST spatial evidence does not match the hashed WASB track: {sample.get('sample_id')}")
    distance = math.dist(teacher_xy, wasb_xy)
    if not _exact_number(evidence.get("agreement_radius_px"), SST_AGREEMENT_RADIUS_PX):
        raise ValueError(f"SST spatial radius is not frozen: {sample.get('sample_id')}")
    if distance > SST_AGREEMENT_RADIUS_PX or not math.isclose(
        _finite_float(evidence.get("distance_px"), "SST spatial distance_px"), distance, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"SST spatial distance evidence mismatch: {sample.get('sample_id')}")


def _validate_sst_temporal_evidence(
    sample: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    teacher_xy: tuple[float, float],
    frame_index: int,
    width: int,
    height: int,
    wasb_frames: Sequence[Mapping[str, Any]],
    teacher_by_source_frame: Mapping[int, Mapping[str, Any]],
) -> None:
    if evidence.get("policy_id") != SST_TEMPORAL_REASON or evidence.get("independent_verifier") != "pinned_frozen_wasb":
        raise ValueError(f"SST temporal evidence is not pinned independent WASB: {sample.get('sample_id')}")
    if int(evidence.get("current_source_frame_index", -1)) != frame_index:
        raise ValueError(f"SST temporal current frame mismatch: {sample.get('sample_id')}")
    current_teacher = _validated_xy(evidence.get("current_teacher_xy"), "SST temporal current_teacher_xy")
    current_confidence = _finite_float(evidence.get("current_teacher_confidence"), "SST temporal current confidence")
    if current_teacher != teacher_xy or current_confidence != float(sample["score"]):
        raise ValueError(f"SST temporal current teacher mismatch: {sample.get('sample_id')}")
    if int(evidence.get("max_gap_source_frames", -1)) != SST_TEMPORAL_MAX_GAP_SOURCE_FRAMES:
        raise ValueError(f"SST temporal gap policy is not frozen: {sample.get('sample_id')}")
    if not _exact_number(evidence.get("anchor_agreement_radius_px"), SST_AGREEMENT_RADIUS_PX):
        raise ValueError(f"SST temporal anchor radius is not frozen: {sample.get('sample_id')}")
    if not _exact_number(evidence.get("interpolation_residual_max_px"), SST_AGREEMENT_RADIUS_PX):
        raise ValueError(f"SST temporal residual cap is not frozen: {sample.get('sample_id')}")
    _validate_sst_current_wasb_gap(
        sample,
        evidence.get("current_wasb"),
        frame_index=frame_index,
        width=width,
        height=height,
        wasb_frames=wasb_frames,
    )
    anchors: list[tuple[int, tuple[float, float]]] = []
    for name in ("prior_anchor", "following_anchor"):
        anchor = _required_mapping(evidence.get(name), f"SST temporal {name}")
        anchor_frame = _nonnegative_int(anchor.get("source_frame_index"), f"SST temporal {name} frame")
        anchor_teacher = _validated_xy(anchor.get("teacher_xy"), f"SST temporal {name} teacher_xy")
        anchor_wasb = _validated_xy(anchor.get("wasb_xy"), f"SST temporal {name} wasb_xy")
        if not _inside_xy(anchor_teacher, width=width, height=height) or not _inside_xy(anchor_wasb, width=width, height=height):
            raise ValueError(f"SST temporal anchor is out of bounds: {sample.get('sample_id')}")
        if _finite_float(anchor.get("teacher_confidence"), f"SST temporal {name} teacher confidence") < SST_TEACHER_CONFIDENCE_MIN:
            raise ValueError(f"SST temporal teacher anchor confidence too low: {sample.get('sample_id')}")
        actual_teacher = teacher_by_source_frame.get(anchor_frame)
        if actual_teacher is None or (
            not _xy_close(anchor_teacher, actual_teacher["xy"])
            or not math.isclose(
                _finite_float(anchor.get("teacher_confidence"), f"SST temporal {name} teacher confidence"),
                float(actual_teacher["confidence"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"SST temporal teacher anchor does not match hashed cv_export: {sample.get('sample_id')}")
        if _finite_float(anchor.get("wasb_confidence"), f"SST temporal {name} WASB confidence") < SST_TEACHER_CONFIDENCE_MIN:
            raise ValueError(f"SST temporal WASB anchor confidence too low: {sample.get('sample_id')}")
        actual_anchor = _wasb_frame_observation(
            wasb_frames,
            anchor_frame,
            sample_id=str(sample.get("sample_id")),
        )
        if (
            actual_anchor["visible"] is not True
            or not _xy_close(anchor_wasb, actual_anchor["xy"])
            or not math.isclose(
                _finite_float(anchor.get("wasb_confidence"), f"SST temporal {name} WASB confidence"),
                actual_anchor["confidence"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"SST temporal anchor does not match hashed WASB track: {sample.get('sample_id')}")
        distance = math.dist(anchor_teacher, anchor_wasb)
        if distance > SST_AGREEMENT_RADIUS_PX or not math.isclose(
            _finite_float(anchor.get("distance_px"), f"SST temporal {name} distance"), distance, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"SST temporal anchor lacks independent WASB agreement: {sample.get('sample_id')}")
        anchors.append((anchor_frame, anchor_wasb))
    prior_frame, prior_xy = anchors[0]
    following_frame, following_xy = anchors[1]
    if not prior_frame < frame_index < following_frame:
        raise ValueError(f"SST temporal anchors do not bracket current frame: {sample.get('sample_id')}")
    teacher_only_gap_length = following_frame - prior_frame - 1
    if teacher_only_gap_length > SST_TEMPORAL_MAX_GAP_SOURCE_FRAMES:
        raise ValueError(f"SST temporal bridge exceeds frozen total teacher-only gap: {sample.get('sample_id')}")
    if evidence.get("gap_length_semantics") != "total_consecutive_teacher_only_interior_frames":
        raise ValueError(f"SST temporal gap semantics mismatch: {sample.get('sample_id')}")
    if int(evidence.get("teacher_only_gap_length_source_frames", -1)) != teacher_only_gap_length:
        raise ValueError(f"SST temporal total gap length mismatch: {sample.get('sample_id')}")
    intermediate = evidence.get("intermediate_frames")
    if not isinstance(intermediate, list) or len(intermediate) != teacher_only_gap_length:
        raise ValueError(f"SST temporal intermediate-frame evidence mismatch: {sample.get('sample_id')}")
    for expected_frame, item_raw in zip(range(prior_frame + 1, following_frame), intermediate):
        item = _required_mapping(item_raw, f"SST temporal intermediate frame {expected_frame}")
        if int(item.get("source_frame_index", -1)) != expected_frame:
            raise ValueError(f"SST temporal intermediate-frame sequence mismatch: {sample.get('sample_id')}")
        intermediate_teacher = _validated_xy(
            item.get("teacher_xy"), f"SST temporal intermediate frame {expected_frame} teacher_xy"
        )
        intermediate_confidence = _finite_float(
            item.get("teacher_confidence"),
            f"SST temporal intermediate frame {expected_frame} teacher confidence",
        )
        actual_teacher = teacher_by_source_frame.get(expected_frame)
        if (
            actual_teacher is None
            or intermediate_confidence < SST_TEACHER_CONFIDENCE_MIN
            or not _inside_xy(intermediate_teacher, width=width, height=height)
            or not _xy_close(intermediate_teacher, actual_teacher["xy"])
            or not math.isclose(
                intermediate_confidence,
                float(actual_teacher["confidence"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"SST temporal intermediate teacher does not match hashed teacher path: {sample.get('sample_id')}"
            )
        actual_wasb = _wasb_frame_observation(
            wasb_frames,
            expected_frame,
            sample_id=str(sample.get("sample_id")),
        )
        if actual_wasb["visible"] is True and actual_wasb["confidence"] >= SST_TEACHER_CONFIDENCE_MIN:
            if math.dist(intermediate_teacher, actual_wasb["xy"]) > SST_AGREEMENT_RADIUS_PX:
                raise ValueError(
                    "SST temporal bridge crosses contradictory high-confidence WASB evidence: "
                    f"{sample.get('sample_id')}:{expected_frame}"
                )
            raise ValueError(
                f"SST temporal intermediate frame is not teacher-only: {sample.get('sample_id')}:{expected_frame}"
            )
        _validate_sst_current_wasb_gap(
            sample,
            item.get("wasb"),
            frame_index=expected_frame,
            width=width,
            height=height,
            wasb_frames=wasb_frames,
        )
        if expected_frame == frame_index:
            if not _xy_close(intermediate_teacher, current_teacher) or not math.isclose(
                intermediate_confidence, current_confidence, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"SST temporal current intermediate teacher mismatch: {sample.get('sample_id')}")
            if item.get("wasb") != evidence.get("current_wasb"):
                raise ValueError(f"SST temporal current intermediate WASB mismatch: {sample.get('sample_id')}")
    fraction = (frame_index - prior_frame) / float(following_frame - prior_frame)
    interpolated = (
        prior_xy[0] + (following_xy[0] - prior_xy[0]) * fraction,
        prior_xy[1] + (following_xy[1] - prior_xy[1]) * fraction,
    )
    evidence_interpolated = _validated_xy(evidence.get("interpolated_wasb_xy"), "SST temporal interpolated_wasb_xy")
    if not _inside_xy(evidence_interpolated, width=width, height=height):
        raise ValueError(f"SST temporal interpolation is out of bounds: {sample.get('sample_id')}")
    if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(interpolated, evidence_interpolated)):
        raise ValueError(f"SST temporal interpolation mismatch: {sample.get('sample_id')}")
    residual = math.dist(teacher_xy, interpolated)
    if residual > SST_AGREEMENT_RADIUS_PX or not math.isclose(
        _finite_float(evidence.get("interpolation_residual_px"), "SST temporal interpolation residual"),
        residual,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"SST temporal bridge residual exceeds frozen radius: {sample.get('sample_id')}")


def _validate_sst_current_wasb_gap(
    sample: Mapping[str, Any],
    evidence_raw: Any,
    *,
    frame_index: int,
    width: int,
    height: int,
    wasb_frames: Sequence[Mapping[str, Any]],
) -> None:
    evidence = _required_mapping(evidence_raw, f"SST temporal current_wasb {sample.get('sample_id')}")
    actual = _wasb_frame_observation(wasb_frames, frame_index, sample_id=str(sample.get("sample_id")))
    if actual["visible"] is True and actual["confidence"] >= SST_TEACHER_CONFIDENCE_MIN:
        raise ValueError(f"SST temporal row is not a teacher-only WASB gap: {sample.get('sample_id')}")
    expected_status = "not_visible" if actual["visible"] is False else "below_confidence_threshold"
    if evidence.get("status") != expected_status or evidence.get("present") is not True:
        raise ValueError(f"SST temporal current-WASB status contradicts hashed track: {sample.get('sample_id')}")
    if int(evidence.get("frame_index", -1)) != frame_index:
        raise ValueError(f"SST temporal current-WASB frame mismatch: {sample.get('sample_id')}")
    evidence_xy = _validated_xy(evidence.get("xy"), "SST temporal current-WASB xy")
    if not _inside_xy(evidence_xy, width=width, height=height) or not _xy_close(evidence_xy, actual["xy"]):
        raise ValueError(f"SST temporal current-WASB coordinate mismatch: {sample.get('sample_id')}")
    evidence_confidence = _finite_float(evidence.get("confidence"), "SST temporal current-WASB confidence")
    if not math.isclose(evidence_confidence, actual["confidence"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"SST temporal current-WASB confidence mismatch: {sample.get('sample_id')}")
    if evidence.get("visible") is not actual["visible"]:
        raise ValueError(f"SST temporal current-WASB visibility mismatch: {sample.get('sample_id')}")


def _wasb_frame_observation(
    frames: Sequence[Mapping[str, Any]],
    frame_index: int,
    *,
    sample_id: str,
) -> dict[str, Any]:
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"SST WASB evidence frame is outside hashed track: {sample_id}:{frame_index}")
    frame = frames[frame_index]
    return {
        "xy": _validated_xy(frame.get("xy"), f"SST WASB frame {frame_index} xy"),
        "confidence": _finite_float(
            frame.get("conf") if frame.get("conf") is not None else 0.0,
            f"SST WASB frame {frame_index} conf",
        ),
        "visible": frame.get("visible"),
    }


def _xy_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12) for a, b in zip(left, right))


def load_trainer_module_from_git_revision(
    revision: str,
    *,
    expected_sha256: str,
) -> tuple[Any, dict[str, str]]:
    """Load one explicitly pinned trainer git blob into an isolated module."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("parity baseline revision must be passed explicitly")
    expected_source_sha256 = _require_sha256(
        expected_sha256,
        "parity baseline trainer sha256",
    )
    commit = _git_commit(revision)
    relative_path = "scripts/racketsport/train_ball_stage2.py"
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    source_bytes = completed.stdout
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "parity baseline git blob SHA-256 mismatch: "
            f"revision={revision!r} resolved_commit={commit} "
            f"expected={expected_source_sha256} actual={source_sha256}"
        )
    module_name = f"_ball_stage2_pinned_{commit[:12]}_{source_sha256[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(ROOT / relative_path)
    module.__package__ = "scripts.racketsport"
    sys.modules[module_name] = module
    exec(compile(source_bytes, module.__file__, "exec"), module.__dict__)
    return module, {
        "requested_revision": revision,
        "commit": commit,
        "source_sha256": source_sha256,
        "expected_source_sha256": expected_source_sha256,
        "source_path": relative_path,
        "load_method": "git show <resolved-commit>:scripts/racketsport/train_ball_stage2.py then isolated exec",
    }


def _resolve_distinct_parity_trainers(
    *,
    baseline_revision: str | None,
    baseline_sha256: str | None,
) -> tuple[Any, dict[str, str], dict[str, str]]:
    if baseline_revision is None:
        raise ValueError("parity requires explicit --baseline-rev; no HEAD default is allowed")
    if baseline_sha256 is None:
        raise ValueError("parity requires explicit --baseline-sha256; no inferred hash is allowed")
    candidate_path = Path(__file__).resolve(strict=True)
    candidate_identity = {
        "path": str(candidate_path),
        "source_sha256": _sha256_file(candidate_path),
        "load_method": "running trainer file",
    }
    baseline_module, baseline_identity = load_trainer_module_from_git_revision(
        baseline_revision,
        expected_sha256=baseline_sha256,
    )
    if baseline_identity["source_sha256"] == candidate_identity["source_sha256"]:
        raise ValueError(
            "parity baseline and candidate trainer SHA-256 must be distinct; "
            "candidate-versus-itself comparisons are invalid"
        )
    return baseline_module, baseline_identity, candidate_identity


def run_revision_explicit_compute_parity(
    batches: Any,
    *,
    baseline_revision: str,
    baseline_sha256: str,
    model_factory: Any,
    optimizer_factory: Any,
    steps: int,
    device: Any,
    torch: Any,
    comparison_config: Mapping[str, Any] | None = None,
    production: bool = False,
) -> dict[str, Any]:
    """Compare a pinned git baseline and running no-SST trainer on shared batches."""

    if production:
        raise ValueError(
            "generic compute parity cannot claim production execution; "
            "use run_revision_explicit_production_parity"
        )
    if steps <= 0:
        raise ValueError("parity steps must be positive")
    baseline_module, baseline_identity, candidate_identity = _resolve_distinct_parity_trainers(
        baseline_revision=baseline_revision,
        baseline_sha256=baseline_sha256,
    )
    base_model = model_factory()
    baseline_model = copy.deepcopy(base_model).to(device)
    candidate_model = copy.deepcopy(base_model).to(device)
    baseline_model.train()
    candidate_model.train()
    baseline_optimizer = optimizer_factory(baseline_model.parameters())
    candidate_optimizer = optimizer_factory(candidate_model.parameters())
    baseline_losses: list[float] = []
    candidate_losses: list[float] = []
    baseline_sample_order: list[list[str]] = []
    candidate_sample_order: list[list[str]] = []
    baseline_batch_iterator = iter(batches)
    candidate_batch_iterator = iter(batches)
    for step_index in range(steps):
        baseline_batch = next(baseline_batch_iterator)
        candidate_batch = next(candidate_batch_iterator)
        baseline_ids = [str(value) for value in baseline_batch.get("sample_id", [])]
        candidate_ids = [str(value) for value in candidate_batch.get("sample_id", [])]
        if len(baseline_ids) != B2_HUMAN_BATCH_SIZE or len(candidate_ids) != B2_HUMAN_BATCH_SIZE:
            raise ValueError(
                f"pinned parity requires {B2_HUMAN_BATCH_SIZE} human rows per arm at step {step_index}, "
                f"got baseline={len(baseline_ids)} candidate={len(candidate_ids)}"
            )
        if baseline_ids != candidate_ids:
            raise RuntimeError(f"baseline/candidate sample order diverged at parity step {step_index}")
        baseline_sample_order.append(baseline_ids)
        candidate_sample_order.append(candidate_ids)
        before_rng = _capture_torch_rng(torch)
        baseline_loss = baseline_module.train_one_stage2_batch(
            baseline_model,
            baseline_batch,
            optimizer=baseline_optimizer,
            device=device,
            torch=torch,
            occluded_prob=0.0,
            occlusion_generator=None,
        )
        after_baseline_rng = _capture_torch_rng(torch)
        _restore_torch_rng(before_rng, torch)
        candidate_loss = train_one_stage2_batch(
            candidate_model,
            candidate_batch,
            optimizer=candidate_optimizer,
            device=device,
            torch=torch,
            occluded_prob=0.0,
            occlusion_generator=None,
        )
        after_candidate_rng = _capture_torch_rng(torch)
        if not _rng_states_equal(after_baseline_rng, after_candidate_rng, torch=torch):
            raise RuntimeError(f"baseline/candidate RNG consumption diverged at parity step {step_index}")
        _restore_torch_rng(after_baseline_rng, torch)
        baseline_losses.append(float(baseline_loss))
        candidate_losses.append(float(candidate_loss))

    baseline_state_sha = state_dict_sha256(baseline_model.state_dict())
    candidate_state_sha = state_dict_sha256(candidate_model.state_dict())
    checkpoint_comparison = _materialize_parity_checkpoint_comparison(
        baseline_module=baseline_module,
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        baseline_optimizer=baseline_optimizer,
        candidate_optimizer=candidate_optimizer,
        step=steps,
        comparison_config=dict(comparison_config or {}),
        torch=torch,
    )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_head_compute_parity_fixture",
        "comparison_scope": "fixture_shared_batches_compute_parity",
        "production_configuration_executed": False,
        "comparison_config": dict(comparison_config or {}),
        "baseline_trainer": baseline_identity,
        "candidate_trainer": candidate_identity,
        "trainer_sources_distinct": True,
        "steps": steps,
        "exact_losses": {
            "baseline": baseline_losses,
            "candidate": candidate_losses,
        },
        "exact_sample_order": {
            "baseline": baseline_sample_order,
            "candidate": candidate_sample_order,
        },
        "sample_order_identical": baseline_sample_order == candidate_sample_order,
        "model_state_sha256": {
            "baseline": baseline_state_sha,
            "candidate": candidate_state_sha,
        },
        "losses_identical": baseline_losses == candidate_losses,
        "model_state_identical": baseline_state_sha == candidate_state_sha,
        "checkpoint_format_comparison": checkpoint_comparison,
        "verdict": (
            "PASS"
            if baseline_losses == candidate_losses
            and baseline_state_sha == candidate_state_sha
            and baseline_sample_order == candidate_sample_order
            else "PARITY_MISMATCH"
        ),
    }


def _materialize_parity_checkpoint_comparison(
    *,
    baseline_module: Any,
    baseline_model: Any,
    candidate_model: Any,
    baseline_optimizer: Any,
    candidate_optimizer: Any,
    step: int,
    comparison_config: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    baseline_args = baseline_module._build_parser().parse_args([])
    candidate_args = _build_parser().parse_args([])
    _apply_checkpoint_comparison_config(baseline_args, comparison_config)
    _apply_checkpoint_comparison_config(candidate_args, comparison_config)
    dataset_summary = {
        "artifact_type": "racketsport_ball_stage2_parity_dataset_binding",
        "comparison_config": dict(comparison_config),
    }
    with tempfile.TemporaryDirectory(prefix="ball_stage2_checkpoint_parity_") as temporary:
        root = Path(temporary)
        baseline_path = baseline_module.save_stage2_checkpoint(
            root / "baseline_checkpoint.pt",
            model=baseline_model,
            optimizer=baseline_optimizer,
            step=step,
            args=baseline_args,
            train_dataset_summary=dataset_summary,
        )
        candidate_path = save_stage2_checkpoint(
            root / "candidate_checkpoint.pt",
            model=candidate_model,
            optimizer=candidate_optimizer,
            step=step,
            args=candidate_args,
            train_dataset_summary=dataset_summary,
        )
        baseline_raw_sha = _sha256_file(baseline_path)
        candidate_raw_sha = _sha256_file(candidate_path)
        baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
        candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    baseline_args_fields = sorted(
        _required_mapping(baseline_payload.get("args"), "baseline parity checkpoint args")
    )
    candidate_args_fields = sorted(
        _required_mapping(candidate_payload.get("args"), "candidate parity checkpoint args")
    )
    baseline_loaded_state_sha = state_dict_sha256(baseline_payload["model_state_dict"])
    candidate_loaded_state_sha = state_dict_sha256(candidate_payload["model_state_dict"])
    return {
        "status": "actual_payloads_materialized_and_loaded",
        "full_checkpoint_bytes_compared": True,
        "full_checkpoint_bytes_expected_identical": False,
        "raw_checkpoint_sha256": {"baseline": baseline_raw_sha, "candidate": candidate_raw_sha},
        "raw_checkpoint_bytes_identical": baseline_raw_sha == candidate_raw_sha,
        "checkpoint_schema": {
            "baseline": {
                "schema_version": baseline_payload.get("schema_version"),
                "artifact_type": baseline_payload.get("artifact_type"),
                "step": baseline_payload.get("step"),
                "model_family": baseline_payload.get("model_family"),
                "frames_in": baseline_payload.get("frames_in"),
                "output_channels": baseline_payload.get("output_channels"),
                "image_size": baseline_payload.get("image_size"),
            },
            "candidate": {
                "schema_version": candidate_payload.get("schema_version"),
                "artifact_type": candidate_payload.get("artifact_type"),
                "step": candidate_payload.get("step"),
                "model_family": candidate_payload.get("model_family"),
                "frames_in": candidate_payload.get("frames_in"),
                "output_channels": candidate_payload.get("output_channels"),
                "image_size": candidate_payload.get("image_size"),
            },
        },
        "top_level_fields": {
            "baseline": sorted(baseline_payload),
            "candidate": sorted(candidate_payload),
        },
        "checkpoint_args": {
            "baseline": dict(
                _required_mapping(baseline_payload.get("args"), "baseline parity checkpoint args")
            ),
            "candidate": dict(
                _required_mapping(candidate_payload.get("args"), "candidate parity checkpoint args")
            ),
        },
        "baseline_args_fields": baseline_args_fields,
        "candidate_args_fields": candidate_args_fields,
        "added_args_fields": sorted(set(candidate_args_fields) - set(baseline_args_fields)),
        "removed_args_fields": sorted(set(baseline_args_fields) - set(candidate_args_fields)),
        "train_dataset_summary": {
            "baseline": dict(
                _required_mapping(
                    baseline_payload.get("train_dataset_summary"),
                    "baseline parity checkpoint train_dataset_summary",
                )
            ),
            "candidate": dict(
                _required_mapping(
                    candidate_payload.get("train_dataset_summary"),
                    "candidate parity checkpoint train_dataset_summary",
                )
            ),
        },
        "train_dataset_summary_structure": {
            "baseline": _payload_structure(baseline_payload.get("train_dataset_summary")),
            "candidate": _payload_structure(candidate_payload.get("train_dataset_summary")),
        },
        "loaded_model_state_sha256": {
            "baseline": baseline_loaded_state_sha,
            "candidate": candidate_loaded_state_sha,
        },
        "loaded_model_state_identical": baseline_loaded_state_sha == candidate_loaded_state_sha,
        "reason": "raw torch checkpoint bytes are recorded separately from semantic model-state parity",
    }


def _apply_checkpoint_comparison_config(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    for key, value in config.items():
        if not hasattr(args, key) or isinstance(value, Mapping):
            continue
        if key == "image_size" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            setattr(args, key, f"{int(value[0])}x{int(value[1])}")
            continue
        current = getattr(args, key)
        if isinstance(current, Path) and isinstance(value, str):
            setattr(args, key, Path(value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            setattr(args, key, value)


def _payload_structure(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _payload_structure(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return ["list", len(value)]
    if isinstance(value, tuple):
        return ["tuple", len(value)]
    return type(value).__name__


def run_revision_explicit_production_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Run the encoded 2,372-step WASB/CUDA parity against an explicit baseline."""

    if args.out_dir is None:
        raise ValueError("--mode verify-head-parity requires --out-dir")
    # Resolve and compare source identities before touching CUDA, datasets, or output.
    # The compute harness resolves them again immediately before execution so the
    # artifact always records identities observed in the same call that computed it.
    _resolve_distinct_parity_trainers(
        baseline_revision=args.baseline_rev,
        baseline_sha256=args.baseline_sha256,
    )
    if args.sst_manifest or args.cvat_export_root is not None or args.resume_checkpoint is not None:
        raise ValueError("production parity is a no-SST frozen-B0 comparison only")
    if args.max_cvat_samples is not None or args.max_sst_samples is not None:
        raise ValueError("production parity cannot truncate either data arm")
    actual_config = {
        "model_family": str(args.model_family),
        "image_size": list(_parse_image_size(args.image_size)),
        "frames_in": int(args.frames_in),
        "output_channels": int(args.output_channels),
        "steps": int(args.steps) if args.steps is not None else None,
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "heatmap_radius_px": float(args.heatmap_radius_px),
        "occluded_prob": float(args.occluded_prob),
        "seed": int(args.seed),
        "device": str(args.device),
        "b0_split_root": str(args.b0_split_root) if args.b0_split_root is not None else None,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "wasb_repo": str(args.wasb_repo),
        "rally_root": str(args.rally_root),
        "image_root_rewrite": list(args.image_root_rewrite or []),
        "num_workers": int(args.num_workers),
    }
    expected_config = dict(PINNED_HEAD_PRODUCTION_PARITY_CONFIG)
    expected_config["b0_split_root"] = str(DEFAULT_B0_SPLIT_ROOT)
    if actual_config != expected_config:
        raise ValueError(
            f"production parity must use the frozen production config: "
            f"expected={expected_config} actual={actual_config}"
        )
    torch = _torch()
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "production parity requires requested CUDA, but torch.cuda.is_available() is false"
        )
    _required_canonical_directory(
        args.b0_split_root,
        "production parity B0 split_root",
        expected=ROOT / DEFAULT_B0_SPLIT_ROOT,
    )
    _required_canonical_directory(
        args.rally_root,
        "production parity rally_root",
        expected=ROOT / DEFAULT_RALLY_ROOT,
    )
    if args.init_checkpoint is None:
        raise ValueError("production parity requires the official WASB --init-checkpoint")
    expected_wasb = _expected_wasb_identity()
    init_checkpoint = _required_canonical_file(
        args.init_checkpoint,
        "production parity init checkpoint",
        expected=ROOT / expected_wasb["checkpoint_path"],
    )
    init_checkpoint_sha = _sha256_file(init_checkpoint)
    if init_checkpoint_sha != expected_wasb["checkpoint_sha256"]:
        raise ValueError("production parity init checkpoint does not match models/MANIFEST.json")
    wasb_repo_identity = _required_clean_git_repo(
        args.wasb_repo,
        "production parity WASB repo",
        expected=ROOT / SST_WASB_REPO_ROOT,
        expected_commit=expected_wasb["repo_commit"],
    )
    seed_summary = _seed_training_process(int(args.seed), torch=torch)
    device = _device(args.device, torch=torch)
    dataset = B0BallStage2Dataset.from_split_root(
        args.b0_split_root,
        rally_root=args.rally_root,
        image_size=_parse_image_size(args.image_size),
        frames_in=int(args.frames_in),
        heatmap_radius_px=float(args.heatmap_radius_px),
        image_path_rewrites=args.image_root_rewrite,
    )
    combined = CombinedStage2Dataset([dataset])
    media_identity = []
    for media_path in sorted({record.video_path.resolve(strict=True) for record in dataset.records}):
        media_identity.append({"path": str(media_path), "sha256": _sha256_file(media_path)})
    loader, _ = _make_training_loader(
        combined,
        batch_size=B2_HUMAN_BATCH_SIZE,
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        torch=torch,
        require_full_batches=True,
    )

    def model_factory() -> Any:
        model = build_model(
            model_family=str(args.model_family),
            frames_in=int(args.frames_in),
            output_channels=int(args.output_channels),
            image_size=_parse_image_size(args.image_size),
            wasb_repo=Path(args.wasb_repo),
        ).to(device)
        load_required_init_checkpoint(
            Path(args.init_checkpoint),
            model=model,
            device=device,
            frames_in=int(args.frames_in),
        )
        return model

    def optimizer_factory(parameters: Any) -> Any:
        return torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    summary = run_revision_explicit_compute_parity(
        loader,
        baseline_revision=args.baseline_rev,
        baseline_sha256=args.baseline_sha256,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        steps=int(args.steps),
        device=device,
        torch=torch,
        comparison_config={
            **expected_config,
            "production_artifact_bindings": {
                "b0_split_root": str((ROOT / DEFAULT_B0_SPLIT_ROOT).resolve(strict=True)),
                "b0_artifact_sha256": {
                    "report": B0_REPORT_SHA256,
                    "train": B0_TRAIN_SHA256,
                    "validation": B0_VALIDATION_SHA256,
                },
                "rally_root": str((ROOT / DEFAULT_RALLY_ROOT).resolve(strict=True)),
                "source_video_identity": media_identity,
                "init_checkpoint": {"path": str(init_checkpoint), "sha256": init_checkpoint_sha},
                "wasb_repo": wasb_repo_identity,
                "deterministic_seed": seed_summary,
                "sst_manifests": [],
            },
        },
        production=False,
    )
    summary.update(
        {
            "artifact_type": "racketsport_ball_stage2_pinned_head_parity",
            "comparison_scope": "production_shared_batches_compute_parity",
            "production_configuration_executed": True,
        }
    )
    summary["ball_verified"] = False
    summary["promotion_claimed"] = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "head_parity.json", summary)
    summary["out_dir"] = str(out_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "build-sst-manifest":
            summary = run_build_sst_manifest(args)
        elif args.mode == "verify-head-parity":
            summary = run_revision_explicit_production_parity(args)
        else:
            summary = run(args)
    except ModuleNotFoundError as exc:
        print(f"torch-gated ball stage2 skipped: missing module {exc.name}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ball stage2 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_cli_summary(summary), indent=2, sort_keys=True))
    if summary.get("artifact_type") == "racketsport_ball_stage2_pinned_head_parity" and summary.get("verdict") != "PASS":
        return 4
    return 0


def run_build_sst_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.sst_manifest_out is None:
        raise ValueError("--mode build-sst-manifest requires --sst-manifest-out")
    protected_eval_hashes = _parse_protected_eval_hashes(args.protected_eval_hash)
    return build_sst_manifest(
        prelabel_root=args.prelabel_root,
        rally_root=args.rally_root,
        out_path=args.sst_manifest_out,
        clips=args.clip,
        max_samples_per_clip=args.max_sst_samples_per_clip,
        protected_eval_hashes=protected_eval_hashes,
        expected_protected_eval_hash_count=args.expected_protected_eval_hash_count,
        eval_root=args.eval_root,
        eval_sample_every_s=args.eval_sample_every_s,
        collision_hamming_threshold=args.collision_hamming_threshold,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir is None:
        raise ValueError("--out-dir is required for training")
    if int(args.epochs) > 30:
        raise ValueError("--epochs must be <= 30 for bounded stage-2 training")
    if args.b0_split_root is not None and args.cvat_export_root is not None:
        raise ValueError("choose exactly one human source: --b0-split-root or --cvat-export-root")
    if args.resume_checkpoint is not None and args.init_checkpoint is not None:
        raise ValueError("--resume-checkpoint and --init-checkpoint are mutually exclusive")
    if args.sst_manifest and args.b0_split_root is None:
        raise ValueError("--sst-manifest requires the frozen --b0-split-root; SST-only or mutable-CVAT training is refused")
    if not args.sst_manifest and args.b0_split_root is None and args.cvat_export_root is None:
        raise ValueError("training requires human rows from --b0-split-root or --cvat-export-root")
    image_path_rewrites = _parse_image_path_rewrites(args.image_root_rewrite)
    if image_path_rewrites and (args.b0_split_root is not None or bool(args.sst_manifest)):
        raise ValueError("frozen B0/SST production training refuses --image-root-rewrite media substitution")
    torch = _torch()
    seed_summary = _seed_training_process(int(args.seed), torch=torch)
    start = time.perf_counter()
    image_size = _parse_image_size(args.image_size)
    if float(args.occluded_prob) > 0.0 and float(args.occluded_prob) != 0.25:
        raise ValueError("stage-2 occlusion augmentation must use the pinned occluded_prob=0.25 or be disabled with 0")
    device = _device(args.device, torch=torch)
    out_dir = Path(args.out_dir)
    human_datasets: list[Any] = []
    sst_datasets: list[Any] = []
    if args.b0_split_root is not None:
        if int(args.batch_size) != B2_HUMAN_BATCH_SIZE:
            raise ValueError(f"frozen B0 training requires exactly --batch-size {B2_HUMAN_BATCH_SIZE}")
        if str(args.model_family) != "wasb_hrnet" or int(args.frames_in) != 3 or int(args.output_channels) != 3:
            raise ValueError("frozen B0 training requires the production WASB-HRNet 3-frame/3-output architecture")
        if float(args.occluded_prob) != 0.0:
            raise ValueError("frozen B0 A/B training requires --occluded-prob 0")
        if args.init_checkpoint is None and args.resume_checkpoint is None:
            raise ValueError("frozen B0 training requires the official WASB --init-checkpoint or an exact-provenance resume")
        if args.init_checkpoint is not None:
            expected_wasb = _expected_wasb_identity()
            init_path = Path(args.init_checkpoint)
            expected_init_path = (ROOT / expected_wasb["checkpoint_path"]).resolve(strict=True)
            if init_path.resolve(strict=True) != expected_init_path or _sha256_file(init_path) != expected_wasb["checkpoint_sha256"]:
                raise ValueError("frozen B0 init checkpoint does not match models/MANIFEST.json")
        if args.max_cvat_samples is not None:
            raise ValueError("the frozen B0 split cannot be truncated")
        human_datasets.append(
            B0BallStage2Dataset.from_split_root(
                args.b0_split_root,
                rally_root=args.rally_root,
                image_size=image_size,
                frames_in=int(args.frames_in),
                heatmap_radius_px=float(args.heatmap_radius_px),
                image_path_rewrites=image_path_rewrites,
            )
        )
    if args.cvat_export_root is not None:
        human_datasets.append(
            CvatBallStage2Dataset.from_export_root(
                args.cvat_export_root,
                rally_root=args.rally_root,
                image_size=image_size,
                frames_in=int(args.frames_in),
                heatmap_radius_px=float(args.heatmap_radius_px),
                image_path_rewrites=image_path_rewrites,
                max_samples=args.max_cvat_samples,
            )
        )
    for manifest_path in args.sst_manifest or []:
        sst_datasets.append(
            SstBallStage2Dataset.from_manifest(
                manifest_path,
                image_size=image_size,
                frames_in=int(args.frames_in),
                heatmap_radius_px=float(args.heatmap_radius_px),
                image_path_rewrites=image_path_rewrites,
                max_samples=args.max_sst_samples,
            )
        )
    if sst_datasets and not human_datasets:
        raise ValueError("SST-only training is refused because it bypasses the human-relative loss cap")
    dual_sst_mode = bool(sst_datasets)
    dataset = CombinedStage2Dataset(human_datasets)
    loader, human_batch_sampler = _make_training_loader(
        dataset,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        torch=torch,
        require_full_batches=args.b0_split_root is not None,
    )
    sst_dataset: CombinedStage2Dataset | None = None
    sst_loader: Any | None = None
    if dual_sst_mode:
        sst_batch_size = int(args.sst_batch_size)
        sst_loss_cap = float(args.sst_loss_cap)
        if sst_batch_size <= 0:
            raise ValueError("--sst-batch-size must be positive")
        if not 0.0 <= sst_loss_cap <= 1.0:
            raise ValueError("--sst-loss-cap must be in [0, 1]")
        if sst_batch_size != B2_SST_BATCH_SIZE:
            raise ValueError(f"production SST training requires exactly --sst-batch-size {B2_SST_BATCH_SIZE}")
        if not _exact_number(sst_loss_cap, B2_SST_LOSS_CAP):
            raise ValueError(f"production SST training requires frozen --sst-loss-cap {B2_SST_LOSS_CAP}")
        sst_dataset = CombinedStage2Dataset(sst_datasets)
        sst_loader, sst_batch_sampler = _make_sst_training_loader(
            sst_dataset,
            batch_size=sst_batch_size,
            seed=int(args.seed) + 1,
            num_workers=int(args.num_workers),
            torch=torch,
        )
    else:
        sst_batch_sampler = None
    train_dataset_summary = _training_data_summary(dataset, sst_dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        model_family=str(args.model_family),
        frames_in=int(args.frames_in),
        output_channels=int(args.output_channels),
        image_size=image_size,
        wasb_repo=Path(args.wasb_repo),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    resume_summary = None
    global_step = 0
    if args.resume_checkpoint is not None:
        resume_summary = load_stage2_checkpoint(
            Path(args.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            device=device,
            args=args,
            train_dataset_summary=train_dataset_summary,
        )
        global_step = int(resume_summary["step"])
    if human_batch_sampler is not None:
        human_batch_sampler.set_start_batch(global_step)
    if sst_batch_sampler is not None:
        sst_batch_sampler.set_start_batch(global_step)
    init_summary = None
    if args.init_checkpoint is not None:
        init_summary = load_required_init_checkpoint(
            Path(args.init_checkpoint),
            model=model,
            device=device,
            frames_in=int(args.frames_in),
        )
    steps = int(args.steps) if args.steps is not None else _steps_for_epochs(len(dataset), int(args.batch_size), int(args.epochs))
    if steps <= 0:
        raise ValueError("--steps must be positive")
    occlusion_generator = _loader_generator(int(args.seed) + 10_000, torch=torch)
    losses: list[float] = []
    human_losses: list[float] = []
    sst_losses_post_weighting: list[float] = []
    sst_losses_applied: list[float] = []
    sst_loss_scales: list[float] = []
    human_sample_order: list[list[str]] = []
    sst_sample_order: list[list[str]] = []
    latest_checkpoint: Path | None = None
    model.train()
    batches = _no_cache_cycle(loader)
    sst_batches = _no_cache_cycle(sst_loader) if sst_loader is not None else None
    for _ in range(steps):
        batch = next(batches)
        _assert_human_training_batch(
            batch,
            exact_count=B2_HUMAN_BATCH_SIZE if args.b0_split_root is not None else None,
        )
        human_sample_order.append([str(value) for value in batch.get("sample_id", [])])
        if sst_batches is None:
            # This is intentionally the historical no-SST path. In particular, the
            # loader, RNG seeds, batch, optimizer call, and returned loss are unchanged.
            loss = train_one_stage2_batch(
                model,
                batch,
                optimizer=optimizer,
                device=device,
                torch=torch,
                occluded_prob=float(args.occluded_prob),
                occlusion_generator=occlusion_generator,
            )
        else:
            sst_batch = next(sst_batches)
            _assert_sst_training_batch(sst_batch, exact_count=B2_SST_BATCH_SIZE)
            sst_sample_order.append([str(value) for value in sst_batch.get("sample_id", [])])
            step_loss = train_one_stage2_human_sst_batch(
                model,
                batch,
                sst_batch,
                optimizer=optimizer,
                device=device,
                torch=torch,
                occluded_prob=float(args.occluded_prob),
                occlusion_generator=occlusion_generator,
                sst_loss_cap=float(args.sst_loss_cap),
            )
            loss = step_loss["total_loss"]
            human_losses.append(step_loss["human_loss"])
            sst_losses_post_weighting.append(step_loss["sst_loss_post_weighting"])
            sst_losses_applied.append(step_loss["sst_loss_applied"])
            sst_loss_scales.append(step_loss["sst_loss_scale"])
        global_step += 1
        losses.append(loss)
        if int(args.checkpoint_every) > 0 and global_step % int(args.checkpoint_every) == 0:
            latest_checkpoint = save_stage2_checkpoint(
                checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=global_step,
                args=args,
                train_dataset_summary=train_dataset_summary,
            )
    latest_checkpoint = save_stage2_checkpoint(
        checkpoint_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        step=global_step,
        args=args,
        train_dataset_summary=train_dataset_summary,
    )
    checkpoint_round_trip = checkpoint_round_trip_summary(
        latest_checkpoint,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    loss_summary = _loss_summary(losses)
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "train_complete" if loss_summary["last"] is not None else "train_failed",
        "ball_verified": False,
        "promotion_claimed": False,
        "heldout_touched": False,
        "out_dir": str(out_dir),
        "model": {
            "family": args.model_family,
            "frames_in": int(args.frames_in),
            "output_channels": int(args.output_channels),
            "image_size": list(image_size),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_summary": init_summary,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "resume_summary": resume_summary,
        },
        "recipe": {
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lr_schedule": "constant",
            "heatmap_radius_px": float(args.heatmap_radius_px),
            "occluded_prob": float(args.occluded_prob),
            "occlusion_policy": "model-space patch after official WASB warp, centered on warped target_xy_px before weighted BCE",
            "epochs": int(args.epochs),
            "steps": steps,
            "checkpoint_every": int(args.checkpoint_every),
            **(
                {
                    "human_batch_size": int(args.batch_size),
                    "sst_batch_size": int(args.sst_batch_size),
                    "sst_loss_cap_relative_to_human": float(args.sst_loss_cap),
                    "sst_loss_cap_policy": (
                        "post_weighting pseudo loss is gradient-scaled so its applied scalar "
                        "contribution is <= sst_loss_cap * detached human loss each step"
                    ),
                }
                if dual_sst_mode
                else {}
            ),
        },
        "data": train_dataset_summary,
        "sample_order": {
            "human_by_step": human_sample_order,
            "sst_by_step": sst_sample_order if dual_sst_mode else [],
        },
        "loss": loss_summary,
        **(
            {
                "human_loss": _loss_summary(human_losses),
                "sst_loss_post_weighting": _loss_summary(sst_losses_post_weighting),
                "sst_loss_applied": _loss_summary(sst_losses_applied),
                "sst_loss_scale": _loss_summary(sst_loss_scales),
            }
            if dual_sst_mode
            else {}
        ),
        "checkpoint": {
            "latest_checkpoint": str(latest_checkpoint),
            **checkpoint_round_trip,
            "state_sha256": state_dict_sha256(model.state_dict()),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - start,
            "device": str(device),
            "seed": int(args.seed),
            "seed_summary": seed_summary,
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "limitations": [
            "Owner labels and SST samples are internal-val/build inputs only; no held-out gate is touched.",
            "Sparse CVAT exports train only reviewed frames; unreviewed frames are not fabricated negatives.",
            "Harness metrics are not BALL product gates.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def sparse_tracknet_labels_from_cvat(path: str | Path) -> list[TrackNetCvatLabel]:
    parsed = validate_artifact_file("cvat_video_annotations", path)
    if not isinstance(parsed, CvatVideoAnnotations):
        raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {path}")
    return sparse_tracknet_labels_from_annotations(parsed)


def sparse_tracknet_labels_from_annotations(annotations: CvatVideoAnnotations) -> list[TrackNetCvatLabel]:
    reviewed_indices = _reviewed_indices(annotations)
    frames_by_index = {frame.frame_index: frame for frame in annotations.frames}
    labels: list[TrackNetCvatLabel] = []
    for frame_index in reviewed_indices:
        frame = frames_by_index.get(frame_index)
        if frame is None:
            raise ValueError(f"{annotations.clip_id} reviewed frame {frame_index} is missing from annotations.frames")
        ball_boxes = [box for box in frame.boxes if box.label == "ball"]
        if len(ball_boxes) > 1:
            raise ValueError(f"multiple ball boxes in {annotations.clip_id} frame {frame_index}")
        frame_visibility_level = _frame_ball_visibility_level(frame)
        if not ball_boxes:
            if frame_visibility_level in {"clear", "partial"}:
                raise ValueError(f"{annotations.clip_id} frame {frame_index} has {frame_visibility_level} without a ball box")
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level" if frame_visibility_level is not None else "reviewed_absent_ball",
                    visibility_level=frame_visibility_level,
                    wbce_weight=_visibility_weight_or_clear(frame_visibility_level),
                    legacy_visibility_state=None if frame_visibility_level is not None else "legacy_hidden",
                )
            )
            continue
        box = ball_boxes[0]
        visibility_level = _merge_box_and_frame_visibility_level(
            box.visibility_level,
            frame_visibility_level,
            clip_id=annotations.clip_id,
            frame_index=frame_index,
        )
        if visibility_level in {"full", "out_of_frame"}:
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level",
                    center_convention=box.center_convention,
                    blur_angle_deg=box.blur_angle_deg,
                    blur_length_px=box.blur_length_px,
                    blur_width_px=box.blur_width_px,
                    blur_label_quality=box.blur_label_quality,
                    visibility_level=visibility_level,
                    wbce_weight=_visibility_weight_or_clear(visibility_level),
                )
            )
            continue
        x, y, width, height = box.bbox_xywh
        labels.append(
            TrackNetCvatLabel(
                frame=frame_index,
                visibility=1,
                x=float(x) + float(width) * 0.5,
                y=float(y) + float(height) * 0.5,
                source="reviewed_cvat_ball_box",
                center_convention=box.center_convention,
                blur_angle_deg=box.blur_angle_deg,
                blur_length_px=box.blur_length_px,
                blur_width_px=box.blur_width_px,
                blur_label_quality=box.blur_label_quality,
                visibility_level=visibility_level,
                wbce_weight=_visibility_weight_or_clear(visibility_level),
                legacy_visibility_state=None if visibility_level is not None else "legacy_visible",
            )
        )
    return labels


def load_cvat_annotations_from_export_clip(clip_dir: str | Path) -> CvatVideoAnnotations:
    path = Path(clip_dir)
    reviewed_json = path / "reviewed_boxes.json"
    if reviewed_json.is_file():
        parsed = validate_artifact_file("cvat_video_annotations", reviewed_json)
        if not isinstance(parsed, CvatVideoAnnotations):
            raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {reviewed_json}")
        return parsed
    xml_path = path / "annotations.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"CVAT clip dir needs reviewed_boxes.json or annotations.xml: {path}")
    with tempfile.TemporaryDirectory(prefix="cvat_video_xml_") as tmp_dir:
        zip_path = Path(tmp_dir) / "annotations.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(xml_path, "annotations.xml")
        annotations, _ = import_cvat_video_zip(zip_path, clip_id=path.name)
    return annotations


def train_one_stage2_batch(
    model: Any,
    batch: Mapping[str, Any],
    *,
    optimizer: Any,
    device: Any,
    torch: Any,
    occluded_prob: float,
    occlusion_generator: Any | None,
) -> float:
    augmented = apply_occlusion_augmentation(
        batch,
        occluded_prob=occluded_prob,
        generator=occlusion_generator,
        torch=torch,
    )
    return train_one_batch(
        model,
        augmented,
        optimizer=optimizer,
        device=device,
        torch=torch,
    )


def train_one_stage2_human_sst_batch(
    model: Any,
    human_batch: Mapping[str, Any],
    sst_batch: Mapping[str, Any],
    *,
    optimizer: Any,
    device: Any,
    torch: Any,
    occluded_prob: float,
    occlusion_generator: Any | None,
    sst_loss_cap: float,
) -> dict[str, float]:
    """Take one optimizer step from separate human and pseudo batches.

    Pseudo sample weights are applied inside ``_stage2_weighted_loss`` first. The
    resulting pseudo scalar is then gradient-scaled against a detached human-loss
    scalar. Detaching the cap reference is important: when the pseudo loss hits the
    cap, it must not create an extra ``cap * human_loss`` gradient and thereby change
    the promised eight-human-rows exposure into a hidden human-loss reweighting.
    """

    cap = float(sst_loss_cap)
    if not 0.0 <= cap <= 1.0:
        raise ValueError("sst_loss_cap must be in [0, 1]")
    augmented_human = apply_occlusion_augmentation(
        human_batch,
        occluded_prob=occluded_prob,
        generator=occlusion_generator,
        torch=torch,
    )
    optimizer.zero_grad(set_to_none=True)
    human_loss = _stage2_weighted_loss(model, augmented_human, device=device, torch=torch)
    sst_weights = sst_batch["wbce_weight"]
    skip_sst_forward = cap == 0.0 or bool(torch.count_nonzero(sst_weights).item() == 0)
    if skip_sst_forward:
        # A zero-contribution pseudo arm must be state- and RNG-identical to the
        # human-only arm. In particular, it may not update BatchNorm buffers.
        sst_loss = human_loss.new_zeros(())
    else:
        # Pseudo gradients remain enabled, but the pseudo forward uses inference
        # behavior so capped supervision cannot make an uncapped state-buffer update.
        with _temporary_model_eval(model):
            sst_loss = _stage2_weighted_loss(model, sst_batch, device=device, torch=torch)

    human_value = float(human_loss.detach().cpu())
    sst_value = float(sst_loss.detach().cpu())
    maximum_sst_value = cap * human_value
    if sst_value <= 0.0 or maximum_sst_value <= 0.0:
        sst_scale = 0.0
    else:
        sst_scale = min(1.0, maximum_sst_value / sst_value)
    applied_sst = sst_loss * sst_scale
    total_loss = human_loss + applied_sst
    total_loss.backward()
    optimizer.step()
    return {
        "human_loss": human_value,
        "sst_loss_post_weighting": sst_value,
        "sst_loss_applied": float(applied_sst.detach().cpu()),
        "sst_loss_scale": float(sst_scale),
        "total_loss": float(total_loss.detach().cpu()),
    }


@contextmanager
def _temporary_model_eval(model: Any) -> Any:
    training_states = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in training_states:
            module.train(was_training)


def _stage2_weighted_loss(model: Any, batch: Mapping[str, Any], *, device: Any, torch: Any) -> Any:
    inputs = batch["input"].to(device)
    target = batch["target"].to(device)
    weights = batch["wbce_weight"].to(device).view(-1)
    logits = _primary_logits(model(inputs))
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    target = target.repeat(1, logits.shape[1], 1, 1)
    loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return (loss_map.flatten(1).mean(dim=1) * weights).mean()


def apply_occlusion_augmentation(
    batch: Mapping[str, Any],
    *,
    occluded_prob: float,
    generator: Any | None,
    torch: Any,
) -> dict[str, Any]:
    if occluded_prob <= 0.0:
        return dict(batch)
    if "wbce_weight" not in batch:
        raise ValueError("occlusion augmentation is allowed only with visibility-weighted WBCE batch['wbce_weight']")
    inputs = batch["input"].clone()
    target_xy = batch["target_xy_px"]
    ball_present = batch["ball_present"]
    if inputs.ndim != 4:
        raise ValueError(f"batch input must be BCHW, got {tuple(inputs.shape)}")
    batch_size, _, height, width = inputs.shape
    selected = torch.rand(batch_size, generator=generator) < float(occluded_prob)
    patch = max(4, int(round(min(height, width) * 0.15)))
    half = max(1, patch // 2)
    for index in range(batch_size):
        if not bool(selected[index]) or float(ball_present[index]) <= 0.0:
            continue
        x = int(round(float(target_xy[index][0])))
        y = int(round(float(target_xy[index][1])))
        x0 = max(0, x - half)
        x1 = min(width, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(height, y + half + 1)
        inputs[index, :, y0:y1, x0:x1] = 0.0
    out = dict(batch)
    out["input"] = inputs
    return out


def load_required_init_checkpoint(path: Path, *, model: Any, device: Any, frames_in: int) -> dict[str, Any]:
    payload = _torch().load(path, map_location=device, weights_only=False)
    if isinstance(payload, Mapping) and payload.get("frames_in") is not None and int(payload["frames_in"]) != int(frames_in):
        raise RuntimeError(f"init checkpoint frames_in mismatch: checkpoint={payload['frames_in']} requested={frames_in}")
    summary = load_model_weights(path, model=model, device=device, strict=False)
    missing = list(summary.get("missing_keys", []))
    unexpected = list(summary.get("unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"init checkpoint key mismatch: missing_keys={missing} unexpected_keys={unexpected}")
    return summary


def save_stage2_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    args: argparse.Namespace,
    train_dataset_summary: Mapping[str, Any],
) -> Path:
    dataset_provenance = train_dataset_summary.get("dataset_provenance")
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_checkpoint",
        "step": int(step),
        "model_family": args.model_family,
        "frames_in": int(args.frames_in),
        "output_channels": int(args.output_channels),
        "image_size": list(_parse_image_size(args.image_size)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_dataset_summary": dict(train_dataset_summary),
        "dataset_provenance": (
            copy.deepcopy(dict(dataset_provenance)) if isinstance(dataset_provenance, Mapping) else None
        ),
    }
    atomic_torch_save(payload, path, torch=_torch())
    return path


def load_stage2_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
    train_dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _torch().load(path, map_location=device, weights_only=False)
    if int(payload.get("frames_in", -1)) != int(args.frames_in):
        raise RuntimeError(f"resume checkpoint frames_in mismatch: checkpoint={payload.get('frames_in')} requested={args.frames_in}")
    if int(payload.get("output_channels", -1)) != int(args.output_channels):
        raise RuntimeError(
            f"resume checkpoint output_channels mismatch: checkpoint={payload.get('output_channels')} requested={args.output_channels}"
        )
    if str(payload.get("model_family")) != str(args.model_family):
        raise RuntimeError(f"resume checkpoint model_family mismatch: checkpoint={payload.get('model_family')} requested={args.model_family}")
    checkpoint_provenance = payload.get("dataset_provenance")
    current_provenance = train_dataset_summary.get("dataset_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise RuntimeError("resume checkpoint is missing required dataset provenance")
    if not isinstance(current_provenance, Mapping):
        raise RuntimeError("current training dataset is missing required dataset provenance")
    if dict(checkpoint_provenance) != dict(current_provenance):
        raise RuntimeError(
            "resume checkpoint dataset provenance mismatch: "
            f"checkpoint={checkpoint_provenance.get('dataset_identity_set_sha256')} "
            f"current={current_provenance.get('dataset_identity_set_sha256')}"
        )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {
        "checkpoint": str(path),
        "step": int(payload["step"]),
        "dataset_identity_set_sha256": str(current_provenance.get("dataset_identity_set_sha256")),
    }


def _record_from_cvat_label(
    annotations: CvatVideoAnnotations,
    label: TrackNetCvatLabel,
    *,
    video_path: Path,
    source_video_sha256: str,
    source_path: str,
) -> Stage2SampleRecord:
    parent_source_id = _parent_source_id(annotations.clip_id)
    return Stage2SampleRecord(
        sample_id=f"cvat:{annotations.clip_id}:{label.frame}",
        source_kind="cvat_owner_sparse",
        clip_id=annotations.clip_id,
        parent_source_id=parent_source_id,
        video_path=video_path,
        source_video_sha256=source_video_sha256,
        frame_index=int(label.frame),
        source_width=int(annotations.task.original_size[0]),
        source_height=int(annotations.task.original_size[1]),
        ball_present=bool(label.visibility == 1),
        source_xy_px=(float(label.x), float(label.y)),
        visibility_level=label.visibility_level,
        wbce_weight=float(label.wbce_weight if label.wbce_weight is not None else 1),
        source_path=source_path,
    )


def _record_to_item(
    record: Stage2SampleRecord,
    *,
    image_size: tuple[int, int],
    frames_in: int,
    heatmap_radius_px: float,
    image_path_rewrites: Mapping[str, str],
) -> dict[str, Any]:
    torch = _torch()
    np = _numpy()
    cv2 = _cv2()
    target_w, target_h = image_size
    video_path = _rewrite_path(record.video_path, image_path_rewrites)
    offsets = _window_offsets(frames_in)
    frames_rgb = [
        _read_video_frame_rgb(video_path, max(0, record.frame_index + offset))
        for offset in offsets
    ]
    trans_input = _wasb_official_input_affine(
        record.source_width,
        record.source_height,
        cv2=cv2,
        np=np,
        output_wh=image_size,
    )
    input_tensor = _preprocess_wasb_window_official(
        frames_rgb,
        trans_input,
        cv2=cv2,
        np=np,
        torch=torch,
        output_wh=image_size,
    )
    if record.ball_present:
        warped_xy = _wasb_affine_transform_xy(record.source_xy_px, trans_input, np=np)
        scaled_x = float(warped_xy[0])
        scaled_y = float(warped_xy[1])
        target = _gaussian_heatmap(scaled_x, scaled_y, width=target_w, height=target_h, radius=heatmap_radius_px, torch=torch)
        target_xy = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
    else:
        target = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        target_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    return {
        "sample_id": record.sample_id,
        "source_slug": record.clip_id,
        "parent_source_id": record.parent_source_id,
        "bucket": record.source_kind,
        "source_split": "train",
        "image_path": str(video_path),
        "window_sample_ids": [f"{record.clip_id}:{max(0, record.frame_index + offset)}" for offset in offsets],
        "temporal_sample_kind": "video_window",
        "input": input_tensor,
        "target": target,
        "target_xy_px": target_xy,
        "source_xy_px": torch.tensor(record.source_xy_px, dtype=torch.float32),
        "ball_present": torch.tensor(1.0 if record.ball_present else 0.0, dtype=torch.float32),
        "wbce_weight": torch.tensor(float(record.wbce_weight), dtype=torch.float32),
        "visibility_level": record.visibility_level,
    }


def _read_video_frame_rgb(path: Path, frame_index: int) -> Any:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise ValueError(f"could not read frame {frame_index} from {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(f"could not read frame {frame_index} from {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb
    finally:
        capture.release()


def _video_size(path: Path) -> tuple[int, int]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"could not read video size: {path}")
    return width, height


def _gaussian_heatmap(x: float, y: float, *, width: int, height: int, radius: float, torch: Any) -> Any:
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    heatmap = torch.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * float(radius) ** 2))
    return heatmap.unsqueeze(0).clamp(0.0, 1.0)


def _reviewed_indices(annotations: CvatVideoAnnotations) -> list[int]:
    if annotations.reviewed_frame_indices is not None:
        return list(annotations.reviewed_frame_indices)
    if annotations.task.frame_filter:
        raise ValueError(
            f"{annotations.clip_id} has frame_filter={annotations.task.frame_filter!r} but no reviewed_frame_indices; "
            "cannot distinguish reviewed-absent frames from never-reviewed frames"
        )
    return [frame.frame_index for frame in sorted(annotations.frames, key=lambda frame: frame.frame_index)]


def _frame_ball_visibility_level(frame: CvatVideoFrame | None) -> BallVisibilityLevel | None:
    if frame is None:
        return None
    return frame.visibility_levels_by_label.get("ball")


def _merge_box_and_frame_visibility_level(
    box_level: BallVisibilityLevel | None,
    frame_level: BallVisibilityLevel | None,
    *,
    clip_id: str,
    frame_index: int,
) -> BallVisibilityLevel | None:
    if box_level is not None and frame_level is not None and box_level != frame_level:
        raise ValueError(f"{clip_id} frame {frame_index} has conflicting ball visibility levels: {box_level} vs {frame_level}")
    return box_level or frame_level


def _visibility_weight_or_clear(level: BallVisibilityLevel | None) -> int:
    if level is None:
        return BALL_VISIBILITY_WBCE_WEIGHTS["clear"]
    return BALL_VISIBILITY_WBCE_WEIGHTS[level]


def _resolve_rally_video(rally_root: Path, clip_id: str) -> Path:
    if "_rally_" not in clip_id:
        raise ValueError(f"cannot infer rally source id from clip id: {clip_id}")
    source_id = clip_id.split("_rally_", 1)[0]
    path = rally_root / source_id / f"{clip_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"missing rally video for {clip_id}: {path}")
    return path


def _parent_source_id(clip_id: str) -> str:
    return clip_id.split("_rally_", 1)[0] if "_rally_" in clip_id else clip_id


def _canonical_b0_media_identity(source_video_sha256: str) -> tuple[str, str] | None:
    return B0_CANONICAL_MEDIA_IDENTITY_BY_SHA256.get(source_video_sha256)


def _rewrite_path(path: Path, rewrites: Mapping[str, str]) -> Path:
    text = str(path)
    for old_prefix, new_prefix in rewrites.items():
        if text == old_prefix or text.startswith(f"{old_prefix}/"):
            return Path(f"{new_prefix}{text[len(old_prefix):]}")
    return path


def _window_offsets(frames_in: int) -> list[int]:
    half = frames_in // 2
    return list(range(-half, half + 1))


def _dataset_summary(
    source_kind: str,
    records: Sequence[Stage2SampleRecord],
    image_size: tuple[int, int],
    frames_in: int,
    heatmap_radius_px: float,
) -> dict[str, Any]:
    weights: dict[str, int] = {}
    visibility: dict[str, int] = {}
    for record in records:
        weights[str(int(record.wbce_weight) if float(record.wbce_weight).is_integer() else record.wbce_weight)] = (
            weights.get(str(int(record.wbce_weight) if float(record.wbce_weight).is_integer() else record.wbce_weight), 0) + 1
        )
        key = record.visibility_level or "none"
        visibility[key] = visibility.get(key, 0) + 1
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_dataset_summary",
        "source_kind": source_kind,
        "selected_sample_count": len(records),
        "clip_count": len({record.clip_id for record in records}),
        "positive_sample_count": sum(1 for record in records if record.ball_present),
        "negative_sample_count": sum(1 for record in records if not record.ball_present),
        "visibility_level_counts": dict(sorted(visibility.items())),
        "wbce_weight_counts": dict(sorted(weights.items())),
        "image_size": list(image_size),
        "input_preprocessing": "wasb_official_affine_imagenet",
        "frames_in": int(frames_in),
        "heatmap_radius_px": float(heatmap_radius_px),
        "sparse_review_policy": "only reviewed_frame_indices are training rows; unreviewed frames are never fabricated negatives",
    }


def _dataset_provenance(records: Sequence[Stage2SampleRecord]) -> dict[str, Any]:
    media_identities: dict[tuple[str, str, str], dict[str, str]] = {}
    sample_identities: list[dict[str, Any]] = []
    for record in records:
        media_sha256 = _require_sha256(record.source_video_sha256, "training source-video SHA-256")
        canonical = _canonical_b0_media_identity(media_sha256)
        canonical_clip, canonical_parent = canonical or (record.clip_id, record.parent_source_id)
        media_key = (media_sha256, canonical_clip, canonical_parent)
        media_identities[media_key] = {
            "source_video_sha256": media_sha256,
            "canonical_clip_id": canonical_clip,
            "canonical_parent_source_id": canonical_parent,
        }
        sample_identities.append(
            {
                "sample_id": record.sample_id,
                "source_kind": record.source_kind,
                "canonical_clip_id": canonical_clip,
                "canonical_parent_source_id": canonical_parent,
                "source_video_sha256": media_sha256,
                "frame_index": int(record.frame_index),
                "ball_present": bool(record.ball_present),
                "source_xy_px": [float(record.source_xy_px[0]), float(record.source_xy_px[1])],
                "wbce_weight": float(record.wbce_weight),
            }
        )
    media_identity_set = [media_identities[key] for key in sorted(media_identities)]
    sample_identity_set_sha256 = _canonical_json_sha256(
        sorted(sample_identities, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    )
    contract = {
        "schema_version": 1,
        "identity_mode": "content_sha256_plus_canonical_clip_parent_and_exact_sample_set",
        "selected_sample_count": len(records),
        "media_identity_set": media_identity_set,
        "sample_identity_set_sha256": sample_identity_set_sha256,
    }
    contract["dataset_identity_set_sha256"] = _canonical_json_sha256(contract)
    return contract


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_dataset_shape(*, image_size: tuple[int, int], frames_in: int, heatmap_radius_px: float) -> None:
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive width,height")
    if frames_in <= 0 or frames_in % 2 == 0:
        raise ValueError("frames_in must be a positive odd integer")
    if heatmap_radius_px <= 0.0:
        raise ValueError("heatmap_radius_px must be positive")


def _steps_for_epochs(sample_count: int, batch_size: int, epochs: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return max(1, int(math.ceil(sample_count / float(batch_size))) * int(epochs))


def _loss_summary(losses: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(losses),
        "first": float(losses[0]) if losses else None,
        "last": float(losses[-1]) if losses else None,
        "strictly_decreased": bool(losses and losses[-1] < losses[0]),
        "values": [float(value) for value in losses],
    }


def _training_data_summary(
    human_dataset: CombinedStage2Dataset,
    sst_dataset: CombinedStage2Dataset | None,
) -> dict[str, Any]:
    if sst_dataset is None:
        return human_dataset.summary
    human_provenance = _required_mapping(
        human_dataset.summary.get("dataset_provenance"), "human dataset_provenance"
    )
    sst_provenance = _required_mapping(
        sst_dataset.summary.get("dataset_provenance"), "SST dataset_provenance"
    )
    combined_identity = {"human": dict(human_provenance), "sst": dict(sst_provenance)}
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_dual_batch_dataset_summary",
        "human": human_dataset.summary,
        "sst": sst_dataset.summary,
        "dataset_provenance": {
            "schema_version": 1,
            "identity_mode": "role_partitioned_human_and_sst_dataset_identity_sets",
            **combined_identity,
            "dataset_identity_set_sha256": _canonical_json_sha256(combined_identity),
        },
        "human_exposure_policy": "one full --batch-size human batch every optimizer step",
        "sst_exposure_policy": (
            "one independent --sst-batch-size pseudo batch every optimizer step; "
            "dataset indices are distinct within every batch"
        ),
    }


class DeterministicFullBatchSampler:
    """Cycle seeded permutations without ever emitting or discarding a short batch."""

    def __init__(self, sample_count: int, batch_size: int, *, seed: int, torch: Any) -> None:
        if sample_count <= 0:
            raise ValueError("full-batch sampler requires at least one sample")
        if batch_size <= 0:
            raise ValueError("full-batch sampler batch_size must be positive")
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.torch = torch
        self.start_batch = 0

    def set_start_batch(self, value: int) -> None:
        if int(value) < 0:
            raise ValueError("full-batch sampler start_batch must be nonnegative")
        self.start_batch = int(value)

    def __iter__(self) -> Any:
        generator = _loader_generator(self.seed, torch=self.torch)
        pending: list[int] = []
        batches_seen = 0
        while True:
            pending.extend(int(index) for index in self.torch.randperm(self.sample_count, generator=generator).tolist())
            while len(pending) >= self.batch_size:
                batch = pending[: self.batch_size]
                del pending[: self.batch_size]
                if batches_seen >= self.start_batch:
                    yield batch
                batches_seen += 1


class DeterministicDistinctFullBatchSampler(DeterministicFullBatchSampler):
    """Cycle seeded permutations with distinct indices in every SST batch.

    Only the SST loader uses this sampler. At a permutation splice, a colliding
    prefix index is swapped with the first later index that is not already in
    the pending tail. The algorithm consumes no additional RNG values and
    preserves every generated permutation's complete multiset.
    """

    def __init__(self, sample_count: int, batch_size: int, *, seed: int, torch: Any) -> None:
        super().__init__(sample_count, batch_size, seed=seed, torch=torch)
        if self.sample_count < self.batch_size:
            raise ValueError(
                "distinct full-batch sampler requires sample_count >= batch_size"
            )

    def __iter__(self) -> Any:
        generator = _loader_generator(self.seed, torch=self.torch)
        pending: list[int] = []
        batches_seen = 0
        while True:
            permutation = [
                int(index)
                for index in self.torch.randperm(
                    self.sample_count,
                    generator=generator,
                ).tolist()
            ]
            if pending:
                prefix_count = self.batch_size - len(pending)
                blocked = set(pending)
                replacement_cursor = prefix_count
                for prefix_index in range(prefix_count):
                    if permutation[prefix_index] not in blocked:
                        continue
                    while (
                        replacement_cursor < len(permutation)
                        and permutation[replacement_cursor] in blocked
                    ):
                        replacement_cursor += 1
                    if replacement_cursor >= len(permutation):
                        raise RuntimeError(
                            "distinct full-batch sampler could not repair a permutation splice"
                        )
                    permutation[prefix_index], permutation[replacement_cursor] = (
                        permutation[replacement_cursor],
                        permutation[prefix_index],
                    )
                    replacement_cursor += 1
            pending.extend(permutation)
            while len(pending) >= self.batch_size:
                batch = pending[: self.batch_size]
                del pending[: self.batch_size]
                if len(set(batch)) != self.batch_size:
                    raise RuntimeError(
                        "distinct full-batch sampler emitted duplicate dataset indices"
                    )
                if batches_seen >= self.start_batch:
                    yield batch
                batches_seen += 1


def _make_training_loader(
    dataset: Any,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    torch: Any,
    require_full_batches: bool,
) -> tuple[Any, DeterministicFullBatchSampler | None]:
    if require_full_batches:
        sampler = DeterministicFullBatchSampler(len(dataset), batch_size, seed=seed, torch=torch)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            generator=_loader_generator(seed + 50_000, torch=torch),
            worker_init_fn=_seed_loader_worker,
            collate_fn=_collate_stage2_batch,
        )
        return loader, sampler
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=_loader_generator(seed, torch=torch),
        worker_init_fn=_seed_loader_worker,
        collate_fn=_collate_stage2_batch,
    )
    return loader, None


def _make_sst_training_loader(
    dataset: Any,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    torch: Any,
) -> tuple[Any, DeterministicDistinctFullBatchSampler]:
    sampler = DeterministicDistinctFullBatchSampler(
        len(dataset),
        batch_size,
        seed=seed,
        torch=torch,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        generator=_loader_generator(seed + 50_000, torch=torch),
        worker_init_fn=_seed_loader_worker,
        collate_fn=_collate_stage2_batch,
    )
    return loader, sampler


def _collate_stage2_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = _collate_batch(items)
    batch["parent_source_id"] = [str(item["parent_source_id"]) for item in items]
    return batch


def _assert_human_training_batch(batch: Mapping[str, Any], *, exact_count: int | None) -> None:
    count = int(batch["input"].shape[0])
    if exact_count is not None and count != exact_count:
        raise RuntimeError(f"human batch must contain exactly {exact_count} rows, got {count}")
    parents = [str(value) for value in batch.get("parent_source_id", [])]
    if len(parents) != count:
        raise RuntimeError("human batch is missing parent_source_id lineage")
    leaked = sorted(set(parents) & B0_JUDGE_PARENT_IDS)
    if leaked:
        raise RuntimeError(f"judge-parent rows can never enter a training batch: {leaked}")
    if exact_count is not None and not set(parents) <= B0_TRAIN_SOURCE_IDS:
        raise RuntimeError(f"frozen B0 batch contains noncanonical parents: {sorted(set(parents) - B0_TRAIN_SOURCE_IDS)}")


def _assert_sst_training_batch(batch: Mapping[str, Any], *, exact_count: int) -> None:
    count = int(batch["input"].shape[0])
    if count != exact_count:
        raise RuntimeError(f"SST batch must contain exactly {exact_count} additive rows, got {count}")
    sample_ids = [str(value) for value in batch.get("sample_id", [])]
    if len(sample_ids) != count:
        raise RuntimeError("SST batch is missing sample_id lineage")
    if len(set(sample_ids)) != count:
        duplicates = sorted(
            sample_id
            for sample_id in set(sample_ids)
            if sample_ids.count(sample_id) > 1
        )
        raise RuntimeError(f"SST batch contains duplicate sample IDs: {duplicates}")
    parents = [str(value) for value in batch.get("parent_source_id", [])]
    if len(parents) != count or not set(parents) <= SST_TRAIN_SOURCE_IDS:
        raise RuntimeError("SST batch contains missing or noncanonical source identities")


def _no_cache_cycle(loader: Any) -> Any:
    while True:
        for batch in loader:
            yield batch


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _exact_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == float(expected)
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _validated_xy(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be [x, y]")
    return (_finite_float(value[0], f"{field}[0]"), _finite_float(value[1], f"{field}[1]"))


def _inside_xy(xy: Sequence[float], *, width: int, height: int) -> bool:
    return 0.0 <= float(xy[0]) < float(width) and 0.0 <= float(xy[1]) < float(height)


def _validated_bbox(
    value: Any,
    *,
    field: str,
    width: int | None = None,
    height: int | None = None,
) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field} must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = tuple(_finite_float(component, f"{field}[{index}]") for index, component in enumerate(value))
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field} is not a positive in-bounds box")
    if width is not None and x2 > float(width):
        raise ValueError(f"{field} exceeds image width {width}")
    if height is not None and y2 > float(height):
        raise ValueError(f"{field} exceeds image height {height}")
    return x1, y1, x2, y2


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _row_mentions_source(value: Any, source_id: str) -> bool:
    if isinstance(value, Mapping):
        return any(_row_mentions_source(item, source_id) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_row_mentions_source(item, source_id) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    tokens = normalized.split("/")
    return (
        value == source_id
        or any(token == source_id for token in tokens)
        or any(token.startswith(f"{source_id}_rally_") for token in tokens)
        or any(token.startswith(f"{source_id}__") for token in tokens)
    )


def _expected_wasb_identity() -> dict[str, str]:
    manifest_path = ROOT / "models/MANIFEST.json"
    payload = _read_json_object(manifest_path)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("models/MANIFEST.json requires models list")
    matches = [row for row in models if isinstance(row, Mapping) and row.get("id") == SST_WASB_MODEL_ID]
    if len(matches) != 1:
        raise ValueError(f"models/MANIFEST.json must contain exactly one {SST_WASB_MODEL_ID}")
    row = matches[0]
    checkpoint_sha = _require_sha256(row.get("sha256"), "models manifest WASB sha256")
    repo_commit = str(row.get("repo_commit") or "")
    if re.fullmatch(r"[0-9a-f]{40}", repo_commit) is None:
        raise ValueError("models manifest WASB repo_commit must be a full git SHA")
    return {
        "models_manifest_sha256": _sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "repo_commit": repo_commit,
        "checkpoint_path": str(row.get("local_path") or ""),
    }


def _validate_gate_count(gate: Mapping[str, Any], key: str, actual: int, target: int) -> None:
    row = _required_mapping(gate.get(key), f"SST gate {key}")
    if int(row.get("after", -1)) != int(actual) or int(row.get("target", -1)) != int(target):
        raise ValueError(f"SST gate {key} does not match independent recount")


def _git_commit(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"git returned invalid commit for {revision}: {commit!r}")
    return commit


def _capture_torch_rng(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_torch_rng(state: Mapping[str, Any], torch: Any) -> None:
    torch.random.set_rng_state(state["cpu"])
    if state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_states_equal(a: Mapping[str, Any], b: Mapping[str, Any], *, torch: Any) -> bool:
    if not bool(torch.equal(a["cpu"], b["cpu"])):
        return False
    a_cuda = a.get("cuda") or []
    b_cuda = b.get("cuda") or []
    return len(a_cuda) == len(b_cuda) and all(bool(torch.equal(x, y)) for x, y in zip(a_cuda, b_cuda))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cli_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("artifact_type") == "racketsport_ball_stage2_pinned_head_parity":
        return {
            "status": "head_parity_complete",
            "verdict": summary.get("verdict"),
            "head_parity_json": str(Path(str(summary["out_dir"])) / "head_parity.json"),
            "baseline_trainer": summary.get("baseline_trainer"),
            "candidate_trainer": summary.get("candidate_trainer"),
            "trainer_sources_distinct": summary.get("trainer_sources_distinct"),
            "model_state_sha256": summary.get("model_state_sha256"),
        }
    if summary.get("artifact_type") == "racketsport_ball_sst_manifest":
        return {
            "status": "sst_manifest_written",
            "summary": summary.get("summary"),
            "weight_policy": summary.get("weight_policy"),
            "protected_eval_hash_check": summary.get("protected_eval_hash_check"),
        }
    return {
        "status": summary.get("status"),
        "summary_json": str(Path(str(summary["out_dir"])) / "summary.json") if summary.get("out_dir") else None,
        "checkpoint": summary.get("checkpoint"),
        "loss": summary.get("loss"),
        "runtime": summary.get("runtime"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BALL2D stage-2 on sparse owner CVAT labels and/or SST pseudo-label manifests.",
    )
    parser.add_argument("--mode", choices=("train", "build-sst-manifest", "verify-head-parity"), default="train")
    parser.add_argument(
        "--baseline-rev",
        default=None,
        help="Explicit git revision for verify-head-parity; no HEAD default is permitted.",
    )
    parser.add_argument(
        "--baseline-sha256",
        default=None,
        help="Expected SHA-256 of the trainer blob loaded from --baseline-rev.",
    )
    parser.add_argument("--cvat-export-root", type=Path, default=None)
    parser.add_argument(
        "--b0-split-root",
        type=Path,
        default=None,
        help="Content-pinned accepted B0 parent-source split; required for B2/SST training.",
    )
    parser.add_argument("--sst-manifest", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="wasb_hrnet")
    parser.add_argument("--wasb-repo", type=Path, default=Path("third_party/WASB-SBDT"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--image-size", default=f"{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[0]}x{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[1]}")
    parser.add_argument("--frames-in", type=int, default=DEFAULT_BALL_PRETRAIN_FRAMES_IN)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--heatmap-radius-px", type=float, default=DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sst-batch-size",
        type=int,
        default=8,
        help="Independent pseudo-label rows per step; does not replace or shrink the human batch.",
    )
    parser.add_argument(
        "--sst-loss-cap",
        type=float,
        default=0.25,
        help="Maximum post-weighting pseudo-loss contribution relative to human loss per step.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--occluded-prob", type=float, default=0.25)
    parser.add_argument("--max-cvat-samples", type=int, default=None)
    parser.add_argument("--max-sst-samples", type=int, default=None)
    parser.add_argument("--rally-root", type=Path, default=DEFAULT_RALLY_ROOT)
    parser.add_argument(
        "--image-root-rewrite",
        action="append",
        default=[],
        help="Rewrite absolute video paths as OLD_PREFIX=NEW_PREFIX for VM/checkouts at a different root.",
    )
    parser.add_argument("--prelabel-root", type=Path, default=DEFAULT_PRELABEL_ROOT)
    parser.add_argument("--sst-manifest-out", type=Path, default=None)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--max-sst-samples-per-clip", type=int, default=None)
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--eval-sample-every-s", type=float, default=DEFAULT_EVAL_SAMPLE_EVERY_S)
    parser.add_argument("--expected-protected-eval-hash-count", type=int, default=DEFAULT_PROTECTED_EVAL_HASH_COUNT)
    parser.add_argument("--collision-hamming-threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD)
    parser.add_argument("--protected-eval-hash", action="append", default=[])
    return parser


def _torch() -> Any:
    import torch

    return torch


def _numpy() -> Any:
    import numpy as np

    return np


def _cv2() -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
