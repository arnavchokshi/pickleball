"""Deep-tier HMR primitives and optional Fast SAM-3D-Body runtime bridge.

The pure helpers validate and package Fast SAM-3D-Body/MHR-style inputs and
outputs. ``FastSam3DBodyRuntime`` is the explicit runtime bridge that can load
external checkpoints and run inference when the H100 environment is available.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from . import mhr_decode
from .model_manifest import load_model_manifest
from .sam3d_body_input_prep import load_mask_prompt_arrays, normalize_body_input_size


SCAFFOLD_NOTE = "cpu_hmr_deep_primitives_no_model_inference"
SCHEMA_VERSION = "body_hmr_deep.v0"
MODEL_FAMILY = "fast_sam_3d_body_mhr_to_smpl"
DEFAULT_BODY_MANIFEST_PATH = Path("models/MANIFEST.json")


class _LazyEnvPathDefault(os.PathLike[str]):
    """Path-like default that reads its environment override at use time."""

    def __init__(self, env_name: str, fallback: str) -> None:
        self.env_name = env_name
        self.fallback = fallback

    def __fspath__(self) -> str:
        return os.environ.get(self.env_name, self.fallback)

    def __str__(self) -> str:
        return os.fspath(self)

    def __repr__(self) -> str:
        return f"Path({str(self)!r})"


DEFAULT_FAST_SAM_REPO = _LazyEnvPathDefault("FAST_SAM_ROOT", "/opt/fast-sam-3d-body")
REQUIRED_FAST_SAM_MODEL_IDS = (
    "fast_sam_3d_body_dinov3",
    "sam_3d_body_mhr_model",
    "moge_2_vitl_normal",
    "yolo26m",
)
CORE_FAST_SAM_MODEL_IDS = (
    "fast_sam_3d_body_dinov3",
    "sam_3d_body_mhr_model",
)
SAM3D_FOOT_KEYPOINT_INDICES = {
    "left_ankle": 13,
    "right_ankle": 14,
    "left_toe": 15,
    "right_toe": 16,
    "left_heel": 17,
    "right_heel": 20,
}


@dataclass(frozen=True)
class VerifiedModelAsset:
    """Manifest-backed checkpoint path verified before runtime setup."""

    model_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class PlayerCropRequest:
    """Validated per-player crop request for the deep mesh tier."""

    frame_idx: int
    player_id: int
    bbox_xyxy: Sequence[float]
    image_size_px: Sequence[int]
    track_confidence: float
    source_track_id: str | None = None
    rally_span_id: str | None = None

    def __post_init__(self) -> None:
        frame_idx = _non_negative_int(self.frame_idx, name="frame_idx")
        player_id = _non_negative_int(self.player_id, name="player_id")
        bbox = _float_vector(self.bbox_xyxy, name="bbox_xyxy", length=4)
        image_size = _image_size(self.image_size_px)
        confidence = _confidence(self.track_confidence, name="track_confidence")

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        if x1 < 0.0 or y1 < 0.0 or x2 > image_size[0] or y2 > image_size[1]:
            raise ValueError("bbox_xyxy must be inside image_size_px")

        object.__setattr__(self, "frame_idx", frame_idx)
        object.__setattr__(self, "player_id", player_id)
        object.__setattr__(self, "bbox_xyxy", tuple(bbox))
        object.__setattr__(self, "image_size_px", tuple(image_size))
        object.__setattr__(self, "track_confidence", confidence)
        if self.source_track_id is not None:
            object.__setattr__(self, "source_track_id", str(self.source_track_id))
        if self.rally_span_id is not None:
            object.__setattr__(self, "rally_span_id", str(self.rally_span_id))

    @property
    def crop_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def area_px(self) -> float:
        return self.crop_xywh[2] * self.crop_xywh[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "player_id": self.player_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "crop_xywh": list(self.crop_xywh),
            "image_size_px": list(self.image_size_px),
            "track_confidence": self.track_confidence,
            "source_track_id": self.source_track_id,
            "rally_span_id": self.rally_span_id,
            "scaffold": SCAFFOLD_NOTE,
        }


def normalize_deep_hmr_payload(
    payload: Mapping[str, Any],
    *,
    request: PlayerCropRequest,
) -> dict[str, Any]:
    """Normalize a model-like MHR/SMPL payload into schema-friendly fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")

    smpl = payload.get("smpl")
    if not isinstance(smpl, Mapping):
        raise ValueError("payload.smpl must be a mapping")

    mhr = payload.get("mhr", {})
    if mhr is None:
        mhr = {}
    if not isinstance(mhr, Mapping):
        raise ValueError("payload.mhr must be a mapping when present")

    model_confidence = _confidence(
        payload.get("confidence", payload.get("model_confidence", request.track_confidence)),
        name="confidence",
    )
    mhr_pose_confidence = _optional_confidence(mhr.get("pose_confidence"), name="mhr.pose_confidence")
    confidence_components = {
        "model_confidence": model_confidence,
        "track_confidence": request.track_confidence,
    }
    confidence_values = [model_confidence, request.track_confidence]
    if mhr_pose_confidence is not None:
        confidence_components["mhr_pose_confidence"] = mhr_pose_confidence
        confidence_values.append(mhr_pose_confidence)

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_idx": request.frame_idx,
        "player_id": request.player_id,
        "model_family": MODEL_FAMILY,
        "representation": "smpl_ish_cpu_normalized",
        "smpl": {
            "global_orient": _float_vector(
                smpl.get("global_orient"),
                name="smpl.global_orient",
                length=3,
            ),
            "body_pose": _float_list(smpl.get("body_pose", []), name="smpl.body_pose"),
            "betas": _float_list(smpl.get("betas", []), name="smpl.betas"),
            "transl": _float_vector(smpl.get("transl"), name="smpl.transl", length=3),
        },
        "mhr": dict(mhr),
        "mesh_vertices_xyz": _vector3_list(
            payload.get("mesh_vertices_xyz", payload.get("vertices", [])),
            name="mesh_vertices_xyz",
        ),
        "joints3d_xyz": _vector3_list(
            payload.get("joints3d_xyz", payload.get("joints3d", [])),
            name="joints3d_xyz",
        ),
        "confidence": min(confidence_values),
        "confidence_components": confidence_components,
        "scaffold": SCAFFOLD_NOTE,
    }


def gate_deep_hmr_artifact(
    hmr_output: Mapping[str, Any],
    *,
    model_inference_ran: bool,
    min_confidence: float = 0.65,
) -> dict[str, Any]:
    """Return deterministic gate metadata for one normalized player output."""

    threshold = _confidence(min_confidence, name="min_confidence")
    confidence = _confidence(hmr_output.get("confidence", 0.0), name="hmr_output.confidence")
    reasons: list[str] = []

    if confidence < threshold:
        reasons.append("low_confidence")
    if not hmr_output.get("mesh_vertices_xyz"):
        reasons.append("missing_mesh_vertices")
    if not hmr_output.get("joints3d_xyz"):
        reasons.append("missing_joints3d")
    if not model_inference_ran:
        reasons.append("scaffold_only_no_model_inference")

    return {
        "decision": "reject" if reasons else "allow",
        "confidence": confidence,
        "threshold": threshold,
        "reasons": reasons,
    }


def build_player_hmr_artifact(
    request: PlayerCropRequest,
    hmr_output: Mapping[str, Any],
    *,
    model_inference_ran: bool = False,
    min_confidence: float = 0.65,
) -> dict[str, Any]:
    """Package one per-player deep-tier HMR artifact."""

    gate = gate_deep_hmr_artifact(
        hmr_output,
        model_inference_ran=model_inference_ran,
        min_confidence=min_confidence,
    )
    return {
        "artifact_type": "deep_hmr_player_frame",
        "schema_version": SCHEMA_VERSION,
        "crop_request": request.to_dict(),
        "hmr_output": dict(hmr_output),
        "gate": gate,
        "metadata": {
            "model_family": MODEL_FAMILY,
            "model_inference_ran": bool(model_inference_ran),
            "scaffold": SCAFFOLD_NOTE,
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fast_sam_manifest_assets(
    manifest_path: str | Path = DEFAULT_BODY_MANIFEST_PATH,
    *,
    required_model_ids: Sequence[str] = REQUIRED_FAST_SAM_MODEL_IDS,
) -> dict[str, VerifiedModelAsset]:
    """Verify every checkpoint the BODY runner must load.

    The BODY runner is intentionally strict: missing files and mismatched
    checksums are configuration errors, not soft fallbacks.
    """

    manifest = load_model_manifest(manifest_path)
    entries = {entry.id: entry for entry in manifest.models}
    verified: dict[str, VerifiedModelAsset] = {}
    for model_id in required_model_ids:
        entry = entries.get(model_id)
        if entry is None:
            raise ValueError(f"models manifest is missing required BODY model id: {model_id}")
        if entry.status != "available_on_h100":
            raise ValueError(f"{model_id} is not available_on_h100 in models manifest: {entry.status}")
        if not entry.local_path or not entry.sha256:
            raise ValueError(f"{model_id} manifest entry must include local_path and sha256")
        path = Path(entry.local_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint for {model_id}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != entry.sha256:
            raise ValueError(f"sha256 mismatch for {model_id}: expected {entry.sha256}, got {actual_sha}")
        verified[model_id] = VerifiedModelAsset(model_id=model_id, path=path, sha256=actual_sha)
    return verified


def fast_sam_required_model_ids(*, detector_name: str = "yolo", fov_name: str = "moge2") -> tuple[str, ...]:
    """Return the manifest assets required for the selected runtime components."""

    model_ids = list(CORE_FAST_SAM_MODEL_IDS)
    if fov_name:
        model_ids.append("moge_2_vitl_normal")
    if detector_name:
        model_ids.append("yolo26m")
    return tuple(model_ids)


class FastSam3DBodyRuntime:
    """Thin loader for the external Fast SAM-3D-Body runtime."""

    def __init__(
        self,
        *,
        assets: Mapping[str, VerifiedModelAsset],
        fast_sam_repo: str | Path = DEFAULT_FAST_SAM_REPO,
        detector_name: str = "yolo",
        fov_name: str = "moge2",
        body_input_size_px: int | None = None,
    ) -> None:
        repo = Path(fast_sam_repo)
        if not repo.is_dir():
            raise FileNotFoundError(f"missing FastSAM-3D-Body repo at {repo}")
        self.fast_sam_repo = repo
        self.detector_name = detector_name
        self.body_input_size_px = normalize_body_input_size(body_input_size_px)
        detector_asset = assets.get("yolo26m") if detector_name else None
        if detector_name and detector_asset is None:
            raise ValueError("yolo26m asset is required when detector_name is enabled")
        self.detector_model = detector_asset.path if detector_asset is not None else None
        self.checkpoint_dir = assets["fast_sam_3d_body_dinov3"].path.parent
        self.fov_name = fov_name
        setup_sam_3d_body = _load_setup_sam_3d_body(repo)
        os.environ.setdefault("FAST_SAM_CHECKPOINT_DIR", str(self.checkpoint_dir))
        os.environ.setdefault("SAM3DBODY_CHECKPOINT_DIR", str(self.checkpoint_dir))
        if self.body_input_size_px is not None:
            os.environ["IMG_SIZE"] = str(self.body_input_size_px)
        self.estimator = setup_sam_3d_body(
            hf_repo_id="facebook/sam-3d-body-dinov3",
            detector_name=detector_name,
            detector_model=str(self.detector_model) if self.detector_model is not None else "",
            fov_name=fov_name,
            local_checkpoint_path=str(self.checkpoint_dir),
        )

    def process_frame(
        self,
        image_path: Path,
        *,
        bboxes_xyxy: list[list[float]],
        mask_paths: Sequence[str | Path | None] | None = None,
        camera_intrinsics: Sequence[Sequence[float]] | None = None,
    ) -> list[dict[str, Any]]:
        bbox_arg: Any = bboxes_xyxy
        try:
            import numpy as np  # type: ignore[import-not-found]

            bbox_arg = np.asarray(bboxes_xyxy, dtype=np.float32)
        except ImportError:
            pass
        masks_arg = load_mask_prompt_arrays(mask_paths)
        cam_int_arg = _camera_intrinsics_tensor(camera_intrinsics)

        raw_output = self.estimator.process_one_image(
            str(image_path.resolve()),
            bboxes=bbox_arg,
            masks=masks_arg,
            cam_int=cam_int_arg,
            use_mask=masks_arg is not None,
            hand_box_source="body_decoder",
        )
        records = extract_fast_sam_person_records(raw_output)
        _attach_mesh_faces(records, getattr(self.estimator, "faces", None))
        return records


class FastSam3DBodySubprocessRuntime:
    """Run Fast SAM-3D-Body in an isolated Python runtime and return raw records."""

    def __init__(
        self,
        *,
        python_executable: str | Path,
        fast_sam_repo: str | Path,
        checkpoint_dir: str | Path,
        detector_name: str = "yolo",
        detector_model: str = "",
        fov_name: str = "moge2",
        body_input_size_px: int | None = None,
        work_dir: str | Path,
        script_path: str | Path | None = None,
    ) -> None:
        self.python_executable = Path(python_executable)
        self.fast_sam_repo = Path(fast_sam_repo)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.detector_name = detector_name
        self.detector_model = detector_model
        self.fov_name = fov_name
        self.body_input_size_px = normalize_body_input_size(body_input_size_px)
        self.work_dir = Path(work_dir)
        repo_root = Path(__file__).resolve().parents[2]
        self.script_path = Path(script_path) if script_path is not None else repo_root / "scripts/racketsport/run_sam3dbody_frame.py"

    def process_frame(
        self,
        image_path: Path,
        *,
        bboxes_xyxy: list[list[float]],
        mask_paths: Sequence[str | Path | None] | None = None,
        camera_intrinsics: Sequence[Sequence[float]] | None = None,
    ) -> list[dict[str, Any]]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.work_dir / f"{Path(image_path).stem}-{uuid.uuid4().hex}.json"
        command = [
            str(self.python_executable),
            str(self.script_path),
            "--image",
            str(Path(image_path)),
            "--out",
            str(out_path),
            "--fast-sam-repo",
            str(self.fast_sam_repo),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--detector-model",
            self.detector_model,
            "--detector-name",
            self.detector_name,
            "--fov-name",
            self.fov_name,
        ]
        if self.body_input_size_px is not None:
            command.extend(["--body-input-size", str(self.body_input_size_px)])
        for mask_path in mask_paths or []:
            if mask_path:
                command.extend(["--mask", str(mask_path)])
        if camera_intrinsics is not None:
            command.extend(["--camera-intrinsics", json.dumps(camera_intrinsics, separators=(",", ":"))])
        for bbox in bboxes_xyxy:
            command.extend(["--bbox", ",".join(str(float(value)) for value in bbox)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit={completed.returncode}"
            raise RuntimeError(f"FastSAM-3D-Body subprocess failed: {detail}")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        records = payload.get("records")
        if not isinstance(records, list):
            raise RuntimeError(f"FastSAM-3D-Body subprocess output missing records: {out_path}")
        return [dict(record) for record in records if isinstance(record, Mapping)]

    def process_frame_batches(
        self,
        requests: list[Any],
        *,
        clip_intrinsics: Mapping[str, Any] | None = None,
        sam3d_body_input_size_px: int | None = None,
        crop_bucket_sizes: Sequence[int] = (),
        torch_compile: bool = False,
        compile_warmup_buckets: Sequence[int] = (),
        steady_state_empty_cache: bool = True,
        inner_bucket_sync: bool = True,
        upstream_env: Mapping[str, Any] | None = None,
        tier2_output_lite: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if not requests:
            return []
        self.work_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.work_dir / f"batch_requests-{uuid.uuid4().hex}.json"
        out_path = self.work_dir / f"batch_outputs-{uuid.uuid4().hex}.json"
        normalized_requests = [_normalize_sam3d_runtime_request(request) for request in requests]
        body_input_size_px = normalize_body_input_size(sam3d_body_input_size_px or self.body_input_size_px)
        request_ids = [str(request.get("request_id") or index) for index, request in enumerate(normalized_requests)]
        request_payload = {
            "schema_version": 1,
            "clip_intrinsics": dict(clip_intrinsics) if clip_intrinsics is not None else None,
            "optimization": {
                "sam3d_body_input_size_px": body_input_size_px,
                "crop_bucket_sizes": [int(value) for value in crop_bucket_sizes],
                "torch_compile": bool(torch_compile),
                "compile_warmup_buckets": [int(value) for value in compile_warmup_buckets],
                "steady_state_empty_cache": bool(steady_state_empty_cache),
                "inner_bucket_sync": bool(inner_bucket_sync),
                "upstream_env": dict(upstream_env or {}),
                "tier2_output_lite": bool(tier2_output_lite),
                "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
            },
            "bucket_plan": _sam3d_bucket_plan(request_ids, bucket_sizes=crop_bucket_sizes),
            "requests": [
                {
                    "request_id": request_ids[index],
                    "image": str(request["image_path"]),
                    "bboxes": [[float(value) for value in bbox] for bbox in request["bboxes"]],
                    "mask_paths": [str(path) for path in request.get("mask_paths", []) if path],
                    "camera_intrinsics": request.get("camera_intrinsics"),
                    "sam3d_body_input_size_px": body_input_size_px,
                    "target_representation": request.get("target_representation", "world_mesh"),
                }
                for index, request in enumerate(normalized_requests)
            ],
        }
        request_path.write_text(json.dumps(request_payload, separators=(",", ":")) + "\n", encoding="utf-8")
        command = [
            str(self.python_executable),
            str(Path(__file__).resolve().parents[2] / "scripts/racketsport/run_sam3dbody_batch.py"),
            "--requests",
            str(request_path),
            "--out",
            str(out_path),
            "--fast-sam-repo",
            str(self.fast_sam_repo),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--detector-model",
            self.detector_model,
            "--detector-name",
            self.detector_name,
            "--fov-name",
            self.fov_name,
        ]
        if self.body_input_size_px is not None:
            command.extend(["--body-input-size", str(self.body_input_size_px)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit={completed.returncode}"
            raise RuntimeError(f"FastSAM-3D-Body batch subprocess failed: {detail}")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise RuntimeError(f"FastSAM-3D-Body batch output missing frames: {out_path}")
        by_id = {
            str(frame.get("request_id")): frame.get("records", [])
            for frame in frames
            if isinstance(frame, Mapping)
        }
        outputs: list[list[dict[str, Any]]] = []
        for index, request_id in enumerate(request_ids):
            records = by_id.get(str(request_id), [])
            if not isinstance(records, list):
                raise RuntimeError(f"FastSAM-3D-Body batch records are not a list for request {index}: {out_path}")
            outputs.append([dict(record) for record in records if isinstance(record, Mapping)])
        return outputs


def _camera_intrinsics_tensor(camera_intrinsics: Sequence[Sequence[float]] | None) -> Any | None:
    if camera_intrinsics is None:
        return None
    import torch  # type: ignore[import-not-found]

    rows = [[float(value) for value in row] for row in camera_intrinsics]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("camera_intrinsics must be a 3x3 matrix")
    return torch.tensor([rows], dtype=torch.float32)


def _normalize_sam3d_runtime_request(request: Any) -> dict[str, Any]:
    if isinstance(request, Mapping):
        image_path = request.get("image_path", request.get("image"))
        bboxes = request.get("bboxes")
        if bboxes is None and request.get("bbox") is not None:
            bboxes = [request["bbox"]]
        if image_path is None or bboxes is None:
            raise ValueError("SAM3D runtime request mapping requires image_path/image and bboxes/bbox")
        return {
            "request_id": str(request.get("request_id", "")),
            "image_path": Path(image_path),
            "bboxes": [[float(value) for value in bbox] for bbox in bboxes],
            "mask_paths": [Path(path) for path in request.get("mask_paths", []) if path],
            "camera_intrinsics": request.get("camera_intrinsics"),
            "target_representation": str(request.get("target_representation", "world_mesh")),
        }
    if isinstance(request, tuple) and len(request) == 2:
        image_path, bboxes = request
        return {
            "request_id": "",
            "image_path": Path(image_path),
            "bboxes": [[float(value) for value in bbox] for bbox in bboxes],
            "mask_paths": [],
            "camera_intrinsics": None,
            "target_representation": "world_mesh",
        }
    raise ValueError("SAM3D runtime request must be a mapping or (image_path, bboxes) tuple")


def _sam3d_bucket_plan(request_ids: Sequence[str], *, bucket_sizes: Sequence[int]) -> list[dict[str, Any]]:
    if not request_ids:
        return []
    buckets = sorted({int(size) for size in bucket_sizes if int(size) > 0})
    if not buckets:
        return [{"bucket_size": len(request_ids), "request_ids": list(request_ids), "padded_request_count": len(request_ids)}]
    plan: list[dict[str, Any]] = []
    pending = list(request_ids)
    while pending:
        remaining = len(pending)
        bucket_size = next((size for size in buckets if size >= remaining), buckets[-1])
        take = min(remaining, bucket_size)
        group = pending[:take]
        del pending[:take]
        plan.append(
            {
                "bucket_size": int(bucket_size),
                "request_ids": group,
                "padded_request_count": int(bucket_size),
            }
        )
    return plan


def extract_fast_sam_person_records(raw_output: Any) -> list[dict[str, Any]]:
    if isinstance(raw_output, Mapping):
        for key in ("people", "persons", "person_results", "humans", "predictions", "outputs", "results"):
            if key in raw_output:
                records = _coerce_person_records(raw_output[key])
                if records:
                    return records
        public_mapping = _as_public_mapping(raw_output)
        if _looks_like_fast_sam_person(public_mapping):
            return [public_mapping]
        return []
    return _coerce_person_records(raw_output)


def _attach_mesh_faces(records: list[dict[str, Any]], faces: Any) -> None:
    if faces is None:
        return
    face_rows = _to_python_container(faces)
    if not isinstance(face_rows, Sequence) or isinstance(face_rows, str | bytes | bytearray):
        return
    for record in records:
        if "mesh_faces" not in record and "faces" not in record:
            record["mesh_faces"] = face_rows


def normalize_fast_sam_body_output(
    person: Mapping[str, Any],
    *,
    request: PlayerCropRequest,
) -> dict[str, Any]:
    """Normalize one real Fast SAM-3D-Body person output for world grounding."""

    public = _as_public_mapping(person)
    if not public:
        raise ValueError("Fast SAM-3D-Body person output must be a mapping-like object")

    joints = _vector3_list(
        _first_present(public, ("pred_keypoints_3d", "pred_joint_coords", "joints3d", "joints3d_xyz")),
        name="pred_keypoints_3d",
    )
    vertices = _vector3_list(
        _first_present(public, ("pred_vertices", "vertices", "mesh_vertices_xyz")),
        name="pred_vertices",
    )
    if not joints and not vertices:
        raise ValueError("Fast SAM-3D-Body output is missing pred_keypoints_3d/pred_vertices")

    confidence = _optional_confidence(
        _first_present(public, ("confidence", "score", "model_confidence")),
        name="confidence",
    )
    if confidence is None:
        confidence = request.track_confidence

    hand_pose = _float_list(_first_present(public, ("hand_pose_params", "hand_pose"), default=[]), name="hand_pose")
    left_hand_pose, right_hand_pose = _split_hands(hand_pose)
    camera_translation = _float_vector(
        _first_present(public, ("pred_cam_t", "camera_translation", "transl"), default=[0.0, 0.0, 0.0]),
        name="camera_translation",
        length=3,
    )
    pred_cam_t_already_applied = _bool_flag(
        _first_present(
            public,
            ("pred_cam_t_already_applied", "camera_translation_already_applied", "translation_already_applied"),
            default=False,
        ),
        name="pred_cam_t_already_applied",
    )
    return {
        "frame_idx": request.frame_idx,
        "player_id": request.player_id,
        "bbox_xyxy": list(request.bbox_xyxy),
        "track_confidence": request.track_confidence,
        "confidence": min(confidence, request.track_confidence),
        "global_orient": _float_vector(
            _first_present(public, ("global_rot", "global_orient", "pred_global_orient"), default=[0.0, 0.0, 0.0]),
            name="global_orient",
            length=3,
        ),
        "body_pose": _float_list(_first_present(public, ("body_pose_params", "body_pose", "pred_pose_raw"), default=[]), name="body_pose"),
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "betas": _float_list(_first_present(public, ("shape_params", "betas"), default=[]), name="betas"),
        # ADDITIVE (P2-2 GATE 1b, w5_p22latent_20260707): the raw estimator
        # output carries `scale_params` (28-dim MHR scale/bone-length
        # correction; see sam_3d_body_estimator.py's `"scale_params":
        # out["scale"][idx]`) but it was previously dropped here entirely --
        # no downstream consumer ever read it. Without it, a decode-time
        # re-synthesis of a frame's mesh/joints from persisted
        # global_orient/body_pose/betas alone defaults to population-mean
        # scale and misses real per-athlete bone-length variation by
        # 100-220mm world-position error (measured live on wolverine, see
        # the lane report GATE 1b evidence) -- far above the <=1mm
        # round-trip fidelity bar. New field, no removals; empty list is a
        # safe/back-compatible default for any existing consumer.
        "scale": _float_list(_first_present(public, ("scale_params", "scale"), default=[]), name="scale"),
        "camera_translation": camera_translation,
        "pred_cam_t_already_applied": pred_cam_t_already_applied,
        "joints_camera": mhr_decode.apply_pred_cam_t_once(
            joints,
            pred_cam_t=camera_translation,
            already_applied=pred_cam_t_already_applied,
        ),
        "vertices_camera": mhr_decode.apply_pred_cam_t_once(
            vertices,
            pred_cam_t=camera_translation,
            already_applied=pred_cam_t_already_applied,
        ),
        "mesh_faces": _face_list(
            _first_present(public, ("mesh_faces", "faces", "pred_faces", "triangles", "mesh_triangles"), default=[]),
            vertex_count=len(vertices),
            name="mesh_faces",
            vertices_name="pred_vertices",
        ),
        "pred_foot_keypoints_2d": _compact_foot_keypoints_2d(
            _first_present(public, ("pred_keypoints_2d",), default=None),
            confidence=min(confidence, request.track_confidence),
        ),
        "model_family": MODEL_FAMILY,
    }


def _image_size(values: Sequence[int]) -> tuple[int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("image_size_px must be a 2-vector")
    width, height = values
    width_int = _positive_int(width, name="image_size_px/0")
    height_int = _positive_int(height, name="image_size_px/1")
    return (width_int, height_int)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value_int


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _float_vector(values: Any, *, name: str, length: int) -> list[float]:
    result = _float_list(values, name=name)
    if len(result) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return result


def _vector3_list(values: Any, *, name: str) -> list[list[float]]:
    values = _to_python_container(values)
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of 3-vectors")
    return [
        _float_vector(vector, name=f"{name}/{idx}", length=3)
        for idx, vector in enumerate(values)
    ]


def _float_list(values: Any, *, name: str) -> list[float]:
    values = _to_python_container(values)
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")

    result: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{name}/{idx} must be finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}/{idx} must be finite") from exc
        if not isfinite(number):
            raise ValueError(f"{name}/{idx} must be finite")
        result.append(number)
    return result


def _face_list(values: Any, *, vertex_count: int, name: str, vertices_name: str) -> list[list[int]]:
    values = _to_python_container(values)
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of triangle index triples")
    faces: list[list[int]] = []
    for face_index, face in enumerate(values):
        face = _to_python_container(face)
        if isinstance(face, (str, bytes)) or not isinstance(face, Sequence) or len(face) != 3:
            raise ValueError(f"{name}/{face_index} must be a triangle index triple")
        parsed_face: list[int] = []
        for raw_index in face:
            raw_index = _to_python_container(raw_index)
            if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                raise ValueError(f"{name}/{face_index} must be a triangle index triple")
            index = int(raw_index)
            if index < 0:
                raise ValueError(f"{name}/{face_index} must be a triangle index triple")
            if index >= vertex_count:
                raise ValueError(f"{name}/{face_index} index {index} is outside {vertices_name}")
            parsed_face.append(index)
        faces.append(parsed_face)
    return faces


def _compact_foot_keypoints_2d(values: Any, *, confidence: float) -> list[dict[str, Any]]:
    values = _to_python_container(values)
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("pred_keypoints_2d must be a sequence of 2-vectors")
    out: list[dict[str, Any]] = []
    for name, index in SAM3D_FOOT_KEYPOINT_INDICES.items():
        if index >= len(values):
            continue
        xy = _float_vector(values[index], name=f"pred_keypoints_2d/{index}", length=2)
        out.append({"name": name, "index": index, "xy_px": xy, "conf": confidence})
    return out


def _confidence(value: Any, *, name: str) -> float:
    value = _to_python_container(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if confidence < 0.0 or confidence > 1.0 or not isfinite(confidence):
        raise ValueError(f"{name} must be between 0 and 1")
    return confidence


def _optional_confidence(value: Any, *, name: str) -> float | None:
    value = _to_python_container(value)
    if value is None:
        return None
    return _confidence(value, name=name)


def _load_setup_sam_3d_body(fast_sam_repo: Path) -> Any:
    repo = str(fast_sam_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        module = importlib.import_module("notebook.utils")
    except Exception as notebook_exc:
        try:
            importlib.import_module("sam_3d_body")
        except Exception as direct_exc:
            raise RuntimeError(
                "could not import FastSAM-3D-Body notebook.utils.setup_sam_3d_body "
                "or direct sam_3d_body runtime "
                f"(notebook_error={type(notebook_exc).__name__}: {notebook_exc}; "
                f"direct_error={type(direct_exc).__name__}: {direct_exc})"
            ) from direct_exc
        return _direct_setup_sam_3d_body
    setup_sam_3d_body = getattr(module, "setup_sam_3d_body", None)
    if not callable(setup_sam_3d_body):
        raise RuntimeError("FastSAM-3D-Body notebook.utils does not expose callable setup_sam_3d_body")
    return setup_sam_3d_body


def _direct_setup_sam_3d_body(
    *,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    detector_name: str = "",
    detector_model: str = "",
    fov_name: str = "",
    device: str = "cuda",
    local_checkpoint_path: str = "",
    local_mhr_path: str = "",
) -> Any:
    import torch  # type: ignore[import-not-found]

    _ensure_torch_amp_custom_decorators(torch)
    _ensure_torch_dynamo_accumulated_cache_limit(torch)
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf

    resolved_device = device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    if local_checkpoint_path:
        checkpoint_dir = Path(local_checkpoint_path)
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(checkpoint_dir / "model.ckpt"),
            device=resolved_device,
            mhr_path=local_mhr_path or str(checkpoint_dir / "assets" / "mhr_model.pt"),
        )
    else:
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=resolved_device)

    human_detector = None
    if detector_name:
        from tools.build_detector import HumanDetector

        detector_kwargs: dict[str, Any] = {}
        if detector_model and detector_name in ("yolo", "yolo11", "yolo_pose"):
            detector_kwargs["model"] = detector_model
        human_detector = HumanDetector(name=detector_name, device=resolved_device, **detector_kwargs)

    fov_estimator = None
    if fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=resolved_device)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )


def _ensure_torch_amp_custom_decorators(torch_module: Any) -> None:
    """Expose torch.amp custom decorators for DINOv3 on older CUDA Torch builds."""

    amp = getattr(torch_module, "amp", None)
    cuda = getattr(torch_module, "cuda", None)
    cuda_amp = getattr(cuda, "amp", None)
    if amp is None or cuda_amp is None:
        return
    if not hasattr(amp, "custom_fwd") and hasattr(cuda_amp, "custom_fwd"):

        def custom_fwd(fwd: Callable | None = None, *, device_type: str | None = None, cast_inputs: Any = None) -> Any:
            del device_type
            return cuda_amp.custom_fwd(fwd, cast_inputs=cast_inputs)

        amp.custom_fwd = custom_fwd
    if not hasattr(amp, "custom_bwd") and hasattr(cuda_amp, "custom_bwd"):

        def custom_bwd(bwd: Callable | None = None, *, device_type: str | None = None) -> Any:
            del device_type
            if bwd is None:
                return lambda fn: cuda_amp.custom_bwd(fn)
            return cuda_amp.custom_bwd(bwd)

        amp.custom_bwd = custom_bwd


def _ensure_torch_dynamo_accumulated_cache_limit(torch_module: Any) -> None:
    """Allow DINOv3 to set a newer torch._dynamo cache knob on Torch 2.1."""

    dynamo = getattr(torch_module, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is None or hasattr(config, "accumulated_cache_size_limit"):
        return
    default_value = getattr(config, "cache_size_limit", 64)
    config_values = getattr(config, "_config", None)
    if isinstance(config_values, dict):
        config_values.setdefault("accumulated_cache_size_limit", default_value)
    default_values = getattr(config, "_default", None)
    if isinstance(default_values, dict):
        default_values.setdefault("accumulated_cache_size_limit", default_value)
    allowed_keys = getattr(config, "_allowed_keys", None)
    if isinstance(allowed_keys, set):
        allowed_keys.add("accumulated_cache_size_limit")


def _coerce_person_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        values = [value[key] for key in sorted(value, key=str)]
        return [_as_public_mapping(item) for item in values if _as_public_mapping(item)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_as_public_mapping(item) for item in value if _as_public_mapping(item)]
    return []


def _as_public_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    public_attrs = getattr(value, "__dict__", None)
    if isinstance(public_attrs, Mapping):
        return {
            str(key): item
            for key, item in public_attrs.items()
            if not str(key).startswith("_") and not callable(item)
        }
    return {}


def _looks_like_fast_sam_person(value: Mapping[str, Any]) -> bool:
    keys = {str(key) for key in value}
    return bool(keys.intersection({"pred_vertices", "pred_keypoints_3d", "pred_joint_coords", "body_pose_params"}))


def _first_present(public: Mapping[str, Any], keys: Sequence[str], *, default: Any = None) -> Any:
    for key in keys:
        if key in public and public[key] is not None:
            return public[key]
    return default


def _bool_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{name} must be a boolean flag")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ValueError(f"{name} must be a boolean flag")


def _split_hands(hand_pose: list[float]) -> tuple[list[float], list[float]]:
    if not hand_pose:
        return [], []
    midpoint = len(hand_pose) // 2
    return hand_pose[:midpoint], hand_pose[midpoint:]


def _to_python_container(value: Any) -> Any:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                item = method()
            except Exception:
                return value
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return value
