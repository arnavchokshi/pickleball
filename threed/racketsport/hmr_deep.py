"""CPU-only deep-tier HMR scaffold primitives.

This module validates and packages Fast SAM-3D-Body/MHR-style inputs and
outputs. It intentionally does not download checkpoints, run inference, select
variants, or touch the GPU.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

from .model_manifest import load_model_manifest


SCAFFOLD_NOTE = "cpu_hmr_deep_primitives_no_model_inference"
SCHEMA_VERSION = "body_hmr_deep.v0"
MODEL_FAMILY = "fast_sam_3d_body_mhr_to_smpl"
DEFAULT_BODY_MANIFEST_PATH = Path("models/MANIFEST.json")
DEFAULT_FAST_SAM_REPO = Path(os.environ.get("FAST_SAM_ROOT", "/opt/fast-sam-3d-body"))
REQUIRED_FAST_SAM_MODEL_IDS = (
    "fast_sam_3d_body_dinov3",
    "sam_3d_body_mhr_model",
    "moge_2_vitl_normal",
    "yolo26m",
)


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


class FastSam3DBodyRuntime:
    """Thin loader for the external Fast SAM-3D-Body runtime."""

    def __init__(
        self,
        *,
        assets: Mapping[str, VerifiedModelAsset],
        fast_sam_repo: str | Path = DEFAULT_FAST_SAM_REPO,
        detector_name: str = "yolo",
        fov_name: str = "moge2",
    ) -> None:
        repo = Path(fast_sam_repo)
        if not repo.is_dir():
            raise FileNotFoundError(f"missing FastSAM-3D-Body repo at {repo}")
        self.fast_sam_repo = repo
        self.detector_name = detector_name
        self.detector_model = assets["yolo26m"].path
        self.checkpoint_dir = assets["fast_sam_3d_body_dinov3"].path.parent
        self.fov_name = fov_name
        setup_sam_3d_body = _load_setup_sam_3d_body(repo)
        os.environ.setdefault("FAST_SAM_CHECKPOINT_DIR", str(self.checkpoint_dir))
        os.environ.setdefault("SAM3DBODY_CHECKPOINT_DIR", str(self.checkpoint_dir))
        self.estimator = setup_sam_3d_body(
            hf_repo_id="facebook/sam-3d-body-dinov3",
            detector_name=detector_name,
            detector_model=str(self.detector_model),
            fov_name=fov_name,
            local_checkpoint_path=str(self.checkpoint_dir),
        )

    def process_frame(self, image_path: Path, *, bboxes_xyxy: list[list[float]]) -> list[dict[str, Any]]:
        bbox_arg: Any = bboxes_xyxy
        try:
            import numpy as np  # type: ignore[import-not-found]

            bbox_arg = np.asarray(bboxes_xyxy, dtype=np.float32)
        except ImportError:
            pass

        raw_output = self.estimator.process_one_image(
            str(image_path.resolve()),
            bboxes=bbox_arg,
            use_mask=False,
            hand_box_source="body_decoder",
        )
        return extract_fast_sam_person_records(raw_output)


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
        "camera_translation": _float_vector(
            _first_present(public, ("pred_cam_t", "camera_translation", "transl"), default=[0.0, 0.0, 0.0]),
            name="camera_translation",
            length=3,
        ),
        "joints_camera": joints,
        "vertices_camera": vertices,
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
    except Exception as exc:
        raise RuntimeError("could not import FastSAM-3D-Body notebook.utils.setup_sam_3d_body") from exc
    setup_sam_3d_body = getattr(module, "setup_sam_3d_body", None)
    if not callable(setup_sam_3d_body):
        raise RuntimeError("FastSAM-3D-Body notebook.utils does not expose callable setup_sam_3d_body")
    return setup_sam_3d_body


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
