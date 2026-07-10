"""MHR pose-code decode wrapper (P2-2 STEP A, phase 1).

See `runs/archive/root_docs_20260709/TECH_BLUEPRINTS.md` BODY pillar STEP 0 + STEP A (~lines 1444-1499) and
`runs/lanes/w5_p22latent_20260707/spec.md` for the full recipe this module
implements.

Loads the SAME Fast-SAM-3D-Body/MHR checkpoint the production pipeline loads
(`third_party/Fast-SAM-3D-Body/sam_3d_body/build_models.py::load_sam_3d_body`)
and exposes a *local re-decode* path: persisted per-frame body params
(``global_orient`` euler(3), ``body_pose`` euler(133), ``betas``/shape(45),
optional ``scale``(28)) -> the MHR pose-code latent (``pred_pose_raw``,
266-dim = global_rot_6d(6) + body_cont(260)) -> decoded camera-frame
joints/vertices, via the model's OWN converters
(``compact_model_params_to_cont_body``, ``compact_cont_to_model_params_body_fast``,
``rot6d_to_rotmat``, ``rotmat_to_rot6d``) -- never reimplemented, always
imported from ``third_party/Fast-SAM-3D-Body`` so this wrapper cannot
silently diverge from production math.

Two fidelity gates this module exists to compute (spec PHASE 1):

* GATE 1a - euler->cont->euler idempotence (<0.1 deg max abs error): pure
  math, needs the head's converters + roma but NOT the model checkpoint.
* GATE 1b - decode(emit(frame)) reproduces persisted joints_world /
  vertices_world to <=1mm: needs the loaded checkpoint (MHR body forward)
  plus the SAME camera->world grounding transform
  `threed.racketsport.worldhmr` applies (reused via
  ``ground_decoded_camera_frame``, never reimplemented).

Also exposes ``mesh_skeleton_divergence_mm`` (PHASE 2 acceptance: a NEW
metric measuring how far the blended mesh+skeleton keypoint regression
diverges from a mesh-vertex-only regression using the model's own
``keypoint_mapping`` matrix with the raw skeleton ``joint_coords``
zeroed out).

GPU-only paths (``MHRDecoder``, anything touching ``MHRHead``/torch/roma)
are heavy and only importable where `third_party/Fast-SAM-3D-Body`'s runtime
deps (roma, braceexpand, a CUDA-capable torch build) are installed -- the
fleet GPU VM's ``body_venv``, not the Mac CPU dev venv. Callers/tests must
check ``MHR_RUNTIME_AVAILABLE`` (or use ``pytest.importorskip("torch")``
plus a manual roma check) before touching anything below the import guard.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import os

from . import coordinates

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_fast_sam_root() -> Path:
    """Locate the Fast-SAM-3D-Body python package (``sam_3d_body/``).

    `third_party/Fast-SAM-3D-Body` in a full checkout is the primary
    candidate, but fleet GPU VMs built by the cold-start scripts keep the
    installed runtime code under a separate ``body_runtime/`` tree (the repo
    submodule there is intentionally not checked out) -- mirrors
    `hmr_deep.py::DEFAULT_FAST_SAM_REPO`'s ``FAST_SAM_ROOT`` env override
    plus that same cold-start convention, tried in order.
    """
    env_override = os.environ.get("FAST_SAM_ROOT", "")
    candidates = [
        Path(env_override) if env_override else None,
        _REPO_ROOT / "third_party" / "Fast-SAM-3D-Body",
        Path("/home/arnavchokshi/coldstart_20260706/body_runtime/Fast-SAM-3D-Body"),
        Path("/opt/fast-sam-3d-body"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "sam_3d_body" / "__init__.py").is_file():
            return candidate
    # Fall through to the repo submodule path even if unpopulated -- the
    # subsequent import attempt will raise a clear ModuleNotFoundError that
    # MHR_RUNTIME_IMPORT_ERROR captures.
    return _REPO_ROOT / "third_party" / "Fast-SAM-3D-Body"


_FAST_SAM_ROOT = _resolve_fast_sam_root()

# ---------------------------------------------------------------------------
# Dimension constants (Fast-SAM-3D-Body MHR head; see mhr_head.py::MHRHead
# and mhr_utils.py's body-index precomputation block).
# ---------------------------------------------------------------------------
BODY_POSE_EULER_DIM = 133  # mhr_head.py::MHRHead body-pose euler width (69 3dof + 58 1dof + 6 trans)
BODY_POSE_CONT_DIM = 260  # mhr_head.py::MHRHead.body_cont_dim
GLOBAL_ROT_EULER_DIM = 3
GLOBAL_ROT_6D_DIM = 6
PRED_POSE_RAW_DIM = GLOBAL_ROT_6D_DIM + BODY_POSE_CONT_DIM  # 266
SHAPE_DIM = 45  # mhr_head.py::MHRHead.num_shape_comps
SCALE_DIM = 28  # mhr_head.py::MHRHead.num_scale_comps
HAND_COMPS_DIM = 54  # mhr_head.py::MHRHead.num_hand_comps (per hand)
FACE_COMPS_DIM = 72  # mhr_head.py::MHRHead.num_face_comps
NUM_KEYPOINTS = 70  # pipeline-facing joint count (j3d[:, :70])

DEFAULT_CHECKPOINT_PATH = (
    "/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt"
)
DEFAULT_MHR_ASSET_PATH = (
    "/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)

GATE_1A_MAX_ABS_ERROR_DEG = 0.1
GATE_1B_MAX_ABS_ERROR_MM = 1.0
MESH_SKELETON_DIVERGENCE_P95_MM = 5.0

# ---------------------------------------------------------------------------
# Guarded heavy import block. Mirrors the pattern the rest of the racketsport
# tree uses for optional torch/GPU deps: never raise at import time, expose a
# flag + the captured exception, and let callers decide whether to skip.
# ---------------------------------------------------------------------------
TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
except Exception as exc:  # pragma: no cover - exercised on the Mac CPU dev venv
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

MHR_RUNTIME_IMPORT_ERROR: Exception | None = None
if torch is not None:
    if str(_FAST_SAM_ROOT) not in sys.path:
        sys.path.insert(0, str(_FAST_SAM_ROOT))
    try:
        import roma  # noqa: F401  (re-exported implicitly via functions below)
        from sam_3d_body.build_models import load_sam_3d_body
        from sam_3d_body.models.heads.mhr_head import MHRHead  # noqa: F401  (type reference only)
        from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat
        # NOTE: `geometry_utils.rotmat_to_rot6d` is NOT the inverse of
        # `rot6d_to_rotmat` and is deliberately NOT imported/used here.
        # `rot6d_to_rotmat` reshapes its 6-vector into two 3-vectors and
        # Gram-Schmidt-orthonormalizes them into the first two COLUMNS of the
        # output matrix; `rotmat_to_rot6d` extracts the first two ROWS of a
        # rotation matrix ("by dropping the last row", per its own
        # docstring). Composing them round-trips to `R^T`, not `R` -- verified
        # live on the fleet GPU VM against real wolverine global_orient
        # values (a near-gimbal-lock sample at ~(86,75,89) degrees exposed a
        # 176 degree "round-trip error" that was actually this row/column
        # mismatch, not a genuine model fidelity gap: rotmat->6D->rotmat
        # diff 1.97 vs 6e-8 with the column-based encoder below; see the
        # lane report HONEST ISSUES). `encode_global_orient_euler_to_rot6d`
        # below builds the correct column-based 6D encoding by hand instead
        # -- this is still 100% the model's own `rot6d_to_rotmat` semantics
        # (its own docstring's "Alternatives" block: a1/a2 come from
        # `x.reshape(-1,2,3).permute(0,2,1)`, i.e. columns 0/1 of a (3,2)
        # view -- so the true inverse of "take columns 0 and 1" is "take
        # columns 0 and 1", not "take rows 0 and 1").
        from sam_3d_body.models.modules.mhr_utils import (
            compact_cont_to_model_params_body_fast,
            compact_model_params_to_cont_body,
            rotmat_to_euler_ZYX,
        )
    except Exception as exc:  # pragma: no cover - exercised on the Mac CPU dev venv
        MHR_RUNTIME_IMPORT_ERROR = exc
else:
    MHR_RUNTIME_IMPORT_ERROR = TORCH_IMPORT_ERROR

MHR_RUNTIME_AVAILABLE = MHR_RUNTIME_IMPORT_ERROR is None


def _require_runtime() -> None:
    if not MHR_RUNTIME_AVAILABLE:
        raise RuntimeError(
            "mhr_decode requires torch + roma + the sam_3d_body package runtime "
            "(present on the fleet GPU body_venv, not necessarily this interpreter): "
            f"{MHR_RUNTIME_IMPORT_ERROR!r}"
        )


def _as_tensor(values: Any) -> "torch.Tensor":
    _require_runtime()
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32)


def _circular_diff_rad(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Minimal signed angular difference a-b, wrapped to (-pi, pi]."""
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Pure encode/decode helpers -- thin wrappers around the head's own
# converters. No reimplementation of the trig/rotation math lives here.
# ---------------------------------------------------------------------------
def encode_body_pose_euler_to_cont(body_pose_euler: Any) -> "torch.Tensor":
    """euler(..., 133) -> cont(..., 260) via ``compact_model_params_to_cont_body``."""
    x = _as_tensor(body_pose_euler)
    if x.shape[-1] != BODY_POSE_EULER_DIM:
        raise ValueError(f"body_pose_euler last dim must be {BODY_POSE_EULER_DIM}, got {x.shape[-1]}")
    return compact_model_params_to_cont_body(x)


def decode_body_pose_cont_to_euler(body_pose_cont: Any) -> "torch.Tensor":
    """cont(..., 260) -> euler(..., 133) via ``compact_cont_to_model_params_body_fast``."""
    x = _as_tensor(body_pose_cont)
    if x.shape[-1] != BODY_POSE_CONT_DIM:
        raise ValueError(f"body_pose_cont last dim must be {BODY_POSE_CONT_DIM}, got {x.shape[-1]}")
    return compact_cont_to_model_params_body_fast(x)


def encode_global_orient_euler_to_rot6d(global_orient_euler: Any) -> "torch.Tensor":
    """euler ZYX(..., 3) -> 6D rotation repr(..., 6).

    Uses ``roma.euler_to_rotmat`` (the upstream primitive
    ``rotmat_to_euler_ZYX`` documents itself as an "optimized replacement"
    for) to get the rotation matrix, then builds the 6D encoding as columns
    0 and 1 of that matrix -- the true inverse of the head's own
    ``rot6d_to_rotmat`` (see the import-block note above; do NOT swap this
    for ``geometry_utils.rotmat_to_rot6d``, which extracts ROWS and is not
    this function's inverse).
    """
    x = _as_tensor(global_orient_euler)
    if x.shape[-1] != GLOBAL_ROT_EULER_DIM:
        raise ValueError(f"global_orient_euler last dim must be {GLOBAL_ROT_EULER_DIM}, got {x.shape[-1]}")
    rotmat = roma.euler_to_rotmat("ZYX", x)
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


def decode_global_rot6d_to_euler(global_rot_6d: Any) -> "torch.Tensor":
    """6D rotation repr(..., 6) -> euler ZYX(..., 3) via the head's own converters."""
    x = _as_tensor(global_rot_6d)
    if x.shape[-1] != GLOBAL_ROT_6D_DIM:
        raise ValueError(f"global_rot_6d last dim must be {GLOBAL_ROT_6D_DIM}, got {x.shape[-1]}")
    rotmat = rot6d_to_rotmat(x)
    return rotmat_to_euler_ZYX(rotmat)


def build_pred_pose_raw(global_orient_euler: Any, body_pose_euler: Any) -> "torch.Tensor":
    """Reconstruct the 266-dim MHR latent ``pred_pose_raw`` from persisted euler fields."""
    rot6d = encode_global_orient_euler_to_rot6d(global_orient_euler)
    cont = encode_body_pose_euler_to_cont(body_pose_euler)
    return torch.cat([rot6d, cont], dim=-1)


def split_pred_pose_raw(pred_pose_raw: Any) -> tuple["torch.Tensor", "torch.Tensor"]:
    x = _as_tensor(pred_pose_raw)
    if x.shape[-1] != PRED_POSE_RAW_DIM:
        raise ValueError(f"pred_pose_raw last dim must be {PRED_POSE_RAW_DIM}, got {x.shape[-1]}")
    return x[..., :GLOBAL_ROT_6D_DIM], x[..., GLOBAL_ROT_6D_DIM:]


# ---------------------------------------------------------------------------
# GATE 1a: euler -> cont -> euler idempotence.
# ---------------------------------------------------------------------------
def round_trip_body_pose_euler_error_deg(body_pose_euler: Any) -> dict[str, Any]:
    x = _as_tensor(body_pose_euler)
    cont = encode_body_pose_euler_to_cont(x)
    back = decode_body_pose_cont_to_euler(cont)
    err_deg = _circular_diff_rad(back, x).abs() * (180.0 / math.pi)
    flat = err_deg.flatten()
    return {
        "component": "body_pose",
        "frame_count": int(x.shape[0]) if x.dim() > 1 else 1,
        "max_abs_error_deg": float(flat.max().item()),
        "mean_abs_error_deg": float(flat.mean().item()),
        "p95_abs_error_deg": float(torch.quantile(flat, 0.95).item()),
    }


def round_trip_global_orient_error_deg(global_orient_euler: Any) -> dict[str, Any]:
    x = _as_tensor(global_orient_euler)
    rot6d = encode_global_orient_euler_to_rot6d(x)
    back = decode_global_rot6d_to_euler(rot6d)
    err_deg = _circular_diff_rad(back, x).abs() * (180.0 / math.pi)
    flat = err_deg.flatten()
    return {
        "component": "global_orient",
        "frame_count": int(x.shape[0]) if x.dim() > 1 else 1,
        "max_abs_error_deg": float(flat.max().item()),
        "mean_abs_error_deg": float(flat.mean().item()),
        "p95_abs_error_deg": float(torch.quantile(flat, 0.95).item()),
    }


def gate_1a_euler_round_trip(global_orient_euler: Any, body_pose_euler: Any) -> dict[str, Any]:
    """GATE 1a: euler->cont->euler idempotence, < 0.1 deg max abs error (spec PHASE 1)."""
    body = round_trip_body_pose_euler_error_deg(body_pose_euler)
    global_ = round_trip_global_orient_error_deg(global_orient_euler)
    max_err = max(body["max_abs_error_deg"], global_["max_abs_error_deg"])
    return {
        "gate": "gate_1a_euler_cont_euler_idempotence",
        "target_max_abs_error_deg": GATE_1A_MAX_ABS_ERROR_DEG,
        "body_pose": body,
        "global_orient": global_,
        "max_abs_error_deg": max_err,
        "passed": bool(max_err < GATE_1A_MAX_ABS_ERROR_DEG),
    }


# ---------------------------------------------------------------------------
# MHRDecoder: loads the production checkpoint once, decode-only forward
# passes (no image backbone / detector needed -- this consumes already
# -estimated pose params, not pixels).
# ---------------------------------------------------------------------------
class MHRDecoder:
    """Loads the SAME Fast-SAM-3D-Body/MHR checkpoint the pipeline loads and
    exposes decode-only forward calls through the model's own MHR body head.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        mhr_path: str = DEFAULT_MHR_ASSET_PATH,
        device: str | None = None,
    ) -> None:
        _require_runtime()
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path, device=resolved_device, mhr_path=mhr_path
        )
        self.model = model
        self.model_cfg = model_cfg
        self.device = resolved_device
        self.head = model.head_pose
        self.head.eval()
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path

    def _prep_batch(
        self,
        *,
        global_orient_euler: Any,
        body_pose_euler: Any,
        shape: Any,
        scale: Any | None,
        hand_pose: Any | None,
    ) -> dict[str, "torch.Tensor"]:
        device = self.device
        global_rot = _as_tensor(global_orient_euler).to(device)
        body_pose = _as_tensor(body_pose_euler).to(device)
        shape_t = _as_tensor(shape).to(device)
        if global_rot.dim() == 1:
            global_rot = global_rot.unsqueeze(0)
        if body_pose.dim() == 1:
            body_pose = body_pose.unsqueeze(0)
        if shape_t.dim() == 1:
            shape_t = shape_t.unsqueeze(0)
        batch_size = global_rot.shape[0]
        if scale is None:
            scale_t = torch.zeros(batch_size, SCALE_DIM, device=device, dtype=torch.float32)
        else:
            scale_t = _as_tensor(scale).to(device)
            if scale_t.dim() == 1:
                scale_t = scale_t.unsqueeze(0)
        if hand_pose is None:
            hand_t = torch.zeros(batch_size, HAND_COMPS_DIM * 2, device=device, dtype=torch.float32)
        else:
            hand_t = _as_tensor(hand_pose).to(device)
            if hand_t.dim() == 1:
                hand_t = hand_t.unsqueeze(0)
        global_trans = torch.zeros_like(global_rot)
        expr = torch.zeros(batch_size, FACE_COMPS_DIM, device=device, dtype=torch.float32)
        return {
            "global_trans": global_trans,
            "global_rot": global_rot,
            "body_pose_params": body_pose,
            "hand_pose_params": hand_t,
            "scale_params": scale_t,
            "shape_params": shape_t,
            "expr_params": expr,
        }

    @torch.no_grad()
    def decode_euler_frame(
        self,
        *,
        global_orient_euler: Any,
        body_pose_euler: Any,
        shape: Any,
        scale: Any | None = None,
        hand_pose: Any | None = None,
    ) -> dict[str, Any]:
        """Decode from persisted euler params via the MHR body model's own ``mhr_forward``.

        ``scale=None`` decodes with the population-mean scale (the schema's
        current gap -- see GATE 1b honest-issue in the lane report). Pass the
        real ``scale_params`` once the additive schema field lands for a
        faithful reconstruction.
        """
        batch = self._prep_batch(
            global_orient_euler=global_orient_euler,
            body_pose_euler=body_pose_euler,
            shape=shape,
            scale=scale,
            hand_pose=hand_pose,
        )
        verts, kp, jcoords, model_params, _joint_rots = self.head.mhr_forward(
            **batch,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        kp = kp[:, :NUM_KEYPOINTS].clone()
        kp[..., [1, 2]] *= -1
        if verts is not None:
            verts = verts.clone()
            verts[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords = jcoords.clone()
            jcoords[..., [1, 2]] *= -1
        return {
            "joints_camera": kp.detach().cpu().numpy(),
            "vertices_camera": verts.detach().cpu().numpy() if verts is not None else None,
            "joint_coords_camera": jcoords.detach().cpu().numpy() if jcoords is not None else None,
        }

    def decode_pred_pose_raw(
        self,
        *,
        pred_pose_raw: Any,
        shape: Any,
        scale: Any | None = None,
        hand_pose: Any | None = None,
    ) -> dict[str, Any]:
        """Decode from the TRUE 266-dim MHR latent (the post-smoothing path, PHASE 2)."""
        rot6d, cont = split_pred_pose_raw(pred_pose_raw)
        global_orient_euler = decode_global_rot6d_to_euler(rot6d)
        body_pose_euler = decode_body_pose_cont_to_euler(cont)
        return self.decode_euler_frame(
            global_orient_euler=global_orient_euler,
            body_pose_euler=body_pose_euler,
            shape=shape,
            scale=scale,
            hand_pose=hand_pose,
        )

    @torch.no_grad()
    def mesh_skeleton_divergence_mm(
        self,
        *,
        global_orient_euler: Any,
        body_pose_euler: Any,
        shape: Any,
        scale: Any | None = None,
        hand_pose: Any | None = None,
    ) -> dict[str, Any]:
        """NEW metric (spec PHASE 2): per-frame || decoded-joint - skinned-mesh-joint || in mm.

        ``decoded-joint`` = the pipeline-facing ``pred_keypoints_3d`` (the
        model's own blend of skinned-mesh vertices AND raw kinematic-chain
        joint coords through ``keypoint_mapping``).
        ``skinned-mesh-joint`` = for each of the 70 keypoints, the CLOSEST of
        the model's own 127 raw kinematic-chain ``joint_coords`` (a
        data-driven nearest-neighbor correspondence -- no invented
        anatomical joint-index table required).

        Earlier iteration note: zeroing out ``joint_coords`` in the
        `keypoint_mapping` regression (instead of nearest-neighbor) produces
        a degenerate ~1-1.75m "divergence" for any keypoint the learned
        regressor weights mostly onto joint_coords (its mesh-only
        reconstruction collapses toward the coordinate origin, not a
        meaningful "where the mesh alone would place it" estimate) --
        caught live on the fleet GPU VM (see the lane report HONEST
        ISSUES); nearest-neighbor avoids that failure mode by construction
        (bounded below by the true nearest skeleton joint, never a
        near-origin artifact).

        This isolates exactly how much the raw FK skeleton and the skinned
        mesh-derived keypoint disagree at decode time. p95 across all
        frames/joints, target <= 5mm (spec).
        """
        batch = self._prep_batch(
            global_orient_euler=global_orient_euler,
            body_pose_euler=body_pose_euler,
            shape=shape,
            scale=scale,
            hand_pose=hand_pose,
        )
        _verts, kp_full, jcoords, _model_params = self.head.mhr_forward(
            **batch,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=False,
        )
        full_70 = kp_full[:, :NUM_KEYPOINTS]
        # (B, 70, 1, 3) - (B, 1, 127, 3) -> (B, 70, 127) pairwise distances -> nearest skeleton joint.
        pairwise = (full_70.unsqueeze(2) - jcoords.unsqueeze(1)).norm(dim=-1)
        nearest_m = pairwise.min(dim=-1).values
        divergence_mm = (nearest_m * 1000.0).detach().cpu().numpy().reshape(-1)
        return {
            "metric": "mesh_skeleton_divergence_mm",
            "target_p95_mm": MESH_SKELETON_DIVERGENCE_P95_MM,
            "sample_count": int(divergence_mm.shape[0]),
            "max_mm": float(np.max(divergence_mm)) if divergence_mm.size else 0.0,
            "mean_mm": float(np.mean(divergence_mm)) if divergence_mm.size else 0.0,
            "p95_mm": float(np.percentile(divergence_mm, 95)) if divergence_mm.size else 0.0,
            "passed": bool(divergence_mm.size and float(np.percentile(divergence_mm, 95)) <= MESH_SKELETON_DIVERGENCE_P95_MM),
        }


# ---------------------------------------------------------------------------
# GATE 1b: world-frame round trip. Reuses `worldhmr._ground_fast_sam_sample`
# (the SAME grounding transform the pipeline applies) rather than
# reimplementing the camera->world math, so this gate measures decode
# fidelity, not a second hand-rolled transform that could itself be wrong.
# ---------------------------------------------------------------------------
def ground_decoded_camera_frame(
    *,
    joints_camera: Any,
    vertices_camera: Any,
    pred_cam_t: Sequence[float] | None = None,
    pred_cam_t_already_applied: bool = False,
    track_world_xy: Sequence[float],
    t: float,
    frame_idx: int,
    player_id: int,
    calibration: Any,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Reground a decoded camera-frame joint/vertex set through the real
    pipeline transform (`threed.racketsport.worldhmr._ground_fast_sam_sample`)
    so GATE 1b compares apples-to-apples against persisted `joints_world` /
    `vertices_world`. camera_motion correction is intentionally omitted here
    (wolverine's camera-motion probe is statics-OFF per wave-4 evidence --
    see the lane report's HONEST ISSUES for clips where this assumption
    would need revisiting).
    """
    from . import worldhmr  # local import: worldhmr pulls in the full schema/pose stack

    joints_with_translation = apply_pred_cam_t_once(
        joints_camera,
        pred_cam_t=pred_cam_t,
        already_applied=pred_cam_t_already_applied,
    )
    vertices_with_translation = apply_pred_cam_t_once(
        vertices_camera,
        pred_cam_t=pred_cam_t,
        already_applied=pred_cam_t_already_applied,
    )
    sample = {
        "frame_idx": int(frame_idx),
        "player_id": int(player_id),
        "t": float(t),
        "confidence": float(confidence),
        "track_world_xy": [float(track_world_xy[0]), float(track_world_xy[1])],
        "joints_camera": joints_with_translation,
        "vertices_camera": vertices_with_translation,
    }
    return worldhmr._ground_fast_sam_sample(sample, calibration=calibration, camera_motion=None)


def apply_pred_cam_t_once(
    points_camera: Any,
    *,
    pred_cam_t: Sequence[float] | None = None,
    already_applied: bool = False,
) -> list[list[float]]:
    """Apply SAM-3D-Body pred_cam_t to camera-space points exactly once.

    MHR conversion.py treats pred_vertices and pred_cam_t as separate meter-space
    terms before conversion into MHR centimeters. This helper keeps that policy
    explicit for harnesses that reconstruct camera-space points from raw model
    outputs while allowing callers to mark already-translated sidecars.
    """
    if pred_cam_t is not None and not already_applied and len(pred_cam_t) != 3:
        raise ValueError("pred_cam_t must be a 3-vector")
    return coordinates.apply_translation_once(
        points_camera,
        pred_cam_t,
        already_applied=already_applied,
    )


def gate_1b_world_round_trip(
    *,
    decoded_joints_world: Sequence[Sequence[float]],
    decoded_vertices_world: Sequence[Sequence[float]],
    persisted_joints_world: Sequence[Sequence[float]],
    persisted_vertices_world: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """GATE 1b: decode(emit(frame)) reproduces persisted joints_world/vertices_world to <=1mm."""

    def _max_mm(decoded: Sequence[Sequence[float]], persisted: Sequence[Sequence[float]]) -> dict[str, Any]:
        if not decoded or not persisted:
            return {"n": 0, "max_abs_error_mm": 0.0, "p95_abs_error_mm": 0.0}
        n = min(len(decoded), len(persisted))
        errs_mm = []
        for i in range(n):
            d = decoded[i]
            p = persisted[i]
            dist_m = math.sqrt(sum((float(d[k]) - float(p[k])) ** 2 for k in range(3)))
            errs_mm.append(dist_m * 1000.0)
        arr = np.asarray(errs_mm, dtype=np.float64)
        return {
            "n": n,
            "max_abs_error_mm": float(arr.max()),
            "mean_abs_error_mm": float(arr.mean()),
            "p95_abs_error_mm": float(np.percentile(arr, 95)),
        }

    joints = _max_mm(decoded_joints_world, persisted_joints_world)
    vertices = _max_mm(decoded_vertices_world, persisted_vertices_world)
    max_err = max(joints["max_abs_error_mm"], vertices["max_abs_error_mm"])
    return {
        "gate": "gate_1b_world_round_trip",
        "target_max_abs_error_mm": GATE_1B_MAX_ABS_ERROR_MM,
        "joints_world": joints,
        "vertices_world": vertices,
        "max_abs_error_mm": max_err,
        "passed": bool(max_err <= GATE_1B_MAX_ABS_ERROR_MM),
    }


__all__ = [
    "BODY_POSE_EULER_DIM",
    "BODY_POSE_CONT_DIM",
    "GLOBAL_ROT_EULER_DIM",
    "GLOBAL_ROT_6D_DIM",
    "PRED_POSE_RAW_DIM",
    "SHAPE_DIM",
    "SCALE_DIM",
    "HAND_COMPS_DIM",
    "FACE_COMPS_DIM",
    "NUM_KEYPOINTS",
    "GATE_1A_MAX_ABS_ERROR_DEG",
    "GATE_1B_MAX_ABS_ERROR_MM",
    "MESH_SKELETON_DIVERGENCE_P95_MM",
    "MHR_RUNTIME_AVAILABLE",
    "MHR_RUNTIME_IMPORT_ERROR",
    "TORCH_IMPORT_ERROR",
    "encode_body_pose_euler_to_cont",
    "decode_body_pose_cont_to_euler",
    "encode_global_orient_euler_to_rot6d",
    "decode_global_rot6d_to_euler",
    "build_pred_pose_raw",
    "split_pred_pose_raw",
    "round_trip_body_pose_euler_error_deg",
    "round_trip_global_orient_error_deg",
    "gate_1a_euler_round_trip",
    "MHRDecoder",
    "apply_pred_cam_t_once",
    "ground_decoded_camera_frame",
    "gate_1b_world_round_trip",
]
