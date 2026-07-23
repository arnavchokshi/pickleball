"""Stable inference adapter for the CAL-MODEL `court_unet_v2` checkpoint (2026-07-05).

`infer_court_model(image_bgr, checkpoint_path, device="cpu")` is a pure function (cv2/numpy/torch
only -- no other repo module carries hidden state) intended as the STABLE CONTRACT later fusion
code (`runs/lanes/court_autofind_20260705/DESIGN.md` Stage 1 "E4 neural evidence") will call per
frame, regardless of which specific `court_unet_v2` checkpoint is loaded:

    infer_court_model(image_bgr, checkpoint_path, device="cpu") -> dict(
        keypoints_xy: dict[name -> [x, y]] in ``image_bgr``'s own pixel space -- the caller may
            pass a frame at any resolution; this adapter resizes to the checkpoint's model input
            size internally and rescales every prediction back into the input image's coordinate
            space, so fusion code never needs to know (or care about) the model's native
            resolution.
        keypoints_conf: dict[name -> float in [0, 1]], the decoded heatmap peak's spatial-softmax
            probability mass (comparable across keypoints/checkpoints since it is normalized).
        keypoints_vis: dict[name -> float in [0, 1]], sigmoid(visibility logit) -- the predicted
            probability this keypoint is cleanly visible (as opposed to occluded/off-frame).
        line_family_mask: HxW uint8 array (``image_bgr``'s resolution), values in
            threed.racketsport.court_synth_stream.LINE_FAMILY_CLASSES {0,1,2,3} =
            {other,pickleball_line,tennis_line,net}.
        surface_mask: HxW uint8 array (``image_bgr``'s resolution), values restricted to
            {0, 2} = {background, interior} from
            threed.racketsport.court_synth_stream.SURFACE_CLASSES -- this checkpoint's 5-class
            segmentation head was never given a separate "apron" class (see
            `threed.racketsport.court_keypoint_net.COURT_UNET_V2_SEG_CLASS_NAMES` and
            `split_line_family_segmentation`), so apron pixels are indistinguishable from
            background here.
        structured_observations: floor-only top-two heatmap evidence with visibility,
            entropy/margin, covariance, and source-pixel coordinates.  Net-top points are
            structurally excluded from the planar evidence set.
        best_court: one regulation-template floor hypothesis with per-point/whole-court
            confidence and explicit inlier/ignored observations.  It is permanently
            review-only and ``measurement_valid=false`` until an independent promotion gate.
    )

Selected production inference remains `court_unet_v2`.  Review-only
`court_structured_v3` evidence checkpoints are also loadable so the structured candidate can be
measured behind the same adapter without changing the selected stack.  This module intentionally
does not attempt to also support the legacy
single-tensor-output architectures (`encoder_decoder_v1`/`local_conv_v1`), which already have
their own inference path in `scripts/racketsport/train_court_keypoint_heatmap.py`
(`predict_source_keypoints`) that this lane does not touch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from threed.racketsport.court_keypoint_net import (
    COURT_UNET_V2_ARCHITECTURE,
    PICKLEBALL_KEYPOINTS,
    court_keypoint_probabilities,
    decode_subpixel_heatmap,
    make_court_keypoint_heatmap_model,
    split_line_family_segmentation,
)
from threed.racketsport.court_confidence_calibration import (
    IsotonicConfidenceCalibrator,
    apply_point_calibration,
)
from threed.racketsport.court_structured_model import (
    COURT_STRUCTURED_V3_ARCHITECTURE,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    make_court_structured_v3_model,
)
from threed.racketsport.court_structured_evidence import extract_court_structured_evidence
from threed.racketsport.court_structured_solver import solve_best_floor_court

__all__ = [
    "CourtModelCheckpointResolution",
    "PICKLEBALL_COURT_UNET_CKPT_ENV",
    "PROMOTED_COURT_UNET_CKPT_PATH",
    "build_court_model_from_checkpoint",
    "court_model_inference_provider",
    "get_current_court_model_infer_provider",
    "infer_court_model",
    "load_court_model_checkpoint",
    "make_court_model_infer_provider",
    "resolve_court_model_checkpoint_path",
]


CourtModelInferProvider = Callable[[Any], Mapping[str, Any]]
PICKLEBALL_COURT_UNET_CKPT_ENV = "PICKLEBALL_COURT_UNET_CKPT"
PROMOTED_COURT_UNET_CKPT_PATH = Path("models/checkpoints/court_unet_v2/court_model_v2.pt")
_LOGGER = logging.getLogger(__name__)
_COURT_MODEL_PROVIDER_CACHE: dict[tuple[str, str], CourtModelInferProvider | None] = {}
_CURRENT_COURT_MODEL_INFER_PROVIDER: ContextVar[CourtModelInferProvider | None] = ContextVar(
    "current_court_model_infer_provider",
    default=None,
)


@dataclass(frozen=True)
class CourtModelCheckpointResolution:
    path: Path
    source: str
    sha256: str | None


def resolve_court_model_checkpoint_path(
    checkpoint_path: str | Path | None = None,
) -> CourtModelCheckpointResolution | None:
    """Resolve the default-on `court_unet_v2` checkpoint path.

    Precedence is explicit argument, then PICKLEBALL_COURT_UNET_CKPT, then the
    promoted repo-local checkpoint. A missing file disables the advisory neural
    provider instead of falling back to a lower-precedence path.
    """

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        source = "explicit_arg"
    else:
        env_value = os.environ.get(PICKLEBALL_COURT_UNET_CKPT_ENV)
        if env_value:
            path = Path(env_value)
            source = PICKLEBALL_COURT_UNET_CKPT_ENV
        else:
            path = PROMOTED_COURT_UNET_CKPT_PATH
            source = "promoted_default"

    if not path.is_file():
        _LOGGER.warning("court_unet_v2 checkpoint unavailable source=%s path=%s; using geometric-only", source, path)
        return None

    sha256 = _read_checkpoint_provenance_sha256(path)
    if sha256:
        _LOGGER.info("court_unet_v2 checkpoint discovered source=%s path=%s sha256=%s", source, path, sha256)
    else:
        _LOGGER.info("court_unet_v2 checkpoint discovered source=%s path=%s sha256=unknown", source, path)
    return CourtModelCheckpointResolution(path=path, source=source, sha256=sha256)


def _read_checkpoint_provenance_sha256(checkpoint_path: Path) -> str | None:
    provenance_path = checkpoint_path.parent / "PROVENANCE.json"
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sha256 = payload.get("sha256")
    return str(sha256) if isinstance(sha256, str) and sha256 else None


def make_court_model_infer_provider(
    *,
    checkpoint_path: str | Path | None = None,
    infer_callable: CourtModelInferProvider | None = None,
    device: str = "cpu",
) -> CourtModelInferProvider | None:
    """Build the optional provider used by E4 fusion.

    A callable is for tests or preloaded runtimes. Otherwise the checkpoint is
    resolved with the default-on precedence rule and loaded once per process for
    the given path/device pair.
    """

    if checkpoint_path is not None and infer_callable is not None:
        raise ValueError("provide either checkpoint_path or infer_callable, not both")
    if infer_callable is not None:
        return infer_callable

    resolved = resolve_court_model_checkpoint_path(checkpoint_path)
    if resolved is None:
        return None
    cache_key = (str(resolved.path), str(device))
    if cache_key in _COURT_MODEL_PROVIDER_CACHE:
        return _COURT_MODEL_PROVIDER_CACHE[cache_key]

    try:
        payload = load_court_model_checkpoint(resolved.path, device=device)
        model, keypoint_names, model_size = build_court_model_from_checkpoint(payload, device=device)
    except Exception as exc:
        _LOGGER.warning(
            "court_unet_v2 provider disabled source=%s path=%s reason=%s; using geometric-only",
            resolved.source,
            resolved.path,
            exc,
        )
        _COURT_MODEL_PROVIDER_CACHE[cache_key] = None
        return None

    def _provider(image_bgr: Any) -> Mapping[str, Any]:
        return _infer_court_model_with_loaded_model(
            image_bgr,
            model=model,
            keypoint_names=keypoint_names,
            model_size=model_size,
            device=device,
        )

    _COURT_MODEL_PROVIDER_CACHE[cache_key] = _provider
    return _provider


def get_current_court_model_infer_provider() -> CourtModelInferProvider | None:
    """Return the context-local E4 neural provider, or None when fusion is off."""

    return _CURRENT_COURT_MODEL_INFER_PROVIDER.get()


@contextmanager
def court_model_inference_provider(
    *,
    checkpoint_path: str | Path | None = None,
    infer_callable: CourtModelInferProvider | None = None,
    device: str = "cpu",
) -> Iterator[CourtModelInferProvider | None]:
    """Temporarily expose an E4 neural provider to lower-level hypothesis code.

    This keeps existing callers default-off and lets the existing multi-frame
    proposal path opt in without changing its source file.
    """

    provider = make_court_model_infer_provider(
        checkpoint_path=checkpoint_path,
        infer_callable=infer_callable,
        device=device,
    )
    token = _CURRENT_COURT_MODEL_INFER_PROVIDER.set(provider)
    try:
        yield provider
    finally:
        _CURRENT_COURT_MODEL_INFER_PROVIDER.reset(token)


def load_court_model_checkpoint(checkpoint_path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a `court_unet_v2` checkpoint file, failing loudly if it is not shaped like one.

    Trainer-authored checkpoints (see `scripts/racketsport/train_court_model_v2.py`) store
    argparse/Path-valued training metadata alongside the model weights, so this cannot use
    `weights_only=True` -- the checkpoint is a repo-owned artifact, never an untrusted download.
    """

    import torch

    map_location = device if device == "cuda" else "cpu"
    payload = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"court model checkpoint must contain a model state dict: {checkpoint_path}")
    return payload


def build_court_model_from_checkpoint(
    payload: dict[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Any, list[str], tuple[int, int]]:
    """Rebuild the `court_unet_v2` module + its keypoint taxonomy + model input size from a
    loaded checkpoint payload (see `load_court_model_checkpoint`)."""

    architecture = str(
        payload.get("network_architecture", payload.get("model_architecture", COURT_UNET_V2_ARCHITECTURE))
    )
    if architecture not in {COURT_UNET_V2_ARCHITECTURE, COURT_STRUCTURED_V3_ARCHITECTURE}:
        raise ValueError(
            "court_model_infer only supports "
            f"{COURT_UNET_V2_ARCHITECTURE!r} and review-only {COURT_STRUCTURED_V3_ARCHITECTURE!r} "
            f"checkpoints, got {architecture!r}"
        )
    default_names = (
        [point.name for point in PICKLEBALL_KEYPOINTS]
        if architecture == COURT_UNET_V2_ARCHITECTURE
        else list(STRUCTURED_FLOOR_KEYPOINT_NAMES)
    )
    keypoint_names = [str(name) for name in payload.get("keypoint_names", default_names)]
    image_size = payload.get("image_size", [640, 360])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("checkpoint image_size must be a two-item [width, height]")
    model_width, model_height = int(image_size[0]), int(image_size[1])
    if model_width <= 0 or model_height <= 0:
        raise ValueError("checkpoint image_size must contain positive dimensions")

    if architecture == COURT_STRUCTURED_V3_ARCHITECTURE:
        if tuple(keypoint_names) != STRUCTURED_FLOOR_KEYPOINT_NAMES:
            raise ValueError("court_structured_v3 checkpoint keypoint_names must match the 30-point floor taxonomy")
        model = make_court_structured_v3_model()
    else:
        model = make_court_keypoint_heatmap_model(len(keypoint_names), architecture=architecture)
    model.load_state_dict(payload["model"])
    calibration_payload = payload.get("point_confidence_calibration")
    point_confidence_calibrator = None
    if calibration_payload is not None:
        if not isinstance(calibration_payload, Mapping):
            raise ValueError("point_confidence_calibration must be a mapping")
        point_confidence_calibrator = IsotonicConfidenceCalibrator.from_dict(
            calibration_payload
        )
    # Keep the public builder tuple stable.  Calibration belongs to this loaded
    # checkpoint instance and is consumed by the per-frame decode path.
    model._point_confidence_calibrator = point_confidence_calibrator
    model.to(device)
    model.eval()
    return model, keypoint_names, (model_width, model_height)


def infer_court_model(image_bgr: Any, checkpoint_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Run a `court_unet_v2` checkpoint on one BGR frame. See module docstring for the exact
    output contract. Reloads and rebuilds the model on every call (no hidden caching state) so
    this stays a straightforward pure function; callers processing many frames from the same
    checkpoint should build the model once themselves via `load_court_model_checkpoint` +
    `build_court_model_from_checkpoint` and only reuse this function's per-frame decode logic if
    they need the exact same numeric behavior -- this convenience wrapper optimizes for a stable,
    trivially-testable contract over raw throughput.
    """

    payload = load_court_model_checkpoint(checkpoint_path, device=device)
    model, keypoint_names, (model_width, model_height) = build_court_model_from_checkpoint(payload, device=device)
    return _infer_court_model_with_loaded_model(
        image_bgr,
        model=model,
        keypoint_names=keypoint_names,
        model_size=(model_width, model_height),
        device=device,
    )


def _infer_court_model_with_loaded_model(
    image_bgr: Any,
    *,
    model: Any,
    keypoint_names: list[str],
    model_size: tuple[int, int],
    device: str = "cpu",
) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch

    if image_bgr is None or getattr(image_bgr, "ndim", None) != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be an HxWx3 array")
    source_height, source_width = image_bgr.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("image_bgr must have positive dimensions")

    model_width, model_height = int(model_size[0]), int(model_size[1])

    resized_bgr = cv2.resize(np.asarray(image_bgr), (model_width, model_height), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = court_keypoint_probabilities(outputs["keypoint_heatmaps"]).detach().cpu()[0]
        vis_probabilities = torch.sigmoid(outputs["keypoint_vis_logits"]).detach().cpu()[0]
        seg_logits = outputs["line_family_logits"].detach().cpu()[0]

    heatmap_stride = int(getattr(model, "heatmap_stride", 4))
    scale_x = source_width / float(model_width)
    scale_y = source_height / float(model_height)

    keypoints_xy: dict[str, list[float]] = {}
    keypoints_conf: dict[str, float] = {}
    keypoints_vis: dict[str, float] = {}
    for index, name in enumerate(keypoint_names):
        decoded = decode_subpixel_heatmap(probabilities[index].tolist())
        keypoints_xy[name] = [
            decoded.x * heatmap_stride * scale_x,
            decoded.y * heatmap_stride * scale_y,
        ]
        keypoints_conf[name] = max(0.0, min(1.0, float(decoded.score)))
        keypoints_vis[name] = float(vis_probabilities[index])

    heatmap_probabilities = {
        name: probabilities[index].numpy() for index, name in enumerate(keypoint_names)
    }
    evidence_records = extract_court_structured_evidence(
        heatmap_probabilities,
        keypoints_vis,
        image_size=(model_width, model_height),
        source_size=(source_width, source_height),
    )
    structured_result = solve_best_floor_court(
        _solver_observations_from_evidence(evidence_records)
    )
    best_court = _best_court_contract(
        structured_result,
        point_confidence_calibrator=getattr(
            model, "_point_confidence_calibrator", None
        ),
    )

    seg_argmax = seg_logits.argmax(dim=0).numpy().astype(np.uint8)
    line_family_head, surface_head = split_line_family_segmentation(seg_argmax)
    line_family_mask = cv2.resize(
        line_family_head, (source_width, source_height), interpolation=cv2.INTER_NEAREST
    )
    surface_mask = cv2.resize(surface_head, (source_width, source_height), interpolation=cv2.INTER_NEAREST)

    return {
        "keypoints_xy": keypoints_xy,
        "keypoints_conf": keypoints_conf,
        "keypoints_vis": keypoints_vis,
        "line_family_mask": line_family_mask,
        "surface_mask": surface_mask,
        "structured_observations": evidence_records,
        "best_court": best_court,
    }


def _solver_observations_from_evidence(
    evidence_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Translate stable heatmap evidence into the solver's bounded top-two candidate contract."""

    observations: dict[str, list[dict[str, Any]]] = {}
    for record in evidence_records:
        name = str(record["keypoint_name"])
        visibility = float(record["visibility"])
        entropy_factor = max(0.05, 1.0 - float(record["normalized_entropy"]))
        covariance = record["covariance_px2"]
        candidates: list[dict[str, Any]] = []
        for rank, field in enumerate(("primary_peak", "secondary_peak"), start=1):
            peak = record[field]
            source_xy = peak.get("source_xy")
            if source_xy is None:
                source_xy = record["observation_xy"] if rank == 1 else peak["heatmap_xy"]
            probability = float(peak["probability"])
            # This is an evidence weight, not calibrated accuracy.  Keep a small positive floor
            # so a diffuse heatmap can still participate in best-effort hypothesis search while
            # naturally ranking below a sharp, visible peak.
            confidence = min(1.0, max(1.0e-8, visibility * probability * entropy_factor))
            candidates.append(
                {
                    "candidate_id": f"{name}:heatmap_peak_{rank}",
                    "xy": [float(source_xy[0]), float(source_xy[1])],
                    "confidence": confidence,
                    "visibility": visibility,
                    "covariance_px2": covariance,
                    "heatmap_probability": probability,
                    "normalized_entropy": float(record["normalized_entropy"]),
                    "peak_margin": float(record["peak_margin"]),
                    "provenance": "court_model_heatmap",
                }
            )
        observations[name] = candidates
    return observations


def _best_court_contract(
    structured_result: Mapping[str, Any],
    *,
    point_confidence_calibrator: IsotonicConfidenceCalibrator | None = None,
) -> dict[str, Any]:
    """Expose the always-best candidate without upgrading it to measurement authority."""

    projected = structured_result.get("projected_floor_keypoints") or {}
    keypoints_xy = {
        str(name): [float(item["xy"][0]), float(item["xy"][1])]
        for name, item in projected.items()
        if isinstance(item, Mapping) and isinstance(item.get("xy"), (list, tuple))
    }
    selected = structured_result.get("selected_hypothesis")
    source = selected.get("source") if isinstance(selected, Mapping) else "no_valid_hypothesis"
    inliers = list(structured_result.get("inliers") or [])
    semantic_count = int((structured_result.get("observation_counts") or {}).get("semantic_count", 0))
    raw_point_confidence = {
        str(name): float(value)
        for name, value in dict(structured_result.get("point_confidence") or {}).items()
    }
    point_confidence = (
        apply_point_calibration(raw_point_confidence, point_confidence_calibrator)
        if point_confidence_calibrator is not None
        else raw_point_confidence
    )
    return {
        "schema_version": 1,
        "status": str(structured_result.get("status", "unknown")),
        "keypoints_xy": keypoints_xy,
        "point_confidence": point_confidence,
        "point_confidence_raw": raw_point_confidence,
        "confidence_status": (
            "calibrated_source_disjoint_dev"
            if point_confidence_calibrator is not None
            else "uncalibrated"
        ),
        "point_confidence_calibration": (
            point_confidence_calibrator.to_dict()
            if point_confidence_calibrator is not None
            else None
        ),
        "court_confidence": float(structured_result.get("court_confidence") or 0.0),
        "hypothesis_margin": structured_result.get("margin"),
        "homography_image_from_court": structured_result.get("homography_image_from_court"),
        "distortion": {
            "model": "not_estimated",
            "k1": None,
            "source": "future_profile_or_joint_fit",
        },
        "transform_covariance": None,
        "inlier_observations": inliers,
        "inlier_count": len(inliers),
        "inlier_ratio": len(inliers) / semantic_count if semantic_count > 0 else 0.0,
        "ignored_observations": list(structured_result.get("ignored_observations") or []),
        "residual_stats_px": dict(structured_result.get("residual_stats_px") or {}),
        "score_components": dict(structured_result.get("score_components") or {}),
        "source": str(source),
        "measurement_valid": False,
        "authority_state": "review_only",
        "solution_role": "best_effort",
        "floor_only": True,
        "diagnostics": dict(structured_result.get("diagnostics") or {}),
    }
