#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.sam3dbody_probe import parse_bbox_arg, summarize_process_one_image_output


EX_CONFIG = 66


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a probe-only FastSAM-3D-Body process_one_image seam and record JSON-safe metadata."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image for FastSAM-3D-Body.")
    parser.add_argument("--out", required=True, type=Path, help="Probe metadata JSON output path.")
    parser.add_argument("--bbox", action="append", default=[], help="Optional repeated bbox as x1,y1,x2,y2.")
    parser.add_argument("--fast-sam-repo", required=True, type=Path, help="FastSAM-3D-Body repository path.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="FastSAM-3D-Body checkpoint directory.")
    parser.add_argument("--detector-model", default="yolo11n.pt", help="Detector model name/path passed to runtime.")
    parser.add_argument(
        "--detector-name",
        default=None,
        help="Detector backend passed to setup_sam_3d_body. Defaults to disabled when --bbox is supplied, otherwise yolo.",
    )
    parser.add_argument("--fov-name", default="moge2", help="FOV estimator name passed to setup_sam_3d_body.")
    args = parser.parse_args(argv)

    try:
        requested_bboxes = [parse_bbox_arg(value) for value in args.bbox]
    except ValueError as exc:
        parser.error(str(exc))

    path_errors = _runtime_path_errors(args.image, args.fast_sam_repo, args.checkpoint_dir)
    if path_errors:
        for error in path_errors:
            print(error, file=sys.stderr)
        return EX_CONFIG

    try:
        setup_sam_3d_body = _load_setup_sam_3d_body(args.fast_sam_repo)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return EX_CONFIG

    os.environ.setdefault("FAST_SAM_CHECKPOINT_DIR", str(args.checkpoint_dir.resolve()))
    os.environ.setdefault("SAM3DBODY_CHECKPOINT_DIR", str(args.checkpoint_dir.resolve()))

    try:
        estimator = _setup_estimator(
            setup_sam_3d_body,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            detector_name=_detector_name(args.detector_name, requested_bboxes),
            detector_model=args.detector_model,
            fov_name=args.fov_name,
        )
        raw_output = estimator.process_one_image(
            str(args.image.resolve()),
            bboxes=_bbox_array_or_list(requested_bboxes),
            use_mask=False,
            hand_box_source="body_decoder",
        )
    except Exception as exc:
        print(f"FastSAM-3D-Body process_one_image failed: {exc}", file=sys.stderr)
        return 1

    try:
        process_one_image = estimator.process_one_image
    except AttributeError:
        process_one_image = None

    payload = summarize_process_one_image_output(
        raw_output,
        provenance={
            "image_path": str(args.image.resolve()),
            "fast_sam_repo": str(args.fast_sam_repo.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "detector_name": _detector_name(args.detector_name, requested_bboxes),
            "detector_model": args.detector_model,
            "fov_name": args.fov_name,
            "runtime_setup_function": _callable_name(setup_sam_3d_body),
            "runtime_function": _callable_name(process_one_image) if process_one_image else "<missing>",
        },
        requested_bboxes=requested_bboxes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def _runtime_path_errors(image: Path, fast_sam_repo: Path, checkpoint_dir: Path) -> list[str]:
    errors: list[str] = []
    if not image.is_file():
        errors.append(f"missing input image at {image}")
    if not fast_sam_repo.is_dir():
        errors.append(f"missing FastSAM-3D-Body repo at {fast_sam_repo}")
    if not checkpoint_dir.is_dir():
        errors.append(f"missing FastSAM-3D-Body checkpoint directory at {checkpoint_dir}")
    return errors


def _load_setup_sam_3d_body(fast_sam_repo: Path) -> Callable[..., Any]:
    repo = str(fast_sam_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        module = importlib.import_module("notebook.utils")
    except Exception as exc:
        try:
            importlib.import_module("sam_3d_body")
        except Exception as direct_exc:
            raise RuntimeError(
                "could not import FastSAM-3D-Body notebook.utils.setup_sam_3d_body "
                "or direct sam_3d_body runtime"
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


def _setup_estimator(
    setup_sam_3d_body: Callable[..., Any],
    *,
    checkpoint_dir: Path,
    detector_name: str,
    detector_model: str,
    fov_name: str,
) -> Any:
    return setup_sam_3d_body(
        hf_repo_id="facebook/sam-3d-body-dinov3",
        detector_name=detector_name,
        detector_model=detector_model,
        fov_name=fov_name,
        local_checkpoint_path=str(checkpoint_dir),
    )


def _detector_name(user_value: str | None, requested_bboxes: list[list[float]]) -> str:
    if user_value is not None:
        return user_value
    if requested_bboxes:
        return ""
    return "yolo"


def _bbox_array_or_list(requested_bboxes: list[list[float]]) -> Any:
    if not requested_bboxes:
        return None
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return requested_bboxes
    return np.asarray(requested_bboxes, dtype=np.float32)


def _callable_name(callable_value: Callable[..., Any]) -> str:
    module = getattr(callable_value, "__module__", "<unknown>")
    name = getattr(callable_value, "__qualname__", getattr(callable_value, "__name__", "<callable>"))
    return f"{module}.{name}"


if __name__ == "__main__":
    raise SystemExit(main())
