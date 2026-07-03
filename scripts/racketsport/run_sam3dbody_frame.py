#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.run_sam3dbody_probe import (  # noqa: E402
    _bbox_array_or_list,
    _detector_name,
    _load_setup_sam_3d_body,
    _runtime_path_errors,
    _setup_estimator,
    parse_bbox_arg,
)
from threed.racketsport.sam3d_body_input_prep import load_mask_prompt_arrays, normalize_body_input_size  # noqa: E402


EX_CONFIG = 66


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run FastSAM-3D-Body on one frame and emit full JSON records.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bbox", action="append", default=[])
    parser.add_argument("--fast-sam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-name", default=None)
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--mask", action="append", default=[], help="Optional repeated binary mask PNG path matching --bbox order.")
    parser.add_argument("--camera-intrinsics", default="", help="Optional static per-clip 3x3 K JSON passed as cam_int.")
    parser.add_argument("--body-input-size", type=int, default=None, help="Optional SAM-3D body crop size: 384, 448, or 512.")
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
        body_input_size = normalize_body_input_size(args.body_input_size)
        camera_intrinsics = _parse_camera_intrinsics(args.camera_intrinsics)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EX_CONFIG
    if body_input_size is not None:
        os.environ["IMG_SIZE"] = str(body_input_size)

    try:
        setup_sam_3d_body = _load_setup_sam_3d_body(args.fast_sam_repo)
        estimator = _setup_estimator(
            setup_sam_3d_body,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            detector_name=_detector_name(args.detector_name, requested_bboxes),
            detector_model=args.detector_model,
            fov_name=args.fov_name,
        )
        masks_arg = load_mask_prompt_arrays(args.mask)
        raw_output = estimator.process_one_image(
            str(args.image.resolve()),
            bboxes=_bbox_array_or_list(requested_bboxes),
            masks=masks_arg,
            cam_int=_camera_intrinsics_tensor(camera_intrinsics),
            use_mask=masks_arg is not None,
            hand_box_source="body_decoder",
        )
    except Exception as exc:
        print(f"FastSAM-3D-Body process_one_image failed: {exc}", file=sys.stderr)
        return 1

    records = [_json_safe(record) for record in _extract_person_records(raw_output)]
    faces = _json_safe(getattr(estimator, "faces", None))
    if faces:
        for record in records:
            if isinstance(record, dict) and "mesh_faces" not in record and "faces" not in record:
                record["mesh_faces"] = faces

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_frame",
        "image_path": str(args.image.resolve()),
        "requested_bboxes": requested_bboxes,
        "requested_masks": [str(Path(path)) for path in args.mask],
        "camera_intrinsics": camera_intrinsics,
        "sam3d_body_input_size_px": body_input_size,
        "records": records,
        "summary": {"record_count": len(records)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def _extract_person_records(raw_output: Any) -> list[Any]:
    if isinstance(raw_output, Mapping):
        for key in ("people", "persons", "person_results", "humans", "predictions", "outputs", "results"):
            if key in raw_output:
                records = _coerce_person_records(raw_output[key])
                if records:
                    return records
        return [raw_output] if _looks_like_person_record(raw_output) else []
    return _coerce_person_records(raw_output)


def _coerce_person_records(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value[key] for key in sorted(value, key=str) if _as_public_mapping(value[key])]
    if _is_sequence(value):
        return [item for item in value if _as_public_mapping(item)]
    return []


def _looks_like_person_record(value: Mapping[str, Any]) -> bool:
    keys = {str(key) for key in value}
    return bool(keys.intersection({"pred_vertices", "vertices", "pred_keypoints_3d", "pred_joint_coords"}))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            item = method()
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    if isinstance(item, Mapping):
        return {str(key): _json_safe(field) for key, field in item.items()}
    if _is_sequence(item):
        return [_json_safe(field) for field in item]
    public = _as_public_mapping(item)
    if public:
        return _json_safe(public)
    return repr(item)


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


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _parse_camera_intrinsics(value: str) -> list[list[float]] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("--camera-intrinsics must be a 3x3 JSON array")
    rows = []
    for row in payload:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("--camera-intrinsics must be a 3x3 JSON array")
        rows.append([float(item) for item in row])
    return rows


def _camera_intrinsics_tensor(camera_intrinsics: list[list[float]] | None) -> Any | None:
    if camera_intrinsics is None:
        return None
    import torch  # type: ignore[import-not-found]

    return torch.tensor([camera_intrinsics], dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(main())
