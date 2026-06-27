from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARTIFACT_TYPE = "racketsport_sam3dbody_probe"
PROBE_STATUS = "probe_only_not_verified"

PERSON_CONTAINER_KEYS = (
    "people",
    "persons",
    "person_results",
    "humans",
    "predictions",
    "outputs",
    "results",
)
BBOX_KEYS = ("bbox", "box", "xyxy", "pred_box", "person_box")
CAMERA_KEYS = ("camera", "pred_cam", "pred_camera", "cam", "weak_perspective_camera", "camera_translation")
FOCAL_KEYS = ("focal", "focal_length", "focal_lengths", "fl")


def parse_bbox_arg(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    try:
        bbox = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("bbox must be x1,y1,x2,y2 with numeric coordinates") from exc
    if not all(math.isfinite(coord) for coord in bbox):
        raise ValueError("bbox coordinates must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox must be x1,y1,x2,y2 with x2>x1 and y2>y1")
    return bbox


def summarize_process_one_image_output(
    raw_output: Any,
    *,
    provenance: Mapping[str, Any] | None = None,
    requested_bboxes: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    persons = [_as_public_mapping(person) for person in _extract_person_records(raw_output)]
    provenance_payload = _json_safe(provenance or {})
    if requested_bboxes is not None:
        provenance_payload["requested_bboxes"] = [[float(coord) for coord in bbox] for bbox in requested_bboxes]

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": PROBE_STATUS,
        "probe_only": True,
        "body_contract": False,
        "verified_body_output": False,
        "generated_artifacts": [],
        "contract_notes": [
            "Probe-only metadata extracted from raw FastSAM-3D-Body process_one_image output.",
            "This is not the BODY contract and is not verified for downstream body evaluation.",
            "This probe does not generate or fabricate smpl_motion.json or skeleton3d.json.",
        ],
        "provenance": provenance_payload,
        "raw_output_type": _type_name(raw_output),
        "top_level_keys": _mapping_keys(raw_output) if isinstance(raw_output, Mapping) else [],
        "raw_array_shapes": _collect_array_shapes(raw_output),
        "person_count": len(persons),
        "persons": [_summarize_person(person, index) for index, person in enumerate(persons)],
    }


def _summarize_person(person: Mapping[str, Any], index: int) -> dict[str, Any]:
    selected_fields: dict[str, Any] = {}
    for label, keys in (
        ("bbox", BBOX_KEYS),
        ("camera", CAMERA_KEYS),
        ("focal", FOCAL_KEYS),
    ):
        selected = _find_first_path(person, keys)
        if selected is not None:
            source_key, value = selected
            selected_fields[label] = {"source_key": source_key, "value": _json_safe(value)}

    return {
        "person_index": index,
        "keys": _mapping_keys(person),
        "selected_fields": selected_fields,
        "array_shapes": _collect_array_shapes(person),
    }


def _extract_person_records(raw_output: Any) -> list[Any]:
    if isinstance(raw_output, Mapping):
        for key in PERSON_CONTAINER_KEYS:
            if key in raw_output:
                records = _coerce_person_records(raw_output[key])
                if records:
                    return records
        if _looks_like_person_record(raw_output):
            return [raw_output]
        return []
    return _coerce_person_records(raw_output)


def _coerce_person_records(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        values = [value[key] for key in sorted(value, key=str)]
        records = [item for item in values if _as_public_mapping(item)]
        return records
    if _is_sequence(value):
        return [item for item in value if _as_public_mapping(item)]
    return []


def _looks_like_person_record(value: Mapping[str, Any]) -> bool:
    keys = {str(key) for key in value}
    if keys.intersection(BBOX_KEYS + CAMERA_KEYS + FOCAL_KEYS):
        return True
    return any(_is_array_like(item) for item in value.values())


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


def _find_first_path(value: Mapping[str, Any], wanted_keys: Sequence[str], prefix: str = "") -> tuple[str, Any] | None:
    wanted = {key.lower() for key in wanted_keys}
    for wanted_key in wanted_keys:
        for key, item in value.items():
            if str(key).lower() == wanted_key.lower():
                source_key = f"{prefix}.{key}" if prefix else str(key)
                return source_key, item

    for key, item in value.items():
        if isinstance(item, Mapping):
            source_prefix = f"{prefix}.{key}" if prefix else str(key)
            found = _find_first_path(_as_public_mapping(item), tuple(wanted), source_prefix)
            if found is not None:
                return found
    return None


def _collect_array_shapes(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if _is_array_like(value):
        return {prefix or "<root>": _array_descriptor(value, include_value=False)}

    shapes: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            item_prefix = f"{prefix}.{key}" if prefix else str(key)
            shapes.update(_collect_array_shapes(value[key], item_prefix))
    elif _is_sequence(value):
        for index, item in enumerate(value):
            item_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            shapes.update(_collect_array_shapes(item, item_prefix))
    else:
        public_mapping = _as_public_mapping(value)
        if public_mapping:
            shapes.update(_collect_array_shapes(public_mapping, prefix))
    return shapes


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if _is_array_like(value):
        return _array_descriptor(value, include_value=True)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_json_safe(item) for item in value]

    public_mapping = _as_public_mapping(value)
    if public_mapping:
        return _json_safe(public_mapping)
    return repr(value)


def _array_descriptor(value: Any, *, include_value: bool) -> dict[str, Any]:
    shape = _shape_list(value)
    descriptor: dict[str, Any] = {"kind": "array_like", "shape": shape}
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        descriptor["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        descriptor["device"] = str(device)
    if include_value and _small_shape(shape):
        small_value = _array_to_list(value)
        if small_value is not None:
            descriptor["value"] = _json_safe(small_value)
    return descriptor


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", ())
    if isinstance(shape, int):
        return [shape]
    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return []


def _array_to_list(value: Any) -> Any | None:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                item = method()
            except Exception:
                return None
    tolist = getattr(item, "tolist", None)
    if not callable(tolist):
        return None
    try:
        return tolist()
    except Exception:
        return None


def _small_shape(shape: Sequence[int]) -> bool:
    if not shape:
        return True
    total = 1
    for dim in shape:
        total *= dim
        if total > 16:
            return False
    return True


def _mapping_keys(value: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in value)


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and not isinstance(value, str | bytes | bytearray)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
