#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.run_sam3dbody_frame import (  # noqa: E402
    EX_CONFIG,
    _bbox_array_or_list,
    _extract_person_records,
    _json_safe,
)
from scripts.racketsport.run_sam3dbody_probe import (  # noqa: E402
    _detector_name,
    _load_setup_sam_3d_body,
    _runtime_path_errors,
    _setup_estimator,
    parse_bbox_arg,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run FastSAM-3D-Body on a batch of frame/bbox requests.")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fast-sam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-name", default=None)
    parser.add_argument("--fov-name", default="")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.requests.read_text(encoding="utf-8"))
        requests = _parse_requests(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid request batch: {exc}", file=sys.stderr)
        return EX_CONFIG

    all_path_errors = []
    for request in requests:
        all_path_errors.extend(_runtime_path_errors(request["image"], args.fast_sam_repo, args.checkpoint_dir))
    if all_path_errors:
        for error in all_path_errors:
            print(error, file=sys.stderr)
        return EX_CONFIG

    try:
        setup_sam_3d_body = _load_setup_sam_3d_body(args.fast_sam_repo)
        estimator = _setup_estimator(
            setup_sam_3d_body,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            detector_name=_detector_name(args.detector_name, [bbox for request in requests for bbox in request["bboxes"]]),
            detector_model=args.detector_model,
            fov_name=args.fov_name,
        )
        faces = _json_safe(getattr(estimator, "faces", None))
        frames = []
        for request in requests:
            raw_output = estimator.process_one_image(
                str(request["image"].resolve()),
                bboxes=_bbox_array_or_list(request["bboxes"]),
                use_mask=False,
                hand_box_source="body_decoder",
            )
            records = [_json_safe(record) for record in _extract_person_records(raw_output)]
            if faces:
                for record in records:
                    if isinstance(record, dict) and "mesh_faces" not in record and "faces" not in record:
                        record["mesh_faces"] = faces
            frames.append(
                {
                    "request_id": request["request_id"],
                    "image_path": str(request["image"].resolve()),
                    "requested_bboxes": request["bboxes"],
                    "records": records,
                    "summary": {"record_count": len(records)},
                }
            )
    except Exception as exc:
        print(f"FastSAM-3D-Body batch failed: {exc}", file=sys.stderr)
        return 1

    out_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch",
        "request_count": len(requests),
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def _parse_requests(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("payload.requests must be a list")
    requests = []
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, Mapping):
            raise ValueError(f"requests/{index} must be an object")
        image = raw.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"requests/{index}/image must be a non-empty string")
        bboxes = raw.get("bboxes")
        if not isinstance(bboxes, list):
            raise ValueError(f"requests/{index}/bboxes must be a list")
        requests.append(
            {
                "request_id": str(raw.get("request_id", index)),
                "image": Path(image),
                "bboxes": [parse_bbox_arg(",".join(str(value) for value in bbox)) for bbox in bboxes],
            }
        )
    return requests


if __name__ == "__main__":
    raise SystemExit(main())
