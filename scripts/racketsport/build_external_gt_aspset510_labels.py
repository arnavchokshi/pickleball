#!/usr/bin/env python3
"""Convert a real ASPset-510 `.c3d` mocap clip into a `body_world_joints.json`-style
external-ground-truth label file for the BODY world-MPJPE gate.

Owner decision (2026-07-02, binding): validate BODY via a public 3D-GT dataset through
the gate's `external_ground_truth` label source. See
`runs/body_external_gt_20260702T*/DATASET_SELECTION.md` for why ASPset-510 was chosen
(CC0 license, freely downloadable, real calibrated 3-camera outdoor mocap, includes
running/jumping/throwing/hitting motions) and `METHODOLOGY.md` for the full write-up.

Requires the optional `ezc3d` dependency (`pip install ezc3d`) to parse the real `.c3d`
biomechanics mocap file -- this is a one-time offline conversion step, not a pipeline
runtime dependency, so `ezc3d` is intentionally NOT added to the main project
requirements; this script fails with a clear message if it is missing.

Usage:
    python3 scripts/racketsport/build_external_gt_aspset510_labels.py \\
        --c3d-path /path/to/ASPset-510/test/joints_3d/1e28/1e28-0001.c3d \\
        --subject-id 1e28 --clip-id 0001 --camera-id left \\
        --player-id 1 --frame-stride 10 \\
        --out runs/body_external_gt_20260702T.../labels/1e28-0001/body_world_joints.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.external_gt_aspset510 import (  # noqa: E402
    ASPSET17J_JOINT_NAMES,
    DATASET_LICENSE,
    DATASET_NAME,
    build_external_gt_label_samples,
)

LABEL_ARTIFACT_TYPE = "racketsport_body_world_joints_labels"
SCHEMA_VERSION = 1


def select_sample_frame_indices(total_frames: int, *, stride: int) -> list[int]:
    """Evenly-spaced frame indices at the given stride, always including the last frame.

    Pure/testable without ezc3d. ``stride=1`` returns every frame.
    """

    if total_frames <= 0:
        return []
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def _load_c3d_frames_mm(c3d_path: Path) -> list[dict[str, tuple[float, float, float]]]:
    """Parse a real ASPset-510 `.c3d` file into a list of {joint_name: (x,y,z)_mm} dicts."""

    try:
        import ezc3d
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "ezc3d is required to parse ASPset-510 .c3d mocap files "
            "(pip install ezc3d) -- this is an offline conversion-only dependency, "
            "not a pipeline runtime dependency."
        ) from exc

    c3d = ezc3d.c3d(str(c3d_path))
    labels = [str(label).strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    missing = set(ASPSET17J_JOINT_NAMES) - set(labels)
    if missing:
        raise ValueError(f"{c3d_path}: .c3d file is missing expected ASPset joint labels: {sorted(missing)}")
    points = c3d["data"]["points"]  # shape (4, n_markers, n_frames); last row is homogeneous 1s
    n_frames = points.shape[2]
    label_index = {label: index for index, label in enumerate(labels)}
    frames: list[dict[str, tuple[float, float, float]]] = []
    for frame_index in range(n_frames):
        frame = {
            name: (
                float(points[0, label_index[name], frame_index]),
                float(points[1, label_index[name], frame_index]),
                float(points[2, label_index[name], frame_index]),
            )
            for name in ASPSET17J_JOINT_NAMES
        }
        frames.append(frame)
    return frames


def build_payload(
    *,
    frames_joint_positions_mm: list[dict[str, tuple[float, float, float]]],
    frame_indices: list[int],
    player_id: int,
    clip_id: str,
    subject_id: str,
    camera_id: str,
) -> dict:
    selected_frames = [frames_joint_positions_mm[index] for index in frame_indices]
    samples = build_external_gt_label_samples(
        frames_joint_positions_mm=selected_frames,
        frame_indices=frame_indices,
        player_id=player_id,
        clip_id=clip_id,
        subject_id=subject_id,
        camera_id=camera_id,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LABEL_ARTIFACT_TYPE,
        "status": "external_ground_truth",
        "not_ground_truth": False,
        "trusted_for_world_mpjpe": True,
        "clip": f"aspset510_{subject_id}_{clip_id}_{camera_id}",
        "dataset": DATASET_NAME,
        "license": DATASET_LICENSE,
        "provenance": {
            "dataset": DATASET_NAME,
            "subject_id": subject_id,
            "clip_id": clip_id,
            "camera_id": camera_id,
            "source_frame_count": len(frames_joint_positions_mm),
            "sampled_frame_count": len(frame_indices),
        },
        "joint_names": samples[0]["joint_names"] if samples else [],
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3d-path", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--player-id", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample every Nth mocap frame.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    frames_mm = _load_c3d_frames_mm(args.c3d_path)
    frame_indices = select_sample_frame_indices(len(frames_mm), stride=args.frame_stride)
    payload = build_payload(
        frames_joint_positions_mm=frames_mm,
        frame_indices=frame_indices,
        player_id=args.player_id,
        clip_id=args.clip_id,
        subject_id=args.subject_id,
        camera_id=args.camera_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "source_frame_count": len(frames_mm),
                "sampled_frame_count": len(frame_indices),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
