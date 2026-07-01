"""Review bundle builder for BODY world-joint labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_world_label_review_bundle"
QUEUE_ARTIFACT_TYPE = "racketsport_body_world_label_review_queue"
LABEL_ARTIFACT_TYPE = "racketsport_body_world_joints_labels"


def build_body_world_label_review_bundle(
    *,
    packet: Mapping[str, Any],
    body_frames_dir: str | Path,
    out_dir: str | Path,
    packet_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write selected BODY world-joint review samples and a safe label template."""

    out = Path(out_dir)
    frames_out = out / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)
    body_frames = Path(body_frames_dir)
    final_label_path = out.parent / str(packet.get("suggested_label_path", "labels/body_world_joints.json"))
    finalization_report_path = out / "body_world_label_finalization.json"
    label_template_path = out / "body_world_joints.template.json"
    selected_ids = _selected_sample_ids(packet)
    samples_by_id = {str(sample.get("sample_id", "")): sample for sample in _samples(packet)}
    selected_samples = [samples_by_id[sample_id] for sample_id in selected_ids if sample_id in samples_by_id]
    missing_selected_ids = [sample_id for sample_id in selected_ids if sample_id not in samples_by_id]

    queue_samples: list[dict[str, Any]] = []
    label_samples: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []
    copied_frames: set[int] = set()

    for sample in selected_samples:
        frame_index = _maybe_int(sample.get("frame_index"))
        sample_id = str(sample.get("sample_id", ""))
        if frame_index is None:
            missing_frames.append({"sample_id": sample_id, "frame_index": None, "source_image_path": ""})
            continue
        source_image = body_frames / f"frame_{frame_index:06d}.jpg"
        image_path = frames_out / source_image.name
        source_image_exists = source_image.is_file()
        if source_image_exists and frame_index not in copied_frames:
            shutil.copy2(source_image, image_path)
            copied_frames.add(frame_index)
        if not source_image_exists:
            missing_frames.append(
                {
                    "sample_id": sample_id,
                    "frame_index": frame_index,
                    "source_image_path": str(source_image),
                    "image_path": str(image_path),
                }
            )
        queue_samples.append(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "t": sample.get("t"),
                "player_id": sample.get("player_id"),
                "track_world_xy": sample.get("track_world_xy"),
                "joint_count": sample.get("joint_count"),
                "predicted_joints_world": sample.get("predicted_joints_world", []),
                "joint_conf": sample.get("joint_conf", []),
                "image_path": str(image_path),
                "source_image_path": str(source_image),
                "source_image_exists": source_image_exists,
            }
        )
        label_samples.append(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "t": sample.get("t"),
                "player_id": sample.get("player_id"),
                "accepted": False,
                "review_status": "needs_review",
                "joints_world": [],
                "predicted_joints_world": sample.get("predicted_joints_world", []),
                "joint_conf": sample.get("joint_conf", []),
                "notes": "",
            }
        )

    status = _status(
        selected_sample_count=len(selected_samples),
        missing_selected_ids=missing_selected_ids,
        missing_frames=missing_frames,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "clip": str(packet.get("clip", "")),
        "source_packet": str(packet_path or ""),
        "source_video": str(packet.get("source_video", "")),
        "suggested_label_path": str(packet.get("suggested_label_path", "labels/body_world_joints.json")),
        "body_frames_dir": str(body_frames),
        "out_dir": str(out),
        "queue_path": str(out / "body_world_label_review_queue.json"),
        "label_template_path": str(label_template_path),
        "final_label_path": str(final_label_path),
        "finalization_report_path": str(finalization_report_path),
        "finalize_command": _finalize_command(
            template_path=label_template_path,
            final_label_path=final_label_path,
            finalization_report_path=finalization_report_path,
        ),
        "selected_sample_count": len(selected_samples),
        "required_sample_count": _review_plan(packet).get("required_sample_count", len(selected_samples)),
        "missing_frame_count": len(missing_frames),
        "missing_selected_sample_count": len(missing_selected_ids),
        "missing_selected_sample_ids": missing_selected_ids,
        "missing_frames": missing_frames,
        "not_ground_truth": True,
    }
    queue = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": QUEUE_ARTIFACT_TYPE,
        "status": status,
        "clip": manifest["clip"],
        "source_packet": manifest["source_packet"],
        "source_video": manifest["source_video"],
        "sample_count": len(queue_samples),
        "samples": queue_samples,
        "not_ground_truth": True,
    }
    template = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LABEL_ARTIFACT_TYPE,
        "status": "draft_review_template",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "clip": manifest["clip"],
        "source_packet": manifest["source_packet"],
        "source_video": manifest["source_video"],
        "joint_names": list(packet.get("joint_names", [])) if isinstance(packet.get("joint_names"), list) else [],
        "selected_sample_ids": [sample["sample_id"] for sample in label_samples],
        "samples": label_samples,
        "review_instructions": [
            "Fill joints_world only after human or trusted teacher review.",
            "Keep accepted=false until a sample is actually reviewed.",
            "Do not use this template for world-MPJPE while not_ground_truth=true.",
            "Write final reviewed labels to the suggested label path only after review is complete.",
        ],
    }
    _write_json(out / "body_world_label_review_bundle.json", manifest)
    _write_json(out / "body_world_label_review_queue.json", queue)
    _write_json(label_template_path, template)
    return manifest


def build_body_world_label_review_bundle_from_paths(
    *,
    packet_path: str | Path,
    body_frames_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    packet = _read_json(packet_path)
    return build_body_world_label_review_bundle(
        packet=packet,
        body_frames_dir=body_frames_dir,
        out_dir=out_dir,
        packet_path=packet_path,
    )


def _selected_sample_ids(packet: Mapping[str, Any]) -> list[str]:
    review_plan = _review_plan(packet)
    selected = review_plan.get("selected_sample_ids")
    if isinstance(selected, list):
        return [str(sample_id) for sample_id in selected if str(sample_id)]
    return [str(sample.get("sample_id", "")) for sample in _samples(packet) if str(sample.get("sample_id", ""))]


def _samples(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples = packet.get("samples")
    if not isinstance(samples, list):
        return []
    return [sample for sample in samples if isinstance(sample, Mapping)]


def _review_plan(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    review_plan = packet.get("review_plan")
    return review_plan if isinstance(review_plan, Mapping) else {}


def _status(
    *,
    selected_sample_count: int,
    missing_selected_ids: list[str],
    missing_frames: list[dict[str, Any]],
) -> str:
    if selected_sample_count == 0:
        return "no_selected_samples"
    if missing_selected_ids:
        return "blocked_missing_selected_samples"
    if missing_frames:
        return "blocked_missing_review_frames"
    return "ready_for_review"


def _finalize_command(
    *,
    template_path: Path,
    final_label_path: Path,
    finalization_report_path: Path,
) -> str:
    return (
        "python scripts/racketsport/finalize_body_world_labels.py "
        f"--template {template_path} "
        f"--out {final_label_path} "
        f"--report-out {finalization_report_path}"
    )


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ARTIFACT_TYPE",
    "build_body_world_label_review_bundle",
    "build_body_world_label_review_bundle_from_paths",
]
