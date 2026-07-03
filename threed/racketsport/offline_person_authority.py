"""Offline authoritative person ReID/global-association runner."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .person_reid_diagnostics import (
    FeatureExtractor,
    ReIDEmbeddingExportConfig,
    assert_reid_checkpoint_clip_safe,
    build_source_reid_embedding_export,
    reid_checkpoint_training_provenance,
)
from .court_templates import Sport
from .person_track_gt_scoring import score_tracks_against_person_ground_truth
from .player_global_association import GlobalAssociationConfig, associate_global_identities, tracks_to_global_detections
from .schemas import PersonGroundTruth, Tracks, validate_artifact_file


@dataclass(frozen=True)
class OfflineAuthorityConfig:
    """Config for the offline (role-locked-track-input) person authority run.

    ``reid_device=None`` auto-detects the fastest available OSNet embedding
    device (cuda > mps > cpu, see
    ``person_reid_diagnostics.resolve_reid_device``); pass an explicit string
    (``"cpu"``, ``"mps"``, ``"cuda:0"``, ...) to override. This is the fix
    for the #1 measured pipeline cost center (CPU-only OSNet ReID embedding
    extraction, ~1,665-7,586 s/min-video, see
    ``runs/glue3_speed_budget_20260702T035746Z/RUNTIME_BUDGET.md``): before,
    the only auto-detected accelerator was CUDA, so every non-CUDA machine
    (including Apple Silicon Macs) silently ran embedding extraction on CPU.
    ``mps`` measured ~16-42x faster than CPU with no found correctness risk
    (200-real-crop proof: ~1e-11 cosine deviation, byte-identical clustering;
    a follow-up full-11,095-detection-clip check that initially looked like a
    device-specific divergence resolved to an unrelated concurrent code
    change once re-measured with matched code -- see
    ``person_reid_diagnostics.resolve_reid_device``'s docstring and
    ``runs/trk_speed_reid_gpu_20260702T045139Z/``).

    ``reid_batch_size=64``: measured near-optimal on both CPU and MPS, see
    ``ReIDEmbeddingExportConfig`` docstring and
    ``runs/trk_speed_reid_gpu_20260702T045139Z/timing/batching_device_benchmark.json``.
    """

    expected_players: int = 4
    reid_backend: str = "osnet"
    reid_model_name: str = "osnet_x1_0"
    reid_batch_size: int = 64
    reid_device: str | None = None
    reid_half: bool | None = None
    sample_stride_frames: int = 1
    crop_padding_px: int = 8
    embedding_bbox_scale: float = 1.0
    max_embedding_bbox_delta_px: float = 2.5
    max_gap_fill_frames: int = 24
    max_merge_gap_frames: int = 240
    max_merge_speed_m_s: float = 9.0
    appearance_weight: float = 1.0
    motion_weight: float = 1.0
    drop_outside_court: bool = True
    court_margin_m: float = 3.0
    post_association_court_margin_m: float | None = None
    cardinality_backfill: bool = False
    backfill_max_cost: float = 2.5
    backfill_iou_threshold: float = 0.25
    iou_threshold: float = 0.5
    sport: Sport = "pickleball"

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.reid_backend not in {"osnet", "ultralytics_yolo"}:
            raise ValueError("reid_backend must be osnet or ultralytics_yolo")
        if self.reid_batch_size <= 0:
            raise ValueError("reid_batch_size must be positive")


def run_offline_authority_candidate(
    *,
    clip_id: str,
    candidate: str,
    video_path: str | Path,
    source_run_dir: str | Path,
    detections_run_dir: str | Path | None = None,
    out_dir: str | Path,
    reid_model_path: str | Path,
    embedding_export_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    expected_players: int = 4,
    config: OfflineAuthorityConfig | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> dict[str, Any]:
    cfg = config or OfflineAuthorityConfig(expected_players=expected_players)
    source_dir = Path(source_run_dir)
    detection_dir = Path(detections_run_dir) if detections_run_dir is not None else source_dir
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracks_path = source_dir / "tracks.json"
    detections_path = _source_detections_path(detection_dir)
    embedding_bbox_scale = _embedding_bbox_scale(
        detection_dir,
        detections_path=detections_path,
        configured_scale=cfg.embedding_bbox_scale,
    )
    score_bbox_scale = _score_bbox_scale(source_dir)
    if score_bbox_scale["source"] == "identity" and detection_dir != source_dir:
        score_bbox_scale = _score_bbox_scale(detection_dir)
    parsed_tracks = validate_artifact_file("tracks", tracks_path)
    if not isinstance(parsed_tracks, Tracks):
        raise ValueError(f"{tracks_path} did not parse as Tracks")
    detections_payload = _read_json_object(detections_path)

    started = time.perf_counter()
    if embedding_export_path is not None:
        output_embedding_export_path = Path(embedding_export_path)
        embedding_payload = _read_json_object(output_embedding_export_path)
    else:
        output_embedding_export_path = out / "reid_embeddings.json"
        embedding_payload = build_source_reid_embedding_export(
            video_path=video_path,
            detections_payload=detections_payload,
            output_path=output_embedding_export_path,
            model_path=reid_model_path,
            command_metadata={
                "clip_id": clip_id,
                "candidate": candidate,
                "source_run_dir": str(source_dir),
                "detections_run_dir": str(detection_dir),
                "tracks_path": str(tracks_path),
                "detections_path": str(detections_path),
            },
            config=ReIDEmbeddingExportConfig(
                backend=cfg.reid_backend,
                sample_stride_frames=cfg.sample_stride_frames,
                crop_padding_px=cfg.crop_padding_px,
                batch_size=cfg.reid_batch_size,
                device=cfg.reid_device,
                half=cfg.reid_half,
                osnet_model_name=cfg.reid_model_name,
            ),
            feature_extractor=feature_extractor,
        )
    reid_provenance = reid_checkpoint_training_provenance(
        _existing_path(embedding_payload.get("model_path")) or reid_model_path
    )
    detections = tracks_to_global_detections(
        parsed_tracks,
        embedding_payload=embedding_payload,
        embedding_bbox_scale=embedding_bbox_scale,
        max_embedding_bbox_delta_px=cfg.max_embedding_bbox_delta_px,
    )
    associated_tracks, association_summary = associate_global_identities(
        detections,
        fps=float(parsed_tracks.fps),
        config=GlobalAssociationConfig(
            expected_players=cfg.expected_players,
            max_gap_fill_frames=cfg.max_gap_fill_frames,
            max_merge_gap_frames=cfg.max_merge_gap_frames,
            max_merge_speed_m_s=cfg.max_merge_speed_m_s,
            appearance_weight=cfg.appearance_weight,
            motion_weight=cfg.motion_weight,
            drop_outside_court=cfg.drop_outside_court,
            court_margin_m=cfg.court_margin_m,
            post_association_court_margin_m=cfg.post_association_court_margin_m,
            cardinality_backfill=cfg.cardinality_backfill,
            backfill_max_cost=cfg.backfill_max_cost,
            backfill_iou_threshold=cfg.backfill_iou_threshold,
            sport=cfg.sport,
        ),
    )
    wall_time_s = time.perf_counter() - started

    output_tracks_path = out / "tracks.json"
    summary_path = out / "offline_authority_summary.json"
    metrics_path = out / "metrics.json"
    output_tracks_path.write_text(
        json.dumps(associated_tracks.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    score: dict[str, Any] | None = None
    score_path: Path | None = None
    if ground_truth_path is not None:
        assert_reid_checkpoint_clip_safe(reid_provenance, clip_id=clip_id)
        parsed_gt = validate_artifact_file("person_ground_truth", Path(ground_truth_path))
        if not isinstance(parsed_gt, PersonGroundTruth):
            raise ValueError(f"{ground_truth_path} did not parse as PersonGroundTruth")
        score = score_tracks_against_person_ground_truth(
            ground_truth=parsed_gt,
            tracks=associated_tracks,
            candidate=f"{candidate}_offline_authority",
            tracks_path=output_tracks_path,
            iou_threshold=cfg.iou_threshold,
            expected_players=cfg.expected_players,
            bbox_scale_x=score_bbox_scale["bbox_scale_x"],
            bbox_scale_y=score_bbox_scale["bbox_scale_y"],
        )
        score_path = out / "person_track_gt_score.json"
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_offline_person_authority_run",
        "status": association_summary.status,
        "clip_id": clip_id,
        "candidate": candidate,
        "source_only": True,
        "uses_cvat_labels_for_candidate": False,
        "ground_truth_scored": score is not None,
        "source_run_dir": str(source_dir),
        "detections_run_dir": str(detection_dir),
        "source_tracks_path": str(tracks_path),
        "source_detections_path": str(detections_path),
        "embedding_bbox_scale": embedding_bbox_scale,
        "score_bbox_scale_x": score_bbox_scale["bbox_scale_x"],
        "score_bbox_scale_y": score_bbox_scale["bbox_scale_y"],
        "score_bbox_scale_source": score_bbox_scale["source"],
        "embedding_export_path": str(output_embedding_export_path),
        "reused_embedding_export": embedding_export_path is not None,
        "tracks_path": str(output_tracks_path),
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "score_path": str(score_path) if score_path is not None else None,
        "wall_time_s": round(wall_time_s, 6),
        "effective_fps": _effective_fps(associated_tracks, wall_time_s),
        "config": asdict(cfg),
        "embedding_export": {
            "feature_type": embedding_payload.get("feature_type"),
            "feature_extractor": embedding_payload.get("feature_extractor"),
            "feature_dim": embedding_payload.get("feature_dim"),
            "detection_count": embedding_payload.get("detection_count"),
            "source_only": embedding_payload.get("source_only"),
            "uses_cvat_labels": embedding_payload.get("uses_cvat_labels"),
        },
        "global_association": association_summary.to_dict(),
        "reid_checkpoint_provenance": reid_provenance,
        "score": score,
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_offline_person_authority_metrics",
                "status": report["status"],
                "clip": clip_id,
                "variant": f"{candidate}_offline_authority",
                "source_run_dir": str(source_dir),
                "detections_run_dir": str(detection_dir),
                "tracks_path": str(output_tracks_path),
                "embedding_export_path": str(output_embedding_export_path),
                "reused_embedding_export": embedding_export_path is not None,
                "wall_time_s": report["wall_time_s"],
                "effective_fps": report["effective_fps"],
                "timing": {
                    "wall_time_s": report["wall_time_s"],
                    "effective_fps": report["effective_fps"],
                },
                "counts": {
                    "source_track_count": len(parsed_tracks.players),
                    "output_player_count": len(associated_tracks.players),
                    "output_track_frame_count": sum(len(player.frames) for player in associated_tracks.players),
                    "score_bbox_scale_x": score_bbox_scale["bbox_scale_x"],
                    "score_bbox_scale_y": score_bbox_scale["bbox_scale_y"],
                    "score_bbox_scale_source": score_bbox_scale["source"],
                },
                "global_association": association_summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _source_detections_path(source_dir: Path) -> Path:
    for filename in ("raw_tracked_detections.json", "tracked_detections.json"):
        path = source_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing tracked_detections.json under {source_dir}")


def _embedding_bbox_scale(source_dir: Path, *, detections_path: Path, configured_scale: float) -> float:
    if detections_path.name != "raw_tracked_detections.json":
        return configured_scale
    scale = _scale_from_metrics(source_dir)
    if scale is not None:
        return _single_bbox_scale(scale["bbox_scale_x"], scale["bbox_scale_y"])
    return configured_scale


def _score_bbox_scale(source_dir: Path) -> dict[str, Any]:
    scale = _scale_from_metrics(source_dir)
    if scale is not None:
        return scale
    return {"bbox_scale_x": 1.0, "bbox_scale_y": 1.0, "source": "identity"}


def _scale_from_metrics(source_dir: Path) -> dict[str, Any] | None:
    metrics_path = source_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    payload = _read_json_object(metrics_path)
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        return None
    source_width = _number(counts.get("source_width"))
    source_height = _number(counts.get("source_height"))
    calibration_width = _number(counts.get("calibration_width"))
    calibration_height = _number(counts.get("calibration_height"))
    if _positive(source_width) and _positive(source_height) and _positive(calibration_width) and _positive(calibration_height):
        return {
            "bbox_scale_x": round(float(source_width / calibration_width), 6),
            "bbox_scale_y": round(float(source_height / calibration_height), 6),
            "source": "metrics_source_over_calibration",
        }
    bbox_scale_x = _number(counts.get("bbox_scale_x"))
    bbox_scale_y = _number(counts.get("bbox_scale_y"))
    if _positive(bbox_scale_x) and _positive(bbox_scale_y):
        return {
            "bbox_scale_x": round(float(1.0 / bbox_scale_x), 6),
            "bbox_scale_y": round(float(1.0 / bbox_scale_y), 6),
            "source": "metrics_inverse_bbox_scale",
        }
    return None


def _single_bbox_scale(scale_x: float, scale_y: float) -> float:
    if abs(scale_x - scale_y) > 1e-3:
        raise ValueError(f"non-uniform raw bbox scale is not supported: x={scale_x:.6f} y={scale_y:.6f}")
    return round(float(scale_x), 6)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _existing_path(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    return str(path) if path.is_file() else None


def _effective_fps(tracks: Tracks, wall_time_s: float) -> float | None:
    if wall_time_s <= 0.0:
        return None
    frame_indexes = {
        int(round(float(frame.t) * float(tracks.fps)))
        for player in tracks.players
        for frame in player.frames
    }
    return round(len(frame_indexes) / wall_time_s, 6)


__all__ = ["OfflineAuthorityConfig", "run_offline_authority_candidate"]
