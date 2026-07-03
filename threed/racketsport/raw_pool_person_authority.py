"""Offline person authority run directly over a raw, pre-role-lock detection pool.

``offline_person_authority.run_offline_authority_candidate`` still reads its
candidate detections from a prior stage's ``tracks.json``, so it can only
re-partition whatever cardinality that stage already locked in -- it cannot
reject a spectator that stage kept, or recover a player detection that stage
dropped. This module instead builds global-association candidates from the
*entire* raw per-frame detection pool (typically far more boxes per frame
than ``expected_players``, including spectators/background), so exactly-N
global selection can do real appearance/motion-based rejection over the full
source pool.

Source-only: ground-truth labels are read only for the optional final score,
never for candidate construction.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .court_templates import Sport, get_court_template
from .person_fast import person_detection_from_bbox
from .person_reid_diagnostics import (
    FeatureExtractor,
    ReIDEmbeddingExportConfig,
    assert_reid_checkpoint_clip_safe,
    build_source_reid_embedding_export,
    reid_checkpoint_training_provenance,
)
from .person_track_gt_scoring import score_tracks_against_person_ground_truth
from .player_global_association import (
    GlobalAssociationConfig,
    associate_global_identities,
    raw_pool_to_global_detections,
)
from .schemas import CourtCalibration, PersonGroundTruth, validate_artifact_file


@dataclass(frozen=True)
class RawPoolAuthorityConfig:
    """Config for the raw-pool (pre-role-lock) person authority run.

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
    max_embedding_bbox_delta_px: float = 2.5
    min_conf: float = 0.0
    split_gap_frames: int = 24
    max_fragment_speed_m_s: float = 12.0
    local_switch_split_distance_m: float = float("inf")
    local_switch_split_embedding_distance: float = float("inf")
    local_switch_split_max_gap_frames: int = 0
    embedding_split_eps: float = 0.35
    embedding_split_min_samples: int = 1
    max_merge_gap_frames: int = 240
    max_merge_speed_m_s: float = 9.0
    max_merge_cost: float = 2.0
    appearance_weight: float = 1.0
    motion_weight: float = 1.0
    side_prior_weight: float = 0.25
    max_fragments_for_global: int = 400
    max_gap_fill_frames: int = 48
    max_gap_fill_speed_m_s: float = 7.0
    cardinality_backfill: bool = False
    backfill_max_cost: float = 2.5
    backfill_iou_threshold: float = 0.25
    drop_outside_court: bool = True
    court_margin_m: float = 2.0
    post_association_court_margin_m: float | None = None
    implausible_motion_speed_threshold_m_s: float = 10.0
    iou_threshold: float = 0.5
    sport: Sport = "pickleball"

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.reid_backend not in {"osnet", "ultralytics_yolo"}:
            raise ValueError("reid_backend must be osnet or ultralytics_yolo")
        if self.reid_batch_size <= 0:
            raise ValueError("reid_batch_size must be positive")
        if self.implausible_motion_speed_threshold_m_s <= 0:
            raise ValueError("implausible_motion_speed_threshold_m_s must be positive")


def run_raw_pool_authority_candidate(
    *,
    clip_id: str,
    candidate: str,
    video_path: str | Path,
    raw_pool_dir: str | Path,
    calibration_path: str | Path,
    out_dir: str | Path,
    reid_model_path: str | Path,
    embedding_export_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    expected_players: int = 4,
    config: RawPoolAuthorityConfig | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> dict[str, Any]:
    cfg = config or RawPoolAuthorityConfig(expected_players=expected_players)
    pool_dir = Path(raw_pool_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration = validate_artifact_file("court_calibration", Path(calibration_path))
    if not isinstance(calibration, CourtCalibration):
        raise ValueError(f"{calibration_path} did not parse as CourtCalibration")

    geometry_path = _geometry_detections_path(pool_dir)
    embedding_source_path = _embedding_source_detections_path(pool_dir)
    geometry_payload = _read_json_object(geometry_path)
    embedding_source_payload = _read_json_object(embedding_source_path)
    fps = _payload_fps(geometry_payload)

    scale = _scale_from_metrics(pool_dir)
    embedding_bbox_scale = _single_bbox_scale(scale) if scale is not None else 1.0
    score_bbox_scale_x = scale["bbox_scale_x"] if scale is not None else 1.0
    score_bbox_scale_y = scale["bbox_scale_y"] if scale is not None else 1.0
    score_image_width = scale.get("source_width") if scale is not None else None
    score_image_height = scale.get("source_height") if scale is not None else None

    started = time.perf_counter()
    if embedding_export_path is not None:
        output_embedding_export_path = Path(embedding_export_path)
        embedding_payload = _read_json_object(output_embedding_export_path)
    else:
        output_embedding_export_path = out / "reid_embeddings.json"
        embedding_payload = build_source_reid_embedding_export(
            video_path=video_path,
            detections_payload=embedding_source_payload,
            output_path=output_embedding_export_path,
            model_path=reid_model_path,
            command_metadata={
                "clip_id": clip_id,
                "candidate": candidate,
                "raw_pool_dir": str(pool_dir),
                "geometry_path": str(geometry_path),
                "embedding_source_path": str(embedding_source_path),
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

    candidate_detections = raw_pool_to_global_detections(
        geometry_payload,
        calibration=calibration,
        embedding_payload=embedding_payload,
        embedding_bbox_scale=embedding_bbox_scale,
        max_embedding_bbox_delta_px=cfg.max_embedding_bbox_delta_px,
        min_conf=cfg.min_conf,
    )
    association_config = GlobalAssociationConfig(
        expected_players=cfg.expected_players,
        split_gap_frames=cfg.split_gap_frames,
        max_fragment_speed_m_s=cfg.max_fragment_speed_m_s,
        local_switch_split_distance_m=cfg.local_switch_split_distance_m,
        local_switch_split_embedding_distance=cfg.local_switch_split_embedding_distance,
        local_switch_split_max_gap_frames=cfg.local_switch_split_max_gap_frames,
        embedding_split_eps=cfg.embedding_split_eps,
        embedding_split_min_samples=cfg.embedding_split_min_samples,
        max_merge_gap_frames=cfg.max_merge_gap_frames,
        max_merge_speed_m_s=cfg.max_merge_speed_m_s,
        max_merge_cost=cfg.max_merge_cost,
        appearance_weight=cfg.appearance_weight,
        motion_weight=cfg.motion_weight,
        side_prior_weight=cfg.side_prior_weight,
        max_fragments_for_global=cfg.max_fragments_for_global,
        max_gap_fill_frames=cfg.max_gap_fill_frames,
        max_gap_fill_speed_m_s=cfg.max_gap_fill_speed_m_s,
        cardinality_backfill=cfg.cardinality_backfill,
        backfill_max_cost=cfg.backfill_max_cost,
        backfill_iou_threshold=cfg.backfill_iou_threshold,
        drop_outside_court=cfg.drop_outside_court,
        court_margin_m=cfg.court_margin_m,
        post_association_court_margin_m=cfg.post_association_court_margin_m,
        sport=cfg.sport,
    )
    associated_tracks, association_summary = associate_global_identities(
        candidate_detections,
        fps=fps,
        config=association_config,
    )
    wall_time_s = time.perf_counter() - started

    output_tracks_path = out / "tracks.json"
    summary_path = out / "raw_pool_authority_summary.json"
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
            candidate=f"{candidate}_raw_pool_authority",
            tracks_path=output_tracks_path,
            iou_threshold=cfg.iou_threshold,
            expected_players=cfg.expected_players,
            bbox_scale_x=score_bbox_scale_x,
            bbox_scale_y=score_bbox_scale_y,
            sport=cfg.sport,
            image_width=score_image_width,
            image_height=score_image_height,
        )
        score_path = out / "person_track_gt_score.json"
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ceiling = raw_pool_four_player_ceiling(
        geometry_payload,
        calibration=calibration,
        expected_players=cfg.expected_players,
        court_margin_m=cfg.court_margin_m,
        sport=cfg.sport,
    )
    cov4_proxy = source_only_cov4_proxy(
        associated_tracks,
        expected_players=cfg.expected_players,
        denominator_frame_count=ceiling["pool_frame_count"],
        denominator_source="raw_pool_frame_count",
    )
    motion_diagnostic = summarize_implausible_motion_spans(
        associated_tracks,
        speed_threshold_m_s=cfg.implausible_motion_speed_threshold_m_s,
    )
    four_player_detection_ceiling = {
        "source_only": True,
        "uses_cvat_labels": False,
        "not_gt": True,
        "value": ceiling["four_player_detection_ceiling"],
        "expected_players": ceiling["expected_players"],
        "frames_with_sufficient_on_court_detections": ceiling["frames_with_sufficient_on_court_detections"],
        "denominator_frame_count": ceiling["pool_frame_count"],
        "court_margin_m": ceiling["court_margin_m"],
        "overlap_iou_threshold": ceiling["overlap_iou_threshold"],
        "notes": [
            "Source-only raw-pool detection ceiling. This does not use GT and does not prove association or identity quality."
        ],
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_raw_pool_person_authority_run",
        "status": association_summary.status,
        "clip_id": clip_id,
        "candidate": candidate,
        "source_only": True,
        "uses_cvat_labels_for_candidate": False,
        "ground_truth_scored": score is not None,
        "raw_pool_dir": str(pool_dir),
        "geometry_path": str(geometry_path),
        "embedding_source_path": str(embedding_source_path),
        "calibration_path": str(calibration_path),
        "embedding_bbox_scale": embedding_bbox_scale,
        "score_bbox_scale_x": score_bbox_scale_x,
        "score_bbox_scale_y": score_bbox_scale_y,
        "score_image_width": score_image_width,
        "score_image_height": score_image_height,
        "embedding_export_path": str(output_embedding_export_path),
        "reused_embedding_export": embedding_export_path is not None,
        "tracks_path": str(output_tracks_path),
        "summary_path": str(summary_path),
        "score_path": str(score_path) if score_path is not None else None,
        "wall_time_s": round(wall_time_s, 6),
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
        "cov4_proxy": cov4_proxy,
        "four_player_detection_ceiling": four_player_detection_ceiling,
        "detection_limited_ceiling": ceiling,
        "implausible_motion_speed_threshold_m_s": motion_diagnostic["speed_threshold_m_s"],
        "implausible_motion_spans": motion_diagnostic["implausible_motion_spans"],
        "implausible_motion_span_count": motion_diagnostic["implausible_motion_span_count"],
        "implausible_motion_step_count": motion_diagnostic["implausible_motion_step_count"],
        "implausible_motion_player_count": motion_diagnostic["implausible_motion_player_count"],
        "implausible_motion_worst_offender": motion_diagnostic["implausible_motion_worst_offender"],
        "implausible_motion_diagnostic": motion_diagnostic,
        "reid_checkpoint_provenance": reid_provenance,
        "score": score,
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def source_only_cov4_proxy(
    tracks: Any,
    *,
    expected_players: int = 4,
    denominator_frame_count: int | None = None,
    denominator_source: str | None = None,
) -> dict[str, Any]:
    """Count exact-N output-track frames without consulting ground truth."""
    if expected_players <= 0:
        raise ValueError("expected_players must be positive")
    fps = float(getattr(tracks, "fps", 0.0) or 0.0)
    if fps <= 0.0:
        raise ValueError("tracks must contain a positive fps")

    frame_players: dict[int, set[int]] = {}
    for player in getattr(tracks, "players", []):
        player_id = int(getattr(player, "id"))
        for frame in getattr(player, "frames", []):
            frame_idx = int(round(float(getattr(frame, "t")) * fps))
            frame_players.setdefault(frame_idx, set()).add(player_id)

    if frame_players:
        first_frame = min(frame_players)
        last_frame = max(frame_players)
        observed_span = last_frame - first_frame + 1
        exact_frames = sum(
            1
            for frame_idx in range(first_frame, last_frame + 1)
            if len(frame_players.get(frame_idx, set())) == expected_players
        )
        histogram: dict[int, int] = {}
        for frame_idx in range(first_frame, last_frame + 1):
            count = len(frame_players.get(frame_idx, set()))
            histogram[count] = histogram.get(count, 0) + 1
    else:
        first_frame = None
        last_frame = None
        observed_span = 0
        exact_frames = 0
        histogram = {}

    denominator = int(denominator_frame_count) if denominator_frame_count is not None else observed_span
    if denominator < 0:
        raise ValueError("denominator_frame_count must be non-negative")
    value = round(exact_frames / denominator, 6) if denominator else 0.0
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_source_only_cov4_proxy",
        "source_only": True,
        "uses_cvat_labels": False,
        "not_gt": True,
        "expected_players": expected_players,
        "value": value,
        "cov4_proxy": value,
        "exact_expected_player_frames": exact_frames,
        "denominator_frame_count": denominator,
        "denominator_source": denominator_source or ("observed_track_frame_span" if denominator_frame_count is None else "provided"),
        "observed_frame_span": observed_span,
        "observed_frame_count": len(frame_players),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "output_player_count_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "notes": [
            "Source-only proxy: counts frames where the output tracks contain exactly expected_players IDs.",
            "This is not GT coverage, IDF1, or a promotion gate; it does not know whether the boxes match real players.",
        ],
    }


def summarize_implausible_motion_spans(
    tracks: Any,
    *,
    speed_threshold_m_s: float = 10.0,
) -> dict[str, Any]:
    """Group consecutive output-track steps whose world_xy speed is physically implausible."""
    if speed_threshold_m_s <= 0.0:
        raise ValueError("speed_threshold_m_s must be positive")
    fps = float(getattr(tracks, "fps", 0.0) or 0.0)
    if fps <= 0.0:
        raise ValueError("tracks must contain a positive fps")

    spans: list[dict[str, Any]] = []
    for player in getattr(tracks, "players", []):
        player_id = int(getattr(player, "id"))
        frames = sorted(getattr(player, "frames", []), key=lambda frame: float(getattr(frame, "t")))
        current_span: dict[str, Any] | None = None
        previous_frame: Any | None = None
        for frame in frames:
            if previous_frame is None:
                previous_frame = frame
                continue
            step = _implausible_motion_step(
                previous_frame=previous_frame,
                frame=frame,
                fps=fps,
                speed_threshold_m_s=speed_threshold_m_s,
            )
            if step is None:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                previous_frame = frame
                continue
            if current_span is not None and current_span["end_frame"] == step["from_frame"]:
                current_span["end_frame"] = step["to_frame"]
                current_span["end_t"] = step["to_t"]
                current_span["step_count"] += 1
                if step["speed_m_s"] > current_span["max_speed_m_s"]:
                    current_span["max_speed_m_s"] = step["speed_m_s"]
                    current_span["max_step"] = step
            else:
                if current_span is not None:
                    spans.append(current_span)
                current_span = {
                    "player_id": player_id,
                    "start_frame": step["from_frame"],
                    "end_frame": step["to_frame"],
                    "start_t": step["from_t"],
                    "end_t": step["to_t"],
                    "step_count": 1,
                    "max_speed_m_s": step["speed_m_s"],
                    "max_step": step,
                }
            previous_frame = frame
        if current_span is not None:
            spans.append(current_span)

    step_count = sum(int(span["step_count"]) for span in spans)
    player_ids = sorted({int(span["player_id"]) for span in spans})
    worst = max(spans, key=lambda span: float(span["max_speed_m_s"])) if spans else None
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_implausible_motion_diagnostic",
        "source_only": True,
        "uses_cvat_labels": False,
        "not_gt": True,
        "speed_threshold_m_s": float(speed_threshold_m_s),
        "implausible_motion_spans": spans,
        "implausible_motion_span_count": len(spans),
        "implausible_motion_step_count": step_count,
        "implausible_motion_player_count": len(player_ids),
        "implausible_motion_player_ids": player_ids,
        "implausible_motion_worst_offender": worst,
        "notes": [
            "Diagnostic-only world_xy speed gate. It flags output-track motion that should not feed foot-lock or replay as physically trustworthy.",
            "No smoothing, holding, deletion, or identity repair is applied by this diagnostic.",
        ],
    }


def _implausible_motion_step(
    *,
    previous_frame: Any,
    frame: Any,
    fps: float,
    speed_threshold_m_s: float,
) -> dict[str, Any] | None:
    previous_xy = _world_xy(previous_frame)
    current_xy = _world_xy(frame)
    if previous_xy is None or current_xy is None:
        return None
    previous_t = float(getattr(previous_frame, "t"))
    current_t = float(getattr(frame, "t"))
    dt_s = current_t - previous_t
    if dt_s <= 0.0:
        return None
    distance_m = math.hypot(current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1])
    speed_m_s = distance_m / dt_s
    if speed_m_s <= speed_threshold_m_s:
        return None
    return {
        "from_frame": int(round(previous_t * fps)),
        "to_frame": int(round(current_t * fps)),
        "from_t": round(previous_t, 6),
        "to_t": round(current_t, 6),
        "dt_s": round(dt_s, 6),
        "distance_m": round(distance_m, 6),
        "speed_m_s": round(speed_m_s, 6),
        "from_world_xy": [round(previous_xy[0], 6), round(previous_xy[1], 6)],
        "to_world_xy": [round(current_xy[0], 6), round(current_xy[1], 6)],
    }


def _world_xy(frame: Any) -> tuple[float, float] | None:
    world_xy = getattr(frame, "world_xy", None)
    if world_xy is None or len(world_xy) != 2:
        return None
    return (float(world_xy[0]), float(world_xy[1]))


def raw_pool_four_player_ceiling(
    detections_payload: dict[str, Any],
    *,
    calibration: CourtCalibration,
    expected_players: int = 4,
    court_margin_m: float = 0.0,
    overlap_iou_threshold: float = 0.3,
    sport: Sport = "pickleball",
) -> dict[str, Any]:
    """Source-only diagnostic: fraction of raw-pool frames with >= N on-court, non-overlapping boxes.

    This bounds how much of the residual coverage/FN gap is a *detection*
    problem (the pool never had enough distinct player-shaped boxes on a
    frame) versus an *association* problem (the boxes existed but selection
    failed to pick them). It never reads ground truth.
    """
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0 + court_margin_m
    half_length_m = template.length_m / 2.0 + court_margin_m

    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")

    frame_count = 0
    sufficient_frame_count = 0
    max_on_court_histogram: dict[int, int] = {}
    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, dict):
            continue
        frame_count += 1
        raw_detections = frame_entry.get("detections", [])
        if not isinstance(raw_detections, list):
            raw_detections = []
        on_court: list[tuple[float, tuple[float, float, float, float]]] = []
        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue
            value = detection.get("class", "person")
            if not (value == 0 or str(value).lower() in {"person", "player", "0"}):
                continue
            bbox_raw = detection.get("bbox") or detection.get("bbox_xyxy")
            if not isinstance(bbox_raw, list | tuple) or len(bbox_raw) != 4:
                continue
            bbox = tuple(float(value) for value in bbox_raw)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            conf = float(detection.get("conf", detection.get("confidence", 1.0)))
            person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=conf)
            x, y = person.foot_world_xy
            if -half_width_m <= x <= half_width_m and -half_length_m <= y <= half_length_m:
                on_court.append((conf, bbox))
        selected = _greedy_non_overlapping(on_court, iou_threshold=overlap_iou_threshold)
        max_on_court_histogram[selected] = max_on_court_histogram.get(selected, 0) + 1
        if selected >= expected_players:
            sufficient_frame_count += 1

    ceiling_fraction = round(sufficient_frame_count / frame_count, 6) if frame_count else 0.0
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_raw_pool_four_player_ceiling",
        "source_only": True,
        "uses_cvat_labels": False,
        "expected_players": expected_players,
        "overlap_iou_threshold": overlap_iou_threshold,
        "court_margin_m": court_margin_m,
        "pool_frame_count": frame_count,
        "frames_with_sufficient_on_court_detections": sufficient_frame_count,
        "four_player_detection_ceiling": ceiling_fraction,
        "on_court_count_histogram": {str(key): value for key, value in sorted(max_on_court_histogram.items())},
        "notes": [
            "Counts frames where the raw pool contains at least `expected_players` "
            "on-court, mutually non-overlapping person boxes (greedy NMS by confidence). "
            "This is a source-only upper bound on association-stage coverage: no "
            "association/selection logic and no ground truth are involved.",
        ],
    }


def _greedy_non_overlapping(
    boxes: list[tuple[float, tuple[float, float, float, float]]],
    *,
    iou_threshold: float,
) -> int:
    ordered = sorted(boxes, key=lambda item: -item[0])
    kept: list[tuple[float, float, float, float]] = []
    for _conf, bbox in ordered:
        if all(_bbox_iou(bbox, other) <= iou_threshold for other in kept):
            kept.append(bbox)
    return len(kept)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _geometry_detections_path(pool_dir: Path) -> Path:
    for filename in ("tracked_detections.json", "raw_tracked_detections.json"):
        path = pool_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing tracked_detections.json under {pool_dir}")


def _embedding_source_detections_path(pool_dir: Path) -> Path:
    for filename in ("raw_tracked_detections.json", "tracked_detections.json"):
        path = pool_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing raw_tracked_detections.json under {pool_dir}")


def _payload_fps(payload: dict[str, Any]) -> float:
    fps = payload.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("detections payload must contain a positive fps")
    return float(fps)


def _scale_from_metrics(pool_dir: Path) -> dict[str, Any] | None:
    metrics_path = pool_dir / "metrics.json"
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
            "source_width": float(source_width),
            "source_height": float(source_height),
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


def _single_bbox_scale(scale: dict[str, Any]) -> float:
    scale_x = float(scale["bbox_scale_x"])
    scale_y = float(scale["bbox_scale_y"])
    if abs(scale_x - scale_y) > 1e-3:
        raise ValueError(f"non-uniform raw bbox scale is not supported: x={scale_x:.6f} y={scale_y:.6f}")
    return round(scale_x, 6)


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


__all__ = [
    "RawPoolAuthorityConfig",
    "raw_pool_four_player_ceiling",
    "run_raw_pool_authority_candidate",
    "source_only_cov4_proxy",
    "summarize_implausible_motion_spans",
]
