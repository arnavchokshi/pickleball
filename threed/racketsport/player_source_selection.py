"""Source-only player selection from tracked detector pools."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
import math
from typing import Any, Sequence

from .person_fast import court_polygon_filter, person_detection_from_bbox
from .schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks

_EMBEDDING_BBOX_MAX_DELTA_PX = 1.0


@dataclass(frozen=True)
class SourceSelectionConfig:
    expected_players: int = 4
    min_detection_conf: float = 0.0
    court_margin_m: float = 0.0
    nms_iou_threshold: float = 0.8
    max_gap_fill_frames: int = 30
    max_fill_distance_m: float = 1.25
    overlap_iou_threshold: float = 0.5
    fill_confidence_cap: float = 0.45
    max_global_step_m: float = 2.5
    continuity_weight: float = 1.0
    source_id_switch_penalty: float = 1.5
    seed_prior_weight: float = 0.75
    cardinality_gap_penalty: float = 8.0
    confidence_reward_weight: float = 0.25
    embedding_weight: float = 0.0
    embedding_bbox_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.max_gap_fill_frames < 0:
            raise ValueError("max_gap_fill_frames must be non-negative")
        if self.max_fill_distance_m < 0.0:
            raise ValueError("max_fill_distance_m must be non-negative")
        if self.max_global_step_m < 0.0:
            raise ValueError("max_global_step_m must be non-negative")
        if self.embedding_weight < 0.0:
            raise ValueError("embedding_weight must be non-negative")
        if self.embedding_bbox_scale <= 0.0:
            raise ValueError("embedding_bbox_scale must be positive")


@dataclass(frozen=True)
class SourceSelectionSummary:
    status: str
    source_candidate_detections: int
    source_candidate_kept: int
    seed_player_count: int
    seed_frame_count: int
    output_player_count: int
    output_frame_count: int
    filled_frame_count: int
    skipped_overlap_count: int
    skipped_distance_count: int
    global_selected_frame_count: int = 0
    exact_cardinality_frame_count: int = 0
    identity_reassignment_count: int = 0
    uses_embeddings: bool = False
    embedding_candidate_count: int = 0
    embedding_joined_count: int = 0
    embedding_missing_count: int = 0
    embedding_cost_applied_count: int = 0
    uses_cvat_labels: bool = False
    source_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CandidateDetection:
    frame_idx: int
    source_track_id: int
    detection_index: int
    bbox: tuple[float, float, float, float]
    world_xy: tuple[float, float]
    conf: float
    embedding: tuple[float, ...] | None = None


@dataclass
class _LaneState:
    player_id: int
    side: str
    role: str
    preferred_source_track_id: int | None
    last_frame_idx: int | None = None
    last_world_xy: tuple[float, float] | None = None
    last_source_track_id: int | None = None
    last_embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class _EmbeddingRecord:
    vector: tuple[float, ...]
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class _EmbeddingJoinStats:
    candidate_count: int = 0
    joined_count: int = 0
    missing_count: int = 0


def source_select_four_player_tracks(
    detections_payload: dict[str, Any],
    calibration: CourtCalibration,
    *,
    seed_tracks: Tracks,
    embedding_payload: dict[str, Any] | None = None,
    config: SourceSelectionConfig | None = None,
) -> tuple[Tracks, SourceSelectionSummary]:
    """Fill short seed-track gaps from tracked detections without using labels."""

    cfg = config or SourceSelectionConfig()
    embedding_index = _embedding_index(embedding_payload)
    candidates_by_frame, source_count, kept_count, embedding_stats = _candidate_detections_by_frame(
        detections_payload,
        calibration,
        config=cfg,
        embedding_index=embedding_index,
    )
    fps = float(seed_tracks.fps)
    if fps <= 0.0:
        raise ValueError("seed_tracks fps must be positive")

    seed_frame_count = sum(len(player.frames) for player in seed_tracks.players)
    existing_by_frame = _tracks_by_frame(seed_tracks, fps=fps)
    output_players: list[PlayerTrack] = []
    filled_frame_count = 0
    skipped_overlap_count = 0
    skipped_distance_count = 0

    for player in sorted(seed_tracks.players, key=lambda item: int(item.id)):
        frames = {
            int(round(float(frame.t) * fps)): TrackFrame(
                t=float(frame.t),
                bbox=tuple(float(value) for value in frame.bbox),  # type: ignore[arg-type]
                world_xy=tuple(float(value) for value in frame.world_xy),  # type: ignore[arg-type]
                conf=float(frame.conf),
            )
            for frame in player.frames
        }
        seed_frame_indices = sorted(frames)
        for left_frame, right_frame in zip(seed_frame_indices, seed_frame_indices[1:], strict=False):
            gap = right_frame - left_frame
            if gap <= 1 or gap - 1 > cfg.max_gap_fill_frames:
                continue
            left = frames[left_frame]
            right = frames[right_frame]
            for frame_idx in range(left_frame + 1, right_frame):
                if frame_idx in frames:
                    continue
                existing = [
                    frame
                    for track_id, frame in existing_by_frame.get(frame_idx, [])
                    if track_id != int(player.id)
                ]
                existing.extend(frame for existing_frame_idx, frame in frames.items() if existing_frame_idx == frame_idx)
                candidate, rejected_overlap, rejected_distance = _best_fill_candidate(
                    frame_idx,
                    left_frame=left_frame,
                    right_frame=right_frame,
                    left=left,
                    right=right,
                    candidates=candidates_by_frame.get(frame_idx, []),
                    existing_frames=existing,
                    config=cfg,
                )
                skipped_overlap_count += rejected_overlap
                skipped_distance_count += rejected_distance
                if candidate is None:
                    continue
                filled = TrackFrame(
                    t=frame_idx / fps,
                    bbox=candidate.bbox,
                    world_xy=candidate.world_xy,
                    conf=max(0.0, min(cfg.fill_confidence_cap, candidate.conf)),
                )
                frames[frame_idx] = filled
                existing_by_frame[frame_idx].append((int(player.id), filled))
                filled_frame_count += 1

        output_players.append(
            PlayerTrack(
                id=int(player.id),
                side=player.side,
                role=player.role,
                frames=[frames[frame_idx] for frame_idx in sorted(frames)],
            )
        )

    output = Tracks(schema_version=1, fps=fps, players=output_players, rally_spans=list(seed_tracks.rally_spans))
    output_frame_count = sum(len(player.frames) for player in output.players)
    status = "ok" if len(output.players) == cfg.expected_players else "unexpected_player_count"
    return output, SourceSelectionSummary(
        status=status,
        source_candidate_detections=source_count,
        source_candidate_kept=kept_count,
        seed_player_count=len(seed_tracks.players),
        seed_frame_count=seed_frame_count,
        output_player_count=len(output.players),
        output_frame_count=output_frame_count,
        filled_frame_count=filled_frame_count,
        skipped_overlap_count=skipped_overlap_count,
        skipped_distance_count=skipped_distance_count,
        uses_embeddings=embedding_payload is not None,
        embedding_candidate_count=embedding_stats.candidate_count,
        embedding_joined_count=embedding_stats.joined_count,
        embedding_missing_count=embedding_stats.missing_count,
    )


def source_select_global_four_player_tracks(
    detections_payload: dict[str, Any],
    calibration: CourtCalibration,
    *,
    seed_tracks: Tracks | None = None,
    embedding_payload: dict[str, Any] | None = None,
    config: SourceSelectionConfig | None = None,
) -> tuple[Tracks, SourceSelectionSummary]:
    """Select bounded source-only four-player tracks from every source frame.

    The selector never reads labels. Seed tracks, when supplied, are only a soft
    model-output prior for lane identity and position; source detections can
    replace the seed on any frame when continuity/cardinality costs prefer them.
    """

    cfg = config or SourceSelectionConfig()
    embedding_index = _embedding_index(embedding_payload)
    candidates_by_frame, source_count, kept_count, embedding_stats = _candidate_detections_by_frame(
        detections_payload,
        calibration,
        config=cfg,
        embedding_index=embedding_index,
    )
    fps = _selection_fps(detections_payload, seed_tracks=seed_tracks)
    lanes = _initial_lane_states(
        candidates_by_frame,
        seed_tracks=seed_tracks,
        expected_players=cfg.expected_players,
    )
    seed_by_lane_frame = _seed_frames_by_lane(seed_tracks, fps=fps) if seed_tracks is not None else {}
    frames_by_player: dict[int, dict[int, TrackFrame]] = {lane.player_id: {} for lane in lanes}
    selected_frame_count = 0
    exact_cardinality_frame_count = 0
    identity_reassignment_count = 0
    embedding_cost_applied_count = 0
    skipped_overlap_count = max(0, source_count - kept_count)
    skipped_distance_count = 0

    for frame_idx in sorted(candidates_by_frame):
        candidates = candidates_by_frame.get(frame_idx, [])
        selected, assignment, distance_rejections, overlap_rejections = _best_global_frame_assignment(
            frame_idx,
            candidates=candidates,
            lanes=lanes,
            seed_by_lane_frame=seed_by_lane_frame,
            config=cfg,
        )
        skipped_distance_count += distance_rejections
        skipped_overlap_count += overlap_rejections
        if not assignment:
            continue
        if len(assignment) == cfg.expected_players:
            exact_cardinality_frame_count += 1
        selected_frame_count += len(assignment)
        for lane_idx, candidate in assignment:
            lane = lanes[lane_idx]
            if cfg.embedding_weight > 0.0 and lane.last_embedding is not None and candidate.embedding is not None:
                embedding_cost_applied_count += 1
            if lane.last_source_track_id is not None and lane.last_source_track_id != candidate.source_track_id:
                identity_reassignment_count += 1
            lane.last_frame_idx = frame_idx
            lane.last_world_xy = candidate.world_xy
            lane.last_source_track_id = candidate.source_track_id
            if candidate.embedding is not None:
                lane.last_embedding = candidate.embedding
            frames_by_player[lane.player_id][frame_idx] = TrackFrame(
                t=frame_idx / fps,
                bbox=candidate.bbox,
                world_xy=candidate.world_xy,
                conf=candidate.conf,
            )

    players = [
        PlayerTrack(
            id=lane.player_id,
            side=lane.side,
            role=lane.role,
            frames=[frames_by_player[lane.player_id][frame_idx] for frame_idx in sorted(frames_by_player[lane.player_id])],
        )
        for lane in lanes
    ]
    output = Tracks(
        schema_version=1,
        fps=fps,
        players=players,
        rally_spans=list(seed_tracks.rally_spans) if seed_tracks is not None else [],
    )
    output_frame_count = sum(len(player.frames) for player in output.players)
    status = "ok" if len(output.players) == cfg.expected_players else "unexpected_player_count"
    return output, SourceSelectionSummary(
        status=status,
        source_candidate_detections=source_count,
        source_candidate_kept=kept_count,
        seed_player_count=len(seed_tracks.players) if seed_tracks is not None else 0,
        seed_frame_count=sum(len(player.frames) for player in seed_tracks.players) if seed_tracks is not None else 0,
        output_player_count=len(output.players),
        output_frame_count=output_frame_count,
        filled_frame_count=max(0, output_frame_count - (sum(len(player.frames) for player in seed_tracks.players) if seed_tracks is not None else 0)),
        skipped_overlap_count=skipped_overlap_count,
        skipped_distance_count=skipped_distance_count,
        global_selected_frame_count=selected_frame_count,
        exact_cardinality_frame_count=exact_cardinality_frame_count,
        identity_reassignment_count=identity_reassignment_count,
        uses_embeddings=embedding_payload is not None,
        embedding_candidate_count=embedding_stats.candidate_count,
        embedding_joined_count=embedding_stats.joined_count,
        embedding_missing_count=embedding_stats.missing_count,
        embedding_cost_applied_count=embedding_cost_applied_count,
    )


def _candidate_detections_by_frame(
    detections_payload: dict[str, Any],
    calibration: CourtCalibration,
    *,
    config: SourceSelectionConfig,
    embedding_index: dict[tuple[int, int, int], _EmbeddingRecord],
) -> tuple[dict[int, list[_CandidateDetection]], int, int, _EmbeddingJoinStats]:
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")

    by_frame: dict[int, list[_CandidateDetection]] = {}
    source_count = 0
    kept_count = 0
    embedding_joined_count = 0
    embedding_missing_count = 0
    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, dict):
            raise ValueError("each frame entry must be an object")
        frame_idx = int(frame_entry.get("frame", frame_entry.get("frame_index", default_frame_idx)))
        detections = frame_entry.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("frame detections must be a list")
        candidates: list[_CandidateDetection] = []
        for det_idx, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValueError("each detection must be an object")
            source_count += 1
            if not _is_person_detection(detection):
                continue
            conf = float(detection.get("conf", detection.get("confidence", 1.0)))
            if conf < config.min_detection_conf:
                continue
            bbox_xyxy = _bbox_xyxy(detection)
            source_track_id = _track_id(detection, det_idx + 1)
            person = person_detection_from_bbox(
                calibration,
                bbox_xyxy=bbox_xyxy,
                confidence=conf,
            )
            if not court_polygon_filter([person], sport=calibration.sport, margin_m=config.court_margin_m):
                continue
            embedding: tuple[float, ...] | None = None
            if embedding_index:
                embedding_record = embedding_index.get((frame_idx, source_track_id, det_idx))
                if embedding_record is None:
                    embedding_missing_count += 1
                else:
                    if embedding_record.bbox is not None:
                        scaled_bbox = tuple(float(value) * config.embedding_bbox_scale for value in bbox_xyxy)
                        bbox_delta = max(abs(float(left) - float(right)) for left, right in zip(scaled_bbox, embedding_record.bbox, strict=True))
                        if bbox_delta > _EMBEDDING_BBOX_MAX_DELTA_PX:
                            raise ValueError(
                                "embedding bbox mismatch for "
                                f"frame={frame_idx} source_track_id={source_track_id} detection_index={det_idx}"
                            )
                    embedding = embedding_record.vector
                    embedding_joined_count += 1
            candidates.append(
                _CandidateDetection(
                    frame_idx=frame_idx,
                    source_track_id=source_track_id,
                    detection_index=det_idx,
                    bbox=tuple(float(value) for value in person.bbox_xyxy),  # type: ignore[arg-type]
                    world_xy=tuple(float(value) for value in person.foot_world_xy),  # type: ignore[arg-type]
                    conf=float(person.confidence),
                    embedding=embedding,
                )
            )
        kept = _nms(candidates, config.nms_iou_threshold)
        by_frame[frame_idx] = kept
        kept_count += len(kept)
    return by_frame, source_count, kept_count, _EmbeddingJoinStats(
        candidate_count=len(embedding_index),
        joined_count=embedding_joined_count,
        missing_count=embedding_missing_count,
    )


def _embedding_index(embedding_payload: dict[str, Any] | None) -> dict[tuple[int, int, int], _EmbeddingRecord]:
    if embedding_payload is None:
        return {}
    if bool(embedding_payload.get("uses_cvat_labels", False)):
        raise ValueError("embedding payload must be source-only and cannot use CVAT labels")
    if embedding_payload.get("source_only") is False:
        raise ValueError("embedding payload must be source-only")
    if bool(embedding_payload.get("promote_trk", False)):
        raise ValueError("embedding payload cannot be a promoted TRK artifact")

    rows = embedding_payload.get("detections")
    if not isinstance(rows, list):
        raise ValueError("embedding payload must contain a detections list")
    global_feature_dim = embedding_payload.get("feature_dim")
    expected_dim = int(global_feature_dim) if global_feature_dim is not None else None
    l2_normalized = bool(embedding_payload.get("l2_normalized", False))

    index: dict[tuple[int, int, int], _EmbeddingRecord] = {}
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("embedding detection rows must be objects")
        frame_idx = _int_field(row, ("frame", "frame_idx", "frame_index"), f"embedding row {row_idx} frame")
        source_track_id = _int_field(row, ("source_track_id", "track_id"), f"embedding row {row_idx} source_track_id")
        detection_index = _int_field(row, ("detection_index", "det_idx"), f"embedding row {row_idx} detection_index")
        embedding = _embedding_vector(row.get("embedding"), expected_dim=expected_dim, l2_normalized=l2_normalized, row_idx=row_idx)
        bbox = _optional_bbox(row.get("bbox", row.get("bbox_xyxy")))
        key = (frame_idx, source_track_id, detection_index)
        if key in index:
            raise ValueError(
                "duplicate embedding row for "
                f"frame={frame_idx} source_track_id={source_track_id} detection_index={detection_index}"
            )
        index[key] = _EmbeddingRecord(vector=embedding, bbox=bbox)
    return index


def _int_field(row: dict[str, Any], keys: Sequence[str], description: str) -> int:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
    raise ValueError(f"{description} must be an integer")


def _embedding_vector(
    raw: Any,
    *,
    expected_dim: int | None,
    l2_normalized: bool,
    row_idx: int,
) -> tuple[float, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise ValueError(f"embedding row {row_idx} must contain a non-empty embedding")
    vector = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"embedding row {row_idx} contains a non-finite value")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"embedding row {row_idx} feature_dim mismatch")
    if l2_normalized:
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"embedding row {row_idx} must be L2-normalized")
    return vector


def _optional_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("embedding bbox must contain four xyxy values when present")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("embedding bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _selection_fps(detections_payload: dict[str, Any], *, seed_tracks: Tracks | None) -> float:
    if seed_tracks is not None:
        fps = float(seed_tracks.fps)
    else:
        fps = float(detections_payload.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError("selection fps must be positive")
    return fps


def _initial_lane_states(
    candidates_by_frame: dict[int, list[_CandidateDetection]],
    *,
    seed_tracks: Tracks | None,
    expected_players: int,
) -> list[_LaneState]:
    if seed_tracks is not None and seed_tracks.players:
        seed_players = sorted(seed_tracks.players, key=lambda player: int(player.id))[:expected_players]
        lanes = [
            _LaneState(
                player_id=int(player.id),
                side=player.side,
                role=player.role,
                preferred_source_track_id=int(player.id),
            )
            for player in seed_players
        ]
        next_id = max((lane.player_id for lane in lanes), default=0) + 1
        while len(lanes) < expected_players:
            lanes.append(_fallback_lane(next_id, len(lanes)))
            next_id += 1
        return lanes

    source_scores: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "confidence": 0.0})
    for candidates in candidates_by_frame.values():
        for candidate in candidates:
            source_scores[candidate.source_track_id]["count"] += 1.0
            source_scores[candidate.source_track_id]["confidence"] += candidate.conf
    source_ids = [
        source_id
        for source_id, _score in sorted(
            source_scores.items(),
            key=lambda item: (-item[1]["count"], -item[1]["confidence"], item[0]),
        )
    ][:expected_players]
    lanes = [
        _LaneState(
            player_id=int(source_id),
            side="unknown",
            role="unknown",
            preferred_source_track_id=int(source_id),
        )
        for source_id in source_ids
    ]
    next_id = 1
    while len(lanes) < expected_players:
        while any(lane.player_id == next_id for lane in lanes):
            next_id += 1
        lanes.append(_fallback_lane(next_id, len(lanes)))
        next_id += 1
    return lanes


def _fallback_lane(player_id: int, lane_index: int) -> _LaneState:
    defaults = [("near", "left"), ("near", "right"), ("far", "left"), ("far", "right")]
    side, role = defaults[lane_index] if lane_index < len(defaults) else ("unknown", "unknown")
    return _LaneState(player_id=player_id, side=side, role=role, preferred_source_track_id=None)


def _seed_frames_by_lane(seed_tracks: Tracks | None, *, fps: float) -> dict[tuple[int, int], TrackFrame]:
    if seed_tracks is None:
        return {}
    frames: dict[tuple[int, int], TrackFrame] = {}
    for player in seed_tracks.players:
        for frame in player.frames:
            frames[(int(player.id), int(round(float(frame.t) * fps)))] = frame
    return frames


def _best_global_frame_assignment(
    frame_idx: int,
    *,
    candidates: Sequence[_CandidateDetection],
    lanes: Sequence[_LaneState],
    seed_by_lane_frame: dict[tuple[int, int], TrackFrame],
    config: SourceSelectionConfig,
) -> tuple[list[_CandidateDetection], list[tuple[int, _CandidateDetection]], int, int]:
    if not candidates:
        return [], [], 0, 0

    max_count = min(config.expected_players, len(candidates), len(lanes))
    best_assignment: list[tuple[int, _CandidateDetection]] = []
    best_selected: list[_CandidateDetection] = []
    best_cost = float("inf")
    distance_rejections = 0
    overlap_rejections = 0

    for count in range(max_count, 0, -1):
        for candidate_group in combinations(candidates, count):
            if _has_overlapping_pair(candidate_group, config.overlap_iou_threshold):
                overlap_rejections += 1
                continue
            if len({candidate.source_track_id for candidate in candidate_group}) != len(candidate_group):
                overlap_rejections += 1
                continue
            for lane_indices in permutations(range(len(lanes)), count):
                cost = config.cardinality_gap_penalty * float((config.expected_players - count) ** 2)
                assignment: list[tuple[int, _CandidateDetection]] = []
                rejected = False
                for lane_idx, candidate in zip(lane_indices, candidate_group, strict=True):
                    lane = lanes[lane_idx]
                    lane_cost = _global_lane_candidate_cost(
                        frame_idx,
                        lane=lane,
                        candidate=candidate,
                        seed_by_lane_frame=seed_by_lane_frame,
                        config=config,
                    )
                    if lane_cost is None:
                        distance_rejections += 1
                        rejected = True
                        break
                    cost += lane_cost
                    assignment.append((lane_idx, candidate))
                if rejected:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment
                    best_selected = list(candidate_group)
        if best_assignment:
            break

    return best_selected, best_assignment, distance_rejections, overlap_rejections


def _global_lane_candidate_cost(
    frame_idx: int,
    *,
    lane: _LaneState,
    candidate: _CandidateDetection,
    seed_by_lane_frame: dict[tuple[int, int], TrackFrame],
    config: SourceSelectionConfig,
) -> float | None:
    cost = -config.confidence_reward_weight * candidate.conf
    if lane.preferred_source_track_id is not None and lane.preferred_source_track_id != candidate.source_track_id:
        cost += 0.5 * config.source_id_switch_penalty
    if lane.last_world_xy is not None and lane.last_frame_idx is not None:
        gap = max(1, frame_idx - lane.last_frame_idx)
        distance = _point_distance(lane.last_world_xy, candidate.world_xy)
        max_distance = config.max_global_step_m * gap
        if distance > max_distance:
            return None
        cost += config.continuity_weight * distance / gap
    if lane.last_source_track_id is not None and lane.last_source_track_id != candidate.source_track_id:
        cost += config.source_id_switch_penalty
    if config.embedding_weight > 0.0 and lane.last_embedding is not None and candidate.embedding is not None:
        cost += config.embedding_weight * _cosine_distance(lane.last_embedding, candidate.embedding)

    seed_frame = seed_by_lane_frame.get((lane.player_id, frame_idx))
    if seed_frame is not None:
        cost += config.seed_prior_weight * _point_distance(seed_frame.world_xy, candidate.world_xy)
    return cost


def _has_overlapping_pair(candidates: Sequence[_CandidateDetection], iou_threshold: float) -> bool:
    for idx, candidate in enumerate(candidates):
        for other in candidates[idx + 1 :]:
            if _bbox_iou(candidate.bbox, other.bbox) > iou_threshold:
                return True
    return False


def _best_fill_candidate(
    frame_idx: int,
    *,
    left_frame: int,
    right_frame: int,
    left: TrackFrame,
    right: TrackFrame,
    candidates: Sequence[_CandidateDetection],
    existing_frames: Sequence[TrackFrame],
    config: SourceSelectionConfig,
) -> tuple[_CandidateDetection | None, int, int]:
    if not candidates:
        return None, 0, 0
    alpha = (frame_idx - left_frame) / (right_frame - left_frame)
    left_xy = tuple(float(value) for value in left.world_xy)
    right_xy = tuple(float(value) for value in right.world_xy)
    predicted = (
        (1.0 - alpha) * left_xy[0] + alpha * right_xy[0],
        (1.0 - alpha) * left_xy[1] + alpha * right_xy[1],
    )

    best: _CandidateDetection | None = None
    best_cost = float("inf")
    rejected_overlap = 0
    rejected_distance = 0
    for candidate in sorted(candidates, key=lambda item: item.conf, reverse=True):
        if any(_bbox_iou(candidate.bbox, existing.bbox) > config.overlap_iou_threshold for existing in existing_frames):
            rejected_overlap += 1
            continue
        distance = _point_distance(predicted, candidate.world_xy)
        if distance > config.max_fill_distance_m:
            rejected_distance += 1
            continue
        cost = distance - 0.1 * candidate.conf
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best, rejected_overlap, rejected_distance


def _tracks_by_frame(tracks: Tracks, *, fps: float) -> dict[int, list[tuple[int, TrackFrame]]]:
    by_frame: dict[int, list[tuple[int, TrackFrame]]] = defaultdict(list)
    for player in tracks.players:
        for frame in player.frames:
            by_frame[int(round(float(frame.t) * fps))].append((int(player.id), frame))
    return by_frame


def _nms(candidates: Sequence[_CandidateDetection], iou_threshold: float) -> list[_CandidateDetection]:
    if iou_threshold <= 0.0:
        return list(candidates)
    kept: list[_CandidateDetection] = []
    for candidate in sorted(candidates, key=lambda item: item.conf, reverse=True):
        if all(_bbox_iou(candidate.bbox, existing.bbox) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _bbox_xyxy(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four xyxy values")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detection bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _is_person_detection(detection: dict[str, Any]) -> bool:
    value = detection.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


def _track_id(detection: dict[str, Any], fallback: int) -> int:
    value = detection.get("track_id", detection.get("player_id", detection.get("id", fallback)))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def _point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding vectors must have matching dimensions")
    dot = sum(float(left) * float(right) for left, right in zip(a, b, strict=True))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in a))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in b))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("embedding vectors must have positive norm")
    similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - similarity


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
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


__all__ = [
    "SourceSelectionConfig",
    "SourceSelectionSummary",
    "source_select_global_four_player_tracks",
    "source_select_four_player_tracks",
]
