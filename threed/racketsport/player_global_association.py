"""Source-only global person identity association for fixed-camera pickleball.

This is an offline post-pass: split mixed tracker fragments, reconnect them
with motion plus appearance, then emit at most the expected player cardinality.
It deliberately does not consume CVAT labels, so it can run on unlabeled clips
before scoring.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .court_templates import Sport, get_court_template
from .doubles_id import assign_doubles_roles
from .person_fast import PersonDetection, court_polygon_filter, person_detection_from_bbox
from .schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks
from .track_lock import TrackCandidate


@dataclass(frozen=True)
class GlobalAssociationDetection:
    frame_idx: int
    source_track_id: int
    bbox: tuple[float, float, float, float]
    world_xy: tuple[float, float]
    conf: float
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GlobalAssociationConfig:
    expected_players: int = 4
    split_gap_frames: int = 24
    max_fragment_speed_m_s: float = 12.0
    local_switch_split_distance_m: float = math.inf
    local_switch_split_embedding_distance: float = math.inf
    local_switch_split_max_gap_frames: int = 0
    embedding_split_eps: float = 0.35
    embedding_split_min_samples: int = 1
    embedding_split_max_clusters: int = 32
    max_merge_gap_frames: int = 240
    max_merge_speed_m_s: float = 9.0
    merge_distance_slack_m: float = 1.25
    max_merge_cost: float = 3.0
    appearance_weight: float = 1.0
    motion_weight: float = 1.0
    side_prior_weight: float = 0.25
    min_fragment_frames: int = 1
    max_fragments_for_global: int = 240
    max_gap_fill_frames: int = 24
    max_gap_fill_speed_m_s: float = 7.0
    gap_fill_iou_threshold: float = 0.25
    cardinality_backfill: bool = False
    backfill_max_cost: float = 2.5
    backfill_iou_threshold: float = 0.25
    drop_outside_court: bool = False
    court_margin_m: float = 0.0
    post_association_court_margin_m: float | None = None
    sport: Sport = "pickleball"

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.split_gap_frames < 1:
            raise ValueError("split_gap_frames must be positive")
        if self.local_switch_split_distance_m < 0.0:
            raise ValueError("local_switch_split_distance_m must be non-negative")
        if self.local_switch_split_embedding_distance < 0.0:
            raise ValueError("local_switch_split_embedding_distance must be non-negative")
        if self.local_switch_split_max_gap_frames < 0:
            raise ValueError("local_switch_split_max_gap_frames must be non-negative")
        if self.embedding_split_eps < 0.0:
            raise ValueError("embedding_split_eps must be non-negative")
        if self.embedding_split_min_samples < 1:
            raise ValueError("embedding_split_min_samples must be positive")
        if self.embedding_split_max_clusters < 1:
            raise ValueError("embedding_split_max_clusters must be positive")
        if self.max_merge_gap_frames < 1:
            raise ValueError("max_merge_gap_frames must be positive")
        if self.max_gap_fill_frames < 0:
            raise ValueError("max_gap_fill_frames must be non-negative")
        if self.backfill_max_cost < 0.0:
            raise ValueError("backfill_max_cost must be non-negative")
        if not 0.0 <= self.backfill_iou_threshold <= 1.0:
            raise ValueError("backfill_iou_threshold must be between 0 and 1")
        if self.court_margin_m < 0.0:
            raise ValueError("court_margin_m must be non-negative")
        if self.post_association_court_margin_m is not None and self.post_association_court_margin_m < 0.0:
            raise ValueError("post_association_court_margin_m must be non-negative")


@dataclass(frozen=True)
class GlobalAssociationSummary:
    status: str
    input_detection_count: int
    fragment_count: int
    selected_fragment_count: int
    dropped_fragment_count: int
    merged_fragment_count: int
    synthetic_frame_count: int
    gap_fill_skipped_overlap_count: int
    cardinality_backfilled_detection_count: int
    output_player_count: int
    embedding_split_count: int
    court_rejected_detection_count: int = 0
    court_filter_skipped_reason: str = ""
    post_association_court_rejected_frame_count: int = 0
    source_only: bool = True
    uses_cvat_labels: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Fragment:
    fragment_id: int
    source_track_id: int
    detections: tuple[GlobalAssociationDetection, ...]

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx

    @property
    def frame_set(self) -> set[int]:
        return {detection.frame_idx for detection in self.detections}

    @property
    def score(self) -> float:
        return len(self.detections) + 0.01 * sum(detection.conf for detection in self.detections)


@dataclass(frozen=True)
class _Cluster:
    fragments: tuple[_Fragment, ...]

    @property
    def frame_set(self) -> set[int]:
        frames: set[int] = set()
        for fragment in self.fragments:
            frames.update(fragment.frame_set)
        return frames

    @property
    def score(self) -> float:
        return sum(fragment.score for fragment in self.fragments)

    @property
    def detections(self) -> list[GlobalAssociationDetection]:
        detections = [detection for fragment in self.fragments for detection in fragment.detections]
        return sorted(detections, key=lambda detection: (detection.frame_idx, -detection.conf))


@dataclass(frozen=True)
class _EmbeddingRecord:
    vector: tuple[float, ...]
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class _BackfillPlayerState:
    embedding: tuple[float, ...] | None
    median_xy: tuple[float, float]
    source_track_ids: set[int]
    frame_indexes: list[int]
    detections: list[GlobalAssociationDetection]


def associate_global_identities(
    detections: Sequence[GlobalAssociationDetection],
    *,
    fps: float,
    config: GlobalAssociationConfig | None = None,
) -> tuple[Tracks, GlobalAssociationSummary]:
    cfg = config or GlobalAssociationConfig()
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    ordered_input = sorted(detections, key=lambda detection: (detection.frame_idx, detection.source_track_id, -detection.conf))
    court_filter_skipped_reason = ""
    if cfg.drop_outside_court:
        ordered, court_filter_skipped_reason = _drop_outside_court_detections(ordered_input, config=cfg)
    else:
        ordered = ordered_input
    court_rejected_detection_count = len(ordered_input) - len(ordered)
    fragments, embedding_split_count = _build_fragments(ordered, fps=fps, config=cfg)
    usable = [fragment for fragment in fragments if len(fragment.detections) >= cfg.min_fragment_frames]
    if len(usable) < cfg.expected_players:
        usable = fragments
    usable = _cap_fragments_for_global(usable, config=cfg)

    clusters, merged_fragment_count = _connect_fragments(usable, fps=fps, config=cfg)
    selected = sorted(clusters, key=lambda cluster: (-cluster.score, _cluster_first_frame(cluster)))[: cfg.expected_players]
    selected_fragment_ids = {fragment.fragment_id for cluster in selected for fragment in cluster.fragments}
    dropped_fragment_count = sum(1 for fragment in fragments if fragment.fragment_id not in selected_fragment_ids)

    tracks, synthetic_count, skipped_overlap, backfilled_count = _clusters_to_tracks(
        selected,
        source_detections=ordered,
        fps=fps,
        config=cfg,
    )

    post_association_court_rejected_frame_count = 0
    if cfg.post_association_court_margin_m is not None:
        tracks, post_association_court_rejected_frame_count, post_filter_note = _drop_outside_court_track_frames(
            tracks,
            config=cfg,
        )
        if post_filter_note and not court_filter_skipped_reason:
            court_filter_skipped_reason = post_filter_note

    status = "ok" if len(tracks.players) == cfg.expected_players else "insufficient_players"
    return tracks, GlobalAssociationSummary(
        status=status,
        input_detection_count=len(detections),
        fragment_count=len(fragments),
        selected_fragment_count=sum(len(cluster.fragments) for cluster in selected),
        dropped_fragment_count=dropped_fragment_count,
        merged_fragment_count=merged_fragment_count,
        synthetic_frame_count=synthetic_count,
        gap_fill_skipped_overlap_count=skipped_overlap,
        cardinality_backfilled_detection_count=backfilled_count,
        output_player_count=len(tracks.players),
        embedding_split_count=embedding_split_count,
        court_rejected_detection_count=court_rejected_detection_count,
        post_association_court_rejected_frame_count=post_association_court_rejected_frame_count,
        court_filter_skipped_reason=court_filter_skipped_reason,
    )


def _drop_outside_court_detections(
    detections: Sequence[GlobalAssociationDetection],
    *,
    config: GlobalAssociationConfig,
) -> tuple[list[GlobalAssociationDetection], str]:
    """Reject candidate detections whose court-projected foot point falls
    outside the sport's court polygon (plus an apron margin).

    This reuses ``person_fast.court_polygon_filter`` -- the same source-only,
    calibration-derived geometry gate already applied in ``track.py`` and
    ``player_source_selection.py`` -- instead of re-deriving the rectangle
    math here, so every candidate-construction path shares one definition of
    "on court". The court polygon is derived only from the sport template and
    each detection's already-projected ``world_xy`` (itself a calibration
    homography projection upstream); no CVAT/ground-truth labels are read.

    Fails open: if the sport's court template is unavailable for any reason,
    the filter is skipped (all detections pass through unchanged) and a
    human-readable reason is returned for the caller to log, rather than
    raising and aborting the whole association run.
    """
    try:
        get_court_template(config.sport)
    except ValueError as exc:
        return list(detections), f"court_template_unavailable_for_sport={config.sport!r}: {exc}"

    kept: list[GlobalAssociationDetection] = []
    for detection in detections:
        person = PersonDetection(
            bbox_xyxy=detection.bbox,
            confidence=detection.conf,
            foot_world_xy=list(detection.world_xy),
        )
        if court_polygon_filter([person], sport=config.sport, margin_m=config.court_margin_m):
            kept.append(detection)
    return kept, ""


def _drop_outside_court_track_frames(
    tracks: Tracks,
    *,
    config: GlobalAssociationConfig,
) -> tuple[Tracks, int, str]:
    """Trim individually off-court frames from already-selected output tracks.

    ``_drop_outside_court_detections`` runs before fragment building and uses
    an apron margin generous enough (``court_margin_m``, typically 1-3 m) to
    avoid rejecting real boundary-line play, so it cannot -- by design --
    catch every off-court frame in the final track (a candidate can sit
    just inside that apron and still be well outside the strict court
    polygon). This second, optional pass runs *after* fragment/ID selection
    is locked in, so it only ever removes frames from an already-chosen
    track; it never changes which candidates get merged into which fragment
    or which fragment wins a player slot, so it cannot introduce new ID
    switches or spectator swaps. It uses a separate (typically tighter)
    ``post_association_court_margin_m`` -- e.g. ``0.0`` to mirror the strict
    court-only definition ``person_track_gt_scoring.py`` uses for
    ``off_court_false_positive_frames``.

    Fails open on an unavailable court template, matching
    ``_drop_outside_court_detections``.
    """
    margin_m = config.post_association_court_margin_m
    if margin_m is None:
        return tracks, 0, ""
    try:
        get_court_template(config.sport)
    except ValueError as exc:
        return tracks, 0, f"court_template_unavailable_for_sport={config.sport!r}: {exc}"

    rejected_count = 0
    players: list[PlayerTrack] = []
    for player in tracks.players:
        kept_frames = []
        for frame in player.frames:
            person = PersonDetection(
                bbox_xyxy=tuple(float(value) for value in frame.bbox),  # type: ignore[arg-type]
                confidence=float(frame.conf),
                foot_world_xy=list(frame.world_xy),
            )
            if court_polygon_filter([person], sport=config.sport, margin_m=margin_m):
                kept_frames.append(frame)
            else:
                rejected_count += 1
        players.append(
            PlayerTrack(id=player.id, side=player.side, role=player.role, frames=kept_frames)
        )
    filtered = Tracks(schema_version=tracks.schema_version, fps=tracks.fps, players=players, rally_spans=tracks.rally_spans)
    return filtered, rejected_count, ""


def tracks_to_global_detections(
    tracks: Tracks,
    *,
    embedding_payload: Mapping[str, Any] | None = None,
    embedding_bbox_scale: float = 1.0,
    max_embedding_bbox_delta_px: float = 2.5,
) -> list[GlobalAssociationDetection]:
    if embedding_bbox_scale <= 0.0:
        raise ValueError("embedding_bbox_scale must be positive")
    if max_embedding_bbox_delta_px < 0.0:
        raise ValueError("max_embedding_bbox_delta_px must be non-negative")
    embedding_index, frame_embedding_index = _embedding_indexes(embedding_payload)
    detections: list[GlobalAssociationDetection] = []
    matched_embedding_count = 0
    fps = float(tracks.fps)
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(float(frame.t) * fps))
            source_track_id = int(player.id)
            bbox = tuple(float(value) for value in frame.bbox)  # type: ignore[arg-type]
            embedding = _match_embedding(
                embedding_index.get((frame_idx, source_track_id), []),
                bbox=bbox,
                embedding_bbox_scale=embedding_bbox_scale,
                max_bbox_delta=max_embedding_bbox_delta_px,
                raise_on_mismatch=False,
            )
            if embedding is None:
                # Repaired tracks may contain synthetic interpolated frames with
                # no persisted source detection; those legitimately carry no
                # appearance embedding, so a per-frame miss must not hard-fail.
                embedding = _match_embedding(
                    frame_embedding_index.get(frame_idx, []),
                    bbox=bbox,
                    embedding_bbox_scale=embedding_bbox_scale,
                    max_bbox_delta=max_embedding_bbox_delta_px,
                    raise_on_mismatch=False,
                )
            if embedding is not None:
                matched_embedding_count += 1
            detections.append(
                GlobalAssociationDetection(
                    frame_idx=frame_idx,
                    source_track_id=source_track_id,
                    bbox=bbox,
                    world_xy=tuple(float(value) for value in frame.world_xy),  # type: ignore[arg-type]
                    conf=float(frame.conf),
                    embedding=embedding,
                )
            )
    return detections


def raw_pool_to_global_detections(
    detections_payload: Mapping[str, Any],
    *,
    calibration: CourtCalibration,
    embedding_payload: Mapping[str, Any] | None = None,
    embedding_bbox_scale: float = 1.0,
    max_embedding_bbox_delta_px: float = 2.5,
    min_conf: float = 0.0,
) -> list[GlobalAssociationDetection]:
    """Build candidates from every raw, pre-role-lock per-frame detection.

    ``tracks_to_global_detections`` only sees whatever a prior role-lock /
    cardinality stage already kept, so it cannot recover a detection that
    stage dropped or reject a spectator it admitted. This reads the raw
    tracked-detections pool directly (every person box on every frame,
    typically far more than ``expected_players`` per frame including
    spectators/background) and projects each box to court world coordinates
    with the same calibration/foot-point convention the rest of the pipeline
    uses, so ``associate_global_identities`` can do real exactly-N selection
    -- appearance/motion cost, same-frame cannot-link, court-polygon prior --
    over the full source pool instead of an already-capped candidate set.
    Source-only: no ground-truth labels are read here.
    """
    if embedding_bbox_scale <= 0.0:
        raise ValueError("embedding_bbox_scale must be positive")
    if max_embedding_bbox_delta_px < 0.0:
        raise ValueError("max_embedding_bbox_delta_px must be non-negative")
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")

    embedding_index, frame_embedding_index = _embedding_indexes(embedding_payload)
    detections: list[GlobalAssociationDetection] = []
    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, Mapping):
            raise ValueError("each frame entry must be an object")
        frame_idx = int(frame_entry.get("frame", frame_entry.get("frame_index", default_frame_idx)))
        frame_detections = frame_entry.get("detections", [])
        if not isinstance(frame_detections, list):
            raise ValueError("frame detections must be a list")
        for det_idx, detection in enumerate(frame_detections):
            if not isinstance(detection, Mapping):
                raise ValueError("each detection must be an object")
            if not _is_raw_person_detection(detection):
                continue
            conf = float(detection.get("conf", detection.get("confidence", 1.0)))
            if conf < min_conf:
                continue
            bbox = _raw_bbox_xyxy(detection)
            source_track_id = _raw_track_id(detection, det_idx + 1)
            person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=conf)
            embedding = _match_embedding(
                embedding_index.get((frame_idx, source_track_id), []),
                bbox=bbox,
                embedding_bbox_scale=embedding_bbox_scale,
                max_bbox_delta=max_embedding_bbox_delta_px,
                raise_on_mismatch=False,
            )
            if embedding is None:
                embedding = _match_embedding(
                    frame_embedding_index.get(frame_idx, []),
                    bbox=bbox,
                    embedding_bbox_scale=embedding_bbox_scale,
                    max_bbox_delta=max_embedding_bbox_delta_px,
                    raise_on_mismatch=False,
                )
            detections.append(
                GlobalAssociationDetection(
                    frame_idx=frame_idx,
                    source_track_id=source_track_id,
                    bbox=person.bbox_xyxy,
                    world_xy=tuple(float(value) for value in person.foot_world_xy),  # type: ignore[arg-type]
                    conf=conf,
                    embedding=embedding,
                )
            )
    return detections


def _is_raw_person_detection(detection: Mapping[str, Any]) -> bool:
    value = detection.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


def _raw_track_id(detection: Mapping[str, Any], fallback: int) -> int:
    value = detection.get("track_id", detection.get("player_id", detection.get("id", fallback)))
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return fallback


def _raw_bbox_xyxy(detection: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four xyxy values")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detection bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _embedding_indexes(
    embedding_payload: Mapping[str, Any] | None,
) -> tuple[dict[tuple[int, int], list[_EmbeddingRecord]], dict[int, list[_EmbeddingRecord]]]:
    if embedding_payload is None:
        return {}, {}
    if bool(embedding_payload.get("uses_cvat_labels", False)) or embedding_payload.get("source_only") is False:
        raise ValueError("embedding payload must be source-only and cannot use CVAT labels")
    if bool(embedding_payload.get("promote_trk", False)):
        raise ValueError("embedding payload cannot be a promoted TRK artifact")
    rows = embedding_payload.get("detections")
    if not isinstance(rows, list):
        raise ValueError("embedding payload must contain a detections list")
    expected_dim_raw = embedding_payload.get("feature_dim")
    expected_dim = int(expected_dim_raw) if expected_dim_raw is not None else None
    l2_normalized = bool(embedding_payload.get("l2_normalized", False))
    index: dict[tuple[int, int], list[_EmbeddingRecord]] = defaultdict(list)
    by_frame: dict[int, list[_EmbeddingRecord]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("embedding detection rows must be objects")
        frame_idx = _int_field(row, ("frame", "frame_idx", "frame_index"), f"embedding row {row_idx} frame")
        source_track_id = _int_field(row, ("source_track_id", "track_id"), f"embedding row {row_idx} source_track_id")
        vector = _embedding_vector(row.get("embedding"), expected_dim=expected_dim, l2_normalized=l2_normalized, row_idx=row_idx)
        bbox = _embedding_bbox(row.get("bbox") or row.get("bbox_xyxy"))
        record = _EmbeddingRecord(vector=vector, bbox=bbox)
        index[(frame_idx, source_track_id)].append(record)
        by_frame[frame_idx].append(record)
    return index, by_frame


def _match_embedding(
    records: Sequence[_EmbeddingRecord],
    *,
    bbox: tuple[float, float, float, float],
    embedding_bbox_scale: float,
    max_bbox_delta: float,
    raise_on_mismatch: bool,
) -> tuple[float, ...] | None:
    if not records:
        return None
    if len(records) == 1 and records[0].bbox is None:
        return records[0].vector
    scaled_bbox = tuple(float(value) * embedding_bbox_scale for value in bbox)
    with_bbox = [record for record in records if record.bbox is not None]
    if not with_bbox:
        return records[0].vector
    best = min(with_bbox, key=lambda record: _bbox_delta(scaled_bbox, record.bbox or scaled_bbox))
    delta = _bbox_delta(scaled_bbox, best.bbox or scaled_bbox)
    if delta > max_bbox_delta:
        if not raise_on_mismatch:
            return None
        raise ValueError(f"embedding bbox mismatch exceeds tolerance: {delta:.3f}px")
    return best.vector


def _int_field(row: Mapping[str, Any], fields: Sequence[str], label: str) -> int:
    for field in fields:
        if field in row:
            return int(row[field])
    raise ValueError(f"{label} missing")


def _embedding_vector(
    raw: Any,
    *,
    expected_dim: int | None,
    l2_normalized: bool,
    row_idx: int,
) -> tuple[float, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise ValueError(f"embedding row {row_idx} must contain a non-empty embedding")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"embedding row {row_idx} contains a non-finite value")
    if expected_dim is not None and len(values) != expected_dim:
        raise ValueError(f"embedding row {row_idx} feature_dim mismatch")
    if l2_normalized:
        norm = math.sqrt(sum(value * value for value in values))
        if not 0.98 <= norm <= 1.02:
            raise ValueError(f"embedding row {row_idx} must be L2-normalized")
    return values


def _embedding_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("embedding bbox must contain four xyxy values when present")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("embedding bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _bbox_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _build_fragments(
    detections: Sequence[GlobalAssociationDetection],
    *,
    fps: float,
    config: GlobalAssociationConfig,
) -> tuple[list[_Fragment], int]:
    grouped: dict[int, list[GlobalAssociationDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.source_track_id].append(detection)

    fragments: list[_Fragment] = []
    embedding_split_count = 0
    next_fragment_id = 1
    for source_track_id, items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda detection: (detection.frame_idx, -detection.conf))
        cluster_labels = _embedding_cluster_labels(sorted_items, config=config)
        labels = {label for label in cluster_labels if label is not None}
        if len(labels) > 1:
            embedding_split_count += len(labels) - 1

        current: list[GlobalAssociationDetection] = []
        current_label: int | None = None
        for idx, detection in enumerate(sorted_items):
            label = cluster_labels[idx]
            if not current:
                current = [detection]
                current_label = label
                continue
            previous = current[-1]
            gap = detection.frame_idx - previous.frame_idx
            split = gap <= 0 or gap > config.split_gap_frames
            if not split:
                dt = max(gap / fps, 1.0 / fps)
                allowed = config.merge_distance_slack_m + config.max_fragment_speed_m_s * dt
                split = _point_distance(previous.world_xy, detection.world_xy) > allowed
            if not split:
                split = _is_local_switch_handoff(previous, detection, gap=gap, config=config)
            if not split and current_label is not None and label is not None:
                split = current_label != label
            if split:
                fragments.append(_make_fragment(next_fragment_id, source_track_id, current))
                next_fragment_id += 1
                current = [detection]
                current_label = label
            else:
                current.append(detection)
                if current_label is None:
                    current_label = label
        if current:
            fragments.append(_make_fragment(next_fragment_id, source_track_id, current))
            next_fragment_id += 1
    return fragments, embedding_split_count


def _is_local_switch_handoff(
    previous: GlobalAssociationDetection,
    current: GlobalAssociationDetection,
    *,
    gap: int,
    config: GlobalAssociationConfig,
) -> bool:
    if config.local_switch_split_max_gap_frames <= 0:
        return False
    if gap <= 0 or gap > config.local_switch_split_max_gap_frames:
        return False
    if not math.isfinite(config.local_switch_split_distance_m) or not math.isfinite(config.local_switch_split_embedding_distance):
        return False
    if previous.embedding is None or current.embedding is None:
        return False
    if _point_distance(previous.world_xy, current.world_xy) < config.local_switch_split_distance_m:
        return False
    return _cosine_distance(previous.embedding, current.embedding) >= config.local_switch_split_embedding_distance


def _embedding_cluster_labels(
    detections: Sequence[GlobalAssociationDetection],
    *,
    config: GlobalAssociationConfig,
) -> list[int | None]:
    embeddings = [detection.embedding for detection in detections]
    labels: list[int | None] = [None] * len(detections)
    embedding_indices = [idx for idx, embedding in enumerate(embeddings) if embedding]
    if not embedding_indices:
        return labels

    components: dict[int, list[int]] = {}
    centroids: dict[int, tuple[float, ...]] = {}
    centroid_sums: dict[int, list[float]] = {}
    centroid_counts: dict[int, int] = {}
    next_label = 0
    for idx in embedding_indices:
        embedding = embeddings[idx]
        if embedding is None:
            continue
        best_label: int | None = None
        best_distance = math.inf
        for label, centroid in centroids.items():
            distance = _cosine_distance(embedding, centroid)
            if distance < best_distance:
                best_distance = distance
                best_label = label
        if (best_label is None or best_distance > config.embedding_split_eps) and len(centroids) < config.embedding_split_max_clusters:
            best_label = next_label
            next_label += 1
            components[best_label] = []
            centroid_sums[best_label] = [0.0] * len(embedding)
            centroid_counts[best_label] = 0
            centroids[best_label] = _normalized_embedding(embedding)
        elif best_label is None:
            best_label = next(iter(centroids))
        components[best_label].append(idx)
        _add_to_centroid(best_label, embedding, centroids=centroids, sums=centroid_sums, counts=centroid_counts)

    label_by_component: dict[int, int] = {}
    next_output_label = 0
    for component_label, component_indices in sorted(components.items(), key=lambda item: min(item[1])):
        if len(component_indices) < config.embedding_split_min_samples:
            continue
        label_by_component[component_label] = next_output_label
        next_output_label += 1

    for component_label, component_indices in components.items():
        output_label = label_by_component.get(component_label)
        if output_label is None:
            continue
        for idx in component_indices:
            labels[idx] = output_label
    return labels


def _normalized_embedding(embedding: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in embedding)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return values
    return tuple(value / norm for value in values)


def _add_to_centroid(
    label: int,
    embedding: Sequence[float],
    *,
    centroids: dict[int, tuple[float, ...]],
    sums: dict[int, list[float]],
    counts: dict[int, int],
) -> None:
    vector = [float(value) for value in embedding]
    totals = sums[label]
    if len(totals) != len(vector):
        return
    for idx, value in enumerate(vector):
        totals[idx] += value
    counts[label] += 1
    count = counts[label]
    mean = tuple(value / count for value in totals)
    centroids[label] = _normalized_embedding(mean)


def _make_fragment(
    fragment_id: int,
    source_track_id: int,
    detections: Sequence[GlobalAssociationDetection],
) -> _Fragment:
    deduped: dict[int, GlobalAssociationDetection] = {}
    for detection in detections:
        existing = deduped.get(detection.frame_idx)
        if existing is None or detection.conf > existing.conf:
            deduped[detection.frame_idx] = detection
    return _Fragment(
        fragment_id=fragment_id,
        source_track_id=source_track_id,
        detections=tuple(sorted(deduped.values(), key=lambda detection: detection.frame_idx)),
    )


def _cap_fragments_for_global(
    fragments: Sequence[_Fragment],
    *,
    config: GlobalAssociationConfig,
) -> list[_Fragment]:
    if config.max_fragments_for_global <= 0 or len(fragments) <= config.max_fragments_for_global:
        return list(fragments)
    return sorted(fragments, key=lambda fragment: (-fragment.score, fragment.start_frame))[: config.max_fragments_for_global]


def _connect_fragments(
    fragments: Sequence[_Fragment],
    *,
    fps: float,
    config: GlobalAssociationConfig,
) -> tuple[list[_Cluster], int]:
    if not fragments:
        return [], 0
    fragment_list = list(fragments)
    parent = {idx: idx for idx in range(len(fragment_list))}
    cluster_fragments: dict[int, tuple[_Fragment, ...]] = {idx: (fragment,) for idx, fragment in enumerate(fragment_list)}
    cluster_frame_sets: dict[int, set[int]] = {idx: fragment.frame_set for idx, fragment in enumerate(fragment_list)}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    edges: list[tuple[float, int, int]] = []
    for i in range(len(fragment_list)):
        for j in range(i + 1, len(fragment_list)):
            if cluster_frame_sets[i] & cluster_frame_sets[j]:
                continue
            cost = _fragment_link_cost(fragment_list[i], fragment_list[j], fps=fps, config=config)
            if math.isfinite(cost) and cost <= config.max_merge_cost:
                edges.append((cost, i, j))
    edges.sort(key=lambda item: (item[0], fragment_list[item[1]].start_frame, fragment_list[item[2]].start_frame))

    merged = 0
    cluster_count = len(fragment_list)
    for _cost, i, j in edges:
        left_root = find(i)
        right_root = find(j)
        if left_root == right_root:
            continue
        if cluster_frame_sets[left_root] & cluster_frame_sets[right_root]:
            continue
        parent[right_root] = left_root
        cluster_fragments[left_root] = cluster_fragments[left_root] + cluster_fragments[right_root]
        cluster_frame_sets[left_root] = cluster_frame_sets[left_root] | cluster_frame_sets[right_root]
        del cluster_fragments[right_root]
        del cluster_frame_sets[right_root]
        merged += 1
        cluster_count -= 1
        if cluster_count <= config.expected_players:
            break

    roots = sorted({find(idx) for idx in range(len(fragment_list))}, key=lambda idx: min(fragment.start_frame for fragment in cluster_fragments[idx]))
    clusters = [_Cluster(cluster_fragments[root]) for root in roots]
    return clusters, merged


def _fragment_link_cost(
    left: _Fragment,
    right: _Fragment,
    *,
    fps: float,
    config: GlobalAssociationConfig,
) -> float:
    if not (left.end_frame < right.start_frame or right.end_frame < left.start_frame):
        return math.inf
    early, late = (left, right) if left.end_frame < right.start_frame else (right, left)
    gap = late.start_frame - early.end_frame
    if gap <= 0 or gap > config.max_merge_gap_frames:
        return math.inf
    dt = max(gap / fps, 1.0 / fps)

    early_last = early.detections[-1].world_xy
    late_first = late.detections[0].world_xy
    direct_dist = _point_distance(early_last, late_first)
    direct_allowed = config.merge_distance_slack_m + config.max_merge_speed_m_s * dt
    if direct_dist > direct_allowed:
        return math.inf

    early_velocity = _fragment_velocity(early, at_end=True, fps=fps)
    late_velocity = _fragment_velocity(late, at_end=False, fps=fps)
    predicted_late = (early_last[0] + early_velocity[0] * dt, early_last[1] + early_velocity[1] * dt)
    predicted_early = (late_first[0] - late_velocity[0] * dt, late_first[1] - late_velocity[1] * dt)
    continuity_dist = 0.5 * (_point_distance(predicted_late, late_first) + _point_distance(predicted_early, early_last))
    continuity_allowed = max(config.merge_distance_slack_m, direct_allowed)
    motion_cost = continuity_dist / continuity_allowed

    appearance_cost = _fragment_appearance_distance(early, late)
    if not math.isfinite(appearance_cost):
        appearance_cost = 0.5

    side_cost = 0.0
    if _median_world_xy(early.detections)[1] * _median_world_xy(late.detections)[1] < 0.0:
        side_cost = 1.0

    source_bonus = -0.20 if early.source_track_id == late.source_track_id else 0.0
    gap_penalty = min(1.0, gap / max(config.max_merge_gap_frames, 1))
    return max(
        0.0,
        config.motion_weight * motion_cost
        + config.appearance_weight * appearance_cost
        + config.side_prior_weight * side_cost
        + 0.10 * gap_penalty
        + source_bonus,
    )


def _fragment_appearance_distance(left: _Fragment, right: _Fragment) -> float:
    left_embedding = _mean_embedding(detection.embedding for detection in left.detections)
    right_embedding = _mean_embedding(detection.embedding for detection in right.detections)
    if left_embedding is None or right_embedding is None:
        return math.inf
    return _cosine_distance(left_embedding, right_embedding)


def _mean_embedding(embeddings: Sequence[Sequence[float] | None] | Any) -> tuple[float, ...] | None:
    vectors = [tuple(float(value) for value in embedding) for embedding in embeddings if embedding]
    if not vectors:
        return None
    dim = len(vectors[0])
    compatible = [vector for vector in vectors if len(vector) == dim]
    if not compatible:
        return None
    mean = tuple(sum(vector[idx] for vector in compatible) / len(compatible) for idx in range(dim))
    norm = math.sqrt(sum(value * value for value in mean))
    if norm <= 0.0:
        return None
    return tuple(value / norm for value in mean)


def _fragment_velocity(fragment: _Fragment, *, at_end: bool, fps: float) -> tuple[float, float]:
    detections = fragment.detections
    if len(detections) < 2:
        return (0.0, 0.0)
    window = detections[-5:] if at_end else detections[:5]
    first = window[0]
    last = window[-1]
    dt = (last.frame_idx - first.frame_idx) / fps
    if dt <= 0.0:
        return (0.0, 0.0)
    return ((last.world_xy[0] - first.world_xy[0]) / dt, (last.world_xy[1] - first.world_xy[1]) / dt)


def _clusters_to_tracks(
    clusters: Sequence[_Cluster],
    *,
    source_detections: Sequence[GlobalAssociationDetection],
    fps: float,
    config: GlobalAssociationConfig,
) -> tuple[Tracks, int, int, int]:
    raw_players = [_dedupe_cluster_detections(cluster) for cluster in clusters]
    raw_players = [detections for detections in raw_players if detections]
    sorted_players = sorted(raw_players, key=_player_sort_key)
    backfilled_count = 0
    if config.cardinality_backfill:
        sorted_players, backfilled_count = _backfill_cardinality_gaps(
            sorted_players,
            source_detections=source_detections,
            fps=fps,
            config=config,
        )

    synthetic_total = 0
    skipped_overlap_total = 0
    filled_players: list[list[GlobalAssociationDetection]] = []
    for idx, detections in enumerate(sorted_players):
        other_detections = [detection for j, player in enumerate(sorted_players) if j != idx for detection in player]
        filled, synthetic_count, skipped_overlap = _fill_short_gaps(
            detections,
            other_detections=other_detections,
            fps=fps,
            config=config,
        )
        filled_players.append(filled)
        synthetic_total += synthetic_count
        skipped_overlap_total += skipped_overlap

    identities = _identity_labels(filled_players)
    players = [
        PlayerTrack(
            id=idx + 1,
            side=identities.get(idx, {}).get("side", "unknown"),
            role=identities.get(idx, {}).get("role", "unknown"),
            frames=[
                TrackFrame(
                    t=detection.frame_idx / fps,
                    bbox=detection.bbox,
                    world_xy=detection.world_xy,
                    conf=max(0.0, min(1.0, detection.conf)),
                )
                for detection in detections
            ],
        )
        for idx, detections in enumerate(filled_players)
    ]
    return Tracks(schema_version=1, fps=fps, players=players, rally_spans=[]), synthetic_total, skipped_overlap_total, backfilled_count


def _backfill_cardinality_gaps(
    players: Sequence[Sequence[GlobalAssociationDetection]],
    *,
    source_detections: Sequence[GlobalAssociationDetection],
    fps: float,
    config: GlobalAssociationConfig,
) -> tuple[list[list[GlobalAssociationDetection]], int]:
    if len(players) != config.expected_players:
        return [list(player) for player in players], 0

    out = [list(player) for player in players]
    states = [_backfill_player_state(player) for player in out]
    used = {_detection_key(detection) for player in out for detection in player}
    player_by_frame = [{detection.frame_idx: detection for detection in player} for player in out]
    source_by_frame: dict[int, list[GlobalAssociationDetection]] = defaultdict(list)
    for detection in source_detections:
        if _detection_key(detection) not in used:
            source_by_frame[detection.frame_idx].append(detection)

    backfilled = 0
    for frame_idx in sorted(source_by_frame):
        present = [
            (player_idx, detection)
            for player_idx, by_frame in enumerate(player_by_frame)
            if (detection := by_frame.get(frame_idx)) is not None
        ]
        if len(present) >= config.expected_players:
            continue
        missing_player_indexes = [idx for idx, by_frame in enumerate(player_by_frame) if frame_idx not in by_frame]
        if not missing_player_indexes:
            continue
        frame_detections = [detection for _, detection in present]
        candidates = sorted(source_by_frame[frame_idx], key=lambda detection: -detection.conf)
        for candidate in candidates:
            if len(present) >= config.expected_players:
                break
            if any(_bbox_iou(candidate.bbox, existing.bbox) > config.backfill_iou_threshold for existing in frame_detections):
                continue
            best_player: int | None = None
            best_cost = math.inf
            for player_idx in list(missing_player_indexes):
                cost = _backfill_player_cost(
                    candidate,
                    out[player_idx],
                    state=states[player_idx],
                    fps=fps,
                    config=config,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_player = player_idx
            if best_player is None or best_cost > config.backfill_max_cost:
                continue
            out[best_player].append(candidate)
            out[best_player].sort(key=lambda detection: detection.frame_idx)
            _add_backfill_state_detection(states[best_player], candidate)
            player_by_frame[best_player][frame_idx] = candidate
            frame_detections.append(candidate)
            present.append((best_player, candidate))
            missing_player_indexes.remove(best_player)
            backfilled += 1
    return out, backfilled


def _backfill_player_state(player: Sequence[GlobalAssociationDetection]) -> _BackfillPlayerState:
    return _BackfillPlayerState(
        embedding=_mean_embedding(detection.embedding for detection in player),
        median_xy=_median_world_xy(player),
        source_track_ids={detection.source_track_id for detection in player},
        frame_indexes=[detection.frame_idx for detection in sorted(player, key=lambda detection: detection.frame_idx)],
        detections=sorted(player, key=lambda detection: detection.frame_idx),
    )


def _add_backfill_state_detection(state: _BackfillPlayerState, detection: GlobalAssociationDetection) -> None:
    insert_at = bisect_left(state.frame_indexes, detection.frame_idx)
    state.frame_indexes.insert(insert_at, detection.frame_idx)
    state.detections.insert(insert_at, detection)
    state.source_track_ids.add(detection.source_track_id)


def _detection_key(detection: GlobalAssociationDetection) -> tuple[int, int, tuple[float, float, float, float]]:
    return (
        detection.frame_idx,
        detection.source_track_id,
        tuple(round(float(value), 4) for value in detection.bbox),  # type: ignore[return-value]
    )


def _backfill_player_cost(
    candidate: GlobalAssociationDetection,
    player: Sequence[GlobalAssociationDetection],
    *,
    state: _BackfillPlayerState,
    fps: float,
    config: GlobalAssociationConfig,
) -> float:
    if not player:
        return math.inf

    candidate_embedding = _normalized_embedding(candidate.embedding) if candidate.embedding else None
    appearance_cost = _cosine_distance(candidate_embedding, state.embedding) if candidate_embedding and state.embedding else 0.5

    median_xy = state.median_xy
    side_cost = 1.0 if median_xy[1] * candidate.world_xy[1] < 0.0 else 0.0
    quadrant_cost = 0.0
    if abs(median_xy[0]) > 0.75 and abs(candidate.world_xy[0]) > 0.75 and median_xy[0] * candidate.world_xy[0] < 0.0:
        quadrant_cost = 0.5

    nearest = _nearest_state_detection(state, candidate.frame_idx)
    if nearest is None:
        return math.inf
    gap = abs(nearest.frame_idx - candidate.frame_idx)
    if gap == 0:
        motion_cost = 0.0
    else:
        dt = max(gap / fps, 1.0 / fps)
        allowed = config.merge_distance_slack_m + config.max_merge_speed_m_s * dt
        motion_cost = min(2.0, _point_distance(nearest.world_xy, candidate.world_xy) / max(allowed, 1e-6))

    source_bonus = -0.25 if candidate.source_track_id in state.source_track_ids else 0.0
    return max(
        0.0,
        config.appearance_weight * appearance_cost
        + 0.35 * motion_cost
        + config.side_prior_weight * side_cost
        + 0.20 * quadrant_cost
        + source_bonus,
    )


def _nearest_state_detection(state: _BackfillPlayerState, frame_idx: int) -> GlobalAssociationDetection | None:
    if not state.frame_indexes:
        return None
    pos = bisect_left(state.frame_indexes, frame_idx)
    candidates: list[GlobalAssociationDetection] = []
    if pos < len(state.detections):
        candidates.append(state.detections[pos])
    if pos > 0:
        candidates.append(state.detections[pos - 1])
    return min(candidates, key=lambda detection: abs(detection.frame_idx - frame_idx)) if candidates else None


def _dedupe_cluster_detections(cluster: _Cluster) -> list[GlobalAssociationDetection]:
    by_frame: dict[int, GlobalAssociationDetection] = {}
    for detection in cluster.detections:
        existing = by_frame.get(detection.frame_idx)
        if existing is None or detection.conf > existing.conf:
            by_frame[detection.frame_idx] = detection
    return [by_frame[frame_idx] for frame_idx in sorted(by_frame)]


def _fill_short_gaps(
    detections: Sequence[GlobalAssociationDetection],
    *,
    other_detections: Sequence[GlobalAssociationDetection],
    fps: float,
    config: GlobalAssociationConfig,
) -> tuple[list[GlobalAssociationDetection], int, int]:
    if config.max_gap_fill_frames <= 0 or len(detections) < 2:
        return list(detections), 0, 0

    other_by_frame: dict[int, list[GlobalAssociationDetection]] = defaultdict(list)
    for detection in other_detections:
        other_by_frame[detection.frame_idx].append(detection)

    out: list[GlobalAssociationDetection] = []
    synthetic_count = 0
    skipped_overlap = 0
    for current, nxt in zip(detections, detections[1:], strict=False):
        out.append(current)
        gap = nxt.frame_idx - current.frame_idx
        if gap <= 1 or gap > config.max_gap_fill_frames:
            continue
        dt = gap / fps
        allowed = config.merge_distance_slack_m + config.max_gap_fill_speed_m_s * dt
        if _point_distance(current.world_xy, nxt.world_xy) > allowed:
            continue
        synthetic = [_interpolate_detection(current, nxt, frame_idx) for frame_idx in range(current.frame_idx + 1, nxt.frame_idx)]
        if any(
            _bbox_iou(det.bbox, other.bbox) > config.gap_fill_iou_threshold
            for det in synthetic
            for other in other_by_frame.get(det.frame_idx, [])
        ):
            skipped_overlap += 1
            continue
        out.extend(synthetic)
        synthetic_count += len(synthetic)
    out.append(detections[-1])
    return sorted(out, key=lambda detection: detection.frame_idx), synthetic_count, skipped_overlap


def _interpolate_detection(
    left: GlobalAssociationDetection,
    right: GlobalAssociationDetection,
    frame_idx: int,
) -> GlobalAssociationDetection:
    alpha = (frame_idx - left.frame_idx) / (right.frame_idx - left.frame_idx)
    bbox = tuple((1.0 - alpha) * left.bbox[i] + alpha * right.bbox[i] for i in range(4))
    world_xy = (
        (1.0 - alpha) * left.world_xy[0] + alpha * right.world_xy[0],
        (1.0 - alpha) * left.world_xy[1] + alpha * right.world_xy[1],
    )
    embedding = left.embedding if left.embedding == right.embedding else None
    conf = max(0.0, min(left.conf, right.conf, 0.35))
    return GlobalAssociationDetection(
        frame_idx=frame_idx,
        source_track_id=left.source_track_id,
        bbox=bbox,  # type: ignore[arg-type]
        world_xy=world_xy,
        conf=conf,
        embedding=embedding,
    )


def _identity_labels(players: Sequence[Sequence[GlobalAssociationDetection]]) -> dict[int, dict[str, str]]:
    candidates = [
        TrackCandidate(track_id=idx, world_xy=list(_median_world_xy(detections)), confidence=_mean_conf(detections))
        for idx, detections in enumerate(players)
        if detections
    ]
    if len(candidates) == 4:
        assigned = assign_doubles_roles(candidates)
        return {int(track_id): {"side": identity.side, "role": identity.role} for track_id, identity in assigned.items()}
    return {
        idx: {
            "side": "near" if _median_world_xy(detections)[1] <= 0.0 else "far",
            "role": "singles" if len(players) == 2 else "unknown",
        }
        for idx, detections in enumerate(players)
        if detections
    }


def _player_sort_key(detections: Sequence[GlobalAssociationDetection]) -> tuple[float, float, int]:
    xy = _median_world_xy(detections)
    return (xy[1], xy[0], detections[0].frame_idx if detections else 0)


def _cluster_first_frame(cluster: _Cluster) -> int:
    return min(fragment.start_frame for fragment in cluster.fragments)


def _median_world_xy(detections: Sequence[GlobalAssociationDetection]) -> tuple[float, float]:
    xs = sorted(detection.world_xy[0] for detection in detections)
    ys = sorted(detection.world_xy[1] for detection in detections)
    if not xs:
        return (0.0, 0.0)
    mid = len(xs) // 2
    return (xs[mid], ys[mid])


def _mean_conf(detections: Sequence[GlobalAssociationDetection]) -> float:
    return sum(detection.conf for detection in detections) / len(detections) if detections else 0.0


def _point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


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


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return math.inf
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return math.inf
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return 1.0 - cosine


__all__ = [
    "GlobalAssociationConfig",
    "GlobalAssociationDetection",
    "GlobalAssociationSummary",
    "associate_global_identities",
    "raw_pool_to_global_detections",
    "tracks_to_global_detections",
]
