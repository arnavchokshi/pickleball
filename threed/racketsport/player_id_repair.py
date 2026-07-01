"""Offline exactly-N player ID repair for racket-sport tracks.

This is a deliberately small post-tracker experiment inspired by the
sam4dbody global ID repair stack: split suspicious tracklets into fragments,
reconnect fragments into exactly N identities with motion continuity and a
same-frame cannot-link, then fill only short safe gaps.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .doubles_id import assign_doubles_roles
from .person_fast import court_polygon_filter, person_detection_from_bbox
from .schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks
from .track_lock import TrackCandidate


@dataclass(frozen=True)
class RepairDetection:
    frame_idx: int
    source_track_id: int
    bbox: tuple[float, float, float, float]
    world_xy: tuple[float, float]
    conf: float


@dataclass(frozen=True)
class RepairConfig:
    expected_players: int = 4
    split_gap_frames: int = 24
    max_fragment_speed_m_s: float = 12.0
    max_merge_gap_frames: int = 240
    max_merge_speed_m_s: float = 9.0
    merge_distance_slack_m: float = 1.25
    max_merge_cost: float = 2.5
    crossover_iou_threshold: float = 0.35
    crossover_world_distance_m: float = 0.65
    min_fragment_frames: int = 1
    max_fragments_for_global: int = 160
    nms_iou_threshold: float = 0.95
    max_gap_fill_frames: int = 24
    max_gap_fill_speed_m_s: float = 7.0
    gap_fill_iou_threshold: float = 0.25
    min_gap_fill_conf: float = 0.25
    court_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.split_gap_frames < 1:
            raise ValueError("split_gap_frames must be positive")
        if self.max_merge_gap_frames < 1:
            raise ValueError("max_merge_gap_frames must be positive")
        if self.max_gap_fill_frames < 0:
            raise ValueError("max_gap_fill_frames must be non-negative")


@dataclass(frozen=True)
class PlayerIdRepairSummary:
    status: str
    input_detection_count: int
    kept_detection_count: int
    fragment_count: int
    selected_fragment_count: int
    dropped_fragment_count: int
    merged_fragment_count: int
    synthetic_frame_count: int
    gap_fill_skipped_overlap_count: int
    output_player_count: int
    impossible_overlap_count: int
    teleport_count: int
    outside_court_count: int = 0
    non_person_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Fragment:
    fragment_id: int
    source_track_id: int
    detections: tuple[RepairDetection, ...]

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
    def detections(self) -> list[RepairDetection]:
        out = [detection for fragment in self.fragments for detection in fragment.detections]
        return sorted(out, key=lambda detection: (detection.frame_idx, -detection.conf))


def repair_tracks(
    tracks: Tracks,
    *,
    config: RepairConfig | None = None,
) -> tuple[Tracks, PlayerIdRepairSummary]:
    detections = [
        RepairDetection(
            frame_idx=int(round(float(frame.t) * float(tracks.fps))),
            source_track_id=int(player.id),
            bbox=tuple(float(value) for value in frame.bbox),  # type: ignore[arg-type]
            world_xy=tuple(float(value) for value in frame.world_xy),  # type: ignore[arg-type]
            conf=float(frame.conf),
        )
        for player in tracks.players
        for frame in player.frames
    ]
    return repair_detections_to_tracks(detections, fps=float(tracks.fps), config=config)


def repair_detection_payload_to_tracks(
    detections_payload: dict[str, Any],
    calibration: CourtCalibration,
    *,
    config: RepairConfig | None = None,
) -> tuple[Tracks, PlayerIdRepairSummary]:
    cfg = config or RepairConfig()
    detections: list[RepairDetection] = []
    string_ids: dict[str, int] = {}
    used_ids: set[int] = set()
    outside_court = 0
    non_person = 0
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")
    fps = float(detections_payload["fps"])

    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, dict):
            raise ValueError("each frame entry must be an object")
        frame_idx = _frame_index(frame_entry, default_frame_idx)
        frame_detections = frame_entry.get("detections", [])
        if not isinstance(frame_detections, list):
            raise ValueError("frame detections must be a list")
        for det_idx, detection in enumerate(frame_detections):
            if not isinstance(detection, dict):
                raise ValueError("each detection must be an object")
            if not _is_person_detection(detection):
                non_person += 1
                continue
            bbox = _bbox_xyxy(detection)
            conf = float(detection.get("conf", detection.get("confidence", 1.0)))
            person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=conf)
            if not court_polygon_filter([person], sport=calibration.sport, margin_m=cfg.court_margin_m):
                outside_court += 1
                continue
            detections.append(
                RepairDetection(
                    frame_idx=frame_idx,
                    source_track_id=_int_track_id(_track_key(detection, det_idx + 1), string_ids, used_ids),
                    bbox=tuple(float(value) for value in person.bbox_xyxy),  # type: ignore[arg-type]
                    world_xy=tuple(float(value) for value in person.foot_world_xy),  # type: ignore[arg-type]
                    conf=float(person.confidence),
                )
            )

    tracks, summary = repair_detections_to_tracks(detections, fps=fps, config=cfg)
    summary = PlayerIdRepairSummary(
        **{**summary.to_dict(), "outside_court_count": outside_court, "non_person_count": non_person}
    )
    return tracks, summary


def repair_detections_to_tracks(
    detections: Sequence[RepairDetection],
    *,
    fps: float,
    config: RepairConfig | None = None,
) -> tuple[Tracks, PlayerIdRepairSummary]:
    cfg = config or RepairConfig()
    if fps <= 0:
        raise ValueError("fps must be positive")
    ordered = sorted(detections, key=lambda detection: (detection.frame_idx, detection.source_track_id, -detection.conf))
    kept = _nms_by_frame(ordered, cfg.nms_iou_threshold)
    fragments = _build_fragments(kept, fps=fps, config=cfg)
    usable = [fragment for fragment in fragments if len(fragment.detections) >= cfg.min_fragment_frames]
    if len(usable) < cfg.expected_players:
        usable = fragments
    usable = _cap_fragments_for_global(usable, config=cfg)

    clusters, merged_fragment_count = _connect_fragments(usable, fps=fps, config=cfg)
    selected = sorted(clusters, key=lambda cluster: (-cluster.score, _cluster_first_frame(cluster)))[: cfg.expected_players]
    selected_fragment_ids = {fragment.fragment_id for cluster in selected for fragment in cluster.fragments}
    dropped_fragment_count = sum(1 for fragment in fragments if fragment.fragment_id not in selected_fragment_ids)

    tracks, synthetic_count, skipped_overlap = _clusters_to_tracks(selected, fps=fps, config=cfg)
    impossible_overlap_count = _count_same_frame_overlaps(tracks, cfg.gap_fill_iou_threshold)
    teleport_count = _count_teleports(tracks, fps=fps, max_speed_m_s=cfg.max_merge_speed_m_s)
    status = "ok"
    if len(tracks.players) != cfg.expected_players:
        status = "insufficient_players"
    elif impossible_overlap_count or teleport_count:
        status = "fail_closed_geometry"

    return tracks, PlayerIdRepairSummary(
        status=status,
        input_detection_count=len(detections),
        kept_detection_count=len(kept),
        fragment_count=len(fragments),
        selected_fragment_count=sum(len(cluster.fragments) for cluster in selected),
        dropped_fragment_count=dropped_fragment_count,
        merged_fragment_count=merged_fragment_count,
        synthetic_frame_count=synthetic_count,
        gap_fill_skipped_overlap_count=skipped_overlap,
        output_player_count=len(tracks.players),
        impossible_overlap_count=impossible_overlap_count,
        teleport_count=teleport_count,
    )


def _build_fragments(
    detections: Sequence[RepairDetection],
    *,
    fps: float,
    config: RepairConfig,
) -> list[_Fragment]:
    ambiguous = _ambiguous_frames_by_source(detections, config)
    grouped: dict[int, list[RepairDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.source_track_id].append(detection)

    fragments: list[_Fragment] = []
    next_fragment_id = 1
    for source_track_id, items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda detection: (detection.frame_idx, -detection.conf))
        current: list[RepairDetection] = []
        for detection in sorted_items:
            if not current:
                current = [detection]
                continue
            previous = current[-1]
            gap = detection.frame_idx - previous.frame_idx
            split = gap <= 0 or gap > config.split_gap_frames
            if not split:
                dt = max(gap / fps, 1.0 / fps)
                max_dist = config.merge_distance_slack_m + config.max_fragment_speed_m_s * dt
                split = _world_distance(previous, detection) > max_dist
            if not split:
                prev_ambiguous = previous.frame_idx in ambiguous.get(source_track_id, set())
                next_ambiguous = detection.frame_idx in ambiguous.get(source_track_id, set())
                split = prev_ambiguous != next_ambiguous
            if split:
                fragments.append(_make_fragment(next_fragment_id, source_track_id, current))
                next_fragment_id += 1
                current = [detection]
            else:
                current.append(detection)
        if current:
            fragments.append(_make_fragment(next_fragment_id, source_track_id, current))
            next_fragment_id += 1
    return fragments


def _make_fragment(fragment_id: int, source_track_id: int, detections: Sequence[RepairDetection]) -> _Fragment:
    deduped: dict[int, RepairDetection] = {}
    for detection in detections:
        existing = deduped.get(detection.frame_idx)
        if existing is None or detection.conf > existing.conf:
            deduped[detection.frame_idx] = detection
    return _Fragment(
        fragment_id=fragment_id,
        source_track_id=source_track_id,
        detections=tuple(sorted(deduped.values(), key=lambda detection: detection.frame_idx)),
    )


def _connect_fragments(
    fragments: Sequence[_Fragment],
    *,
    fps: float,
    config: RepairConfig,
) -> tuple[list[_Cluster], int]:
    clusters = [_Cluster((fragment,)) for fragment in fragments]
    merged = 0
    while len(clusters) > config.expected_players:
        best_cost = math.inf
        best_pair: tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i].frame_set & clusters[j].frame_set:
                    continue
                cost = _cluster_link_cost(clusters[i], clusters[j], fps=fps, config=config)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)
        if best_pair is None or not math.isfinite(best_cost) or best_cost > config.max_merge_cost:
            break
        i, j = best_pair
        merged_cluster = _Cluster(clusters[i].fragments + clusters[j].fragments)
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in {i, j}]
        clusters.append(merged_cluster)
        merged += 1
    return clusters, merged


def _cap_fragments_for_global(
    fragments: Sequence[_Fragment],
    *,
    config: RepairConfig,
) -> list[_Fragment]:
    if config.max_fragments_for_global <= 0 or len(fragments) <= config.max_fragments_for_global:
        return list(fragments)
    by_source: dict[int, list[_Fragment]] = defaultdict(list)
    for fragment in fragments:
        by_source[fragment.source_track_id].append(fragment)
    kept: list[_Fragment] = []
    per_source_quota = max(1, config.max_fragments_for_global // max(config.expected_players * 4, 1))
    for source_fragments in by_source.values():
        kept.extend(sorted(source_fragments, key=lambda fragment: (-fragment.score, fragment.start_frame))[:per_source_quota])
    if len(kept) < config.max_fragments_for_global:
        kept_ids = {fragment.fragment_id for fragment in kept}
        remainder = [fragment for fragment in fragments if fragment.fragment_id not in kept_ids]
        kept.extend(
            sorted(remainder, key=lambda fragment: (-fragment.score, fragment.start_frame))[
                : config.max_fragments_for_global - len(kept)
            ]
        )
    return sorted(kept, key=lambda fragment: fragment.fragment_id)[: config.max_fragments_for_global]


def _cluster_link_cost(
    left: _Cluster,
    right: _Cluster,
    *,
    fps: float,
    config: RepairConfig,
) -> float:
    costs = [
        _fragment_link_cost(a, b, fps=fps, config=config)
        for a in left.fragments
        for b in right.fragments
    ]
    finite = [cost for cost in costs if math.isfinite(cost)]
    return min(finite) if finite else math.inf


def _fragment_link_cost(
    left: _Fragment,
    right: _Fragment,
    *,
    fps: float,
    config: RepairConfig,
) -> float:
    if not (left.end_frame < right.start_frame or right.end_frame < left.start_frame):
        return math.inf
    early, late = (left, right) if left.end_frame < right.start_frame else (right, left)
    gap = late.start_frame - early.end_frame
    if gap <= 0 or gap > config.max_merge_gap_frames:
        return math.inf
    dt = max(gap / fps, 1.0 / fps)
    early_velocity = _fragment_velocity(early, at_end=True, fps=fps)
    late_velocity = _fragment_velocity(late, at_end=False, fps=fps)
    early_last = early.detections[-1].world_xy
    late_first = late.detections[0].world_xy
    predicted_late = (early_last[0] + early_velocity[0] * dt, early_last[1] + early_velocity[1] * dt)
    predicted_early = (late_first[0] - late_velocity[0] * dt, late_first[1] - late_velocity[1] * dt)
    continuity_dist = 0.5 * (_point_distance(predicted_late, late_first) + _point_distance(predicted_early, early_last))
    direct_dist = _point_distance(early_last, late_first)
    direct_allowed = config.merge_distance_slack_m + config.max_merge_speed_m_s * dt
    if direct_dist > direct_allowed:
        return math.inf
    size_cost = _bbox_size_cost(early.detections[-1].bbox, late.detections[0].bbox)
    if not math.isfinite(size_cost):
        return math.inf
    continuity_allowed = max(config.merge_distance_slack_m, direct_allowed)
    source_bonus = -0.25 if early.source_track_id == late.source_track_id else 0.0
    gap_penalty = min(1.0, gap / max(config.max_merge_gap_frames, 1))
    return max(0.0, continuity_dist / continuity_allowed) + 0.25 * size_cost + 0.15 * gap_penalty + source_bonus


def _fragment_velocity(fragment: _Fragment, *, at_end: bool, fps: float) -> tuple[float, float]:
    detections = fragment.detections
    if len(detections) < 2:
        return (0.0, 0.0)
    window = detections[-5:] if at_end else detections[:5]
    first = window[0]
    last = window[-1]
    dt = (last.frame_idx - first.frame_idx) / fps
    if dt <= 0:
        return (0.0, 0.0)
    return ((last.world_xy[0] - first.world_xy[0]) / dt, (last.world_xy[1] - first.world_xy[1]) / dt)


def _clusters_to_tracks(
    clusters: Sequence[_Cluster],
    *,
    fps: float,
    config: RepairConfig,
) -> tuple[Tracks, int, int]:
    raw_players = [_dedupe_cluster_detections(cluster) for cluster in clusters]
    raw_players = [items for items in raw_players if items]
    sorted_players = sorted(raw_players, key=_player_sort_key)
    synthetic_total = 0
    skipped_overlap_total = 0
    filled_players: list[list[RepairDetection]] = []
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
    return Tracks(schema_version=1, fps=fps, players=players, rally_spans=[]), synthetic_total, skipped_overlap_total


def _dedupe_cluster_detections(cluster: _Cluster) -> list[RepairDetection]:
    by_frame: dict[int, RepairDetection] = {}
    for detection in cluster.detections:
        existing = by_frame.get(detection.frame_idx)
        if existing is None or detection.conf > existing.conf:
            by_frame[detection.frame_idx] = detection
    return [by_frame[frame_idx] for frame_idx in sorted(by_frame)]


def _fill_short_gaps(
    detections: Sequence[RepairDetection],
    *,
    other_detections: Sequence[RepairDetection],
    fps: float,
    config: RepairConfig,
) -> tuple[list[RepairDetection], int, int]:
    if config.max_gap_fill_frames <= 0 or len(detections) < 2:
        return list(detections), 0, 0
    other_by_frame: dict[int, list[RepairDetection]] = defaultdict(list)
    for detection in other_detections:
        other_by_frame[detection.frame_idx].append(detection)

    out: list[RepairDetection] = []
    synthetic_count = 0
    skipped_overlap = 0
    for current, nxt in zip(detections, detections[1:], strict=False):
        out.append(current)
        gap = nxt.frame_idx - current.frame_idx
        if gap <= 1 or gap > config.max_gap_fill_frames:
            continue
        dt = gap / fps
        if _point_distance(current.world_xy, nxt.world_xy) > config.merge_distance_slack_m + config.max_gap_fill_speed_m_s * dt:
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


def _interpolate_detection(left: RepairDetection, right: RepairDetection, frame_idx: int) -> RepairDetection:
    alpha = (frame_idx - left.frame_idx) / (right.frame_idx - left.frame_idx)
    bbox = tuple((1.0 - alpha) * left.bbox[i] + alpha * right.bbox[i] for i in range(4))
    world_xy = (
        (1.0 - alpha) * left.world_xy[0] + alpha * right.world_xy[0],
        (1.0 - alpha) * left.world_xy[1] + alpha * right.world_xy[1],
    )
    conf = max(0.0, min(left.conf, right.conf, 0.35))
    return RepairDetection(
        frame_idx=frame_idx,
        source_track_id=left.source_track_id,
        bbox=bbox,  # type: ignore[arg-type]
        world_xy=world_xy,
        conf=max(conf, 0.0),
    )


def _identity_labels(players: Sequence[Sequence[RepairDetection]]) -> dict[int, dict[str, str]]:
    candidates = [
        TrackCandidate(track_id=idx, world_xy=list(_median_world_xy(detections)), confidence=_mean_conf(detections))
        for idx, detections in enumerate(players)
        if detections
    ]
    if len(candidates) == 4:
        assigned = assign_doubles_roles(candidates)
        return {
            int(track_id): {"side": identity.side, "role": identity.role}
            for track_id, identity in assigned.items()
        }
    return {
        idx: {
            "side": "near" if _median_world_xy(detections)[1] <= 0.0 else "far",
            "role": "singles" if len(players) == 2 else "unknown",
        }
        for idx, detections in enumerate(players)
        if detections
    }


def _player_sort_key(detections: Sequence[RepairDetection]) -> tuple[float, float, int]:
    xy = _median_world_xy(detections)
    return (xy[1], xy[0], detections[0].frame_idx if detections else 0)


def _cluster_first_frame(cluster: _Cluster) -> int:
    return min(fragment.start_frame for fragment in cluster.fragments)


def _median_world_xy(detections: Sequence[RepairDetection]) -> tuple[float, float]:
    xs = sorted(detection.world_xy[0] for detection in detections)
    ys = sorted(detection.world_xy[1] for detection in detections)
    if not xs:
        return (0.0, 0.0)
    mid = len(xs) // 2
    return (xs[mid], ys[mid])


def _mean_conf(detections: Sequence[RepairDetection]) -> float:
    return sum(detection.conf for detection in detections) / len(detections) if detections else 0.0


def _ambiguous_frames_by_source(
    detections: Sequence[RepairDetection],
    config: RepairConfig,
) -> dict[int, set[int]]:
    by_frame: dict[int, list[RepairDetection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame_idx].append(detection)
    ambiguous: dict[int, set[int]] = defaultdict(set)
    for frame_idx, items in by_frame.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if (
                    _bbox_iou(items[i].bbox, items[j].bbox) >= config.crossover_iou_threshold
                    or _point_distance(items[i].world_xy, items[j].world_xy) <= config.crossover_world_distance_m
                ):
                    ambiguous[items[i].source_track_id].add(frame_idx)
                    ambiguous[items[j].source_track_id].add(frame_idx)
    return ambiguous


def _nms_by_frame(detections: Sequence[RepairDetection], iou_threshold: float) -> list[RepairDetection]:
    if iou_threshold <= 0:
        return list(detections)
    by_frame: dict[int, list[RepairDetection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame_idx].append(detection)
    kept: list[RepairDetection] = []
    for frame_idx in sorted(by_frame):
        frame_kept: list[RepairDetection] = []
        for candidate in sorted(by_frame[frame_idx], key=lambda detection: detection.conf, reverse=True):
            if all(_bbox_iou(candidate.bbox, existing.bbox) <= iou_threshold for existing in frame_kept):
                frame_kept.append(candidate)
        kept.extend(frame_kept)
    return sorted(kept, key=lambda detection: (detection.frame_idx, detection.source_track_id, -detection.conf))


def _count_same_frame_overlaps(tracks: Tracks, iou_threshold: float) -> int:
    by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(float(frame.t) * float(tracks.fps)))
            by_frame[frame_idx].append((int(player.id), tuple(float(value) for value in frame.bbox)))  # type: ignore[arg-type]
    count = 0
    for items in by_frame.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0] and _bbox_iou(items[i][1], items[j][1]) > iou_threshold:
                    count += 1
    return count


def _count_teleports(tracks: Tracks, *, fps: float, max_speed_m_s: float) -> int:
    count = 0
    for player in tracks.players:
        frames = sorted(player.frames, key=lambda frame: frame.t)
        for previous, current in zip(frames, frames[1:], strict=False):
            frame_gap = int(round((float(current.t) - float(previous.t)) * fps))
            if frame_gap <= 0:
                continue
            dt = frame_gap / fps
            previous_xy = tuple(float(value) for value in previous.world_xy)
            current_xy = tuple(float(value) for value in current.world_xy)
            if _point_distance(previous_xy, current_xy) > max_speed_m_s * dt + 1.25:
                count += 1
    return count


def _bbox_size_cost(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    aw, ah = max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])
    bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    wr = max(aw / bw, bw / aw)
    hr = max(ah / bh, bh / ah)
    if wr > 4.0 or hr > 4.0:
        return math.inf
    return math.log(wr) + math.log(hr)


def _world_distance(a: RepairDetection, b: RepairDetection) -> float:
    return _point_distance(a.world_xy, b.world_xy)


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


def _bbox_xyxy(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four xyxy values")
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("detection bbox must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _track_key(detection: dict[str, Any], fallback: int) -> int | str:
    for field in ("player_id", "track_id", "temp_track_id", "temp_id", "id"):
        value = detection.get(field)
        if value is not None:
            return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else str(value)
    return fallback


def _int_track_id(key: int | str, mapping: dict[str, int], used_ids: set[int]) -> int:
    if isinstance(key, int):
        used_ids.add(key)
        return key
    if key not in mapping:
        next_id = 1
        while next_id in used_ids or next_id in mapping.values():
            next_id += 1
        mapping[key] = next_id
    return mapping[key]


def _frame_index(frame_entry: dict[str, Any], default: int) -> int:
    return int(frame_entry.get("frame", frame_entry.get("frame_index", default)))


def _is_person_detection(detection: dict[str, Any]) -> bool:
    value = detection.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


__all__ = [
    "PlayerIdRepairSummary",
    "RepairConfig",
    "RepairDetection",
    "repair_detection_payload_to_tracks",
    "repair_detections_to_tracks",
    "repair_tracks",
]
