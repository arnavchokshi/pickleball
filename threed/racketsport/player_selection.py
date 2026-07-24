"""Preview-band, source-only player selection over a high-recall person pool.

The selector is deliberately downstream of global association.  It treats every
frame that can be joined back to the raw pool as measured, treats unjoined frames
as synthetic, and never lets a long/identity-ambiguous geometric bridge survive.
No labels are consumed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import copy
import math
from typing import Any, Iterable, Mapping, Sequence

from .court_templates import Sport, get_court_template
from .doubles_id import assign_doubles_roles
from .person_fast import person_detection_from_bbox
from .schemas import CourtCalibration
from .track_lock import TrackCandidate


class OpenSetDecision(str, Enum):
    ACCEPT = "accept"
    DEFER = "defer"
    REJECT = "reject"


EXACTLY_FOUR_HARD_CAP = 4
COURT_REGION_HARD_BOUND_M = 1.0


@dataclass(frozen=True)
class HardConstraintConfig:
    """Pre-registered, non-tunable player-export constraints."""

    exactly_four_hard_cap: int = EXACTLY_FOUR_HARD_CAP
    court_region_hard_bound_m: float = COURT_REGION_HARD_BOUND_M

    def __post_init__(self) -> None:
        if self.exactly_four_hard_cap != EXACTLY_FOUR_HARD_CAP:
            raise ValueError(
                f"exactly_four_hard_cap is frozen at {EXACTLY_FOUR_HARD_CAP}"
            )
        if not math.isclose(
            self.court_region_hard_bound_m,
            COURT_REGION_HARD_BOUND_M,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "court_region_hard_bound_m is frozen at "
                f"{COURT_REGION_HARD_BOUND_M}"
            )


_REID_POLICY_SCOPE = "player_selection_post_association"
_REID_GALLERY = "gallery_only_not_consulted"
_REID_ELIGIBLE = "reid_eligible_ambiguous"
_REID_INVOKED = "reid_invoked_ambiguous"
_REID_SKIPPED = "reid_skipped_unambiguous"
_REID_UNAVAILABLE = "reid_unavailable"


@dataclass(frozen=True)
class PlayerSelectionConfig:
    """Frozen registered operating point from DESIGN_selection_layer.md."""

    expected_players: int = 4
    sigma_court_m: float = 0.5
    court_ema_half_life_s: float = 2.0
    identity_accept_distance: float = 0.35
    identity_reject_distance: float = 0.42
    max_displacement_m: float = 2.5
    max_micro_fill_frames: int = 12
    court_weight: float = 0.4
    identity_weight: float = 0.4
    persistence_weight: float = 0.2
    selection_score_min: float = 0.5
    enrollment_court_presence_min: float = 0.8
    enrollment_max_pairwise_iou: float = 0.2
    enrollment_min_window_s: float = 1.0
    recovery_max_speed_m_s: float = 7.0
    raw_match_bbox_delta_px: float = 3.0
    raw_match_iou_min: float = 0.98
    sport: Sport = "pickleball"

    def __post_init__(self) -> None:
        if self.expected_players <= 0:
            raise ValueError("expected_players must be positive")
        if self.sigma_court_m <= 0.0 or self.court_ema_half_life_s <= 0.0:
            raise ValueError("court sigma and EMA half-life must be positive")
        if not 0.0 <= self.identity_accept_distance < self.identity_reject_distance:
            raise ValueError(
                "identity thresholds must define an ordered non-negative band"
            )
        if self.max_displacement_m < 0.0 or self.max_micro_fill_frames < 0:
            raise ValueError("gap limits must be non-negative")
        if any(weight < 0.0 for weight in self.fusion_weights):
            raise ValueError("fusion weights must be non-negative")
        if not math.isclose(sum(self.fusion_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fusion weights must sum to one")
        if not 0.0 <= self.selection_score_min <= 1.0:
            raise ValueError("selection_score_min must be in [0, 1]")
        if self.enrollment_min_window_s <= 0.0 or self.recovery_max_speed_m_s <= 0.0:
            raise ValueError("enrollment window and recovery speed must be positive")

    @property
    def fusion_weights(self) -> tuple[float, float, float]:
        return (self.court_weight, self.identity_weight, self.persistence_weight)


@dataclass(frozen=True)
class SelectionDetection:
    frame_idx: int
    source_track_id: int
    bbox: tuple[float, float, float, float]
    world_xy: tuple[float, float]
    conf: float
    embedding: tuple[float, ...] | None = None
    interpolated: bool = False
    raw_detection_uid: str | None = None
    payload: Mapping[str, Any] | None = field(default=None, compare=False)
    reid_policy: str = field(default=_REID_GALLERY, compare=False)


@dataclass(frozen=True)
class TrackFragment:
    fragment_id: str
    source_track_id: int
    detections: tuple[SelectionDetection, ...]
    seed_player_id: int | None = None

    def __post_init__(self) -> None:
        if not self.detections:
            raise ValueError("fragment detections cannot be empty")
        frames = [detection.frame_idx for detection in self.detections]
        if frames != sorted(frames) or len(frames) != len(set(frames)):
            raise ValueError(
                "fragment detections must have unique sorted frame indexes"
            )

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx


@dataclass(frozen=True)
class SelectionSlot:
    slot_id: int
    side: str
    role: str
    embedding_centroid: tuple[float, ...] | None
    enrolled_frames: tuple[int, ...]
    source_fragment_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class PresenceEvidence:
    court_presence: float
    persistence: float
    mean_visibility: float
    real_detection_count: int
    span_frames: int


@dataclass(frozen=True)
class StitchDecision:
    refused: bool
    open_set: OpenSetDecision
    embedding_distance: float | None
    displacement_m: float
    net_crossing: bool
    real_support_in_gap: int
    evidence_classes: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _SlotCandidate:
    fragment_id: str
    slot_id: int
    fusion_score: float
    embedding_distance: float | None
    open_set: OpenSetDecision
    role_consistent: bool
    temporal_motion_consistent: bool
    stitch_vetoed: bool
    registered_owner_continuity: bool
    feasible: bool


@dataclass
class _SelectiveReIDAudit:
    """Mutable run audit for ambiguity-triggered identity comparisons."""

    provider_available: bool = True
    provider_reason: str | None = None
    frames_ambiguous: set[int] = field(default_factory=set)
    frames_invoked: set[int] = field(default_factory=set)
    frames_skipped: set[int] = field(default_factory=set)
    frames_unavailable: set[int] = field(default_factory=set)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.provider_available:
            self.decisions.append(
                {
                    "action": _REID_UNAVAILABLE,
                    "policy_scope": _REID_POLICY_SCOPE,
                    "level": "provider",
                    "reasons": [self.provider_reason or "embedding_provider_unavailable"],
                }
            )

    def identity_distance(
        self,
        left: Sequence[float] | None,
        right: Sequence[float] | None,
        *,
        trigger_frames: Iterable[int],
        comparison: str,
        context: Mapping[str, Any] | None = None,
    ) -> float | None:
        """Run one comparison only when an ambiguity trigger names its frames."""

        frames = tuple(sorted({int(frame_idx) for frame_idx in trigger_frames}))
        if not frames:
            return None
        non_ambiguous = set(frames) - self.frames_ambiguous
        if non_ambiguous:
            raise ValueError(
                "selective ReID comparison attributed to non-ambiguous frame(s): "
                + ", ".join(str(frame_idx) for frame_idx in sorted(non_ambiguous))
            )
        distance = (
            cosine_distance(left, right)
            if self.provider_available and left is not None and right is not None
            else None
        )
        action = _REID_INVOKED if distance is not None else _REID_UNAVAILABLE
        target = self.frames_invoked if distance is not None else self.frames_unavailable
        target.update(frames)
        for frame_idx in frames:
            decision: dict[str, Any] = copy.deepcopy(dict(context or {}))
            decision.update(
                {
                    "action": action,
                    "policy_scope": _REID_POLICY_SCOPE,
                    "frame_idx": frame_idx,
                    "comparison": comparison,
                    "reasons": [
                        "identity_comparison_executed_on_ambiguous_frame"
                        if distance is not None
                        else "identity_evidence_unavailable_on_ambiguous_frame"
                    ],
                }
            )
            if distance is not None:
                decision["embedding_distance"] = distance
            self.decisions.append(decision)
        return distance

    def summary_decision(self) -> dict[str, Any]:
        if self.frames_invoked - self.frames_ambiguous:
            raise ValueError("ReID invoked outside the registered ambiguous-frame set")
        if self.frames_skipped & self.frames_ambiguous:
            raise ValueError("ReID frame cannot be both ambiguous and unambiguous")
        unavailable_count = len(self.frames_unavailable)
        if not self.provider_available:
            unavailable_count = max(1, unavailable_count)
        return {
            "action": "selective_reid_policy_summary",
            "policy_scope": _REID_POLICY_SCOPE,
            "policy": "consult_only_on_zero_or_two_plus_in_court_slot_candidates",
            "counter_unit": "unique_frame_plus_provider_event",
            "frames_ambiguous": len(self.frames_ambiguous),
            "reid_invoked": len(self.frames_invoked),
            "reid_skipped_unambiguous": len(self.frames_skipped),
            "reid_unavailable": unavailable_count,
            "provider_available": self.provider_available,
            "provider_reason": self.provider_reason,
            "warning_codes": [_REID_UNAVAILABLE] if unavailable_count else [],
        }


_DESTRUCTIVE_EVIDENCE_CLASSES = frozenset({"appearance", "geometry"})

DROP_TRIGGER_REASON_VOCABULARY = (
    "appearance_all_detections_all_slots_at_or_above_identity_reject_distance_0_42",
    "geometry_detection_persistence_below_selection_score_min_0_5",
    "geometry_all_slots_temporal_overlap_or_speed_above_7_0_m_s",
)


def open_set_decision(
    distance: float | None, config: PlayerSelectionConfig | None = None
) -> OpenSetDecision:
    cfg = config or PlayerSelectionConfig()
    if distance is None:
        return OpenSetDecision.DEFER
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("cosine distance must be finite and non-negative")
    if distance <= cfg.identity_accept_distance:
        return OpenSetDecision.ACCEPT
    if distance >= cfg.identity_reject_distance:
        return OpenSetDecision.REJECT
    return OpenSetDecision.DEFER


def cosine_distance(
    left: Sequence[float] | None, right: Sequence[float] | None
) -> float | None:
    if left is None or right is None:
        return None
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    similarity = sum(float(a) * float(b) for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(0.0, min(2.0, 1.0 - similarity))


def _identity_distance(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    *,
    reid_audit: _SelectiveReIDAudit | None,
    trigger_frames: Iterable[int] = (),
    comparison: str,
    context: Mapping[str, Any] | None = None,
) -> float | None:
    if reid_audit is None:
        return cosine_distance(left, right)
    return reid_audit.identity_distance(
        left,
        right,
        trigger_frames=trigger_frames,
        comparison=comparison,
        context=context,
    )


def embedding_centroid(
    detections: Sequence[SelectionDetection],
) -> tuple[float, ...] | None:
    vectors = [
        _consultable_embedding(detection)
        for detection in detections
        if _consultable_embedding(detection) is not None
        and not detection.interpolated
    ]
    return _normalized_embedding_centroid(vectors)


def _gallery_embedding_centroid(
    detections: Sequence[SelectionDetection],
) -> tuple[float, ...] | None:
    """Build enrollment/reference state without making an identity decision."""

    vectors = [
        detection.embedding
        for detection in detections
        if detection.embedding is not None and not detection.interpolated
    ]
    return _normalized_embedding_centroid(vectors)


def _normalized_embedding_centroid(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("fragment embeddings must have one non-zero dimension")
    mean = [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm <= 0.0:
        return None
    return tuple(value / norm for value in mean)


def off_court_excess_m(
    world_xy: Sequence[float], *, sport: Sport = "pickleball"
) -> float:
    """Match the frozen scorer's rectangular excess calculation."""

    if len(world_xy) != 2:
        raise ValueError("world_xy must contain two values")
    template = get_court_template(sport)
    dx = max(0.0, abs(float(world_xy[0])) - template.width_m / 2.0)
    dy = max(0.0, abs(float(world_xy[1])) - template.length_m / 2.0)
    return math.hypot(dx, dy)


def median_real_footpoint_court_excess_m(
    detections: Sequence[SelectionDetection],
    *,
    sport: Sport = "pickleball",
) -> float | None:
    """Return lifetime median excess over measured footpoints only."""

    excesses = sorted(
        off_court_excess_m(detection.world_xy, sport=sport)
        for detection in detections
        if not detection.interpolated
    )
    return _median(excesses) if excesses else None


def _consultable_embedding(
    detection: SelectionDetection,
) -> tuple[float, ...] | None:
    if detection.reid_policy in {_REID_SKIPPED, _REID_UNAVAILABLE}:
        return None
    return detection.embedding


def _provisional_slot_key(detection: SelectionDetection) -> tuple[str, str]:
    x, y = detection.world_xy
    return ("near" if y < 0.0 else "far", "left" if x < 0.0 else "right")


def _apply_selective_reid_policy(
    detections: Sequence[SelectionDetection],
    *,
    hard_config: HardConstraintConfig,
    sport: Sport,
    frame_indexes: Iterable[int],
    provider_available: bool,
    provider_reason: str | None,
) -> tuple[tuple[SelectionDetection, ...], _SelectiveReIDAudit]:
    """Classify appearance consultability using geometry/cardinality only."""

    audit = _SelectiveReIDAudit(
        provider_available=provider_available,
        provider_reason=provider_reason,
    )
    by_frame: dict[int, list[SelectionDetection]] = defaultdict(list)
    for frame_idx in frame_indexes:
        by_frame[int(frame_idx)]
    for detection in detections:
        if not detection.interpolated:
            by_frame[detection.frame_idx].append(detection)

    slot_keys = (
        ("near", "left"),
        ("near", "right"),
        ("far", "left"),
        ("far", "right"),
    )
    policy_by_uid: dict[str, str] = {}
    for frame_idx in sorted(by_frame):
        frame_detections = by_frame[frame_idx]
        in_region = [
            detection
            for detection in frame_detections
            if off_court_excess_m(detection.world_xy, sport=sport)
            <= hard_config.court_region_hard_bound_m
        ]
        by_slot: dict[tuple[str, str], list[SelectionDetection]] = {
            key: [] for key in slot_keys
        }
        for detection in in_region:
            by_slot[_provisional_slot_key(detection)].append(detection)

        frame_is_ambiguous = any(len(by_slot[key]) != 1 for key in slot_keys)
        if frame_is_ambiguous:
            audit.frames_ambiguous.add(frame_idx)
        else:
            audit.frames_skipped.add(frame_idx)

        for side, role in slot_keys:
            candidates = sorted(
                by_slot[(side, role)],
                key=lambda item: item.raw_detection_uid or "",
            )
            candidate_uids = [_required_raw_uid(item) for item in candidates]
            if len(candidates) == 1 and not frame_is_ambiguous:
                candidate = candidates[0]
                policy_by_uid[_required_raw_uid(candidate)] = _REID_SKIPPED
                audit.decisions.append(
                    {
                        "action": _REID_SKIPPED,
                        "policy_scope": _REID_POLICY_SCOPE,
                        "frame_idx": frame_idx,
                        "slot_side": side,
                        "slot_role": role,
                        "candidate_real_detections": 1,
                        "raw_detection_uids": candidate_uids,
                        "reasons": ["exactly_one_in_court_candidate_for_slot"],
                    }
                )
                continue

            missing_identity = (
                not provider_available
                or not candidates
                or any(candidate.embedding is None for candidate in candidates)
            )
            action = _REID_UNAVAILABLE if missing_identity else _REID_ELIGIBLE
            if missing_identity:
                audit.frames_unavailable.add(frame_idx)
            reason = (
                "embedding_provider_unavailable"
                if not provider_available
                else "zero_in_court_candidates_for_slot"
                if not candidates
                else "embedding_absent_for_ambiguous_slot_frame"
                if any(candidate.embedding is None for candidate in candidates)
                else "two_or_more_in_court_candidates_for_slot"
                if len(candidates) >= 2
                else "frame_ambiguous_for_another_slot"
            )
            for candidate in candidates:
                policy_by_uid[_required_raw_uid(candidate)] = action
            audit.decisions.append(
                {
                    "action": action,
                    "policy_scope": _REID_POLICY_SCOPE,
                    "frame_idx": frame_idx,
                    "slot_side": side,
                    "slot_role": role,
                    "candidate_real_detections": len(candidates),
                    "raw_detection_uids": candidate_uids,
                    "reasons": [reason],
                }
            )

        for detection in frame_detections:
            uid = _required_raw_uid(detection)
            if uid in policy_by_uid:
                continue
            action = (
                _REID_UNAVAILABLE
                if frame_is_ambiguous
                and (not provider_available or detection.embedding is None)
                else _REID_ELIGIBLE
                if frame_is_ambiguous
                else _REID_SKIPPED
            )
            policy_by_uid[uid] = action
            if action == _REID_UNAVAILABLE:
                audit.frames_unavailable.add(frame_idx)
            audit.decisions.append(
                {
                    "action": action,
                    "policy_scope": _REID_POLICY_SCOPE,
                    "frame_idx": frame_idx,
                    "candidate_scope": "off_court_not_competing_for_slot",
                    "candidate_real_detections": 0,
                    "raw_detection_uids": [uid],
                    "reasons": [
                        "off_court_detection_consultable_only_because_in_court_slot_is_ambiguous"
                        if frame_is_ambiguous and action == _REID_ELIGIBLE
                        else "embedding_unavailable_on_ambiguous_frame"
                        if frame_is_ambiguous
                        else "off_court_detection_cannot_trigger_reid"
                    ],
                }
            )

    classified = tuple(
        replace(
            detection,
            reid_policy=policy_by_uid.get(
                _required_raw_uid(detection),
                _REID_UNAVAILABLE if not provider_available else _REID_GALLERY,
            ),
        )
        for detection in detections
    )
    return classified, audit


def registered_hard_constraint_exclusions(
    detections: Sequence[SelectionDetection],
    *,
    hard_config: HardConstraintConfig | None = None,
    sport: Sport = "pickleball",
) -> tuple[frozenset[int], list[dict[str, Any]]]:
    """Register lifetime court-bound exclusions without changing soft fusion."""

    cfg = hard_config or HardConstraintConfig()
    by_source: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in detections:
        if not detection.interpolated:
            by_source[detection.source_track_id].append(detection)
    excluded: set[int] = set()
    decisions: list[dict[str, Any]] = []
    for source_track_id in sorted(by_source):
        median_excess = median_real_footpoint_court_excess_m(
            by_source[source_track_id],
            sport=sport,
        )
        assert median_excess is not None
        hard_excluded = median_excess > cfg.court_region_hard_bound_m
        if hard_excluded:
            excluded.add(source_track_id)
        decisions.append(
            {
                "action": (
                    "hard_exclude_court_region"
                    if hard_excluded
                    else "hard_constraint_court_region_pass"
                ),
                "source_track_id": source_track_id,
                "real_detection_count": len(by_source[source_track_id]),
                "median_real_footpoint_court_excess_m": median_excess,
                "court_region_hard_bound_m": cfg.court_region_hard_bound_m,
                "comparison": "strictly_greater_than",
                "reasons": [
                    "lifetime_real_detection_median_excess_above_registered_bound"
                    if hard_excluded
                    else "lifetime_real_detection_median_excess_not_above_registered_bound"
                ],
            }
        )
    return frozenset(excluded), decisions


def frame_court_evidence(
    world_xy: Sequence[float], config: PlayerSelectionConfig | None = None
) -> float:
    cfg = config or PlayerSelectionConfig()
    return math.exp(-off_court_excess_m(world_xy, sport=cfg.sport) / cfg.sigma_court_m)


def presence_evidence(
    detections: Sequence[SelectionDetection],
    *,
    fps: float,
    config: PlayerSelectionConfig | None = None,
) -> PresenceEvidence:
    cfg = config or PlayerSelectionConfig()
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    real = sorted(
        (detection for detection in detections if not detection.interpolated),
        key=lambda item: item.frame_idx,
    )
    if not real:
        return PresenceEvidence(0.0, 0.0, 0.0, 0, 0)
    ema = frame_court_evidence(real[0].world_xy, cfg)
    previous_frame = real[0].frame_idx
    for detection in real[1:]:
        dt_s = max(0.0, (detection.frame_idx - previous_frame) / fps)
        alpha = 1.0 - math.exp(-math.log(2.0) * dt_s / cfg.court_ema_half_life_s)
        ema = (1.0 - alpha) * ema + alpha * frame_court_evidence(
            detection.world_xy, cfg
        )
        previous_frame = detection.frame_idx
    span = real[-1].frame_idx - real[0].frame_idx + 1
    persistence = len(real) / span
    visibility = sum(detection.conf for detection in real) / len(real)
    return PresenceEvidence(
        court_presence=max(0.0, min(1.0, ema)),
        persistence=max(0.0, min(1.0, persistence)),
        mean_visibility=max(0.0, min(1.0, visibility)),
        real_detection_count=len(real),
        span_frames=span,
    )


def fusion_score(
    *,
    court_presence: float,
    identity_match: float,
    persistence: float,
    config: PlayerSelectionConfig | None = None,
) -> float:
    cfg = config or PlayerSelectionConfig()
    values = (court_presence, identity_match, persistence)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("fusion evidence must be in [0, 1]")
    return sum(
        weight * value for weight, value in zip(cfg.fusion_weights, values, strict=True)
    )


def identity_match_score(
    distance: float | None, config: PlayerSelectionConfig | None = None
) -> float:
    cfg = config or PlayerSelectionConfig()
    if distance is None:
        return 0.5
    if distance <= cfg.identity_accept_distance:
        return 1.0
    if distance >= cfg.identity_reject_distance:
        return 0.0
    width = cfg.identity_reject_distance - cfg.identity_accept_distance
    return (cfg.identity_reject_distance - distance) / width


def destructive_action_allowed(evidence: Mapping[str, bool]) -> bool:
    """Require independent appearance and geometry disagreement."""

    unknown = set(evidence) - _DESTRUCTIVE_EVIDENCE_CLASSES
    if unknown:
        raise ValueError(
            "unknown or non-independent destructive evidence classes: "
            + ", ".join(sorted(unknown))
        )
    return all(evidence.get(name, False) for name in _DESTRUCTIVE_EVIDENCE_CLASSES)


def _destructive_drop_evidence(
    fragment: TrackFragment,
    candidates: Sequence[_SlotCandidate],
    *,
    slots: Sequence[SelectionSlot],
    presence: PresenceEvidence,
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    """Collapse every world-derived signal into one geometry evidence class."""

    appearance_rejected = all(
        _consultable_embedding(detection) is not None
        and all(
            open_set_decision(
                _identity_distance(
                    slot.embedding_centroid,
                    _consultable_embedding(detection),
                    reid_audit=reid_audit,
                    trigger_frames=(detection.frame_idx,),
                    comparison="destructive_drop",
                    context={
                        "slot_id": slot.slot_id,
                        "raw_detection_uid": detection.raw_detection_uid,
                    },
                ),
                config,
            )
            is OpenSetDecision.REJECT
            for slot in slots
        )
        for detection in fragment.detections
    )
    persistence_rejected = presence.persistence < config.selection_score_min
    temporal_motion_rejected = all(
        not candidate.temporal_motion_consistent for candidate in candidates
    )
    geometry_rejected = persistence_rejected or temporal_motion_rejected
    reasons: list[str] = []
    if appearance_rejected:
        reasons.append(DROP_TRIGGER_REASON_VOCABULARY[0])
    if persistence_rejected:
        reasons.append(DROP_TRIGGER_REASON_VOCABULARY[1])
    if temporal_motion_rejected:
        reasons.append(DROP_TRIGGER_REASON_VOCABULARY[2])
    return (
        {
            "appearance": appearance_rejected,
            "geometry": geometry_rejected,
        },
        tuple(reasons),
    )


def evaluate_stitch(
    left: TrackFragment,
    right: TrackFragment,
    *,
    real_support_in_gap: Sequence[SelectionDetection] = (),
    config: PlayerSelectionConfig | None = None,
    ambiguity_triggered: bool = True,
    reid_audit: _SelectiveReIDAudit | None = None,
    trigger_frames: Iterable[int] = (),
) -> StitchDecision:
    cfg = config or PlayerSelectionConfig()
    left_endpoint = left.detections[-1]
    right_endpoint = right.detections[0]
    if right_endpoint.frame_idx <= left_endpoint.frame_idx:
        raise ValueError("right fragment must start after left fragment")
    distance = _identity_distance(
        _gallery_embedding_centroid(left.detections),
        _gallery_embedding_centroid(right.detections),
        reid_audit=reid_audit,
        trigger_frames=trigger_frames if ambiguity_triggered else (),
        comparison="fragment_stitch",
        context={
            "left_fragment_id": left.fragment_id,
            "right_fragment_id": right.fragment_id,
        },
    )
    open_set = open_set_decision(distance, cfg)
    displacement = _point_distance(left_endpoint.world_xy, right_endpoint.world_xy)
    crossing = _net_crossing(left_endpoint.world_xy, right_endpoint.world_xy)
    support_count = sum(
        left_endpoint.frame_idx < detection.frame_idx < right_endpoint.frame_idx
        for detection in real_support_in_gap
        if not detection.interpolated
    )
    kinematic_implausible = displacement > cfg.max_displacement_m or (
        crossing and support_count == 0
    )
    appearance_wrong = open_set is OpenSetDecision.REJECT
    evidence = {
        "appearance": appearance_wrong,
        "geometry": kinematic_implausible,
    }
    # The registered defer rule is a reversible merge refusal, not a fragment drop.
    refused = open_set is OpenSetDecision.DEFER or destructive_action_allowed(evidence)
    reasons: list[str] = []
    if open_set is OpenSetDecision.REJECT:
        reasons.append("embedding_distance_at_or_above_reject")
    elif open_set is OpenSetDecision.DEFER:
        reasons.append("embedding_in_defer_band_or_unmeasurable")
    if displacement > cfg.max_displacement_m:
        reasons.append("endpoint_displacement_above_cap")
    if crossing and support_count == 0:
        reasons.append("unsupported_net_crossing")
    if not refused:
        reasons.append("stitch_not_vetoed")
    return StitchDecision(
        refused=refused,
        open_set=open_set,
        embedding_distance=distance,
        displacement_m=displacement,
        net_crossing=crossing,
        real_support_in_gap=support_count,
        evidence_classes=tuple(name for name, agrees in evidence.items() if agrees),
        reasons=tuple(reasons),
    )


def micro_fill_allowed(
    left: SelectionDetection,
    right: SelectionDetection,
    *,
    identity_distance: float | None,
    config: PlayerSelectionConfig | None = None,
) -> tuple[bool, tuple[str, ...]]:
    cfg = config or PlayerSelectionConfig()
    missing_frames = right.frame_idx - left.frame_idx - 1
    reasons: list[str] = []
    if missing_frames < 1:
        reasons.append("no_gap")
    if missing_frames > cfg.max_micro_fill_frames:
        reasons.append("gap_above_micro_fill_cap")
    if _point_distance(left.world_xy, right.world_xy) > cfg.max_displacement_m:
        reasons.append("endpoint_displacement_above_cap")
    if _net_crossing(left.world_xy, right.world_xy):
        reasons.append("net_crossing")
    if open_set_decision(identity_distance, cfg) is not OpenSetDecision.ACCEPT:
        reasons.append("same_identity_not_accepted")
    return not reasons, tuple(reasons)


def mark_micro_fill_provenance(
    frames: Sequence[Mapping[str, Any]],
    *,
    left: SelectionDetection,
    right: SelectionDetection,
    identity_distance: float | None,
    fps: float = 1.0,
    config: PlayerSelectionConfig | None = None,
) -> list[dict[str, Any]]:
    allowed, reasons = micro_fill_allowed(
        left, right, identity_distance=identity_distance, config=config
    )
    if not allowed:
        raise ValueError("micro-fill refused: " + ", ".join(reasons))
    expected = set(range(left.frame_idx + 1, right.frame_idx))
    if len(frames) != len(expected):
        raise ValueError(
            "micro-fill payload must contain every missing frame exactly once"
        )
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in frames:
        row = copy.deepcopy(dict(payload))
        frame_idx = _frame_index(row, fps=fps)
        if frame_idx not in expected or frame_idx in seen:
            raise ValueError("micro-fill payload frame indexes do not match the gap")
        row["interpolated"] = True
        output.append(row)
        seen.add(frame_idx)
    return output


def infer_active_player_count(
    detections: Sequence[SelectionDetection],
    *,
    fps: float,
    config: PlayerSelectionConfig | None = None,
) -> int | None:
    """Infer singles or doubles from measured, persistent on-court evidence.

    Doubles requires four distinct measured identities to coexist for either
    one source second or 20 percent of accepted frames (whichever is smaller).
    Otherwise two-player evidence selects singles.  This never invents a
    missing third or fourth identity.
    """

    cfg = config or PlayerSelectionConfig()
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    identities_by_frame: dict[int, set[int]] = defaultdict(set)
    for detection in detections:
        if detection.interpolated:
            continue
        if (
            frame_court_evidence(detection.world_xy, cfg)
            < cfg.enrollment_court_presence_min
        ):
            continue
        identities_by_frame[detection.frame_idx].add(detection.source_track_id)
    accepted_frames = [
        identities
        for identities in identities_by_frame.values()
        if len(identities) >= 2
    ]
    if not accepted_frames:
        return None
    one_second = max(1, int(math.ceil(fps)))
    twenty_percent = max(1, int(math.ceil(0.2 * len(accepted_frames))))
    doubles_threshold = min(one_second, twenty_percent)
    doubles_frames = sum(len(identities) >= 4 for identities in accepted_frames)
    if doubles_frames >= doubles_threshold:
        return 4
    return 2


def enroll_slots(
    fragments: Sequence[TrackFragment],
    *,
    fps: float,
    config: PlayerSelectionConfig | None = None,
) -> tuple[SelectionSlot, ...]:
    """Deterministically enroll the configured two or four active-player slots."""

    cfg = config or PlayerSelectionConfig()
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    window_frames = max(1, int(math.ceil(cfg.enrollment_min_window_s * fps)))
    by_fragment_frame = {
        fragment.fragment_id: {
            detection.frame_idx: detection
            for detection in fragment.detections
            if not detection.interpolated
        }
        for fragment in fragments
    }
    all_frames = sorted(
        {frame for frames in by_fragment_frame.values() for frame in frames}
    )
    for start in all_frames:
        required = tuple(range(start, start + window_frames))
        alive = [
            fragment
            for fragment in fragments
            if all(
                frame in by_fragment_frame[fragment.fragment_id] for frame in required
            )
            and all(
                frame_court_evidence(
                    by_fragment_frame[fragment.fragment_id][frame].world_xy, cfg
                )
                >= cfg.enrollment_court_presence_min
                for frame in required
            )
        ]
        if len(alive) != cfg.expected_players:
            continue
        if any(
            _bbox_iou(
                by_fragment_frame[left.fragment_id][frame].bbox,
                by_fragment_frame[right.fragment_id][frame].bbox,
            )
            > cfg.enrollment_max_pairwise_iou
            for index, left in enumerate(alive)
            for right in alive[index + 1 :]
            for frame in required
        ):
            continue
        enrollment_detections = {
            fragment.fragment_id: tuple(
                by_fragment_frame[fragment.fragment_id][frame] for frame in required
            )
            for fragment in alive
        }
        enrollment_centroids = {
            fragment.fragment_id: _gallery_embedding_centroid(
                enrollment_detections[fragment.fragment_id]
            )
            for fragment in alive
        }
        # An enrollment owner without a usable appearance centroid cannot support
        # open-set re-entry.  Treat that window as ineligible instead of creating
        # a slot that later ranks every fragment from geometry alone.
        if any(centroid is None for centroid in enrollment_centroids.values()):
            continue
        world_centroids = {
            fragment.fragment_id: _median_world_xy(
                enrollment_detections[fragment.fragment_id]
            )
            for fragment in alive
        }
        if any(
            not all(math.isfinite(value) for value in centroid)
            for centroid in world_centroids.values()
        ):
            continue
        ordered = sorted(
            alive,
            key=lambda fragment: (
                world_centroids[fragment.fragment_id][1],
                world_centroids[fragment.fragment_id][0],
                fragment.fragment_id,
            ),
        )
        candidates = [
            TrackCandidate(
                track_id=index + 1,
                world_xy=list(world_centroids[fragment.fragment_id]),
                confidence=presence_evidence(
                    fragment.detections, fps=fps, config=cfg
                ).mean_visibility,
            )
            for index, fragment in enumerate(ordered)
        ]
        roles = assign_doubles_roles(candidates)
        slots: list[SelectionSlot] = []
        for index, fragment in enumerate(ordered, start=1):
            evidence = presence_evidence(fragment.detections, fps=fps, config=cfg)
            identity = roles[index]
            slots.append(
                SelectionSlot(
                    slot_id=index,
                    side=identity.side,
                    role=identity.role,
                    embedding_centroid=enrollment_centroids[fragment.fragment_id],
                    enrolled_frames=required,
                    source_fragment_ids=(fragment.fragment_id,),
                    confidence=(
                        evidence.court_presence
                        + evidence.persistence
                        + evidence.mean_visibility
                    )
                    / 3.0,
                )
            )
        return tuple(slots)
    return ()


def recover_identity_conditioned_pool(
    slot: SelectionSlot,
    *,
    last_detection: SelectionDetection,
    pool: Sequence[SelectionDetection],
    fps: float,
    occupied_frames: Iterable[int] = (),
    next_detection: SelectionDetection | None = None,
    preferred_source_track_ids: Iterable[int] = (),
    config: PlayerSelectionConfig | None = None,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> tuple[SelectionDetection, ...]:
    """Recover measured detections one frame at a time; never create geometry."""

    cfg = config or PlayerSelectionConfig()
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    occupied = set(occupied_frames)
    preferred_sources = frozenset(preferred_source_track_ids)
    recovered: list[SelectionDetection] = []
    anchor = last_detection
    by_frame: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in pool:
        if detection.interpolated:
            continue
        if detection.frame_idx <= last_detection.frame_idx:
            continue
        if (
            next_detection is not None
            and detection.frame_idx >= next_detection.frame_idx
        ):
            continue
        by_frame[detection.frame_idx].append(detection)
    for frame_idx in sorted(by_frame):
        if frame_idx in occupied:
            continue
        feasible: list[tuple[int, float, float, str, SelectionDetection]] = []
        for detection in by_frame[frame_idx]:
            dt_s = (detection.frame_idx - anchor.frame_idx) / fps
            if (
                _point_distance(anchor.world_xy, detection.world_xy)
                > cfg.recovery_max_speed_m_s * dt_s
            ):
                continue
            if next_detection is not None:
                remaining_s = (next_detection.frame_idx - detection.frame_idx) / fps
                if remaining_s <= 0.0:
                    continue
                if (
                    _point_distance(detection.world_xy, next_detection.world_xy)
                    > cfg.recovery_max_speed_m_s * remaining_s
                ):
                    continue
            consultable = _consultable_embedding(detection)
            if consultable is None:
                continue
            distance = _identity_distance(
                slot.embedding_centroid,
                consultable,
                reid_audit=reid_audit,
                trigger_frames=(detection.frame_idx,),
                comparison="identity_conditioned_pool_recovery",
                context={
                    "slot_id": slot.slot_id,
                    "raw_detection_uid": detection.raw_detection_uid,
                },
            )
            if open_set_decision(distance, cfg) is not OpenSetDecision.ACCEPT:
                continue
            feasible.append(
                (
                    0 if detection.source_track_id in preferred_sources else 1,
                    float("inf") if distance is None else distance,
                    -detection.conf,
                    detection.raw_detection_uid or "",
                    detection,
                )
            )
        if not feasible:
            continue
        detection = min(
            feasible,
            key=lambda row: (row[0], row[1], row[2], row[3], row[4].source_track_id),
        )[4]
        recovered.append(detection)
        occupied.add(detection.frame_idx)
        anchor = detection
    return tuple(recovered)


def select_players_payload(
    tracks_payload: Mapping[str, Any],
    *,
    raw_pool_payload: Mapping[str, Any] | None = None,
    embedding_payload: Mapping[str, Any] | None = None,
    calibration: CourtCalibration | None = None,
    enabled: bool = False,
    config: PlayerSelectionConfig | None = None,
    hard_config: HardConstraintConfig | None = None,
    embedding_bbox_scale: float = 1.0,
    reid_provider_available: bool | None = None,
    reid_provider_reason: str | None = None,
    auto_player_count: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clean association output and emit a decision-complete preview report.

    Disabled mode is a semantic no-op.  Byte identity is implemented by the CLI
    with a direct file copy because parsing cannot preserve whitespace.
    """

    cfg = config or PlayerSelectionConfig()
    hard_cfg = hard_config or HardConstraintConfig()
    if not enabled:
        report = _base_report(cfg, status="disabled_noop")
        players = tracks_payload.get("players")
        if isinstance(players, list):
            player_count = len(players)
            track_frame_count = sum(
                len(player.get("frames", []))
                for player in players
                if isinstance(player, Mapping)
                and isinstance(player.get("frames", []), list)
            )
            report["input_counts"]["players"] = player_count
            report["input_counts"]["track_frames"] = track_frame_count
            report["output_counts"]["players"] = player_count
            report["output_counts"]["track_frames"] = track_frame_count
        return copy.deepcopy(dict(tracks_payload)), report
    if raw_pool_payload is None or calibration is None:
        raise ValueError("enabled selection requires raw pool and calibration")
    if not math.isfinite(embedding_bbox_scale) or embedding_bbox_scale <= 0.0:
        raise ValueError("embedding_bbox_scale must be finite and positive")
    provider_available = (
        embedding_payload is not None
        if reid_provider_available is None
        else bool(reid_provider_available)
    )
    if embedding_payload is None:
        embedding_payload = {
            "source_only": True,
            "uses_cvat_labels": False,
            "promote_trk": False,
            "feature_dim": None,
            "detections": [],
        }
    attestations = _validate_source_only_embeddings(embedding_payload)

    real_pool = _raw_pool_detections(
        raw_pool_payload,
        embedding_payload=embedding_payload,
        calibration=calibration,
        config=cfg,
        embedding_bbox_scale=embedding_bbox_scale,
    )
    raw_uids = [detection.raw_detection_uid for detection in real_pool]
    if any(uid is None for uid in raw_uids) or len(raw_uids) != len(set(raw_uids)):
        raise ValueError("raw person detections must carry unique immutable UIDs")
    real_by_frame: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in real_pool:
        real_by_frame[detection.frame_idx].append(detection)

    fps = float(tracks_payload.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("tracks payload fps must be finite and positive")
    players = tracks_payload.get("players")
    if not isinstance(players, list):
        raise ValueError("tracks payload must contain a players list")
    for player in players:
        if not isinstance(player, Mapping):
            raise ValueError("track players must be objects")
        if not isinstance(player.get("frames"), list):
            raise ValueError("track player frames must be a list")

    frame_indexes = {
        _frame_index(frame, fps=fps)
        for player in players
        for frame in player.get("frames", [])
        if isinstance(frame, Mapping)
    }
    real_pool, reid_audit = _apply_selective_reid_policy(
        real_pool,
        hard_config=hard_cfg,
        sport=cfg.sport,
        frame_indexes=frame_indexes,
        provider_available=provider_available,
        provider_reason=reid_provider_reason,
    )
    real_by_frame = defaultdict(list)
    for detection in real_pool:
        real_by_frame[detection.frame_idx].append(detection)
    hard_excluded_source_ids, hard_decisions = registered_hard_constraint_exclusions(
        real_pool,
        hard_config=hard_cfg,
        sport=cfg.sport,
    )
    inferred_player_count: int | None = None
    if auto_player_count:
        inferred_player_count = infer_active_player_count(
            [
                detection
                for detection in real_pool
                if detection.source_track_id not in hard_excluded_source_ids
            ],
            fps=fps,
            config=cfg,
        )
        cfg = replace(cfg, expected_players=inferred_player_count or 2)

    output = copy.deepcopy(dict(tracks_payload))
    report = _base_report(cfg, status="preview_selection_complete")
    report["source_attestations"] = attestations
    report["source_only"] = True
    report["selection_mode"] = (
        f"auto_{cfg.expected_players}_player_enrollment"
        if auto_player_count
        else "enrolled_four_slot"
    )
    if auto_player_count:
        report["player_cardinality"] = {
            "mode": "auto_two_or_four",
            "inferred_players": inferred_player_count,
            "selected_target": cfg.expected_players,
            "hard_maximum": hard_cfg.exactly_four_hard_cap,
        }
    association_join_count = _association_raw_join_count(
        players,
        fps=fps,
        real_by_frame=real_by_frame,
        config=cfg,
    )
    report["input_counts"] = {
        "players": len(players),
        "track_frames": sum(
            len(player.get("frames", []))
            for player in players
            if isinstance(player, dict)
        ),
        "raw_pool_real_detections": len(real_pool),
        "association_frames_joined_to_raw": association_join_count,
    }

    pool_fragments = _pool_fragments(real_pool, config=cfg, reid_audit=reid_audit)
    enrollment_fragments = tuple(
        fragment
        for fragment in pool_fragments
        if fragment.source_track_id not in hard_excluded_source_ids
    )
    slots = enroll_slots(enrollment_fragments, fps=fps, config=cfg)
    stitch_decisions, stitch_vetoes = _fragment_stitch_audit(
        pool_fragments,
        real_by_frame=real_by_frame,
        config=cfg,
        reid_audit=reid_audit,
    )
    report["decisions"].extend(stitch_decisions)
    report["decisions"].extend(hard_decisions)
    report["enrollment"] = {
        "status": (
            "enrolled"
            if slots
            else f"no_valid_exact_{cfg.expected_players}_window"
            if auto_player_count
            else "no_valid_exact_four_window"
        ),
        "slot_count": len(slots),
        "slots": [asdict(slot) for slot in slots],
    }

    if slots:
        (
            output_players,
            unbound_observations,
            binding_decisions,
            track_rows,
            selection_counts,
        ) = _select_slot_players(
            slots,
            pool_fragments=pool_fragments,
            real_pool=real_pool,
            stitch_vetoes=stitch_vetoes,
            fps=fps,
            config=cfg,
            hard_excluded_source_ids=hard_excluded_source_ids,
            reid_audit=reid_audit,
        )
        report["decisions"].extend(binding_decisions)
        report["tracks"] = track_rows
        if selection_counts["unbound_real_detections"]:
            report["status"] = "preview_selection_partial_with_abstentions"
    else:
        measured_fallback = (
            _select_measured_association_fallback(
                players,
                real_by_frame=real_by_frame,
                pool_fragments=pool_fragments,
                fps=fps,
                config=cfg,
            )
            if auto_player_count and inferred_player_count in {2, 4}
            else None
        )
        if measured_fallback is not None:
            (
                output_players,
                unbound_observations,
                fallback_decisions,
                track_rows,
                selection_counts,
            ) = measured_fallback
            report["status"] = "preview_selection_measured_association_fallback"
            report["selection_mode"] = "auto_measured_association_fallback"
        else:
            (
                unbound_observations,
                fallback_decisions,
                track_rows,
                selection_counts,
            ) = _select_without_enrollment(
                pool_fragments,
                fps=fps,
                config=cfg,
            )
            output_players = []
            report["status"] = "preview_selection_partial_no_enrollment"
            report["selection_mode"] = "partial_unbound_real_only"
        report["decisions"].extend(fallback_decisions)
        report["tracks"] = track_rows

    output["players"] = output_players
    output["unbound_observations"] = unbound_observations
    _enforce_hard_export_invariants(output_players, hard_config=hard_cfg)
    report["decisions"].extend(reid_audit.decisions)
    report["decisions"].append(reid_audit.summary_decision())
    input_track_frames = report["input_counts"]["track_frames"]
    report["output_counts"] = {
        "players": len(output_players),
        "unbound_observations": len(unbound_observations),
        "track_frames": sum(len(player["frames"]) for player in output_players),
        "interpolated_frames": selection_counts["interpolated_frames"],
        "synthetic_frames_removed": max(0, input_track_frames - association_join_count),
        "unbound_real_detections": selection_counts["unbound_real_detections"],
        "dropped_real_detections": selection_counts["dropped_real_detections"],
        "recovered_real_detections": selection_counts["recovered_real_detections"],
    }
    return output, report


def _raw_pool_detections(
    raw_pool_payload: Mapping[str, Any],
    *,
    embedding_payload: Mapping[str, Any],
    calibration: CourtCalibration,
    config: PlayerSelectionConfig,
    embedding_bbox_scale: float = 1.0,
) -> tuple[SelectionDetection, ...]:
    embedding_index = _selection_embedding_index(embedding_payload)
    frames = raw_pool_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("raw pool payload must contain a frames list")
    seen_frame_indexes: set[int] = set()
    used_embedding_keys: set[tuple[int, int, int]] = set()
    detections: list[SelectionDetection] = []
    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, Mapping):
            raise ValueError("raw pool frame entries must be objects")
        frame_idx = int(
            frame_entry.get("frame", frame_entry.get("frame_index", default_frame_idx))
        )
        if frame_idx in seen_frame_indexes:
            raise ValueError(f"raw pool contains duplicate frame index {frame_idx}")
        seen_frame_indexes.add(frame_idx)
        frame_detections = frame_entry.get("detections", [])
        if not isinstance(frame_detections, list):
            raise ValueError("raw pool frame detections must be a list")
        for detection_index, raw_detection in enumerate(frame_detections):
            if not isinstance(raw_detection, Mapping):
                raise ValueError("raw pool detections must be objects")
            if not _is_person_detection(raw_detection):
                continue
            conf = float(
                raw_detection.get("conf", raw_detection.get("confidence", 1.0))
            )
            if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
                raise ValueError(
                    "raw detection confidence must be finite and in [0, 1]"
                )
            bbox = _raw_bbox(raw_detection)
            source_track_id = _raw_track_id(raw_detection, detection_index + 1)
            key = (frame_idx, source_track_id, detection_index)
            embedding: tuple[float, ...] | None = None
            record = embedding_index.get(key)
            if record is not None:
                embedding, embedding_bbox = record
                if embedding_bbox is not None:
                    bbox_in_embedding_space = tuple(
                        value * embedding_bbox_scale for value in bbox
                    )
                    delta = max(
                        abs(left - right)
                        for left, right in zip(
                            bbox_in_embedding_space,
                            embedding_bbox,
                            strict=True,
                        )
                    )
                    if delta > config.raw_match_bbox_delta_px:
                        raise ValueError(
                            "embedding bbox mismatch for "
                            f"frame={frame_idx} source_track_id={source_track_id} "
                            f"detection_index={detection_index}: {delta:.3f}px"
                        )
                used_embedding_keys.add(key)
            projected = person_detection_from_bbox(
                calibration,
                bbox_xyxy=bbox,
                confidence=conf,
            )
            detections.append(
                SelectionDetection(
                    frame_idx=frame_idx,
                    source_track_id=source_track_id,
                    bbox=tuple(float(value) for value in projected.bbox_xyxy),  # type: ignore[arg-type]
                    world_xy=tuple(float(value) for value in projected.foot_world_xy),  # type: ignore[arg-type]
                    conf=float(projected.confidence),
                    embedding=embedding,
                    interpolated=False,
                    raw_detection_uid=f"raw:{frame_idx}:{detection_index}",
                    payload=raw_detection,
                )
            )
    unused_embedding_keys = set(embedding_index) - used_embedding_keys
    if unused_embedding_keys:
        first = min(unused_embedding_keys)
        raise ValueError(
            "embedding artifact is not keyed to this raw pool; first unmatched key is "
            f"frame={first[0]} source_track_id={first[1]} detection_index={first[2]}"
        )
    return tuple(detections)


def _selection_embedding_index(
    embedding_payload: Mapping[str, Any],
) -> dict[
    tuple[int, int, int],
    tuple[tuple[float, ...], tuple[float, float, float, float] | None],
]:
    rows = embedding_payload.get("detections")
    if not isinstance(rows, list):
        raise ValueError("embedding payload must contain a detections list")
    expected_dim_raw = embedding_payload.get("feature_dim")
    expected_dim = int(expected_dim_raw) if expected_dim_raw is not None else None
    index: dict[
        tuple[int, int, int],
        tuple[tuple[float, ...], tuple[float, float, float, float] | None],
    ] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("embedding detection rows must be objects")
        frame_idx = _required_int(
            row,
            ("frame", "frame_idx", "frame_index"),
            f"embedding row {row_index} frame",
        )
        source_track_id = _required_int(
            row,
            ("source_track_id", "track_id"),
            f"embedding row {row_index} source_track_id",
        )
        detection_index = _required_int(
            row,
            ("detection_index", "det_idx"),
            f"embedding row {row_index} detection_index",
        )
        raw_vector = row.get("embedding")
        if not isinstance(raw_vector, (list, tuple)) or not raw_vector:
            raise ValueError(
                f"embedding row {row_index} must contain a non-empty embedding"
            )
        vector = tuple(float(value) for value in raw_vector)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError(f"embedding row {row_index} contains non-finite values")
        if expected_dim is not None and len(vector) != expected_dim:
            raise ValueError(
                f"embedding row {row_index} has dimension {len(vector)}, expected {expected_dim}"
            )
        raw_bbox = row.get("bbox", row.get("bbox_xyxy"))
        bbox: tuple[float, float, float, float] | None
        if raw_bbox is None:
            bbox = None
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox = tuple(float(value) for value in raw_bbox)  # type: ignore[assignment]
            if any(not math.isfinite(value) for value in bbox):
                raise ValueError(
                    f"embedding row {row_index} bbox contains non-finite values"
                )
        else:
            raise ValueError(f"embedding row {row_index} bbox must contain four values")
        key = (frame_idx, source_track_id, detection_index)
        if key in index:
            raise ValueError(
                "duplicate embedding row for "
                f"frame={frame_idx} source_track_id={source_track_id} "
                f"detection_index={detection_index}"
            )
        index[key] = (vector, bbox)
    return index


def _association_raw_join_count(
    players: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    real_by_frame: Mapping[int, Sequence[SelectionDetection]],
    config: PlayerSelectionConfig,
) -> int:
    return len(
        _association_raw_matches(
            players,
            fps=fps,
            real_by_frame=real_by_frame,
            config=config,
        )
    )


def _association_raw_matches(
    players: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    real_by_frame: Mapping[int, Sequence[SelectionDetection]],
    config: PlayerSelectionConfig,
) -> list[tuple[int, int, SelectionDetection]]:
    """One-to-one measured raw detections matching existing association rows."""

    requests_by_frame: dict[
        int, list[tuple[int, int, tuple[float, float, float, float]]]
    ] = defaultdict(list)
    for player_index, player in enumerate(players):
        frames = player.get("frames")
        if not isinstance(frames, list):
            raise ValueError("track player frames must be a list")
        for frame_position, payload in enumerate(frames):
            if not isinstance(payload, Mapping):
                raise ValueError("track frames must be objects")
            bbox = _track_frame_bbox(payload)
            frame_idx = _frame_index(payload, fps=fps)
            requests_by_frame[frame_idx].append((player_index, frame_position, bbox))

    joined: list[tuple[int, int, SelectionDetection]] = []
    for frame_idx, requests in requests_by_frame.items():
        candidates = list(real_by_frame.get(frame_idx, ()))
        preferences: dict[int, list[int]] = {}
        for request_index, (_player_index, _frame_position, bbox) in enumerate(
            requests
        ):
            eligible: list[tuple[float, int]] = []
            for candidate_index, candidate in enumerate(candidates):
                delta = max(
                    abs(left - right)
                    for left, right in zip(bbox, candidate.bbox, strict=True)
                )
                iou = _bbox_iou(bbox, candidate.bbox)
                if (
                    delta > config.raw_match_bbox_delta_px
                    and iou < config.raw_match_iou_min
                ):
                    continue
                quality = iou + (1.0 / (1.0 + delta)) * 1e-3 + candidate.conf * 1e-6
                eligible.append((quality, candidate_index))
            preferences[request_index] = [
                candidate_index
                for _quality, candidate_index in sorted(
                    eligible,
                    key=lambda row: (
                        -row[0],
                        candidates[row[1]].raw_detection_uid or "",
                    ),
                )
            ]

        candidate_owner: dict[int, int] = {}

        def assign(request_index: int, seen: set[int]) -> bool:
            for candidate_index in preferences[request_index]:
                if candidate_index in seen:
                    continue
                seen.add(candidate_index)
                previous = candidate_owner.get(candidate_index)
                if previous is None or assign(previous, seen):
                    candidate_owner[candidate_index] = request_index
                    return True
            return False

        for request_index in sorted(
            range(len(requests)),
            key=lambda index: (
                len(preferences[index]),
                requests[index][0],
                requests[index][1],
            ),
        ):
            assign(request_index, set())
        for candidate_index, request_index in sorted(candidate_owner.items()):
            player_index, frame_position, _bbox = requests[request_index]
            joined.append(
                (player_index, frame_position, candidates[candidate_index])
            )
    return joined


def _select_measured_association_fallback(
    players: Sequence[Mapping[str, Any]],
    *,
    real_by_frame: Mapping[int, Sequence[SelectionDetection]],
    pool_fragments: Sequence[TrackFragment],
    fps: float,
    config: PlayerSelectionConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
] | None:
    """Preserve established IDs using measured associations when enrollment fragments.

    This fallback is intentionally narrower than slot enrollment. It only retains
    existing association rows that one-to-one match immutable raw detections; it
    never emits an interpolated bridge or creates a new identity.
    """

    expected = config.expected_players
    if expected not in {2, 4}:
        return None
    matches = _association_raw_matches(
        players,
        fps=fps,
        real_by_frame=real_by_frame,
        config=config,
    )
    matched_by_player: dict[int, dict[int, SelectionDetection]] = defaultdict(dict)
    for player_index, _frame_position, detection in matches:
        matched_by_player[player_index][detection.frame_idx] = detection

    accepted_frame_count = len(
        {
            detection.frame_idx
            for per_player in matched_by_player.values()
            for detection in per_player.values()
        }
    )
    persistence_floor = min(
        max(1, int(math.ceil(fps))),
        max(1, int(math.ceil(0.2 * accepted_frame_count))),
    )
    eligible = [
        player_index
        for player_index, detections in matched_by_player.items()
        if len(detections) >= persistence_floor
        and any(
            frame_court_evidence(detection.world_xy, config)
            >= config.enrollment_court_presence_min
            for detection in detections.values()
        )
    ]
    if len(eligible) < expected:
        return None

    def rank(player_index: int) -> tuple[int, float, int]:
        detections = tuple(matched_by_player[player_index].values())
        return (
            len(detections),
            sum(detection.conf for detection in detections) / len(detections),
            -player_index,
        )

    ranked = sorted(eligible, key=rank, reverse=True)
    if expected == 2:
        by_side: dict[str, list[int]] = defaultdict(list)
        for player_index in ranked:
            side = str(players[player_index].get("side", ""))
            if side not in {"near", "far"}:
                _x, y = _median_world_xy(
                    tuple(matched_by_player[player_index].values())
                )
                side = "near" if y < 0.0 else "far"
            by_side[side].append(player_index)
        if by_side["near"] and by_side["far"]:
            selected_indexes = [by_side["near"][0], by_side["far"][0]]
        else:
            selected_indexes = ranked[:2]
    else:
        selected_indexes = ranked[:4]
    selected_indexes.sort(key=lambda index: int(players[index].get("id", index + 1)))

    output_players: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    for player_index in selected_indexes:
        source_player = players[player_index]
        detections = [
            matched_by_player[player_index][frame_idx]
            for frame_idx in sorted(matched_by_player[player_index])
        ]
        raw_uids = [_required_raw_uid(detection) for detection in detections]
        used_uids.update(raw_uids)
        player_id = int(source_player.get("id", player_index + 1))
        side = str(source_player.get("side", ""))
        role = str(source_player.get("role", ""))
        if side not in {"near", "far"} or role not in {"left", "right"}:
            x, y = _median_world_xy(tuple(detections))
            side = "near" if y < 0.0 else "far"
            role = "left" if x < 0.0 else "right"
        output_players.append(
            {
                "id": player_id,
                "side": side,
                "role": role,
                "frames": [
                    _detection_payload(detection, fps=fps)
                    for detection in detections
                ],
            }
        )
        track_rows.append(
            {
                "player_id": player_id,
                "selection_state": "bound_measured_association_fallback",
                "raw_detection_uids": raw_uids,
                "output_frame_count": len(detections),
            }
        )
        decisions.append(
            {
                "action": "retain_measured_association_identity",
                "player_id": player_id,
                "measured_frame_count": len(detections),
                "persistence_floor_frames": persistence_floor,
                "reasons": [
                    "automatic_two_or_four_cardinality",
                    "one_to_one_raw_detection_match",
                    "existing_identity_preserved",
                    "interpolated_rows_excluded",
                ],
            }
        )

    unbound_observations, unbound_track_rows = _unbound_observations(
        pool_fragments,
        excluded_uids=used_uids,
        abstention_reasons_by_fragment={
            fragment.fragment_id: [
                "not_selected_by_measured_association_fallback",
                "preserve_as_unbound_real_observation",
            ]
            for fragment in pool_fragments
        },
        fps=fps,
    )
    track_rows.extend(unbound_track_rows)
    all_uids = {
        _required_raw_uid(detection)
        for fragment in pool_fragments
        for detection in fragment.detections
    }
    unbound_uids = {
        uid
        for row in unbound_track_rows
        for uid in row["raw_detection_uids"]
    }
    if used_uids & unbound_uids or used_uids | unbound_uids != all_uids:
        raise ValueError(
            "measured association fallback must account for every raw detection once"
        )
    return (
        output_players,
        unbound_observations,
        decisions,
        track_rows,
        {
            "interpolated_frames": 0,
            "unbound_real_detections": len(unbound_uids),
            "dropped_real_detections": 0,
            "recovered_real_detections": 0,
        },
    )


def _pool_fragments(
    detections: Sequence[SelectionDetection],
    *,
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> list[TrackFragment]:
    by_source: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in detections:
        if not detection.interpolated:
            by_source[detection.source_track_id].append(detection)
    fragments: list[TrackFragment] = []
    for source_track_id in sorted(by_source):
        run: list[SelectionDetection] = []
        embedding_sum: list[float] | None = None
        embedding_count = 0
        source_fragment_ordinal = 0

        def flush() -> None:
            nonlocal embedding_count, embedding_sum, source_fragment_ordinal
            if not run:
                return
            source_fragment_ordinal += 1
            fragments.append(
                TrackFragment(
                    fragment_id=(
                        f"pool-{source_track_id}-{source_fragment_ordinal}-"
                        f"{run[0].frame_idx}-{run[-1].frame_idx}"
                    ),
                    source_track_id=source_track_id,
                    detections=tuple(run),
                )
            )
            run.clear()
            embedding_sum = None
            embedding_count = 0

        def current_centroid() -> tuple[float, ...] | None:
            if embedding_sum is None or embedding_count == 0:
                return None
            norm = math.sqrt(sum(value * value for value in embedding_sum))
            if norm <= 0.0:
                return None
            return tuple(value / norm for value in embedding_sum)

        def accumulate_embedding(detection: SelectionDetection) -> None:
            nonlocal embedding_count, embedding_sum
            embedding = detection.embedding
            if embedding is None:
                return
            if not embedding:
                raise ValueError("fragment embeddings must have one non-zero dimension")
            if embedding_sum is None:
                embedding_sum = [0.0] * len(embedding)
            if len(embedding) != len(embedding_sum):
                raise ValueError("fragment embeddings must have one non-zero dimension")
            for index, value in enumerate(embedding):
                embedding_sum[index] += float(value)
            embedding_count += 1

        for detection in sorted(
            by_source[source_track_id], key=lambda item: item.frame_idx
        ):
            if run and (
                detection.frame_idx != run[-1].frame_idx + 1
                or _appearance_discontinuity(
                    centroid=current_centroid(),
                    previous=run[-1],
                    detection=detection,
                    config=config,
                    reid_audit=reid_audit,
                )
            ):
                flush()
            run.append(detection)
            accumulate_embedding(detection)
        flush()
    return fragments


def _appearance_discontinuity(
    *,
    centroid: Sequence[float] | None,
    previous: SelectionDetection,
    detection: SelectionDetection,
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> bool:
    detection_embedding = _consultable_embedding(detection)
    if centroid is None or detection_embedding is None:
        return False
    distances = [
        _identity_distance(
            centroid,
            detection_embedding,
            reid_audit=reid_audit,
            trigger_frames=(detection.frame_idx,),
            comparison="fragment_centroid_discontinuity",
            context={"raw_detection_uid": detection.raw_detection_uid},
        )
    ]
    if previous.embedding is not None:
        distances.append(
            _identity_distance(
                previous.embedding,
                detection_embedding,
                reid_audit=reid_audit,
                trigger_frames=(detection.frame_idx,),
                comparison="fragment_previous_detection_discontinuity",
                context={"raw_detection_uid": detection.raw_detection_uid},
            )
        )
    return any(
        open_set_decision(distance, config) is not OpenSetDecision.ACCEPT
        for distance in distances
        if distance is not None
    )


def _fragment_stitch_audit(
    fragments: Sequence[TrackFragment],
    *,
    real_by_frame: Mapping[int, Sequence[SelectionDetection]],
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    by_source: dict[int, list[TrackFragment]] = defaultdict(list)
    for fragment in fragments:
        by_source[fragment.source_track_id].append(fragment)
    decisions: list[dict[str, Any]] = []
    vetoes: set[tuple[str, str]] = set()
    for source_track_id in sorted(by_source):
        ordered = sorted(
            by_source[source_track_id],
            key=lambda fragment: (
                fragment.start_frame,
                fragment.end_frame,
                fragment.fragment_id,
            ),
        )
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right.start_frame <= left.end_frame:
                continue
            support = [
                detection
                for frame_idx in range(left.end_frame + 1, right.start_frame)
                for detection in real_by_frame.get(frame_idx, ())
                if detection.source_track_id == source_track_id
            ]
            trigger_frames = (
                tuple(
                    frame_idx
                    for frame_idx in range(left.end_frame + 1, right.start_frame + 1)
                    if frame_idx in reid_audit.frames_ambiguous
                )
                if reid_audit is not None
                else tuple(range(left.end_frame + 1, right.start_frame + 1))
            )
            stitch = evaluate_stitch(
                left,
                right,
                real_support_in_gap=support,
                config=config,
                ambiguity_triggered=bool(trigger_frames),
                reid_audit=reid_audit,
                trigger_frames=trigger_frames,
            )
            decision = {
                "action": "stitch_refused" if stitch.refused else "stitch_retained",
                "source_track_id": source_track_id,
                "left_fragment_id": left.fragment_id,
                "right_fragment_id": right.fragment_id,
                "left_raw_detection_uid": left.detections[-1].raw_detection_uid,
                "right_raw_detection_uid": right.detections[0].raw_detection_uid,
                **asdict(stitch),
                "open_set": stitch.open_set.value,
                "reid_trigger_frames": list(trigger_frames),
                "reid_policy_scope": _REID_POLICY_SCOPE,
            }
            decisions.append(decision)
            if stitch.refused:
                vetoes.add((left.fragment_id, right.fragment_id))
    return decisions, vetoes


def _select_without_enrollment(
    pool_fragments: Sequence[TrackFragment],
    *,
    fps: float,
    config: PlayerSelectionConfig,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]
]:
    decisions: list[dict[str, Any]] = []
    for fragment in sorted(
        pool_fragments,
        key=lambda item: (item.start_frame, item.end_frame, item.fragment_id),
    ):
        evidence = presence_evidence(fragment.detections, fps=fps, config=config)
        decisions.append(
            {
                "action": "leave_unbound",
                "fragment_id": fragment.fragment_id,
                "slot_id": None,
                "open_set": OpenSetDecision.DEFER.value,
                "embedding_distance": None,
                "court_presence": evidence.court_presence,
                "persistence": evidence.persistence,
                "identity_match": 0.5,
                "fusion_score": fusion_score(
                    court_presence=evidence.court_presence,
                    identity_match=0.5,
                    persistence=evidence.persistence,
                    config=config,
                ),
                "role_consistent": None,
                "motion_envelope_consistent": None,
                "evidence_classes": [
                    "geometry"
                    for _ in [0]
                    if evidence.court_presence < config.selection_score_min
                    or evidence.persistence < config.selection_score_min
                ],
                "reasons": [
                    "no_usable_exact_four_enrollment",
                    "typed_partial_abstention",
                ],
                "raw_detection_uids": [
                    detection.raw_detection_uid for detection in fragment.detections
                ],
            }
        )
    unbound_observations, track_rows = _unbound_observations(
        pool_fragments,
        excluded_uids=set(),
        abstention_reasons_by_fragment={
            str(decision["fragment_id"]): list(decision["reasons"])
            for decision in decisions
            if decision["action"] == "leave_unbound"
        },
        fps=fps,
    )
    return (
        unbound_observations,
        decisions,
        track_rows,
        {
            "interpolated_frames": 0,
            "unbound_real_detections": sum(
                len(fragment.detections) for fragment in pool_fragments
            ),
            "dropped_real_detections": 0,
            "recovered_real_detections": 0,
        },
    )


def _select_slot_players(
    slots: Sequence[SelectionSlot],
    *,
    pool_fragments: Sequence[TrackFragment],
    real_pool: Sequence[SelectionDetection],
    stitch_vetoes: set[tuple[str, str]],
    fps: float,
    config: PlayerSelectionConfig,
    hard_excluded_source_ids: frozenset[int] = frozenset(),
    reid_audit: _SelectiveReIDAudit | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Bind/re-bind real pool fragments to slots, then micro-fill only accepted identity gaps."""

    real_by_frame: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in real_pool:
        real_by_frame[detection.frame_idx].append(detection)

    def real_detections_between(
        left_frame: int,
        right_frame: int,
    ) -> list[SelectionDetection]:
        return [
            detection
            for frame_idx in range(left_frame + 1, right_frame)
            for detection in real_by_frame.get(frame_idx, ())
        ]

    assignments: dict[int, list[TrackFragment]] = {slot.slot_id: [] for slot in slots}
    decisions: list[dict[str, Any]] = []
    fragment_by_id = {fragment.fragment_id: fragment for fragment in pool_fragments}
    owner_source_ids_by_slot = {
        slot.slot_id: frozenset(
            fragment_by_id[fragment_id].source_track_id
            for fragment_id in slot.source_fragment_ids
        )
        for slot in slots
    }
    owner_slots_by_source: dict[int, set[int]] = defaultdict(set)
    for slot_id, source_track_ids in owner_source_ids_by_slot.items():
        for source_track_id in source_track_ids:
            owner_slots_by_source[source_track_id].add(slot_id)
    owner_slot_by_source = {
        source_track_id: next(iter(slot_ids))
        for source_track_id, slot_ids in owner_slots_by_source.items()
        if len(slot_ids) == 1
    }
    assigned_fragment_ids: set[str] = set()
    enrollment_owned_uids: set[str] = set()
    for slot in sorted(slots, key=lambda item: item.slot_id):
        if slot.embedding_centroid is None or any(
            not math.isfinite(value) for value in slot.embedding_centroid
        ):
            raise ValueError(f"slot {slot.slot_id} has no usable appearance centroid")
        for fragment_id in slot.source_fragment_ids:
            fragment = fragment_by_id.get(fragment_id)
            if fragment is None:
                raise ValueError(
                    f"enrollment fragment {fragment_id} is absent from the raw pool"
                )
            if fragment_id in assigned_fragment_ids:
                raise ValueError(
                    f"enrollment fragment {fragment_id} has more than one owner"
                )
            enrollment_detections = tuple(
                detection
                for detection in fragment.detections
                if detection.frame_idx in slot.enrolled_frames
            )
            if (
                tuple(detection.frame_idx for detection in enrollment_detections)
                != slot.enrolled_frames
            ):
                raise ValueError(
                    f"enrollment fragment {fragment_id} does not own every registered frame"
                )
            registered_fragment = TrackFragment(
                fragment_id=fragment.fragment_id,
                source_track_id=fragment.source_track_id,
                detections=enrollment_detections,
                seed_player_id=fragment.seed_player_id,
            )
            assignments[slot.slot_id].append(registered_fragment)
            assigned_fragment_ids.add(fragment_id)
            enrollment_owned_uids.update(
                _required_raw_uid(detection)
                for detection in registered_fragment.detections
            )
            decisions.append(
                {
                    "action": "bind",
                    "fragment_id": fragment_id,
                    "slot_id": slot.slot_id,
                    "open_set": OpenSetDecision.ACCEPT.value,
                    "embedding_distance": None,
                    "court_presence": presence_evidence(
                        registered_fragment.detections,
                        fps=fps,
                        config=config,
                    ).court_presence,
                    "persistence": presence_evidence(
                        registered_fragment.detections,
                        fps=fps,
                        config=config,
                    ).persistence,
                    "identity_match": 1.0,
                    "fusion_score": 1.0,
                    "role_consistent": True,
                    "motion_envelope_consistent": True,
                    "evidence_classes": [],
                    "reasons": ["enrollment_registered_owner"],
                    "raw_detection_uids": [
                        detection.raw_detection_uid
                        for detection in registered_fragment.detections
                    ],
                }
            )

    unbound_fragments: list[TrackFragment] = []
    dropped_fragments: list[TrackFragment] = []
    remaining = [
        fragment
        for fragment in pool_fragments
        if fragment.fragment_id not in assigned_fragment_ids
    ]
    remaining.extend(
        _residual_fragments(
            [
                fragment_by_id[fragment_id]
                for fragment_id in sorted(assigned_fragment_ids)
            ],
            excluded_uids=enrollment_owned_uids,
        )
    )
    by_start: dict[int, list[TrackFragment]] = defaultdict(list)
    for fragment in remaining:
        by_start[fragment.start_frame].append(fragment)
    for start_frame in sorted(by_start):
        batch = sorted(
            by_start[start_frame],
            key=lambda item: (item.end_frame, item.fragment_id),
        )
        # Enrollment supplies a source-ID continuity hint, not identity authority.
        # Resolve appearance-accepted owner hints first in each chronological batch;
        # infeasible hints fall through to the ordinary one-to-one assignment.
        owner_chosen: dict[str, _SlotCandidate] = {}
        owner_slots_used: set[int] = set()
        for fragment in batch:
            owner_slot_id = owner_slot_by_source.get(fragment.source_track_id)
            if owner_slot_id is None or owner_slot_id in owner_slots_used:
                continue
            owner_candidates = _binding_candidates(
                fragment,
                slots=slots,
                assignments=assignments,
                stitch_vetoes=stitch_vetoes,
                fps=fps,
                config=config,
                owner_source_ids_by_slot=owner_source_ids_by_slot,
                reid_audit=reid_audit,
            )
            owner_candidate = next(
                candidate
                for candidate in owner_candidates
                if candidate.slot_id == owner_slot_id
            )
            if not owner_candidate.feasible:
                continue
            owner_chosen[fragment.fragment_id] = owner_candidate
            owner_slots_used.add(owner_slot_id)
            assignments[owner_slot_id].append(fragment)
            assigned_fragment_ids.add(fragment.fragment_id)

        generic_batch = [
            fragment
            for fragment in batch
            if fragment.fragment_id not in owner_chosen
        ]
        candidates_by_fragment = {
            fragment.fragment_id: _binding_candidates(
                fragment,
                slots=slots,
                assignments=assignments,
                stitch_vetoes=stitch_vetoes,
                fps=fps,
                config=config,
                owner_source_ids_by_slot=owner_source_ids_by_slot,
                reid_audit=reid_audit,
            )
            for fragment in generic_batch
        }
        chosen = {
            **owner_chosen,
            **_choose_one_to_one_bindings(generic_batch, candidates_by_fragment),
        }
        for fragment in batch:
            evidence = presence_evidence(fragment.detections, fps=fps, config=config)
            candidates = (
                [owner_chosen[fragment.fragment_id]]
                if fragment.fragment_id in owner_chosen
                else candidates_by_fragment[fragment.fragment_id]
            )
            selected_candidate = chosen.get(fragment.fragment_id)
            destructive_evidence: dict[str, bool] = {}
            drop_trigger_reasons: tuple[str, ...] = ()
            best = min(
                candidates,
                key=lambda candidate: (
                    -candidate.fusion_score,
                    float("inf")
                    if candidate.embedding_distance is None
                    else candidate.embedding_distance,
                    candidate.slot_id,
                ),
            )
            if selected_candidate is not None:
                if fragment.fragment_id not in owner_chosen:
                    assignments[selected_candidate.slot_id].append(fragment)
                    assigned_fragment_ids.add(fragment.fragment_id)
                action = "rebind"
                output_slot_id: int | None = selected_candidate.slot_id
                chosen_candidate = selected_candidate
            else:
                destructive_evidence, drop_trigger_reasons = (
                    _destructive_drop_evidence(
                        fragment,
                        candidates,
                        slots=slots,
                        presence=evidence,
                        config=config,
                        reid_audit=reid_audit,
                    )
                )
                if destructive_action_allowed(destructive_evidence):
                    action = "drop"
                    dropped_fragments.append(fragment)
                else:
                    action = "leave_unbound"
                    unbound_fragments.append(fragment)
                output_slot_id = None
                chosen_candidate = best
            decisions.append(
                {
                    "action": action,
                    "fragment_id": fragment.fragment_id,
                    "slot_id": output_slot_id,
                    "candidate_slot_id": chosen_candidate.slot_id,
                    "open_set": chosen_candidate.open_set.value,
                    "embedding_distance": chosen_candidate.embedding_distance,
                    "court_presence": evidence.court_presence,
                    "persistence": evidence.persistence,
                    "identity_match": identity_match_score(
                        chosen_candidate.embedding_distance,
                        config,
                    ),
                    "fusion_score": chosen_candidate.fusion_score,
                    "role_consistent": chosen_candidate.role_consistent,
                    "motion_envelope_consistent": chosen_candidate.temporal_motion_consistent,
                    "stitch_vetoed": chosen_candidate.stitch_vetoed,
                    "registered_owner_continuity": (
                        chosen_candidate.registered_owner_continuity
                    ),
                    "evidence_classes": (
                        []
                        if selected_candidate is not None
                        else [
                            name
                            for name, agrees in destructive_evidence.items()
                            if agrees
                        ]
                    ),
                    "reasons": (
                        list(drop_trigger_reasons)
                        if action == "drop"
                        else [
                            "registered_enrollment_source_hint",
                            "identity_accept",
                            "same_net_side",
                            "fusion_at_or_above_0_5",
                            "stitch_veto_absent",
                            "owner_priority_before_generic_binding",
                            (
                                "temporal_motion_advisory_pass"
                                if chosen_candidate.temporal_motion_consistent
                                else "temporal_motion_disagreement_preserved_without_fill"
                            ),
                        ]
                        if selected_candidate is not None
                        and chosen_candidate.registered_owner_continuity
                        else [
                            reason
                            for reason, applies in (
                                (
                                    "identity_accept",
                                    chosen_candidate.open_set
                                    is OpenSetDecision.ACCEPT,
                                ),
                                (
                                    "identity_defer",
                                    chosen_candidate.open_set
                                    is OpenSetDecision.DEFER,
                                ),
                                (
                                    "identity_reject",
                                    chosen_candidate.open_set
                                    is OpenSetDecision.REJECT,
                                ),
                                (
                                    "side_role_consistent",
                                    chosen_candidate.role_consistent,
                                ),
                                (
                                    "inside_7m_s_temporal_motion_envelope",
                                    chosen_candidate.temporal_motion_consistent,
                                ),
                                ("stitch_veto", chosen_candidate.stitch_vetoed),
                                (
                                    "fusion_at_or_above_0_5",
                                    chosen_candidate.fusion_score
                                    >= config.selection_score_min,
                                ),
                                (
                                    "one_to_one_feasible_slot_assignment",
                                    selected_candidate is not None,
                                ),
                            )
                            if applies
                        ]
                    ),
                    "raw_detection_uids": [
                        detection.raw_detection_uid for detection in fragment.detections
                    ],
                }
            )

    # Binding is chronological, so a veto counterpart may be assigned after an
    # earlier fragment was left unbound. Recompute against final assignments in
    # both edge directions. If both unbound endpoints appearance-accept the same
    # slot, abstain on both rather than let Layer C merge a vetoed pair.
    layer_c_accepted_fragment_ids_by_slot = {
        slot.slot_id: {
            fragment.fragment_id
            for fragment in unbound_fragments
            if any(
                _uniquely_accepts_slot(
                    detection,
                    slots=slots,
                    config=config,
                    reid_audit=reid_audit,
                )
                == slot.slot_id
                for detection in fragment.detections
            )
        }
        for slot in slots
    }
    stitch_blocked_fragment_ids_by_slot: dict[int, set[str]] = {
        slot.slot_id: set() for slot in slots
    }
    for slot in slots:
        blocked = stitch_blocked_fragment_ids_by_slot[slot.slot_id]
        for fragment in unbound_fragments:
            if _fragment_stitch_vetoed_for_assignments(
                fragment,
                assignments[slot.slot_id],
                stitch_vetoes,
            ) or (
                fragment.source_track_id in owner_source_ids_by_slot[slot.slot_id]
                and _fragment_is_incoming_stitch_veto(fragment, stitch_vetoes)
            ):
                blocked.add(fragment.fragment_id)
        accepted = layer_c_accepted_fragment_ids_by_slot[slot.slot_id]
        for left_fragment_id, right_fragment_id in stitch_vetoes:
            left_matches = {
                fragment_id
                for fragment_id in accepted
                if _fragment_id_is_or_residual(fragment_id, left_fragment_id)
            }
            right_matches = {
                fragment_id
                for fragment_id in accepted
                if _fragment_id_is_or_residual(fragment_id, right_fragment_id)
            }
            if left_matches and right_matches:
                blocked.update(left_matches)
                blocked.update(right_matches)
    stitch_vetoed_uids_by_slot: dict[int, set[str]] = {
        slot.slot_id: {
            _required_raw_uid(detection)
            for fragment in unbound_fragments
            if fragment.fragment_id
            in stitch_blocked_fragment_ids_by_slot[slot.slot_id]
            for detection in fragment.detections
        }
        for slot in slots
    }

    measured_by_slot: dict[int, dict[int, SelectionDetection]] = {
        slot.slot_id: {} for slot in slots
    }
    used_uids: set[str] = set()
    for slot in slots:
        for fragment in assignments[slot.slot_id]:
            for detection in fragment.detections:
                uid = _required_raw_uid(detection)
                if uid in used_uids:
                    raise ValueError(
                        f"raw detection UID {uid} was assigned more than once"
                    )
                if detection.frame_idx in measured_by_slot[slot.slot_id]:
                    raise ValueError(
                        f"slot {slot.slot_id} has multiple real detections at frame "
                        f"{detection.frame_idx}"
                    )
                used_uids.add(uid)
                measured_by_slot[slot.slot_id][detection.frame_idx] = detection

    unbound_uids = {
        _required_raw_uid(detection)
        for fragment in unbound_fragments
        for detection in fragment.detections
    }
    recovered_count = 0
    slot_by_id = {slot.slot_id: slot for slot in slots}
    for slot in sorted(slots, key=lambda item: item.slot_id):
        initial_indexes = sorted(measured_by_slot[slot.slot_id])
        for left_index, right_index in zip(
            initial_indexes, initial_indexes[1:], strict=False
        ):
            if right_index <= left_index + 1:
                continue
            left = measured_by_slot[slot.slot_id][left_index]
            right = measured_by_slot[slot.slot_id][right_index]
            candidate_pool: list[SelectionDetection] = []
            for detection in real_detections_between(left_index, right_index):
                uid = _required_raw_uid(detection)
                if uid not in unbound_uids or uid in used_uids:
                    continue
                if uid in stitch_vetoed_uids_by_slot[slot.slot_id]:
                    continue
                accepted_slot_id = _uniquely_accepts_slot(
                    detection,
                    slots=slots,
                    config=config,
                    reid_audit=reid_audit,
                )
                if accepted_slot_id == slot.slot_id:
                    candidate_pool.append(detection)
            recovered = recover_identity_conditioned_pool(
                slot,
                last_detection=left,
                next_detection=right,
                pool=candidate_pool,
                fps=fps,
                occupied_frames=measured_by_slot[slot.slot_id],
                preferred_source_track_ids=owner_source_ids_by_slot[slot.slot_id],
                config=config,
                reid_audit=reid_audit,
            )
            for detection in recovered:
                uid = _required_raw_uid(detection)
                if uid in used_uids:
                    raise ValueError(
                        f"Layer C attempted to reuse raw detection UID {uid}"
                    )
                used_uids.add(uid)
                unbound_uids.remove(uid)
                measured_by_slot[slot.slot_id][detection.frame_idx] = detection
                recovered_count += 1
                decisions.append(
                    {
                        "action": "recover_real",
                        "slot_id": slot.slot_id,
                        "frame_idx": detection.frame_idx,
                        "raw_detection_uid": uid,
                        "interpolated": False,
                        "reasons": [
                            "identity_accept",
                            "inside_forward_and_reverse_7m_s_motion_envelope",
                            "one_to_one_raw_uid_consumption",
                        ],
                    }
                )

    hard_dropped_uids: set[str] = set()
    for slot_id in sorted(measured_by_slot):
        for frame_idx, detection in list(measured_by_slot[slot_id].items()):
            if detection.source_track_id not in hard_excluded_source_ids:
                continue
            uid = _required_raw_uid(detection)
            del measured_by_slot[slot_id][frame_idx]
            used_uids.remove(uid)
            hard_dropped_uids.add(uid)
            decisions.append(
                {
                    "action": "hard_drop_bound_detection",
                    "slot_id": slot_id,
                    "source_track_id": detection.source_track_id,
                    "frame_idx": frame_idx,
                    "raw_detection_uid": uid,
                    "reasons": ["source_track_hard_excluded_after_soft_binding"],
                }
            )

    slot_players: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    interpolated_total = 0
    for slot in sorted(slots, key=lambda item: item.slot_id):
        measured = measured_by_slot[slot.slot_id]
        frames: dict[int, dict[str, Any]] = {
            frame_idx: _detection_payload(detection, fps=fps)
            for frame_idx, detection in measured.items()
        }
        current_slot_uids = {
            _required_raw_uid(detection) for detection in measured.values()
        }
        measured_indexes = sorted(measured)
        for left_index, right_index in zip(
            measured_indexes, measured_indexes[1:], strict=False
        ):
            if right_index <= left_index + 1:
                continue
            left = measured[left_index]
            right = measured[right_index]
            trigger_frames = (
                tuple(
                    frame_idx
                    for frame_idx in range(left_index + 1, right_index)
                    if frame_idx in reid_audit.frames_ambiguous
                )
                if reid_audit is not None
                else tuple(range(left_index + 1, right_index))
            )
            distance = _identity_distance(
                left.embedding,
                right.embedding,
                reid_audit=reid_audit,
                trigger_frames=trigger_frames,
                comparison="micro_fill_endpoint",
                context={
                    "slot_id": slot.slot_id,
                    "left_frame": left_index,
                    "right_frame": right_index,
                },
            )
            allowed, reasons = micro_fill_allowed(
                left,
                right,
                identity_distance=distance,
                config=config,
            )
            if not allowed:
                decisions.append(
                    {
                        "action": "micro_fill_refused",
                        "slot_id": slot.slot_id,
                        "left_frame": left_index,
                        "right_frame": right_index,
                        "reasons": list(reasons),
                        "reid_trigger_frames": list(trigger_frames),
                    }
                )
                continue
            ambiguous = _ambiguous_real_observations(
                slot_by_id[slot.slot_id],
                left=left,
                right=right,
                pool=real_detections_between(left_index, right_index),
                current_slot_uids=current_slot_uids,
                fps=fps,
                config=config,
                reid_audit=reid_audit,
            )
            if ambiguous:
                decisions.append(
                    {
                        "action": "micro_fill_refused",
                        "slot_id": slot.slot_id,
                        "left_frame": left_index,
                        "right_frame": right_index,
                        "reasons": ["ambiguous_real_observation_in_gap"],
                        "ambiguous_raw_detection_uids": [
                            _required_raw_uid(detection) for detection in ambiguous
                        ],
                    }
                )
                continue
            for frame_idx in range(left_index + 1, right_index):
                alpha = (frame_idx - left_index) / (right_index - left_index)
                interpolated = SelectionDetection(
                    frame_idx=frame_idx,
                    source_track_id=left.source_track_id,
                    bbox=tuple(
                        (1.0 - alpha) * left.bbox[index] + alpha * right.bbox[index]
                        for index in range(4)
                    ),  # type: ignore[arg-type]
                    world_xy=(
                        (1.0 - alpha) * left.world_xy[0] + alpha * right.world_xy[0],
                        (1.0 - alpha) * left.world_xy[1] + alpha * right.world_xy[1],
                    ),
                    conf=min(left.conf, right.conf, 0.35),
                    embedding=None,
                    interpolated=True,
                    raw_detection_uid=None,
                )
                frames[frame_idx] = _detection_payload(interpolated, fps=fps)
                interpolated_total += 1
            decisions.append(
                {
                    "action": "micro_fill",
                    "slot_id": slot.slot_id,
                    "left_frame": left_index,
                    "right_frame": right_index,
                    "interpolated_frames": right_index - left_index - 1,
                    "reid_trigger_frames": list(trigger_frames),
                    "reasons": [
                        "identity_accept",
                        "within_12_frame_2_5m_no_net_caps",
                        "no_ambiguous_real_observation_in_gap",
                    ],
                }
            )
        player = {
            "id": slot.slot_id,
            "side": slot.side,
            "role": slot.role,
            "frames": [frames[frame_idx] for frame_idx in sorted(frames)],
        }
        slot_players.append(player)
        slot_raw_uids = [
            _required_raw_uid(detection)
            for detection in sorted(measured.values(), key=lambda item: item.frame_idx)
        ]
        track_rows.append(
            {
                "player_id": slot.slot_id,
                "selection_state": "bound_slot",
                "slot_id": slot.slot_id,
                "raw_detection_uids": slot_raw_uids,
                "output_frame_count": len(player["frames"]),
            }
        )

    residual_unbound = _residual_fragments(unbound_fragments, excluded_uids=used_uids)
    unbound_observations, unbound_track_rows = _unbound_observations(
        residual_unbound,
        excluded_uids=set(),
        abstention_reasons_by_fragment={
            str(decision["fragment_id"]): list(decision["reasons"])
            for decision in decisions
            if decision["action"] == "leave_unbound"
        },
        fps=fps,
    )
    unbound_output_id = {
        row["source_fragment_id"]: row["unbound_observation_id"]
        for row in unbound_track_rows
    }
    for decision in decisions:
        if decision.get("action") == "leave_unbound":
            output_ids = [
                observation_id
                for fragment_id, observation_id in unbound_output_id.items()
                if fragment_id == decision.get("fragment_id")
                or str(fragment_id).startswith(f"{decision.get('fragment_id')}:")
            ]
            decision["output_unbound_observation_ids"] = output_ids

    dropped_uids = {
        _required_raw_uid(detection)
        for fragment in dropped_fragments
        for detection in fragment.detections
    }
    dropped_uids.update(hard_dropped_uids)
    residual_uids = {
        uid for row in unbound_track_rows for uid in row["raw_detection_uids"]
    }
    all_uids = {_required_raw_uid(detection) for detection in real_pool}
    if (
        used_uids & residual_uids
        or used_uids & dropped_uids
        or residual_uids & dropped_uids
    ):
        raise ValueError("bound, unbound, and dropped raw UID sets must be disjoint")
    if used_uids | residual_uids | dropped_uids != all_uids:
        raise ValueError("every raw UID must be bound, unbound, or explicitly dropped")

    track_rows.extend(unbound_track_rows)
    return (
        slot_players,
        unbound_observations,
        decisions,
        track_rows,
        {
            "interpolated_frames": interpolated_total,
            "unbound_real_detections": len(residual_uids),
            "dropped_real_detections": len(dropped_uids),
            "recovered_real_detections": recovered_count,
        },
    )


def _binding_candidates(
    fragment: TrackFragment,
    *,
    slots: Sequence[SelectionSlot],
    assignments: Mapping[int, Sequence[TrackFragment]],
    stitch_vetoes: set[tuple[str, str]],
    fps: float,
    config: PlayerSelectionConfig,
    owner_source_ids_by_slot: Mapping[int, frozenset[int]] | None = None,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> list[_SlotCandidate]:
    evidence = presence_evidence(fragment.detections, fps=fps, config=config)
    role = _fragment_role(fragment)
    trigger_frames = tuple(
        detection.frame_idx
        for detection in fragment.detections
        if not detection.interpolated
        and _consultable_embedding(detection) is not None
    )
    centroid = (
        _gallery_embedding_centroid(fragment.detections)
        if trigger_frames
        else None
    )
    appearance_skipped_unambiguous = bool(fragment.detections) and all(
        detection.reid_policy == _REID_SKIPPED
        for detection in fragment.detections
    )
    candidates: list[_SlotCandidate] = []
    for slot in slots:
        distance = (
            _identity_distance(
                slot.embedding_centroid,
                centroid,
                reid_audit=reid_audit,
                trigger_frames=trigger_frames,
                comparison="fragment_slot_binding",
                context={
                    "fragment_id": fragment.fragment_id,
                    "slot_id": slot.slot_id,
                },
            )
            if trigger_frames
            else None
        )
        open_set = open_set_decision(distance, config)
        score = fusion_score(
            court_presence=evidence.court_presence,
            identity_match=identity_match_score(distance, config),
            persistence=evidence.persistence,
            config=config,
        )
        temporal_motion_consistent = _fragment_inside_slot_motion_envelope(
            fragment,
            assignments[slot.slot_id],
            fps=fps,
            max_speed_m_s=config.recovery_max_speed_m_s,
        )
        role_consistent = role == (slot.side, slot.role)
        registered_owner_continuity = (
            fragment.source_track_id
            in (owner_source_ids_by_slot or {}).get(slot.slot_id, frozenset())
            or any(
                fragment.fragment_id.startswith(f"{source_fragment_id}:residual-")
                for source_fragment_id in slot.source_fragment_ids
            )
        )
        stitch_vetoed = (
            _fragment_stitch_vetoed_for_assignments(
                fragment,
                assignments[slot.slot_id],
                stitch_vetoes,
            )
            or (
                registered_owner_continuity
                and _fragment_is_incoming_stitch_veto(fragment, stitch_vetoes)
            )
        )
        unambiguous_geometry_for_slot = (
            appearance_skipped_unambiguous and role_consistent
        )
        common_feasible = (
            (
                (centroid is not None and open_set is OpenSetDecision.ACCEPT)
                or unambiguous_geometry_for_slot
            )
            and not stitch_vetoed
            and score >= config.selection_score_min
        )
        frame_overlap = _fragment_overlaps_assignments(
            fragment,
            assignments[slot.slot_id],
        )
        feasible = common_feasible and (
            (
                registered_owner_continuity
                and role[0] == slot.side
                and not frame_overlap
            )
            or (
                not registered_owner_continuity
                and role_consistent
                and temporal_motion_consistent
            )
        )
        candidates.append(
            _SlotCandidate(
                fragment_id=fragment.fragment_id,
                slot_id=slot.slot_id,
                fusion_score=score,
                embedding_distance=distance,
                open_set=open_set,
                role_consistent=role_consistent,
                temporal_motion_consistent=temporal_motion_consistent,
                stitch_vetoed=stitch_vetoed,
                registered_owner_continuity=registered_owner_continuity,
                feasible=feasible,
            )
        )
    return candidates


def _choose_one_to_one_bindings(
    fragments: Sequence[TrackFragment],
    candidates_by_fragment: Mapping[str, Sequence[_SlotCandidate]],
) -> dict[str, _SlotCandidate]:
    ordered = sorted(fragments, key=lambda item: (item.end_frame, item.fragment_id))
    ordered_index = {
        fragment.fragment_id: index for index, fragment in enumerate(ordered)
    }

    def mapping_key(
        mapping: Mapping[str, _SlotCandidate],
    ) -> tuple[int, int, float, tuple[tuple[int, int], ...]]:
        return (
            len(mapping),
            sum(
                candidate.registered_owner_continuity for candidate in mapping.values()
            ),
            sum(candidate.fusion_score for candidate in mapping.values()),
            tuple(
                (-ordered_index[fragment_id], -candidate.slot_id)
                for fragment_id, candidate in sorted(
                    mapping.items(),
                    key=lambda item: ordered_index[item[0]],
                )
            ),
        )

    # Only four slots exist, so retain the best partial mapping for each slot
    # bitmask instead of enumerating every fragment/slot combination.
    states: dict[int, dict[str, _SlotCandidate]] = {0: {}}
    for fragment in ordered:
        next_states = dict(states)
        for used_slot_mask, mapping in states.items():
            for candidate in sorted(
                candidates_by_fragment[fragment.fragment_id],
                key=lambda item: (
                    -item.registered_owner_continuity,
                    -item.fusion_score,
                    float("inf")
                    if item.embedding_distance is None
                    else item.embedding_distance,
                    item.slot_id,
                ),
            ):
                if not candidate.feasible:
                    continue
                slot_bit = 1 << candidate.slot_id
                if used_slot_mask & slot_bit:
                    continue
                candidate_mapping = dict(mapping)
                candidate_mapping[fragment.fragment_id] = candidate
                candidate_mask = used_slot_mask | slot_bit
                previous = next_states.get(candidate_mask)
                if previous is None or mapping_key(candidate_mapping) > mapping_key(
                    previous
                ):
                    next_states[candidate_mask] = candidate_mapping
        states = next_states
    return max(states.values(), key=mapping_key)


def _uniquely_accepts_slot(
    detection: SelectionDetection,
    *,
    slots: Sequence[SelectionSlot],
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> int | None:
    if detection.reid_policy == _REID_SKIPPED:
        side, role = _provisional_slot_key(detection)
        matches = [
            slot.slot_id
            for slot in slots
            if (slot.side, slot.role) == (side, role)
        ]
        return matches[0] if len(matches) == 1 else None
    consultable = _consultable_embedding(detection)
    if consultable is None:
        return None
    accepted = [
        slot.slot_id
        for slot in slots
        if open_set_decision(
            _identity_distance(
                slot.embedding_centroid,
                consultable,
                reid_audit=reid_audit,
                trigger_frames=(detection.frame_idx,),
                comparison="single_detection_slot_acceptance",
                context={
                    "slot_id": slot.slot_id,
                    "raw_detection_uid": detection.raw_detection_uid,
                },
            ),
            config,
        )
        is OpenSetDecision.ACCEPT
    ]
    return accepted[0] if len(accepted) == 1 else None


def _ambiguous_real_observations(
    slot: SelectionSlot,
    *,
    left: SelectionDetection,
    right: SelectionDetection,
    pool: Sequence[SelectionDetection],
    current_slot_uids: set[str],
    fps: float,
    config: PlayerSelectionConfig,
    reid_audit: _SelectiveReIDAudit | None = None,
) -> tuple[SelectionDetection, ...]:
    ambiguous: list[SelectionDetection] = []
    for detection in pool:
        uid = _required_raw_uid(detection)
        if uid in current_slot_uids:
            continue
        if not left.frame_idx < detection.frame_idx < right.frame_idx:
            continue
        forward_s = (detection.frame_idx - left.frame_idx) / fps
        reverse_s = (right.frame_idx - detection.frame_idx) / fps
        if (
            _point_distance(left.world_xy, detection.world_xy)
            > config.recovery_max_speed_m_s * forward_s
        ):
            continue
        if (
            _point_distance(detection.world_xy, right.world_xy)
            > config.recovery_max_speed_m_s * reverse_s
        ):
            continue
        if detection.reid_policy != _REID_SKIPPED:
            consultable = _consultable_embedding(detection)
            if consultable is None:
                ambiguous.append(detection)
                continue
            identity = open_set_decision(
                _identity_distance(
                    slot.embedding_centroid,
                    consultable,
                    reid_audit=reid_audit,
                    trigger_frames=(detection.frame_idx,),
                    comparison="micro_fill_ambiguous_observation",
                    context={
                        "slot_id": slot.slot_id,
                        "raw_detection_uid": detection.raw_detection_uid,
                    },
                ),
                config,
            )
            if identity is OpenSetDecision.REJECT:
                continue
        ambiguous.append(detection)
    return tuple(
        sorted(
            ambiguous,
            key=lambda item: (item.frame_idx, item.raw_detection_uid or ""),
        )
    )


def _residual_fragments(
    fragments: Sequence[TrackFragment],
    *,
    excluded_uids: set[str],
) -> list[TrackFragment]:
    residual: list[TrackFragment] = []
    for fragment in fragments:
        run: list[SelectionDetection] = []
        ordinal = 0

        def flush() -> None:
            nonlocal ordinal
            if not run:
                return
            ordinal += 1
            residual.append(
                TrackFragment(
                    fragment_id=f"{fragment.fragment_id}:residual-{ordinal}",
                    source_track_id=fragment.source_track_id,
                    detections=tuple(run),
                    seed_player_id=fragment.seed_player_id,
                )
            )
            run.clear()

        for detection in fragment.detections:
            if _required_raw_uid(detection) in excluded_uids:
                flush()
                continue
            if run and detection.frame_idx != run[-1].frame_idx + 1:
                flush()
            run.append(detection)
        flush()
    return residual


def _unbound_observations(
    fragments: Sequence[TrackFragment],
    *,
    excluded_uids: set[str],
    abstention_reasons_by_fragment: Mapping[str, Sequence[str]],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    observation_id = 1
    for fragment in sorted(
        fragments,
        key=lambda item: (item.start_frame, item.end_frame, item.fragment_id),
    ):
        detections = [
            detection
            for detection in fragment.detections
            if _required_raw_uid(detection) not in excluded_uids
        ]
        if not detections:
            continue
        raw_detection_uids = [_required_raw_uid(detection) for detection in detections]
        parent_fragment_id, decision_reasons = max(
            (
                (source_fragment_id, reasons)
                for source_fragment_id, reasons in abstention_reasons_by_fragment.items()
                if fragment.fragment_id == source_fragment_id
                or fragment.fragment_id.startswith(f"{source_fragment_id}:")
            ),
            key=lambda item: len(item[0]),
            default=("", ()),
        )
        if not parent_fragment_id:
            raise ValueError(
                f"unbound fragment {fragment.fragment_id} has no abstention decision"
            )
        abstention_reasons = ["not_bound_to_claimed_slot", *decision_reasons]
        observation = {
            "observation_id": observation_id,
            "selection_state": "unbound_abstention",
            "source_fragment_id": fragment.fragment_id,
            "abstention_reasons": abstention_reasons,
            "raw_detection_uids": raw_detection_uids,
            "frames": [
                _detection_payload(detection, fps=fps) for detection in detections
            ],
        }
        observations.append(observation)
        track_rows.append(
            {
                "unbound_observation_id": observation_id,
                "selection_state": "unbound_abstention",
                "source_fragment_id": fragment.fragment_id,
                "abstention_reasons": abstention_reasons,
                "raw_detection_uids": raw_detection_uids,
                "output_frame_count": len(detections),
            }
        )
        observation_id += 1
    return observations, track_rows


def _fragment_role(fragment: TrackFragment) -> tuple[str, str]:
    x, y = _median_world_xy(fragment.detections)
    return ("near" if y < 0.0 else "far", "left" if x < 0.0 else "right")


def _fragment_inside_slot_motion_envelope(
    fragment: TrackFragment,
    assigned: Sequence[TrackFragment],
    *,
    fps: float,
    max_speed_m_s: float,
) -> bool:
    if not assigned:
        return True
    if _fragment_overlaps_assignments(fragment, assigned):
        return False
    predecessors = [
        other for other in assigned if other.end_frame < fragment.start_frame
    ]
    successors = [
        other for other in assigned if fragment.end_frame < other.start_frame
    ]
    comparisons: list[tuple[SelectionDetection, SelectionDetection]] = []
    if predecessors:
        predecessor = max(
            predecessors,
            key=lambda item: (item.end_frame, item.fragment_id),
        )
        comparisons.append((predecessor.detections[-1], fragment.detections[0]))
    if successors:
        successor = min(
            successors,
            key=lambda item: (item.start_frame, item.fragment_id),
        )
        comparisons.append((fragment.detections[-1], successor.detections[0]))
    return all(
        (right.frame_idx - left.frame_idx) / fps > 0.0
        and _point_distance(left.world_xy, right.world_xy)
        <= max_speed_m_s * ((right.frame_idx - left.frame_idx) / fps)
        for left, right in comparisons
    )


def _fragment_stitch_vetoed_for_assignments(
    fragment: TrackFragment,
    assigned: Sequence[TrackFragment],
    stitch_vetoes: set[tuple[str, str]],
) -> bool:
    return any(
        _fragment_pair_has_stitch_veto(
            fragment.fragment_id,
            other.fragment_id,
            stitch_vetoes,
        )
        for other in assigned
    )


def _fragment_is_incoming_stitch_veto(
    fragment: TrackFragment,
    stitch_vetoes: set[tuple[str, str]],
) -> bool:
    return any(
        _fragment_id_is_or_residual(fragment.fragment_id, right_fragment_id)
        for _left_fragment_id, right_fragment_id in stitch_vetoes
    )


def _fragment_pair_has_stitch_veto(
    left_fragment_id: str,
    right_fragment_id: str,
    stitch_vetoes: set[tuple[str, str]],
) -> bool:
    return any(
        (
            _fragment_id_is_or_residual(left_fragment_id, registered_left)
            and _fragment_id_is_or_residual(right_fragment_id, registered_right)
        )
        or (
            _fragment_id_is_or_residual(left_fragment_id, registered_right)
            and _fragment_id_is_or_residual(right_fragment_id, registered_left)
        )
        for registered_left, registered_right in stitch_vetoes
    )


def _fragment_id_is_or_residual(fragment_id: str, registered_id: str) -> bool:
    return fragment_id == registered_id or fragment_id.startswith(
        f"{registered_id}:residual-"
    )


def _fragment_overlaps_assignments(
    fragment: TrackFragment,
    assigned: Sequence[TrackFragment],
) -> bool:
    return any(
        not (
            other.end_frame < fragment.start_frame
            or fragment.end_frame < other.start_frame
        )
        for other in assigned
    )


def _detection_payload(detection: SelectionDetection, *, fps: float) -> dict[str, Any]:
    if detection.interpolated and detection.raw_detection_uid is not None:
        raise ValueError("interpolated geometry cannot claim a raw detection UID")
    if not detection.interpolated and detection.raw_detection_uid is None:
        raise ValueError("measured geometry must carry an immutable raw detection UID")
    return {
        "frame_idx": detection.frame_idx,
        "t": detection.frame_idx / fps,
        "bbox": list(detection.bbox),
        "world_xy": list(detection.world_xy),
        "conf": detection.conf,
        "interpolated": detection.interpolated,
    }


def _enforce_hard_export_invariants(
    players: Sequence[Mapping[str, Any]],
    *,
    hard_config: HardConstraintConfig,
) -> None:
    if len(players) > hard_config.exactly_four_hard_cap:
        raise ValueError(
            "EXACTLY_FOUR_HARD_CAP invariant violated: "
            f"exported {len(players)} bound players"
        )
    player_ids: list[int] = []
    per_frame: Counter[int] = Counter()
    for player in players:
        if not isinstance(player.get("id"), int):
            raise ValueError("player export contains a non-bound-slot player")
        player_id = int(player["id"])
        if not 1 <= player_id <= hard_config.exactly_four_hard_cap:
            raise ValueError(f"player {player_id} is not a registered bound slot")
        player_ids.append(player_id)
        frames = player.get("frames")
        if not isinstance(frames, list):
            raise ValueError("bound-slot player export requires a frames list")
        for payload in frames:
            if not isinstance(payload, Mapping):
                raise ValueError("bound-slot player frames must be objects")
            per_frame[_frame_index(payload, fps=1.0)] += 1
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("player export contains duplicate bound slot IDs")
    violating = [
        (frame_idx, count)
        for frame_idx, count in sorted(per_frame.items())
        if count > hard_config.exactly_four_hard_cap
    ]
    if violating:
        frame_idx, count = violating[0]
        raise ValueError(
            "EXACTLY_FOUR_HARD_CAP frame invariant violated: "
            f"frame {frame_idx} carries {count} player detections"
        )


def _base_report(config: PlayerSelectionConfig, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_player_selection_report",
        "status": status,
        "VERIFIED": 0,
        "preview_only": True,
        "source_only": None,
        "source_attestations": {
            "status": "not_applicable_disabled",
            "source_only": None,
            "uses_cvat_labels": None,
            "promote_trk": None,
        },
        "selection_mode": "disabled",
        "not_for_training": True,
        "not_a_promotion": True,
        "config": asdict(config),
        "input_counts": {
            "players": 0,
            "track_frames": 0,
            "raw_pool_real_detections": 0,
            "association_frames_joined_to_raw": 0,
        },
        "output_counts": {
            "players": 0,
            "unbound_observations": 0,
            "track_frames": 0,
            "interpolated_frames": 0,
            "synthetic_frames_removed": 0,
            "unbound_real_detections": 0,
            "dropped_real_detections": 0,
            "recovered_real_detections": 0,
        },
        "enrollment": {"status": "not_run", "slot_count": 0, "slots": []},
        "tracks": [],
        "decisions": [],
    }


def _validate_source_only_embeddings(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
    }
    for field_name, expected_value in expected.items():
        if field_name not in payload:
            raise ValueError(
                f"embedding artifact requires explicit {field_name}={expected_value!r} attestation"
            )
        if payload[field_name] is not expected_value:
            raise ValueError(
                f"embedding artifact attestation {field_name} must be literal "
                f"{expected_value!r}"
            )
    return {
        "status": "explicitly_attested",
        **expected,
    }


def _required_int(
    payload: Mapping[str, Any],
    names: Sequence[str],
    description: str,
) -> int:
    for name in names:
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(value, bool):
            break
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        break
    raise ValueError(f"{description} must be an integer")


def _is_person_detection(payload: Mapping[str, Any]) -> bool:
    value = payload.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


def _raw_track_id(payload: Mapping[str, Any], fallback: int) -> int:
    value = payload.get(
        "track_id", payload.get("player_id", payload.get("id", fallback))
    )
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return fallback


def _raw_bbox(payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = payload.get("bbox", payload.get("bbox_xyxy"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("raw detection bbox must contain four xyxy values")
    bbox = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in bbox):
        raise ValueError("raw detection bbox values must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("raw detection bbox must be ordered as x1, y1, x2, y2")
    return bbox  # type: ignore[return-value]


def _track_frame_bbox(payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = payload.get("bbox", payload.get("bbox_xyxy"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("track frame bbox must contain four xyxy values")
    bbox = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in bbox):
        raise ValueError("track frame bbox values must be finite")
    return bbox  # type: ignore[return-value]


def _required_raw_uid(detection: SelectionDetection) -> str:
    uid = detection.raw_detection_uid
    if detection.interpolated or uid is None:
        raise ValueError("expected a measured detection with an immutable raw UID")
    return uid


def _frame_index(payload: Mapping[str, Any], *, fps: float) -> int:
    explicit = payload.get(
        "frame_idx", payload.get("frame", payload.get("frame_index"))
    )
    if explicit is not None:
        return int(explicit)
    if "t" not in payload:
        raise ValueError("frame payload requires frame_idx/frame/frame_index or t")
    return int(round(float(payload["t"]) * fps))


def _median_world_xy(detections: Sequence[SelectionDetection]) -> tuple[float, float]:
    xs = sorted(detection.world_xy[0] for detection in detections)
    ys = sorted(detection.world_xy[1] for detection in detections)
    return (_median(xs), _median(ys))


def _median(values: Sequence[float]) -> float:
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return 0.5 * (float(values[middle - 1]) + float(values[middle]))


def _point_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(left[0]) - float(right[0]), float(left[1]) - float(right[1])
    )


def _net_crossing(left: Sequence[float], right: Sequence[float]) -> bool:
    left_y = float(left[1])
    right_y = float(right[1])
    return (left_y < 0.0 < right_y) or (right_y < 0.0 < left_y)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


__all__ = [
    "COURT_REGION_HARD_BOUND_M",
    "DROP_TRIGGER_REASON_VOCABULARY",
    "EXACTLY_FOUR_HARD_CAP",
    "HardConstraintConfig",
    "OpenSetDecision",
    "PlayerSelectionConfig",
    "PresenceEvidence",
    "SelectionDetection",
    "SelectionSlot",
    "StitchDecision",
    "TrackFragment",
    "cosine_distance",
    "destructive_action_allowed",
    "embedding_centroid",
    "enroll_slots",
    "evaluate_stitch",
    "frame_court_evidence",
    "fusion_score",
    "identity_match_score",
    "infer_active_player_count",
    "mark_micro_fill_provenance",
    "median_real_footpoint_court_excess_m",
    "micro_fill_allowed",
    "off_court_excess_m",
    "open_set_decision",
    "presence_evidence",
    "recover_identity_conditioned_pool",
    "registered_hard_constraint_exclusions",
    "select_players_payload",
]
