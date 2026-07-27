"""Versioned contract for human-placed 3D ball labels (REVIEW-ONLY, not verified GT).

This module fixes the shape of ``ball_human_labels.json``: the artifact the
owner produces with ``scripts/racketsport/ball_label_studio.py``. It is a
*separate* artifact from every pipeline output. Raw observations
(``ball_track.json``, ``ball_candidates.json``, ``skeleton3d.json``,
``court_calibration.json``) stay immutable; nothing here ever writes into a
run directory.

Conventions are deliberately borrowed from the A-3 metric-3D observation
contract (``ball_metric3d_contract.py`` on ``ball-lane-20260723``): the same
``court_netcenter_z_up_m`` world frame, required per-axis ``sigma_xyz_m``
never a bare xyz, fail-closed validation, and deterministic serialization
(sorted keys, floats rounded to ``FLOAT_DECIMALS``, no wall-clock timestamps
or hostnames in the payload). Wall-clock session state lives in a separate,
explicitly non-deterministic session file.

The one thing this contract adds over the triangulated-GT contract is the
**accuracy tier**, because these labels are NOT all of the same quality:

===================  =======================  =========================================
label kind           accuracy tier            how depth along the camera ray was fixed
===================  =======================  =========================================
``bounce``           ``plane_solved``         ray-plane intersection at z=BALL_RADIUS_M
``near_player``      ``player_referenced``    human judgement against a tracked skeleton
``free_flight``      ``unreferenced_estimate``  human guess with no depth reference
===================  =======================  =========================================

The tier is derived from the kind and validated to match, and
``is_ground_truth_candidate`` is true *only* for ``bounce``. A free-flight
estimate can therefore never be promoted into the bounce tier by editing a
field: the validator rejects the payload. ``VERIFIED=0`` stays binding —
every payload carries ``verified_ground_truth: false`` and ``review_only:
true``, and validation rejects any attempt to set them otherwise.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_ball_human_label_set"
LABEL_FILE_NAME = "ball_human_labels.json"
SESSION_FILE_NAME = "ball_label_session.json"
FLOAT_DECIMALS = 6

# Same metric world frame as the production court calibration artifacts and
# the A-3 contract: x width, y along the baseline direction (net plane y=0),
# z up, origin under the net center, court surface z=0, metres.
WORLD_FRAME = "court_netcenter_z_up_m"

# Pickleball radius used by the production arc solver (ball_arc_solver.BALL_RADIUS_M).
BALL_RADIUS_M = 0.0371

KIND_BOUNCE = "bounce"
KIND_NEAR_PLAYER = "near_player"
KIND_FREE_FLIGHT = "free_flight"
LABEL_KINDS = (KIND_BOUNCE, KIND_NEAR_PLAYER, KIND_FREE_FLIGHT)

# The tier is a function of the kind. It is stored explicitly so downstream
# consumers never have to re-derive it, and validated so it can never lie.
ACCURACY_TIER_BY_KIND = {
    KIND_BOUNCE: "plane_solved",
    KIND_NEAR_PLAYER: "player_referenced",
    KIND_FREE_FLIGHT: "unreferenced_estimate",
}

# Only a bounce label has a geometrically solved depth. Everything else is a
# human depth judgement and must never be consumed as metric ground truth.
GROUND_TRUTH_CANDIDATE_KINDS = frozenset({KIND_BOUNCE})

DEPTH_SOURCES = frozenset(
    {
        "ray_plane_intersection",  # bounce: solved, no human depth input
        "human_drag",  # near_player / free_flight: human moved the marker
        "interpolated_arc",  # accepted from a ballistic interpolation proposal
    }
)

ORIGINS = frozenset(
    {
        "fresh",  # created from nothing by the human
        "prefill_confirmed",  # a pipeline prefill the human accepted unchanged
        "prefill_corrected",  # a pipeline prefill the human moved before saving
    }
)

HUMAN_CONFIDENCE = ("low", "medium", "high")

_UNIT_NORM_TOLERANCE = 1e-4
_BOUNCE_HEIGHT_TOLERANCE_M = 1e-6

# Verdicts a label's optional ``extrapolation`` block may carry. Mirrors
# ``calibration_extrapolation.VERDICT_*``; duplicated as literals so this
# contract module keeps its no-import-cycle independence, and pinned equal by
# test.
_EXTRAPOLATION_VERDICTS = frozenset(
    {"within_calibrated_envelope", "extrapolated", "far_extrapolated"}
)


class LabelContractError(ValueError):
    """Raised when a payload violates the human-label contract."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BallLabel:
    """One human-placed 3D ball position, with its evidence and its honesty."""

    frame: int
    timestamp_s: float
    pixel_xy: tuple[float, float]
    world_xyz_m: tuple[float, float, float]
    kind: str
    depth_along_ray_m: float
    ray_origin_m: tuple[float, float, float]
    ray_direction_unit: tuple[float, float, float]
    depth_source: str
    sigma_xyz_m: tuple[float, float, float]
    sigma_along_ray_m: float
    sigma_perp_m: float
    uncertainty_basis: str
    human_confidence: str
    origin: str
    prefill: Mapping[str, Any] | None = None
    near_player: Mapping[str, Any] | None = None
    notes: str = ""
    # Where this pixel sits relative to the image region the calibration's own
    # correspondences cover (``calibration_extrapolation.evaluate_pixel``). A
    # click can be perfectly correct and still land where the camera model was
    # never fit; the owner needs to see that on the label rather than have the
    # label quietly disappear. Optional so every label set written before this
    # existed still validates and still round-trips byte-identically.
    extrapolation: Mapping[str, Any] | None = None

    @property
    def accuracy_tier(self) -> str:
        return ACCURACY_TIER_BY_KIND[self.kind]

    @property
    def is_ground_truth_candidate(self) -> bool:
        return self.kind in GROUND_TRUTH_CANDIDATE_KINDS

    @property
    def label_id(self) -> str:
        return f"f{int(self.frame):06d}"

    def validate(self, *, path: str = "label") -> None:
        if not isinstance(self.frame, int) or self.frame < 0:
            raise LabelContractError(f"{path}.frame: expected non-negative int, got {self.frame!r}")
        _require_finite(self.timestamp_s, path=f"{path}.timestamp_s")
        if float(self.timestamp_s) < 0.0:
            raise LabelContractError(f"{path}.timestamp_s: must be >= 0")
        _require_vec(self.pixel_xy, 2, path=f"{path}.pixel_xy")
        _require_vec(self.world_xyz_m, 3, path=f"{path}.world_xyz_m")
        if self.kind not in LABEL_KINDS:
            raise LabelContractError(
                f"{path}.kind: unknown kind {self.kind!r}; known: {sorted(LABEL_KINDS)}"
            )
        if self.depth_source not in DEPTH_SOURCES:
            raise LabelContractError(
                f"{path}.depth_source: unknown {self.depth_source!r}; "
                f"known: {sorted(DEPTH_SOURCES)}"
            )
        if self.origin not in ORIGINS:
            raise LabelContractError(
                f"{path}.origin: unknown {self.origin!r}; known: {sorted(ORIGINS)}"
            )
        if self.human_confidence not in HUMAN_CONFIDENCE:
            raise LabelContractError(
                f"{path}.human_confidence: unknown {self.human_confidence!r}; "
                f"known: {list(HUMAN_CONFIDENCE)}"
            )

        _require_vec(self.ray_origin_m, 3, path=f"{path}.ray_origin_m")
        _require_vec(self.ray_direction_unit, 3, path=f"{path}.ray_direction_unit")
        norm = math.sqrt(sum(float(v) * float(v) for v in self.ray_direction_unit))
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise LabelContractError(
                f"{path}.ray_direction_unit: expected unit norm, got {norm!r}"
            )

        # Sign guard. Depth is measured in metres from the camera origin ALONG
        # the unit ray direction, so a valid label is always in front of the
        # camera. A negative or zero depth means the ray parameterisation was
        # inverted somewhere, which would silently mirror every label through
        # the camera centre.
        _require_finite(self.depth_along_ray_m, path=f"{path}.depth_along_ray_m")
        if float(self.depth_along_ray_m) <= 0.0:
            raise LabelContractError(
                f"{path}.depth_along_ray_m: must be > 0 (label must be in front of the "
                f"camera), got {self.depth_along_ray_m!r}"
            )

        # The stored world point must actually lie on the stored ray at the
        # stored depth. This is the second half of the sign/consistency guard:
        # it makes an inverted or mismatched parameterisation impossible to
        # persist even if the three fields were written independently.
        expected = tuple(
            float(self.ray_origin_m[i]) + float(self.depth_along_ray_m) * float(self.ray_direction_unit[i])
            for i in range(3)
        )
        offset = math.dist(expected, tuple(float(v) for v in self.world_xyz_m))
        if offset > 1e-3:
            raise LabelContractError(
                f"{path}.world_xyz_m: does not lie on ray_origin_m + depth_along_ray_m * "
                f"ray_direction_unit (off by {offset:.6f} m)"
            )

        _require_vec(self.sigma_xyz_m, 3, path=f"{path}.sigma_xyz_m")
        for axis, sigma in zip("xyz", self.sigma_xyz_m):
            if not float(sigma) > 0.0:
                raise LabelContractError(
                    f"{path}.sigma_xyz_m.{axis}: per-axis sigma must be > 0, got {sigma!r}"
                )
        for name, value in (
            ("sigma_along_ray_m", self.sigma_along_ray_m),
            ("sigma_perp_m", self.sigma_perp_m),
        ):
            _require_finite(value, path=f"{path}.{name}")
            if not float(value) > 0.0:
                raise LabelContractError(f"{path}.{name}: must be > 0, got {value!r}")
        if not self.uncertainty_basis:
            raise LabelContractError(f"{path}.uncertainty_basis: must be a non-empty string")

        if self.kind == KIND_BOUNCE:
            if self.depth_source != "ray_plane_intersection":
                raise LabelContractError(
                    f"{path}.depth_source: bounce labels must be solved by "
                    f"ray_plane_intersection, got {self.depth_source!r}"
                )
            height = float(self.world_xyz_m[2])
            if abs(height - BALL_RADIUS_M) > _BOUNCE_HEIGHT_TOLERANCE_M:
                raise LabelContractError(
                    f"{path}.world_xyz_m.z: a bounce label sits one ball radius above the "
                    f"court ({BALL_RADIUS_M} m), got {height!r}"
                )
        elif self.depth_source == "ray_plane_intersection":
            raise LabelContractError(
                f"{path}.depth_source: only bounce labels may claim ray_plane_intersection"
            )

        if self.kind == KIND_NEAR_PLAYER:
            reference = self.near_player
            if not isinstance(reference, Mapping):
                raise LabelContractError(
                    f"{path}.near_player: a near_player label must record the tracked "
                    f"player/joint it was judged against"
                )
            for key in ("player_id", "joint_index", "joint_name", "offset_from_ray_m"):
                if key not in reference:
                    raise LabelContractError(f"{path}.near_player.{key}: required")
        if self.prefill is not None and not isinstance(self.prefill, Mapping):
            raise LabelContractError(f"{path}.prefill: expected object or null")
        if self.extrapolation is not None:
            if not isinstance(self.extrapolation, Mapping):
                raise LabelContractError(f"{path}.extrapolation: expected object or null")
            verdict = self.extrapolation.get("verdict")
            if verdict not in _EXTRAPOLATION_VERDICTS:
                raise LabelContractError(
                    f"{path}.extrapolation.verdict: unknown {verdict!r}; "
                    f"known: {sorted(_EXTRAPOLATION_VERDICTS)}"
                )
        if self.origin != "fresh" and self.prefill is None:
            raise LabelContractError(
                f"{path}.prefill: origin={self.origin!r} claims a prefill was used but none "
                f"is recorded; a prefill must never be silently promoted"
            )

    @property
    def is_extrapolated(self) -> bool:
        """True when this pixel lies outside the calibrated image envelope.

        False when no envelope was recorded: an old label set says nothing
        about extrapolation, and silence is not evidence of safety. Read
        ``extrapolation`` itself to tell "checked and inside" from "never
        checked".
        """

        record = self.extrapolation
        if not isinstance(record, Mapping):
            return False
        return bool(record.get("extrapolated", False))

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "accuracy_tier": self.accuracy_tier,
            "depth_along_ray_m": _round(self.depth_along_ray_m),
            "depth_source": self.depth_source,
            "frame": int(self.frame),
            "human_confidence": self.human_confidence,
            "is_ground_truth_candidate": self.is_ground_truth_candidate,
            "kind": self.kind,
            "label_id": self.label_id,
            "near_player": _round_tree(self.near_player),
            "notes": str(self.notes),
            "origin": self.origin,
            "pixel_xy": [_round(v) for v in self.pixel_xy],
            "prefill": _round_tree(self.prefill),
            "ray_direction_unit": [_round(v) for v in self.ray_direction_unit],
            "ray_origin_m": [_round(v) for v in self.ray_origin_m],
            "sigma_along_ray_m": _round(self.sigma_along_ray_m),
            "sigma_perp_m": _round(self.sigma_perp_m),
            "sigma_xyz_m": [_round(v) for v in self.sigma_xyz_m],
            "timestamp_s": _round(self.timestamp_s),
            "uncertainty_basis": str(self.uncertainty_basis),
            "world_xyz_m": [_round(v) for v in self.world_xyz_m],
        }
        # Emitted only when it was actually computed, so a label set written
        # before the envelope existed serializes byte-identically and an
        # absent key stays honest about never having been checked.
        if self.extrapolation is not None:
            payload["extrapolation"] = _round_tree(self.extrapolation)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "label") -> "BallLabel":
        record = _require_mapping(payload, path=path)
        try:
            label = cls(
                frame=int(record["frame"]),
                timestamp_s=float(record["timestamp_s"]),
                pixel_xy=_tuple(record["pixel_xy"], 2),
                world_xyz_m=_tuple(record["world_xyz_m"], 3),
                kind=str(record["kind"]),
                depth_along_ray_m=float(record["depth_along_ray_m"]),
                ray_origin_m=_tuple(record["ray_origin_m"], 3),
                ray_direction_unit=_tuple(record["ray_direction_unit"], 3),
                depth_source=str(record["depth_source"]),
                sigma_xyz_m=_tuple(record["sigma_xyz_m"], 3),
                sigma_along_ray_m=float(record["sigma_along_ray_m"]),
                sigma_perp_m=float(record["sigma_perp_m"]),
                uncertainty_basis=str(record["uncertainty_basis"]),
                human_confidence=str(record["human_confidence"]),
                origin=str(record["origin"]),
                prefill=record.get("prefill"),
                near_player=record.get("near_player"),
                notes=str(record.get("notes") or ""),
                extrapolation=record.get("extrapolation"),
            )
        except KeyError as exc:
            raise LabelContractError(f"{path}.{exc.args[0]}: required") from exc
        except (TypeError, ValueError) as exc:
            raise LabelContractError(f"{path}: {exc}") from exc

        # Reject a stored tier/flag that disagrees with the kind. This is the
        # fence that stops a free-flight estimate being relabelled as a
        # plane-solved bounce by hand-editing the JSON.
        stored_tier = record.get("accuracy_tier")
        if stored_tier is not None and stored_tier != label.accuracy_tier:
            raise LabelContractError(
                f"{path}.accuracy_tier: {stored_tier!r} does not match kind={label.kind!r} "
                f"(expected {label.accuracy_tier!r})"
            )
        stored_flag = record.get("is_ground_truth_candidate")
        if stored_flag is not None and bool(stored_flag) != label.is_ground_truth_candidate:
            raise LabelContractError(
                f"{path}.is_ground_truth_candidate: {stored_flag!r} does not match "
                f"kind={label.kind!r} (expected {label.is_ground_truth_candidate})"
            )
        label.validate(path=path)
        return label


@dataclass
class BallLabelSet:
    """The whole ``ball_human_labels.json`` payload."""

    clip_id: str
    fps: float
    frame_count: int
    image_size: tuple[int, int]
    source_artifacts: Mapping[str, Any] = field(default_factory=dict)
    calibration_evidence: Mapping[str, Any] = field(default_factory=dict)
    labels: list[BallLabel] = field(default_factory=list)

    POLICY_NOTES = (
        "Human review-only labels. VERIFIED=0: this artifact is not verified ground truth.",
        "Depth along the camera ray is SOLVED only for kind=bounce (ray-plane intersection "
        "at z=BALL_RADIUS_M). near_player depth is a human judgement against a tracked "
        "skeleton; free_flight depth is an unreferenced human estimate.",
        "Only kind=bounce carries is_ground_truth_candidate=true. Do not aggregate the three "
        "kinds into one accuracy number.",
        "Per-label sigma_xyz_m is an honest estimate, not a measured covariance.",
        "Raw pipeline observations are immutable and were read, never written.",
    )

    def validate(self) -> None:
        if not self.clip_id:
            raise LabelContractError("clip_id: must be a non-empty string")
        _require_finite(self.fps, path="fps")
        if float(self.fps) <= 0.0:
            raise LabelContractError("fps: must be > 0")
        if int(self.frame_count) <= 0:
            raise LabelContractError("frame_count: must be > 0")
        seen: set[int] = set()
        previous_frame = -1
        for index, label in enumerate(self.labels):
            label.validate(path=f"labels[{index}]")
            if label.frame in seen:
                raise LabelContractError(
                    f"labels[{index}].frame: duplicate frame {label.frame} "
                    f"(one label per frame)"
                )
            if label.frame <= previous_frame:
                raise LabelContractError(
                    f"labels[{index}].frame: labels must be sorted by frame "
                    f"({label.frame} follows {previous_frame})"
                )
            if label.frame >= int(self.frame_count):
                raise LabelContractError(
                    f"labels[{index}].frame: {label.frame} is outside the clip "
                    f"(frame_count={self.frame_count})"
                )
            seen.add(label.frame)
            previous_frame = label.frame

    def summary(self) -> dict[str, Any]:
        by_kind = {kind: 0 for kind in LABEL_KINDS}
        by_tier = {tier: 0 for tier in sorted(set(ACCURACY_TIER_BY_KIND.values()))}
        by_origin = {origin: 0 for origin in sorted(ORIGINS)}
        by_confidence = {level: 0 for level in HUMAN_CONFIDENCE}
        by_depth_source = {source: 0 for source in sorted(DEPTH_SOURCES)}
        by_extrapolation = {verdict: 0 for verdict in sorted(_EXTRAPOLATION_VERDICTS)}
        unchecked = 0
        for label in self.labels:
            by_kind[label.kind] += 1
            by_tier[label.accuracy_tier] += 1
            by_origin[label.origin] += 1
            by_confidence[label.human_confidence] += 1
            by_depth_source[label.depth_source] += 1
            record = label.extrapolation
            if isinstance(record, Mapping) and record.get("verdict") in by_extrapolation:
                by_extrapolation[str(record["verdict"])] += 1
            else:
                unchecked += 1
        by_extrapolation["not_checked"] = unchecked
        frames = [label.frame for label in self.labels]
        return {
            "label_count": len(self.labels),
            "ground_truth_candidate_count": sum(
                1 for label in self.labels if label.is_ground_truth_candidate
            ),
            # A ground-truth candidate whose pixel sits outside the calibrated
            # image envelope is still a correct click, but its 3D position
            # rests on an extrapolated camera model. Counted separately so no
            # consumer aggregates the two without noticing.
            "extrapolated_ground_truth_candidate_count": sum(
                1
                for label in self.labels
                if label.is_ground_truth_candidate and label.is_extrapolated
            ),
            "by_extrapolation_verdict": by_extrapolation,
            "by_kind": by_kind,
            "by_accuracy_tier": by_tier,
            "by_origin": by_origin,
            "by_human_confidence": by_confidence,
            "by_depth_source": by_depth_source,
            "labeled_frame_span": [min(frames), max(frames)] if frames else None,
            "frame_coverage_fraction": _round(
                len(frames) / float(self.frame_count) if self.frame_count else 0.0
            ),
        }

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "artifact_type": ARTIFACT_TYPE,
            "calibration_evidence": _round_tree(dict(self.calibration_evidence)),
            "clip_id": str(self.clip_id),
            "fps": _round(self.fps),
            "frame_count": int(self.frame_count),
            "generated_by": "scripts/racketsport/ball_label_studio.py",
            "human_reviewed": True,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "labels": [label.to_json_dict() for label in self.labels],
            "not_ground_truth": True,
            "policy": {
                "accuracy_tier_by_kind": dict(ACCURACY_TIER_BY_KIND),
                "ground_truth_candidate_kinds": sorted(GROUND_TRUTH_CANDIDATE_KINDS),
                "notes": list(self.POLICY_NOTES),
            },
            "review_only": True,
            "schema_version": SCHEMA_VERSION,
            "source_artifacts": _round_tree(dict(self.source_artifacts)),
            "summary": self.summary(),
            "verified_ground_truth": False,
            "world_frame": WORLD_FRAME,
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "label_set") -> "BallLabelSet":
        record = _require_mapping(payload, path=path)
        artifact_type = record.get("artifact_type")
        if artifact_type != ARTIFACT_TYPE:
            raise LabelContractError(
                f"{path}.artifact_type: expected {ARTIFACT_TYPE!r}, got {artifact_type!r}"
            )
        version = record.get("schema_version")
        if version != SCHEMA_VERSION:
            raise LabelContractError(
                f"{path}.schema_version: unsupported version {version!r} "
                f"(this build understands {SCHEMA_VERSION})"
            )
        if record.get("world_frame") != WORLD_FRAME:
            raise LabelContractError(
                f"{path}.world_frame: expected {WORLD_FRAME!r}, got {record.get('world_frame')!r}"
            )
        for flag, expected in (
            ("verified_ground_truth", False),
            ("review_only", True),
            ("not_ground_truth", True),
        ):
            if flag in record and bool(record[flag]) is not expected:
                raise LabelContractError(
                    f"{path}.{flag}: must be {expected} for a human review-only label set"
                )
        raw_labels = record.get("labels")
        if not isinstance(raw_labels, list):
            raise LabelContractError(f"{path}.labels: expected a list")
        image_size = record.get("image_size") or [0, 0]
        label_set = cls(
            clip_id=str(record.get("clip_id") or ""),
            fps=float(record.get("fps") or 0.0),
            frame_count=int(record.get("frame_count") or 0),
            image_size=(int(image_size[0]), int(image_size[1])),
            source_artifacts=record.get("source_artifacts") or {},
            calibration_evidence=record.get("calibration_evidence") or {},
            labels=[
                BallLabel.from_json_dict(item, path=f"{path}.labels[{index}]")
                for index, item in enumerate(raw_labels)
            ],
        )
        label_set.validate()
        return label_set

    def upsert(self, label: BallLabel) -> None:
        """Replace-or-insert a label, keeping the list sorted by frame."""

        label.validate()
        self.labels = [item for item in self.labels if item.frame != label.frame]
        self.labels.append(label)
        self.labels.sort(key=lambda item: item.frame)

    def remove(self, frame: int) -> bool:
        before = len(self.labels)
        self.labels = [item for item in self.labels if item.frame != int(frame)]
        return len(self.labels) != before

    def get(self, frame: int) -> BallLabel | None:
        for label in self.labels:
            if label.frame == int(frame):
                return label
        return None


# ---------------------------------------------------------------------------
# Persistence (atomic, so an autosave can never truncate an hour of work)
# ---------------------------------------------------------------------------


def dumps(label_set: BallLabelSet) -> str:
    return json.dumps(label_set.to_json_dict(), indent=2, sort_keys=True) + "\n"


def write_label_set(path: str | Path, label_set: BallLabelSet) -> Path:
    """Serialize atomically: write a sibling temp file, then ``os.replace``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(label_set)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def read_label_set(path: str | Path) -> BallLabelSet:
    return BallLabelSet.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_mapping(payload: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise LabelContractError(f"{path}: expected an object, got {type(payload).__name__}")
    return payload


def _require_finite(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelContractError(f"{path}: expected a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise LabelContractError(f"{path}: expected a finite number, got {value!r}")
    return number


def _require_vec(value: Any, length: int, *, path: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LabelContractError(f"{path}: expected a sequence of {length} numbers")
    if len(value) != length:
        raise LabelContractError(f"{path}: expected {length} numbers, got {len(value)}")
    return tuple(_require_finite(item, path=f"{path}[{index}]") for index, item in enumerate(value))


def _tuple(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"expected {length} numbers, got {value!r}")
    return tuple(float(item) for item in value)


def _round(value: Any) -> float:
    return round(float(value), FLOAT_DECIMALS)


def _round_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _round_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round_tree(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _round(value)
    return value


__all__ = [
    "ACCURACY_TIER_BY_KIND",
    "ARTIFACT_TYPE",
    "BALL_RADIUS_M",
    "BallLabel",
    "BallLabelSet",
    "DEPTH_SOURCES",
    "GROUND_TRUTH_CANDIDATE_KINDS",
    "HUMAN_CONFIDENCE",
    "KIND_BOUNCE",
    "KIND_FREE_FLIGHT",
    "KIND_NEAR_PLAYER",
    "LABEL_FILE_NAME",
    "LABEL_KINDS",
    "LabelContractError",
    "ORIGINS",
    "SCHEMA_VERSION",
    "SESSION_FILE_NAME",
    "WORLD_FRAME",
    "dumps",
    "read_label_set",
    "write_label_set",
]
