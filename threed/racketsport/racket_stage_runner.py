"""Fail-closed PADDLE/RACKET StageRunner from explicit four-corner candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .racket6dof import estimate_planar_paddle_pose_with_diagnostics, validate_paddle_dimensions
from .racket_pose_readiness import build_racket_pose_readiness, write_racket_pose_readiness
from .racket_promotion_audit import build_racket_promotion_audit, write_racket_promotion_audit
from .racket_true_corners import is_box_derived_source
from .schemas import CourtCalibration, RacketCandidateFrame, RacketCandidates, RacketPose, validate_artifact_file


DEFAULT_RACKET_CANDIDATES_FILENAME = "racket_candidates.json"
RACKET_STAGE_DIAGNOSTICS_FILENAME = "racket_stage_diagnostics.json"
RACKET_POSE_HYPOTHESES_FILENAME = "racket_pose_hypotheses.json"


@dataclass(frozen=True)
class RacketStageRun:
    stage: str
    status: str
    real_model: bool
    source_mode: str
    produced_artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "real_model": self.real_model,
            "source_mode": self.source_mode,
            "produced_artifacts": list(self.produced_artifacts),
            "notes": list(self.notes),
            "metrics": self.metrics,
        }


class RacketStageRunner:
    stage = "racket"
    real_model = False
    source_mode = "explicit_four_corner_candidates_pnp_ippe"

    def __init__(
        self,
        *,
        candidate_path: str | Path | None = None,
        max_reprojection_error_px: float = 6.0,
        ambiguity_margin_threshold_px: float = 1.0,
        reject_ambiguous: bool = False,
    ) -> None:
        if max_reprojection_error_px < 0.0:
            raise ValueError("max_reprojection_error_px must be non-negative")
        if ambiguity_margin_threshold_px < 0.0:
            raise ValueError("ambiguity_margin_threshold_px must be non-negative")
        self.candidate_path = Path(candidate_path) if candidate_path is not None else None
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.ambiguity_margin_threshold_px = float(ambiguity_margin_threshold_px)
        self.reject_ambiguous = reject_ambiguous

    def run(self, context: Any) -> RacketStageRun:
        candidates_path = self._resolve_candidate_path(context)
        calibration = validate_artifact_file("court_calibration", context.run_dir / "court_calibration.json")
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        candidates = validate_artifact_file("racket_candidates", candidates_path)
        if not isinstance(candidates, RacketCandidates):
            raise ValueError("racket_candidates.json did not validate as RacketCandidates")
        fps = candidates.fps
        camera_matrix = _camera_matrix(calibration)
        dist_coeffs = calibration.intrinsics.dist

        players: list[dict[str, Any]] = []
        hypothesis_players: list[dict[str, Any]] = []
        metrics = {
            "candidate_path": str(candidates_path),
            "candidate_frame_count": 0,
            "accepted_frame_count": 0,
            "rejected_high_reprojection_count": 0,
            "rejected_ambiguous_count": 0,
            "carried_ambiguous_count": 0,
            "invalid_candidate_count": 0,
            "rejected_box_derived_source_count": 0,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "ambiguity_margin_threshold_px": self.ambiguity_margin_threshold_px,
            "reject_ambiguous": self.reject_ambiguous,
            "not_gate_verified": True,
        }

        box_sources = _box_derived_candidate_sources(candidates)
        if box_sources:
            metrics["candidate_frame_count"] = sum(len(player.frames) for player in candidates.players)
            metrics["rejected_box_derived_source_count"] = sum(box_sources.values())
            metrics["box_derived_sources"] = box_sources
            _write_json(
                context.run_dir / RACKET_STAGE_DIAGNOSTICS_FILENAME,
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_racket_stage_diagnostics",
                    "stage": self.stage,
                    "status": "failed",
                    "source_mode": self.source_mode,
                    "produced_artifacts": [],
                    "metrics": metrics,
                    "notes": [
                        "no racket_pose.json written because candidates are box-derived preview evidence",
                        "collect reviewed true paddle corners, CAD/reference, or keypoint/mask corners before RKT promotion",
                    ],
                },
            )
            raise ValueError(f"box-derived racket candidates cannot promote to racket_pose.json: {box_sources}")

        for player_payload in candidates.players:
            player_id = player_payload.id
            paddle_dims = validate_paddle_dimensions(player_payload.paddle_dims_in)
            frames = []
            hypothesis_frames = []
            for frame_payload in player_payload.frames:
                metrics["candidate_frame_count"] += 1
                try:
                    frame, hypothesis_frame = self._pose_frame(
                        frame_payload,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        paddle_dims_in={"length": paddle_dims.length_in, "width": paddle_dims.width_in},
                    )
                except (TypeError, ValueError):
                    metrics["invalid_candidate_count"] += 1
                    continue
                if frame["reprojection_error_px"] > self.max_reprojection_error_px:
                    metrics["rejected_high_reprojection_count"] += 1
                    continue
                if self.reject_ambiguous and frame["ambiguous"]:
                    metrics["rejected_ambiguous_count"] += 1
                    continue
                if frame["ambiguous"]:
                    metrics["carried_ambiguous_count"] += 1
                metrics["accepted_frame_count"] += 1
                frames.append(frame)
                hypothesis_frames.append(hypothesis_frame)
            if frames:
                players.append(
                    {
                        "id": player_id,
                        "paddle_dims_in": {"length": paddle_dims.length_in, "width": paddle_dims.width_in},
                        "frames": frames,
                        "contacts": [],
                    }
                )
                hypothesis_players.append(
                    {
                        "id": player_id,
                        "paddle_dims_in": {"length": paddle_dims.length_in, "width": paddle_dims.width_in},
                        "frames": hypothesis_frames,
                    }
                )

        if not players:
            _write_json(
                context.run_dir / RACKET_STAGE_DIAGNOSTICS_FILENAME,
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_racket_stage_diagnostics",
                    "stage": self.stage,
                    "status": "failed",
                    "source_mode": self.source_mode,
                    "produced_artifacts": [],
                    "metrics": metrics,
                    "notes": [
                        "no racket_pose.json written because all candidates failed fail-closed checks",
                        "prototype diagnostics only; not a RKT accuracy gate",
                    ],
                },
            )
            raise ValueError(
                "no accepted racket pose frames; "
                f"candidate_frame_count={metrics['candidate_frame_count']}, "
                f"rejected_high_reprojection_count={metrics['rejected_high_reprojection_count']}, "
                f"rejected_ambiguous_count={metrics['rejected_ambiguous_count']}, "
                f"invalid_candidate_count={metrics['invalid_candidate_count']}"
            )

        racket_pose = RacketPose.model_validate(
            {
                "schema_version": 1,
                "fps": fps,
                "world_frame": "camera",
                "translation_unit": "cm",
                "players": players,
            }
        )
        _write_json(context.run_dir / "racket_pose.json", racket_pose.model_dump(mode="json"))
        _write_json(
            context.run_dir / RACKET_POSE_HYPOTHESES_FILENAME,
            {
                "schema_version": 1,
                "artifact_type": "racketsport_racket_pose_hypotheses",
                "fps": fps,
                "world_frame": "camera",
                "translation_unit": "cm",
                "players": hypothesis_players,
            },
        )
        readiness = build_racket_pose_readiness(
            clip=str(getattr(context, "clip", "")),
            racket_candidates=candidates,
            racket_pose=racket_pose,
        )
        write_racket_pose_readiness(context.run_dir / "racket_pose_readiness.json", readiness)
        promotion_audit = build_racket_promotion_audit(
            clip=str(getattr(context, "clip", "")),
            racket_candidates=candidates,
            racket_pose=racket_pose,
        )
        write_racket_promotion_audit(context.run_dir / "racket_promotion_audit.json", promotion_audit)
        return RacketStageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=(
                "racket_pose.json",
                RACKET_POSE_HYPOTHESES_FILENAME,
                "racket_pose_readiness.json",
                "racket_promotion_audit.json",
            ),
            notes=(
                "consumed explicit four-corner paddle candidates and solved planar PnP/IPPE",
                "retains both finite IPPE hypotheses; ambiguous candidates are carried by default rather than ranked away by reprojection alone",
                "fails closed when candidates are missing, invalid, degenerate, or above reprojection threshold",
                "prototype integration only; real PADDLE verification still requires detector/SAM2 candidates and ArUco/AprilTag GT",
            ),
            metrics=metrics,
        )

    def _resolve_candidate_path(self, context: Any) -> Path:
        candidates = [self.candidate_path] if self.candidate_path is not None else [
            Path(context.inputs_dir) / DEFAULT_RACKET_CANDIDATES_FILENAME,
            Path(context.run_dir) / DEFAULT_RACKET_CANDIDATES_FILENAME,
        ]
        searched: list[Path] = []
        for candidate in candidates:
            if candidate is None:
                continue
            searched.append(candidate)
            if candidate.is_file():
                return candidate
        searched_text = ", ".join(str(path) for path in searched)
        raise FileNotFoundError(
            f"missing racket candidate artifact: {DEFAULT_RACKET_CANDIDATES_FILENAME}; searched: {searched_text}"
        )

    def _pose_frame(
        self,
        frame_payload: RacketCandidateFrame,
        *,
        camera_matrix: Sequence[Sequence[float]],
        dist_coeffs: Sequence[float],
        paddle_dims_in: Mapping[str, float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        estimate = estimate_planar_paddle_pose_with_diagnostics(
            frame_payload.corners_px,
            camera_matrix,
            paddle_dims_in,
            dist_coeffs=dist_coeffs,
            ambiguity_margin_threshold_px=self.ambiguity_margin_threshold_px,
        )
        pose = estimate.pose
        frame_confidence = max(0.0, min(1.0, frame_payload.conf * pose.confidence))
        frame = {
            "t": frame_payload.t,
            "pose_se3": {
                "R": [list(row) for row in pose.R],
                "t": list(pose.t),
            },
            "conf": frame_confidence,
            "world_frame": "camera",
            "translation_unit": "cm",
            "source": f"{frame_payload.source}:pnp_ippe",
            "reprojection_error_px": estimate.reprojection_error_px,
            "ambiguous": estimate.ambiguous,
        }
        alt_pose = estimate.alt_pose
        hypothesis_frame = {
            "t": frame_payload.t,
            "primary_pose": {
                "pose_se3": {"R": [list(row) for row in pose.R], "t": list(pose.t)},
                "confidence": pose.confidence,
                "frame_conf": frame_confidence,
                "reprojection_error_px": estimate.reprojection_error_px,
                "source": pose.source,
            },
            "alt_pose": (
                {
                    "pose_se3": {"R": [list(row) for row in alt_pose.R], "t": list(alt_pose.t)},
                    "confidence": alt_pose.confidence,
                    "frame_conf": max(0.0, min(1.0, frame_payload.conf * alt_pose.confidence)),
                    "reprojection_error_px": estimate.candidate_reprojection_errors_px[1],
                    "source": alt_pose.source,
                }
                if alt_pose is not None
                else None
            ),
            "candidate_reprojection_errors_px": list(estimate.candidate_reprojection_errors_px),
            "ambiguity_margin_px": estimate.ambiguity_margin_px,
            "ambiguous": estimate.ambiguous,
        }
        return frame, hypothesis_frame


def _camera_matrix(calibration: CourtCalibration) -> list[list[float]]:
    intrinsics = calibration.intrinsics
    return [
        [intrinsics.fx, 0.0, intrinsics.cx],
        [0.0, intrinsics.fy, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ]


def _box_derived_candidate_sources(candidates: RacketCandidates) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in candidates.players:
        for frame in player.frames:
            if is_box_derived_source(frame.source):
                counts[frame.source] = counts.get(frame.source, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_RACKET_CANDIDATES_FILENAME",
    "RACKET_STAGE_DIAGNOSTICS_FILENAME",
    "RACKET_POSE_HYPOTHESES_FILENAME",
    "RacketStageRun",
    "RacketStageRunner",
]
