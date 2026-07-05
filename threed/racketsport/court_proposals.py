"""Fail-closed court proposal artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


Point2D = tuple[float, float]


@dataclass(frozen=True)
class CourtProposal:
    proposal_id: str
    source: str
    court_keypoints: dict[str, Point2D]
    scores: dict[str, float | int | None]
    homography_image_from_court: list[list[float]] | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        gate = {
            "auto_usable": False,
            "review_usable": True,
            "failed": ["not_verified"],
            "warnings": [],
        }
        gate.update(self.gate)
        gate["auto_usable"] = False
        failed = [str(item) for item in gate.get("failed", [])]
        if "not_verified" not in failed:
            failed.append("not_verified")
        gate["failed"] = failed
        gate["warnings"] = [str(item) for item in gate.get("warnings", [])]
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "verified": False,
            "not_cal3_verified": True,
            "court_keypoints": {
                name: [float(x), float(y)]
                for name, (x, y) in sorted(self.court_keypoints.items())
            },
            "homography_image_from_court": self.homography_image_from_court,
            "scores": dict(sorted(self.scores.items())),
            "gate": gate,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CourtProposalReport:
    clip: str
    image_size: tuple[int, int]
    frame_indices: list[int]
    proposals: list[CourtProposal]
    video: str | None = None
    motion_mode: str = "unknown"
    assist: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        selected = self.proposals[0].proposal_id if self.proposals else None
        return {
            "artifact_type": "racketsport_court_proposals",
            "schema_version": 1,
            "clip": self.clip,
            "status": "ranked_not_verified",
            "verified": False,
            "not_cal3_verified": True,
            "input": {
                "video": self.video,
                "frame_indices": [int(index) for index in self.frame_indices],
                "image_size": [int(self.image_size[0]), int(self.image_size[1])],
                "motion_mode": self.motion_mode,
            },
            "assist": self.assist or {"mode": "none", "tap_points": [], "line_label": None},
            "ranking": {
                "selected_proposal_id": selected,
                "selection_reason": "best_score_but_review_required" if selected else "no_proposals",
                "abstain": True,
                "abstain_reasons": ["not_cal3_verified"],
            },
            "proposals": [proposal.to_json_dict() for proposal in self.proposals],
        }


def write_court_proposal_report(path: str | Path, report: CourtProposalReport) -> None:
    Path(path).write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
