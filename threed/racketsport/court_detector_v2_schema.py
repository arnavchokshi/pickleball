"""Schema helpers for court detector v2 proposal artifacts."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DETECTOR_V2_ARTIFACT_TYPE = "racketsport_court_detector_v2_proposals"
DETECTOR_V2_SCHEMA_VERSION = 1


class CourtDetectorV2Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifact_type: Literal["racketsport_court_detector_v2_proposals"]
    clip: str
    source_frame: str | None = None
    image_size: tuple[int, int]
    promoted: bool
    verified: bool
    not_cal3_verified: bool
    promotion_status: Literal["promoted", "needs_user_input", "blocked"]
    promotion_blockers: list[str] = Field(default_factory=list)
    selected_hypothesis_id: str | None = None
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    net_evidence: dict[str, Any] = Field(default_factory=dict)
    surface_evidence: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    needs_user_input: list[str] = Field(default_factory=list)

    @field_validator("clip")
    @classmethod
    def _non_empty_clip(cls, value: str) -> str:
        if not value:
            raise ValueError("clip must be non-empty")
        return value

    @field_validator("image_size")
    @classmethod
    def _positive_image_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or int(value[0]) <= 0 or int(value[1]) <= 0:
            raise ValueError("image_size must contain positive width and height")
        return (int(value[0]), int(value[1]))

    @model_validator(mode="after")
    def _promotion_requires_verified(self) -> "CourtDetectorV2Proposal":
        if self.promoted and (not self.verified or self.not_cal3_verified):
            raise ValueError("promoted detector v2 proposals must be verified and CAL-3 eligible")
        if self.promoted and self.promotion_status != "promoted":
            raise ValueError("promoted detector v2 proposals must use promotion_status=promoted")
        if not self.promoted and not self.not_cal3_verified:
            raise ValueError("blocked detector v2 proposals must preserve not_cal3_verified")
        if not self.promoted and self.verified:
            raise ValueError("blocked detector v2 proposals cannot be verified")
        return self


def build_blocked_detector_v2_proposal(
    *,
    clip: str,
    source_frame: str | None,
    image_size: tuple[int, int],
    blockers: Sequence[str],
    needs_user_input: Sequence[str],
    selected_hypothesis_id: str | None = None,
    hypotheses: Sequence[dict[str, Any]] = (),
    net_evidence: dict[str, Any] | None = None,
    surface_evidence: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "schema_version": DETECTOR_V2_SCHEMA_VERSION,
        "artifact_type": DETECTOR_V2_ARTIFACT_TYPE,
        "clip": clip,
        "source_frame": source_frame,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "promoted": False,
        "verified": False,
        "not_cal3_verified": True,
        "promotion_status": "needs_user_input" if needs_user_input else "blocked",
        "promotion_blockers": [str(blocker) for blocker in blockers],
        "selected_hypothesis_id": selected_hypothesis_id,
        "hypotheses": [dict(hypothesis) for hypothesis in hypotheses],
        "net_evidence": dict(net_evidence or {}),
        "surface_evidence": dict(surface_evidence or {}),
        "verification": dict(verification or {}),
        "needs_user_input": [str(name) for name in needs_user_input],
    }
    return CourtDetectorV2Proposal.model_validate(artifact).model_dump(mode="json")


def build_promoted_detector_v2_proposal(
    *,
    clip: str,
    source_frame: str | None,
    image_size: tuple[int, int],
    selected_hypothesis_id: str,
    hypotheses: Sequence[dict[str, Any]],
    net_evidence: dict[str, Any] | None = None,
    surface_evidence: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "schema_version": DETECTOR_V2_SCHEMA_VERSION,
        "artifact_type": DETECTOR_V2_ARTIFACT_TYPE,
        "clip": clip,
        "source_frame": source_frame,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "promoted": True,
        "verified": True,
        "not_cal3_verified": False,
        "promotion_status": "promoted",
        "promotion_blockers": [],
        "selected_hypothesis_id": selected_hypothesis_id,
        "hypotheses": [dict(hypothesis) for hypothesis in hypotheses],
        "net_evidence": dict(net_evidence or {}),
        "surface_evidence": dict(surface_evidence or {}),
        "verification": dict(verification or {}),
        "needs_user_input": [],
    }
    return CourtDetectorV2Proposal.model_validate(artifact).model_dump(mode="json")
