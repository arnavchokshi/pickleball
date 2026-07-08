"""BODY post-chain knob configuration shared by local and remote runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RAW_GROUNDED_JOINTS_ARTIFACT = "body_raw_grounded_joints.json"

POSTCHAIN_STAGE_ORDER: tuple[str, ...] = (
    "temporal_smoothing",
    "foot_lock",
    "foot_pin",
    "contact_splice",
    "wrist_lock",
    "world_joint_visual_smoothing",
)


@dataclass(frozen=True)
class BodyPostChainConfig:
    temporal_smoothing: bool = True
    foot_lock: bool = True
    foot_pin: bool = True
    contact_splice: bool = True
    wrist_lock: bool = True
    world_joint_visual_smoothing: bool = True

    @classmethod
    def raw(cls) -> "BodyPostChainConfig":
        return cls(
            temporal_smoothing=False,
            foot_lock=False,
            foot_pin=False,
            contact_splice=False,
            wrist_lock=False,
            world_joint_visual_smoothing=False,
        )

    @property
    def is_default(self) -> bool:
        return not self.bypassed_stages()

    @property
    def is_raw(self) -> bool:
        return set(self.bypassed_stages()) == set(POSTCHAIN_STAGE_ORDER)

    @property
    def raw_grounded_joints_sidecar(self) -> str:
        return RAW_GROUNDED_JOINTS_ARTIFACT if self.is_raw else ""

    def bypassed_stages(self) -> list[str]:
        values = {
            "temporal_smoothing": self.temporal_smoothing,
            "foot_lock": self.foot_lock,
            "foot_pin": self.foot_pin,
            "contact_splice": self.contact_splice,
            "wrist_lock": self.wrist_lock,
            "world_joint_visual_smoothing": self.world_joint_visual_smoothing,
        }
        return [stage for stage in POSTCHAIN_STAGE_ORDER if not bool(values[stage])]

    def to_artifact_dict(self, *, mode: str | None = None) -> dict[str, Any]:
        return {
            "mode": mode or ("raw" if self.is_raw else "default"),
            "temporal_smoothing": bool(self.temporal_smoothing),
            "foot_lock": bool(self.foot_lock),
            "foot_pin": bool(self.foot_pin),
            "contact_splice": bool(self.contact_splice),
            "wrist_lock": bool(self.wrist_lock),
            "world_joint_visual_smoothing": bool(self.world_joint_visual_smoothing),
            "raw_grounded_joints_sidecar": self.raw_grounded_joints_sidecar,
        }

    def bypass_summary(self) -> dict[str, Any] | None:
        stages = self.bypassed_stages()
        if not stages:
            return None
        return {
            "status": "postchain_bypassed",
            "stages": stages,
            "raw_grounded_joints_sidecar": self.raw_grounded_joints_sidecar,
        }
