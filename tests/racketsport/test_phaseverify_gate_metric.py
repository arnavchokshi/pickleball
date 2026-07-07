from __future__ import annotations

from tests.racketsport.test_foot_contact import JOINT_NAMES_65, _frame
from threed.racketsport import worldhmr
from threed.racketsport.body_grounding_quality import (
    DEFAULT_MAX_FOOT_SLIDE_M,
    build_body_grounding_quality,
)


def test_phaseverify_rejected_overthreshold_contact_still_blocks_grounding_gate() -> None:
    frames = [_frame(idx, left_x=x_m, left_z=0.0) for idx, x_m in enumerate([0.00, 0.01, 0.02, 0.03, 0.04])]
    skeleton3d = {
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "joint_names": list(JOINT_NAMES_65),
        "provenance": {"refined_stance_phase_lock": {"source": "phaseverify_synthetic"}},
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                        "transl_world": [0.0, 0.0, 0.0],
                        "track_world_xy": [0.0, 0.0],
                    }
                    for frame in frames
                ],
            }
        ],
    }

    metrics, gate_stream = worldhmr._contact_gate_stream_for_skeleton3d(
        skeleton3d,
        clip="phaseverify_overthreshold",
        threshold_m=DEFAULT_MAX_FOOT_SLIDE_M,
    )
    max_slide_m = max(
        (float(row.get("slide_mm", 0.0)) / 1000.0 for row in metrics.get("phase_metrics", [])),
        default=0.0,
    )
    quality = build_body_grounding_quality(
        clip="phaseverify_overthreshold",
        grounding_metrics={**metrics, "max_foot_lock_slide_m": max_slide_m, "foot_lock_gate_stream": gate_stream},
    )

    assert gate_stream["summary"]["phases_over_threshold"][0]["slide_m"] == 0.04
    assert quality["foot_slide_gate"]["passed"] is False
