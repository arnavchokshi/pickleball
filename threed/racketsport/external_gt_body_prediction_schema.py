"""Real joint-name order for Fast-SAM-3D-Body's raw ``pred_keypoints_3d`` output.

`threed.racketsport.worldhmr`/`threed.racketsport.hmr_deep` carry this project's real
BODY-stage predictions straight through from the Fast-SAM-3D-Body subprocess into
`smpl_motion.json`'s ``players[].frames[].joints_world`` **without attaching a top-level
`joint_names` field** (confirmed empirically: a real `smpl_motion.json` produced by this
lane's own inference run has keys
``['fps', 'mesh_faces', 'model', 'players', 'schema_version', 'world_frame']`` -- no
`joint_names`). This means
`threed.racketsport.eval.body_gate_report._payload_joint_names` returns ``[]`` for
`smpl_motion.json`, and `_joint_errors`'s name-keyed matching (the F1 fix) silently falls
back to legacy positional matching for this producer. That fallback is *wrong* here: the
70-joint MHR ("Momentum Human Rig") layout does **not** start with a COCO-17-style
ordering (wrists sit at indices 41/62, not 9/10; hips sit at 9/10, not 11/12) -- positional
alignment against a 12-joint, COCO-order GT array would silently pair unrelated joints.

This module records the real, verified MHR-70 joint order so external-GT scoring
(`threed.racketsport.external_gt_alignment`) can do genuine name-keyed selection of the
`threed.racketsport.external_gt_aspset510.SHARED_CORE_JOINT_NAMES` subset out of a raw
70-joint prediction array, instead of relying on a same-order assumption.

**Provenance (verified, not guessed):** copied programmatically from
``sam_3d_body/metadata/mhr70.py``'s ``pose_info["original_keypoint_info"]`` in the real
Fast-SAM-3D-Body repository this project's BODY stage subprocess actually runs
(`/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body` on the A100 runtime host; Meta
Platforms, Inc. copyright header) -- re-derived with
``[pose_info["original_keypoint_info"][i] for i in range(70)]`` against the live file, not
transcribed from memory or from a different/assumed convention. The first 17 entries are
byte-for-byte the same 17 names as `threed.racketsport.eval.body_gate_report.
CORE_BODY_JOINT_NAMES` / `threed.racketsport.joint_schema.BODY_17_JOINT_NAMES`, just in a
different order (MHR is body-part-grouped: face, then shoulders/elbows, then
hips/knees/ankles/feet, then fingers, then wrists near the fingers they belong to, then a
few extra biomechanical landmarks and neck).
"""

from __future__ import annotations

MHR70_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe_tip",
    "left_small_toe_tip",
    "left_heel",
    "right_big_toe_tip",
    "right_small_toe_tip",
    "right_heel",
    "right_thumb_tip",
    "right_thumb_first_joint",
    "right_thumb_second_joint",
    "right_thumb_third_joint",
    "right_index_tip",
    "right_index_first_joint",
    "right_index_second_joint",
    "right_index_third_joint",
    "right_middle_tip",
    "right_middle_first_joint",
    "right_middle_second_joint",
    "right_middle_third_joint",
    "right_ring_tip",
    "right_ring_first_joint",
    "right_ring_second_joint",
    "right_ring_third_joint",
    "right_pinky_tip",
    "right_pinky_first_joint",
    "right_pinky_second_joint",
    "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip",
    "left_thumb_first_joint",
    "left_thumb_second_joint",
    "left_thumb_third_joint",
    "left_index_tip",
    "left_index_first_joint",
    "left_index_second_joint",
    "left_index_third_joint",
    "left_middle_tip",
    "left_middle_first_joint",
    "left_middle_second_joint",
    "left_middle_third_joint",
    "left_ring_tip",
    "left_ring_first_joint",
    "left_ring_second_joint",
    "left_ring_third_joint",
    "left_pinky_tip",
    "left_pinky_first_joint",
    "left_pinky_second_joint",
    "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
)

assert len(MHR70_JOINT_NAMES) == 70, "MHR70_JOINT_NAMES must have exactly 70 entries"
assert len(set(MHR70_JOINT_NAMES)) == 70, "MHR70_JOINT_NAMES must have no duplicate names"

__all__ = ["MHR70_JOINT_NAMES"]
