# w6_gate1b_knob_20260708 GPU Instrument Command

This is the follow-on one-clip BODY raw-postchain instrument run for
`wolverine_mixed_0200_mid_steep_corner`. It is an internal-val clip. Do not use
Outdoor/Indoor labels for this run.

## Required Environment

```bash
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
```

`--remote-host REMOTE_HOST_PLACEHOLDER` is intentionally literal here. Replace
`REMOTE_HOST_PLACEHOLDER` with the freshly refreshed SSH target only at execution
time. Do not omit `--remote-host`; fleet IPs recycle.

## Exact Command

```bash
MPLBACKEND=Agg PYTHONUNBUFFERED=1 .venv/bin/python scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --court-calibration eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json \
  --out runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain \
  --force \
  --body-schedule serial \
  --remote-host REMOTE_HOST_PLACEHOLDER \
  --remote-ssh-key ~/.ssh/google_compute_engine \
  --remote-repo /home/arnavchokshi/coldstart_20260706/repo \
  --remote-python /home/arnavchokshi/coldstart_20260706/body_runtime/body_venv/bin/python \
  --remote-fast-sam-python /home/arnavchokshi/coldstart_20260706/body_runtime/body_venv/bin/python \
  --remote-fast-sam-root /home/arnavchokshi/coldstart_20260706/body_runtime/Fast-SAM-3D-Body \
  --remote-lock-wait-timeout-s 3600 \
  --remote-command-timeout-s 21600 \
  --sam3d-body-input-size-px 384 \
  --sam3d-crop-bucket-sizes 8,16 \
  --sam3d-compile-warmup-buckets 8,16 \
  --serialize-tier2-mesh-vertices \
  --fetch-body-monoliths \
  --body-postchain raw \
  --no-body-temporal-smoothing \
  --no-body-foot-lock \
  --no-body-foot-pin \
  --no-body-contact-splice \
  --no-body-wrist-lock \
  --no-body-world-joint-visual-smoothing \
  --no-sam3d-wrist-bone-lock
```

## Expected Output Paths

Clip output directory:

```text
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner
```

Raw-persist sidecar required by the latent-smoothing acceptance harness:

```text
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_raw_grounded_joints.json
```

Bypass/runtime proof paths:

```text
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/pipeline_summary.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_stage_phase_timing.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/skeleton3d.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/contact_splice.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/remote_sam3d_tier2_dispatch_config.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/sam3d_tier2_config.json
```

Expected bypass keys and values:

```text
pipeline_summary.json -> stages[stage=="body"].metrics.postchain_bypassed_stages
body_stage_phase_timing.json -> postchain_bypasses.stages
skeleton3d.json -> provenance.body_postchain_bypass.stages
contact_splice.json -> summary.status == "bypassed"
remote_sam3d_tier2_dispatch_config.json -> optimization.body_postchain
sam3d_tier2_config.json -> optimization.body_postchain
```

Expected bypass stage list:

```json
[
  "temporal_smoothing",
  "foot_lock",
  "foot_pin",
  "contact_splice",
  "wrist_lock",
  "world_joint_visual_smoothing"
]
```

World/BODY outputs to preserve for the P2-2 comparison:

```text
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/skeleton3d.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/smpl_motion.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_mesh.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_mesh_index/body_mesh_index.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_mesh_index/body_mesh_faces.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_mesh_index/body_mesh_chunks/
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_joint_quality.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_full_clip_gate.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/body_grounding_quality.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/virtual_world.json
runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_wolverine_mixed_0200_raw_postchain/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json
```

## Raw Sidecar Field Mapping

The latent-smoothing acceptance harness from commit `2db0d1b4e` should consume
the raw sidecar at:

```text
body_raw_grounded_joints.json
```

Top-level keys:

| Key | Meaning for harness |
|---|---|
| `schema_version` | Raw sidecar schema version. Required value: `1`. |
| `artifact_type` | Required value: `racketsport_body_raw_grounded_joints`. |
| `source` | Raw extraction source. Expected: `worldhmr._ground_fast_sam_sample`. |
| `model` | BODY model/source namespace. Expected: `sam3dbody_world_joints`. |
| `fps` | Clip frame rate used for `t` alignment. |
| `world_frame` | Coordinate frame. Required value: `court_Z0`. |
| `grounding` | Grounding transform description. |
| `grounding_anchor_source` | Anchor source used before post-chain stages. |
| `postchain` | Serialized knob values; see nested mapping below. |
| `postchain_bypassed_stages` | Ordered list of disabled stages. Must match the six-stage list above for this run. |
| `joint_names` | Joint-name vector; index-aligned to each frame's `joints_world` and `joint_conf`. |
| `players` | Player records; join by `players[].id` and `players[].frames[].frame_idx`. |
| `summary` | Count sanity checks; see nested mapping below. |

`postchain` keys:

| Key | Expected value for this run |
|---|---|
| `mode` | `raw` |
| `temporal_smoothing` | `false` |
| `foot_lock` | `false` |
| `foot_pin` | `false` |
| `contact_splice` | `false` |
| `wrist_lock` | `false` |
| `world_joint_visual_smoothing` | `false` |
| `raw_grounded_joints_sidecar` | `body_raw_grounded_joints.json` |

`players[]` keys:

| Key | Meaning for harness |
|---|---|
| `id` | Stable player id. Use with `frame_idx` as the primary join key. |
| `frames` | Raw grounded BODY samples for that player. |

`players[].frames[]` keys:

| Key | Meaning for harness |
|---|---|
| `frame_idx` | Integer source frame index. Join key against `skeleton3d.json`, `smpl_motion.json`, and `body_mesh.json`. |
| `t` | Timestamp in seconds. Secondary alignment key; prefer `frame_idx` for exact joins. |
| `track_world_xy` | Raw track anchor `[x, y]` used by grounding before BODY post-chain stages. |
| `transl_world` | Raw grounded root translation `[x, y, z]` before temporal smoothing, foot-lock, foot-pin, contact-splice, wrist-lock, or visual smoothing. |
| `joints_world` | Raw grounded joint coordinates, shape `[joint_count][3]`, in `court_Z0`. This is the primary raw BODY joint tensor for decode(emit). |
| `joint_conf` | Confidence vector index-aligned to `joint_names` and `joints_world`. |
| `grounding_anchor` | Per-frame grounding anchor tag when present; empty string when absent. |

`summary` keys:

| Key | Meaning for harness |
|---|---|
| `player_count` | Number of player records in `players`. |
| `frame_count` | Number of unique `frame_idx` values in the sidecar. |
| `sample_count` | Number of player-frame samples. |
| `joint_count_min` | Minimum raw joint count across samples. |
| `joint_count_max` | Maximum raw joint count across samples. |

Harness join rule:

```text
raw sample key = (players[].id, players[].frames[].frame_idx)
raw joints = players[].frames[].joints_world
raw translation = players[].frames[].transl_world
joint order = joint_names
world frame = court_Z0
```

For GATE 1b, reject the run if any of these are missing:

```text
body_raw_grounded_joints.json
body_raw_grounded_joints.json -> postchain.mode == "raw"
body_raw_grounded_joints.json -> postchain_bypassed_stages includes all six expected stages
body_stage_phase_timing.json -> postchain_bypasses.stages includes all six expected stages
skeleton3d.json -> provenance.body_postchain_bypass.strict_mode_loud == true
remote_sam3d_tier2_dispatch_config.json -> optimization.body_postchain.raw_grounded_joints_sidecar == "body_raw_grounded_joints.json"
```
