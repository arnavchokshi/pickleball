# Shot Classification External Eval - 2026-06-28

## Status

The current shot path is a scaffold/transfer baseline for review overlays, not a trained pickleball classifier. It should not be treated as SHOT-1 complete or as a claimable shot-classification product feature.

Canonical labels:

`serve`, `fh_shot`, `bh_shot`, `fh_drive`, `bh_drive`, `dink`, `lob`, `overhead`, `third_shot_drop`, `reset_block`

`fh_shot` and `bh_shot` are abstract side labels. They are allowed when evidence supports forehand/backhand side but not a specific subtype. Specific labels require stronger body, ball, court, contact, and ideally racket evidence.

## External Score Snapshot

### THETIS Pose Eval

Artifact directory: `runs/shot_external_eval/thetis_pose_eval_60/`

Source: THETIS tennis action clips, sampled at 5 clips each from 12 classes and collapsed to `fh_shot`, `bh_shot`, `serve`, and `overhead` families. Pose came from YOLO pose extraction. No pickleball ball/court/racket evidence was available, so this tests the current semantic-pose fallback rather than the full intended pipeline.

| metric | value |
|---|---:|
| samples | 60 |
| family accuracy | 31/60 = 51.7% |
| top-2 family accuracy | 40/60 = 66.7% |
| `bh_shot` | 18/20 = 90.0% |
| `fh_shot` | 13/20 = 65.0% |
| `serve` | 0/15 = 0.0% |
| `overhead` | 0/5 = 0.0% |

Confusion summary: every serve and overhead was mapped to `fh_shot` or `bh_shot`. This proves the current baseline has no reliable serve/overhead phase concept.

### OpenSportsLab Tennis Localization Eval

Artifact directory: `runs/shot_external_eval/opensportslab_tennis_eval_100/`

Source: OpenSportsLab tennis localization events, excluding bounce events and mapping labels to broad `serve` vs `swing`. This checks whether the current event-window baseline recognizes serves versus rally swings.

| metric | value |
|---|---:|
| samples | 100 |
| broad family accuracy | 81/100 = 81.0% |
| `swing` | 81/81 = 100.0% |
| `serve` | 0/19 = 0.0% |

The 81% headline is misleading: the classifier labeled every sampled event as a swing. It is useful for review coverage, not for serve detection.

### Local Pickleball Review Coverage

Current regenerated accepted-four outputs have 34/34 non-unknown shots after the abstract fallback change:

| clip | shots | unknown | label counts |
|---|---:|---:|---|
| Burlington | 9 | 0 | `fh_shot`: 3, `fh_drive`: 1, `reset_block`: 3, `bh_shot`: 1, `third_shot_drop`: 1 |
| Indoor | 10 | 0 | `fh_shot`: 4, `third_shot_drop`: 5, `bh_shot`: 1 |
| Outdoor | 9 | 0 | `fh_shot`: 1, `fh_drive`: 8 |
| Wolverine | 6 | 0 | `bh_shot`: 2, `fh_shot`: 2, `reset_block`: 1, `bh_drive`: 1 |

This is coverage, not accuracy. The current local path uses ball/track image-side fallbacks heavily, and many labels are abstract or low-confidence.

## Root Causes

1. The active baseline is not a trained sequence classifier. It emits heuristic or transfer labels.
2. Local BODY skeletons use generic names such as `sam3dbody_joint_###`, while the shot baseline expects semantic names such as `left_wrist`, `right_wrist`, `left_shoulder`, and `right_shoulder`; therefore the joint-aware path is mostly unused on local clips.
3. The baseline often falls back to ball position versus player bbox/image center. That can guess side, but it cannot reliably classify dink, drop, reset, lob, serve, or overhead.
4. Contact windows often lack stable `player_id`, forcing brittle hitter attribution from nearest box/ball evidence.
5. Ball inflection matching is sparse and approximate, and the current path does not consume audio onsets, wrist velocity peaks, racket candidates, court zones, net plane, or virtual-world trajectory as a single learned feature window.
6. Serve and overhead need explicit phase/event modeling over a longer pre-contact window. Single-frame side heuristics will keep collapsing them to FH/BH swings.

## Recommended Build Path

1. Add a canonical event-level shot evaluator: match predictions to labels by time/frame, report coverage, unknown/gated rate, accuracy, macro-F1, top-2 accuracy, per-class precision/recall/F1, confusion, and calibration.
2. Add a SAM-3D-Body semantic joint adapter or emit named joints from BODY so wrists, elbows, shoulders, hips, knees, and ankles are available to shot classification.
3. Build `data/pb_shots/` from reviewed pickleball contact windows with truth labels, abstraction labels, player id, contact frame/time, and source artifact references.
4. Train a first H100 baseline over fixed windows: pose sequence + velocities + court zone + ball pre/post trajectory + contact/audio confidence + player track features. Use a small TCN/GRU/Transformer and PoseConv3D/PoseC3D before bigger fusion.
5. Add a Stage-0 phase head for `serve`, `overhead_candidate`, `normal_hit`, and `unknown`, using longer serve windows around `[-1.2s, +0.4s]`.
6. Add BST-style pose+ball/player fusion only after the pose, ball, court-zone, and player tensors are clean enough to evaluate.
7. Keep the abstraction policy: exact class only above calibrated confidence; otherwise emit `fh_shot`/`bh_shot` when side is supported; only emit `unknown` when evidence is genuinely missing or gated.

## Sources Used For Model/Data Direction

- PoseC3D paper: https://arxiv.org/abs/2104.13586
- MMAction2 PoseC3D docs/configs: https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d
- BST paper: https://arxiv.org/html/2502.21085v3
- THETIS dataset: https://github.com/THETIS-dataset/dataset
- OpenSportsLab tennis localization dataset: https://huggingface.co/datasets/OpenSportsLab/soccernetpro-localization-tennis
