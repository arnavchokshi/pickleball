# Paddle Pose Research Snapshot - 2026-06-28

Archived evidence snapshot. Current truth lives in `CAPABILITIES.md` and `BUILD_CHECKLIST.md`; do not use this file as an active runbook.

## Recommendation

Use a two-stage paddle stack:

1. **Find and track the paddle in video.** Prefer SAM 3 concept prompting for
   detection, segmentation, and identity tracking when the runtime/checkpoint
   license is approved. Fall back to DINO-X or Grounded-SAM-2 with SAM 2 for
   open-vocabulary detection plus video mask tracking.
2. **Estimate 6DoF from geometry, then gate hard.** Use CAD/reference-image
   object pose methods (FoundationPose, FoundPose, or GigaPose) to propose
   pose, then keep the existing PnP/IPPE, reprojection, ambiguity, UKF motion,
   grip prior, and rebound-consistency checks as fail-closed gates.

The current accepted-four artifacts are not yet gate-quality paddle pose:
`racket_candidates.json` is still box-derived, `racket_pose_preview.json` is
preview-only, and `virtual_world_paddle_preview.html` is for inspection only.
The next real RKT promotion blocker is not rendering; it is true paddle masks,
keypoints/corners, or CAD/reference pose evidence plus ground-truth evaluation.
`racket_pose_readiness.json` now makes that distinction machine-readable:
accepted-four clips report box-derived candidate frames, but
`true_corner_frame_count=0` and `reference_gt_frame_count=0`.

## Source Refresh - 2026-06-28

- SAM 3 is still the preferred detector/segmenter/tracker to try first because
  the official Meta repo describes one model for text/visual-prompt concept
  detection, segmentation, and video tracking.
- FoundationPose remains the best premium 6DoF default when RGB-D, CAD, or a
  small reference-image set is available; it is explicitly designed for
  model-based and model-free novel-object pose estimation/tracking.
- FoundPose and GigaPose remain strong RGB/CAD fallbacks: FoundPose onboards
  unseen objects from 3D models using foundation features, and GigaPose uses
  rendered CAD templates plus patch correspondences for fast RGB pose.
- None of those sources change the current local gate: accepted-four has no
  true paddle keypoints/masks, CAD/reference pose proposal, or reference/GT
  evaluation yet.

## Detection, Segmentation, and Tracking

- **Primary 2026 path:** Meta SAM 3. Meta describes SAM 3 as a unified model
  that detects, segments, and tracks visual concepts from text, exemplars, or
  visual prompts across images and video. That matches a "pickleball paddle"
  concept better than the older separate detector-plus-mask pipeline. Source:
  https://ai.meta.com/research/sam3/ and https://github.com/facebookresearch/sam3
- **Fallback path:** DINO-X/Grounded-SAM-2. The Grounded-SAM-2 repo supports
  Grounding DINO, Grounding DINO 1.5/1.6, Florence-2, DINO-X, and SAM 2 for
  grounding and tracking objects in videos. DINO-X adds text, visual, and
  customized prompts for open-world object detection. Sources:
  https://github.com/IDEA-Research/Grounded-SAM-2 and
  https://arxiv.org/abs/2411.14347
- **Video mask reason:** SAM 2 introduced streaming video memory and promptable
  image/video segmentation; Meta reports fewer interactions than prior video
  segmentation methods. This is useful for paddle masks once a paddle prompt or
  detection is seeded. Sources: https://ai.meta.com/research/sam2/ and
  https://arxiv.org/abs/2408.00714

## 6DoF Pose

- **FoundationPose:** best default for a premium GPU path when a paddle CAD
  model or small reference set is available. NVIDIA presents it as unified 6D
  pose estimation and tracking for novel objects, usable with CAD or reference
  images at test time. Sources:
  https://research.nvidia.com/publication/2024-06_foundationpose-unified-6d-pose-estimation-and-tracking-novel-objects
  and https://github.com/NVlabs/FoundationPose
- **GigaPose / FoundPose:** useful RGB/CAD alternatives. GigaPose is CAD-based
  and uses rendered templates plus patch correspondences; FoundPose targets
  unseen object pose from a single RGB image with foundation features and fast
  object onboarding from 3D models. Sources: https://nv-nguyen.github.io/gigapose/
  and https://arxiv.org/html/2311.18809v2
- **Deterministic geometry stays mandatory:** OpenCV `solvePnP` estimates pose
  from 3D object points and 2D image projections; IPPE is appropriate for
  planar objects with four or more correspondences. The repo already uses this
  correctly for explicit four-corner candidates, but box corners are not true
  paddle face corners. Sources:
  https://docs.opencv.org/3.4.20/d5/d1f/calib3d_solvePnP.html and
  https://github.com/tobycollins/IPPE
- **Benchmarking reference:** BOP remains the relevant 6DoF benchmark family,
  and its 2024/2025 materials track model-based and model-free 6D object pose
  progress. Source: https://bop.felk.cvut.cz/challenges/

## Pipeline Policy

- Do not promote `racket_pose_preview.json` to `racket_pose.json` when its
  source is `label_bbox:*` or `*:pnp_ippe_preview`.
- Treat box-derived candidate corners as a visualization seed only. A valid
  promotion needs at least one of:
  - true face-corner/keypoint labels or detector output,
  - CAD/reference pose proposal from FoundationPose/GigaPose/FoundPose,
  - ArUco/AprilTag or measured paddle ground truth for evaluation.
- Use `source_evidence_counts` in `racket_pose_readiness.json` as the promotion
  gate summary: `box_derived` is preview-only, `keypoint_or_mask` and
  `synthetic_or_cad` can satisfy true paddle evidence, and `reference_gt` is
  required before RKT can move from "pose present" to evaluated.
- Keep BODY and replay allowed to display preview paddles, but keep metrics
  and paddle-face angles gated until RKT evaluation passes.

## Historical Local-Work Notes; Cross-Check Active Docs First

- Keep generating `racket_candidate_overlay.mp4` and
  `virtual_world_paddle_preview.html` for human review.
- Keep readiness diagnostics current so every packet explicitly separates
  box-derived preview from true keypoint/mask, synthetic/CAD, and reference/GT
  pose evidence.
- Keep `racket_promotion_audit.json` current so every packet proves whether
  canonical `racket_pose.json` is absent, safe, or contaminated by box-derived
  or preview sources.
- Keep the global `racket_model_runtime_readiness.json` report current after
  model-manifest or CAD/reference-asset changes. It should remain CPU-only and
  must not claim GPU inference or runtime imports.
- Use `racket_model_adapters.py` as the fail-closed seam before any premium
  paddle runtime touches GPU code. It converts readiness into an adapter plan
  and raises if SAM 3, DINO-X/Grounded-SAM-2, FoundationPose, GigaPose, or
  FoundPose are requested before manifest, runtime, CAD/reference, and GT
  prerequisites are ready.
- Add data/eval manifest checks for paddle CAD/reference images, ArUco/AprilTag
  captures, face-corner labels, and held-out video clips.
- Add model-adapter seams for SAM 3, Grounded-SAM-2/DINO-X, FoundationPose,
  GigaPose, and FoundPose that fail closed when repos, checkpoints, licenses,
  or GPU runtime are missing.
