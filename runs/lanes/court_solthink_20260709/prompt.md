# COURT AUTO-FIND SYSTEM DESIGN — deep ideation lane (gpt-5.6-sol, xhigh)

You are the deep-thinking design partner for DinkVision's court auto-detection system.
The owner has ordered a dedicated push: court auto-find is our weakest pillar and we now
have real supervision to attack it. Your job: think VERY deeply and produce the best
possible system design + experiment ladder. You have repo read access and web search.

## HARD RULES
- Write ONLY inside runs/lanes/court_solthink_20260709/. No other file changes, no git.
- Protected eval clips (eval_clips/**, Burlington/Wolverine/Outdoor/Indoor) may be read
  for context but are EVAL-ONLY; never propose training on them.
- Honest reporting: separate established fact / literature claim / your hypothesis.

## PRODUCT + GATE
iPhone single-camera pickleball recording -> server GPU (H100) pipeline -> metric 3D
court + players + ball. Court calibration is the ONLY hard dependency of the whole
pipeline (stage 2 of 19). Today v1 ships device profiles + guided human corner-tap
confirm. The owner wants true AUTO-find with best possible accuracy.
Named gate (NORTH_STAR_ROADMAP.md, CAL row): owner-viewpoint PCK@5px >= 0.95 over the
15 metric court keypoints + net-height error <= 2cm + reprojection/distortion/handheld
gates. Typical user camera: phone on tripod or lightly handheld, behind/beside court,
often STEEP/LOW angle. Common hard cases: adjacent identical courts, tennis-overlay
courts (pickleball lines painted inside tennis courts), partial occlusion by players,
varied surface colors/lighting. Owner ANGLE POLICY: courts not fully visible / camera
too low are REJECTED at intake (out of scope for the gate).

## CURRENT TRUTH (measured, do not re-derive)
- Best learned candidate court_unet_v2 (24M UNet heatmap, 15 channels, trained ONLY on
  our synthetic generator): synthetic val 1.90px / PCK@5 0.751, but on real owner GT:
  PCK@5 = 0.0, mean error ~976px (runs/lanes/w7_courtkpingest_20260709/owner_gate_rescore_summary_r2.json).
  Synthetic-only training failed decisively TWICE (random-init and ImageNet-init).
  STANDING KILL RULE: no third synthetic-ONLY retrain. Synthetic+real mixing is allowed.
- Geometric line-solver (cal_geo/GEO-r3): pool of homography candidates NEVER contains
  the true court on the steep/low protected clips (0/16); adjacent-court lock-on;
  Outdoor (high/long view) got to 4.4px though. Line detection is not the bottleneck;
  line-to-court-semantics ASSIGNMENT is.
- Fused (neural seed + geo fallback) on protected clips: Burlington 310px / Wolverine
  207px vs <=200px bars — FAIL (runs/lanes/calv1_fuseval2_20260708/).
- Impact: bad calibration poisons placement, in/out, BODY grounding, fusion — everything.

## REAL SUPERVISION INVENTORY (the new unlock)
1. Owner CVAT FULL15 labels, corrected r2 (runs/lanes/w7_courtkpingest_20260709/gt_roots/corrected_r2/):
   5 frames / 3 static-tripod YouTube sources: 73VurrTKCZ8 f3808; HyUqT7zFiwk f10195,f14564;
   zwCtH_i1_S4 f3636,f6363. THE named owner gate scores against these. 3 more sources
   were owner-REJECTED for angle policy (Ezz6HDNHlnk, _L0HVmAlCQI, wBu8bC4OfUY) — their
   partial labels exist and could still be TRAIN data.
2. Harvest court calibrations (data/online_harvest_20260706/court_calibrations/):
   73VurrTKCZ8 solved at 2.93px median (manual_bar). HyUqT7zFiwk and zwCtH_i1_S4 FAILED
   the bar — but that solve used the OLD (pre-correction) labels; the corrected task-18
   relabels landed later and were never re-solved. Re-solving with r2 labels likely
   unlocks 3 calibrated static-camera sources.
3. 40 rally mp4s across 6 sources (data/online_harvest_20260706/rallies/), static tripod
   cameras -> one calibration per source projects all 15 keypoints onto EVERY frame =
   thousands of real pseudo-labeled frames (but few distinct viewpoints — that is the
   diversity bottleneck to reason about).
4. External assets already on disk (models/checkpoints/court_external/): yastrebksv
   TennisCourtDetector weights (14kp tennis, trained on 8.8k real tennis frames),
   PnLCalib SV_kp + SV_lines weights, DeepLSD, ScaleLSD, resnet34. LICENSES.md there.
5. Roboflow "pickle-court-keypoints" dataset (~1,135 real pickleball images) known to
   exist on Roboflow Universe; may already be on our GPU snapshot as "roboflow corpus"
   (110,749 files) — verify + license-check before relying on it.
6. Synthetic generator (scripts/racketsport/generate_synthetic_court_keypoints.py):
   7 families incl tennis-overlay dual line-family masks, subpixel-exact, 64-87 imgs/s.
7. Protected eval clips with metric15 court GT (eval_clips/**): EVAL-ONLY.
8. Product-side human tap confirmations accumulate over time (owner clips: zwCtH tapped
   run, IMG_1605 tennis-overlay portrait, IMG_9545 fresh) — a future data flywheel.
9. Device/lens distortion profiles + ChArUco calibration tools exist and work.

## KEY CODE TO READ (ground your design in what exists)
- scripts/racketsport/train_court_keypoint_heatmap.py (current trainer, streams synthetic)
- scripts/racketsport/evaluate_court_keypoint_owner_gate.py (THE gate harness; raw vs
  aggregated_static_camera_median modes)
- scripts/racketsport/calibrate_harvest_courts.py (solver + manual_bar/auto_bar thresholds)
- threed/racketsport/court_calibration_metric15.py (metric solver, MIN 6 correspondences)
- scripts/racketsport/generate_synthetic_court_keypoints.py
- runs/lanes/court_autofind_20260705/DESIGN.md (old master design)
- runs/lanes/calv1_geor3_20260708/ + calv1_seeddiag_20260708/ (failure forensics)
- NORTH_STAR_ROADMAP.md sections 2.2 (CAL row), 3, NS-03.CAL.

## CONSTRAINTS
- Commercial-clean licenses for anything in the shipped stack (research-only diagnostics OK).
- AnyCalib only as an import prior (standing ruling). No SfM/DPVO as court authority.
- Eval integrity: the 5 owner-gate frames and their sources must stay source-disjoint
  from training, or you must design an explicit LoSO protocol — think hard about how to
  train on harvest pseudo-labels WITHOUT contaminating the owner gate (this is subtle:
  the 3 calibratable sources ARE the gate sources). Propose a defensible protocol.
- Server-side deep pass: latency secondary (seconds are fine), accuracy first.
- Product needs calibrated UNCERTAINTY + abstention: fall back to guided taps when unsure.

## WHAT I WANT FROM YOU
Write runs/lanes/court_solthink_20260709/DESIGN_PROPOSAL.md with:
1. RANKED system architectures (>=3 candidates, with why #1 wins): consider keypoint
   heatmaps + homography snap-to-template; line/segmentation + semantic assignment;
   tennis-transfer fine-tune; foundation-feature correspondence; temporal aggregation
   over the video (static camera = huge prior); net-post anchors; court-surface color
   segmentation; hybrid neural-seed + geometric-refine; test-time optimization per clip.
   Exploit VIDEO structure: we calibrate a CLIP, not a frame.
2. DATA RECIPE: exact mix (harvest pseudo-labels, owner labels, rejected-angle labels,
   tennis data via TennisCourtDetector dataset/weights, Roboflow sets + licenses,
   synthetic + domain randomization, style/appearance augmentation to fake viewpoint
   diversity); how many real viewpoints do we actually need? Cheapest path to +20
   diverse calibrated viewpoints (e.g. auto-calibrate MORE YouTube tripod sources with
   the geometric solver on easy/high views + human spot-check?).
3. EVAL PROTOCOL that survives an adversarial review: source-disjoint splits, LoSO,
   what number PCK@5 we report, how we use protected clips (eval-only), abstention metrics.
4. EXPERIMENT LADDER for the next 24-48h on 1 H100: ordered rungs, each with train
   data, steps, expected outcome, KILL criterion, GPU-hours. First rungs should be
   cheap decisive probes (e.g. zero-shot TennisCourtDetector on our frames; fine-tune
   it 30min on pseudo-labels; PnLCalib zero-shot).
5. FAILURE-MODE DESIGN: adjacent identical courts, tennis overlay, occlusion, steep
   angle, handheld drift, night/indoor lighting. Which architecture component handles each?
6. UNCERTAINTY + PRODUCT INTEGRATION: per-keypoint confidence -> solver covariance ->
   trust band; when to abstain to guided taps; how auto-find seeds the confirm UI.
7. NUMBERED DECISION ASKS for the manager (things only the manager/owner can rule on).
Use web search to check any SOTA claim you rely on (PnLCalib/TVCalib/TennisCourtDetector
accuracies, Roboflow dataset licenses, newer 2025-2026 field-registration work).
Be exhaustive; this is the thinking document the build wave executes against.
