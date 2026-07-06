# SOTA research fan-out — 2026-07-05

Four adversarially-verified web-research reports (Sonnet agent fan-out: 5-6 search angles -> 14+ primary-source deep-reads -> completeness critic -> gap-fill + 2-vote refute attempts -> synthesis; ~28 agents each).
Feeds NORTH_STAR_ROADMAP.md. Status: research evidence only — ran_not_verified, no promotion claims.

- `ball_report.md` — Ball detection, tracking, and monocular 3D trajectory reconstruction for racket sports (pickleball) (20 key facts, 24 sources)
- `body_report.md` — Temporally-stable, world-grounded 3D human mesh recovery from monocular sports video (eliminating jitter and foot-sliding) (23 key facts, 20 sources)
- `paddle_report.md` — High-fidelity 6-DOF paddle/racket pose estimation from monocular sports video (+ hand-object interaction) (19 key facts, 21 sources)
- `product_report.md` — Competitive landscape (pb.vision, SwingVision, ...) + automated coaching from 3D reconstructions + cloud video pipeline economics (23 key facts, 27 sources)

## PASS 2 — citation-graph deep dive (2026-07-06, 159 agents)
Triggered by owner catching a missed sub-task (RacketVision TrajPred). Method: deep-read every pass-1 seed, enumerate its sub-tasks/baselines + backward refs + forward successors, dedup vs pass-1, deep-read the NEW frontier + a 6-angle cutting-edge recency sweep per domain, adversarial-verify, synthesize missed_in_pass1 + new_adoptions mapped to task IDs.
- `pass2_ball_report.md`, `pass2_body_report.md`, `pass2_paddle_report.md`, `pass2_product_report.md`
Consolidated into NORTH_STAR_ROADMAP.md PART II-B (corrections + ranked new adoptions).
Headlines: pickleball aerodynamics DO exist (Lindsey/Steyn — corrects P0-7); SAM-3D latent-smoothing published blueprint (2512.21573 = P2-2); RacketVision multi-task (TrajPred cross-attn + RTMPose keypoints); SOMA-X MHR<->SMPL-X; RF-DETR; Grounding DINO zero-shot paddle; Uplifting-TT lift; LATTE-MV data engine; Human3R/DuoMo/JOSH3R body challengers; OnePoseViaGen/RGBTrack/Image-as-IMU paddle; CoachMe/BioCoach coaching; AnyCalib/BroadTrack court.


## PASS 3 — court/net + global fusion + production (2026-07-06, 87 agents)
Owner flagged court calibration + the combine-everything (fusion) pillar as under-researched. 3 angle-based deep dives:
- `pass3_court_net_report.md` — NO pickleball court/net prior art; net 3D geometry = biggest open gap in the field; PnLCalib points+lines lineage, PoseGravity (ARKit-gravity closed-form pose, BSD-3), 3D catenary net-sag (ICRA'19), BroadTrack temporal. Sobering: SOTA 1.4-4.5px on EASIER domains vs our 19.8px — pilot before port.
- `pass3_fusion_report.md` — THE missing pillar: JOSH (ICLR'26) contact-coupled joint optimization (per-subsystem outputs as init -> joint opt with contact residual; WA-MPJPE 314->120mm, floating 9->2.9%); still beats feed-forward on accuracy; ARKit removes most SLAM apparatus (feed gravity/pose into GVHMR). No system fuses ball+body+paddle+court for any racket sport — we'd be first.
- `pass3_crosspillar_report.md` — production: provable-bound tracker failure detection (auto-QA), hybrid physics+NN trajectory (pickleball split, ~90 traj), ForeHOI (feedforward held-object 6DoF, code), UniCon3R/SHOW/VGGT/Depth-Anything-3 backbones, Synthesizing-the-Expert coaching trust. CORRECTIONS: arXiv 2409.19000 = SIMULATION (Cd~0.6 borrowed) NOT measured — keep separate from Lindsey/Steyn; US11045705B2/US12478848B2 = Nex Team/HomeCourt NOT pb.vision.
Consolidated into NORTH_STAR PART II-C + new PHASE F (global fusion).
