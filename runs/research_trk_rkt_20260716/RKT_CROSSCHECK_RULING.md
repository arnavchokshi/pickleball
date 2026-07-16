# RKT dual-survey cross-check ruling — 2026-07-16 (Track F manager)

Sources: rkt_survey_A_20260716/SURVEY.md x rkt_survey_B_20260716/SURVEY.md (independent codex
gpt-5.6-sol high lanes, sibling-fenced; A method-first, B data-first) + rkt_refute_20260716/
REFUTATION.md (2nd-vote adversarial verification). Pattern per
runs/research_eventdata_20260713/CROSSCHECK_RULING.md. VERIFIED=0.

STATUS: FINAL — refutation lane landed 2026-07-16 (C1/C3/C4/C5/C6/C10 CONFIRM, C2 REFUTE-speed,
C7 REFUTE-in-part, C8/C9 PARTIAL; full evidence rkt_refute_20260716/REFUTATION.md).

## Refutation results folded in (supersede any conflicting line below)

- **RacketVision bundle CONFIRMED** (counts, five-keypoint schema, RTMDet-M+RTMPose-M baseline,
  PCK/MPJPE incl. side-keypoint 64.8-80.1 weakness, YouTube provenance, MIT code/annotation
  card, no model card/license on weights). Nuance: HF "unsafe" flags = pickle-import scan
  (`__builtin__.getattr`); JFrog reports safe, VirusTotal 0/74 — a serialization warning, not
  malware proof. Weight sizes pinned: epoch_300.pth 411,293,859 B; best_PCK_epoch_90.pth
  106,524,759 B.
- **RACE-6D speed attribution REFUTED:** 76.7 BOP AR runs at **84.0 FPS** (RTX 3090, 640x480);
  16.6 FPS was the CRT-6D comparison row. Mean rotation 10.3° RGB / 8.0° RGB-D confirmed;
  Apache-2.0 code confirmed; NO checkpoints/releases confirmed (empty releases, no weights in
  tree). Consequence: RACE-6D is runtime-VIABLE if we ever train it — its blockers are now only
  missing weights + ambiguity-collapse, not speed.
- **TT4D 26.4±4.4° / 0.58±0.40 m/s CONFIRMED**; no public artifact found (bounded search).
- **Event-camera contact numbers CONFIRMED** (tennis <15mm on recovered contours, sunlight
  collapse 24/26→3/20; badminton 116/124, biases 1.84ms/+3.45mm/−1.92mm; LoA asymmetric
  −3.35..+10.24 / −10.63..+6.78 mm).
- **Sim2real numbers CONFIRMED** (Self6D chains exact; DOPE 77.00/66.64/62.94; MegaPose 88.6% =
  refiner-from-initial-pose only; ROCK 59.4 is the FIVE-object YCB-V subset, not all 21).
- **Metrology precedents CONFIRMED with limits:** Vicon 0.15mm/2mm = cited system capability;
  PhoCaL 0.20mm/0.38° = ICP-sim RMSE, not total GT uncertainty; Anipose >90% <1° = board-pose
  validation. Our rig must prove its OWN held-out error — precedents motivate, never substitute.
- **Generator corrections (C7):** BlenderProc GPL-3.0 + live motion-blur/rolling-shutter example
  CONFIRMED. **Kubric is ACTIVE** (not archived; commits 2026-05) — a stronger Apache fallback
  than surveyed. **Isaac Sim is split-licensed**: GitHub source Apache-2.0; Kit/assets under
  NVIDIA proprietary terms (not blanket Omniverse EULA).
- **License bundle:** FoundationPose non-commercial+RGB-D, FoundPose CC-BY-NC coarse-only,
  GigaPose MIT + small-segment failure, KV-Tracker Imperial non-commercial, GRAB NC,
  GraspXL CC-BY-NC — all CONFIRMED. Correction: **SAM-6D has a nested MIT license** (its
  Instance_Segmentation_Model) but no root/project-wide grant — still `unknown-needs-review`.
  IPPE BSD-3 + exactly-two-solutions CONFIRMED.
- **Blur direction (C9) PARTIAL with a new asset:** BlurHandNet + Human-from-Blur confirmed as
  described. BUT a released MIT generic rigid-object package EXISTS — **ShapeFromBlur**
  (rozumden/ShapeFromBlur): textured 3D shape + sub-frame motion of fast-moving objects from one
  blurred image. Only the sports-equipment-SPECIFIC absence is supportable. ADD ShapeFromBlur to
  Gap C (exposure-integrated planar render refinement) as prior art / candidate component to
  evaluate before building from scratch.
- **Sync math CONFIRMED (C10):** 0.5 frame @60fps = 8.33ms = 8.3-16.7cm at 10-20 m/s; ≤1ms is a
  conservative target (motion-only budget alone would allow 1.5-3.0ms; calibration/RS/labeling
  consume the rest). √2·σ/L angle formula valid first-order (σ≪L, independent endpoints):
  σ=5px → 10.13° @L=40px, 5.06° @L=80px.

## Convergent (adopt with confidence — independently fetched by both lanes)

1. **The product regime (<80px, motion-blurred, hand-occluded, near-planar, single elevated RGB
   camera) has NO published off-the-shelf solution and NO published evidence.** No paper/repo/
   checkpoint reports pose accuracy binned at small pixel extent + blur + edge-on. BOP results do
   not transfer; 2026 high-speed racket research switched sensors (event cameras + mocap) rather
   than solving single-RGB. This is a genuine build-our-own seat. [CORROBORATED x2]
2. **Synthetic-only training is UNSUPPORTED at face-angle p90 ≤5°; synthetic + small real
   fine-tune is plausible but unproven.** Strongest positive sim2real (DOPE sugar-box, MegaPose
   BOP transfer, ROCK) is household-object recall/AUC, never angular p90 on thin planar objects
   [PENDING C5]. Strongest negative (Self6D occlusion chain) shows large residual gaps.
   [CORROBORATED x2 at verdict level]
3. **RacketVision reality:** five 2D keypoints + box only — NOT metric 6DoF, NOT contact GT;
   single-frame RTMDet-M+RTMPose-M baseline; side keypoints (the face-width cue we need) are its
   WEAKEST points (PCK ~64.8-80.1 vs ~82-90 structural); MIT code/annotation card but YouTube
   broadcast provenance unresolved → data/weights `unknown-needs-review`; HF weights live but
   no model card + unsafe-pickle flags [PENDING C1]. It remains the ruled first challenger as a
   keypoint SOURCE feeding our own pose machinery, not a pose solution. [CORROBORATED x2]
4. **Rank-1 method route: keypoint/heatmap evidence → both-IPPE planar solutions → temporal
   ambiguity-preserving hypothesis graph (factor graph/Viterbi w/ wrist-grip, silhouette,
   velocity, ball-contact soft factors; missing/abstain node; never collapse to one mode
   pre-fusion).** Plausible for the interim ≤30° milestone; NOT supported to ≤5° by any published
   number — the 5° gate rides on our own GT + stratified measurement. [CORROBORATED x2]
5. **Render-and-compare / template methods (MegaPose, GigaPose, FoundPose, RefPose/Pos3R class)
   = offline oracles and proposal refiners only**: seconds-per-frame, no small/blur evidence,
   GigaPose has a documented small-segment failure mode [PENDING C8c]. Use as upper-bound probes
   on sharp large crops; if they fail there, the information isn't in the pixels. [CORROBORATED x2]
6. **Eliminated as product path:** FoundationPose (RGB-D + NVIDIA non-commercial license)
   [PENDING C8a]; SAM-6D (RGB-D, no license); category-level pose (RGB-D household datasets, no
   paddle category exists — revisit only after owner-instance proof); direct RGB rotation
   regression as primary (RACE-6D class mean rotation ~10.3° on EASIER objects → not gate-class;
   also collapses ambiguity) [PENDING C2]; trajectory/inverse-control as precision source (TT4D
   26.4±4.4° mean = interim-milestone class; keep as prior/hypothesis-selector) [PENDING C3].
   [CORROBORATED x2]
7. **Blur direction: blur-aware training + (later) exposure-integrated render refinement beats
   deblur-first.** Deblurring hallucinates edges — dangerous for a two-solution planar problem.
   (A-only citations BlurHandNet/Human-from-Blur [PENDING C9]; B independently reached the same
   posture via image-formation reasoning.) [CONVERGENT at posture level]
8. **GT capture is the binding bottleneck and must precede model benchmarking.** Marker/ChArUco
   multi-view rig with a mandatory held-out metrology gate: face-normal p90 ≤1-1.5°, point p90
   ≤1cm, sync ≤~1ms at impact; markers registered to exact CAD; static jig audit before/after;
   markers/extra cameras GT-only. Neither lane found ANY commercially usable dataset combining
   tiny blurred paddle 6DoF + impact geometry + synced contact GT. [CORROBORATED x2]
9. **Contact-point budget: BALL 3D accuracy dominates.** To leave margin under 3cm p90, the
   contact-window BALL point needs ~≤1cm p90 per face-relevant axis + face origin ≤1cm + normal
   ≤1-1.5° + timing ~≤1ms. Audio = time anchor only (no face location). The contact estimator
   must be validated by an oracle ladder (GT time/GT ball/GT face, then swap one input at a time)
   so the limiting sensor is identified before fusion work. [CORROBORATED x2]

## Single-source items sent to refutation (2nd vote)

- C1 RacketVision numbers/weights/provenance bundle; C2 RACE-6D (code live, Apache, NO ckpts,
  76.7 AR / 16.6 FPS / 10.3° mean rot); C3 TT4D 26.4±4.4°; C4 event-camera contact numbers
  (<15mm tennis w/ sunlight collapse; 3.45mm/1.84ms badminton dual-event); C5 sim2real numbers
  (Self6D/DOPE/MegaPose-ModelNet-refinement caveat/ROCK); C6 GT metrology precedents
  (Vicon/PhoCaL/Imitrob/Anipose); C7 BlenderProc GPL + rolling-shutter example, Kubric/Isaac
  posture; C8 license/elimination bundle (FoundationPose, FoundPose, GigaPose, KV-Tracker, GRAB/
  GraspXL, SAM-6D, IPPE BSD-3); C9 blur-aware precedents; C10 sync + keypoint→angle error math.

## Disagreements resolved by manager ruling

- **Minimum gate-credible GT rig.** A: rented 6-8-camera mocap ≥240Hz preferred, ≥4 synced
  high-speed cameras acceptable, two unsynchronized phones NOT gate-credible. B: three phones
  (product + two 120/240fps GT views, 60-100° convergence) + ChArUco volume sweeps + audio/LED
  sync + flat both-face decals + held-out jig metrology gate. RULING: **B's 3-phone rig is the
  owner ask, with A's discipline made binding** — the rig certifies NOTHING until it passes the
  held-out metrology gate (face-normal p90 ≤1-1.5°, point ≤1cm, sync residual ≤1ms, re-checked
  after remount/drift); if it fails, the session downgrades to training-evidence-only (still
  valuable for fine-tune). Mocap rental remains the escalation if the phone rig can't pass. This
  is consistent with both lanes: A's "not gate-credible" referred to UNVALIDATED phone captures.
- **Sync bar correction (B).** NS-02.1's "inter-camera sync ≤0.5 frame" is fine for CAL/BODY but
  INSUFFICIENT for RKT contact GT: 0.5 frame @60fps = 8.33ms = 8-17cm at 10-20 m/s [PENDING C10].
  RULING: RKT contact GT requires ≤1ms (≤0.25 of a 240fps frame) at impact; carry this into the
  NS-02 gold-capture checklist as an RKT-specific addendum (owner-facing; no North Star edit from
  this lane).
- **RACE-6D placement.** A ranks it #2 challenger; B eliminates direct regression as next primary.
  RULING: reconcile — RACE-6D is NOT a first-wave arm (no released checkpoints; direct-regression
  ambiguity collapse) but stays the named architecture-class challenger IF its checkpoints appear
  or we train it on our synthetic+real GT, with the required modification both lanes agree on:
  expose multi-mode/heatmap output rather than single rotation.
- **Synthetic generator.** Both rank BlenderProc first. GPL-3.0 is a TOOL license — generated
  images/labels are not GPL-encumbered; the obligation risk is only in distributing modified
  generator code. RULING: BlenderProc for the MVP synthetic set (owner CAD + measured camera
  match + real court plates + hand rig from owner GT trajectories), Kubric as the Apache-clean
  fallback; every third-party asset (HDRI, body/hand model, court plates) gets its own license
  row in the dataset ledger.

## What this changes vs the 2026-07-09 register

The register's NS-03.RKT sequence (GT capture → RacketVision zero-shot → fine-tune → both-IPPE →
cross-cue resolution) SURVIVES and is sharpened: (a) GT capture is confirmed as strictly first —
no candidate can be scored against the gates without it, and the rig now has a concrete
metrology-gated 3-phone spec; (b) RacketVision is a keypoint source, not a pose method — its
side-keypoint weakness is exactly our hard case; (c) both-IPPE retention (already wired,
evidence17) gets its consumer: the temporal hypothesis graph is the main build; (d) synthetic
pretraining enters as a supervised route with a falsifiable first experiment, not a platform
build-out; (e) GigaPose-class stays an offline oracle, demoted from "later candidate" to
"diagnostic probe"; (f) new named challengers RACE-6D (conditional) and KV-Tracker (R&D-only
license, architecture reference) enter the watch list.

## Where we build our own tech (RKT) — the gap map (union of both lanes)

A. Tiny blur-aware paddle evidence extractor (native-res temporal ROI, heatmaps + visibility +
   blur + uncertainty; synthetic pretrain + real fine-tune; never collapse heatmaps pre-PnP).
B. Ambiguity-preserving trajectory inference (two IPPE modes + learned proposals + missing node;
   wrist/silhouette/velocity/contact soft factors; posterior + abstention out).
C. Exposure-integrated planar render refinement (SE(3) spline across exposure/rolling shutter;
   must be allowed to return broad uncertainty).
D. Paddle-shape adapter for unknown paddles (owner exact-CAD first; category later, only after
   instance proof).
E. Continuous-time contact estimator + oracle-ladder evaluator (piezo/impact-tape/multiview GT;
   kill geometry-only route if GT-pose+GT-time+predicted-BALL misses 3cm — that isolates BALL).
F. Commercial-clean synthetic hand+paddle domain (BlenderProc, owner kinematics, licensed hand
   rig; stop-rule: kill scale-up if synthetic-only misses 30° on held-out real).

These six specs (full details in the two SURVEY.md files, reconciled here) are the novel-tech
program; nothing on the market replaces them.
