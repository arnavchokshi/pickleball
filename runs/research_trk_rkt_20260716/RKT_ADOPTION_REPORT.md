# ADOPTION REPORT — NS-03.RKT paddle 6DoF research (Track F, 2026-07-16)

Method: dual independent codex gpt-5.6-sol surveys (A method-first, B data-first,
sibling-fenced) + manager cross-check + 2-vote adversarial refutation (rkt_refute_20260716).
Research-only: VERIFIED=0; published numbers are motivation, never pickleball accuracy.
Standing rules intact: rectangle IoU never promotes as 6DoF; both IPPE poses survive to fusion.

STATUS: FINAL — 2-vote refutation complete 2026-07-16 (C1/C3/C4/C5/C6/C10 CONFIRM, C2 speed
REFUTED-corrected, C7 REFUTED-in-part, C8/C9 PARTIAL; corrections folded below and in
RKT_CROSSCHECK_RULING.md "Refutation results" section, which supersedes on any conflict).

## Headline

The market cannot solve our regime. Both surveys independently concluded that NO published
method, dataset, or checkpoint addresses <80px, motion-blurred, hand-occluded, near-planar
paddle 6DoF from one elevated consumer RGB camera — the 2026 frontier for fast racket sensing
moved to event cameras + mocap instead [CORROBORATED x2]. NS-03.RKT is therefore a
build-our-own program with off-the-shelf parts only at the EVIDENCE layer (keypoints) and the
ORACLE layer (offline render-and-compare probes). The binding first move is unchanged and
sharpened: gate-credible owner GT before any model benchmarking.

## Ranked adoption list

| Rank | Item | Status | License posture | Why / evidence |
|---:|---|---|---|---|
| 1 | **Owner GT capture, metrology-gated 3-phone rig** | FIRST — blocks everything | owned data = commercial-clean | [CORROBORATED x2] No usable dataset exists combining tiny/blurred paddle 6DoF + contact GT. Rig: product phone untouched + two 120/240fps GT phones (60-100° convergence), ChArUco volume sweeps, flat both-face decals registered to exact CAD, audio/LED sync, held-out jig audit. GATE: face-normal p90 ≤1-1.5°, point ≤1cm, sync ≤1ms at impact — else the session is training-evidence only. Mocap rental is the escalation path. [refute-CONFIRMED: sync math exact (8.33ms @0.5-frame/60fps = 8-17cm); metrology precedents confirmed w/ limits — the rig must prove its OWN held-out error] |
| 2 | **RacketVision 5-keypoint zero-shot → pickleball fine-tune** (already-ruled challenger) | Tier-1 benchmark arm | code+annotations MIT; weights no-card/unsafe-pickles; YouTube video provenance `unknown-needs-review` → R&D-only posture for weights/data | [CORROBORATED x2] It is a 2D keypoint SOURCE, not a pose method (no 6DoF/intrinsics/contact). Its side keypoints — our face-width cue — are its weakest (PCK ~64.8-80.1). Keypoint-error math says ~5px error supports ~30°-class at 40-80px, not 5° (formula refute-confirmed: 10.13°@40px, 5.06°@80px). [refute-CONFIRMED C1; HF "unsafe" = pickle-import warning, JFrog safe/VT 0/74] |
| 3 | **Both-IPPE temporal hypothesis graph** (build ours) | core build, consumes evidence17 wiring | ours; IPPE ref impl BSD-3 (refute-confirmed, returns exactly two solutions) | [CORROBORATED x2] Rank-1 method route: heatmaps → two IPPE solutions + missing/abstain node → factor-graph/Viterbi with wrist-grip, silhouette, velocity, ball-contact soft factors; posterior + abstention out. Plausible to the ≤30° interim milestone; ≤5° rides on GT + strata, no published support either way. |
| 4 | **Synthetic owner-CAD pretraining (BlenderProc MVP)** | falsifiable experiment, not a platform | BlenderProc GPL-3.0 (tool; outputs unencumbered); assets each reviewed; Kubric (Apache, refute-corrected: ACTIVELY maintained) strong fallback; Isaac Sim source now Apache w/ proprietary Kit/assets | [CORROBORATED x2 + refute-CONFIRMED numbers] Synthetic-ONLY is unsupported at ≤5° (positives are household-object recall: DOPE 77.0 AUC, ROCK 59.4 on a 5-object subset; negative Self6D 40.1→58.9 vs 86.9 real). Synthetic + small-real is plausible-unproven → run the 4-way ablation (real-small / synth-only / synth→real / mixed) against one frozen real split. Kill scale-up if synth-only misses 30° on held-out real. Must render: exact CAD, measured intrinsics/RS/exposure (BlenderProc rolling-shutter example confirmed live), real court plates, hand rig from owner GT trajectories, both face-normal signs. |
| 5 | **Offline CAD oracle probes: MegaPose (Apache code) + GigaPose (MIT code)** | diagnostic only, never product path | weights/template data `unknown-needs-review`; runtime seconds/frame | [CORROBORATED x2 + refute-CONFIRMED] Purpose: measure whether the INFORMATION exists in sharp large crops (oracle boxes/masks). GigaPose's small-segment failure is documented in its own paper. If oracles fail on sharp crops, no per-frame method will save the regime — the temporal graph carries it. |
| 6 | **RACE-6D** (exact-CAD RT-DETR-lineage joint det+pose) | conditional watchlist — UPGRADED runtime | Apache-2.0 code; NO released checkpoints [refute-CONFIRMED] | [refute-CORRECTED] Speed attribution was wrong in survey A: RACE-6D runs **84.0 FPS** (16.6 was CRT-6D) — runtime-viable at 60fps if we ever train it. Still not first-wave: no ckpts + direct rotation regression collapses ambiguity (~10.3° mean rotation on EASIER objects). Challenger iff trained on our synthetic+real GT with multi-mode output exposed. |
| 7 | **Contact estimator + oracle ladder** | after face-angle evidence exists | ours | [CORROBORATED x2 + refute-CONFIRMED C4] BALL 3D dominates the 3cm budget (needs ~≤1cm p90 per face axis + ≤1-1.5° normal + ~≤1ms timing). Event-camera precedents confirm mm-scale contact is physically recoverable at high temporal resolution (badminton 116/124 @ ~3.5mm bias) but with specialized sensors. Oracle ladder isolates the limiting input: (GT time, GT ball, GT face) then swap one at a time; kill geometry-only route per spec. Audio = time anchor only. Piezo/impact-tape = GT-side signals. |
| 8 | **ShapeFromBlur (NEW from refutation)** | evaluate before building Gap C | MIT (rozumden/ShapeFromBlur) | [refute-FOUND] Released generic rigid-object package recovering textured 3D shape + SUB-FRAME MOTION from one blurred image — direct prior art for our exposure-integrated planar refinement gap. Evaluate on synthetic paddle crops before writing Gap C from scratch. Sports-equipment-specific solutions still do not exist (bounded search). |
| — | **Eliminated**: FoundationPose (RGB-D + NVIDIA non-commercial license §research/evaluation-only — refute-confirmed); SAM-6D (RGB-D; nested MIT component but NO root license — refute-corrected); FoundPose (CC-BY-NC, coarse-only release); category-level pose (no paddle category exists; revisit only after owner-instance proof); TT4D-class inverse-control as precision source (26.4±4.4° mean, refute-confirmed — keep only as prior/hypothesis-selector); deblur-first preprocessing (edge hallucination risk; blur-aware training preferred — BlurHandNet/Human-from-Blur confirmed); KV-Tracker (Imperial non-commercial — architecture reference only). | — | — | Evidence in both surveys + REFUTATION.md. |

## Where NO off-the-shelf solution exists — the build-our-own program

Six reconciled gap specs (full detail in RKT_CROSSCHECK_RULING.md + surveys):
A. tiny blur-aware paddle evidence extractor (native-res temporal ROI; heatmaps+visibility+blur+
   uncertainty; never collapse heatmaps pre-PnP);
B. ambiguity-preserving trajectory inference (the both-IPPE graph — rank 3 above);
C. exposure-integrated planar render refinement (SE(3) spline over exposure/rolling-shutter;
   allowed to return broad uncertainty) — evaluate MIT ShapeFromBlur as prior art first
   (refutation find);
D. paddle-shape adapter for unknown paddles (owner exact-CAD first; category later);
E. continuous-time contact estimator + oracle-ladder evaluator (rank 7 above);
F. commercial-clean synthetic hand+paddle domain (rank 4 above).
Each carries a kill criterion in the ruling; none has a market substitute.

## Owner-facing consequences

1. The NS-02 gold-capture protocol needs an RKT addendum: "inter-camera sync ≤0.5 frame" is
   NOT sufficient for contact GT (8.33ms @60fps ≈ 8-17cm at swing speed; refute-confirmed
   exact); the RKT blocks need ≤1ms audio/LED-verified sync (conservative; motion-only budget
   alone would allow 1.5-3ms) and the jig-audit metrology gate.
2. The half-day capture shot list (jig poses → shadow swings → ~150 fed contacts → rally
   footage → re-audit) is specified in benchmark_spec_rkt.md Tier-2 and rkt_survey_B §2;
   expected yield ~100-180 contact events + thousands of pose frames — enough to score the
   interim 30° milestone credibly and to fine-tune; the 5°/3cm gates additionally require the
   rig to PASS its metrology gate.

## What to benchmark first and why

Tier 1 (now, no GT): baseline-reproduction + RacketVision zero-shot integration card per
benchmark_spec_rkt.md — integration/runtime/proxy metrics only, explicitly NOT accuracy.
Tier 2 (after GT): the real gates, stratified by px-width/blur/edge-on/occlusion, oracle probes,
4-way synthetic ablation, then the temporal graph A/B. The spec pins the protocol so the owner
capture is collected to fit it.

## Paths

- Spec: runs/research_trk_rkt_20260716/benchmark_spec_rkt.md
- Ruling: runs/research_trk_rkt_20260716/RKT_CROSSCHECK_RULING.md
- Surveys: rkt_survey_A_20260716/SURVEY.md, rkt_survey_B_20260716/SURVEY.md (+livechecks)
- Refutation: rkt_refute_20260716/REFUTATION.md (+livechecks)
