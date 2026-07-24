# WS5 lifter prep — notes (2026-07-23, VERIFIED=0, measurement-only)

Measured: WASB tennis zero-shot (thr 0.5) vs human labels — 566 labeled frames (10 clips: 4 eval_clips click sets + 6 online-harvest CVAT sets), 152 independent pairs after excluding 126 CVAT labels that are unmoved detector-seeded prelabels (circular; ≤0.5 px from prediction).
Headline: inlier jitter ~unbiased, MAD-σ 3.7/2.6 px (dx/dy); false-peak mode 15.9% at court-scale displacement; miss 21.6%; FP-at-hidden 34.3% pooled, 50.5% outdoor-night, FPs confident (median 0.844); 4,111 missing-detection gaps, mostly 1–5 frames with a 100+ frame tail. All inputs sha256-recorded in residual_analysis.json (deterministic, re-runnable via analyze_residuals.py).
Spec: GENERATOR_INTERFACE_SPEC.md binds the pose-conditioned generator to the landed A-3 contract (threed/racketsport/ball_metric3d_contract.py), maps every injected effect to a deliverable-1 field, fixes the B3 gated cross-attention I/O (Q=ball, K/V=player+paddle, gate=hit-prob), and lists pre-training acceptance gates.
Deferred (freeze): generator code, any training, any solver/anchor change, owner-retrain (w7) noise re-measurement (per-frame predictions not local), occlusion-cause splits (needs A-1 capture).
HOLD 1: Codex court-lane K,D,R,t (+Σ) calibration — generator projection/ray stage blocked without it.
HOLD 2: Gate 2.2 / Phase-A exit — reproducible metric report for the current physics-only solver; zero training use of generator output before it holds.
No spin anywhere (killed approach); physics limited to the existing gravity+drag ODE constants.
