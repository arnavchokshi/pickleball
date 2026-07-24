# Detector-Noise + Missing-Detection Characterization (WS5, freeze-safe)

Date: 2026-07-23. Status: measurement-only, `VERIFIED=0`, `ball_verified=false`.
Produced by `analyze_residuals.py` in this directory; all numbers below are read from
`residual_analysis.json` (deterministic: sorted keys, 6-decimal floats, no timestamps;
re-running the script reproduces the identical file byte-for-byte). Every input file is
recorded with root-relative path + sha256 in the `inputs` block of the JSON (26 files).

This characterizes the noise a future pose-conditioned synthetic generator (PLAN.md §5.B1)
must inject. It trains nothing and changes no solver behavior.

## 1. Data actually used (all pre-existing, main checkout, read-only)

| Set | Labels (human) | Predictions (detector) | Clips | Labeled frames |
|---|---|---|---|---|
| A | `eval_clips/ball/{clip}/labels/ball_points.json` — sparse click reviews, 30/clip, `status=human_reviewed`, `not_ground_truth=true` | `runs/lanes/ball_tracking_track_regen_20260704/tracks/{clip}/wasb_tennis_zeroshot_thr_0_500/source_wasb_predictions.csv` — dense per-frame WASB tennis zero-shot, thr 0.5 | 4 (burlington, indoor_doubles_fwuks, outdoor_webcam_iynbd, wolverine) | 120 (89 visible) |
| B | `runs/lanes/w3_reviewimport_20260707/normalized_cvat/{clip}/annotations.xml` — sparse CVAT keyframes, stride ~18 frames; `outside=1` keyframes read as human "not visible here" | `data/online_harvest_20260706/prelabels/{clip}/ball_track.json` — dense raw WASB tracks (the same files scored in the w3 benchmark) | 6 online-harvest rally clips | 446 (267 visible) |
| Blur | — | `runs/lanes/ball_anchor_boost_20260712/{burlington,wolverine}_ball_blur_sidecar.json` (per-frame blur measurements) | 2 | 36 joinable label frames |

Totals: 566 human-labeled frames evaluated (356 visible / 210 not-visible), 279 paired
detections, 73,448 dense detector frames, 4,111 interior missing-detection gaps.

Detector identity: **WASB tennis zero-shot at heatmap threshold 0.5** in both sets. This is
NOT the w7 owner-retrained production model — see §7.

## 2. Label-provenance finding (affects everything downstream)

**126 of 209 SET B paired labels sit ≤ 0.5 px from the WASB output.** The CVAT tasks were
seeded from these same WASB prelabel tracks, so those keyframes are unmoved seeds:
circular evidence, uninformative about localization noise. They are excluded from all
residual estimates (`coincident_pair` in the JSON). Per clip: `_L0HVmAlCQI` 47/51,
`HyUqT7zFiwk` 26/27, `zwCtH_i1_S4` 32/34, `Ezz6HDNHlnk` 21/29 coincident;
`73VurrTKCZ8` 0/44 and `wBu8bC4OfUY` 0/24 (independently placed). SET A is clean (1/70
within 0.5 px; that pair is retained-excluded by the same rule for consistency).

Consequences: (a) residual estimates use only the 152 independent pairs (69 SET A + 83
SET B human-moved); (b) SET B human-moved residuals are conditioned on the annotator
having chosen to move the seed, so they over-represent error — SET A is the primary
localization-noise source; (c) the w3 benchmark's headline "median error 1.88 px" is
dominated by unmoved seeds and should not be used as a detector-noise estimate.

## 3. Localization noise (independent pairs, labeled-visible frames)

Two-mode structure. Mode 1 — **inlier jitter** (residual ≤ 20 px):

| Source | n | rate | dx bias / MAD-σ (px) | dy bias / MAD-σ (px) | radial p50 / p90 / p95 (px) |
|---|---|---|---|---|---|
| SET A (independent clicks) | 58 | 0.841 | +1.67 / 3.68 | +0.38 / 2.56 | 4.73 / 7.71 / 8.33 |
| SET B (human-moved, biased high) | 61 | 0.735 | — / 4.17 | — / 3.19 | 4.08 / 9.24 / 13.14 |

Inlier jitter is near-unbiased, roughly isotropic, σ ≈ 2.5–4 px at 1080p. Within the
inlier mode the tails are mild (SET A inlier excess kurtosis: dx −0.60, dy +0.17).

Mode 2 — **false-peak / mis-association** (residual > 20 px at a labeled-visible frame):
rate 0.159 in SET A (11/69); displacement is court-scale, not local — SET A p50 155 px,
p90 1284 px. There is essentially no mass between 20 and 50 px (`frac_gt_20px` ≈
`frac_gt_50px`), i.e. the detector is either on the ball or somewhere else entirely.

**Heavy-tail quantification (all independent pairs, SET A):** dx excess kurtosis 18.1,
dy 7.31; |dx| p50 3.98 px vs p95 293.7 px; radial p50 5.02 px vs p95 302.0 px. A single
Gaussian is a wrong model; inject as mixture: (1−p_out)·N(bias, σ_inlier) + p_out·(far
false-peak draw), with p_out ≈ 0.12–0.16 at this operating point.

## 4. Miss and false-positive rates (all 566 labeled frames)

| Slice | labeled-visible n | miss rate | labeled-hidden n | FP rate at hidden |
|---|---|---|---|---|
| pooled | 356 | 0.216 | 210 | 0.343 |
| indoor | 86 | 0.291 | 53 | 0.132 |
| outdoor_day | 110 | 0.145 | 62 | 0.274 |
| outdoor_night | 160 | 0.225 | 95 | 0.505 |

False positives are **confident**: FP confidence median 0.844, p90 0.917 — confidence
alone cannot gate them. Confidence does correlate with localization quality on
independent pairs (inlier rate 0.55 below conf 0.7, 0.86 above 0.9; see
`confidence_bins_independent_pairs`), so confidence→pixel-covariance calibration (A-3)
is supported, but a confident-FP mode must be injected regardless of confidence.
Regime tags: SET A from `clip_metadata.json`; SET B visually classified from one
screening thumbnail per video during WS5 (approximate; flagged `visual_screen`).

## 5. Missing-detection gap lengths (dense tracks, 73,448 frames)

Interior runs of consecutive frames with no visible detection. This is an **upper bound
on occlusion**: it mixes true occlusion with plain detector misses on visible balls
(no occlusion-cause labels exist locally). 4,111 gaps across 10 clips:

- Pooled histogram: 1–2 f: 1,852 · 3–5 f: 795 · 6–11 f: 599 · 12–29 f: 594 · 30–59 f: 193 · 60+ f: 78.
- Typical clip p50 is 1–5 frames; p95 ranges 8 f (burlington, indoor 60 fps) to 58 f
  (outdoor_webcam); worst observed gaps 178 f (`HyUqT7zFiwk`, indoor 30 fps ≈ 5.9 s) and
  182 f (`zwCtH_i1_S4`). Per-clip distributions + ms conversions are in
  `per_clip.*.missing_detection_gaps` (fps recorded per clip).
- Detection coverage varies 0.45–0.83 per clip; the lowest-coverage clips are the long
  online-harvest recordings (`HyUqT7zFiwk` 0.485, `zwCtH_i1_S4` 0.454, `wBu8bC4OfUY`
  0.461 — these include dead time between rallies, so coverage there is not a per-rally
  statement) plus the `indoor_doubles_fwuks` (0.446) and `outdoor_webcam_iynbd` (0.487)
  eval clips.
- Boundary (censored) runs are counted separately and excluded from gap statistics.

Generator guidance: gap lengths need at least two regimes — a geometric-ish short mode
(1–5 f) plus a long-tail mode reaching 100–600+ ms, matching PLAN §5.B1's "contiguous
100–600 ms" masking plus rarer multi-second dropouts.

## 6. Blur association (small-n, directional only)

36 labeled-visible frames on burlington + wolverine join a blur-sidecar frame.
Split at median blur length (21.5 px): median radial residual 4.73 px (low blur) vs
4.16 px (high blur) — **no detectable association at n=36**; do not condition injected
jitter on blur from this evidence. Caveat: sidecars were computed along the
`ballcand_20260710` baseline track, not the labels (frame-index join only). Sidecar blur
lengths themselves: p50 21.5 px, p90 36.3 px — usable as a marginal blur distribution.

## 7. Gaps — what does NOT exist locally (stated, not synthesized)

1. **No per-frame predictions from the w7 owner-retrained model** paired with any local
   label set: `runs/lanes/w7_ballretrain_20260709/vm_pull/` holds only aggregate LOSO
   reports. This characterization is therefore of the *zero-shot* WASB operating point;
   the production model's noise must be re-measured with the same script once its
   per-frame outputs exist locally (the script is detector-agnostic given the same CSV
   or track-JSON shapes).
2. **No dense human labels**: all labels are sparse (stride ~10–20 frames), so miss/FP
   rates are sampled-frame estimates, and per-event (hit/bounce-window) noise cannot be
   sliced with usable n.
3. **No occlusion-cause labels**: gap lengths cannot be split into player-occlusion vs
   plain miss; PLAN §5.C occlusion-as-supervision will need that split from A-1 capture.
4. **pb.vision teacher tracks were deliberately excluded** (teacher pseudo-labels, not
   human review), as were `ball_gt_rescore_20260713/real_wasb` reruns (only 2 of 4 eval
   clips, second config; the track_regen set covers all 4 with one config).
5. Sample sizes are small (152 independent pairs); treat σ estimates as ±30–40 % and
   revisit against A-1 Tier-1 data before using them as acceptance thresholds.
