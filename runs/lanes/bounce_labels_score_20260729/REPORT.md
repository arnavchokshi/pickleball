# Owner bounce-label round 2 — first fresh solver-vs-human 3D numbers since the pilot

Lane: `bounce_labels_score_20260729`. Date: 2026-07-29 (overnight). Commit `096619f`.
Status: review-only evidence; `VERIFIED=0`; human labels are not verified GT.
Written by the orchestrator from the lane's returned findings; full per-label
tables in `bounce_labels_score_report.json`, headline numbers in
`bounce_labels_score_summary.json`, method in `score_round2_labels.py`.

## Labels scored (owner round 2, in progress — 53 at scoring time)

| Clip | Total | Bounce | Near-player | Free-flight | Prefill-corrected | Calib floor (median) |
|---|---:|---:|---:|---:|---:|---:|
| wolverine | 10 | 6 | 0 | 4 | 90% | 0.1268 m |
| burlington | 12 | 6 | 0 | 6 | 67% | 0.1907 m |
| outdoor_webcam | 13 | 4 | 9 | 0 | 62% | 0.1006 m |
| pbv11 | 18 | 17 | 0 | 1 | 0% (no prefill exists) | 0.1444 m |

## Bounce vs live solver (`ball_track_arc_solved.json`; 14/33 bounces comparable)

Pooled: median **0.268 m**, p90 1.479 m. By band:

| Band | n | median | p90 | max |
|---|---:|---:|---:|---:|
| `anchored_measured` | 5 | **0.149 m** | 0.227 m | 0.235 m |
| `arc_interpolated` | 1 | 0.474 m | — | — |
| `arc_weak` | 8 | 0.632 m | 1.835 m | 2.056 m |

**Headline: anchored_measured bounces sit at the calibration floor** (0.10–0.19 m
measured per clip) — on fresh labels the anchored solve is calibration-limited,
not solver-limited, consistent with the 2026-07-26 finding that calibration is
the binding floor on bounce accuracy.

**Band separation replicates in direction** (arc_weak ~4.2× worse than
anchored; `separation_holds_on_fresh_labels: true`) but not the pilot's
catastrophic magnitudes (pilot saw 2.5–24.8 m on extreme-z frames; this sample
tops out at 2.06 m — none of these labels landed on such frames). Honest
sampling note, not a method discrepancy: every computed error cross-checked
against the label's own recorded `prefill.delta_m` to 6 decimals.

Near-player/free-flight comparisons reported separately as estimate-vs-estimate
per §2.3; never aggregated with bounce. pbv11 has no solver arc (ball track
covers only its first ~180 s) — its 17 bounce labels are banked for future
scoring.

## What this feeds

- Round target ≥150 bounces: 33/150 at scoring time — round stays open.
- The calibration k1/distortion lever (queue row 3) is now demonstrably the
  main lever for anchored-bounce accuracy on fresh labels.
- pbv11's banked labels become scorable once a full-length ball track exists
  for the demo clip.
