# Plant-phase plausibility check — spurious 4-frame plants no longer poison the slide metric

Lane: `plant_phase_plausibility_20260728`. Date: 2026-07-28.
Commits: `55f6761` (the fix + tests, lane agent) + the follow-up allowlist
one-liner in `threed/racketsport/body_grounding_quality.py` (orchestrator, this
commit). Status: measurement-robustness engineering; `VERIFIED=0`. Written by
the orchestrator from the lane agent's returned findings (its harness refused
report-file writes). Recompute script + results:
`recompute_slide_metric.py`, `recompute_results.json`.

## The check

A candidate plant phase is trusted for the slide metric only if EITHER:
- `frame_count >= 5` (`plant_plausibility_min_frames`; the diagnosed spurious
  phase was 4 frames), OR
- `max_speed_mps <= 0.66 m/s` (`enter_speed_mps 0.75 × plant_plausibility_speed_fraction 0.88`).

New pure function `foot_contact.phase_speed_duration_plausible()`, wired into
both `foot_contact._body_phase_rejection_reason()` and
`worldhmr._gate_phase_rejection_reasons()` (the slide-gate feed for
`max_foot_lock_slide_m`). New typed reason
`implausible_short_high_speed_plant`, registered in
`foot_contact.INDEPENDENT_BODY_PHASE_REJECTION_REASONS` and (follow-up
one-liner) `body_grounding_quality.INDEPENDENT_PHASE_REJECTION_REASONS` so the
top-level grounding status honors it.

The check NEVER reads `slide_m`/displacement — only duration and speed — so a
genuinely planted-then-sliding foot always stays in the metric (two pre-existing
regression tests covering a real 5-frame/0.3 m/s/40 mm slide remain green).

Threshold derivation was population-based, not two-point fitted: spurious phase
= 4 frames @ 0.6996 m/s; shortest reproducible REAL slide phase across all six
target-clip runs = 6 frames @ 0.59–0.62 m/s; fastest reproducible real short
phase = 0.623–0.640 m/s. The floor (5 frames) and ceiling (0.66 m/s) sit
strictly between with margin, verified against the full phase population.

## Per-clip before → after (recomputed from existing artifacts)

| Clip | Run | Before | After |
|---|---|---:|---:|
| wolverine 0-10 | tonight | 0.037769 m FAIL | **0.008654 m PASS** |
| wolverine 0-10 | baseline | 1.79e-15 m PASS | 0.0 m PASS |
| indoor_diagonal 100-110 | tonight | 0.054561 m FAIL | 0.054561 m FAIL (unchanged) |
| indoor_diagonal 100-110 | baseline | 0.054476 m FAIL | 0.054476 m FAIL (unchanged) |
| outdoor_pbvision 50-60 | tonight | 0.002604 m PASS | 0.0 m PASS |
| outdoor_pbvision 50-60 | baseline | 0.016871 m PASS | 1.8e-15 m PASS |

## indoor_diagonal verdict: REAL slide, not the spurious class

Its failing phase (`3:right:438-443`, 6 frames) reproduces almost identically
in the baseline AND tonight (0.054476 vs 0.054561 m — 0.09 mm apart; speeds
0.619 vs 0.591 m/s) — the opposite signature from the diagnosed defect. At 6
frames it clears the duration floor; the fix correctly leaves it failing. This
is now the one genuine foot-slide fault in the six-clip suite — a real accuracy
item for the daytime queue, not a measurement artifact.

## Tests

8 new tests (6 in `test_foot_contact.py`, 2 through the real
`_contact_gate_stream_for_skeleton3d` in `test_worldhmr_stance_grounding.py`).
Focused sweep `-k "foot or ground or contact or slide"`: 322 passed, 1
pre-existing unrelated failure (`test_attribute_body_decode_residual`,
reproduced identically on a clean pre-lane checkout). `test_process_video.py`:
186 passed.

## Honesty note

The slide-metric definition changed: prior numbers are not directly comparable
without the recomputation table above. This is measurement robustness, not an
accuracy promotion; the frozen-gate discipline requires citing this lane
whenever pre/post numbers are compared.
