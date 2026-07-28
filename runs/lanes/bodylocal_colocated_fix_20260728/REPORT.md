# Co-located `--body-local` reuse fix + first full co-located BODY measurement

Lane: `bodylocal_colocated_fix_20260728`. Date: 2026-07-28. Fix commit: `757da51`.
Status: measured engineering evidence; `VERIFIED=0`. Written by the orchestrator
from the lane agent's returned findings (its harness refused report-file writes);
two-sided sha256-verified run evidence lives under this directory
(`colocated_wolverine/`, `stretch_full_ball_oneworld/`).

## The bug (proven, not hypothesized)

Co-located `--body-local` silently degraded BODY to skeleton-only on every GPU
run. Root cause, reproduced byte-for-byte in an instrumented local repro: the
NVZ/kitchen line-posterior persistence added by `alwayson_defaults_20260728`
(`_persist_nvz_line_posteriors` in `threed/racketsport/placement.py`)
legitimately rewrites `court_lock.json` in place later in the same run — after
calibration's `RunIdentityStore` transaction fingerprinted it. `_run_body_local()`'s
`orchestrator.run_pipeline(..., require_content_identity_for_reuse=True)` then
saw calibration as falsely stale, fell through to the default-registered
`ManualCalibrationRunner`, and failed on `capture_sidecar.json` — absent on bare
eval `.mp4` clips. (The 2026-07-05 `pipeline_speed` log shows the same terminal
symptom; tonight's identity-store interaction is the proven mechanism for this
failure.)

## The fix (surgical, `scripts/racketsport/process_video.py` only)

1. `_identity_artifacts()`: `court_lock.json` added to calibration's
   `mutable_later` set — the exact fix shape already used for `tracks.json`
   (tracking/player_selection rewrite pattern).
2. `_stage_body()` + new `_local_body_calibration_available()`: a fail-closed
   typed pre-check (`reason_code=local_body_missing_calibration`,
   `status="blocked"`) so genuinely missing/stale calibration never reaches
   `ManualCalibrationRunner`'s sidecar requirement. Uses `RunIdentityStore`
   exact generation references — never lexical. Remote dispatch path untouched.

Tests: 5 new focused tests + 1 fixture update; `test_process_video.py` **186
passed** (181 pre-existing + 5 new, 0 regressions); `test_run_identity.py` +
`test_orchestrator_spine.py` spot-check 42 passed.

## Measured results (night1, A100-40GB, compute mode Default)

**Complete co-located pipeline including real BODY, wolverine
(`court_skeletons`, fresh `--force`, source-video-only): total wall 266.5 s** —
24% faster than the 352.5 s July-25 baseline median and 47% faster than
tonight's contended ~500 s split-mode median. Stage breakdown: ingest 1.04,
calibration 37.86, tracking 33.83, player_selection 2.20, placement 3.02,
frames 4.50, **body 158.23 (ran, not degraded: 1136 player-frames, coverage
1.0, 0 blockers, ~63 crops/s steady-state)**, placement_refine 8.75,
grounding_refine 3.56, placement_trajectory_refine 3.97, world 3.68,
confidence_gate 3.39, manifest 1.20. All three always-on stages ran for real.
Foot-slide `max_foot_lock_slide_m = 0.0158 m` — passes the 0.03 m bar.

**court23 attempts (2): fix confirmed working** (calibration reused, BODY
genuinely launched) but both failed with `CUDA error: device(s)
busy/unavailable` from the GPU's `Exclusive_Process` compute mode — a separate,
pre-existing infra issue (same signature documented in `gpu_fleet.md`
2026-07-20). Follow-up flagged: co-located runs need compute mode Default
(`nvidia-smi -c 0`) or a single shared CUDA context.

## Stretch: the "everything together" integration demo (night1)

Full preset + ball stages + default ball-aware mesh scheduling + `--one-world`
on the canonical `wolverine_mixed_0200_mid_steep_corner` + its committed
`court_calibration_metric15pt.json` seed: **wall 987.3 s**, real mesh BODY
(584.5 s), `one_world` ran, `match_stats`/`paddle_pose` ran. Five stages
honestly typed-degraded, all pre-existing documented product behaviors, none
caused by this fix: input_quality (low camera angle), ball_arc +
ball_arc_refined (known solver/segment-budget limits), grounding_refine
(worsened-residual sanity gate restored originals), coaching_facts
(zero-fabrication audit correctly rejected free-form language). Nothing
fabricated. Evidence: `stretch_full_ball_oneworld/`.

## Honest caveats

Single clip, single night, shared repo tree during a multi-lane session;
timing is engineering evidence, not accuracy promotion; `VERIFIED=0` binding.
