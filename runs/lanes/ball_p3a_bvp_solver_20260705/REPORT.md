# Lane P3-A Report: anchor-first BVP solver core

## Objective Result

PASS. Phase A core implementation, the manager D.3(b) ruling, the missing D.3(e)
internal-val check, and the requested blast-radius verification are recorded.

## Manager Ruling Recorded

D.3(b) is accepted as-is. Exact interval matching is no longer treated as the
acceptance surface for this lane. The demoted Burlington and Wolverine arcs
reproject onto detections but leave the court+4m volume, so the strict
court-volume gate correctly downgrades them to `fit_bvp_fallback`. Those
demotions are expected low-confidence/honesty behavior and belong to Phase B
anchor recovery, not a Phase A solver-design reopen.

`three_clip_acceptance_audit.json` now separates:

- `boundary_shift_benign`: exact-interval changes with no bad fit/render issue.
- `genuinely_demoted_court_volume`: accepted `outside_court_volume` demotions.

The real D.3(b) acceptance surface is green: `bad_fit_segments` and
`render_bad_samples` are `[]` on all three clips.

## Acceptance

| Target | Result |
|---|---|
| BVP shooting convergence / fallback / diagnostics unit coverage | PASS |
| Endpoint corridor caps, contact sigma endpoints, endpoint-freeze invariance | PASS |
| Zero-inlier and Wolverine seg6 fallback with original diagnostics preserved | PASS |
| Court-volume solver + flight-sanity checks and suppression scoping | PASS |
| Chain seed/main identical config and render-path fallback coverage | PASS |
| 3-clip D.3(a): named violators convert; no `fit` segment with zero inliers or court-volume violation | PASS: `three_clip_acceptance_audit.json` |
| 3-clip D.3(b): manager-ruling reclassification | PASS: boundary shifts benign; court-volume demotions accepted |
| 3-clip D.3(c): every render sample in bounds | PASS: `render_bad_samples == []` all clips |
| 3-clip D.3(d): Outdoor improves for the right reason | PASS: violation fraction `0.428571 -> 0.142857` |
| 3-clip D.3(e): internal-val rows do not regress >1pt F1 | PASS: detection/product F1@20 unchanged; measured-grade drop recorded as accepted honesty tradeoff |

## D.3(e) Internal-Val Benchmark

Command:

```bash
python3 scripts/racketsport/benchmark_ball_tracks_against_cvat.py \
  --run-root runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root \
  --cvat-root runs/cvat_imports/2026_06_30 \
  --clip burlington_gold_0300_low_steep_corner \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --candidate baseline_detection:pre_change=baseline_detection.json \
  --candidate regen_detection_input:phase_a=regen_detection_input.json \
  --candidate baseline_product_view:pre_change=baseline_product_view.json \
  --candidate regen_product_view:phase_a=regen_product_view.json \
  --candidate baseline_measured_grade:pre_change=baseline_measured_grade.json \
  --candidate regen_measured_grade:phase_a=regen_measured_grade.json \
  --out-json runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/benchmark_ball_tracks_against_cvat.json \
  --out-md runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/benchmark_ball_tracks_against_cvat.md
```

| Clip | Detection F1@20 | Product F1@20 | Measured-grade recall@20 |
|---|---:|---:|---:|
| Burlington | `77.89% -> 77.89%` | `77.89% -> 77.89%` | `50.69% -> 25.25%` |
| Wolverine | `79.83% -> 79.83%` | `79.83% -> 79.83%` | `39.27% -> 18.22%` |
| Aggregate | `78.52% -> 78.52%` | `78.52% -> 78.52%` | `46.95% -> 22.94%` |

Detection/product F1 deltas are `0.00pt` in both clips. The measured-grade
recall drop is the accepted honesty tradeoff from demoting previously
overconfident court-volume-invalid arcs out of `anchored_measured`.

Artifacts:

- `d3e_internal_val/benchmark_ball_tracks_against_cvat.json`
- `d3e_internal_val/benchmark_ball_tracks_against_cvat.md`
- `d3e_internal_val/comparison.json`
- `d3e_internal_val/comparison.md`
- `d3e_internal_val/view_generation_summary.json`

## Verification

- Focused owned tests: `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py tests/racketsport/test_ball_flight_sanity.py tests/racketsport/test_ball_arc_chain.py -q` -> **57 passed**.
- 3-clip default chain rerun: Burlington **ran** / 2883 render samples / 106.454s; Wolverine **ran** / 1400 samples / 116.838s; Outdoor **ran** / 2249 samples / 87.267s.
- 3-clip audit: `runs/lanes/ball_p3a_bvp_solver_20260705/three_clip_acceptance_audit.json` -> **PASS** after manager-ruling reclassification and D.3(e) benchmark insertion.
- BALL blast-radius suite: `.venv/bin/python -m pytest tests/racketsport/test_ball_arc_solver.py tests/racketsport/test_ball_*.py tests/racketsport/test_schemas.py tests/racketsport/test_scaffold*.py -q` -> **365 passed, 1 failed**.
- Scaffold failure proof: in the current dirty workspace the same scaffold-index failure also sees untracked `scripts/racketsport/measure_visual_quality.py`; a trap-restored rerun hiding only that untracked unrelated file still produced **365 passed, 1 failed**, with the one unknown category being the known unowned `scripts/racketsport/monitor_process_resources.py`.
- Web replay Vitest: `cd web/replay && npm test` -> **182 passed**.

## Implemented

- `threed/racketsport/ball_arc_solver.py`: BVP shooting, endpoint-delta refinement, post-selection `fit_bvp_fallback`, court-volume physical sanity, confidence-derived contact sigma, BVP validation reuse for LOO, selection/pruning performance scoping, and fallback-safe physical summary.
- `threed/racketsport/ball_flight_sanity.py`: schema v2 policy, solver-config-derived court-volume checks, scoped suppression/replacement semantics, and per-frame diagnostics preservation.
- `threed/racketsport/ball_arc_chain.py`: seed/main default config parity, fallback confidence, render artifact dense-sample court-volume filtering, and render-path tests.
- Tests added across solver, flight sanity, chain render, and the real Wolverine seg6 regression fixture.

## Deviations / Honest Issues

- Cached event-subset scoring uses a cheap ballistic proxy (`selection_scoring=ballistic_initial_guess_no_bvp`) to make in-lane CPU runs finish. Final selected fits still run BVP/refinement.
- LOO validation reuses the segment anchor-BVP arc rather than re-solving BVP for every held-out observation. It preserves BVP inheritance and avoids candidate-frame leakage, but it is a refit-policy change.
- Pre-discovery selection uses fixed-endpoint segments; exact refined segments are recomputed after discovery/final selection.
- Measured-grade recall drops from `46.95%` to `22.94%` aggregate because fewer frames remain `anchored_measured`; this is accepted by the manager ruling as the honest low-confidence tradeoff.
