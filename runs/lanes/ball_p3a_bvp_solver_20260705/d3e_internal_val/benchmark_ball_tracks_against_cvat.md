# CVAT Ball Tracker Benchmark

BALL is not verified by this report. It scores existing artifacts against reviewed CVAT labels and contact review timestamps only.

## Verification Blockers

- No reviewed-label BALL acceptance gate is defined and passed here; every row remains a scored candidate, not a verification result.
- Best scored candidate is baseline_measured_grade with score 0.459, F1@20 0.624, hit recall 0.479, hidden FP 0.130, and 0 teleports; this is not a BALL gate pass.

## Full-Horizon Coverage

| CVAT labels | Evaluated labels | Excluded labels | All labels evaluated |
| ---: | ---: | ---: | --- |
| 754 | 754 | 0 | yes |

## Aggregate

| Candidate | Category | Clips | Eval labels | Excl labels | F1@20 | Precision@20 | Recall@20 | Hit recall | P90 px | P95 px | Hidden FP | Hidden FP/min | Coverage | P95 step px | Teleports | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_detection | pre_change | 2 | 754 | 0 | 0.785 | 0.811 | 0.761 | 0.782 | 498.762 | 689.587 | 0.349 | 941.378 | 0.781 | 62.200 | 22 | 0.323 |
| baseline_measured_grade | pre_change | 2 | 754 | 0 | 0.624 | 0.929 | 0.469 | 0.479 | 12.601 | 14.805 | 0.130 | 346.185 | 0.411 | 35.317 | 0 | 0.459 |
| baseline_product_view | pre_change | 2 | 754 | 0 | 0.785 | 0.811 | 0.761 | 0.782 | 498.762 | 689.587 | 0.349 | 941.378 | 0.781 | 62.200 | 22 | 0.323 |
| regen_detection_input | phase_a | 2 | 754 | 0 | 0.785 | 0.811 | 0.761 | 0.782 | 498.762 | 689.587 | 0.349 | 941.378 | 0.781 | 62.200 | 22 | 0.323 |
| regen_measured_grade | phase_a | 2 | 754 | 0 | 0.370 | 0.956 | 0.229 | 0.239 | 14.502 | 16.046 | 0.007 | 19.336 | 0.193 | 16.476 | 0 | 0.271 |
| regen_product_view | phase_a | 2 | 754 | 0 | 0.785 | 0.811 | 0.761 | 0.782 | 498.762 | 689.587 | 0.349 | 941.378 | 0.781 | 62.200 | 22 | 0.323 |

## Per Clip

| Clip | Candidate | CVAT frames | Evaluated frames | Excl frames | Eval labels | Excl labels | F1@20 | Hit recall | P90 px | P95 px | Hidden FP | Coverage | Teleports | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| burlington_gold_0300_low_steep_corner | baseline_detection | 600 | 600 | 0 | 507 | 0 | 0.779 | 0.761 | 978.057 | 1346.374 | 0.344 | 0.798 | 19 | 0.078 |
| burlington_gold_0300_low_steep_corner | baseline_measured_grade | 600 | 600 | 0 | 507 | 0 | 0.662 | 0.509 | 8.205 | 9.357 | 0.108 | 0.448 | 0 | 0.540 |
| burlington_gold_0300_low_steep_corner | baseline_product_view | 600 | 600 | 0 | 507 | 0 | 0.779 | 0.761 | 978.057 | 1346.374 | 0.344 | 0.798 | 19 | 0.078 |
| burlington_gold_0300_low_steep_corner | regen_detection_input | 600 | 600 | 0 | 507 | 0 | 0.779 | 0.761 | 978.057 | 1346.374 | 0.344 | 0.798 | 19 | 0.078 |
| burlington_gold_0300_low_steep_corner | regen_measured_grade | 600 | 600 | 0 | 507 | 0 | 0.402 | 0.254 | 8.661 | 10.031 | 0.011 | 0.217 | 0 | 0.324 |
| burlington_gold_0300_low_steep_corner | regen_product_view | 600 | 600 | 0 | 507 | 0 | 0.779 | 0.761 | 978.057 | 1346.374 | 0.344 | 0.798 | 19 | 0.078 |
| wolverine_mixed_0200_mid_steep_corner | baseline_detection | 300 | 300 | 0 | 247 | 0 | 0.798 | 0.826 | 19.468 | 32.800 | 0.358 | 0.763 | 3 | 0.568 |
| wolverine_mixed_0200_mid_steep_corner | baseline_measured_grade | 300 | 300 | 0 | 247 | 0 | 0.540 | 0.417 | 16.998 | 20.254 | 0.170 | 0.373 | 0 | 0.378 |
| wolverine_mixed_0200_mid_steep_corner | baseline_product_view | 300 | 300 | 0 | 247 | 0 | 0.798 | 0.826 | 19.468 | 32.800 | 0.358 | 0.763 | 3 | 0.568 |
| wolverine_mixed_0200_mid_steep_corner | regen_detection_input | 300 | 300 | 0 | 247 | 0 | 0.798 | 0.826 | 19.468 | 32.800 | 0.358 | 0.763 | 3 | 0.568 |
| wolverine_mixed_0200_mid_steep_corner | regen_measured_grade | 300 | 300 | 0 | 247 | 0 | 0.302 | 0.206 | 20.343 | 22.061 | 0.000 | 0.170 | 0 | 0.218 |
| wolverine_mixed_0200_mid_steep_corner | regen_product_view | 300 | 300 | 0 | 247 | 0 | 0.798 | 0.826 | 19.468 | 32.800 | 0.358 | 0.763 | 3 | 0.568 |

## Next Training/Eval Recommendation

Next: keep the best full-span candidate as a benchmark candidate only, inspect its false positives and misses, and train only against dense reviewed labels or hard negatives if the held-out CVAT metrics still miss the BALL gate.

## Candidate Paths

| Clip | Candidate | Category | Track frames | Excluded labels | Path |
| --- | --- | --- | ---: | ---: | --- |
| burlington_gold_0300_low_steep_corner | baseline_detection | pre_change | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/baseline_detection.json` |
| burlington_gold_0300_low_steep_corner | baseline_measured_grade | pre_change | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/baseline_measured_grade.json` |
| burlington_gold_0300_low_steep_corner | baseline_product_view | pre_change | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/baseline_product_view.json` |
| burlington_gold_0300_low_steep_corner | regen_detection_input | phase_a | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/regen_detection_input.json` |
| burlington_gold_0300_low_steep_corner | regen_measured_grade | phase_a | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/regen_measured_grade.json` |
| burlington_gold_0300_low_steep_corner | regen_product_view | phase_a | 600 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/burlington_gold_0300_low_steep_corner/regen_product_view.json` |
| wolverine_mixed_0200_mid_steep_corner | baseline_detection | pre_change | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/baseline_detection.json` |
| wolverine_mixed_0200_mid_steep_corner | baseline_measured_grade | pre_change | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/baseline_measured_grade.json` |
| wolverine_mixed_0200_mid_steep_corner | baseline_product_view | pre_change | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/baseline_product_view.json` |
| wolverine_mixed_0200_mid_steep_corner | regen_detection_input | phase_a | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/regen_detection_input.json` |
| wolverine_mixed_0200_mid_steep_corner | regen_measured_grade | phase_a | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/regen_measured_grade.json` |
| wolverine_mixed_0200_mid_steep_corner | regen_product_view | phase_a | 300 | 0 | `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/comparison_root/wolverine_mixed_0200_mid_steep_corner/regen_product_view.json` |
