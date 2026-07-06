# D.3(e) Internal-Val Comparison

Benchmark: `runs/lanes/ball_p3a_bvp_solver_20260705/d3e_internal_val/benchmark_ball_tracks_against_cvat.json`
CVAT root: `runs/cvat_imports/2026_06_30`

| Clip | Detection F1@20 baseline -> after | Product F1@20 baseline -> after | Measured-grade recall@20 baseline -> after | Measured-grade F1@20 baseline -> after |
|---|---:|---:|---:|---:|
| burlington_gold_0300_low_steep_corner | 77.89% -> 77.89% (+0.00pt) | 77.89% -> 77.89% (+0.00pt) | 50.69% -> 25.25% (-25.44pt) | 66.24% -> 40.19% (-26.05pt) |
| wolverine_mixed_0200_mid_steep_corner | 79.83% -> 79.83% (+0.00pt) | 79.83% -> 79.83% (+0.00pt) | 39.27% -> 18.22% (-21.05pt) | 54.04% -> 30.20% (-23.84pt) |

## Aggregate

- Detection F1@20: 78.52% -> 78.52% (+0.00pt).
- Product-view F1@20: 78.52% -> 78.52% (+0.00pt).
- Measured-grade recall@20: 46.95% -> 22.94% (-24.01pt).
- Measured-grade F1@20: 62.38% -> 37.01% (-25.37pt).

## Acceptance

- PASS: no detection or product-view F1@20 row regressed more than 1pt.
- ACCEPTED HONESTY TRADEOFF: measured-grade recall drops because fewer frames remain anchored_measured after court-volume demotion.
