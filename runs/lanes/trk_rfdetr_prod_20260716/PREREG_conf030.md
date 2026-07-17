# PREREGISTRATION — RF-DETR-L conf-floor 0.30 single-shot (Track F, 2026-07-17)

Written BEFORE execution. This is the SECOND and FINAL frozen-threshold attempt permitted by the
detbench stop rule ("two frozen-threshold attempts"). No iteration, no grid, no post-hoc floor
adjustment: one run at exactly conf ≥ 0.30, both clips.

- Inputs: the pinned detbench arm-1 RF-DETR-L raw detections (checkpoint sha256 0f4e20e1…,
  person class id 1, native 704; md5-pinned dumps) filtered to conf ≥ 0.30. NO re-inference.
- Pipeline: identical to variant P otherwise — per-frame BOTSORT (botsort_no_reid_loose.yaml)
  feeder → frozen association (margin 1.0 + OSNet, defaults) → frozen scorer (IoU 0.5,
  expected-players 4, frozen GT). GPU-class environment mandatory (Mac CPU not score-faithful).
- PASS criteria (coordinator's 2a, verbatim intent): wolverine switches = 0 AND spectator FP = 0
  AND far-off-court FP = 0, WHILE burlington keeps an all-clean row (0 sw / 0 spectFP / 0 farFP)
  with a material gain over baseline (cov4 ≥ 0.95-class, IDF1 > baseline).
- FAIL → coordinator's 2b executes: ship-for-demo flip proposal at the production-equivalent
  0.18 operating point with the wolverine regression stated verbatim.
- All numbers reported regardless (IDF1/cov4/HOTA per clip); one-shot result retained whether
  pass or fail. VERIFIED=0 either way; the fresh-clip full gate remains unmet on this
  historical card.
