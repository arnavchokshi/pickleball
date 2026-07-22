# ULTRA ADVERSARIAL REVIEW of lane ball_b1b2_prep_20260721 (read-only)

Ground truth: runs/lanes/ball_b1b2_prep_20260721/spec.md, report.json, log.txt; owned files:
build_pbvision_ball_sst.py (new), train_ball_stage2.py (modified), ball_loso_validation.py
(modified), test_build_pbvision_ball_sst.py + test additions. Doctrine: EXACT_PLAN §3.2 B1/B2 +
§2.1 BALL teacher ruling.

Attack surfaces:
1. Compare-ID refusal: truly structural (before any path/read), or bypassable via CLI/symlink/
   manifest aliasing?
2. Eligibility: can a row slip in WITHOUT frozen-WASB spatial agreement (<=20px @ conf>=0.90) or
   the preregistered temporal/geometry check? Is that check actually preregistered (parameters
   frozen in the emitted manifest) or tunable post-hoc? Teacher absence as negative — truly
   impossible?
3. THE A-ARM PARITY CLAIM (highest stakes): 'no-SST invocation byte-identical to today' — verify
   the proof is real (exact loss array + model-state SHA with flags absent vs pre-change behavior),
   not a self-comparison of the new code with itself. Check git: does the modified trainer with no
   SST flags follow EXACTLY the old code path (RNG draws, dataloader order, batch composition)?
4. SST math: batch composition (8 human + N pseudo), loss cap post-weighting <=0.25×human — probe
   edge cases (zero pseudo loss, cap binding, weight interaction with sample_weight 0.25/0.5).
5. Parent-source scorer: grouping correctness on the live B0 artifact (167 rows / 2 parents);
   could a clip be mapped to the wrong parent?
6. Media-absent refusal path: safe and honest?
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT.
