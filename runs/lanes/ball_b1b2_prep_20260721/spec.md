# LANE ball_b1b2_prep_20260721 — B1 SST builder + B2 trainer/scorer plumbing (CPU code; GPU runs come later)

## HARD RULES
- No branches/commits. Read runs/regroup_20260721/EXACT_PLAN.md §3.2 B1+B2 and §2.1 (BALL teacher
  eligibility ruling) first. pb.vision compare-only IDs (83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr)
  must be structurally unreadable by the builder. Protected clips EVAL-ONLY.
- Honest reporting; WIDE suite (MPLBACKEND=Agg, full tests/racketsport), exact counts; known
  environmental failure set (~31) is pre-existing — no NEW failures.
- Artifacts under runs/lanes/ball_b1b2_prep_20260721/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/build_pbvision_ball_sst.py (new)
- scripts/racketsport/train_ball_stage2.py (modify: add --sst-manifest consumption if absent,
  --sst-batch-size, --sst-loss-cap)
- scripts/racketsport/ball_loso_validation.py (modify: add parent-source grouping mode)
- tests/racketsport/test_build_pbvision_ball_sst.py (new) + minimal additions to the existing
  trainer/validation test files for the new flags/mode ONLY.
Nothing else. Do NOT touch build_ball_regroup_split.py (owned by ball_b0_split fix lane).

## OBJECTIVE (EXACT_PLAN B1+B2, verbatim contracts)
1. build_pbvision_ball_sst.py per the B1 command block (gallery-root, media-root, split-manifest
   runs/lanes/pbv_pickleball_corpus_20260720/manifest.json, wasb-checkpoint,
   --teacher-confidence-min 0.90 --agreement-radius-px 20 --pseudo-weight 0.25). Requirements:
   only the 7 frozen train IDs (143sf3gdwxsa, 98z43hspqz13, bewqc0glhgpq, st0epgnab7dr,
   td2szayjwtrj, tqjlrcntpjvt, xkadsq9bli3h); pldtjpw3h0jw/utasf5hnozwz teacher-val-only,
   0tmdeghtfvjx teacher-test-only, 3 compare IDs UNREAD (hard structural refusal + test);
   SHA-bind media and PTS; positives only; teacher absence is NEVER a negative; eligibility
   requires frozen-WASB spatial agreement (<=20px at conf>=0.90) OR the preregistered
   temporal/geometry check (define it precisely in-code + docstring; preregister parameters in the
   emitted manifest); every row teacher_derived=true, ground_truth=false, with agreement reason +
   dependency hashes. Emitted manifest must satisfy train_ball_stage2.py --sst-manifest schema.
   B1 GATE fields the CLI must self-report: accepted_windows (>=1,000 target), accepted_sources
   (>=5 target), holdout_rows_present (must be 0), decode_failures.
2. train_ball_stage2.py: --sst-batch-size N (B receives the exact same 8 human rows/step as A;
   pseudo adds its own batch) and --sst-loss-cap 0.25 (pseudo loss <=25% of human loss per step,
   post-weighting); deterministic; a no-SST invocation must remain BYTE-IDENTICAL in behavior to
   today (A-arm parity — prove with a test comparing short-run losses with/without the new flags
   absent).
3. ball_loso_validation.py: parent-source mode — group scoring by original parent source video
   (not per-clip); per-source and pooled F1@20/recall/precision/hidden-FP; consumes B0's split
   artifact format (read it from runs/lanes/ball_b0_split_20260721/split/ if present, else fixture).
4. Tests: fixtures for eligibility (agree/disagree/low-conf/absent-teacher), compare-ID refusal,
   SST batch/loss-cap math, parent-source grouping, no-SST parity.

## DATA CONTRACT
CPU only; media absent locally is EXPECTED (builder must fail gracefully listing missing media —
the GPU lane stages media on its VM from public GCS). End-of-lane number: test counts + a fixture
dry-run of the builder emitting a valid empty-media refusal + schema-valid sample manifest.

## CROSS-SIGNAL
Consumes: WASB 2D, pb.vision teacher ball, PTS chain. Feeds: BALL B2 A/B arms, parent-source scorer,
event-head ball-kink family (shared upstream).

## BEST-STACK DELTA
None yet — B2's arms are candidates; any stack change waits for the B2 gate + review.

## MANDATORY STRUCTURED REPORT
objective_result; full_suite; HONEST ISSUES; artifacts; the parity proof result.
