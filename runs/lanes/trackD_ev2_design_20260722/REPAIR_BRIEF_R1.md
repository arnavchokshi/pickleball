# Repair round 1 — Track-D manager rulings on review findings (trackD_ev2_review_20260722)

Read the FULL review report first: runs/lanes/trackD_ev2_review_20260722/report.json (16 findings,
each with file:line evidence + minimal fix). Fix ALL of EV2-R1..R4 and EV2-F1..F9 per the
reviewer's minimal-fix text, EXCEPT where a manager ruling below overrides/decides a design choice.
One repair round; your fixes will be re-scored by the SAME reviewer's unmodified harness
(including the R1 synthetic proof). Re-register every changed bound/cardinality/policy in
REGISTRATION.md — the registration must match the code EXACTLY after repair.

## Manager rulings per REJECT-class finding

- EV2-R1 (vacuous p90 gate): DO NOT touch the judge. Fix = REMOVE the standalone
  "timing p90 <= 2f" row from the Stage-V gate table and add an explicit registration note that
  macro-F1@+-2 > 0 mathematically implies every matched timing error <= 2f (subsumption), with the
  no-matches sentinel (=window_frames) documented; note E1 comparability (E1-B's p90=2 was equally
  implied; E1-A/C's 64 = the zero-match sentinel). Do NOT invent a wider-tolerance timing gate —
  that would require judge changes, forbidden.
- EV2-R2 (held-out source leak): exclude ALL held-out-source (st0epgnab7dr) rows from the
  hard-negative candidate pool. Re-register the pool cardinality with exact new counts (expected
  292 - 30 = 262; ASSERT and record the actual). Top-96 selection unchanged, tie-breaks unchanged.
  Add a hard assertion: no hard-negative row's video_id equals the Stage-P held-out source.
- EV2-R3 (5+4 batches): enforce EXACT 8 owner + 4 hard-negative composition on every Stage-F step
  (deterministic wrap-around/reshuffle of owner windows at epoch boundary, seeded — NOT drop-last
  silent step-count change; total steps stay 1000). Re-register the sampler policy text.
- EV2-R4 (silent legacy defaults): final-step mode must HARD-REQUIRE the registered recipe values
  (sqrt-frequency weighting, dilation=1, temporal weight 0.25, offset loss 0.2, Hungarian OFF in
  Stage F as registered) — typed error if any is absent or divergent; a final-step run without the
  full registered recipe must be impossible, not merely unlikely. Add the missing test asserting a
  divergent-recipe final-step run FAILS.

## FIX-before-commit items
Apply the reviewer's minimal fixes verbatim where unambiguous (F2 threshold-lock hard assert, F5
reviewed-bytes-in-commit preflight, F6 complete teardown routes, F7 price-proof source rule, F8
KILL-backed hard walls, F9 atomic PASS-only best-stack step). For F1: restructure the owner-manifest
read to sanitize/strip validation-row event fields BEFORE any token/whole-file scan (split-only
projection first; hash checks may use bytes but must be documented as content-blind). For F3: make
the firing-rate media inventory independent of owner-manifest conditioning or re-register the exact
subset rule with its derivation; state which you chose and why.

## Constraints unchanged
No judge edits (eval_event_head.py/matcher.py byte-frozen); fences as before; focused suite must
stay green + new tests for each fix; update CODE_SHA256SUMS + CONTROL_RESULTS.json; report via the
same schema with a per-finding fix table (finding id -> commit-ready evidence).
