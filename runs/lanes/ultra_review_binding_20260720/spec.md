# ultra_review_binding_20260720 — final review of the binding fix (gates commit + the one-shot GPU card)

REVIEW-ONLY, gpt-5.6-sol ultra. The binding fix (runs/lanes/trkL_binding_fix_20260720/report.json) claims a projection result EXCEEDING the design counterfactual: wolverine bound-slot spectFP 47→0, switches→0, cov4 0.2433→0.7800 (CF 0.6167); burlington clean; OFF byte-identical; thresholds untouched; 74 tests. Verify it is REAL and not gamed before the manager commits and burns the one-shot GPU card.

VERIFY (live tree: threed/racketsport/player_selection.py, tests/racketsport/test_player_selection.py):
1. HOW did cov4 jump 0.24→0.78? Trace the binding/recovery changes — is coverage built from REAL pool detections (Layer-C recovery of real UIDs) or any synthesis/relabeling? Every recovered frame must map to a real raw-detection UID; sample-verify from the lane's projection artifacts.
2. spectFP 47→0: are spectator fragments now correctly rejected by appearance+geometry, or excluded by something that would ALSO exclude real players on other clips (over-filtering)? Check the burlington bounded-export caveat doesn't hide over-filtering (128 near-miss localization FPs are a DIFFERENT axis — confirm).
3. No threshold changes (diff-verify against registered values); no test weakened; OFF byte-identity test genuine.
4. The projection harness itself: same scorer as the frozen card, honest CPU-embedding caveat, no judge contamination (nothing fit against the scorer output).
5. Prior 8 findings from earlier reviews: still resolved (spot-check stitch-veto, unbound-preserved, raw-UID one-to-one, no-enrollment no-bypass, interpolated export).
6. The report's own limitation list is accurate (seed_player_id provenance robustness note).

OUTPUT: verdict COMMIT_OK / COMMIT_WITH_FIXES (exact) / DO_NOT_COMMIT + file:line evidence; explicit answer per item; one-line GPU-card go/no-go recommendation.
