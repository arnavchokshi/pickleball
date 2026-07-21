# ultra_review_w2a_20260721 — review the honest binding rebuild (gates commit + the one-shot holdout eval)

REVIEW-ONLY, gpt-5.6-sol ultra. w2a_binding_honest_20260721 (report.json) claims the round-5 binding must-fixes are done (runs/lanes/ultra_review_binding_20260720/log.txt was the DO_NOT_COMMIT). Verify in the live tree (player_selection.py, select_players_from_pool.py, test_player_selection.py):
1. Registered SOFT fusion restored (S>=0.5 combined; NO hard court_presence gate anywhere on a destructive path) — or if a revised rule exists, it is formally preregistered from NON-protected data with the derivation documented.
2. Owner-rebind respects stitch-veto GENERALLY (vetoed fragments cannot rebind absent independent appearance provenance) — trace all rebind paths, not just the f44/f87 test.
3. The demanded regressions exist and bite: boundary-player, source-ID-reuse, owner-veto, recovery_max_speed 7.0 assert.
4. Scorer-input projection + sha256 emission correct (field-stripped, hashed).
5. NO judge peeking in the lane's log (spot-check for any GT-scorer invocation during development — the lane claims fixtures-only; verify).
6. Prior invariants intact (stitch-veto fixture, unbound-preserved, no-enrollment no-bypass, raw-UID one-to-one, interpolated export, OFF byte-identity).
OUTPUT: verdict COMMIT_OK / COMMIT_WITH_FIXES (exact) / DO_NOT_COMMIT + file:line; and a go/no-go for the ONE-SHOT preregistered holdout evaluation (untouched protected pair) — including whether the heldout_eval_ledger prereg row I will write should pin anything specific (code sha, config, scorer, thresholds).
