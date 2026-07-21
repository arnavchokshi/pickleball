# ultra_review_abc_gate3_20260721 — A/B/C launch gate, round 3 (FINAL go/no-go)

REVIEW-ONLY, gpt-5.6-sol ultra. Round-2 (runs/lanes/ultra_review_abc_gate2_20260721/log.txt) left 4 launch blockers; w1d_abc_final_20260721 (report.json) claims all 4 fixed. Verify ONLY those 4 in the live tree (rounds 1-2 resolved items need only spot-checks):
1. abc_decision_gate.py now implements the FULL registered causal gate incl. paired-bootstrap 95% lower bound, timing-p90 non-regression, negFP(B)<=negFP(A)+1, equal-final-steps — and the round-2 counterexample (B w/ +2 negFP, 1 step) now FAILS (run the test).
2. Interrupted runs cannot leave a stale finetune_manifest.json (atomic-last write or delete-on-interrupt; test).
3. C-placebo exposure parity: loss-valid frame counts IDENTICAL B vs C (mask-aware parity guard; the 64-vs-63 probe now fails-then-fixed; run the test).
4. Mandatory SHA chain (audio/ball dependency hashes REQUIRED) + VM_ABC_RUN.md complete (derivative-to-media SHA at every hop, protected-50 one-touch + refusal, final frozen eval sequence).
Also: no judge contact in w1d's log.
OUTPUT: verdict LAUNCH_OK / LAUNCH_WITH_FIXES (exact) / DO_NOT_LAUNCH + file:line; one-line GPU go/no-go.
