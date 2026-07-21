# ultra_review_abc_gate4_20260721 — A/B/C gate round 4 (verify ONLY the round-3 blocker-4 sub-items)

REVIEW-ONLY, gpt-5.6-sol ultra, TIGHT: rounds 1-3 resolved everything except blocker 4; w1e_abc_chain_20260721 (report.json) claims its 4 sub-items fixed. Verify ONLY these in the live tree (+one spot-check that nothing previously-fixed regressed):
1. build_audio_onsets_v2.py + ball_inflections payloads now emit media/PTS sha256 identity the materializer's mandatory-hash check accepts (trace producer→materializer end-to-end on a fixture).
2. Running build_abc_arm_manifests.py can NEVER write/overwrite VM_ABC_RUN.md (the stale-generator import/write path is gone; test proves runbook untouched).
3. eval_event_head.py --mode owner-val: consumes the owner_102 val split, emits per-seed/arm macro-F1@±2 per class + negFP + timing-p90 + firing rate — exactly what abc_decision_gate.py consumes; deterministic on fixtures.
4. One-touch guard: atomic mkdir lock + correct grep exit-code branches (0=REFUSE,1=proceed,>=2=refuse); the three branches tested.
Also: no judge contact in w1e's log.
OUTPUT: LAUNCH_OK / LAUNCH_WITH_FIXES (exact) / DO_NOT_LAUNCH + file:line; one-line go/no-go.
