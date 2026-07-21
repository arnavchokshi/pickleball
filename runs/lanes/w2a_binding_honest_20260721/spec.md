# w2a_binding_honest_20260721 — people binding done HONESTLY (ULTRA; fixture-only development)

Codex gpt-5.6-sol ULTRA. Re-land the binding improvement per the review's exact must-fixes (runs/lanes/ultra_review_binding_20260720/log.txt), starting from the preserved patch (runs/lanes/trkL_binding_fix_20260720/binding_fix_UNCOMMITTED.patch) as REFERENCE (not gospel):
1. FUSION RULE: restore the registered soft fusion (S>=0.5 combined; no unregistered hard court_presence>=0.5 gate) OR formally preregister a revised rule derived from NON-protected data (document the derivation; the registered thresholds stay).
2. OWNER-REBIND MUST RESPECT STITCH-VETO: no rebind of stitch_vetoed fragments absent independent (appearance) provenance — the general invariant, tested beyond f44/f87.
3. Regressions the review demanded: boundary-player, source-ID-reuse, owner-veto, recovery_max_speed_m_s==7.0 default assert.
4. Archive/hash exact scorer INPUTS support: the selection CLI emits the field-stripped scoring projection with its sha256 alongside the full output.
5. DISCLOSE in your report that the prior round was scorer-guided; your development uses FIXTURES ONLY (the committed diagnosis JSONs + synthetic cases). **ABSOLUTE: never run person_track_gt_scoring or any GT during this lane.** The one-shot evaluation happens later on the UNTOUCHED holdout pair via heldout_eval_ledger prereg (manager runs it).
## HARD RULES
- NO commits/branches/pushes (manager commits after ultra review). VERIFIED=0. Honest reporting; misses are misses.
- **NO JUDGE PEEKING (new standing rule after a judge-contamination catch): develop against fixtures/synthetic cases ONLY. You may NOT run any GT scorer / frozen gate / protected eval during development. ONE final scored run happens later, by the manager, on frozen code. A log showing peek-tweak-rescore = automatic rejection.**
- Focused tests + wide suite (MPLBACKEND=Agg), real exit codes, attribute failures. Artifacts under YOUR lane dir.
- CROSS-SIGNAL ROW required in your report (what you consume/feed — North Star §3.1).
- Concurrent file-disjoint lanes are live — touch ONLY your fenced files.
FENCE: threed/racketsport/player_selection.py, scripts/racketsport/select_players_from_pool.py, tests/racketsport/test_player_selection.py, lane dir. Nothing else.
