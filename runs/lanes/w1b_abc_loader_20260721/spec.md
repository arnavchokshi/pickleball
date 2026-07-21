# w1b_abc_loader_20260721 — A/B/C blockers 3+4: UNKNOWN-mask loader + agreement/B/C materializers

Codex gpt-5.6-sol xhigh. From the launch gate fix EXACTLY:
1. LOADER SCHEMA-V2 / UNKNOWN MASK: apply + finish runs/lanes/pbv_corpus_rebuild_20260720/LOADER_CHANGE_REQUIRED.diff to threed/racketsport/event_head/datasets.py — per-frame UNKNOWN mask honored (unknown frames EXCLUDED from loss, never background); schema v2 accepted, v1 still works (additive); determinism gate stays green; tests.
2. MATERIALIZERS: new scripts/racketsport/build_abc_arm_manifests.py producing from the rebuilt teacher corpus: B-arm manifest (agreement-weighted 0/0.25/0.5 per the frozen policy — agreement inputs = per-clip audio-onset times + ball-velocity-kink times, consumed from artifact files whose PATHS are parameters; SHA-bound: every consumed artifact + media file sha256 recorded) and C-arm placebo (same rows/pixels/weights, event times shuffled within rally, frozen seed). Emit needs list for the VM (which per-clip artifacts must exist). Unit-test with synthetic agreement inputs (NO real scoring).
3. PTS SHA-BINDING: extend the corpus builder output so每 PTS-verification file is sha256-bound to its media (per the gate's finding). Also emit VM_ABC_RUN.md: exact staging→audio-onset→ball-2D→agreement→arms sequence with SHA checks at each hop.
## HARD RULES
- NO commits/branches/pushes (manager commits after ultra review). VERIFIED=0. Honest reporting; misses are misses.
- **NO JUDGE PEEKING (new standing rule after a judge-contamination catch): develop against fixtures/synthetic cases ONLY. You may NOT run any GT scorer / frozen gate / protected eval during development. ONE final scored run happens later, by the manager, on frozen code. A log showing peek-tweak-rescore = automatic rejection.**
- Focused tests + wide suite (MPLBACKEND=Agg), real exit codes, attribute failures. Artifacts under YOUR lane dir.
- CROSS-SIGNAL ROW required in your report (what you consume/feed — North Star §3.1).
- Concurrent file-disjoint lanes are live — touch ONLY your fenced files.
FENCE: threed/racketsport/event_head/datasets.py, scripts/racketsport/build_pbvision_event_corpus.py (additive), NEW build_abc_arm_manifests.py, their tests, lane dir. NOT: finetune_event_head.py/owner manifest (w1a), player_selection (w2a).
