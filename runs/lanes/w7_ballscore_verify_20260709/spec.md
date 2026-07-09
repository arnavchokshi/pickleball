# LANE w7_ballscore_verify_20260709 — ADVERSARIAL VERIFY of w7_ballretrain (PRE-STAGED; dispatch after its REPORT.md lands)

## HARD RULES
Read-only on the repo: you may NOT edit any repo source. Your executable defect proofs (scripts/tests) live under runs/lanes/w7_ballscore_verify_20260709/ ONLY. No commits. Protected clips untouchable. Honest reporting; a stub finding ("looks fine") without executed evidence is a failed mission.

## MISSION
Attack the w7_ballretrain_20260709 results (runs/lanes/w7_ballretrain_20260709/REPORT.md + pulled artifacts) before the manager rules on base choice. You are trying to REFUTE the lane's claims. Known fraud classes to hunt (each caught a real would-have-shipped defect in waves 3-6):
1. CIRCULAR CONTROL: does the control row's reproduction share code/artifacts with the candidate scoring such that both would move together? Verify control numbers came from a fresh scoring pass, not copied from wave-6 outputs (checksum/mtime provenance on the VM-pulled artifacts).
2. PROVENANCE: are the 4 fine-tuned checkpoints actually DISTINCT models (md5s differ; per-arm loss curves exist; steps counts match the budget formula)? Numbers matching a prior run TO THE DIGIT = the "re-ran the diagnosis" tell.
3. LoSO INTEGRITY: recompute fold disjointness from runs/lanes/w6_labelingest_20260708/loso_fold_manifest.json vs the corpus manifest md5 37a5d43ab537a15bd12d382bb882a5fe; confirm the OUTDOOR fold was actually scored (per-fold rows present, none silently dropped). Any excluded frame/fold needs a reason that does NOT reference any score threshold (gate-referencing exclusion = definitionally circular, forbidden).
4. CONTRACT CHECK: did the train/inference tensor-contract check actually execute (evidence artifact, not a claim)? If the lane says "identical tensors", find the assertion output.
5. METRIC KEYS: acceptance numbers use the harness's exact keys; any paraphrased statistic (e.g. a different F1 aggregation than wave-6's micro-F1) = wrong-statistic class.
6. 486-ANOMALY ARM: was it run under the W5 protocol (same folds/corpus), or a lookalike? A protocol swap invalidates the lineage claim.
Write EXECUTABLE proofs for anything you allege (a script the manager can run that demonstrates the defect). Verdict per claim: CONFIRMED-VALID / REFUTED (with proof) / UNVERIFIABLE (with what's missing).

## REPORT
Self-write runs/lanes/w7_ballscore_verify_20260709/report.json (lane_report.schema.json structure): verdict table per attack 1-6, defect proofs' paths, honest_issues, next. The manager scores any repair against YOUR unmodified harness.
