# ROUND-2 RE-REVIEW (retry) of abc_audiofix_20260721 — correctness verification (read-only)

Prior review: runs/lanes/abc_audiofix_20260721_review/review.json (REJECT with four findings).
Fix claim: report_fix1.json (status BLOCKED pending VM preflight). Target files:
scripts/racketsport/build_abc_arm_manifests.py + tests/racketsport/test_abc_arm_manifests.py.

Verify each prior finding is closed by exercising the validation logic with NEGATIVE INPUTS
(malformed or mislabeled artifacts must be refused). Use the existing test suite where possible;
prefer reading the code and the shipped tests over writing new fixture programs:
1. Family/provenance validation: a JSON with the wrong artifact_type/source/world_frame bound to
   the --ball-velocity-kinks flag must be refused; same for the audio flag. Confirm the shipped
   tests cover wrong-type, wrong-source, and case-variant family strings.
2. Event-ID uniqueness: duplicate or coercion-colliding event IDs must be refused; matching uses
   internal indices with rechecked deltas. Confirm test coverage.
3. Null-window period: the circular-shift period must derive from verified PTS; an inconsistent
   declared duration must be refused. Confirm the inflated-duration refusal test exists and is real.
4. Frame-times identity: missing media declaration must be refused; declared vs staged identity
   recorded separately. Confirm coverage.
5. OVER-STRICTNESS CHECK (important): compare the newly required fields against the REAL artifacts
   (pulled samples under runs/lanes/abc_experiment_20260721/vm_pull/ and the emitters
   build_ball_inflections.py / build_audio_onsets_v2.py). List any required field the real
   emitters do NOT produce — that would wrongly block the real VM rebuild.
6. Invariants: 0 audio-only accepted; 1,189 corrected recount; weight tiers (0.25/0.5/0.25/0.0);
   C parity; determinism; CLI flags unchanged.
7. Is the lane's VM preflight list (six ball-track provenance chains) complete and sufficient?
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT, plus DISPATCH_DECISION: may the VM
rebuild+train proceed conditional on the stated VM preflight passing?
