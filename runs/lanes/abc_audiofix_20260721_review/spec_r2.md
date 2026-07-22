# ROUND-2 ULTRA RE-REVIEW of abc_audiofix_20260721 after FIX round (read-only)

Prior review: runs/lanes/abc_audiofix_20260721_review/review.json (REJECT: BALL_FAMILY_SPOOF,
DUPLICATE_EVENT_ID_ALIAS, NULL_DURATION_GAMING, FRAME_TIMES_IDENTITY_OMISSION). Fix claim:
report_fix1.json (status BLOCKED pending VM preflight). Target: build_abc_arm_manifests.py +
test_abc_arm_manifests.py.

Re-run your original exploits EXACTLY:
1. exotic/pb.vision/audio-disguised-as-ball artifacts bound to --ball-velocity-kinks — must refuse.
2. DUP@1s/DUP@8s one-kink alias — must refuse or match uniquely; 1 vs "1" coercion — must refuse.
3. duration=1000s inflation — must refuse; verify the period now derives from verified PTS.
4. frame-times missing media identity — must refuse.
Then probe for NEW holes the fix may have opened (over-strict contracts that would reject the REAL
VM artifacts: check the pulled kink/audio samples under runs/lanes/abc_experiment_20260721/vm_pull/
and the emitting builders build_ball_inflections.py / build_audio_onsets_v2.py — do the real
artifacts carry every newly-required field? Flag any field the real emitters do NOT write).
Verify invariants: 0 audio-only; 1,189 recount; weight tiers; C parity; determinism; CLI unchanged.
Assess the lane's own BLOCKED item: is the VM preflight list (six ball-track provenance chains)
complete and correct?
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT — plus DISPATCH_DECISION: may the VM
rebuild+train proceed conditional on the stated VM preflight passing?
