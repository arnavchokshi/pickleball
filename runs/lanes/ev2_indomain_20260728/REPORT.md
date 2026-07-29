# E-v2 in-domain (pb.vision) — Step-0 gate FAILED closed; training correctly refused

Lane: `ev2_indomain_20260728`. Date: 2026-07-28 (evening).
Status: honest refusal, $≈0 GPU; `VERIFIED=0`. Written by the orchestrator from
the lane agent's returned findings (its harness refused report-file writes).
Evidence: `gate_proof_pbvision_20260728.json`, `training_inputs_pbvision.json`
(mirrored local + VM).

## What the lane verified first (all held)

night2 at 34.55.78.184 with the committed host-key pin; VM-side checkpoint
sha256 exact-match `e11529bc16…`; rail decoded to 2026-07-29T18:06:21Z; the
recovered cache-data mount read-only at /mnt/cachedata with all 13 pb.vision
video dirs; VM repo fast-forwarded 15 commits to origin/main cleanly.

## The refusal (the data-governance system working as designed)

`verify_training_inputs.py` on a 7-train-partition-ID manifest pointed at
`/mnt/cachedata/media/pbvision/<id>/max.mp4`: **status FAIL, 7/7 inputs**, two
reasons each:

1. `LEDGER_COMPONENT_NOT_AUTHORIZED` — `pbvision_gallery_20260719` carries
   `EVENT=CONDITIONAL`, not `ALLOW`.
2. `LEDGER_PATH_UNBOUND` — the raw `max.mp4` bytes were never hash-registered
   in the ledger (only the JSON export inventory + 8 court-review stills are).

The alternate label path, `pbv_pickleball_teacher_events_20260720`
(teacher-event corpus), is independently `state: BLOCKED`: "0/7314 windows are
locally decodable and the recovered B/C materialization accepted 292 audio-only
rows, so no valid training consumption is recorded" — its EVENT ruling is
CONDITIONAL on a non-audio-agreement/PTS-hash gate that has not passed.

The lane did NOT edit the ledger, fabricate path bindings, or drop the
`component` field to dodge the check. No training ran; nothing was uploaded.

## Unblock (owned by a data-steward lane, not a GPU lane)

The blocking facts predate tonight's media recovery — the pixels now EXIST
again on `pickleball-cache-data-usc1f`. The steward work: hash-bind the
recovered `max.mp4` bytes into the ledger (verifying against any recorded
expected hashes; `83gyqyc10y8f` stays COMPARE_ONLY_NEVER_TRAIN and is known
sha-mismatched), re-materialize the teacher-corpus decode/PTS-hash gates
against the recovered media, evaluate the EVENT `CONDITIONAL` conditions
honestly (usage rights signed 2026-07-20; agreement-filter consumption rules),
and land it as a reviewed, committed ledger change (new RUN_COMMIT). Then this
lane's staging/training/eval/S3 steps run exactly as specified. Follow-up
steward lane: `pbv_ledger_steward_20260728`.
