# WS3.0 — E-v2 unblock lane notes (ball-lane-20260723)

Date: 2026-07-24. Status: all three documented E-v2 process blockers are CLEAR at the
level verifiable locally without GPU/VM action. VERIFIED=0 stands; no GPU, no VM, no
training, no protected-row scoring. Worktree branch `ball-lane-20260723`.

## State found vs. state produced

Two of the three blockers had already been landed on origin/main after the blocker
reports were written (ledger authorization commit `c28951b`, cache-swap plan commits
incl. `8cb96fd`/`8f6e5d1`/`71d7568`, deciding review `trackD_ev2_review4_20260723`
verdict ACCEPT with `gpu_dispatch_authorized: true`). This lane's work was to VERIFY
each landing against its own gate, close the residual test/proof gaps, and produce a
fresh preflight dry-run proof under the current committed ledger.

## Blocker 1 — CACHE SWAP (landed; canonical suite now run and green)

The CACHE_SWAP_BRIEF.md rulings are present in the committed plan docs
(`runs/lanes/trackD_ev2_design_20260722/`, byte-identical between this worktree and the
main checkout):

1. BOOT image `pickleball-cache-image-20260722` family `pickleball-cache` (VM_RUN_PLAN.md:50,485).
2. ZONE `us-central1-f` with fail-closed stockout behavior: "usc1f A100 Spot create
   failed; registered failure, ABORT with no zone fallback" (VM_RUN_PLAN.md:446); usc1f
   shape/quota assert ABORT (VM_RUN_PLAN.md:136,849).
3. DATA: RO cache disk `pickleball-cache-data-usc1f` mounted read-only at /cache
   (VM_RUN_PLAN.md:497,619); conservative scope: only the six Stage-P teacher clips whose
   CACHE_MANIFEST rows carry `sha256_matches` are swapped, per-item sha256 re-verified
   before use; SHA256_MISMATCH / COMPARE_ONLY_NEVER_TRAIN / QUARANTINED* rows are
   ineligible fail-closed and `83gyqyc10y8f` is explicitly excluded
   (VM_RUN_PLAN.md:715-716,748-750,769-770; INPUT_LOCK.json `transport_notes_20260723`,
   pins unchanged: `"pins_changed": false`). Everything not proven on the disk keeps its
   registered SCP/tar transport verbatim (INPUT_LOCK.json `registered_transport_retained_verbatim`).
4. TERMINATION: teardown detaches the shared cache disk and deletes only the VM + its
   auto-delete boot disk (VM_RUN_PLAN.md:412-414,569-570).
5. LABELS lowercase `fable-fleet=pickleball,fable-lane=trackd_ev2_20260722,owner=arnavchokshi`
   (VM_RUN_PLAN.md:442).
6. TIMINGS: disk-attach staging with recomputed 20-min setup / 300-min compute / 30-min
   teardown arithmetic (VM_RUN_PLAN.md:990 region).

CODE_SHA256SUMS confirmed to list no plan docs (0 matches in the 9-line file), as the
brief required.

### Canonical E-v2 suite under pinned manifest hashes (the residual this lane closed)

Command (from the worktree root, exit 0):

```
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /Users/arnavchokshi/Desktop/pickleball/.venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/racketsport/test_ball_3d_events_gate.py tests/racketsport/test_eval_event_head.py \
  tests/racketsport/test_event_fusion.py tests/racketsport/test_event_head_anchor_candidates.py \
  tests/racketsport/test_event_head_assignment.py tests/racketsport/test_event_head_dataset.py \
  tests/racketsport/test_event_head_ev2_judge_control.py tests/racketsport/test_event_head_ev2_repair.py \
  tests/racketsport/test_event_head_eval.py tests/racketsport/test_event_head_finetune.py \
  tests/racketsport/test_event_head_matcher.py tests/racketsport/test_event_head_model.py \
  tests/racketsport/test_event_head_recipe_utils.py tests/racketsport/test_event_head_stage_p_ev2.py \
  tests/racketsport/test_event_head_training_controls.py tests/racketsport/test_event_head_training.py \
  tests/racketsport/test_event_review_ingest.py tests/racketsport/test_event_review_session.py \
  tests/racketsport/test_event_sequence_dp.py tests/racketsport/test_finetune_event_head_ev2.py \
  tests/racketsport/test_finetune_event_head.py tests/racketsport/test_measure_audio_event_alignment.py \
  tests/racketsport/test_owner_event_manifest.py tests/racketsport/test_pbvision_event_corpus.py \
  tests/racketsport/test_shot_event_eval.py tests/racketsport/test_verify_training_inputs.py
```

Result: **217 passed, 0 failed, 0 errors** (`canonical_suite_worktree.log` in this dir).

Pinned hashes the suite and preflight ran against (all sha256):

| artifact | hash |
|---|---|
| Stage-P corrected manifest `arm_b_manifest.json` (1,189 rows) | `f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7` |
| Frozen T20 checkpoint `frozen_t20_event_head.pt` | `f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082` |
| `owner_102_manifest.json` | `84a0062c776029bc33b01381add8c0b6ecbe9fc018732d6cff2bb8bdcd194e9b` |
| `data/online_harvest_20260706/rallies` (directory_tree_v1) | `a9bb912cb99deac90a5f61240b0aa164125c05d1574a0ca2f87d82e15787ec83` |
| `runs/manager/data_ledger.json` (this branch, committed) | `e3fcba22bb1281e0c2e257f6d96cbbdf45e03a595c7bd22827dc06b06310ae70` |
| `runs/lanes/trackD_ev2_design_20260722/INPUT_LOCK.json` | `ac587e75e220af99a8f766985f23fc9b7f6f3969e55ecc5eeb9eacf1b5ea129c` |
| E1 judge `scripts/racketsport/eval_event_head.py` (byte-pinned by test) | `a0c172f73231113af3c14bcfb8b91dd83415e5406ab89d0439b697d27848e22f` |
| E1 judge `threed/racketsport/event_head/matcher.py` (byte-pinned by test) | `2272a01d94a02d6663764b3fc7018f43b70bec428a8ad7c2c3fc125373149b62` |

Suite venue notes (honesty): the worktree carries only tracked content, so untracked
media/corpora fixtures were made visible locally before the run via untracked
symlinks/hardlink-copies from the main checkout (`.venv`, `data/event_public_20260713`,
`data/pbv_replay_20260720` + its `vm_pull` lane artifacts, `data/pbvision_gallery_20260719`,
`data/event_labels_owner_20260719`, `data/event_bootstrap_20260713`,
`data/online_harvest_20260706/{rallies,manifest.json,prelabels}`,
`runs/lanes/event_bootstrap_20260713/*`, the untracked `abc_experiment_20260721`
vm_pull/vm_pull_v2 artifacts). None of these local shims is committed. The identical
26-file suite also runs in the main checkout (committed tree + native data): 212 passed
with exactly 1 failure, which is the pre-existing
`test_public_eval_persists_probability_maxima_and_nonfinite_count` fixture bug this lane
fixes (see blocker 3).

## Blocker 2 — LEDGER AUTH (landed on this branch; preflight dry-run PASS)

The queue-authorization enrichment reached origin/main in commit `c28951b` ("EVENT
ledger authorization — 4 inputs queue-authorized") and is present in this branch's
committed `runs/manager/data_ledger.json`:

- `event_abc_vm_pull_20260721`: state `READY` (`ready_for_named_consumer`),
  `trainer_forbidden: false`, EVENT-use queued disposition, corrected 1,189-row manifest
  and T20 checkpoint SHA-bound.
- `event_abc_inputs_20260720` and `online_harvest_20260706`: queued training
  dispositions; harvest registration narrowed to the exact rallies root + manifest.

No ledger edit was needed from this lane; the derived views were verified instead of
regenerated: `build_data_inventory.py --check` passes ("DATA_INVENTORY.md is in sync"),
and the enriched `audit_data_utilization.py` (main working tree copy, see residual R2)
run against THIS branch's ledger + DATA_LEDGER.md reports PASS with 0
ledger/hash/view/queue violations, NEVER-QUEUED (0).

### Preflight dry-run (NO GPU, NO VM)

```
/Users/arnavchokshi/Desktop/pickleball/.venv/bin/python scripts/racketsport/verify_training_inputs.py \
  --inputs runs/ball_lane_20260723/ev2_unblock/training_inputs_ev2.json \
  --ledger runs/manager/data_ledger.json \
  --repo-root /Users/arnavchokshi/Desktop/pickleball \
  --gate-proof runs/ball_lane_20260723/ev2_unblock/gate_proof_PREFLIGHT_DRYRUN.json
```

Result: exit 0, `status: PASS`, 4/4 inputs PASS with zero reasons
(`gate_proof_PREFLIGHT_DRYRUN.json`). `--repo-root` points at the main checkout because
the T20 checkpoint bytes and the 40 rally MP4s are untracked artifacts that exist only
there; the ledger under test is this branch's committed one. Every recorded input hash
matched the pinned table above. The verifier itself was NOT modified.

Fail-closed refusals re-proven against the same ledger
(`gate_proof_REFUSAL_PROBES.json`, exit 1, all 3 probes FAIL as required):

- Old E0 audio-only manifest (`vm_pull/abc_out/arm_b_manifest.json`, old sha
  `9d3d31aa...`): refused `LEDGER_PATH_UNBOUND`.
- Compare-only pb.vision `83gyqyc10y8f`: refused `LEDGER_PROVENANCE_FORBIDS_TRAINING`,
  input bytes never read.
- Protected eval manifest: refused on state/provenance/queue, input bytes never read.

### Audio-only defect stays excluded (E0_VERDICT.md original sin)

Direct audit of the corrected `arm_b_manifest.json` (sha `f5c1e3d8...`): 1,189 rows /
1,189 events; **0 events lack a `ball_velocity_kink` independent agreement; 0 events are
audio_onset-only** (vs. 292 in the rejected 1,481-row E0 manifest). Weight tiers:
803 rows at 0.25, 386 at 0.5; totals HIT 636 / BOUNCE 553. `audio_weight_eligible`
false rows carry no audio-derived weight. Teacher rows remain `teacher_derived`,
`training_ready: false`, `verified: false` in the artifact itself (authorization lives
in the ledger, not the artifact).

## Blocker 3 — EVAL WINDOW (fix already landed; proof tests added by this lane)

The historic harness bug (hardcoded 15-frame windows against the 64-frame-context
model) is fixed in the frozen E1 judge: `eval_event_head.py` derives the eval window
from checkpoint provenance (`_checkpoint_window_frames`/`_resolve_window_frames`),
refuses any mismatched `--window-frames` request, refuses internally inconsistent
checkpoints, and `eval_owner_val` refuses rows whose context differs from the
checkpoint window. The judge files are byte-pinned to E1 commit `9bbd8011` by
`test_event_head_ev2_judge_control.py` and were NOT modified by this lane.

Tests added by this lane (TDD: written against the required properties, then run):

- `tests/racketsport/test_event_head_eval.py::test_public_eval_builds_every_window_at_the_matched_checkpoint_context`
  proves end-to-end that a 64-frame checkpoint yields exactly 64 decoded frames per
  evaluated clip (even from 128-frame source rows) and that the legacy 15-frame request
  raises.
- `tests/racketsport/test_event_head_eval.py::test_owner_val_refuses_row_context_that_differs_from_checkpoint_window`
  proves the frozen owner-41 gate refuses 15-frame rows against a 64-frame checkpoint.
- `tests/racketsport/test_event_head_eval.py::test_frozen_e1_judge_peak_pick_nms_radius_is_2`
  pins NMS radius 2 (signature default + radius-2 suppression behavior).
- `tests/racketsport/test_eval_event_head.py::test_frozen_judge_macro_f1_at_2_protocol_is_unchanged`
  pins macro-F1@±2 semantics with a hand-computed boundary case: +2-frame offset counts,
  +3-frame offset does not, `macro_f1_at_2 == (14/15 + 1.0)/2`.

Test-only repairs (no judge/production code touched):

- `test_event_head_eval.py::test_public_eval_persists_probability_maxima_and_nonfinite_count`:
  pre-existing failure reproduced byte-identically at origin/main (StopIteration in the
  pinned judge's `next(model.parameters())` because the fixture model had no
  parameters). Fixed by registering a `device_anchor` parameter on the fixture model.
- `test_verify_training_inputs.py::test_finetune_cli_refuses_forged_proof_before_training_input_read`:
  fixture wrote `gitdir: <CODE_ROOT>/.git`, which double-indirects and breaks
  `git rev-parse HEAD` when CODE_ROOT is a linked worktree (worktree `.git` is itself a
  pointer file). Fixed to resolve `git rev-parse --absolute-git-dir`. Identical
  behavior in the main checkout; assertions unchanged; the gate itself untouched.

## Residuals / preconditions for the GPU dispatch task (fail-closed, not faked)

- R1 (cloud-side, unverifiable locally): actual usc1f A100 spot capacity, the live
  cache image/disk state, and `/cache/CACHE_MANIFEST.json` on the attached disk can only
  be proven at dispatch. The plan's Step-0 asserts fail closed on all of them.
- R2 (cross-track, Track E / roadmap order 2): origin/main carries a test/script skew.
  Commit `c28951b` landed `tests/racketsport/test_audit_data_utilization.py` expecting
  `REQUIRED_CONTRACT_ASSET_IDS` etc., but the matching enriched
  `scripts/racketsport/audit_data_utilization.py` (+583/-13) is still an UNCOMMITTED
  change in the main checkout's working tree. The committed audit script cannot even
  collect that test module, and its older queue notion flags the READY corrected corpus
  as NEVER-QUEUED. This is the separately-reviewed Track-E safety-revision landing that
  VM_RUN_PLAN.md's step-0 gate already names as a dispatch-time input
  ("REVIEWED_SAFETY_REVISION_SHA ... filled by the serialized integration owner at
  dispatch time"). NOT landed by this lane: it is a governance-critical gate script
  awaiting its own adversarial review, outside this lane's fence.
  `test_audit_data_utilization.py` is therefore excluded from the canonical E-v2 suite
  above; every other ledger/gate test (test_verify_training_inputs.py, 13/13) is green.
- R3: a fresh RUN_COMMIT must be cut after R2 lands so the VM's fresh checkout carries
  ledger + gate + audit-script coherently; the preflight proof here binds ledger
  `e3fcba22...` and must be regenerated at that RUN_COMMIT per the plan (proofs expire
  in 900 s by design).

## Files

- `training_inputs_ev2.json` — the exact four-input manifest (VM_RUN_PLAN Step-0 pairs).
- `gate_proof_PREFLIGHT_DRYRUN.json` — PASS proof, 4/4 inputs.
- `gate_proof_REFUSAL_PROBES.json` — FAIL proof (expected), 3/3 refusals typed.
- `canonical_suite_worktree.log` — 217 passed, exit 0.
- Modified tests: `tests/racketsport/test_event_head_eval.py`,
  `tests/racketsport/test_eval_event_head.py`,
  `tests/racketsport/test_verify_training_inputs.py`.
