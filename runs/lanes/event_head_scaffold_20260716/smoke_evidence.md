# Event-head scaffold smoke evidence — 2026-07-16

Status: **PARTIAL / scoped scaffold only. `VERIFIED=0`.** Five of six
acceptance items passed. The full wide suite finished but remained red for ten
out-of-lane failures detailed below. No checkpoint is promoted.

## Disk and cache policy

- Initial live preflight: `df -h .` reported 4.0 GiB free, already below the
  prompt's 5 GiB cleanup threshold before this lane wrote anything.
- The lane therefore used on-the-fly video decode and created no frame cache.
- Before builder evidence: 9.2 GiB free. After builder evidence: 9.1 GiB free.
- Check immediately after the wide suite: 5.4 GiB free. Final report validation
  later showed 12 GiB free as concurrent workspace usage changed.
- Lane artifact directory: 15 MiB. Vendored committed-only reference: 23 MiB.
  Both are below the 300 MiB lane-artifact cap; no public media was copied.

## 1. CPU smoke training

```bash
.venv/bin/python scripts/racketsport/train_event_head.py --smoke --weights none --steps 30 --image-size 64 --window-frames 5 --out runs/lanes/event_head_scaffold_20260716/train
echo $?
# 0
```

Evidence: 30/30 losses finite; first-five mean `0.9264168978`; last-five mean
`0.4576649249`; elapsed optimizer time `3.4913s`. The batch contained three
jhong93 windows and one OpenTTGames window. Checkpoint and train manifest were
written. Checkpoint posture is `RD_ONLY` because it saw uncleared jhong93
broadcast pixels and CC-BY-NC-SA OpenTTGames pixels.

## 2. Evaluation

Public held-out slice:

```bash
.venv/bin/python scripts/racketsport/eval_event_head.py --checkpoint runs/lanes/event_head_scaffold_20260716/train/smoke_event_head.pt --mode public --out runs/lanes/event_head_scaffold_20260716/eval/public_smoke_metrics.json
echo $?
# 0
```

Protected owner seed, eval-only:

```bash
.venv/bin/python scripts/racketsport/eval_event_head.py --checkpoint runs/lanes/event_head_scaffold_20260716/train/smoke_event_head.pt --mode protected-seed --out runs/lanes/event_head_scaffold_20260716/eval/protected_seed_smoke_metrics.json
echo $?
# 0
```

The protected output contains `eval_only=true`, `review_only=true`, and
`never_training=true`. It evaluated all 50 rows: 17 HIT, 11 BOUNCE, one
separately reported `other`, and 21 negatives. Honest random-init smoke result:
HIT F1 `0.0` and BOUNCE F1 `0.0` for both k=1 and k=2. No events crossed the
0.5 threshold, so negative FP rate was also `0.0`; that is abstention, not useful
accuracy.

## 3. Dataloader/inference parity and direct tests

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_event_head_*.py -q
echo $?
# 12 passed; 0
```

The suite includes byte-identical training-dataloader versus inference
preprocessing for a real decoded sample, empirical jhong93 source-frame offset
arithmetic, matcher/model tests, and direct CLI references for all four CLIs.

## 4. Fixture fine-tune and provenance failures

```bash
.venv/bin/python scripts/racketsport/finetune_event_head.py --reviewed tests/racketsport/fixtures/event_head/reviewed_labels_v2.jsonl --manifest tests/racketsport/fixtures/event_head/dataset_manifest.json --pretrain runs/lanes/event_head_scaffold_20260716/train/smoke_event_head.pt --out runs/lanes/event_head_scaffold_20260716/finetune_fixture --steps 2 --image-size 32 --window-frames 3
echo $?
# 0
```

The direct CLI test subprocesses assert typed nonzero exits: rejected bootstrap
provenance `21`, protected-seed overlap `22`, and duplicate IDs `23`. Missing
reviewed input exits `2` with the ingest instruction. The encompassing focused
test command exited `0`.

## 5. Builder determinism

```bash
.venv/bin/python scripts/racketsport/build_event_head_dataset.py --out runs/lanes/event_head_scaffold_20260716/dataset/manifest_a.json
echo $?
# 0
.venv/bin/python scripts/racketsport/build_event_head_dataset.py --out runs/lanes/event_head_scaffold_20260716/dataset/manifest_b.json
echo $?
# 0
cmp -s runs/lanes/event_head_scaffold_20260716/dataset/manifest_a.json runs/lanes/event_head_scaffold_20260716/dataset/manifest_b.json
echo $?
# 0
```

The byte-identical manifests reconcile inventory universes exactly: jhong93
`33,791`, OpenTTGames `4,271`, ShuttleSet `36,484`. Media-present rows were 641
jhong93 clips, two OpenTT videos, and zero ShuttleSet rows.

## 6. Hygiene and wide suite

```bash
.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
echo $?
# 0
.venv/bin/python scripts/racketsport/audit_dead_code.py --root .
echo $?
# 0
python3 scripts/racketsport/audit_storage_policy.py --root . --json
echo $?
# 1
```

All four owned CLIs have direct-CLI and related tests. Dead-code audit reported
582 Python sources and zero unknown sources. Storage audit had zero unknown
large tracked/untracked files; literal exit `1` is the pre-existing stale
allowlist state (missing formerly allowlisted files), which this lane could not
edit.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q
echo $?
# 3733 passed, 10 failed, 24 skipped in 3889.20s; 1
```

Failures:

- Three `test_court_keypoint_review_server.py` tests: managed sandbox denied
  TCP bind with `PermissionError: [Errno 1] Operation not permitted`.
- Three `test_review_input_server.py` tests: same TCP-bind denial.
- Two `test_sam3dbody_persistent_worker.py` tests: managed sandbox denied Unix
  socket bind; the client accordingly could not find the worker socket.
- `test_flight_simulator.py::test_generate_corpus_is_deterministic_and_fast_for_small_cpu_sample`:
  8.079s in the wide run versus a 5s threshold; isolated rerun still failed at
  7.327s. This lane did not modify flight code or its test.
- `test_scaffold_tool_index.py::test_real_scaffold_tool_index_matches_checked_in_schema`:
  concurrent untracked `scripts/racketsport/build_event_head_anchor_candidates.py`
  was categorized `unknown`. It belongs to another active lane and is outside
  this lane's exhaustive ownership fence. This lane's four CLIs were all
  registered and categorized.

Because the last failure is concurrent rather than proven pre-existing, the
wide gate is honestly FAIL and the overall lane report is PARTIAL.

## Fine-tune command for the owner drop

```bash
.venv/bin/python scripts/racketsport/finetune_event_head.py --reviewed runs/lanes/owner_event_labels_20260715/reviewed_v2/reviewed_labels_v2.jsonl --manifest runs/lanes/owner_event_labels_20260715/reviewed_v2/dataset_manifest.json --pretrain runs/lanes/event_head_scaffold_20260716/train/smoke_event_head.pt --out runs/lanes/event_head_scaffold_20260716/owner_finetune
```

## Best-stack delta and ledger bullet

**BEST-STACK DELTA: (c) NO stack delta.** This lane is scaffold-only; no model
is promoted and `best_stack.json` is unchanged. A future manager-run GPU
pretrain checkpoint must receive its own **PENDING** `models/MANIFEST.json`
entry in that future lane before it is used as lineage.

- **2026-07-16 — event_head_scaffold_20260716:** dataset/model/eval/fine-tune
  scaffolding wired with scoped CPU smoke evidence; `VERIFIED=0`, protected seed
  remained eval-only, zero-shot owner-seed F1 was 0, no stack delta; lane PARTIAL
  because the mandatory wide suite ended 3733 passed / 10 failed / 24 skipped.
