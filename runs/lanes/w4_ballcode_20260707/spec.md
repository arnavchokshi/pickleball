# LANE w4_ballcode_20260707 — BALL2D stage-2 training path + SST round-1 feed (the wave-4 build gap)

## OBJECTIVE
Close the BALL2D STEP-3 BUILD GAP so a GPU lane can run (i) an owner-label seed fine-tune and
(ii) SST round 1, purely by CLI, with zero improvisation on the VM. Today
`scripts/racketsport/train_ball_pretrain.py` reads ONLY the Roboflow `corpus_index`; the
owner-CVAT loader and an SST pseudo-label feed are not wired into any trainer. INTERNAL-VAL ONLY
world: nothing you build touches held-out anything.

## EVIDENCE TO READ FIRST
- `TECH_BLUEPRINTS.md` BALL-2D pillar (STEP 3 build gap, STEP 4 SST recipe, the SCORING BRIDGE, §4
  DO-NOTs — the kill list binds you).
- `threed/racketsport/ball_tracknet_cvat_dataset.py` (`build_ball_tracknet_cvat_dataset`,
  `dense_tracknet_labels_from_cvat`, `_visibility_wbce_weight`) and
  `threed/racketsport/schemas/__init__.py` `BALL_VISIBILITY_LEVELS` + WBCE weights
  (clear=1/partial=2/full=3/out_of_frame=3).
- `scripts/racketsport/train_ball_pretrain.py` (model/opt/checkpoint/eval plumbing you will REUSE).
- The owner label export: `cvat_upload/exports/harvest_review_20260707/` (6 clip dirs + README +
  per-clip MANAGER_NOTE.md — read the visibility-remap note). ~274 human-verified boxes, sparse.
- Teacher sidecars: `data/online_harvest_20260706/prelabels/<clip>/` (40 clips, local). CHECK the
  exact per-clip file layout FIRST (ls one dir; expect WASB-style ball_track/candidates JSON) and
  the matching rally frames under `data/online_harvest_20260706/rallies/` — state what you found.

## DESIGN (pinned shape; implementation details yours)
A. NEW `scripts/racketsport/train_ball_stage2.py` — a sibling CLI that REUSES
   train_ball_pretrain.py's plumbing via imports (factor shared helpers if needed with MINIMAL
   edits to train_ball_pretrain.py that keep its CLI byte-compatible — its stage-1 behavior must
   not change; do not fork-copy hundreds of lines). Two data sources, combinable:
   - `--cvat-export-root <dir>`: owner labels via the existing CVAT loader with real per-sample
     `wbce_weight` flowing into the loss (ASSERT in a test that batch["wbce_weight"] carries 2/3/3
     values for partial/full/out_of_frame samples).
   - `--sst-manifest <json>`: pseudo-label samples from (B).
   - **SPARSE-REVIEW SEMANTICS (the trap — get this right):** the wave-3 import used sparse-review
     semantics: only REVIEWED frames carry truth; an unreviewed frame is NOT a negative. Training
     rows must come ONLY from reviewed frames (ball present OR explicitly reviewed-absent). CHECK
     how the export/loader distinguishes reviewed-absent from never-reviewed; if
     `dense_tracknet_labels_from_cvat` would fabricate absent-ball rows for unreviewed frames,
     do NOT use it for sparse exports — build the sparse path correctly. If the export format
     cannot distinguish the two cases at all, STOP and report (this poisons training if guessed).
   - Occlusion augmentation at `occluded_prob=0.25` in the training batch path, IMPLEMENTED (the
     harness lacks it), deterministic under `--seed`, and ALWAYS paired with the visibility WBCE
     (aug alone is kill-listed: RMSE 29.6→54.3).
   - Init: `--init-checkpoint` through the same load path that reports
     `missing_keys`/`unexpected_keys`; ABORT (nonzero exit, clear message) if either is non-empty.
     `frames_in` must equal the checkpoint's (3) — assert, don't assume.
   - Optimizer/recipe defaults: AdamW lr=5e-4 wd=5e-5, 512x288, radius 4, constant LR (do NOT
     invent a scheduler), checkpoint_every 500, bounded epochs (≤30 default).
B. NEW `threed/racketsport/ball_sst_dataset.py` — builds student samples from teacher sidecars +
   rally frames. Per-sample confidence weight from the teacher detection score: pin
   `weight = clamp(score, 0.0, 1.0)` (boring by design — document it; no cleverness). Emits/reads
   the `--sst-manifest` JSON (clips, frame refs, teacher xy, score, weight). Fail-closed parsing.
   PROTECTED-EVAL GUARD: before emitting any manifest, assert no source frame/video sha collides
   with the 35 protected eval hashes — REUSE the existing guard machinery (grep
   `assert_no_protected_eval_hash_collisions` in `threed/racketsport/roboflow_corpus.py` and the
   harvest dedup path; do not reimplement).
C. Disagreement emitter — NEW CLI `scripts/racketsport/export_sst_disagreements.py` (or a
   subcommand of A; your call): consumes TWO prediction sets (teacher sidecars + student
   predictions in the same JSON shape) and emits a ranked per-clip disagreement-frame queue JSON
   (frame refs + disagreement type: teacher-only / student-only / large-offset) for the P0-4 CVAT
   label queue. Unit-test with synthetic fixtures (student predictions won't exist until the GPU
   lane runs).

## ACCEPTANCE (all measured, in the report table)
1. Unit tests (new `tests/racketsport/test_ball_stage2_*.py`): WBCE weight flow (2/3/3 reach the
   loss); occlusion-aug determinism under seed + paired-with-WBCE enforcement; init key-diff abort
   on a deliberately-mismatched checkpoint; sparse-review semantics (unreviewed frame yields NO
   training row; reviewed-absent yields a negative row); SST manifest build from ONE real local
   prelabel clip (skip-if-missing pattern); protected-hash guard fires on a synthetic collision;
   disagreement emitter fixture output.
2. CPU smoke: `train_ball_stage2.py --cvat-export-root cvat_upload/exports/harvest_review_20260707
   --init-checkpoint runs/lanes/w3_p11_train_20260707/checkpoints/latest.pt --steps 20 --device cpu
   ...` runs end-to-end and training loss strictly decreases from step ~1 to ~20 (assert final <
   first). If loading the real checkpoint on CPU trips a torch/env limitation, prove it and fall
   back to a tiny randomly-initialized model config for the smoke — but say so in the report.
3. Full blast radius: `.venv/bin/python -m pytest tests/racketsport/test_ball_pretrain_harness.py
   tests/racketsport/test_ball_tracknet_cvat_dataset.py tests/racketsport/test_ball_wasb_dataset.py
   tests/racketsport/test_scaffold_tool_index.py <your new test files> -q` PLUS every test file your
   grep shows importing `train_ball_pretrain` (list them; run them ALL). Register new CLIs in the
   scaffold index in this SAME lane.

## OWNED FILES (anti-collision fence)
NEW: `scripts/racketsport/train_ball_stage2.py`, `threed/racketsport/ball_sst_dataset.py`,
`scripts/racketsport/export_sst_disagreements.py`, `tests/racketsport/test_ball_stage2_*.py`.
EDIT-MINIMAL: `scripts/racketsport/train_ball_pretrain.py` (shared-helper factoring only; stage-1
CLI byte-compatible). READ-ONLY: `ball_tracknet_cvat_dataset.py` unless a sparse-semantics fix is
REQUIRED there — if so, keep it surgical and test it. DO NOT TOUCH: `ball_arc_solver.py` (another
live lane owns it), `run_wasb_ball.py`, `fuse_ball_tracks.py`, `run_ball_tracking_eval_suite.py`,
`process_video.py`/`orchestrator.py` (fenced), `ios/**`, `runs/manager/**`, eval labels, ledger.

## KILL
- Sparse-review semantics unresolvable from the export format → STOP: report with evidence.
- CVAT loader structurally cannot feed the trainer without >1-day rework → STOP: propose design.

## DISCIPLINE
`.venv/bin/python`; `pytest.importorskip("torch")` for torch tests; no git branch/commit/push; no
network (weights/data are local or the GPU lane's job); never touch Outdoor/Indoor/held-out
anything; no new root-level .md; prove any pre-existing failure fails at HEAD.

## STRUCTURED REPORT
Acceptance table per item above; CHANGES file:line; full_suite counts with named failures;
honest_issues (esp. any semantics you had to interpret); next = the exact CLI invocations the GPU
lane should run (fine-tune, SST manifest build, student train, disagreement export).
