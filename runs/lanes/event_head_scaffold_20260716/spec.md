# Lane event_head_scaffold_20260716 — Track G: event-head training+eval scaffold (dataset / model / eval / fine-tune entrypoint / CPU smoke)

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 is binding; "wired"/"scoped pass" language at
most; nothing you produce is a promotion.

## HARD RULES
- NO branches, NO commits, NO pushes. The manager rules and commits fence-only.
- Read first: NORTH_STAR_ROADMAP.md (§2.2 BALL + EVENTS/PHYS rows), AGENTS.md, RUNBOOK.md,
  runs/research_eventdata_20260713/CROSSCHECK_RULING.md (the ruled recipe — binding),
  data/event_public_20260713/INVENTORY.md + licenses_ledger.json,
  runs/manager/inflight_lanes.md (live-lane fences).
- CONCURRENT-LANE FENCES (hard): Track A is LIVE editing threed/racketsport/ball_arc_solver.py,
  ball_arc_chain.py, tests/racketsport/test_ball_arc_solver.py, test_ball_arc_scale_guard.py —
  do not touch. Track D is LIVE in ios/** — do not touch. scripts/racketsport/process_video.py is
  FORBIDDEN (integration-owner file; this lane needs no runner hunks). Preserve ALL unrelated dirty
  worktree state (git status shows Track A/D edits — leave them byte-identical).
- PROTECTED DATA (absolute):
  - runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json +
    owner_spot_check_results_20260715.json = the 50-row PROTECTED EVAL SEED. READ-ONLY,
    EVAL-ONLY, NEVER training, never copied into any training manifest. Your eval outputs that
    touch it are stamped `"eval_only": true, "review_only": true, "never_training": true`.
  - data/event_bootstrap_20260713/** Tier-A/B bootstrap auto-labels FAILED the owner gate
    (29/50) and are DEAD as a label source. No training path may read them. Your fine-tune
    entrypoint must HARD-FAIL (typed nonzero exit) on any input carrying their provenance.
  - The 4 protected clips (Outdoor/Indoor) and cvat_upload/**, brand-exploration/** untouched.
- DISK: host volume is 99% full (~7.2GiB free). NO bulk frame materialization. Decode frames
  on-the-fly in the dataloaders. Any cached artifact ≤300MB total, under your lane dir, counted
  in your report. If free disk ever <5GiB from your own writes, delete your caches and go pure
  on-the-fly. Check `df -h` before/after.
- No network assumption: the sandbox has no network. Backbone weights default `--weights none`
  (random init) for smoke; accept `--weights imagenet` for later GPU use (documented, untested
  here is fine — but the flag path must exist and fail loudly if download impossible).
- Every new CLI ships its direct-CLI reference test same-lane AND all three hygiene checks must
  pass at the end (an unregistered CLI turns the whole wide suite red — this bit us yesterday):
  `.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .` EXIT 0
  `.venv/bin/python scripts/racketsport/audit_dead_code.py --root .` EXIT 0
  `python3 scripts/racketsport/audit_storage_policy.py --root . --json` — no NEW violations vs
  its current output (a pre-existing stale-allowlist failure exists repo-wide; document, don't fix).
  Read scripts/racketsport/list_scaffold_tools.py first to learn the registration mechanism
  (RELATED_TEST_OVERRIDES / SCHEMA_OVERRIDES) and register every new CLI properly.
- Run the WIDE blast-radius suite at the end (`MPLBACKEND=Agg .venv/bin/python -m pytest
  tests/racketsport -x -q` is NOT acceptable — no -x, full run), report real exit code +
  passed/failed counts; failed>0 while claiming PASS = rejected unless every failure is proven
  pre-existing (name them; current baseline ~3684 passed / 24 skipped, plus known sandbox
  socket-bind denials).
- All lane artifacts under runs/lanes/event_head_scaffold_20260716/. No new root .md files.
  NEVER a .patch deliverable.
- Honest reporting. Expect near-zero zero-shot transfer to pickleball — report it as-is.

## FILE OWNERSHIP (exhaustive — you may create/edit ONLY these)
- threed/racketsport/event_head/ (NEW package — all files yours)
- scripts/racketsport/build_event_head_dataset.py (NEW)
- scripts/racketsport/train_event_head.py (NEW)
- scripts/racketsport/finetune_event_head.py (NEW)
- scripts/racketsport/eval_event_head.py (NEW)
- scripts/racketsport/list_scaffold_tools.py (ONLY the registration dict entries for your 4 CLIs)
- tests/racketsport/test_event_head_*.py (NEW) + tests/racketsport/fixtures/event_head/ (NEW)
- third_party/spot (NEW vendored clone — see Deliverable 2) + third_party/VENDOR_PINS.md (add ONE row)
- runs/lanes/event_head_scaffold_20260716/** (lane dir)
Everything else read-only. If you believe you need another file, STOP and record it in the report
as a blocker instead of editing it.

## Context you do not need to rediscover (manager-verified tonight)
- E2E-Spot reference repo is ALREADY ON DISK as a git clone:
  data/event_public_20260713/jhong93_spot/ @ sha edec4201471beed631bed374bd0b95fcdc8a2f4f,
  origin https://github.com/jhong93/spot, BSD-3 LICENSE. Contains model/, util/eval.py (the
  ±frame-window eval protocol), train_e2e.py, data/tennis/{train,val,test}.json + class.txt,
  and videos_pilot/ (6 of 28 source videos on disk, 360p, audio kept).
- jhong93 tennis label format: list of clip dicts {video, fps, height, width, num_frames,
  num_events, events:[{frame, label}]}, clip name embeds the source-video absolute frame range
  (e.g. usopen_2020_womens_final_osaka_azarenka_166072_166372). Total 33,791 events
  (16,277 bounce via *_bounce; 17,514 hit via *_swing + *_serve). Only clips whose parent video
  is one of the 6 in videos_pilot/ are trainable tonight — count both universes honestly.
  Follow the vendored repo's own dataset/frame code for the clip→video frame-offset convention;
  validate it empirically (decode one labeled frame + neighbors, assert ball-near-court-surface
  style sanity is NOT required — just assert the convention matches the reference code paths).
- OpenTTGames on disk: data/event_public_20260713/openttgames/ videos/{game_4.mp4 (train),
  test_2.mp4 (test)} + markup/ (zips + extracted/). Events: bounce 1,777 / net 1,350 /
  empty_event 1,144 (bounce→BOUNCE; net+empty→background). License CC BY-NC-SA = RD_ONLY_STRICT.
- Extended OpenTTGames: data/event_public_20260713/extended_openttgames/ — stroke frame = CONTACT
  frame (evidence in data/event_public_20260713/ANSWERS.md Q2) → HIT supervision where it covers
  the on-disk videos. Same NC license.
- ShuttleSet (MIT): data/event_public_20260713/coachai_shuttleset/ShuttleSet — 36,484 hit strokes,
  NO media on disk → loader + manifest rows with media_absent=true, excluded from tonight's
  trainable set, included in the label-universe counts.
- License ledger: data/event_public_20260713/licenses_ledger.json. Every dataset manifest row and
  every checkpoint manifest you write carries the license posture; NC-flagged sets stamped
  RD_ONLY; any checkpoint trained on ANY broadcast pixels (incl. jhong93 videos) is stamped
  `"license_posture": "RD_ONLY"` with a one-line reason. Put the posture in code comments at each
  loader too.
- Owner fine-tune labels (NOT YET ON DISK — arrive within a day): produced by
  scripts/racketsport/ingest_event_review_results.py into
  runs/lanes/owner_event_labels_20260715/reviewed_v2/reviewed_labels_v2.jsonl +
  dataset_manifest.json. Row schema (verbatim from the ingest CLI, schema_version 2): label_id,
  clip_id, source_group, video_path, video_sha256, anchor_pts_s, stratum, score_band,
  decision ∈ {paddle,ground,other,none,unclear}, contact {x_norm,y_norm,x_px,y_px,source_width,
  source_height}|null, dt_s|null, corrected_contact_pts_s|null, suggested_split, review{session_id,
  reviewed_by, ingested_at}, provenance{seed, generator_version, generator_sha256, manifest_sha256,
  results_sha256}. video_path roots at data/online_harvest_20260706/rallies/<source>/… (all 6
  source dirs verified present). paddle→HIT, ground→BOUNCE, other→excluded-from-typed (reported),
  none/unclear→hard negatives.
- Protected seed GT construction (for eval ONLY): spot_check_tier_a_50.json has 50 labels with
  anchor{frame, pts_s (normalized to first video pts), proposal_center_time_s} + clip ids;
  owner_spot_check_results_20260715.json has answers keyed "1"-"50" with decision/x/y/dt.
  corrected_contact_pts_s = anchor.pts_s + dt. 29 typed contacts (17 paddle / 11 ground /
  1 other) + 21 negatives (none). Source videos are the same online_harvest rally clips.
- .venv has torch 2.12.1 + torchvision 0.27.1 (CPU), Python 3.14. Use .venv/bin/python for
  everything.

## Deliverables

### 1. Dataset layer — threed/racketsport/event_head/datasets.py (+ build_event_head_dataset.py CLI)
Unified sample schema: (frames[T,C,H,W] at modest resolution — pick and pin e.g. short-side 224,
document), per-frame targets over classes {0:background, 1:HIT, 2:BOUNCE}, per-sample per-class
loss-validity mask (loss-masked union per the ruling: a source that only labels BOUNCE masks the
HIT column and vice versa; jhong93 labels both → full mask on).
Sources v1: jhong93/spot (media-present subset), OpenTTGames (+Extended stroke HIT where it maps),
ShuttleSet label-only rows. Deterministic splits: keep jhong93's canonical train/val/test; for
OpenTT keep its train/test video identity; split assignment must be source-video-disjoint and
byte-identically reproducible from a seed (prove: run the builder twice, diff manifests).
The builder CLI writes a dataset manifest (JSON under your lane dir or a path arg): per-row source,
video, media_present, split, event counts by class, license posture, and totals that reconcile
EXACTLY against INVENTORY.md numbers above (33,791 / 4,271 / 36,484 — assert in a test).

### 2. Model layer — vendor + head
- Vendor the reference: `git clone data/event_public_20260713/jhong93_spot third_party/spot`
  then `git -C third_party/spot checkout edec4201471beed631bed374bd0b95fcdc8a2f4f` (local clone,
  no network; committed content only — media stays out). Add the VENDOR_PINS.md row (dir, pinned
  sha, origin https://github.com/jhong93/spot.git, role: E2E-Spot reference for event spotting —
  eval protocol + architecture reference; labels BSD-3; local-only large files: none).
- threed/racketsport/event_head/model.py: OUR compact 2-class+background temporal spotting head in
  the E2E-Spot family — per-frame 2D feature backbone at modest resolution (small torchvision
  backbone, weights arg as above) + temporal head (GRU) + per-frame 3-way logits; masked
  cross-entropy honoring the per-class validity mask; provenance header comment citing
  third_party/spot@edec4201 as the architectural reference. Do NOT import third_party/spot code
  into the production path (their repo has dataset-specific deps); it is vendored as reference +
  eval-protocol source. Keep it small enough that a CPU smoke step runs in seconds.

### 3. Eval layer — threed/racketsport/event_head/matcher.py + eval_event_head.py CLI
- Type-aware tolerance matcher: greedy one-to-one matching of NMS-peaked predictions to GT events,
  same type AND within ±k frames (default k=2 at the source fps; report a small tolerance sweep
  k∈{1,2} plus the ms equivalent). Per-class precision/recall/F1 + matched timing-error stats.
- Public held-out eval: jhong93 val/test media-present clips (+ OpenTT test_2) with the vendored
  repo's protocol as reference.
- PROTECTED owner-seed eval (eval-only mode): for each of the 50 rows decode
  [anchor_pts_s−1.0s, +1.0s] from the named rally video, run the model, peak-pick; typed contacts
  match at same-type ±2 frames vs corrected_contact_pts_s; the 1 'other' row reported separately;
  negatives: any prediction within ±0.3s of anchor = false positive (report FP rate over the 21).
  Output JSON stamped eval_only/review_only/never_training, under runs/lanes/…/eval/. The CLI
  refuses to run seed eval in any mode that could write training artifacts.

### 4. Fine-tune entrypoint — finetune_event_head.py CLI (one-command for the owner drop)
- Consumes --reviewed <reviewed_labels_v2.jsonl> --manifest <dataset_manifest.json> --pretrain
  <ckpt> --out <dir>. Validates schema_version 2, uses suggested_split, ASSERTS source-disjoint
  splits (source_group never straddles splits — typed failure otherwise).
- PROVENANCE HARD-FAILS (typed nonzero exits, tested): (a) any row whose provenance/paths/dataset
  version matches Tier-A bootstrap (event_bootstrap_v0*, data/event_bootstrap_20260713/**, tier
  fields); (b) any row overlapping a protected-seed anchor (same source video within ±0.75s of any
  of the 50 seed anchors — load the seed read-only to check); (c) missing/duplicate label_ids.
- If the reviewed file is absent: clear actionable message, nonzero exit.
- Trains HIT/BOUNCE on contact rows + hard negatives from none/unclear rows; window sampling
  around corrected_contact_pts_s with background context; config recorded in the out-dir manifest
  with full provenance chain (input shas, pretrain ckpt sha, git head).
- Because the real file doesn't exist yet: prove with fixtures (schema-exact tiny
  reviewed_labels_v2.jsonl marked synthetic_fixture, pointing at a tiny locally-generated test
  video under tests/racketsport/fixtures/event_head/) — full tiny fine-tune step must run CPU
  EXIT 0, plus the three hard-fail cases each proven by test.

### 5. CPU smoke (the acceptance gate — all real unpiped exit codes)
1. `train_event_head.py --smoke` (tiny subset: e.g. 2-3 jhong93 media-present clips + 1 OpenTT
   window; short T; random-init weights): ≥30 optimizer steps, every loss finite, mean of last 5
   losses < mean of first 5 (tiny-overfit sanity), checkpoint + train manifest (with license
   posture RD_ONLY + provenance) written. EXIT 0.
2. `eval_event_head.py` with that smoke checkpoint: (a) tiny public held-out slice → metrics JSON;
   (b) protected owner seed eval-only → metrics JSON (expect ~0 — report honestly). Both EXIT 0.
3. Dataloader↔inference parity (MANDATORY training-lane item): push ONE sample through the training
   dataloader AND the eval/inference preprocessor; assert identical tensors (test).
4. `finetune_event_head.py` fixture run EXIT 0 + all three provenance hard-fails EXIT nonzero (tests).
5. Builder determinism: two runs byte-identical manifests.
6. Hygiene trio + wide suite as in HARD RULES.
Record every command + real `echo $?` in the report and in runs/lanes/event_head_scaffold_20260716/
smoke_evidence.md.

## Kill criteria (report BLOCKED honestly instead of faking)
- jhong93 clip→video frame-offset convention cannot be reconciled with the vendored reference code.
- CPU smoke train step cannot complete ≥30 steps in reasonable wall time even at minimum config.
- Disk pressure makes even on-the-fly decode infeasible.

## BEST-STACK DELTA (mandatory in report)
Expected: (c) NO stack delta — this lane lands scaffolding only; no model is promoted, no
best_stack.json change. The eventual pretrain checkpoint (GPU, manager-run) will get its own
PENDING models/MANIFEST.json entry in that lane — name this explicitly in your report.

## Mandatory structured report (the manager rules ONLY on report.json)
objective_result PASS/FAIL vs the 6 smoke-gate items (each with its real exit code), full_suite
counts + exit code with pre-existing failures named, HONEST ISSUES (incl. zero-shot transfer
numbers however weak, any convention you had to assume), artifacts list (absolute paths), the
exact one-command fine-tune instruction for when reviewed_labels_v2.jsonl lands, BEST-STACK DELTA
line, and a dated bullet for the lane ledger.

## Anti-passive-wait
Ending your turn to wait = lane death; you will NOT be re-woken. Poll any long step with bounded
foreground loops; end only with the final report or a hard blocker.
