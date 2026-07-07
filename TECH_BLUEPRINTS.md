# TECH BLUEPRINTS — the executable per-pillar technical specs (successor handoff edition)

Written 2026-07-07 by Fable 5 on owner request, on the eve of the manager-model transition. This is
the companion to `NORTH_STAR_ROADMAP.md`: the roadmap says WHAT to build and WHEN (Part III tasks,
PART VI waves); THIS file says HOW, exactly — final algorithm choices, exact recipes and
hyperparameters, file-level integration points, decision trees for every foreseeable branch, and
DO-NOT lists — so that a less-judgment-strong manager can execute without inventing architecture.

**Provenance:** every pillar section was drafted by a deep-design agent grounded in the repo + the
3-pass adversarially-verified SOTA research (`runs/research_sota_20260705/`), then attacked by two
independent verifiers (repo-reality + successor-confusion), fixed, and cross-checked by a
completeness critic. External load-bearing bets were re-verified against the live web on 2026-07-07.
Staging artifacts: `runs/manager/blueprints_20260707/`.

**Status discipline:** same as the roadmap — `VERIFIED=0`; nothing here claims a passed gate.
Numbers carry evidence paths. This file is maintained under NORTH_STAR Part IV rule 14: every wave
closeout updates the affected pillar sections as evidence lands; a blueprint that drifts from
reality poisons the successor it exists to protect.

**How to use:** boot per PART A §A.5. When executing a PART VI wave lane, open this file's matching
pillar, follow its build plan and decision tree literally, and copy its acceptance keys into the
lane spec. If reality diverges from a blueprint (a step impossible, a number contradicted), that is
a diagnosis trigger + doc fix, never silent improvisation.

# PART A — SUCCESSOR MANAGER PRIMER (read this before touching anything)

Written 2026-07-07 by Fable 5 (the manager model being retired) explicitly for a less-judgment-strong
successor (Opus-class). NORTH_STAR_ROADMAP.md tells you WHAT and WHEN (Part III tasks, PART VI
waves). THIS document tells you HOW, exactly, so you never have to invent architecture. The single
most important sentence in this file: **when a decision is pre-ruled anywhere in the docs, follow
it; when it isn't and it's consequential, surface a typed STOP — do not synthesize a "reasonable"
answer.** Fable's edge was knowing which decisions were consequential. You get that as a bright-line
test instead (§A.3).

## A.1 Doc precedence (when two documents disagree)
1. The LATEST dated BUILD_CHECKLIST.md bullet (bottom of file) — live truth.
2. NORTH_STAR_ROADMAP.md PART VI (waves) and Part III STATUS lines — plan truth.
3. CAPABILITIES.md — truth-claim ceiling (what may be called working). If it is stale-dated,
   trust the newer BUILD_CHECKLIST bullet and book a docs-reconciliation lane (Part IV rule 14).
4. Everything else (older sections, memories, handoffs) — historical context; LATER CORRECTS EARLIER.
Never resolve a conflict by picking the version that lets you proceed. Resolve by date, or STOP.

## A.2 Judgment heuristics, converted to rules (each one was paid for)
- **No ruling without a number and a path.** A lane report that "sounds done" but cites no metric
  value + artifact path is NOT done — resume the lane and demand the report format. Never re-run its
  work yourself.
- **Internal-val wins mean nothing.** Four internal→held-out inversions are on record. Enthusiasm
  from internal numbers must never move a held-out shot forward — only the pre-registered protocol
  does (ledger row FIRST, then one shot, then STOP either way).
- **Suspect the harness before the model.** The biggest wins so far were harness bugs, not model
  limits: the 30fps frame-index bug (root jumps), constant-fps assumptions (1363-hit audit), the
  wrong viewer manifest file, a 16-commit-stale VM checkout. When a metric looks insane, first ask
  "what is the measurement pipeline assuming?" — only then consider the model.
- **Two symptoms, one cause.** Diagnose before fixing; run cheap read-only diagnosis lanes and look
  for convergence (weak bilateral phases explained BOTH the slide-gate failure and the
  grounding_refine self-kill). If you are about to dispatch two fix lanes for two symptoms, first
  check whether one root cause explains both.
- **Never trust a VM-computed number without the version stamp** (`remote_body_dispatch.py
  --verify-version-stamp`). Fleet VMs pin their cold-start commit forever.
- **Copy the exact gated metric key into every acceptance criterion** (e.g. `max_foot_lock_slide_m`,
  not "slide p95"). A verify lane caught a fix lane passing the wrong statistic. Budget one
  independent adversarial verify for every gate-adjacent claim.
- **Kill criteria are commitments.** If the criterion fires, the answer is whatever the spec said
  (flag OFF, lane dead, approach banked) — never "let me tune it a little."
- **The kill list (NORTH_STAR Part IV rule 5) is load-bearing.** Re-attempting a killed approach
  without NEW evidence is a failed session, even if it "seems worth retrying."
- **Resume, don't re-dispatch.** `codex exec resume <session-id>` and SendMessage keep full context
  for one sentence of correction. A fresh lane costs a full re-spec and re-orientation.
- **Prefer boring composition.** When choosing between a clever new mechanism and the established
  house pattern (foot_pin bounded corrections, scipy least_squares TRF+huber, flat per-account JSON,
  trust bands), take the house pattern. Every solver in this repo is scipy; every correction is
  bounded; every uncertainty is a band. Blend in, don't innovate on plumbing.
- **Agents that push back are gold.** A lane refusing to produce a metric because the request is
  structurally impossible (stale baseline, metric computed elsewhere) is doing its job — re-rule the
  request, never override the pushback.
- **Never read a big file to "get oriented."** Commission a summary or read the exact range a
  decision needs. Your context is the scarcest resource after GPU dollars.
- **Write state to disk at every milestone.** BUILD_CHECKLIST bullet, gpu_fleet.md, inflight_lanes.md,
  memory. Assume your session dies at any moment; the next session must re-derive everything from files.

## A.3 The bright-line STOP test (run it whenever unsure; any YES = typed STOP per manual §13)
1. Would this touch Outdoor/Indoor labels or any held-out clip without a pre-registered ledger row?
2. Would this change a gate threshold, gate statistic, or eval protocol?
3. Is this on the kill list, or equivalent to something on it?
4. Does this spend beyond ≤$5/GPU/hr × 4 GPUs, or need money/access only the owner has?
5. Did a lane self-report PASS while its own numbers say otherwise (full_suite.failed>0, metric
   under bar)? (→ needs-validation)
6. Am I about to build something no NORTH_STAR task ID / blueprint section covers?
7. Are two docs in conflict with no clear later-date winner?
If all NO: proceed and log the ruling in one line. A surfaced STOP is a SUCCESS state, not a failure.

## A.4 The owner (how to work with him)
Types fast with typos — infer intent, never ask about obvious meaning. Gives directives then leaves
for hours — run autonomously, never block mid-flight on questions. Wants numbered asks with
one-command unblocks (`! <command>`). Values parallelism ("what else can run?") and product framing
("how far from the finished product"). Money: envelope above; flag brief overages; log exact costs.
He explicitly granted full commit+push (2026-07-07, .claude/settings.json) — commit at checkpoints,
push after; never force-push. When you change a policy he sets, update the CLAUDE.md text in the
same pass — the permission classifier enforces the TEXT, not just settings.

## A.5 Session boot ritual (every session, no exceptions, ~10 minutes)
1. CLAUDE.md auto-loads → read FABLE_OPERATING_MANUAL.md (§12-§19 minimum) → NORTH_STAR PART 0
   (blank item = STOP) + I.7 + PART VI current wave → BUILD_CHECKLIST last ~15 bullets →
   runs/manager/gpu_fleet.md + inflight_lanes.md → memory index.
2. Reconcile fleet: `gcloud compute instances list --filter=labels.fable-fleet=pickleball`
   (auth check via the impersonated call in PART 0; orphaned VM = resume its lane or STOP+DELETE it).
3. Find your wave: the latest `[WAVE-N ...]` bullet names the queue; PART VI names the exit contract;
   the boot prompt at runs/manager/wave<N>_boot_prompt.md is your marching order. Execute VI.0 steps
   1-10 literally. Do not invent a different wave shape.

## A.6 Dispatch cheat-sheet (who does what — never deviate)
- **Codex lanes** (abundant): ALL code, self-verification, debugging, docs. Sandbox: no network, no
  MPS, no localhost binds, cannot write .git/index. Full lane contract = manual §3; report schema §5.
- **Sonnet subagents**: ONLY GPU/SSH/browser/network work. Always pin `model`; always include the
  anti-passive-wait phrasing; budget 1-2 SendMessage resumes per GPU lane.
- **You (the manager)**: decide, spec, rule, verify the decisive artifact once, book state. You never
  edit .py files (repo Edit/Write is guard-blocked anyway — coordination doc edits go via Bash
  python3 exact-replace with fail-loud anchor counts), never run wide suites (one clean adjudication
  run per wave is the exception), never parse big JSON by hand.
- **Workflows**: research fan-outs and adversarial verification panels (research-fanout skill).

# PART B — CROSS-PILLAR RULINGS (the completeness critic's findings, ruled by Fable — final)

A cross-pillar completeness critic read all nine blueprints against the I.7 definition of done.
Its findings are RULED here; these rulings bind every pillar section below when they conflict.

## B.1 Rulings on the contradictions
1. **`process_video.py` / `orchestrator.py` stage edits.** The "FENCED" warning in the COURT pillar
   is TEMPORAL (live wave-3 lanes), not permanent. Standing rule: ALL stage insertions into
   `_build_suffix_stage_fns` (paddle `paddle_pose`, fusion `consistency`, speedprod pre-flight,
   body IDF1 wiring) are serialized through the wave's ONE integration micro-lane (PART VI VI.0
   step 7) — never two concurrent lanes on this file. Blueprint line numbers are baseline hints;
   re-grep at HEAD before every edit.
2. **Which number flips BALL VERIFIED.** Beating held-out 0.7248 (pre-registered) is the P1-1
   PROMOTION MILESTONE — the chain candidate may then replace the zero-shot anchor as pipeline
   default. **BALL VERIFIED flips ONLY at the full M1 gate: F1@20 ≥ 0.90, recall ≥ 0.75, hidden-FP
   ≤ 0.05 on held-out.** 0.7248 is necessary, never sufficient. (ball3d §6 is patched accordingly.)
3. **Paddle artifact for fusion.** `racket_pose_estimate.json` is BUILT-NOT-WIRED (manual CLI only)
   until paddle P3-1 lands. Fusion PF-1 therefore ships its foot↔ground term first and adds the
   ball↔paddle term only after P3-1; PF-1 must not be blocked waiting. (fusion dep line patched.)
4. **One contact producer, ever.** P1-6's learned contact classifier is a new SOURCE inside
   `event_fusion.fuse_contact_windows` — `contact_windows.json` remains THE artifact; the heuristic
   sources stay as fallback weights. Never a parallel contact artifact. (ball3d STEP 6 patched.)
5. **RKT VERIFIED needs the owner marker-GT session and none is booked.** Correct and intended:
   surface it as an owner ask at EVERY wave boot until booked (pair it with the W5 capture block).
   Internal IoU never promotes RKT — that is the honesty design, not a bug.

## B.2 Rulings on the gaps
1. **Data engine + player identity** — now the NINTH pillar (PART C first section): harvest
   recurrence, owner-capture day-1 runbook, labeling scale-up to budget, label QC, cross-clip
   ReID/role assignment (consent-gated), corpus bookkeeping.
2. **Cross-clip identity ownership:** WITHIN-clip tracking/IDF1 = BODY (P2-5); CROSS-clip ReID
   galleries + owner/partner/opponent roles + per-account stability = DATA pillar, on the H4
   player profile, consent-gated for non-owners.
3. **P0-10 capture app** stays a NORTH_STAR task (its P0-10 block is already the spec, plus EDGE
   H28-H34); its server-side consumers are already specced in COURT (PoseGravity/AnyCalib) and
   FUSION (camera seed). Owner+device dependency; hard-pinned ≤ wave 8.
4. **The composite v1-DONE harness** (3 consecutive fresh games, all ladder gates + QA + coaching
   card together) is OWNED BY FUSION PF-4, extended: one named harness run
   (`v1_done_harness`) = PF-4 consistency assertions + P5-6 QA green + P6-4 fabrication audit on
   the same three games. Build it in the wave that attempts M5, not before.
5. **Rally-end-cause attribution** (who ended the rally, how) is OWNED BY THE BALL CHAIN (ball3d
   pillar): emitted with the rally/contact artifacts (the rally_gating stage already segments
   rallies); coaching consumes it for unforced-error stats — it never derives it itself.
6. **`profile_registry_schema.json` has ONE owner: the DATA pillar.** Court/coaching/speedprod
   CONSUME the schema; any schema change is a single lane touching the schema file with a version
   bump — three pillars evolving it concurrently is forbidden.

## B.3 The cross-pillar dependency spine (order of artifacts — memorize this chain)
confident per-foot contact phases (w3_phasefix fresh-GPU promotion) → paddle P3-1 in-pipeline
wiring → ball P1-4 full-flight arcs (`ball_track_arc_solved.json` with real velocities) → THREE
consumers unlock together: paddle P3-5 reflection, fusion PF-1 ball terms, coaching S1 shot
features → PF-2 joint optimizer. A slip in P1-4 stalls three pillars — it is the highest-leverage
single artifact in the plan. P5-6 auto-QA ships WITHOUT the Phase-F consistency residuals first
(they don't exist until PF-1) and adds them later — QA-green before PF-1 means "green minus
Phase-F residuals" and must say so in its report.

## B.4 The successor's twelve most likely mistakes (from the critic; guardrails are binding)
1. Promoting a ball checkpoint on the wrong scorer — only `label_f1_at_20px` from
   `run_ball_tracking_eval_suite.py` gates; the training-harness proxy `f1_at_20px` never does.
2. Taking a held-out shot with a public-only student, or re-tuning past an inversion — an
   internal→held-out inversion is an automatic needs-validation STOP (4 on record).
3. Trusting a GPU/VM number without `--verify-version-stamp` (fleet1 sat 16 commits stale).
4. Clearing the foot-slide gate from offline replays — `max_foot_lock_slide_m` is fresh-GPU-only.
5. Silently moving a threshold or paraphrasing a metric — statistic disputes are owner
   needs-decision STOPs with banked evidence; always copy the exact gated key.
6. Two parallel lanes editing `process_video.py`/`orchestrator.py` — one integration lane, always.
7. Re-deciding frozen rulings (SAM-3D backbone, scalar-spin-only pre-H13, physics-gated ball
   selection not voting, scipy TRF/huber house solver, render-only paddle) — a challenger reverses
   one ONLY via its pre-registered decision rule.
8. Touching an Outdoor/Indoor label or spending a held-out shot early — ledger row + owner go
   first; iterate on Burlington/Wolverine only.
9. Claiming VERIFIED/DONE on internal or self-reported numbers — VERIFIED=0 until a product gate
   passes on real labels.
10. Building a second producer that orphans existing consumers — grep the existing artifact_type
    and EXTEND it (contact_windows, shots) — see ruling B.1.4.
11. Filling a wave with only P2/P4/P5 polish — every wave needs ≥1 critical-path lane
    (DATA→BALL→flight→contacts→paddle-impact→stats→fusion→coaching); zero is a planning bug.
12. Feeding the coach LLM raw numbers or skipping the fabrication firewall — the LLM sees ONLY
    comparator verdicts; every cited metric must trace to a finding or the card is rejected.

# PART C — THE NINE PILLAR BLUEPRINTS

## PILLAR: DATA ENGINE + PLAYER IDENTITY — the critical path's first link (P0-1b/3/4/5, P0-9 identity, cross-clip ReID)

> Audience: the successor manager. Every file/function/flag below is grep-verified 2026-07-07.
> Reserved word `VERIFIED` = a passed PRODUCT gate on real labels; current state **VERIFIED=0** — do
> not overclaim. You DELEGATE these to Codex/Sonnet lanes (use `run-lane`); you never implement. This
> pillar is the FIRST link of the critical path DATA→BALL→flight→contacts→paddle→stats→coaching:
> nothing downstream fine-tunes until the label budgets here are met.

### 0. Final ruling — the stack in one paragraph
Two distinct fuels feed one funnel. **Harvest** (`yt-dlp` tripod games → auto-clip → prelabel) is
OUT-of-domain diversity/pretrain + broad-test fuel; **owner captures** are the in-domain finisher —
NEVER swap the roles. Every clip gets its role (`train` | `internal_val` | `heldout`) assigned AT
INGEST; held-out clips are reserved in `runs/manager/heldout_eval_ledger.md` BEFORE any prelabel/
detector pass ever touches them. Corrections land in a **self-hosted CVAT** at
`/Users/arnavchokshi/cvat_labelfactory/cvat_src` (localhost:8080, docker compose), export→import
round-trip proven via `scripts/racketsport/import_cvat_video_annotations.py`. Budgets are DIVERSITY-
first: ball ≥10-20k corrected frames across ≥4 distinct sessions/courts before the first fine-tune
shot, paddle ≥1-2k keypoint frames, contacts ≥500 events — single-session fine-tunes are kill-listed.
Identity splits cleanly: **within-clip** tracking/IDF1 is the BODY pillar (P2-5); **cross-clip**
identity (per-player ReID gallery, owner/partner/opponent roles, per-account stability for P6-6 trends)
is THIS pillar, built on the H4 player profile and CONSENT-GATED for everyone but the owner. Storage
is flat per-account JSON under `runs/profiles/<account_id>/` (schema
`docs/racketsport/profile_registry_schema.json`); the pipeline consumes a profile when present and
falls back to generic solvers when absent. NO persistent non-owner biometrics until the P7-4b consent
decision; session-only tracking meanwhile.

### 1. Current measured state (numbers + evidence paths only)
- **Harvest pipeline DONE through prelabels (P0-1b).** 8 games → 43 rally clips, roles train-29 /
  internal-val-6, + 2 held-out reservations; 40/40 WASB prelabel sidecars on fleet2 ~$1.3.
  Evidence: `runs/lanes/p01b_harvest_ingest_20260706/` (corpus_card.json/.md, dedup_report.json) +
  `runs/lanes/p01b_prelabel_20260707/`. Role assigner = `online_harvest_ingest.assign_clip_roles`
  (`internal_val_modulo=5`; held-out sources → role `heldout_candidate_proposed`, excluded from shards).
- **Held-out reservations registered, ZERO exposure.** HARVEST-1 (`pwxNwFfYQlQ`), HARVEST-2
  (`vQhtz8l6VqU`) — ledger rows 305/307, exposure = download+ffprobe+manifest screening only, no
  labels exist. Do NOT open their content.
- **Leakage guard = 35 protected eval hashes.** `roboflow_corpus.DEFAULT_PROTECTED_EVAL_HASH_COUNT=35`
  (`roboflow_corpus.py:41`); collision asserts `assert_no_protected_eval_hash_collisions` (:626),
  `assert_no_index_marked_protected_eval_collisions` (:660). Roboflow P1-0: 61,260 kept samples, 0/35
  leakage. Video dedup for harvest = `online_harvest_ingest.build_dedup_report` (:736) via
  `perceptual_hash_video` (:686).
- **Label factory LIVE (P0-4), first owner batch returned.** CVAT project id 2
  `racketsport_online_harvest_wave3_review`, tasks 7-12 (2 train / 4 internal-val, ~81 frames each).
  Export→import→schema-validate round-trip PASS (`runs/lanes/w3_labelfactory_20260707/roundtrip_report.json`,
  returncode 0). OPEN: no labels/hour number yet; corpus ~486 review frames vs the ≥10-20k budget.
- **Visibility UI trap caught + remapped.** CVAT UI default label reads `full` but the schema `full`
  means fully-occluded-but-in-frame; raw exports were deterministically `full→clear` remapped, raw
  preserved. Evidence: `cvat_upload/exports/harvest_review_20260707/*/MANAGER_NOTE.md`. Schema =
  `threed/racketsport/schemas/__init__.py::BALL_VISIBILITY_LEVELS` {clear,partial,full,out_of_frame},
  WBCE weights clear=1/partial=2/full=3/out_of_frame=3.
- **Profile registry SCHEMA+STORAGE DONE (P0-9), no pipeline wiring.** `threed/racketsport/profile_registry.py`
  (5 models; `PlayerProfile` carries `height_m`, `frozen_shape_betas_ref`, `reid_gallery_ref`,
  `handedness`, `consent_status`, :163-178), flat JSON `runs/profiles/<account_id>/` (`DEFAULT_PROFILE_ROOT`
  :16), consent enforced by `_validate_persistence_rules` (:387; non-owner biometric persistence needs
  `consent_status='granted'`+`consent_source_trace`). Evidence: `runs/lanes/p09_registry_20260706/REPORT.md`.
- **Owner-capture intake EXISTS, NOT YET FIRED (P0-3 not started).** `ingest_owner_capture`
  (`owner_capture_intake.py:75`) + `prelabel_owner_capture` (:216) + `apply_reviewed_cvat_export`
  (:317) with eval-leak guards (`collect_protected_eval_video_shas`, `PROTECTED_OWNER_EVAL_SLUGS`,
  `ProtectedEvalCaptureError`). GAP: the owner manifest row (:111) has NO `role` field — only
  `review_status`∈{unreviewed,reviewed} and `train_eligible` bool (grep `"role"` = 0 hits). Roles-at-
  ingest for owner captures is a TO-BUILD (step D2).
- **Cross-clip identity NOT built.** `reid_gallery_ref` is a schema SLOT only. What exists is
  WITHIN-clip: `player_global_association.associate_global_identities` (:178, fragments→clusters within
  one clip), `doubles_id.assign_doubles_roles` (near/far + left/right lateral, NOT owner/partner/
  opponent), IDF1 via `mobile_person_eval.py:79`. Crop exporter `person_reid_dataset.export_person_reid_crop_dataset`
  (:53) emits train/query/gallery folders — the input format a TorchReID trainer consumes, but no
  persistent per-player gallery, no cross-clip role assignment. TRK gate blocker still open
  (`person_tracking_promotion_audit.py:42` `labeled_idf1_spectator_gate_missing`).
- **No corpus dashboard script yet.** `ls scripts/racketsport | grep corpus` → only
  `aggregate_roboflow_corpus.py`; P0-4 gate's dashboard is unbuilt (step D8).

### 2. The exact build plan
Each step: objective · file targets · recipe · lane sizing · acceptance (exact keys) · kill.

**D1 — Recurring harvest batches on VMs (a).** *Obj:* keep out-of-domain fuel flowing as label demand
grows, zero owner time. *Files:* `threed/racketsport/online_harvest_ingest.py` (`assign_clip_roles`,
`build_dedup_report`, `perceptual_hash_video`), reuse `runs/lanes/p01b_harvest_ingest_20260706/`
scaffold (sources/, rally_clip_manifest.json, prelabel_shard_manifest.json). *Recipe:* new dated lane
`runs/lanes/p01b_harvest_<date>/`; yt-dlp bulk pull NEW channels/courts (widen diversity, not depth);
auto-clip to rallies; `build_dedup_report` against the 35 eval hashes AND prior harvest hashes (dedup
BEFORE prelabel); `assign_clip_roles` with 2 fresh held-out proposals per batch; prelabel with the
PHYSICS-GATED chain on a spot GPU (fleet pattern proven ~$1.3/40 clips). *Lane sizing:* 1 Sonnet
network-capable lane per batch (GPU prelabel shard); CPU ingest/dedup runs on Codex. *Accept:*
`corpus_card.json` with `dedup.eval_hash_collisions == 0`, roles logged, held-out proposals written to
ledger by the MANAGER (never the lane — see `heldout_proposals[].ledger_action`
`manager_registers_only`). *Kill:* dedup collision >0 vs the 35 → drop the colliding clip, never
override the assert.

**D2 — Owner-capture ingest day-1 runbook (b) [FIRES ~2026-07-09, W4-E/W5 entry].** *Obj:* first real
in-domain batch through the loop with roles-at-ingest + ≥2 held-out-with-audio reserved. *Files:*
`scripts/racketsport/ingest_owner_capture.py`, `scripts/racketsport/prelabel_owner_capture.py`,
`owner_capture_intake.py` (add a `role` field to the row at :111 + a `--role` CLI arg — the manifest
GAP). *Runbook, exact order:*
  1. Per NON-held-out capture: `python3 scripts/racketsport/ingest_owner_capture.py <package_dir>`
     (package = clip.mov + capture_sidecar.json). Confirm stdout `status:"registered"` + a
     `camera_fingerprint`; the eval-leak guard fires automatically (sha + `PROTECTED_OWNER_EVAL_SLUGS`).
  2. `python3 scripts/racketsport/prelabel_owner_capture.py --capture-id <id> --dry-run` first (wiring),
     then without `--dry-run` on a GPU lane → candidate-only prelabels (status enforced by
     `enforce_candidate_prediction_status`; NEVER used as training rows).
  3. Load prelabels into CVAT (D3 factory), owner reviews, export zip,
     `import_cvat_video_annotations.py --cvat-zip … --clip-id … --fps <real PTS fps>`, then
     `ingest_owner_capture.py --capture-id <id> --reviewed-cvat-export <zip>` → flips
     `train_eligible=true` + appends to `runs/training_corpora_20260702/owner_capture/manifest.json`.
  4. **Held-out captures (≥2, WITH audio, ≥1 deliberately handheld): DO NOT ingest into owner_data.**
     Mirror the harvest pattern — screen with ffprobe only, reserve a pre-registered
     `heldout_eval_ledger.md` row (candidate + decision rule + "no labels exist") BEFORE any prelabel.
     Audio matters: they gate P1-6 contacts + the BALL M4 sub-gate (P0-5).
  *Lane sizing:* manager runs ingest CLIs (tiny); 1 GPU lane for prelabel; owner does review. *Accept:*
one capture fully ingested (court keypoints present, role registered, prelabels in CVAT); ≥2 held-out
rows in ledger with audio flagged; `validate_owner_data_manifest` passes. *Kill:* any capture sha
matches a protected eval clip → `ProtectedEvalCaptureError` (intended; investigate, do not bypass).

**D3 — Labeling factory scale-up to budget (c).** *Obj:* reach ball ≥10-20k / paddle ≥1-2k / contacts
≥500 with measured throughput. *Files:* `runs/lanes/w3_labelfactory_20260707/` (create_project_and_tasks.py,
build_and_import_prelabels.py, OWNER_LABELING_GUIDE.md), `import_cvat_video_annotations.py`,
`check_cvat_video_annotations.py`. *Recipe:* (i) **Throughput math** — instrument labels/hour by
diffing CVAT task `updated_date` deltas vs corrected-box counts per session (owner marks start/stop);
record labels/hr per label type separately (ball-correct is fast, paddle-keypoint slow). Extrapolate
to budget → owner+helpers plan (owner does train + all held-out; helpers do internal-val only, never
held-out). (ii) **Active-learning queue** — feed SST teacher↔student disagreement frames (W4-B,
`disagreement frames → P0-4 label queue`) to the FRONT of the queue; label the model's blind spots,
not uniform frames. (iii) SAM3 per-frame assist for ball is the DEFAULT plan (§5). *Lane sizing:*
Codex micro-lane builds the throughput+queue tooling; labeling is owner/helper wall-clock. *Accept:*
`labels_per_hour` reported per type; ≥4 distinct sessions/courts represented before any fine-tune is
scheduled. *Kill:* if projected labels/hr can't reach the ball budget in the owner's available hours,
the fine-tune WAITS (do not shrink the budget — diversity beats volume is the ruling).

**D4 — Label QC (d).** *Obj:* trustworthy labels. *Files:* `check_cvat_video_annotations.py`,
`MANAGER_NOTE.md` remap pattern, `schemas/__init__.py::BALL_VISIBILITY_LEVELS`. *Recipe:* (i)
**Visibility remap** — on every export apply the deterministic `full→clear` remap where the CVAT UI
default `full` was never explicitly reviewed; preserve raw. (ii) **Duplicate-box** — run the sanity
checker to catch >1 ball box/frame or overlapping player boxes; reject the task export on any dup. (iii)
**Inter-rater** — for helper-labeled internal-val, owner spot-checks a ≥10% random sample; log
agreement; if <90% box-IoU agreement, retrain the helper before more volume. *Accept:* `sanity_report.json`
clean; remap applied+logged; inter-rater agreement recorded. *Kill:* systematic label error found →
quarantine that helper's batch, re-review before it enters the corpus.

**D5 — Cross-clip ReID gallery + role assignment (e) [CONSENT-GATED; owner-only until P7-4b].** *Obj:*
stable per-player identity across an account's clips so P6-6 trends attach to the right person. *Files
to BUILD on:* `person_reid_dataset.export_person_reid_crop_dataset` (:53, already emits train/query/
gallery), `profile_registry.PlayerProfile.reid_gallery_ref` (the storage slot), `player_global_association`
(within-clip embeddings source), `_validate_persistence_rules` (:387, the consent gate). *Recipe:* (1)
build a per-account gallery = frozen ReID embeddings of the OWNER's reviewed player crops, persisted as
a `reid_gallery` artifact referenced by `reid_gallery_ref`; (2) at pipeline time, match each within-clip
track's mean embedding against the account gallery → assign the OWNER identity; (3) owner/partner/
opponent ROLE = owner-identity (gallery match) + near/far side from `assign_doubles_roles`; partner =
same side as owner, opponents = far side (session-only labels for non-owner slots). Backbone: TorchReID/
SoccerNet-class model (§5). NON-owner persistence stays BLOCKED by `_validate_persistence_rules` until
consent. *Lane sizing:* 1 GPU lane (train/eval ReID on owner crops), gated on ≥owner labels across ≥2
sessions. *Accept:* owner gallery match precision on a held-out owner clip ≥ agreed bar (register the
key before running); non-owner write attempt raises `ProfileConsentError`. *Kill:* owner match unstable
across sessions after 2 iters → bank negative, keep session-only ReID (`associate_global_identities`),
do not ship a persistent gallery.

**D6 — Corpus/dashboard bookkeeping + leakage (f).** *Obj:* one place that answers "how much labeled
data, what diversity, any leakage." *Files to BUILD:* `scripts/racketsport/corpus_dashboard.py` (does
NOT exist — P0-4 gate requires it). *Recipe:* aggregate corpus manifests (harvest corpus_card.json +
`runs/training_corpora_20260702/owner_capture/manifest.json` + Roboflow index) → counts by label type,
distinct sessions/courts, roles; re-run `assert_no_protected_eval_hash_collisions` + a video-hash check
of every train/val clip vs the 35 eval hashes + the 2 harvest held-out + owner held-out; emit a corpus
card per new batch. *Lane sizing:* 1 Codex lane. *Accept:* dashboard prints per-type counts + distinct-
session count + `eval_hash_collisions:0`; a corpus card exists for every registered batch. *Kill:* any
nonzero leakage → STOP, quarantine the batch, surface to manager.

### 3. Decision trees
- **Owner can't record yet (state before ~2026-07-09):** run D1 (recurring harvest), D3 tooling
  (throughput+active-learning), D6 (dashboard), D5 owner-gallery DESIGN only — all idle-safe, no owner
  dependency. W4-E "fires the moment captures land, else idles harmlessly." Do NOT block BALL pretrain
  (W4-A) on owner data — it runs on harvest+Roboflow now.
- **Labels below budget when a fine-tune is proposed:** if <10-20k ball across <4 sessions/courts →
  the fine-tune WAITS (W5 entry is HARD: single-session fine-tunes kill-listed). Meanwhile pretrain
  (W4-A) + SST (W4-B) + recall levers (W4-C) proceed — they don't need the in-domain finisher. Paddle
  <1-2k or contacts <500 → those specific downstream gates (P3-7, P1-6) wait, not the whole wave.
- **Consent unanswered (P7-4b blank):** BLOCKED = any persistent non-owner biometric (frozen betas or
  reid_gallery for a non-owner), any cross-account player-profile share, processing the FIRST non-owner
  footage. ALLOWED = owner profile #1 (is_account_owner exempt), session-only within-clip tracking of
  anyone, harvest processing (the ruling: harvest = video processing, NOT biometric persistence of
  people). The block is enforced in code (`_validate_persistence_rules`), not just policy.

### 4. DO-NOT list
- Do NOT prelabel/detector-pass/label a held-out clip before its ledger row exists (harvest OR owner);
  reservation ALWAYS precedes exposure.
- Do NOT train on competitor-PROCESSED video (pb.vision / SwingVision outputs) or any labels you can't
  trust — raw tripod video is fine, their inferred tracks are not.
- Do NOT fine-tune on a single session/court — it is a proven failure mode (kill-listed). ≥4 distinct
  sessions/courts before the first ball fine-tune, no exceptions for volume.
- Do NOT persist ANY non-owner biometric profile (frozen betas, ReID gallery) until P7-4b consent —
  `_validate_persistence_rules` will (and must) raise `ProfileConsentError`.
- Do NOT let a train/val clip collide with the 35 protected eval hashes or the reserved held-out games;
  the assert is a guardrail, never override it.
- Do NOT reuse owner-capture `candidate_prediction` prelabels as training rows (enforced by
  `enforce_candidate_prediction_status`) — only REVIEWED exports flip `train_eligible`.
- Do NOT confuse the CVAT UI `full` default with the schema `full` (fully-occluded-in-frame) — always
  apply the documented remap, preserve raw.

### 5. External bets verified today (WebSearch 2026-07-07)
- **CVAT SAM 3 (released Nov 2025) DOES now propagate through video** (seed a keyframe with a text
  prompt, tracker propagates masks; SAM 3.1 multiplexing does up to 16 objects/pass). BUT the built-in
  managed feature is **CVAT Online (paid) / Enterprise only**. Our factory is self-hosted OSS Community
  → SAM 3 propagation is NOT free out-of-the-box; it requires **self-deploying SAM 3 as a Nuclio
  serverless function** (Docker Compose, some setup). RULING UNCHANGED as the default plan: treat CVAT
  assist as per-frame; the Nuclio SAM 3 tracker is an OPTIONAL throughput lever to evaluate in D3 IF
  ball-labeling wall-clock becomes the bottleneck — not a plan dependency, and it segments (masks), our
  ball label is a box+visibility, so payoff is unproven. Sources: cvat.ai/resources/changelog,
  docs.cvat.ai serverless-tutorial.
- **Cross-clip ReID with code (D5 backbone options):** TorchReID (the de-facto library), SoccerNet
  `sn-reid` benchmark + baselines, SportsReID (built on TorchReID, SoccerNet-ranked), CLIP-ReIdent
  (2022 challenge winner, OpenCLIP ensemble). All open-source, all consume the exact train/query/gallery
  crop layout our `person_reid_dataset.export_person_reid_crop_dataset` already emits. Pick TorchReID
  first (lightest integration). Sources: github.com/KaiyangZhou/deep-person-reid (TorchReID),
  github.com/SoccerNet/sn-reid, github.com/shallowlearn/sportsreid.

### 6. Acceptance gates + owner dependencies
- **P0-3 gate:** one owner capture fully ingested (court keypoints, role registered, prelabels in CVAT);
  ≥2 held-out-with-audio ledger rows (≥1 handheld). *Owner dep:* first capture batch (~2026-07-09).
- **P0-4 gate:** `labels_per_hour` measured per type; `scripts/racketsport/corpus_dashboard.py` exists;
  budgets tracked toward ball 10-20k/≥4 sessions, paddle 1-2k, contacts 500. *Owner dep:* labeling hours
  (owner + helpers).
- **P0-5 gate:** Indoor CVAT export fixed; ≥2 owner held-out clips + ≥1 handheld registered; eval_guard
  passes on all new clips; `eval_clips/ball/README.md` updated. *Owner dep:* the held-out captures.
- **P0-9 gate:** a court profile round-trips (store→match next upload→skip re-calibration); missing-
  profile clip degrades to generic path. *Owner dep:* profile-capture session (W5-C).
- **Cross-clip ReID gate (this pillar):** owner gallery match precision ≥ pre-registered bar on a
  held-out owner clip; non-owner persistence raises `ProfileConsentError`. *Owner dep:* owner labels
  ≥2 sessions + the P7-4b consent decision (PART 0) before ANY non-owner gallery.
- **Standing owner-dependency ladder (surface at wave boot only):** captures → labeling passes →
  profile/paddle-GT sessions → consent decision. Everything else keeps running; harvest carries the
  interim.

## PILLAR: BALL 2D — detection/tracking from 0.6969 to the M1 bar (F1@20>=0.90, recall>=0.75, hidden-FP<=0.05)

**[WAVE-3 CLOSEOUT CORRECTIONS 2026-07-07 — these supersede conflicting lines below]:** (1) STAGE-1
pretrain is DONE (H100, harness internal_val f1@20 0.0615→0.6104, precision@20 0.848, recall 0.477;
ckpts runs/lanes/w3_p11_train_20260707/checkpoints/; cycle-caching + output_channels harness bugs
fixed in repo) — the missing-WASB-checkpoint blocker is resolved. (2) SST teacher RE-RULED by
measurement: raw single-WASB (pooled F1 0.680) BEATS 2D-gated teachers (0.395) on human GT — the
blessed seed is raw single-WASB; consensus-fusion ban intact; physics-gated-chain teacher deferred
behind P4 court auto-cal (3D chain hard-requires calibration). (3) Owner labeling throughput
measured: ~240 frames/hr (480 frames / ~2h first session).**

> Audience: the successor manager. Every step here is pre-ruled. Where judgment could creep in, the branch
> ends in "typed STOP: <bucket>". Do NOT re-decide the standing rulings in §0/§4 — they are owner+Fable-settled.
> Two scoring worlds exist; confusing them is the #1 trap — read §1 before running anything.

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
We locked WASB-HRNet (nttcom/WASB-SBDT, 1.5M params, MIT) as the architecture anchor because its zero-shot
tennis checkpoint is the standing held-out bar (F1@20 0.7248, ledger row 4) and no runnable public detector
beat it (SAM3.1, volleyball-WASB, public fine-tunes all inverted on held-out). We adopted the TOTNet recipe
(AugustRushG/TOTNet, MIT) — visibility-weighted 4-level WBCE (weights clear=1/partial=2/full=3/out_of_frame=3)
+ occlusion augmentation TOGETHER (aug alone hurts: RMSE 29.6->54.3). We ruled a 3-stage path: STAGE-1 warm-start
pretrain on the 61,260-sample Roboflow corpus + adjacent-sport aux at 8:1, INTERNAL-VAL ONLY; STAGE-2 fine-tune
ONLY on owner in-domain CVAT labels (the only lever that ever moved held-out) — only stage-2 candidates may take
a pre-registered held-out shot; STAGE-3 SST bootstrap with the PHYSICS-GATED chain as teacher. Detectors are
CANDIDATE GENERATORS, never spatial-consensus voters (voting hidden-FP 0.349 vs single WASB 0.063); final
selection is physics-consistency (arc solver), flipped only after a pre-registered A/B. Recall to 0.75 is a
separate composable lane (P1-3) measured lever-by-lever. VERIFIED=0 and stays 0 until a product gate passes on
real labels — nothing in this doc is VERIFIED.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- STANDING HELD-OUT BAR: WASB-tennis zero-shot Outdoor F1@20 **0.7248**, recall@20 0.626, precision 0.861,
  hidden-FP 0.063, teleports 7. Evidence: `runs/manager/heldout_eval_ledger.md` row 4.
- BEST QUALITY CHAIN (held-out): fused+auto-anchor+arc, Outdoor product_veto_v40_weak F1@20 **0.6969**, P 0.878,
  R 0.578, hidden-FP 0.021, P95 34.3px, teleports 1. Below bar (lost on recall). Evidence: ledger row 23.
- INTERNAL-VAL reference (Burlington+Wolverine, 754 visible): WASB-tennis F1@20 **0.6685** (ledger BALL-IV-1).
- TWO SCORING WORLDS — DO NOT CONFUSE:
  1. HARNESS PROXY (fast, during training): `scripts/racketsport/train_ball_pretrain.py::evaluate()` emits metric
     key **`f1_at_20px`** (also `recall_at_20px`,`precision_at_20px`,`median_error_px`) on the RoboflowBallPretrain
     internal-val split (peak-of-heatmap vs `target_xy_px`, threshold 0.5). This is a PROXY on public frames, NOT
     the product gate. Never promote on it.
  2. PRODUCT GATE (the real number): `scripts/racketsport/run_ball_tracking_eval_suite.py` ->
     `threed/racketsport/ball_benchmark.py`. Metric keys: **`label_f1_at_20px`** (per clip),
     **`visible_hit_recall`** / **`mean_visible_hit_recall`** (recall), **`hidden_false_positive_rate`** /
     **`mean_hidden_false_positive_rate`** (hidden-FP), plus `visible_recall_at_20px`. DEFAULT_CLIPS =
     `burlington_gold_0300_low_steep_corner`, `wolverine_mixed_0200_mid_steep_corner` (INTERNAL-val reference —
     safe to score any time). Held-out Outdoor/Indoor are scored ONLY via a pre-registered ledger row through
     `benchmark_ball_tracks_against_cvat.py` under `run_ball_chain.py --heldout-authorized`.
- FAILURE EVIDENCE (why public-only fine-tunes are dead): `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`
  — TrackNetV3 -17pt on Burlington ref; WASB tennis-seed 0.0018 static-distractor lock. Root causes: broadcast
  domain, 84% temporally-dead frames, too-few-sources distractor-lock.
- CORPUS (stage-1 fuel, DONE): 61,260 kept samples, `data/roboflow_universe_20260706/aggregated/corpus_index.json`,
  buckets by SOURCE-count core_pickleball=59 sources / adjacent_sport_aux=3 sources (pre-filter per-SAMPLE counts
  80,967 / 29,036; corpus_index keeps 61,260 total after de-dup/dead-link filtering), 0/35 eval-hash leakage.
  Card: NORTH_STAR P1-0 (row [x]).
- OWNER IN-DOMAIN LABELS (first batch, DONE 2026-07-07): `cvat_upload/exports/harvest_review_20260707/` — ~274
  human-verified ball boxes across 6 harvest-source clips, 4-level visibility populated (clear-vs-absent dominant;
  `partial` under-used, honest). Held-out clean (pwxNwFfYQlQ/vQhtz8l6VqU absent). This is P1-1/P1-2 SEED material,
  far below the STAGE-2 volume budget (>=10-20k frames, NORTH_STAR P0-4) — insufficient FUEL for a stage-2 shot,
  NOT "zero labels." Evidence: BUILD_CHECKLIST 2026-07-07 "HARVEST REVIEW LABELS COMPLETE".
- LOCAL BLOCKER: `models/checkpoints/wasb/wasb_tennis_best.pth.tar` is ABSENT on the Mac (confirmed). Stage-1
  init + the true zero-shot baseline need a VM/network prestage.
- 4-LEVEL VISIBILITY SCHEMA: landed end-to-end (`p11_visibility_schema_20260706`). Pretrain harness
  (`train_ball_pretrain.py`, `RoboflowBallPretrainDataset`) landed in wave-3 `w3_p11_prep` (CPU smoke loss
  0.75->0.21; UNRULED). These are the uncommitted working-tree files this blueprint plans against.

### 2. The exact build plan
Paths/functions below were grep-verified on 2026-07-07. Run everything on a VM (fleet), never on Fable.
SEQUENCING (do not halt the whole pillar on one blocker): STEP 1 and STEP 2 are runnable NOW (no owner
dependency). STEP 3/4 are owner-label-volume-blocked -> typed STOP: labeling, but that does NOT freeze the pillar:
STEP 5 (fusion A/B, internal-val only) and STEP 6 (a)-(c) recall levers are composable and SHOULD proceed in
parallel on Burlington/Wolverine internal-val while awaiting labels. Only the held-out shots and STAGE-2/3 wait
on owner P0-4 volume.

**STEP 1 — Prestage the WASB anchor + zero-shot baseline (SONNET network lane, no GPU).**
- Objective: get `models/checkpoints/wasb/wasb_tennis_best.pth.tar` onto the target VM and record its sha256.
  Source = nttcom/WASB-SBDT MODEL_ZOO tennis checkpoint (already the row-4 anchor).
- Acceptance: file present; sha256 == `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb`, the
  authoritative value in `models/MANIFEST.json` entry `wasb_tennis_bmvc2023` (the ONLY place this hash is recorded;
  ledger row 4 and the never-present `models/checkpoints/candidates_t6/` dir do NOT carry it). Kill: sha mismatch ->
  typed STOP: needs-validation (do not train on an unverified anchor; do not infer a hash from any other source).

**STEP 2 — STAGE-1 warm-start pretrain on the 61k corpus (GPU lane, A100 sufficient).**
- Objective: a public-warm-started WASB checkpoint that raises the harness proxy without corrupting the anchor.
- File target: `scripts/racketsport/train_ball_pretrain.py` (as-is) + `configs/racketsport/ball_pretrain_roboflow_wasb.json`.
- Exact command (VM; rewrite the image-root prefix to the VM checkout):
  ```
  python3 scripts/racketsport/train_ball_pretrain.py \
    --config configs/racketsport/ball_pretrain_roboflow_wasb.json \
    --out-dir runs/lanes/p11_stage1_pretrain_<TS>/ \
    --model-family wasb_hrnet --wasb-repo third_party/WASB-SBDT \
    --init-checkpoint models/checkpoints/wasb/wasb_tennis_best.pth.tar \
    --device cuda --steps 12000 --batch-size 8 \
    --learning-rate 5e-4 --weight-decay 5e-5 \
    --image-size 512x288 --frames-in 3 --heatmap-radius-px 4.0 \
    --core-to-aux-ratio 8 --checkpoint-every 500 --num-workers 4 \
    --seed 1337 --zero-shot-baseline \
    --image-root-rewrite /Users/arnavchokshi/Desktop/pickleball=<VM_CHECKOUT_ROOT>
  ```
- Recipe rationale (portable, from `third_party/TOTNet/src/train.sh`): AdamW lr=5e-4 wd=5e-5, 512x288,
  radius 4 (train radius in the harness; TOTNet uses 8 train / 4 test — harness has one radius, keep 4).
  Constant LR (harness has NO scheduler — do not fabricate one). 12000 steps x bs8 = 96,000 samples ~= 1.5 epochs
  over the 61,260-sample corpus (raise --steps ~proportionally for more coverage: ~40k steps ~= 5 epochs).
- KEY-DIFF TRAP (mandatory check): `load_model_weights` runs with `strict=--strict-init` (default False). Read
  `summary.json`'s `model.init_summary` block: `missing_keys` and `unexpected_keys` MUST both be empty for the tennis->WASB-HRNet
  load (same family). If either lists SE keys, `strict=False` silently amputated modules (the row-14 disaster) ->
  typed STOP: needs-validation. Do NOT proceed on an amputated init.
- Visibility note (why 4-level WBCE is INERT here): public Roboflow labels carry `visibility_level=None`, so
  `RoboflowBallPretrainDataset` sets `wbce_weight = PUBLIC_UNKNOWN_VISIBILITY_WBCE_WEIGHT =
  BALL_VISIBILITY_WBCE_WEIGHTS["clear"] = 1` for every sample. Stage-1 is therefore a plain warm-start; the
  4-level WBCE and occlusion aug only bite at STAGE-2. This is expected, not a bug.
- Acceptance (PROXY only): `summary.json` `internal_val.metrics.f1_at_20px` > `zero_shot_baseline.f1_at_20px`
  (or read the sibling files `internal_val_metrics.json` / `zero_shot_baseline.json`). PLUS checkpoint round-trip
  `round_trip_state_sha256_match == true`.
- REAL acceptance (still INTERNAL, never held-out): score the stage-1 checkpoint on the two DEFAULT_CLIPS via the
  SCORING BRIDGE below (a raw checkpoint CANNOT be fed to `run_ball_tracking_eval_suite.py` — it has no weights arg).
  Require `label_f1_at_20px` >= 0.6685 (WASB-tennis internal-val F1@20, ledger BALL-IV-1) AND `mean_visible_hit_recall`
  >= the WASB-tennis internal-val RECALL reference (NOT 0.6685 — that is an F1 figure; if the recall value is not
  recorded in BALL-IV-1/the run, obtain it before gating -> typed STOP: needs-validation) AND
  `mean_hidden_false_positive_rate` not worse than the WASB-tennis internal reference.
- Kill: proxy up but internal product F1 DOWN on Burlington/Wolverine = distractor-lock reproduced -> STOP,
  analyze source mix, do NOT take a held-out shot. A public-only student NEVER takes a held-out shot (4 inversions
  on record). Lane sizing: A100 spot, 1 GPU, ~1-2h wall for 12k steps (1.5M-param net, tiny). H100 unnecessary.

**SCORING BRIDGE — trained checkpoint -> product-gate score (load-bearing; every product-gate step routes here).**
`run_ball_tracking_eval_suite.py` has NO checkpoint arg — it consumes precomputed detector sidecars from a
`<run_root>/<clip>/` layout. To score ANY trained WASB checkpoint on the gate, regenerate the sidecars first:
1. Per DEFAULT_CLIP, run WASB inference from the checkpoint into the run-root:
   `python3 scripts/racketsport/run_wasb_ball.py --checkpoint <stage>/checkpoints/latest.pt
   --wasb-repo third_party/WASB-SBDT --video <clip.mp4> --fps <clip_fps> --candidate-top-k 5
   --out <run_root>/<clip>/wasb/ball_track.json` (writes `ball_candidates.json` alongside; the trainer saves
   `model_state_dict`, which run_wasb_ball loads via `wasb_adapter._checkpoint_state_dict`).
2. Fuse into the track the suite reads: `python3 scripts/racketsport/fuse_ball_tracks.py --primary-ball-track
   <run_root>/<clip>/wasb/ball_track.json --stable-ball-track <...> --out <run_root>/<clip>/<sidecar>.json
   --summary-out <...>` — the exact sidecar FILENAMES the suite ingests are registered in the suite's
   `_add_existing_candidate`/external-candidate/`--overlay-candidate` wiring (grep them, place files to match).
3. THEN score: `python3 scripts/racketsport/run_ball_tracking_eval_suite.py --run-root <run_root>
   --review-root <labels> --out-root <out> --clip burlington_gold_0300_low_steep_corner
   --clip wolverine_mixed_0200_mid_steep_corner`.
Held-out shots use the same bridge through `run_ball_chain.py --heldout-authorized` +
`benchmark_ball_tracks_against_cvat.py` under a pre-registered ledger row. STEPS 3/4/5/6 all route their
product-gate scoring through THIS bridge — there is no other checkpoint->score path.

**STEP 3 — STAGE-2 owner fine-tune (GPU lane) — BLOCKED until owner in-domain CVAT labels reach P0-4 VOLUME (a
~274-box first seed exists at `cvat_upload/exports/harvest_review_20260707/`; STAGE-2 needs the >=10-20k-frame budget).**
- Objective: fine-tune the stage-1 checkpoint on owner labels; this is the ONLY stage that has ever moved held-out.
- BUILD GAP (must be closed first, Codex lane): `train_ball_pretrain.py` reads ONLY `corpus_index` (Roboflow). The
  owner-CVAT feed lives in `threed/racketsport/ball_tracknet_cvat_dataset.py`
  (`build_ball_tracknet_cvat_dataset`, `dense_tracknet_labels_from_cvat`) which already emits `visibility_level` +
  `wbce_weight` per label via `_visibility_wbce_weight` -> `BALL_VISIBILITY_WBCE_WEIGHTS`
  (`threed/racketsport/schemas/__init__.py` lines 47-53: clear=1/partial=2/full=3/out_of_frame=3). Wire this
  loader into the trainer (or a stage-2 sibling script) AND add the two TOTNet pieces the harness lacks:
  (i) occlusion augmentation at `occluded_prob=0.25` (NOT implemented in `train_one_batch`), (ii) confirm the
  real per-sample `wbce_weight` (2/3/3) flows — it does, via the same `weights = batch["wbce_weight"]` multiply.
- Recipe: init from stage-1 checkpoint (`--init-checkpoint <stage1>/checkpoints/latest.pt`), AdamW lr=5e-4
  wd=5e-5, frames_in=3 (MUST equal the stage-1 checkpoint — frames_in sets input channels = frames_in*3, so
  changing it makes the init non-loadable and trips STEP 1's own KEY-DIFF STOP), 512x288, occluded_prob=0.25,
  WBCE 1/2/3/3, radius 4. Epochs: bounded (<=30, TOTNet default), checkpoint_every 500. (If dense-sequence 5-frame
  training is later wanted, it needs a FRESH stage-1 pretrain at frames_in=5, not a re-init of this checkpoint.)
- Acceptance: PRE-REGISTER a held-out ledger row FIRST. Then exactly ONE held-out shot. Bars (INTERIM, not full M1):
  Outdoor product-view `label_f1_at_20px` > 0.7248 AND recall (`visible_hit_recall`) >= 0.70 AND
  `hidden_false_positive_rate` <= 0.05. Select the candidate on Burlington/Wolverine internal-val BEFORE the shot.
- Kill: ANY internal->held-out inversion = STOP and analyze, never re-tune past it (the 4-inversions rule).
  Lane sizing: A100, 1 GPU. Owner-dependency: this step CANNOT start until owner labels reach P0-4 VOLUME (the
  ~274-box `harvest_review_20260707` seed is far short) -> typed STOP: labeling.

**STEP 4 — STAGE-3 SST bootstrap (GPU lane; nightly flywheel later).**
- Objective: lift the stage-2 student using owner UNLABELED footage via teacher-student pseudo-labels.
- Teacher = PHYSICS-GATED chain output (`scripts/racketsport/run_ball_chain.py`, hidden-FP 0.021), falling back to
  single zero-shot WASB-tennis (0.063) when the arc solver self-kills. NEVER the raw un-gated fusion (0.349 would
  bake correlated hallucinations into labels).
- Recipe: Vandeghen SST (github.com/rvandeghen/SST), 2 rounds, doubt/confidence-weighted student loss; active
  learning — frames where detectors disagree go to the human label queue first. Seed already exists: 40 WASB
  prelabel sidecars from the `p01b_prelabel_20260707` lane (40/40 shards DONE). DATA-LOCALITY caveat: only 6/40
  local; the producing VM `pickleball-a100-fleet2` is DECOMMISSIONED (DELETED 2026-07-07T04:49Z per `gpu_fleet.md`),
  so the 34 remote sidecars are gone — REGENERATE them on a fresh fleet VM via
  `run_wasb_ball.py --checkpoint <stage2 or wasb_tennis_best> ...` over the owner footage list before scaling.
- Acceptance: student beats teacher on INTERNAL-val (`label_f1_at_20px` on Burlington/Wolverine) WITHOUT held-out
  regression (pre-registered). Kill: pseudo-label recall inherits teacher blind spots and does not move recall
  after round 2 -> stop iterating. Lane sizing: A100; pseudo-labeling fans out ONE CLIP PER GPU across the fleet.

**STEP 5 — Fusion selection A/B (candidates vs voting) — pre-registered flip.**
- Objective: replace spatial-consensus voting with physics-consistency selection as the DEFAULT, only if measured.
- Protocol: pre-register a ledger row. Hold detectors as candidate generators (top-K sidecars already emit).
  Arm A = current voting/fusion default; Arm B = arc-solver physics-consistency selection over the same candidate
  pool. Score BOTH on Burlington/Wolverine internal-val first (`label_f1_at_20px`, `mean_hidden_false_positive_rate`).
  Flip the default to B ONLY if B's internal hidden-FP is materially lower at equal-or-better F1. THEN one
  pre-registered held-out shot to confirm no inversion. Kill: B inverts on held-out -> keep A, STOP.

**STEP 6 — Recall rescue to >=0.75 (P1-3), composable, MEASURE EACH SEPARATELY.**
Union candidate recall ceiling measured 0.8793; fused recall 0.578 must climb to >=0.75. Integration ORDER and
expected direction (measure, do not assume magnitudes):
  (a) SAHI-style tiled inference on hitter-adjacent regions during RALLY frames only (rally frames = in-rally
      `foot_contact_phases` from the phase producer in `threed/racketsport/placement.py`/`orchestrator.py`;
      hitter-adjacent region = bbox from the body/pose track; if those producers are absent, treat (a) as
      recipe-only and gate on full-frame inference) — biggest expected recall gain on small/distant balls;
      risk = hidden-FP up (gate on `mean_hidden_false_positive_rate`).
  (b) motion-channel input (frame differencing, TrackNetV4 pattern) — recover motion-blurred balls; recipe only.
  (c) lower detection threshold + physics-gated acceptance — arc solver prunes the added FPs (0.021 has headroom).
  (d) occlusion-recipe recovery (the stage-2 TOTNet model) — recover partial/full-occlusion frames.
  (e) RIFE-interpolated sub-frame recovery — RENDER-ONLY/derived, NEVER counted as measured evidence.
- Harness for each: add the lever, re-run `run_ball_tracking_eval_suite.py` on Burlington/Wolverine, diff
  `mean_visible_hit_recall` and `mean_hidden_false_positive_rate` vs the prior candidate. Keep a lever only if
  recall up AND hidden-FP not worse. Acceptance: BALL M1 recall bar on held-out with no hidden-FP regression.

### 3. Decision trees (if X -> do exactly Y)
- Stage-1 init reports non-empty `missing_keys`/`unexpected_keys` -> checkpoint amputated -> typed STOP: needs-validation.
- Stage-1 proxy `f1_at_20px` UP but internal product `label_f1_at_20px` on Burlington/Wolverine DOWN ->
  distractor-lock -> inspect per-source counts, drop the offending sources, re-pretrain; do NOT take a held-out shot.
- `wasb_tennis_best.pth.tar` missing on VM -> run STEP 1 prestage; if the MODEL_ZOO URL is dead ->
  typed STOP: needs-validation.
- Owner in-domain labels below STAGE-2 volume (a ~274-box seed exists at `cvat_upload/exports/harvest_review_20260707/`;
  STAGE-2 needs the >=10-20k-frame P0-4 budget) -> STAGE-2 cannot run yet -> typed STOP: labeling (public-only
  student is forbidden a held-out shot; do not substitute more public data for owner data).
- Any candidate: internal-val beats reference but held-out < internal (inversion) -> STOP, log the 5th inversion,
  do NOT re-tune; keep WASB-tennis 0.7248 as standing best.
- Stage-2 clears interim bar (F1>0.7248, R>=0.70, hFP<=0.05) but recall < 0.75 -> proceed to STEP 6 recall levers;
  the interim bar is a milestone, NOT the M1 gate.
- SST student does not beat teacher after round 2 -> stop SST iteration; the binding lever is more owner labels ->
  typed STOP: labeling.
- Fusion A/B: B not clearly better on internal hidden-FP -> keep voting/current default, do NOT spend a held-out shot.
- A recall lever raises `mean_hidden_false_positive_rate` above the prior candidate -> drop that lever, keep the rest.
- Wall-clock/quota: no idle A100 in `runs/manager/gpu_fleet.md` -> use the gpu-fleet-provision skill (a lane runs
  gcloud, never Fable/successor directly). Cost/speed unclear -> typed STOP: purchase-approval only if beyond cap.

### 4. DO-NOT list (kill-listed approaches + one-line reason/evidence)
- DO NOT ship or held-out-test a PUBLIC-ONLY student. Evidence: 4 internal->held-out inversions
  (`ball_t4_train_20260704/EVIDENCE_REPORT.md`, ledger rows 14/16/20/23).
- DO NOT use 2D spatial-consensus VOTING as the final answer. Hidden-FP 0.349 vs single WASB 0.063 — ensemble artifact.
- DO NOT feed raw un-gated fusion as the SST teacher. It bakes correlated hallucinations (0.349 hidden-FP) at scale.
- DO NOT cross-load blurball<->WASB-SBDT checkpoints without a key-diff; 36 SE keys, `strict=False` silently amputates
  (row 14 catastrophe). Always read `missing_keys`/`unexpected_keys`.
- DO NOT apply occlusion augmentation WITHOUT the visibility-weighted WBCE. Aug alone hurts: RMSE 29.6->54.3.
- DO NOT treat harness `f1_at_20px` as the product gate. It is a public-frame proxy; the gate is `label_f1_at_20px`.
- DO NOT count RIFE-interpolated recoveries as measured evidence — render-only/derived, always trust-band honest.
- DO NOT touch Outdoor/Indoor labels without a pre-registered `heldout_eval_ledger.md` row. eval_guard keeps
  Burlington/Wolverine never-gradient-trained.
- DO NOT re-tune past any inversion. Kill rule is absolute.
- DO NOT invent an LR scheduler/warmup for the harness — it uses constant AdamW by design (matches TOTNet).

### 5. External bets verified today (claim -> verdict -> source/URL -> license -> code availability)
- TOTNet occlusion recipe (WBCE 4-level + occlusion aug) -> REAL, current. arXiv 2508.09650 (Aug 2025); RMSE
  37.30->7.19, fully-occluded 0.63->0.80. https://github.com/AugustRushG/TOTNet -> MIT. Code+weights present
  (`/weights` dir; loss `third_party/TOTNet/src/losses_metrics/losses.py`, default weighted_list [1,2,2,3];
  our TTA recipe uses 1 2 3 3 per `src/train.sh`). Vendored at `third_party/TOTNet/`. Weights are TT-domain (recipe
  only, not a drop-in detector).
- WASB-SBDT (anchor + HLSM) -> REAL. arXiv 2311.05237, BMVC2023. https://github.com/nttcom/WASB-SBDT -> MIT
  (`third_party/WASB-SBDT/LICENSE.md`, NTT Communications 2023). HLSM = Hard-to-Localize Sample Mining
  (position-aware GT map applied only to mined hard samples). Vendored; anchor tennis checkpoint absent on Mac.
- TrackNetV4 motion-prompt layer -> REAL (recipe only). arXiv 2409.14543 (Sep 2024). Motion attention via frame
  differencing + motion prompt layer; +F1/+recall for TrackNetV2/V3. Upstream weights unusable (standing note);
  re-implement the layer, do not vendor weights.
- RacketVision -> REAL and NEWLY DOWNLOADABLE (material update, see §"new"). AAAI 2026 Oral, arXiv 2511.17045.
  https://github.com/OrcustD/RacketVision (MIT) + HF `linfeng302/RacketVision-Models`. Checkpoints downloadable via
  `python download_checkpoints.py`: ball `balltrack_best.pth`, racket pose `epoch_300.pth`, keypoint
  `best_PCK_epoch_90.pth`. Multi-sport joint-train +14-19% mAP; cross-attention (racket K/V, ball Q) beats concat.
- Vandeghen SST -> REAL. arXiv 2204.06859, CVPRW2022. https://github.com/rvandeghen/SST -> BSD-3. Teacher-student
  with 3 doubt-weighted loss parametrizations; the STAGE-3 recipe.

### 6. Acceptance gates + owner dependencies
- EXACT metric keys (copy verbatim): product gate reads `label_f1_at_20px`, `visible_hit_recall`
  (`mean_visible_hit_recall`), `hidden_false_positive_rate` (`mean_hidden_false_positive_rate`),
  `visible_recall_at_20px` from `threed/racketsport/ball_benchmark.py`. Harness proxy reads `f1_at_20px`,
  `recall_at_20px` from `train_ball_pretrain.py::evaluate()`.
- INTERIM gate (stage-2, one pre-registered held-out shot): Outdoor `label_f1_at_20px` > 0.7248 AND
  `visible_hit_recall` >= 0.70 AND `hidden_false_positive_rate` <= 0.05.
- TRUE M1 gate (I.2): F1@20 >= 0.90, recall >= 0.75, hidden-FP <= 0.05 on held-out; VERIFIED flips only here.
- OWNER-ONLY inputs (cannot be manufactured): (1) P0-4 in-domain CVAT ball labels at STAGE-2 VOLUME — a ~274-box
  first seed (4-level visibility) is DONE at `cvat_upload/exports/harvest_review_20260707/`, but STAGE-2's real fuel
  needs the >=10-20k-frame P0-4 budget (NORTH_STAR P0-4); (2) owner unlabeled footage for STAGE-3 SST; (3) explicit
  authorization for each held-out ledger row (joint-commit + eval discipline). Missing any -> typed STOP: labeling
  / decision. Do not proceed past a real blocker.

## PILLAR: BALL 3D — full-flight arcs, spin, contacts, in/out with honest uncertainty

Audience: the successor manager (Opus-class). Every step names exact files/functions (grep-verified
2026-07-07), exact hyperparameters, and exact metric keys. Where judgment could creep in it is
pre-ruled or marked "typed STOP". VERIFIED=0 for this pillar until a product gate passes on real
labels; nothing below claims otherwise.

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
Fable + owner decided a SEQUENCED single-system stack, not a bake-off. The production truth is the
event-anchored per-segment ODE solver (`threed/racketsport/ball_arc_solver.py`), which fits drag+gravity
free-flight arcs by reprojection against the 2D track with stored-anchor BVP shooting and a 74mm
size-depth range residual (already default-on). We (1) FIRST stabilize the anchor-first BVP path so
currently-good baseline intervals keep fit status, (2) THEN add Magnus as a single SCALAR spin S on a
fixed horizontal axis perpendicular to horizontal velocity (Cl=0.195·S Steyn; Cd 0.33/0.45 Lindsey
already in `PhysicsParameters`), (3) use the landed P0-7 flight simulator (`flight_simulator.py`,
reuses `_rk4_step`) to make noise/dropout-matched training data, (4) add a learned lift model
(UpliftingTT-style RoPE transformer trained purely on sim) ONLY as a narrow rescue for segments the
anchored solver cannot fit — never a parallel system. Full 3D spin vector is UNIDENTIFIABLE from our
single-view data (6 unknowns on as few as 3 obs) → scalar top/backspin only; spin-SIGN (P1-5) is
parked until H13 owner-measured court friction/restitution exists. Contacts (P1-6) = temporal
classifier over (track-window, audio onsets, wrist-cue distance) with heuristic cusp+gap fallback.
In/out (P1-7) always keeps a `too_close_to_call` gray zone — advisory, never officiating.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- Held-out BALL bar NOT yet beaten: 0.6969 vs 0.7248 (honest MISS), evidence in memory
  `pickleball-ball-chain-state.md`; zero-shot Roboflow fine-tune DEGRADED held-out (TrackNetV3 −17pt;
  WASB 0.0018 lock) — `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`.
- BVP anchor-first solver landed WIP, objective_result=PARTIAL:
  `runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json`. PASS: owned pytest 57, Wolverine seg6
  fallback in-court, D.3(a) violators all convert, D.3(c) 0 out-of-bounds render, D.3(d) Outdoor
  violation_fraction 0.428571→0.142857. FAIL (the open bug): D.3(b) — Burlington 4 + Wolverine 1
  exact-baseline currently-good intervals LOSE fit status after reselection; D.3(e) internal-val F1
  not run. Root-cause noted in report: cached event-subset scoring uses a ballistic proxy; LOO reuses
  the segment anchor-BVP arc instead of re-solving. Full BALL suite 365 passed / 1 preexisting-unowned.
- P0-7 flight simulator landed, objective_result=PARTIAL: `runs/lanes/p07_flightsim_20260706/report.json`.
  PASS: 1k corpus failed_segments=0; round-trip clean fit p95=0.063596m; noise match
  p95=34.21px / recall=0.5771 / hidden-FP=0.02148 (targets 34px/0.578/0.021); 1k in 40.7s; deterministic.
  FAIL: wide suite had 7 unrelated failures (6 sandbox TCP-bind EPERM, 1 A100 known_hosts drift) —
  NOT P0-7-owned. Corpus: `runs/lanes/p07_flightsim_20260706/flight_corpus_1000.jsonl`.
- Magnus math ALREADY exists in the simulator (`flight_simulator.py:729 _rk4_step_with_magnus`,
  `STEYN_CL_PER_SPIN=0.195` at :36, `_spin_axis_for_velocity` at :902) but is NOT in the solver fit path.
- Bounce restitution=0.58 / friction=0.16 are UNMEASURED priors (H13-pending), `flight_simulator.py:93`.
- Contact testbed: IMG_1605 (`eval_clips/ball/owner_IMG_1605_8a193402780b`) has 30 real audio onsets
  (`audio_onsets_v2.json` `onsets` field, first labels). No trained contact classifier yet.
- Confidence tiers live: `ball_arc_chain.py:793 _segment_confidence` → fit_bvp_fallback base 0.30,
  fit_weak 0.38, else 0.92. Viewer bands `web/replay/src/ballArcRender.ts`: anchored_measured /
  arc_interpolated / arc_extrapolated / arc_weak / hidden; `leastCertainBand` merges neighbors.

### 2. The exact build plan
Ordering is MANDATORY (do not reorder): STEP 1 gates STEP 2; STEP 3 exists but extends; STEP 4/5/6 are
downstream. Size guide: Codex = single-file deterministic-CPU numerics/tests; Sonnet/GPU = training or
multi-file. Every lane: dispatch via `/run-lane`, pin an explicit `model`, no commit unless owner says.

**STEP 1 — P1-4a: stabilize anchor-first BVP so good intervals keep fit status (Codex).**
- Objective: eliminate the D.3(b) regression BEFORE adding spin. Exact bug: reselection demotes 5
  currently-good baseline intervals (Burlington seg0/seg13-adjacent x4, Wolverine seg6-region x1).
- File targets: `threed/racketsport/ball_arc_solver.py` — `_fit_flight_segment_once` (:486), the
  event-subset selector `_select_event_subset` (:2234, invoked at :1887/:1925). The report says cached
  event-subset scoring uses a "ballistic proxy" while final fits run BVP/refinement — the suspect is the
  `selection_scoring="ballistic_initial_guess_no_bvp"` branch at :841 inside `_solve_bvp_shooting`
  (finite-diff Newton, def :845). LOO validation is `_leave_one_out_validation` (:3623); endpoint
  refinement is `_refine_bvp_endpoints` (:963). Grep the function NAMES — the lane report.json line
  numbers (:1866/:2279/:3422/:838/:960) have DRIFTED; re-grep against current HEAD before STEP 1 edits.
- Recipe (order MANDATORY): (a2 — try FIRST, cheap) an anchor-preservation rule: any interval that was
  `fit*` in the frozen baseline and still has ≥`min_segment_observations` (3) inliers may not be demoted
  below `fit_bvp_fallback` without a strictly-worse endpoint_error_m. (a1 — ONLY if a2 leaves D.3(b)
  failing; expensive, see §3) make selection scoring use the SAME BVP/refinement cost the final fit uses
  (remove the ballistic-proxy shortcut) on JUST the affected intervals. (b) Re-solve BVP per-holdout in LOO
  rather than reusing the segment arc. Keep `integrator_max_step_s=1/240`, `min_segment_dt_s=0.045`,
  `max_reprojection_inlier_px=18.0`, `robust_pixel_sigma=6.0` UNCHANGED.
- Acceptance (exact, from the lane report's own gate labels): D.3(b) — the 5 exact Burlington+Wolverine
  baseline good intervals (Burlington seg0/seg13-adjacent x4 + Wolverine seg6-region x1) stay
  `status.startswith("fit")` with `endpoint_error_m <= baseline`. Frozen-baseline source: the D.3(a)/(b)
  entries in the `acceptance` list of `runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json` (each
  dict's `baseline`/`after` fields name the intervals + endpoint_error_m); compare post-fix per-interval
  status/endpoint_error_m to those. If that report lacks exact per-interval numbers, git-stash the fix
  and regenerate the baseline from pre-change code FIRST. D.3(e) internal-val F1 no >1pt regression (RUN
  it this time); D.3(a)/(c)/(d) must NOT regress; owned pytest stays green.
- Fresh-GPU re-run (NOT cached): provision/reuse a GPU via `/gpu-fleet-provision`, then run the BALL
  chain on each of the 3 named clips (§STEP-2 list) via `scripts/racketsport/run_ball_chain.py` (or
  `process_video.py`) writing to `runs/lanes/<lane>/<clip>/`; the D.3 gates read the emitted
  `ball_track_arc_solved.json` per-segment `status`/`endpoint_error_m`. Diff against the frozen baseline.
- Kill criterion: if making selection BVP-exact blows CPU runtime unbounded (report already flags this
  risk), keep the proxy for SEARCH but add the anchor-preservation guard as the fix, and record the
  runtime delta. If good intervals still cannot be preserved without breaking D.3(a) → typed STOP: advice.

**STEP 2 — P1-4b: add scalar Magnus S to the solver fit (Codex).**
- Objective: let the solver fit one extra scalar S per segment so top/backspin arcs stop being
  free-fit approximations. Port EXACTLY the simulator's proven derivative — do not invent new physics.
- State extension: NONE. Keep 6-state (x,y,z,vx,vy,vz). S is a per-segment CONSTANT parameter, not a
  7th integrated state. Spin axis is FIXED per segment: `axis = (vy0/norm, -vx0/norm, 0)` from the
  segment's initial horizontal velocity (copy `_spin_axis_for_velocity`, `flight_simulator.py:902`).
- Force term (copy from `_rk4_step_with_magnus`, `flight_simulator.py:729`): inside `deriv` of a new
  `_rk4_step_magnus` in `ball_arc_solver.py` (mirror `_rk4_step` at :3922), after computing drag accel,
  add lift: `lift_dir = unit(cross(axis, v_hat))`; `lift_k = 0.5*rho*π*r²/mass`;
  `lift_acc = lift_k * speed² * (0.195 * S)`; add `lift_acc*lift_dir` to (ax,ay,az). Define
  `STEYN_CL_PER_SPIN=0.195` in the solver too (do NOT import from simulator — solver must not depend on
  it). S-threading DECISION (do exactly this, no re-litigation): add `spin_scalar: float = 0.0` as a
  keyword arg to `_integrate_positions` (:3888) AND `_rk4_step` (:3922) plus the new `_rk4_step_magnus`;
  route to the magnus stepper when `abs(spin_scalar)>1e-12`, else the plain `_rk4_step` (analytic no-drag
  shortcut only for spin_scalar==0.0). Thread spin_scalar ONLY from the three fit paths that OWN a
  segment S (`_fit_free_flight_segment_once` :611, `_refine_bvp_endpoints` :963, `_solve_bvp_shooting`
  :845); ALL other ~15 `_integrate_positions` call sites pass the default 0.0. Do NOT put S on
  `PhysicsParameters` (shared object → cross-segment spin leakage).
- Fit-procedure change: add S as the extra least-squares parameter in ALL THREE paths a segment can
  traverse — the velocity fit in `_fit_free_flight_segment_once` (:611), the [dp0,dp1,dt0,dt1] vector in
  `_refine_bvp_endpoints` (:963), and `_solve_bvp_shooting` (:845) must receive the current S so its
  integration matches (every path carries the SAME S). Initialize S0=0.0. BOUNDS on S: `|S| <= 0.8`
  (matches the simulator's `rng.uniform(-0.8,0.8)` shot-family band; hard-clip). Regularize toward 0:
  add residual `sqrt(lambda)*S` with `lambda=0.05` so S only moves when the data demands it (prevents
  spin absorbing detector noise on short/side-view segments). Gate: only FIT S when segment has
  `>= 6` inlier observations AND view-geometry confidence is not back-view-degraded (STEP-2 uses the
  same per-segment view-confidence STEP 5 computes; until STEP 5 lands, require `>=8` inliers as proxy).
- Acceptance: (a) owned pytest green incl a new test that a known-S simulator trajectory
  (`generate_trajectory_pair`, spin_scalar=0.5) round-trips to recovered `|S_hat-0.5|<=0.15`; (b) on
  the 3-clip set (= `eval_clips/ball/{burlington_gold_0300_low_steep_corner,
  wolverine_mixed_0200_mid_steep_corner, outdoor_webcam_iynbd_1500_long_high_baseline}`, per the report's
  "3-clip D.3(a)" set), `reprojection_rmse_px` mean does NOT increase and improves on the steepest-launch
  high-arc segment `wolverine_mixed_0200 seg6`; (c) D.3(a)/(b)/(c)/(d) all still PASS; (d) internal-val
  F1 no >1pt regression.
- Kill criterion: if enabling S regresses any D.3 gate or inflates rmse on ≥1 clip, default S OFF
  (`fit_spin_scalar=False` config flag, S pinned 0) and ship the plumbing dormant; record which
  segments benefited. Do NOT attempt full 3-axis spin — UNIDENTIFIABLE (see DO-NOT). Do NOT attempt
  spin-SIGN disambiguation — parked on H13.

**STEP 3 — P0-7 extension: harden sim as the training oracle (Codex, CPU).**
- Objective: the simulator already emits noise-matched pairs; extend it so its label distribution
  matches our real error profile tightly enough to train STEP 4. File: `flight_simulator.py`
  (`DetectorNoiseProfile` p95_jitter_px/recall=0.578/hidden_fp_rate=0.021; `generate_corpus` :640;
  CLI `scripts/racketsport/generate_flight_corpus.py`).
- Recipe: (a) widen shot-family sampler `sample_shot_family` (:235) to cover serve/drive/dink/lob
  speed+launch-angle bands (document each as unmeasured prior, not constant); (b) add
  occlusion-BURST dropout (contiguous missed frames, not just iid) to match real tracker gaps; (c)
  keep bounce restitution=0.58/friction=0.16 but TAG every bounced record `bounce_params_measured=false`
  so STEP 4 can exclude bounces until H13. Generate via `scripts/racketsport/generate_flight_corpus.py`:
  `--count 50000 --seed 20260707 --out runs/lanes/<lane>/flight_corpus_train_50000.jsonl` and `--count
  5000 --seed 20260708 --out runs/lanes/<lane>/flight_corpus_val_5000.jsonl` (seeds MUST differ; record
  both in the lane report). STEP 4 reads exactly these two paths.
- Acceptance (exact keys from report): corpus `failed_segments=0`, `demoted_frames=0`; noise stats
  `jitter_p95_px` within 20% of 34, `recall` within 20% of 0.578, `hidden_fp_rate` within 20% of
  0.021 (`within_20_percent=true`); determinism test passes.
- Kill criterion: if a widened family produces physically-impossible arcs (fails
  `ball_flight_sanity`), narrow that band and log it. If bounce dynamics look wrong → keep bounces out
  of the training corpus (STEP 4 fits free-flight only) — this is fine, not a blocker.

**STEP 4 — P1-4c: learned-rescue lift model, trained purely on sim (Sonnet + GPU).**
- Objective: a narrow rescue ONLY for segments the anchored solver returns non-`fit` (i.e.
  `fit_bvp_fallback`, `fit_weak`, or blocked) — never runs on segments the solver already fits.
- Architecture: small RoPE-timestamp transformer (UpliftingTT-style, see §5), ~4-6 layers, d_model
  128-256, trained to lift an unsegmented noisy 2D track → 3D positions (+ optional scalar S head).
  New files: `threed/racketsport/ball_lift_net.py` (model + inference) and
  `scripts/racketsport/train_ball_lift_net.py`. Do NOT import UpliftingTT weights (GPL-3.0 — internal
  use OK per owner ruling but re-implement architecture; note license in the lane report).
- Input featurization (exact): per visible frame — normalized 2D `xy_px` (÷image_w,÷image_h), a
  visible/dropout flag, and camera ray direction from `calibration` (back-project pixel through
  intrinsics+extrinsics — reuse `project_world_points` inverse / the calibration in the sim records);
  timestamps go in via RoPE using the record `t` (seconds), NOT frame index (VFR-safe). Target:
  per-frame `world_xyz_m` from the sim `truth_3d`.
- Training-data protocol: STEP-3 50k corpus for train, 5k different-seed for val; free-flight only
  (exclude `bounce_params_measured=false` bounces); loss = mean 3D position L2 + `0.1`·S-head L1.
- Gating rule (exact, non-negotiable): rescue applies IFF `status == "fit_bvp_fallback"` OR
  `status == "fit_weak"` OR `status.startswith("blocked:")` (NOT `== "blocked"` — `_blocked_segment`
  :4175 emits `f"blocked:{reason}"`, e.g. "blocked:nonfinite_anchor"; exact equality never matches).
  If solver status `startswith("fit")` and not weak/fallback,
  the SOLVER output wins — the net never overrides it. When the net produces a rescue arc, it is
  emitted at confidence tier `arc_weak` band and `_segment_confidence` base capped at 0.38 (same as
  fit_weak); never `anchored_measured`.
- Acceptance (make-or-break kill-gate — exact protocol): run `scripts/racketsport/solve_ball_arcs.py`
  over the 5k val corpus, record per-segment `status` and 3D-position p95 vs `truth_3d`; the comparison
  set = segments where solver `status=="fit_bvp_fallback"`; compute the net's p95 on that SAME subset,
  emit both p95s side-by-side in the lane report. Net p95 MUST be <= solver `fit_bvp_fallback` p95 on
  that matched set (net must beat the thing it replaces); on the 3-clip real set, rescued segments
  produce in-court arcs (0 `outside_court_volume`) and do NOT worsen any D.3 gate; internal-val F1 no
  >1pt regression.
- Kill criterion: if the net does not beat `fit_bvp_fallback` on degraded segments, SHIP NOTHING —
  keep the fallback. A learned system that only ties the fallback is dead weight. Do not widen its
  gating to compete with the solver → typed STOP: decision if tempted.

**STEP 5 — P1-4d: per-segment view-geometry confidence bands (Codex).**
- Objective: trust bands from camera geometry (TT3D: 12.4cm side-view vs 29.8cm back-view error).
- File: `ball_arc_chain.py:_segment_confidence` (:793) + `ballArcRender.ts`. Compute per segment the
  angle between the segment's mean velocity direction and the camera optical axis; scale `base`
  confidence down when the arc is viewed near-end-on (back-view, depth poorly observed). Feed the same
  view-confidence into STEP 2's S-fit gate (back-view → do not fit S). AFTER this lands, RETURN to
  STEP 2 and replace its interim `>=8`-inlier proxy S-gate with the computed back-view test, then re-run
  STEP 2 acceptance (b)/(c).
- Acceptance: known side-view segments keep high band; synthetic back-view segment demotes to
  `arc_weak`; no D.3 regression.

**STEP 6 — P1-6 contact-event classifier v0 (Sonnet, CPU; owner audio labels).**
- Objective: temporal classifier emitting contact timestamps; timing gate <=40ms p90.
- Features (exact): sliding window over (a) 2D track curvature/kink + inter-frame gap flags, (b) audio
  onset times, (c) wrist-cue distance = min 3D distance ball↔player wrists per frame. Wrist 3D positions
  come from the body/world-HMR skeleton `<clip>/skeleton3d_pre_grounding_refine.json` (dict with
  `joint_names` incl `left_wrist`/`right_wrist`, `players`, metric `world_frame`); if that artifact is
  absent for a clip, drop the wrist-cue feature and fall back to track-kink + audio only. Labels source:
  IMG_1605 = `eval_clips/ball/owner_IMG_1605_8a193402780b`; its 30 real audio onsets are the `onsets`
  field of `<run>/owner_IMG_1605_8a193402780b/audio_onsets_v2.json` (e.g.
  runs/fix3_img1605_20260704T124052Z/…; `onset_count`=30, ONLY testbed today). New files:
  `threed/racketsport/contact_classifier.py`, `scripts/racketsport/detect_contacts.py`. **EMISSION CONTRACT (PART B ruling B.1.4): the classifier is a new SOURCE inside `event_fusion.fuse_contact_windows` — `contact_windows.json` stays THE artifact; heuristic sources remain as fallback weights; never a parallel contact artifact.**
- Train/val: leave-one-onset-out or temporal split on the 30 IMG_1605 onsets (tiny — expect a
  threshold/logistic model, NOT a deep net). Keep the existing heuristic cusp+gap anchors as FALLBACK
  whenever the classifier confidence is low or audio is absent.
- Acceptance: contact timing error p90 <= 40ms on IMG_1605 held-back onsets; fallback path still
  emits anchors when classifier abstains.
- Kill criterion: 30 onsets from ONE clip is not enough to generalize — do NOT claim VERIFIED; ship as
  advisory v0 and mark it. More clips with audio onsets → typed STOP: labeling (owner must provide).

**P1-7 in/out (fold into STEP 5/6 outputs, Codex):** classify ball-landing vs court lines using the
3D bounce point + its uncertainty ellipse. Ellipse source: the BVP endpoint covariance from the
`least_squares` Jacobian at the bounce anchor (`_refine_bvp_endpoints` :963), projected to the court
plane; `too_close_to_call` triggers when the 2-sigma ellipse overlaps a line. ALWAYS keep the gray zone.
NEVER emit an officiating-grade boolean. No new gate beyond "gray-zone present and honest".

### 3. Decision trees (if X → do exactly Y)
- STEP 1 D.3(b) still fails after anchor-preservation guard → try selection-scoring BVP-exact on just
  the affected intervals; if runtime explodes AND intervals still demote → typed STOP: advice (surface
  the runtime-vs-fidelity tradeoff to owner).
- STEP 2 recovered S from a spin_scalar=0.5 sim traj lands `|S_hat-0.5|>0.15` → the port has a sign or
  axis bug; diff your `deriv` against `flight_simulator.py:729` line-by-line; the cross-product order
  `cross(axis, v_hat)` and axis `(vy,-vx,0)` are load-bearing. Do NOT ship S until the round-trip test
  passes.
- STEP 2 enabling S regresses any D.3 gate → set `fit_spin_scalar=False`, ship plumbing dormant, log
  benefited segments. Not a STOP — this is the pre-ruled fallback.
- STEP 4 net ties but does not beat `fit_bvp_fallback` → ship nothing (kill). If someone proposes
  making it a parallel primary system → typed STOP: decision (violates standing ruling).
- Anyone proposes fitting full 3-axis spin or spin-SIGN before H13 → typed STOP: decision (both
  UNIDENTIFIABLE / parked; see DO-NOT).
- Contact classifier asked to generalize beyond IMG_1605 → typed STOP: labeling (owner audio clips).
- Any request to touch Outdoor/Indoor held-out CVAT labels for a metric → typed STOP: needs-validation
  (requires a pre-registered `heldout_eval_ledger.md` row FIRST; never touch labels otherwise).
- Bounce dynamics needed (restitution/friction) for spin-sign or in/out precision → typed STOP:
  needs-validation (H13 owner court-surface measurement does not exist).

### 4. DO-NOT list (kill-listed approaches + one-line reason)
- DO NOT fit a full 3D spin vector — 6 unknowns on as few as 3 obs, UNIDENTIFIABLE from single view
  (standing ruling; TT4D needs a learned prior + more views to even attempt spin).
- DO NOT attempt spin-SIGN (top vs back disambiguation via bounce) before H13 measured
  restitution/friction — unidentifiable without surface constants (`flight_simulator.py:93` priors
  are unmeasured).
- DO NOT make the learned lift net a parallel/primary system — it is a NARROW rescue only, gated to
  non-fit solver segments (standing ruling).
- DO NOT rebuild the 74mm size-depth range residual — already default-on in all three fit paths
  (`enable_size_depth_residual`).
- DO NOT import UpliftingTT / TT3D / TT4D weights into the shipped product path — GPL-3.0 (UpliftingTT)
  is copyleft; re-implement the architecture, train on OUR sim. Internal experiments fine per owner
  ruling; note license in every lane report.
- DO NOT fine-tune the 2D detector on Roboflow-only data again — measured DEGRADATION held-out
  (`ball_t4_train_20260704/EVIDENCE_REPORT.md`), wrong domain + 84% temporally-dead frames.
- DO NOT emit officiating-grade in/out — single-camera officiating does not exist anywhere
  (PlayReplay=4 cams, Hawk-Eye 6+); keep `too_close_to_call`.
- DO NOT add spin as a 7th integrated ODE state — it is a per-segment constant parameter; extending
  state changes the whole solver needlessly.
- DO NOT claim VERIFIED for the contact classifier on 30 onsets from one clip — advisory v0 only.

### 5. External bets verified today (2026-07-07)
- Uplifting Table Tennis (WACV'26) → REAL, code+weights available. github.com/KieDani/UpliftingTableTennis;
  arXiv 2511.20250; site kiedani.github.io/WACV2026. License GPL-3.0 (copyleft — re-implement, don't
  ship their weights). Trains 2D→3D uplift on MuJoCo synthetic data; weights via torch.hub; training
  code present. RoPE-timestamp not explicitly confirmed in README (paper-level; treat RoPE as OUR
  design choice, not a claim about theirs). → use as architecture template for STEP 4.
- TT3D (CVPRW'25) → REAL, code available. github.com/cogsys-tuebingen/tt3d; arXiv 2504.10035. Per-
  segment drag+Magnus ODE reprojection fit = OUR arc-solver architecture (external validation).
  License NOT confirmed from search (org repos vary MIT/BSD/Apache) — verify LICENSE file before any
  reuse; we only borrow the METHOD, already independently built.
- Hybrid physics+NN (arXiv:2503.18584, "A Universal Model Combining Differential Equations and Neural
  Networks") → REAL, pickleball split CONFIRMED: 90 train / 103 test trajectories, 300–1000ms,
  0.97ms inference. No public code link found in search → treat as method inspiration only, NOT a
  dependency.
- MuJoCo MJX (phase-2 sim) → REAL, active. `pip install mujoco-mjx`; `from mujoco import mjx`; latest
  release 2026-06-22; Python>=3.10; NVIDIA/AMD/Apple-Silicon/TPU. Only needed if/when we move sim to
  GPU physics — Phase-1 sim is pure-numpy and does NOT need it yet.
- NEW since Jul 5 — TT4D (arXiv:2605.01234, May 2026) → REAL paper, learned lifting net on
  UNSEGMENTED 2D track, infers spin, handles occlusion; 140+hr dataset. Code/weights NOT confirmed
  released → WATCH-LIST, validates STEP-4 direction, do NOT build on until code lands.
- NEW since Jul 5 — MFS "Multi-Focus Temporal Shifting" event spotting (arXiv:2507.07381) →
  plug-and-play temporal module; Table Tennis Australia PES benchmark 4,878 events incl serve/bounce.
  Relevant to STEP-6 contact spotting IF we later need a learned spotter; code availability not
  confirmed → note only, our v0 stays the audio+kink+wrist fusion.

### 6. Acceptance gates + owner dependencies (exact metric keys)
- STEP 1 gate keys (from `ball_p3a_bvp_anchor_first_20260705/report.json`): D.3(a) all violators
  convert; D.3(b) exact baseline good intervals `status.startswith("fit")` & `endpoint_error_m<=baseline`;
  D.3(c) 0 out-of-bounds render samples; D.3(d) Outdoor `violation_fraction` improves; D.3(e)
  internal-val F1 no >1pt regression.
- STEP 2 keys: `reprojection_rmse_px` (non-increasing), round-trip `|S_hat-0.5|<=0.15`.
- STEP 3 keys: corpus `failed_segments=0`, `demoted_frames=0`; noise `within_20_percent=true` for
  `jitter_p95_px`/`recall`/`hidden_fp_rate`.
- STEP 4 keys: sim-val 3D position p95 (beat `fit_bvp_fallback`); real-clip `outside_court_volume`=0.
- STEP 6 key: contact timing error p90 <= 40ms.
- PROMOTION MILESTONE (necessary, NOT the VERIFIED flip — see PART B ruling B.1.2: BALL VERIFIED flips only at the full M1 gate F1>=0.90/recall>=0.75/hidden-FP<=0.05 held-out): held-out BALL F1 >= 0.7248 on the pre-registered heldout ledger
  row — NOT any internal number. VERIFIED stays 0 until this passes.
- OWNER-ONLY dependencies (cannot proceed without): (a) H13 court friction/restitution measurement →
  unblocks spin-SIGN + precise bounce/in-out; (b) more clips with audio onsets → unblocks contact
  classifier generalization beyond IMG_1605; (c) a pre-registered `heldout_eval_ledger.md` row before
  ANY held-out label is read; (d) commit/push permission (joint-commit rule); (e) in-domain owner game
  data → the real unlock for beating the held-out BALL bar (Roboflow alone proven insufficient).

## PILLAR: BODY — raw-noise kill (latent smoothing), camera motion, far players, GT, challenger protocol

**[WAVE-3 CLOSEOUT CORRECTIONS 2026-07-07 — supersede conflicting lines below]:** (1) STEP 0 is
DONE: slide gate GREEN 4/4 on fresh GPU proof @ ad75c875c (max_foot_lock_slide_m 20.25/22.50/17.98/
16.66mm vs 30mm frozen bar; p95 <12mm; 0 root jumps; fix survived 3 adversarial-verify rounds).
(2) grounding_refine = honest NO-OP posture (0 confident phases on eval clips); un-kill requires
UPSTREAM per-foot attribution at source — wave-4 queue #2. (3) Camera-motion motion-conditional
landed but PARTIAL: img1605 in-pipeline probe scored 0.329 vs 53.7 offline (probe-context bug) →
auto-OFF; diagnosis = wave-4 queue #1. (4) Version-stamp/code-sync LIVE-PROVEN (73 files, 0 drift,
4/4 stamp echo).**

> Audience: the successor manager. Every step names exact files/functions (grep-verified 2026-07-07),
> exact metric keys copied from gate code, and a decision tree. Reserved word `VERIFIED` = passed
> PRODUCT gate on real labels. Current state: **VERIFIED=0**. Do not write text that overclaims.
> You DELEGATE these to Codex/Sonnet lanes (use the `run-lane` skill). You never implement.

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
Fable + owner FROZE the backbone: **SAM-3D-Body + MHR70** (per-frame mesh model), scored by OUR
classical world-grounding chain (person-masked LK+MAD camera tracking, foot-plane anchoring, MAD+
Gaussian smoothing) — the exact architecture that WON the FIFA WorldPose/SMART challenge (+38.6%),
whose learned temporal-refiner LOST on held-out by over-smoothing. Challengers (GVHMR, PromptHMR-Vid,
Human3R, DuoMo, WHAM, JOSH3R) are **benchmark-only** and may replace the stack ONLY via the P2-7
pre-registered decision rule (§2 step E). The five open BODY jobs are: **P2-2** latent smoothing (THE
raw-noise fix — smooth the MHR pose-code sequence, decode via a wrapped vendored MHRHead), **P2-1**
camera-motion motion-conditional default, **P2-3** far-player high-res crop re-inference, **P2-5**
IDF1 wiring, **P2-6** independent GT. All standing rulings below are DECIDED — re-deciding any is a
failed mission.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- **Root-jump: WON.** outdoor 55→0, burlington 24→1 (survivor 10.04 vs 10.0 review floor). Root cause
  = hardcoded 30fps frame-index in `placement.py` visual root-step rewrite on 60fps clips (ABAB).
  Evidence: `runs/lanes/wave2_freshworlds_20260707/`, BUILD_CHECKLIST [CLOSING RUN RULED 2026-07-07].
- **Foot-slide MAX gate: OPEN (the sole carried BODY blocker).** Gated key
  `grounding_metrics.max_foot_lock_slide_m` ≤ 0.03 (`body_grounding_quality.py:12 DEFAULT_MAX_FOOT_SLIDE_M`).
  Closing run: burlington 40.6mm, outdoor 56.0mm, wolverine 18.4mm (PASS), img1605 25.6mm (PASS).
  p95 UNDER bar everywhere ⇒ **outlier-frame driven**. Root cause (2 independent diagnoses converged):
  100% of consumed contact phases are confidence-free `bilateral_from_player_stance` placeholders with
  `source_phase_foot: unknown` (exact-foot BODY agreement 0.363–0.651). Evidence:
  `runs/lanes/w3_slidediag_20260707/REPORT.md` §6, `runs/lanes/w3_groundref_diag_20260707/REPORT.md` §5.
  MAD bone-length ruled INNOCENT (engages 0 frames; [MAD A/B RULED 2026-07-07]).
- **grounding_refine: self-kills 4/4** (predates wave-2). Same root cause as slide-max (weak phases).
- **Camera-motion module: HARDENED, default-OFF.** `camera_motion.py` person-masked LK+MAD+2-pass
  smoothing; img1605 handheld wins all 3 proxies (inlier .767→.895, jerk 2.62→2.45, court-line 14.84→
  11.75px), 2× faster (98→50ms/f). Default-stage kill-criterion FIRED (wolverine placement jitter p90
  +1%). Motion-CONDITIONAL AUTO probe landed UNRULED: `estimate_camera_motion_probe` @
  `CAMERA_MOTION_AUTO_THRESHOLD = 2.5` (camera_motion.py:17). Evidence: `runs/lanes/p21_cammotion_20260706/`,
  `runs/lanes/w3_cammotion_conditional_20260707/`.
- **P2-2 emission prerequisite: DONE.** `global_orient`/`body_pose`(euler)/`betas`/`left_hand_pose`/
  `right_hand_pose` schema'd end-to-end in the per-sample frame dict returned by
  `worldhmr.py::_ground_fast_sam_sample` (function def line 2231; schema keys returned lines 2274-2278 —
  there is NO function named `_frame_from_sample`; grep the string `grounding_anchor` to land on the
  return dict). Decode-back path is the gap (§2-A).
- **IDF1: NOT wired.** 3 hardcoded `idf1=None` at `scripts/racketsport/process_video.py:954, 968, 1041`
  (`derive_track_trust_band`). Working scorer exists: Burlington IDF1 0.9112 recorded.
- **GVHMR spike: DONE, premise CORRECTED** (`runs/lanes/p27a_gvhmr_spike_20260706/`): external-pose+
  gravity is train-dataloader-only; no tripod benefit; single-person-hardcoded (4 players = 4 runs).

### 2. The exact build plan
> **EXECUTION ORDER ≠ letter order.** Do **STEP 0 FIRST** (it closes the sole carried blocker), then the
> §6 sequencing: **STEP D** (P2-5) and **STEP B**, then **STEP A** (P2-2, after its decode wrapper +
> round-trip gate), then **STEP C** and **STEP E**. Letters A–E are labels, not sequence.

**STEP 0 — close the carried foot-slide MAX blocker FIRST (w3_phasefix promotion run).**
- The `w3_phasefix_20260707` lane has ALREADY LANDED the fix code (per-foot confidence-bearing phases +
  weak-bilateral demotion + `foot_lock_gate_stream` instrumentation; offline replays PASS at 0.000m
  consumed surrogate on all 4 clips — report.json `objective_result: PARTIAL`, blocked ONLY on the WIDE
  suite, not on missing implementation). It is NOT promoted: `max_foot_lock_slide_m` is a fresh-GPU-only
  measurement (offline replays CANNOT clear it — [ROOTJUMP VERIFY RULED]).
- NEXT ACTION: dispatch a **Sonnet GPU** fresh-worlds run (acceptance in
  `runs/lanes/w3_phasefix_20260707/spec.md`) to measure the gated max on burlington/outdoor/wolverine/
  img1605 with the phasefix code. If all ≤0.03 with confident per-foot phases consumed → blocker CLOSES.
  If it still FAILs → follow the §3 slide-max branch (max-vs-p99 statistic = owner needs-decision STOP;
  NEVER re-tune 0.03).

**STEP A — P2-2 Latent-space temporal smoothing (THE raw-noise fix; the big one).**
- Objective: replace world-JOINT smoothing (mesh/skeleton can diverge) with smoothing of the MHR
  **pose-code sequence**, then decode → mesh+skeleton+70 joints move coherently by construction.
- Where codes emit: MHR head `third_party/Fast-SAM-3D-Body/sam_3d_body/models/heads/mhr_head.py::MHRHead`
  (class line 87, `forward` line 802). The latent/pose-code = `pred = self.proj(x)` (npose dims);
  the coherent-continuous slice is `pred_pose_raw = cat([global_rot_6d(6), pred_pose_cont(260)])` =
  266-dim (`body_cont_dim = 260`). Shape=`num_shape_comps 45`, scale=`num_scale_comps 28` are the
  per-subject lock targets. Our schema already carries `global_orient`/`body_pose`(euler)/`betas`.
- Decode wrapper (THE real gap): expose a local re-decode. Do NOT edit vendored files in place —
  add `threed/racketsport/mhr_decode.py` that imports `MHRHead`, loads the same checkpoint the
  pipeline loads, and calls the FK/skinning from `pred_pose_raw`+shape+scale → joints/vertices. Reuse
  the existing euler↔cont converters in the head (`compact_cont_to_model_params_body_fast`).
  **DATA-FLOW GAP:** the schema persists `body_pose` as EULER (~63 dims), NOT the 260-dim continuous
  code — so the wrapper must first re-encode euler→cont via `compact_model_params_to_cont_body`
  (mhr_head.py:30) to rebuild `pred_pose_raw`, then decode via `compact_cont_to_model_params_body_fast`.
  Verify euler→cont→euler is idempotent to <0.1° BEFORE smoothing; if not, PERSIST the raw 266-dim cont
  code in the emission schema (GPU-side schema change) so smoothing acts on the TRUE latent, not a lossy
  reconstruction. If neither is faithful → typed STOP: needs-validation. Verify
  round-trip: decode(emit(frame)) reproduces persisted `joints_world`/`vertices_world` to ≤1mm before
  touching smoothing (regression gate). This runs GPU-side (BODY dispatch), so the lane is a
  **Sonnet GPU lane** on fleet, NOT Codex — decode needs the real MHR checkpoint + torch/CUDA.
- Smoothing recipe (blueprint = arXiv 2512.21573, qualitative-only — prototype against OUR defects):
  (1) **Shape/scale lock**: per player per clip, fix `pred_shape`(45) + `pred_scale`(28) to the
  per-track median (constant bone length — same intent as the MAD bone-length detector already
  wired). (2) **Sliding-window optimization in MHR latent** over `pred_pose_raw`(266): window W=**9**
  frames, stride 1, objective = data term ‖code_t − code_t^raw‖² + λ_smooth·‖Δ²code_t‖² (2nd-diff /
  acceleration) + λ_foot·soft-foot-contact penalty (zero-velocity at contact frames from
  `foot_contact_phases`). Start λ_smooth=**0.3**, λ_foot=**1.0**; sweep λ_smooth∈{0.1,0.3,0.6}.
  (3) Decode smoothed codes → joints/vertices → feed existing world-grounding.
- **λ_foot dependency:** the soft-foot-contact term is only meaningful once CONFIDENT per-foot phases
  exist — i.e. AFTER STEP 0 (the w3_phasefix fresh-GPU promotion) passes. Today 100% of consumed phases
  are weak `bilateral_from_player_stance` placeholders that w3_phasefix REJECTS (consumed=0). Until STEP
  0 lands, set λ_foot=0 and prototype the smoothness term alone, or serialize P2-2 after STEP 0 — do NOT
  tune λ_foot against all-rejected placeholder phases.
- SOMA-X interop (Apache): use `py-soma-x` (§5) only if you need an SMPL-X prior (SmoothNet/DPoser-X)
  — NOT required for the core spike. Keep it optional to bound scope.
- Acceptance (EXACT keys, `visual_quality.py::measure_visual_quality`, per-player):
  `players[pid].world_jitter_mm_per_frame2.{feet,wrists,root}` raw 2–8 cm/f → **≤1 cm effective**;
  `foot_slide_mm_per_frame.stance.p95`; **wrist swing-peak 0-frame-delta** (harness:
  `pose_temporal.py::compare_wrist_peak_timing`, `refine_sam3d_skeleton3d(..., max_wrist_peak_delta_frames=1)`);
  **mesh-skeleton divergence ≤ 5mm p95** (new metric = per-frame ‖decoded-joint − skinned-mesh-joint‖
  p95 in the decode wrapper); downstream **paddle-stability** (paddle lane measured skeleton noise as
  its binding constraint). No regression on `foot_slide_gate` (0.03) or root-jump.
- Kill → hybrid: if latent smoothing over-smooths fast swings (SMART's failure mode; wrist peak delta
  >1 frame or wrist jitter reduction erases the swing), fall back to **hybrid**: latent smoothing for
  torso/legs joint groups, classical one-euro wrist protection unchanged (the near-pass-through "feet"/
  wrist params in `pose_temporal.py` are already tuned — reuse `_joint_smoothing_group`). Do NOT ship
  a global latent smoother that touches wrists if the peak-timing gate moves.

**STEP B — P2-1 camera-motion motion-conditional default: ALREADY IMPLEMENTED — RULE + COMMIT it.**
- Status: the probe-then-conditional-enable recipe is ALREADY WIRED (uncommitted working-tree change —
  `git status` shows `M scripts/racketsport/process_video.py`) and ALREADY MEASURED end-to-end on all 4
  eval clips by `runs/lanes/w3_cammotion_conditional_20260707/` (report.json `objective_result: PARTIAL`
  — every acceptance row PASS except the WIDE-suite classification). This is NOT a job to re-wire.
- **DO NOT** re-dispatch a Codex "CPU wiring" lane against
  `runs/lanes/p21_cammotion_20260706/deferred_patches/` — those patches are STALE (`git apply --check`
  FAILS on both `default_stage_wiring` and `placement_consumption_hook` today, forward and reverse);
  re-running them collides with the landed uncommitted work.
- NEXT ACTION: **RULE** the existing lane — reconcile the WIDE-suite failures as cross-lane/sandbox-
  suspect (the report already classifies them) or fix them, then **COMMIT** (owner joint-commit rule).
  Files it touched: `process_video.py`, `camera_motion.py::estimate_camera_motion_probe` (threshold 2.5,
  camera_motion.py:17), placement consumption hook.
- Measured acceptance (from the lane report — all PASS): img1605 AUTO enabled (score 53.7 > 2.5) with
  3/3 hardened handheld proxies retained (inlier .767→.895, jerk 2.62→2.45, court-line 14.84→11.75px);
  wolverine AUTO OFF (score 0.13 ≤ 2.5), static path bit-identical (`jitter_after_p90_mean` 2.244 mm/f²
  unchanged); burlington/outdoor AUTO OFF (score 0.52/0.57). NOTE: img1605 foot-slide already PASSes at
  25.6mm (§1) — camera-motion's win is the handheld PROXIES above, NOT a foot-slide fix (the 330mm
  figure from earlier drafts was unsourced; drop it). Kill guard: if any static clip's placement jitter
  p90 regresses >1% vs its OWN default-OFF baseline p90 (record all 4 baselines from the lane report
  before enabling AUTO, so the 1% threshold is computable per clip), keep default-OFF and ship the probe
  as advisory only.
- RAFT-small upgrade (pending): weights `raft_small_C_T_V2-01064c6d.pth`, sha256 staged
  `runs/lanes/w3_labelfactory_20260707/raft_sha256.txt`, URL confirmed §5. Prefetch on a
  network-capable Sonnet lane, then A/B RAFT vs LK+MAD; adopt only if it beats LK+MAD (which already
  wins). This is opportunistic, NOT blocking.

**STEP C — P2-3 Far/small-player high-res crop re-inference.**
- Objective: re-run SAM-3D only on far players at upscaled crop res. Input is HARD-CAPPED 384/448/512px
  — whole-frame res is NOT a lever; the crop path is the only family.
- Scheduler integration: `threed/racketsport/frame_rating.py::build_frame_compute_plan`
  (`DEFAULT_TARGET_MESH_FRAME_BUDGET = 200`, line 22) already prioritizes the hitter via contact-dense
  logic; the executor is `threed/racketsport/body_compute.py::_contact_dense_execution_frames` /
  `_contact_dense_player_targets`. Extend the SAME budget logic: add a bbox-height threshold selector
  (player bbox height < **T px**, start T=**160**) that flags far players for a high-res crop re-run
  pass on already-scheduled mesh frames only (do not add frames). Crop sizing: upscale the bbox crop to
  the model cap (512px) with bicubic; light SR (Real-ESRGAN x2) is OPTIONAL and gated on the ≤60s
  budget — try bicubic first.
- Lane sizing: **Codex** for the scheduler selector + tests; **Sonnet GPU** for the re-inference cost
  measurement. Acceptance: far-player jitter RMS within **1.5×** near players
  (`visual_quality.players[pid].world_jitter_mm_per_frame2` far-vs-near ratio; today far worst 41–78
  mm/f²); **wall-time increase ≤ 60s/clip** (measure honestly on fleet). Kill: >60s/clip or no jitter
  gain → drop SR, keep bicubic-only or shelve.

**STEP D — P2-5 IDF1 wiring (cheap TRK-gate win; Codex).**
- Objective: replace the 3 hardcoded `idf1=None`. Sites: `scripts/racketsport/process_video.py:954,
  968, 1041` calling `derive_track_trust_band(*, idf1, evidence_path)` (`trust_band.py:247`).
- Scorer EXISTS: `threed/racketsport/person_track_gt_scoring.py::score_tracks_against_person_ground_truth`
  (line 275) → returns dict with `"idf1"` (line 359); `DEFAULT_IDF1_THRESHOLD = 0.85` (line 188).
  Wire: when person GT is present for the clip, compute IDF1 via the scorer and pass the float; else
  keep `None` (honest absence — do NOT fabricate). Generalize spectator/court-membership prior
  (`off_court_apron_margin_m`, `outside_gt`) so the img1605 membership-exclusion case becomes a general
  court-membership prior.
- Acceptance (TRK gate): IDF1 ≥ 0.85 all eval clips, 0 switches, 0 spectator-FP, coverage ≥ 0.95
  (`person_track_gt_scoring.py:467` promotion policy). Note: this only wires the number into trust
  bands; it does NOT change tracking. VERIFIED requires real GT rows.

**STEP E — P2-7 Challenger benchmark (the ruling re-opener; benchmark-ONLY, NO integration).**
- Run AFTER P2-1/2/3 land. Models: **GVHMR + PromptHMR-Vid** (core), + **Human3R** (code live, §5),
  + **DuoMo** (code NOW live, §5 — material change), optionally WHAM/JOSH3R. Clips: 4 eval + owner
  handheld. Metrics: `visual_quality.py` keys + P2-6 GT world-MPJPE + placement gates.
- **Pre-registered decision rule (verbatim — do NOT edit):** "challenger must beat the hardened
  SAM-3D stack on ≥3 of 4: world-MPJPE, jitter, foot-slide, far-player accuracy — by ≥20% each — AND
  integrate multi-person + mesh-skeleton coherence. Otherwise the ruling stands and we bank the
  benchmark as evidence." Kill: benchmark-only lane; NO pipeline integration before the rule passes.
- Sizing: **Sonnet GPU** (4 players = 4 crop-runs for single-person challengers — real cost, log it).

### 3. Decision trees (if X → do exactly Y)
- **Latent decode round-trip > 1mm vs persisted joints** → checkpoint/config mismatch. Tolerance:
  the numerical floor from euler↔cont conversion + float32 skinning is ~0.1–0.5mm; treat ≤0.5mm as PASS.
  Only >1mm indicates a real mismatch — verify the wrapper loads the SAME MHR checkpoint the dispatch
  loads (grep dispatch for the checkpoint path). If 1–3mm with a confirmed-identical checkpoint, first
  rule out euler↔cont non-idempotency (§2-A) before declaring mismatch; if still off → typed STOP:
  needs-validation (do NOT proceed to smoothing on an unfaithful decoder).
- **Latent smoothing moves wrist swing-peak >1 frame** → switch to hybrid (torso/legs latent, wrist
  classical). If hybrid still fails the 5mm mesh-skeleton divergence → typed STOP: advice (the coherent
  decode is fighting the wrist protection — owner/architect call).
- **Slide-max still FAILs after the w3_phasefix fresh-GPU promotion run passes (STEP 0)** → this is a FRESH-GPU-only measurement
  (`max_foot_lock_slide_m` computed GPU-side on non-persisted samples — offline replays CANNOT clear
  it; [ROOTJUMP VERIFY RULED] lesson). Dispatch a Sonnet GPU fresh-worlds run to measure. If still
  FAIL with confident per-foot phases → the slide-statistic question (max vs p99+outlier-cap) is a
  typed STOP: needs-decision (owner) — NEVER silently re-tune the 0.03 threshold or the review floor.
- **Camera-motion AUTO regresses a static clip** → keep default-OFF, ship probe as advisory. Do not
  force-enable.
- **Far-player crop re-run > 60s/clip** → drop SR (bicubic only); if still over → shelve P2-3, it is
  not on the M-milestone critical path.
- **A challenger appears to win** → re-read the decision rule verbatim; if it does not meet ALL of
  (≥3/4 by ≥20% + multi-person + mesh-skeleton coherence) → bank as evidence, ruling STANDS. If it
  DOES → typed STOP: decision (owner ratifies a backbone swap; this reverses a frozen ruling).
- **fleet remote checkout drift suspected** (any GPU metric surprise) → the BODY dispatch ships DATA
  never `threed/` code; fleet1 was found 16 commits stale ([MAD A/B RULED]). Verify remote md5 of
  `worldhmr.py`/`orchestrator.py` == local BEFORE trusting any GPU number. This is wave-3 #2
  (code-sync/version-stamp; `remote_body_dispatch.py::build_version_stamp` line 534 exists — wire it
  fail-loud). If drift found → re-sync and re-run before ruling.
- **Any new default stage proposed** → it MUST carry a static-clip regression guard in its acceptance
  (the wave-2 lesson that cost the camera-motion default). No guard → reject the lane spec.

### 4. DO-NOT list (kill-listed approaches + traps, each with the one-line reason)
- **DO NOT smooth world joints as the primary raw-noise fix** — mesh/skeleton diverge (the bug class
  we already hit); latent smoothing is the ruled fix.
- **DO NOT edit vendored `third_party/Fast-SAM-3D-Body/.../mhr_head.py` in place** — wrap it in
  `threed/racketsport/mhr_decode.py`; in-place edits break the frozen-backbone contract + resync.
- **DO NOT run whole-frame at higher res for far players** — input hard-capped 384/448/512px; only
  crop re-inference moves the needle.
- **DO NOT retrain/adopt a learned temporal-refiner net** — SMART shipped classical Gaussian+MAD after
  the learned net LOST on held-out by over-smoothing; our classical chain is the evidence-backed choice.
- **DO NOT trust offline replays for `max_foot_lock_slide_m`** — computed GPU-side on non-persisted
  samples; only a fresh GPU dispatch measures the gated max ([ROOTJUMP VERIFY RULED]).
- **DO NOT re-tune foot_slide_gate 0.03 / root-jump floor 10.0 / refine's better/worse predicate** to
  pass a gate — auto-reject; the statistic question is an owner needs-decision STOP.
- **DO NOT integrate any challenger before the P2-7 decision rule passes** — benchmark-only.
- **DO NOT re-diagnose the weak-bilateral-phase root cause** — two diagnoses converged; build on
  `w3_phasefix` (per-foot confidence-bearing phases + weak-bilateral demotion; code landed
  offline, promote via STEP 0's fresh-GPU run).
- **DO NOT assume GVHMR gravity-injection helps tripod clips** — premise CORRECTED (train-dataloader-
  only); retest only extreme-tilt/handheld.
- **DO NOT touch Outdoor/Indoor eval labels** without a pre-registered `heldout_eval_ledger.md` row.
- **DO NOT skip the remote md5/version-stamp check** before trusting a GPU metric (fleet drift trap).

### 5. External bets verified today (2026-07-07) — claim → verdict → source → license → code
- arXiv 2512.21573 retargeting blueprint → **REAL, NO code released** (abstract confirms method:
  frozen SAM-3D-Body + MHR, shape/scale lock, sliding-window MHR-latent smoothing, differentiable soft
  foot-contact; no GitHub/project-page link). Use as DESIGN reference only; window size/objective NOT
  published — the W=9 / λ values in §2-A are OUR starting points to sweep. Source:
  https://arxiv.org/abs/2512.21573 . License: n/a (paper).
- SOMA-X → **REAL + code + PyPI**. `py-soma-x==0.2.0`; supports SMPL/SMPL-X/MHR/Anny interop
  (MHR↔SMPL-X pivot). Source: https://github.com/NVlabs/SOMA-X . License: Apache-2.0. Optional for
  P2-2 (only if an SMPL-X prior is wanted).
- Human3R (ICLR'26) → **REAL + code/models/demos live**. Feed-forward multi-person SMPL-X + scene +
  camera, 15 FPS/8GB, 1-GPU-day (BEDLAM), built on CUT3R. Source: https://fanegg.github.io/Human3R/
  (arXiv 2510.06219). License: check repo before any bench (SMPL-X drags NC). → P2-7 bracket.
- DuoMo (CVPR'26) → **CODE NOW RELEASED — material change vs roadmap "no code → watch".** Two-stage
  camera→world diffusion; −16% EMDB/−30% RICH reported. Source: https://github.com/facebookresearch/DuoMo
  (arXiv 2603.03265). License: check repo (facebookresearch, likely permissive; verify). → add to P2-7.
- RAFT-small weights → **URL CONFIRMED**: https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth
  (torchvision `Raft_Small_Weights.C_T_V2`); sha256 matches staged
  `runs/lanes/w3_labelfactory_20260707/raft_sha256.txt` (01064c6d…9680e27). License: BSD (torchvision).
  Prefetch on a network lane; A/B vs LK+MAD (opportunistic, non-blocking).
- Recency sweep (since 2026-07-05): no code-backed paper materially changes this pillar; DuoMo's code
  drop is the one substantive delta. (Skeptical: SMART/classical stance holds.)

### 6. Acceptance gates + owner dependencies (exact metric keys; what only the owner can provide)
- **Gated keys (copy exactly):** `grounding_metrics.max_foot_lock_slide_m` ≤ 0.03
  (`body_grounding_quality.py::DEFAULT_MAX_FOOT_SLIDE_M`; blocker `foot_slide_gate_failed`);
  `foot_lock_slide_p95_m` (advisory, gated by NOTHING — do not confuse with the max);
  `visual_quality.players[pid].world_jitter_mm_per_frame2.{feet,wrists,root}`;
  `foot_slide_mm_per_frame.{all,stance,non_stance}.{p95,max}`; `root_step_m.{p95,max}`;
  `smoothing_reset_count`; `summary.root_motion_temporal_jump_count` (root-jump, review floor 10.0);
  TRK: `idf1` ≥ 0.85 (`DEFAULT_IDF1_THRESHOLD`), 0 switches, coverage ≥ 0.95; world-MPJPE ≤ 0.05m
  (P2-6 gate). New (P2-2): mesh-skeleton divergence ≤ 5mm p95 (define in the decode wrapper).
- **P2-6 Independent GT — ONLY the owner can provide (typed STOP: labeling until captured).** GT MUST
  be independent of the calibration BODY is scored against.
  - **WHEN to raise it:** surface this labeling STOP **AT SESSION START** (owner turnaround ~10 min to
    multi-hour; it gates STEP E world-MPJPE + every world-position VERIFIED claim). STEPs 0/D/B/A/C do
    NOT depend on it and proceed in parallel. STEP E (P2-7) and any world-MPJPE VERIFIED claim are
    BLOCKED until GT lands. If owner unavailable, run STEP E's other metrics and mark world-MPJPE
    INCOMPLETE (never VERIFIED).
  - **(a) Cheap — surveyed court-landmark protocol (owner capture script):** tape-measure/survey a set
    of court points FIRST (4 corners, both kitchen/non-volley lines at the sidelines, both baselines'
    centers, net-post bases) and record their world XY on paper — this is the independent frame.
    Then on a single tripod clip the owner performs **scripted static poses**: (1) stand with BOTH
    feet together on each surveyed point, hold **3 s**, arms at sides; (2) stand with one foot on the
    point, hold 3 s; (3) T-pose on the near baseline center, 3 s. Call out the point name at each pose
    (audio marker for the frame). Yields ankle world-position GT at known frames → world-MPJPE for
    ankles. Duration ~10 min. Deliverable = the surveyed-points sheet + the clip.
  - **(b) Better — two-iPhone stereo rig (training-time-only; product stays single-cam):** two iPhones
    on tripods ~2–3 m apart both seeing the court. **Sync**: a sharp **audio clap** (or LED flash) at
    session start AND end — align by the clap onset in each audio track (our audio-onset tooling
    exists). **Extrinsics**: wave a **ChArUco board** through the overlap volume for ~20 s at start →
    solve relative extrinsics offline. Then owner does ~5 scripted rally/pose sequences. Triangulate
    joints on ≥ **20 samples** → world-MPJPE with per-player far/near breakdown. Deliverables the owner
    must produce: the two clips + the clap timestamps + the ChArUco capture. Everything downstream is
    a lane.
- **Owner ratification gates:** any backbone swap from a P2-7 challenger win = owner decision STOP; the
  slide-max statistic question (max vs p99+outlier-cap) = owner needs-decision STOP; commits/pushes =
  owner joint-commit rule (currently granted per `.claude/settings.json`, re-confirm per wave).
- **Sequencing note:** P2-5 (IDF1, Codex) and Step B (camera-motion default) are the cheapest wins —
  land first. P2-2 (latent smoothing, Sonnet GPU) is the highest-value but needs the decode wrapper +
  round-trip gate before smoothing. P2-3 far-player and P2-7 bracket follow. P2-2/P2-1 both touch
  `worldhmr.py`/`pose_temporal.py` — file-fence them apart or serialize (they collided in wave-2).

## PILLAR: PADDLE — 6-DOF to the RKT face-angle gate

> Audience: the successor manager. Every step below is pre-decided. Where judgment could creep in,
> the branch ends in "typed STOP: <bucket>". VERIFIED=0 for RKT and stays 0 until owner 4-marker GT
> passes (P3-7). Everything this pillar ships renders in the ESTIMATED/preview band, never truth.
> All paths absolute-from-repo-root `/Users/arnavchokshi/Desktop/pickleball/`.

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
Fable + owner ruled (2026-07-05/07) that the paddle is a **rigidly-gripped rectangle solved off the
wrist**: `X_paddle(t) = W_hand(t) ∘ G`, one per-segment constant grip transform `G`, sparse strong
evidence locking `G`, wrist carrying full 6-DOF through evidence-free frames. Phase-1 fused estimator
(`paddle_pose_fused.py`, render-only) LANDED needing **no new model and no GPU**: palm frames from the
discarded MHR70 finger joints + per-segment `G` + wrist-gated YOLO26s detector-box correction + a
soft ball-reflection cone (built, dormant) + SO(3)/one-euro smoothing. The forward plan was SEQUENCED
by tech-audit: **P3-1 wire it into the default pipeline (fail-closed, ESTIMATED band) → P3-3 WiLoR
hand crops (fixes the MEASURED palm rest-pose defect, palm-only IoU 0.065) → P3-5 activate the
ball-reflection factor the instant P1-4 3D ball velocities exist**. Everything richer (P3-2 seg→IPPE
PnP, P3-4 5-keypoint detector, P3-4b face-texture homography, P3-6 nvdiffrast) is
DEFERRED-PENDING-GT-GAP: do NOT build until owner marker GT shows a residual gap, EXCEPT the one free
probe — eval RacketVision's public MIT RTMPose-M checkpoint at zero training cost. Kill-listed for
good: FoundationPose-class zero-shot (HANDAL 0.04-0.26 AP), HOISDF/MOHO (no license), HOLD (25 GPU-hr/
video). The constant-grip-transform assumption STAYS; per-segment grip re-fit is measured inside the
WiLoR lane, not re-architected.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
Evidence root: `runs/lanes/racket_6dof_20260705/` (STATUS.md = trail;
`i1_fused_estimator/acceptance_record_v2.json` = final numbers). RACKET_6DOF_GOAL.md §Log is canonical.
- Fused estimator SHIPPED, render-only, RKT board = SCAFFOLD, VERIFIED=0. Source string
  `wrist_palm_grip_fused` (`paddle_pose_fused.py:82 SOURCE`); artifact contract =
  `racket_pose_estimate.json` (`ARTIFACT_TYPE="racketsport_racket_pose_estimate"`), consumed by an
  UNMODIFIED `virtual_world.py`.
- Internal-val vs CVAT paddle rectangles (scoring-only, R1 scorer functions verbatim): **Wolverine**
  IoU 0.2356, center-error median 23.4px, rotation-jitter p95 max 27.9°/f; **Burlington** IoU 0.3424
  (12.7× the 0.0269 wrist-proxy baseline), center-error median 13.4px. Face-normal jitter median
  ~5°/f (baseline 23.4-53.0). Coverage 100% on scored clips.
- Bands: 100% `palm_fitted` everywhere. `contact_locked`=0 (no usable 3D-velocity contacts exist yet);
  reflection channel true-zero → warning `reflection_channel_dormant_no_usable_ball_contacts`
  (`paddle_pose_fused.py:1776`).
- MEASURED per-factor truth (the load-bearing facts that set the sequence): detector-box evidence =
  **+0.19 IoU** (the whole game); palm channel = smoothness but weak absolute orientation, palm-only
  IoU **0.065** (SAM3D fingers rest-pose-dominated; MHR70 = the 70 MANO-hand-region latent joints of
  SAM-3D-Body, so "MHR palm frame" and "SAM3D fingers" name the SAME source everywhere below) — this
  is the defect P3-3 targets; reflection
  contributed nothing (no contacts).
- Teleport census: 29 → 0 undeclared one-frame jumps >0.35m across 4 clips; 5 remain, all at declared
  hand-switch boundaries. Upstream skeleton position noise (raw wrist Δpos med 0.020-0.076 m/f @30fps)
  is now the binding constraint on perceived paddle stability — NOT a paddle-solver bug.
- Detector assets: `runs/rkt_train_20260702T072800Z/det_yolo26_external_split/weights/best.pt` (box),
  `.../seg_yolo_external_split/weights/best.pt` (seg, IDLE). YOLO26s box predictions currently exist
  ONLY for the 3 CVAT clips. **TRAP:** `runs/` is gitignored → these weights are LOCAL untracked and
  will NOT survive a fresh clone. Owner-clip pooled AP50 0.27 (Outdoor 0.50 / Burlington 0.13) —
  weak-evidence grade only.

### 2. The exact build plan
Lane-sizing rule: mechanical file work with a written spec = SONNET, checkpointed (Codex quota is
exhausted until Jul 9 per STATUS.md; do not assume Codex). GPU inference (WiLoR, detector) = provision
via `gpu-fleet-provision` skill, reconcile `runs/manager/gpu_fleet.md` first. Use the `run-lane` skill
to template every dispatch. NEVER run subagents on Fable; pin an explicit `model`.

**STEP P3-1 — Wire the fused estimator into the default pipeline (DO FIRST).**
- Objective: a fresh E2E on ANY clip emits the fused paddle track by default, fail-closed when
  evidence is absent, ESTIMATED band, suite green. Today it is BUILT-NOT-WIRED — the CLI
  `scripts/racketsport/build_paddle_pose_fused.py` is manual-only; `process_video.py` only *reads* a
  pre-existing `racket_pose_estimate.json` at `_stage_world` (line 2962/2976) and `_stage_confidence_gate`
  (line 3033). Nothing BUILDS it in-pipeline.
- File targets:
  1. `scripts/racketsport/process_video.py` — add a new stage method (mirror the `_stage_*` pattern,
     e.g. `_stage_paddle_pose`). Insert `("paddle_pose", self._stage_paddle_pose)` into the suffix
     stage list (`_build_suffix_stage_fns`, ~line 513) immediately AFTER `("grounding_refine", ...)` —
     the `skeleton3d.json` producer — and after the ball_arc/physics stages, and BEFORE `("world", ...)`
     (world reads paddle output at line 2962). Verify `skeleton3d.json` exists at stage entry; absent →
     blocked/degraded (fail-closed). Call `build_paddle_pose_fused_from_file` (import from
     `threed.racketsport.paddle_pose_fused`; signature verified at `paddle_pose_fused.py:1351`). Pass the
     skeleton path POSITIONALLY as the first arg — it is `skeleton3d_path`, NOT a keyword `skeleton=`:
     `build_paddle_pose_fused_from_file(str(self.clip_dir/"skeleton3d.json"), clip_id=<run id>, ...)`.
     All of `physics_estimate=`, `detector_boxes=`, `calibration=`, `membership=` are typed `Mapping`
     (LOADED DICTS), NOT file paths — `json.load` each file into a dict first (or pass `None` if absent),
     mirroring `scripts/racketsport/build_paddle_pose_fused.py`'s `_read_optional_json`. Sources:
     `physics_estimate` ← `racket_physics_estimate.json`, `detector_boxes` ← per-clip YOLO26s boxes,
     `calibration` ← `court_calibration.json`, `membership` ← `membership.json` (the
     `player_court_membership.py`/`virtual_world.py:604` artifact). Set `use_reflection=True` (default);
     `use_detector_boxes=True` ONLY when both detector_boxes+calibration dicts loaded. Write via
     `write_paddle_pose_fused(self.clip_dir/"racket_pose_estimate.json", payload)`.
  2. Per-clip YOLO26s paddle-box inference producer (SONNET-authored script; Ultralytics inference =
     MPS locally OR `cuda:0` on a VM — NOT a mandatory GPU lane. Predictions exist only for the 3 CVAT
     clips today). Add `scripts/racketsport/build_paddle_detector_boxes.py` that loads
     `runs/rkt_train_20260702T072800Z/det_yolo26_external_split/weights/best.pt` via Ultralytics `YOLO`
     and predicts on the clip's UNDISTORTED per-frame images (the frames the solver reprojects into),
     single class `0=paddle` (`runs/rkt_prep_20260702T000000Z/det/data.yaml`, nc:1), `imgsz=1280`
     (train config), `conf=0.25` fixed, `device=mps`|`cuda:0`. Write a detector_boxes JSON in the schema
     `_detector_box_records` accepts (`paddle_pose_fused.py:771`): top-level key one of
     `records|detections|frames|boxes` → list of `{"frame_idx": int, "bbox_xyxy":[x1,y1,x2,y2],
     "conf": float}` (keep highest-conf box per frame). Example:
     `{"records":[{"frame_idx":42,"bbox_xyxy":[812.0,430.5,905.0,540.0],"conf":0.71}, ...]}`. Wrist-gate is internal to the solver
     (`detector_box_wrist_gate_radius_px`, default 130px). **Because the weights are gitignored+local**,
     gate the box factor on file existence; absence → fall through to palm-only (still fail-closed, still
     emits a track). Do NOT hard-require boxes.
  3. Trust-band wording patch (the ONE authorized `virtual_world.py` edit, I1c ruling (c)):
     `_paddle_estimate_trust_band` / `_paddle_frame_trust_band` (`virtual_world.py:1548/1575`).
     Today source `wrist_palm_grip_fused` fails `_is_wrist_proxy_source` (`:1544`, matches only
     `wrist_proxy`), so `source_kind` collapses to `"racket_pose_estimate"` (`:1484`) and the paddle
     renders a generic PHYS-RACKET `physics_derived` band — the QA "all frames proxy fallback" misread
     root cause. Patch: recognize `wrist_palm_grip_fused` as its own `source_kind` with a distinct,
     honest reason string ("fused wrist+palm+grip render-only estimate; not true 6-DoF, does not score
     or promote RKT"), badge `low_confidence`, gate_id `wrist_palm_grip_fused_estimated_paddle` (the
     artifact already self-declares this at `paddle_pose_fused.py` trust_band). ONE stanza, no
     eval-guard override, no Outdoor label touch.
- Suite coverage: extend `tests/racketsport/test_process_video.py` (new stage present + fail-closed on
  missing skeleton); put the viewer-band assertion in `tests/racketsport/test_trust_band.py` (the paddle
  trust-band suite, NOT test_process_video): assert `source_kind == "wrist_palm_grip_fused"`, badge
  `low_confidence`, gate_id `wrist_palm_grip_fused_estimated_paddle`. Run only the racketsport suite locally (`pytest
  tests/racketsport -q`); do NOT run the full repo or GPU jobs.
- Acceptance (EXACT): fresh E2E on Wolverine emits `racket_pose_estimate.json` with
  `source=="wrist_palm_grip_fused"`, `render_only==True`, `not_for_detection_metrics==True`,
  `trusted_for_rkt_promotion==False`, `rkt_gate_unscoreable==True`; on a clip with NO skeleton the
  stage returns blocked/degraded and world has zero paddles (fail-closed); viewer band on the paddle
  reads the fused wording, NOT `wrist_proxy`; racketsport suite green (allow the 1 known pre-existing
  `monitor_process_resources` fail). Non-regression floor: Wolverine internal-val IoU ≥ 0.235,
  Burlington ≥ 0.34 — re-score (scoring-only, no held-out clip) with the canonical fused scorer
  `runs/lanes/racket_6dof_20260705/i1_fused_estimator/score_fused.py` (it reuses
  `r1_evidence/measure_r1_evidence.py`'s IoU/center-error/jitter method via importlib; emits
  IoU/center-error/jitter/coverage). CVAT paddle-rectangle GT =
  `runs/cvat_imports/2026_06_30/<clip>/reviewed_boxes.json` (Burlington + Wolverine only).
- Kill criteria: if wiring drops Wolverine IoU below 0.235 or introduces any undeclared teleport
  (>0.35m non-switch), revert to reading a pre-built artifact and STOP → typed STOP: advice.

**STEP P3-3 — WiLoR hand crops (DO SECOND, right after P3-1).**
- Objective: replace the rest-pose-biased MHR palm frame (palm-only IoU 0.065) with WiLoR observed
  pronation on wrist crops; downstream fused IoU +≥0.03, jitter not worse.
- Crop source: wrist pixel from `skeleton3d.json` reprojected to image; crop a fixed SQUARE box,
  half-size ≈ 1.5× the wrist→elbow pixel length (a whole hand fits), floor 120px, clamped to image
  bounds — this is a CROP size, NOT the solver's 130px `detector_box_wrist_gate_radius_px` (that is a
  box-to-wrist MATCHING gate, a different quantity; do not reuse its value as the crop). WiLoR
  (`github.com/rolpotamias/WiLoR`) emits a MANO hand frame → derive palm normal.
- Blend rule (PRE-RULED, no judgment): use WiLoR palm frame where WiLoR crop confidence ≥ threshold
  (start 0.5, single value, tune once); else fall back to the existing MHR palm frame. This is a
  per-frame source swap feeding the SAME solver channel — no new solver architecture. The
  constant-grip-transform assumption STAYS; additionally MEASURE per-segment grip re-fit (compare `G`
  fit with WiLoR palm vs MHR palm per segment) and report drift — do not change the assumption unless
  the owner rules on the measured drift (typed STOP: decision if drift is large).
- File targets (P3-3): the palm/pronation frame is derived in `_build_hand_frames`
  (`paddle_pose_fused.py:198`; `rotation` columns = hand X, Y=grip axis, Z=palm normal). Add a new kwarg
  `wilor_palm_frames: Mapping | None = None` to `build_paddle_pose_fused_from_skeleton` and thread it
  into `_build_hand_frames`; where a confident WiLoR frame exists for that (player, frame), substitute
  its palm-normal as the Z-candidate BEFORE orthonormalization (`_orthonormal_frame_from_y_z`, :299),
  else keep the MHR candidate. `process_video.py` loads the WiLoR artifact dict and passes it via that
  kwarg. The WiLoR crop pass is a SEPARATE producer artifact (one batched A100 pass → JSON), NOT inline.
- Lane sizing: GPU (A100) — WiLoR is CUDA, no MPS. Provision via `gpu-fleet-provision`. Batch wrist
  crops across all frames/players per clip in one A100 pass (detector 138-175fps, recon fast) — do NOT
  run per-frame subprocess.
- License inventory line (REQUIRED before pulling weights): note WiLoR's license as a one-line "what we
  used" inventory item in the P3-3 lane report, per NORTH_STAR Part IV rule 6 / line 638 ("record
  license VERBATIM in the lane report"). Do NOT edit `EDGE_PLAYBOOK.md` — its license column was removed
  by owner ruling (`EDGE_PLAYBOOK.md:371`) and is not where license lines go. Verbatim: "WiLoR
  CC-BY-NC-ND, internal R&D use only per owner 2026-07-04 ruling; also inherits Ultralytics (AGPL) +
  MANO licenses; NEVER ship weights in the product, derived pose only." No commercial launch without
  counsel (mirror the US11615540B2 watch-item discipline).
- Acceptance (EXACT): absolute pronation improves vs the CVAT-box orientation proxy; **downstream
  fused Wolverine IoU +≥0.03** over the P3-1 number; jitter median ≤ prior (≤ ~5°/f) and p95 not
  worse; coverage ≥99% of P3-1. Report is scoring-only on internal clips.
- Kill criteria: if fused IoU gain <0.03 after one tuning round OR jitter regresses → keep WiLoR OFF
  by default, retain MHR palm; book the negative result; STOP → typed STOP: advice.

**STEP P3-5 — Ball-reflection factor activation (FAST-TRACK, unlocks with P1-4).**
- Objective: at contact frames, fuse the ball-reflection cone as the impact-window orientation prior
  (strongest evidence, exactly the frames coaches care about). Code is BUILT + unit-tested + DORMANT.
- Exact dormant path (grep-verified): `paddle_pose_fused.py` — `_reflection_records_for_player`
  (:706), `_solve_grip_transform` reflection pairs (:677-695), `use_reflection` default True (:1383),
  `any_reflection_available` gate (:1465), warning at :1776. Constants:
  `DEFAULT_REFLECTION_WEIGHT_SCALE=8.0` (:108), `DEFAULT_REFLECTION_MAX_TIME_GAP_S=0.12` (:109).
- Input contract from P1-4 (the unlock): the reflection channel reads a `racket_physics_estimate.json`-
  shaped payload via the `physics_estimate=` arg; it is dormant because `physics_estimate["estimates"]`
  is empty — no 3D ball velocities exist. P1-4 must produce a real 3D ball arc
  (`ball_track_arc_solved.json`, already a pipeline artifact — see `process_video.py:2961`) with
  in/out velocities at contacts; `racket_physics_estimate.py` (`build`, consumes ball impulse + wrist
  motion) then populates `estimates`. **Do NOT hand-fabricate velocities.** First real test bed =
  IMG_1605 (30 confirmed audio onsets, no detector ball track yet → needs P2c GPU ball track).
- Impact-window fusion weights: keep `reflection_weight_scale=8.0` and `max_time_gap_s=0.12` at
  defaults for the first activation (inverse-uncertainty weighted, soft cone — spin/friction tolerant,
  no hard lock). Frames within `contact_lock_window_s` of a contact where reflection agrees → band
  promotes to `contact_locked` (currently 0). Reference band at impact = the TT4D table-tennis research
  existence proof, 26.4°±4.4 orientation error from trajectory inversion alone (NORTH_STAR lines
  178/417) — expectation-setting ONLY, NOT a pass/fail gate; the gate is impact-frame face-angle error
  ≤30° p90 vs owner 4-marker GT.
- Gate protocol vs marker GT (EXACT): impact-frame face-angle error vs owner 4-marker GT (P3-7)
  ≤ 30° p90 on the first pass, tightening with data. Until marker GT exists, the ONLY honest internal
  check is: reflection ON vs OFF changes `band_distribution[contact_locked]` from 0 to >0 on IMG_1605,
  and held-out-contact residual (fit without half the contacts → reflection residual on the held-out
  half) is bounded. Report `evidence_channels.reflection_contacts_available==True`.
- Lane sizing: solver activation = SONNET (config + validation); the ball-track prerequisite is the
  GPU cost (shared with P1). Kill criteria: reflection worsens impact-frame agreement vs GT once GT
  exists → drop weight_scale toward 0, keep as diagnostic only; STOP → typed STOP: decision.

**STEP P3-4-PROBE — RacketVision RTMPose-M zero-cost checkpoint eval (the ONLY deferred-item probe allowed now).**
- Objective: measure, for free, whether the public MIT 5-keypoint racket head beats our current
  orientation before committing to any P3-4 training. This does NOT build the detector; it is a probe.
- Protocol (EXACT): clone `github.com/OrcustD/RacketVision` (MIT). `cd source && python
  download_checkpoints.py` → RTMDet-M `epoch_300.pth` + RTMPose-M `best_PCK_epoch_90.pth` land under
  `source/RacketPose/checkpoints/`. ENV: RTMDet/RTMPose need a version-pinned mmpose/mmdet/mmcv stack —
  build in an isolated venv per the repo's pinned requirements (or its container if provided); target
  device MPS/CPU locally. FAILURE BRANCH: if the mm* stack cannot be built locally within the lane, run
  the probe on a provisioned CPU/GPU VM (`gpu-fleet-provision`) and TIME-BOX to 1 lane — do not
  rabbit-hole on the env. Run inference on Burlington/Wolverine frames ONLY (scoring-only
  clips; NEVER Outdoor/Indoor). Map the 5 keypoints (Top/Bottom/Handle/Left/Right) to our paddle
  rectangle with this FIXED rule: long axis = unit(Top − Bottom) with Handle disambiguating the handle
  end (Handle is nearest Bottom); short axis = unit(Right − Left); head center = midpoint(Top,Bottom);
  the 4 oriented-rectangle corners are the {Top,Bottom}×{Left,Right} extents about that center;
  face-normal = unit(long_axis × short_axis), sign continuity-locked frame to frame. Measure face-normal
  error vs the CVAT-derived orientation proxy and center error in px. Expected ceiling (their reported numbers, NOT ours): PCK@0.2 81.8-89.6 overall, handle
  92.6-97.9, but **side/edge (face-orientation) keypoints only 64.8-80.1** — the same weak axis we
  measured, so temper expectations. Report as a decision input only; do NOT wire into fusion yet.
- Decision output: if the RTMPose keypoints give face-normal error materially better than our current
  palm+box fusion on sharp frames → P3-4 (fine-tune on owner data) becomes worth a lane AFTER marker
  GT. If not → P3-4 stays deferred. Either way, write the number; typed STOP: decision to the owner
  with the measured comparison.

### 3. Decision trees
- P3-1 wiring drops IoU <0.235 OR adds an undeclared teleport → revert to read-only artifact
  consumption, keep the manual CLI path, book the regression → typed STOP: advice.
- YOLO26s box weights absent on the run machine (gitignored, fresh clone) → box factor auto-disables
  on file-existence check; pipeline emits palm-only fused track (still fail-closed). If owner clips
  need boxes at scale → re-stage weights from the rkt train lane or promote to `scripts/`/`models/`;
  typed STOP: decision (where to persist the weights).
- Skeleton3d.json missing/empty for a clip → paddle stage returns blocked; world has zero paddles;
  viewer shows no paddle (correct fail-closed). No STOP — this is the designed degrade.
- WiLoR fused IoU gain <0.03 after one tune → WiLoR OFF by default, MHR palm retained, negative result
  booked → typed STOP: advice.
- WiLoR per-segment grip re-fit shows large `G` drift (grip not actually constant) → do NOT silently
  change the constant-grip assumption (standing ruling) → typed STOP: decision (owner rules on
  re-fit cadence).
- P3-5: P1-4 has NOT landed real 3D ball velocities → reflection STAYS dormant; do not fake it; the
  warning `reflection_channel_dormant_no_usable_ball_contacts` is CORRECT, not a bug. Ship P3-1/P3-3
  without it.
- P3-5: IMG_1605 ball track needs GPU (30 audio onsets, no detector track) → provision via
  `gpu-fleet-provision`; this is shared with P1's ball work — coordinate, do not double-provision.
- Sequencing tie-break (P3-3 vs P3-5): P3-3 is DO-SECOND, P3-5 is FAST-TRACK on the P1-4 unlock. If
  P1-4 lands (or a GPU frees) while P3-3 is mid-flight, do NOT pause P3-3 — finish P3-1's
  non-regression re-score and P3-3's tuning round; P3-5 activation runs as a SEPARATE parallel lane
  (it is SONNET config + a shared ball GPU cost) and never blocks P3-3.
- Any request to score against Outdoor/Indoor CVAT labels to "prove" the paddle → FORBIDDEN without a
  pre-registered `runs/manager/heldout_eval_ledger.md` row (use `threed/racketsport/append_lock.py`);
  RKT is not gate-ready anyway (no marker GT) → typed STOP: needs-validation.
- Anyone proposes promoting the paddle to canonical `racket_pose.json` or scoring the RKT gate off the
  fused estimate → HARD BLOCK: `never_canonical_racket_pose==True`, `rkt_gate_unscoreable==True`,
  box-to-6DoF promotion is MASTER_PLAN-killed. Refuse → typed STOP: decision.
- Marker GT (P3-7) not captured → RKT stays SCAFFOLD/VERIFIED=0 forever; no amount of internal IoU
  promotes it. W5 = NORTH_STAR PART VI wave 5 (W5-D P3-7 paddle GT + hi-def asset, line 1712) — no date
  is booked. Scheduling mechanism: raise a typed STOP: decision (owner capture date) requesting the
  owner set the marker session.

### 4. DO-NOT list
- DO NOT build P3-2 (seg→IPPE PnP), P3-4 (5-kp detector training), P3-4b (face-texture homography),
  or P3-6 (nvdiffrast) before owner marker GT shows a residual gap. Reason: DEFERRED-PENDING-GT-GAP
  ruling — masks fail exactly at impact (blur), keypoint pixel-math only favorable <6-8m on sharp
  frames; no evidence yet that current fusion needs them.
- DO NOT adopt FoundationPose / GigaPose / FoundPose / any BOP model-free zero-shot pose. Reason:
  HANDAL benchmark 0.04-0.26 AP on handheld tools ≈ paddles; NVIDIA restrictive license. (Kill-listed.)
- DO NOT use HOISDF / MOHO (no license files = hard block, YCB domain mismatch) or HOLD (25 GPU-hr/
  video). Reason: cost/license, ruled.
- DO NOT re-derive paddle pose from detector boxes alone (rectangle→6DoF promotion). Reason: IPPE-
  ambiguous, MASTER_PLAN-killed (proof: `runs/rkt_paddle_lane_20260704T204142Z_wolverine/`, 544/544
  ambiguous). Boxes are a wrist-gated *correction* factor only.
- DO NOT tighten the phase-1 position-smoothness bars below the upstream skeleton noise floor
  (raw wrist Δpos med 0.020-0.076 m/f @30fps). Reason: MEASURED unreachable; paddle stability is now a
  skeleton-smoothing problem, not a paddle-solver one.
- DO NOT ship WiLoR weights in the product. Reason: CC-BY-NC-ND + AGPL Ultralytics; derived pose only,
  internal R&D use per owner ruling; counsel review before any commercial launch.
- DO NOT touch or read Outdoor/Indoor CVAT labels for any paddle scoring. Reason: strict held-out;
  eval_guard + ledger discipline. CVAT paddle rectangles (Burlington 13 / Wolverine 14 / Outdoor 17)
  are scoring-only forever, never solver input on the clip being scored.

### 5. External bets verified today (2026-07-07)
- WiLoR → REAL. `github.com/rolpotamias/WiLoR` (CVPR'25). License **CC-BY-NC-ND** on model weights;
  repo also inherits Ultralytics + MANO licenses. Detector 138-175 FPS (Proposed-M/Proposed-S; matches
  NORTH_STAR line 414); code + weights released. CUDA (no MPS). Source: repo README + emergentmind/arXiv
  2409.12259. Verdict: usable for P3-3 internal-only, ledger it.
- RacketVision RTMPose-M 5-keypoint checkpoint → REAL + DOWNLOADABLE NOW. `github.com/OrcustD/
  RacketVision`, license **MIT** (verified on repo). `python download_checkpoints.py` fetches RTMDet-M
  `epoch_300.pth` + RTMPose-M `best_PCK_epoch_90.pth` to `source/RacketPose/checkpoints/`; dataset on
  HF `linfeng302/RacketVision`. Face/edge keypoints weak (64.8-80.1 PCK). Verdict: run the zero-cost
  probe (P3-4-PROBE); MIT means it could even ship.
- Image-as-an-IMU (Oxford, H22/P3-8) → REAL, code released. `github.com/jerredchen/image-as-an-imu`
  (arXiv 2503.17358). Inverts motion blur → instantaneous 6-DOF angular velocity from ONE blurred
  image (known focal length + exposure). Verdict: valid P3-8 stretch spike only; time-box 1 lane,
  adopt on measured impact-frame gain. Not on the P3-1/3/5 critical path.
- OnePoseViaGen → REAL, code released. `github.com/GZWSAMA/OnePoseviaGen` (CoRL'25 Oral, arXiv
  2509.07978). 1 reference photo → generative domain randomization → 6D pose training set. Verdict:
  a P3-4 *label-bootstrap* option, DEFERRED with P3-4; do not build now.
- Recency sweep (monocular racket/paddle 6-DOF, since 2026-07-05): NO new competing released code
  found; "we are first" holds (surveys + RacketVision only). No change to the pillar.

### 6. Acceptance gates + owner dependencies
- Internal (scoring-only, Burlington/Wolverine, NOT held-out): metric keys are the R1 scorer outputs —
  paddle-rectangle **IoU**, **center-error px** (median/p95), **face-normal jitter deg/f**
  (median/p95), **coverage %**, and `band_distribution` (`contact_locked|palm_fitted|
  grip_extrapolated`). P3-1 floor: IoU ≥0.235 / ≥0.34, no undeclared teleport. P3-3 target:
  fused IoU +≥0.03, jitter not worse. These are RENDER-QUALITY gates, NOT RKT promotion.
- Artifact honesty keys (must all hold on every emitted `racket_pose_estimate.json`):
  `render_only==True`, `not_for_detection_metrics==True`, `trusted_for_rkt_promotion==False`,
  `never_canonical_racket_pose==True`, `rkt_gate_unscoreable==True`, `source=="wrist_palm_grip_fused"`.
- The RKT VERIFIED gate (the real one): **face-angle / contact-point error vs owner 4-corner-marker
  GT**. P3-5 first-pass gate = impact-frame face-angle error ≤ **30° p90**. NO internal metric ever
  promotes RKT; VERIFIED=0 until this passes on real markers.
- OWNER-ONLY dependencies (schedule with the W5 captures — NORTH_STAR PART VI wave 5, W5-C/W5-D block,
  lines 1709-1713 — the ONLY path to RKT VERIFIED):
  (1) **P3-7 hi-def paddle asset**: photo set of the owner's actual paddle → textured mesh, correct
  dimensions (LOD glTF for the viewer). (2) **4-corner-marker GT session**: same session films clips
  with fiducial markers on the 4 paddle corners → the scoring harness input (marker corners in image →
  reprojected face angle = GT). Harness consumes: marker GT + our fused `racket_pose_estimate.json` →
  per-impact face-angle error. (3) Owner decision on where to persist the gitignored YOLO26s weights
  for production clips. Until (1)+(2) land: paddle ships ESTIMATED/preview only, RKT = SCAFFOLD.

## PILLAR: COURT + NET — profiles-first calibration, distortion fix, auto-find epic, net geometry

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
We DECIDED the v1 court path is **profiles-first, not auto-find**. For the owner's ≤3 courts we store one
frozen calibration + line-paint color + per-lens intrinsics/distortion in the already-landed
`CourtProfile` (P0-9, `threed/racketsport/profile_registry.py`), re-identify the court on upload by
`camera_fingerprint` + color + a cheap 4-line reprojection check, and REUSE the frozen calibration —
no re-solving. We RULED the single highest-leverage accuracy fix is **distortion**: metric15 already
estimates gated k1/k2, but the ARKit back-projection path silently drops `intrinsics.dist` (verified
below) — fixing that + a one-afternoon ChArUco k1/k2 capture is worth more than any detector work.
Auto-find (neural + geometric multi-hypothesis solving, Wave A) is an **UNKNOWN-COURT EPIC behind a
preview flag**, NOT in DONE-v1; it resumes only when a real non-profile court upload happens. Net v1
is a **~1-line data change** (tape-measured heights from the profile override regulation defaults at
`build_net_plane` call sites); the seg+catenary 3D net is v2, staged with its own gates. When ARKit
lands (P0-10), PoseGravity (BSD-3, closed-form pose from points+lines+gravity) and AnyCalib/GeoCalib
(Apache-2.0 intrinsic sanity) become the priority integrations.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- **Two SEPARATE error budgets — do not conflate:**
  - Owner/metric15 v1 path: **12.3px p95 / 4.8px median** reproj; ITS floor is IMG_1605-class
    edge-of-frame distortion (x̄≈53px at frame edges) — source: NORTH_STAR P4-4 (line 1164-1173).
  - Auto-find/GEO path: round-2 geometric solver — Outdoor **4.4px median (7.1px p95) PASS ≤20px**;
    Burlington 366.5/1211.3, Wolverine 318.8/812.8 (adjacent-court lock-on, temporal gate correctly
    FAILS them), Indoor 93.3/340.1. The oft-cited **19.8px p95 is THIS path's discrete lock-on bug**,
    v1-irrelevant. Evidence: `runs/lanes/court_autofind_20260705/handoff/cal_geo_r2_report.md`.
- **P4-0 registry: SCHEMA+STORAGE DONE 2026-07-06** (`p09_registry_20260706`). `CourtProfile` +
  `lookup_court_profile()` in `profile_registry.py`; `camera_fingerprint()` in
  `owner_capture_intake.py:211` — all exist. ⚠️ NORTH_STAR P4-0's "0% built / no
  CourtProfile module exists" note is STALE — reconcile: what is unbuilt is the MATCHING ALGORITHM
  (embedding + color histogram), the 4-line verification, trust-banding, and pipeline consumption.
- **Distortion leak CONFIRMED (the P4-4 audit):** `court_positioning.back_project_pixel_to_floor`
  (`threed/racketsport/court_positioning.py:87-108`) uses ONLY fx/fy/cx/cy — never reads
  `intrinsics.dist`; its docstring says it expects an *already-undistorted* pixel. The ARKit metric
  caller `court_calibration.py:443-446` back-projects RAW keypoint uv with NO prior undistort → dist
  dropped. The homography path (`placement.py:253` `undistort_applied`, `:1711` `cv2.undistortPoints`)
  DOES undistort. metric15 (`court_calibration_metric15.py:303 fit_single_view_metric_camera`,
  `:429 _try_refine_with_distortion`) DOES gated k1/k2 via `cv2.calibrateCamera` but defaults to
  `distortion_model="zero_distortion_grid_search_focal"` (k1=k2=0) on clean views.
- **Net:** `net_plane.py:10 build_net_plane(sport)` + `:33 net_top_height_m_at_x` already do linear
  post→center sag (pickleball 36in post / 34in center). Heights are REGULATION DEFAULTS from
  `court_templates.py:159-167`; the profile fields `net_post_height_in`/`net_center_height_in` exist
  but are NOT threaded into `build_net_plane`. VERIFIED=0 everywhere (no product gate passed on labels).

### 2. The exact build plan (numbered; objective · files · recipe · lane · acceptance keys · kill)
**STEP 1 — P4-4a: plug the ARKit distortion leak (do FIRST; smallest, highest-leverage).**
- Objective: apply `intrinsics.dist` in the ARKit metric back-projection path.
- Files: `threed/racketsport/court_calibration.py` (the `world_keypoints` comprehension at :443-446);
  do NOT change `back_project_pixel_to_floor`'s contract (its docstring correctly promises undistorted
  input). Undistort AT THE CALLER, mirroring `placement.py:1704-1711`.
- Recipe: before back-projecting, if `sidecar.intrinsics.dist` is nonzero, map each keypoint uv through
  `cv2.undistortPoints(pts, K, dist, P=K)` (K from `camera_matrix_from_intrinsics`) to get
  undistorted pixels, then back-project those. Guard with a `_dist_nonzero` check (reuse placement's).
- Lane: **Codex**, single file + test. CPU only.
- Acceptance: add a unit test proving a synthetic k1<0 barrel case moves a frame-edge keypoint's
  back-projected floor point toward truth; existing court_calibration tests stay green. Emit an
  `undistort_applied` flag into the artifact for parity with placement.
- Kill: if undistorting REGRESSES the 12.3px metric15 p95 → the sidecar dist is wrong/garbage; revert,
  and gate the undistort behind `distortion_model != "zero_distortion..."`. Measure with
  `python scripts/racketsport/validate_metric_calibration_15pt.py --run-dir <out>` (fits all 4 eval
  clips, writes `reprojection_p95_px` per clip); compare pre/post-undistort. NOTE real ARKit sidecars
  are NOT yet captured (P0-10 records the sidecar but is not proven on a real device) — so this
  real-clip kill check is DEFERRED; until P0-10, validate Step 1 on SYNTHETIC sidecars only (Step 3).

**STEP 1.5 — BRIDGE: get ChArUco dist onto the sidecar the metric path reads (wires Step 2→Step 1).**
- WHY (load-bearing): Step 1 reads `sidecar.intrinsics.dist`, but Step 2 writes k1/k2 into
  `DeviceProfile.intrinsics_by_lens_zoom` — a DIFFERENT object. ARKit intrinsics arrive rectified
  (dist=0), so WITHOUT this bridge `metric_calibration_from_sidecar_and_keypoints`
  (court_calibration.py:399) sees dist=[0,0,0,0] and Step 1 is a permanent no-op. Distortion lives on
  `DeviceProfile` (keyed device_key/lens/zoom); the v1 match path (4a) keys on
  `CourtProfile.camera_fingerprint` — the two MUST be linked at intake.
- OWNER of the injection: the owner-capture INTAKE path (`owner_capture_intake.py`, alongside
  `camera_fingerprint()` at :211). At intake: resolve the `DeviceProfile` by device_key (from the
  sidecar/metadata device field), select the `LensZoomIntrinsics` whose (lens,zoom) matches the
  sidecar's lens/zoom fields, and INJECT its `dist` into `sidecar.intrinsics.dist` BEFORE the metric
  calibration runs. FALLBACK: no matching DeviceProfile / lens / zoom entry → leave dist=0 → Step 1
  inert BY DESIGN (regulation-clean, no regression).
- Lane: **Codex**, small (intake wiring + a test that a matched DeviceProfile populates sidecar dist).
- Acceptance: an intake carrying a DeviceProfile ChArUco entry for the sidecar's lens yields a sidecar
  whose `intrinsics.dist` is nonzero and equals the profile entry; unmatched lens → dist stays zero.

**STEP 2 — P4-4b: ChArUco k1/k2 owner capture protocol (owner-dependent; runs in parallel).**
- Objective: get a clean, independent k1/k2 per lens/zoom into `DeviceProfile.intrinsics_by_lens_zoom`
  (`LensZoomIntrinsics`, profile_registry.py:146-153), decoupled from the ARKit timeline.
- OWNER PROTOCOL (typed STOP: labeling/capture — the owner must physically shoot this):
  print a **ChArUco board, 5×7 squares, ~40mm square / ~30mm marker, DICT_4X4_50** on rigid flat
  foam-core (A2/A1). For EACH lens preset the phone actually uses in match capture (0.5×/1×/2× —
  confirm with owner which they use), record ~20s slow sweep holding the board at **3 distances
  (~1m, ~2.5m, ~4m)** and **≥6 orientations** (tilts ±30°, board filling different image regions
  INCLUDING the four corners — edge coverage is what constrains k1/k2). Keep the board static per
  shot; move the phone. No zoom changes mid-shot.
- Fit: `cv2.aruco.detectMarkers` → `interpolateCornersCharuco` → `cv2.calibrateCameraCharuco` per
  lens/zoom → store fx,fy,cx,cy,dist=[k1,k2,p1,p2] as a `LensZoomIntrinsics` via `update_profile`.
- Lane: **Sonnet** to write the capture-processing script (`scripts/racketsport/`), **owner** to shoot.
- Acceptance: RMS reproj from `calibrateCameraCharuco` ≤ **1.0px**; k1/k2 stable across the 3 distances
  (relative spread <20%). Store `source_trace` pointing to the capture clip.
- Kill: if RMS >2px or k1/k2 unstable → board too small/blurry/insufficient edge coverage; re-shoot,
  do NOT persist. typed STOP: advice if it fails twice.

**STEP 3 — P4-4c: back_project audit AS A STANDING TEST (pass/fail check, no judgment).**
- Objective: make the leak un-reintroducible.
- Recipe: add a test targeting `metric_calibration_from_sidecar_and_keypoints` (court_calibration.py:399)
  with a fixture sidecar carrying nonzero `intrinsics.dist` (source=='arkit'), OR unit-test the
  caller-level undistort helper from Step 1 directly — assert its edge floor points DIFFER from the
  zero-dist result at an edge pixel by >2px (proves dist is consumed). NOTE: a bare `CameraFloorGeometry`
  tests `back_project_pixel_to_floor` in isolation, NOT the entrypoint — use the entrypoint (file paths)
  or the Step-1 helper. PASS = dist changes output; FAIL = leak present.
- Lane: **Codex**, bundled with Step 1.

**STEP 4 — P4-0: court-profile MATCHING + verification + consumption (the v1 court path).**
- 4a MATCHING (Codex): extend `lookup_court_profile` (profile_registry.py:349) from exact-equality to
  ranked match. NEW param: add `line_color_lab: LabColor | None = None` to its signature. Signals, in
  order: (1) `camera_fingerprint` exact prefilter (already there); (2) **line-paint color** — computed
  by the CALLER, not inside lookup: from the upload's calibration-chosen keyframe (the frame the
  candidate profile's frozen calibration was solved on), project the 4 outer court-line polylines with
  that frozen calibration (`project_world_points`) and sample pixels ALONG the projected polylines
  (NOT raw HSV line detection — §4 forbids it; the frozen calibration already gives line locations),
  convert BGR→CIELAB (`cv2.cvtColor`), take the mean → `line_color_lab`; lookup compares it to
  `profile.line_paint_color_lab` (LabColor l/a/b) via ΔE2000, accept ≤ **10**; (3) optional
  `gps_hint`/`wifi_hint` if the sidecar carries them (radius match on GPS). At N≤3 courts a color +
  fingerprint match is sufficient — DO NOT build a learned embedding yet (see DO-NOT). Return
  `(profile, match_confidence)`.
- 4b 4-LINE VERIFICATION (Codex): reuse `project_world_points`/reproj machinery — take the frozen
  calibration, project the 4 outer court lines (2 baselines + 2 sidelines) into the upload frame,
  measure pixel support against detected lines (existing line bank / hough support). Accept reuse iff
  **4-line reproj median ≤4.8px AND p95 ≤12.3px** (the manual bars, §1) AND ≥3 of 4 lines recovered —
  use the median bar for the median and the p95 bar for the p95, never a median vs the p95 bar. This is
  the cheap "is this really the same court, unmoved camera" gate.
- 4c TRUST-BANDING (pre-ruled — no judgment). The GOVERNING reuse gate is the concrete conjunction
  everywhere (NOT an abstract "match_conf high"): **REUSE iff (camera_fingerprint exact) AND (ΔE2000
  ≤10) AND (4-line reproj median ≤4.8px AND p95 ≤12.3px AND ≥3/4 lines)**. The returned scalar
  `match_confidence` (define as `1 - ΔE2000/10` if returned) is ADVISORY ONLY — it never gates reuse.
  - color+fingerprint match AND 4-line PASS → **reuse frozen calibration**, mark `court_source="profile_reuse"`.
  - color+fingerprint match AND 4-line FAIL → camera moved / court repainted → fall through to
    metric15/manual solve, and (if solve passes) OFFER to refresh the profile (do NOT auto-overwrite;
    typed STOP: decision).
  - no match → generic path (metric15/manual/auto-find preview). NEVER ride the TRUSTED channel on an
    unverified guess.
- 4d CONSUMPTION: on profile_reuse the upload path must skip re-calibration and write
  `court_source='profile_reuse'`. ⚠️ The calibration-source seam AND the `court_assist_seed`/`court_review`
  channels live in `orchestrator.py` (FENCED) + `scripts/racketsport/review_input_server.py` +
  `replay_export.py`; `process_video.py` is FENCED too. There is NO non-fenced seam a fresh session can
  thread trusted reuse through today → **4d is BLOCKED on the fenced session.** SCOPE this lane to
  4a-4c (pure functions + tests) ONLY; deliver 4d as a typed handoff to the fenced-orchestrator owner:
  "matcher returns `(profile, match_confidence)` + the §3 reuse decision; wire it into the orchestrator
  calibration seam to emit `court_source='profile_reuse'` and short-circuit metric15 when the gate
  passes." Do NOT edit the fenced files.
- Lane: **Codex** (4a-4c pure functions + tests). Product wiring (4d) is a SEPARATE lane, BLOCKED on
  the fenced-orchestrator session — do not start until the fence lifts.
- Acceptance (NORTH_STAR P4-0 gate, exact): "a court profile **round-trips** (store → match on next
  upload → skip re-calibration with reproj ≤ the manual bar); missing-profile clip degrades to the
  generic path." Test with an owner clip pair from the SAME court (internal clips only).
- Kill: if ΔE2000 color match cannot separate the owner's courts (e.g. two identical blue courts) →
  add the 4-line geometry check as the tiebreaker (it already disambiguates); if STILL ambiguous →
  typed STOP: decision (owner tags the court at upload).

**STEP 5 — P4-6.0: net v1 data change (~1 line at each call site).**
- Objective: tape-measured net heights override regulation defaults.
- Files/signatures (grep-verified NON-fenced callers of `build_net_plane`):
  `external_gt_precomputed_calibration_runner.py:82`, `court_auto_evidence.py:149`,
  `calibration_overlay.py:59` (and `:272`), `court_corner_review.py:61`,
  `scripts/racketsport/calibrate.py:70`. `orchestrator.py:200/:339` are ALSO callers but are FENCED
  (§4) — do NOT edit them; the orchestrator picks up profile heights only once the fenced session
  threads the profile through (follow-up handoff). Deliver Step 5 for the NON-fenced callers only.
  NOTE: `court_keypoint_net.py:157` is NOT a `build_net_plane` caller — it reads the template constant
  `_PICKLEBALL_TEMPLATE.post_net_height_m` directly (module const at :150), so adding overrides to
  `build_net_plane` does nothing there. Leave that synthetic net-keypoint geometry on regulation
  defaults for v1; if tape heights are ever needed there, thread the profile in separately.
- Recipe: give `build_net_plane` optional `post_height_in`/`center_height_in` overrides (default None →
  template). At each call site, when a `CourtProfile` is in scope with
  `net_height_provenance == "tape_measured"`, pass `profile.net_post_height_in` /
  `profile.net_center_height_in`. Profile fields ALREADY exist (profile_registry.py:134-137) with the
  `_tape_measurement_requires_trace` validator — no schema change.
- Owner dependency (typed STOP: labeling): owner tape-measures post + center height ONCE per court,
  stored with `net_height_provenance="tape_measured"` + `net_height_source_trace`.
- Lane: **Codex**, tiny. Acceptance: `build_net_plane(sport, post_height_in=35.5, center_height_in=33.5)`
  yields a plane whose center height == 33.5in and posts == 35.5in in meters (not the 34/36in defaults);
  regression clips unchanged when provenance is `regulation_default`. NOTE `net_top_height_m_at_x(sport,
  x_m)` (net_plane.py:33) takes ONLY (sport, x_m) and reads `get_court_template(sport)` directly — it
  never sees a profile, so the acceptance MUST assert on the `build_net_plane(...)` result (center/post
  endpoints), NOT on `net_top_height_m_at_x`. Only extend that function's signature (optional overrides
  + thread at every caller) if a caller truly needs per-x profile heights; for v1 the plane endpoints
  suffice. Kill: none (pure data pass-through).

**STEP 6 — P4-6 net v2 catenary (STAGED, NOT v1-blocking; each stage its OWN gate).**
- 6a net-cord/post SEGMENTATION bootstrap: Grounding DINO "net cord/tape" prompt + owner labels → tiny
  seg head. Gate: its OWN seg-quality gate (IoU on owner labels). FALLBACK if seg fails: keep single
  `net_plane` + trust-band, DO NOT block the pipeline.
- 6b 3D CATENARY fit: Madaan ICRA'19 5-param planar catenary + rigid transform, distance-transform
  reprojection loss, using ARKit per-frame poses; degrades under occlusion. Gate: validate on
  SYNTHETIC/known geometry FIRST before any real clip.
- 6c real-clip integration + tape-measure GT. Gate (FINAL, not the only one): **net top-cord height
  error ≤ 2cm vs tape-measured GT at posts+center** on an owner clip. Secondary (quantified, not vague):
  cross-net ball-track fragmentation — number of track breaks within Npx of the net plane — does NOT
  increase vs the no-net-occlusion baseline on the same clip. Lane: **GPU/Codex**, staged,
  manager-gated between stages.

**STEP 7 — Auto-find epic (P4-1/2/3): DO NOT resume unless the entry condition fires (see §3).**

### 3. Decision trees (if X -> do exactly Y; ambiguous branches end in typed STOP)
- **Upload arrives:**
  - camera_fingerprint matches a CourtProfile AND ΔE2000 ≤10 AND 4-line reproj (median ≤4.8px AND
    p95 ≤12.3px, ≥3/4 lines) → REUSE frozen calibration (`court_source="profile_reuse"`). Done.
  - fingerprint matches BUT 4-line FAIL → metric15/manual solve; if it passes, surface "refresh this
    court profile?" → typed STOP: decision (never auto-overwrite a frozen calibration).
  - no profile, sidecar has ARKit floor plane → metric15 ARKit path (with Step-1 undistort). 
  - no profile, no ARKit → manual 15pt / guess+confirm UI (Wave A product path, trusted channel only
    on human confirm).
- **A non-profile / unknown court is actually uploaded (the auto-find entry condition):** ONLY THEN is
  the auto-find epic worth resuming, in this order: P4-1 (regenerate + land Wave A patch) → P4-3 GEO r3
  (top-3 cross-frame court-identity vote — fixes the 19.8px adjacent-court lock-on) → P4-2 train
  court_unet_v2 ONLY IF geometric still fails IMG_1605-class overlay. If no unknown-court upload ever
  happens → auto-find stays parked; spend the effort on profiles/distortion/net instead.
- **P4-1 patch application:** `git apply --check` on
  `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch` FAILS (bogus self-symlink
  hunks eval_clips/eval_clips, BUILD_CHECKLIST conflict, package.json conflict; baseline 501a1114 not
  an ancestor of main). DO NOT force-apply. → Regenerate: rebase `worktree-court-autofind-20260705`
  onto main, `git format-patch`, strip symlink hunks; OR cherry-pick with explicit conflict resolution
  (stash BUILD_CHECKLIST, resolve package.json). If the branch is gone/unrebasable → typed STOP: advice.
- **ChArUco fit RMS >2px or k1/k2 unstable twice** → typed STOP: advice (board/lighting/edge coverage).
- **Distortion undistort regresses metric15 p95** → gate undistort behind
  `distortion_model != "zero_distortion_grid_search_focal"`; if still bad → typed STOP: needs-validation.
- **Any held-out (Outdoor/Indoor) scoring of a court checkpoint** → prereg row in
  `runs/manager/heldout_eval_ledger.md` (CAL section) + manager go BEFORE the run. No exceptions.
- **Two owner courts are visually identical (color match cannot separate)** → 4-line geometry check is
  the tiebreaker; if still tied → typed STOP: decision (owner tags court at upload).

### 4. DO-NOT list (kill-listed approaches + the one-line reason)
- DO NOT build a learned image-retrieval EMBEDDING for court re-ID at v1 — N≤3 courts; fingerprint +
  ΔE2000 color + 4-line check is sufficient and has zero training cost (owner ruling: profiles-first, N small).
- DO NOT change `back_project_pixel_to_floor`'s signature/contract to take dist — its docstring
  correctly promises undistorted input; undistort at the CALLER (mirrors placement.py). Changing it
  would desync the two paths.
- DO NOT re-run court_keypoint heatmap training on point-only supervision — CAL-2/CAL-3 (ledger rows
  70-72) KILLED heatmap-then-points as architecturally wrong (PCK@5 0.017-0.056 vs 0.95 gate). Any
  neural retry is P4-2's 24M kp+line model at 640×360, NOT the killed 160×90 encoder.
- DO NOT HSV-mask LINES for detection — dead (0.00-0.03 support); mask the court SURFACE instead
  (DESIGN.md E2). Shadow normalization HURTS — don't.
- DO NOT assume a PnLCalib port beats our bar — even SOTA sports calib hits 1.4-4.5px on EASIER
  broadcast domains vs our 19.8px handheld/distorted case; a pilot must PROVE it clears the bar first.
- DO NOT auto-overwrite a frozen CourtProfile calibration on a 4-line mismatch — surface a refresh
  decision; a silent overwrite loses the golden calibration.
- DO NOT let unconfirmed auto-find guesses ride the TRUSTED `court_corners` channel — advisory
  (`court_assist_seed`/`court_review`) only until human confirm (Wave A closed this fail-open hole).
- DO NOT edit `orchestrator.py` / `process_video.py` — FENCED (active other session); route via the
  advisory/trusted split.

### 5. External bets verified today (2026-07-07)
- **PnLCalib / No-Bells-Just-Whistles** → REAL, code + weights public, **GPL-2.0**, SOCCER-ONLY.
  `github.com/mguti97/PnLCalib` + `.../No-Bells-Just-Whistles`. Weights (SoccerNet/WorldCup14/
  TS-WorldCup/WorldPose) on releases page. **NEW since Jul5: March 2026 update added LENS DISTORTION
  OPTIMIZATION to the refinement module + a weighting param α; accepted to CVIU journal** — directly
  relevant to P4-4 if we ever pilot the neural path. GPL internal-R&D-OK per owner 2026-07-04 ruling;
  ledger it. Weights were local at `models/checkpoints/court_external/` as of P4-3 (roadmap ~line 1162)
  but are NOT present on this Mac now (only `yolo26m.pt`) — reconfirm (GPU VM? swept in wave-2 disk
  cleanup?) before Step-4/pilot lanes assume a local checkpoint.
- **PoseGravity** → REAL, `github.com/akschion/PoseGravity`, **BSD-3-Clause**, C++ core with PyBind11
  Python bindings, low activity but available (arXiv:2405.12646). Closed-form O(n) pose from
  points+lines with a known gravity axis = exactly our ARKit sidecar case. PRIORITY integration WHEN
  ARKit lands (P0-10). Needs a compile step (not pip-install).
- **AnyCalib (ICCV'25)** → REAL, `github.com/javrtg/AnyCalib`, **Apache-2.0**, code + weights, pip -e
  installable (Py≥3.10 + PyTorch), arXiv:2503.12701. Model-agnostic single-view intrinsic+distortion
  sanity-check per clip.
- **GeoCalib (ECCV'24)** → REAL, `github.com/cvg/GeoCalib`, **Apache-2.0**, code + weights. Single-image
  intrinsics + gravity. AnyCalib builds on its siclib. Intrinsic sanity companion to AnyCalib.
- **BroadTrack (WACV'25)** → REAL, code released `github.com/evs-broadcast/BroadTrack`, arXiv:2412.01721.
  Halves reproj (10.28→5.02px) via a broadcast camera+tripod motion model on SoccerNet. SOCCER-specific;
  LICENSE NOT VERIFIED (check LICENSE file before any use). Concept (feed ARKit VIO as the motion prior)
  is open territory — reference only, not a drop-in.
- **NEW-court-domain scan (skeptical):** NO pickleball court/net calibration prior art exists anywhere
  (re-confirmed 2026-07-07). Adjacent monocular-3D racket work: TT3D (table tennis, arXiv:2504.10035),
  MonoTrack (badminton), **CalTennis (tennis multi-view dataset, arXiv:2606.20542, NEW)** — none are a
  court-calibration drop-in; CalTennis is a pose dataset, not code for our floor path. No change to the
  ruling.

### 6. Acceptance gates + owner dependencies (exact metric keys; what only the owner can provide)
- **P4-4 (distortion):** IMG_1605 placement residual at frame EDGES ≤ **2× center** residual; foot-pin
  `cap_exceeded_skips → 0` (emitted by `foot_pin.py`, read from the placement artifact). metric15 p95
  via `scripts/racketsport/validate_metric_calibration_15pt.py --run-dir <out>` (`reprojection_p95_px`);
  IMG_1605 material under `runs/*img1605*` (e.g. `runs/vp1_img1605_20260704T215924Z`). Edge/center
  split: edge = keypoints with radius >Npx from the principal point (cx,cy) — if no artifact emits
  per-region residual today, the Step-1 lane MUST add that split to its harness (typed sub-task, not
  assumed). Standing test: nonzero-dist floor point differs from zero-dist by >2px at an edge pixel.
  Artifact parity: `undistort_applied=true` in the ARKit path. (Real-clip edge/center gate DEFERRED
  until P0-10 ARKit capture; synthetic until then.)
- **P4-0 (profiles):** court profile ROUND-TRIPS (store → match on next upload → skip re-calibration)
  with reuse reproj **≤12.3px p95** (the manual bar); missing-profile clip degrades to generic path.
  Match keys: `camera_fingerprint` exact, line-color **ΔE2000 ≤10**, 4-line reproj median ≤ manual bar,
  ≥3/4 lines recovered.
- **P4-6.0 (net v1):** tape-measured heights change the `build_net_plane(...)` plane (center/post
  endpoints); `regulation_default` clips unchanged. **P4-6c (net v2):** net top-cord height error
  **≤ 2cm** vs tape GT at posts+center.
- **Auto-find promotion (P4-5, only if epic resumes):** held-out **PCK@5px ≥ 0.95** on owner-reviewed
  viewpoints, via `scripts/racketsport/evaluate_court_keypoint_owner_gate.py`; fail-closed
  mandatory-review below bar; CAL promotion row in the ledger or a documented miss.
- **VERIFIED discipline:** every one of the above is VERIFIED=0 until a passed documented gate on REAL
  labels. Any Outdoor/Indoor touch needs a pre-registered `heldout_eval_ledger.md` CAL row + manager go.
- **OWNER-ONLY inputs (typed STOP: purchase/labeling/decision as noted):**
  (1) ChArUco board print + the per-lens sweep capture (Step 2) — nobody else can shoot it.
  (2) Tape-measure each court's post + center net height ONCE (Step 5) — into the H0 profile.
  (3) Confirm WHICH phone lens presets (0.5/1/2×) are used in real match capture.
  (4) Decide profile-refresh on a 4-line mismatch; tag court if two courts are visually identical.
  (5) Roboflow key (only if the auto-find neural epic resumes) — external corpus re-fetch.

# BLUEPRINT — PHASE F (global fusion). Author: deep-design agent, 2026-07-07.
# Audience: the successor Opus-class MANAGER. Every step is pre-ruled; where judgment
# is unavoidable it says "typed STOP: <bucket>". Do not re-decide standing rulings.

## PILLAR: PHASE F — global fusion: one mutually-consistent metric 3D world (the capstone)

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
We DECIDED to ship Phase F in two committed steps and a gate, never as a from-scratch solve:
**PF-1 first** — two cheap post-hoc consistency corrections built in the `foot_pin.py` HOUSE PATTERN
(bounded max-correction caps + confidence-gated application, "can-only-nudge"): (a) ball<->paddle
impact snap (<=37mm each side, tapered +/-2 frames, gated on fused contact confidence); (b)
foot<->ground clamp + mesh non-penetration. **PF-2** — ONE offline contact-coupled joint optimizer
built as `scipy.optimize.least_squares(method='trf', loss='huber')` with a `jac_sparsity` block-sparse
Jacobian (the house idiom — every solver in this repo is scipy least_squares), whose weights are the
EXISTING confidence fields (event_fusion contact confidence, `SE3PoseConfidence.confidence`,
ball-chain trust band) with ZERO new modeling; **v0 is coordinate-descent** (re-run each subsystem's
own solver with cross-terms as soft constraints) because contacts are sparse. **PF-3** world temporal
smoothing at the seams; **PF-4** the trust-banded bundle + a `verify_process_video_viewer.py`
consistency assertion. We REJECTED ceres/GTSAM/differentiable rendering (uninstalled, not the house
idiom, not what JOSH uses). JOSH (arXiv:2501.02158, ICLR'26) is the pattern and its **code is now
released** (github.com/genforce/JOSH) — we PORT THE IDEA (contact-coupled joint opt), not the codebase,
and EXTEND it to ball+paddle+court which no published system does.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- VERIFIED=0 for Phase F. Nothing here has passed a product gate on real labels. Do not write otherwise.
- The house nudge pattern EXISTS and is proven at stance-only: `threed/racketsport/foot_pin.py`
  (`apply_foot_pin_to_payload`, `FootPinSettings`; caps `STANCE_MAX_XY_CORRECTION_M=0.30`,
  `R3_MAX_XY_CORRECTION_M=0.02`, `min_phase_confidence=0.20`; always emits `foot_pin_audit.json`).
  CLI: `scripts/racketsport/apply_foot_pin.py` (`--world`, `--max-correction-m`, `--taper-frames`).
- Foot-pin live metrics already land in the visual-polish memory (resets 14->2, feet jitter -60..75%,
  root p95 0.267->0.100 m) — evidence: memory `pickleball-visual-polish-20260705.md`. These are
  foot<->ground only; the ball<->paddle half of PF-1 is NOT built.
- Contact windows artifact EXISTS: `contact_windows.json` from
  `event_fusion.fuse_contact_windows` — each event has keys `frame`, `confidence`,
  `window{t0,t1,importance}`, `sources{wrist_vel,ball_inflection,audio,human_review}`
  (`threed/racketsport/contact_windows.py:build_contact_event`). Contact detection is HEURISTIC —
  a spurious window would create false "consistency"; this is why PF-1(a) is confidence-gated.
- Ball 3D world position EXISTS per frame in `ball_track_arc_solved.json` (authoritative `world_xyz`;
  frames outside solver coverage are `band=="hidden"`, `world_xyz=null`) overlaid onto
  `ball_track_physics_filled.json` — see `virtual_world.py:apply_ball_track_arc_solved_overlay`.
  Ball is trust-banded `low_confidence` everywhere today (`trust_band.derive_ball_trust_band`).
- Paddle 6-DoF EXISTS per frame in `racket_pose_estimate.json` (SE3 `{R,t}` + confidence via
  `racket6dof.SE3PoseConfidence.confidence`, computed `1/(1+reproj_px/3)`); world face corners via
  `virtual_world._paddle_mesh_vertices_world` + `racket6dof.paddle_face_corners_object_cm`. Paddle
  banded `low_confidence` (0 true-corner labels: `trust_band.derive_paddle_trust_band`).
- Player body: `virtual_world.json` `players[].frames[].joints_world` (list of [x,y,z]) + `joint_conf`,
  MHR70 names (`external_gt_body_prediction_schema.MHR70_JOINT_NAMES`).
- Court plane + net: `court_calibration.json` (`court_plane` = `Plane{point,normal}`,
  `metric_confidence`, `grade`) and `net_plane.json` (schema `NetPlane`: `plane`, `endpoints`,
  `center_height_in`, `post_height_in`) — net is a SINGLE PLANE today (3D catenary is P4-6, NOT a
  Phase-F dependency). The ball's clearance over the net (`height_over_net_m`) is a SEPARATE derived
  field on `BallArcRenderShot` in `ball_arc_chain.py` (from `net_clearance_m`), NOT part of the
  NetPlane schema; the PF-2 net residual uses the plane geometry + that derived clearance.
- Confidence/trust plumbing EXISTS and PF-2 REUSES it verbatim: `confidence_gate.py`
  (`ConfidenceGateConfig.confidence_threshold=0.5`, `band_from_sigma`, `apply_confidence_gate_to_world`),
  `trust_band.py` badges `verified|preview|low_confidence`.
- NO ball-paddle-gap or floor-penetration metric exists yet (grep: zero hits). PF-1 step 0 BUILDS the
  measurement first (baseline), then the correction.

### 2. The exact build plan (numbered steps)
Each step: objective / file targets / algorithm+recipe / lane sizing / acceptance metric keys / kill.

--- STEP F0 (PREREQUISITE, do before any PF-1 correction) — the baseline meter ---
- Objective: a read-only metric module that measures, on `virtual_world.json`, the two quantities
  PF-1 must improve, so "measurable reduction" is checkable.
- Files (NEW): `threed/racketsport/consistency_metrics.py` + `tests/racketsport/test_consistency_metrics.py`.
- Algorithm: (1) `ball_paddle_impact_gap_m(world, contact_windows)` = iterate
  `contact_windows["events"]` (artifact = `{"schema_version":1,"events":[...]}`); for each event with
  `confidence >= tau_contact` (DEFAULT 0.5 = `ConfidenceGateConfig.confidence_threshold`, SAME value
  used in F1), at `frame`, Euclidean distance between ball `world_xyz` and nearest
  paddle face-corner-quad point (nearest point = closest point on the FINITE paddle-face quad
  polygon, clamping the plane projection to the quad boundary — NOT the infinite face plane); return
  per-impact list + median.
  (2) `floor_penetration(world, court)` = fraction of penetrating points over all
  `players[].frames[].joints_world` z PLUS `mesh_vertices_world` when present, where a point
  penetrates iff its SIGNED DISTANCE to the calibrated court plane is < `-floor_eps_m`
  (`floor_eps_m=0.01`). Signed distance = `dot(court_plane.normal, x - court_plane.point)` read from
  `court_calibration.json` `court_plane` (`Plane{point,normal}`) — do NOT compare raw world-z (the
  plane may be tilted/offset). Return rate + max depth. Emit `consistency_baseline.json`.
- Lane sizing: ONE Codex lane (pure-python, deterministic, unit-tested; no GPU).
- Acceptance: unit tests pass on a synthetic world; running on the STANDING PHASE-F ACCEPTANCE SET
  — the two NON-held-out development clips under `eval_clips/ball/`: `owner_IMG_1605_8a193402780b`
  and `wolverine_mixed_0200_mid_steep_corner` (produce each clip's `virtual_world.json` via
  `process_video`) — yields finite numbers. NEVER use the held-out `indoor_doubles_*` /
  `outdoor_webcam_*` clips for this gate (held-out label rule). Metric keys emitted:
  `ball_paddle_impact_gap_m.median`,
  `ball_paddle_impact_gap_m.per_impact`, `floor_penetration.rate`, `floor_penetration.max_depth_m`.
- Kill: none (measurement only).

--- STEP F1 (PF-1) — cheap consistency priors, foot_pin house pattern ---
- Objective: two bounded, confidence-gated post-hoc corrections that reduce the F0 metrics with zero
  regression to standalone metrics.
- Files (NEW): `threed/racketsport/consistency_priors.py`
  (`apply_consistency_priors_to_payload(world, *, settings, contact_windows) -> ConsistencyResult`
  mirroring `apply_foot_pin_to_payload`'s (payload, audit) shape), CLI
  `scripts/racketsport/apply_consistency_priors.py`, tests
  `tests/racketsport/test_consistency_priors.py`. STAGE INSERTION: insert the tuple
  `("consistency", self._stage_consistency)` into the sequence built by `_build_suffix_stage_fns()`
  in `scripts/racketsport/process_video.py`, between `("world", self._stage_world)` (line 516) and
  `("confidence_gate", self._stage_confidence_gate)` (line 517). Consistency MUST run BEFORE
  confidence_gate so trust badges reflect the CORRECTED geometry. (`_stage_world` is defined at line
  2950 — that is the method being wired, NOT the insertion point; do NOT edit the prefix
  `("placement", ...)` list at line 497.) It needs ball world_xyz + paddle pose + player joints
  together. Reuse the `foot_pin` gating literals; do NOT modify `foot_pin.py`.
- Algorithm (a) ball<->paddle impact snap: iterate `contact_windows["events"]` (the artifact is
  `{"schema_version":1,"events":[...]}`); for each event with `confidence >= tau_contact` (DEFAULT
  0.5 = `ConfidenceGateConfig.confidence_threshold`), SELECT the striking paddle: the one belonging to
  `event["player_id"]`; if `player_id is None`, pick the paddle whose face-nearest-point is closest to
  the ball `world_xyz` at `frame`. NEVER move both paddles for one event. Let
  d = ball_center - paddle_face_nearest_point at the impact frame (nearest point = closest point on the
  FINITE paddle-face quad polygon, clamped to the quad boundary, NOT the infinite plane). Move ball by
  `+w_b * clamp(d/2, |.|<=0.037m)` and that paddle's translation `t` by `-w_p * clamp(d/2,
  |.|<=0.037m)` where
  `w_b = ball_conf/(ball_conf+paddle_conf)`, `w_p = 1-w_b` (confidence fields already exist: ball
  frame `conf`, paddle `SE3PoseConfidence.confidence`). Taper linearly over `taper_frames=2` each side
  (weight 1, 2/3, 1/3). Skip events where ball frame `band=="hidden"` (no measured ball -> do NOT
  fabricate). (b) foot<->ground + non-penetration: run the EXISTING `apply_foot_pin_to_payload` on the
  world payload — its ACTUAL upstream call site is `worldhmr.py` (line 662), NOT placement. If
  `virtual_world.json` already reflects foot_pin (applied in worldhmr), F1(b) MUST perform ONLY the
  non-penetration z-clamp and MUST NOT re-run `apply_foot_pin_to_payload` (avoids double-correction);
  re-run foot_pin here ONLY if the audit shows it was not applied upstream. The z-clamp lifts any
  body-mesh/joint below the court plane up to the plane, capped at `max_penetration_fix_m=0.05`.
  Always emit `consistency_audit.json`
  (before/after F0 metrics, per-correction magnitude, cap-exceeded skips) — copy the audit discipline
  from `foot_pin._audit_payload`.
- Hyperparameters (defaults, CLI-overridable): `tau_contact=0.5`, `max_ball_paddle_correction_m=0.037`,
  `taper_frames=2`, `max_penetration_fix_m=0.05`, `floor_eps_m=0.01`.
- Lane sizing: ONE Codex lane (decomposable, unit-testable, no GPU). Follow the run-lane skill template.
- Acceptance (gate keys from F0): on the Phase-F acceptance set (F0 standing clips),
  `ball_paddle_impact_gap_m.median` strictly decreases,
  `floor_penetration.rate` decreases, AND standalone metrics do not regress — run
  `python scripts/racketsport/build_body_gate_report.py` -> `body_gate_report.json` and compare the
  clip's `world_mpjpe.core_mean_error_m` (unchanged within tol; regression = any increase),
  `visual_quality.py` jitter/reset counts not worse. Corrections that would
  exceed a cap are SKIPPED and logged, never clamped-then-applied silently.
- Kill: if a correction increases any standalone metric, disable that sub-correction (gate it off) and
  ship only the passing half; if contact windows are too noisy (>30% skipped as low-conf), raise
  `tau_contact` to 0.6 and re-measure. If both halves regress -> typed STOP: needs-validation.

--- STEP F2-DEP (gate) — the dependency checklist that must be GREEN before PF-2 starts ---
PF-2 is NOT worth starting until ALL of these are true (current status in brackets):
  1. PF-1 landed + its gate passed on the Phase-F acceptance set (F0 standing clips) [NOT STARTED].
  2. Per-frame paddle pose present + non-degenerate on the target clips (`racket_pose_estimate.json`
     with real `SE3PoseConfidence.confidence`) [CORRECTED per PART B ruling B.1.3: BUILT-NOT-WIRED — manual CLI only until paddle P3-1 lands in-pipeline; PF-1 ships its foot<->ground term FIRST and adds the ball<->paddle term after P3-1].
  3. Ball arc solver produces >=2 confident anchored segments per rally on target clips
     (`ball_track_arc_solved.json` not all-hidden) [EXISTS but clip-dependent — VERIFY per clip].
  4. Court metric calibration `grade` acceptable + `metric_confidence` present [EXISTS].
  5. P2-2 latent MHR pose-code + FROZEN decoder callable as a python function returning joints from a
     latent [VERIFY: grep `threed/racketsport` for a decoder entrypoint such as
     `decode_latent`/`latent_decoder`/`pose_code`/`decode_pose`. As of 2026-07-07 this returns ZERO
     hits — the decoder does NOT exist. RULING: the ONLY supported PF-2 is the root+global-orient
     degrade (decision tree); do NOT attempt the latent-code variable vector. Full-latent fusion is a
     typed STOP: decision, blocked on P2-2 landing].
  6. ARKit sidecar present+valid OR the P2-1 RAFT+MAD camera seed exists (`camera_motion` path via
     `process_video._placement_camera_motion_path`) [camera_motion EXISTS; ARKit P0-10 is owner-gated].
If any of 2-6 is red on a clip, PF-2 runs in DEGRADED mode for that clip (trust-band the whole world),
never blocks the others.

--- STEP F2-v0 (PF-2 coordinate descent) — the legitimate cheaper first optimizer ---
- Objective: alternating-block refinement that adds cross-terms as soft constraints and measurably
  beats the PF-1 baseline, using existing per-subsystem solvers.
- Files (NEW): `threed/racketsport/fusion_solver.py` (`solve_world_coordinate_descent(world, *,
  weights, budget) -> FusionResult`), reusing: camera seed from `camera_motion`; body from the
  placement/worldhmr solver; ball from `ball_arc_solver`/`ball_physics3d.reconstruct_bounce_arcs_from_image_track`;
  paddle from `racket6dof.camera_paddle_pose_to_court_world`. Tests + a small offline harness.
- Block update ORDER per sweep (fixed). No standalone solver today accepts external anchors, so build
  a thin residual-augmentation wrapper per block, `fusion_solver._augment_<block>`, that STACKS the
  cross-term residuals onto that solver's own residual vector (do NOT rewrite the solvers). ACCEPTANCE
  CLAUSE (every wrapper): with an EMPTY anchor set the wrapper MUST reproduce the standalone solver's
  output BIT-FOR-BIT. Blocks: (1) camera trajectory — `camera_motion` seed (ARKit-locked-ish when
  sidecar valid: optimize only a global similarity + slow drift; else full 6-DoF/frame from RAFT+MAD
  seed AND set world trust band degraded). (2) each player root+pose — `_augment_body` wraps the
  placement/worldhmr body solver with the F0/F1 contact residuals as soft anchors. (3) ball arc
  segments — `_augment_ball` wraps `ball_arc_solver` /
  `ball_physics3d.reconstruct_bounce_arcs_from_image_track`, re-fitting parabolas with paddle-impact +
  ground-bounce anchor points. (4) paddle 6-DoF — `_augment_paddle` wraps
  `racket6dof.camera_paddle_pose_to_court_world` PnP with grip-offset + impact soft constraints.
  Repeat.
- Cross-terms injected as soft constraints (weights = existing confidence): ball<->paddle impact,
  ball<->ground bounce, foot<->ground, hand<->paddle grip (bounded slowly-varying offset from the
  gripping hand palm frame — reuse the grip transform `G = (R_g, t_g)`, hand->paddle SE3, produced by
  `paddle_pose_fused._fit_grip_transform`; hand anchor residual = `H(t) @ G` vs the paddle pose),
  net-height consistency (ball/player behind `net_plane` must respect the derived `height_over_net_m`
  clearance on `BallArcRenderShot`, NOT a NetPlane field; occluded frames trust-banded, NOT
  fabricated), non-penetration, known-scale anchors (court 20x44ft, net 36in ends/34in center, ball
  74mm=37mm radius, per-player heights from H4).
- Convergence: stop when max root delta < 1mm AND max change in any reprojection residual < 0.1px
  between sweeps, OR sweep cap = 5. Runtime budget: offline, whole-clip; JOSH-speed ~0.8 FPS on A100
  is acceptable (a 10s/300-frame clip ~ several minutes per sweep — fine).
- Lane sizing: ONE GPU/Sonnet build lane on an A100 (needs the frozen decoder + real clips); provision
  via gpu-fleet-provision skill only if no idle A100 in `runs/manager/gpu_fleet.md`.
- Acceptance (ALL must hold on the Phase-F acceptance set (F0 standing clips), none regress):
  world-MPJPE improves vs PF-1 baseline (`body_gate_report.json` `world_mpjpe.core_mean_error_m`),
  `floor_penetration.rate` improves, `ball_paddle_impact_gap_m.median` improves, runtime within
  offline budget. JOSH's 314->149mm is a WITHIN-optimizer ablation — cite DIRECTIONALLY only, never as
  our number.
- Kill: JOSH needs VISIBLE contact — for frames where contact is occluded, fail CLOSED to PF-1 output +
  trust-band those frames (`band` low_confidence), do not fabricate. If coordinate descent oscillates
  (residual not monotone over 3 sweeps) -> reduce step (damp block updates 0.5x) once; still oscillating
  -> typed STOP: advice.

--- STEP F2-v1 (PF-2 full joint) — only if v0 passes and v0's seams motivate it ---
- PRECONDITION: the full-latent variable vector REQUIRES the P2-2 frozen MHR decoder (F2-DEP #5),
  which does NOT exist today (zero grep hits). While it is absent, F2-v1 is NOT buildable as specced —
  stay on v0 (a shippable PF-2); attempting full-latent fusion is a typed STOP: decision.
- Objective: single `scipy.optimize.least_squares(method='trf', loss='huber', jac_sparsity=S)` over
  the stacked variable vector, initialized from v0's output (v0 IS the init).
- Variable dims (10s @30fps = 300 frames, 2 players, 2 paddles): camera 6*300=1800 (or ~7 total if
  ARKit-locked to global similarity); body 2*300*(6 root + ~40 latent MHR code)=~27.6k; ball arc
  ~10 segments*6=60; paddle 2*300*6=3600. Total ~33k free params.
- Residual dims: 2D reprojection body ~2*300*25*2=30k, ball ~300*2, paddle ~2*300*4*2=4.8k, court lines
  static; contact residuals sparse (impacts ~10*3, bounces ~10*3, foot-ground stance frames);
  scale anchors handful. Total ~40k residuals.
- Sparsity S (`jac_sparsity`, shape 40k x 33k, ~99% zero): each reprojection residual depends only on
  its own frame's camera + that subsystem's frame vars; each contact residual couples exactly the two
  frames it joins. Build S as a scipy.sparse.lil_matrix of 1s at those blocks. Providing S forces the
  'lsmr' trust-region solver (confirmed in SciPy docs) — required at this scale.
- Init order: camera (ARKit/RAFT) -> body root -> body latent -> ball arc -> paddle, then hand v0.
- Lane sizing: ONE A100 lane. Acceptance = same keys as v0, must beat v0. Kill: if trf does not
  converge in the offline budget with the sparsity pattern, STAY on v0 (v0 is a shippable PF-2) —
  typed STOP only if the OWNER asks for v1 accuracy beyond v0.

--- STEP F3 (PF-3) — world temporal smoothing at the seams --- (do after F2-v0 ships)
- Reuse P2-2 latent smoothing + PF-2 residuals across time. Gate: no cross-stage seam artifacts in the
  viewer; world jitter/slide must be <= the best per-stage value (NO regression vs any single stage),
  measured by `visual_quality.py` + the new world-consistency metric (F0 module). Lane: ONE Codex/GPU
  lane. This blueprint does not further spec F3 (out of scope focus).

--- STEP F4 (PF-4) — the deliverable + the honesty assertion ---
- Objective: extend `scripts/racketsport/verify_process_video_viewer.py` with a cross-system
  consistency assertion, mirroring the existing `assert_ball_honesty` fail-closed pattern (line 113).
- File targets: add `assert_cross_system_consistency(consistency_hud, *, audit, kill_reasons)` next to
  `assert_ball_honesty`; call it inside `verify_viewer_loads` alongside the existing
  `assert_ball_honesty` (line ~337) and append to `assertion_errors`.
- Assertion content (fail-closed): for every contact event above `tau_contact`, the viewer world's
  ball<->paddle gap at that frame must be <= a gate (`max_rendered_impact_gap_m=0.05`); no rendered
  body vertex more than `floor_eps_m` below the court plane; any element without a measured value must
  carry a non-`verified` trust band (never render fabricated consistency as `verified`). Read the truth
  from `consistency_audit.json` (F1) — do NOT recompute geometry in JS. CAP-LIMITED BRANCH: the PF-1
  snap moves each side by at most `clamp(d/2,<=0.037m)`, so an initial gap >~120mm cannot reach the
  50mm render gate. If a confident contact's post-correction gap still exceeds
  `max_rendered_impact_gap_m` BECAUSE the per-side correction hit its cap, trust-band that frame
  `low_confidence` and record `cap_limited=true` in `consistency_audit.json`; the assertion treats
  cap-limited frames as banded (NOT verified) and does NOT fail on them.
- Lane sizing: ONE Codex lane (python + a small viewer HUD hook, unit-tested with a synthetic audit).
- Acceptance: on a fresh owner clip, viewer loads, ball visibly meets paddle at every detected contact,
  feet planted, unmeasured elements banded; `assertion_errors == []`. Metric surfaced:
  `headless_verify.json` gains `consistency.impact_gap_max_m`, `consistency.floor_penetration_rate`.
- Kill: if the assertion fires on a healthy run because contacts are spurious -> raise `tau_contact`,
  do not weaken the gap gate. If it cannot pass on any clip -> typed STOP: needs-validation.

### 3. Decision trees (if X -> do exactly Y)
- ARKit sidecar present + valid? YES -> camera block optimizes only global similarity + slow drift,
  world trust band NORMAL. NO -> seed from `camera_motion` (RAFT+MAD), optimize full 6-DoF/frame, AND
  degrade the WHOLE-world trust band (not just occluded frames).
- Contact window confidence < `tau_contact` at an impact? -> SKIP the ball<->paddle snap for that event,
  log it in the audit, leave both entities untouched (never fabricate coincidence).
- Ball frame `band=="hidden"` (no measured 3D)? -> never snap ball to paddle; paddle unchanged; frame
  stays trust-banded. -> also applies inside PF-2 (fail-closed to PF-1 for occluded contact).
- Frozen MHR latent decoder callable (F2-DEP #5)? YES -> body block optimizes root + latent code.
  NO -> body block optimizes root + global orient only (fewer DOF); note the degrade in the audit; do
  NOT invent a decoder -> if owner wants full-latent fusion, typed STOP: decision.
- PF-1 sub-correction regresses a standalone metric? -> gate that sub-correction OFF, ship the passing
  half, re-measure. Both regress -> typed STOP: needs-validation.
- PF-2 v0 oscillates (residual non-monotone 3 sweeps)? -> damp block updates 0.5x once; still
  oscillates -> typed STOP: advice.
- trf/lsmr v1 will not converge in offline budget? -> stay on v0 (shippable); v1 only on owner request.
- Any live wave-3 codex lane already editing `process_video.py` / `virtual_world.py`? -> do NOT dispatch
  a colliding lane; serialize behind it (check `BUILD_CHECKLIST.md` + `runs/manager/gpu_fleet.md`).
- Someone proposes adding ceres/GTSAM/nerfstudio/a new SE3 lib? -> typed STOP: decision (violates the
  house-idiom ruling; default answer is NO).

### 4. DO-NOT list (kill-listed approaches + traps)
- DO NOT use ceres, GTSAM, or differentiable rendering. Reason: uninstalled, not the house idiom, not
  what JOSH uses; solver is RULED to scipy trf/huber or torch-LM (roadmap PF-2, tech-audit).
- DO NOT snap ball<->paddle on low-confidence or hidden-ball frames. Reason: contact detection is
  heuristic; a spurious window fabricates false consistency (roadmap PF-1 note).
- DO NOT introduce any NEW confidence/weight modeling. Reason: weights are RULED to be the EXISTING
  fields (event_fusion confidence, `SE3PoseConfidence.confidence`, ball-chain trust) — zero new modeling.
- DO NOT clamp-then-apply a correction that exceeds a cap. Reason: house pattern SKIPS + logs
  (`foot_pin` cap_exceeded_skips); silent clamping hides errors.
- DO NOT chunk the clip for PF-2. Reason: whole-clip beats chunked ~12% (TT4D/JOSH); offline budget
  allows whole-clip.
- DO NOT modify `foot_pin.py` (a live wave-3 lane may touch it). Reason: reuse via import, not edit.
- DO NOT cite JOSH's 314->149mm as our result. Reason: it is a within-optimizer loss ablation — cite
  directionally only. VERIFIED stays 0 until OUR gate passes on real labels.
- DO NOT treat the net as 3D catenary for PF-2. Reason: net catenary is P4-6 (separate); PF-2 uses the
  existing `net_plane` + `height_over_net_m`.
- DO NOT open, read, or reference Outdoor/Indoor held-out eval label files. Reason: standing hard rule.

### 5. External bets verified today (2026-07-07)
- Claim: JOSH is the SOTA contact-coupled joint-opt pattern and its CODE is released.
  Verdict: TRUE. Source: https://github.com/genforce/JOSH ("[ICLR 2026] Official Implementation",
  217 stars, last push 2026-04-15, not archived) + https://arxiv.org/abs/2501.02158 +
  https://openreview.net/forum?id=7eLE4mfEpz. License: NO LICENSE file present yet (research code;
  per project's "licenses-don't-matter" ruling we PORT THE IDEA, not the code — no legal blocker).
  Repo now also supports Pi3X point-cloud init + ships JOSH3R (feed-forward variant). Use as REFERENCE.
- Claim: scipy least_squares supports our block-sparse scale via `jac_sparsity`.
  Verdict: TRUE. Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
  — `jac_sparsity` (shape m x n) speeds finite-diff for few-nonzero rows and FORCES the 'lsmr'
  trust-region solver; 'trf' is documented as suited to large sparse bounded problems. API fits ~40k x
  33k. License: BSD-3 (scipy, already a dependency). Code availability: installed.
- Claim: no 2026 work does contact-coupled multi-object fusion for racket sports (our novelty).
  Verdict: HOLDS. Nearest neighbors are table-tennis reconstruction (TT4D arXiv:2605.01234, TT3D
  arXiv:2504.10035) and a tennis serve rigid-body inverse-dynamics model (arXiv:2512.18320) — none
  JOINTLY optimize ball+body+paddle+court with contact coupling; TT4D does inverse-control stroke
  fitting, not a joint world solve, and is table-tennis. No pickleball court/net fusion prior art.
  Sources: https://arxiv.org/abs/2605.01234 , https://arxiv.org/html/2504.10035 ,
  https://arxiv.org/html/2512.18320v1 . Nothing NEW since 2026-07-05 changes this pillar.

### 6. Acceptance gates + owner dependencies
- PF-1 gate keys (from `consistency_metrics.py`, step F0): `ball_paddle_impact_gap_m.median` (down),
  `floor_penetration.rate` (down), `floor_penetration.max_depth_m`; plus no-regression on
  `body_gate_report.json` `world_mpjpe.core_mean_error_m` + `visual_quality.py` jitter/reset counts.
  Phase-F acceptance set (F0 standing clips).
- PF-2 gate keys: `world_mpjpe.core_mean_error_m` (down vs PF-1), `floor_penetration.rate` (down),
  `ball_paddle_impact_gap_m.median` (down), runtime within offline budget; none regress; Phase-F
  acceptance set (F0 standing clips).
- PF-4 gate keys: `verify_process_video_viewer.py` `assertion_errors == []`, `headless_verify.json`
  `consistency.impact_gap_max_m <= 0.05`, `consistency.floor_penetration_rate` at floor.
- OWNER-ONLY dependencies (typed STOP: purchase-approval / decision / labeling if missing):
  (1) ARKit P0-10 sidecar capture (owner phone build) — unblocks the camera-locked path; absence forces
  the RAFT+MAD degraded path (works, lower trust). (2) Per-player heights (H4) for the metric scale
  anchor — owner-provided. (3) Any true paddle-corner or contact labels needed to ever move ball/paddle
  trust bands ABOVE `low_confidence` — none exist today; VERIFIED stays 0 without them.
  (4) Commit/push permission — joint-commit rule; do not commit Phase-F code until the owner says so.

## PILLAR: COACHING + STATS — shots, the stat layer, reference ranges, grounded-LLM coach, visual feedback

Audience: the successor MANAGER (Opus-class). This is executable. Every path/flag/metric key below is
grep-verified against the repo on 2026-07-07. Where judgment could enter, it is PRE-RULED or marked
"typed STOP". VERIFIED=0 for this pillar today — nothing here is VERIFIED until a P6-4 gate passes on
real owner labels.

### 0. Final ruling — the stack in one paragraph (what we use and why, past tense of decision)
We locked a **3-stage causally-validated pipeline** (NORTH_STAR II.4, PHASE 6): (1) a **deterministic
feature extractor** over existing 3D artifacts (virtual_world, contact_windows, ball_track_arc_solved,
court_zones, paddle_pose_fused, foot_contact_phases, rally_spans) → (2) a **rule/reference-range
comparator that is NOT the LLM** — typed findings with severity + evidence pointers (frame ranges +
artifact paths) → (3) a **format-locked LLM** (`claude-opus-4-8`, structured JSON: score + exactly-N
corrections + one drill) that NEVER sees a raw number without the comparator's verdict, and NEVER invents
metrics. Shots start **rule-based** on 3D features (learned classifier only when owner labels allow). Stats
start **minimal** (unforced errors, third-shot success, dink-rally win rate), each with a trust band + "how
we measured" popover. The reference-range library is **versioned JSON with per-range provenance** — the
moat; none exists publicly. Free LLM narration is kill-listed (SportsGPT ablation). Decided by Fable+owner.

### 1. Current measured state (numbers + evidence paths only — no aspirations)
- **Built:** `rally_metrics.py::build_rally_metrics(run_dir)` emits `rally_metrics` + `coaching_card_facts`;
  **position-based only** (`policy.pose_biomechanics_used=false`, `policy.protected_eval_labels_read=false`).
  Metric keys (exact): `distance_covered_m`, `avg_speed_mps`, `p95_speed_mps`, `zone_occupancy` (sub-keys
  `kitchen`/`transition`/`baseline`/`out_of_court`), `kitchen_proximity_s`, `contact_count`,
  `contact_positions_world`. Each dict carries `value/unit/frames_used/frames_total/coverage_fraction/trust`;
  trust ∈ {`"ok"`,`"estimated"`,`"unverified_cue"`}; `MIN_OK_COVERAGE=0.8`.
- **Contact schema** (`contact_windows.py::build_contact_event`): `type="contact"`, `t`, `frame`,
  `player_id`, `confidence`, `sources{wrist_vel, ball_inflection, audio?, human_review?}`,
  `window{t0,t1,importance}`, optional `trust_band_note`. Wrist-cue-only → `unverified_cue` (`rally_metrics._is_wrist_cue_only`).
- **Ball arc** (`ball_arc_solver.py`, ARTIFACT_TYPE `racketsport_ball_track_arc_solved`): only top-level
  status `"ran"` is trusted (self-kill statuses `degenerate_zero_segments` etc. are untrusted — see
  `verify_process_video_viewer.py` `TRUSTED_BALL_ARC_SOLVER_STATUSES`). Physics bands present: `min/max_plausible_speed_mps` (3–35),
  `max_plausible_apex_m=8.0`. Bounce anchors + free-flight segments are render-only today.
- **In/out uncertainty** (`ball_inout_uncertainty.py`): camera-geometry margins 0.94–2.26 m documented;
  `binding_boundary_axis` picks sideline vs baseline. Honest band, NOT an officiating claim.
- **Paddle** (`paddle_pose_fused.py`, ARTIFACT_TYPE `racketsport_racket_pose_estimate`): emits
  `face_normal_world`, provenance band `contact_locked`, render style `paddle_face_with_handle`.
- **Viewer** (`web/replay/README.md`): loads `replay_viewer_manifest.json`; trust badges, court-map, mesh,
  moment-jump, `ball_arc_solved_url`, net_plane→arc all landed. Honesty gate: `verify_process_video_viewer.py --manifest <run>/replay_viewer_manifest.json`.
- **Partially built (P6-1):** a rule-based shot classifier ALREADY exists — `shot_taxonomy.py::classify_
  shots_from_payloads` (ARTIFACT_TYPE `racketsport_shots`; SHOT_TYPES smash/lob/dink/drop/drive/atp/erne/
  tweener + serve/return/third-shot detection; carries `schema_version`), CLI `scripts/racketsport/classify_
  shots.py`, siblings `shot_classifier.py`/`shot_trainable_baseline.py`/`shot_transfer_baseline.py` (+ tests).
  S1 MUST reconcile with / extend these, NOT spin up a colliding `racketsport_shots` producer.
- **NOT built yet:** value stats (P6-2), reference-range library (P6-3), comparator + LLM harness (P6-4),
  coaching overlays (P6-5), session report (P6-6).
- **Owner-label volume for shots:** unknown/zero registered. → learned classifier is OUT until a
  `heldout_eval_ledger.md` row exists (typed STOP: labeling).

### 2. The exact build plan
Order is fixed: **S1 features → S2 reference library → S3 comparator → S4 LLM harness → S5 overlays**.
S2 can run in parallel with S1 (it is pure JSON + owner review). Do NOT start S4 before S3's finding
schema is frozen.

**S1 — Shot segmentation + feature vector (P6-1) [Codex lane, CPU-only, no GPU].**
- Objective: from existing artifacts, produce `shots.json` = ordered list of shots, each with a feature
  vector + a rule-based class + confidence. No new model. No held-out labels touched.
- **STEP 0 (mandatory): read `shot_taxonomy.py` first.** It already emits ARTIFACT_TYPE `racketsport_shots`
  with per-shot landing (`_landing_from_segment`), apex (`_peak_height`), ground-intersection
  (`_ground_intersection_dt`), rally index (`_rally_index`/`third_shot`), outcome + uncertainty. Decide
  EXTEND-vs-REPLACE: if it already emits the S1 feature keys below, extend IN PLACE — do NOT create a second
  module. Fork only with a documented reason + a DISTINCT artifact_type (e.g. `racketsport_shot_features`) +
  a migration note for existing `racketsport_shots` consumers (viewer gates key off the type). Reuse arc helpers.
- File targets: extend `shot_taxonomy.py` (preferred) or a new module reusing `rally_metrics` loaders
  (`read_virtual_world_tracks`, `_load_court_zones`, `_load_contact_events`, `_load_rally_spans`,
  `_zone_for_point`). Test: `tests/racketsport/test_shots.py`. Keep ARTIFACT_TYPE `"racketsport_shots"` ONLY
  when extending the existing producer; a fork MUST use a new type. Bump `schema_version`.
- A shot = one contact event (`type=="contact"`) attributed to a player, paired with the ball arc
  segment that STARTS at that contact (nearest arc segment whose first observation `t` ≥ contact `t`,
  within 0.20 s; else `arc=null`, trust downgraded).
- Feature vector per shot (exact keys):
  `contact_t`, `contact_frame`, `player_id`, `contact_world_xy` (from nearest player frame),
  `contact_zone` (`kitchen`/`transition`/`baseline`/`out_of_court` via `_zone_for_point`),
  `contact_height_m` (ball arc z at contact via the arc integrator — see Arc-segment access; else null),
  `apex_height_m` (arc max z on segment via `_peak_height`; else null),
  `landing_world_xy` + `landing_zone` (next-bounce court-plane point via `_landing_from_segment`; else null),
  `net_crossing_height_m` (arc z where the segment crosses the arc artifact's `net_plane` — reuse the
  `shot_taxonomy` net-plane handling, NOT y=0; null if no arc or no net_plane),
  `horizontal_speed_mps` (m/s) + `outgoing_dir` (unit 2D [dx,dy] in world court-plane coords, horizontal
  component only, from the first ~3 arc obs; null if no arc),
  `rally_shot_index` (1-based position within its `rally_spans` rally),
  `is_serve_candidate` (rally_shot_index==1), `prev_shot_bounced` (bool: was there a bounce between the
  previous contact and this one — needed for volley vs groundstroke),
  `trust` (min of contributing artifact trusts; `unverified_cue` if contact is wrist-cue-only,
  `estimated` if arc absent or coverage<0.8).
- **Arc-segment access (z is NOT a stored field).** Segments live under the arc artifact `segments[]`; each
  carries (`FlightSegmentFit.to_json`): `segment_id`, `t0`/`t1`, `frame_start`/`frame_end`, `start_anchor`/
  `end_anchor` + `anchors_used[]`=`{anchor_id,kind,t,frame,world_xyz,status}`, `initial_position_m`,
  `initial_velocity_mps`, `initial_speed_mps`, `net_clearance_m`, `net_clearance_ok`. No stored z — get z(t)
  via the integrator (`FlightSegmentFit.predict` or `shot_taxonomy` `_peak_height`/`_ground_intersection_dt`/
  `_landing_from_segment`). Consume ONLY when arc status==`"ran"`; bounce anchors give landing, else null.
- Rule table v0 (evaluate top-to-bottom, FIRST match wins; every class carries `confidence` in [0,1] and
  `rule_fired`). Thresholds are hand-tuned constants IN the classifier module (see kill-criterion); the S2
  library holds skill-band success RANGES, NOT these apex/height/speed cutoffs — do not conflate them:
  1. **serve**: `rally_shot_index==1` AND `contact_zone==baseline`. conf 0.9.
  2. **return**: `rally_shot_index==2` AND `prev_shot_bounced`. conf 0.85.
  3. **lob**: `apex_height_m >= 3.0` AND `landing_zone in {baseline,transition}`. conf 0.7.
  4. **smash**: `contact_height_m >= 1.6` AND `horizontal_speed_mps >= 12` AND downward
     (`net_crossing_height_m` < `contact_height_m`). conf 0.7.
  5. **volley**: `NOT prev_shot_bounced` AND `contact_zone in {kitchen,transition}`. conf 0.65.
  6. **dink**: `prev_shot_bounced` AND `contact_zone==kitchen` AND `apex_height_m < 1.2` AND
     `landing_zone==kitchen`. conf 0.75.
  7. **drop**: `(rally_shot_index==3 OR contact_zone=="baseline") AND landing_zone=="kitchen" AND
     apex_height_m < 2.5`. conf 0.7. ("hitter at baseline" = `contact_zone=="baseline"`; 3rd-shot-drop is
     the flagship — see S2.)
  8. **drive**: fallback for any remaining shot with `horizontal_speed_mps >= 10` AND
     `apex_height_m < 1.5`. conf 0.6.
  9. else **unknown**, conf 0.3.
- **Null-feature semantics:** any rule condition referencing a null feature evaluates to False (the rule
  cannot fire; never `None>=x`). An arc-null shot can only match arc-free rules (serve/return/volley via
  `rally_shot_index`/`prev_shot_bounced`/`contact_zone`) or falls to unknown — never crashes. A fixture must
  assert an arc-less mid-rally shot lands on a rally-index/zone rule or unknown.
- Confidence rule: multiply the seed conf by `0.5` when `trust=="estimated"` and by `0.3` when
  `trust=="unverified_cue"`. Never emit conf > seed.
- Lane sizing: **Codex, 1 lane**, ~1 file + 1 test, CPU. Use `run-lane` skill template.
- Acceptance (S1): pytest `tests/racketsport/test_shots.py` green on ≥2 synthetic fixtures covering all
  8 classes + unknown + no-arc downgrade; `build_shots` runs on an existing run_dir without reading any
  Outdoor/Indoor label file (assert `policy.protected_eval_labels_read==false` in the artifact).
  NO agreement-with-owner metric yet (that needs labels → S1b).
- Kill criteria: if >30% of shots on a real owner clip land `unknown`, the rule order/thresholds are
  wrong → adjust the in-module threshold constants and re-run; do NOT add ML.
- **S1b (deferred, typed STOP: labeling):** learned classifier + the `≥0.85 agreement with
  owner-labeled shot types on held-out` gate (NORTH_STAR P6-1) ONLY after an owner labels shots and a
  `heldout_eval_ledger.md` row is pre-registered.

**S1c — Value-stats producer (P6-2) [Codex lane, CPU].**
- Objective: aggregate per-shot `shots.json` classes + outcomes into the RATE metrics S2/S3 compare against:
  `third_shot_drop_success`, `unforced_error_share`, `dink_rally_win_rate` (exact output keys), each with the
  rally_metrics trust envelope. **Dependency:** unforced-error / rally-outcome attribution needs rally-end-
  cause from the BALL/rally chain — absent → emit those rates as `insufficient_evidence` + typed STOP:
  needs-validation (never fabricate a rate). S2 `measured_by`/S5 overlays #1/#5 use THESE keys; gate S3/S5 on it.

**S2 — Reference-range library v0 (P6-3) [tiny Codex/Fable-edit lane, no GPU, then owner review].**
- Objective: versioned JSON contract of skill-band ranges with per-range provenance. This is the moat.
- File target (new): `docs/racketsport/reference_ranges_v0.json` + loader
  `threed/racketsport/reference_ranges.py::load_reference_ranges(path) -> dict` +
  `tests/racketsport/test_reference_ranges.py` (schema + provenance-present assertions).
- Schema (exact top-level): `{"schema_version":1, "library_version":"v0",
  "created_utc":..., "signed_off_by": null, "ranges":[ ... ]}`. `signed_off_by` STAYS null until a coach
  signs (gate). Each range object:
  `{"metric_id","shot_class"|null,"skill_band" (one of "3.0","3.5","4.0","4.5+"),
    "band" {"lo","hi","unit","direction" ("higher_better"|"lower_better"|"target_window")},
    "provenance" {"source_type" ("trade_benchmark"|"coach_review"|"our_user_data"),
                  "source_ref" (URL or citation string), "confidence" ("seed"|"reviewed"|"data_backed"),
                  "notes"},
    "measured_by" (exact S1/rally_metrics key this compares against)}`.
- v0 seed rows (VERIFIED-today sources; source_type=trade_benchmark, confidence=seed):
  - `third_shot_drop_success` by band: 3.0 lo0.40 hi0.50 · 3.5 lo0.70 hi0.80 · 4.0 lo0.85 hi0.90,
    unit=fraction, direction=higher_better. source_ref: thedinkpickleball.com "Third Shot Drop by Skill
    Levels 3.0 to 4.0". NOTE row: subtract 0.10–0.20 for live-play vs drill.
  - `unforced_error_share`: amateur ≈0.40 of points lost, direction=lower_better (NORTH_STAR II.4).
  - `dink_apex_over_net_m`: target_window lo0.10 hi0.40 (a "good dink" clears low, forces no attack).
    confidence=seed, provenance notes: derived, needs coach review.
  - `kitchen_arrival_after_return_s`: target_window (faster better); seed placeholder, mark
    confidence=seed + notes "needs owner/coach calibration".
  - `serve_depth_fraction` (fraction of court depth reached): higher_better, seed placeholder.
  Any range without a real source_ref MUST carry `confidence:"seed"` and a notes string saying so.
- Lane sizing: Fable tiny-edit OR 1 Codex lane (JSON + loader + test). CPU.
- Acceptance (S2): test asserts every range has non-empty `provenance.source_ref` OR
  `confidence=="seed"` with a notes string; loader rejects unknown `skill_band`. **Product gate
  (P6-3):** `signed_off_by` set by owner + coach review → that flips confidences to `reviewed`. Until
  then library is USABLE but UNSIGNED (comparator must surface "range unsigned" in trust).
- Kill criteria: if a seed range contradicts owner intuition on their own game → mark that row
  `confidence:seed` + notes, never silently ship as authoritative.

**S3 — Comparator engine (P6-4 stage 2) [Codex lane, CPU].**
- Objective: deterministic function mapping (shots.json + rally_metrics + reference_ranges) → typed
  `coaching_findings.json`. This is the ONLY component that decides pass/fail vs a range. The LLM never
  does comparison.
- File target (new): `threed/racketsport/coaching_comparator.py::build_findings(run_dir, ranges_path,
  skill_band) -> dict` + `tests/racketsport/test_coaching_comparator.py`. ARTIFACT_TYPE
  `"racketsport_coaching_findings"`, `schema_version=1`.
- **`skill_band` source:** read from the P0-9 profile registry (`threed/racketsport/profile_registry.py`,
  profile_registry.json, per-account); default `"3.5"` with trust downgraded + a "skill band assumed" note
  when absent. Raise typed STOP: needs-validation only if the owner requires a certified band.
- Finding schema (exact keys):
  `{"finding_id","metric_id","shot_class"|null,"severity" ("info"|"minor"|"major"),
    "verdict" ("below_range"|"in_range"|"above_range"|"insufficient_evidence"),
    "measured_value","band_lo","band_hi","unit","skill_band",
    "evidence": {"frame_range":[f0,f1],"artifact_paths":[...],"shot_ids":[...],"trust"},
    "range_provenance": {...copied from the range...},
    "how_we_measured": "<one plain sentence naming the artifact + formula>"}`.
- **Verdict table (deterministic, per `band.direction`):** `target_window` → `below_range` if measured<lo,
  `above_range` if measured>hi, else `in_range` (`band_width=hi-lo`; `nearest_band_edge`=violated bound).
  `higher_better` → `below_range` if <lo, `above_range` if >hi (better than target, still info), else
  `in_range` (`band_width=hi-lo`). `lower_better` → mirror. If only one bound present, use it as
  `nearest_band_edge`, `band_width`=that bound's own scale (fallback 1.0).
- Rules: severity = major if `abs(measured - nearest_band_edge)/band_width >= 0.5` AND trust=="ok"; minor if
  smaller gap OR trust=="estimated"; info if `in_range`. If supporting trust is `unverified_cue` OR
  coverage<0.8 OR arc absent → verdict=`insufficient_evidence`, severity=`info` (no coaching claim on weak
  evidence). Unsigned library range → append `"(reference range unsigned)"` to `how_we_measured`.
- Every finding MUST carry a `frame_range` and ≥1 `artifact_paths` entry — a finding with no evidence
  pointer is a bug; the test asserts this for all findings.
- Lane sizing: **Codex, 1 lane**, 1 file + 1 test. CPU.
- Acceptance (S3): test proves (a) every finding has non-empty evidence.frame_range +
  artifact_paths; (b) `insufficient_evidence` is forced when trust∈{estimated,unverified_cue} or
  coverage<0.8; (c) no finding references an Outdoor/Indoor label path.
- Kill criteria: if the comparator ever emits a numeric coaching claim (`below_range`/`above_range`)
  on `insufficient_evidence` inputs → hard bug, block S4.

**S4 — Format-locked LLM harness (P6-4 stage 3) [Codex lane + owner API key; small live-API budget].**
- Objective: turn findings → a locked coaching card. Claude sees ONLY the comparator's verdicts, never raw artifacts.
- File targets (new): `threed/racketsport/coaching_llm.py::generate_coaching_card(findings_dict,
  skill_band, *, model="claude-opus-4-8", n_corrections=3) -> dict` +
  `scripts/racketsport/audit_coaching_fabrication.py` + `tests/racketsport/test_coaching_llm.py`
  (test uses a STUBBED client — no live API in pytest).
- Model + call (claude-api skill 2026-07-07): `claude-opus-4-8`, $5 in / $25 out per 1M. Anthropic SDK
  `client.messages.create` with **structured outputs** `output_config={"format":{"type":"json_schema",
  "schema":CARD_SCHEMA}}` (no assistant prefill — 400s on opus-4-8; no `temperature`/`top_p` — 400).
  Adaptive thinking optional (`thinking={"type":"adaptive"}`); `output_config={"effort":"low"}` is fine.
- Prompt contract:
  - system: `"You are a precise, evidence-based pickleball coach. You may ONLY reference the numbers and
    verdicts in the supplied findings. You MUST NOT invent, estimate, or infer any numeric value not
    present. If findings are insufficient, say so."` (mirrors Talking Tennis's system role.)
  - user: the findings JSON (verdicts + measured values + bands + how_we_measured), the skill_band, and
    `"Return exactly N corrections."`.
- Response schema CARD_SCHEMA (json_schema, additionalProperties:false, all required):
  `{"headline_score": integer 0-10, "summary": string (<=280 chars),
    "corrections": array (length==n_corrections normally; ==real-finding-count when
    insufficient_evidence) of {"finding_id": string, "text": string,
      "cites_metric_id": string}, "drill": {"name": string, "why": string, "target_metric_id": string},
    "insufficient_evidence": boolean}`.
- Refusal/fallback behavior (exact): (a) if `stop_reason=="refusal"` → return
  `{"insufficient_evidence": true, ...}` and log; do NOT retry with looser prompt. (b) if any
  `corrections[].cites_metric_id` is NOT present in the input findings → REJECT the card (raise), retry
  once; on 2nd failure return the deterministic fallback card (comparator's top-3 major findings phrased
  by a template, no LLM). (c) if total findings < n_corrections, do NOT fabricate filler: return the
  template card with `insufficient_evidence=true` and `corrections` length == number of real findings
  (CARD_SCHEMA relaxes the length constraint here; each correction still carries a real `finding_id`/
  `cites_metric_id`). This cross-check is the fabrication firewall.
- Fabrication audit (`audit_coaching_fabrication.py`): input = dir of N cards + source findings; per card,
  assert every number/metric_id in `summary`+`corrections` traces to a finding value (regex numerics, exact
  match to a finding measured/band value or a whitelisted token). Output: `fabrication_count`/`total`/
  `pass=fabrication_count==0`. Gate sample = **300 outputs** (NORTH_STAR P6-4), bar = **0/300**.
- Cost estimate: per session ≈ 2–4k in + 300 out tokens ≈ **$0.028**; a 300-output audit ≈ **$8** one-time
  (Batch API 50% off → ~$4). Unit economics are a non-issue (engineering time is the scarce resource).
- Lane sizing: **Codex, 1 lane** for harness+audit+stubbed test (CPU). The live 300-output audit is a
  **separate owner-gated run** (needs owner API key + owner+4.0-reviewer time).
- Acceptance (S4): stubbed pytest proves citation cross-check rejects a fabricated card; the
  fabrication-audit script scores 0 on a hand-built fabricated fixture and 0 on a clean fixture. Product
  gate (P6-4): coach rubric ≥8/10 usefulness AND fabrication audit **0/300** — owner-run only.
- Kill criteria: any card that passes schema but cites a metric_id not in findings and is NOT caught →
  the cross-check is broken; block release.

**S5 — Visual feedback overlays (P6-5) [Codex/web lane, browser-verified].**
- Objective: ≥5 finding types render in `web/replay` with jump-to-moment, each reusing a LANDED layer.
- File targets: extend `web/replay` viewer manifest builder + `verify_process_video_viewer.py` to assert
  coaching-overlay honesty (no overlay on `insufficient_evidence` findings; every overlay has a
  frame_range). Feed = `coaching_findings.json`.
- ≥5 finding types v0 → reused layer:
  1. **third-shot-drop landing vs kitchen target** → court-map layer (target zone box vs `landing_world_xy`).
  2. **dink apex over net** → net_plane→arc layer (arc apex marker + net line).
  3. **kitchen-arrival timing** → court-map + moment-jump (player track ribbon to kitchen line).
  4. **contact height / smash contact point** → mesh + paddle `face_normal_world` marker at `contact_locked` frame.
  5. **unforced-error moment** → moment-jump to the shot's frame_range + trust badge.
  6. (bonus) **serve depth** → court-map depth ribbon.
- Every overlay MUST show the trust badge + `how_we_measured` on hover (reuse trust-band badge).
  `insufficient_evidence` findings render as GREY "not enough evidence", never a coaching claim.
- Lane sizing: 1 web Codex lane; browser-verify per README (`npm test -- --run --dir web/replay`,
  `npm run typecheck --prefix web/replay`, then the python viewer verifier).
- Acceptance (P6-5 gate): browser-verified demo of ≥5 finding types on an OWNER game (not eval labels);
  `verify_process_video_viewer.py` extended assertion green.
- Kill criteria: any overlay drawn for an `insufficient_evidence` finding → honesty violation, block.

### 3. Decision trees (if X → do exactly Y)
- Shot rules leave >30% `unknown` on a real clip → adjust in-module S1 threshold constants (apex/height
  cutoffs), re-run; still >30% after two passes → typed STOP: advice (is arc coverage the cause? check status=="ran").
- Owner asks for a learned shot classifier → require a pre-registered `heldout_eval_ledger.md` row +
  owner shot labels first → typed STOP: labeling. Never touch Outdoor/Indoor labels to build it.
- Comparator wants to compare a metric with no matching reference range → emit
  `verdict=insufficient_evidence`, do NOT guess a band → and file the missing range as an S2 backlog row.
- LLM returns a `cites_metric_id` absent from findings → reject+retry once → then deterministic template
  fallback card. Never ship the fabricated card.
- Fabrication audit finds ≥1/300 → BLOCK P6-4 sign-off; inspect which finding leaked; tighten the
  citation cross-check or the prompt; re-run full 300. Bar is exactly 0.
- Reference library still `signed_off_by:null` at coach-review time → coach unavailable is a typed STOP:
  needs-validation. Ship UNSIGNED only for internal dogfood, comparator surfaces "(range unsigned)".
- A metric's trust is `unverified_cue` (wrist-cue-only contact) → comparator forces
  `insufficient_evidence`; overlay greys out. Never coach on it.
- Owner wants a single-number stat (e.g. "your speed is 27 mph") → REFUSE single-number over-precision
  (pb.vision cautionary tale, NORTH_STAR P6-2); report a band + trust. This is a standing ruling.
- Any pillar work would read an Outdoor/Indoor CVAT label file → STOP immediately, do not proceed;
  every artifact must keep `protected_eval_labels_read=false`.

### 4. DO-NOT list (each with the one-line reason)
- DO NOT let the LLM see raw artifacts or compute comparisons — SportsGPT ablation: raw numbers into the
  LLM tank accuracy 3.9→2.85; removing grounding tanks feasibility 3.9→1.65.
- DO NOT allow free-form narration or "insights" beyond the locked schema — fabrication risk; kill-listed.
- DO NOT emit single-number stats without a band + trust — pb.vision's 60-vs-27mph blunder.
- DO NOT build the learned shot classifier before owner labels + a heldout ledger row exist — VERIFIED=0
  rule; no eval-label leakage.
- DO NOT coach on `unverified_cue` / coverage<0.8 / arc-absent shots — force `insufficient_evidence`.
- DO NOT ship a reference range as authoritative without provenance — the library is the moat only if
  every range is traceable; unsigned ranges must self-label.
- DO NOT claim officiating/line-calls — `ball_inout_uncertainty` is an honest band (0.94–2.26 m), not a
  call; US11615540B2 patent watch is a commercial-launch concern (internal use fine).
- DO NOT use assistant prefill or `temperature`/`top_p` with `claude-opus-4-8` — both 400 (claude-api).
- DO NOT run the live 300-output fabrication audit inside pytest — stub the client; live audit is a
  separate owner-gated run with an API key.

### 5. External bets verified today (claim → verdict → source/URL → license → code)
- **Talking Tennis** (rubric + no-fabrication method) → REAL, method confirmed. arxiv 2510.03921 (Oct
  2025). Constrained output = overall score + concise summary + **exactly three corrections** + "no
  fabrication of numerical values"; system role "precise, evidence-based tennis coach"; CNN-LSTM biomech
  features. License: arXiv preprint, no code repo (reproducible from paper). The "100% no-fab / 317 outputs
  / coaches 8.4–8.9" figure is NORTH_STAR-attested (couldn't re-extract from binary PDF) — the METHOD is
  what we copy.
- **SportsGPT ablation** → REAL. arxiv 2512.14121 (KISMAM + SportsRAG on Qwy3). Ablations: grounding
  improves diagnostic accuracy + feasibility vs a general LLM. Exact 3.9→2.85 / 3.9→1.65 deltas NORTH_STAR-
  attested (directionally confirmed by abstract). arXiv preprint; code not confirmed public.
- **CoachMe** → REAL + CODE. arxiv 2509.11698 (ACL 2025), github.com/MotionXperts/MotionExpert. Reference-
  based motion-diff coaching; beats GPT-4o +31.6% (skating)/+58.3% (boxing). Reinforces our reference-range
  design. License: check repo before reuse — we borrow the DESIGN, not weights.
- **BioCoach** → REAL (benchmark), no code. arxiv 2603.26938; biomechanics-grounded VLM coaching, quant ROM
  + phase-aware cues, beats Stream-VLM (METEOR +262.8%). Evidence for biomech phrasing, not a dependency.
- **Claude API for the coach** → CURRENT. `claude-opus-4-8` default, **$5 in / $25 out per 1M**, 1M ctx,
  128K out (claude-api, cached 2026-06-24). Structured outputs via `output_config.format` json_schema;
  adaptive thinking; Batch API 50% off. No prefill, no sampling params (400). Per-session ≈ $0.03.
- **3rd-shot-drop seeds** → REAL. thedinkpickleball.com: 3.0 ≈40–50% (drill), 3.5 70–80%, 4.0 85–90%; live
  −10–20%. S2 v0 seeds (trade_benchmark, confidence=seed).
- **NEW since 2026-07-05:** no new paper with code changes this pillar. TennisExpert (2603.13397) +
  "Quantifying Player Skill in Table Tennis" (2603.25736) are video-understanding/skill-scoring, not
  grounded-coaching stacks with released code — no action; our 3-stage design stands.

### 6. Acceptance gates + owner dependencies (exact metric keys; owner-only items)
- **P6-1 gate:** `≥0.85 agreement with owner-labeled shot types on held-out; every classification carries
  confidence`. Metric to compute: per-shot class agreement fraction vs owner labels. OWNER MUST PROVIDE:
  shot-type labels on their own clips + a pre-registered `heldout_eval_ledger.md` row. Until then only
  the rule-based S1 (with confidence) ships; the 0.85 number is NOT claimable.
- **P6-2 gate:** every stat has a trust band + a "how we measured" popover; rally-metrics facts JSON
  feeds it. Enforced by S3 (`how_we_measured` + `evidence.trust` required on every finding). Value stats
  first: unforced errors, third-shot success, dink-rally win rate. (Unforced-error / rally-outcome
  attribution needs rally-end-cause — a dependency on the BALL/rally chain, not this pillar.)
- **P6-3 gate:** coach sign-off on v1 ranges; every range carries provenance. Enforced by the schema
  (`provenance.source_ref`/`confidence`). OWNER + COACH MUST: sign `signed_off_by` (flips seed→reviewed).
- **P6-4 gate (the product gate — this is what "VERIFIED" would require):** coach rubric **≥8/10**
  usefulness AND fabrication audit **0/300**; every claim traces to a trust-banded artifact. OWNER MUST:
  provide the API key for the live run, and be one of the ≥2 reviewers (owner + ≥1 rated 4.0+ player),
  scoring on the Talking-Tennis rubric template. This gate is 100% owner-gated; the harness + audit
  script are ours to build, the sign-off is theirs.
- **P6-5 gate:** browser-verified demo of ≥5 finding types on an owner game;
  `verify_process_video_viewer.py` extended to assert coaching-overlay honesty. OWNER MUST: provide a
  game clip + eyeball the demo.
- **P6-6 (session report):** owner dogfood approval on their own games. Player identity is partially wired —
  `threed/racketsport/profile_registry.py` (profile_registry.json, per-account, source-trace validation);
  verify it exposes stable per-account player identity before STOPping, typed STOP: decision only if a
  specific capability is confirmed missing.
- **Standing owner-only inputs:** shot labels (P6-1 learned), coach range sign-off (P6-3), the P6-4
  rubric review + API key, dogfood approval (P6-6). Everything else (S1–S5 code) is buildable by Codex
  lanes without owner input, but ships UNVERIFIED until these gates pass.

## PILLAR: SPEED + QA + PRODUCTION — <=2x duration SLA, auto-QA, durable service, consent/privacy

> Author: deep-design agent, 2026-07-07. Audience: the successor manager. Every step is pre-ruled; where
> a real blocker exists it is marked **typed STOP: <bucket>**. VERIFIED=0 across this pillar today — do not
> let any step's self-report promote a capability without a documented gate on real labels.

### 0. Final ruling — the stack in one paragraph (what we decided and why)
We RULED the pillar's SLA as **<=2x owner-video duration** (a 12-min game -> <=24min), because the four
booked speed levers were *measured* to floor at ~6-8 min/clip (BODY = 96-99% of E2E), so <=1x is an
**un-booked STRETCH** (P5-7), not a promise. We RULED the remaining P5-1 levers land largest-first as
flat-array/mmap handoff + gates-from-arrays + dispatch auto-clean (chunked binary transport was measured
REGRESSION, killed). We RULED TensorRT scope to **WASB + YOLO26 only via Ultralytics `half=True`** (detectors
<3% of wall; never SAM-3D ViT/MHR). We RULED auto-QA = **e-process sequential-hypothesis-testing** (arXiv
2602.12983) around trackers + Phase-F residuals + trust bands, implemented from scratch (paper ships NO code).
We RULED the durable service = **Render Persistent Disk (replace /tmp) + Key Value queue + Background Worker
(replace in-process BackgroundTasks/JobStore) + token auth + SQLite per-user library** — sized for owner+friends,
not enterprise. We RULED consent gates only the PERSISTENT biometric storage (ReID galleries / frozen shape betas) of non-owner
people, NOT processing itself — third-party/harvested footage processing is already owner-approved (P0-1b, PART 0
[x]). Default until the consent flow (P7-4b) lands: **session-only, non-persistent tracking** for every non-owner
person; delete-cascade designed into P0-9 now. Cost is **fully-loaded** (GPU +
storage + LLM + review labor + idle), never the BODY-only $0.117/clip slice.

### 1. Current measured state (numbers + evidence paths only)
- E2E Wolverine **2141s -> 1144s (1.87x)**, zero quality change; BODY stage 2106s -> 1134s (~99% of E2E).
  Evidence: `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md`.
- Booked floor after P5-1 lands: **BODY ~300-400s -> E2E ~6-8 min/clip** (same report, "measured ceiling").
- Handoff subprocess->orchestrator **~489s** for ~400MB pickle floats; payload assembly **~335s** (gates consume
  it); model load + compile warmup **~67s/dispatch** (23.3 load + 42.3 warmup). Same report, "honest misses".
- S4 chunk/binary transport = REGRESSION (1057.4->1300.7s; handoff 376->489s); reverted to pickle. Same report.
- Quality proof harness live: `body_full_clip_gate` (artifact `racketsport_body_full_clip_gate`), foot-slide key
  **`max_foot_lock_slide_m`**, gate `foot_slide_max_m` threshold **`DEFAULT_MAX_FOOT_SLIDE_M = 0.03` (30mm)**,
  0 root-motion jumps. Code: `threed/racketsport/body_full_clip_gate.py`, `body_grounding_quality.py:12,24`.
- Version-stamp/sync system LANDED (wave-3 `w3_codesync`, live-VM proof PENDING): `remote_body_dispatch.py`
  `build_version_stamp`/`verify_version_stamp_file`/`sync_remote_checkout_to_local_head`; CLI `--sync-remote-code`,
  `--verify-version-stamp`, `--allow-dirty`; critical-file set via AST import closure `_remote_runtime_critical_files`.
  Dispatch is **one-shot per clip** under a flock **shared-slot lease** (`scripts/gpu-eval-run.sh`), NO queue.
- **How to run E2E + where the number lives:** the full pipeline runs via `scripts/gpu-eval-run.sh` (flock lease +
  `--verify-version-stamp`) wrapping `scripts/racketsport/process_video.py`; E2E wall + per-stage timing land in the
  `PIPELINE_SUMMARY` block (grep `pipeline_summary` in `process_video.py`/`threed/racketsport/pipeline_cli.py`) and,
  for the BODY handoff/prep/warmup breakdown, in per-clip `remote_body_dispatch_timing.json` /
  `body_stage_phase_timing.json`. Reproduce the 1144s baseline from `wolverine_speed*.log` + `timing_facts.json`;
  copy the exact `process_video.py` arg list from `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md` before quoting.
- Server = single-process FastAPI `create_app` (`server/render_app.py`); `JobStore` writes job JSON to **/tmp**
  (`DEFAULT_UPLOAD_ROOT`); jobs run via **in-process `BackgroundTasks`** (`_execute_job`); **ZERO client auth**.
  `render.yaml` sets `PICKLEBALL_UPLOAD_ROOT=/tmp/pickleball_uploads` (free-plan docker web service); the code
  default is `/tmp/pickleball_render_uploads` (env `PICKLEBALL_UPLOAD_ROOT`, `server/render_app.py:39`).
- Orientation is a **hardcoded `"orientation": "landscape"` stub** in 3 places:
  `scripts/racketsport/process_video.py:4205`, `scripts/racketsport/process_video.py:4283`,
  `threed/racketsport/court_corner_review.py:189`. `ball_capture_protocol.py:74` already checks
  `orientation != "landscape"` -> emits `orientation_not_landscape`, but it never sees a real value.
- Trust bands live: `threed/racketsport/trust_band.py`, `TRUST_BADGES = ("verified","preview","low_confidence")`,
  `derive_{body,court,ball,track,paddle}_trust_band`. BODY never claims `verified` until its gate passes.
- Cost metering: NOT built as a fully-loaded $/clip line (only BODY-only $0.117 slice exists).

### 2. The exact build plan (numbered; each = objective / files / recipe / lane / acceptance / kill)

**Land Steps 1-5 in the numbered order exactly** (precondition first, then ascending risk). This is deliberately
NOT strictly largest-saving-first: Step 4 (mmap, ~430s) is the biggest but highest-risk lever and lands only AFTER
the cheap low-risk wins (Steps 2-3) are banked and verified. The numbered order governs, not the saving-size heuristic.

**Step 1 — Dispatch-dir auto-clean (do FIRST, blocks unattended runs).**
- Objective: remove ENOSPC recurrence (A100 disk hit 100%, killed run #7). No speed gain; a REQUIRED precondition.
- Files: `scripts/racketsport/remote_body_dispatch.py` — insert cleanup after the sync-back call
  (`synced_outputs = _sync_body_outputs(...)` at ~L1163) near the end of `dispatch_body_stage` (which closes ~L1222);
  `_sync_body_outputs` itself is defined at L1560.
- Recipe: after successful sync-back + version-verify, `rm -rf` the remote `remote_run_dir`; guard behind a
  `--keep-remote-run-dir` escape hatch for debugging. Emit freed-bytes into `remote_body_dispatch_timing.json`.
- Lane: **Codex** (local edit + unit test). Acceptance: remote run dir absent after a dispatch; disk stays <80%.
- Kill: none — safety fix.

**Step 2 — Gates-from-arrays (recover ~335s, biggest cheap win).**
- Objective: feed the quality gates from the in-memory float arrays instead of the assembled ~1GB smpl monolith,
  so payload assembly stops running in slim mode.
- Files: `threed/racketsport/body_grounding_quality.py`, `body_full_clip_gate.py`, the assembly boundary in the
  BODY runner (`scripts/racketsport/run_sam3dbody_batch.py`) + orchestrator gate call site.
- Correctness proof: **bit-identical** — gate numbers must be unchanged. Assert `max_foot_lock_slide_m` equal to
  >=6 decimals vs pre-change; `body_full_clip_gate`==TRUE; 0 root jumps.
- Lane: **Codex** local + **1 GPU verify run** (synthetic byte-identity misses finalizer interactions — verify on a
  REAL A100 run per known-trap #8). Acceptance: `max_foot_lock_slide_m` identical to 6 dp; assembly time ~0 in slim.
- Kill: if arrays can't reproduce the gate bit-identically, STOP the lever (do not ship a changed metric).

**Step 3 — Restore S3 monolithic subprocess output path (recover ~90-120s).**
- Objective: revert the S4 chunk-streaming residue back to the monolithic writer to re-establish a clean S3
  correctness baseline. **Relationship to Step 4:** Step 3 restores the monolithic pickle-shaped output; Step 4 then
  swaps THAT format for a flat mmap array of the SAME logical shape and supersedes it. Land+verify Step 3 first so
  Step 4 regression-checks against a clean S3 baseline, not against S4 residue.
- Files: `scripts/racketsport/run_sam3dbody_batch.py` (subprocess output), handoff read in `remote_body_dispatch.py`.
- Recipe: single monolithic subprocess output artifact (S3 shape), not chunk-streamed. Correctness: **bit-identical**.
- Lane: **Codex + 1 GPU verify**. Acceptance: handoff+prep back to S3 profile; foot-slide 6-dp identical.
- Kill: chunked streaming is KILL-LISTED (Step in DO-NOT). Do not re-introduce it.

**Step 4 — Flat-array / mmap subprocess->orchestrator handoff (target 489s -> <60s, ~430s).**
- Objective: the single biggest lever. Replace the ~400MB pickle round-trip with **one large flat contiguous
  array written to a single mmap file** (NOT chunked — chunking was S4's root cause).
- Files: `scripts/racketsport/run_sam3dbody_batch.py` (writer), `remote_body_dispatch.py` (reader/`_collect_body_inputs`).
- Recipe: pack all per-frame float tensors into one preallocated `numpy.memmap` (or a single `np.save` flat array +
  offset index), mmap-read in the orchestrator; no per-object pickle, no streaming loop.
- Correctness proof: **bit-identical** — float round-trip must be bitwise equal. Assert arrays `np.array_equal`
  pre/post; foot-slide 6-dp; gate TRUE; 0 root jumps; **six-run variance report** (a single run cannot prove it).
- Lane: **Codex build + dedicated GPU verify on its OWN spot GPU** (file-disjoint; worktree MANDATORY for VM/rsync).
  Before trusting ANY VM number: `remote_body_dispatch.py --verify-version-stamp` must pass (remote md5==HEAD).
- Kill: if measured slower than S3 pickle baseline on a real run (as S4 was), REVERT immediately and log the miss;
  do not tune-in-place past one regression.

**Step 5 — Compiled-graph disk cache (recover ~42s/clip, near-zero risk; bridges to P5-7).**
- Objective: cache the `torch.compile` graph keyed by **model version + input shape** so warmup (42.3s) is paid once.
- Files: the compile-warmup path in `scripts/racketsport/run_sam3dbody_batch.py` (consumes
  `compile_warmup_buckets`/`compile_warmup_passes` from the request payload), driven by the
  `--sam3d-compile-warmup-buckets` CLI flag in `remote_body_dispatch.py`/`process_video.py`.
- Recipe: persist the compiled artifact to VM disk under a key `{model_md5}_{input_size}_{bucket_sizes}`; load on hit.
- Correctness: **score-identical** = exact 6-dp equality of `max_foot_lock_slide_m` vs the un-cached run
  (bit-identity expected; a compiled graph must not move the metric).
- Lane: **Codex + GPU verify**. Acceptance: warmup ~0 on a 2nd clip same shape; metrics identical.
- Kill: if cached graph ever changes a metric, invalidate the cache key and STOP.

**P5-1 GATE (copy exactly):** P5-1 passes iff Wolverine E2E **<=400s**. A **400-500s** result WITH BODY floored at
its measured 300-400s AND every other stage <50s is a **CONDITIONAL pass** — continue downstream but log a
needs-decision note that 400 may paraphrase the real ~450s floor. **>500s is a hard STOP** (needs-decision). Also
required: `max_foot_lock_slide_m` **bit-identical** (6 dp) vs pre-change; `body_full_clip_gate`==TRUE; 0 root-motion
jumps; the six-run variance report. Outdoor **<=2x its video duration** needs a `heldout_eval_ledger.md` row (see 6).
**Six-run pass condition:** across ALL six runs `max_foot_lock_slide_m` identical to 6 dp in every run AND
`body_full_clip_gate`==TRUE in every run AND 0 root jumps in every run AND E2E wall within +/-5% of the six-run
mean; any single run breaking 6-dp bit-identity FAILS the gate.

**Post-P5-1 dispatch order (which lane first / concurrency):** land **P5-5b + P7-4b first** (pre-flight + PART-0
consent-persistence gate — cheap, no GPU); then run **P5-3 and P5-4 as parallel file-disjoint Codex lanes** (P5-3
needs 1 GPU, P5-4 is CPU-only) within the fleet cap; **P5-6 after P5-4** (needs the cost/trust wiring); **P7-1 only
after the needs-purchase-approval STOP clears**; **P5-7 last** as the stretch. Concurrency: at most one clip per
GPU; a P5-3 GPU verify + a P5-1 Step-4 verify may run on separate spot GPUs (<=4 concurrent, <=$5/hr each, else
typed STOP: needs-purchase-approval).

**P5-3 — Detector TensorRT (WASB + YOLO26 ONLY).**
- Objective: ball+track stage wall -50% at identical F1. Files: WASB + YOLO26 detector wrappers under
  `threed/racketsport/` (grep the ball/track detector call sites). Recipe: official Ultralytics export,
  `model.export(format="engine", half=True)` (FP16); batch rally-window frames.
- Correctness: **score-identical** := `|dF1| <= 0.005` absolute AND `|dIdF1| <= 0.01` on internal-val
  (Wolverine/Burlington, never held-out); FP16 bitwise WILL differ, bit-identity NOT required. Exceeding either
  bound -> keep the FP32 engine (still batched).
- Lane: **Codex + GPU**. Acceptance: ball+track stage wall **-50%** at score-identical internal-val F1.
- Kill: NEVER convert SAM-3D ViT/MHR (highest risk, <3% wall). If FP16 drops F1 below tolerance, keep FP32 engine.

**P5-4 — Fully-loaded cost metering.**
- Objective: one $/clip figure = GPU-seconds x spot-price + storage $/GB-mo (retained video+profiles) + per-session
  LLM coaching cost (P6-4) + amortized human-review labor (from P5-6 flags) + idle/orchestration VM time.
- Files: emit into `PIPELINE_SUMMARY` (grep `pipeline_summary` in `scripts/racketsport/process_video.py`,
  `threed/racketsport/pipeline_cli.py`); `threed/racketsport/stage_runtime_budget.py` already exists — extend it. Recipe: sum, tag spot price from fleet
  ledger. Add spot-preemption auto-resume (VM restart mid-verify bit us; NEW-IP handling per `configs/ssh/a100_known_hosts`).
  **Degraded mode:** until P6-4 (LLM) and P5-6 (review-flag rate) land, emit their two terms as explicit
  `0`-with-TODO placeholders (GPU-seconds x spot-price + storage are computable now); tag each term's source. Do NOT
  block P5-4 on P6-4/P5-6, and do NOT let a partial cost line feed P7-3 pricing.
- Lane: **Codex**. Acceptance: cost line on EVERY run; simulated preemption recovers without human help.
- Kill: do NOT feed P7-3 pricing the BODY-only $0.117 slice.

**P5-5b — Pre-flight sanity gate (burns ZERO GPU on bad clips).**
- Objective: reject bad clips before any GPU stage. Files: pipeline entry in `process_video.py` (before dispatch,
  near `_clip_duration_seconds`/L1578); REPLACE the 3 hardcoded `"orientation":"landscape"` stubs (L4205, L4283,
  `court_corner_review.py:189`). Emit `preflight_gate.json` (`status: passed|rejected`, `reason`).
- Exact checks: (1) **ffprobe integrity** — `ffprobe -v error -show_entries format=duration
  -show_entries stream=codec_type,width,height,r_frame_rate`; reject if nonzero return, duration<=0, or no video
  stream. (2) **REAL orientation** — read display-matrix `rotate`/`side_data_list` rotation + compare effective
  W vs H; if effective H>W -> portrait -> reject; write the true value so `ball_capture_protocol.py:74` fires
  `orientation_not_landscape` for real. (3) **court-presence** — sample ~8 frames (1 per 2s, capped), run
  `threed/racketsport/court_detector_v2.py::detect_court_v2_from_frame`; require >=1 frame court-line conf >= 0.5,
  else reject "no court detected". (4) **sport** — same frames, run the person detector
  `threed/racketsport/mobile_person_yolo_replay.py::run_replay_yolo_candidate` (the candidate's `conf` field);
  require >=1 person box conf >= 0.25. Checks 3-4 = a few CPU/tiny-detector frames, not the GPU BODY stage.
- Lane: **Codex** (deterministic, no GPU). Acceptance: 4 injected bad clips (portrait, wrong-sport, corrupt,
  no-court) each rejected pre-GPU with the correct `reason` in `preflight_gate.json`.
- Kill: keep it minimal — do NOT build a full sport classifier here (over-scope).

**P5-6 — Per-clip auto-QA via sequential hypothesis testing.**
- Objective: catch a bad output BEFORE the user sees it; flag for reprocess/human review, never silently show.
- Method: **e-process / e-value sequential test** (arXiv 2602.12983); anytime-valid, false-alarm bounded by
  Ville's inequality at level **alpha=0.01** regardless of stopping. **Paper ships NO code — implement from
  scratch** (a small accumulator; not a dependency). Files: new `threed/racketsport/qa_sequential_test.py`;
  wire into `PIPELINE_SUMMARY`.
- **e-process spec (build exactly — do NOT hand-wave "grows when residual exceeds null"):** per stage, null
  `H0` = the calibrated residual distribution from PASSING clips (via `scripts/racketsport/calibrate_confidence_bands.py`).
  Each frame `t` yields a betting factor `e_t` = a test-supermartingale increment: the likelihood ratio
  `p1(r_t)/p0(r_t)` of the observed residual `r_t` under a fixed alternative `p1` (a sub-exponential / mean-shifted
  form of `H0`) vs the null `p0`, built so `E_H0[e_t | past] <= 1`. Wealth process `E_t = prod_{i<=t} e_i`,
  `E_0 = 1`. Flag when `E_t >= 1/alpha` (= 100 at alpha=0.01) — Ville's inequality then bounds `P_H0(ever flag) <=
  alpha`. Use a **mixture/GRO** betting weight (mix over a grid of shift sizes) so power holds without knowing the
  failure magnitude. Reference sketch: accumulate `log E_t += log(mean_k p1_k(r_t)) - log p0(r_t)`; fire when
  `log E_t >= log(1/alpha)`.
- Per-stage statistic choices (the e-value grows when the observed residual exceeds the null):
  - Ball: per-frame physics residual = distance from `ball_physics_fill.py` prediction + detection-confidence drop.
  - Body: streaming `max_foot_lock_slide_m` + Phase-F consistency residuals (foot penetration/float, root-jump count).
  - Track: ID-switch-rate / IdF1-proxy stream (`derive_track_trust_band` inputs).
- False-alarm budget: **alpha = 0.01 per stage per clip** (Ville guarantees P(false flag) <= alpha). Combine SHT
  flag with Phase-F residuals + trust-band coverage (`scripts/racketsport/calibrate_confidence_bands.py`). Output keys:
  `qa_sht_evalue` (per stage), `qa_flag` (bool), `qa_status: passed|qa_failed`.
- Lane: **Codex build**; validate on injected-failure clips (synthetic corrupt tracks). Acceptance: injected-failure
  clips caught at **TPR >= 0.90** on the injected-failure suite (>= 5 clips per failure mode: corrupt track,
  foot-slide blowout, ID-switch storm) with false alarms **<= alpha=0.01**; below 0.90 the residual statistic is
  wrong — fix the residual, do NOT lower alpha. Ties into P7-5 ladder.
- Kill: do NOT retrain any model for QA (the whole point is no-retraining). If e-values never fire on injected
  failures, the statistic is wrong — fix the residual, not the threshold.

**P5-7 — BODY-runtime redesign (the un-booked <=1x STRETCH).** See decision tree in 3. Cheap-first = Step 5
compiled-graph cache (already in P5-1). Bigger lever = **batched multi-clip inference** (pack crops from N clips
into one forward batch). A persistent warm worker is KILL-LISTED as a naive daemon (see DO-NOT) — reconsider a
pooled-with-batching worker ONLY if it processes clips back-to-back with <1h idle between them (never idles a GPU
>1h, per fleet doctrine); if clips do not arrive back-to-back the warm-daemon KILL applies and you fall back to the
compiled-graph disk cache. Gate: BODY wall <= video duration on a full owner game, gates green.
Kill (roadmap's own): if two candidates fail to beat the booked floor >=2x, **accept <=2x as the product SLA and
close the stretch** (owner already pre-ruled <=2x is the real target — this is NOT a STOP).

**P7-1 — Minimal durable service (owner+friends scale).**
- (a) Object storage: **Render Persistent Disk** ($0.25/GB-mo) mounted on the web service; point
  `PICKLEBALL_UPLOAD_ROOT`/`DEFAULT_UPLOAD_ROOT` at the mount, replacing `/tmp`. (S3/R2 only if it ever multi-instances.)
- (b) Durable queue: **Render Key Value** (managed Redis; free 25MB tier, $10/mo paid) + a **Render Background
  Worker** ($7/mo) that pulls jobs; replaces in-process `BackgroundTasks` + `_execute_job`. Survives web restart/preemption.
- (c) Job store: replace `JobStore` JSON-in-/tmp with **SQLite on the persistent disk** (ACID, single-writer fine at
  this scale) — also backs the per-user library. Files: `server/render_app.py` (`JobStore` L54-113).
- (d) Auth: **bearer token / signed key** on `/api/jobs*` + a `users` table in SQLite. Not enterprise OAuth.
- (e) Per-user library: SQLite `clips` table keyed by `user_id`, FK job + artifacts; design P0-9 delete-cascade in now.
- Lane: **Codex** (server, no GPU). Acceptance: a job survives a worker restart; unauth request to `/api/jobs` rejected.
- **typed STOP: needs-purchase-approval** before enabling any paid Render tier (Key Value $10, Worker $7, Disk).

**P7-4b — Consent / retention flow v0.**
- H0 wizard asks: (1) "Is everyone visible you, or have they consented?" -> **gates PERSISTING non-owner biometric
  profiles until YES** (processing still runs session-only; PART 0 persistence gate, not a processing block).
  (2) biometric disclosure — ReID galleries + frozen shape betas (H4) are biometric-category
  under BIPA/CCPA/GDPR. (3) retention choice: keep profile embeddings vs session-only-discard.
- Default = **session-only**: ReID embeddings + shape betas discarded post-processing unless affirmatively kept;
  a "second-person detected" consent prompt at capture. Purge: delete-clip cascades to derived artifacts (tracks,
  ReID, betas, meshes, world JSON); delete-account cascades all clips; cross-account shared ReID gallery =
  reference-counted (purge only when last holder deletes). P0-9 schema: every artifact traces to source clip_id+profile_id.
- Lane: **Codex** (schema + server), but the go/no-go + legal basis are owner-only (see 6).

### 3. Decision trees (if X -> do exactly Y)
- P5-1 lever measured SLOWER than baseline on a real GPU run -> **REVERT that lever, log the miss**, do not tune past
  one regression (S4 lesson). Continue with the next lever.
- VM speed/quality number arrives but `--verify-version-stamp` FAILS (md5!=HEAD) -> **discard the number**, run
  `--sync-remote-code`, re-run; fail-loud on drift (fleet1 sat 16 commits stale). Never trust an unverified VM metric.
- P5-1 lands but Wolverine E2E > 400s -> if BODY floor is still 300-400s and everything else <50s, the gate is met
  in spirit; if E2E > 500s, **typed STOP: needs-decision** (is 400s still the right bar or a paraphrase artifact?).
- P5-7: post-P5-1 BODY <= video duration on a full owner game -> <=1x WON, close stretch. Else compiled-cache +
  batched-multi-clip (N>=2 owner clips) amortize BODY <= video duration -> adopt. Else two candidates fail to beat
  floor >=2x -> **accept <=2x as SLA, close stretch** (pre-ruled, not a STOP).
- A clip flags `qa_failed` but the owner insists it looks fine -> the statistic or its null is miscalibrated;
  **typed STOP: needs-validation** (a self-reported PASS contradicting numbers) — do not silently lower alpha.
- Owner uploads NON-OWNER footage before P7-4b consent lands -> **process it** (harvest is owner-approved, P0-1b)
  but run **session-only, non-persistent** tracking; do NOT persist ReID galleries / shape betas. Only if the owner
  asks to PERSIST a non-owner biometric profile -> **typed STOP: needs-decision** (PART 0 persistence gate).
- Any GPU sits idle >1h between lanes -> tear it down (fleet doctrine); >$5/hr or a 5th concurrent GPU ->
  **typed STOP: needs-purchase-approval**.
- TensorRT FP16 drops ball F1 below internal-val tolerance -> keep the FP32 engine (still batched); do not ship a
  score regression to chase wall time.
- Pricing decision needs the fully-loaded number but P5-4 not built -> **do NOT price on BODY-only $0.117**;
  block P7-3 until P5-4 emits the cost line.

### 4. DO-NOT list (each with its one-line reason/evidence)
- Do NOT convert SAM-3D ViT/MHR to TensorRT — highest risk, detectors <3% of wall (P5-3 scope; doesn't touch the floor).
- Do NOT re-attempt chunked/streaming binary transport — S4 measured REGRESSION 1057->1300s (FINAL_REPORT miss #1). Use flat arrays/mmap.
- Do NOT build an always-on warm-worker daemon — KILL-LISTED (PART IV rule 5) + idle GPU spend violates fleet doctrine; only the compiled-graph disk cache is sanctioned.
- Do NOT trust any VM metric without `--verify-version-stamp` passing — fleet1 ran 16 commits stale (PART IV rule 11).
- Do NOT price on BODY-only $0.117/clip — must be fully-loaded (P5-4 ruling); it omits storage/LLM/labor/idle.
- Do NOT PERSIST any non-owner person's biometric profile (ReID gallery / shape betas) before P7-4b consent lands
  — PART 0 gates biometric PERSISTENCE, not processing (harvest is owner-approved, P0-1b); interim default is
  session-only, non-persistent tracking.
- Do NOT claim any SLA or <=1x VERIFIED without a gate on real labels — VERIFIED=0 (PART IV rule 3).
- Do NOT parallelize eval-clip STAGES — KILL-LISTED; the parallelism unit is one clip per GPU (fleet doctrine).
- Do NOT accept a single-run "bit-identical" claim — six-run variance is the gate (foot-slide identical to 6 dp).
- Do NOT paraphrase a gate metric — the exact key is `max_foot_lock_slide_m`, not "slide p95" (PART IV rule 12; a wave-2 lane "passed" the wrong statistic).
- Do NOT depend on a QA library for P5-6 — arXiv 2602.12983 ships NO code; implement the small e-process yourself.

### 5. External bets verified today (claim -> verdict -> source -> license -> code)
- SHT failure detection (arXiv:2602.12983) -> **REAL, exactly as cited** ("Detecting Object Tracking Failure via
  Sequential Hypothesis Testing", e-process, Ville's inequality, anytime-valid, 2 models x 4 video benchmarks) ->
  https://arxiv.org/abs/2602.12983 -> **CC-BY-4.0** -> **NO code release found** (implement from scratch; low effort).
- Ultralytics TensorRT export -> **REAL, official, current** — `model.export(format="engine", half=True)` FP16
  (TRT11+ adds ModelOpt INT8/AutoCast) -> https://docs.ultralytics.com/integrations/tensorrt -> AGPL-3.0 (Ultralytics;
  internal-use OK per owner license stance) -> code available (official).
- Render durable primitives -> **REAL** — Persistent Disk **$0.25/GB-mo**, Background Worker from **$7/mo**, Key
  Value (Redis) free 25MB / paid from **$10/mo** -> https://render.com/pricing , https://render.com/docs/background-workers
  -> commercial SaaS -> managed (no code). Sized for owner+friends; a paid tier = purchase-approval STOP.
- GPU spot pricing sanity vs $5/hr cap -> **H100 SXM 80GB GCP spot ~$3.69/hr (us-central1, ~May 2026) — UNDER cap**;
  A100-80GB spot typically ~$1.5-2/hr -> https://cloud.google.com/spot-vms/pricing , spheron/synpix trackers ->
  n/a -> H100-first ruling holds; a >$5/hr VM = needs-purchase-approval STOP.
- NEW-since-2026-07-05 scan: nothing with code materially changes this pillar (Ultralytics TRT path stable; no new
  anytime-valid QA lib; Render pricing unchanged). No adoption.

### 6. Acceptance gates + owner dependencies
- **Benchmark clips (paths + held-out status — never confuse these):** Wolverine =
  `wolverine_mixed_0200_mid_steep_corner` and Burlington = **internal-val** (freely runnable, six-run OK); Outdoor =
  `outdoor_webcam_iynbd_1500_long_high_baseline` (`cvat_upload/03_outdoor_webcam_iynbd_1500_long_high_baseline_frames_0000_1150.mp4`)
  and Indoor = `indoor_doubles_fwuks_0500_long_mid_baseline` = **STRICT held-out** (one pre-committed scoring run
  only; `heldout_eval_ledger.md` row + owner STOP required). Hammering a held-out clip spends the one eval asset —
  iterate on internal-val only.
- P5-1: E2E Wolverine `<=400s`; `max_foot_lock_slide_m` bit-identical (>=6 dp) vs pre-change; `body_full_clip_gate`==TRUE;
  0 root-motion jumps; six-run variance report. Outdoor `<=2x video duration` **requires an owner-approved
  `runs/manager/heldout_eval_ledger.md` row + STOP-for-go** (held-out label protection).
- P5-3: ball+track stage wall `-50%` at **score-identical** internal-val ball F1 / track IdF1 (never held-out).
- P5-5b: 4 injected bad clips rejected pre-GPU; `preflight_gate.json.reason` correct each.
- P5-6: injected-failure **TPR >= 0.90** (>= 5 clips/failure mode); false alarms `<= alpha=0.01` (Ville). Keys `qa_sht_evalue`, `qa_flag`, `qa_status`.
- P5-7: BODY wall `<= video duration` on a full owner game, gates green — OR the pre-ruled close-at-<=2x.
- P7-1: job survives worker restart; unauth `/api/jobs` rejected.
- **Owner-only (typed STOPs):** (needs-decision) PART 0 consent go/no-go before ANY non-owner footage; retention-window
  values per artifact type; (needs-labeling) Outdoor held-out ledger row + go; (needs-purchase-approval) any paid Render
  tier, any >$5/hr GPU or a 5th concurrent GPU; (needs-advice, P7-4) counsel review of biometric legal basis
  (BIPA/CCPA/GDPR) — cannot be lane-decided.

---
*Maintained under NORTH_STAR Part IV rule 14 — update pillar sections as gates pass/evidence lands; never let this file claim more than the linked evidence shows.*
