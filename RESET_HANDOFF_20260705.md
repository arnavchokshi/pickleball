# RESET HANDOFF — 2026-07-05 (wind-down, GPU teardown, fresh-agent bootstrap)

**Purpose:** the owner ended all active work on 2026-07-05 to tear down the A100 spot VM and restart
later with a new GPU and a fresh manager agent. This document is the single entry point for that
restart: what is DONE (verified, with evidence paths), what is NOT done (ranked), every known
failure case, and the exact restart runbook. Anything marked ⏳PENDING was in flight at write time
and is reconciled in the final section before the closing commit.

## 0. Read order for a fresh agent
1. This file, fully.
2. `FABLE_OPERATING_MANUAL.md` — HOW to run work here (delegation, verification discipline).
3. `CAPABILITIES.md` — CANONICAL capability/truth table (wins over every other doc on conflict).
4. `PIPELINE_STATUS.md` — living status + §3 FAILURE CASES for placement/identity.
5. `BUILD_CHECKLIST.md` — coordination channel; read the last ~15 dated bullets.
6. `MASTER_PLAN.md` — the product goal; every lane must trace to it.
7. `runs/manager/heldout_eval_ledger.md` — pre-registration ledger; the held-out discipline record.

**Non-negotiable standing rules (owner-ratified):**
- 4 protected eval clips (Burlington / Wolverine / Outdoor / Indoor). Burlington+Wolverine internal
  scoring allowed; **Outdoor/Indoor labels NEVER touched** without a pre-registered ledger row.
  CVAT labels = SCORING ONLY, never construction (leakage).
- VERIFIED=0 today. VERIFIED requires the documented real-clip acceptance gate; scaffold/diagnostic
  artifacts never promote. Trust bands are the product's honesty mechanism — nothing silently fakes.
- Coordination between concurrent agents happens ONLY via BUILD_CHECKLIST.md bullets + commit
  messages. Explicit file ownership per lane; file-disjoint concurrent lanes.
- Honest kills are wins. Do not re-attempt anything in §5 without NEW evidence or owner data.
- All work is PRIVATE/internal-use (owner 2026-07-04): research-only/NC/GPL licenses usable
  internally; record license verbatim per source; access reality is the bar.

## 1. State in one paragraph
The end-to-end product path works on real clips: one command
(`scripts/racketsport/process_video.py --video … --rally-gating --ball-track …`) →
player tracking → court-placed, stance-grounded, temporally-smoothed 3D skeletons + meshes in a
metric world → default 3D ball chain (fused 2D → candidates → auto-bounce anchors → frozen arc
solver → flight-sanity gate → honest trail) → fused 6-DOF paddle estimate (render-only) → browser
replay viewer with trust bands, honesty KPIs, mesh layer, 2x-FPS interpolation. Wolverine E2E wall
went 2141s → ~530-700s this week with zero quality change (gates green, slide bit-identical where
required). VERIFIED remains 0 by discipline. Every measured accuracy wall (ball F1, tracking
coverage, BODY grounding on handheld, paddle stability, court auto-find on overlays) traces to the
same root: **no in-domain owner-captured training data yet**. That capture pipeline is the single
highest-value next investment.

## 2. DONE — verified, by workstream (evidence per claim)

### 2.1 BALL — 3D chain shipped as pipeline default (session "ball-tracking-long-run") — COMMITTED
- Default chain: candidates sidecar emission (WASB/TrackNet top-K), label-free auto-bounce anchors
  (cusp + gap-ballistic; beat the human-bounce baseline on Burlington LOO 0.0226 vs 0.0313),
  frozen row-22 arc solver config (fixture-guarded against drift), flight-sanity parabolic
  demotion gate, fail-closed self-kill excluded from world, dual-artifact runner
  (`run_ball_chain.py`, `--heldout-authorized` guard). Commits `790930ed`+`faf70a0e`+`2381ce88`.
- 3-clip browser-verified (18-26 FPS, trail + honesty KPI live):
  `runs/lanes/ball_v2_viewer_polish_20260705/`, `runs/lanes/ball_final_outdoor_20260705/`.
- Viewer fail-closed gate landed: self-killed solves can no longer render as "measured"
  (`runs/lanes/ball_viewer_failclosed_fix_20260705/`); `verify_process_video_viewer.py` now
  asserts real grid labels + ball honesty. net_plane now consumed by the default arc solve
  (`runs/lanes/ball_arc_netplane_20260705/`).
- Held-out shot taken honestly and MISSED: product F1@20 0.6969 vs 0.7248 bar (ledger rows 22-23;
  4th internal→held-out inversion). Standing best ball candidate: zero-shot WASB-tennis 0.7248
  (row 4). Chain beats the anchor on precision/hidden-FP/P95/teleports; loses recall only.
- Training campaign CONCLUDED NEGATIVE with measured proof (public data can't beat baselines):
  `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`. Deferred checkpoints:
  `models/checkpoints/candidates_t6/` (LOCAL, survive VM teardown).
- ⏳PENDING: P3-A anchor-first BVP solver lane (Codex, live at write time) — makes out-of-court
  arc fits impossible by construction (court-volume gates, endpoint-corridor refinement). P2 arc
  render/courtmap presentation layer coded+tested (`runs/lanes/ball_p2_arc_render_courtmap_20260705/`).

### 2.2 SPEED — Wolverine E2E 2141s → ~530s equivalent, zero quality change — ⏳final verdict pending
- S1-S4 slim BODY monoliths + VM-built mesh index + batched rsync + full phase instrumentation:
  2141→1144s, transfers 1.96GB→76MB/clip (`runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md`).
- CHUNKFIX (pickle-chunk subprocess handoff — binary .npy exists but is OPT-IN, not default; killed a PRE-EXISTING cross-venv numpy pickle incompat that
  silently faked all prior "pickle" numbers): 1144→702.4s
  (`runs/body_chunkfix_verify_20260705T204618Z`, `runs/lanes/body_chunkfix_20260705/REPORT.md`).
- SCHED-A `--body-schedule=overlap` opt-in landed, serial default byte-identical
  (`runs/lanes/sched_parallel_body_20260705/`).
- Payload-collapse (feed gates from arrays, kill 171s smpl payload assembly): speed CONFIRMED —
  BODY stage wall 618.5→473.0s (−23.5%), E2E ≈533s after backing out a measured 170s GPU-lock
  queue confound (`runs/lanes/body_payload_collapse_livecheck_20260705/REPORT.md`). Accuracy
  invariants diverged in that run, attributed with code+test evidence to the concurrently-landed
  `worldhmr.py` smoothing feature contaminating the baseline, NOT the payload-collapse refactor;
  an apples-to-apples isolation rerun was executing at write time
  (`runs/lanes/payload_collapse_isolation_20260705/`). ⏳Verdict reconciled in §8.
- Measured truth: actual GPU inference is ~18s/clip (15.3ms/person); everything else is data
  plumbing. Warm-worker daemon measured not-worth-it (~67s).
- GPU inference quality gates: SAM3D Phase D 32.23ms/person steady (≤55 bar), BODY GPU ~$0.117/clip.

### 2.3 BODY/PLACEMENT/VISUAL — grounded, smoothed, mesh-aligned worlds on all 4 clips
- Foot-slide disease killed at refine time (FIX3): Wolverine 9/9 gates PASS, slide 122.4mm p95→~0
  (pipeline gate 25.2mm), Burlington 7.6mm, Outdoor 6.2mm. IMG_1605 = honest attributed FAIL
  (handheld camera drift; see §5). `runs/lanes/joint_placement_4videos_20260704/FINAL_REPORT.md`.
- Visual polish live-verified (`runs/visual1_wolverine_20260705T220517Z`, 532.3s E2E, gates green):
  smoothing resets 14→2, feet jitter RMS −60-75%, wrists −40-46%, worst root-step p95 0.267→0.100m.
  Measurement harness: `threed/racketsport/visual_quality.py` + `measure_visual_quality.py`.
- Viewer: mesh layer consumable in ALL previews (30MB index, no more 950MB monolith parse fail),
  VP-C2 mesh-root fix (mesh no longer sinks into floor), skeleton-inside-mesh per-frame alignment,
  presence-hold flicker fix, "2x FPS (interpolated)" button honest + reversible.
- Contact-dense mesh scheduling (hitter dense ±0.5s around contacts, ball_aware mode) landed +
  unit-proven in `threed/racketsport/body_compute.py` — NOT live-proven yet (needs ball-chain
  runtime config present; first fresh E2E run after this handoff should confirm).

### 2.4 COURT AUTO-FIND — Wave A complete on branch `worktree-court-autofind-20260705` (pushed; NOT on main)
- Upload guess+confirm UI with the trust hole closed (unconfirmed guesses can never ride the
  TRUSTED calibration channel); geometric multi-frame solver: Outdoor 4.4px NO-TAP (old best
  12.7px), aggregate 213.3px vs 289.5 baseline (hard bar 200 missed by 13.3 — honest PARTIAL);
  synthetic generator v2 (7 families incl. tennis-overlay dual line-family); court_unet_v2 24M
  trainer + eval harness, A100 training STAGED NOT RUN (one command, see §4).
- Apply path: `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch` (42 files,
  +8865; drop BUILD_CHECKLIST hunks on conflict) or cherry-pick from the pushed branch.
  Instructions + next-session runbook: `OWNER_CHECKIN_20260705.md` (court section).

### 2.5 RACKET 6-DOF — phase 1 fused paddle estimator shipped (render-only, NOT pipeline-wired by design)
- `threed/racketsport/paddle_pose_fused.py` + builder CLI + 35 tests. Final shippable set
  `runs/lanes/racket_6dof_20260705/i1_fused_estimator/final_v3/` (4 clips): Wolverine IoU 0.2356
  (proxy 0.111) teleport-free (29→0 undeclared teleports), Burlington 0.3424 (13×), center error
  47→20px / 111→12px, jitter 23-53→~5 deg/f. Same `racket_pose_estimate.json` contract as
  paddle_proxy → renders through UNMODIFIED viewer, ESTIMATED band. RKT stays SCAFFOLD until
  owner 4-marker/true-corner capture.
- Ball-direction factor built+tested but DORMANT until mid-air 3D ball velocities flow (arc stage
  now default → activates on first post-reset E2E with both stages verified together).
- Pipeline integration = deliberate deferred patch (production clips need per-clip YOLO26s
  inference or P2a masks; current box predictions only cover the 3 CVAT clips).

### 2.6 Foundation landed earlier this week (still true, don't rebuild)
E2E glue + stale-derivation invalidation + config extraction; confidence framework
(calibrated error-vs-horizon bands); DATA-ENGINE (`ingest_owner_capture.py` /
`prelabel_owner_capture.py`, eval-clip guards attack-tested); rally metrics (7 position-based
per rally×player, per-metric trust); schema provenance; fusion_arbiter_v1 frozen
(F1 0.7852 internal); fail-closed honesty hardening (approx never BAND_MEASURED, no fabricated
world_xyz, bounce-lift markers, provenance firewall).

## 3. INTEGRATION TRUTH TABLE — what is actually wired vs dormant
⏳PENDING at write time — being generated by the wiring-audit lane. Canonical outputs:
- `runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md` (feature → DEFAULT-ON / OPT-IN /
  BUILT-NOT-WIRED / WORKTREE-ONLY / DEAD-CODE, with file:line evidence)
- `runs/lanes/wiring_audit_20260705/DEFERRED_PATCH_LEDGER.md` (every unapplied patch, whether it
  still applies, recommendation)
Known-before-audit integration debts (verify against the table): racket estimator not invoked by
process_video (by design, §2.5); contact-dense mesh scheduling not live-proven; `grounding_refine`
dead code (consumes `foot_contact_phases.json` nothing produces — wire or remove);
`ball_failclosed_fixes_20260704/deferred_virtual_world_arc_status_and_raw_ball_flags.patch`;
racket `_paddle_estimate_trust_band` one-line wording patch; court Wave A entirely off-main.

## 4. NOT DONE — ranked queue for the next session
**P0 — the data engine (the measured unlock for every accuracy wall):**
owner captures → `ingest_owner_capture.py` → prelabel → CVAT review → in-domain fine-tunes.
Everything in §5 says zero-shot/public-data is exhausted. IMG_1605 already carries 30 REAL audio
onsets = first full-confidence contact test bed (racket P2c + BALL M4 both want it).
**P1 — speed to the 6-8 min/clip floor (path fully booked, measured):**
resolve payload-collapse isolation verdict → land it; then P2 shared-memory/mmap subprocess
handoff (~489→<40s already proven partly), P3+P5 (gates from arrays everywhere), P7 freshness,
dispatch dir auto-clean (A100 disk filled to 100% once — §5). Combined overlap proof queued.
**P1 — court auto-find Wave B:** A100 train (one command, staged:
`bash scripts/gpu-train-lock.sh bash runs/lanes/cal_model_20260705/train_a100.sh` in the worktree)
→ eval → wire model into solver E4 evidence channel (contract ready: `court_model_infer.infer_court_model`)
→ GEO r3 (temporal-median fallback trigger + top-3 cross-frame vote for adjacent courts) →
downstream impact harness → browser QA of review UI. All 4 Codex specs reusable verbatim:
`runs/lanes/cal_{geo,synth,model,product}_20260705/spec.md`.
**P2 — racket phase 2:** P2a wrist-gated masks (seg checkpoint exists:
`runs/rkt_train_20260702T072800Z/seg_yolo_external_split/`), P2b WiLoR hand frames (A100; fixes
rest-pose pronation weakness), P2c IMG_1605 GPU ball track. CAD refiners ruled premature.
**P2 — upstream skeleton smoothing lane:** raw skeleton position noise (2-8cm/frame @30fps) is the
measured binding constraint on paddle stability; helps bodies too.
**P2 — integration debts:** apply everything in the DEFERRED_PATCH_LEDGER; wire-or-remove
grounding_refine; TRK follow-ups (Outdoor teleport frame, IMG_1605 adjacent-court spectator-FP,
IDF1 scoring never wired — `idf1=None` hardcoded).
**P3:** IMG_1605 camera-motion compensation + owner-capture association profile; hand-switch
paddle "snap" cosmetic; BALL M4 audio sub-gate (needs re-capture with audio); Indoor CVAT export.

## 5. FAILURE-CASE CATALOG — measured walls and dead ends (do NOT re-attempt without new evidence)
**Held-out discipline record (BALL): 4 internal→held-out inversions.** Consensus/fusion tuning on
internal clips does not transfer (rows 19-20 SAM3.1 0.7557→0.5137; rows 22-23 chain 0.6969 vs
0.7248). Fusion re-tuning without owner data is FORBIDDEN (parked B06/B15).
**Public-data fine-tune wall (measured, $9.70 spot A100):** TNv3 +2.7 Wolverine but −17pt
Burlington; WASB-blurball fine-tune 0.0018 on Burlington (real distractor lock). Roboflow corpus
recipe preserved (see §7). `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`.
**hidden-FP consensus floor ~0.349:** 33.6% of hidden frames have ≥2 detectors spuriously
agreeing — 2D consensus cannot fix; only the physics lift prunes those.
**Loader trap:** blurball↔WASB-SBDT checkpoints strict-incompatible (36 SE keys);
`strict=False` silently amputates trained modules (T6 scored 0.1479 as an artifact until SE-aware
rescore). Always key-diff checkpoints across forks.
**Dead/parked approaches (with kill evidence):** SAM 3.1 ball (held-out inversion, row 19-20);
TrackNetV4 (no usable weights upstream); TrackNetV5 (proprietary); PB-MAT (does not exist
publicly); point-tracker gap-fill CoTracker3/Track-On2 (blur weakness, demoted); SAM2 mask
propagation (dead); PhysPT (license + 20fps hardcode + root-ownership conflict; parked);
RTMW skeletons (dead — SAM-3D body-mode is the only skeleton path, owner-ruled); rectangle→6DoF
paddle promotion (killed; box-only world suppression stays); 3 CAL neural architectures (kill was
ARCHITECTURAL: 160×90 input — recipe that replaces it: kp+line heatmaps @640×360, staged in court
lane); warm-worker daemon (saves only ~67s); pipeline-level stage parallelism on eval clips (<2s;
BODY interior is the only lever — async value is for owner captures with dead time).
**Court auto-find open walls:** Burlington/Wolverine adjacent-identical-court lock-on (top-3
cross-frame vote is the next idea); IMG_1605 tennis-overlay needs the neural E4 channel; Indoor
46→93px trade; fallback trigger saturates (temporal-median proposed, unactivated).
**Placement/identity (see PIPELINE_STATUS.md §3):** IMG_1605 handheld 41px camera drift → foot
slide 0.330m attributed FAIL + tracked far-court players are ADJACENT-COURT people (spectator-FP);
Outdoor single-frame 1.52m root teleport t=13.55s (fail-closed catches it; TRK root cause open).
**Calibration noise floor:** Burlington/Wolverine reproj p95 ≈ 19.8/19.9px ≈ the 20px F1 radius —
internal 3D fits carry that floor; court label review kit exists to push it down
(`runs/manager/owner_court_label_review_kit_20260705/`).
**Paddle absolute pronation:** SAM-3D finger joints trend rest-pose (palm-only IoU 0.065 < proxy
0.111); detector boxes dominate evidence (+0.19 of +0.19 IoU) — why P2b WiLoR is the fix.
**~11% of Wolverine paddle boxes structurally unreachable** (far-from-wrist track, sub-2s hand
switches) — ceiling on box-evidence approaches.
**Eval-clip limits:** none have audio (M4 unpassable as-is); zero dead time (rally gating is a
no-op on them — its value shows only on owner captures); Indoor CVAT export missing.
**Infra gotchas that cost real time:** cross-venv numpy pickle incompat silently faked all
pre-chunkfix "pickle" timings; A100 root disk hit 100% (dispatch dirs ~2GB each accumulate under
runs/process_video_body_dispatch/ — auto-clean still not implemented); `--clip` must be passed to
process_video or clip id defaults to "source"; monolithic ~1GB JSON kills the viewer (use
windowed refs/mesh index); Codex sandbox: no network/MPS/localhost-binds/xcodebuild; Claude agents
die passive-waiting (resume via SendMessage with "no idle-wait" order; budget 1-2 resumes per GPU
lane); background Bash tasks get externally kill-swept on this Mac (run long jobs
nohup+disown+monitor); bg-session Write/Edit blocked pre-worktree (Bash writes allowed; or
EnterWorktree; or owner sets worktree.bgIsolation=none); gcloud auth fragile — direct ssh
(`ssh -i ~/.ssh/google_compute_engine arnavchokshi@<ip>`) bypasses; anaconda python lacks MPS —
use the repo .venv; zsh has no `timeout` builtin.
**2026-07-04 data-loss event (unresolved):** ~26GB local runs/ + VM disk cleanup by an unknown
actor; human-reviewed bounce labels lost (superseded by auto-anchors); WASB anchor raw artifacts
lost (numbers preserved in ledger). If the owner didn't do it, a rogue cleanup process existed.

## 6. GIT STATE AT CLOSE
⏳Finalized in §8 after the last in-flight lanes land. Target state: `main` fully committed +
pushed (all 5 sessions' work); court Wave A on pushed branch `worktree-court-autofind-20260705`
(apply via §2.4); `third_party/` = pinned gitlinks per `third_party/VENDOR_PINS.md` with our
in-vendor additions backed up under `third_party/pickleball_vendor_additions/`; ball model weights
gitignored under `models/checkpoints/` (sha256s in ledger row 22).

## 7. GPU TEARDOWN + NEW-GPU COLD START
**Old VM:** `pickleball-a100-spot-ase1a` (project `gifted-electron-498923-h1`, asia-southeast1-a,
A100-SXM4-40GB spot, ~$1.1-1.3/hr). ⏳Final inventory + termination log in §8.
**Intentionally abandoned on teardown:** `~/ball_training_data` Roboflow corpus (8.6k frames —
fine-tunes measured negative; rebuild recipe = `runs/lanes/ball_t4_train_20260704/` scripts +
owner's Roboflow key, ~1h); sam31/physpt/sam3 venvs (SAM3.1 killed, PhysPT parked); dispatch dirs.
Anything unique-and-valuable found by the final inventory is listed in §8.
**Cold start (new GPU), in order:**
1. Create spot VM (one steady GPU only; <$2/hr policy; prefer L4 for detector inference, A100
   when VRAM-bound). If gcloud auth is broken: owner runs `gcloud auth login` (hello@ account).
2. `git clone` the repo (it is now fully pushed — DO NOT resurrect the old scp-file-sync
   workflow; the md5-sync discipline in old notes is obsolete post-commit).
3. Restore vendor checkouts: `third_party/VENDOR_PINS.md` table (clone + checkout pinned sha),
   then re-apply our additions from `third_party/pickleball_vendor_additions/`.
4. `scripts/racketsport/gpu_cold_start.sh` (proven 258s cold start) + venv per RUNBOOK.md.
5. Pull model weights per `models/MANIFEST.json` + ledger sha256s.
6. Serialize GPU jobs via `scripts/gpu-train-lock.sh` / `gpu-eval-run.sh` (the /tmp/gpu-lease
   slot mechanism). Verify with nvidia-smi before claiming availability.
7. First job recommendation: court model training (staged, §4) — it's one command and unblocks
   the E4 channel; then a fresh 4-clip E2E to live-prove contact-dense mesh scheduling + racket
   arc-stage activation together.

## 8. CLOSE-OUT RECONCILIATION (⏳ filled at final commit)
- Payload-collapse isolation verdict: **SETTLED (manager-run diff).** Same-worldhmr legacy vs
  array-native: gates value-identical; skeleton3d differs mm-scale on 14,863 values; root cause =
  array-native drops stance lower-body smoothing protection (828→0 protected frames). Speed win
  real; array-native stays OPT-IN OFF; fix recipe in
  `runs/lanes/payload_collapse_isolation_20260705/REPORT.md`. Default pipeline unaffected.
- Ball P3-A lane outcome: **PARTIAL, committed as documented WIP.** BVP shooting + endpoint-corridor
  refinement + court-volume sanity landed test-green (547P/1F known-unowned; frozen row-22 fixture
  guard passes; no promotion). Open: 5 previously-good baseline intervals (4 Burlington, 1 Wolverine)
  lose `fit` status after reselection; internal-val F1 check not rerun. Finish-line steps in the lane
  report's `next`. The lane itself refused to commit (correct per its own bar); the wind-down sweep
  committed the tree state with this labeling.
- Wiring truth table + ledger: **DONE** (`runs/lanes/wiring_audit_20260705/`). Headlines: ball chain /
  viewer honesty / net_plane / slim monoliths / mesh index / confidence-gate fixes = DEFAULT-ON;
  overlap schedule + binary chunks + array-native = OPT-IN; paddle_pose_fused + grounding_refine =
  BUILT-NOT-WIRED (grounding_refine stage exists but always skips — no default producer of
  foot_contact_phases.json; run_physics_footlock.py is the standalone producer); court Wave A =
  branch-only. All 4 old deferred patches verified SUPERSEDED (content already at HEAD). Orchestrator
  chunk-format status-string lie fixed in the sweep.
- Docs-recon outcome: **DONE** — MASTER_PLAN/CAPABILITIES/RUNBOOK/TECH_STACK/RACKET_6DOF_GOAL
  reconciled with July-5 evidence, GPU marked reset-pending (`runs/lanes/docs_recon_winddown_20260705/
  RECON_NOTES.md`). The 5 doc-test failures found at baseline were all fixed by the wind-down sweep
  (doc allowlist, storage untracking, checklist heading, monitor CLI reference test).
- Final full-suite count: **wide pytest (tests/racketsport + tests/render_service, MPLBACKEND=Agg):
  2898 passed / 16 skipped at first pass with 6 failures; 4 were fixed in-session (2 tier2 body-runner
  tests now opt in to `write_body_monoliths=True` since slim is the default; replay-readiness test got
  labels_root isolation; stale 35.240.205.82 block removed from configs/ssh/a100_known_hosts) and
  re-proven green (117/117 across all affected files). Web vitest 182/182. Doc-consistency + storage
  audit + scaffold index: ALL GREEN.**
  **2 KNOWN FAILURES remain, both pre-existing and booked (do not paper over):**
  `tests/racketsport/test_overlapping_court_calibration_eval.py` (2 tests) — root cause: the
  owner-reviewed `eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints.json` (written
  2026-07-04) has no `frames` metadata block, which the dormant overlapping-court eval loader
  requires (`court_calibration_metric15.py:126`). Label-schema drift between two efforts; resolving
  means an owner-level decision on the label file schema or a loader-side compat shim — next session.
- VM archive: **DONE** — 10 VM-only checkpoints sha256-verified into
  `models/checkpoints/vm_archive_20260705/` (268MB; 6 more already existed locally); full git bundle
  of the VM repo (69MB, 5 refs) + dirty-state captures in `runs/lanes/vm_archive_20260705/`
  (see ARCHIVE_MANIFEST.md). Termination: **VM powered off 2026-07-05 17:00 PDT** via ssh `sudo shutdown -h now` after
  idle verification (0% GPU / 0 MiB / no processes); confirmed unreachable. GPU-hour billing stopped.
  OWNER: to fully delete (removes the boot disk + any residual disk cost):
  `gcloud auth login` then
  `gcloud compute instances delete pickleball-a100-spot-ase1a --project gifted-electron-498923-h1 --zone asia-southeast1-a`
- Final main sha: **the commit that added this line** (wind-down sweep commit, pushed to
  origin/main 2026-07-05 evening). Everything from all five sessions is in it; court Wave A remains
  on pushed branch `worktree-court-autofind-20260705` by design.

## 9. OWNER ACTION QUEUE (only-owner-can-do, easiest first)
1. Roboflow: export court keypoint datasets (court lane check-in item #3) and/or re-issue an API
   key when ball fine-tunes resume on owner data.
2. `gcloud auth login` (hello@…) — needed for any VM create/delete; ssh works without it.
3. Court label review kit, 5-10 min: `runs/manager/owner_court_label_review_kit_20260705/README.md`
   (4-16 suspect labels block the 0.2ft calibration target).
4. Capture plan: phone clips WITH AUDIO (unlocks M4 + contact confidence), static tripod when
   possible (IMG_1605 handheld drift is a measured failure), both courts + varied lighting.
   Import path already built: camera-roll import / `ingest_owner_capture.py`.
5. Racket 4-marker/true-corner capture — the only path to VERIFIED paddle pose.
6. Codex weekly quota resets Jul 9 1:31PM — court specs are reusable verbatim on reset.
