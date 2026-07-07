# Owner check-in — 2026-07-05 (racket 6-DOF goal session)

## ⭐ Headline
**Racket goal PHASE 1 SHIPPED same-day.** The new fused 6-DOF paddle estimator beats the old
wrist-proxy on every bar, with one config across clips: paddle box IoU 0.11→0.26 (Wolverine) and
0.03→0.36 (Burlington), center error 47→20px / 111→12px, face-normal jitter 23-53→~5 deg/frame
median. 100% frame coverage, honest ESTIMATED banding, renders in the existing viewer with zero
viewer changes. Visual QA screenshots in
runs/lanes/racket_6dof_20260705/i1_fused_estimator/qa_visual/. Ball-direction factor is built +
tested but dormant: no mid-air 3D ball velocities exist yet — it auto-activates once the ball
session's arc stage output flows (their lane landed today too). Full story: RACKET_6DOF_GOAL.md.

## Blockers
(nothing hard-blocking — work continues on Sonnet)

- ⚠️ CODEX WEEKLY CREDITS EXHAUSTED until Jul 9, 1:31 PM (hit ~12:03 today mid-goal). One-click
  unblock if you want the workhorse back sooner: https://chatgpt.com/codex/settings/usage
  (buy credits). Meanwhile racket implementation runs on Sonnet in manager-checkpointed legs.

Optional whenever you want the real promotion gate unlocked (not blocking rendering work):
the 4-marker/true-corner paddle capture is still the only path to VERIFIED paddle pose claims.

## Verify when back
1. LOOK at the paddles: viewer on final_v3/ worlds — ALL 4 clips now viewable
   (runs/lanes/racket_6dof_20260705/i1_fused_estimator/final_v3/<clip>/virtual_world.json).
   All 29 undeclared teleports eliminated; remaining 5 jumps are at justified hand switches.
   IMG_1605 correctly shows only your 2 real players. (qa_visual/ has screenshots of the v1 world.)
2. RACKET_6DOF_GOAL.md (2 min) — phase-1 numbers + the measured truth: detector-box evidence
   dominates; SAM-3D finger joints give smoothness but weak absolute pronation (they trend
   rest-pose); that's why phase 2 = masks (P2a) + WiLoR hand model (P2b) + IMG_1605 GPU ball track
   (P2c, your 30 audio onsets are waiting).
3. Everything renders as ESTIMATED band; rectangle-to-6DoF kill respected; Outdoor/Indoor labels
   untouched; RKT promotion still needs your 4-marker/true-corner capture.
4. Decisions when you're back: (i) green-light phase 2 — masks + WiLoR (A100 hours) + IMG_1605
   GPU ball track; (ii) NEW measured fact: raw skeleton position noise (2-8cm/frame at 30fps) is
   now the binding constraint on how steady the paddle can look — an upstream skeleton-smoothing
   lane would help paddles AND bodies; (iii) final Wolverine numbers after the teleport fixes:
   IoU 0.236 (was 0.111 baseline), teleport-free — I traded 0.004 IoU for zero teleports.

## Money/GPU log
- No GPU spend this session so far (research lanes are local CPU + web). A100 untouched by racket work.

## In flight (other sessions, untouched by me)
- ball_i1_default_integration lane (LIVE since ~08:00Z): making the 3D ball chain a default
  pipeline stage; owns process_video.py/virtual_world.py/web/replay — racket lanes fenced off those.
- Speed lane (runs/lanes/pipeline_speed_20260705/) open in measurement phase.

## Overnight log
- ~08:00Z: goal opened, repo scout dispatched.
- ~18:40Z: scout landed (inventory in runs/lanes/racket_6dof_20260705/STATUS.md); goal doc written;
  BUILD_CHECKLIST handoff posted; R1 (evidence quantification) + R2 (external SOTA, web) Codex
  lanes dispatched with monitors.

---

# COURT AUTO-FIND LANE (court manager session, opened ~08:00Z Jul 5)

⭐ **Headline:** Court auto-guess + calibration lane is designed and running. Full architecture in
`runs/lanes/court_autofind_20260705/DESIGN.md` (evidence ensemble → multi-hypothesis
pickleball-vs-tennis template competition → fail-closed verify → upload review UI; neural
keypoint+line model retrain on synthetic data — the old model was killed by a 160x90-input
architecture, not by the idea). Implementation lanes are executing in an isolated worktree
(`.claude/worktrees/court-autofind-20260705`) because this background session's direct main-tree
writes are blocked by a new harness guard — results come back as one consolidated diff for you to
apply (or I apply it once unblocked, see Unblocks).

## Blockers (none hard — work continues)
- **Codex weekly limit exhausted until Jul 9 1:31PM** (other sessions consumed it overnight).
  Implementation shifted to Sonnet agents — watch Claude spend; I'm keeping lanes narrow.

## One-command unblocks (in priority order)
1. **Let background sessions write the main tree again (restores your playbook's no-worktree mode):**
   `! python3 -c "import json,pathlib;p=pathlib.Path('.claude/settings.json');c=json.loads(p.read_text()) if p.exists() else {};c.setdefault('worktree',{})['bgIsolation']='none';p.write_text(json.dumps(c,indent=2)+'\n')"`
2. **gcloud reauth** (only needed if I must create a 2nd GPU VM; VM1 reachable via direct ssh):
   `! gcloud auth login`
3. **Roboflow court datasets** (1,135-image pickleball keypoint corpus — 35x our real training data;
   needs your account): export `pickle-court-keypoints-nluo7` (xuann-bacc-ujr91) and
   `pickleball-court` (gideons) from Roboflow Universe in COCO-keypoints format, drop the zips in
   `runs/owner_data/incoming/`.

## Verify when back
- **5-10 min unblock: court label review kit** — runs/manager/owner_court_label_review_kit_20260705/README.md
  (4-16 suspect keypoint labels are what's blocking the 0.2 ft calibration target; you rule on each).
- Overlay contact sheets from the geometric solver: `runs/lanes/cal_geo_20260705/overlays/` (worktree
  copy if not yet applied to main) — do the guessed courts LOOK right on all 5 clips?
- Synthetic corpus contact sheets: `runs/lanes/cal_synth_20260705/samples/` — do tennis-overlay +
  adjacent-court fakes look like your IMG_1605 reality?

## Money / GPU log
- $0 spent so far this lane. VM1 A100 idle at last check (0% util, 19G disk free — tight).
  Court model training will use VM1 via train-lock (yields to speed-session BODY jobs); no 2nd VM
  unless contention demands it.

## Overnight log (court lane)
- ~08:0xZ recon: 4 scouts (product path, prior evidence, training assets, external SOTA) + eval-frame
  visual review. Key finds: upload "court prediction" today is a fixed template rectangle that flows
  into a TRUSTED calibration channel (fail-open — fix designed, routes around the other session's
  files); IMG_1605 is the tennis-overlay hard case; old neural kill was architectural (160x90 input).
- ~08:3xZ DESIGN.md + BUILD_CHECKLIST ownership fence posted.
- ~08:4xZ 4 Codex lanes dispatched → all died on the Codex usage limit (resets Jul 9). Pivoted to
  Sonnet.
- ~09:0xZ worktree isolation set up (baseline = 152 dirty files imported so lanes build/test against
  the true tree state); lanes relaunching; CAL-EXT (external weights/datasets download) running.


## COURT LANE — FINAL WRAP (~14:5xZ, owner stop order received)

⭐ **Session outcome:** the court auto-guess system went from a fixed template rectangle to a real
multi-frame solver + human-confirm product flow + a trainable 24M-param model, in one session.
Outdoor clip is now PIXEL-ACCURATE no-tap (4.4px vs 12.7px old best); aggregate 213px vs 289.5px
baseline (26% better); the trust hole is closed (unconfirmed guesses can never ride the trusted
channel); zero false-confident promotions anywhere. See it yourself: worktree
runs/lanes/cal_geo_20260705/overlays_r2/contact_sheet.jpg.

**All code lives on worktree branch `worktree-court-autofind-20260705`** (background-session guard
blocked main-tree writes). To bring it onto main (after you're happy):
```
cd ~/Desktop/pickleball/.claude/worktrees/court-autofind-20260705
git add -A && git commit -m "court autofind lanes (scratch)"
git diff 501a1114..HEAD > /tmp/court_autofind.patch
cd ~/Desktop/pickleball && git apply --stat /tmp/court_autofind.patch   # review
git apply /tmp/court_autofind.patch                                     # apply
```
(or run unblock #1 from the list above and tell the next session to apply it for you).
Ready-made copy: `runs/lanes/court_autofind_20260705/handoff/` has the patch
(court_autofind_wave_a.patch, 42 files/+8865), the solver overlay sheet, all 7 synthetic
contact sheets, the CAL-GEO report, and train_a100.sh. Apply caveat: drop any
BUILD_CHECKLIST.md hunks if they conflict — equivalents are already posted on main.

**Next session runbook (in order):**
1. A100 training (everything staged): `bash scripts/gpu-train-lock.sh bash runs/lanes/cal_model_20260705/train_a100.sh`
   then eval: `python scripts/racketsport/evaluate_court_model_v2.py --checkpoint <ckpt> --out <dir>/owner_gate_report_v2.json --device cuda`
2. Wire the trained model into the solver's E4 evidence channel (contract ready: court_model_infer.infer_court_model)
   — this is the expected unlock for IMG_1605-style tennis overlays + Burlington/Wolverine.
3. CAL-GEO round 3 (specs + measured signals ready): temporal-median fallback trigger + top-3
   cross-frame consistency vote for adjacent-court scenes.
4. Downstream impact harness (calibration variants -> placement/ball gate deltas on Wolverine/Burlington).
5. Browser QA of the new upload review UI (drag + Confirm/Re-predict/Skip).
6. Codex resets Jul 9 — all 4 lane specs in runs/lanes/cal_*_20260705/spec.md are reusable verbatim.

**Money/GPU:** $0 GPU spend this lane (VM1 untouched, verified ready). All spend was Claude-side
(4 recon scouts + 5 implementation/download lanes + 2 resumes).


---
**[WIND-DOWN CORRECTION 2026-07-05 ~17:1x PDT]** The court worktree `.claude/worktrees/court-autofind-20260705`
has been REMOVED (it was breaking the markdown-inventory doc test). The Wave A work is fully preserved in:
(1) pushed branch `worktree-court-autofind-20260705` on origin — apply via
`git diff 501a1114..worktree-court-autofind-20260705 | git apply` (drop BUILD_CHECKLIST hunks on conflict),
(2) the ready-made patch `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch`,
(3) the VM git bundle `runs/lanes/vm_archive_20260705/pickleball_git_all.bundle`.
Full wind-down state: `RESET_HANDOFF_20260705.md`.
