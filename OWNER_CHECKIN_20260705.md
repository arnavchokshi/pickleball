# Owner check-in — 2026-07-05 (racket 6-DOF goal session)

## ⭐ Headline
Your new goal is OPEN and moving: **full 6-DOF paddle in the 3D world from wrist + ball direction**
(RACKET_6DOF_GOAL.md). Big head start discovered: a wrist-proxy paddle already renders in the
viewer, and a ball-reflection face-normal estimator already exists — they were just never fused.
Also: the paddle's missing orientation DOF (forearm roll / pronation) is recoverable from data we
ALREADY store — SAM-3D's 70-joint output includes 20 finger joints per hand that nothing reads yet.
No new capture, no new model, no GPU needed for the core upgrade.

## Blockers
(nothing hard-blocking — work continues on Sonnet)

- ⚠️ CODEX WEEKLY CREDITS EXHAUSTED until Jul 9, 1:31 PM (hit ~12:03 today mid-goal). One-click
  unblock if you want the workhorse back sooner: https://chatgpt.com/codex/settings/usage
  (buy credits). Meanwhile racket implementation runs on Sonnet in manager-checkpointed legs.

Optional whenever you want the real promotion gate unlocked (not blocking rendering work):
the 4-marker/true-corner paddle capture is still the only path to VERIFIED paddle pose claims.

## Verify when back
1. Read RACKET_6DOF_GOAL.md (2 min) — the architecture: paddle = hand frame × a grip transform
   held constant per grip segment; ball reflection at contacts + finger-derived palm frame +
   (later) wrist-gated masks lock it; wrist alone then carries the paddle through frames with no
   direct evidence ("whenever possible", honestly banded).
2. Research wave results: runs/lanes/racket_6dof_20260705/{r1_evidence,r2_sota}/REPORT/FINDINGS.
3. Everything renders as ESTIMATED band; the killed rectangle-to-6DoF promotion stays killed.

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
