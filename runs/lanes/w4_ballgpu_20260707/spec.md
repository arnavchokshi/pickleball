# LANE w4_ballgpu_20260707 — Sonnet GPU: seed fine-tune + SST round 1 + threshold sweep (INTERNAL-VAL ONLY)

STATUS: FINAL (w4_ballcode RULED PASS + LANDED at 5b268aa6d). Exact CLIs spliced below.
CODE SYNC NOTE: sync the VM from COMMITTED main HEAD (>= 5b268aa6d) via the git-bundle method —
NEVER rsync the Mac working tree (it carries other live lanes' half-done edits, e.g. an in-flight
ball_arc_solver.py change that must NOT ship).

## OBJECTIVE
Bank internal-val-scored ball checkpoints on a self-provisioned H100 spot VM, then DELETE it:
(1) SEED FINE-TUNE: stage-2 trainer on the ~274-box owner CVAT labels, init from the stage-1
    warm-start checkpoint (`runs/lanes/w3_p11_train_20260707/checkpoints/latest.pt`).
(2) SST ROUND 1: student warm-started from the SAME stage-1 checkpoint, trained on pseudo-labels
    from the 40 local raw-WASB prelabel sidecars (the blessed teacher — raw single-WASB per the
    wave-3 measurement ruling; NEVER raw multi-detector fusion).
(3) THRESHOLD SWEEP: detection-threshold recall/hidden-FP trade on the best candidate + stage-1.
(4) SST disagreement queue export for the owner's next labeling session (active learning).
Discipline: **INTERNAL-VAL ONLY — no held-out anything. A public/thin-seed student NEVER takes a
held-out shot (4 inversions on record).** Nothing here changes any pipeline default.

## SCORING RULE (the #1 trap — read twice)
The training harness proxy `f1_at_20px` NEVER promotes anything. Every candidate is scored through
the SCORING BRIDGE on the two DEFAULT_CLIPS (`burlington_gold_0300_low_steep_corner`,
`wolverine_mixed_0200_mid_steep_corner`):
  1. `scripts/racketsport/run_wasb_ball.py --checkpoint <ckpt> --wasb-repo third_party/WASB-SBDT
     --video <clip> --fps <real fps> --candidate-top-k 5 --out <run_root>/<clip>/wasb/ball_track.json`
  2. `scripts/racketsport/fuse_ball_tracks.py` into the exact sidecar filenames the suite ingests
     (grep the suite's `_add_existing_candidate`/`--overlay-candidate` wiring FIRST; place to match)
  3. `scripts/racketsport/run_ball_tracking_eval_suite.py --run-root ... --clip <both>`
Exact keys to report per candidate: `label_f1_at_20px` (per clip), `mean_visible_hit_recall`,
`mean_hidden_false_positive_rate`, `visible_recall_at_20px`. References: WASB-tennis internal-val
F1@20 **0.6685** (ledger BALL-IV-1); stage-1 harness-proxy state was f1 0.6104 / recall 0.477 /
precision 0.848 (internal_val 2640). Report every candidate vs the 0.6685 reference and vs
stage-1's bridge score (measure stage-1 through the bridge too — it has never been bridge-scored).

## FLEET RULES (all mandatory)
- H100-80GB spot (a3-highgpu-1g) — asia-southeast1-b or -c (NOT -a), else us-central1/us-east4/
  us-west1/us-west4/europe-west4 (quota: 2/region approved). Attempt-create is the definitive
  quota test (describe lags). ≤$5/hr or STOP: purchase-approval.
- Verify gcloud auth FIRST with one cheap list call (impersonation
  `--impersonate-service-account=pickleball-fleet@gifted-electron-498923-h1.iam.gserviceaccount.com`
  or owner token); on a reauth challenge → STOP (typed needs-decision), never retry-storm.
- Create with `--provisioning-model=SPOT --instance-termination-action=STOP
  --labels=fable-lane=w4-ballgpu,fable-fleet=pickleball
  --metadata-from-file=startup-script=scripts/fleet/lane_vm_startup.sh` (EXCLUSIVE_PROCESS +
  preemption watcher come from the startup script — verify it did both).
- ALWAYS address the VM by its fresh external IP explicitly (IPs recycle); refresh
  `configs/ssh/a100_known_hosts` for the new IP (use `scripts/fleet/refresh_remote_host.*` if the
  w4_fleethosts lane landed it; else ssh-keyscan manually).
- CODE SYNC + PROOF: sync the VM checkout to the Mac's main HEAD via the proven git-bundle method
  (w3_fleetseed pattern); record md5 + `git status` whole-tree proof + a version_stamp.json in the
  lane dir. NO metric from the VM is trusted without this stamp.
- SELF-TEARDOWN: pull ALL checkpoints/summaries/queue files back to the Mac under
  `runs/lanes/w4_ballgpu_20260707/` and md5-verify BEFORE `gcloud compute instances delete`;
  list-confirm deletion; report uptime + $ estimate.

## PRESTAGE (STEP 1 of the BALL2D pillar — do this on the VM first, it has network)
- Fetch `wasb_tennis_best.pth.tar` (nttcom/WASB-SBDT MODEL_ZOO tennis checkpoint). sha256 MUST
  equal `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb` (authoritative:
  `models/MANIFEST.json` entry `wasb_tennis_bmvc2023`). Mismatch → STOP: needs-validation (do not
  train, do not infer a hash from anywhere else). ALSO scp it back to the Mac at
  `models/checkpoints/wasb/wasb_tennis_best.pth.tar` (closes the standing LOCAL BLOCKER).
- Upload: owner export `cvat_upload/exports/harvest_review_20260707/`, stage-1 `latest.pt`,
  `data/online_harvest_20260706/` prelabels + the rally frames the SST manifest needs (CHECK size
  first with du -sh; use the proven tar_batch + bounded-retries + rsync-fallback pattern for >10MB),
  eval clips + review labels for the two DEFAULT_CLIPS (scoring only — eval_guard enforces
  never-gradient-trained; run the protected-hash collision assert on the training corpus before any
  training step and report 35/0).

## RUN (exact CLIs from the w4_ballcode report; substitute <TS>; --device cuda; run on the VM)
1. SEED FINE-TUNE (init key-diff must be empty — the CLI aborts otherwise; bank ckpts):
   `.venv/bin/python scripts/racketsport/train_ball_stage2.py --cvat-export-root cvat_upload/exports/harvest_review_20260707 --init-checkpoint runs/lanes/w3_p11_train_20260707/checkpoints/latest.pt --out-dir runs/lanes/w4_ball_stage2_owner_<TS> --model-family wasb_hrnet --wasb-repo third_party/WASB-SBDT --device cuda --batch-size 8 --epochs 30 --learning-rate 5e-4 --weight-decay 5e-5 --image-size 512x288 --frames-in 3 --heatmap-radius-px 4 --checkpoint-every 500 --num-workers 4 --seed 1337`
2. BRIDGE-SCORE stage-1 AND the seed-fine-tune on both DEFAULT_CLIPS.
3. SST MANIFEST (protected-hash assert 35 expected):
   `.venv/bin/python scripts/racketsport/train_ball_stage2.py --mode build-sst-manifest --prelabel-root data/online_harvest_20260706/prelabels --rally-root data/online_harvest_20260706/rallies --sst-manifest-out runs/lanes/w4_ball_stage2_sst_round1_<TS>/sst_manifest.json --expected-protected-eval-hash-count 35`
   (Teacher sidecar layout per clip dir: ball_track.json + metadata JSON + WASB CSV; rally videos
   under data/online_harvest_20260706/rallies/<source_id>/.)
4. SST STUDENT TRAIN — MANAGER RULING on init (conditional, decided by step 2's bridge scores):
   IF seed-fine-tune >= stage-1 on bridge `label_f1_at_20px` (both clips, no hidden-FP regression)
   → `--init-checkpoint runs/lanes/w4_ball_stage2_owner_<TS>/checkpoints/latest.pt`;
   ELSE → `--init-checkpoint runs/lanes/w3_p11_train_20260707/checkpoints/latest.pt`.
   State which branch fired and why in the report. Rest of the command verbatim:
   `.venv/bin/python scripts/racketsport/train_ball_stage2.py --cvat-export-root cvat_upload/exports/harvest_review_20260707 --sst-manifest runs/lanes/w4_ball_stage2_sst_round1_<TS>/sst_manifest.json --init-checkpoint <PER RULING> --out-dir runs/lanes/w4_ball_stage2_sst_round1_<TS>/student_train --model-family wasb_hrnet --wasb-repo third_party/WASB-SBDT --device cuda --batch-size 8 --epochs 30 --learning-rate 5e-4 --weight-decay 5e-5 --image-size 512x288 --frames-in 3 --heatmap-radius-px 4 --checkpoint-every 500 --num-workers 4 --seed 1337`
5. BRIDGE-SCORE the SST student.
6. THRESHOLD SWEEP on the best-by-bridge candidate + stage-1: sweep the detector's candidate
   threshold (find the exact knob in run_wasb_ball/its config — report which) across ~5 values;
   re-run the bridge per point; table of recall vs hidden-FP (`mean_visible_hit_recall` /
   `mean_hidden_false_positive_rate`). Measure and report — no re-tuning spiral.
7. DISAGREEMENT QUEUE (teacher vs best student; student predictions in the same JSON shape as
   teacher sidecars — generate them with run_wasb_ball from the student checkpoint over the 40
   harvest clips, or the subset time allows — state coverage honestly):
   `.venv/bin/python scripts/racketsport/export_sst_disagreements.py --teacher-predictions data/online_harvest_20260706/prelabels --student-predictions <student_predictions_root_same_shape> --out runs/lanes/w4_ball_stage2_sst_round1_<TS>/sst_disagreements.json --large-offset-px 25` → pull to Mac.
KILL (commitments): internal-val bridge F1 regression vs 0.6685 for a candidate after 2 recipe
iterations → bank the negative, stop that arm (no spiral). Preemption mid-train → resume from the
atomically-saved checkpoint (checkpoint_every 500 gives ≤~1min loss); if the SPOT pool is hostile
(≥2 preemptions), fall back to A100-80/40 in-region and note the SKU change.

## ANTI-PASSIVE-WAIT (binding)
Poll all long steps in the FOREGROUND with bounded sleeps (`sleep 60` loops with progress checks);
NEVER end your turn to "wait/monitor". If you believe you are blocked, state the exact blocker and
what you need — do not idle.

## REPORT (final message = structured report; also write REPORT.md in the lane dir)
OBJECTIVE RESULT · acceptance table (per candidate: label_f1_at_20px both clips,
mean_visible_hit_recall, mean_hidden_false_positive_rate, vs 0.6685 + vs stage-1-bridge) ·
threshold-sweep table · artifacts pulled (paths + md5) · VM lifecycle (created/deleted proof,
uptime, $) · version-stamp proof · HONEST ISSUES · NEXT.
