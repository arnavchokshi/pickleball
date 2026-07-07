# WAVE 3 PLAN (opened 2026-07-07 UTC / 2026-07-06 22:2x PDT, manager)

Queue source: owner boot message + BUILD_CHECKLIST [WAVE-2 COMPLETE 2026-07-07] bullet. HEAD at
wave start: 2f7336598. Fleet at start: fleet1 STOPPED (repo pinned 5b9f132ee = 16 stale), fleet2
DELETED. Auth: SA impersonation verified live at session start. `body4d-waker-ctrl` e2-micro in
us-central1-a is NOT ours (labels cost-center=body4d, created 2026-06-14) — untouched.

Session mechanics note: bg-session write guard active; manager coordination files are authored on
worktree branch `worktree-wave3-manager` (ff'd to main) and merged to main at commit checkpoints.
Lane artifacts + specs' runtime copies live in the shared checkout under runs/lanes/ as always
(codex/agent processes are unaffected by the session guard). Settings self-modification was
classifier-denied — respected; worktree route used instead.

Owner constraints this wave: owner may run labeling passes from laptop; owner CANNOT record new
videos for a couple days — all lanes run on public/harvest data. Commits pre-authorized
(.claude/settings.json); pushes owner-gated. Slide-gate thresholds FROZEN.

## Lanes (dispatch batch 1 — all safe-parallelism-checked: file-disjoint, data-disjoint, resource-disjoint)

| lane | kind | queue item | owns (files) | GPU |
|---|---|---|---|---|
| w3_codesync_20260707 | Codex | #0 code-sync/version-stamp hardening | scripts/racketsport/remote_body_dispatch.py, scripts/fleet/*, tests/racketsport/test_remote_body_dispatch.py (+new tests) | no |
| w3_slidediag_20260707 | Codex diagnosis | #1 slide-MAX outliers (diagnosis only; fix lane after ruling) | NONE (runs/ only) | no |
| w3_groundref_diag_20260707 | Codex diagnosis | #2 grounding_refine 4/4 self-kill | NONE (runs/ only) | no |
| w3_cammotion_conditional_20260707 | Codex | #6 motion-conditional default | threed/racketsport/camera_motion.py, scripts/racketsport/estimate_camera_motion.py, scripts/racketsport/process_video.py (SOLE owner), tests/racketsport/test_camera_motion.py, tests/racketsport/test_process_video.py | no |
| w3_img1605_mesh_diag_20260707 | Codex diagnosis | #7 img1605 zero-mesh-frames | NONE (runs/ only) | no |
| w3_p11_prep_20260707 | Codex | #4 P1-1 harness prep (GPU run = later lane) | threed/racketsport/roboflow_corpus.py (extensions), threed/racketsport/ball_tracknet_cvat_dataset.py, NEW scripts/racketsport/train_ball_pretrain.py + configs + tests/racketsport/test_roboflow_corpus.py + new test files | no (CPU smoke) |
| w3_labelfactory_20260707 | Sonnet (local docker/browser) | #3 P0-4 launch + #8 RAFT prefetch | none in repo (docker state, runs/lanes/w3_labelfactory_20260707/, docs/racketsport/OWNER_LABELING_GUIDE.md) | no |
| w3_fleetseed_20260707 | Sonnet (gcloud/SSH) | #0b fleet1 restart+sync-to-HEAD + #5 P1-2 teacher DRY-RUN (2 clips) | none in repo (runs/lanes/w3_fleetseed_20260707/) | fleet1 (reuse; STOP after) |

Cross-lane fence list (given to every Codex lane): remote_body_dispatch.py+scripts/fleet → codesync;
camera_motion.py+estimate_camera_motion.py+process_video.py+their tests → cammotion; roboflow_corpus.py+
ball_tracknet_cvat_dataset.py+train harness → p11_prep. Diagnosis lanes own NO repo files.

## Mid-wave events (booked)
- 34/40 p01b raw WASB prelabel sidecars pruned on-disk ~23:21 PDT (owner Finder cleanup; survivors ==
  the 6 CVAT review clips). Teacher tuning proceeds on 8 clips (6 survivors + 2 fleetseed raw);
  REGENERATE the 34 raw sidecars on the next fleet1 GPU cycle (p01b recipe banked, ~40 GPU-min ~$1),
  then mass-produce teacher sets at the blessed operating point (single local command, MISSING_34
  manifest in teachertune lane dir).
- P0-4 LAUNCHED: CVAT 2.69.0 live at localhost:8080 (docker compose at ~/cvat_labelfactory/cvat_src);
  owner labels tasks 7,8 first. Structural P1-2 finding: 3D physics-gated teacher on harvest REQUIRES
  court auto-cal (P4) — until then harvest teacher = 2D gate chain (ballistic+RANSAC-arc+Kalman).
- fleet1 IP changed on restart: 34.143.175.207 → 35.240.183.195 (same host key/disk). known_hosts +
  remote_body_dispatch DEFAULT_REMOTE_HOST update pending w3_codesync land.
- codex v0.142.5 resume gotcha: `codex exec resume` accepts only -c overrides (no --cd/--sandbox/
  --output-schema/-o; sandbox+cwd persist from the recorded session; lane must self-write the report
  file). Manual §10 resume template needs this correction at wave close.

## Held for rulings (dispatch later)
- Slide-MAX FIX lane (after w3_slidediag ruling) → adversarial verify lane → fresh GPU proof.
- P1-1 GPU training lane: H100-80GB-spot-first ≤$5/hr (owner directive 2; one-time cold-start
  validation) — provision AFTER w3_p11_prep lands (until its runbook exists, provisioning is speculative).
- P1-2 MASS seeding (40 clips, fan per owner directive 1): after teacher dry-run + teacher-quality
  check vs owner review labels (ruling: teacher validated before mass pseudo-labels).
- Wave-end composed fresh-worlds GPU run (decisive for grounding_metrics.max_foot_lock_slide_m ≤ 0.03,
  img1605 mesh fix, cammotion auto-decisions, code-sync stamp live proof) + clean wide suite +
  manager browser verify + handoff.

## Data discipline
No lane touches held-out (pwxNwFfYQlQ / vQhtz8l6VqU excluded everywhere, asserted). Outdoor/Indoor
labels untouched (no ledger row this wave). P1-1 internal-val ONLY.
