# GPU fleet ledger (FABLE_OPERATING_MANUAL §12)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).

WAVE-4 FLEET LOG (2026-07-07, manager):
- pickleball-h100-w4body (H100 a3 spot) — CREATED from snapshot pickleball-fleet1-snap-20260707
  (attempt 2: a3 REQUIRES --boot-disk-type=pd-balanced, pd-standard rejected) → BODY-compat
  validation PASS (27/27 GPU tests 0 skipped; wolverine BODY dispatch real w/ 82-84% util;
  version-stamp 73/73 @ c8c4c0425) → DELETED list-confirmed. Uptime 58m01s ≈ $0.55-4.11.
  **SKU POLICY UPDATE: H100 = BODY-VALIDATED 2026-07-07, ≈2.37× A100 (479.6s vs 1134.4s
  wolverine ref; caveat: reused clip-dir, not clean-room). H100 now default for BOTH training
  and BODY when ≤$5/hr; decisive gate runs stay on proven SKUs per standing rule.** Evidence:
  runs/lanes/w4_h100body_20260707/REPORT.md. refresh_remote_host.sh first LIVE proof: works
  (sshd needs ~90s post-create before first keyscan).
- pickleball-h100-w4ballgpu (H100 a3 spot) — BUSY, lane w4_ballgpu (seed fine-tune + SST r1 +
  threshold sweep; syncs from committed HEAD ≥ 5b268aa6d), self-DELETE on completion.
- fleet1 stays STOPPED (reserved for the wave-close A100 fresh-worlds proof).
- TOOL BUGS found live (fix lane w4_syncstamp dispatched): --sync-remote-code dirty-check hashes
  working tree not shipped committed blobs (false drift under concurrent-session dirt); tar_batch
  transport failure exit-1 non-retryable while ssh transport bug (`ssh_packet_write_poll: Result
  too large`) recurs on large transfers — rsync fallback exit-255 IS retryable.
WAVE-5 FLEET LOG (2026-07-07, manager):
- pickleball-h100-w5p22 (H100 a3-highgpu-1g SPOT, asia-southeast1-b, pd-balanced 200GB boot disk
  FROM pickleball-fleet1-snap-20260707) — lane w5_p22latent_20260707 DONE: CREATED
  2026-07-08T~05:35Z (1st attempt succeeded, no quota fallback needed) → P2-2 MHR decode wrapper +
  fidelity gates + latent smoothing prototype (GATE 1a PASS 2.7e-5 deg; GATE 1b measured-partial,
  see report) → new fan snapshot `pickleball-fleet-snap-20260708-ffmpeg` (READY, 200GB, ffmpeg
  4.4.2 baked in) created from the VM disk BEFORE teardown, VM repo reset to clean committed HEAD
  first (lane's uncommitted schema patch stays Mac-side only) → DELETED 2026-07-08T~06:56Z,
  list-confirmed empty. Uptime ≈1.3-1.5h × $0.6-4.3/hr ≈ $0.8-6.5 (mid ~$2-3). Zero preemptions.
  in-VM idle watchdog (60min no-heartbeat -> shutdown) armed + never triggered (heartbeat touched
  every SSH round-trip).

- pickleball-h100-w5ball — DONE+DELETED (list-confirmed by lane, 2026-07-08T08:29:51Z): created
  06:14:38Z, SPOT-preempted 07:43:31Z (3m42s outage, disk-safe, watchdog-caught + manager restart),
  deleted 08:29:51Z; billed 2.192h ~ $1.25-9.32. Deliverables: stage1_official NOT-degenerate
  (queue #1 closed), seed_official honest-mixed, control exact, LoSO-mean all rows, 20/20 contract
  tests on VM. Gap: snapshot lacks corpus+rally videos (16min transfer tax) — next snapshot cut.

- pickleball-h100-w5fastbody — DONE+DELETED (list-confirmed, 2026-07-08T08:31:59Z): created
  07:32:45Z, 0.99h ~ $0.59-4.25 (mid ~$2) vs $15 owner cap, zero preemptions. VERDICT NOT-ADOPT
  (accuracy regression, kill pre-ruled; wall-clock net-slower per-clip). 3 footgun catches folded
  into w5_transport lane.
- TEMPLATE POLICY 2026-07-08: `pickleball-fleet-snap-20260708-ffmpeg` (READY, 200GB, ffmpeg 4.4.2,
  clean committed HEAD) SUPERSEDES pickleball-fleet1-snap-20260707 as the default boot template.
- TEMPLATE POLICY 2026-07-09: `pickleball-fleet-snap-20260708-w6close` (READY, 200GB disk / 45.6GB stored,
  ffmpeg + roboflow corpus + rally videos BAKED, verified-clean tree @ 37b8ecd3f) SUPERSEDES the ffmpeg
  snapshot as default boot template. Known by-design git-status lines: 2 vendor-submodule overlays
  (RESTORE.md); anything ELSE dirty on boot = reset --hard first (truncated-file class seen twice).
- No `pickleball-a100-w5proof` row yet: `runs/lanes/w5_closeproof_20260708/spec.md` is pre-staged,
  but no close-proof VM/report exists in this checkout at the w5_closedocs pass.

WAVE-4 FINAL FLEET ACCOUNTING (2026-07-08 close): ALL wave-4 VMs DELETED, list-confirmed by
manager + each lane. (1) pickleball-h100-w4body: 58m01s, $0.55-4.11 (BODY-validated 2.37×).
(2) pickleball-h100-w4ballgpu: created 17:43:49Z deleted 23:38:18Z, 5.90h, $3.4-25 honest span
($12-24 mid-band) — ~2.5h of uptime was transfer/outage-idle (macOS ssh EMSGSIZE flake + two
API stalls; nohup pattern preserved all VM-side work), zero preemptions. (3) w4fan1/2/3 (A100
snapshot fans): 1.824h each, 5.47 GPU-h, $6.0-7.1 total. WAVE TOTAL ≈ $10-36 (mid-band ~$19-32)
vs the stated ~$12-25 budget — top-band overage driven entirely by the outage/transfer idle;
per-VM ≤$5/hr and ≤4-concurrent caps NEVER breached. fleet1 remains STOPPED disk-intact
(snapshot source). Snapshot template gap: lacks ffmpeg (wave-5 recipe item).

Cap compliance: ≤$5/hr × ≤4 concurrent holds; wave budget ~$12-25 per wave4_boot_prompt.md.

| vm_name | zone | gpu | model | status (provisioning/idle/busy/preempted/tearing-down) | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| pickleball-h100-w5p22 | asia-southeast1-b | H100-80GB (a3-highgpu-1g) | SPOT | DELETED 2026-07-08T~06:56Z, list-confirmed | w5_p22latent_20260707 DONE | ~$0.6-4.3/hr; uptime ≈1.3-1.5h, ≈$0.8-6.5 mid ~$2-3 | 2026-07-08T~05:35Z | Created from `pickleball-fleet1-snap-20260707` with pd-balanced 200GB. P2-2 MHR decode wrapper + latent smoothing prototype; GATE 1a PASS 2.7e-5deg, GATE 1b measured-partial. Snapshot `pickleball-fleet-snap-20260708-ffmpeg` created before teardown after VM repo reset to clean committed HEAD. Evidence: BUILD_CHECKLIST [W5 P22 PHASE1 RULED + OWNER UNBLOCKS 2026-07-08]; 62d785ce3. |
| pickleball-h100-w5ball | asia-southeast1-b | H100-80GB (a3-highgpu-1g) | SPOT | DELETED 2026-07-08T08:29:51Z, list-confirmed | w5_ballretrain_20260707 DONE | ~$0.57-4.25/hr; uptime 2.192h, ≈$1.25-9.32 | 2026-07-08T06:14:38Z | SPOT-preempted 2026-07-08T07:43:31Z for 3m42s, disk-safe, watchdog-caught + manager restart. Delivered stage1_official non-degenerate, seed_official honest-mixed, control exact, LoSO-mean rows, and 20/20 VM contract tests. Snapshot gap: corpus+rally videos not baked, 16min transfer tax. Evidence: BUILD_CHECKLIST [W5 BALLRETRAIN PASS 2026-07-08]; `runs/lanes/w5_ballretrain_20260707/`. |
| pickleball-h100-w5fastbody | asia-southeast1-b | H100-80GB (a3-highgpu-1g) | SPOT | DELETED 2026-07-08T08:31:59Z, list-confirmed | w5_fastbody_bench_20260708 DONE | ~$0.57-4.25/hr; uptime 0.99h, mid ~$2 vs $15 cap | 2026-07-08T07:32:45Z | Verdict NOT-ADOPT: challenger was 1.31x slower full-stage per clip and regressed accuracy; zero preemptions. Three dispatch footguns folded into w5_transport. Evidence: BUILD_CHECKLIST [W5 FASTBODY BENCH 2026-07-08]; af16e27c7. |
| pickleball-a100-fleet1 | asia-southeast1-a | A100-SXM4-40GB (a2-highgpu-1g) | SPOT | STOPPED (wave-3 w3_fleetseed cycle 2026-07-07T05:44→05:58Z, 14m23s ~$0.30: restarted, repo SYNCED 5b9f132ee→2f7336598 via git-bundle w/ md5+git-status whole-tree proof (version_stamp.json in lane dir), WASB teacher dry-run 2 clips, STOPPED list-confirmed. NEW IP 35.240.183.195 (same host key/disk; old 34.143.175.207 stale in configs/ssh/a100_known_hosts + remote_body_dispatch DEFAULT_REMOTE_HOST — update pending w3_codesync land). Per-SKU seed (A100-40GB, WASB detector): 23.3-29.9s GPU/clip = 0.49-1.11× realtime) | w3_fleetseed_20260707 DONE (prev: p27a_gvhmr_spike_20260706 (prev: p06-ground DONE (4/4 fresh BODY re-dispatches: stance_aware ACTIVE on all 4 (anchor_source=placement_track_world_xy); foot-slide PASS wolverine ~0m + img1605 25.6mm, FAIL burlington 46.9mm + outdoor 40.5mm w/ root_motion_temporal_jump blockers; body gate PASS 2/4. VM left RUNNING per manager — fleet decision at closeout. Details: runs/lanes/p06_freshworlds_20260706/) | ~$1.1-1.3/hr (est., matches prior same-zone/same-shape SPOT rate per RESET_HANDOFF §7; well under $5/hr cap) | 2026-07-06T18:55:16Z | P0-1 cold start DONE: nvidia-smi OK, EXCLUSIVE_PROCESS set, vendor pins restored, 27/27 BODY pytest GPU tests pass (0 skipped), inference smoke produced pred_keypoints_3d w/ GPU util confirmed. IP 34.143.175.207. Full detail: `runs/lanes/gpu_coldstart_20260706/report.md`. P0-6 lane (2026-07-06) added SSH host key to configs/ssh/a100_known_hosts + a VM-local symlink (~/body_runtime/Fast-SAM-3D-Body/checkpoints -> ~/coldstart_20260706/body_runtime/checkpoints) so the checked-in models/MANIFEST.json's hardcoded checkpoint local_path resolves; ran 4 fresh clips, all `partial` (BODY blocked by a local-machine rsync/openrsync transport bug, NOT a VM problem -- VM confirmed healthy post-run, nvidia-smi 0% util/0 MiB). Full detail: `runs/lanes/p06_freshworlds_20260706/report.md`. Left RUNNING + idle per mission. |

| pickleball-a100-fleet2 | asia-southeast1-a | A100-SXM4-40GB (a2-highgpu-1g) | SPOT | DELETED 2026-07-07T04:49:57Z (lane self-teardown, list-confirmed) | p01b_prelabel_20260707 DONE (40/40 shards, 46.7 GPU-min, ~$1.3 total, uptime 1.104h) | ~$1.2/hr | 2026-07-07T03:43:41Z | Created by Sonnet lane w/ lane_vm_startup.sh (EXCLUSIVE_PROCESS + preemption watcher); labels fable-lane=p01b-prelabel,fable-fleet=pickleball. Lane DELETES VM at end + reports cost. |

| pickleball-a100-w3p11train | asia-southeast1-a | A100-SXM4-40GB (a2-highgpu-1g) | SPOT | DELETED 2026-07-07T08:25:39Z (lane self-teardown on H100 migration, list-confirmed; uptime 1.187h ≈ $1.4) | w3_p11_train_20260707 (superseded by H100 VM below) | ~$1.2/hr (est.; web spot-pricing check found a2-highgpu-1g in the $0.39-$1.99/hr range depending on source/region, and fleet1/fleet2's own prior billing in this exact zone/shape is $1.1-1.3/hr — both well under $5/hr cap) | 2026-07-07T07:14:21Z | SKU DECISION (cost-gate evidence at the time): full-region quota scan found ZERO H100 and ZERO A100-80GB quota, so fell through to A100-40GB. Did real work before migrating: cold-start validation PASS (driver 580.159.03, CUDA 13.0/torch cu129, GPU smoke loss 1.551→0.270 strictly decreasing, checkpoint round-trip sha256 match), TRUE zero-shot baseline computed on full internal_val (2640 samples: f1@20px=0.0615, median_error_px=167.96 — see runs/lanes/w3_p11_train_20260707/zero_shot_wasb_pretrained/), caught+fixed a real output_channels=1-vs-3 checkpoint-shape-mismatch bug via CLI override (`--output-channels 3`, not a repo edit), and proved `--resume-checkpoint` end-to-end (resumed cleanly from step 10→20, loss stayed in the same low band, no reset/blowup). No 12k-step training had started yet when the migration order arrived, so nothing real was lost. |
| pickleball-h100-w3p11train | asia-southeast1-b | H100-80GB (a3-highgpu-1g) | SPOT | DELETED 2026-07-07T09:03:12Z (lane self-teardown, list-confirmed; uptime 1.027h) | w3_p11_train_20260707 DONE (P1-1 warm-start COMPLETE: 12000 steps in 25.8 min, internal_val f1@20px 0.0615→0.6104 (~10x vs true zero-shot), median_error 167.9px→2.73px, heldout_touched=false, ckpts md5-verified local. FIRST-H100 cold-start validation PASS: driver 580.159.03/CUDA 13.0/torch 2.9.1+cu129, smoke loss 0.885→0.038 strictly-decreasing, checkpoint round-trip sha match, ~8 steps/s sustained ≈ 2.3x A100-40GB on identical workload. Artifacts: runs/lanes/w3_p11_train_20260707/) | ~$0.57-4.25/hr (web sources conflict: $0.5325/hr cloudprice.net vs ~$3.69/hr spheron/thunder us-central1 spot baselines, +7-15% ase1 premium; either way under $5 cap) × 1.027h uptime ≈ $0.59-4.36 this VM | 2026-07-07T08:01:36Z | QUOTA GOTCHA: region `describe` still showed ZERO H100 quota AFTER the owner's grant — real create succeeded anyway (propagation lag on the describe/read path). Lesson: after a fresh grant, attempt create as the definitive quota test. a3-highgpu-1g needs ase1-b or -c (NOT -a — no nvidia-h100-80gb accelerator there). HARNESS GOTCHA (booked for repo fix by a Codex lane): scripts/racketsport/train_ball_pretrain.py uses itertools.cycle(train_loader) which CACHES the entire first epoch (~5.7k batches ≈ 240GB at full corpus) — kills any full-corpus run on ANY SKU (FD exhaustion at default ulimit 1024 first, RAM/shm blowup next). Lane ran a 3-hunk patched COPY under its lane dir (repo source untouched; diff at runs/lanes/w3_p11_train_20260707/lane_patch.diff): cycle→non-caching re-iterating generator + ROOT hardcode; launch also needs ulimit -n 1048576 and --output-channels 3 (official WASB tennis ckpt final_layers have 3 channels; config default 1 fails state_dict load). |

QUOTA UNLOCK (owner-filed, approved 2026-07-07 ~12:47-12:50 AM): Preemptible NVIDIA H100 GPUs = 2/region
in asia-southeast1 + us-east4 + us-central1 + us-west1 + us-west4 + europe-west4; NVIDIA A100 80GB = 2/region
in asia-southeast1 + us-central1 + us-east4 + europe-west4 (us-west1/us-west4 denied). H100-first
(owner directive 2) is now ACTIONABLE — and VALIDATED 2026-07-07: first-fleet H100 cold-start PASS
(driver 580.159.03 / CUDA 13.0 / torch cu129; ~8 steps/s ≈ 2.3× A100-40GB on identical training
work; evidence runs/lanes/w3_p11_train_20260707/cold_start_validation.json). STANDING SKU POLICY:
H100-80GB spot = DEFAULT heavy worker for TRAINING-class jobs (a3-highgpu-1g lives in
asia-southeast1-b/-c, NOT -a; describe-quota lags admission control — attempt create as the
definitive test); A100-80GB = middle tier (quota exists, unused); A100-40GB = proven BODY-stage
default until H100 BODY-runtime compat is separately validated (wave-4 queue #7) — decisive gate
runs stay on proven SKUs. Fleet IPs RECYCLE across VMs on restart — always --remote-host; refresh
known_hosts + DEFAULT_REMOTE_HOST each restart (wave-4 queue #6). Snapshot→fan is the standard
multi-GPU pattern (template snapshot: pickleball-fleet1-snap-20260707, READY, 46.56GB).

Fleet cost cap (owner 2026-07-06): ≤$5/GPU/hr, max 4 concurrent GPUs; teardown/DELETE the moment a
lane ends (idle spend never acceptable); 5th GPU or >$5/hr = `needs-purchase-approval` STOP.
Auth: owner gcloud refresh token (hello@; SA key creation org-blocked); dead auth = typed STOP for one owner login.

| pickleball-calv1unet-a100-spot | asia-southeast1-a | a2-highgpu-1g A100-40GB | SPOT | DELETED 2026-07-08T23:45Z list-confirmed (lane COMPLETE: 2.87 GPU-h, ~$4.4-6, zero preemptions) | calv1_unet_train_20260708 | ~$1.1-1.5/hr | ckpt pulled+sha-verified |

WAVE-6 FLEET LOG (2026-07-08, manager):
- pickleball-h100-w6gate1b (H100 a3-highgpu-1g SPOT, asia-southeast1-b, pd-balanced 200GB from
  pickleball-fleet-snap-20260708-ffmpeg) — DONE+DELETED 2026-07-08T21:01Z list-confirmed (created 19:52Z, uptime 1.158h ~= $0.66-4.92, zero preemptions; 3 BODY arms + GATE-1b harness + 541/276 evidence + mesh-cap proof pulled w/ md5; TEMPLATE DIRTY ON BOOT: 5 truncated core files, reset --hard recovered — re-cut template next errand). (Sonnet lane w6_gpu_instrument_20260708,
  self-tearing provision→run→verify→DELETE→report). Mission: (1) snapshot-hygiene check (git status
  CLEAN on boot — gap on record) + sync to committed dd5e5980d + version-stamp verify; (2) wolverine
  raw-postchain instrument dispatch (GATE-1b: decode(emit) <=1mm, mesh-skel <=5mm p95 vs raw sidecar
  via 2db0d1b4e harness; roma importable check first); (3) 541/276 frame-count adjudication data
  (VM-side monolith; per runs/lanes/w6_instrudocs_20260708/frame_scheduling_adjudication.md);
  (4) pull artifacts + md5 to Mac lane dir; DELETE + list-confirm + cost. Budget: ~1-2h × $0.6-4.3/hr
  ≈ $1-8 (+50% contingency ceiling ~$12); in-VM 60-min no-heartbeat self-stop armed.
- pickleball-h100-w6close (H100 a3-highgpu-1g SPOT, ase1-b, pd-balanced 200GB from
  pickleball-fleet-snap-20260708-ffmpeg) — DONE+DELETED 2026-07-09T00:57Z list-confirmed (created 22:07Z,
  uptime 2.833h ≈ $1.61-12.04, zero preemptions, survived a Mac power-cycle via nohup+transcript-resume).
  All 4 sub-tasks delivered: (A) re-score 3 candidates x 20 clips OFFICIAL+LoSO — seed_official micro-F1
  0.5329/hFP 0.2255 WINNER, stage1_official 0.2971 BELOW control 0.3611 (disagreement-corpus caveat booked);
  (B) legitimate GATE-1b raw arm: gate_1a PASS 4.098e-05deg, gate_1b FAIL 262.35mm (<=1mm), mesh-skel FAIL
  53.50mm p95 (<=5mm) w/ provenance + scale/hand_pose flowing; (C) outdoor mesh byte-budget-400: 409/409
  eligible frames, 112.6MiB actual (-14% vs estimate), 5.21->21.32 mesh fps; (D) TEMPLATE
  pickleball-fleet-snap-20260708-w6close READY 45.6GB w/ roboflow corpus (6.9GB/110,749 files) + rally
  videos (1.09GB) BAKED, 5-truncated-file dirt reset, vendor-submodule lines adjudicated BY-DESIGN
  (third_party/pickleball_vendor_additions/RESTORE.md). WAVE-6 GPU TOTAL: $2.27-16.96 (mid ~$5-9); caps
  never breached (max 2/4 concurrent incl. the foreign CALV1 VM, which tore itself down). (Sonnet lane w6_close_errand_20260708,
  self-tearing). CONSOLIDATED CLOSE ERRAND: (a) label re-score GPU_RESCORE_COMMANDS.sh (3 candidates
  x 20 clips, OFFICIAL bridge + LoSO-mean w/ outdoor fold, control row); (b) TEMPLATE RE-CUT clean-tree
  + bake corpus/rally videos; (c) legitimate GATE-1b raw arm (fixed knob + canonical harness w/
  provenance); (d) mesh-cap outdoor budget-400 proof. Budget ~2-3h x $0.6-4.3/hr ≈ $2-13 (wave total
  may reach the ~$12-15 band — flagged in checkin); 60-min idle self-stop armed.
- FLEET NOTE 2026-07-08: pickleball-calv1unet-a100-spot (A100 spot, ase1-a) RUNNING = the CONCURRENT
  CALV1 court session's VM (their ledger row/teardown; global concurrency 2/4 with w6close).

| pickleball-calv1unet2-a100-spot | asia-southeast1-a | a2-highgpu-1g A100-40GB | SPOT | DELETED 2026-07-09T04:35Z list-confirmed (retrain PASS: 3.19h, ~$3.5-4.8, zero preemptions) | calv1_unet_retrain_20260708 | ~$1.1-1.5/hr | ckpt pulled+sha-verified |

WAVE-7 FLEET LOG (2026-07-09, manager):
- pickleball-h100-w7ball (H100 a3-highgpu-1g SPOT, ase1-b/-c ladder, pd-balanced 200GB FROM
  pickleball-fleet-snap-20260708-w6close) — PROVISIONING (Sonnet lane w7_ballretrain_20260709,
  self-tearing). Mission: control-row reproduction (0.3611/0.5991) -> 486-row seed LoSO anomaly
  re-run (0.6404 vs 0.6858, RULINGS R2e) -> ~100-step probe (budget formula: min(12000,
  45min*rate)) -> 4 fine-tune arms (seed_official/rawWASB/stage1_official +aug, seed_official
  -aug ablation) on the 1,121-row corpus (md5 37a5d43a...) -> OFFICIAL bridge + LoSO scoring w/
  control row = THE 1k label-checkpoint eval. Wall cap 5h; drop order D,B; budget ~$1.5-17
  (+50% contingency ceiling ~$26); in-VM 60-min idle self-stop; nohup everything; DELETE +
  list-confirm + cost honesty at end. NO promotion; VERIFIED=0.
- BOOT RECONCILE 2026-07-09: fleet-filter list = fleet1 TERMINATED only (ZERO running) — matches
  ledger. Foreign VM body4d-waker-ctrl (e2-micro, us-central1-a, cost-center=body4d, running
  since Jun 14) is NOT fleet — surfaced to owner as FYI in OWNER_CHECKIN_20260709.md.
- pickleball-h100-w7p22 (H100 a3-highgpu-1g SPOT, ase1-b/-c ladder, pd-balanced 200GB FROM
  pickleball-fleet-snap-20260708-w6close) — PROVISIONING (Sonnet lane w7_p22gate_20260709,
  self-tearing). Mission: decisive post-pred_cam_t-fix GATE-1b measurement — fresh wolverine
  raw arm w/ ALL raw sidecars preserved (pred_cam_t/pred_vertices/pred_keypoints_3d/
  pred_joint_coords) -> canonical gate_check_body_decode -> alternate-field quantification ->
  synthetic gate --decoder sam3d. Sync gate: md5-verified hmr_deep/mhr_decode/gate CLIs at
  >=22be98100 (the fix MUST be in the measured code). Wall cap 2h, ~$0.6-6.5 (ceiling ~$10),
  60-min idle self-stop, DELETE+list-confirm at end. Concurrency 2/4 with w7ball.
- pickleball-h100-w7p22 — DONE+DELETED 2026-07-09T07:18:23Z list-confirmed (created 06:31:45Z,
  uptime 0.777h, $0.47-5.05, zero preemptions, first-attempt ase1-b create, idle watchdog armed
  never fired). DECISIVE: GATE-1b unchanged post-fix (262.348mm/53.50mm p95, FP-identical to w6)
  BECAUSE the canonical harness itself omits pred_cam_t in its re-ground call
  (gate_check_body_decode.py:385 — code fact); raw pred_keypoints_3d grounded WITH cam_t = p95
  23.4mm/max 26.7mm vs persisted joints_world; pred_joint_coords NOT a drop-in (p95 328.7mm).
  sam3d synthetic-gate adapter = honest stub (unwired). SNAPSHOT GAP: w6close template lacks 2
  best_stack-pinned files (waveb calibration_curves.json + court_model_v2.pt 287MB) and
  remote_body_dispatch validates the whole manifest at import — re-bake at next snapshot cut.
  Evidence: runs/lanes/w7_p22gate_20260709/ (31MB, md5-manifested).
- pickleball-h100-w7masklet (H100 a3 SPOT from w6close snapshot) — PROVISIONING (Sonnet lane
  w7_masklet_20260709, self-tearing, time-boxed 2.5h). P2-4 masklet-conditioning ADOPT-ONLY-ON-WIN
  spike: baseline-vs-candidate BODY on wolverine, pre-ruled kills K1 setup>60min / K2 slower /
  K3 p95 divergence>10mm or schema breakage / K4 uptime>2.5h. Budget ~$1.5-11. Concurrency
  2-3/4 with w7ball. NO-ATTEMPT is a valid honest verdict if the evidence base lacks a concrete
  recipe (Mac-side check happens BEFORE provisioning).
- pickleball-h100-w7masklet — DONE+DELETED 2026-07-09T08:49Z list-confirmed (created 08:19:45Z,
  uptime 0.489h ~= $0.28-2.08 mid ~$1, zero preemptions). VERDICT NO-ATTEMPT on candidate arm:
  permission classifier denied executing third-party sam-body4d code on the VM x3 (named prereq:
  owner grant; HF checkpoints also gated; full pipeline = 5 model families, NOT a cheap spike;
  masklets-only A/B is the real cheap shape). BASELINE BANKED: 307.179s H100 BODY wall wolverine
  (reproduces w5_fastbody 316.05s within 2.8%) at runs/lanes/w7_masklet_20260709/. SNAPSHOT GAP
  re-confirmed (2nd lane): calibration_curves.json + court_model_v2.pt missing from w6close
  template — MUST bake at next cut. ~10min honest idle from 2 passive-wait turns (1 nudge used).
- pickleball-h100-w7ball — DONE+DELETED 2026-07-09T09:17:18Z list-confirmed (created 05:40:25Z,
  uptime 3.615h, $2.06-15.36 mid ~$4-8, zero preemptions, first-attempt ase1-b create, snapshot
  had corpus/videos/ckpts baked = near-zero transfer tax, code identity byte-exact 6/6 scripts).
  PARTIAL-honest: ARM0 control BIT-EXACT (0.361111/0.599089); ARM1 486-anomaly REPRODUCES
  (0.6868 vs 0.6426; lineage clean); probe 0.8787 steps/s -> 2372 steps/arm; arms A+C trained
  (B,D dropped per pre-agreed order at the 5h cap); 1k-CHECKPOINT CARD: A_seed_official_aug
  0.6152/0.2506, C_stage1_official_aug 0.6121/0.2597 vs control 0.3611/0.5991 — label curve
  NOT plateaued (+15.5% rel F1 from 486->1121 rows); hFP regressed (needs arm D to attribute).
  LoSO harness = per-CLIP folds not per-source grouping (pre-existing, flagged for ruling).
- pickleball-h100-w7ballc (H100 a3 SPOT from w6close snapshot) — PROVISIONING (Sonnet lane
  w7_ballcomplete_20260709, self-tearing, wall cap 4.5h, ~$2-15). Completion errand: on-VM
  contract check w/ PULLED log (closes verifier attack-4 gap) -> arms B (rawWASB+aug) + D
  (seed no-aug ablation) at 2372 steps on the 1121 corpus -> complete 5-row 1121 card + full
  re-score on the 1750 corpus (prepared block, control-first). Owner visibility ruling encoded
  (w6-session rows = box-position-only). Drop order if tight: bottom-up from 1750 re-score.
- pickleball-h100-w7ballc — DONE+DELETED 2026-07-09T12:39:49Z list-confirmed (created 09:41:36Z,
  uptime 2.953h, $1.68-12.55 mid ~$3-6, zero preemptions). PARTIAL-honest: contract-check log
  pulled+md5-verified BOTH SIDES (verifier gap CLOSED); ARM D landed via bit-exact
  checkpoint-resume after a self-reported concurrency mistake (D+B co-scheduled on one GPU ->
  ~3x throughput collapse, persistent — NEW STANDING GOTCHA: never co-schedule training arms;
  re-probe rate per fresh VM before trusting the budget formula); ARM B dropped at step 500
  (ckpt banked); 1750 re-score prepped not dispatched. AUG-ATTRIBUTION ANSWERED: D(no-aug)
  0.6241 F1/0.3554 hFP vs A(aug) 0.6152/0.2506 — aug = the hFP MITIGATION (labels alone drive
  hFP +57.6% rel; aug pulls back 29.5%). Recipe ruling: occlusion-aug STAYS.
- pickleball-h100-w7crit (H100 a3 SPOT from w6close snapshot) — PROVISIONING (Sonnet lane
  w7_critique_20260709, self-tearing, owner-waiting priority). OWNER CRITIQUE E2E: owner's own
  IMG_9545.mov (3m33s/30fps/223MB, never processed) COLD through the full rev-9 production stack;
  overrides per owner order: --body-skeleton-stride 1 (no frame skipping) + max mesh byte budget;
  fresh clip id, zero banked knowledge (verified). Deliverable: full viewer-servable world pulled
  to Mac + provenance report. Concurrency 2/4 with w7speed. ~1-1.5h, ~$0.6-6.5.
- pickleball-h100-w7speed (H100 a3 SPOT from w6close snapshot) — RUNNING (Sonnet lane
  w7_speedgate_20260709, dispatched ~09:5x post-reauth): ARM1 clean-room timing (bars now
  INFORMATIONAL per owner morning ruling) + ARM2 corrected GATE-1b numeric + ARM3 w7close
  template re-bake (2 artifacts + 1750 corpus baked) + post-timing critique-world pull (scope add).
- pickleball-h100-w7crit — DELETED 2026-07-09 ~20:5x by manager, list-confirmed (created 11:19 local,
  ~1.7h, ~$1-7). Owner critique E2E thread: zwCtH cold + tapped + metric15 runs (trust-chain gauntlet
  findings booked in NORTH_STAR dated note); worlds pulled to runs/lanes/w7_critique_20260709/.
- pickleball-h100-w7speed — DONE+DELETED 2026-07-09T21:39:40Z list-confirmed (created 17:49:42Z,
  uptime 3.833h, $2.18-16.29 mid ~$7.7, zero preemptions, first-attempt create). ARM1 wolverine x6
  full-stack 483.6-497.5s mean 489.4s stdev 5.5s (informational vs 400s bar per owner ruling; 4.4x vs
  2141s historical); BODY 78.6% of wall, steady inference only 5.5s/380s (=the efficiency-wave lever);
  outdoor x2 ~1515s. Foot-slide spread 4.1e-7m (GPU nondeterminism scale — tolerance ruling needed,
  not a regression). ARM2 corrected GATE-1b: pred_cam_t consumed 705/705 exactly-once, per-player p95
  22-58mm, 262.17mm = SINGLE worst-frame outlier player 20 (frame-level phenomenon; decode stays
  NOT-WIRING-READY). ARM3: pickleball-fleet-snap-20260709-w7close READY 46.2GB — bakes ALL THREE
  discovered manifest-pinned gaps (calibration_curves, court_model_v2.pt, ball latest.pt) + yolo26m.pt
  + ultralytics venv + 1750 corpus. INTEGRITY CATCH: first 6 timing runs invalid (tracking dead on old
  template) — lane discarded its own faster-but-false numbers; gate checks worked. VM-side known_hosts
  self-dispatch keyscan = boot-ritual candidate. FLEET NOTE: foreign pickleball-h100-ns06eff-bx1
  RUNNING under fable-fleet=pickleball = the NS-consolidation session's NS-06 lane (theirs, untouched).
- pickleball-h100-w7ball2 (H100 a3 SPOT, FIRST USE of pickleball-fleet-snap-20260709-w7close) —
  PROVISIONING (Sonnet lane w7_ballretrain2_20260709, self-tearing, wall cap 5h, ~$3-15).
  Owner-ordered retrain on the fresh 2,388-row corpus (md5 0ae65f01..., job_21=ball_session_03
  ingested +638): control-first -> per-VM probe -> ONE-ARM-AT-A-TIME: E=seed_official+aug on 2388
  (the ~2.4k curve point; w6/w7 rows box-position-only per owner ruling), B-complete from banked
  step-500, F=warm-from-A optional -> score all candidates on the new 40-clip block w/ true source
  groups. Template-bake validation = part of the mission. Drop order: F, B, C/D rows.
- pickleball-h100-ns06eff-bx1 — DONE+DELETED 2026-07-09T23:04:46Z, list-confirmed absent
  (created 21:22:01Z, ~1.71h, a3-highgpu-1g H100 SPOT, zero preemptions; ase1-c first attempt
  stockout, ase1-b succeeded). NS-06 speed/storage lane: matched Wolverine 244 BODY frames / 705
  player-frames; selected delta-v2 level-6 candidate cut indexed mesh 51,921,156 -> 21,538,180
  bytes (-58.5%) with index build 5.765s vs 22.068s prior baseline (-73.9%). Candidate full wall
  470.492s is scoped n=1, not a global speed promotion. Self-dispatch required H100 compute mode
  Default because the coordinator's CUDA context conflicts with Exclusive_Process; detailed import
  errors exposed that root cause. Evidence: runs/lanes/ns06_efficiency_20260709/.
- pickleball-h100-ns06cpu-bx1 — DONE+DELETED and list-confirmed absent by 2026-07-10T00:34:37Z
  (created 2026-07-09T23:38:34Z, <0.94h, a3-highgpu-1g H100 SPOT, zero observed
  preemptions). NS-06 CPU/storage follow-up: same-VM warm Wolverine control 502.810s pipeline /
  384.035s BODY phase / 15.34 GiB peak RSS; selected vector+clip-topology candidate 366.810s /
  241.090s / 15.18 GiB, with exact 244 BODY frames / 705 player-frames and scoped BODY plus
  grounding gates passing. Vector-only intermediate's 19.98 GiB RSS was rejected; topology
  interning recovered 4.80 GiB and another 49.56s BODY. Compute mode Default was required for
  self-dispatch. `VERIFIED=0`; evidence: runs/lanes/ns06_cpu_efficiency_20260709/.

WAVE-DEMO FLEET LOG (2026-07-10, demo_beststack lane):
- pickleball-h100-demo1 (H100 a3-highgpu-1g SPOT, ase1-b/-c ladder, pd-balanced 200GB FROM
  pickleball-fleet-snap-20260709-w7close) — PROVISIONING (Sonnet lane demo_beststack_20260710,
  self-tearing, wall cap 2h, ~$1-6). Mission: git-bundle sync to demo-beststack branch (fail-closed
  ball emission) -> compute-mode DEFAULT -> cold best-stack E2E wolverine + zwcth45s excerpt ->
  attestation (best_stack.resolved + fail_closed provenance) -> batched artifact pull w/ md5 ->
  DELETE + list-confirm + cost. Concurrency 2/4 with w7ball2 (foreign, 3K training).
- pickleball-h100-demo1 -- DONE+DELETED 2026-07-10T~03:02:30Z list-confirmed absent (created
  2026-07-10T02:20:11Z, uptime ~0.71h, a3-highgpu-1g H100 SPOT ase1-b, zero preemptions,
  first-attempt create). Sonnet lane demo_beststack_gpu_20260710: consumed best_stack rev 11
  (ball.world_emission_fail_closed WIRED_DEFAULT, demo-beststack-20260709 branch HEAD
  0c110deac26190e883af7cf5830aa5d4cb83ec01). Code sync via git bundle, 6/6 md5 identity match.
  Wolverine full-stack cold run (--body-local): wall 379.5s, all mandatory stages ran (partial
  only via input_quality degraded_input advisory), ball_arc_render.json
  fail_closed_enabled=true suppressed_segment_ids=[0,2,3,4,6,8] (exact commit-claim match),
  virtual_world/confidence_gated_world max ball z=0.968m (matches commit's claimed
  23.53m->0.968m exactly, n=75/300 frames). zwcth45s 45s harvest excerpt run twice (R1
  --remote-host missing-host degraded BODY; R2 --body-local hit a real 'missing BODY frame
  image' bug specific to this harvest clip) -- ball fail-closed IDENTICAL+reproducible both
  attempts (suppressed=[0,2,3,4,5], max z=1.69m) but BODY/paddle_pose never completed for this
  clip in either attempt (flagged for follow-up, distinct from the ball fix). Manager (Fable)
  took over parallel wolverine pull+QA-render work mid-lane
  (runs/lanes/demo_beststack_render_20260710/); this lane's own vm_pull/ holds zwcth45s R1+R2
  with full two-sided md5 verification (wolverine independently re-verified live via SSH, not
  re-pulled to this lane's dir). Cost ~$0.42-3.03 (0.71h x $0.6-4.3/hr H100 spot). Evidence:
  runs/lanes/demo_beststack_gpu_20260710/report.json.

- pickleball-h100-demo2 (H100 a3 SPOT, us-central1-a — ase1-b/-c STOCKOUT at attempt, snapshot is
  global so cross-region boot from pickleball-fleet-snap-20260709-w7close) — RUNNING (manager-run
  r3: zwcth45s stride-1 BODY rerun after demo1's early teardown race killed the first r3; manager
  tears down on pull). demo1 accounting: created 2026-07-10T02:22Z, deleted ~03:0xZ by lane,
  ~0.71h, ~$0.42-3.03, list-confirmed (lane report.json).

NS-014 FLEET LOG (2026-07-09/10, ns014 manager):
- pickleball-h100-ns014rescore (H100 a3-highgpu-1g SPOT, ase1-b/-c ladder, pd-balanced 200GB FROM
  pickleball-fleet-snap-20260709-w7close) — PROVISIONING (Sonnet lane ns014_gpu_rescore, self-tearing,
  wall cap 2.5h). Mission: fresh production-mode wolverine full-stack run at 8cd810a53 -> canonical
  gate_check_body_decode rescore w/ --attribution-report embedding -> attribute_body_decode_residual
  full-frame decomposition (grounding determinism / per-postchain-stage deltas / FK-vs-head) ->
  synthetic_body_decode_gate --decoder sam3d first real measurement (blocked_* = valid honest result)
  -> pull + md5 -> DELETE + list-confirm + cost. Budget ~1.5-2h x $0.6-4.3/hr ~= $1-9 (+50% ceiling
  ~$13); 60-min idle self-stop; compute-mode DEFAULT (ns06 self-dispatch finding). Concurrency 3/4
  with w7ball2 + demo2 (both foreign sessions').
- pickleball-h100-ns014rescore — DONE+DELETED 2026-07-10T04:43:51Z list-confirmed (created 03:25:07Z,
  uptime 1.312h, $0.79-5.64, zero preemptions; ase1-b stockout -> ase1-c first-attempt). Delivered:
  gate_1a exact reproduction; mesh-skel ~53mm p95 REPRODUCES ARM2 digit-close (stable); synthetic
  sam3d instrument FIRST REAL MEASUREMENT (3/3 valid detections; joints p95 313mm; mesh-skel 39.5mm);
  FK-vs-head ~0 (persisted params decode EXACTLY to pred_keypoints_3d — decisive). Replay-attribution
  arm CONFOUNDED (stale chunk index vs later self-dispatch body_mesh — 527mm p95 NOT valid evidence);
  gate_1b fail-closed blocked_missing_pred_cam_t as designed. STRUCTURAL CATCHES: --body-local never
  writes body_mesh monoliths (code-confirmed process_video.py:2734/5750/5805); remote-dispatch
  sync-back EXCLUDES fast_sam_subprocess/ (gate/attribution must run VM-side against the DISPATCH
  dir); coordinates.py StrEnum breaks py3.10 fleet venvs (repo fix in flight). FYI: demo2 VM vanished
  mid-lane (its own session's teardown; this lane issued no command at it). Second corrected arm
  planned (~$1-6). Evidence: runs/lanes/ns014_p22residual_20260709/gpu/pulled/ (13 files, md5 both-sides).

- pickleball-h100-demo2 — DONE+DELETED 2026-07-10T~04:0xZ list-confirmed (created ~03:33Z,
  ~0.5-0.6h, ~$0.3-2.6, zero preemptions; cross-region snapshot boot us-central1-a WORKS).
  r3 stride-1 rerun reproduced the SAME BODY failure as r2 stride-2 ("missing BODY frame image
  for frame 41" — frames stage never materializes it on this cold harvest clip regardless of
  stride) => reproducible cold-clip frames/BODY scheduling bug, 3 signatures banked in
  runs/lanes/demo_beststack_20260710/REPORT.md; follow-up lane queued. DEMO WAVE GPU TOTAL
  (demo1+demo2): ~1.2-1.3h, ~$0.7-5.7; caps never breached (max 2/4 concurrent).
- pickleball-h100-ns014rescore2 (H100 a3 SPOT, ase1-c-first ladder, FROM w7close snapshot) —
  PROVISIONING (Sonnet lane ns014_gpu_rescore2, self-tearing, wall cap 2h, ~$0.6-4.5 ceiling ~$7).
  CORRECTED arm-2 procedure: sync 4a3cbc60a (py3.10 fix + coherence guard) -> pre-seed VM self
  hostkey -> self-dispatch remote BODY (the body_mesh-writing route) -> attribution + gate against
  the DISPATCH-side run dir (fresh chunks + body_mesh same-execution; coherence guard active) ->
  pull + md5 -> DELETE + list-confirm. Concurrency <=3/4 (w7ball2 foreign still running).
  OUTCOME: BLOCKED-STOCKOUT 2026-07-10T05:05-05:07Z — ase1-c AND ase1-b both
  ZONE_RESOURCE_POOL_EXHAUSTED (inverse of arm 1's pattern 100min earlier; volatile SKU). $0 spent,
  zero resources allocated, list-confirmed. Evidence runs/lanes/ns014_p22residual_20260709/gpu2/.
  RETRY DISPATCHED as ns014_gpu_rescore3 w/ owner-granted broadened H100 zone ladder
  (ase1-c, ase1-b, us-central1-a, us-central1-b + one bounded 15-min second pass).
- pickleball-h100-ns014rescore3 — DONE+DELETED 2026-07-10T06:10:44Z list-confirmed (created
  05:18:01Z, uptime 0.879h, $0.53-3.78, zero preemptions, ase1-c first-attempt). Coherence guard
  LIVE-exercised (705/705 coherent on dispatch-dir pairing); gate_1b MEASURED (worst joints p95
  112.43mm, scale_source=None caveat vs ARM2 lineage); gate_1a + FK-vs-head exact reproductions;
  synthetic instrument stable replicate (<=0.5mm vs arm 1); mesh-skel 53.41 third reproduction.
  Per-stage replay table ruled VOID as production attribution (replay infidelity self-declared +
  foot-lock 20mm-cap impossibility) — production postchain total stays w7-armC 23.4mm p95. Full
  ruling: runs/lanes/ns014_p22residual_20260709/REPORT.md. ns014 lane GPU total ~2.19h ~$1.3-9.4.


WAVE-COURT FLEET LOG (2026-07-10, manager court session):
- pickleball-h100-court1 — DONE+DELETED 2026-07-10T13:15:53Z list-confirmed (created 11:27:42Z
  after 27 create attempts / ~65min ase1-b+c stockout — zero cost while failing; uptime 1.803h,
  $1.03-7.66 mid ~$2-4, zero preemptions). HONEST FAIL vs R2 kill bar + DECISIVE repo-gap find:
  court_model_v2.pt (court_unet_v2/resnet34/dict-output, trained by train_court_model_v2.py) is
  unloadable by train_court_keypoint_heatmap.py (hardcoded legacy archs, no init flag) — literal
  ARM-A/B impossible; substitute legacy-arch arms learned Roboflow-family real signal (val median
  423->13.5px) but 0.0 PCK@5 on owner cards (CARD-A median 197-226px). Control rows banked
  (frozen model: CARD-A 0.0 PCK/942px, CARD-B 0.0 PCK/675px). homography-refine = exact no-op at
  garbage-pred level (booked). court_external absent from w7close snapshot (bake next cut).
  Evidence: runs/lanes/court_train1_20260709/. (Sonnet lane court_train1_20260709, self-torn.) Mission: control-row-first court real-transfer probe ladder (frozen
  court_model_v2 control on harvest-GT + Burl/Wolv cards -> ~100-step probe -> ARM-A fine-tune
  on 3,921-row external partial corpus (Protocol S: no gate-source/eval-clip rows) -> eval w/
  homography-refinement variants -> optional ARM-B imagenet-init). Wall cap 3.5h, ~$2-15
  (ceiling ~$22), 60-min idle self-stop, DELETE+list-confirm at end. Code pin 497b64dbd.
- pickleball-h100-court2 — BLOCKED-STOCKOUT 2026-07-10T16:27Z, $0.00 billed, list-confirmed
  clean (52 create attempts / 110min ase1-b+c: 18 true stockouts + ~20 snapshot-clone
  'Operation rate exceeded' throttles — NEW failure mode, add inter-attempt backoff; 1 ghost
  STOPPING instance never RUNNING). Pre-flight banked: pin verified, v2-trainer + gate-eval
  court_unet_v2 dispatch confirmed. Retry = court_train4 w/ manager-authorized multi-region
  ladder within owner quota grant. (Sonnet lane court_train3_20260710.) Mission: court_unet_v2 real-transfer arms (the rung TRAIN-1 couldn't run):
  control-reproduction -> probe -> ARM-A init court_model_v2.pt / ARM-B imagenet resnet34 on
  the 3,921-row partial corpus (split-enforced) + synthetic 35% -> CARD-A/B + external val
  + homography variants + overlays. Wall cap 3h, ~$2-13 (ceiling $20). Code pin be5db7078.
- pickleball-h100-court2 — BLOCKED-STOCKOUT 2026-07-10T~16:27Z, list-confirmed absent, $0 cost
  (never reached RUNNING; never SSH-reachable). 52 create attempts across asia-southeast1-b/-c
  (26 each) over 110.25 minutes (2026-07-10T14:36:47Z -> 16:27:02Z), exceeding the spec's ~90min
  retry cap by ~20min before the lane stopped and reported. 18/52 genuine
  ZONE_RESOURCE_POOL_EXHAUSTED/STOCKOUT; ~20/52 a NEW failure mode -- "Operation rate exceeded"
  on the pickleball-fleet-snap-20260709-w7close snapshot clone itself (client-side throttle,
  distinct from zone capacity; worth watching if multiple same-day lanes boot from this snapshot
  concurrently). One anomalous instance briefly appeared (STOPPING) then vanished without ever
  reaching RUNNING/SSH-reachable -- consistent with async rollback of a rate-limited create, zero
  billing impact, confirmed via clean disk/instance list checks. Pre-flight verification banked
  without a VM: code pin be5db7078 = current Mac HEAD (zero diff); confirmed
  train_court_model_v2.py is a genuine separate court_unet_v2 trainer with --init-from-checkpoint/
  --encoder-weights-path reaching the resnet34 dict-output architecture (resolves TRAIN-1's
  decisive architecture-mismatch blocker); confirmed evaluate_court_keypoint_owner_gate.py already
  bridges to court_unet_v2 checkpoints via build_model_from_checkpoint's architecture dispatch;
  checkpoint/corpus/CARD-A/CARD-B source integrity re-verified against TRAIN-1's banked hashes.
  No training, no promotion, no best-stack delta. Evidence: runs/lanes/court_train3_20260710/REPORT.md.
