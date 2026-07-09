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
