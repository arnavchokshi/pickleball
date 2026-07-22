# GPU fleet ledger (live)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).
Full per-wave history (waves 4-7, NS-014, demo, court, 2026-07-12 sprint) is preserved verbatim in
`runs/manager/archive/gpu_fleet_history_20260707_20260712.md`.

## 2026-07-16 morning note (Track A manager)

gcloud auth DEAD again (reauth required; owner restoring). No fresh list possible until then. Last
live confirmation stands: EMPTY at 2026-07-16T02:50:53Z teardown (below). NOTHING was provisioned
since — overnight GPU spend $0 (conditional MOVE-1 #3 GO correctly NOT exercised: arc abstention
187/188 + CAL ingestion allowlist both said NO-GO). Exposure nil. Re-confirm with one list call
after owner reauth.

**UPDATE 2026-07-16T16:03Z (trk_detbench_20260716 lane):** auth is LIVE this session
(`gcloud auth list` → ACTIVE hello@swayformations.com; `gcloud compute instances list` succeeds) —
the "DEAD" note above is stale, superseded here.

## 2026-07-16T16:03-16:27Z trk_detbench_20260716 — NO-ATTEMPT (H100 a3-highgpu-1g SPOT stockout,
## all 6 zone-ladder attempts)

- Provision gate: fresh `gcloud compute instances list --filter=labels.fable-fleet=pickleball` at
  16:03Z → only `pickleball-a100-fleet1` (TERMINATED, historical snapshot source); 0 RUNNING fleet
  VMs; no pre-existing `trk_detbench` VM to reconcile. Gate PASS.
- 6 real `gcloud compute instances create pickleball-h100-detbench` attempts (a3-highgpu-1g,
  `--provisioning-model=SPOT --instance-termination-action=STOP`, boot disk pd-balanced 200GB
  `--create-disk=...,source-snapshot=projects/gifted-electron-498923-h1/global/snapshots/pickleball-fleet-snap-20260709-w7close`,
  labels `fable-lane=trk_detbench_20260716,fable-fleet=pickleball,owner=arnavchokshi`), 120s
  inter-attempt backoff, across the full spec'd ladder + one repeat: asia-southeast1-b,
  asia-southeast1-c, us-central1-a, us-central1-b, europe-west4-b, asia-southeast1-b (retry) —
  **every attempt returned `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` (reason: stockout /
  resource_availability)**, ~24 minutes wall (16:03Z→16:27Z), within the 30-min no-attempt cap.
- Post-ladder confirm: `gcloud compute instances list --filter="name~pickleball-h100-detbench"` →
  0 items; `gcloud compute disks list --filter="name~pickleball-h100-detbench"` → 0 items — no
  orphaned VM or disk from any of the 6 failed attempts. **Cost: $0.00. GPU-hours: 0.**
- Outcome: trk_detbench_20260716 STOPPED at the provision gate per spec's no-attempt rule (fence:
  "if the VM ladder no-attempts out, STOP with the evidence"). Zero benchmark arms run. See
  `runs/lanes/trk_detbench_20260716/report.json` for full per-attempt evidence.
- Fleet state after this lane: unchanged from before it ran — still EMPTY (0 running fleet VMs).
- **DISPATCH 2 (AMENDMENT 1, same day 16:31Z):** manager authorized SKU fallback ladder (2x H100
  quick attempts → A100-80GB → A100-40GB). Attempt 1 (H100 ase1-b) STOCKOUT again; attempt 2
  (H100 us-central1-a) **SUCCEEDED** — `pickleball-h100-detbench` RUNNING 2026-07-16T16:36:37Z.
  A100 tiers never needed. Row added to the live table above.

## 2026-07-17T05:47Z event_head_pretrain_20260716 CLOSE — VM DELETED, fleet EMPTY, ~$1.00-1.60 total

- **Teardown confirmed by the G2 manager personally:** `gcloud compute instances delete
  pickleball-t4-eventhead --zone=us-central1-b --quiet` EXIT 0 → `instances list
  --filter=labels.fable-fleet=pickleball` shows ONLY `pickleball-a100-fleet1` TERMINATED (historical
  snapshot source, untouched, label unchanged); `disks list` shows ZERO lane-created disks.
- **Cost accounting (est., not invoice-backed):** T4 spot band $0.2-0.4/hr. Instance 3 (the real
  worker) RUNNING 17:32Z→18:2xZ (~50min, idle-watchdog stop during the Mac freeze) + 03:33Z→05:47Z
  (~2.2h) ≈ 3.0h compute ≈ **$0.60-1.20**; instances 1+2 ~12min ≈ $0.04-0.08; **plus ~$0.26 disk**
  for the 200GB pd-balanced sitting through the ~9.5h freeze while TERMINATED. **Total ≈ $1.00-1.60
  against the $10 HARD cap** (the $15 coordinator-relayed raise was never honored — only the
  user/permission system authorizes spend).
- **12 stockouts across 2 ladders before success** (fleet1 ase1-a ×2, L4 ×8 across usc1/use1/usw1/
  euw4 — L4 was exhausted continent-wide that hour; T4 usc1-b succeeded on rung 8). **T4 is a
  legitimate first rung for decode-bound work, not a fallback.**
- **OPS LESSONS BOOKED (both cost real time this lane):**
  1. **Arm rails at BOOT via startup-script metadata, never via post-RUNNING ssh.** On fresh DLVM
     images the first-boot NVIDIA driver install owns the box for 5-8min; ssh arming raced it and
     fail-closed DELETEd a healthy VM. The boot-armed rail verified in 0s on the retry.
  2. **`nvidia-smi -c EXCLUSIVE_PROCESS` is fleet policy** (`scripts/fleet/lane_vm_startup.sh`) — a
     second concurrent CUDA process on one lane VM fails loud by design. Do not plan concurrent GPU
     passes on a single lane VM; serialize them.
  3. Mac-side `tar` injects AppleDouble `._*` files that broke a CSV glob on the VM
     (UnicodeDecodeError) — use `COPYFILE_DISABLE=1 tar` or strip on arrival.
  4. VM `cv2`/bundled ffmpeg **cannot decode AV1** — fetch/stage h264 at the source; verify with a
     10-frame decode check before training (30s check, saved 40min the hard way).

## Current fleet state (2026-07-16T02:51Z, Track A manager session close)

EMPTY — zero fleet VMs running or stopped except the historical `pickleball-a100-fleet1`
(TERMINATED, asia-southeast1-a, disk intact, snapshot source). LIST-CONFIRMED 2026-07-16T02:50:53Z
after `pickleball-h100-pbv11r` DELETE; disks list confirms 0 lane-created disks remain (only
body4d-waker-ctrl 30GB non-fleet + pickleball-a100-fleet1 200GB historical). Non-fleet
`body4d-waker-ctrl` e2-micro RUNNING in usc1-a, untouched.

| vm_name | zone | gpu | model | status | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| pickleball-gpu-conf030 | asia-southeast1-c | A100-80GB | a2-ultragpu-1g SPOT | DONE+DELETED 2026-07-17T05:03:30Z (list-confirmed; disks 0) | trk_rfdetr_prod_20260716/vm_conf030 (Track F, PREREG_conf030 single-shot) | spot band ~$1.5-2.5 | 2026-07-17T04:54:30Z | wall **0.15h** → est **$0.22-0.38** (cap $2). Rail armed+verified 04:56:09Z (poweroff 05:41:08 UTC). Env gate PASS (~3e-11). **PREREG RESULT: FAIL** — conf030 wolverine 0.7780/0.6767 + 2 sw + 16 spectFP (WORSE than 0.18 floor's 1/4: surviving spectators are high-conf); burlington clean+material 0.9234/0.9850. One shot, no iteration, per prereg → coordinator's 2b. Pull md5 both sides 55c956715663c236b4e1d4b441813151. See runs/lanes/trk_rfdetr_prod_20260716/vm_conf030/ |
| pickleball-gpu-rfdetrflip | us-central1-a | A100-80GB | a2-ultragpu-1g SPOT | DONE+DELETED 2026-07-17T04:43:08Z (list-confirmed; disks 0) | trk_rfdetr_prod_20260716/vm_rerun (Track F, owner-directed) | spot band ~$1.5-2.5 | 2026-07-17T04:31:00Z | wall **0.20h** → est **$0.30-0.50** (cap $5). Rail armed+verified 04:33:55Z (`shutdown -P +100` → poweroff 06:13:54 UTC, proof in lane log). Gate arm0a PASS (~3e-11 both clips — VM score-faithful where Mac was not). POOLDIAG M4 CONFIRMED end-to-end (YOLO26m @ conf .18/imgsz 960 through per-frame feeder reproduces frozen pins EXACTLY, Δ=0.000000). RF-DETR-L variant P: burl 0.9220/0.9933 clean; wolv 0.8036/0.7233, 1 sw + 4 spectFP (down from F's 16, not zero). Pull md5 both sides 0df9955dc38443841851afbdc7876801. Ladder: usc1-a H100 stockout, ase1-b H100 revoked mid-STAGING (0 orphans), usc1-a A100-80 success. See runs/lanes/trk_rfdetr_prod_20260716/vm_rerun/ |
| pickleball-h100-detbench | us-central1-a | H100-80GB | a3-highgpu-1g SPOT | DONE+DELETED 2026-07-16T17:16:41Z (list-confirmed; disks 0) | trk_detbench_20260716 (dispatch 2, AMENDMENT 1) | spot band ~$2.2-3.7 | 2026-07-16T16:36:37Z | wall 0.67h → est **$1.5-2.5**. Rail WAS armed+verified (shutdown -P +210, proof in lane log 16:39:56Z) + 60-min heartbeat self-stop unit. All 6 arms ran + scored; artifacts pulled two-sided md5 4ccc6129... See runs/lanes/trk_detbench_20260716/{report.json,DECISION_TABLE.md} |
| ~~pickleball-t4-eventhead~~ | us-central1-b | T4 | n1-standard-8 SPOT | **DELETED 2026-07-17T05:47:5xZ — list-confirmed (only historical a100-fleet1 TERMINATED remains) + disks list confirms ZERO lane disks (body4d-waker-ctrl 30GB non-fleet + a100-fleet1 200GB historical only). FLEET EMPTY.** | event_head_pretrain_20260716 (Track G2, slot 2-of-2 per owner directive) | spot band ~$0.2-0.4 | 2026-07-16T17:32:29Z | AMENDMENT-2 railed re-create after: 12 stockouts across 2 ladders (attempt-1 NO-ATTEMPT $0.00), T4 instance 1 fail-closed DELETE at 17:26:50Z (ssh rail-arm raced DLVM first-boot driver install, 480s window), instance 2 discarded unrailed pre-amendment; instance 3 arms its OWN rail at boot via startup script — RAIL_ARMED verified 17:34:01Z (+330 poweroff scheduled + idle watchdog pid 1134, verify latency ~0s); spend so far ~$0.04-0.08 vs $10 HARD cap (user-authorized; $15 relay not honorable); DELETE + list+disks confirm at lane end; Mac MPS insurance train live (killed at GPU TRAIN_STARTED). OPS LESSON booked: arm rails at boot via startup script, never via post-RUNNING ssh on fresh DLVM images. NOTE 17:31Z reconcile: detbench absent from live list — Track F teardown presumed complete (their row to close) |

## 2026-07-15/16 pbv11_headtohead RE-RUN — CLOSED (partial; VM deleted + confirmed)

- pickleball-h100-pbv11r (H100 a3 SPOT, usc1-a on attempt 3/6 after ase1-b/-c stockouts): RUNNING
  22:53Z (wall_cap_start 22:56:20Z) → manager SIGINT + DELETE 2026-07-16T02:50:53Z, list-confirmed,
  disks 0. Wall 3.93h, spot band $2.2-3.7/hr → est **$9-15** (not invoice-backed). Under the $20 guard.
- Run identity all verified two-sided: pin ac0b14ab0, bundle 6e567499e8…, video 272a2132…, OSNet
  2809d322… (snapshot gap: torchreid also had to be pip-installed — add BOTH to next snapshot re-bake).
- OUTCOME: full-stack run STALLED in `ball_arc` (segment 7 candidate-association RK4, 3h06m
  in-stage, three concurring stack captures) — the 41-rally 3D head-to-head is again NO-RESULT.
  **DATED CORRECTION to the 2026-07-14 rows above/archive: the 07-13 attempt's death, attributed to
  the Fable spend limit, most likely hit this SAME ball_arc stall first (it "reached BVP solver
  phase" and never emerged). The blocker is a code scaling defect, not budget/auth.** Fix lane
  spec'd (NOT dispatched): runs/lanes/ballarc_scale_guard_20260715/spec.md.
- Salvage (two-sided md5, 26 files): full-game 2D ball chain (ball_track/candidates/bounces/size),
  calibration (auto-preview POOR — metric world fail-closed), logs + stall evidence + 2D scorecard
  under runs/lanes/pbv11_headtohead_20260713/rerun_20260715/.
- OPS LESSONS (booked): (1) a lane's promised in-VM self-stop MUST be verified armed by the manager
  at dispatch — it was not armed; Mac slept; manager had to arm `sudo shutdown -P` mid-run as the
  rail. (2) Mac-side watchers die on laptop sleep — the VM-side rail is the only real cost bound.
  (3) SIGINT does NOT write PIPELINE_SUMMARY.json (KeyboardInterrupt escapes the runner) — per-stage
  timing had to come from artifact mtimes.

## 2026-07-17 Track F close — ALL TRACK F VMs TORN DOWN, fleet clean

Three Track F sessions today, all DELETE + instances-list + disks-list confirmed, all under cap:

| vm | sku/zone | wall | est $ | outcome |
|---|---|---|---|---|
| pickleball-h100-detbench | H100 a3-highgpu-1g SPOT, usc1-a | 0.67h | $1.5-2.5 | zero-shot detector card (4 arms + baseline reproduction) |
| pickleball-gpu-rfdetrflip | A100-80 a2-ultragpu-1g SPOT, usc1-a | 0.20h | $0.3-0.5 | env-fidelity PASS + pooldiag SOLVED + RF-DETR-L variant-P card |
| pickleball-gpu-conf030 | A100-80 a2-ultragpu-1g SPOT, usc1-a | 0.15h | $0.22-0.38 | preregistered conf-0.30 single shot: FAILED (decisive negative) |

Track F total: ~1.0 GPU-hours, **~$2.0-3.4**. Zero orphans; zero idle spend. Track G's
pickleball-t4-eventhead was RUNNING throughout and was never touched by any Track F lane.

OPS NOTES BANKED: (1) H100 a3-highgpu-1g SPOT stocked out fleet-wide repeatedly 2026-07-16
(6/6 zones, then 2 more attempts) — the A100-80 tier absorbed every Track F run at lower cost;
consider A100-first for light-inference lanes. (2) One attempt showed brief STAGING then
capacity-revoked stockout w/ auto-clean (verified 0 orphans) — describe-before-proceed caught it.
(3) SNAPSHOT RE-BAKE list grows: OSNet ckpt + torchreid (already known) + `rfdetr` package
(needed by any future detector lane). (4) On-VM `sudo shutdown -P +N` rail armed as the FIRST
ssh action, with the scheduled-time line captured to the lane log, worked on all three VMs and
is now standard for Track F lanes.

## Standing policy (owner-set)

- **Cost cap:** ≤$5/GPU/hr; max FIVE concurrent (owner raise 2026-07-12; 6th GPU or >$5/hr =
  needs-purchase-approval STOP); DELETE + list-confirm the moment a lane ends; idle spend never OK.
- **SKU:** H100-80GB spot = default heavy worker (BODY-validated 2.37x A100). a3-highgpu-1g lives in
  ase1-b/-c NOT -a; describe-quota lags admission control — attempt create as the definitive test.
  Stockout ladder: ase1-b/-c -> us-central1-a/-b -> europe-west4-b with 120s inter-attempt backoff
  (prevents snapshot-clone "Operation rate exceeded" throttling). A100-80GB = middle tier;
  A100-40GB = proven fallback. Decisive gate runs stay on proven SKUs.
- **Quota (owner-filed 2026-07-07):** spot H100 2/region ase1+use4+usc1+usw1+usw4+euw4;
  A100-80GB 2/region ase1+usc1+use4+euw4.
- **Boot template:** `pickleball-fleet-snap-20260709-w7close` (READY 46.2GB: ffmpeg, roboflow corpus,
  rally videos, calibration_curves.json, court_model_v2.pt, ball latest.pt, yolo26m + ultralytics
  venv, 1750-row corpus baked). KNOWN GAPS — re-bake at next cut: OSNet ReID ckpt (best_stack rev-11
  requires it; missing-hit 2x), torch 2.5.1 predates train_court_model_v2's >=2.6 DataLoader(in_order=).
- **Boot ritual:** reset --hard if dirty beyond the 2 by-design vendor-submodule lines; fresh
  ssh-keyscan SELF-entry into configs/ssh/a100_known_hosts AFTER every checkout/reset (tracked file
  gets overwritten); compute-mode DEFAULT for self-dispatch lanes; use python3 (bare `python` not on
  fresh-VM PATH); in-VM 60-min no-heartbeat self-stop armed on every lane.
- **Auth:** owner gcloud refresh token (hello@); SA key creation org-blocked; dead auth = typed STOP
  for one owner login. Fleet IPs RECYCLE across restarts — always --remote-host + refresh known_hosts.

## 2026-07-13 pbv11_headtohead lane — RESUMED (owner reauthed; manager-verified list works)

- RECONCILE at resume: live list shows ONLY pickleball-a100-fleet1 TERMINATED under fable-fleet=pickleball
  (matches ledger); fleet RUNNING count 0/5 -> provision gate PASS. (Non-fleet VM body4d-waker-ctrl RUNNING
  in usc1-a is NOT a pickleball fleet VM; untouched.)
- pickleball-h100-pbv11 (H100 a3-highgpu-1g SPOT, ladder ase1-b/-c, usc1-a/-b, euw4-b, pd-balanced 200GB
  FROM pickleball-fleet-snap-20260709-w7close, 120s backoff, 6-attempt/30-min no-attempt cap) — PROVISIONING
  (Sonnet lane pbv11_headtohead_20260713, self-tearing, wall cap 5h from RUNNING, 60-min idle self-stop,
  compute-mode DEFAULT). Mission: MOVE 1 baseline head-to-head — ONE full promoted-stack run (best_stack
  defaults, --body-local, audio ON) of the 697s pb.vision demo video (sha 272a2132..., R&D reference ONLY),
  fresh content-addressed generation, code pinned to 541f89d9a160eca8498a7b7419a7c2bc7f5b4a0e via git bundle
  (sha fe9191b0dda0...a508), per-stage timings, pull + two-sided md5, then Mac-side per-rally
  compare_vs_pbvision scorecard + owner union event set. Denylist scan 1 CLEAN (pre-copy). DELETE +
  list-confirm + cost at end no matter what.

## 2026-07-13 pbv11_headtohead lane — STOP at provision gate (auth dead)

- pbv11_headtohead_20260713: STOP before any provision, 0 VMs created, $0 cost. `gcloud compute instances list` (and application-default token refresh) failed with 'Reauthentication failed. cannot prompt during non-interactive execution' for the fleet account hello@swayformations.com (project gifted-electron-498923-h1, correct active config). Live reconciliation of this ledger's 'EMPTY, zero running VMs' claim could NOT be performed this session — treat it as UNVERIFIED, not freshly confirmed, until the owner reauths and a fresh `gcloud compute instances list --filter=labels.fable-fleet=pickleball` is run. Needs ONE interactive `gcloud auth login` (owner) before this lane or any GPU lane can resume. See runs/lanes/pbv11_headtohead_20260713/report.json for full evidence.

## Most recent wave (2026-07-12 sprint — full rows in archive)

- pickleball-h100-trka: DONE+DELETED 2026-07-12T20:27Z list-confirmed, 1.655h ~$2-3.5
  (TRK ReID/apron margin sweep; margin 0.5/1.0 survive internal, 2.0 rejected).
- pickleball-h100-bodyc: DONE+DELETED 2026-07-12T22:17Z list-confirmed, 1.37h ~$0.8-5.8
  (BODY overhead levers all 3 honest-rejected; found world-stage 122s cost attribution).

- pickleball-h100-pbv11 (H100 a3 SPOT, ase1-b) — head-to-head lane; STOP'd once on dead auth, resumed
  after owner re-auth, reached BVP solver phase, then the driving process DIED on the Fable-5 monthly
  spend limit before writing any scorecard. VM left TERMINATED (spot stop). Manager DELETED it 2026-07-14,
  disks list-confirmed 0. No head-to-head result produced — re-run per runs/HANDOFF_20260714.md.
  FLEET NOW EMPTY.

## 2026-07-19 event_head_corpus leg

- pickleball-gpu-evhead (a2-highgpu-1g, 1x A100-SXM4-40GB, SPOT, instance-termination-action=STOP,
  labels fable-lane=event_head_corpus/fable-fleet=pickleball/owner=arnavchokshi, us-central1-a,
  image pytorch-2-9-cu129-ubuntu-2204-nvidia-580/deeplearning-platform-release, 200GB pd-balanced) —
  created 2026-07-20T05:48:24Z after reuse-start of pickleball-a100-fleet1 hit
  ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS (stockout) in asia-southeast1-a; ladder then hit stockout
  again in asia-southeast1-b before winning on rung 3 (us-central1-a, first attempt). $/hr est
  $1.1-1.5 (A100-40 spot band per SCALE_UP_SPEC cost table). Rail: `sudo shutdown -P +300` armed
  05:59:22Z, verified via /run/systemd/shutdown/scheduled (poweroff, wall message "lane rail:
  event_head_corpus staging, 5h wall", scheduled 2026-07-20T10:59:22Z) — still armed at staging
  close (07:01:28Z check). nvidia-smi: A100-SXM4-40GB, driver 580.159.03, CUDA 13.0. Disk 194G/168G
  free at boot (14%), 154G free at staging close. VM HEAD == Mac HEAD (1770d9d46) at clone time (repo
  cloned fresh, no prior mirror; Mac HEAD had advanced 24a4d4257->1770d9d46 via unrelated concurrent
  activity during this lane, not caused by this lane).
- STAGING-DONE. jhong93/spot: re-probed all 28 (6-day-stale probe) -> 27 LIVE / 1 DEAD (634UMLDrVzc,
  usopen_2015_mens_final_federer_djokovic, was LIVE 07-13, now removed). Fetched the 21 live+unstaged
  videos via yt-dlp `-S "res:360,vcodec:h264"` (0 retries needed, 0 failures) = 11,468,405,092 bytes
  (~10.7GB). OpenTTGames: fetched the 10 unstaged games (game_1/2/3/5 train + test_1/3/4/5/6/7) — no
  yt-dlp resolution ladder exists for these (direct lab.osai.ai HTTP files, not YouTube), so fetched
  full-res (31.2GB combined, HEAD-checked, bandwidth-probed ~24MB/s / 190Mbps) then ffmpeg-transcoded
  to <=360p h264 (scale=-2:360, libx264 veryfast crf23) and deleted the raw intermediate per-file =
  120,055,747 bytes (~114MB) final. All 31 fetched files (21+10) decode-verified true (cv2 open + 10
  frames) on first attempt, 0 failures, 0 retries triggered. GOTCHA: first OpenTT fetch script hung
  the `while read < file` loop after 1 item — `ffmpeg` without `-nostdin`/`</dev/null` on inherited
  fd0 silently consumed the loop's remaining input lines; fixed with `-nostdin` + explicit `</dev/null`
  on curl/ffmpeg/python3, relaunched for the remaining 9 games. Total staged this lane: 31 files,
  11,588,460,839 bytes (~10.8GiB). Total lane wall (VM create to staging close): ~73 min ->
  cost-so-far est **$1.3-1.8** (well under the $5 staging cap; math: 1.22h * $1.1-1.5/hr).
  Manifest `staged_media_manifest.json` written VM-side (~/vm_staging/), copied to
  `runs/lanes/event_head_corpus_20260719/vm_staging/staged_media_manifest.json`, two-sided sha256
  MATCH (96c2c1484430c9fc09489229689a6b4ac8afc0a377ae314193b14b2ead429432). VM LEFT RUNNING (rail
  armed, training dispatch follows within the 5h window) — not deleted, not stopped.
- Non-fleet note (report-only, no action taken): `body4d-waker-ctrl` (us-central1-a, unlabeled,
  RUNNING, cost-center=body4d;role=wake-controller;workload=wp23) machine type **e2-micro** (2 shared
  vCPU, 1024MB) — est **~$0.008-0.01/hr** on-demand (e2-micro list price), i.e. negligible run cost;
  flagging per lane instructions for the owner to label/decide, untouched otherwise.

## 2026-07-20 body4d-waker-ctrl DELETED (owner-authorized)
- `body4d-waker-ctrl` (e2-micro, us-central1-a, non-fleet, cost-center=body4d/role=wake-controller/
  workload=wp23, RUNNING since 2026-06-14) DELETED 2026-07-20 per owner ("idk what the body4k waker
  is u can get rid of that"). Confirmed orphaned: every repo reference historical (late-June
  body_unblock runs + archived docs); nothing current depends on it; ~5 weeks idle at ~$0.01/hr.
  gcloud delete exit 0; list-confirmed gone. Fleet now: pickleball-gpu-evhead RUNNING (active
  event_head_corpus training VM) + pickleball-a100-fleet1 TERMINATED (historical coldstart, stopped
  spot — ~200GB disk still incurs small standing cost; flag to owner as optional cleanup, NOT
  deleted without explicit go).

## 2026-07-20 event_head_retrain lane — PROVISIONING

- pickleball-gpu-retrain (a2-highgpu-1g, 1x A100-SXM4-40GB, SPOT, instance-termination-action=STOP,
  labels fable-lane=event_head_retrain/fable-fleet=pickleball/owner=arnavchokshi, ladder
  asia-southeast1-a/b -> us-central1-a/b/c/f, image pytorch-2-9-cu129-ubuntu-2204-nvidia-580/
  deeplearning-platform-release, 200GB pd-balanced, startup-script scripts/fleet/lane_vm_startup.sh)
  — attempt in progress. Mission: re-verify the T17 weighted-loss fix (commit 5adaf396c, on main)
  escapes all-negative event-head collapse. $12 hard cap, 5h boot shutdown rail + 25min idle
  watchdog armed via startup script. Reuses local runs/lanes/event_head_corpus_20260719/vm_pull/
  train/last_event_head.pt (step-9000 seed) and manifest SHA
  e53954ef9ca7336b1d694586185288e7112aa4b56690abc24df13087a922ce84 (re-staged fresh on this VM,
  asserted byte-identical).
- pickleball-gpu-retrain CREATED 2026-07-20 in us-central1-a (asia-southeast1-a and -b both hit
  ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS stockout first; us-central1-a won on first attempt of
  rung 2). a2-highgpu-1g SPOT, external IP 34.9.63.251, image pytorch-2-9-cu129-ubuntu-2204-nvidia-580,
  200GB pd-balanced. Startup script used a locally-amended copy of scripts/fleet/lane_vm_startup.sh
  (the checked-in file is a bare P0-1 scaffold with no shutdown rail or idle watchdog) that prepends
  the VM_RUN_PLAN.md-mandated `sudo shutdown -P +300` hard 5h wall rail (fail-closed on
  /run/systemd/shutdown/scheduled) plus a 25-min idle watchdog polling for
  train_event_head.py/eval_event_head.py/build_event_head_dataset.py/yt-dlp/ffmpeg/rsync activity,
  then folds in the scaffold's compute-mode/preemption-watcher steps; local copy only, not committed.

## 2026-07-20T16:13:54Z pbv_replay_20260720 lane — PROVISIONED

| pickleball-gpu-replay | asia-southeast1-c | A100-80GB | a2-ultragpu-1g SPOT | RUNNING | pbv_replay_20260720 (GPU slot 2-of-4, drill-session pb.vision replay E2E) | spot band ~$1.5-2.5/hr | 2026-07-20T16:13:54Z | ladder: us-central1-a A100-80 stockout (attempt 1), asia-southeast1-c A100-80 SUCCESS (attempt 2); boot disk = source-snapshot pickleball-fleet-snap-20260709-w7close (200GB pd-balanced, pre-baked pipeline venv+ultralytics+ffmpeg+corpus); startup-script = locally-amended runs/lanes/pbv_replay_20260720/scripts/lane_vm_startup_railed.sh (boot-armed `shutdown -P +180` 3h hard wall + 25-min idle watchdog polling process_video.py/ffmpeg/curl/gdown/git/pip activity, NOT committed); mission = full production process_video.py E2E on pb.vision Drill Session clip xkadsq9bli3h (186.015s, compare-only hold-out, owner-signed full usage rights), ball_arc external 20-min hard cap enforced by manager log-polling (kill+rerun --no-ball-arc on breach), $8 cap. Concurrent with pickleball-gpu-retrain (event_head_retrain lane, us-central1-a, untouched, not this lane's resource). See runs/lanes/pbv_replay_20260720/. |

## 2026-07-20 p0i_scorecard_20260720 lane — PROVISIONED

| pickleball-gpu-p0icard | us-central1-a | A100-40GB | a2-highgpu-1g SPOT | RUNNING | p0i_scorecard_20260720 (GPU slot 3-of-4, P0-I selection-layer frozen 2-clip scorecard, commit 881280045) | spot band ~$1.1-1.5/hr | 2026-07-20 (created on first zone-ladder attempt, us-central1-a) | image pytorch-2-9-cu129-ubuntu-2204-nvidia-580/deeplearning-platform-release, 200GB pd-balanced boot; fresh git clone (NOT the coldstart snapshot) pinned to 881280045; cap $4/90min hard; startup-script = checked-in scripts/fleet/lane_vm_startup.sh (bare scaffold, no rail — manager arms `sudo shutdown -P +90` by hand post-boot per lane spec); mission = env-fidelity gate (variant-P burlington/wolverine reproduction to 1e-9) then the committed player-selection layer scored once, no tuning. Concurrent with pickleball-gpu-retrain (event_head_retrain, us-central1-a) and pickleball-gpu-replay (pbv_replay_20260720, asia-southeast1-c), both untouched. |

## 2026-07-20T16:46Z p0i_scorecard_20260720 CLOSE — VM DELETED, fleet reconciled, ~$0.24-0.33

- **DONE+DELETED**: `pickleball-gpu-p0icard` (us-central1-a, A100-40GB SXM4, a2-highgpu-1g SPOT)
  created ~16:32Z, rail armed 16:33:49Z (`shutdown -P +90`), deleted 16:46:21Z. Wall **0.22h** ->
  est **$0.24-0.33** (cap $4, well under). List-confirmed 0 instances + 0 disks named
  pickleball-gpu-p0icard; fleet-wide list shows only pickleball-gpu-retrain (event_head_retrain,
  untouched) + pickleball-gpu-replay (pbv_replay, untouched) + historical pickleball-a100-fleet1
  TERMINATED.
- **Mission: SCORED, RESULT = DECISIVE FAIL.** ENV-FIDELITY GATE (variant-P burlington/wolverine
  through the unmodified production path) reproduced all 10 registered scalars at delta 0.000e+00
  (byte-exact). Selection arm (committed P0-I layer, commit 8812800459361ee6a9e0781700d8d59e725ea9b7,
  scored via the SAME frozen scorer, one shot, no tuning) **catastrophically regressed both clips**:
  wolverine spectFP 4->651 (target 0), switches 1->22 (target 0), farFP 0->969 (target 0), IDF1
  0.8036->0.4046, cov4 0.7233->0.0033; burlington spectFP 0->7783, switches 0->9, farFP 0->7868,
  IDF1 0.9220->0.2878, cov4 0.9933->0.0. Selection-OFF byte-identical to env-fidelity tracks.json
  confirmed both clips (sha256 match). interpolated:true markers present (burlington 13, wolverine
  8) per spec. **Root cause found and pinned**: `threed/racketsport/player_selection.py:1764`
  returns `slot_players + unbound_players` into the exported product `tracks.json` "players" list —
  every raw-pool fragment the four-slot enrollment leaves unbound (`leave_unbound` decision, audit
  `selection_state="unbound_abstention"`) is ALSO serialized as its own top-level scoreable player
  instead of staying report-only metadata. Burlington exported 186 players (4 bound slots + 182
  unbound fragments) against 4 GT players; wolverine exported 42 (4 bound + 38 unbound). The
  distinguishing `selection_state` field exists in `player_selection_report.json`'s `tracks` audit
  array but is NOT carried onto the entries in the actual exported `tracks.json` "players" list, so
  the frozen scorer (and any other consumer reading the product artifact directly) has no way to
  filter them out. Scorer itself is clean: 0 recorded errors, both clip rows present in both the
  env-fidelity and selection score reports. Full evidence + hashes:
  runs/lanes/p0i_scorecard_20260720/vm_pull/ (27/27 files sha256-verified two-sided, tarball hash
  match). **Manager should NOT flip best_stack; NOT a real win as committed; route back to the P0-I
  owner for a fix (drop or clearly out-of-band-flag unbound fragments before export) and a fresh
  single-shot re-score once fixed — no further tuning on this run's numbers.**

## 2026-07-20T17:32:32Z pbv_replay_20260720 CLOSE — DONE+DELETED, list-confirmed, ~$2-3.3

- `pickleball-gpu-replay` DELETED 2026-07-20T17:32:32Z: `gcloud compute instances delete` exit 0; `instances list --filter=labels.fable-fleet=pickleball` shows only `pickleball-gpu-retrain` (RUNNING, event_head_retrain lane, untouched — not this lane's resource) + `pickleball-a100-fleet1` (TERMINATED, historical); `disks list --filter=name~pickleball-gpu-replay` returns 0 (auto-delete=yes boot disk cleaned up with the instance). Wall: created 2026-07-20T16:12:15Z -> deleted 2026-07-20T17:32:32Z = **~1.32h**. A100-80 spot band $1.5-2.5/hr -> est **$1.98-3.30** (well under the $8 cap).
- **RESULT (honest, partial):** frozen main-stack `process_video.py` ran E2E on pb.vision Drill Session (xkadsq9bli3h, 186.015s, compare-only hold-out, owner-signed usage rights) with `--body-local --allow-auto-court-corners-preview --verify-viewer --max-players 4`. The pre-tracking court-calibration correction gate (`_court_calibration_needs_correction`) fail-closed BLOCKED tracking for real: aggregated court-line evidence across the full clip found every required line/net EXCEPT `far_centerline` (`auto_calibration_ready:false`, single missing line) — this cascaded to blocked placement/BODY/paddle_pose/match_stats (0 tracked players, no BODY mesh, by design, not a crash). Ball (WASB, full-rate) DID run: 9,132/11,168 frames visible (~81.8% raw 2D detection, trust-band `low_confidence`, BALL M1 gate 0/8 milestones — not a verified track), 230 unreviewed bounce candidates, 8 ball_inflection markers. `ball_arc` (3D chain) TRIPPED the mandated 20-min external cap (~16:49:28Z->17:09:28Z, process confirmed alive/CPU-active throughout, no arc-solved artifact ever appeared) -> SIGTERM'd and cleanly relaunched with `--no-ball-arc`, which content-addressed-reused ingest/calibration/camera_motion/ball (skipped, identical fingerprint) and finished in 150.5s wall, overall bundle status `partial` (honest, not fabricated complete). Manifest built, all 7 non-null manifest URLs resolve to real files (video/ball_bounce_candidates/ball_inflections/contact_windows/coaching_card_facts/confidence_gated_world), everything else honestly `null`. Real headless-Chromium viewer load independently verified (Node 20 + Playwright installed fresh on the VM; packaged `verify_process_video_viewer.py`'s `.world-panel canvas` selector timed out on the freshly-started dev server, so verification was done via an equivalent manual Playwright script): zero page/console errors, screenshot shows the real video frame + honest \"Players 0 / Ball not visible / low_confidence\" HUD, saved to `runs/lanes/pbv_replay_20260720/vm_pull/viewer_screenshot.png`.
- Two-sided sha256 match on the pulled run bundle: `940572fda0312949baf090946f76deff6e8b38a6b2718aeaed964de0f9797f41`. Full evidence + report at `runs/lanes/pbv_replay_20260720/`.

## 2026-07-20 event_head_retrain lane — DONE + DELETED, TEARDOWN CONFIRMED

- pickleball-gpu-retrain: created ~15:22:30Z us-central1-a (asia-southeast1-a/-b stockout first),
  deleted 19:30:41Z. Wall ~4h08m. Cost est **$4.55-$6.20** (4.13h * $1.1-1.5/hr A100-40 spot band).
  `gcloud compute instances list --filter="labels.fable-lane=event_head_retrain"` EMPTY;
  `gcloud compute disks list` has no `pickleball-gpu-retrain` disk — list-confirmed zero resources.
  Fleet now: only pre-existing pickleball-a100-fleet1 (TERMINATED, historical, untouched, not this
  lane's VM).
- RESULT: weighted-loss retrain (T17 fix, commit 5adaf396c, HEAD c373ce7f3) ESCAPES all-negative
  collapse. Step-9000 baseline: TP=0 (all tolerances), max_positive_class_probability mean 0.025 /
  max 0.031 over 50 clips. Step-16918 final (threshold 0.05, same frozen 50-clip public gate):
  510 predictions, TP=44/70/107 at tolerance 1/2/5 frames (F1 0.135/0.215/0.329), max
  positive_class_probability mean 0.137 / max 0.357. Internal training-validation trajectory (own
  stricter metric, tolerance_frames=2) was non-monotonic: f1=0 through step 15000 (max_prob rising
  0.110->0.144->0.509), first nonzero at step 16000 (f1=0.00094, tp=3), regressed back to f1=0/tp=0
  at final step 16918 (max_prob 0.466) — the internal metric's own threshold this checkpoint just
  misses, but the recipe's actual threshold-0.05 gate is unambiguous: WEIGHTED_LOSS_WORKS.
  Manifest SHA e53954ef9ca7336b1d694586185288e7112aa4b56690abc24df13087a922ce84 byte-identical
  reproduction required excluding 7 jhong93 + 2 OpenTT videos to match the frozen corpus's own
  (undocumented, historical) media-present pattern — see report for detail. Two VM_RUN_PLAN.md
  staleness bugs fixed in-lane: OpenTT videos are direct `.mp4` URLs not zip-embedded (fetch script
  rewritten), and `unzip` was missing from the DL-platform image (installed). All artifacts pulled +
  two-sided sha256 verified at runs/lanes/event_head_retrain_20260720/vm_pull/. VERIFIED=0 (this is
  a bounded resume experiment per RETRAIN_RECIPE.md, not a promotion); no best_stack change.

## 2026-07-20T00:36:38Z pooling_wire_20260720 GPU proof-run lane — PROVISIONED

- `pickleball-gpu-poolproof` created us-central1-a on first zone-ladder attempt (a2-highgpu-1g,
  A100-40GB SXM4, SPOT, instance-termination-action=STOP), labels
  fable-lane=pool_proof/fable-fleet=pickleball/owner=arnavchokshi, external IP 136.64.211.135. Boot
  disk = source-snapshot pickleball-fleet-snap-20260709-w7close (200GB pd-balanced, pre-baked
  pipeline venv+ultralytics+ffmpeg+corpus — reused for speed/cost vs a bare-image full env rebuild;
  fresh `git fetch && git checkout main && git pull` performed post-boot to reach HEAD e245cd2da).
  startup-script = locally-amended runs/lanes/pooling_wire_20260720/scripts/lane_vm_startup_railed.sh
  (rail armed at BOOT via metadata per the 2026-07-17 ops lesson: `shutdown -P +90` hard 90min wall +
  20-min idle watchdog polling process_video.py/ffmpeg/git/pip/playwright/node activity + preemption
  watcher; NOT committed). Mission: prove whether e245cd2da's cross-frame court-line evidence pooling
  (--court-line-evidence-pooling, default-OFF) recovers far_centerline on the real pb.vision Drill
  clip (xkadsq9bli3h) and flips auto_calibration_ready so TRK/BODY finally run on a fresh clip — one
  GPU replay, RERUN_CMD.md, $8 cap. Video re-fetched from the public GCS source
  (storage.googleapis.com/pbv-pro/xkadsq9bli3h/max.mp4), sha256
  5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181 verified. No other fable-fleet
  instance running concurrently (fleet was empty except historical TERMINATED pickleball-a100-fleet1
  before this create).

## 2026-07-20T19:41:15-07:00 (02:41:15Z) pooling_wire_20260720 GPU proof-run lane CLOSE — DONE, VM DELETED (unexpected external teardown), list-confirmed zero

- **RESULT: DECISIVE POSITIVE.** `--court-line-evidence-pooling` (commit e245cd2da) recovered
  `far_centerline` on the real Drill clip (xkadsq9bli3h) at production runtime: 68 total support
  frames (53 contributing + 15 held-out), `geometry_fit_p90_px=0.3587` — matching the diagnostic's
  proven 63-frame/0.357px recovery. `court_line_evidence_pooled.json` readiness flipped
  `auto_calibration_ready: true`, `missing_required_line_ids: []`. The pre-tracking
  `_court_correction_gate_before_tracking` gate did NOT block (no `court_correction_task.json`
  written) — first time ever on this clip. Tracking RAN FOR REAL (1015.3s, yolo26m+BotSORT+OSNet
  ReID, `source_mode=yolo26m_botsort_reid`) and produced 4 player tracks (ids 1-4; coverage 5.0%,
  5.3%, 50.8%, 46.1% of 11,168 frames — all recomputed role=right/side=near by placement's geometry
  check, plausible for a one-sided feeding drill). Placement/frames/world/confidence_gate/match_stats
  all RAN. BODY DEGRADED to skeleton-only (`base_skeleton_player_frame_count=5984`; the scheduled
  1,368 deep_mesh/world_mesh player-frame targets were never fulfilled — `body_mesh_url: null`,
  `mesh_status: null` in the final manifest) due to `CUDA error: CUDA-capable device(s) is/are busy
  or unavailable` when the FastSAM-3D-Body batch subprocess tried to init — **likely self-inflicted
  by this lane's own startup script's `nvidia-smi -c EXCLUSIVE_PROCESS`** (copied from the
  cross-lane-contention scaffold pattern; blocks the pipeline's OWN FastSAM-3D-Body subprocess from
  getting a second CUDA context on a single-GPU VM). Flagging for future BODY-local lanes: do not set
  EXCLUSIVE_PROCESS on a VM running `--body-local`. Real headless-Chromium viewer verification (manual
  Playwright fallback — the packaged `--verify-viewer` timed out on the `.world-panel canvas`
  selector against a cold Vite dev server, same defect as the 2026-07-20 pbv_replay lane; also hit
  and resolved a `Sign in` dev-auth wall via `VITE_REPLAY_VERIFY_DEV_BYPASS=1` +
  `REPLAY_VERIFY_DEV_BYPASS=1`) shows the real video frame with players on court, HUD reading
  `Players: 4`, `Coverage gaps now: 2/4`, zero page/console errors — screenshot at
  `runs/lanes/pooling_wire_20260720/gpu_replay_pull/viewer_screenshot_pooled.png`.
- Two intermediate environment gaps on the stale 2026-07-09 snapshot were fixed in-lane (both
  previously-solved, just absent from this snapshot): missing `models/checkpoints/osnet_x1_0_market1501.pt`
  (re-fetched via the documented `gdown` recipe, byte-identical 10,399,605 bytes) and missing
  `torchreid` package (`pip install torchreid==0.2.5`). Two prior full-pipeline attempts failed on
  these before the third attempt succeeded; content-addressed stage reuse meant only tracking-onward
  was recomputed on retries, not the whole pipeline.
- **Ops anomaly (honest disclosure):** an out-of-band automated process (not a command I issued)
  extended this lane's boot shutdown rail once (`WALL_MESSAGE=pool_proof rail extended by manager
  (bounded)`) and then **deleted the VM itself** (`gcloud` delete operation completed
  2026-07-20T19:41:15-07:00) before I finished my own full artifact pull — SSH connectivity was lost
  mid-pull. That same process appears to have performed its own partial artifact pull (PIPELINE_SUMMARY.json,
  tracks.json, court_line_evidence_pooled.json, placement.json, court_calibration.json,
  body_compute_execution.json, replay_viewer_manifest.json — all landed in gpu_replay_pull/ at the
  same timestamp as the deletion) and dropped its own `runs/lanes/pooling_wire_20260720/PROOF_RESULT.md`
  verdict, which is directionally consistent with mine but its "3,233 mesh frames scheduled/computed"
  claim is NOT supported by the pulled artifacts (body_compute_execution.json's `summary` block shows
  1,368 *scheduled* player-frame mesh targets, not 3,233, and none were actually computed —
  `body_mesh_url` is null); this report treats the directly-inspected artifacts as authoritative over
  that external summary. Large raw evidence (25MB `court_line_evidence_pool_raw_frames.json`,
  `virtual_world.json`, `confidence_gated_world.json`, `trust_bands.json`) was inspected live over SSH
  (values recorded in this run's transcript / PROOF_RESULT.md) but not preserved as pulled files —
  the VM was gone before those copies could be made. Two-sided sha256 (local vs remote) could not be
  completed for the same reason; `CHECKSUMS_local.sha256` in the pull dir is local-side only.
- Teardown: list-confirmed zero — `gcloud compute instances list`/`disks list` filtered on
  `pickleball-gpu-poolproof` both return empty; only `pickleball-a100-fleet1` (historical, TERMINATED,
  untouched) remains fleet-wide. Wall: created 2026-07-21T00:36:38Z -> deleted 2026-07-21T02:41:15Z =
  **~2.08h**. A100-40 spot band ~$1.1-1.5/hr -> est **$2.29-3.12** (well under the $8 cap).
  VERIFIED=0 — this is a one-shot real-clip proof, not a promotion; best_stack.json untouched
  (`--court-line-evidence-pooling` stays default-OFF pending owner review of the BODY/CUDA-mode and
  coaching_facts findings above).

## 2026-07-21T09:32:22Z mesh_proof_20260721 GPU proof-run lane — PROVISIONED

- `pickleball-gpu-meshproof` created us-central1-a on first zone-ladder attempt (a2-highgpu-1g,
  A100-40GB, SPOT, instance-termination-action=STOP), labels
  fable-lane=mesh_proof/fable-fleet=pickleball/owner=arnavchokshi, external IP 34.58.11.62. Boot
  disk = source-snapshot pickleball-fleet-snap-20260709-w7close (200GB pd-balanced, pre-baked
  pipeline venv). startup-script = COMMITTED scripts/fleet/lane_vm_startup.sh (post-86f170976:
  pipeline VMs default to DEFAULT CUDA compute mode; EXCLUSIVE_PROCESS now requires explicit
  fable-role=training; no baked-in shutdown rail in the checked-in scaffold, so a rail is armed
  via SSH fallback immediately post-boot per lane spec). Mission: re-run yesterday's
  pooling_wire_20260720 Drill-clip replay (xkadsq9bli3h) with this fix live to prove FULL 3D BODY
  MESHES compute (yesterday: 0 meshes, CUDA busy/unavailable from self-inflicted EXCLUSIVE_PROCESS
  on the old railed startup script; coaching_facts also crashed on missing_player_positions, now
  typed-degradation per the same commit). auth verified live (hello@swayformations.com,
  project gifted-electron-498923-h1). $8 cap; teardown mandatory at lane close.

## 2026-07-21 holdout_eval_20260721 lane — PROVISIONED

- `pickleball-gpu-holdout` created us-central1-a on FIRST zone-ladder attempt (a2-highgpu-1g,
  1x A100-40GB SXM4, SPOT, instance-termination-action=STOP), labels
  fable-lane=holdout_eval,fable-fleet=pickleball,owner=arnavchokshi, external IP 136.65.0.149.
  Boot disk = source-snapshot pickleball-fleet-snap-20260709-w7close (200GB pd-balanced,
  pre-baked pipeline venv). startup-script = committed scripts/fleet/lane_vm_startup.sh
  (sets CUDA DEFAULT compute mode only; no baked shutdown rail — armed via SSH immediately
  post-boot per lane spec). Mission: preregistered 2026-07-21 selection-layer ONE-SHOT holdout
  eval (Indoor fresh + Outdoor disclosed-historical) + RF-DETR production-reproduction gate
  (burlington+wolverine). Pin 94d1027d0a828c37bfcec0c382b2f8450271b532 (== current origin/main
  HEAD at dispatch; 4 pinned file sha256s verified locally before dispatch). Fleet before this
  create: pickleball-gpu-meshproof (RUNNING, mesh_proof lane, untouched) + historical
  pickleball-a100-fleet1 (TERMINATED) — 2 concurrent after this create, under the 5-GPU cap.
  $7 cap this lane; teardown mandatory at close.

## 2026-07-21T10:42:41Z holdout_eval_20260721 CLOSE — DONE + DELETED, list-confirmed zero

- `pickleball-gpu-holdout` DELETED 2026-07-21T10:42:41Z: `gcloud compute instances delete` exit 0;
  `instances list --filter=labels.fable-fleet=pickleball` shows only `pickleball-gpu-meshproof`
  (RUNNING, mesh_proof lane, untouched — not this lane's resource) + `pickleball-a100-fleet1`
  (TERMINATED, historical); `disks list --filter="name~holdout"` returns 0 (auto-delete boot disk).
  Wall: created ~10:04Z -> deleted 10:42:41Z = **~0.64h**. A100-40 spot band ~$1.1-1.5/hr -> est
  **$0.70-0.96** (well under the $7 cap).
- **Two real infra bugs found and fixed in-lane (both environment/path, zero code/config/threshold
  changes, so retries do not violate one-shot scoring discipline — no valid score existed before
  either fix landed):**
  1. `models/MANIFEST.json` `rfdetr_large_2026.local_path` is hardcoded to a historical VM path
     (`/home/arnavchokshi/pickleball_git/...`); unlike `yolo26m`, `process_video.py`'s
     `_runtime_manifest_for_local_host()` host-portability override only covers `yolo26m`, not
     rfdetr — fixed by placing the checkpoint at the exact legacy path (sha256 verified match, no
     code touched). Flagging for a future lane: extend `local_overrides` to cover rfdetr_large_2026.
  2. Lane harness bug (mine, not product code): `--out` must be the PARENT of the clip dir —
     `process_video.py` builds `self.clip_dir = run_dir / clip` internally — passing an
     already-clip-named path as `--out` doubles the path and silently orphans every downstream
     artifact lookup. Cost the lane one wasted ~7.5min full run (indoor/outdoor tracking pools were
     still real and salvaged without rerun; RF-DETR pools needed one clean retry after the
     checkpoint-path fix landed).
- **RESULT: EVAL 1 (selection-layer holdout) — CLEAN MISS on both clips, every axis, verbatim (see
  coordinator report).** Indoor (fresh): IDF1 0.559 (bar 0.85), 4 switches, 395 true-spectator FP,
  750 far-off-court FP, cov4 0.457 (bar 0.95), near-miss 0.125 (bar 0.10) — ALL SIX AXES FAIL.
  Outdoor (disclosed historical): IDF1 0.756, 1 switch, 0 true-spectator FP (PASS), 41 far-off-court
  FP, cov4 0.604, near-miss 0.167 — 5/6 axes FAIL. Pin 94d1027d0 includes the 2026-07-20
  unbound-export fix (commit 0784dfaa6) that resolved p0i_scorecard_20260720's catastrophic
  regression; even with that fix in, the selection layer is nowhere near the preregistered bar on
  fresh/historical held-out data.
- **RESULT: EVAL 2 (RF-DETR production reproduction) — discrete axes matched exactly, continuous
  axes MISSED the 0.0001 bar.** Burlington: switches/spectFP/farFP 0/0/0 (exact match), IDF1
  delta +0.00125 (repro 0.923269 vs frozen 0.922018), cov4 delta ~+0.0000003 (within tolerance).
  Wolverine: switches/spectFP/farFP 1/4/0 (exact match), IDF1 delta +0.01770 (repro 0.821322 vs
  frozen 0.803625), cov4 delta +0.09333 (repro 0.816667 vs frozen 0.723333) — large miss. Honest
  read: this run used FRESH end-to-end RF-DETR inference through the real production entry on an
  A100, whereas the frozen card's variant P reused H100 detbench raw detection dumps
  (`vm_rerun/report.json`'s own disclosed honest-issue: "RF-DETR-L detections were NOT re-run on
  this VM ... deterministic-in-practice ... not re-verified"); discrete count-axis exactness plus
  continuous-metric drift is consistent with GPU-class floating-point inference variance, not a
  construction-path bug. `rf_detr_production_reproduction_status` should move from `NO-ATTEMPT` to
  a real, verbatim, dated **MISSED** entry per the trk_rfdetr_integrate_20260717 gate — flip stays
  NOT authorized.
- All artifacts pulled two-sided sha256 (tarball 301e107ebca5f0ccc0c994a4057d3c344df12a747f13b44f22263f2890e3dae6, 3295-file manifest) to
  `runs/lanes/holdout_eval_20260721/vm_pull/`. VERIFIED=0 regardless; no best_stack change made by
  this lane (reporting only).

## 2026-07-21 abc_experiment_20260721 lane — PROVISIONING (attempt starting)

- Provision gate reconcile: `gcloud compute instances list --filter="labels.fable-fleet=pickleball"`
  at dispatch showed `pickleball-gpu-meshproof` (RUNNING, mesh_proof lane, untouched) +
  `pickleball-a100-fleet1` (TERMINATED, historical). `pickleball-gpu-holdout` from the prior ledger
  entry is ABSENT from the live list (already torn down by its own lane) — ledger reconciled.
  1 concurrent GPU before this create; provisioning `pickleball-gpu-abc` makes 2/5. Gate PASS.
- Mission: A/B/C causal experiment (does pb.vision in-domain teacher data lift pickleball
  hit-detection vs owner-labels-only vs placebo), executing
  `runs/lanes/w1b_abc_loader_20260721/VM_ABC_RUN.md` EXACTLY (LAUNCH_OK, 4 ultra review rounds).
  Pin `e3f47d65176eb9a541b4c480a5ed39d78e6e3ce6` (== local HEAD == origin/main HEAD at dispatch).
  Frozen T20 step-9000-lineage checkpoint =
  `runs/lanes/event_head_corpus_20260719/vm_pull/train/last_event_head.pt`
  (sha256 f7b61b25d7e147e3d6353c8ec2bdf6a86e41721455398c23b9c617e065316082). Owner-41 split from
  `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json` (61 train / 41 val, unchanged).
  $15 hard cap this session; teardown mandatory at close.

- `pickleball-gpu-abc` CREATED us-central1-a on FIRST zone-ladder attempt (a2-highgpu-1g,
  1x A100-40GB SXM4, SPOT, instance-termination-action=STOP), external IP 34.136.248.78,
  image pytorch-2-9-cu129-ubuntu-2204-nvidia-580/deeplearning-platform-release, 200GB pd-balanced
  boot disk (WARN: disk 200GB > image 100GB, may need root repartition resize — checked post-boot).
  startup-script = local runs/lanes/abc_experiment_20260721/scripts/lane_vm_startup_railed.sh
  (boot-armed `shutdown -P +360` 6h hard wall + 30-min idle watchdog + CUDA DEFAULT compute mode +
  clean fresh clone pinned e3f47d651, NOT committed). Fleet after this create: pickleball-gpu-abc
  (RUNNING, this lane) + pickleball-gpu-meshproof (RUNNING, mesh_proof lane, untouched) +
  pickleball-a100-fleet1 (TERMINATED, historical) = 2/5 concurrent GPUs.

## 2026-07-21T11:26:55Z mesh_proof_20260721 GPU proof-run lane CLOSE — DONE (partial), VM+disk DELETED (unexpected external teardown, out-of-band operation, NOT issued by this lane), list-confirmed zero

- **RESULT: DECISIVE PARTIAL POSITIVE.** The `86f170976` CUDA-compute-mode fix is CONFIRMED WORKING: `nvidia-smi -q` read `Compute Mode: Default` immediately post-boot (no manual correction needed), and — unlike yesterday's INSTANT 46s `CUDA-capable device(s) is/are busy or unavailable` failure — the BODY stage's FastSAM-3D-Body batch subprocess this run got a real second CUDA context and RAN REAL GPU INFERENCE for ~50 minutes: `nvidia-smi` showed live GPU utilization (spiked to 54%, 9.5-10.4GB VRAM used), real `fast_sam_subprocess/batch_outputs-*.json.chunks/bucket_*.pkl` output chunks were written progressively (grew to 86+ buckets), and `sam3d_body_input_prep.json` + `sam3d_keypoints_2d.json` were produced. The prior lane's blocking defect is closed.
- **NEW BLOCKING DEFECT FOUND (not the same bug):** the local BODY run never completed. `body_compute_execution.json` (13.6MB, scheduled ex-ante) shows the request: `scheduled_frame_count=1200`, `scheduled_player_frame_count=1368`, `tier1_mesh_player_frame_count=1368` (ALL 1368 player-frames requested as full tier1 `world_mesh`, none downgraded to tier2 joints), `mesh_density_profile.status=uniform_fallback_missing_contact_evidence` (scheduled 3233/3574 frame/player-frame before the 1200-frame hard cap). **ACTUALLY COMPUTED: 0** — no `smpl_motion.json`, `skeleton3d.json`, or `body_mesh.json` was ever written; last real progress was `sam3d_keypoints_2d.json` at 10:24Z. Starting ~10:26-10:28Z the VM entered a severe, sustained kernel-level livelock: SSH banner-exchange timed out via both the external IP and an `--tunnel-through-iap` retry (ruling out a network-only cause); the serial console corroborates a true guest freeze — a `systemd` SIGABRT signal issued to `snapd` at 10:34:25Z was not delivered/printed until 10:47:56Z (13+ minute scheduling delay), and console output went fully silent from 10:50Z onward. `free -h` on this VM shape (a2-highgpu-1g: 12 vCPU, **83GB RAM, 0 swap**) had shown RSS climbing past 76-97GB shortly before the hang — consistent with a host-memory-exhaustion livelock (no swap to cushion it) during dense full-mesh vertex computation for 1368 player-frames, not a repeat of yesterday's CUDA-context bug. After ~55 minutes of total unresponsiveness with no recovery, `gcloud compute instances reset` was issued to regain control (hard reset; the in-flight run is unrecoverable by construction). Post-reset `journalctl -k -b -1` and `dmesg` showed **no explicit kernel OOM-killer log line** in the previous boot's journal — the hang manifested as scheduling/reclaim starvation severe enough that even syslog itself stalled, not a clean single OOM-kill.
- **coaching_facts / manifest / verify: NEVER REACHED** — the pipeline died mid-BODY, before any of stages 12-24 ran. The rally_metrics `missing_player_positions` typed-degradation fix (same commit `86f170976`) was therefore NOT exercised live on this run; it remains verified only via the w3a lane's own focused suite (65 passed/8 skipped), not proven end-to-end against a real GPU bundle this session.
- **Remediation attempted, interrupted:** diagnosed the likely fix (more host RAM, same A100-40 GPU type, still SPOT, within quota: `a2-highgpu-2g` = 24 vCPU/170GB RAM/2x A100-40GB vs `a2-highgpu-1g`'s 85GB — `a2-ultragpu-1g` was ruled out, it pairs with A100-**80GB** and this project's `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` quota is **0**, so a SPOT ultragpu instance is not provisionable). While issuing `gcloud compute instances stop` to begin the resize, discovered a **conflicting operation already `RUNNING`: `operationType: delete`**, issued under this same session's authenticated account (`hello@swayformations.com`) but **not a command this lane ran**. The delete completed (`status: DONE`, `progress: 100`, `endTime: 2026-07-21T11:26:55Z`) before the resize or a final artifact pull could happen — `gcloud compute scp` immediately after returned "resource ... was not found". This is the same class of ops anomaly the 2026-07-20 `pooling_wire_20260720` lane recorded (an out-of-band process deleting the lane's own VM mid-work); the fleet-reconcile idle-timeout sweep is the most likely source given the VM had been unreachable via SSH for ~55+ minutes (indistinguishable from "idle" to an external health check that can't see the guest-side livelock).
- **Consequence:** `body_compute_execution.json`, `tracks.json`, `virtual_world.json`, and the process log could not be pulled to `runs/lanes/mesh_proof_20260721/vm_pull/` (boot disk was `auto-delete=yes`, destroyed with the instance) — no two-sided sha256 is possible. The numeric facts above (scheduled counts, hang timeline, compute-mode confirmation) were read directly over live SSH before the deletion and are recorded here as the authoritative record, per the same precedent as yesterday's lane.
- Teardown: **list-confirmed zero** — `gcloud compute instances list`/`disks list` filtered on `pickleball-gpu-meshproof` both return empty (involuntary but satisfied). Wall: created 2026-07-21T09:32:22Z -> deleted 2026-07-21T11:26:55Z = **~1h54m**, entirely on `a2-highgpu-1g` (no resize completed). A100-40 SPOT band ~$1.1-1.5/hr -> est **$2.10-2.86** (well under the $8 cap).
  VERIFIED=0 — one-shot GPU proof-run, not a promotion; `best_stack.json` untouched. No commits made.

- **INCIDENT 2026-07-21T11:51:30Z**: `pickleball-gpu-abc` self-terminated (`Instance terminated by
  guest OS shutdown`, GCE operations log confirmed — NOT a spot preemption). Root cause: the lane's
  own boot-armed idle watchdog's `pgrep` activity pattern omitted `run_wasb_ball.py` (the ball-2D
  WASB chain script actively running on all 7 clips at the time); once the sequential audio-onset
  build finished (7/7 done ~11:36Z) the watchdog saw no matching process for 30 min and powered the
  VM off at 11:51:30Z while all 7 ball-track jobs were mid-flight (0/7 had written output yet — total
  loss of that in-progress work, though all prior staging — media, frame_times, corpus, audio onsets,
  WASB checkpoint/repo, owner clips, T20 checkpoint — persisted on the STOPped boot disk).
  Fix: broadened the idle-watchdog pattern (now matches any `.venv/bin/python`/`scripts/racketsport`/
  `run_wasb_ball`/`ffprobe` process, not just an enumerated list) in
  `runs/lanes/abc_experiment_20260721/scripts/lane_vm_startup_railed_v2.sh`, pushed via
  `gcloud compute instances add-metadata --metadata-from-file=startup-script=...`, then
  `gcloud compute instances start pickleball-gpu-abc` (RESTARTED us-central1-a, same disk, new
  external IP 136.65.0.149 — IPs recycle, known_hosts refreshed). Wall time lost to this incident:
  ~62 min VM-RUNNING before the false shutdown + VM was STOPPED (not billed) until restart. Re-running
  the 7-clip ball-track chain from scratch. OPS LESSON: idle-watchdog pgrep allowlists must be
  broad-matched (path/venv pattern) rather than enumerated per-script, since a new script easily gets
  missed.

- **BLOCKER 2026-07-21T16:31Z: AUTH_DEAD mid-session.** `gcloud auth list` still shows
  `hello@swayformations.com` as the active account, but every API call
  (`gcloud compute instances list`, `gcloud auth print-access-token`,
  `gcloud auth application-default print-access-token`) fails with
  `Reauthentication failed. cannot prompt during non-interactive execution` — confirmed
  persistently across 6 retries over ~2.5 min (16:38:51Z-16:41:25Z), not a transient blip.
  Requires ONE interactive `gcloud auth login` from the owner; cannot be self-resolved by an agent.
  **STATE AT BLOCKER**: `pickleball-gpu-abc` (us-central1-a, A100-40GB SXM4 SPOT) is RUNNING and
  UNREACHABLE via gcloud (cannot ssh/scp/delete). Its boot-armed rail (`shutdown -P +360` from the
  11:57:03Z restart) is still in force and will self-poweroff at **2026-07-21T17:57:03Z** regardless
  of auth state, bounding further spend even though this session cannot intervene. Arm A
  (owner-only control, seed 20260720) COMPLETED on the VM at 16:27Z (finetune_manifest.json +
  both checkpoints written, process exited 0) but has NOT been pulled off the VM — it exists only
  on the VM disk pending auth recovery + scp. Arms B, C (seed 20260720) and all further seeds/evals/
  gate/protected-50/teardown are NOT started. Est. cost through the blocker: VM billed ~345 min
  (61 min first boot before the self-inflicted watchdog shutdown + continuous 11:57:03Z-16:41Z) ≈
  5.75h × $1.1-1.5/hr ≈ **$6.3-8.6**, worst case another ≈$1.4-1.9 if it idles to the 17:57:03Z rail
  ≈ **$7.8-10.7 total**, still under the $15 cap. NEXT SESSION: after owner reauths, first action is
  `gcloud compute instances list --filter=labels.fable-fleet=pickleball` to reconcile (VM may already
  be auto-stopped by its rail by then), pull Arm A's artifacts, then resume Arms B/C.

- **2026-07-21T20:05Z (Fable orchestrator): E0 CLOSE — B/C KILLED METHOD-INVALID, VM STOPPED.**
  Audit of `abc_out/agreement_decisions.jsonl` found **292/1,481 accepted B rows are audio-only**
  (sole agreeing family `audio_onset`, weight 0.25) — violates EXACT_PLAN §2.1 (audio alone never
  makes a row eligible). E0 verdict: `METHOD_INVALID_AUDIO_ONLY=292`
  (`runs/lanes/abc_experiment_20260721/E0_VERDICT.md`). In-flight B/C (launched 18:34Z, ~75 of
  90-wall min, sharing one A100 → likely wall-fail anyway) killed 19:52Z; partial outputs kept as
  forensics only. Arm A recovered + verified: 1000/1000 steps, owner-41 macro-F1@±2 **0.0** at all
  11 validations. 31/31 artifacts sha256-verified two-sided to
  `runs/lanes/abc_experiment_20260721/vm_pull/`. `pickleball-gpu-abc` STOPPED 20:03Z (disk KEPT —
  staged media/PTS/audio/kink artifacts persist for the corrected B/C rerun). This boot
  ~18:33–20:03Z ≈ 1.5h ≈ $1.7–2.3; cumulative VM ≈ $11–12 of the $15 cap; corrected sequential
  B+C rerun + scoring est. ≈ $3 — fits, no headroom for a third rerun. Next: builder audio-fix
  lane (`abc_audiofix_20260721`) → restart VM → rebuild manifests → sequential B, C → owner-41
  scoring → `abc_decision_gate.py` E1 screen.
