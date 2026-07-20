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
