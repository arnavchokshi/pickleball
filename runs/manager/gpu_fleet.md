# GPU fleet ledger (live)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).
Full per-wave history (waves 4-7, NS-014, demo, court, 2026-07-12 sprint) is preserved verbatim in
`runs/manager/archive/gpu_fleet_history_20260707_20260712.md`.

## Current fleet state (2026-07-15, Track A manager session)

RECONCILED LIVE 2026-07-15: `gcloud compute instances list` succeeded (hello@, project
gifted-electron-498923-h1). Only `pickleball-a100-fleet1` TERMINATED under fable-fleet=pickleball
(matches prior EMPTY claim — now freshly confirmed); non-fleet `body4d-waker-ctrl` e2-micro RUNNING
in usc1-a, untouched. Fleet RUNNING count 0/5 → provision gate PASS for the pbv11 re-run.

| vm_name | zone | gpu | model | status | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| pickleball-h100-pbv11r | us-central1-a (attempt 3; ase1-b/-c stockout) | H100-80GB | a3-highgpu-1g SPOT | RUNNING (rail armed) | pbv11_headtohead_20260713 RE-RUN | ≤$5 | 2026-07-15T22:53Z (wall_cap_start 22:56:20Z) | 2026-07-16T02:50Z manager takeover after Mac sleep killed the Sonnet lane: pipeline alive (PID 3999, ball_arc BVP solve, GPU 0%/CPU-bound), calibration auto-preview grade POOR (ball_world/virtual_world_metric fail-closed; tracks.json absent). Lane's promised in-VM self-stop was NEVER armed — manager armed `sudo shutdown -P 03:56` (UTC) as the hard cost rail, verified in systemd. Partial artifacts protectively pulled to runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/. Cap ruling: NO full-run extension (post-events stages near-worthless w/o tracks + metric fail-close); ≤45-min one-time rail push ONLY if arc done + events mid-flight at 03:45Z check. DELETE + list-confirm + disks 0 + cost at end no matter what — manager-owned. |

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
