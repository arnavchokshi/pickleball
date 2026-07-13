# GPU fleet ledger (live)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).
Full per-wave history (waves 4-7, NS-014, demo, court, 2026-07-12 sprint) is preserved verbatim in
`runs/manager/archive/gpu_fleet_history_20260707_20260712.md`.

## Current fleet state (2026-07-13, doc/org session)

EMPTY — zero running VMs. Only `pickleball-a100-fleet1` exists, TERMINATED (disk intact, historical
snapshot source). Last list-reconcile: 2026-07-12 sprint close.

| vm_name | zone | gpu | model | status | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| (none running) | | | | | | | | |

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

## Most recent wave (2026-07-12 sprint — full rows in archive)

- pickleball-h100-trka: DONE+DELETED 2026-07-12T20:27Z list-confirmed, 1.655h ~$2-3.5
  (TRK ReID/apron margin sweep; margin 0.5/1.0 survive internal, 2.0 rejected).
- pickleball-h100-bodyc: DONE+DELETED 2026-07-12T22:17Z list-confirmed, 1.37h ~$0.8-5.8
  (BODY overhead levers all 3 honest-rejected; found world-stage 122s cost attribution).
