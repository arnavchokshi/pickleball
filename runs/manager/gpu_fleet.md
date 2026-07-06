# GPU fleet ledger (FABLE_OPERATING_MANUAL §12)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).

| vm_name | zone | gpu | model | status (provisioning/idle/busy/preempted/tearing-down) | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| _(none — fleet empty; old `pickleball-a100-spot-ase1a` powered off 2026-07-05, delete staged w/ owner)_ | | | | | | | | |

Fleet cost cap (owner 2026-07-06): ≤$5/GPU/hr, max 4 concurrent GPUs; teardown/DELETE the moment a
lane ends (idle spend never acceptable); 5th GPU or >$5/hr = `needs-purchase-approval` STOP.
Auth: service-account key `~/.secrets/pickleball-fleet-sa.json` (never in git/chat).
