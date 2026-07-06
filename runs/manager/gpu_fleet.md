# GPU fleet ledger (FABLE_OPERATING_MANUAL §12)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).

| vm_name | zone | gpu | model | status (provisioning/idle/busy/preempted/tearing-down) | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| _(none — fleet empty; old `pickleball-a100-spot-ase1a` powered off 2026-07-05, delete staged w/ owner)_ | | | | | | | | |

Fleet cost cap: ≈$2/hr × active-lane-count; any single VM >$3/hr or a 5th concurrent lane = `needs-purchase-approval` STOP.
