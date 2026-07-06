#!/usr/bin/env bash
# Fleet reconcile sweep (FABLE_OPERATING_MANUAL §12) — run by a Sonnet lane or a local scheduled job
# (needs network; NEVER Codex). STATUS: SCAFFOLD — flesh out with gpu_fleet.md parsing in P0-1 lane.
set -euo pipefail
# Intent: (1) list live VMs: gcloud compute instances list --filter=labels.fable-fleet=pickleball
# (2) diff vs runs/manager/gpu_fleet.md; (3) restart STOP'd (preempted) VMs whose lane is unfinished;
# (4) tear down VMs idle >15min with no assigned lane; (5) write a structured report for the manager.
echo "reconcile: scaffold — implement in P0-1 lane"
