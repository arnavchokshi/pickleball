---
name: gpu-fleet-provision
description: Use when a GPU-bound pickleball lane needs a VM and no idle matching GPU exists in runs/manager/gpu_fleet.md. Runs the safe-parallelism + reuse-vs-new decision, preflights GCP quota, issues the labeled spot-create via a delegated SONNET (network-capable) lane, and records the VM in the fleet ledger. Do NOT use to run the gcloud calls on Fable directly — Fable decides, a lane executes.
---

# gpu-fleet-provision

Owner ruling 2026-07-06: buy more GPUs to parallelize. Fable runs many lanes across separate GPUs so
they never contend. Fable DECIDES; a SONNET subagent or manager-run detached script runs `gcloud` —
NEVER Codex (its sandbox has no network). Full model: FABLE_OPERATING_MANUAL §12.

## Decision (before provisioning)
1. **Safe-parallelism check** (run-lane pre-dispatch): file/data/resource-disjoint. If it fails on
   file/data → sequence, don't parallelize. If data touches held-out without a ledger row → STOP.
2. **Reuse vs new:** read `runs/manager/gpu_fleet.md`. Reuse an idle VM whose GPU type/driver/CUDA
   matches (verify `nvidia-smi` on it first). Provision NEW only when no idle match AND ≥2 GPU-bound
   lanes are genuinely safe-parallel. One physical GPU per lane. **Hard cap 4 concurrent lanes** — a
   5th is a `needs-purchase-approval` STOP (stop-and-ask). Never provision speculatively.
3. **Cost (owner 2026-07-06):** ≤$5/GPU/hr, max 4 GPUs; teardown/DELETE on lane completion — idle
   spend never acceptable; 5th GPU or >$5/hr = needs-purchase-approval STOP.
4. **Auth:** owner's gcloud refresh token (SA key creation is ORG-BLOCKED — don't retry). Verify
   with one cheap list call before provisioning; dead auth = typed STOP for one owner login.

## Provision (delegated to a Sonnet/network-capable lane — never hand-run on Fable, never Codex)
The lane runs, with a preflight quota check (fall back to next region on SKU exhaustion, don't
retry-storm):
```
gcloud compute instances create "$LANE_VM" --zone="$ZONE" \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --accelerator="type=$GPU_TYPE,count=1" \
  --labels="fable-lane=$LANE,fable-fleet=pickleball,owner=arnavchokshi" \
  --metadata-from-file=startup-script="$ROOT/scripts/fleet/lane_vm_startup.sh" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT"
```
- `--instance-termination-action=STOP` (never DELETE) — preempted VM keeps its disk for cheap resume.
- Every fleet VM MUST carry `fable-lane=<lane>` — teardown sweep + billing breaker key off it.
- `lane_vm_startup.sh` sets `nvidia-smi --compute-mode=EXCLUSIVE_PROCESS` (2nd CUDA context fails
  loud, no silent contention), mounts code, starts the preemption watcher.

## Record + teardown
- Write the lane→VM row to `runs/manager/gpu_fleet.md` (name, zone, type, spot, status, lane, $/hr,
  created_at) BEFORE dispatching work.
- Tear the VM down the moment its lane ends. `fleet-reconcile` (scheduled ~15-20min) sweeps idle-timeout
  VMs and restarts STOP'd (preempted) ones. Reconcile orphaned prior-session VMs at session start.
- If gcloud auth is broken: `ssh -i ~/.ssh/google_compute_engine arnavchokshi@<ip>` bypasses it, but
  create/delete needs `gcloud auth login` (owner action — a `needs-decision` STOP if missing).
