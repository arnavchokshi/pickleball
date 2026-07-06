---
name: run-lane
description: Use when Fable is about to dispatch any Codex or Sonnet implementation lane (build / fix / verify / docs work) on the pickleball project. Templates a complete, collision-safe lane spec so the lane self-verifies and returns a rulable structured report. Do NOT use for Fable's own tiny coordination edits or for research fan-outs (use research-fanout).
---

# run-lane

Every implementation lane gets this spec. Fable writes it, a Codex lane (default) or Sonnet lane
(only for GPU/SSH/browser/network) executes it. Fable never implements directly (§1).

## Pre-dispatch: the safe-parallelism check (FABLE_OPERATING_MANUAL §12.1)
Before writing the spec, confirm the lane is: **file-disjoint** (owned files overlap 0 with every
in-flight lane per BUILD_CHECKLIST), **data-disjoint** (no held-out/protected label without a ledger
row — else STOP), **resource-disjoint** (GPU? idle one or provision new per §12). If it needs a GPU,
provision/assign the VM FIRST (gpu-fleet-provision skill) and record it in `runs/manager/gpu_fleet.md`.

## The spec (write to `runs/lanes/<lane>_<date>/spec.md`)
1. **HARD RULES block:** no branches/commits (owner joint-commit rule); read NORTH_STAR + BUILD_CHECKLIST
   last note; 4 protected clips EVAL-ONLY (Burlington/Wolverine internal-val OK; Outdoor/Indoor labels
   NEVER without a pre-registered ledger row + STOP for Fable go); honest reporting; run the WIDE
   blast-radius test suite (`MPLBACKEND=Agg`), not a hand-picked subset; register any new root .md in
   the doc allowlist same-lane; every new CLI ships its direct-CLI reference test same-lane; artifacts
   under `runs/lanes/<lane>_<date>/`.
2. **EXPLICIT FILE OWNERSHIP:** name exactly which files this lane owns; concurrent lanes must be
   file-disjoint (process_video.py contention has bitten us).
3. **Objective + acceptance NUMBERS:** the measurable gate from the roadmap task, verbatim. Kill criteria.
4. **Evidence to read first:** the exact run paths / reports the lane must ground on.
5. **Mandatory structured report** (Fable rules on this, never the transcript): `objective_result`
   (PASS/FAIL vs the numbers), `full_suite` (passed/failed — failed>0 while claiming PASS = rejected
   unless all failures proven pre-existing), HONEST ISSUES, artifacts, dated BUILD_CHECKLIST bullet.
6. **Anti-passive-wait** (for any lane touching a >10min GPU/remote job): "ending your turn to wait =
   lane death; you will NOT be re-woken; poll with a bounded foreground until-loop; end only with the
   final report or a hard blocker." Budget 1-2 SendMessage resumes.

## Dispatch (match FABLE_OPERATING_MANUAL §10 exactly — schema-validated report, no watchers)
```bash
LANE=<short-name>; ROOT=/Users/arnavchokshi/Desktop/pickleball
mkdir -p "$ROOT/runs/lanes/$LANE"   # spec.md written first
codex exec \
  --cd "$ROOT" --sandbox workspace-write \
  -c model_reasoning_effort=xhigh \
  --output-schema "$ROOT/docs/racketsport/lane_report.schema.json" \
  -o "$ROOT/runs/lanes/$LANE/report.json" \
  < "$ROOT/runs/lanes/$LANE/spec.md" \
  > "$ROOT/runs/lanes/$LANE/log.txt" 2>&1
```
run_in_background: true; absolute paths ALWAYS; add `-c tools.web_search=true` for research lanes.
The harness notifies on process exit — read `report.json` and rule. Do NOT set up Monitor/done-marker
watchers (they false-fire; manual §10). Prefer big self-iterating lanes over many small round-trips.

## After
Verify personally: run its tests in the real env, open its artifacts, check its numbers. Codex
"failures" are often sandbox-only (socket/MPS) — re-verify locally. Then update the task board.
