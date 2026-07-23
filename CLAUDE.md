# CLAUDE.md — session bootstrap

Read in this order before acting:

1. `AGENTS.md` for durable repository rules.
2. `NORTH_STAR_ROADMAP.md` for the sole product definition, current truth,
   ordered program, gates, owner asks, and next-agent queue.
3. The relevant `RUNBOOK.md` section for commands and actual runtime behavior.
4. `runs/manager/inflight_lanes.md` and `runs/manager/gpu_fleet.md` only when
   coordinating live lanes or cloud workers.
5. `DATA_INVENTORY.md` to see what training data exists on every lane and whether
   it is used (generated from `runs/manager/data_ledger.json`; regenerate with
   `scripts/racketsport/build_data_inventory.py` after any ledger change).

Do not create or revive a separate master plan, build checklist, capability
matrix, wave roadmap, operating manual, edge playbook, or technical blueprint.
Dated work goes under `runs/`; only material product truth, gate, or sequencing
changes belong in the North Star.

`VERIFIED=0` remains binding until a named independent-data gate passes.
Outdoor is historical rather than fresh promotion evidence. Best-stack changes
must be represented in `configs/racketsport/best_stack.json`, but the North Star
alone decides what should be attempted next.
