# Pickleball Repo Status

This repo is the active CV/client implementation for the Sway Body racket-sport pipeline.

Canonical read order:

1. `CAPABILITIES.md` - what is actually invoked, gate-checked, blocked, or scaffold-only.
2. `BUILD_CHECKLIST.md` - operational task status and current handoff checklist.
3. `IMPLEMENTATION_PHASES.md` - target build plan and phase gates.
4. `ACCURACY_AND_TRAINING.md`, `TECH_STACK.md`, `SWAY_BODY_PICKLEBALL_MVP.md` - target product, model, data, and training context.

Current truth: `VERIFIED = 0`; many iOS, model, replay, and on-device items are scaffold or prototype-gate only until their checklist rows are promoted by tests and real label/device evidence. `CAPABILITIES.md` section "Canonical Tier Split" is the single source of truth for ON-DEVICE LIVE / fast tier versus SERVER OFFLINE / deep tier placement; if any other doc conflicts with it, `CAPABILITIES.md` wins and the conflicting doc must be fixed. Accepted-four artifacts under `runs/eval0/prototype_gate_h100_v2` are prototype packet evidence, not a populated `DATA-1` test-clip dataset. Files under `docs/racketsport/archive/` are evidence snapshots, not active runbooks.
