# MICRO-LANE w6_beststack_recon_20260708 — best-stack doctrine reconciliation for wave-6 gains (Part IV rule 15, owner standing directive 2026-07-08)

## HARD RULES
- NO branches/commits/pushes; no board edits (bullet text in report). Artifacts under runs/lanes/w6_beststack_recon_20260708/ only.
- The doctrine arrived MID-wave-6 (encoded by a concurrent session): this lane retroactively reconciles wave-6's landed gains into configs/racketsport/best_stack.json. A manifest entry is a DEFAULT selection, NEVER a VERIFIED claim.
- Follow the EXISTING schema exactly (28 entries; keys stage.name; status/gate/provenance shape — read several entries first). Bump revision; update the updated date.

## FILE OWNERSHIP
- OWNED: configs/racketsport/best_stack.json + any test that validates it (grep for the manifest's coverage tests; update coverage lists if the schema demands).
- READ-ONLY: everything else. DO NOT TOUCH court/calibration files, web/replay, cvat_upload.

## THE WAVE-6 GAINS TO RECONCILE (manager's classification — encode these, adjusting only if the schema forces)
1. mesh.byte_budget_policy — the --mesh-byte-budget-mib selection policy (commit 091e91196). STATUS: PENDING. gate: owner decision on default promotion (transfer/UX cost: wolverine 77MiB, outdoor 112.6MiB vs ~13MiB at legacy cap-100; measured playback gain 1.67x/4.1x; evidence runs/lanes/w6_meshcap_20260708/report.json + runs/lanes/w6_close_errand_20260708/). Current default remains legacy target_mesh_frame_budget=100 until promoted.
2. body.postchain_raw_knob — the --body-postchain raw instrument family (dd5e5980d + footpinstub fix). STATUS: DORMANT (instrument-only, never a pipeline default; strict loud-bypass provenance). Evidence: runs/lanes/w6_gate1b_knob_20260708/, w6_footpinstub report.
3. instrument.gate_check_body_decode — canonical decode-fidelity harness (scripts/racketsport/gate_check_body_decode.py). STATUS: DORMANT instrument (not a pipeline stage). Evidence: w6_gatecheckfix report + w6_close_errand gate1b_raw_arm_report.json (gate_1b FAIL 262.35mm — decode NOT wiring-ready; the manifest must NOT reference latent smoothing as available).
4. ball.arc_solver_spin — fit_spin_scalar plumbing (dormant by kill ruling). STATUS: DORMANT w/ kill provenance (revisit gated on STEP 5 view confidence). Evidence: runs/lanes/w6_magnus_20260708/report.json.
5. training.stage2_resume — --resume-checkpoint (infra, not a default selection): include ONLY if the schema has an infra/instrument class; else report as out-of-scope with one line.
6. ball.candidate_ordering NOTE: seed_official > control > stage1_official on the 1121-row owner corpus is CANDIDATE EVIDENCE, not a default change — ball.wasb_checkpoint default stays the raw WASB tennis zero-shot per its existing entry. Do NOT change it; add nothing unless the schema has an evidence/notes field on the existing entry (then append the pointer runs/lanes/w6_labelingest_20260708/gpu_rescore/loso/loso_report.json).

## ACCEPTANCE
1. Manifest validates against its own tests (find + run them: coverage lists, schema checks, scaffold/doc guardrails).
2. Each new entry carries status/gate/provenance in the house shape with evidence paths that EXIST (verify).
3. No default behavior changes (this is bookkeeping; the only WIRED_DEFAULT entries remain the pre-existing ones).
4. Focused suite for the manifest tests green.
## REPORT (schema-enforced): objective_result; full_suite (scoped); CHANGES; per-gain entry table; HONEST ISSUES; proposed bullet; NEXT.
