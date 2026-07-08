# LANE w6_gate1b_harnessdiag_20260708 — READ-ONLY diagnosis: is the GATE-1b harness measuring what it claims?

## HARD RULES
- STRICTLY READ-ONLY outside runs/lanes/w6_gate1b_harnessdiag_20260708/ (you may RUN existing code CPU-side; you may not edit/create/delete any repo file). No GPU. Protected clips EVAL-ONLY. Honest numbers with paths; label estimates vs measurements.

## THE QUESTION (needs-validation class; diagnosis BEFORE any GPU re-spend)
The H100 errand ran the decode-fidelity harness (runs/lanes/w5_p22latent_20260707/scripts/gate_check.py, calling threed.racketsport.mhr_decode) against the default-arm body_mesh.json and got gate_1b worst_joints_world_max_abs_error_mm=233.01 / mesh_skel worst_p95_mm=53.50 (gate_1a passed 4.098e-05 deg). TWO ANOMALIES make the numbers suspect before any interpretation:
(A) Checkpoint load logged LARGE missing-key lists (character_torch skeleton/mesh/parameter_transform/blend_shape components) — the harness may be decoding with an incomplete character model. The errand had to override mhr_decode.py's hardcoded default checkpoint path to /home/arnavchokshi/coldstart_20260706/body_runtime/checkpoints/sam-3d-body-dinov3/{model.ckpt,assets/mhr_model.pt} (VM paths; evidence copies under runs/lanes/w6_gpu_instrument_20260708/gate1b_harness/).
(B) --scale-source none vs field produced BYTE-IDENTICAL results despite body_mesh.json carrying real non-zero per-frame scale — contradicting the code comment that population-mean scale explains the divergence. Either the flag is a no-op (bug), or scale genuinely doesn't enter the world round-trip (then the comment is wrong).

## EVIDENCE TO READ FIRST
- runs/lanes/w6_gpu_instrument_20260708/gate1b_harness/ (both harness reports + logs incl. the missing-key dump)
- runs/lanes/w5_p22latent_20260707/ (spec, vm_evidence — how GATE 1a/1b were measured in wave-5, what checkpoint/assets THAT run loaded, and what its partial GATE-1b numbers were)
- threed/racketsport/mhr_decode.py + runs/lanes/w5_p22latent_20260707/scripts/gate_check.py (code paths for checkpoint load, scale handling, world round-trip)
- runs/manager/w5_rider2_score/latent_smoothing_acceptance_report.md

## DELIVER (facts, no recommendation)
1. Whether the missing character_torch keys are (a) expected (keys unused by the decode round-trip), or (b) load into randomly-init/zero modules that the round-trip DOES use — trace the exact modules consumed by decode(emit) and the mesh-skel comparison. If determinable CPU-side with a tiny latent sample from the pulled artifacts, demonstrate.
2. The scale question: trace --scale-source through gate_check.py/mhr_decode — is it a no-op on the world round-trip path (bug) or semantically unused (comment wrong)? Show the code path.
3. Compare wave-5's GATE-1b partial measurement setup vs the errand's: same checkpoint? same assets? same metric keys? Did wave-5 ALSO see ~hundreds-of-mm round-trip on real frames (i.e., is 233mm consistent with the known 'measured-partial' state) or is this new?
4. A one-page adjudication table: for each anomaly — mechanism, evidence path, and what a CORRECT harness invocation would look like (exact flags/paths) IF one exists; or 'harness fix required in <file>' if not.
## REPORT (schema-enforced): objective_result (PASS = all 4 delivered); full_suite N/A read-only; HONEST ISSUES; artifact paths; proposed bullet; NEXT.
