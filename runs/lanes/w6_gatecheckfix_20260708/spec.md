# LANE w6_gatecheckfix_20260708 — canonicalize + fix the GATE-1b decode-fidelity harness (scale/hand_pose/provenance)

## HARD RULES
- NO branches/commits/pushes; no board edits (bullet text in report). Protected clips EVAL-ONLY. .venv/bin/python; MPLBACKEND=Agg. Artifacts under runs/lanes/w6_gatecheckfix_20260708/ only. Other lanes' run dirs are READ-ONLY EVIDENCE — you fix a COPY in a canonical home, never edit runs/lanes/w5_p22latent_20260707/scripts/gate_check.py in place.

## FILE OWNERSHIP (exclusive)
- OWNED: a NEW canonical harness under scripts/racketsport/ (e.g. gate_check_body_decode.py — pick a house-consistent name), its scaffold-index/direct-CLI reference tests, its unit tests.
- READ-ONLY: runs/lanes/w5_p22latent_20260707/** (the lane-local harness you supersede + vm_evidence), runs/lanes/w6_gate1b_harnessdiag_20260708/** (the adjudication table — your requirements doc), runs/lanes/w6_gpu_instrument_20260708/gate1b_harness/** (the flawed W6 reports), threed/racketsport/mhr_decode.py (consume, do not edit).
- DO NOT TOUCH: threed/racketsport/worldhmr.py + body_grounding_quality.py (w6_footpinstub just landed there), owner-label ingest files (w6_labelingest RUNNING), court/calibration files (CALV1), cvat_upload/**, web/replay/**.

## OBJECTIVE (from the harnessdiag adjudication — read its report + table FIRST)
The GATE-1b harness (w5 lane-local gate_check.py) produces round-trip numbers that are NOT a decode-fidelity verdict: (1) _extract_frames reads smplx_params but never stores smplx_params.scale, so --scale-source field is a NO-OP (real per-frame scale silently dropped); (2) nonzero left/right hand_pose is also dropped; (3) no decoder checkpoint/asset provenance is recorded, so runs are not comparable across waves. Build the canonical fixed harness so the next GPU errand can measure GATE-1b legitimately on the raw-postchain arm.

## THE DESIGN (pinned)
- Port gate_check.py into the canonical CLI with: full field plumbing (scale + left/right hand_pose + any other smplx_params fields mhr_decode consumes — enumerate them from MHRHead.mhr_forward's signature and document the mapping), checkpoint/asset provenance block in the output (paths + sha256 of model.ckpt and mhr_model.pt + mhr_decode module version stamp), explicit --scale-source semantics that FAIL LOUD if requested-but-absent, and unchanged metric key names (gate_1a_euler_cont_euler_idempotence, gate_1b_world_round_trip.worst_joints_world_max_abs_error_mm, mesh_skeleton_divergence.worst_p95_mm_over_sample, total_real_frame_sample_count) so wave-5/6 comparisons stay possible.
- Decoder unavailability locally (no roma, no checkpoints) is EXPECTED: the CLI must degrade loudly (clear MHR_RUNTIME_AVAILABLE=False message) and its field-plumbing must be testable WITHOUT the decoder.
## ACCEPTANCE
1. Unit tests (decoder-free, fixture body_mesh.json snippets): extracted frames carry scale + hand_pose verbatim; requested-but-absent scale FAILS LOUD; provenance block present; metric-key names asserted.
2. A --self-check mode proving on a synthetic fixture that fields flow end-to-end to the decode call boundary (mock/stub at the mhr_decode interface).
3. Scaffold-index/direct-CLI reference tests same-lane; focused suites green (your files + scaffold/doc guardrails). Full wide suite NOT required (concurrently dirty tree; wave-close adjudication final).
4. Report the EXACT GPU-errand invocation for the legitimate GATE-1b raw-arm measurement (fixed harness + raw body_mesh from a post-footpinstub raw run + checkpoint paths from the errand evidence).
## REPORT (schema-enforced): objective_result; full_suite (scoped); CHANGES; field-mapping table; GPU invocation; HONEST ISSUES; proposed bullet; NEXT.
