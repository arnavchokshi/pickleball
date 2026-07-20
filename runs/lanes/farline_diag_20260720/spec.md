# farline_diag_20260720 — can cross-frame pooling recover far_centerline on the REAL Drill clip?

Codex gpt-5.6-sol xhigh, CPU. THE live court question from tonight's replay (runs/lanes/pbv_replay_20260720/MANAGER_RULING.md): on a fresh real pickleball clip, auto-calibration found EVERY required line EXCEPT far_centerline across 186s → the fail-closed gate blocked the entire people/BODY stack. T14 built default-off pooling/lookalike levers (runs/lanes/court_line_hardening_20260720/, in the working tree) but never tested this clip. Answer ONE question with a number: does cross-frame pooling (or any T14 lever) recover a usable far_centerline here?

## INPUTS (all local)
- Video: data/pbv_replay_20260720/xkadsq9bli3h/max.mp4 (sha 5085ae6e..., 1920x1080@60fps, 186s, STATIC camera).
- The replay run's pulled line evidence: runs/lanes/pbv_replay_20260720/vm_pull/process_video_pbv_replay_xkadsq9bli3h_20260720/court_line_evidence.json (+ court_correction_task.json) — shows exactly what the frozen detector found per line class.
- T14's modules in the tree: threed/racketsport/court_line_robustness.py (or as named), court_line_keypoints.py levers.

## DO
1. Reproduce the failure: run the frozen line detection on a sample of frames (e.g. 60-120 frames spread over the clip) and confirm far_centerline is missing/never-accepted; report per-line acceptance counts.
2. Apply T14's cross-frame pooling + lookalike rejection over the sampled frames (static camera ⇒ pooling is valid). Report: is a far_centerline candidate recovered? With what support/frames/residual? Does the pooled line pass the evidence-readiness bar that gated tracking (auto_calibration_ready)?
3. If pooling alone fails: diagnose WHY (line not visible at all? contrast? occlusion? detector never proposes it? lookalike suppression?) with 2-3 annotated evidence frames (save JPEGs to the lane dir).
4. HONEST output either way: RECOVERED (report the would-be calibration readiness) or NOT-RECOVERABLE-BY-POOLING (with the visual reason).

## RULES: no commits; read-only on all shared code except your lane dir (run T14 levers via imports/copies in the lane dir if needed); no GPU; the frozen harness verdict on the 5 venues stands (this is a DIFFERENT question — a 6th real-world case). Report: {far_centerline_baseline_accept_count, pooled_recovered:bool, support_frames, residual_px, would_pass_gate:bool, reason_if_not, evidence_jpegs:[]}.
