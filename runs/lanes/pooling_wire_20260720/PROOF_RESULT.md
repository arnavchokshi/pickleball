# Pooled-calibration proof run — RESULT (2026-07-20): PLAYERS ON A FRESH CLIP ✓

Drill Session (xkadsq9bli3h, 186s, fresh pb.vision clip, static camera). Yesterday: Players 0 —
tracking hard-blocked by missing far_centerline. Today, with --court-line-evidence-pooling:

- calibration gate OPENED (pooled evidence; reused content-addressed calibration from the run chain)
- tracking RAN (1015s): 4 tracks — the two real players carry 5,669 and 5,143 frames of ~11,168
  (two minor fragments 563/592 frames, honest)
- placement RAN, frames RAN, world RAN; BODY degraded: 0 full 3D meshes COMPUTED (1,368 deep targets scheduled; CUDA context conflict — the startup script's EXCLUSIVE_PROCESS mode blocked the BODY subprocess's second context, self-inflicted, fix queued); skeleton fallback covered 5,984 player-frames; paddle blocked (unchanged); ball_arc skipped
  (typed, per the known stage-cap defect); manifest/verify degraded (missing optional pieces, marked)
- run status: PARTIAL (honest) — but the people chain on a fresh clip is ALIVE end-to-end for the
  first time, zero human input.

LIVE-RUN CONFIRMATION (agent report): far_centerline recovered in PRODUCTION at 68 support / 0.359px (diagnostic said 63/0.357 — reproduced on real decode); auto_calibration_ready TRUE; NO court_correction_task written (first time on this clip); viewer browser-verified w/ screenshot (Players=4, zero page errors).
NEW DEFECTS EXPOSED (players flowing deep for the first time): coaching_facts ValueError missing players[].frames[].track_world_xy; EXCLUSIVE_PROCESS vs BODY subprocess CUDA conflict; verify-viewer cold-start selector timeout. All queued.
RECORD NOTE: the "out-of-band process" the agent observed (rail extension + VM deletion + partial pull + this file) was the MANAGER acting directly — coordinated, not anomalous. Where this file and the agent's pulled artifacts disagreed (mesh count), the agent's directly-inspected artifacts are authoritative and this file is corrected.
Artifacts: gpu_replay_pull/ (summary, tracks, pooled evidence, placement, body execution,
calibration). VM deleted. ~$3. VERIFIED=0 — this is a scoped real-clip proof, not a promotion.
