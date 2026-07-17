# Lane spec — trk_pooldiag_20260716 (SPEC-ONLY, booked NOT dispatched)

Booked by Track F manager 2026-07-16 per coordinator directive, from the trk_detbench finding:
the SAME detector (YOLO26m, conf=0.05, imgsz=1536, cls 0) produces materially different pools via
the production path (process_video → orchestrator `model.track`, frozen pools → post-association
cov4 0.7117/0.76) vs a per-frame BOTSORT-`update()` feeder (cov4 0.9733/0.9267, but wolverine
+2 switches +19 spectator FP). Pool CONSTRUCTION is a coverage variable of the same order as the
detector swap itself. This lane ATTRIBUTES the delta; it does not tune anything.

## Question

Which mechanical difference(s) between the two pool-construction paths explain the coverage gap
and the wolverine FP/switch cost? Candidate mechanisms to test (enumerated up front; no fishing):
M1 saved-pool filtering (production may persist only confirmed/pruned tracks into
tracked_detections.json rather than all detections); M2 BoT-SORT lifecycle differences between
ultralytics' track pipeline and raw `update()` (activation/new_track thresholds, first-frame
handling); M3 GMC/camera-motion-compensation on in one path, off/different in the other; M4
frame preprocessing (letterbox/stride/color/vid_stride/batch) differences; M5 ultralytics version
divergence Mac/VM/snapshot.

## Protocol (bounded, two phases, CPU/MPS-first)

- Phase 1 — pure artifact forensics, NO new inference: diff the frozen production pools
  (runs/lanes/trk_flip_20260713/{default,preflip}_production/<clip>/) against the feeder pools
  (runs/lanes/trk_detbench_20260716/vm_pull/detbench_out/pools/arm0b_*/). Per frame: detection
  counts, conf histograms, which GT-matched boxes exist in one pool and not the other (use frozen
  GT read-only for matching), track_id fragmentation stats. Expected outcome: directly implicates
  M1 (count asymmetry concentrated in low-conf/short-track detections) or points at M2/M3
  (systematic early-track/motion-window differences). Deliverable: attribution table per clip.
- Phase 2 — ONLY if Phase 1 cannot separate M2-M5: single local rerun of the production tracking
  stage on the two eval clips (Mac MPS or CPU; 900 frames total is feasible) with a debug dump of
  pre-tracker detections + post-tracker persisted set, plus one feeder rerun pinned to the SAME
  ultralytics version; compare stage-by-stage. NO GPU provisioning for this.

## Rails

- NOT an association sweep: association knobs, margins, and thresholds are frozen; nothing is
  tuned; the output is an attribution REPORT (runs/lanes/trk_pooldiag_20260716/REPORT.md), not a
  candidate. Any follow-up change to the production pool builder must be its own ruled lane with
  the full frozen card + stop rules.
- Fence: runs/lanes/trk_pooldiag_20260716/** only; read-only elsewhere; no pipeline edits.
- Interaction with the fine-tune lane: the fine-tune evaluates through the PRODUCTION pool path
  (per the FEEDER_DRIFT ruling); this diagnostic tells us whether a later pool-builder fix could
  stack with detector gains. Sequencing: after the fine-tune lane, or parallel (file-disjoint).
- Budget: half-day Codex/Sonnet CPU lane; no VM.

## Success criterion

Each mechanism M1-M5 marked CONFIRMED / EXCLUDED / UNRESOLVED with file-level evidence; a one-line
answer to "is the production pool builder leaving recoverable coverage on the table, and at what
FP/switch price?" VERIFIED=0; diagnostic only.
