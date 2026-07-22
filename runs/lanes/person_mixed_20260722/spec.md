# LANE person_mixed_20260722 — owner-directed mixed-pool self-training PERSON pack builder (CPU code phase)

## AUTHORITY + AMENDMENT RULING
Owner directive 2026-07-21 (via main): EXACT_PLAN P2's closure is AMENDED — closed for
Roboflow-only, OPEN for a mixed-pool self-training EXPERIMENT. PERSON_RF_POOL_TOO_THIN stands.
No promotion path: VERIFIED=0, best_stack.json untouched, do_not_promote everywhere. The
self-training blind-spot caveat (teacher misses become background holes in pseudo frames) is
OWNER-ACCEPTED and must be recorded verbatim in the pack manifest.

## HARD RULES
- No branches/commits. Quarantines UNCHANGED and binding: the 4 protected eval clips, the 3
  compare-only pb.vision IDs (83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr), and every IYnbdRs1Jdk
  derivative are structurally excluded (refusal + tests). Pseudo-labels NEVER appear in any
  validation/eval set. Honest reporting; WIDE suite; no NEW failures beyond the known set.
- Artifacts under runs/lanes/person_mixed_20260722/.

## FILE OWNERSHIP (exclusive)
- scripts/racketsport/build_person_mixed_pseudo_pack.py (new)
- tests/racketsport/test_build_person_mixed_pseudo_pack.py (new)
Nothing else. Do NOT touch export_roboflow_person_yolo_dataset.py (closed lane) — READ its
leak-fixed split artifacts as input evidence only.

## DESIGN (pre-registered; constants in code, CLI overrides mark production_eligible=false)
1. ANCHOR SET: the leak-fixed Roboflow 7-family train split from
   runs/lanes/person_p1_roboflow_20260721/roboflow_person* (read its manifests; do not recompute
   families). Validation = its held-out human families (od8al mega-family val + hemel test),
   human-only, byte-identical to the closed lane's split.
2. PSEUDO SOURCES: (a) the non-compare pb.vision gallery videos (10 IDs incl. xkadsq9bli3h);
   (b) the online_harvest_20260706 downloaded sources (all 8; note HyUqT7zFiwk/Ezz6HDNHlnk are
   BALL-judge holdouts, NOT person-protected — include them, but stamp each row with its
   cross-component holdout roles so the ledger can track). Media may be absent locally — the
   builder emits a decode-plan manifest with per-source frame indices + expected media SHAs; the
   GPU lane materializes crops/labels on the VM.
3. SAMPLING (preregistered): uniform temporal stride per source, cap 400 sampled frames/source
   video and cap 15% of total pack per venue/source family; target 6,000-12,000 pseudo frames
   across >=15 distinct source families (report exact achievable counts).
4. TEACHER (preregistered): stock YOLO26m (models/checkpoints/yolo26m.pt, SHA-pinned vs
   models/MANIFEST.json), person class only, confidence >= 0.60, NMS defaults; every pseudo row
   carries teacher_derived=true, ground_truth=false, teacher_conf, and the teacher checkpoint SHA.
5. OUTPUT: a YOLO-trainable pack manifest (pseudo train shard + anchor train shard, deterministic
   interleave plan with anchor:pseudo exposure recorded) + a data.yaml TEMPLATE gated on the GPU
   lane completing teacher inference; per-source/per-family count tables; the blind-spot caveat +
   experiment bars recorded: mixed must beat anchor-only control on held-out-family F1/mAP50 with
   BOTH val families non-negative.
6. Tests: quarantine refusals (protected/compare/IYnbdRs1Jdk), caps enforced, determinism,
   pseudo-never-in-val structural check, anchor split byte-fidelity vs the closed lane's artifact.

## DATA CONTRACT
CPU only this lane; GPU teacher-inference + retrain is a separate lane after ultra review.
End-of-lane numbers: planned pseudo-frame counts per source family; anchor/val fidelity hashes.

## CROSS-SIGNAL
Consumes: stock person detector (teacher), pb.vision + harvest media inventory, closed P1 splits.
Feeds: PERSON mixed-pool experiment arm; data-steward ledger rows for every consumed source.

## BEST-STACK DELTA
None — experiment with no promotion path (owner-directed).

## MANDATORY STRUCTURED REPORT
objective_result; full_suite counts; HONEST ISSUES; artifacts; the count tables.
