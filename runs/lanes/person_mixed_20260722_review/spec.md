# ULTRA REVIEW of lane person_mixed_20260722 (read-only) — owner-directed mixed-pool experiment pack

Ground truth: runs/lanes/person_mixed_20260722/spec.md (incl. the AMENDMENT RULING header),
report.json, owned files build_person_mixed_pseudo_pack.py + its test, artifacts in the lane dir.

Verify with negative-input validation: (1) quarantine refusals are structural (protected clips,
3 compare-only pb.vision IDs, IYnbdRs1Jdk) — try aliased paths/renamed IDs; (2) pseudo rows can
never enter validation (structural, not conventional); (3) anchor/val split byte-fidelity vs the
closed P1 lane artifacts (hashes must match its committed manifests); (4) caps math (400/source,
15%/family) on the real plan; (5) teacher constants preregistered (conf 0.60, SHA-pinned YOLO26m)
with overrides marking non-production; (6) determinism; (7) the blind-spot caveat + bars recorded
in the pack manifest verbatim; (8) the decode-plan honesty (7,200 are candidates, not labels —
report must not claim materialized data). Note license_field=null on harvest sources is recorded
as experiment-only per owner directive — confirm the recording exists; do not relitigate the
owner's scope decision.
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT, plus GPU_DISPATCH_DECISION with
on-VM preconditions for the teacher-inference + retrain phase.
