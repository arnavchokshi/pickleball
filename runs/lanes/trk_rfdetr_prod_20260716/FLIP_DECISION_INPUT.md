# RF-DETR-L preview-flip decision input — Track F manager, 2026-07-17

Per the coordinator's decision frame ("pass → proposal; FPs survive → honest numbers +
ship-for-demo posture unless egregious — manager judgment, coordinator ruling"). VERIFIED=0;
this is an owner-directed stack-selection question, never an accuracy promotion.

## The numbers (frozen card, frozen scorer, VM-faithful environment; full table vm_rerun/report.json)

| clip | arm | IDF1 | cov4 | sw | spectFP | farFP |
|---|---|---|---|---|---|---|
| burlington | YOLO26m baseline | 0.8831 | 0.7117 | 0 | 0 | 0 |
| burlington | **RF-DETR-L (production-equivalent P)** | **0.9220** | **0.9933** | 0 | 0 | 0 |
| wolverine | YOLO26m baseline | 0.8516 | 0.7600 | 0 | 0 | 0 |
| wolverine | **RF-DETR-L (production-equivalent P)** | 0.8036 | 0.7233 | **1** | **4** | 0 |

Owner hypothesis HALF-CONFIRMED: the production conf floor (0.18 — see pool finding below) cut
wolverine spectator FPs 16→4, but not to zero; burlington is the best row ever recorded on the
clip and fully clean.

## Manager judgment: NOT egregious, but NOT a clean blanket flip

- The regression is confined to the worst clip, and it is exactly the owner's most-hated visual
  axes: ghost spectators (4 FP frames) + one identity switch, plus IDF1/cov4 below the incumbent
  there. A blanket flip makes one demo clip dramatically better and the other visibly worse in
  the way the owner complains about most. That is a per-clip trade, not a win.
- RECOMMENDATION: do not flip blanket as-is. Two rule-compliant options for the coordinator:
  1. **One remaining preregistered frozen-threshold attempt** (within the detbench stop rule's
     "two frozen-threshold attempts"): single declared conf floor 0.30 for RF-DETR-L variant P,
     one shot, no grid. Plausible mechanics: the 4 surviving FPs carry conf 0.18-0.30-ish while
     on-court players at 704 native run higher — could zero the FP axes at modest cov4 cost. If
     it zeroes wolverine's FP/switch axes while keeping burlington's gain: flip proposal follows
     immediately. If not: kill, and the flip question waits for the (parked) fine-tune arm.
  2. **Ship-for-demo anyway** per the owner's "good enough" posture, with the FP counts on the
     table: preview band, do_not_promote, and the wolverine per-clip regression stated in the
     best_stack notes verbatim. Mechanically real (Apache-2.0 weights, 31ms/f) but carries the
     visible-ghost risk on wolverine-like content.
- Either way the flip's code seam remains: orchestrator's RealYOLO26BoTSORTReIDTrackingRunner
  hardcodes yolo26m via MANIFEST — a detector flip needs a ruled integration lane (detector
  entry + pool-construction parity at conf 0.18 semantics + production reproduction bar).

## Load-bearing process findings banked this lane

1. **Frozen-pool provenance corrected:** the frozen production pools = benchmark_person_trackers
   DEFAULTS (conf 0.18, imgsz 960), NOT the orchestrator 0.05/1536 the detbench spec asserted
   (my spec error, now fixed in the record). FEEDER_DRIFT = 100% operating point; construction
   path (BOTSORT update vs model.track) is EXCLUDED as a factor — proven by exact-zero-delta
   reproduction through the feeder at 0.18/960. trk_pooldiag_20260716 spec: CLOSED by evidence
   (M4 CONFIRMED TOTAL; M1/M2/M3/M5 EXCLUDED).
2. **Platform pin required:** Mac CPU association is not score-faithful (gap-fill divergence);
   H100/A100 agree exactly. The frozen card protocol now implicitly includes: score on
   GPU-class VM environments only, never local CPU.
3. Honest caveats: variant P reused the pinned detbench RF-DETR detections (sha-pinned, not
   re-inferred on A100); the 4 wolverine FPs are scorer-labeled, not visually audited.

Evidence: vm_rerun/report.json, vm_rerun/POOLDIAG_PHASE1.md, pulled artifacts (50-file sha256
manifest, two-sided md5), incremental scores s01-s04. Cost $0.3-0.5 / 0.2h; teardown
list-confirmed; Track G untouched.
