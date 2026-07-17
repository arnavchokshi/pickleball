# POOLDIAG Phase 1 — vm_rerun (trk_rfdetr_prod_20260716)

Status: **EXECUTED** (Mac-side, pure artifact forensics — deterministic JSON reads + IoU matching
only; no association, no scorer, no inference; the Mac unfaithfulness finding applies to the
CPU association/scorer path, not to reading frozen artifacts). Wall: ~14 min (bar 30).
Diagnostic only; VERIFIED=0; nothing tuned.

## One-line answer

The frozen production pools were built by `scripts/racketsport/benchmark_person_trackers.py` at
its DEFAULTS — **detector conf floor 0.18, imgsz 960** — not at the orchestrator production
operating point (conf 0.05, imgsz 1536) that the detbench arm-0b feeder used per its spec; the
coverage delta comes almost entirely from this operating-point difference changing track
stability/fragmentation, and its price on wolverine is +2 switches +19 spectator FPs.

## Evidence

1. **Config identity (code):** `benchmark_person_trackers.py` argparse defaults: `--imgsz 960`,
   `--conf 0.18`, `--iou 0.6`. The frozen pool metrics.json records
   `tracker_config=configs/racketsport/botsort_no_reid_loose.yaml`, `variant=loose_pool`,
   `model=yolo26m.pt` (same yaml + weights as the feeder — tracker config is NOT the variable).
2. **Conf floor (artifact):** production pool min conf = **0.1800** (burlington) / **0.1846**
   (wolverine) — exactly the 0.18 floor. Feeder pool min conf = 0.1204/0.1220 (the BOTSORT
   track_high_thresh 0.12 boundary, detector floor 0.05).
3. **Persistence (artifact + code):** production writes ALL tracker-output person boxes —
   `untracked_person_boxes=0`, `tracked_detections.json` content-identical to
   `raw_tracked_detections.json` (scale 1.0). No hidden post-filter.
4. **Box-level diff (IoU≥0.6 cross-match, per frame):**
   - burlington: 8,745 boxes in both; 1,511 feeder-only (1,223 at conf≥.18 + 288 below);
     920 production-only. Of the 1,511 feeder-only boxes, only **11 match GT** (IoU≥.5) —
     the extra feeder boxes are overwhelmingly spectators/background, NOT players.
   - wolverine: 1,973 both; 90 feeder-only (3 GT-matched); 33 production-only (10 GT-matched).
5. **Fragmentation (artifact):** burlington track_ids: production **68** vs feeder **52**;
   wolverine 23 vs 26. Feeder tracks on burlington are markedly longer/steadier.

## Interpretation (why cov4 jumped 0.71→0.97 without new GT boxes)

The cov4 gain is NOT recall of previously-missed players (only ~11 new GT-matched boxes).
It comes from the association stage receiving **fewer, longer, steadier fragments** (68→52 ids)
— imgsz 1536 (vs 960) yields more stable far-player boxes frame-to-frame, and the lower conf
floor keeps tracks alive through conf dips instead of fragmenting them. Better fragments →
better global association → more frames with all four selected players. The price appears on
wolverine, where steadier spectator tracks survive association: +2 switches, +19 true-spectator
FPs (detbench arm 0b).

## Mechanism table

| mechanism | status | evidence |
|---|---|---|
| M1 saved-pool filtering | **EXCLUDED** | untracked=0; tracked==raw content-identical; all tracker output persisted |
| M2 BoT-SORT lifecycle (track() vs update()) | **EXCLUDED** (upgraded from "excluded as primary" by the end-to-end confirmation below) | same yaml both paths; per-frame update() at the pool operating point reproduces the frozen result EXACTLY |
| M3 GMC | **EXCLUDED** | same `gmc_method: sparseOptFlow` in the single shared yaml; same video frames |
| M4 preprocessing / operating point | **CONFIRMED (total, end-to-end)** | conf floor 0.18@960 vs 0.05@1536; conf_min in artifacts matches the 0.18 default exactly; AND the on-VM confirmation run (below) shows this is the ENTIRE difference |
| M5 ultralytics version divergence | **EXCLUDED** | 8.4.87 (VM) at the pool operating point reproduces the frozen pins exactly; whatever generated the 07-13 pools, 8.4.87 is score-equivalent to it |

## End-to-end confirmation (VM, score-faithful environment, 2026-07-17)

Fresh YOLO26m detections at **conf 0.18 / imgsz 960** (the `benchmark_person_trackers.py`
defaults) through the SAME per-frame BOTSORT-`update()` feeder that produced FEEDER_DRIFT at
0.05/1536, then the frozen association + scorer:

| clip | IDF1 | Δ vs frozen pin | cov4 | Δ | sw | spFP | farFP |
|---|---|---|---|---|---|---|---|
| burlington | 0.883078 | **+0.000000** | 0.711667 | **+0.000000** | 0 | 0 | 0 |
| wolverine | 0.851596 | **+0.000000** | 0.760000 | **+0.000000** | 0 | 0 | 0 |

The feeder at the pool operating point reproduces the frozen baseline EXACTLY. The detbench
FEEDER_DRIFT was therefore 100% operating-point (conf floor + input size), 0% construction-path.
The per-frame `update()` feeder is production-equivalent when driven at the pool operating point.

## Consequence for variant P (RF-DETR-L production-equivalent)

Production-equivalence for a fixed-resolution detector (RF-DETR-L native 704) reduces to:
**conf floor 0.18 before tracking + same botsort_no_reid_loose per-frame BOTSORT + persist-all**.
P inputs built from the PINNED arm-1 raw detections (sha-verified checkpoint, person id 1):
burlington 37,175→11,697 dets, wolverine 5,259→2,204 dets after the 0.18 filter
(`vm_rerun/p_inputs/*.rfdetr_p.json`, md5 24b835a68f76f981ed822f89558b9a42 /
ea144397231f5ac31fd00e834e04e32d). The BOTSORT+association+scoring steps run ONLY on the VM
(the environment proved score-faithful at ~3e-11), never on this Mac.
