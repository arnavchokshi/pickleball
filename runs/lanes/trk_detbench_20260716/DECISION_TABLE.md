# DECISION_TABLE — trk_detbench_20260716

Status: **COMPLETE** (dispatch 2, post-AMENDMENT-1). All arms ran on `pickleball-h100-detbench`
(H100 SPOT, us-central1-a, RUNNING 16:36:37Z → DELETED 17:16:41Z, wall **0.67h**, est **$1.5-2.5**,
under the $15 cap). Diagnostic only; historical-internal clips; **VERIFIED=0 unchanged; no
promotion; no best_stack change.** Full evidence: `report.json`; pulled artifacts two-sided md5
`4ccc612921eba5daa047eacf7571d3c4` (+ 112-file sha256 manifest).

## Gate results first

- **Arm 0a (baseline reproduction from frozen pools): PASS.** Both clips match the preflip_score
  pins to ~3e-11 (bar 0.0001): burlington IDF1 0.8830775881 / cov4 0.7116666667, wolverine IDF1
  0.8515962036 / cov4 0.76, 0 switches, 0 spectator FP, 0 far-off-court FP. The frozen
  scorer+association harness is byte-faithful on the VM (all 5 script md5s + both ckpt sha256s
  matched two-sided).
- **Arm 0b (feeder confound check): FEEDER_DRIFT** (|Δ| ≫ 0.005 both clips). Same YOLO26m at the
  production operating point (conf .05 / imgsz 1536 / cls 0) through the lane's per-frame
  BOTSORT-`update()` feeder produces a materially different pool than the frozen production raw
  pools: burlington cov4 0.9733 vs 0.7117 (ΔIDF1 +0.0166), wolverine cov4 0.9267 vs 0.76 but with
  **2 new switches + 19 true-spectator FPs** (ΔIDF1 −0.0324). Per spec, all candidate arms below
  are therefore read **paired vs arm 0b** (same feeder protocol), with deltas vs the frozen 0a
  baseline also shown.

## The table (baseline rows first; det = detector ms/frame batch-1)

### burlington_gold_0300_low_steep_corner (600f)

| arm | IDF1 | cov4 | switches | spectator FP | far-off-court FP | HOTA | DetA | det ms/f |
|---|---|---|---|---|---|---|---|---|
| **0a YOLO26m frozen baseline** | 0.8831 | 0.7117 | 0 | 0 | 0 | 0.8892 | 0.7906 | — (frozen pool) |
| 0b YOLO26m lane feeder | 0.8997 | 0.9733 | 0 | 0 | 0 | 0.9042 | 0.8176 | 34.6 |
| 1 RF-DETR-L | **0.9204** | **0.9967** | 0 | 0 | 0 | 0.9233 | 0.8525 | 31.2 |
| 2 RF-DETR-Seg-L (boxes) | 0.8771 | 0.6617 | 0 | 0 | 0 | 0.8838 | 0.7811 | 263.3 |
| 3 D-FINE-L | 0.8893 | 0.7100 | 0 | 0 | 0 | 0.8948 | 0.8007 | 24.5 |
| 4 DEIMv2-L | 0.9088 | 0.9117 | 0 | 0 | 0 | 0.9126 | 0.8328 | 21.0 |

### wolverine_mixed_0200_mid_steep_corner (300f, worst clip)

| arm | IDF1 | cov4 | switches | spectator FP | far-off-court FP | HOTA | DetA | det ms/f |
|---|---|---|---|---|---|---|---|---|
| **0a YOLO26m frozen baseline** | 0.8516 | 0.7600 | 0 | 0 | 0 | 0.8611 | 0.7415 | — (frozen pool) |
| 0b YOLO26m lane feeder | 0.8192 | 0.9267 | 2 | 19 | 0 | 0.8104 | 0.7866 | 38.1 |
| 1 RF-DETR-L | 0.7955 | 0.7267 | 1 | 16 | 0 | 0.7947 | 0.7224 | 31.8 |
| 2 RF-DETR-Seg-L (boxes) | 0.7433 | 0.3900 | 2 | 0 | 0 | 0.7479 | 0.6701 | 253.6 |
| 3 D-FINE-L | 0.8157 | 0.5767 | 1 | 0 | 0 | 0.8130 | 0.7446 | 25.0 |
| 4 DEIMv2-L | 0.7005 | 0.3100 | 5 | 18 | 0 | 0.7101 | 0.6281 | 21.6 |

(Every switch/spectator-FP in this bench occurs on wolverine and only under the feeder protocol —
including for YOLO26m itself (arm 0b). far-off-court FP was 0 for every arm on both clips.)

## Verdicts (spec vocabulary: adopt-next-step / reject / no-attempt)

| arm | verdict | basis |
|---|---|---|
| 1 RF-DETR-L | **reject** as zero-shot drop-in; **adopt-next-step** as the fine-tune base | wolverine fails gate axes vs frozen baseline (1 switch, 16 spectator FP) and loses cov4 −0.20 vs paired 0b; but burlington is the best row of the bench (0.9204/0.9967, clean), it beats its paired baseline on wolverine FP axes (1<2 sw, 16<19 spectFP), it is FASTER than YOLO26m (31 vs 35-38 ms/f), and it is Apache-2.0 |
| 2 RF-DETR-Seg-L | **reject** (boxes) | cov4 collapse both clips (0.66/0.39 — the predicted 504-native far-player recall regression), 2 switches, detector wall ~7.4× (>20% kill rule). Masks archived (coco_rle, both clips) for the future mask-cue lane — mission accomplished for this arm's other purpose |
| 3 D-FINE-L | **reject** | par with frozen baseline on burlington, worse everywhere on wolverine (cov4 0.577, 1 switch); no gate-relevant gain |
| 4 DEIMv2-L | **reject** | worst wolverine row (5 switches, 18 spectator FP, cov4 0.31) despite good burlington + fastest detector; far-side instability disqualifies |

## Fine-tune decision arm: **GO** (RF-DETR-L base)

1. **The ceiling is real:** zero-shot RF-DETR-L hits cov4 0.9967 on burlington with zero
   FP-axis violations — above even the 0.95 promotion-bar level on that clip.
2. **Speed is free:** 31 ms/f fp32 unoptimized beats the 35-38 ms/f incumbent feeder — no runtime
   tax; headroom via fp16 `optimize_for_inference` remains unexploited.
3. **The failure is domain-shaped, not architecture-shaped:** every detector including YOLO26m
   itself degrades on wolverine's far-side small players + spectator-adjacent pool under the
   feeder protocol. That is precisely the target of the ruled fine-tune recipe (single
   `on_court_player` class + labeled spectator/passer hard negatives + empty-court frames).
4. **Independent controls confirmed rank-1:** D-FINE-L and DEIMv2-L (different labs, different
   training pipelines) did not beat RF-DETR-L — the crosscheck ruling's ordering held on our data.
5. **License:** Apache-2.0 vs the AGPL-3.0 incumbent (NS-07.3 relicense pressure).

**Cautions binding the fine-tune lane:** (a) zero-shot RF-DETR-L is NOT adoptable today — the
pre-registered stop rule stands (two frozen-threshold attempts without material coverage gain, or
ANY new switch/spectator/far FP → stop the branch); (b) FEEDER_DRIFT means fine-tune evals must
run through the production pool path or the pinned paired protocol — never mix; (c) rfdetr 1.8.3
partial-load warnings (group_detr flat-slice, `_kp_active_mask` random-init) were empirically
benign for inference here but must be pinned down before training.

## Dispatch history (dated)

- **Dispatch 1 (16:03-16:27Z): NO-ATTEMPT.** 6/6 H100-only zone-ladder attempts stockout
  (`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`: ase1-b, ase1-c, usc1-a, usc1-b, euw4-b, ase1-b
  retry), $0, zero orphans (instances+disks list-confirmed). Evidence in
  `logs/provision_attempts.log`.
- **AMENDMENT 1 (manager):** SKU fallback ladder authorized (2× H100 quick → A100-80 → A100-40).
- **Dispatch 2 (16:31Z):** amended attempt 1 (H100 ase1-b) stockout; amended attempt 2 (H100
  us-central1-a) SUCCESS 16:36:37Z. A100 tiers never needed.
- **Teardown:** DELETE 17:16:41Z; instances list → only historical `pickleball-a100-fleet1`
  TERMINATED (at delete time); `name~pickleball-h100-detbench` → 0 items; disks `name~detbench` →
  0 items. Wall 0.67h, est $1.5-2.5.
