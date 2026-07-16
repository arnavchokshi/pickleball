# ADOPTION REPORT — NS-03.TRK detector/domain research (Track F, 2026-07-16)

Method: dual independent codex gpt-5.6-sol surveys (sibling-fenced) + manager cross-check +
2-vote adversarial refutation (trk_refute_20260716). Every ranked artifact HEAD-live-checked
2026-07-16. Statuses: [CORROBORATED x2] = both surveys independently; [REFUTED→CORRECTED] =
survivor after refutation. Research-only: VERIFIED=0; published numbers are motivation, never
pickleball accuracy. Ruled sequence unchanged: detector/domain → ReID → mask cue; association
sweeps stay banned.

## Ranked adoption list

| Rank | Candidate | Status | License posture (NS-07.3) | Why / evidence |
|---:|---|---|---|---|
| 1 | **RF-DETR-L detection zero-shot** (`rf-detr-large-2026.pth`, 704px) | benchmark FIRST | code+weights Apache-2.0; COCO data `unknown-needs-review` | [CORROBORATED x2 + refute-confirmed] 56.5 COCO AP, mature fine-tune path, exact live artifact. NO official crowded-person evidence exists — our frozen worst-clip card is the deciding test. |
| 2 | **RF-DETR-L owned-data fine-tune** w/ explicit spectator negatives | the decision arm | our data = only commercial-clean supervision | [CORROBORATED x2] Both surveys independently ranked domain supervision above any model swap for the cov4 0.71→0.95 gap. Small-data sports precedents exist but are weaker than quoted (refutation C8: transductive domain labels, not a boxes-volume law). |
| 3 | **RF-DETR-Seg-L zero-shot** (boxes scored; masks archived) | benchmark same session | Apache-2.0 weights (ALL seg sizes) | [CORROBORATED x2] Per-frame det+mask without temporal drift; feeds the ruled step-3 mask cue. Risk: 504 native res may cost far-player recall — scored separately. |
| 4 | **D-FINE-L (`dfine_l_obj2coco_e25.pth`) + DEIMv2-L (`DEIMv2_DINOv3_L_COCO`)** | controls, one run each | Apache code; weights undeclared; data review | [REFUTED→CORRECTED pins] Independent DETR-lineage points to attribute any RF-DETR gain. EdgeCrafter EXCLUDED as a DEIMv2 same-lab sibling (refutation C7). |
| 5 | **Owned 4-player enrollment gallery ReID** (centroids + open-set reject) | next lane after detector freeze | commercial-clean if owned crops + cleared backbone | [CORROBORATED x2] NO public ReID checkpoint is commercial-clean (Market-1501 terms unresolved-at-best, refutation C9c; KPR Hippocratic; SapiensID CC-BY-NC). Published sports support: ~6 tracklets/player sufficed (rugby, confirmed). SOLIDER = R&D diagnostic only. |
| 6 | **McByte mask cue** — worst-clip forensics ONLY | bounded step-3 probe | MIT code; SAM/Cutie deps permissive; its SportsMOT-trained detector asset R&D-only | [refute-CONFIRMED] 3-5 FPS end-to-end on A100 from its own paper (≈12-20x source duration derived) — can prove mask-cue VALUE, can never ship as-is. Feed our frozen detections + archived Seg-L masks. |
| 7 | **Selective-window mask propagation** (detection-gap + ambiguity triggers) | design to adopt if McByte shows value | paper CC-BY-4.0; third-party MIT impl (holma91/selective-mask-propagation) unofficial | [REFUTED→UPGRADED] Numbers confirmed (masks +1.5 HOTA vs GTA +7.8 — association-dominated, which we're banned from re-sweeping anyway); code claim corrected: MIT reimplementation appeared 2026-07-14, validate before trust. Our variant must add a DETECTION-GAP trigger since coverage is our failure mode. |
| 8 | **SAM2.1-B+ selective windows** (Apache) | fallback mask source | code+ckpt Apache-2.0 | [refute-CONFIRMED] Multi-target reality: 17.8/12.4 FPS @3/5 targets (A6000) — always-on VOS is out; selective re-prompted windows only. |
| — | Rejected/blocked: SAM-MT (CC-BY-NC-SA ckpt, no code license), SapiensID (CC-BY-NC), KPR (HL3), CAMELTrack global ckpt (R&D-trained mix; association anyway), open-vocab as primary detector, EdgeCrafter (non-independent) | — | — | Evidence in surveys + REFUTATION.md. |

## Load-bearing facts for the program

1. **Our incumbent detector is AGPL-3.0** (Ultralytics YOLO26m, confirmed real public family,
   exact asset pinned). The RF-DETR swap is simultaneously an accuracy experiment AND the
   NS-07.3 commercial-clean move for the detection seat.
2. **Coverage (cov4 0.71-0.76 vs 0.95), not switches, is the whole gap** — worst clips already
   have 0 switches. Everything association-side is a later safeguard, exactly as the North Star
   ruled.
3. **Public sports MOT data cannot enter the product stack** (SportsMOT CC-BY-NC, DanceTrack
   videos NC, SoccerNet NDA). Owned labeled frames are the only clean fine-tune source; public
   sets are R&D diagnostics.
4. **No off-the-shelf tracker solves 4-enrolled-players + open-set spectator rejection**
   [CORROBORATED x2] — that layer (enrollment gallery, open-set thresholds, soft court-footpoint
   prior with truncation-aware uncertainty) is ours to build.

## What to benchmark first and why

Arms 1-4 of `benchmark_spec_trk.md` (this dir): RF-DETR-L zero-shot locates the floor in <1
GPU-hour; Seg-L adds the mask archive; D-FINE/DEIMv2 controls attribute architecture vs
resolution; the owned-data fine-tune is the arm both surveys independently predict moves cov4.
Protocol, baseline-reproduction bar, stop rules, and exact artifact pins are in the spec.
ReID enrollment and McByte forensics are the two follow-on lanes, in that ruled order.

## Paths

- Spec: runs/research_trk_rkt_20260716/benchmark_spec_trk.md
- Ruling: runs/research_trk_rkt_20260716/TRK_CROSSCHECK_RULING.md
- Surveys: trk_survey_A_20260716/SURVEY.md, trk_survey_B_20260716/SURVEY.md (+livechecks)
- Refutation: trk_refute_20260716/REFUTATION.md (+livechecks)
