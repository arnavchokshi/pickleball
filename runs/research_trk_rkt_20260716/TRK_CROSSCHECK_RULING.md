# TRK dual-survey cross-check ruling — 2026-07-16 (Track F manager)

Sources: trk_survey_A_20260716/SURVEY.md x trk_survey_B_20260716/SURVEY.md (independent codex
gpt-5.6-sol high lanes, sibling-fenced) + trk_refute_20260716/REFUTATION.md (2nd-vote adversarial
verification). Pattern per runs/research_eventdata_20260713/CROSSCHECK_RULING.md. VERIFIED=0.

STATUS: FINAL — refutation lane landed 2026-07-16 (verdicts C1 CONFIRM, C2 CONFIRM, C3 REFUTE-in-
part, C4/C5/C6/C7/C8/C9 PARTIAL w/ corrections, C10 CONFIRM; full evidence
trk_refute_20260716/REFUTATION.md).

## Refutation results folded in (supersede any conflicting line below)

- **McByte 3-5 FPS on A100 CONFIRMED from the paper + gating constants confirmed in source**
  (mc=0.6/mb=0.9/mf=0.05 present in mcbyte_tracker.py). 12-20x source duration is derived
  arithmetic, not a separately benchmarked figure. McByte = worst-clip forensics only.
- **SAM2.1-B+ multi-target collapse CONFIRMED** (17.8/12.4 FPS @3/5 targets, A6000, SAM-MT
  Table 3) with the caveat that Meta's 64.1 FPS is a different protocol/hardware — not a
  controlled scaling curve. Always-on VOS is not viable at 60fps; selective windows are.
- **Selective Mask Propagation: numbers CONFIRMED (87.2 HOTA; gain mostly GTA +7.8 vs masks
  +1.5), but "no code" REFUTED** — a third-party MIT implementation exists
  (github.com/holma91/selective-mask-propagation, pushed 2026-07-14, training-free, pulls
  upstream SAM/tracker assets). Upgrade: step-3 mask-cue experiments may evaluate this repo
  directly instead of reimplementing; treat as UNOFFICIAL reimplementation requiring validation
  against paper numbers before trusting.
- **RF-DETR bundle CONFIRMED**: license split (det N-L Apache / det XL-2XL PML-1.0 / seg all
  Apache); `rf-detr-large-2026.pth` HEAD 200 Content-Length 135,954,129; `scale_jitter` commit
  full SHA 69b12dbf8d40a739ff22a8463f682fa4a066c2ba (authored 2026-07-15; False → direct-resize,
  border annotations never clipped); latest release tag 1.8.3 (2026-06-29). Crowded-person
  absence is a bounded negative ("no official result located 2026-07-16"), not a universal.
- **Exact spec pins corrected**: D-FINE-L arm uses `dfine_l_obj2coco_e25.pth` (57.3 AP; the
  non-e25 asset is live legacy). DEIMv2 has EIGHT sizes; the L arm pins HF
  `Intellindust/DEIMv2_DINOv3_L_COCO` (56.0 AP, 32.2M, 10.47ms, DINOv3-S backbone) — NOT the
  `_S_COCO` artifact one survey mapped to L; S/M use ViT-Tiny distilled backbones.
- **EdgeCrafter is real (canonical repo Intellindust-AI-Lab/EdgeCrafter, Apache-2.0, live
  ecdet_l/ecseg_l assets) but same-lab/authors as DEIMv2** — it is NOT an independent
  architecture-family vote. Demoted to optional same-family control; running both DEIMv2 and
  EdgeCrafter adds little independence.
- **Small-data adaptation evidence PARTIAL**: Gadde WACV22 "+16 mAP from ~100 samples/video" is
  transductive DOMAIN LABELS ON PREDICTED BOXES, not 100 drawn boxes — do not quote it as a
  labeling-volume law. Vandeghen SST BSD-3 and Maglo ~6 tracklets/player CONFIRMED. Our recipe
  stands but its data-volume expectations are inference, not published law.
- **Market-1501 correction**: the CURRENT official page states no license terms at all —
  posture is "terms unresolved from official page" (previously "academic-only per page").
  Practical consequence UNCHANGED: not commercial-clean; OSNet market1501 checkpoint stays
  R&D-only pending an authoritative agreement.
- **YOLO26m baseline pin CONFIRMED**: asset
  github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt, 44,255,705 bytes,
  52.5 COCO AP on the default e2e head (do not substitute the 53.1 non-e2e number).

## Convergent (adopt with confidence — independently fetched by both lanes)

1. **RF-DETR-L detection = first zero-shot challenger; RF-DETR-Seg-L = first mask-producing A/B.**
   Exact artifact `rf-detr-large-2026.pth` (storage.googleapis.com/rfdetr/, HEAD 200 both lanes).
   Detection N/S/M/L weights Apache-designated; detection XL/2XL = PML-1.0 via `rfdetr_plus`;
   ALL segmentation sizes N-2XL Apache-designated. Fine-tune path mature (COCO/YOLO formats,
   frozen-encoder/LoRA, negative images). [CORROBORATED x2]
2. **No official RF-DETR person-class / CrowdHuman / MOT20-det evidence exists.** Aggregate COCO
   AP is the only official number; occlusion superiority over YOLO26m is UNPROVEN and our frozen
   worst-clip benchmark is the deciding test. [CORROBORATED x2]
3. **"YOLO26" is a real public Ultralytics family (AGPL-3.0/Enterprise), NMS-free default.** Our
   baseline identity must pin package version, weight SHA-256, head mode, resolution, thresholds.
   The incumbent detector is itself NOT commercial-clean → RF-DETR would improve licensing AND
   possibly accuracy. [CORROBORATED x2]
4. **No commercially clean public person-ReID checkpoint exists (2026-07).** OSNet default: MIT
   code but Market-1501 checkpoint academic-only; KPR = Hippocratic 3.0 (reconfirmed x2);
   SOLIDER/CLIP-ReID/TransReID/SapiensID/Pose2ID all R&D-only via training data or repo license.
   Clean route = session/enrollment embedding trained on owned consented crops from a cleared
   backbone. [CORROBORATED x2]
5. **Public sports MOT data is R&D-only:** SportsMOT CC BY-NC 4.0 (owner README beats the
   third-party CC-BY table cited by one aggregator), DanceTrack videos noncommercial, SoccerNet
   research-only/NDA. Owned clips are the only commercial-clean fine-tune source; public sets are
   diagnostics/pretrain-for-R&D only. [CORROBORATED x2]
6. **SAM-MT (2026-07-09/10) is the one substantive post-register release** — multi-target mask
   propagation ~36.5 FPS @5 targets/A6000 vs SAM2.1-B+ collapse at multi-target — but checkpoint
   CC BY-NC-SA 4.0 + no repo license + no training code → R&D throughput control only.
   [CORROBORATED x2]
7. **Owned-domain fine-tune with explicit spectator/negative supervision is the highest-leverage
   move** for the coverage gap (cov4 0.71-0.76 vs 0.95): single positive class `on_court_player`,
   labeled off-court/spectator negatives, empty frames, edge-truncation care, court geometry as a
   scored post-rule not a training crop. Both lanes converged on ~1k-2k boxes first tranche +
   stratified error-driven growth, game/session-disjoint splits. [CORROBORATED x2 at recipe level;
   the underlying "small fine-tune wins" is INFERENCE backed by soccer/rugby small-data precedents
   [PENDING C8]]
8. **McByte stays the ruled step-3 bounded diagnostic** (MIT, runnable, external detections
   supported, inactive since 2025-07) — with a runtime caveat that reframes it as
   worst-clip-forensics only, never a default: published end-to-end 3-5 FPS on A100
   [PENDING C1]. CAMELTrack stays last (Apache code but global ckpt trained on R&D-only mix).
   [CORROBORATED x2 on posture]

## Single-source items sent to refutation (2nd vote)

- C1 McByte 3-5 FPS A100 end-to-end (B-only; reshapes step-3 framing).
- C2 SAM2.1-B+ multi-target throughput collapse 17.8/12.4 FPS @3/5 targets (B-only).
- C3 Selective Mask Propagation v3: 87.2 SportsMOT HOTA, gain mostly GTA association, no code
  (B-only; its "selective windows + detection-gap trigger" design idea informs our step-3 shape).
- C4 RF-DETR facts bundle incl. `scale_jitter=False` develop commit (A-only) + latest release tag.
- C5 D-FINE exact ckpt filename (A/B disagree: `dfine_l_obj2coco.pth` vs `_e25.pth`).
- C6 DEIMv2 variant table (A/B quote different sizes/numbers — pin exact artifacts).
- C7 EdgeCrafter ECDet/ECSeg existence + numbers (A-only candidate).
- C8 Small-data sports adaptation precedents (B-only citations).
- C9 License texts re-verify (both lanes agree; refutation re-fetches the texts).
- C10 YOLO26 family + `yolo26m.pt` asset pin.

## Disagreements resolved by manager ruling

- **Benchmark order (A: zero-shot controls before fine-tune; B: fine-tune immediately after
  RF-DETR-L zero-shot).** RULING: B's order. Both lanes agree domain supervision is the leverage;
  zero-shot controls (D-FINE-L, DEIMv2) are cheap and run in the same GPU session as arms 2-3,
  but the fine-tune decision must not wait on them. Order: (1) RF-DETR-L zero-shot,
  (2) RF-DETR-Seg-L zero-shot (boxes scored first, masks archived), (3) D-FINE-L + DEIMv2
  controls same session, (4) RF-DETR-L owned-data fine-tune (the real experiment),
  (5) ReID enrollment A/B, (6) McByte forensics on worst clips w/ archived masks.
- **SportsMOT license conflict (owner README CC BY-NC 4.0 vs third-party table CC-BY-4.0).**
  RULING: owner README controls. R&D-only. (B surfaced and resolved this correctly.)
- **Open-vocabulary detectors as spectator filter (A ranked low, B did not rank).** RULING:
  diagnostic-only, never the primary detector; "player vs spectator" is scene role, not
  appearance class — court geometry + labeled negatives is the direct attack. [Both lanes'
  reasoning converges here.]

## What this changes vs the 2026-07-09 register

Nothing in the ruled NS-03.TRK sequence is overturned. The register's RF-DETR → ReID → McByte
order survives with sharpened details: exact Apache artifact set, the PML-1.0 XL/2XL boundary,
the fine-tune-first-after-zero-shot ordering, McByte demoted to bounded forensics by runtime
evidence [pending C1], SAM2.1 multi-target cost realism [pending C2], and a selective-window +
detection-gap-trigger design (no code; reimplement bounded) as the modern mask-cue shape if
McByte shows value but can't ship. The genuinely new fact: our baseline detector is AGPL — the
detector swap is now ALSO a licensing move (NS-07.3).

## Where we build our own tech (TRK)

No off-the-shelf tracker solves 4-enrolled-players + open-set spectator rejection at a fixed
elevated court camera [CORROBORATED x2]. The build list: (a) owned-data detector fine-tune with
explicit negative supervision; (b) four-slot enrollment gallery with absolute + margin open-set
rejection and no online gallery writes from uncertain frames; (c) soft court-footpoint prior with
truncation-aware uncertainty + hysteresis (replacing the hard margin heuristic only via a later
controlled experiment); (d) provenance-rich person-authority artifact feeding BODY/paddle/events
(already register-mandated).
