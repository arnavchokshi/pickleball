# TRK refutation lane — 2nd vote on load-bearing single-source claims (adversarial verification)

You are the REFUTATION lane in a 2-vote primary-source verification pattern. Two independent
surveys ran; the claims below are load-bearing for a benchmark spec and currently have only ONE
vote. Your job: independently CONFIRM or REFUTE each claim from primary sources you fetch
yourself. Be adversarial — try to break each claim, do not rubber-stamp. Do NOT read
`runs/research_trk_rkt_20260716/trk_survey_A_20260716/` or `trk_survey_B_20260716/` (the claims
are restated fully below; independence of your fetches is the point).

For EACH claim output: verdict CONFIRM / REFUTE / PARTIAL / UNRESOLVED, the primary-source URL(s)
you fetched, a short quote or concrete observation as evidence, and fetch date. If REFUTE or
PARTIAL, state the corrected fact.

## Claims to verify

C1. McByte end-to-end runtime is ~3-5 FPS on a single A100 per its own paper (appendix), i.e.
    ~12-20x source duration for 60fps video. Source to check: arXiv 2506.01373 (McByte paper).
    Also confirm: repo github.com/tstanczyk95/McByte is MIT, last pushed ~2025-07, supports
    externally supplied detections, uses SAM ViT-B init + Cutie base-mega propagation, and its
    paper gating constants are mask confidence 0.6 / box coverage 0.9 / fill ratio 0.05.

C2. SAM 2.1 Base+ multi-target video throughput drops to ~17.8 FPS at three targets and
    ~12.4 FPS at five targets on an RTX A6000 (per the SAM-MT paper's comparison table,
    arXiv 2607.08688), even though Meta's own single-target table says 64.1 FPS on A100.

C3. "Selective Mask Propagation" paper (arXiv 2606.13033, v3 dated ~2026-07-09) reports
    87.2 HOTA on SportsMOT test with SAM3-Deep-EIoU + GTA; its ablations attribute most gain to
    global tracklet association (+7.8 HOTA) vs selective masks (~+1.5); it has NO released code
    or weights.

C4. RF-DETR facts bundle (repo github.com/roboflow/rf-detr): (a) detection N/S/M/L weights are
    Apache-designated and detection XL/2XL are PML-1.0 via `rfdetr_plus`; (b) ALL segmentation
    sizes N-2XL are Apache-designated; (c) current Large class resolves to
    `rf-detr-large-2026.pth` at storage.googleapis.com/rfdetr/rf-detr-large-2026.pth (HEAD it);
    (d) a July 2026 develop commit adds `scale_jitter=False` control that prevents random-crop
    clipping of border annotations (claimed commit 69b12dbf8d40 — verify it exists and what it
    does); (e) latest release tag as of 2026-07-16 (one lane says v1.8.3 June 29, another cites
    v1.7.1 fixes); (f) NO official COCO-person, CrowdHuman, or MOT20-det benchmark is published
    by Roboflow for RF-DETR anywhere official.

C5. D-FINE exact released L checkpoint for Objects365→COCO: is the artifact
    `dfine_l_obj2coco.pth` or `dfine_l_obj2coco_e25.pth` (or both)? Pin the exact URL + HTTP
    status + claimed AP (repo github.com/Peterande/D-FINE).

C6. DEIMv2 variant reality-check (github.com/Intellindust-AI-Lab/DEIMv2): enumerate the actual
    released sizes with their COCO AP / params / latency (one lane claims "DEIMv2-L 56.0 AP,
    32.2M, 10.47ms w/ DINOv3-S backbone + HF artifact DEIMv2_DINOv3_S_COCO"; another claims
    "S 50.9 AP 9.7M / M 53.0 AP 18.1M"). Which artifacts exist on HF/Drive, and which backbone
    maps to which size? Pin exact names for a benchmark spec.

C7. EdgeCrafter (github.com/capsule2077/edgecrafter): does it exist with live release assets,
    ECDet-L 57.0 box AP / ECSeg-L 47.1 mask AP at 640² T4-TRT claims, Apache-2.0 repo? Is it a
    real independent candidate or a repackaging?

C8. Small-data sports adaptation evidence bundle: (a) WACV 2022 transductive soccer player
    detection reports ~+16 mAP from ~100 annotated samples per video (Gadde et al.);
    (b) Vandeghen et al. CVPRW 2022 semi-supervised soccer detection code is BSD-3;
    (c) Maglo et al. CVPRW 2022 rugby tracking needed only ~6 short game-specific tracklets
    per player for identity. Verify each paper exists and states this.

C9. License bundle re-verify (fetch the actual license texts/pages): (a) SportsMOT official
    README = CC BY-NC 4.0, no redistribution; (b) DanceTrack videos noncommercial research-only
    while annotations CC BY 4.0, code MIT; (c) Market-1501 official page = academic use
    only/no redistribution; (d) KPR repo license = Hippocratic 3.0 (HL3 variant);
    (e) SAM-MT HF checkpoint card = CC BY-NC-SA 4.0 AND its GitHub repo has no LICENSE file;
    (f) Ultralytics YOLO26 = AGPL-3.0 or paid Enterprise; (g) SAM 2.1 code + checkpoints =
    Apache-2.0 per facebookresearch/sam2.

C10. Ultralytics YOLO26 family reality-check: n/s/m/l/x detection + seg variants exist publicly,
     `yolo26m.pt` is a real release asset (HEAD it), default head is NMS-free one-to-one, and
     the m-detection COCO AP is ~52.x. Pin the exact asset URL for a spec.

## Deliverables (write ONLY into /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/trk_refute_20260716/)

1. `REFUTATION.md` — one section per claim C1-C10: verdict, evidence quote, URL, date, corrected
   fact if any.
2. `livechecks.md` — every URL fetched: HTTP status/bytes, date.
3. Final message: ≤25 lines — verdict list C1-C10 + anything you found that materially changes
   the picture.

Rules: no GPU work, no multi-GB downloads (HEAD/partial fetch), no pipeline/config edits, no
other runs/ dirs. Published numbers are motivation, never pickleball accuracy. VERIFIED=0 stands.
