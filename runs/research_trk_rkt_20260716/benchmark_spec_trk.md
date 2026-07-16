# Benchmark lane spec — NS-03.TRK detector/domain leverage (SPEC-ONLY, no GPU dispatched)

Status: FINAL (ready to dispatch as a GPU lane when the coordinator sequences it)
Author: Track F manager, 2026-07-16, per TRK_CROSSCHECK_RULING.md (dual survey + 2-vote
refutation). VERIFIED=0 binding; nothing here is a promotion.

## Objective

Score detector/domain candidates against the current YOLO26m baseline on the frozen worst-clip
set with the frozen TRK scorer, per the NS-03.TRK ruled sequence (detector/domain first; ReID
second; McByte mask cue third; association-only sweeps BANNED).

## Frozen protocol (identical for every arm — no exceptions)

- Scorer: `threed/racketsport/person_track_gt_scoring.py` (gate v2.1 FP decomposition,
  far-off-court = >1.0m apron), invoked via `scripts/racketsport/score_person_track_sources.py`,
  IoU threshold 0.5, expected players 4. Scorer files byte-verified against commit `3d5125d58`
  md5s before any run (READ-ONLY, precedent: `runs/lanes/trk_reid_apron_20260712/spec.md`).
- Eval set (all labeled clips; no selection):
  - `wolverine_mixed_0200_mid_steep_corner` (300 frames) — worst clip
  - `burlington_gold_0300_low_steep_corner` (600 frames)
  - GT: `runs/lanes/trk_flip_20260713/frozen_gt/<clip>/person_ground_truth.json` (immutable)
  - Videos: `eval_clips/ball/<clip>/source.mp4`
  - NOTE (honesty): these are historical-internal clips, NOT fresh. Results steer the next
    candidate only; the full NS-03.TRK gate requires fresh clips under NS-02. No promotion from
    this card.
- Baseline arm (must run first, same scorer, same day, same machine class):
  - Detector `models/checkpoints/yolo26m.pt` sha256 `401cea9ab23a…0745d0b7` (AGPL-3.0 — flag:
    the incumbent itself is NOT commercial-clean; NS-07.3 relicense pressure applies to the
    baseline too, which raises the value of Apache-2.0 challengers).
  - Association: WIRED_DEFAULT margin-1.0m + OSNet (`models/checkpoints/osnet_x1_0_market1501.pt`
    sha256 `2809d322…9154`), best_stack rev 12 `owner_directed_margin1p0_osnet`.
  - Reproduction bar: match `runs/lanes/trk_flip_20260713/preflip_score/
    person_track_gt_scoring_report.json` within 0.0001 (burlington IDF1 0.8830775881 / cov4
    0.7116666667; wolverine IDF1 0.8515962036 / cov4 0.76; 0 switches) or STOP and diagnose
    before scoring any candidate.
- Detector-swap discipline: candidate arms replace ONLY the detection stage. Association, ReID,
  margin, and every threshold stay frozen at rev 12 defaults. Where the harness supports it,
  export candidate detections once and reuse (`runs/lanes/trk_flip_20260713/*_production/<clip>/
  tracked_detections.json` shape), so later ReID/McByte steps run on fixed detections per the
  North Star ("each later step keeps detections fixed").
- Metrics recorded per arm per clip: IDF1, cov4, id_switches, true_spectator_or_background FP,
  far_off_court FP frames, near-miss/no-gt-frame FP (diagnostic), HOTA/DetA/AssA (diagnostic),
  plus batch-1 detector ms/frame and all-in TRK wall per clip.
- Runtime bar (kill rule from North Star NS-03.TRK): reject any candidate adding >20% detector
  wall or failing any per-clip gate metric relative to baseline without a full-gate-relevant gain.

## Candidate arms (FINAL — per TRK_CROSSCHECK_RULING.md; all artifacts HEAD-200-verified
## 2026-07-16 by two independent lanes + refutation)

Baseline pin sharpened by refutation: public `yolo26m.pt` =
github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt (44,255,705 bytes; 52.5 COCO
AP on the default NMS-free e2e head). Record our local yolo26m.pt sha256 401cea9a… against it,
plus ultralytics package version, head mode, imgsz, conf threshold — "YOLO26m" alone is not a
reproducible identity.

1. **RF-DETR-L detection zero-shot** (rank-1). `RFDETRLarge`, weights `rf-detr-large-2026.pth`
   from storage.googleapis.com/rfdetr/rf-detr-large-2026.pth (HEAD 200, 135,954,129 bytes),
   Apache-designated, native 704. Replace ONLY detections; association/ReID frozen. Report
   coverage/recall stratified near/far side, net-occlusion, truncation; spectator +
   far-off-court FP separately.
2. **RF-DETR-Seg-L zero-shot** (`rf-detr-seg-l-ft.pth`, Apache, 504 native). Score its BOXES as
   arm 2 with the same frozen tracker; ARCHIVE per-detection masks for the later mask-cue lane.
   Watch for far-player recall regression vs arm 1 (lower native res).
3. **Independent controls, same GPU session:** D-FINE-L Objects365→COCO pinned to
   `dfine_l_obj2coco_e25.pth` (github.com/Peterande/storage/releases/download/dfinev1.0/…,
   HEAD 200, 57.3 AP claim) and DEIMv2-L pinned to HF `Intellindust/DEIMv2_DINOv3_L_COCO`
   (56.0 AP / 32.2M / 10.47ms, DINOv3-S backbone — NOT the `_S_COCO` artifact). One frozen
   default-threshold run each; no threshold grid. EdgeCrafter is EXCLUDED (same-lab DEIMv2
   sibling — adds no independence; refutation C7).
4. **RF-DETR-L owned-domain fine-tune (the decision arm).** Runs after arms 1-3 locate the
   zero-shot floor; do not block on their verdicts to prepare data. Recipe per ruling: single
   positive class `on_court_player`; explicit labeled negatives (spectators/passers/far-court),
   empty-court frames; game/session-disjoint from the two eval clips (eval clips NEVER in
   training); first tranche ~1-2k boxes + 500-1k hard negatives from owned footage
   (harvest-clip frames are in-domain candidates); frozen-encoder/LoRA and full-FT branches from
   one seed; pin rf-detr at release 1.8.3 or, if the border-truncation branch is used, at
   develop commit 69b12dbf8d40a739ff22a8463f682fa4a066c2ba (`scale_jitter=False`) — record which.
   Budget ≤8 A100/H100-hours. Stop rule: two frozen-threshold attempts without material coverage
   gain over YOLO26m, or ANY new switch/spectator/far-off-court FP → stop the branch.
5. **Deferred out of this lane (ruled):** ReID enrollment A/B (next lane after detector freeze);
   McByte forensics (bounded, worst clips only, 3-5 FPS A100 confirmed — needs the archived
   masks + frozen detections from arms 1-4); selective-window mask cue via the third-party MIT
   repo holma91/selective-mask-propagation (unofficial reimplementation — validate against paper
   numbers first); CAMELTrack (association change, banned at this step); SAM-MT/SapiensID/KPR
   (license-blocked diagnostics); open-vocab spectator filters (diagnostic only).

Numbers, licenses, and livecheck evidence: `TRK_CROSSCHECK_RULING.md`, both `SURVEY.md`s, and
`trk_refute_20260716/REFUTATION.md`.

## Outputs required from the lane

- `runs/lanes/trk_detbench_<date>/report.json`: per-arm per-clip full metric table + runtime +
  checkpoint sha256s + scorer byte-verify proof + baseline-reproduction proof.
- Verdict per candidate: `adopt-next-step` / `reject` / `no-attempt`, against the kill rules
  above. No best_stack change from this lane alone (detector flip needs the full ruled sequence
  + fresh-clip gate; this card is historical-internal).

## Provision

H100 spot per `.claude/skills/gpu-fleet-provision/` from fleet snapshot (precedent:
`runs/lanes/trk_reid_apron_20260712/spec.md`); ledger row in `runs/manager/gpu_fleet.md`;
teardown + disk-confirm at close. Estimated wall: 2-4h (2 clips × ≤6 arms + fine-tune budget).
