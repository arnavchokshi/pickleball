# Track F manager notes — research_trk_rkt_20260716 (reconcile-from-disk on every resume)

Mission: TOPIC 1 = NS-03.TRK person det/seg/ReID/mask-cue research; TOPIC 2 = NS-03.RKT paddle
6DoF + synthetic-data research. Each topic ends in an ADOPTION_REPORT + ready-to-run benchmark
lane spec (spec only, NO GPU dispatch). Pattern: dual independent surveys → cross-check →
2-vote primary-source refutation → manager ruling (precedent:
runs/research_eventdata_20260713/CROSSCHECK_RULING.md).

## Phase state (update on every transition)

- [x] Phase 0: context read (North Star §2.2/2.3/NS-03.TRK/NS-03.RKT/§6, CV_SOTA_RESEARCH_20260709,
      eventdata precedent), dirs created, baseline identity pinned from best_stack.json rev 12.
- [x] Phase 1 DISPATCHED 2026-07-16 ~02:0x PDT: 4 independent codex gpt-5.6-sol high survey lanes,
      nohup-detached, workspace-write + network, web_search=live (user config):
      | lane | pid | codex session id | expected outputs |
      |---|---|---|---|
      | trk_survey_A_20260716 | 56427 | 019f69f1-cb88-7a20-9a8b-464487603d0f | SURVEY.md, livechecks.md, last_message.txt |
      | trk_survey_B_20260716 | 56429 | 019f69f1-cb88-7550-8712-4b7c8a2bcdef | same |
      | rkt_survey_A_20260716 | 56431 | 019f69f1-cb88-7fd0-9110-b764f70a4dca | same |
      | rkt_survey_B_20260716 | 56433 | 019f69f1-cb88-70d3-bdae-f9fe236e93d8 | same |
      Resume on death: `codex exec --cd /Users/arnavchokshi/Desktop/pickleball -s workspace-write \
      -c model=gpt-5.6-sol -c model_reasoning_effort=high -c sandbox_workspace_write.network_access=true \
      -o <lanedir>/last_message.txt resume <session-id>` with a brief "reconcile from your lane dir,
      finish SURVEY.md" message, nohup-detached (flags BEFORE resume).
- [x] Phase 1 COMPLETE ~01:24-02:0x: all 4 surveys landed w/ SURVEY.md + livechecks.md.
- [~] Phase 2 IN FLIGHT: manager cross-checks done in-session (convergences + single-source
      claims enumerated in the refutation PROMPTs); refutation lanes dispatched:
      | lane | pid | codex session id |
      |---|---|---|
      | trk_refute_20260716 | 69818 | 019f6a0e-19a5-77c2-aa6a-c052724949bd |
      | rkt_refute_20260716 | (see codex.pid) | (see log.txt session id line) |
      Same resume template as Phase 1.
      Key cross-check notes for the rulings:
      - TRK convergent: RF-DETR-L first (Apache weights; XL/2XL det = PML-1.0; ALL seg Apache);
        no official crowded-person numbers anywhere; YOLO26 AGPL; OSNet ckpt academic-only; no
        commercial-clean public ReID; SportsMOT/DanceTrack/SoccerNet R&D-only; SAM-MT new but
        license-blocked; owned-data fine-tune w/ spectator negatives = top leverage.
      - TRK order divergence: A = controls before fine-tune; B = fine-tune at #2. Manager leans
        B (both agree domain supervision is the leverage; controls cheap, same GPU session).
      - RKT convergent: no off-the-shelf for <80px blur planar regime; synthetic-only
        unsupported at 5°, synth+small-real plausible-unproven; RacketVision = 2D kpts only,
        side-kpts weak, YouTube provenance unresolved; both-IPPE+temporal graph rank-1;
        render-and-compare = offline oracle; category-level eliminated for now; TT4D ~26°;
        FoundationPose eliminated (RGB-D + NC); BALL 3D dominates contact budget.
      - RKT divergence to rule: GT rig minimum (A: mocap/≥4 synced cams, 2 unsynced phones not
        gate-credible; B: 3-phone rig + mandatory held-out jig metrology gate ≤1-1.5°/≤1cm/≤1ms).
        Also B's correction: 0.5-frame sync @60fps = 8.33ms = 8-17cm at swing speed — NS-02.1
        sync bar insufficient for RKT contact GT; carry into ruling + owner ask.
- [~] Phase 3: **TRK COMPLETE** — trk_refute landed (C1/C2/C10 CONFIRM, C3 REFUTE-in-part:
      MIT third-party selective-mask-propagation impl exists; C4-C9 PARTIAL w/ pin corrections:
      D-FINE `_e25`, DEIMv2-L = `DEIMv2_DINOv3_L_COCO`, EdgeCrafter non-independent,
      Market-1501 terms-unresolved, yolo26m.pt asset pinned). TRK_CROSSCHECK_RULING.md FINAL,
      benchmark_spec_trk.md FINAL, TRK_ADOPTION_REPORT.md written.
      **RKT COMPLETE** (morning reconciliation after Mac sleep ~01:30: lane had FINISHED its
      deliverables before dying — REFUTATION.md 01:59, last_message 02:03 — harvested, no
      resume needed). Verdicts: C1/C3/C4/C5/C6/C10 CONFIRM; C2 REFUTE-speed (RACE-6D = 84.0
      FPS, 16.6 was CRT-6D); C7 REFUTE-in-part (Kubric active; Isaac split Apache/proprietary);
      C8 PARTIAL (SAM-6D nested MIT, no root); C9 PARTIAL (ShapeFromBlur MIT generic
      rigid-object blur package EXISTS → added to Gap C as prior art). All folded:
      RKT_CROSSCHECK_RULING.md FINAL, RKT_ADOPTION_REPORT.md FINAL, benchmark_spec_rkt.md FINAL.
- [x] Phase 4 CLOSE: both topics delivered (2 rulings, 2 adoption reports, 2 specs, 6 lane
      dirs w/ surveys+livechecks+refutations). Committed fence-only; ledger row closed.

## Liveness checks (for resume after Mac sleep)

- `pgrep -fl "codex exec"`; per lane: pid in codex.pid, tail log.txt. Lane DONE when last_message.txt
  exists and process gone. If process dead + no last_message.txt → resume per above.
- Explore agent (repo anchors for benchmark specs: frozen TRK scorer path, worst-clip manifests,
  RKT preview measurement paths) was running at Phase-1 dispatch; findings get folded into the
  benchmark specs. If lost, re-derive: best_stack.json tracking.* entries + runs/lanes/trk_flip_20260713/
  + runs/lanes/racket_6dof_20260705/.

## Pinned baseline identity (for the specs)

- TRK: YOLO26m + BoT-SORT + OSNet x1_0 market1501 (models/checkpoints/osnet_x1_0_market1501.pt) +
  margin-1.0m raw-pool global association = best_stack rev 12 `tracking.association_court_margin` /
  `tracking.global_association_profile` (owner_directed_margin1p0_osnet, preview). Worst-clip flip:
  IDF1 0.6425→0.8516, cov4 0.0433→0.7117, 0 switches. Gate: every fresh clip IDF1 ≥0.85, 0 switches,
  0 spectator FP, 0 far-off-court FP, coverage ≥0.95. Production reproduction evidence:
  runs/lanes/trk_flip_20260713/preflip_score/person_track_gt_scoring_report.json.
- RKT: paddle.fused_estimator WIRED_DEFAULT render-only estimated_preview, proven_against
  wolverine_mean_iou 0.2356 / burlington_mean_iou 0.3424; gates face-angle p90 ≤5°, contact p90 ≤3cm,
  interim 30°; both-IPPE retention landed in evidence17_20260716 (committed 8a282d4db).

## Rules being enforced

R&D-only vs commercial-clean flag per candidate (NS-07.3); primary sources live-checked by lanes;
no promotion claims; fence = this research dir only; ledger rows in runs/manager/inflight_lanes.md;
no GPU dispatch (specs only); association-only TRK sweeps banned; rectangle-IoU never 6DoF.
