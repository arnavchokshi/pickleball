# COURT WAVE 2026-07-09 — manager design ruling (Fable)

Inputs re-derived, not forwarded: runs/research_court_20260709/SYNTHESIS.md (28-agent SOTA fanout,
adversarially refuted), runs/lanes/court_solthink_20260709/DESIGN_PROPOSAL.md (gpt-5.6-sol xhigh),
4-agent repo census, COURT-EXT-1 probe results, COURT-DATA-1 mid-flight finding.

## Rulings (effective now)
R1. EVAL PROTOCOL = Protocol S (sealed sources). 73VurrTKCZ8 / HyUqT7zFiwk / zwCtH_i1_S4 pixels,
    labels, calibrations, projected frames, and styles are EVALUATION-ONLY for any promotion
    candidate. Consequence: with Roboflow-only training, the 5 corrected owner FULL15 rows are a
    legitimately source-disjoint dev gate. The 4 protected eval clips remain the promotion surface
    (Burlington/Wolverine internal-dev allowed; Outdoor/Indoor only via prereg ledger row).
R2. Tonight's decisive probe = REAL-ONLY FLOOR TRANSFER (Sol rung 4): fine-tune court_unet_v2 on
    audited Roboflow 12-floor-point rows (net channels MASKED, never imputed), synthetic minority
    mix per Sol §2.4 R0 proportions. Two arms: (A) init from court_model_v2.pt (synthetic-pretrained),
    (B) init ImageNet/torchvision resnet34 (commercial-clean lineage). Kill criterion per arm:
    source-disjoint median <25px AND PCK@5 >= +0.30 absolute over frozen baseline (0.0), else the
    recipe (not the whole program) dies.
R3. Tennis warm-start rung SKIPPED (TennisCourtDetector unlicensed + zero-shot = noise per EXT-1);
    PnLCalib/TVCalib oracle probes DEFERRED (GPL/research diagnostics, not tonight's critical path).
R4. A1 (multi-instance + semantic segments + clip-level top-K metric optimizer) is ADOPTED as the
    target architecture for the follow-on wave, gated on R2 firing. A2 (segment-first) is its
    first ablation. Loss additions adopted with it: geometric-consistency loss, masked multi-head
    training, reprojection-RMSE hypothesis voter.
R5. Masked-loss loader extension (per-keypoint visibility mask) is REQUIRED before training on
    12-point Roboflow rows; owned by the training lane; label_status must stay honest
    (roboflow rows are 'reviewed' human labels but their provenance travels per-row; derived/
    projected points are NEVER 'reviewed').
R6. Harvest pseudo-label corpus (DATA-1, single source 73V) = dev/eval ammunition only under R1.
    Its value tonight: temporal-aggregation eval + occlusion robustness diagnostics.
R7. Net-height: template stays 36in-uniform for now, and ANY net-height output remains
    template-prior, NOT a measured claim, until the owner rules on the 34in-center contract
    (owner ask #3). The 2cm net gate is not honestly passable before that ruling.
R8. GPU budget tonight: <=6 H100-hours, single spot VM from pickleball-fleet-snap-20260709-w7close,
    control-row-first, ~100-step probe before step caps, no co-scheduled arms, in-VM idle watchdog.

## Deferred to owner (staged asks, safe defaults active)
O1. +20 diverse calibrated viewpoints acquisition (Sol §2.6 strata) — reviewer time + rights ruling.
    Default meanwhile: R&D-only training on CC-BY Roboflow + owned data.
O2. Auto-coverage product gate (recommend >=80% auto-accept at <=1% false-confident-accept).
O3. Net-height truth contract (34in center sag vs 36in template; surveyed GT).
O4. Commercial rights: CC BY 4.0 Roboflow accepted for shipped stack w/ attribution? YouTube
    harvest promotion-training rights? TennisCourtDetector permission ask?
O5. New fresh lockbox capture for the eventual promotion claim (existing gates are inspected-many-times).
