# Cited Synthesis — Ball Chain / Coaching SOTA Scan (2026-07-08)

Scope note: only claims that were adversarially refuted **and returned `survives:true`** are marked **[CORROBORATED]** below. Everything else — including findings that were never run through refute, and findings that were refuted with `survives:false` even where most of the underlying facts checked out — is labeled **UNVERIFIED**, per instruction. Two findings (TrackNetV5, TOTNet) fall in a middle case: the adversarial pass confirmed most of the claim but caught a specific factual error, so I report the **corrected** version and still tag it UNVERIFIED overall since `survives:false`.

---

## (1) WHAT CHANGES OUR PLAN

**1a. Labels-vs-accuracy curves for specialized domains plateau early (~100–150 shots/class) in published SSOD benchmarks — we have not checked whether ours already has.**
[CORROBORATED] — arxiv.org/html/2601.13380 (submitted 2026-01-19, v2 2026-01-29). Verified quote: "structured per-class supervision yields stronger performance than percentage-based sampling," 150-shot MixPL (41.6 mAP) beats a COCO 5%-of-dataset SSOD baseline (40.1 mAP); on their specialized "Beetle" domain, MixPL hits 71.1 mAP by 100 shots/class "after which performance plateaus. Additional labeled data did not yield further improvements."

**Exact recommendation:** Before continuing the CVAT disagreement-queue push from 1,121 → 10-20k frames on current budget, insert explicit internal checkpoint evals at **1k / 3k / 6k / 10k reviewed rows** against the frozen held-out bar (F1@20px). If the curve is already flattening by 3-6k (as this paper's evidence suggests is common for specialized single-domain tasks), redirect surplus labeling budget toward court/lighting/venue diversity sampling instead of raw volume, or toward the coaching-wave label needs. This is a direct amendment to the P-series labeling-target task in NORTH_STAR — currently written as a flat 10-20k target with no interim decision gate.

**1b. Our own internal→held-out generalization gap (4 inversions, F1@20px 0.7248 never beaten) should be explicitly tracked as a "seen-environment vs unseen-environment" split, not just a single scalar bar.**
[CORROBORATED] — arxiv.org/abs/2603.06691 (v1 2026-03-04, v2 2026-03-17). Verified: YOLOv8 fine-tuned on 20,510 frames/11 backgrounds for egocentric shuttlecock detection scores F1=0.86 in similar-to-training environments vs F1=0.70 in entirely unseen environments (a ~16pt gap attributed to object size + background-texture shift), at a labeled-frame scale matching our own 10-20k target almost exactly.

**Exact recommendation:** Add an explicit "unseen-environment" stratum to `heldout_eval_ledger.md` (distinct courts/lighting/camera setups never touched during training or disagreement-mining) alongside the existing held-out row schema, and report F1@20px separately for seen-vs-unseen. This external result recalibrates what to expect: a 15-20pt seen/unseen gap at our label scale is normal for small racket-sport-object detection, not evidence the pipeline is broken — so the ledger should stop treating a single held-out number as the full picture and start tracking the gap itself as a first-class metric.

---

## (2) WHAT CONFIRMS OUR PLAN

**2a. Commodity RGB reprojection genuinely cannot reach usable spin accuracy at our noise floor — the actual unlock requires a sensing upgrade, not a better solver.**
[CORROBORATED] — arxiv.org/abs/2606.26780v1 (submitted 2026-06-25). Verified: an event camera + pan/tilt galvanometer mirrors + focus-tunable telephoto lens hits 2.1% magnitude / 4.0° axis error on static balls, and 8.8% magnitude / 6.4° axis error at 3ms latency / 750Hz throughput in live 3-view professional table-tennis. This is a **hardware-dependent** result (event sensor + actuated optics), not achievable from standard smartphone RGB.
**Read on our decision:** This directly supports the wave-5 honest kill of scalar-Magnus-spin (spin-on inflated reprojection RMSE on 2/3 clips at lambda=0.05, ≥8-inlier gate) and the frozen "gate on per-segment view-geometry confidence" ruling. No action needed — this is confirmatory evidence we killed the right thing for the right reason, and that further solver tuning (vs. a sensing upgrade we don't have) would not have closed the gap.

**2b. Our CVAT disagreement-queue design (large-offset / student-only / teacher-only buckets) mirrors validated active-learning literature and is not ad hoc.**
UNVERIFIED (not run through adversarial refute this pass, but pre-2025 and structurally uncontested) — arxiv.org/pdf/2308.08476 ("Classification Committee for Active Deep Object Detection"). Argues localization disagreement must be modeled alongside classification disagreement for effective detection active-learning — matches our "large-offset" bucket design. Treat as directional confirmation only; not independently re-verified this round.

**2c. The seed-trained student beating the 61k-public warm-start on hard frames is consistent with published "simple-label bias" theory.**
UNVERIFIED (not adversarially refuted) — arxiv.org/html/2507.00608v1 (DeSimPL, 2025-07). Proposes that self-labeling/large public corpora systematically over-represent easy/simple examples ("simple-label bias"), which — if true — mechanistically explains why raw tennis-ckpt zero-shot and the 61k-frame warm-start underperform our small, adversarially hard-mined (486-row) seed student on hard frames. Consistent with, but does not independently prove, the wave-3 owner ruling that killed the 2D-teacher plan in favor of raw WASB seed. Flag as supporting theory, not proof.

---

## (3) WATCH ITEMS (interesting, unverified)

**Detection/tracking architecture:**
- TrackNetV5 claims F1 0.9859 / accuracy 0.9733, a new SOTA beating TrackNetV4 by 2.78% F1 — UNVERIFIED overall (refute caught that the paper's own text never says "badminton"; it frames its target application as **tennis**, despite evaluating on the dataset conventionally called "TrackNetV2" which originates from badminton footage). Corrected framing: arxiv.org/html/2512.02789v1 (2025-12-02) — same-domain (non-pickleball) in-distribution benchmark either way; does not falsify our zero-shot cross-domain 0.7248 bar. Not urgent.
- TOTNet (occlusion-aware, 3D convs + visibility-weighted loss + occlusion augmentation; RMSE 37.30→7.19, occluded-frame accuracy 0.63→0.80) — UNVERIFIED overall; refute caught that the "no runtime numbers published" sub-claim is **false** — the paper's Table 2 reports ~28.08 FPS (TOTNet) / ~12.19 FPS (TOTNet-OF), 8.65-8.66M params. Corrected: arxiv.org/abs/2508.09650 (2025-08; journal version Computer Vision and Image Understanding, DOI 10.1016/j.cviu.2026.104657, Feb 2026). Occlusion-augmentation training recipe (independent of full architecture adoption) remains a plausible candidate to fold into our own seed-student training given occlusion is implicated in some of our inversions — but re-verify runtime/architecture claims directly before citing further.
- TrackNetV4 motion-attention-map fusion (arxiv.org/abs/2409.14543, 2025 ICASSP) — drop-in motion-difference channel on TrackNet backbones; not adversarially checked, pre-known-adjacent lineage.
- RF-DETR (Roboflow, ICLR 2026; github.com/roboflow/rf-detr) — DINOv2-backbone real-time DETR, no published F1@20px on a tennis/pickleball ball dataset; not a proven beat of our bar, just a bench candidate.
- Roboflow's May 2026 pickleball tutorial (RF-DETR + Claude Sonnet 4.5 commentary, 80.7% F1 general pickleball detection, no ball-tracking/trajectory work) — blog.roboflow.com/automate-pickleball-analytics, 2026-05-11. Below our ball-specific bar; relevant only as coaching-stage competitive color.

**Spin/trajectory:**
- TT3D (arxiv.org/html/2504.10035v2, 2025-04/06) — infers spin from **bounce-induced trajectory kinks**, not free-flight curvature; requires ≥3 visible frames post-bounce. Potentially a stronger identifiability cue than our killed free-flight-only Magnus fit, given pickleball's frequent bounces/paddle contacts. Worth a scoped look before any spin revisit — not yet verified.
- Synthetic-to-real spin classifier, zero real spin labels, 92.0% spin-class accuracy (arxiv.org/pdf/2504.19863, CVPRW 2025) — alternative to our least-squares physics solver; unverified.
- "Uplifting Table Tennis" (arxiv.org/abs/2511.20250, 2025-11-25) — synthetic-only 2D→3D+spin uplift net engineered for detector-noise robustness; PDF text not fully extractable, flagged for follow-up read.
- "Where Is The Ball" (arxiv.org/html/2506.05763v1, 2025-06) — explicitly skips spin/Magnus entirely, still gets 87.21% landing accuracy on tennis via gravity + ground-plane constraints only; weak indirect support for keeping spin non-load-bearing.
- TT4D (arxiv.org/html/2605.01234, 2026-05, ACM MM 2026 submission) — confirms the field still has no formal fps/segment-length/view-geometry identifiability threshold for monocular spin; supports continuing our own empirically-gated approach rather than searching for an external formula.

**Coaching / LLM grounding:**
- BioCoach (arxiv.org/abs/2603.26938, 2026-03-31) — 3-stage DOF-selector → structured biomechanical context → vision-conditioned feedback, architecturally near-identical to our planned 3-stage grounded coach. Worth a direct read of their DOF-scoping design.
- QEVD-bio-fit-coach benchmark + "LLM-Bio-Acc" judge metric (letsdatascience.com, 2026-06) — candidate eval-methodology pattern for our 0-fabrication bar; unverified, not primary-sourced directly.
- CHI 2026 survey (dl.acm.org/doi/10.1145/3772318.3791652) — documents ungrounded LLM sports feedback underperforming human coaches; directionally supports comparator-before-LLM design, not independently re-verified.
- RubricRAG (arxiv.org/pdf/2603.20882, 2026-03) — retrieval-grounded rubrics for LLM feedback constraint; candidate technique for the rule-comparator stage, unverified.
- PMC scoping review (ncbi.nlm.nih.gov/pmc/articles/PMC12520646, 2025-2026) — no existing benchmark evaluates real-time corrective coaching adequacy; confirms we likely need our own eval harness.

**Competitive landscape (last full survey 2026-07-05; these are newer/unsurveyed names per that ruling):**
- PB Vision: $20/$50/mo tiers, "3D trajectory analysis" = shot-path-level, not full-scene 3D; DUPR rating-integration partnership (softwarefinder.com/artificial-intelligence/pb-vision). Unverified this pass.
- SwingVision: AI coaching bundled in Pro tier, already ships post-session coaching advice, appears stat-summary-based not fabrication-audited (swing.vision). Unverified this pass.
- SwingVantage "Motion Lab": free phone-only 3D body-motion reconstruction with phase scores, pickleball-specific (swingvantage.com/pickleball). Load-bearing competitive signal if accurate — closest existing product to our "3D" claim; our differentiation would need to rest on full-scene (ball+court+multi-player) fidelity and trust bands, not "3D" alone. Unverified this pass — recommend a direct hands-on check before using this to reposition messaging.
- Wingfield: hardware SKU, ~€6,999 one-time or ~€187/mo financing (wingfield.io/en/products). Different business model, informs a possible venue-hardware pricing anchor only.
- PlayReplay (via Pickleball Inc. distribution): 4-camera line-calling, ITF Silver Classification, 100M+ calls/250k+ matches logged, also ball speed/spin/shot-type (playreplay.io news release). Benchmark for "mature" ball-physics data volume vs our 1,121 frames.
- Owl AI + MLP: broadcast-camera-only AI officiating, live 2026 Dallas season (sportsvideo.org, 2026-05-22). Officiating segment, low direct relevance; useful "software-only, existing-camera" messaging parallel.

---

## (4) DEAD ENDS SEARCHED

- paperswithcode.com's sports-ball-detection-and-tracking-on-tennis leaderboard page redirected to an unrelated page — no independent leaderboard corroboration obtained beyond what the TrackNetV4/V5/TOTNet papers self-report.
- No paper found benchmarking directly against "WASB-SBDT" by name, nor any claiming cross-sport zero-shot transfer comparable to our own WASB-on-pickleball setup — the domain-transfer angle we care about most remains unaddressed in the literature.
- "Active learning disagreement sampling object detection domain adaptation 2026" search surfaced only pre-2025 general-AL background and adjacent-but-mismatched hits (Bi3D, DELTA, source-free ADA); no 2026 racket/ball-sport-specific disagreement-AL result.
- TrackNet-lineage search returned no new 2025-2026 domain-adaptation paper beyond already-known TrackNetV4.
- WebFetch of arxiv PDF 2409.09412 ("Label Convergence... Contradictory Annotations," potentially relevant to our F1@20px ceiling/inversions) returned corrupted binary content — needs retry via the abstract/HTML page, not the raw PDF.
- Leave-one-domain-out generalization search returned only generic PACS/OfficeHome/DomainNet benchmarks — no sports/ball-tracking LOSO result newer or more specific than our own standing LoSO practice.
- Direct WebFetch of arxiv PDF binaries for 2511.20250 and 2504.19863 (pdf variant) failed to parse — quantitative spin-on/spin-off ablation numbers for "Uplifting Table Tennis" remain unconfirmed pending a direct read via the HTML mirror.
- No paper found with a direct spin-on-vs-spin-off reprojection-error ablation matching our exact experiment shape (TT3D comes closest but doesn't quantify this either).
- Event-based gaze-control spin paper was noted but not deep-dived on hardware-adoption feasibility, given it requires specialized (non-smartphone) optics — out of scope for our pipeline as-is.
- General RAG hallucination-mitigation literature is generic factuality work, not sports/coaching-specific — confirms direction, no sport-specific technique to borrow.
- AgentCoach, SoleCoach, BoxingPro, GPTCoach (CHI 2026 / IoT-LLM coaching systems) are motivational/adherence or wearable-sensor systems, not vision/3D-pose-grounded technique coaching — adjacent, not closer matches than BioCoach.
- Rubric-generation-for-LLM-judge papers (Auto-Rubric, OptimSyn, FeedEval, iRULER) are general NLG/essay-feedback grounding work, not sports-specific — RubricRAG banked, rest are dead ends.
- No published processing-latency figures found for pb.vision, SwingVision, Wingfield, PlayReplay, or Owl AI — all qualitative "fast turnaround" claims only.
- No public px-level or F1-level accuracy claim found for any competitor's ball-tracking beyond PlayReplay's ITF Silver Classification and "100M+ calls" scale claim — nothing directly comparable to our F1@20px bar.
- No competitor found publicly claiming a full multi-body + ball + court joint 3D scene reconstruction with trust/confidence bands (our core claim) — SwingVantage Motion Lab is closest but appears single-player, body-only.
- No new entrants found beyond SwingVantage/Wingfield/PlayReplay/Owl AI since the 2026-07-05 survey; a Twitter/X or Product Hunt pass might surface earlier-stage/stealth entrants not yet web-indexed.