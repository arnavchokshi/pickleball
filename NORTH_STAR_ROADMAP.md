# NORTH STAR ROADMAP — the most accurate + fastest single-camera pickleball 3D engine

Owner-requested 2026-07-05. This is the master TO-DO document: what we still have to build,
in what order, and why — grounded in (a) our own measured state (every number below has a run
path) and (b) a 4-domain, ~114-agent adversarially-verified SOTA research sweep
(`runs/research_sota_20260705/` — ball, body, paddle, product/competitive reports with sources).

**Status discipline:** this is a PLANNING artifact. `CAPABILITIES.md` is canonical on any truth
conflict. `VERIFIED=0` today — nothing below claims a passed promotion gate. Checkboxes here mean
"work item exists", never "capability verified". Standing rules in Part IV bind every task.

**Companion doc:** `EDGE_PLAYBOOK.md` (owner-requested second pass, 2026-07-05) — the
profile-first advantages (N starts at 1: owner is profile #1, friends onboard through the same H0
setup phases; nothing hardcoded to one person), pickleball-rulebook logic hacks (H0-H26), the
iPhone-capture exploitation tier (H27-H34 — all video comes from iPhones: metadata harvest, and a
minimal capture-logger app that records per-frame intrinsics/exposure/ARKit-pose/gravity, solving
handheld + blur-speed + calibration at the source), exact per-stage technology bill of materials,
and exact data sources. Its §5 lists the task deltas it
applies to this roadmap. Owner ruling recorded there: private use for now → licenses are NOT a
constraint (revisit only IF the product expands beyond friends; held-out eval discipline unchanged
— that's about truth, not law).

---

**Map of this document:** PART 0 owner-setup (the ignition key) · PART I owner summary (I.0
built-vs-left · I.1 verdict · I.2 state-vs-bar · I.3 strategic calls · I.4 phases · I.5 owner actions · I.6 first
steps · I.7 definition-of-done / critical path / demo milestones) · PART II + II-B + II-C research
verdicts (chronological addenda — where they disagree, the LATER pass corrects the earlier) ·
PART III phase checklists (P0 foundation+data → P1 ball → P2 body → P3 paddle → P4 court/net →
**PF global fusion** → P5 speed/QA → P6 coaching → P7 productization) · PART IV standing rules ·
PART V evidence map.

# PART 0 — BEFORE THIS RUN STARTS, OWNER MUST (read first; agents STOP here if any is blank)

A run cannot begin until these are set. A manager agent that finds any of these missing does NOT
guess or work around it — it STOPS and surfaces the blocker (Part IV rule 9). **But verify before
stopping:** check each item against live evidence (BUILD_CHECKLIST last bullets + runs/) — several
may already be satisfied; tick an item only WITH a dated evidence pointer. Re-verify at every run
start: stale-blank boxes cause false STOPs, silently-ignored boxes destroy the rule:

- [ ] **GPU access:** SPEND CAP APPROVED (owner 2026-07-06): ≤$5/GPU/hr, max 4 GPUs, always
  teardown/delete on lane completion. AUTH: one final `gcloud auth login` by the owner, after which
  the manager creates a SERVICE ACCOUNT + key at `~/.secrets/` (outside repo, chmod 600, never in
  git/chat) so all future sessions auth non-interactively — this box ticks when the service-account
  key is verified working. Old VM `pickleball-a100-spot-ase1a`: owner ruled DELETE + start fresh.
- [x] **Commit the docs of record:** SATISFIED by the owner-authorized 2026-07-06 docs-of-record
  commit (this file's presence in git history is the evidence; includes EDGE_PLAYBOOK, CLAUDE.md,
  the manual, skills, fleet scaffolds, allowlist test).
- [x] **First data:** OWNER RULED 2026-07-06 — **BROAD online-video harvest APPROVED** (any public
  pickleball footage; private internal use only, never redistributed; copyright explicitly waived as
  a concern by the owner). First batch dispatched same day (P0-1b; `data/online_harvest_20260706/`).
  Owner capture batches (P0-3/§I.5) remain the in-domain finisher. Unchanged: NO persistent
  ReID/biometric profiles of non-owner people (that's the separate consent item below).
- [ ] **Roboflow API key** re-issued (for P1-0 public corpus + court keypoints), if using public data.
- [ ] **Biometric consent decision (blocking before PERSISTING any non-owner person's biometric
  profile — ReID gallery / shape betas, H4):** how consent is obtained + recorded for friends
  (P7-4b). SCOPE (default ruling until owner refines): gates biometric PERSISTENCE of identifiable
  people, NOT the mere processing of harvested third-party video (that's the P0-1b question above).
  Until answered: session-only, non-persistent tracking of every non-owner person.
- [ ] **Any task-specific unblock** the first wave needs (labeling, validation, a decision) — the
  manager lists these as typed STOPs at run start, per Part IV rule 9.

Everything below is the plan; this block is the ignition key. If a field is blank, that is a
`needs-decision`/`needs-purchase-approval` STOP, not a proceed-anyway.

# PART I — OWNER SUMMARY (read this, skip the rest until you need it)

## I.0 What is ALREADY BUILT vs. what is LEFT (read this first)

A lot exists. The honest distinction this whole doc rests on: **"built" means the code/app/stage
runs and produces artifacts; "VERIFIED" means it passed its promotion gate on real labels.**
Today `VERIFIED=0` — so everything below is "built, not yet gate-passed." That is NOT the same as
"nothing works." It means the machine is assembled and the remaining work is accuracy + proof, not
green-field construction.

### ✅ BUILT & RUNNING (server pipeline) — needs accuracy + gate proof, not construction
- **End-to-end orchestrator**: one command (`process_video.py`) → 17 stages → 3D world bundle with
  trust bands, artifact reuse, fail-closed gaps, `PIPELINE_SUMMARY.json`. Runs on real clips.
- **Ball 3D chain (pipeline default)**: detector zoo (WASB+blurball+TrackNet; TECH-AUDIT: value is
  the CANDIDATE POOL — union recall 0.879; the consensus-VOTING fusion is a measured liability,
  hidden-FP 0.349 vs single-WASB 0.063) → candidate
  sidecars → label-free auto-bounce anchors → frozen arc solver → flight-sanity gate → **top-down
  court-map view (pb.vision-style shot lines + bounce dots)** → viewer trail + honesty KPI,
  fail-closed parsing. *(Accuracy below bar — that's P1.)*
- **Body/world**: SAM-3D-Body (MHR70 skeleton+mesh) on A100, per-frame; classical post = person
  track → court placement → stance detection + foot-pin → stance-aware smoothing → world grounding;
  viewer-consumable 30MB mesh index; contact-dense (`ball_aware`) mesh scheduling. Foot-slide p95
  8-23mm on tripod clips. *(Raw jitter + handheld + GT gate = P2.)*
- **Paddle**: fused 6-DOF estimator (`paddle_pose_fused`, render-only). *(IoU 0.24-0.34, not wired
  by default = P3.)*
- **Court**: manual + metric-15pt calibration paths (work today); auto-find Wave A on a branch
  (guess+confirm UI, geometric solver, `court_unet_v2` trainer staged, synthetic generator).
- **Speed**: 2141s → ~532-565s (3.8×) with zero quality change; path to 6-8 min/clip booked.
- **Data engine**: `ingest_owner_capture.py` + `prelabel_owner_capture.py` + attack-tested
  eval-guards; **and it already consumes the app's capture sidecar** (provenance, intrinsics
  fingerprint, capture-id, court corners, manual taps).
- **Viewer** (`web/replay`): trust bands, honesty KPIs, mesh layer, 2×-FPS interpolation button,
  ball trail, court-map. **Confidence framework**, 7 rally metrics, coaching-facts JSON (scaffold).
- **Eval infra**: held-out pre-registration ledger, gate scripts (TRK/BALL-M1/BODY), ~2,900-test suite.

### ✅ BUILT (our iOS app — it EXISTS) — needs on-device proof + deeper server wiring
- **110 Swift files across 7 modules** (`ios/`): Capture, Core, Calibration, FastTier, Guidance,
  Upload, Replay — app shell, record/stop, camera-roll import, module boundaries, unit tests.
- **Capture sidecar contract already carries the good sensors**: per-clip camera **intrinsics**,
  **ARKit camera pose**, **gravity vector**, **court plane**, **locked exposure/ISO/focus/WB**,
  **LiDAR depth refs**, and **capture modes incl. `ballPhysics240` (240fps) + `swing120`** — i.e.
  most of the "iPhone hacks" in EDGE_PLAYBOOK are already *designed into the app's data model*.
- **On-device fast tier**: CoreMotion gravity sampler, live court-overlay engine, live guidance
  evaluator, on-device person detect/track, CoreML person detector + ~288p ball-heatmap spike,
  court dot-map, live ball/foot overlays. Calibration seed (ARKit + manual taps). Upload manifest +
  render-gateway client. Replay playback boundaries.

### ⬜ NOT DONE (this is the roadmap — P0→P7)
- **`VERIFIED=0`**: no stage has passed its promotion gate on real labels. This is the headline gap.
- **In-domain training data** (the measured unlock for every accuracy wall) — **P0**.
- **Physical-device capture proof**: the app is scaffold/simulator-tested, not validated recording
  a real game on a real phone — **P0-10 / P7-2**.
- **Server doesn't yet USE the app's richest signals**: per-frame ARKit pose + gravity are in the
  sidecar contract but not yet consumed for world-grounding / handheld robustness — **P2-1 / P0-10**.
- **Profile registry** (per-user courts/gear/players, the multi-user backbone) — **P0-9**.
- Accuracy to bar on ball / body / paddle / court, the coaching product, and productization —
  **P1-P7**. Per-phase "Already built vs. To build" lines head each phase in Part III.

**One-line status:** the car is assembled and drives; it is not yet a race car. P0-P7 is the tuning.

## I.1 The one-paragraph verdict

The end-to-end machine works: one command turns a video into a 3D world with players, ball chain,
paddle, court map, and an honest trust-banded viewer — and this week it got 3.8x faster
(2141s → ~532-565s on Wolverine). But accuracy is below product bar in all three things you care
about, and **every measured accuracy wall points at the same root cause: we have zero in-domain
training data.** Public/zero-shot approaches are provably exhausted (4 pre-registered held-out
failures on ball alone — including one we built from Roboflow: an 8,631-frame pickleball corpus on
which BOTH ball architectures fine-tuned WORSE on held-out, one catastrophically). The precise
lesson isn't "no pickleball data exists" — Roboflow has plenty; it's that public pickleball data is
broadcast/detection-style, not our-camera tracking-style, so it doesn't transfer (the U. Rochester
attempt hit the same wall). What's genuinely scarce is IN-DOMAIN, temporal, our-camera data.
Meanwhile the competitive news is good: **nobody — including pb.vision — ships single-camera 3D
player meshes or 6-DOF paddles. pb.vision's "3D" is ball-trajectory-only, and it demands a
perfectly stationary tripod.** Our full 3D world + motion tolerance + honest trust bands is a real,
defensible differentiator. And we already own the capture side: our iPhone app exists (110 Swift
files) and its capture sidecar already records intrinsics, ARKit camera pose, gravity, and 240fps
modes — signals the server largely doesn't consume yet, which is upside sitting on the table. The
plan: turn your captures into a data engine (Phase 0), then run four parallel accuracy campaigns
(ball, body, paddle, court) against pre-registered gates, then speed (already mostly booked), then
the coaching product.

## I.2 Where we stand vs. the bar (all numbers sourced)

| Capability | Us today (measured) | Competitor bar (researched) | Target gate |
|---|---|---|---|
| **Ball 2D track** | Held-out F1@20px 0.6969 chain / 0.7248 zero-shot anchor; recall 0.578 (misses ~40%) | pb.vision self-claims "95%+ shot detection" under strict guideline conditions (no third-party benchmark exists) | BALL M1: F1@20 ≥ 0.90, recall ≥ 0.75, hidden-FP ≤ 0.05 on held-out |
| **Ball 3D flight** | Ground events + net-plane only; no true mid-air 3D; no spin | pb.vision ships 3D shot trajectories (ball only); table-tennis SOTA (TT4D) gets 2.35cm synthetic / lift-first transformer | Full-flight 3D arcs on every trusted segment + spin estimate, physics-sane, fail-closed |
| **Bounce/contact events** | Heuristic cusp+gap anchors; audio unused (eval clips have none) | TTNet-class learned event detectors; audio fusion standard in commercial systems | Contact timing ≤ 40ms vs. reviewed labels; in/out with gray zone |
| **Player 3D (jitter/slide)** | Foot-slide p95 8-23mm on tripod eval clips (gates pass); raw skeleton noise 2-8cm/frame; far players worst; handheld FAILS (330mm) | Nobody ships single-cam player meshes. Broadcast-sports academic bar (SMART/FIFA): global MPJPE 0.324m | BODY world-MPJPE gate on independent GT + jitter/slide bars + handheld pass |
| **Paddle 6-DOF** | IoU 0.24-0.34, ~5°/frame jitter, render-only, not wired into pipeline | Nobody ships this. Research existence proof: 26.4° orientation error from ball-trajectory inversion alone (TT4D) | Face-angle error vs. owner 4-marker GT; wired default; hi-def asset |
| **Court auto-find** | Wave A: Outdoor 4.4px no-tap (best); aggregate 213px vs 200px bar (miss); worktree-only | pb.vision REQUIRES manual-grade stationary setup, breaks on any camera movement | Held-out PCK@5px on owner viewpoints; tennis-overlay + handheld tolerated |
| **Speed** | ~532-565s for a 10s clip (Wolverine); 3163s for ~2min (Outdoor, old baseline); floor 6-8 min/clip booked | pb.vision: "about 30 minutes" typical turnaround (their own docs) | ≤ 2× video duration E2E, then ≤ 1×; < $0.50 GPU cost per game-hour tracked |
| **Coaching output** | 7 rally metrics + facts JSON exist (scaffold); nothing user-facing verified | pb.vision Coach's Insights = clips + stats; users call AI coaching "getting there"; skill-metric ratings are their ROADMAP, not shipped | Grounded-LLM coach with cited 3D moments; zero fabricated numbers (architecture proven in research: 100% no-hallucination compliance achievable) |

## I.3 The six strategic calls (from research + our measurements)

1. **Data engine first — everything else compounds on it.** Measured internally 4×, confirmed
   externally: zero-shot and public-data are exhausted. Your captures (with audio!) are the unlock
   for ball recall, contact events, paddle keypoints, court robustness, and eventually GT gates.
2. **Bodies: keep SAM-3D-Body, kill noise at the source, benchmark challengers honestly.** The
   FIFA-challenge paper (SMART) — architecturally our twin on real broadcast sports — tried a
   learned temporal net, watched it overfit, and reverted to classical smoothing. Our architecture
   is validated; what's missing is (a) camera tracking à la RAFT+MAD, (b) per-track shape locking,
   (c) smoothing in MHR latent-pose space (keeps mesh+skeleton aligned BY CONSTRUCTION), (d) a
   far-player high-res crop pass. GVHMR/PromptHMR-Vid get benchmarked as challengers on OUR footage
   before any ruling reopens.
3. **Ball: fine-tune on owner data with the occlusion recipe + semi-supervised bootstrap; lift to
   3D with synthetic physics.** TOTNet's visibility-weighted loss (occlusion acc 0.63→0.80) targets
   exactly our recall hole; teacher-student pseudo-labeling (SST) stretches every labeled frame
   using our ensemble as teacher on your unlabeled footage; and the mid-air-3D + spin gap is
   solvable with ZERO real 3D labels via a pickleball flight simulator + lift network
   (TT4D/UpliftingTT pattern — proven in table tennis, and our arc solver is already the right
   skeleton).
4. **Paddle: 2D keypoints + masks + planar PnP, not foundation-pose models.** BOP's own handheld-
   tools benchmark (HANDAL) shows the FoundationPose generation scoring 0.04-0.26 AP on exactly our
   object class — skip it. RacketVision proves the 5-keypoint racket schema works; our idle YOLO
   seg checkpoint + IPPE planar PnP is an unexploited, nearly-free 6-DOF signal; WiLoR fixes the
   pronation weakness; the ball-impact inversion factor (already built, dormant) is externally
   validated at 26.4° accuracy.
5. **Court auto-find + motion tolerance is our #1 marketable differentiator.** pb.vision's own docs
   admit any camera movement breaks them, and academic SOTA (PnLCalib) is single-frame-only. Wave B
   (train the staged model, fuse into solver, temporal vote) + distortion-aware handheld calibration
   directly attacks a documented competitor weakness.
6. **Coaching: 3-stage grounded-LLM, never free narration.** Causal ablation evidence (SportsGPT):
   feeding raw numbers to an LLM craters accuracy; rule-grounded comparator + format-locked LLM gets
   100% no-fabrication compliance (Talking Tennis) and coach ratings 8.4-8.9/10. We must build the
   pickleball reference-range library ourselves (none exists publicly) — our trust-band discipline
   is exactly the right substrate for it.

## I.4 Phases at a glance

| Phase | Outcome | Gate that ends it | Runs in parallel with |
|---|---|---|---|
| **P0 Data engine + reset** | GPU back up; capture→ingest→prelabel→CVAT→train loop live; eval suite expanded (audio!); fresh SAM-3D worlds everywhere | First owner capture fully ingested + labeled + registered as eval/train | — (everything depends on it) |
| **P1 Ball** | In-domain detector beats 0.7248 anchor; full 3D flight + spin; learned bounce/contact w/ audio; in/out calls | BALL M1 + contact-timing + 3D-plausibility gates, pre-registered held-out | P2, P3, P4 |
| **P2 Body** | Raw noise ≤1cm/frame effective; far players fixed; handheld placement fix; challenger benchmark ruled | BODY world-MPJPE on independent GT + jitter/slide/handheld bars | P1, P3, P4 |
| **P3 Paddle** | Wired-by-default 6-DOF paddle, keypoint+mask+hand fusion, impact-corrected, hi-def asset | Face-angle vs. owner 4-marker GT + stability bars | P1, P2 |
| **P4 Court + TRK** | No-tap calibration (all points + NET 3D geometry) incl. tennis-overlay + handheld; identity/coverage gates pass | CAL held-out PCK@5px; net height ≤2cm; TRK IDF1 ≥ 0.85, 0 switches | P1, P2, P3 |
| **PF Global fusion** | ONE consistent metric 3D world — ball meets paddle, feet planted, one camera; contact-coupled joint optimization (the combine-everything pillar/capstone) | World-MPJPE + foot-float + ball-paddle-gap all improve vs standalone; cross-system consistency asserted | after P1-P4 partials |
| **P5 Speed + cost** | ≤2× video duration per clip; per-clip $ metered; booked levers landed; per-clip auto-QA/failure-detection | Timed E2E on owner captures + cost dashboard + provable-bound failure gate | continuous |
| **P6 Coaching product** | Shot classification, stats, grounded-LLM coach, visual feedback in viewer | Coach-review rubric + zero-fabrication audit + user-visible demo | after P1/P2 partials |
| **P7 Product scale** | Upload service, accounts, iOS capture guidance, pricing | Real-device E2E + billing dry-run | last |

## I.5 What only YOU can do (highest-leverage owner actions, easiest first)

1. **Record games.** Tripod (≥5ft, whole court + all 4 corners; landscape; 1080p60 preferred), audio ON,
   both courts, varied lighting, full games not rallies. Also: a handful of deliberately-handheld
   clips (that's our motion-tolerance test set). This single action unblocks P1, P3, P4 and the GT
   work in P2. (pb.vision requires this exact tripod setup from every user — we film it once to
   BEAT the requirement.)
2. **Paddle GT capture**: 4 corner markers (tape dots) on your paddle + one slow-mo orbit video of it
   (this doubles as the reference scan for pose evaluation) → unlocks the only path to RKT VERIFIED.
3. **10-min court-label review kit**: `runs/manager/owner_court_label_review_kit_20260705/README.md`
   (4-16 suspect labels currently block the 0.2ft calibration target).
4. `gcloud auth login` (hello@) so the manager can create/delete the GPU VM; then approve one steady
   spot GPU (<$2/hr policy stands).
5. Roboflow API key re-issue (court keypoint corpus export; ball corpus rebuild recipe exists).
6. **One-time profile captures** (EDGE_PLAYBOOK §4.1, ~an afternoon total): ChArUco lens sweep per
   zoom preset (H3); tape-measure player heights (H4); pick ONE ball SKU with max contrast vs your
   court colors and stick to it (H5); 30-60s empty-court clip per court (H6); ball-drop +
   known-speed drill for physics constants (H13); paddle photo orbit + 4-corner-marker clips (H7).
7. Monthly 10-min slow-mo (1080p240) drill session — precision contact/spin GT (H25).

*Owner ETA (2026-07-06): items 1-6 in "a few days" — no lane blocks on them meanwhile (harvest +
eval clips carry the interim); the moment they land, P4-4/P4-0/H13/P3-7 unlock.*

## I.6 Direct next steps (the first ~2 weeks, in order)

1. P0-1 GPU cold start + P0-2 vendor/dirty-tree hygiene (same day).
2. P0-6 fresh 4-clip E2E worlds (fixes stale 65-joint skeleton cache; live-proves contact-dense mesh
   scheduling + paddle arc-stage activation that are already landed-but-unproven).
3. Court Wave B kickoff: P4-1 regenerate+land the Wave A patch (it does NOT apply as-is — see P4-1's
   harsh-review note), then P4-2 train court_unet_v2 (script present locally; see P4-2 note).
4. Owner records first capture batch → P0-3/P0-4 ingest + label factory on it.
5. P1-1 ball fine-tune campaign v2 (owner data, TOTNet recipe, SST bootstrap) the moment ≥1 labeled
   owner clip exists; P1-4 flight simulator build starts in parallel (needs no data at all).
6. P2-1 SMART-recipe hardening lane + P2-2 latent-space smoothing spike (no GPU dependency conflicts
   with P1 training if scheduled via gpu-train-lock).
7. P0-10 — our app ALREADY records the sidecar (intrinsics, ARKit pose, gravity, 240fps modes); the
   work is (a) prove it on a real device recording a real game, (b) wire the server to actually
   CONSUME per-frame ARKit pose + gravity (world-grounding/handheld), (c) add profile-capture steps.
   Plus H27b PTS/VFR ingest audit ships server-side regardless (iPhone = variable frame rate).

---

## I.7 Definition of Done, the critical path, and demo milestones

**DEFINITION OF DONE (v1).** A user with an H0 profile (you first, a friend later) records a full
game in OUR app and uploads it. With zero human intervention the pipeline returns, within ≤2× game
duration: (1) a QA-passed 3D world (P5-5b pre-flight + P5-6 auto-QA green) — court+net, 4 identified
players, full-flight 3D ball, both paddles — where every rendered element is gate-passed or honestly
trust-banded and the Phase-F consistency checks hold (ball meets paddle at contacts, feet planted,
no penetration); (2) a coaching card with ≥5 finding types tied to jump-to 3D moments and 0
fabricated numbers; (3) the component gates on the promotion ladder VERIFIED (BALL M1, TRK, BODY
world-MPJPE, RKT face-angle, CAL PCK). **v1 is DONE when this repeats on 3 consecutive fresh games.**

**THE CRITICAL PATH (the one chain whose delay delays the product):**
DATA (P0-1b harvest + P0-3/P0-4 owner captures/labels) → BALL to bar (P1-0..P1-3) → 3D flight
(P1-4) → contacts (P1-6) → paddle impact (P3-5) + shots/stats (P6-1/P6-2) → fusion (PF-1/PF-2) →
coaching + visual feedback (P6-4/P6-5). Everything else multiplies quality/speed in PARALLEL but
does not gate the first full-value demo: P2 (body noise/far/handheld — current bodies already pass
gates on tripod clips), P4 (generic/unknown courts — P4-0 profiles cover the owner's own courts
immediately), P5 (speed/cost), P3-1..P3-4 (paddle quality beyond impact), P7 (scale).

**CAN START TODAY (zero external dependency — fleet-parallel candidates):** P0-1 GPU · P0-2 hygiene ·
P0-1b harvest · P0-7 simulator (seeded with the measured Cd/Cl) · P0-8 VFR audit · P0-9 registry
schema · P2-1 SMART hardening (GPU-light; schedulable without blocking P1 training) · P6-3 reference-library v0 · P0-10(b) sidecar
server-consumption design; P4-0 the moment the owner does the 15-minute profile captures.

**DEMO MILESTONES (what the owner SEES, in order — march to these):**
- **M1 — "it all runs together" (days):** fresh GPU; fresh 4-clip worlds co-proving every
  landed-but-unproven feature (contact-dense meshes, paddle arc-factor plumbing, court-map, viewer
  honesty). [P0-1..P0-6]
- **M2 — "the data engine is alive":** first harvest corpus card + first owner game ingested with
  profiles; pseudo-labels flowing; nightly flywheel scheduled. [P0-1b, P0-3/P0-4, P1-2]
- **M3 — "the ball is fixed":** held-out ≥0.7248 finally beaten + full-flight 3D arcs visible in the
  viewer. [P1-1..P1-4]
- **M4 — "it coaches me":** first coaching card on an owner game — stats + grounded-LLM v0,
  fabrication-audited. [P6-1..P6-4 partial]
- **M5 — "one world, and a friend can use it":** fused consistent world (ball provably meets paddle)
  + a friend onboards via H0. [PF-1/PF-2, P7-1 partial]

# PART II — RESEARCH VERDICTS (what the sweep found, and what we do about it)

Full cited reports: `runs/research_sota_20260705/{ball,body,paddle,product}_report.md`.
Each ran ~28 Sonnet agents: 5-6 search angles → 14+ primary-source deep reads → completeness critic
→ gap-fill → 2-vote adversarial refutation on load-bearing claims → synthesis. Corrections from the
refutation pass are already applied below (several "known facts" died in verification — e.g. the
supposed pb.vision patent belongs to a cricket company; a famous 94%-TPR pickleball fine-tune blog
is an overfit artifact of a 65%-TPR project).

## II.1 Ball (ball_report.md)

- **Precise version of "no pickleball data" (the phrasing to stop overstating).** What's true:
  no pickleball *academic tracking benchmark or SOTA paper* exists (2023-2026), and no *published
  temporal ball-tracking dataset*. What DOES exist: many Roboflow pickleball **detection** datasets
  (bounding boxes on frames) + court-keypoint sets. **We already used them:** we aggregated an
  8,631-frame Roboflow-only corpus (leakage-checked 0/41,866, licensed) and fine-tuned both ball
  architectures on an A100 — **both DEGRADED held-out** (TrackNetV3 −17pt on the reference clip;
  WASB catastrophic 0.0018 static-distractor lock). Evidence:
  `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`. Three measured reasons Roboflow data
  didn't beat the baseline: (1) **wrong domain** — broadcast/tournament stills, not phone-on-tripod
  amateur games → the internal→held-out inversion; (2) **wrong format** — 84% are temporally-dead
  isolated frames, but our detectors (TrackNet/WASB) are TRACKING models that need 3 consecutive
  labeled frames to exploit motion; (3) **too few sources** → overfit/distractor-lock. Externally
  replicated: U. Rochester got 53%→65% TPR on 12k single-match frames, no cross-match validation.
  **So Roboflow data is NOT useless — it's just not sufficient ALONE** (see P1-1's corrected use).
- **Adopt:** TOTNet occlusion recipe (MIT) — visibility-weighted 4-level BCE **+** occlusion
  augmentation together (aug alone measurably hurts); fully-occluded accuracy 0.63→0.80 in tennis/TT.
  TrackNetV4 motion-prompt layer (+1.6 F1, +1.8 recall, drop-in for heatmap nets — paper recipe, not
  their unusable weights). RacketVision findings: background modeling −54-61% error; **multi-sport
  joint training +14-19% mAP** (use tennis/badminton/TT corpora as auxiliary data WITH owner data,
  never alone); ball↔racket cross-attention helps, naive concat hurts.
- **Semi-supervised bootstrap (SST, BSD-3):** teacher-student pseudo-labeling, mAP 18.1→26.2 at 1%
  labels in soccer; our WASB+BlurBall+TrackNet ensemble is the teacher over owner unlabeled footage.
  Caveat: inherits teacher false-negatives — pair with active learning on disagreement frames.
- **3D lift with zero real 3D labels:** TT3D (per-segment drag+Magnus ODE fit on reprojection — our
  arc solver's architecture, externally validated; beware 2× error at unfavorable camera angles) and
  TT4D / UpliftingTT (transformers trained purely on MuJoCo-synthetic trajectories; 2.35cm position,
  ~97% spin accuracy under degraded detections; >500 pts/s). **Build a pickleball flight simulator**
  (drag/Magnus coefficients for a holey plastic ball differ from anything published — measure from
  owner captures' 2D tracks + court constraints).
- **Contact events:** trajectory-kink + audio-onset + pose-cue fusion is the standard; our IMG_1605
  already has 30 real audio onsets as the first testbed. TT4D proves paddle-impact inversion works.
- **Calibrate ambition:** pb.vision discloses zero tech; the sport's own line-calling deployment
  (PlayReplay) uses FOUR net-post cameras; Hawk-Eye 6+; Zenniz adds 30 audio sensors. Single-camera
  officiating-grade in/out does not exist anywhere — our gray-zone trust-band design is correct.
  Keep in/out "advisory with uncertainty", never "officiating".

## II.2 Body (body_report.md)

- **The decisive same-domain datapoint (SMART, FIFA WorldPose challenge, broadcast soccer):**
  per-frame mesh model + optical-flow camera tracking + foot-plane anchoring + classical smoothing —
  i.e., OUR architecture — beat the baseline by 38.6%; their learned temporal-refinement net won on
  validation (+22%) and LOST on held-out (over-smoothing), so they shipped Gaussian+MAD classical
  smoothing. Direct evidence our classical stack is not a dead end. Portable pieces: RAFT-small +
  MAD-outlier camera tracking (0.041° rotation error, 55ms/frame, beats homography), per-track
  shape/scale locking, two-pass MAD+Gaussian smoothing.
- **Challengers exist but are unproven on our footage class:** GVHMR (gravity-view frame; RICH/EMDB
  foot-slide 3.0-3.5mm, jitter 12.8-16.7×10m/s³ — best-in-class; non-causal, NC license, SMPL out);
  PromptHMR-Vid (best EMDB-2 jitter 16.3 / skate 3.5mm; full-image prompts built for small/far/
  crowded people — the most interesting for us; no speed numbers, NC). ALL benchmarks are
  pedestrian-scale ground-level datasets; NONE tested at 4-player-court scale, elevated/far camera.
  A swap without our own benchmark would be faith, not engineering.
- **Frozen-backbone add-ons match the owner ruling:** SAM-Body4D (MIT, training-free) feeds SAM3
  video masklets as temporally-consistent prompts to SAM-3D-Body — plausible input-side jitter fix,
  zero published numbers → cheap experiment. The Dec-2025 retargeting paper (no code/metrics) is a
  blueprint identical to our stack and endorses **latent-space smoothing on the MHR pose code** —
  the one idea that reduces joint noise while keeping mesh+skeleton aligned by construction.
- **Handheld:** TRAM's masked-DROID-SLAM ablation (camera ATE 2.42m→0.32m by masking humans out of
  SLAM) is the proven fix pattern for our IMG_1605 41px-drift failure; we can adopt the
  mask-the-players + optical-flow idea without adopting TRAM's SMPL half.
- **Small/far players:** controlled study: MPJPE 33.3→42.6mm from large→small crops, cause = 2D
  keypoint degradation at low res → high-res crop re-inference for far players is a plausible lever.
- **Licensing [CORROBORATED]:** MHR is Apache-2.0 (fully safe); SAM-3D-Body is a custom "SAM License"
  (fine for internal use; not Apache — noted); every serious challenger (GVHMR/WHAM/TRAM/PromptHMR/
  DPoser) drags in NC SMPL — fine internally per owner ruling, blocks commercial shipping later.
- **Skip:** physics refiners (PhysPT class — parked blockers unchanged, lab-only evidence); DPoser-X
  (per-frame prior, ~2cm residual ≈ no better than our filtering); OnlineHMR (3.3 FPS); MultiPly
  (24 GPU-hr/person); CoMotion/RAM tracking are watch-list (RAM: 15 ID-switches vs 344-349 for
  competitors, but no code released yet; CoMotion admits track collapse in close-proximity scenes —
  exactly doubles).

## II.3 Paddle (paddle_report.md)

- **We are first.** No modern 6-DOF racket/paddle work exists. Closest is RacketVision (MIT):
  2D-only 5-keypoint racket schema (Top/Bottom/Handle/Left/Right) via RTMDet+RTMPose; PCK@0.2
  81.8-89.6% overall but **side/edge keypoints (face orientation) only 64.8-80.1%** — the same weak
  axis we measured. That's the realistic expectation-setter.
- **Foundation-pose models are the wrong tool [strong negative evidence]:** BOP's handheld-tools
  benchmark (HANDAL — hammers/spatulas ≈ paddles) shows the FoundationPose/GigaPose generation at
  0.04-0.26 AP; the model-free in-hand track got ZERO submissions in 2024; SAM-6D/FreeZe need RGB-D.
  FoundationPose also carries a restrictive NVIDIA Source Code License. Do not build on these.
- **Do:** (a) paddle 5-keypoint detector fine-tuned on owner CVAT corners → extra PnP constraints in
  our existing fusion [highest-confidence option]; (b) exploit the IDLE YOLO seg checkpoint:
  mask → oriented quadrilateral → IPPE planar PnP (BSD-3, 50-80× faster for rectangles) — a
  white-space nobody has published, nearly free; risk = mask quality exactly at impact (the spoon-
  tracking analog found segmentation is the binding constraint); (c) WiLoR hand crops (CC-BY-NC-ND
  ok; detector 138-175fps; per-frame-smooth without temporal parts) to fix rest-pose pronation;
  (d) nvdiffrast render-and-compare refinement at SPARSE impact keyframes only (Diff-DOPE: 3.5s/
  frame → per-frame impossible, keyframes fine); (e) activate the ball-reflection factor when P1's
  3D velocities land — TT4D existence proof: 26.4°±4.4 orientation from trajectory inversion alone.
- **Novel niches available to us:** blur-axis-as-orientation for an elongated implement (nobody has
  done it; BlurBall's PCA blur cue works only for symmetric balls) — optional spike, publishable.
- **Skip:** HOISDF/MOHO joint hand-object nets (NO license files = hard block + YCB domain mismatch);
  HOLD (25 GPU-hours/video).

## II.4 Product / competitive / coaching (product_report.md)

- **pb.vision, precisely:** 3D = ball trajectory ONLY (no player mesh/skeleton anywhere); requires
  fully stationary camera ≥5ft with all corners+players always in frame, no stabilization/HDR; any
  movement breaks alignment (their docs admit it); self-reported "95%+ shot detection" under those
  guideline conditions, zero third-party benchmarks; ~30min typical processing; minutes-based pricing
  ($19.99/mo 100min, $49.99/mo 400min, 5-min floor; API $8/hr HD video). Their skill-rating CV
  metrics are roadmap aspiration; injury/posture coaching is "exploring" — unclaimed space. Users:
  coaching "getting there", only 4-5 metrics actually used, best for competitive players.
- **SwingVision** = on-device iPhone (Neural Engine), 500M+ shots of training data, 2-camera
  line-calling coming "99%" (single-cam admittedly lower), $179.99/yr. **PlaySight** = venue
  hardware + Azure-OpenAI "5 strengths/weaknesses in minutes" for rec players. **SportAI** =
  hardware-agnostic technique overlays (Shark pickleball partnership). **Zenniz** = 4 cams + 30
  audio sensors, ITF Silver, ~7mm — a different (installed-hardware) category.
- **→ Our differentiators, in order:** (1) full 3D world (players+paddle+ball) from ONE camera —
  no one ships it; (2) camera-motion tolerance — attacks pb.vision's documented weakness, and
  academic SOTA calibration is single-frame-only; (3) honest trust bands (their error modes are
  silent); (4) coaching tied to 3D moments with visual feedback.
- **Coaching architecture (causally validated):** 3 stages — deterministic feature extraction →
  rule/reference-range comparator (NOT the LLM) → format-locked LLM (score + exactly-N corrections).
  Talking Tennis: 100% no-fabrication over 317 outputs, coaches 8.4-8.9/10. SportsGPT ablations: raw
  numbers into the LLM tank accuracy 3.9→2.85; removing grounding tanks feasibility 3.9→1.65.
  **No pickleball reference-range library exists — building it is a moat.** Trade-press benchmark
  seeds: 3rd-shot-drop success ~50% (3.0) → 85-90% (4.0, drill); ~40% amateur points from unforced
  errors; minimal valuable stat set = unforced errors, third-shot success, dink-rally win rate; a
  "good drop" is defined by the OPPONENT's forced upward trajectory — requires our 3D ball data.
- **Legal watch item:** US11615540B2 (Maiden AI — cricket, NOT pb.vision) claims broad single-camera
  3D ball-tracking pipeline shape to 2042. Internal-only use is fine; get counsel review before any
  commercial launch.
- **Economics context:** H100 ~$2-3.35/hr on-demand, spot 60-80% off; pb.vision's $8/hr-of-video API
  price is the market reference ceiling; our GPU cost today ≈ $0.117/clip BODY — unit economics are
  already fine, engineering time is the scarce resource.

---

# PART II-B — PASS 2 DEEP-DIVE ADDENDA (citation-graph traversal, 2026-07-06)

A second, deeper pass (159 agents; `runs/research_sota_20260705/pass2_*_report.md`) traversed every
pass-1 paper's sub-tasks + reference graph + a cutting-edge recency sweep. It was triggered by the
owner noticing we'd missed RacketVision's **TrajPred**. It found that plus two corrections to THIS
doc and a set of new adoptions. Verdicts are adversarially checked; "no code yet" flagged explicitly.
**[CORROBORATED]** marks a research claim that survived our adversarial fact-check — it is NOT the reserved
word `VERIFIED` (which means a passed PRODUCT gate; still zero of those).

### Corrections to our own plan (we stated these wrong)
1. **Pickleball aerodynamics DO exist — stop saying "measure, don't assume" as if there's no prior.**
   Two independent studies measured real pickleball drag/lift: Lindsey (TWU, May 2025) outdoor
   40-hole Cd≈0.33, indoor 26-hole Cd≈0.45, with **asymmetric topspin>backspin lift (violates
   classical Magnus symmetry)**; Steyn et al. (arXiv:2501.00163, Dec 2024) outdoor Cd=0.30±0.02,
   Cl=0.195·S. The two outdoor numbers **corroborate** [CORROBORATED]. → P0-7 now seeds the simulator +
   arc solver with these constants (still refine from owner captures, but we have a real prior, not
   a blank). This tightens the 3D fit with zero data/training — a closed-form constant swap.
2. **The "latent-space smoothing" idea (P2-2) now has a published blueprint on our EXACT backbone.**
   "World-Coordinate Human Motion Retargeting via SAM 3D Body" (arXiv:2512.21573) wraps frozen
   SAM-3D-Body+MHR with (a) per-subject shape/scale locking, (b) sliding-window optimization in MHR
   **latent** space, (c) a differentiable soft foot-contact model — exactly P2-2. Use as the design
   reference (robotics-retarget paper, qualitative-only evidence — prototype against our floor-sink/
   root-jitter defects, don't assume the numbers).

### New adoptions mapped to tasks (ranked; full evidence in pass2 reports)
**NOW (cheap, high-confidence, or corrections):**
- **Pickleball Cd/Cl constants** → P0-7 / P1-4 (above).
- **SOMA-X** (NVlabs, Apache-2.0, `py-soma-x`): GPU canonical MHR↔SMPL-X pivot → P2 interop layer so
  every SMPL-X-based prior/tool (SmoothNet, DPoser-X, coaching-viz) runs on our MHR output for free.
- **RacketVision is multi-task, not just TrajPred** (MIT, public checkpoints): **TrajPred** =
  cross-attention ball+racket forecasting (racket K/V, ball Q — beats naive concat, which HURTS);
  **RTMPose 5-keypoint** racket head (PCK 81.8-89.6, handle 92.6-97.9); **MS-TrackNetV3** ball head.
  → new P1-8 (forecasting) + P3-4 (keypoints) + P1-1 (aux).
- **SoccerNet-v3D ball-diameter depth anchoring** (IoU 0.57→0.66 via known diameter) → P1-4 + it
  CITES our EDGE_PLAYBOOK H23 hack — pickleball's fixed 74mm diameter makes the formula drop-in.
- **RF-DETR** (Apache, DINOv2, fast fine-tune, strong cross-domain, T4 2-4ms → leaves A100 headroom)
  → P3-1/P4-2 detection/keypoint backbone.
- **Grounding DINO** zero-shot "pickleball paddle" boxes with ZERO in-domain labels → P0-4/P3-1
  bootstrap.

**SOON (real code, needs a lane):**
- **Uplifting Table Tennis** (WACV'26, code): RoPE-**timestamp**-keyed transformer 2D→3D+spin trained
  synthetic-only, robust to missing detections + variable FPS (pairs with our P0-8 VFR/PTS work) →
  P1-4 learned-lift track.
- **LATTE-MV** (Berkeley, code+data): turn ~50h ordinary monocular video into a 73k-exchange 3D
  corpus + anticipation model → P0 data-engine template + P6 prediction (raises sim return 49.9→59.0%).
- **BLADE** (NVlabs CVPR'25, code) + **PersPose** (ICCV'25, 60.1mm 3DPW, code) + **KASportsFormer**
  (58mm SportsPose/34.3 WorldPose, code): close-range perspective + sports bone-constraints → P2-3
  (our single close iPhone is exactly their regime).
- **OnePoseViaGen** (CoRL'25 oral, code): 1 reference photo → generative texture/view randomization →
  paddle pose head, zero labels → P3-2/P3-4 (matches our paddle-scan asset H7).
- **RGBTrack** (IROS'25, code): depth-free FoundationPose + tracking → P3 paddle frame-to-frame
  stability (feed the flat-rectangle paddle proxy).
- **Image-as-an-IMU** (Oxford, code): invert motion blur → instantaneous 6-DOF angular velocity →
  P3-8 (this is our blur-axis hack H22, with released code) — fuse at fast swing/contact frames.
- **CoachMe** (reference-motion skeletal-graph deviation → instruction) + **BioCoach** (VLM w/
  per-skill DOF selector) → P6-4 grounded coaching reference+feature-selection layer.
- **AnyCalib** (ICCV'25, code): single-image intrinsics/distortion for unknown lens → P0/P4 across
  iPhone models; **BroadTrack / SoccerNet-GSR** temporal calibration (LK flow + RANSAC per-frame
  homography) → P4 handheld court stability.
- **WASB HLSM** (Hard-to-Localize Sample Mining): retrain WASB focusing on missed/blurred frames →
  P1-1 fine-tune recipe on owner data. **TrackNetV5-SDK** (Motion-Direction-Decoupling + refinement
  transformer, F1 0.9859 tennis, +3.7% FLOPs) → P1-1 detector-upgrade candidate.
- **MoPO** (occlusion de-occlusion motion prior) → P2 occlusion-specific jitter (net/paddle/overlap).
- **MFS event spotting** (frame-exact hit/bounce/serve) + **RANSAC pre-filter + LM/Ceres robust loss
  audit** → P1-6/P2/P4 cheap hardening.

**SPIKE / WATCH (no code yet — monitor for release):**
- **Human3R** (ICLR'26): feed-forward multi-person SMPL-X + scene + camera in ONE pass, 15 FPS/8GB,
  1-GPU-day training — could COLLAPSE our mesh+camera+smoothing stages; bake-off in P2-7.
- **DuoMo** (CVPR'26): camera+world diffusion, biggest reported margin (−16% EMDB/−30% RICH, low
  foot-skate), no code → watch.
- **Where Is The Ball** (camera-independent plane-point lift; Height/depth net is the dominant error
  term), **TAPNext++** (point tracking with re-detection for ball reappearance), **HMRMamba**,
  **ComPose** (adaptive hand-vs-object trust — our exact fusion weakness), **PMGS/PEGS** (3DGS
  projectile w/ Newtonian loss = differentiable arc solver), Gaussian-splat pose family
  (GSGTrack/GHOST/GTR/6DOPE-GS, tuned for thin/symmetric/low-texture = paddle) → spikes/bake-offs.

### Competitive update
**Owl AI × Major League Pickleball** launched automated officiating (line calls + challenge review)
**live May 22 2026**, cloud CV on existing broadcast cameras — first pickleball-specific automated
officiating in production (multi-camera broadcast, not single-consumer — doesn't change our wedge).

### Naming/infra cautions surfaced
- TWO different papers named **"SMART"** (pass-1 FIFA one vs a newer SMPLest-X+RAFT one, arXiv
  2605.31551) — disambiguate before citing.
- Our tree vendors **`third_party/SAT-HMR`** while the ruling backbone is SAM-3D-Body — a pass-2
  agent flagged uncertainty about which is actually live; verify in P0-6 (don't assume).

# PART II-C — PASS 3 DEEP-DIVE ADDENDA (court/net + global fusion + production, 2026-07-06)

A third pass (87 agents; `runs/research_sota_20260705/pass3_{court_net,fusion,crosspillar}_report.md`)
went deep on the two under-researched pillars the owner named — court/net calibration and the
"combine-everything" fusion step — plus a production sweep. Headlines:

### The biggest architectural finding: we're missing a GLOBAL FUSION stage (now Phase F)
Our pipeline solves court → camera → human → ball → paddle **independently and composites at the
end**, so nothing guarantees the ball meets the paddle or feet don't float/penetrate. The published
SOTA answer is **contact-coupled joint optimization** (JOSH, ICLR'26, arXiv:2501.02158): take each
subsystem's output as *initialization*, then run ONE optimization over scene+camera+humans using
**contact as the coupling residual**. Evidence: contact-scale loss alone cut WA-MPJPE 314→149mm
(−53%) on SLOPER4D; full joint opt → 120mm; foot-floating 9.0→2.9%; beats TRAM/WHAM/SLAHMR on
EMDB/RICH [CORROBORATED, SLOPER4D-specific]. **Iterative optimization still beats every feed-forward
method on accuracy as of mid-2026** (JOSH W-MPJPE 174.7 vs SHOW 262.3 vs JOSH3R 661.7 on EMDB-2) —
so for our offline pipeline, joint optimization is the right bet. **No published system fuses
ball+body+paddle+court for any racket sport** — extending JOSH's contact-coupling to also include
ball↔paddle-impact and ball↔ground-bounce residuals, initialized from our ARKit pose/gravity + known
metric priors (court 20×44ft, net 36/34in, ball 74mm, player heights), is genuinely novel work and
the capstone of the whole project → new **PHASE F**.

### Court/net (deepen P4) — and net 3D is the field's single biggest open gap
- **No pickleball court/net calibration prior art exists anywhere** — this is novel integration, not
  a port. The mature floor architecture is the **PnLCalib / No-Bells-Just-Whistles** points+lines
  lineage (dual-HRNet keypoint+line heatmaps + PnL refinement, +5.7-8.3 FS; GPL-2.0 code). **Reality
  check:** even SOTA sports calibration reaches only 1.4-4.5px reprojection on EASIER broadcast
  domains vs our **19.8px p95** on the harder handheld/close/distorted/overlaid-line pickleball case
  — so **a pilot must prove a retrained net clears our bar before we commit** (don't assume a port wins).
- **Exploit ARKit (our unfair advantage):** **PoseGravity** (BSD-3 code) is a closed-form solver for
  camera pose from points+lines given a KNOWN gravity axis — exactly our sidecar case; nobody has
  fused ARKit gravity+intrinsics+plane-anchors with a court-line detector end-to-end (novel for us).
  **AnyCalib/GeoCalib** sanity-check intrinsics/distortion per clip.
- **NET as full 3D geometry (new P4-6), not a plane.** Today `net_plane` is a single plane; real net
  = 2 posts, top cord 36in ends/34in center, center strap, sag. Nobody reconstructs this in any sport
  (TennisCourtDetector has ZERO net logic — verified). Path: a net-cord segmentation model (thin/
  low-contrast/occluded — the hard part) + **3D catenary curve fit** (Madaan ICRA'19, 5-param planar
  catenary, works with known ARKit camera poses + degrades gracefully under occlusion). The net cord
  then doubles as a calibration anchor and a 3D occluder for ball/player.
- **Temporal:** BroadTrack (halves reproj error 10.28→5.02px via camera-motion model) / BHITK
  (Kalman homography) — but feed ARKit VIO as the motion prior (open territory). **Court re-ID**
  (H1 profile library): PlanaReLoc (plane-primitive relocalization) is the right shape but no code yet.

### Cross-pillar + production (map to tasks)
- **NOW/HIGH — production auto-QA:** wrap the ball tracker (and any tracker) in **sequential-
  hypothesis-testing failure detection** (arXiv:2602.12983 — anytime-valid, provable false-alarm
  bound via Ville's inequality, no retraining, negligible compute) → P5/P7 per-clip self-eval that
  catches a bad output before the user sees it. This is the reliability primitive our plan lacked.
- **Hybrid physics+NN trajectory** (arXiv:2503.18584): closed-form drag/Magnus + tiny MLP (3 coeffs),
  **includes a pickleball test split**, needs only ~90 trajectories, 0.97ms → P0-7/P1-4 (less data
  than a pure-learned lift).
- **ForeHOI** (arXiv:2602.06226, code+weights) [CORRECTED by tech-audit: runs ~1 MINUTE/clip not <1s,
  and its pose comes from bolting on FoundationPose — inheriting the documented HANDAL weakness
  (0.04-0.26 AP on thin handheld objects); demoted to backlog spike]: feed-forward 3D geometry of an arbitrary
  HELD object from monocular hand-object video in <1s → P3 paddle (feed-forward alternative to the
  fused estimator; trained on synthetic GraspXL).
- **Body scene+human backbones** (P2-7 bake-off additions): UniCon3R (contact-aware, floating
  33.6→5.5cm, penetration 6.4→1.6cm — but no code yet), SHOW (scale-fix, no code), Multi-HMR 2
  (SAM2-memory multi-person ID without video supervision — relevant to 4-player ID), all on the
  **VGGT** backbone; **Depth Anything 3** (code+weights) claims to beat VGGT — evaluate as a scene/
  camera backbone. **PhysDynPose+MoviCam** (MIT code): non-flat-ground + handheld + contact,
  foot-slide 3.22mm (but 68% penetration remains — partial).
- **Feed ARKit gravity/pose into GVHMR's Gravity-View frame** (replaces its DPVO weakest link) →
  P0-10/P2 low-risk integration that our device data uniquely enables.
- **AI-coaching trust:** "Synthesizing the Expert" (arXiv:2605.12799) — 4-stage LLM pipeline w/ 12
  domain rejection rules, 97.4% auto-accept on 1,914 synthetic Q&A → P6-4 grounding+validation blueprint.
- **Integration references** (released, modular single-camera pipelines to study): SoccerNet-GSR
  (calibration→detection→ReID→tracking→OCR + GS-HOTA metric) and TrackID3x3 (fixed-roster multi-player
  ID + court-relative identity, Apache-2.0) → P2-5 / Phase F reference architectures.
- **Latency lesson:** GPU-accelerated postprocessing cut a video-analytics pipeline 40.2→11.4ms/frame
  (arXiv:2512.07009) — echoes our own "plumbing dominates" finding → P5.

### Two more corrections (keep our facts clean)
1. **Do NOT confuse pickleball aerodynamics papers.** The MEASURED coefficients are Lindsey (TWU 2025)
   + Steyn (arXiv:2501.00163). A THIRD paper, "Pickleball Flight Dynamics" (arXiv:2409.19000), is a
   **simulation** with Cd≈0.6 borrowed from wiffle-ball (±0.5) and spin explicitly ignored — do NOT
   use it as a calibrated prior. P0-7 uses Lindsey/Steyn only.
2. **Patent misattribution (again).** US11045705B2 / US12478848B2 belong to **Nex Team (HomeCourt
   basketball)**, not pb.vision (like the earlier cricket-patent mixup) — treat as analogous method
   only. pb.vision still discloses zero verifiable architecture.

# PART III — PHASE CHECKLISTS (agent-facing; every task self-contained)

Conventions: each task has an ID (stable — reference it in BUILD_CHECKLIST bullets), file pointers,
an acceptance gate, and kill criteria. Lanes must be file-disjoint; coordinate via BUILD_CHECKLIST
only. Protected clips: Outdoor/Indoor labels NEVER without a pre-registered
`runs/manager/heldout_eval_ledger.md` row. New models/datasets: record license VERBATIM in the lane
report (internal-use policy: research/NC acceptable; NO-LICENSE-FILE repos are hard-blocked).

## PHASE 0 — Foundation reset + the data engine (unblocks everything)

**Already built:** E2E orchestrator, ball 3D chain default, SAM-3D body/world, fused paddle, viewer,
data-engine ingest+prelabel+guards, the iOS app (110 Swift files, sidecar with intrinsics/ARKit-pose/
gravity/240fps), server ingest of that sidecar (intrinsics/provenance/taps). **To build:** GPU back
up, tree hygiene, the profile registry (P0-9), on-device capture proof + ARKit-pose/gravity server
consumption (P0-10), expanded eval suite with audio, fresh gate-passing worlds.

- [ ] **P0-1 GPU cold start.** Follow `RESET_HANDOFF_20260705.md` §7: create ONE spot VM (<$2/hr;
  L4 for detector work, A100-40GB when VRAM-bound e.g. BODY/SAM-3D), `git clone` for the GPU-VM compute env (code is pushed — do NOT resurrect scp-sync; NOTE: the
  docs-of-record incl. this file + EDGE_PLAYBOOK are pending the owner joint-commit, so a fresh clone
  lacks them until that lands — §0 owner-setup), restore vendor pins per `third_party/VENDOR_PINS.md`, run
  `scripts/racketsport/gpu_cold_start.sh` (258s proven), pull weights per `models/MANIFEST.json` +
  ledger sha256s, serialize via `scripts/gpu-train-lock.sh`. Gate: `nvidia-smi` + one smoke BODY
  dispatch completes. *Owner dep: gcloud auth (I.5 #4).*
- [ ] **P0-2 Tree hygiene.** Reconcile the current dirty tree: `runs/lanes/ball_tracking_long_run_STATUS.md`
  mod + vendor content changes (SAT-HMR/WASB-SBDT/blurball content, untracked `third_party/TrackNetV4`)
  → commit or pin per VENDOR_PINS policy. Resolve the 2 known pre-existing test failures
  (`tests/racketsport/test_overlapping_court_calibration_eval.py`): the reviewed IMG_1605
  `court_keypoints.json` lacks the `frames` metadata block the loader requires — a `needs-validation`
  owner call: (a) loader-side compat shim (fast default) vs (b) re-export the label. Until answered,
  proceed on everything else; the pair stays a booked pre-registered failure.
  Gate: wide suite (MPLBACKEND=Agg) green minus explicitly-booked failures; `git status` clean.
- [ ] **P0-1b Online-video harvest → clip → label (data source that needs ZERO owner recording time;
  owner 2026-07-06).** There are countless standard-tripod pickleball games online. Build a
  harvest→clip→prelabel pipeline: `yt-dlp` bulk pull of tripod-setup full-court games (varied courts/
  lighting/skill), auto-clip to rallies, prelabel with the current ensemble (feeds SST P1-2), review a
  subset in CVAT. Use for BOTH training AND broad testing. **Discipline:** online clips are OUT-of-domain
  (their cameras, not ours) → they are a diversity/pretrain + broad-test asset, NOT the in-domain
  finisher (owner captures still do that, P0-3). Assign roles at ingest (train / internal-val / a couple
  reserved as fresh held-out with the ledger); DO NOT train on competitor-processed videos (pb.vision/
  SwingVision outputs) or anything whose labels we can't trust. OWNER RULING 2026-07-06: BROAD harvest approved (private
  internal use, never redistributed — copyright waived as a concern by the owner); still no
  non-owner biometric persistence. Gate: N harvested games ingested with
  roles + provenance, dedup vs eval clips, a corpus card (counts/sources/courts).
- [ ] **P0-3 Owner-capture ingest loop, live.** First real batch through
  `scripts/racketsport/ingest_owner_capture.py` → `prelabel_owner_capture.py` → CVAT project.
  Verify eval-clip guards fire (they are attack-tested — keep them). Register every new clip's role
  ON INGEST: `train` | `internal-val` | `held-out` (held-out clips get ledger protection immediately;
  we need ≥2 new held-out clips with audio). Gate: one capture fully ingested with court keypoints,
  role registered, prelabels loaded in CVAT.
- [ ] **P0-4 Labeling factory.** Throughput plan for owner+helpers: (a) ball: prelabel with current
  ensemble, human corrects (SST teacher output = prelabels — see P1-2; CVAT SAM3 integration is
  per-frame only, no video propagation — don't plan around it); (b) paddle: boxes + 4 visible-corner
  keypoints on contact-window frames (RacketVision needed 24.6k annotations for 3 sports — we need
  low-thousands to start); (c) contacts: audio-onset-seeded event marking (fast); (d) court: corners
  per new viewpoint. Budget targets (research-informed): ball ≥ 10-20k corrected frames across ≥4
  distinct sessions/courts before the first fine-tune shot (single-match fine-tunes are a proven
  failure mode — diversity beats volume); paddle ≥ 1-2k keypoint frames; contacts ≥ 500 events.
  Gate: labels/hour measured; corpus dashboard script in `scripts/racketsport/`.
- [ ] **P0-5 Eval-suite expansion.** Fix Indoor CVAT export (missing); add ≥2 owner held-out clips
  (WITH audio — unblocks BALL M4 sub-gate + contact gates) + ≥1 deliberately-handheld clip as the
  motion-tolerance benchmark; document in `eval_clips/ball/README.md`. Gate: held-out ledger updated;
  eval_guard passes on all new clips.
- [ ] **P0-6 Fresh worlds + stale-cache purge.** Rerun full E2E on all 4 eval clips on the new GPU
  (`--clip` flag! `--remote-command-timeout-s 7200` for Outdoor): purges the stale 65-joint RTMW
  skeleton caches (found by the ball-render lane `runs/lanes/ball_p4_render_fix_20260706` — its 'P4'
  is that lane's own phase name, NOT Phase 4 of this roadmap), live-proves
  contact-dense mesh scheduling (`ball_aware`) + paddle arc-stage activation (both landed, never
  co-proven). Wire `events_selected.json` production from the default arc chain (wiring gap:
  `threed/racketsport/ball_arc_chain.py:158` writes arc-solved but not events_selected —
  see `runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md`). Gate: 4 fresh worlds, browser-
  verified via `verify_process_video_viewer.py`; contact-dense scheduling reports non-uniform
  coverage; ball-reflection factor PLUMBING verified pre-activation (it stays dormant until P1-4's 3D
  velocities exist — a Phase 1 task — so non-dormancy is P3-5's gate, not P0-6's); AND confirm which
  body backbone is actually live from the run artifacts — verify, don't assume (Part II-B caution;
  tech-audit 2026-07-06 verified the CURRENT answer: SAM-3D-Body is the live default, SAT-HMR is
  vendored-only — keep the cheap per-run check anyway).
- [ ] **P0-7 Pickleball flight simulator (no data needed — start immediately).** Ballistic + drag +
  Magnus generator for a 26g holey plastic ball; **SEED with the real measured pickleball
  coefficients** (Lindsey TWU 2025: outdoor Cd≈0.33 / indoor Cd≈0.45, asymmetric topspin>backspin
  lift; Steyn arXiv:2501.00163: Cd=0.30±0.02, Cl=0.195·S — Part II-B), then refine by fitting our
  own 2D tracks + court geometry. Phase 1 = pure-numpy ODE (no new installs — Codex-safe); Phase 2 =
  MuJoCo (Apache; TT4D's stitching pattern samples physically-valid flight/bounce/hit segments) —
  NOTE MuJoCo is NOT installed in `.venv` and Codex has no network: a network-capable prestage
  (Sonnet/manager) must run `scripts/racketsport/install_mujoco_mjx_env.sh` first. Camera-projection
  module reusing our calibration schema; noise/dropout models matched to
  our detector error profile (p95 34px, recall 0.578, hidden-FP 0.021). Output: unlimited synthetic
  (2D track ↔ 3D flight + spin + bounce) pairs. This powers P1-4, P1-5, and P3-6.
  Gate: simulated 2D tracks pass our own flight-sanity checker; error-profile match report.
- [ ] **P0-8 iPhone metadata harvest + VFR-correctness audit (server-side, any video).**
  EDGE_PLAYBOOK H27/H27b: parse QuickTime metadata at ingest (GPS→court, timestamp→sun/lighting,
  lens→device profile, HDR-transfer fix before color gating); and the MANDATORY fix — iPhone shoots
  VARIABLE frame rate, so emit a PTS-accurate frame-time table and make every timing consumer
  (contact times, velocities, arc-solver dt, audio sync) use real timestamps, not assumed fps.
  Gate: PTS table on every clip; audit shows no constant-fps assumption survives; a synthetic VFR
  clip yields correct contact timing.
- [ ] **P0-9 Profile registry (the multi-user backbone; EDGE_PLAYBOOK H0).** Per-account data model
  for 5 profile types — court (calibration/line-color/background/net), device (intrinsics/exposure
  constant), player (height/frozen-betas/ReID-gallery/handedness, shareable across accounts), gear
  (paddle scan + ball SKU), session cache. RULED (don't re-decide): storage = flat per-account JSON
  under `runs/profiles/<account_id>/` matching existing artifact conventions; schema file =
  `docs/racketsport/profile_registry_schema.json`; per-account storage;
  pipeline consumes a profile when present, falls back to generic solvers when absent (that's why
  the generic lanes still get built); `ingest_owner_capture.py` generalizes to `ingest_user_capture`.
  Owner is profile #1; a friend onboards through the identical setup steps. Gate: a court profile
  round-trips (store → match on next upload → skip re-calibration); missing-profile clip degrades to
  the generic path with the right trust band.
- [ ] **P0-10 iPhone app: BUILD the ARKit session + on-device proof + server wiring. TECH-AUDIT
  CORRECTION to this doc's earlier claim: the sidecar SCHEMA carries ARKit fields, but zero `import
  ARKit` exists anywhere in ios/ — `ARKitSetupPassSidecar` is schema-only/orphaned; only CoreMotion
  gravity is real today, and 0/20 real capture sidecars carry any pose. So (a) below is real
  engineering (ARSession/ARWorldTrackingConfiguration + plane anchors), not just wiring. The GOOD
  news: the server consumer ALREADY exists (`metric_calibration_from_sidecar_and_keypoints` consumes
  arkit_camera_pose/court_plane/intrinsics) — it's waiting on the producer.** Our app (`ios/`, 110 Swift files) already writes a
  capture sidecar carrying per-frame intrinsics, ARKit camera pose, gravity, court plane, locked
  exposure/ISO/focus/WB, LiDAR refs, and 240/120fps modes; the server ingest already reads intrinsics/
  provenance/taps. The work: (a) **physical-device proof** — record a real game, land the sidecar,
  verify fields populate (the BUILD_CHECKLIST IOS-1 row is SCOPED PASS/simulator only today; CAPABILITIES.md has no iOS row — a gap in the canonical doc); (b) **wire
  server consumption of the signals we ignore** — per-frame ARKit pose + gravity feed world-grounding
  and handheld placement (H29: gravity is exactly what GVHMR-class methods fight to estimate — we get
  it from the IMU; masked-SLAM in P2-1 becomes the fallback for stock-camera video only); per-frame
  exposure makes the H13 blur-speedometer exact (H30); LiDAR refs refine court/paddle scale (H33);
  (c) **add profile-capture flows** (H0/P0-9 setup steps: empty-court clip, ChArUco/AprilGrid sweep,
  paddle orbit, height entry, ball pick) into the existing capture UI; (d) enforce capture policy
  (EIS off, AE/AF/WB lock, landscape) which the `CapturePolicy`/`CaptureMode` modules already model.
  Gate: one real-device game recording → sidecar with populated ARKit-pose+gravity → server world
  built USING them → measurably better handheld placement vs. the ARKit-blind path on the same clip.

## PHASE 1 — Ball to (and past) the pb.vision bar

**Already built:** detector zoo (WASB+blurball+TrackNet — running VANILLA zero-shot weights + a
naive nearest-neighbor gate; the paper's HLSM training lever + online-filter are TO-BUILD, tech-audit
verified) + consensus fusion (candidate pool = real value; the voting mechanism = measured liability,
see P1-1), label-free
auto-bounce anchors, frozen arc solver + flight-sanity gate, candidate sidecars, court-map view,
viewer trail + honesty KPI, P3-A BVP solver (WIP), held-out ledger discipline. **To build:**
in-domain fine-tune that beats 0.7248, recall rescue, true 3D flight + spin, learned contact events,
in/out with gray zone.

- [ ] **P1-0 Harvest + aggregate ALL Roboflow (and public) pickleball data (licenses no longer a
  gate).** The T4 corpus was 8.6k frames from a limited pull; the measured failure modes were
  domain-gap + too-few-sources (distractor-lock). Fix the second one: scrape EVERY Roboflow
  Universe pickleball project (ball detection, court, player, paddle) + any other public set, dedup
  (perceptual hash), normalize labels to our schema, and build ONE large diverse corpus — diversity
  is the specific antidote to the distractor-lock we measured. Convert detection boxes → point
  labels; flag which frames have temporal neighbors (usable for tracking) vs. isolated stills
  (detection-pretrain only). Gate: corpus card with per-source counts, dedup rate, temporal-vs-still
  split, leakage check vs. eval clips (the T4 0/41,866 discipline). This is PRETRAINING/aux fuel for
  P1-1, NOT a standalone candidate.
- [ ] **P1-1 Fine-tune campaign v2 — owner data as the finisher, public data as the primer.** The
  corrected recipe after the T4 lesson (public-only fine-tunes degraded held-out): **pretrain/warm-
  start on the P1-0 aggregated public corpus + multi-sport auxiliary (RacketVision/TrackNet, +14-19%
  mAP evidence), then FINE-TUNE on owner-labeled in-domain data (P0-4) — never ship a public-only
  student.** Public data teaches "what a pickleball looks like across many courts"; owner data
  teaches "what it looks like on THIS camera" — the second is what moved held-out in every measured
  case. Architectures in bake-off: WASB (anchor, 1.5M params) with TOTNet visibility-weighted 4-level
  BCE + occlusion augmentation TOGETHER (aug alone hurts: RMSE 29.6→54.3) + TrackNetV4 motion-prompt
  (NOT yet re-implemented — vendored raw upstream only, tech-audit verified) + RacketVision background
  modeling. **PREREQUISITE (tech-audit):** `BallFrame.visible` is a plain bool — extend schema + CVAT
  export/labeling to the 4-level occlusion taxonomy (clear/partial/full/out-of-frame) BEFORE the WBCE
  loss is trainable as specified. **Portable starting recipe (from third_party/TOTNet/src/train.sh):**
  AdamW lr=5e-4 wd=5e-5, WBCE weights 1/2/3/3, occluded_prob=0.25, frames_in=5, 288×512, heatmap
  radius 8 train / 4 test. **FUSION CHANGE (measured):** stop using 2D spatial-consensus voting as the
  final answer (hidden-FP floor 0.349 = 5.5× worse than single WASB 0.063 — an ensemble ARTIFACT);
  keep all detectors as CANDIDATE GENERATORS (top-K sidecars already emit) and select via
  physics-consistency (the arc solver) — run as a pre-registered A/B before flipping the default. Eval discipline: eval_guard keeps Burlington/Wolverine
  never-gradient-trained; every held-out shot pre-registered. Kill criteria: any internal→held-out
  inversion = stop, analyze, don't re-tune (that's the 4-inversions rule).
  Gate (INTERIM MILESTONE, not the full BALL M1): beat 0.7248 F1@20 held-out, recall ≥ 0.70, hidden-FP
  ≤ 0.05. P1-3 closes the rest to the true BALL M1 in I.2 (F1≥0.90/recall≥0.75). Loader trap: blurball↔WASB strict-incompatible
  (36 SE keys) — always key-diff checkpoints (`strict=False` silently amputates).
- [ ] **P1-2 SST semi-supervised bootstrap.** Teacher = the PHYSICS-GATED chain output (hidden-FP
  0.021), falling back to single zero-shot WASB-tennis (0.063) when the solver self-kills — NEVER the
  raw un-gated fusion (0.349 hidden-FP would bake correlated multi-detector hallucinations into
  pseudo-labels at scale; tech-audit). Run over ALL owner
  unlabeled footage → confidence/doubt-weighted pseudo-labels (Vandeghen recipe, BSD-3,
  github.com/rvandeghen/SST) → student trains on labeled+pseudo → iterate 2×; active-learning
  loop: frames where detectors disagree go to the human label queue first. Once running, schedule it
  as a NIGHTLY FLYWHEEL (cron routine on the fleet, manual §14): harvest new clips → pseudo-label →
  candidate retrain → internal-val report, zero-attention. Gate: student beats
  teacher on internal-val WITHOUT held-out regression (pre-registered). Kill: pseudo-label recall
  inherits teacher's blind spots and doesn't move recall after round 2.
- [ ] **P1-3 Recall rescue at inference.** Independent lane, composable wins, measure each: (a)
  high-res/tiled inference near players (SAHI-style slicing on hitter-adjacent regions during rally
  frames only), (b) motion-channel input (frame differencing) per TrackNetV4 evidence, (c) lower
  detection threshold + physics-gated acceptance (the arc solver already prunes FPs: hidden-FP 0.021
  with room to trade), (d) occluded-frame recovery via the P1-1 occlusion-recipe model, (e) RIFE-
  interpolated sub-frame recovery (H15): run the detector on RIFE-interpolated frames where the real
  frame missed the ball, marked render-only/derived — NEVER counted as measured evidence. Budget:
  candidate union recall ceiling measured at 0.8793 — get fused recall from 0.578 toward ≥0.75.
  Gate: BALL M1 recall bar on held-out without hidden-FP regression.
- [ ] **P1-4 True 3D flight lift.** SEQUENCED (tech-audit: a symmetric bake-off is premature — P1-4a is itself PARTIAL with a live
  regression and P0-7 has zero code): (1) STABILIZE (P1-4a) — finish the
  P3-A BVP anchor-first solver (committed WIP — 5 baseline intervals lose `fit` status on
  reselection + internal F1 never rerun; finish steps in
  `runs/lanes/ball_p3a_bvp_anchor_first_20260705/report.json` `next`), upgraded with drag+Magnus
  from P0-7. (2) ADD MAGNUS to the existing solver: extend `_rk4_step` state with a SCALAR spin number
  S on a FIXED horizontal axis ⊥ to segment velocity (classic top/backspin; skip sidespin/3-axis for
  v1 — the free fit has only 6 unknowns on as few as 3 observations, a full spin vector is
  unidentifiable); F_magnus = Cl(S)·½ρA|v|²·n̂, Cl=0.195·S (Steyn). NOTE (tech-audit): Cd constants
  0.33/0.45 are ALREADY in `PhysicsParameters.for_ball_type` (that half is DONE); Cl/Magnus is 0%
  built. The size-depth (74mm) residual is ALSO already wired default-on (plan under-credited it);
  blur-speed + direct ARKit residuals have zero code. (3) BUILD P0-7 reusing `_rk4_step`/deriv() as
  the literal simulator core (kills sim/solver physics-mismatch bugs). (4) ONLY THEN (P1-4b) learned
  lift as a NARROW RESCUE model — applied solely to segments the anchored solver can't fit
  (status∈{fit_weak,fit_bvp_fallback,blocked_*} or observations<min) — not a parallel system: transformer
  (UpliftingTT pattern, time-aware RoPE, GPL-3.0 code reference — internal-use OK) trained on P0-7
  synthetic pairs, robust-to-gaps by construction (97.1% spin accuracy under half-FPS + 10% missing
  in TT). Camera-angle sensitivity is real (TT3D: 12.4cm side vs 29.8cm back view) — emit a
  per-segment confidence from view geometry; trust bands degrade accordingly. Gate: physically-sane
  full-flight 3D on ≥90% of trusted rally segments across 4 clips; net-clearance/court-volume checks;
  zero teleports; held-out F1 unharmed. THEN: activate paddle ball-reflection factor (P3-5) and 3D
  speed/spin stats (P6).
- [ ] **P1-5 Spin estimate (stretch).** PRECONDITION (tech-audit): H13's measured court-surface
  friction/restitution MUST exist first — spin-sign-from-bounce-kink is UNIDENTIFIABLE without it
  (the same kink fits a family of (friction, spin) pairs). 3-way sign only (top/back/flat), never
  magnitude, never user-facing without the gray zone. From the learned lift + rebound deviation
  (spin changes bounce direction — measurable from our court-plane events). Report with wide trust
  bands; never user-facing without the gray zone. Gate: sign-accuracy on labeled slow-mo owner clips.
- [ ] **P1-6 Learned bounce/contact events.** Replace heuristic cusp+gap anchors as the PRIMARY
  event source (keep as fallback): temporal classifier on (track window, audio onsets, wrist-cue
  distance) triples; train on owner captures (audio!) seeded by P0-4 contact labels; IMG_1605's 30
  real onsets = first testbed (racket P2c wanted this too). Gate: contact timing ≤ 40ms p90 vs
  reviewed events; bounce-vs-hit classification ≥ 0.9 F1 internal-val; held-out pre-registered.
- [ ] **P1-7 In/out calls with modeled uncertainty.** Combine P1-4 3D bounce point + calibration
  reprojection error + ball-radius/contact-patch ambiguity into an explicit gray zone
  (`too_close_to_call`) per `BALL_TRACKING_PIPELINE.md` §5.6. NEVER officiating-grade claims
  (research: everyone who does line-calling for real uses 4+ cameras and/or audio sensor arrays).
  Gate: reviewed-label agreement on clear calls ≥ 0.95; zero confident-wrong calls on held-out.
- [ ] **P1-8 Ball trajectory forecasting (the TrajPred we missed — recall + prediction).** A learned
  forecaster predicting the ball's near-future path from (ball history, hitter pose, 6-DOF paddle at
  contact) serves two roles: (1) **recall** — bridge frames where the detector loses the ball behind
  a player/paddle by predicting where it should be, marked derived/render-only; (2) **prediction** —
  the anticipation signal for coaching ("you were out of position for the likely return"). Use the
  RacketVision **cross-attention** design (racket/paddle pose as K/V, ball trajectory as Q — proven
  to beat naive concat, which measurably HURTS; MIT code+checkpoints) and LATTE-MV's anticipation
  framing (raises simulated return rate 49.9→59.0%). Data: owner captures + our own 3D reconstruction
  as self-labels (LATTE-MV pattern). Gate: forecast ADE beats a constant-velocity baseline on
  held-out rallies; recall-bridge additions never counted as measured (trust-band honest).

## PHASE 2 — Bodies: kill the noise at the source

**Already built:** SAM-3D-Body runtime (local + remote A100), MHR70 skeleton+mesh, person tracking
(YOLO26m+BoT-SORT+ReID), court placement, stance detection + foot-pin, stance-aware smoothing (slide
p95 8-23mm tripod), mesh index, contact-dense scheduling. **To build:** raw-noise kill at source
(latent smoothing), camera-motion/handheld robustness, far-player quality, IDF1 scoring, independent
GT + world-MPJPE gate, challenger benchmark.

- [ ] **P2-1 SMART-recipe hardening lane.** Port the validated pieces into our stack (all classical,
  no backbone change): (a) camera tracking — TECH-AUDIT: `camera_motion.py`/`estimate_camera_motion.py`
  ALREADY implement classical LK+RANSAC with person-masking (TRAM's masking insight partially landed)
  but are NOT a default stage, not RAFT, not MAD-gated → upgrade the existing module (RAFT-small +
  MAD outlier filter, 0.041° rot err @55ms/frame) and make it default, feeding placement (attacks
  IMG_1605 handheld drift; combine with (b)) ; (b) mask-the-players before camera estimation (TRAM's ablated
  insight: camera ATE 2.42m→0.32m) — we already have person masks; (c) per-track shape/scale locking
  (constant bone lengths per player per clip) — QUICK WIN (tech-audit): a correct MAD bone-length
  outlier detector ALREADY exists but feeds only trust-band metadata; wire it into the smoothing
  weights; (d) two-pass MAD-then-Gaussian smoothing where our
  one-euro chain currently stacks. Files: `threed/racketsport/worldhmr.py` (VP-A2 owns
  smoothing — coordinate!), `placement.py`, `pose_temporal.py`. Measure with
  `threed/racketsport/visual_quality.py` (+ new camera-motion metric). Gate: IMG_1605 foot-slide
  FAIL (330mm) → PASS ≤ 30mm with camera-motion compensation; no regression on the 3 static clips
  (slide p95 ≤ current 8-23mm; resets stay ≤ 2).
- [ ] **P2-2 Latent-space temporal smoothing (the big one for raw noise) — now has a published
  blueprint on our exact backbone.** Today we smooth world joints — mesh and skeleton can diverge
  (bug class we already hit). Instead smooth SAM-3D's MHR pose-code sequence (blueprint: arXiv
  2512.21573, which wraps frozen SAM-3D-Body+MHR with shape/scale locking + MHR-latent sliding-window
  optimization + differentiable soft foot-contact — Part II-B; qualitative-only evidence, prototype
  against our defects), then decode: mesh, skeleton, and all 70 joints move COHERENTLY. Use SOMA-X
  (Apache, `py-soma-x`) for MHR↔SMPL-X so SMPL-X-space priors (SmoothNet/DPoser-X) compose here.
  TECH-AUDIT (verified): the emission prerequisite is ALREADY DONE — `global_orient`/`body_pose`/
  `betas` are emitted and schema'd end-to-end. The REAL gap is the decode-back (FK/skinning) path,
  which exists only inside vendored `third_party/Fast-SAM-3D-Body/.../mhr_head.py::MHRHead` — expose/
  wrap it for local re-decode after latent smoothing. Evaluate AGAINST the current classical chain with
  visual_quality.py + skeleton-vs-mesh divergence + paddle-stability downstream metric (paddle lane
  measured skeleton noise as ITS binding constraint — this is the fix). Gate: raw-equivalent joint
  noise 2-8cm/frame → ≤ 1cm effective without lag artifacts (wrist swing-peak timing must stay
  0-frame-delta — regression harness exists); mesh-skeleton divergence ≤ 5mm p95. Kill: if latent
  smoothing over-smooths fast swings (SMART's failure mode) — fall back to hybrid (latent for
  torso/legs, classical wrist protection).
- [ ] **P2-3 Far/small-player quality.** TECH-AUDIT: the body model's input is hard-capped at
  384/448/512px regardless of source resolution — so "run the whole frame at higher res" is NOT a
  distinct lever; the crop-based path below is the right family. High-res crop re-inference pass for players whose bbox
  height < threshold (research: MPJPE 33→43mm large→small, cause = input resolution): upscale crops
  (optionally light SR) → SAM-3D re-run on far players only (contact-dense scheduling already
  prioritizes the hitter; extend the same budget logic). Gate: far-player jitter RMS within 1.5× of
  near players (today: "far players worst" qualitative + 41-78mm/f² band); wall-time increase ≤ 60s/clip.
- [ ] **P2-4 SAM-Body4D masklet-prompt experiment (cheap, time-boxed).** Training-free: SAM3 video
  masklets as temporally-consistent prompts instead of per-frame boxes (MIT,
  github.com/gaomingqi/sam-body4d — zero published numbers, so WE benchmark it). Gate to adopt: ≥20%
  raw-noise reduction on a 2-clip A/B at acceptable cost. Kill: no measurable gain or >90s/clip.
- [ ] **P2-5 Identity/tracking upgrades (TRK gate).** Wire IDF1 scoring (the scorer EXISTS + works in `person_track_gt_scoring.py`/`mobile_person_eval.py`,
  e.g. Burlington IDF1 0.9112 recorded — but `process_video.py` hardcodes `idf1=None` at 3
  `derive_track_trust_band` call sites (~lines 896/910/983); wire the existing scorer to them); root-cause the
  Outdoor single-frame 1.52m teleport (t=13.55s, fail-closed already catches it); spectator/adjacent-
  court FP hardening (IMG_1605 case is solved by membership exclusion — make it a general court-
  membership prior). Watch-list: RAM (CVPR'26; 15 ID-switches vs 344-349 — adopt IF code releases);
  CoMotion needs Apple-license review + has close-proximity collapse (= doubles) — bench only.
  Gate: TRK gate on all eval clips: IDF1 ≥ 0.85, 0 switches, 0 spectator-FP, coverage ≥ 0.95.
- [ ] **P2-6 Independent BODY GT + world-MPJPE gate.** The gate exists; independent GT doesn't.
  Plan (harsh-review: GT must be INDEPENDENT of the calibration BODY is scored against — derive ≥a
  subset of GT points from an independently surveyed/tape-measured court frame, not the pipeline's own
  solved calibration): (a) cheap: measured court-landmark protocol — owner performs scripted standing/touching
  poses at surveyed court points (corners, kitchen line) on video → world-position GT for ankles at
  known frames; (b) better: one session with a second camera (training-time-only rig; REQUIRES a
  stated time-sync method (audio-clap/LED-flash) + extrinsic calibration (ChArUco) before it's a real
  deliverable) (training-time-only rig is allowed;
  product stays single-cam) for triangulated joint GT on ≥ 20 samples. Gate: world-MPJPE ≤ 0.05m
  threshold pass per `pickleball-manager-setup` gate; per-player far/near breakdown reported.
- [ ] **P2-7a [EARLY, decoupled — tech-audit: don't trap cheap narrow tests inside the all-or-nothing
  backbone gate] GVHMR gravity-view spike.** VERIFIED at source: `get_R_c2gv(R_w2c, axis_gravity_in_w)`
  cleanly accepts EXTERNAL camera pose + gravity (no DPVO needed for that path) — and on TRIPOD clips
  both are already known WITHOUT ARKit (constant R from court calibration; gravity = court-plane
  normal), so this spike can run TODAY on eval clips. Caveat (verified): GVHMR is single-person-
  hardcoded (`get_one_track`; a rumored demo_multi.py does NOT exist) → 4 players = 4 crop-track runs;
  measure that cost honestly. Gate: single-player world-metrics vs our stack on 2 clips; informs P2-7.
- [ ] **P2-7b [EARLY, decoupled] Far-player full-image-conditioning probe.** PromptHMR-Vid/BLADE-style
  full-image context vs our crop pipeline, specifically for the far pair. Gate: far-player jitter/
  accuracy delta on 2 clips; informs both P2-3 and P2-7.
- [ ] **P2-7 Challenger benchmark (the ruling re-opener, AFTER P2-1/2/3 land).** GVHMR +
  PromptHMR-Vid (+ optionally WHAM) on our 4 eval clips + owner handheld clip, scored with OUR
  metrics (visual_quality.py + P2-6 GT + placement gates). Licenses are a non-constraint now (owner
  ruling) — SMPL/NC just get a ledger line. **Add the pass-2 finds to the bracket: Human3R** (ICLR'26,
  feed-forward multi-person SMPL-X + scene + camera in ONE 15-FPS pass — could collapse stages;
  gravity/camera-conditioned variants can CONSUME our ARKit extrinsics), **DuoMo** when code releases
  (−16% EMDB/−30% RICH, biggest reported margin), and **JOSH3R** (feed-forward, 15.4 FPS, TRAM-based,
  EMDB W-MPJPE 174.7mm). Decision rule (pre-registered): challenger must beat the hardened
  SAM-3D stack on ≥3 of 4: world-MPJPE, jitter, foot-slide, far-player accuracy — by ≥20% each —
  AND integrate multi-person + mesh-skeleton coherence. Otherwise the ruling stands and we bank the
  benchmark as evidence. Kill: benchmark-only lane; NO pipeline integration before the decision rule
  passes.
- [ ] **P2-8 Wire-or-remove `grounding_refine`.** Dead default stage (consumes
  `foot_contact_phases.json` that nothing produces; `run_physics_footlock.py` is the standalone
  producer). Either produce phases in-pipeline (post P2-1 stance work) or delete the stage.
  Gate: wiring truth table updated; no always-skipping default stages remain.

## PHASE 3 — Paddle: high-definition 6-DOF

**Already built:** fused 6-DOF estimator (`paddle_pose_fused`, render-only, IoU 0.24-0.34, jitter
~5°/f), YOLO26s box + idle seg checkpoint, ball-reflection factor (built, dormant), same viewer
contract as the proxy. **To build:** wire-by-default per-clip inference, seg→IPPE planar PnP, WiLoR
hands, 5-keypoint detector, impact-factor activation, hi-def scanned asset + marker GT.

- [ ] **P3-1 Wire the fused estimator into the default pipeline.** `build_paddle_pose_fused.py`
  currently manual-only (BUILT-NOT-WIRED; `process_video.py:2681` only reads an existing
  `racket_pose_estimate.json`). Add per-clip YOLO26s paddle-box inference (predictions currently
  exist only for the 3 CVAT clips) + the `_paddle_estimate_trust_band` wording patch + ESTIMATED
  band enforced. Gate: fresh E2E on any clip emits paddle track by default, fail-closed when
  evidence is absent; suite green.
- [ ] **P3-2 [DEFERRED-PENDING-GT-GAP — tech-audit resequence] P2a wrist-gated masks → silhouette factor.** The seg checkpoint exists
  (`runs/rkt_train_20260702T072800Z/seg_yolo_external_split/`) and is IDLE. Run it wrist-gated,
  add mask-silhouette residual to the fusion. THEN the white-space play: mask → oriented
  quadrilateral corners → IPPE planar PnP (BSD-3; 4-point coplanar; 50-80× faster) as a direct
  6-DOF factor with per-frame quality gating (expect masks to fail near impact — blur; the gate
  must know). Gate: Wolverine IoU 0.236 → ≥ 0.35 AND Burlington 0.342 → ≥ 0.45 internal-val, jitter
  ≤ 5°/f held, no new teleports. Kill: mask Dice < 0.6 on contact-window frames after one tuning
  round → keep silhouette factor only where masks are clean.
- [ ] **P3-3 [DO SECOND, right after P3-1 — tech-audit: this targets the MEASURED defect (palm-only
  IoU 0.065 rest-pose bias); boxes supply ~100% of current gain] P2b WiLoR hand crops.** A100; CC-BY-NC-ND (internal OK — ledger it). Replaces
  rest-pose-biased MHR fingers as palm-frame source (measured: palm-only IoU 0.065 — the weak
  link). Blend: WiLoR palm frame where crop confidence high, else MHR. Gate: absolute pronation
  improves on CVAT-box orientation proxy; downstream fused IoU +≥0.03; jitter not worse. Note: the
  constant-grip-transform assumption stays — measure per-segment grip re-fit as part of this lane.
- [ ] **P3-4 [DEFERRED-PENDING-GT-GAP; pixel-math gate: only favorable <6-8m on sharp frames — worst
  exactly at impact; eval RacketVision's public RTMPose-M checkpoint at ZERO training cost first]
  Paddle 5-keypoint detector (owner data).** RacketVision schema (Top/Bottom/Handle/
  Left/Right); **start from RacketVision's released RTMPose-M racket-keypoint checkpoint** (PCK@0.2
  81.8-89.6, handle 92.6-97.9 — MIT, public) rather than from scratch, then fine-tune on P0-4 labels
  (low-thousands of frames; their side/edge-keypoint weakness [64.8-80.1% PCK] tells us to oversample
  face-on + blurred frames). **Bootstrap labels cheaply**: Grounding DINO zero-shot "pickleball
  paddle" boxes + OnePoseViaGen (single paddle photo → generative texture/view-randomized training
  set, CoRL'25 code) + our H7 face-texture auto-labels. Feed keypoints as PnP constraints into the
  fusion (keypoint→corner correspondence with the known paddle rectangle). Cross-attention (ball as
  query) if we later joint-train — naive concat hurts (RacketVision ablation). Gate: face-normal error
  vs CVAT-derived orientation proxies improves ≥25%; center error ≤ 10px.
- [ ] **P3-4b [DEFERRED-PENDING-GT-GAP] Face-texture homography anchors (EDGE H7 — the self-labeling pose factor).** On sharp
  wrist-crop frames, feature-match the scanned paddle face graphic (SuperPoint+LightGlue) → planar
  homography → direct 6-DOF pose. Use as (a) gold anchors that continuously re-fit the per-segment
  grip transform, (b) free auto-labels for P3-4 keypoint training. Gate: anchor-frame pose agrees
  with marker GT ≤10° where both exist; grip-transform drift measurably reduced between anchors.
- [ ] **P3-5 [FAST-TRACK — tech-audit: the strongest orientation evidence at exactly the frames
  coaches care about; cheaper than any new vision signal. Unlock = P2c IMG_1605 GPU ball track (30
  real audio onsets), shared with P1] Ball-reflection factor activation.** Already built + tested, DORMANT pending P1-4 3D
  velocities. TT4D existence proof: 26.4°±4.4 orientation from trajectory inversion alone — treat as
  target band at impact frames. Fuse as impact-window prior (highest-value frames). Gate: impact-
  frame face-angle error vs owner 4-marker GT ≤ 30° p90 (first pass), tightening with data.
- [ ] **P3-6 [DEFERRED — last, only if marker GT shows a residual impact-frame gap] Sparse
  differentiable-render refinement (optional, after P3-2).** nvdiffrast render-
  and-compare at impact keyframes only (Diff-DOPE evidence: 3.5s/frame full-config — keyframes only;
  resolution-insensitive so cheapen aggressively). Uses the P3-7 textured asset. Gate: measurable
  face-angle improvement at impacts on GT; ≤ 20s/clip added. Kill if masks/keypoints already
  saturate GT accuracy.
- [ ] **P3-7 Hi-def paddle asset + owner GT capture.** From owner's paddle: photo set → textured
  mesh (photogrammetry or manual CAD + texture bake), correct dimensions; the SAME session films
  4-corner-marker GT clips (I.5 #2) → the promotion dataset for RKT. Viewer: LOD-ed glTF, brand
  face optional per-user later. Gate: asset renders in viewer at full detail ≥ 60fps; RKT scoring
  harness consumes marker GT.
- [ ] **P3-8 (Stretch/novel) blur-axis orientation spike.** PCA on paddle blur streaks as an
  orientation cue for the fastest frames (nobody has published this for elongated implements —
  BlurBall proves the primitive for balls). Time-boxed 1 lane; adopt only on measured gain at
  impact frames.

## PHASE 4 — Court auto-find + placement everywhere

**Already built:** manual + metric-15pt calibration (work today), Wave A on a branch (guess+confirm
UI with trust hole closed, geometric solver Outdoor 4.4px no-tap, `court_unet_v2` 24M trainer staged,
synthetic generator v2 w/ tennis overlays), external checkpoints local (PnLCalib/TennisCourtDetector/
DeepLSD/ScaleLSD). **To build:** land Wave A on main, train the model, GEO r3 (adjacent-court vote),
distortion-aware handheld calibration, court-profile library (P4-0), downstream impact gate.

- [ ] **P4-0 [START HERE — 0% built today (no CourtProfile module exists; `camera_fingerprint()` is a
  device key only, tech-audit verified). THE v1 court path.] Court-profile library + color +
  intrinsics-at-ingest (EDGE_PLAYBOOK H1/H2/H3; do BEFORE
  Wave B matters for owner clips).** Per registered court: store one precise calibration + line-paint
  color + lens intrinsics/distortion (from the H3 ChArUco sweep or the app sidecar); on upload,
  re-identify the court (image-retrieval embedding + color histogram, trivial at N≤3 courts; reuse the
  existing `camera_fingerprint()` in `owner_capture_intake.py` as the device key) and reuse the frozen
  calibration, verified by a cheap 4-line reprojection check. Generic no-tap solving (P4-2/3) then only
  runs for NEW/unknown courts. Gate: a court profile round-trips (store → match on next upload → skip
  re-calibration with reproj ≤ the manual bar); missing-profile clip degrades to the generic path.
- [ ] **P4-1 Land Wave A on main.** ⚠️ NOTE (harsh-review 2026-07-06): the handoff patch does NOT apply
  as-is — `git apply --check` fails on 4 bogus self-referential symlink hunks (eval_clips/eval_clips
  etc.), a BUILD_CHECKLIST.md conflict, and a web/replay/package.json typecheck-script conflict; its
  baseline is not an ancestor of current main. FIRST regenerate the patch (rebase
  `worktree-court-autofind-20260705` onto main + `git format-patch`, strip the symlink hunks), OR
  cherry-pick with explicit conflict resolution (stash BUILD_CHECKLIST first, resolve package.json).
  Then apply
  `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch` (42 files, baseline
  501a1114; drop BUILD_CHECKLIST hunks on conflict) or cherry-pick branch
  `worktree-court-autofind-20260705`. Gate: 31 pytest + 156 vitest from the lane green on main;
  upload guess+confirm UI keeps the trust hole closed (unconfirmed guesses NEVER ride the TRUSTED
  channel).
- [ ] **P4-2 [UNKNOWN-COURT EPIC — NOT in DONE-v1 (tech-audit: real clips calibrate via manual/
  sidecar precedence; the auto-find solver only runs behind an explicit preview flag; v1 = P4-0
  profiles + P4-4 distortion + measured net heights)] Train court_unet_v2 (STAGED — one command).**
  `bash scripts/gpu-train-lock.sh bash runs/lanes/cal_model_20260705/train_a100.sh` (script now present
  at that path; NOTE `runs/` is gitignored → it is a LOCAL untracked artifact and won't survive a fresh
  clone — promote to `scripts/` or re-stage from the court lane on a clean machine).
  (24M params, kp+line heatmaps @640×360 — the architecture that replaces the 3 killed 160×90
  attempts). Then eval harness + E4 fusion into the geometric solver
  (`court_model_infer.infer_court_model` contract is ready). Gate: aggregate ≤ 200px hard bar
  (currently 213.3); IMG_1605 tennis-overlay case solved by the neural channel.
- [ ] **P4-3 [UNKNOWN-COURT EPIC — NOT in DONE-v1; the one real fix here is the top-3 cross-frame
  court-identity vote (the 19.8px number is THIS path's discrete lock-on bug)] GEO r3.** Temporal-median fallback trigger (predeclare this time) + top-3 cross-frame
  vote for the adjacent-identical-court lock-on (Burlington/Wolverine failure). PnLCalib SV_kp/
  SV_lines weights already local (`models/checkpoints/court_external/` — GPL v2, internal OK,
  ledger it). Gate: all 5 samples ≤ 200px; Outdoor stays ≤ 5px; no Indoor regression past 93px.
- [ ] **P4-4 [TECH-AUDIT: the SINGLE highest-leverage v1 accuracy fix. Two SEPARATE error budgets —
  the oft-cited 19.8px p95 is the GEO/auto-find path (a discrete adjacent-court lock-on bug, v1-
  irrelevant); the owner/metric15 v1 path already runs 12.3px p95 / 4.8px median, and ITS floor is
  IMG_1605-class edge-of-frame distortion (~53px). Fix = ChArUco k1/k2 per lens NOW (decoupled from
  ARKit timeline) + VERIFY `back_project_pixel_to_floor` actually applies `intrinsics.dist` (audit
  suggests it may not).] Distortion-aware calibration for owner captures.** IMG_1605's 330mm foot-slide FAIL is
  edge-of-frame lens distortion (zero-distortion 15pt model breaks at x̄≈53px). Add k1/k2 estimation
  to the metric15 fit (or per-camera profile from the owner-capture association profile); handheld
  composes with P2-1 camera tracking. Gate: IMG_1605 placement residual at frame edges ≤ 2× center;
  foot-pin cap_exceeded_skips → 0.
- [ ] **P4-5 Downstream impact harness + no-tap promotion path.** Every CAL change scored not just
  in px but in downstream foot-slide/placement/ball-3D deltas (the synergy audit's
  calibration-noise-floor finding: reproj p95 ≈ 19.8px ≈ the F1@20 radius — pushing CAL down lifts
  BALL headroom). Held-out PCK@5px gate on owner-reviewed viewpoints for promotion; fail-closed
  mandatory-review below bar. Gate: CAL promotion row in ledger or documented miss.
- [ ] **P4-6 NET geometry. TECH-AUDIT CORRECTION: `net_plane.py` ALREADY does linear post-to-center
  sag interpolation (36/34in regulation constants) — NOT a single flat plane. The v1 gap is (a)
  heights are regulation DEFAULTS not per-court tape measurements → **P4-6.0 (v1, ~1-line data
  change): add tape-measured `net_post_height_in`/`net_center_height_in` to the H0 court profile and
  override the template at `net_plane_from_template` call sites** — and (b) zero pixel-level net
  verification. The full seg+catenary build below is the v2 path — novel (nobody does it in any
  sport) but NOT v1-blocking. STAGED
  (don't bundle into one checkbox): P4-6a net-cord/post detection bootstrap with its OWN segmentation-
  quality gate (fallback if seg fails: keep the current single net_plane + trust-band, don't block the
  pipeline); P4-6b catenary fit validated on synthetic/known geometry first; P4-6c real-clip
  integration + tape-measure GT (the ≤2cm bar is the FINAL gate, not the only one).** Today
  `net_plane.json` is a single plane. Build the real net: 2 posts, top cord (36in ends / 34in center),
  center strap, sag curve. Steps (Part II-C): (a) a net-cord segmentation model (the hard part —
  thin, low-contrast, often player-occluded; bootstrap with Grounding DINO "net cord/tape" + owner
  labels); (b) **3D catenary curve fit** (Madaan ICRA'19, 5-param planar catenary + rigid transform,
  distance-transform reprojection loss) using ARKit per-frame camera poses (we have them) — degrades
  gracefully under partial/occluded detection; (c) publish the net as both a calibration anchor and a
  3D occluder consumed by ball 3D (P1-4) + placement (Phase F). Gate: net top-cord height error ≤ 2cm
  vs. tape-measured GT at posts+center on an owner clip; net occlusion improves ball-track continuity
  across the net. Owner GT: tape-measure net heights once per court (into the H0 court profile).

## PHASE F — GLOBAL FUSION: one mutually-consistent metric 3D world (the combine-everything pillar)

**Why this phase exists (Part II-C):** every subsystem above is currently solved INDEPENDENTLY and
composited, so errors compound and nothing enforces cross-system physical consistency (ball meets
paddle at impact, foot meets ground, no interpenetration, one shared camera trajectory). This is the
owner's pillar #6 and the capstone — the step that turns "several good stages" into "one accurate 3D
world." The published SOTA pattern is **contact-coupled joint optimization** (JOSH), and **no one has
done it for ball+body+paddle+court together** — so this is our novel frontier. Runs AFTER P1-P4 land
enough to initialize from; it is a REFINEMENT layer, never a from-scratch solve.

**Already built:** per-subsystem outputs that become the initialization (court homography, SAM-3D
skeleton/mesh, ball arc, paddle estimate, stance foot-pin, ARKit camera pose/gravity from P0-10).
**To build:** the joint optimizer itself + its contact/consistency residuals + world-level temporal
smoothing.

- [ ] **PF-1 Cheap consistency priors first (stopgap before the full optimizer).** Before building a
  joint optimizer, add the two highest-value cross-system residuals as post-hoc corrections and
  measure them — implement in the `foot_pin.py` HOUSE PATTERN (bounded max-correction caps +
  confidence-gated application, the repo's established can-only-nudge idiom): (a) **ball↔paddle
  impact** — at each P1-6 contact window ABOVE a fused-confidence threshold (contact detection is
  heuristic today — a spurious window would create false 'consistency'), snap ball 3D and paddle face
  toward coincidence, each moved ≤ half a ball radius (~37mm), tapered ±2 frames; (b)
  **foot↔ground + non-penetration** — clamp feet to the calibrated court plane at stance (extends
  foot-pin) and forbid mesh-below-floor. Gate: measurable reduction in ball-paddle gap at impacts +
  floor penetration, zero regression to standalone metrics. Confidence: high (cheap, decomposable).
- [ ] **PF-2 Contact-coupled joint optimizer (JOSH-pattern, our metric priors + ARKit init).** One
  offline optimization — SOLVER RULED (tech-audit): torch-autograd Levenberg-Marquardt or
  scipy.optimize.least_squares(method='trf', loss='huber') with block-sparse Jacobians — the house
  idiom (every solver in this repo is scipy least_squares; ceres/GTSAM are uninstalled, unused, and
  not even what JOSH uses; differentiable render fits none of the residuals). Optimize over {camera trajectory
  (seeded + locked-ish from ARKit when the P0-10 sidecar is present+valid; ELSE seeded from P2-1
  RAFT+MAD for stock-camera/capture-failure clips — and degrade the WHOLE-world trust band on the
  fallback path, not just occluded frames), each player's root+pose, ball arc segments, paddle 6-DoF} with
  residuals (weights = the pipeline's EXISTING per-subsystem confidence fields — event_fusion,
  SE3PoseConfidence, ball-chain trust — zero new modeling): 2D reprojection of every subsystem +
  human-scene contact (JOSH; NOTE the 314→149mm number is a WITHIN-optimizer loss ablation, cite as
  directional only) + ball↔paddle impact +
  ball↔ground bounce + **net-occlusion/height consistency** (ball/player behind the P4-6 net cord must
  respect its fitted height; occluded frames trust-banded, not fabricated) + **hand↔paddle grip**
  (paddle handle bounded to a slowly-varying offset from the gripping hand's palm frame) +
  non-penetration + known-scale anchors (court 20×44ft, net 36/34in, ball 74mm,
  per-player heights from H4). Whole-clip (offline) — JOSH shows whole-clip beats chunked by ~12%
  (TT4D). Per-player variable: reuse P2-2's latent MHR pose-code through the frozen decoder (anatomical
  validity by construction, drastically fewer DOF). De-risk path: contact events are SPARSE → a
  coordinate-descent/alternating-block approximation (re-run each subsystem's own solver with
  cross-terms as soft constraints) is a legitimate cheaper PF-2 v0. Kill/guard: JOSH's own limitation
  is it needs VISIBLE contact — fail-closed to PF-1 +
  trust-band the frames where contact is occluded (don't fabricate consistency). Gate: world-MPJPE +
  foot-float-rate + ball-paddle-impact-gap all improve vs the PF-1 baseline on ≥2 clips; nothing
  regresses; runtime within the offline budget (JOSH full opt ≈ 0.8 FPS is acceptable on A100).
- [ ] **PF-3 World-level temporal consistency (replace per-stage smoothing at the seams).** A final
  global trajectory optimization over the whole clip that jointly respects court + ball + players +
  contacts, instead of each stage smoothing in isolation (which is where seams/lag appear). Reuses
  the P2-2 latent-smoothing + PF-2 residuals across time. Gate: no cross-stage seam artifacts in the
  viewer; jitter/slide at least as good as per-stage, measured by visual_quality.py + a new
  world-consistency metric.
- [ ] **PF-4 The final deliverable: the accurate 3D world + its honesty.** Assemble PF-1..3 into the
  single trust-banded 3D world bundle (court+net, 4 meshes, ball 3D flight, both paddles, contacts),
  browser-verified, every element consistency-checked and banded. This is the "state-of-the-art
  extremely accurate 3D mesh of the video" end state that feeds Phase 6 coaching. Gate: a fresh
  owner clip → one consistent world where ball visibly meets paddle at every detected contact, feet
  are planted, and every unmeasured element is honestly banded; `verify_process_video_viewer.py`
  extended to assert cross-system consistency.

## PHASE 5 — Speed + cost (realistic gate: ≤ evidenced floor; ≤1× is an un-booked STRETCH)

**Already built:** 2141→~532-565s (3.8×, zero quality change), slim BODY monoliths, VM-built mesh
index, batched rsync, chunkfix, worldhmr-split fast path, overlap-schedule opt-in, full phase
instrumentation. **REALITY (harsh-review 2026-07-06):** the four booked levers floor at ≈6-8 min/clip
(≈300-400s BODY, which is 96-98% of E2E) — that is NOT ≤2×/≤1× of a short clip. So: the phase GATE is
the **evidenced floor** (P5-1); **≤2× video duration is the real target for full-length owner games**
(a 12-min game → ≤24min); **≤1× is an un-booked STRETCH** needing the BODY-runtime redesign in P5-7.
**To build:** booked levers (mmap handoff, gates-from-arrays, dispatch auto-clean), TensorRT engines,
per-court cache, cost metering, pre-flight QA, and the P5-7 BODY redesign.

- [ ] **P5-1 Land the remaining booked levers** (all measured, from
  `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md` + chunkfix/payload lanes): shared-memory/mmap
  subprocess handoff — TECH-AUDIT CORRECTION: the one live attempt (S4 chunked binary transport) was
  a measured REGRESSION (1057.4→1300.7s; handoff 376→489s) and was reverted; only round-trip
  correctness is proven, the <40-60s target is NOT banked. Next attempt = large flat arrays, not
  chunked streaming (S4's own root-cause). Plus gates-from-arrays everywhere (P3+P5 of that plan),
  dispatch-dir auto-clean (A100 disk hit 100% once — REQUIRED before unattended runs), P7 freshness.
  Gate: Wolverine ≤ 400s; Outdoor ≤ 2× its video duration; six-run variance report; foot-slide
  bit-identical vs. pre-change.
- [ ] **P5-2 Overlap scheduling by default for owner captures.** `--body-schedule=overlap` is
  landed opt-in, byte-identical serial default; rally gating actually pays off on real captures
  (dead time — eval clips have none). Flip default for owner-profile clips after a 3-clip
  correctness A/B. Gate: byte-identical worlds; wall reduction measured.
- [ ] **P5-3 Detector engine optimization — SCOPE (tech-audit): WASB (1.5M CNN) + YOLO26 ONLY via
  the official Ultralytics TensorRT/FP16 path (2-5× realistic). Do NOT convert SAM-3D's ViT/MHR
  (highest risk, and detectors are <3% of wall — this lever does NOT touch the 6-8min floor; don't
  conflate). Batch rally-window frames.**
  Gate: ball+track stage wall −50% at identical F1 (bitwise-identical not required — score-identical
  on internal-val is).
- [ ] **P5-4 FULLY-LOADED cost metering (not just GPU).** A per-clip $ tracker summing: GPU-seconds ×
  spot price, **storage $/GB-month of retained video+profiles, per-session LLM coaching API cost
  (P6-4), amortized human-review labor from P5-6 QA flags, and idle/orchestration VM time** → one
  fully-loaded $/clip figure that P7-3 pricing MUST use (not the BODY-only $0.12 slice). Also: GPU-
  seconds × spot price + egress emitted into
  PIPELINE_SUMMARY + dashboard; spot-preemption resilience (auto-resume dispatch — VM restart mid-
  verify already bit us once; NEW IP handling per `configs/ssh/a100_known_hosts` lesson). Gate:
  cost line on every run; simulated preemption recovers without human help.
- [ ] **P5-5 (Later, product-gated) iOS live-tier distillation.** CoreML ball heatmap distillation
  (task #8, owner-capture-gated) + capture guidance. Only after P1 lands (distill the GOOD teacher).
- [ ] **P5-5b Pre-flight sanity gate (cheap, BEFORE any GPU stage — harsh-review gap).** At pipeline
  entry, before dispatch: ffprobe/opencv integrity (0-duration/corrupt), a REAL orientation/rotation
  check (today it's a hardcoded `landscape` stub in 2 places), and a 1-few-frame court-presence + sport
  check. A clip failing any short-circuits with a clear user message and burns ZERO GPU. Gate:
  injected bad clips (portrait, wrong-sport, corrupt, no-court) rejected pre-GPU with the right message.
- [ ] **P5-6 Per-clip automated QA / failure detection (the reliability primitive we lacked).**
  Before a processed world reaches the user, auto-detect a bad output: wrap each tracker/stage in
  **sequential-hypothesis-testing failure detection** (arXiv:2602.12983 — anytime-valid, provable
  false-alarm bound via Ville's inequality, no retraining, negligible compute) + cross-check against
  the Phase-F consistency residuals + trust-band coverage. A clip failing QA is flagged for
  reprocessing/human review, never silently shown. Gate: injected-failure clips are caught at a
  documented true-positive rate with bounded false alarms; ties into P7-5 promotion ladder.

- [ ] **P5-7 BODY-runtime redesign (the unbooked ≤1× stretch — this task is what could unlock it).**
  Booked levers floor at ~300-400s BODY; ≤1× video duration needs an order-of-magnitude BODY change.
  HONEST CEILING (tech-audit): warm-worker-addressable fixed cost = model_load 23.3s + compile_warmup
  42.3s = 65.6s/clip; queue-amortized best case 65.6·(N−1)/N (~59s/clip at N=10) — 15-20% of the
  post-P5-1 floor, NOT an order of magnitude. So the warm pool MUST pair with **batched multi-clip
  inference** (the bigger lever) + cheap first: **disk-cache the compiled graph keyed by model
  version + input shape (~42s/clip back, near-zero risk)**. Also evaluate Triton-style persistent
  serving vs extending remote_body_dispatch (currently strictly one-shot-per-clip, flock lease, no
  queue). Other candidates: frame-plan sparsification beyond ball_aware. Gate: BODY wall ≤ video duration on a
  full owner game, gates green. Kill: if two candidates fail to beat the booked floor ≥2×, accept ≤2×
  as the product SLA and close the stretch.
- [ ] **P5-8 Per-court warm caches + cascade inference (EDGE H17).** Cache per-court/per-session
  immutables (H1 profile, H6 background, H4 ReID galleries, TensorRT engines) so a clip's marginal
  work is only its rallies; tiny detector every frame, full ensemble only on uncertainty/rally
  windows. Gate: owner-clip wall −25% at identical internal-val scores.

## PHASE 6 — Coaching + product output (the end goal)

**Already built:** 7 position-based rally metrics + coaching-facts JSON (scaffold), confidence
framework with trust bands, court-map view, viewer moment-jump surface. **To build:** shot
classification, the stat layer users value, the pickleball reference-range library (the moat),
the grounded-LLM coach, direct visual feedback overlays, session reports.

- [ ] **P6-1 Shot classification.** Serve/return/drive/drop/dink/lob/volley/smash from (ball 3D arc,
  contact events, hitter pose features, court zones). Start rule-based on P1 outputs (interpretable,
  trust-banded), add a learned classifier when owner-label volume allows (Talking-Tennis used
  EfficientNet-B0+LSTM at 79% on 12 tennis classes — our 3D features should beat video-only).
  Gate: ≥ 0.85 agreement with owner-labeled shot types on held-out; every classification carries
  confidence.
- [ ] **P6-2 The stat layer users actually value (research-ranked).** Minimal set FIRST: unforced
  errors (needs shot outcome + rally end cause), third-shot success (needs P6-1 + landing zones),
  dink-rally win rate; then serve/return depth, kitchen-arrival time, court-coverage maps (court-map
  view already landed in the ball-render lane `ball_p4_render_fix_20260706` — extend), shot-speed (needs P1-4; report bands, never
  single-number over-precision — pb.vision's 60-vs-27mph blunder is the cautionary tale). Gate:
  every stat has a trust band + a "how we measured" popover; rally-metrics facts JSON feeds it.
- [ ] **P6-3 Pickleball reference-range library (the moat).** Per skill band (3.0/3.5/4.0/4.5+):
  third-shot-drop success and apex/landing bands, serve depth distribution, kitchen-arrival timing,
  ready-position paddle height, contact-point-vs-body position, dink apex over net. Sources: trade
  benchmarks as seeds (drill 50%@3.0 → 85-90%@4.0), owner/coach review, then OUR OWN accumulated
  user data. Stored as versioned JSON contracts (`docs/racketsport/` schema). Gate: coach sign-off
  on v1 ranges; every range carries provenance.
- [ ] **P6-4 Grounded coaching engine (3-stage, causally-validated architecture).** (1) feature
  extractor over our 3D world (kinematics, positions, events — deterministic, tested); (2) rule
  comparator vs P6-3 ranges → structured findings with severity + evidence pointers (frame ranges,
  3D moments); (3) format-locked LLM (Claude; score + exactly-N corrections + drill suggestion),
  which NEVER sees raw numbers without the comparator's verdicts, NEVER invents metrics; audited for
  0-fabrication on a 300-output sample (Talking Tennis achieved 100% — that's the bar). Euler-angle
  phrasing for human-readable joint feedback. **Define the acceptance protocol (harsh-review):** reviewer
  = owner + ≥1 rated (4.0+) player (name the role + count); rubric = Talking-Tennis's published
  questions/scale as the template; fabrication audit = a fixed random sample of 300 outputs with a
  stated adjudication rule. Gate: coach rubric ≥ 8/10 on usefulness; fabrication
  audit 0/300; every claim traces to a trust-banded artifact.
- [ ] **P6-5 Direct visual feedback in the viewer.** For each coaching finding: jump-to-moment, draw
  the evidence (target zone vs actual landing, ghost ideal-contact-point, kitchen-line distance
  ribbon, paddle-face angle at impact), before/after comparison scrubbing. Reuses trust-band + court-
  map + mesh layers (all landed). Gate: browser-verified demo of ≥5 finding types on an owner game;
  verify_process_video_viewer.py extended to assert honesty of coaching overlays.
- [ ] **P6-6 Session report + progress tracking.** Per-game report (headline wins/errors, 3
  priorities, drills) + cross-session trends (needs stable player identity per account). Gate: owner
  dogfood approval on their own games.

## PHASE 7 — Productization (brief; expand when P6 demos)

**Already built:** the iOS app shell + capture/upload/replay modules (`ios/`, 110 Swift files),
upload manifest + render-gateway client, `server/` + `render.yaml` seeds, camera-roll import.
**To build:** device-proven capture, accounts + per-user library, the H0 onboarding wizard, pricing,
legal review, the VERIFIED promotion ladder.

- [ ] **P7-1** Upload service + accounts + the H0 onboarding wizard. The current server is single-
  process with an in-process JobStore, uploads in ephemeral `/tmp`, and ZERO client-facing auth
  (harsh-review). Explicit sub-tasks: (a) durable object storage replacing `/tmp`; (b) a durable job
  queue replacing in-process BackgroundTasks/JobStore (survives restart/preemption); (c) client-facing
  auth/authz; (d) per-user clip library + CLI-populated-then-wizard profile registry (P0-9). **P7-2** iOS capture guidance E2E on device (framing/stability coach at record time —
  our tolerance is a differentiator, but guided capture still maximizes quality; modules exist under
  `ios/`). **P7-3** Pricing vs pb.vision anchors ($19.99/mo 100min; our GPU cost ≈$0.12/clip BODY
  today — margin is fine); processing-SLA: match/beat pb.vision's ~30min via P5's ≤2× target (~24min on a 12-min game), ≤1×
  (~12min) stretch; a sub-10min/game SLA is a separate un-booked target, not asserted here. **P7-4b DATA-PRIVACY + retention (BLOCKING before the first NON-OWNER footage is processed — NOT
  gated on commercial launch).** (a) inventory every per-person artifact captured/retained (video,
  ReID galleries, frozen shape betas — H4); (b) legal-basis check vs BIPA/CCPA/GDPR biometric-category
  statutes; (c) a consent/disclosure step in the H0 wizard + a 'second-person detected' consent prompt
  in capture (default: session-only, discard embeddings post-processing unless affirmatively kept);
  (d) a retention window per artifact type + a delete-account/delete-clip flow that CASCADES to derived
  artifacts (design P0-9's schema now so every artifact traces to its source clip/profile for purge;
  handle the cross-account-shared ReID gallery case). **P7-4** Legal:
  counsel review of US11615540B2 claim scope BEFORE commercial launch; license ledger re-audit for
  anything NC that became ship-critical (SMPL-dependent challengers, WiLoR-ND, GPL components
  isolation). **P7-5** VERIFIED ladder: promote stages one by one through their documented gates —
  the product markets ONLY VERIFIED capabilities as accurate; everything else ships behind
  preview/trust-band labels (this honesty is a feature — pb.vision's silent failure modes are
  documented user pain).

---

# PART IV — STANDING RULES FOR EVERY AGENT ON THIS ROADMAP

1. **Read order for a fresh session:** `CLAUDE.md` (auto-loads the pointer) → `FABLE_OPERATING_MANUAL.md`
   → this file PART 0 (any blank owner item = typed STOP) + Part I (incl. I.7 critical path) →
   `BUILD_CHECKLIST.md` (last ~15 bullets) → `runs/manager/gpu_fleet.md` (reconcile orphaned VMs) →
   `CAPABILITIES.md` (canonical truth) + `PIPELINE_STATUS.md` as truth-checks → the linked lane/run
   evidence. `RESET_HANDOFF_20260705.md` = historical restart context when needed.
2. **Protected data:** Outdoor + Indoor labels NEVER touched without a pre-registered
   `runs/manager/heldout_eval_ledger.md` row + explicit STOP for manager go. Burlington/Wolverine =
   internal-val only. CVAT labels = scoring only, never construction. New owner captures get a role
   (train/internal-val/held-out) AT INGEST and held-out ones inherit full protection.
3. **Truth discipline:** VERIFIED requires the documented gate on real labels. Scaffold/diagnostic
   artifacts never promote. Trust bands: every degraded/predicted/estimated output is banded, never
   silently faked; fail-closed stays fail-closed. Honest kills are wins — log them with evidence.
   **Calibrate the bands (harsh-review):** wire `calibrate_confidence_bands.py` into PIPELINE_SUMMARY;
   bin held-out outputs by badge and require empirical error monotonic (verified ≤ preview ≤
   low_confidence); periodically re-verify a once-passed gate still holds as capture conditions drift.
4. **Lane protocol:** file-disjoint concurrent lanes; explicit file ownership in every lane spec;
   coordination ONLY via BUILD_CHECKLIST.md bullets + commit messages; artifacts under
   `runs/lanes/<lane>_<date>/`; every lane runs its blast-radius tests WIDE (MPLBACKEND=Agg) and
   reports honestly; every new root .md registers in the doc-consistency allowlist; every new CLI
   ships its direct-CLI reference test same-lane.
5. **Kill list (do NOT re-attempt without new evidence):** SAM3.1 ball; TrackNetV4 upstream weights;
   TrackNetV5 (proprietary); PB-MAT; CoTracker3/Track-On2 ball gap-fill; SAM2 ball mask propagation;
   PhysPT (blockers unchanged); RTMW/RTMPose skeletons; rectangle→6DoF paddle promotion; 160×90 CAL
   architectures; warm-worker daemon; eval-clip stage parallelism; fusion re-tuning without owner
   data; VNDetectTrajectories; small-single-match supervised fine-tunes (now externally corroborated);
   NEW from research: FoundationPose-class zero-shot on the paddle (HANDAL evidence), HOISDF/MOHO
   (no license), HOLD/MultiPly (GPU-hours per video), free-narration LLM coaching (SportsGPT
   ablation), officiating-grade single-cam line calls (nobody on earth does it without hardware).
6. **License stance (owner ruling 2026-07-05, EDGE_PLAYBOOK header):** private use for now
   (owner + possibly friends) — licenses are NOT a constraint; use whatever helps (SMPL/NC/GPL/ND/
   no-license/scraped video all fine). Keep only a one-line "what we used" inventory per lane as
   future-proofing; revisit only IF the product ever expands beyond friends. Technical kills stay
   killed on technical grounds (e.g. FoundationPose died on HANDAL evidence, not its license).
   Build nothing hardcoded to the owner: person/court/gear specificity lives in H0 profiles with
   generic-path fallbacks.
7. **GPU FLEET (multi-GPU — owner 2026-07-06: buy more GPUs to parallelize).** Before EVERY lane run
   the **safe-parallelism check**: is it file-disjoint (owned-files overlap 0 with every in-flight
   lane, per BUILD_CHECKLIST), data-disjoint (doesn't touch held-out/protected labels without a ledger
   row), and resource-disjoint? If it passes and no idle matching GPU exists, **provision a NEW GCP
   spot GPU and run this lane isolated on its own GPU so lanes never contend** (one physical GPU per
   lane; `EXCLUSIVE_PROCESS` compute mode). Fable DECIDES provision/reuse/teardown; a Codex/script lane
   runs the `gcloud` calls (SPOT + `--instance-termination-action=STOP` + `fable-lane=<lane>` label) and
   returns a VM report. Track the fleet in `runs/manager/gpu_fleet.md`; tear down a VM the moment its
   lane ends; cost cap (OWNER RULING 2026-07-06): **≤$5/GPU/hr, max 4 concurrent GPUs; teardown/delete the moment
   a lane ends — idle spend is never acceptable** (a 5th GPU or any >$5/hr VM = a
   `needs-purchase-approval` STOP). Full model: `FABLE_OPERATING_MANUAL.md` §12 + `runs/research_sota_20260705/fable5_manager_setup.md`. (Legacy single-GPU serialize via
   `scripts/gpu-train-lock.sh` / `gpu-eval-run.sh`; verify with nvidia-smi before claiming
   availability; auto-clean dispatch dirs (until P5-1 lands, clean manually); md5/schema-sync
   discipline whenever artifact fields change (`schemas/__init__.py` → VM or remote BODY dies with
   extra_forbidden)).
8. **Known traps (cost real days — respect them):** `--clip` must be passed or clip id = "source";
   Outdoor needs `--remote-command-timeout-s 7200`; blurball↔WASB checkpoint key amputation under
   strict=False; monolithic ~1GB JSON kills the viewer (windowed refs/mesh index only); LOCAL disk
   preflight before any harvest/ingest lane (`df -h` — this Mac has sat at ~95%; per-lane disk budget,
   big corpora belong on VM disks); background
   Bash tasks get kill-swept on the manager Mac (nohup+disown+Monitor); Codex sandbox has no
   network/MPS/localhost; Sonnet agents die passive-waiting (bounded poll loops + SendMessage
   resume); headless far-camera screenshots miss body-scale misalignment (zoomed browser checks
   required); synthetic byte-identity tests miss real finalizer interactions (verify on a real GPU
   run); doc-consistency tests drift within days (register new docs same-lane).
10. **Signal-adoption discipline (tech-audit 2026-07-06).** Before adopting ANY new vision signal
   into a fusion stack: (a) re-derive the CURRENT ablation from repo artifacts — which existing
   signal actually moves the metric (the paddle lesson: palm-only IoU 0.065 vs boxes carrying +0.19
   — the aspirational signal list didn't match measured reality); (b) run the 10-line pixel-math
   conditioning check (arctan(keypoint-noise px / apparent-baseline px) at real working distance)
   — if the geometry can't beat the incumbent at the ranges/blur that matter, don't build it.
9. **STOP-AND-ASK when genuinely blocked (owner 2026-07-06).** Blocked = a decision needing info,
   judgment, money, or authority ONLY the owner has AND no standing rule/kill-list/ruling covers it.
   If a standing rule answers it, proceed and log the ruling — do NOT stop. Otherwise classify into
   exactly one bucket and STOP, surfacing the blocker AS THE RESULT (lead your check-in with it, never
   bury it): **needs-validation** (a self-reported PASS contradicts its numbers, or a pre-registered
   held-out metric MISSED), **needs-advice** (two valid approaches, product-taste trade-off),
   **needs-labeling** (next unlock needs owner in-domain data/labels no lane can make),
   **needs-decision** (scope/priority/re-attempt-a-kill-listed-thing), **needs-purchase-approval**
   (spend beyond the envelope). Shape: one-line ask · why-no-rule-covers-it · minimal evidence ·
   options+your-leaning · safe-default-if-unanswered · what-still-runs-unaffected. Never guess past a
   real blocker. Full protocol: `FABLE_OPERATING_MANUAL.md` §13.

# PART V — EVIDENCE MAP

- Companion: `EDGE_PLAYBOOK.md` — profile-first hacks H0-H26 + iPhone-capture hacks H27-H34, exact stack, exact data sources, task deltas
- Pass 2/3 research addenda: PART II-B (citation-graph deep dive) + PART II-C (court/net + global fusion + production); reports in `runs/research_sota_20260705/pass2_*.md` + `pass3_*.md`
- Research (this doc's Part II): `runs/research_sota_20260705/{README,ball_report,body_report,paddle_report,product_report}.md`
- Current-state canon: `RESET_HANDOFF_20260705.md`, `CAPABILITIES.md`, `PIPELINE_STATUS.md`,
  `MASTER_PLAN.md`, `runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md`
- Ball: `runs/manager/heldout_eval_ledger.md` (rows 4, 19-23), `runs/lanes/ball_tracking_long_run_STATUS.md`,
  `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`, `runs/lanes/ball_p3a_bvp_anchor_first_20260705/`
- Body/visual: `runs/lanes/joint_placement_4videos_20260704/FINAL_REPORT.md`,
  `runs/lanes/visual_polish_20260705/`, `runs/visual1_wolverine_20260705T220517Z/`
- Paddle: `RACKET_6DOF_GOAL.md`, `runs/lanes/racket_6dof_20260705/STATUS.md` (+ final_v3)
- Court: `runs/lanes/court_autofind_20260705/` (+ handoff patch), `OVERLAPPING_COURT_CALIBRATION_GOAL.md`
- Speed: `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md`, `runs/lanes/body_chunkfix_20260705/REPORT.md`,
  `runs/lanes/payload_collapse_isolation_20260705/REPORT.md`
- Synergy/dead-code: `runs/lanes/e2e_synergy_audit_20260705/`

*Maintained by the manager session. Update checkboxes + add dated notes as lanes land; never let this
document claim more than the linked evidence shows.*
